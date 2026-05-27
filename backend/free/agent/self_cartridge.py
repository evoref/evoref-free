"""``@self`` 仮想カートリッジ — EvorefMem

エージェントが「現在の自分自身」を参照するための仮想カートリッジ実装。
``SemanticFactStore`` 直参照を廃止し、収集は
:class:`~backend.free.memory.views.loop.LoopFactView` (agent は loop pillar 配下)
経由に統一した。

責務:

1. SemMem (global / project スコープ) から以下を集約した
   :class:`AgentConstants` を構築する
   - ユーザープロファイル (``personal_fact`` / ``preference``)
     — FACT_OWNERSHIP.readers に ``"loop"`` を追加済
   - active な ``policy`` ファクト (``confidence >= 閾値``)
   - pinned ファクト (global / project)
2. ``@self`` トークンを含むテキストを上記の整形済ブロックで置換する
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.free.memory.types import SemanticFact
from backend.free.memory.views.loop import LoopFactView
from backend.log_config import get_logger

logger = get_logger("agent.self_cartridge")


# ──────────────────────────────────────────────────────────────────────────
# 公開定数
# ──────────────────────────────────────────────────────────────────────────


SELF_CARTRIDGE_REF = "@self"
"""仮想カートリッジ参照トークン"""

# `@self` の単語境界マッチ用パターン (例: ``@selfish`` は除外)
SELF_CARTRIDGE_PATTERN = re.compile(r"@self(?![A-Za-z0-9_])")

# デフォルト上限値 (token budget 暴発を防ぐ)
DEFAULT_MAX_USER_PROFILE_FACTS = 20
DEFAULT_MAX_ACTIVE_POLICY_FACTS = 30
DEFAULT_MAX_PINNED_GLOBAL_FACTS = 30
DEFAULT_MAX_PINNED_PROJECT_FACTS = 30
DEFAULT_POLICY_ACTIVATION_MIN_CONFIDENCE = 0.7


# ──────────────────────────────────────────────────────────────────────────
# DTO
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentConstants:
    """エージェントの「定数 (constants)」 — process 1 回中で不変な参照素材。"""

    project_id: str | None = None
    user_profile: tuple[SemanticFact, ...] = field(default_factory=tuple)
    active_policies: tuple[SemanticFact, ...] = field(default_factory=tuple)
    pinned_global: tuple[SemanticFact, ...] = field(default_factory=tuple)
    pinned_project: tuple[SemanticFact, ...] = field(default_factory=tuple)
    policy_activation_min_confidence: float = (
        DEFAULT_POLICY_ACTIVATION_MIN_CONFIDENCE
    )
    skipped_below_confidence: int = 0

    def is_empty(self) -> bool:
        return not (
            self.user_profile
            or self.active_policies
            or self.pinned_global
            or self.pinned_project
        )

    def all_facts(self) -> list[SemanticFact]:
        """全ファクトを id 重複排除して返す (順序: profile → policy → pinned)。"""
        seen: set[str] = set()
        out: list[SemanticFact] = []
        for bucket in (
            self.user_profile,
            self.active_policies,
            self.pinned_global,
            self.pinned_project,
        ):
            for fact in bucket:
                if fact.id in seen:
                    continue
                seen.add(fact.id)
                out.append(fact)
        return out

    def total_facts(self) -> int:
        return len(self.all_facts())

    def as_summary(self) -> dict[str, int | str | None | float]:
        return {
            "project_id": self.project_id,
            "user_profile": len(self.user_profile),
            "active_policies": len(self.active_policies),
            "pinned_global": len(self.pinned_global),
            "pinned_project": len(self.pinned_project),
            "total": self.total_facts(),
            "policy_activation_min_confidence": (
                self.policy_activation_min_confidence
            ),
            "skipped_below_confidence": self.skipped_below_confidence,
        }


EMPTY_CONSTANTS = AgentConstants()


# ──────────────────────────────────────────────────────────────────────────
# 収集ロジック
# ──────────────────────────────────────────────────────────────────────────


def gather_constants(
    *,
    view: LoopFactView | None = None,
    project_id: str | None = None,
    policy_activation_min_confidence: float = (
        DEFAULT_POLICY_ACTIVATION_MIN_CONFIDENCE
    ),
    max_user_profile: int = DEFAULT_MAX_USER_PROFILE_FACTS,
    max_active_policies: int = DEFAULT_MAX_ACTIVE_POLICY_FACTS,
    max_pinned_global: int = DEFAULT_MAX_PINNED_GLOBAL_FACTS,
    max_pinned_project: int = DEFAULT_MAX_PINNED_PROJECT_FACTS,
) -> AgentConstants:
    """LoopFactView から :class:`AgentConstants` を構築する。

    ``view`` が ``None`` の場合は :data:`EMPTY_CONSTANTS` を返す (chat モード
    などで SemMem 未配線時)。``project_id`` が非空なら bootstrap_context で
    active_policies + pinned を一括収集し、``None`` なら汎用メソッドで
    代替取得する。

    Args:
        view: SemMem への :class:`LoopFactView` (読取可能なら書込不要な形でも可)。
        project_id: 現在のプロジェクト ID。bootstrap_context で
            pinned_global / pinned_project 分離に使用。
        policy_activation_min_confidence: ``policy`` ファクトの active 判定閾値
        max_user_profile: ユーザープロファイル上限件数
        max_active_policies: active policy 上限件数
        max_pinned_global: global pinned 上限件数
        max_pinned_project: project pinned 上限件数
    """
    if view is None:
        return EMPTY_CONSTANTS

    user_profile = view.get_user_profile(limit=max_user_profile)

    active_policies: list[SemanticFact] = []
    pinned_global: list[SemanticFact] = []
    pinned_project: list[SemanticFact] = []
    skipped_below = 0

    if project_id:
        bootstrap = view.bootstrap_context(
            project_id=project_id,
            policy_activation_min_confidence=policy_activation_min_confidence,
        )
        active_policies = list(bootstrap.active_policies)
        pinned_global = list(bootstrap.pinned_global)
        pinned_project = list(bootstrap.pinned_project)
        skipped_below = bootstrap.skipped_below_confidence
    else:
        active_policies = view.get_active_policies(
            min_confidence=policy_activation_min_confidence,
        )
        pinned_all = view.get_pinned_facts()
        for fact in pinned_all:
            if fact.scope == "global":
                pinned_global.append(fact)
            else:
                pinned_project.append(fact)

    active_policies.sort(key=lambda f: (f.created_at, f.id))
    pinned_global.sort(key=lambda f: (-f.accessed_at, -f.created_at, f.id))
    pinned_project.sort(key=lambda f: (-f.accessed_at, -f.created_at, f.id))

    if max_active_policies > 0:
        active_policies = active_policies[:max_active_policies]
    if max_pinned_global > 0:
        pinned_global = pinned_global[:max_pinned_global]
    if max_pinned_project > 0:
        pinned_project = pinned_project[:max_pinned_project]

    constants = AgentConstants(
        project_id=project_id,
        user_profile=tuple(user_profile),
        active_policies=tuple(active_policies),
        pinned_global=tuple(pinned_global),
        pinned_project=tuple(pinned_project),
        policy_activation_min_confidence=policy_activation_min_confidence,
        skipped_below_confidence=skipped_below,
    )
    logger.debug(
        "@self gather_constants: %s", constants.as_summary(),
    )
    return constants


# ──────────────────────────────────────────────────────────────────────────
# 整形・展開
# ──────────────────────────────────────────────────────────────────────────


def _fact_line(fact: SemanticFact) -> str:
    """1 ファクトを ``subject.predicate = object`` 形式の 1 行に整形する。"""
    subject = fact.subject or "?"
    predicate = fact.predicate or "?"
    obj = fact.object or ""
    obj = re.sub(r"\s+", " ", obj).strip()
    if len(obj) > 200:
        obj = obj[:197] + "..."
    return f"- {subject}.{predicate} = {obj}"


def _format_section(title: str, facts: tuple[SemanticFact, ...]) -> list[str]:
    if not facts:
        return []
    lines = [f"## {title} ({len(facts)})"]
    lines.extend(_fact_line(f) for f in facts)
    return lines


def format_self_cartridge(constants: AgentConstants) -> str:
    """:class:`AgentConstants` を ``@self`` 展開用テキストブロックに整形する。"""
    if constants.is_empty():
        return "[self: no constants available]"

    lines: list[str] = ["[self virtual cartridge]"]
    if constants.project_id:
        lines.append(f"project_id: {constants.project_id}")
    lines.extend(_format_section("User profile", constants.user_profile))
    lines.extend(_format_section("Active policies", constants.active_policies))
    lines.extend(_format_section("Pinned (global)", constants.pinned_global))
    lines.extend(_format_section("Pinned (project)", constants.pinned_project))
    lines.append("[/self]")
    return "\n".join(lines)


def contains_self_reference(text: str | None) -> bool:
    """``text`` に ``@self`` トークンが含まれるか判定する。"""
    if not text:
        return False
    return SELF_CARTRIDGE_PATTERN.search(text) is not None


def expand_self_references(text: str, constants: AgentConstants) -> str:
    """``text`` 内の ``@self`` を整形済カートリッジブロックで置換する。"""
    if not contains_self_reference(text):
        return text
    block = format_self_cartridge(constants)
    return SELF_CARTRIDGE_PATTERN.sub(lambda _m: block, text)


__all__ = [
    "AgentConstants",
    "DEFAULT_MAX_ACTIVE_POLICY_FACTS",
    "DEFAULT_MAX_PINNED_GLOBAL_FACTS",
    "DEFAULT_MAX_PINNED_PROJECT_FACTS",
    "DEFAULT_MAX_USER_PROFILE_FACTS",
    "DEFAULT_POLICY_ACTIVATION_MIN_CONFIDENCE",
    "EMPTY_CONSTANTS",
    "SELF_CARTRIDGE_PATTERN",
    "SELF_CARTRIDGE_REF",
    "contains_self_reference",
    "expand_self_references",
    "format_self_cartridge",
    "gather_constants",
]
