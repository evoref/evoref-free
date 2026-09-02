"""

EvorefMem 統合仕様 における自律ループの **クリーンコンテキスト再起動**
層。``SemanticFactStore`` 直参照を廃止し、bootstrap ロジックは
:class:`~backend.free.memory.views.loop.LoopFactView` へ委譲した。

提供する API:

1. ``estimate_episodic_tokens(wm, stm)`` — 現在の episodic 層 (Working Memory +
   Short-Term Memory) の推定トークン量を返す純粋関数
2. ``reset_episodic_context(wm, stm, ...)`` — episodic 層 (WM/STM) を破棄する
   (SemMem には触らない)
3. ``bootstrap_project_context(view, project_id, ...)`` — LoopFactView 経由で
   SemMem の active ``policy`` / ``pinned`` / ``failure_pattern`` を収集する
   純粋関数
4. ``maybe_reset_and_bootstrap(view, ...)`` — 閾値超過判定 + reset + bootstrap
   を一括実行するオーケストレータ

設計原則 (CLAUDE.md / .claude/rules/backend.md):

- Python 3.12+ 型表現
- LLM 不要 (filtering とソートのみ)
- View 経由で SemMem 操作
- 後方互換不要
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from backend.free.memory.stores.short_term import ShortTermMemory
from backend.free.memory.types import SemanticFact
from backend.free.memory.views.loop import LoopFactView
from backend.free.memory.stores.working import WorkingMemory
from backend.log_config import get_logger
from backend.utils import estimate_tokens

logger = get_logger("loop.bootstrap")


# ──────────────────────────────────────────────────────────────────────────
# 結果 DTO
# ──────────────────────────────────────────────────────────────────────────


ResetTrigger = Literal["threshold", "manual", "startup"]


@dataclass(frozen=True)
class ResetReport:
    """``reset_episodic_context`` の実行結果"""

    triggered_by: ResetTrigger
    wm_turns_dropped: int
    wm_evicted_dropped: int
    stm_notes_dropped: int
    threshold_tokens: int | None
    observed_tokens: int | None

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            "triggered_by": self.triggered_by,
            "wm_turns_dropped": self.wm_turns_dropped,
            "wm_evicted_dropped": self.wm_evicted_dropped,
            "stm_notes_dropped": self.stm_notes_dropped,
            "threshold_tokens": self.threshold_tokens,
            "observed_tokens": self.observed_tokens,
        }


@dataclass(frozen=True)
class BootstrapResult:
    """``bootstrap_project_context`` の実行結果。

    ``tier1_facts()`` は active policy + pinned のみで構成され、``failure_pattern``
    は ``candidate_failure_patterns`` として別フィールドに分離している。
    """

    project_id: str
    active_policies: list[SemanticFact] = field(default_factory=list)
    pinned_global: list[SemanticFact] = field(default_factory=list)
    pinned_project: list[SemanticFact] = field(default_factory=list)
    candidate_failure_patterns: list[SemanticFact] = field(default_factory=list)
    skipped_below_confidence: int = 0
    policy_activation_min_confidence: float = 0.7
    artifact_count: int = 0

    def tier1_facts(self) -> list[SemanticFact]:
        """Tier 1 注入候補 (active policy + pinned) を id 重複排除して返す。"""
        seen: set[str] = set()
        out: list[SemanticFact] = []
        for fact in (
            *self.active_policies,
            *self.pinned_global,
            *self.pinned_project,
        ):
            if fact.id in seen:
                continue
            seen.add(fact.id)
            out.append(fact)
        return out

    def total_tier1(self) -> int:
        return len(self.tier1_facts())

    def as_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "active_policies": len(self.active_policies),
            "pinned_global": len(self.pinned_global),
            "pinned_project": len(self.pinned_project),
            "candidate_failure_patterns": len(self.candidate_failure_patterns),
            "skipped_below_confidence": self.skipped_below_confidence,
            "policy_activation_min_confidence": (
                self.policy_activation_min_confidence
            ),
            "total_tier1": self.total_tier1(),
            "artifact_count": self.artifact_count,
        }


# ──────────────────────────────────────────────────────────────────────────
# Episodic 層トークン推定 / リセット
# ──────────────────────────────────────────────────────────────────────────


def estimate_episodic_tokens(
    wm: WorkingMemory | None,
    stm: ShortTermMemory | None,
) -> int:
    """Working Memory + Short-Term Memory の推定トークン数を返す。"""
    total = 0
    if wm is not None:
        for turn in wm.turns:
            content = turn.get("content") or ""
            total += estimate_tokens(content)
    if stm is not None:
        for note in stm.notes.values():
            total += estimate_tokens(note.content or "")
    return total


def should_reset_episodic(
    observed_tokens: int,
    threshold_tokens: int,
) -> bool:
    """``observed_tokens`` が ``threshold_tokens`` 以上なら ``True``。"""
    if threshold_tokens <= 0:
        return False
    return observed_tokens >= threshold_tokens


def reset_episodic_context(
    wm: WorkingMemory | None,
    stm: ShortTermMemory | None,
    *,
    triggered_by: ResetTrigger = "manual",
    threshold_tokens: int | None = None,
    observed_tokens: int | None = None,
) -> ResetReport:
    """episodic context (WM + STM) を破棄する。"""
    wm_turns_dropped = 0
    wm_evicted_dropped = 0
    if wm is not None:
        wm_turns_dropped = len(wm.turns)
        wm_evicted_dropped = len(wm._evicted)
        wm.turns.clear()
        wm._evicted.clear()

    stm_notes_dropped = 0
    if stm is not None:
        stm_notes_dropped = len(stm.notes)
        stm.notes.clear()
        stm._cache.clear()
        stm._cache_dirty = True

    report = ResetReport(
        triggered_by=triggered_by,
        wm_turns_dropped=wm_turns_dropped,
        wm_evicted_dropped=wm_evicted_dropped,
        stm_notes_dropped=stm_notes_dropped,
        threshold_tokens=threshold_tokens,
        observed_tokens=observed_tokens,
    )
    logger.info(
        "reset_episodic_context: trigger=%s wm_turns=%d wm_evicted=%d "
        "stm_notes=%d observed_tokens=%s threshold_tokens=%s",
        triggered_by,
        wm_turns_dropped,
        wm_evicted_dropped,
        stm_notes_dropped,
        observed_tokens,
        threshold_tokens,
    )
    return report


# ──────────────────────────────────────────────────────────────────────────
# bootstrap_project_context (LoopFactView 委譲版)
# ──────────────────────────────────────────────────────────────────────────


def bootstrap_project_context(
    view: LoopFactView,
    *,
    project_id: str,
    policy_activation_min_confidence: float = 0.7,
) -> BootstrapResult:
    """SemMem からクリーンコンテキスト再起動後の Tier 1 素材を構築する。

    LoopFactView.bootstrap_context へ委譲し、結果を既存の
    :class:`BootstrapResult` 型に変換して返す。LoopDriverState.last_bootstrap
    の型互換を維持する。

    Args:
        view: ``stores=[global, project]`` + ``writeback_store=project`` 構成の
            :class:`LoopFactView`。
        project_id: 対象プロジェクト ID (空文字列は不可)
        policy_activation_min_confidence: active 判定の最小 confidence

    Returns:
        :class:`BootstrapResult`
    """
    if not project_id:
        raise ValueError("project_id must be non-empty")
    result = view.bootstrap_context(
        project_id=project_id,
        policy_activation_min_confidence=policy_activation_min_confidence,
    )
    converted = BootstrapResult(
        project_id=result.project_id,
        active_policies=list(result.active_policies),
        pinned_global=list(result.pinned_global),
        pinned_project=list(result.pinned_project),
        candidate_failure_patterns=list(result.candidate_failure_patterns),
        skipped_below_confidence=result.skipped_below_confidence,
        policy_activation_min_confidence=(
            result.policy_activation_min_confidence
        ),
        artifact_count=result.artifact_count,
    )
    logger.info(
        "bootstrap_project_context: project=%s policies=%d "
        "pinned_global=%d pinned_project=%d failure_candidates=%d "
        "artifact_count=%d skipped_below_confidence=%d (min_conf=%.2f)",
        project_id,
        len(converted.active_policies),
        len(converted.pinned_global),
        len(converted.pinned_project),
        len(converted.candidate_failure_patterns),
        converted.artifact_count,
        converted.skipped_below_confidence,
        policy_activation_min_confidence,
    )
    return converted


# ──────────────────────────────────────────────────────────────────────────
# オーケストレータ: 閾値判定 + reset + bootstrap
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MaybeResetReport:
    """``maybe_reset_and_bootstrap`` の実行結果"""

    observed_tokens: int
    threshold_tokens: int
    triggered: bool
    reset: ResetReport | None
    bootstrap: BootstrapResult | None

    def as_dict(self) -> dict[str, object]:
        return {
            "observed_tokens": self.observed_tokens,
            "threshold_tokens": self.threshold_tokens,
            "triggered": self.triggered,
            "reset": self.reset.as_dict() if self.reset else None,
            "bootstrap": self.bootstrap.as_dict() if self.bootstrap else None,
        }


def maybe_reset_and_bootstrap(
    view: LoopFactView,
    *,
    wm: WorkingMemory | None,
    stm: ShortTermMemory | None,
    project_id: str,
    threshold_tokens: int,
    policy_activation_min_confidence: float = 0.7,
    force: bool = False,
) -> MaybeResetReport:
    """トークン閾値超過時に ``reset`` + ``bootstrap`` を行うオーケストレータ。

    Args:
        view: ``stores=[global, project]`` + ``writeback_store=project`` 構成の
            :class:`LoopFactView`。
        wm: 対象 WorkingMemory
        stm: 対象 ShortTermMemory
        project_id: bootstrap 対象プロジェクト ID
        threshold_tokens: ``config.loop.context_reset_threshold_tokens``
        policy_activation_min_confidence: active 判定の最小 confidence
        force: ``True`` の場合は閾値判定を飛ばして強制リセットする

    Returns:
        :class:`MaybeResetReport` — ``triggered=False`` の場合は ``reset`` /
        ``bootstrap`` が ``None``。
    """
    if not project_id:
        raise ValueError("project_id must be non-empty")

    observed = estimate_episodic_tokens(wm, stm)
    triggered = force or should_reset_episodic(observed, threshold_tokens)
    if not triggered:
        return MaybeResetReport(
            observed_tokens=observed,
            threshold_tokens=threshold_tokens,
            triggered=False,
            reset=None,
            bootstrap=None,
        )

    trigger: ResetTrigger = "startup" if force else "threshold"
    reset = reset_episodic_context(
        wm,
        stm,
        triggered_by=trigger,
        threshold_tokens=threshold_tokens,
        observed_tokens=observed,
    )
    bootstrap = bootstrap_project_context(
        view,
        project_id=project_id,
        policy_activation_min_confidence=policy_activation_min_confidence,
    )
    return MaybeResetReport(
        observed_tokens=observed,
        threshold_tokens=threshold_tokens,
        triggered=True,
        reset=reset,
        bootstrap=bootstrap,
    )


__all__ = [
    "BootstrapResult",
    "MaybeResetReport",
    "ResetReport",
    "ResetTrigger",
    "bootstrap_project_context",
    "estimate_episodic_tokens",
    "maybe_reset_and_bootstrap",
    "reset_episodic_context",
    "should_reset_episodic",
]
