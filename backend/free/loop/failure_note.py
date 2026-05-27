"""

``SemanticFactStore`` 直参照を廃止し、書込は
:class:`~backend.free.memory.views.loop.LoopFactView` 経由に統一した。

提供する API:

1. ``compute_failure_signature_from_gate(gate_result, mdp_steps)`` —
   品質ゲート結果 と MDP ステップ (agent_tracer) から
   ``(error_type, normalized_file_path, last_3_step_actions)`` の SHA1 先頭
   12 桁ハッシュを算出する。
2. ``write_failure_note(view, ...)`` — 自律ループ 1 イテレーションで品質
   ゲートが失敗した時に LoopFactView 経由で ``failure_pattern`` ファクトを
   書き込む。
3. ``consolidate_failure_patterns(view, ...)`` — sleep-time **Step 13** 本体。
   LoopFactView.consolidate_failure_patterns を呼び出し、``ConsolidationSummary``
   に要約する。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from backend.free.loop.quality_gate import GateResult, QualityGateOutcome
from backend.free.memory.extractors.mdp_trace import (
    LOOP_FAILURE_PREFIX,
    compute_failure_signature,
)
from backend.free.memory.types import SemanticFact
from backend.free.memory.views.loop import LoopFactView
from backend.log_config import get_logger

logger = get_logger("loop.failure_note")


# ──────────────────────────────────────────────────────────────────────────
# 定数・パターン
# ──────────────────────────────────────────────────────────────────────────

FAILURE_PREDICATE = "failed_with"
"""failure_pattern ファクトの predicate (固定)"""

#: 代表的な Python / TypeScript / pytest エラータイプ名の抽出パターン。
_ERROR_TYPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Failure))\b"),
    re.compile(r"\b(error\s+TS\d+)\b"),
    re.compile(r"\b(SyntaxError|TypeError|ValueError|RuntimeError|AssertionError)\b"),
)

#: ソースファイルパス抽出用。
_FILE_PATH_PATTERN = re.compile(
    r"(?P<path>[\w.+/\\-]+\.(?:py|pyi|ts|tsx|js|jsx|mjs|svelte|json))(?::\d+)?",
)

_PATH_SEPARATOR = "/"

#: 保持する outcomes_history 最大エントリ数
MAX_OUTCOMES_HISTORY = 5

#: object 文字列の最大長
MAX_OBJECT_LEN = 2000


# ──────────────────────────────────────────────────────────────────────────
# データ型
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FailurePayload:
    """failure_pattern ファクトの ``object`` フィールド JSON 表現。

    :class:`LoopFactView` 側が JSON 生成を担うが、従来の
    ``FailurePayload.to_json`` 出力とは互換を保つ。テスト用および
    ``parse_failure_object`` の逆写像として保持する。
    """

    error_type: str
    normalized_file_path: str
    last_actions: list[str]
    occurrences: int
    outcomes_history: list[str]
    mitigation: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "error_type": self.error_type,
            "normalized_file_path": self.normalized_file_path,
            "last_actions": list(self.last_actions),
            "occurrences": int(self.occurrences),
            "outcomes_history": list(self.outcomes_history),
        }
        if self.mitigation is not None:
            payload["mitigation"] = self.mitigation
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(text) > MAX_OBJECT_LEN:
            trimmed = FailurePayload(
                error_type=self.error_type,
                normalized_file_path=self.normalized_file_path,
                last_actions=self.last_actions,
                occurrences=self.occurrences,
                outcomes_history=self.outcomes_history[-1:],
                mitigation=(
                    self.mitigation[:200] if self.mitigation is not None else None
                ),
            )
            text = json.dumps(
                {
                    "error_type": trimmed.error_type,
                    "normalized_file_path": trimmed.normalized_file_path,
                    "last_actions": trimmed.last_actions,
                    "occurrences": trimmed.occurrences,
                    "outcomes_history": trimmed.outcomes_history,
                    **(
                        {"mitigation": trimmed.mitigation}
                        if trimmed.mitigation is not None
                        else {}
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        return text


# ──────────────────────────────────────────────────────────────────────────
# GateResult / MDPStep からの情報抽出
# ──────────────────────────────────────────────────────────────────────────


def extract_error_type(gate_result: GateResult) -> str:
    """``GateResult`` から ``error_type`` 文字列を推定する。"""
    err = (gate_result.error or "").strip()
    if err:
        head = err.split(":", 1)[0].strip()
        if head:
            return head
    haystack = "\n".join(
        [gate_result.stderr_tail or "", gate_result.stdout_tail or ""],
    )
    for pattern in _ERROR_TYPE_PATTERNS:
        m = pattern.search(haystack)
        if m:
            return m.group(1).strip()
    rc = gate_result.returncode
    rc_text = str(rc) if rc is not None else "?"
    return f"gate:{gate_result.name}:{rc_text}"


def extract_file_path(gate_result: GateResult) -> str:
    """``GateResult`` から関連ソースファイルの正規化パスを推定する。"""
    for source in (gate_result.stdout_tail or "", gate_result.stderr_tail or ""):
        if not source:
            continue
        m = _FILE_PATH_PATTERN.search(source)
        if m:
            return _normalize_file_path(m.group("path"))
    return ""


def _normalize_file_path(raw: str) -> str:
    path = raw.replace("\\", _PATH_SEPARATOR).strip()
    path = path.strip("'\"(),")
    if path.startswith("./"):
        path = path[2:]
    return path


def extract_actions_from_steps(steps: Iterable[Any]) -> list[str]:
    """MDP ステップ列から ``action`` 文字列のリストを抽出する。"""
    actions: list[str] = []
    for s in steps:
        if isinstance(s, dict):
            value = s.get("action", "")
        else:
            value = getattr(s, "action", "")
        actions.append(str(value or "").strip())
    return actions


def compute_failure_signature_from_gate(
    gate_result: GateResult,
    mdp_steps: Iterable[Any] | None = None,
) -> str:
    """``(error_type, normalized_file_path, last_3_step_actions)`` の SHA1[:12] を返す。"""
    steps = list(mdp_steps) if mdp_steps is not None else []
    return compute_failure_signature(
        error_type=extract_error_type(gate_result),
        normalized_file_path=extract_file_path(gate_result),
        last_actions=extract_actions_from_steps(steps),
    )


def compute_failure_signatures_from_outcome(
    outcome: QualityGateOutcome,
    mdp_steps: Iterable[Any] | None = None,
) -> list[str]:
    """``QualityGateOutcome`` 内の失敗ゲート全てについて signature を返す。"""
    if outcome.ok:
        return []
    steps = list(mdp_steps) if mdp_steps is not None else []
    out: list[str] = []
    for r in outcome.results:
        if r.is_failure():
            out.append(compute_failure_signature_from_gate(r, steps))
    return out


# ──────────────────────────────────────────────────────────────────────────
# object フィールドのエンコード / パース
# ──────────────────────────────────────────────────────────────────────────


def parse_failure_object(text: str) -> dict[str, Any]:
    """``failure_pattern.object`` を dict にパースする。

    - 新形式 (JSON): そのまま dict 化して返す
    - レガシー形式 (``outcome=...; last_actions=...``): ベスト
      エフォートでパースして dict を返す
    - どちらにも該当しない場合は ``{"raw": text, "occurrences": 1}``
    """
    if not text:
        return {"occurrences": 1}
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        data = None
    if isinstance(data, dict):
        data.setdefault("occurrences", 1)
        data.setdefault("outcomes_history", [])
        data.setdefault("last_actions", [])
        return data
    legacy: dict[str, Any] = {
        "occurrences": 1,
        "outcomes_history": [],
        "last_actions": [],
    }
    segments = [seg.strip() for seg in text.split(";") if seg.strip()]
    for seg in segments:
        if "=" not in seg:
            continue
        key, _, value = seg.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "outcome" and value:
            legacy["outcomes_history"] = [value]
        elif key == "last_actions" and value:
            legacy["last_actions"] = [a for a in value.split("/") if a]
    legacy["legacy_text"] = text
    return legacy


# ──────────────────────────────────────────────────────────────────────────
# 即時書き込み (LoopFactView 経由)
# ──────────────────────────────────────────────────────────────────────────


def _gate_outcome_label(gate_result: GateResult) -> str:
    """``GateResult`` を人間可読な 1 行ラベルに要約する。"""
    parts = [f"gate={gate_result.name}"]
    if gate_result.returncode is not None:
        parts.append(f"rc={gate_result.returncode}")
    if gate_result.error:
        parts.append(f"err={gate_result.error[:80]}")
    if gate_result.stderr_tail:
        head = gate_result.stderr_tail.strip().splitlines()[0][:80]
        if head:
            parts.append(f"stderr={head}")
    return "; ".join(parts)


def write_failure_note(
    view: LoopFactView,
    *,
    project_id: str,
    gate_result: GateResult,
    mdp_steps: Iterable[Any] | None = None,
    trace_id: str | None = None,
    mitigation: str | None = None,
    confidence: float = 0.7,
    now: float | None = None,
) -> SemanticFact:
    """自律ループ失敗時に ``failure_pattern`` ファクトを即時書き込みする。

    Args:
        view: 対象スコープの :class:`LoopFactView`
        project_id: ``scope = project:<project_id>`` に設定する
        gate_result: 失敗した品質ゲートの結果
        mdp_steps: 直近 MDP ステップ列。最後の 3 件の ``action`` を signature に使う
        trace_id: ``SemanticFact.trace_id`` に格納するエピソード ID
        mitigation: 人間 / LLM から提供された回避策 (任意)
        confidence: ファクトの初期 confidence (デフォルト 0.7)
        now: 現在時刻 (テスト差し替え用)

    Returns:
        永続化された :class:`SemanticFact`

    Raises:
        ValueError: ``project_id`` が空
    """
    if not project_id:
        raise ValueError("project_id must be non-empty")
    steps = list(mdp_steps) if mdp_steps is not None else []
    error_type = extract_error_type(gate_result)
    file_path = extract_file_path(gate_result)
    last_actions = extract_actions_from_steps(steps)
    signature = compute_failure_signature(
        error_type=error_type,
        normalized_file_path=file_path,
        last_actions=last_actions,
    )
    outcome_label = _gate_outcome_label(gate_result)
    fact = view.write_failure_pattern(
        project_id=project_id,
        signature=signature,
        error_type=error_type,
        normalized_file_path=file_path,
        last_actions=last_actions,
        mitigation=mitigation,
        outcome_label=outcome_label if outcome_label else None,
        confidence=confidence,
        trace_id=trace_id,
        now=now,
    )
    logger.info(
        "write_failure_note: subject=%s%s signature=%s gate=%s project=%s",
        LOOP_FAILURE_PREFIX, signature, signature, gate_result.name, project_id,
    )
    return fact


# ──────────────────────────────────────────────────────────────────────────
# Step 13: 統合 (LoopFactView 委譲)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConsolidationSummary:
    """``consolidate_failure_patterns`` の集約結果"""

    groups_scanned: int
    merged_groups: int
    superseded_facts: int
    kept_facts: int
    mitigation_updated: int

    def as_dict(self) -> dict[str, int]:
        return {
            "groups_scanned": self.groups_scanned,
            "merged_groups": self.merged_groups,
            "superseded_facts": self.superseded_facts,
            "kept_facts": self.kept_facts,
            "mitigation_updated": self.mitigation_updated,
        }


def consolidate_failure_patterns(
    view: LoopFactView,
    *,
    scope: str | None = None,
    project_id: str | None = None,
) -> ConsolidationSummary:
    """同一 ``failure_signature`` の failure_pattern ファクトを統合する (Step 13)。

    LoopFactView.consolidate_failure_patterns に委譲する。``scope`` が指定された
    場合は ``project:<id>`` 形式から project_id を抽出する (後方互換のため
    両方の引数を受け付ける)。

    Args:
        view: 対象スコープの :class:`LoopFactView`
        scope: ``project:<id>`` 形式のスコープ文字列 (後方互換)
        project_id: 対象 project_id (新 API)

    Returns:
        :class:`ConsolidationSummary`

    Note:
        LoopFactView の consolidate_failure_patterns が
        ``mitigation_updated`` を別途記録しないため、常に 0 を返す。
        ``outcomes_history`` / ``mitigation`` の個別マージロジックは
        LoopFactView 側が未実装のため、sleep_update.py
        分離時に対応予定。
    """
    target_project_id = project_id
    if target_project_id is None and scope is not None:
        # scope="project:<id>" から project_id を抽出
        if scope.startswith("project:"):
            target_project_id = scope[len("project:"):]
    summary = view.consolidate_failure_patterns(project_id=target_project_id)
    result = ConsolidationSummary(
        groups_scanned=int(summary.get("groups_scanned", 0)),
        merged_groups=int(summary.get("merged_groups", 0)),
        superseded_facts=int(summary.get("superseded_facts", 0)),
        kept_facts=int(summary.get("kept_facts", 0)),
        mitigation_updated=int(summary.get("mitigation_updated", 0)),
    )
    if result.merged_groups:
        logger.info(
            "Step 13: consolidated %d failure_pattern group(s) "
            "(superseded=%d)",
            result.merged_groups, result.superseded_facts,
        )
    return result


__all__ = [
    "ConsolidationSummary",
    "FAILURE_PREDICATE",
    "FailurePayload",
    "MAX_OBJECT_LEN",
    "MAX_OUTCOMES_HISTORY",
    "compute_failure_signature_from_gate",
    "compute_failure_signatures_from_outcome",
    "consolidate_failure_patterns",
    "extract_actions_from_steps",
    "extract_error_type",
    "extract_file_path",
    "parse_failure_object",
    "write_failure_note",
]
