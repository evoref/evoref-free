"""学習の事例 (few-shot の手本 / 採用ゲートの評価ケース) から外すターンの述語

few-shot プール (``FewShotPool.add_from_experiences``) と Level 1 の採用ゲート
(``optimizer.prompt_eval.select_prompt_eval_cases``) は、どちらも「system prompt
と問いだけでは再現できないターン」を事例にしない。以前は両者が別々に判定して
いて、ツール実行結果を根拠にしたターン (``tool_grounded``) を few-shot は除き
採用ゲートは除かない、という非対称があった。判定をここに 1 つだけ持つ
(不変則 #14 (a) 同一性)。

ツール根拠のターンは、記録したツール結果ブロック (``gen_config.tool_context``)
を問いに添えれば採用ゲートで再生できる (:func:`replay_context`、2026-10-08)。
手本には引き続き使わない (:func:`grounding_reason` は変えない)。

どの関数も純関数で、外す理由 (文字列) か ``None`` を返す。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.free.core.verifier_events import decided_by_shortcut
from backend.free.learning.corrected_pairs import resolve_corrected_turn
from backend.free.learning.level0_instant import (
    TOOL_CONTEXT_KEY,
    TOOL_CONTEXT_TRUNCATED_KEY,
    used_corpus_evidence,
)

#: ツール実行結果を根拠にした応答 (``signals.tool_grounded``)。
REASON_TOOL_GROUNDED = "tool_grounded"
#: 文書チャンク (corpus) を注入した応答 (``gen_config.evidence_ids`` の ``corpus:``)。
REASON_CORPUS_GROUNDED = "corpus_grounded"
#: 長文生成の経路を通った応答 (``signals.long_form_used``)。
REASON_LONG_FORM = "long_form"
#: ツール根拠のターンだがツール結果ブロックが記録されていない (旧データ等)。
REASON_TOOL_CONTEXT_MISSING = "tool_context_missing"
#: 記録したツール結果ブロックが上限で切られている (同じ入力にならない)。
REASON_TOOL_CONTEXT_TRUNCATED = "tool_context_truncated"
#: ツールを近道 (recall / learned) が選んだ。結果そのものが誤りでありうる (不変則 #15)。
REASON_SHORTCUT_TOOL = "shortcut_tool"


def grounding_reason(exp: dict) -> str | None:
    """応答が外部の根拠 (ツール結果 / 文書チャンク) に依存していれば理由を返す。

    値はその問いのものであって文体ではない。手本に載ると同じ形の問いに
    根拠を引かず値を復唱し、評価ケースにすると system prompt だけの再生成には
    根拠が無く、どの候補でも同じ点になる。
    """
    signals = exp.get("signals") or {}
    if signals.get("tool_grounded"):
        return REASON_TOOL_GROUNDED
    if used_corpus_evidence(exp):
        return REASON_CORPUS_GROUNDED
    return None


def regeneration_mismatch_reason(exp: dict) -> str | None:
    """system prompt と問い (+ 記録したツール結果) で再生成しても同じ入力にならないなら理由を返す。

    採用ゲート (f_04 §4.5) の評価ケースから外す判定。文書チャンクを注入した
    ターンと、長文生成の経路 (計画・分割生成・検証) を通ったターン — ゲートの
    再生成は 1 回の短い生成なので、長文の結果 (成否を問わず) は再現できない
    (docs/f_04 §2.5)。ツール根拠のターンは :func:`replay_context` が判定する。
    """
    if used_corpus_evidence(exp):
        return REASON_CORPUS_GROUNDED
    if (exp.get("signals") or {}).get("long_form_used"):
        return REASON_LONG_FORM
    return None


def replay_context(exp: dict) -> tuple[str, str | None]:
    """ツール根拠のターンを再生するためのツール結果ブロックと、再生できない理由を返す。

    ``tool_grounded`` でないターンは ``("", None)`` (添えるものが無い)。ツール根拠の
    ターンは、ブロックが記録されていて切られておらず、ツールを近道が選んでいない
    ときだけ ``(ブロック, None)``。それ以外は ``("", 理由)``。
    """
    signals = exp.get("signals") or {}
    if not signals.get("tool_grounded"):
        return "", None
    decided_by = signals.get("decided_by")
    if decided_by_shortcut(decided_by if isinstance(decided_by, dict) else None):
        return "", REASON_SHORTCUT_TOOL
    gen_config = exp.get("gen_config")
    if not isinstance(gen_config, dict):
        gen_config = {}
    if gen_config.get(TOOL_CONTEXT_TRUNCATED_KEY):
        return "", REASON_TOOL_CONTEXT_TRUNCATED
    context = str(gen_config.get(TOOL_CONTEXT_KEY) or "")
    if not context.strip():
        return "", REASON_TOOL_CONTEXT_MISSING
    return context, None


def has_failure_evidence(signals: Mapping[str, Any]) -> bool:
    """訂正以外の失敗の証拠 (👎 / ``failed`` / 言い直し) があるか。

    採用ゲートの失敗ケース (``optimizer.prompt_eval``) と session snapshot の
    全文保持 (:func:`case_eligible_ids`) が同じ判定を使う。
    """
    return (
        signals.get("user_negative") is True
        or signals.get("turn_outcome") == "failed"
        or bool(signals.get("rephrased_query"))
    )


def case_eligible_ids(experiences: list[dict]) -> set[str]:
    """採用ゲートの失敗ケースになりうる経験の ``id`` を返す。

    失敗の証拠がある行と、検証済みの訂正 (``user_correction``) が指す宛先の行。
    宛先はゲートと同じくモード内で :func:`resolve_corrected_turn` で解く。
    session snapshot はこの行だけ問いとツール結果ブロックを残す (f_04 §4.5)。
    """
    by_mode: dict[str, list[dict]] = {}
    for exp in experiences:
        by_mode.setdefault(str(exp.get("mode") or ""), []).append(exp)
    eligible: set[str] = set()
    for rows in by_mode.values():
        for index, exp in enumerate(rows):
            signals = exp.get("signals") or {}
            if signals.get("user_correction"):
                target = resolve_corrected_turn(rows, index)
                if target and target.get("id"):
                    eligible.add(str(target["id"]))
            elif has_failure_evidence(signals) and exp.get("id"):
                eligible.add(str(exp["id"]))
    return eligible


__all__ = [
    "REASON_CORPUS_GROUNDED",
    "REASON_LONG_FORM",
    "REASON_SHORTCUT_TOOL",
    "REASON_TOOL_CONTEXT_MISSING",
    "REASON_TOOL_CONTEXT_TRUNCATED",
    "REASON_TOOL_GROUNDED",
    "case_eligible_ids",
    "grounding_reason",
    "has_failure_evidence",
    "regeneration_mismatch_reason",
    "replay_context",
]
