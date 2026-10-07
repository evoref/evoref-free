"""学習の事例 (few-shot の手本 / 採用ゲートの評価ケース) から外すターンの述語

few-shot プール (``FewShotPool.add_from_experiences``) と Level 1 の採用ゲート
(``optimizer.prompt_eval.select_prompt_eval_cases``) は、どちらも「system prompt
と問いだけでは再現できないターン」を事例にしない。以前は両者が別々に判定して
いて、ツール実行結果を根拠にしたターン (``tool_grounded``) を few-shot は除き
採用ゲートは除かない、という非対称があった。判定をここに 1 つだけ持つ
(不変則 #14 (a) 同一性)。

どの関数も純関数で、外す理由 (文字列) か ``None`` を返す。
"""

from __future__ import annotations

from backend.free.learning.level0_instant import used_corpus_evidence

#: ツール実行結果を根拠にした応答 (``signals.tool_grounded``)。
REASON_TOOL_GROUNDED = "tool_grounded"
#: 文書チャンク (corpus) を注入した応答 (``gen_config.evidence_ids`` の ``corpus:``)。
REASON_CORPUS_GROUNDED = "corpus_grounded"
#: 長文生成の経路を通った応答 (``signals.long_form_used``)。
REASON_LONG_FORM = "long_form"


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
    """system prompt だけで短く再生成しても同じ入力にならないターンなら理由を返す。

    採用ゲート (f_04 §4.5) の評価ケースから外す判定。:func:`grounding_reason`
    に加え、長文生成の経路 (計画・分割生成・検証) を通ったターンも外す —
    ゲートの再生成は 1 回の短い生成なので、長文の結果 (成否を問わず) は
    再現できない (docs/f_04 §2.5)。
    """
    reason = grounding_reason(exp)
    if reason is not None:
        return reason
    if (exp.get("signals") or {}).get("long_form_used"):
        return REASON_LONG_FORM
    return None


__all__ = [
    "REASON_CORPUS_GROUNDED",
    "REASON_LONG_FORM",
    "REASON_TOOL_GROUNDED",
    "grounding_reason",
    "regeneration_mismatch_reason",
]
