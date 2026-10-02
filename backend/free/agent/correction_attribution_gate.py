"""訂正候補が **誰の誤り** を指しているかの判定点 ``correction_attribution`` (c_17 §3.17)。

字句で立った訂正候補の帰属 (``assistant`` / ``self`` / ``not_correction``) は
:func:`agent.feedback.correction_attribution_reason` が決める。規則で決まらない
対比は、旧値 X を **誰が先に述べたか** (前のユーザー発話の平叙の文か、直前の
アシスタント応答か) で決める。それも無いときは従来どおり ``assistant`` に倒すが、
記録では **棄権** として残す — 「判定した結果 assistant」と「判定できずに
assistant へ倒した」を ``decision.jsonl`` で分ける (不変則 #14 (b)(c))。

2026-10-02 監査 D01#4「やっぱり妻ではなく母と行くことになりました」は既定で
``assistant`` に倒れていた (妻は本人の申告で、アシスタントは復唱しただけ)。
記録するのは学習側の候補判定 (:meth:`FeedbackCollector._detect_correction`) の
1 か所だけで、応答パスの注記などの読み手は記録しない純粋関数を使う。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.free.agent.feedback import (
    ATTRIBUTION_DEFAULT_EVIDENCE,
    correction_attribution_reason,
)
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "correction_attribution"

#: アシスタントの誤りの指摘。
ASSISTANT_LABEL = "assistant"
#: ユーザー自身の前の値の言い直し。
SELF_LABEL = "self"


class _AttributionRule:
    """字句段 (``Predicate`` プロトコル)。根拠を規則ごとに書き分けるため
    ``LexicalPredicate`` ではなく ``Verdict`` を直接返す。

    ``ctx`` の ``prev_user`` (訂正より前のユーザー発話) と ``prev_response``
    (直前のアシスタント応答)、``prev_query`` (その応答が答えたユーザー発話) を
    対比の旧値の出所の判定に使う。
    """

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        ctx = ctx or {}
        label, evidence = correction_attribution_reason(
            text or "",
            prev_user=str(ctx.get("prev_user") or ""),
            prev_response=str(ctx.get("prev_response") or ""),
            prev_query=str(ctx.get("prev_query") or ""),
        )
        if evidence == ATTRIBUTION_DEFAULT_EVIDENCE:
            return Verdict(
                value=None, score=0.0, band="abstain",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        if label == "not_correction":
            return Verdict(
                value=NEGATIVE_LABEL, score=0.0, band="skip",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=label, score=1.0, band="fire",
            evidence=evidence, predicate=self.name, stage="lexical",
        )


_RULE = _AttributionRule()

#: プロセス共通の判定点。学習側が訂正候補の立ったターンで 1 回だけ引く。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="complement",
        candidates=[ASSISTANT_LABEL, SELF_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def correction_attribution_verdict(
    text: str, *, prev_user: str = "", prev_response: str = "", prev_query: str = "",
) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する。ターンに 1 回)。"""
    return predicate.evaluate(
        text or "",
        {"prev_user": prev_user, "prev_response": prev_response, "prev_query": prev_query},
    )


def attribution_target(verdict: Verdict) -> str:
    """判定を帰属 (``assistant`` / ``self`` / ``not_correction``) へ戻す (純粋関数)。

    棄権は従来どおり ``assistant`` に倒す (判別できないものを落とすと、本物の
    指摘を学習から取りこぼす)。
    """
    if verdict.band == "abstain":
        return ASSISTANT_LABEL
    if verdict.value == NEGATIVE_LABEL:
        return "not_correction"
    return str(verdict.value)


__all__ = [
    "ASSISTANT_LABEL",
    "PREDICATE_NAME",
    "SELF_LABEL",
    "attribution_target",
    "bind_debug_logger",
    "correction_attribution_verdict",
    "predicate",
]
