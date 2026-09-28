"""「2つ目に教えてもらったクエリ」がアシスタントの N 番目の回答を指すかの判定点
``session_answer_ordinal`` (c_17 §3.12)。

N 番目の回答は deliberative が「確定事実」(一字一句そのまま示せ) として差し込む
(f_03 §3.5)。誤って発火すると、ユーザー自身の発言の序数 (「私が2番目に言ったこと」)・
苦情 (「2回目に言ったけど」)・末尾からの序数 (「最後から2番目」)・回答の中の項目
(「リストの3番目に挙げた項目」) にまで、別の回答を正答として差し込む (2026-09-28
レビュー H1)。型の判定は :func:`intent_vocab.session_answer_ordinal_reason`
(構造と格で見る)、ここは根拠を ``decision.jsonl`` に残す字句段だけのカスケード
(不変則 #14 (b)(c))。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.free.core.intent_vocab import session_answer_ordinal_reason
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "session_answer_ordinal"

#: アシスタントの N 番目の回答を指している。
ANSWER_LABEL = "answer"

#: 候補ですらない (序数 + アシスタントの動詞が無い) ことを示す根拠。
NO_CANDIDATE_EVIDENCE = "no_match"


class _SessionAnswerRule:
    """字句段 (``Predicate`` プロトコル)。根拠を型ごとに書き分けるため
    ``LexicalPredicate`` ではなく ``Verdict`` を直接返す。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:  # noqa: ARG002 - Predicate プロトコルの引数
        n, evidence = session_answer_ordinal_reason(text or "")
        if n is None:
            return Verdict(
                value=NEGATIVE_LABEL, score=0.0, band="skip",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=ANSWER_LABEL, score=1.0, band="fire",
            evidence=evidence, predicate=self.name, stage="lexical",
        )


_RULE = _SessionAnswerRule()

#: プロセス共通の判定点。deliberative が候補のあるターンで 1 回だけ引く。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="complement",
        candidates=[ANSWER_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def session_answer_verdict(text: str) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する。ターンに 1 回)。"""
    return predicate.evaluate(text or "")


def session_answer_rule(text: str) -> Verdict:
    """字句段だけを **記録せずに** 評価する (純粋関数。投機判定の抑止など 2 度目以降の読み手)。"""
    return _RULE.evaluate(text or "")


__all__ = [
    "ANSWER_LABEL",
    "NO_CANDIDATE_EVIDENCE",
    "PREDICATE_NAME",
    "bind_debug_logger",
    "predicate",
    "session_answer_rule",
    "session_answer_verdict",
]
