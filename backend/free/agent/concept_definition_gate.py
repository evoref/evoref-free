"""実行可能コマンドの規則層を定義・知識の問いで棄権させる判定点 (docs/f_03 §3.1、c_17)。

規則層 (``executable_command_rule``) は「IP アドレス」「ホスト名」のような語から
実コマンドを組む。語は概念を尋ねる問い (「IPアドレスとは？」) にも現れるので、
2026-10-09 ライブ監査 (trace 9d836069a0c9) で定義を聞いただけのターンに
``run_command_readonly`` が撃たれた。語を外すと「この PC の IP は？」が撃てなく
なる (#14: 語形で直さない)。判定は問いの形 (:func:`asks_concept_definition`) で
行い、発火したら規則層は **棄権** して後段 (門・分類器) へ渡す。

- 字句段だけのカスケード。``decision.jsonl`` に根拠が残る。
- 記録は候補のあるターン (規則表がコマンドを組めたターン) で 1 回だけ。
  ``_infer_executable_command`` からは記録しない字句段 (:func:`concept_definition_rule`)
  を読む (``_infer_tool`` などから 1 ターンに何度も呼ばれるため)。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.free.core.intent_vocab import asks_concept_definition
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "concept_definition_query"

#: 対象を一般概念として尋ねている (規則層は棄権する)。
DEFINITION_LABEL = "definition"


class _ConceptDefinitionRule:
    """字句段 (``Predicate`` プロトコル)。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:  # noqa: ARG002
        if asks_concept_definition(text or ""):
            return Verdict(
                value=DEFINITION_LABEL, score=1.0, band="fire",
                evidence="definition_frame", predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=NEGATIVE_LABEL, score=0.0, band="skip",
            evidence="no_definition_frame", predicate=self.name, stage="lexical",
        )


_RULE = _ConceptDefinitionRule()

#: プロセス共通の判定点。規則層がコマンドを組めたターンで 1 回だけ引く。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="complement",
        candidates=[DEFINITION_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def concept_definition_verdict(query: str) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する。ターンに 1 回)。"""
    return predicate.evaluate(query or "")


def concept_definition_rule(query: str) -> Verdict:
    """字句段だけを **記録せずに** 評価する (純粋関数)。"""
    return _RULE.evaluate(query or "")


__all__ = [
    "DEFINITION_LABEL",
    "PREDICATE_NAME",
    "bind_debug_logger",
    "concept_definition_rule",
    "concept_definition_verdict",
    "predicate",
]
