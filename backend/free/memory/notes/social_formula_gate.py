"""社交の定型だけの発話をノート化から外す判定点 (sleep-time)。

命題を 1 つも運ばない発話 (「了解しました。」「お疲れさまです。」) は、後で
想起しても何も答えられない。にもかかわらずエピソード記憶のノートになると、
注入枠を食い、較正が確定するまでの窓では **その枠を独占する**。

実インシデント (2026-09-16 ライブ監査 F-03): 全リセット直後の 50 ターンで
episodic 37 ノート中 9 件がこの形になり、較正確定までの 13 ターンは
``[参考情報]`` の 4〜5 枠が全部これで埋まった。短い日本語の問いに対する
cosine が 0.47〜0.57 と **高い帯に居座る** ため、関連度の棒を上げる方向では
分離できない (棒を上げると本物の証拠も一緒に落ちる)。書き込み側で止める。

判定は :func:`~backend.free.core.intent_vocab.is_contentless_social_formula`
の **全文被覆** で、字句段だけのカスケードとして不変則 #14 の契約に載せる
(``score`` / ``band`` / ``evidence`` を持ち、``log_decision`` を出す)。事例段は
未装着 — 全文被覆は誤発火が構造的に出ない形なので、``confirm`` の相手を作る
必要が今のところ無い。付けるときは ``exemplar=`` を渡すだけで足りる。
"""

from __future__ import annotations

from typing import Any

from backend.free.core.intent_vocab import is_contentless_social_formula
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    LexicalPredicate,
    register_predicate,
)

PREDICATE_NAME = "contentless_social_formula"

#: 発火ラベル。``candidates`` にも同じ綴りで載せる。
SOCIAL_FORMULA_LABEL = "social_formula"


def _rule(text: str) -> str:
    """全文被覆なら発火ラベル、そうでなければ陰性ラベル。

    ``bool`` ではなく **ラベル文字列** を返す (事例段を足したときに
    ``_same_decision`` が ``"True"`` と ``"social_formula"`` を比べて
    常に不一致になる事故を先に潰しておく — c_17 の警告)。
    """
    return SOCIAL_FORMULA_LABEL if is_contentless_social_formula(text) else NEGATIVE_LABEL


#: プロセス共通の判定点。sleep-time の同期経路から ``evaluate`` で引く。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=LexicalPredicate(
            f"{PREDICATE_NAME}_rule", _rule, evidence="full_coverage",
        ),
        policy="complement",
        candidates=[SOCIAL_FORMULA_LABEL, NEGATIVE_LABEL],
        scope="sleep",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def is_note_worthy(text: str) -> bool:
    """ノートにする価値があるか。社交の定型だけなら False。"""
    return not predicate.evaluate(text or "").fired


__all__ = [
    "PREDICATE_NAME",
    "SOCIAL_FORMULA_LABEL",
    "bind_debug_logger",
    "is_note_worthy",
    "predicate",
]
