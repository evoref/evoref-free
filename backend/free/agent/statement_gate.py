"""申告 (plain_statement) の確認ゲートと、申告ターンへの言い換え確認の注記。

``intent_vocab.is_plain_statement`` は「問い・依頼のマーカーが無く、平叙の文末
(です / ます …) で終わる」を字句で見る。文末の語形でモダリティを判定するので、
**文末が「です」の依頼・希望** を申告として拾う — 「社内勉強会「Git入門」の案内文を
作りたいです。日時は10月15日18時から、場所は会議室Bです。」が申告扱いになり、
静的 system の「申告は復唱せず言い換えて確認する」に従って、案内文を作らずに
言い換え確認だけで終わった (2026-09-26 監査 C08#1)。

**「たい」を依頼マーカーに足すのは却下した** — 「来週大阪に行きたいです」のような
本当の申告まで巻き込む (語形を足す方向、不変則 #12 / #14)。

**方針は ``confirm``** — 字句が発火したターンだけ事例の近傍に確認させ、反対され
たら棄権へ倒す。棄権 / 反対のターンには注記を付けない (静的 system からは申告の
規則を外してあるので、注記が無ければモデルは依頼として普通に応じる)。事例が確認
した / 事例が棄権した / 未 warmup のターンは字句のとおり注記を付ける (従来の挙動)。

**他の消費者は字句値のまま** (ツール判定のゲート・記憶の述べ直し判定など)。
ここは注記の要否だけを決める (docs/c_17 §3.9、f_03 §7.1)。

**チャット応答パスのレイテンシ**: 字句が発火したターンだけ事例段へ進み、検索で
計算済みの ``query_vec`` を渡せば埋め込みの往復は無い (軽量パスは ``query_vec`` が
無いので 1 回引く。埋め込みキャッシュに当たることが多い)。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.core.intent_vocab import is_plain_statement, memorize_request
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Exemplar,
    ExemplarPredicate,
    LexicalPredicate,
    load_exemplars,
)
from backend.i18n_helper import prompt_locale
from backend.log_config import get_logger

logger = get_logger("agent.statement_gate")

#: 同梱事例 (tracked)。
DEFAULT_EXEMPLARS_FILE = (
    Path(__file__).resolve().parent / "_defaults" / "statement_exemplars.jsonl"
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "plain_statement"

#: 質問を含まない申告 (予定・数値・事実・自己紹介) を表す陽性ラベル。
STATEMENT_LABEL = "statement"

#: 近傍投票数と発火に必要な得票率。**実測で決める** (bench_predicate_gate.py)。
#:
#: 2026-09-27 / bge-m3-q8_0 / 事例 55 件 (陽性 31 = 申告 24 + 本人の希望 7 /
#: 陰性 24 = 「〜たいです」の依頼 12 + それ以外の依頼 12) の LOO::
#:
#:     k=3 ratio=0.6  acc(判定)=0.925  棄権  2/55
#:     k=5 ratio=0.6  acc(判定)=0.943  棄権  2/55
#:     k=5 ratio=0.8  acc(判定)=1.000  棄権 11/55   ← 採用
#:     k=7 ratio=0.6  acc(判定)=1.000  棄権 11/55
#:     k=7 ratio=0.8  acc(判定)=1.000  棄権 19/55
#:
#: 棄権は字句のまま (注記あり = 従来の挙動) に倒れるので、誤った反対を 0 にする
#: 設定の中で棄権の少ない k=5 / 0.8 を採る (k=7 / 0.6 と同率。近傍が少ない方が
#: 事例の追加に対して安定する)。
DEFAULT_K = 5
DEFAULT_FIRE_RATIO = 0.8

#: 事例段を使うか。LOO が ``DEFAULT_MIN_LOO_ACCURACY`` (0.95) に届かない間は
#: False にして字句だけで注記を決める (従来の挙動と同じ)。
EXEMPLAR_STAGE_ENABLED = True

#: 申告ターンに付ける注記 (静的 system から外した規則の本文)。
STATEMENT_NOTES: dict[str, str] = {
    "ja": (
        "この発言は質問を含まない申告 (予定・数値・事実) である。復唱ではなく、"
        "受け取った内容を自分の言葉で言い換えて確認すること "
        "(例:「今週の定例会議は火曜日の15時です。」→"
        "「今週の定例会議は火曜の15時ですね。承知しました。」)。"
    ),
    "en": (
        "This message states a fact, number, or schedule without asking a "
        "question. Acknowledge it in your own words instead of restating it "
        "(e.g. \"The weekly meeting is Tuesday at 15:00.\" -> \"Got it - the "
        "weekly meeting is set for Tuesday at 3 PM.\")."
    ),
}


#: 「覚えておいて」型の保存指示に付ける注記。依頼マーカーを持つので字句は発火しないが、
#: 「受け取った内容を言い換えて確認する」は静的 system から外した規則に依存していた
#: (router の ``memorize_request`` 規則、docs/c_17 §3.9)。
MEMORIZE_NOTES: dict[str, str] = {
    "ja": (
        "この発言は内容を覚えておくよう頼む保存指示である。復唱ではなく、"
        "覚える内容を自分の言葉で言い換えて確認すること。"
    ),
    "en": (
        "This message asks you to remember something. Acknowledge what you will "
        "remember in your own words instead of restating the message."
    ),
}


def statement_note() -> str:
    """現在の prompt locale の申告注記。"""
    return STATEMENT_NOTES.get(prompt_locale(), STATEMENT_NOTES["ja"])


def memorize_note() -> str:
    """現在の prompt locale の保存指示の注記。"""
    return MEMORIZE_NOTES.get(prompt_locale(), MEMORIZE_NOTES["ja"])


def _lexical_label(text: str) -> str:
    """字句段: 申告なら陽性ラベル、そうでなければ陰性ラベル (事例段と同じ語彙)。"""
    return STATEMENT_LABEL if is_plain_statement(text) else NEGATIVE_LABEL


class StatementGate:
    """字句の申告判定に事例の確認を掛け、注記の要否を返す。"""

    def __init__(
        self,
        embedder: Any,
        *,
        exemplars: Sequence[Exemplar] | None = None,
        k: int = DEFAULT_K,
        fire_ratio: float = DEFAULT_FIRE_RATIO,
        debug_logger: Any = None,
        exemplar_stage: bool = EXEMPLAR_STAGE_ENABLED,
    ) -> None:
        records = (
            list(exemplars)
            if exemplars is not None
            else load_exemplars(DEFAULT_EXEMPLARS_FILE)
        )
        self._exemplar = ExemplarPredicate(
            PREDICATE_NAME,
            embedder,
            exemplars=records,
            k=k,
            mode="chat",
            fire_ratio=fire_ratio,
        )
        self._cascade = CascadePredicate(
            PREDICATE_NAME,
            lexical=LexicalPredicate(
                f"{PREDICATE_NAME}_rule", _lexical_label,
                evidence="statement_tail",
            ),
            exemplar=self._exemplar if exemplar_stage else None,
            policy="confirm",
            debug_logger=debug_logger,
            candidates=[STATEMENT_LABEL, NEGATIVE_LABEL],
            scope="request",
        )

    def is_ready(self) -> bool:
        return self._exemplar.is_ready()

    def bind_debug_logger(self, debug_logger: Any) -> None:
        self._cascade.bind_debug_logger(debug_logger)

    def reset(self, embedder: Any = None) -> None:
        self._exemplar.reset(embedder)

    async def warmup(self) -> bool:
        return await self._exemplar.warmup()

    def calibration(self) -> dict[str, float | bool | int]:
        return self._exemplar.calibration

    def self_check(self) -> dict[str, object]:
        """warmup 時の LOO 自己診断 (正解率 / 被覆 / 閾値を満たしたか)。"""
        return self._exemplar.self_check

    def leave_one_out(self, *, with_errors: bool = False) -> dict[str, object]:
        return self._exemplar.leave_one_out(with_errors=with_errors)

    async def needs_note(
        self, query: str, *, query_vec: np.ndarray | None = None,
    ) -> bool:
        """申告の言い換え確認の注記を付けるか (判定点 ``plain_statement`` が fire)。

        字句が不発のターンも字句段だけで評価して記録する (判定点は毎ターン
        ``log_decision`` を出す。事例段へは進まない)。
        """
        if not is_plain_statement(query):
            self._cascade.evaluate(query)
            return False
        verdict = await self._cascade.aevaluate(query, query_vec=query_vec)
        return verdict.band == "fire"


async def statement_note_for(
    gate: StatementGate | None, query: str, *, query_vec: np.ndarray | None = None,
) -> str:
    """申告ターンなら注記を、そうでなければ空文字を返す。

    ゲートが無い (埋め込み無しの構成) ときは字句のとおりに決める。ゲートの失敗は
    字句へ縮退する (注記の有無で応答を止めない)。「覚えておいて」型の保存指示には
    保存指示用の注記を付ける。
    """
    if memorize_request(query):
        return memorize_note()
    if gate is None:
        return statement_note() if is_plain_statement(query) else ""
    try:
        fires = await gate.needs_note(query, query_vec=query_vec)
    except Exception as e:  # pragma: no cover - 縮退で吸収する
        logger.warning("Statement gate failed, falling back to the rule: %s", e)
        fires = is_plain_statement(query)
    return statement_note() if fires else ""
