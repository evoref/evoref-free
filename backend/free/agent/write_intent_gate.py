"""ローカルファイル書込み意図を事例で補完するゲート。

``ComplexityClassifier._is_local_write_intent`` の書込み動詞 (``_WRITE_VERB_RE``)
は ``追記`` / ``書き足`` を持つが ``追加`` を持たない。そのため
「``E:\\tmp\\sales.xlsx`` に 10月 の行を1行だけ追加してください」が
``deliberative`` へ落ち、書込みプランが組まれず「上書き保存しますか？」と
聞き返すだけで終わっていた (2026-09-16 ライブ監査)。

**語形として ``追加`` を足すのは測って却下した。** 同日の実測で、取りこぼし
6 件中 5 件は回収できるが陰性 9 件中 4 件が誤発火した — 「``main.py`` に
型ヒントを追加するべきでしょうか」「``requirements.txt`` に依存を追加すると
どうなりますか」のように、**同じ語で目的語だけが違う**形が巻き添えになる。
CLAUDE.md 不変則 #12 / #14 が禁じる「語形を足す方向」そのものなので、
c_17 の事例ゲートへ載せる。

**方針は ``complement``** — 規則が発火したターンには触れず、**規則が黙った
ターンだけ**事例に聞く。規則の発火を上書きしないので、既存の書込み経路は
1 ターンも挙動が変わらない。

**チャット応答パスのレイテンシ**: 埋め込みを引くのは
:func:`needs_write_intent_hint` が真のターン、すなわち

- chat モードで
- URL を含まず
- how-to (「作り方を教えて」) でもなく
- **宛先 (パス / ``に`` 格のファイル名) は既に立っていて**
- 書込み動詞だけが無い

という狭い集合だけ。ファイル名を 1 つも含まない通常の会話では
文字列判定だけで終わり、埋め込みは 1 度も引かれない。

**棄権は安全側**。棄権すると規則の ``False`` がそのまま通り、従来どおり
``deliberative`` が聞き返す (データは壊れない)。したがって被覆より正解率を
優先した設定にしてある。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Exemplar,
    ExemplarPredicate,
    LexicalPredicate,
    load_exemplars,
)
from backend.log_config import get_logger

logger = get_logger("agent.write_intent_gate")

#: 同梱事例 (tracked)。
DEFAULT_EXEMPLARS_FILE = (
    Path(__file__).resolve().parent / "_defaults" / "write_intent_exemplars.jsonl"
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "local_write_intent"

#: 「このファイルを書き換えてほしい」を表す陽性ラベル。
WRITE_LABEL = "write"

#: 近傍投票数と発火に必要な得票率。**実測で決める** (bench_predicate_gate.py)。
#:
#: 2026-09-16 / bge-m3-q8_0 / 事例 44 件 (陽性 20 / 陰性 24) の LOO::
#:
#:     k=3 ratio=0.6  acc(判定)=0.833  棄権  2/44
#:     k=5 ratio=0.6  acc(判定)=0.929  棄権  2/44
#:     k=5 ratio=0.8  acc(判定)=0.966  棄権 15/44
#:     k=7 ratio=0.6  acc(判定)=1.000  棄権 13/44   ← 採用
#:     k=7 ratio=0.8  acc(判定)=1.000  棄権 26/44
#:
#: **棄権が安い判定点なので正解率を採る。** 棄権すると規則の ``False`` が
#: そのまま通り、従来どおり deliberative が「上書き保存しますか？」と聞き返す
#: — データは壊れず、ユーザーが「はい」と答えれば書ける。誤って発火する方が
#: 重い (読取依頼が書込みプランに乗る) ので、被覆 70% で正解率 1.000 を採る。
#:
#: 陰性を 20 → 24 件に増やしたとき k=5/0.6 は 0.947 → 0.929 に **下がった** が、
#: 同じ事例で k=7/0.6 は 0.936 → 1.000 に上がった。c_17 §8.1 の「足したら
#: 測り直す」は **(k, ratio) の再選択まで含む** — 事例だけ戻すと今の精度は出ない。
DEFAULT_K = 7
DEFAULT_FIRE_RATIO = 0.6


class WriteIntentGate:
    """規則が黙ったターンだけ、書込み意図を事例の近傍で補う。

    未 warmup / 棄権 / 失敗はすべて「補わない」に倒れ、規則の判定
    (``False`` = 書込みではない) がそのまま通る。
    """

    def __init__(
        self,
        embedder: Any,
        *,
        exemplars: Sequence[Exemplar] | None = None,
        k: int = DEFAULT_K,
        fire_ratio: float = DEFAULT_FIRE_RATIO,
        debug_logger: Any = None,
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
                f"{PREDICATE_NAME}_rule",
                # 事例段と **同じラベル語彙** を返すこと。bool を返すと
                # ``_same_decision`` がラベル文字列と比べられなくなる。
                # ここへ来るのは規則が黙ったターンだけなので常に棄権する。
                lambda _text, ctx: (
                    WRITE_LABEL if (ctx and ctx.get("rule_says_write")) else None
                ),
                takes_ctx=True,
                evidence="rule_write_verb",
            ),
            exemplar=self._exemplar,
            policy="complement",
            debug_logger=debug_logger,
            candidates=[WRITE_LABEL, NEGATIVE_LABEL],
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

    async def decide(self, query: str) -> bool | None:
        """書込み意図か。棄権 / 未 warmup は ``None`` (規則の判定を尊重)。

        呼出側は :func:`~backend.free.agent.router.needs_write_intent_hint` が
        真のときだけ呼ぶこと。真でないターンで呼ぶと、規則が既に答えを出して
        いる判定に無駄な埋め込みを 1 回足すだけになる。
        """
        if not self.is_ready():
            return None
        verdict = await self._cascade.aevaluate(query, {"rule_says_write": False})
        if not verdict.decided:
            return None
        return verdict.value == WRITE_LABEL
