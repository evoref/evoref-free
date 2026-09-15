"""few-shot の「文脈依存の問い」棄却を事例で確認するゲート。

``find_content_rejection`` は :func:`~backend.free.learning.corrected_pairs.
refers_to_previous_turn` が真なら手本候補を落とす。この述語は照応語
(``_ANAPHORA_RE``) と想起形 (``_RECALL_FORM_RE``) の 2 本の正規表現で、
**想起形のほうに裸の名詞が混ざっている** — ``覚え`` / ``言いました`` /
``でしたか`` は問いの述語だが、``記憶`` は項にもなる。

実インシデント (2026-09-15 ライブ監査): 「あなたが記憶を書き込むのはいつ
ですか。」は ``記憶`` が ``書き込む`` の目的語で、問いの述語は「いつですか」。
完全に自立した問いなのに「照応・継続」として棄却された。50 ターンで棄却
17 件のうち 1 件がこれで、**手本プールは 50 ターンから 1 件しか残らなかった**
(内容ゲート 44 件棄却 + 品質床 5 件) ので、精度の 1 件は軽くない。

``記憶`` を語彙から抜く方向では直せない — 「記憶にありますか」「記憶して
いますか」は本物の想起形で、同じ語に依存している。語形の網目を細かくする
のは不変則 #12 / #14 が名指しで禁じている方向でもある。判定は文中の語では
なく **その語が問いの述語かどうか** にかかっており、それは事例でしか引けない。

そこで ``confirm`` 方針を採る: **規則が「文脈依存」と言ったときだけ**事例の
近傍に確認を取り、反対されたら棄権へ倒す。棄権は「この理由では落とさない」
の意味で、候補は残りの内容ゲート (逐語抜粋 / 崩れ / 算術矛盾 / 定型文 …) と
LLM の品質床を通常どおり通る — 二の矢があるので、開く側に倒しても
「文脈依存の手本が素通りする」ことにはならない。

背景で走る (Level 1 tick / sleep-time) のでレイテンシ予算は無い。チャット
応答パスからは呼ばない。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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
from backend.free.learning.corrected_pairs import refers_to_previous_turn
from backend.log_config import get_logger

logger = get_logger("learning.context_bound_gate")

#: 同梱事例 (tracked)。
DEFAULT_EXEMPLARS_FILE = (
    Path(__file__).resolve().parent / "_defaults" / "context_bound_exemplars.jsonl"
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "fewshot_context_bound"

#: 「直前ターンを前提にした問い」を表す陽性ラベル。
CONTEXT_BOUND_LABEL = "context_bound"

#: 近傍投票数と発火に必要な得票率。**実測で決める**
#: (``scripts/bench/predicate_gate/bench_predicate_gate.py``)。
#:
#: 2026-09-15 / bge-m3-q8_0 / 事例 64 件 (陽性 35 / 陰性 29) の LOO::
#:
#:     k=3 ratio=0.6  acc(判定)=0.934  棄権  3/64
#:     k=5 ratio=0.6  acc(判定)=0.934  棄権  3/64
#:     k=5 ratio=0.8  acc(判定)=0.982  棄権 10/64   ← 採用
#:     k=7 ratio=0.6  acc(判定)=0.982  棄権  9/64
#:     k=7 ratio=0.8  acc(判定)=1.000  棄権 15/64
#:
#: :mod:`~backend.free.rag.retrieval_skip_gate` と違い、このゲートは
#: **棄権が安全側** — 棄権すれば規則の棄却がそのまま通り、従来の挙動に戻る
#: だけで、文脈依存の手本が流れ込むことはない。したがって「棄権の少なさ」
#: ではなく **危険側の誤りの数** で選ぶ。危険側は ``context_bound`` を
#: ``none`` と読む向き (規則の棄却を誤って覆す) で、0.8 ではこれが **0 件**
#: — 残る 1 件は ``none`` を ``context_bound`` と読む安全側 (覆せないだけ)。
#: 0.6 は誤り 4 件でこの安全性が崩れるので採らない。
#:
#: 事例を足したら ``scripts/bench/predicate_gate/bench_predicate_gate.py
#: --gate fewshot_context_bound --errors`` を回して測り直すこと。初版は
#: 汎用の技術質問を陰性に並べて 0.922 だった — **陰性は「規則が発火する
#: のに自立している」形**でなければ幾何の役に立たない (規則が黙る発話は
#: 事例段に到達しないので、幾何を歪めるだけで一度も投票に参加しない)。
#: これは ``TestShippedExemplars::
#: test_every_exemplar_actually_reaches_the_exemplar_stage`` が固定する。
DEFAULT_K = 5
DEFAULT_FIRE_RATIO = 0.8


class ContextBoundGate:
    """規則の「文脈依存」棄却に事例の確認を掛ける。

    未 warmup / 棄権 / 失敗のときは規則の判定をそのまま通す (誤って開かない)。
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
                # 事例段と **同じラベル語彙** を返す。bool を返すと
                # ``_same_decision`` が "True" と "context_bound" を比べて
                # 常に不一致になり、規則が正しい問いまで棄権へ降りる。
                lambda text: (
                    CONTEXT_BOUND_LABEL if refers_to_previous_turn(text) else None
                ),
                evidence="rule_context_bound",
            ),
            exemplar=self._exemplar,
            policy="confirm",
            debug_logger=debug_logger,
            candidates=[CONTEXT_BOUND_LABEL, NEGATIVE_LABEL],
            scope="cycle",
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

    async def clear_queries(self, queries: Iterable[str]) -> set[str]:
        """規則の棄却を覆せる問いの集合を返す。

        規則が「文脈依存」と言った問いだけを事例に掛け、**反対された
        (棄権へ倒れた) もの** を返す。呼出側はこの集合に入る問いについてだけ
        「文脈依存」を棄却理由にしない。

        未 warmup なら空集合 — 規則の挙動そのまま。
        """
        if not self.is_ready():
            return set()
        cleared: set[str] = set()
        seen: set[str] = set()
        for query in queries:
            q = (query or "").strip()
            if not q or q in seen:
                continue
            seen.add(q)
            if not refers_to_previous_turn(q):
                # 規則が発火しないなら、この理由では落ちていない。
                continue
            try:
                verdict = await self._cascade.aevaluate(q)
            except Exception as e:  # noqa: BLE001 - 判定の失敗で候補を落とさない
                logger.warning("Context-bound confirmation failed: %s", e)
                continue
            if verdict.band == "abstain" and verdict.stage == "exemplar":
                cleared.add(q)
        if cleared:
            logger.info(
                "Context-bound gate cleared %d rule-rejected quer(ies)",
                len(cleared),
            )
        return cleared


__all__ = [
    "CONTEXT_BOUND_LABEL",
    "DEFAULT_EXEMPLARS_FILE",
    "DEFAULT_FIRE_RATIO",
    "DEFAULT_K",
    "PREDICATE_NAME",
    "ContextBoundGate",
]
