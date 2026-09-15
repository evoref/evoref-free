"""RAG 要否の ``skip`` を事例で確認するゲート。

``self_rag_judge`` の規則判定で最も重いのは **誤って skip すること**。skip は
検索を丸ごと飛ばすので、記憶にしかない値を持つ問いがモデルの事前知識だけで
答えられ、そのまま作話になる。

実インシデント (2026-08-23 ライブ監査 セット 1 ターン 91):
「私が来月出張する都市を、確信度を付けて答えてください。」が質問マーカーを
持たずルール 6 (``context_count >= 3``) に落ち、``skip (sufficient context:
37 turns)``。記憶検索が一度も走らないまま「確信度は 100% です。…東京です。」
と作話した (正解は大阪)。同一セッションでルール 6 の skip は 35/94 ターン。

そこで **規則が skip と言ったターンだけ** 事例の近傍に確認を取る
(:data:`~backend.free.core.predicate.CascadePolicy` の ``confirm``)。
事例が反対したら判定を ``uncertain`` へ降ろす — ``uncertain`` は呼出側で
埋め込みリコールに回る安全側の値で、``retrieve`` を強制するわけではない。

**追加のレイテンシはゼロ**。検索パイプラインは ``query_vec`` を引数で受け取って
いるので、その同じベクトルを事例行列に当てるだけで済む (``embed_query`` を
呼ばない)。これが :meth:`~backend.free.core.predicate.ExemplarPredicate.aevaluate`
に ``query_vec`` を通す口を設けた理由。

``fetch`` は確認しない。URL や明示的 fetch 動詞は確定シグナルで、降ろすと
取得依頼が RAG へ流れてしまう。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.core.predicate import (
    DISPUTED_EVIDENCE_PREFIX,
    NEGATIVE_LABEL,
    CascadePredicate,
    Exemplar,
    ExemplarPredicate,
    LexicalPredicate,
    load_exemplars,
)
from backend.log_config import get_logger

logger = get_logger("rag.retrieval_skip_gate")

#: 同梱事例 (tracked)。
DEFAULT_EXEMPLARS_FILE = (
    Path(__file__).resolve().parent / "_defaults" / "retrieval_skip_exemplars.jsonl"
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "retrieval_skip"

#: 「検索しなくてよい」を表す陽性ラベル。
SKIP_LABEL = "skip"

#: 近傍投票数と発火に必要な得票率。**実測で決める** (bench_predicate_gate.py)。
#:
#: 2026-09-15 / bge-m3-q8_0 / 事例 49 件の LOO::
#:
#:     k=3 ratio=0.6  acc(判定)=0.957  棄権  3/49
#:     k=5 ratio=0.6  acc(判定)=0.957  棄権  3/49   ← 採用
#:     k=3 ratio=0.8  acc(判定)=1.000  棄権 20/49
#:     k=5 ratio=0.8  acc(判定)=1.000  棄権 18/49
#:
#: **事例を足すと下がることがある** (2026-09-15 の実測): 自己構成の問いを
#: 5 件足したら「設定」「一覧」が記憶想起側 (「設定のどの項目を変更したか
#: 思い出せますか」「この会話で出た数字をまとめて」) を巻き込み、0.951 →
#: 0.895 に落ちた。語彙の重なる 2 件を外して 0.957。**足したら必ず測り直す**。
#:
#: このゲートは **棄権が危険側** — 棄権すると規則の ``skip`` がそのまま通り、
#: 塞ぎたかった作話の経路が残る。得票率を 0.8 に上げると正解率は 1.0 になるが
#: 棄権が 5 倍になり、**捕まえられる誤った skip がむしろ減る** (23 件の陰性例
#: のうち 0.6 なら 22 件、0.8 なら 17 件しか反対できない)。正解率が 0.95 を
#: 満たす範囲で棄権の少ない方を採る。
DEFAULT_K = 5
DEFAULT_FIRE_RATIO = 0.6


class RetrievalSkipGate:
    """規則の ``skip`` に事例の確認を掛ける。

    確認が取れなければ判定を ``uncertain`` へ降ろすだけで、``retrieve`` を
    強制しない。未 warmup / 棄権 / 失敗のときは規則の判定をそのまま返す
    (誤って開かない・誤って閉じない)。
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
                # ``_same_decision`` が "True" と "skip" を比べて常に不一致に
                # なり、規則が正しいターンまで uncertain へ降りる。
                lambda _text, ctx: (
                    SKIP_LABEL if (ctx and ctx.get("rule_says_skip")) else None
                ),
                takes_ctx=True,
                evidence="rule_skip",
            ),
            exemplar=self._exemplar,
            policy="confirm",
            debug_logger=debug_logger,
            candidates=[SKIP_LABEL, NEGATIVE_LABEL],
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

    async def confirm(
        self,
        query: str,
        *,
        necessity: str,
        query_vec: np.ndarray | None = None,
    ) -> str:
        """規則の判定を返す。``skip`` の確認が取れなければ ``uncertain``。

        Args:
            query: ユーザークエリ。
            necessity: ``RetrievalNecessityJudge`` の 3 値 + ``uncertain``。
            query_vec: 検索パイプラインが既に計算したクエリベクトル。渡せば
                埋め込みの往復は起きない。

        Returns:
            ``necessity`` そのまま、または ``"uncertain"``。
        """
        if necessity != SKIP_LABEL or not self.is_ready():
            return necessity
        verdict = await self._cascade.aevaluate(
            query, {"rule_says_skip": True}, query_vec=query_vec,
        )
        if verdict.band == "abstain" and verdict.evidence.startswith(
            DISPUTED_EVIDENCE_PREFIX,
        ):
            logger.info(
                "Necessity: skip downgraded to uncertain by the exemplar gate "
                "(%s) for query: %s",
                verdict.evidence, query[:50],
            )
            return "uncertain"
        return necessity
