"""層振り分けの shadow 評価 — 挙動は変えず、不一致だけを記録する。

``router._CLASSIFY_RULES`` は 24 ルールの表で、**並び順が優先度**。在庫の中で
最も脆い判定点で、前方ルール (``long_form`` / ``local_write_intent``) の誤爆は
生成経路ごと奪われ、ツール判定にも RAG にも記憶注入にも一度も到達しない。
代替経路は 1 つも無い。

にもかかわらず、ここを事例へ置き換えるのは **今はやらない**:

- ``router`` ドメインは ``policy_evolver.EVOLVABLE_DOMAINS`` から **意図的に
  凍結** されており (``test_router_domain_stays_frozen`` が固定)、自動で動かす
  ことに対する明確な判断が既にある。
- 誤りのコストが非対称で、層を奪う向きの誤爆は 1 件でも重い。

そこで :data:`~backend.free.core.predicate.CascadePolicy` の ``shadow`` を使う。
規則の結果を **必ずそのまま返し**、事例の判定との不一致だけを
``decision.jsonl`` に残す。切り替えるかどうかは、貯まった不一致を人が見てから
決める (The Replay Gap: 旧方針の下で集めたログを素朴に再生すると分布シフトを
無視するので、shadow の同一入力比較が唯一まともに測れる形になる)。

チャット応答パスで走るが、**追加の埋め込み往復は無い**: 層振り分けの後に
検索パイプラインが同じクエリを埋め込むので、そこで計算済みの ``query_vec`` を
渡す。まだ無い経路では ``None`` を渡してよく、その場合ゲートは棄権する。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.core.predicate import (
    CascadePredicate,
    Exemplar,
    ExemplarPredicate,
    LexicalPredicate,
    load_exemplars,
)
from backend.log_config import get_logger

logger = get_logger("agent.layer_shadow")

#: 同梱事例 (tracked)。
DEFAULT_EXEMPLARS_FILE = (
    Path(__file__).resolve().parent / "_defaults" / "layer_exemplars.jsonl"
)

#: 判定点の名前。既存の ``layer_classification`` とは **別名** にする —
#: 同じ名前で 2 つの決定が ``decision.jsonl`` に混ざると、どちらの記録か
#: 分からなくなる (レジストリが名前の一意性を強制する理由と同じ)。
PREDICATE_NAME = "layer_classification_shadow"

#: 層のラベル。``router.classify`` の戻り値と同じ語彙。
LAYERS: tuple[str, ...] = ("reactive", "deliberative", "meta_cognitive")

#: 近傍投票数と発火に必要な得票率。**実測で決める** (bench_predicate_gate.py)。
#:
#: 2026-09-15 / bge-m3-q8_0 / 事例 43 件の LOO::
#:
#:     k=3 ratio=0.6  acc(判定)=0.878  棄権  2/43
#:     k=3 ratio=0.8  acc(判定)=1.000  棄権 28/43   ← 採用 (k=3 の全会一致)
#:     k=5 ratio=0.6  acc(判定)=0.756  棄権  2/43
#:     k=5 ratio=0.8  acc(判定)=0.885  棄権 17/43
#:
#: shadow は挙動を変えないので、観測数 (被覆) より **不一致の信頼度** を採る —
#: 偽の不一致は人のレビュー時間を無駄にするだけで、見逃しても次のターンがある。
#: k=3 の全会一致は意味も明快 ("近傍 3 件が揃って別の層を指した")。
#:
#: このゲートは陰性クラス (``none``) を持たない。層振り分けは 3 値の網羅選択で
#: 「どれでもない」が存在しないため。較正の null 側は **違うラベルの最近傍**
#: (= 最も近い誤答) で定義されているので縮退しない
#: (:func:`~backend.free.core.predicate.calibrate_exemplar_gate`)。
DEFAULT_K = 3
DEFAULT_FIRE_RATIO = 0.8


class LayerClassificationShadow:
    """層振り分けの規則と事例を並走させ、不一致を記録する。

    :meth:`observe` は **常に規則の結果を返す**。返り値を使う側から見れば
    何も変わらない。
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
                lambda _text, ctx: (ctx or {}).get("layer") or None,
                takes_ctx=True,
                evidence="rule",
            ),
            exemplar=self._exemplar,
            policy="shadow",
            debug_logger=debug_logger,
            candidates=list(LAYERS),
            scope="request",
        )
        self._agreements = 0
        self._disagreements = 0

    @property
    def stats(self) -> dict[str, int]:
        """一致 / 不一致の累計 (プロセス内。``/api/status`` 等の観測用)。"""
        return {
            "agreed": self._agreements,
            "disagreed": self._disagreements,
        }

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

    def leave_one_out(self, *, with_errors: bool = False) -> dict[str, object]:
        return self._exemplar.leave_one_out(with_errors=with_errors)

    async def observe(
        self,
        query: str,
        layer: str,
        *,
        query_vec: np.ndarray | None = None,
    ) -> str:
        """規則の判定 ``layer`` を **そのまま返す**。不一致だけ記録する。

        Args:
            query: ユーザークエリ。
            layer: ``ComplexityClassifier.classify`` が返した層。
            query_vec: 計算済みのクエリベクトル。``None`` なら事例段は
                埋め込みを 1 回行う (shadow のためにチャットを待たせたくない
                経路では渡すこと)。

        Returns:
            ``layer`` (常に入力そのまま)。
        """
        if not self.is_ready():
            return layer
        try:
            _chosen, shadow = await self._cascade.aevaluate_pair(
                query, {"layer": layer}, query_vec=query_vec,
            )
        except Exception as e:  # pragma: no cover - 縮退で吸収する
            logger.info("Layer shadow failed: %s", e)
            return layer
        if shadow is not None and shadow.decided:
            if str(shadow.value) == layer:
                self._agreements += 1
            else:
                self._disagreements += 1
                logger.info(
                    "Layer shadow disagreement: rule=%s exemplar=%s (%.2f) for %r",
                    layer, shadow.value, shadow.score, query[:60],
                )
        # ``policy="shadow"`` なので採択されるのは常に規則の結果。
        return layer
