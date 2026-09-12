"""候補検索 instruction の offline 実測評価 (EmbedInstructionEval)

EmbedInstructionEvolver (EvorefLearn) が進化中の候補 instruction の検索品質を
実測するための実体。候補 instruction を ``query_template`` の ``{task}`` に
差し込んで記録済みクエリを再埋め込みし、既存のドキュメントベクトル
(``VectorStore``) と検索して top1 cosine スコアを測る。

本クラスは ``backend.free.optimizer.embed_eval.EmbedEvalProtocol`` を**明示継承
しない** (構造的部分型で満たす)。Gen pillar が Learn pillar の Protocol を
import すると依存方向 (Gen→Learn) を逆転させ pillar 境界を侵すため、duck typing
で満たし wire 時に注入する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.exceptions import VectorDimensionMismatchError
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.vector_store import VectorStore

logger = get_logger("rag.embed_instruction_eval")


class EmbedInstructionEval:
    """候補 instruction で query を再埋め込み・再検索し top1 cosine を実測する。"""

    def __init__(
        self,
        embedder: "EmbeddingBackend",
        vector_store: "VectorStore",
        query_template: str = "Instruct: {task}\nQuery: {query}",
        top_k: int = 1,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._query_template = query_template
        self._top_k = max(1, top_k)

    @property
    def can_measure(self) -> bool:
        """候補 instruction を埋め込みに差し込めるか (instruction-aware モデルか)。

        bge-m3 のような非 instruction-aware モデルは ``query_template`` が空で、
        :meth:`score_candidate` は常に ``None`` を返す。その場合 evolver は候補に
        無関係な定数 fitness へ degrade し、10 世代の変異生成 (LLM 呼出) が
        選択圧ゼロで空回りする (2026-09-05 実機: 初期集団 5 件とも 0.6260、
        履歴 10 回すべて before == after)。呼出側はこれで phase ごと skip する。
        """
        return "{task}" in self._query_template and "{query}" in self._query_template

    async def score_candidate(
        self, candidate: str, queries: list[str],
    ) -> float | None:
        """``candidate`` を検索 instruction として ``queries`` の top1 平均を返す。

        実測不能 (ベクトルストア空 / query 無し / テンプレート不正 / 埋め込み
        または検索の例外) は ``None`` を返し degrade させる。
        """
        vs = self._vector_store
        if vs is None or vs.count == 0 or not queries:
            return None
        if "{task}" not in self._query_template or "{query}" not in self._query_template:
            # instruction を差し込めないテンプレート (非 instruction-aware モデル)
            return None

        # キャッシュをバイパスして raw embedder を使う (候補ごとの eval 専用埋め込みで
        # ディスクキャッシュを汚さない)。doc_template は instruction-aware モデルでは
        # 空のため、整形済みテキストを is_query=False で素通り送出できる。
        raw = getattr(self._embedder, "inner", self._embedder)
        scores: list[float] = []
        try:
            for q in queries:
                formatted = self._query_template.format(task=candidate, query=q)
                vecs = await raw.embed([formatted], is_query=False)
                hits = vs.search(vecs[0], top_k=self._top_k)
                if hits:
                    scores.append(float(hits[0][1]))
        except (VectorDimensionMismatchError, Exception) as exc:
            logger.warning("embed instruction eval failed, degrading: %s", exc)
            return None

        if not scores:
            return None
        return sum(scores) / len(scores)


#: 疑似クエリ評価で使う (問い, 正解チャンク) ペアの上限。候補ごとにこの件数を
#: 再埋め込みするので、世代 × 集団のコストを抑える。
PSEUDO_QUERY_EVAL_MAX_PAIRS = 64


class PseudoQueryEmbedEval:
    """疑似クエリ索引 (f_01 §6) を **ラベル付き** の実測評価に使う評価器。

    :class:`EmbedInstructionEval` は記録済みクエリの top1 cosine 平均
    (正解が何かを知らない指標) だったが、疑似クエリは「問い → 正解チャンク」
    の対応を持つので、候補 instruction で問いを再埋め込みして正解チャンクが
    top-k に入る率 (recall@k) を fitness にできる。ペアは疑似クエリ id で
    決定論的にサンプリングし、同じランの候補間で同じ集合を使う。

    ``EmbedEvalProtocol`` の ``queries`` 引数 (記録済みクエリ) は使わない —
    ラベルを持つ疑似クエリの方が弁別力が高い。
    """

    def __init__(
        self,
        cartridge_manager: Any,
        embedder: Any,
        *,
        query_template: str,
        top_k: int = 5,
        max_pairs: int = PSEUDO_QUERY_EVAL_MAX_PAIRS,
    ) -> None:
        self._manager = cartridge_manager
        self._embedder = embedder
        self._query_template = query_template
        self._top_k = max(1, int(top_k))
        self._max_pairs = max(1, int(max_pairs))

    @property
    def can_measure(self) -> bool:
        return "{task}" in self._query_template and "{query}" in self._query_template

    def _pairs(self) -> list[tuple[str, str, str]]:
        """``(package_id, 問い, 正解チャンク id)`` を id 順で最大 ``max_pairs`` 件。"""
        out: list[tuple[str, str, str]] = []
        corpus = getattr(self._manager, "corpus", None)
        if corpus is None:
            return out
        for package_id in corpus.loaded_ids:
            package = corpus.get(package_id)
            index = getattr(package, "pseudo_queries", None)
            if package is None or index is None or len(index) == 0:
                continue
            snapshot = index.store.snapshot
            if snapshot is None:
                continue
            for row in range(len(snapshot)):
                raw = snapshot.raw_at(row) or {}
                target = (raw.get("attrs") or {}).get("target_id")
                text = str(raw.get("text") or "")
                if isinstance(target, str) and target and text:
                    out.append((package_id, text, target))
        out.sort(key=lambda t: (t[0], t[2], t[1]))
        if len(out) <= self._max_pairs:
            return out
        step = len(out) / self._max_pairs
        return [out[int(i * step)] for i in range(self._max_pairs)]

    async def score_candidate(
        self, candidate: str, queries: list[str],  # noqa: ARG002 — Protocol の面
    ) -> float | None:
        """候補 instruction での recall@k (0.0〜1.0)。実測不能は ``None``。"""
        if not self.can_measure:
            return None
        pairs = self._pairs()
        if not pairs:
            return None
        corpus = self._manager.corpus
        raw = getattr(self._embedder, "inner", self._embedder)
        hits = 0
        try:
            for package_id, question, target in pairs:
                package = corpus.get(package_id)
                if package is None:
                    continue
                formatted = self._query_template.format(task=candidate, query=question)
                vecs = await raw.embed([formatted], is_query=False)
                rows = package.store.search("", vecs[0], top_k=self._top_k)
                snapshot = package.store.snapshot
                found = {
                    snapshot.id_at(row) for row, _c, _s in rows
                } if snapshot is not None else set()
                if target in found:
                    hits += 1
        except (VectorDimensionMismatchError, Exception) as exc:  # noqa: BLE001
            logger.warning("pseudo-query embed eval failed, degrading: %s", exc)
            return None
        return hits / len(pairs)
