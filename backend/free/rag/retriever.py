"""ハイブリッド検索（BM25 + Vector RRF 融合 + リランカー統合）"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.free.rag.vector_store import VectorStore
from backend.free.rag.bm25_retriever import BM25Retriever
from backend.free.rag.reranker_null import NullReranker
from backend.free.rag.reranker_skip import (
    evaluate_reranker_skip,
    is_skip_config_active,
)
from backend.log_config import get_logger
from backend.policy_helpers import get_policy_value

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.core.policy_interpreter import PolicyInterpreter

from backend.free.rag.reranker_backend import RerankerBackend

logger = get_logger("rag.retriever")


def _accumulate_normalized(
    scores: dict[str, float],
    results: list[tuple[str, float]],
    weight: float,
) -> None:
    """``results`` を max 正規化し、``weight`` 倍して ``scores`` に加算する。

    max スコアが 0 以下 (全て 0 / 空) の場合は寄与なし。重み付き融合で
    ベクトル側 / BM25 側を同一ロジックで処理するための共通ヘルパ。
    """
    if not results:
        return
    max_s = max(s for _, s in results)
    for cid, s in results:
        norm_s = s / max_s if max_s > 0 else 0
        scores[cid] = scores.get(cid, 0.0) + norm_s * weight


class HybridRetriever:
    """BM25 + Vector のハイブリッド検索（async + リランカー統合）"""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        embedder: EmbeddingBackend,
        reranker: RerankerBackend | None = None,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        candidates_multiplier: int = 3,
        debug_logger: DebugLogger | None = None,
        policy: PolicyInterpreter | None = None,
        config: dict | None = None,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.embedder = embedder
        self.reranker: RerankerBackend = reranker or NullReranker()
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.candidates_multiplier = candidates_multiplier
        self._debug_logger = debug_logger
        self._policy = policy
        # reranker quality-aware skip 用に config 全体を保持する
        # None のときは skip 無効 (従来挙動)。
        self._config = config

    async def search(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = True,
        mode: str = "chat",
    ) -> list[tuple[str, float, str]]:
        """ハイブリッド検索を実行（async）

        クエリの埋め込みを内部で生成し、リランカー統合も含む。
        """
        # ポリシーからパラメータを解決（コンストラクタ値をフォールバック）
        rrf_k = self._resolve("rrf_k", self.rrf_k, mode)
        bm25_weight = self._resolve("bm25_weight", self.bm25_weight, mode)
        vector_weight = self._resolve("vector_weight", self.vector_weight, mode)
        candidates_multiplier = self._resolve(
            "candidates_multiplier", self.candidates_multiplier, mode,
        )

        logger.debug(
            "Hybrid search: query=%r, top_k=%d, hybrid=%s, fusion=%s",
            query[:50], top_k, hybrid, self.fusion_method,
        )

        # クエリ埋め込み
        query_vec = await self.embedder.embed_query(query, mode=mode)

        # リランカー有効時はより多くの候補を取得し、リランカーで最終選別
        has_reranker = self.reranker.is_active
        if has_reranker:
            fetch_k = top_k * (candidates_multiplier + 2)
        else:
            fetch_k = top_k * 3

        vec_results = self.vector_store.search(query_vec, top_k=fetch_k)
        logger.debug("Vector search: %d results (fetch_k=%d)", len(vec_results), fetch_k)

        if not hybrid or self.bm25.count == 0:
            logger.debug(
                "Skipping BM25 (hybrid=%s, bm25_count=%d), returning vector-only",
                hybrid, self.bm25.count,
            )
            n_vec = top_k * candidates_multiplier if has_reranker else top_k
            candidates = vec_results[:n_vec]
        else:
            # BM25 検索
            bm25_results = self.bm25.search(query, top_k=fetch_k)
            logger.debug("BM25 search: %d results", len(bm25_results))

            # 融合
            if self.fusion_method == "rrf":
                fused_ids = self.rrf_fuse(
                    [(cid, score) for cid, score, _ in vec_results],
                    bm25_results,
                    k=rrf_k,
                )
            else:
                fused_ids = self._weighted_fuse(
                    [(cid, score) for cid, score, _ in vec_results],
                    bm25_results,
                    bm25_w=bm25_weight,
                    vector_w=vector_weight,
                )
            logger.debug(
                "Fusion (%s): %d unique results, top scores=[%s]",
                self.fusion_method, len(fused_ids),
                ", ".join(f"{s:.4f}" for _, s in fused_ids[:5]),
            )

            # RRF 融合後の候補数: リランカー有効時は candidates_multiplier 倍
            n_candidates = top_k * candidates_multiplier if has_reranker else top_k

            # チャンクテキストを付与
            chunk_texts = {cid: text for cid, _, text in vec_results}
            candidates = []
            for chunk_id, score in fused_ids[:n_candidates]:
                text = chunk_texts.get(chunk_id, "")
                if not text:
                    text = self.vector_store.load_chunk(chunk_id)
                candidates.append((chunk_id, score, text))

        # リランカー適用（is_active=False の場合はスキップ）
        # reranker.skip の quality-aware 条件を先に評価し
        # top スコア / gap / 候補数の信頼度が十分高い場合は rerank を短絡する。
        if self.reranker.is_active and candidates:
            skip_decision = None
            if is_skip_config_active(self._config):
                skip_decision = evaluate_reranker_skip(candidates, self._config)
            if skip_decision is not None and skip_decision.should_skip:
                logger.debug(
                    "Reranker skipped by quality gate: reason=%s, top_score=%.4f, "
                    "gap=%.4f, candidates=%d",
                    skip_decision.reason, skip_decision.top_score,
                    skip_decision.gap, skip_decision.candidates_count,
                )
                if self._debug_logger is not None:
                    self._debug_logger.log_rerank_skipped(
                        query_preview=query,
                        candidates_count=skip_decision.candidates_count,
                        reason=skip_decision.reason,
                        top_score=skip_decision.top_score,
                        gap=skip_decision.gap,
                        source="hybrid_retriever",
                    )
                results = candidates[:top_k]
            else:
                docs = [text for _, _, text in candidates]
                # mode 別 instruction を reranker へ伝搬
                reranked = await self.reranker.rerank(query, docs, top_k, mode=mode)
                results = [
                    (candidates[idx][0], score, candidates[idx][2])
                    for idx, score in reranked
                ]
                logger.debug("Reranker applied: %d results", len(results))
        else:
            results = candidates[:top_k]

        # デバッグログ: RAG 検索結果記録
        dl = self._debug_logger
        if dl:
            dl.log_rag_result(n=top_k, query=query, chunks=results)

        return results

    @staticmethod
    def rrf_fuse(
        vec_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion"""
        scores: dict[str, float] = {}
        for ranked_list in (vec_results, bm25_results):
            for rank, (chunk_id, _) in enumerate(ranked_list, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _weighted_fuse(
        self,
        vec_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        bm25_w: float | None = None,
        vector_w: float | None = None,
    ) -> list[tuple[str, float]]:
        """重み付き融合 (max 正規化 + 重み付き加算)"""
        _bm25_w = bm25_w if bm25_w is not None else self.bm25_weight
        _vec_w = vector_w if vector_w is not None else self.vector_weight
        scores: dict[str, float] = {}
        _accumulate_normalized(scores, vec_results, _vec_w)
        _accumulate_normalized(scores, bm25_results, _bm25_w)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _resolve(self, key: str, default: int | float, mode: str) -> int | float:
        """ポリシーからパラメータ取得（フォールバック付き）"""
        return get_policy_value(self._policy, "search", key, default, mode=mode)
