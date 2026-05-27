"""Lazy Contextual Retrieval サービス

Contextual Retrieval のプレフィックスを retrieval 時に on-demand で生成する
サービス。Sleep-time で一括生成する eager モードと対になる lazy モードを
提供する。

## モード

- ``eager``: Sleep-time Step 5.8 で ``has_context=False`` の全チャンクを
  順次処理し一括でプレフィックスを埋める (従来動作)
- ``lazy``: retrieval 時にヒットした chunk のうち ``has_context=False`` かつ
  ``tokens >= min_chunk_tokens`` のものに対し、hit 回数が
  ``lazy_hit_threshold`` を超えた時点で on-demand 生成する

## 設計方針

- main VectorStore (LTM) のみを対象とする。カートリッジは chunk_id が
  store 間で衝突するため逆引きが難しく、eager モード縛りとする
  (将来的に拡張可)。
- retrieval レスポンスをブロックしないため、呼び出し側は
  ``asyncio.create_task(service.on_retrieval_hits(...))`` で fire-and-forget
  するのが推奨。ただし単体テストは直接 await して ``<500ms/hit`` を検証。
- chunk 長閾値 (``min_chunk_tokens``) 未満、または既にプレフィックス済み
  (``has_context=True``) の chunk は早期 short-circuit する。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.rag.bm25_retriever import BM25Retriever
    from backend.free.rag.contextual_prefix import ContextualPrefixGenerator
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.vector_store import VectorStore

logger = get_logger("rag.lazy_contextual")


class LazyContextualPrefixService:
    """Lazy モードの Contextual Retrieval プレフィックス生成サービス"""

    def __init__(
        self,
        generator: "ContextualPrefixGenerator",
        embedder: "EmbeddingBackend",
        vector_store: "VectorStore",
        *,
        bm25_retriever: "BM25Retriever | None" = None,
        config: dict,
    ) -> None:
        self._generator = generator
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25 = bm25_retriever

        rag_cfg = (config or {}).get("rag", {})
        cp_cfg = rag_cfg.get("contextual_prefix", {}) or {}
        self.enabled: bool = bool(cp_cfg.get("enabled", True))
        self.mode: str = str(cp_cfg.get("mode", "eager"))
        self.min_chunk_tokens: int = int(cp_cfg.get("min_chunk_tokens", 200))
        self.lazy_hit_threshold: int = int(cp_cfg.get("lazy_hit_threshold", 2))
        # 生成処理の直列化 (同時実行は assist model の負荷増大を招くため)
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def is_active(self) -> bool:
        """lazy モードが有効なときのみ True。eager / disabled は False。"""
        return self.enabled and self.mode == "lazy"

    async def on_retrieval_hits(self, chunk_ids: Iterable[str]) -> int:
        """retrieval でヒットした main VectorStore の chunk を評価・生成する。

        ``is_active=False`` の場合は何もしない。対象 chunk は以下を全て
        満たすもののみ:

        - main VectorStore の metadata に存在
        - ``has_context`` が未セット/False
        - ``tokens >= min_chunk_tokens``
        - ``access_count`` が ``lazy_hit_threshold`` 以上に達した (本呼び出しで
          インクリメント後に評価)

        Returns:
            実際にプレフィックスを生成・永続化した件数。
        """
        if not self.is_active:
            return 0

        store = self._vector_store
        if store is None:
            return 0

        targets: list[dict] = []
        for cid in chunk_ids:
            meta = store._find_meta(cid)  # noqa: SLF001 (internal helper)
            if meta is None:
                continue
            if meta.get("has_context"):
                continue
            if int(meta.get("tokens", 0)) < self.min_chunk_tokens:
                continue
            new_count = store.increment_access_count(cid)
            if new_count < self.lazy_hit_threshold:
                continue
            targets.append(meta)

        if not targets:
            return 0

        # 同時実行を直列化し、同一 document の KV キャッシュヒット率を保護
        async with self._lock:
            generated = await self._generate_targets(targets)

        if generated > 0:
            # Lazy 生成では metadata.json + vectors を即時永続化する
            # (次回 retrieval から反映されるように)
            try:
                store.save()
            except Exception as e:  # pragma: no cover - save failure は非致命
                logger.warning("lazy contextual save failed: %s", e)

            # BM25 も contextual text で再構築 (存在する場合のみ)
            if self._bm25 is not None:
                try:
                    chunk_ids_all = [m["id"] for m in store.metadata]
                    texts = [store.get_contextual_text(cid) for cid in chunk_ids_all]
                    self._bm25.build(chunk_ids_all, texts)
                except Exception as e:  # pragma: no cover
                    logger.warning("lazy contextual bm25 rebuild failed: %s", e)

            logger.info(
                "Lazy contextual prefix generated for %d chunks", generated,
            )

        return generated

    async def _generate_targets(self, targets: list[dict]) -> int:
        """対象 metadata リストに対してプレフィックスを 1 件ずつ生成する。

        同一 source の chunk を連続で処理し、llama-server 側の KV キャッシュ
        再利用率を最大化する。
        """
        # source 単位でまとめ、document prefill の KV キャッシュを再利用する
        by_source: dict[str, list[dict]] = {}
        for meta in targets:
            by_source.setdefault(meta.get("source", ""), []).append(meta)

        store = self._vector_store
        generated = 0
        for source, metas in by_source.items():
            source_text = store.load_source_text(source) if source else ""
            if not source_text:
                logger.warning(
                    "lazy contextual: source text missing for %r — skipping %d chunk(s)",
                    source, len(metas),
                )
                continue
            for meta in metas:
                chunk_id = meta["id"]
                chunk_text = store.load_chunk(chunk_id)
                if not chunk_text:
                    continue
                start = time.monotonic()
                prefix = await self._generator.generate_prefix(
                    source_text, chunk_text,
                )
                elapsed_ms = (time.monotonic() - start) * 1000.0
                if not prefix:
                    logger.debug(
                        "lazy contextual: generate_prefix empty for %s (%.1fms)",
                        chunk_id, elapsed_ms,
                    )
                    continue
                try:
                    new_vectors = await self._embedder.embed(
                        [prefix + "\n" + chunk_text], is_query=False,
                    )
                    store.update_context_prefix(
                        chunk_id, prefix, new_vector=new_vectors[0],
                    )
                    generated += 1
                    logger.debug(
                        "lazy contextual: prefix generated for %s in %.1fms",
                        chunk_id, elapsed_ms,
                    )
                except Exception as e:  # pragma: no cover - embed failure
                    logger.warning(
                        "lazy contextual: embed/update failed for %s: %s",
                        chunk_id, e,
                    )
        return generated


__all__ = ["LazyContextualPrefixService"]
