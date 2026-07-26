"""Step 5.8: Contextual Retrieval プレフィックス生成

``sleep_update.SleepTimeWorker._step5_8_contextual_prefixes`` / ``_generate_prefixes_for_store``
として実装されていた contextual prefix 生成ロジックを独立 module に切り出した。

VectorStore と CartridgeManager 内の ``has_context=false`` チャンクに対し、
アシストモデルでコンテキストプレフィックスを生成し、プレフィックス付き
テキストで再埋め込み → BM25 再構築を行う。

本 module は EvorefMem pillar 内部扱いだが、実質は EvorefGen pillar の
VectorStore / BM25Retriever / EmbeddingBackend / ContextualPrefixGenerator を
ほぼ直接操作するオーケストレーション層。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.rag.bm25_retriever import BM25Retriever
    from backend.free.rag.cartridge_manager import CartridgeManager
    from backend.free.rag.contextual_prefix import ContextualPrefixGenerator
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.vector_store import VectorStore

logger = get_logger("memory.sleep.contextual")


async def generate_prefixes_for_store(
    store: "VectorStore | None",
    generator: "ContextualPrefixGenerator",
    embedder: "EmbeddingBackend",
    batch_size: int,
    label: str,
    *,
    main_store: "VectorStore | None" = None,
    bm25_retriever: "BM25Retriever | None" = None,
    is_cancelled: Callable[[], bool] | None = None,
    min_chunk_tokens: int = 0,
) -> int:
    """1 つの VectorStore 内のプレフィックス未生成チャンクを処理する。

    ``store is main_store`` かつ ``bm25_retriever`` が指定されていれば、
    プレフィックス生成後にメイン BM25 を再構築する (カートリッジは
    独自の BM25 インデックスを持たないため再構築しない)。

    ``min_chunk_tokens`` が 1 以上のときは、metadata の ``tokens`` が
    その値未満の chunk をスキップする。短文 chunk は
    プレフィックスの retrieval 寄与が薄いためアシストモデル呼び出しを節約する。
    """
    if store is None:
        return 0

    pending = store.get_chunks_without_context()
    if not pending:
        return 0

    # 短文 chunk をスキップ
    if min_chunk_tokens > 0:
        before = len(pending)
        pending = [
            m for m in pending if int(m.get("tokens", 0)) >= min_chunk_tokens
        ]
        skipped = before - len(pending)
        if skipped > 0:
            logger.debug(
                "Step 5.8 [%s]: skipped %d short chunks (<%d tokens)",
                label, skipped, min_chunk_tokens,
            )

    if not pending:
        return 0

    pending = pending[:batch_size]
    logger.info(
        "Step 5.8 [%s]: processing %d chunks for contextual prefix",
        label, len(pending),
    )

    # 同一 document の chunk を連続処理することで
    # llama-server 側の KV キャッシュ命中率を最大化する (cache_prompt=True)。
    by_source: dict[str, list[dict]] = {}
    for meta in pending:
        src = meta.get("source", "")
        by_source.setdefault(src, []).append(meta)

    generated = 0
    backfilled = 0
    for source, metas in by_source.items():
        if is_cancelled is not None and is_cancelled():
            break

        source_text = store.load_source_text(source)
        if not source_text:
            # memory:<session_id> ソースは設計上 source text を保存しない
            # (long_term.py の absorb_from_short_term 参照) ため恒久的に
            # 見つからない。無限リトライを避けるため has_context=True で
            # 確定させ、以後の Step 5.8 スキャン対象から外す。
            if source.startswith("memory:"):
                for meta in metas:
                    store.mark_has_context(meta["id"])
                backfilled += len(metas)
                logger.info(
                    "Step 5.8 [%s]: %s has no source text by design "
                    "(memory-sourced chunk); marked %d chunk(s) "
                    "has_context=True to exclude from future scans",
                    label, source, len(metas),
                )
            else:
                logger.warning(
                    "Step 5.8 [%s]: source text not found for %s, skipping",
                    label, source,
                )
            continue

        for meta in metas:
            if is_cancelled is not None and is_cancelled():
                break

            chunk_id = meta["id"]
            chunk_text = store.load_chunk(chunk_id)
            if not chunk_text:
                continue

            prefix = await generator.generate_prefix(source_text, chunk_text)
            if not prefix:
                continue

            contextual_text = prefix + "\n" + chunk_text
            new_vectors = await embedder.embed(
                [contextual_text], is_query=False,
            )
            store.update_context_prefix(
                chunk_id, prefix, new_vector=new_vectors[0],
            )
            generated += 1

    if generated > 0 or backfilled > 0:
        store.save()

        if store is main_store and bm25_retriever is not None:
            chunk_ids = [m["id"] for m in store.metadata]
            texts = [store.get_contextual_text(cid) for cid in chunk_ids]
            bm25_retriever.build(chunk_ids, texts)
            logger.info(
                "Step 5.8 [%s]: rebuilt BM25 index with %d contextual texts",
                label, len(chunk_ids),
            )

    return generated


async def generate_contextual_prefixes(
    llm_client: Any,
    *,
    config: dict,
    embedder: "EmbeddingBackend",
    vector_store: "VectorStore | None",
    cartridge_manager: "CartridgeManager | None",
    bm25_retriever: "BM25Retriever | None",
    is_cancelled: Callable[[], bool] | None = None,
) -> int:
    """Step 5.8 本体 — メイン VectorStore + 全カートリッジを順次処理する。

    ``rag.contextual_prefix.enabled=false`` または ``mode=lazy`` の場合は
    no-op で ``0`` を返す。lazy モードでは
    :class:`~backend.free.rag.lazy_contextual.LazyContextualPrefixService` が
    retrieval 時に on-demand でプレフィックスを生成するため、Sleep-time
    での一括生成は行わない

    Args:
        llm_client: プレフィックス生成に使う LLM クライアント
            (AssistModelClient 相当)。
        config: ``rag.contextual_prefix`` セクション (enabled / mode /
            batch_size / min_chunk_tokens 等) を含む設定 dict。
        embedder: 再埋め込みに使う
            :class:`~backend.free.rag.embedding_backend.EmbeddingBackend`。
        vector_store: メイン VectorStore。
        cartridge_manager: 任意の CartridgeManager。
        bm25_retriever: メイン BM25 を再構築するために渡す BM25Retriever。
        is_cancelled: キャンセル判定コールバック。

    Returns:
        生成したプレフィックス総数。
    """
    rag_cfg = (config or {}).get("rag", {})
    cp_cfg = rag_cfg.get("contextual_prefix", {}) or {}
    enabled = bool(cp_cfg.get("enabled", True))
    mode = str(cp_cfg.get("mode", "eager"))
    if not enabled:
        logger.debug("Step 5.8: contextual_prefix.enabled=false, skipping")
        return 0
    # lazy モードでは Sleep-time での一括生成はしない
    # (retrieval 時に LazyContextualPrefixService が on-demand で生成)
    if mode == "lazy":
        logger.debug("Step 5.8: lazy mode — skipping eager batch generation")
        return 0

    from backend.free.rag.contextual_prefix import ContextualPrefixGenerator
    generator = ContextualPrefixGenerator(llm_client, config)
    batch_size = int(cp_cfg.get("batch_size", 10))
    min_chunk_tokens = int(cp_cfg.get("min_chunk_tokens", 200))

    total_generated = 0

    total_generated += await generate_prefixes_for_store(
        vector_store,
        generator,
        embedder,
        batch_size,
        "main",
        main_store=vector_store,
        bm25_retriever=bm25_retriever,
        is_cancelled=is_cancelled,
        min_chunk_tokens=min_chunk_tokens,
    )

    if cartridge_manager is not None:
        for cart_id, cart_store in cartridge_manager.get_loaded_stores().items():
            if is_cancelled is not None and is_cancelled():
                break
            total_generated += await generate_prefixes_for_store(
                cart_store,
                generator,
                embedder,
                batch_size,
                f"cartridge:{cart_id}",
                main_store=vector_store,
                bm25_retriever=bm25_retriever,
                is_cancelled=is_cancelled,
                min_chunk_tokens=min_chunk_tokens,
            )

    if total_generated > 0:
        logger.info("Step 5.8: generated %d contextual prefixes", total_generated)
    else:
        logger.debug("Step 5.8: no chunks need contextual prefixes")

    return total_generated


__all__ = [
    "generate_contextual_prefixes",
    "generate_prefixes_for_store",
]
