"""ベクトルインデックス再構築

埋め込みモデル変更後に呼び出され、保存済みソーステキストから
新しい Embedder で全ベクトルを再生成する。

対象:
- メイン VectorStore（RAG）
- カートリッジ VectorStore
- STM ノートの埋め込み（再埋め込みは sleep-time worker に委ねる：
  ここでは embedding を None にリセットして再計算をトリガするだけ）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.vector_store import VectorStore

logger = get_logger("rag.reindex")


@dataclass
class ReindexPlan:
    """再構築対象のドライランサマリ"""
    rag_chunks: int = 0
    cartridge_chunks: int = 0
    cartridges: list[str] = field(default_factory=list)
    memory_notes: int = 0


@dataclass
class ReindexResult:
    """再構築結果"""
    rag_chunks: int = 0
    cartridge_chunks: int = 0
    cartridges_rebuilt: list[str] = field(default_factory=list)
    memory_notes_reset: int = 0
    elapsed_sec: float = 0.0


def plan_reindex(
    state: "AppState",
    cartridge_id: str | None = None,
) -> ReindexPlan:
    """再構築対象を集計する（実際の再構築は行わない）"""
    plan = ReindexPlan()
    if cartridge_id is None:
        if state.vector_store is not None:
            plan.rag_chunks = state.vector_store.count
        if state.short_term_memory is not None:
            plan.memory_notes = sum(
                1 for n in state.short_term_memory.notes.values()
                if n.embedding is not None
            )
    if state.cartridge_manager is not None:
        for cart_id, info in state.cartridge_manager._registry.items():
            if cartridge_id is not None and cart_id != cartridge_id:
                continue
            plan.cartridges.append(cart_id)
            plan.cartridge_chunks += info.chunks
    return plan


async def run_reindex(
    state: "AppState",
    cartridge_id: str | None = None,
) -> ReindexResult:
    """全ベクトルストアを再構築する

    Args:
        state: AppState
        cartridge_id: 指定があればそのカートリッジのみ再構築。
                      None なら RAG / 全カートリッジ / メモリを再構築する。
    """
    embedder = state.embedder
    if embedder is None:
        raise RuntimeError("Embedder is not initialized")

    t0 = time.monotonic()
    result = ReindexResult()

    if cartridge_id is None:
        # メイン RAG ストアを再構築
        if state.vector_store is not None:
            rag_count = await _reindex_rag_store(state.vector_store, embedder)
            result.rag_chunks = rag_count

        # STM ノートの embedding をクリア（sleep-time worker が再計算）
        if state.short_term_memory is not None:
            reset = 0
            for note in state.short_term_memory.notes.values():
                if note.embedding is not None:
                    note.embedding = None
                    reset += 1
            state.short_term_memory._cache_dirty = True
            result.memory_notes_reset = reset

    # カートリッジ
    if state.cartridge_manager is not None:
        target_ids = (
            [cartridge_id] if cartridge_id is not None
            else list(state.cartridge_manager._registry.keys())
        )
        for cart_id in target_ids:
            try:
                info = await state.cartridge_manager.rebuild(cart_id, embedder)
                result.cartridges_rebuilt.append(cart_id)
                result.cartridge_chunks += info.chunks
            except Exception as exc:
                logger.error("Reindex failed for cartridge %s: %s", cart_id, exc)

    # 状態更新: embed 切替 reindex マーカーを消してから次元不一致フラグを再評価。
    # (フル reindex 時のみ。特定カートリッジのみの reindex では全体の stale 状態は
    # 解消しないためマーカーは残す。)
    try:
        from backend.free.rag.dimension_check import (
            check_embedding_dim_consistency,
            clear_embed_reindex_required,
        )
        if cartridge_id is None:
            clear_embed_reindex_required()
        check_embedding_dim_consistency(state)
    except Exception as exc:
        logger.warning("Post-reindex dimension check failed: %s", exc)

    result.elapsed_sec = round(time.monotonic() - t0, 3)
    logger.info(
        "Reindex complete: rag=%d, cartridge=%d (%d carts), memory_reset=%d, %.2fs",
        result.rag_chunks, result.cartridge_chunks,
        len(result.cartridges_rebuilt), result.memory_notes_reset,
        result.elapsed_sec,
    )
    return result


async def _reindex_rag_store(
    store: "VectorStore",
    embedder: "EmbeddingBackend",
) -> int:
    """メイン RAG VectorStore を再構築する

    既存のチャンクテキスト（chunks/*.txt）から新しい Embedder で再埋め込み
    し、新規 metadata と一緒にディスクに書き戻す。
    """
    if store.count == 0:
        # 空ストア: store_info だけ更新
        store.mark_reindexed(
            embedding_model=embedder.model_name(),
            embedding_backend=embedder.backend_type(),
            embedding_dim=embedder.dim(),
        )
        store.save()
        return 0

    # 既存メタデータと本文を回収
    items: list[tuple[dict, str]] = []
    for meta in store.metadata:
        chunk_id = meta.get("id", "")
        text = store.load_chunk(chunk_id)
        if text:
            items.append((meta, text))

    if not items:
        return 0

    texts = [t for _, t in items]
    new_vecs = await embedder.embed(texts, is_query=False)

    # 既存ベクトルをクリアして新規追加
    from backend.free.rag.vector_store import quantize_int8
    q8, scales = quantize_int8(np.asarray(new_vecs, dtype=np.float32))

    # 直接書き換え（add_vectors はメタデータを新規作成してしまうため使わない）
    store._ensure_writable()
    store.vectors_q8 = q8
    store.scales = scales
    # 既存メタデータの embedding_model/backend を更新
    for meta, _ in items:
        meta["embedding_model"] = embedder.model_name()
        meta["embedding_backend"] = embedder.backend_type()

    store.mark_reindexed(
        embedding_model=embedder.model_name(),
        embedding_backend=embedder.backend_type(),
        embedding_dim=embedder.dim(),
    )
    store.save()
    return len(items)
