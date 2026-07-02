"""RAG 管理 API"""

import os
import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from backend.app_state import AppState, get_app_state
from backend.i18n_helper import msg
from backend.free.api.content._rag_helpers import (
    aggregate_sources,
    compute_index_size_mb,
    embedder_not_initialized_error,
    rag_file_empty_error,
    rag_filename_required_error,
    rag_no_chunks_error,
    rag_unsupported_format_error,
    vector_store_not_initialized_error,
)
from backend.free.api.schemas import RagIngestResponse, RagStatsResponse, RagSourceInfo
from backend.config import get_config
from backend.log_config import get_logger
from backend.free.rag.chunker import SemanticChunker
from backend.free.rag.text_extractor import (
    SUPPORTED_DOC_EXTENSIONS,
    extract_text_from_bytes,
    parse_csv_bytes_to_chunks,
)

logger = get_logger("api.rag")

router = APIRouter(prefix="/api/rag", tags=["rag"])

SUPPORTED_EXTENSIONS = SUPPORTED_DOC_EXTENSIONS


@router.post("/ingest", response_model=RagIngestResponse, status_code=201)
async def ingest_document(
    state: AppState = Depends(get_app_state),
    file: UploadFile = File(...),
    category: str = Form("document"),
):
    """ドキュメントを RAG ベクトル DB に追加"""
    if not file.filename:
        raise rag_filename_required_error()

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise rag_unsupported_format_error(ext)

    store = state.vector_store
    if store is None:
        raise vector_store_not_initialized_error()

    start = time.time()

    content = await file.read()

    cfg = get_config()
    rag_cfg = cfg.get("rag", {})

    # CSV は行ごとに1チャンク（ヘッダー付与）で分割（§4.9.9）
    if ext == ".csv":
        chunks = parse_csv_bytes_to_chunks(content)
        source_text = "\n".join(chunks)
    else:
        text = extract_text_from_bytes(content, file.filename)
        if not text.strip():
            raise rag_file_empty_error()
        source_text = text

        chunker = SemanticChunker(
            chunk_size=rag_cfg.get("chunk_size", 512),
            chunk_overlap=rag_cfg.get("chunk_overlap", 128),
            min_chunk=rag_cfg.get("semantic_min_chunk", 64),
            max_chunk=rag_cfg.get("semantic_max_chunk", 512),
            strategy=rag_cfg.get("chunking_strategy", "semantic"),
        )
        chunks = chunker.chunk(text)

    if not chunks:
        raise rag_no_chunks_error()

    # ソーステキスト保存（Contextual Retrieval のプレフィックス生成用）
    store.save_source_text(file.filename, source_text)

    # 実 Embedder で埋め込みを生成
    # 旧実装はダミー zeros + sleep-time 再計算だったが、ストア次元が
    # 設定値と乖離する不整合の温床だったため、ingest 時に実埋め込みを行う
    embedder = state.embedder
    if embedder is None:
        raise embedder_not_initialized_error()

    vectors = await embedder.embed(chunks, is_query=False)

    # 既存ストアの store_info が空なら現在の Embedder 情報で初期化
    store.ensure_store_info(
        embedding_model=embedder.model_name(),
        embedding_backend=embedder.backend_type(),
        embedding_dim=embedder.dim(),
    )

    store.add_vectors(
        vectors, chunks, source=file.filename, category=category,
        embedding_model=embedder.model_name(),
        embedding_backend=embedder.backend_type(),
    )
    store.save()

    tokens_total = sum(max(1, len(c) // 2) for c in chunks)
    elapsed = time.time() - start

    logger.info("Ingested %s: %d chunks in %.2fs", file.filename, len(chunks), elapsed)

    return RagIngestResponse(
        source=file.filename,
        chunks_created=len(chunks),
        tokens_total=tokens_total,
        ingest_time_sec=round(elapsed, 3),
    )


@router.get("/stats", response_model=RagStatsResponse)
async def get_rag_stats(state: AppState = Depends(get_app_state)):
    """RAG ベクトル DB の統計情報"""
    logger.debug("GET /api/rag/stats")
    store = state.vector_store
    cfg = get_config()
    rag_cfg = cfg.get("rag", {})
    embedding_cfg = cfg.get("embedding", {})

    total_chunks = 0
    total_vectors = 0
    index_size_mb = 0.0
    sources_map: dict[str, dict] = {}
    stored_dim: int | None = None
    store_info: dict = {}

    if store is not None:
        total_chunks = store.count
        total_vectors = store.count
        stored_dim = store.stored_dim()
        index_size_mb = compute_index_size_mb(store.index_path)
        sources_map = aggregate_sources(store.metadata)
        store_info = store.store_info

    embedder_dim = (
        state.embedder.dim() if state.embedder is not None
        else int(embedding_cfg.get("dim", 1024))
    )

    return RagStatsResponse(
        total_chunks=total_chunks,
        total_vectors=total_vectors,
        total_sources=len(sources_map),
        index_size_mb=index_size_mb,
        embedding_dim=embedder_dim,
        embedding_dim_stored=stored_dim,
        embedding_dim_mismatch=bool(state.embedding_dim_mismatch),
        chunking_strategy=rag_cfg.get("chunking_strategy", "semantic"),
        hybrid_search=rag_cfg.get("hybrid_search", True),
        fusion_method=rag_cfg.get("fusion_method", "rrf"),
        created_at=store_info.get("created_at"),
        last_reindex_at=store_info.get("last_reindex_at"),
        embedding_model=store_info.get("embedding_model"),
        embedding_backend=store_info.get("embedding_backend"),
        sources=[RagSourceInfo(**s) for s in sources_map.values()],
    )


@router.post("/reindex")
async def reindex_vectors(
    state: AppState = Depends(get_app_state),
    dry_run: bool = False,
    cartridge: str | None = None,
):
    """ベクトルインデックスを現在の Embedder で再構築する

    Query params:
        dry_run: True なら対象件数だけ返して実行しない
        cartridge: カートリッジ ID を指定すると当該カートリッジのみ
    """
    from backend.free.rag.dimension_check import embedder_config_mismatch
    from backend.free.rag.reindex import plan_reindex, run_reindex

    if state.embedder is None:
        raise embedder_not_initialized_error()

    plan = plan_reindex(state, cartridge_id=cartridge)
    if dry_run:
        return {
            "dry_run": True,
            "rag_chunks": plan.rag_chunks,
            "cartridge_chunks": plan.cartridge_chunks,
            "cartridges": plan.cartridges,
            "memory_notes": plan.memory_notes,
        }

    # migrate 直後で embedder が旧モデルのまま実行すると、旧モデルのベクトルで
    # 再構築して stale マーカーまでクリアしてしまう (順序依存の罠)。embed
    # サーバ再起動 + embedder reload が済むまで実行を拒否する。
    stale = embedder_config_mismatch(state)
    if stale is not None:
        raise HTTPException(
            status_code=409,
            detail=msg(
                "error.rag.stale_embedder", current=stale[0], expected=stale[1],
            ),
        )

    result = await run_reindex(state, cartridge_id=cartridge)
    return {
        "dry_run": False,
        "rag_chunks": result.rag_chunks,
        "cartridge_chunks": result.cartridge_chunks,
        "cartridges_rebuilt": result.cartridges_rebuilt,
        "cartridges_failed": result.cartridges_failed,
        "memory_notes_reset": result.memory_notes_reset,
        "elapsed_sec": result.elapsed_sec,
        "embedding_dim_mismatch": state.embedding_dim_mismatch,
    }


@router.post("/calibrate-thresholds")
async def calibrate_thresholds_endpoint(
    state: AppState = Depends(get_app_state),
):
    """再構築済み RAG ベクトルのスコア分布から rag.* 閾値の推奨値を返す。

    埋め込みモデル切替後にスコアスケールへ閾値を合わせるための分布ヒューリスティック。
    正解ラベル不在のため **自動適用はせず提案のみ** を返す (UI でレビュー → 適用)。
    """
    from backend.free.rag.threshold_calibration import calibrate_thresholds

    if state.embedder is None:
        raise embedder_not_initialized_error()
    return calibrate_thresholds(state)
