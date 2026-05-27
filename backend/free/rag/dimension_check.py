"""埋め込みモデル次元の整合性チェック

Embedder と VectorStore / カートリッジの次元を突合し、不一致があれば
AppState.embedding_dim_mismatch をセットする共通ヘルパー。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState

logger = get_logger("rag.dimension_check")


def check_embedding_dim_consistency(state: "AppState") -> bool:
    """Embedder と各ストアの次元整合性をチェック

    結果は state.embedding_dim_mismatch / embedding_dim_stored /
    embedding_dim_current にセットされる。

    Returns:
        次元不一致が検出されたら True、整合していれば False
    """
    embedder = state.embedder
    if embedder is None:
        state.embedding_dim_mismatch = False
        state.embedding_dim_stored = None
        state.embedding_dim_current = None
        return False

    try:
        embedder_dim = int(embedder.dim())
    except Exception as exc:
        logger.warning("Failed to read embedder dim: %s", exc)
        state.embedding_dim_mismatch = False
        return False

    state.embedding_dim_current = embedder_dim
    mismatch = False

    # メイン VectorStore
    vs = state.vector_store
    stored_dim: int | None = None
    if vs is not None:
        stored_dim = vs.stored_dim()
        if stored_dim is None:
            # 空ストアなら現在の Embedder 情報で store_info を初期化
            try:
                if vs.ensure_store_info(
                    embedding_model=embedder.model_name(),
                    embedding_backend=embedder.backend_type(),
                    embedding_dim=embedder_dim,
                ):
                    # 既存メタデータがある場合だけ書き戻す
                    if vs.metadata or vs.store_info:
                        vs.save()
            except Exception as exc:
                logger.warning("Failed to bootstrap store_info: %s", exc)
        elif stored_dim != embedder_dim:
            logger.warning(
                "DIMENSION MISMATCH: stored vector dim=%d, current embedder dim=%d. "
                "Search will be blocked. Run 'evoref reindex' to rebuild.",
                stored_dim, embedder_dim,
            )
            mismatch = True

    state.embedding_dim_stored = stored_dim

    # カートリッジマネージャ
    cart_mgr = state.cartridge_manager
    if cart_mgr is not None:
        try:
            mismatched = cart_mgr.check_dimension_consistency(embedder_dim)
            if mismatched:
                mismatch = True
        except Exception as exc:
            logger.warning("Cartridge dimension check failed: %s", exc)

    state.embedding_dim_mismatch = mismatch
    if not mismatch:
        logger.info(
            "Embedding dimension OK: embedder=%d, stored=%s",
            embedder_dim, stored_dim,
        )
    return mismatch
