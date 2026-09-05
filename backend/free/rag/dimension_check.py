"""埋め込みモデル次元の整合性チェック

Embedder と VectorStore / カートリッジの次元を突合し、不一致があれば
AppState.embedding_dim_mismatch をセットする共通ヘルパー。

次元 (dim) は埋め込みモデル同一性の十分条件ではない: 同 dim の別モデルへ
切り替える (例 Qwen3-Embedding-0.6B 1024 → LFM2.5-Embedding-350M 1024) と
dim だけでは検知できず、既存ベクトルが stale のまま新クエリで検索されて
RAG が静かに壊れる。これを補うため、embed モデル切替時に
:func:`set_embed_reindex_required` で reindex 要求マーカーを立て、本チェックが
モデル同一性ベースの mismatch として扱う (WARNING + stats 表示 + reindex 案内、
``auto_reindex_on_mismatch`` 有効時は自動 reindex)。マーカーは reindex 成功で
:func:`clear_embed_reindex_required` により消える。さらに二重防御として、
store_info の embedding_model と現行 embedder のモデル名も突合する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from backend.io.atomic import atomic_write_text
from backend.log_config import get_logger
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.app_state import AppState

logger = get_logger("rag.dimension_check")

#: embed モデル切替により既存ベクトルが stale になったことを示すマーカー名。
#: vectors_dir 直下に置く (RAG ベクトルの再構築要否フラグ)。
_EMBED_REINDEX_MARKER = ".embed_reindex_required"


def _embed_reindex_marker_path() -> Path | None:
    """reindex 要求マーカーの絶対パス (resolver 未初期化時は ``None``)。"""
    try:
        from backend.config import get_path_resolver

        return get_path_resolver().resolve_local("vectors_dir") / _EMBED_REINDEX_MARKER
    except Exception:
        return None


def set_embed_reindex_required(new_model: str) -> None:
    """embed モデル変更により既存ベクトルが stale になったことを記録する。

    embed component-migrate (model 変更時) から呼ぶ。次回の
    :func:`check_embedding_dim_consistency` で dim が一致していても mismatch
    扱いになり、search ブロック + reindex 案内 (auto_reindex 有効なら自動 reindex)
    が走る。同 dim swap の「検知不発の罠」を防ぐ。
    """
    p = _embed_reindex_marker_path()
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            p,
            json.dumps({"new_model": new_model, "at": utc_now()}, ensure_ascii=False),
        )
        logger.info(
            "Embed reindex marker set (model changed to %s); search results are "
            "unreliable until reindex (run 'evoref reindex' or set "
            "embedding.auto_reindex_on_mismatch)",
            new_model,
        )
    except Exception as exc:
        logger.warning("Failed to write embed reindex marker: %s", exc)


def clear_embed_reindex_required() -> None:
    """reindex 要求マーカーを消す (reindex 成功後に呼ぶ)。"""
    p = _embed_reindex_marker_path()
    if p is None:
        return
    try:
        p.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to clear embed reindex marker: %s", exc)


def embedder_config_mismatch(state: "AppState") -> tuple[str, str] | None:
    """embedder の model_name と config の embedding.model_name の不一致を検出する。

    migrate 成功直後は config は新モデルに切り替わっているが、embed サーバと
    ``state.embedder`` は再起動 / reload まで旧モデルのまま残る。この状態で
    reindex / reembed を実行すると旧モデルのベクトルでストアを再構築し、
    stale マーカーまでクリアしてしまう (順序依存の罠)。実行前の前提チェック
    として双方非空の場合のみ突合し、不一致なら ``(current, expected)`` を返す。
    判定不能 (embedder 未初期化 / config 未ロード / 片方空) は ``None``。
    """
    embedder = state.embedder
    if embedder is None:
        return None
    try:
        current = str(embedder.model_name() or "")
    except Exception:
        return None
    try:
        from backend.config import get_config

        expected = str((get_config().get("embedding") or {}).get("model_name") or "")
    except Exception:
        return None
    if not current or not expected:
        return None
    if current != expected:
        return (current, expected)
    return None


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

    # embed モデル切替マーカー: dim が一致していてもモデルが変われば既存ベクトルは
    # stale。dim のみの比較では検知できない同一性変化をマーカーで補う。
    marker = _embed_reindex_marker_path()
    if marker is not None and marker.exists():
        logger.warning(
            "EMBED MODEL CHANGED since last index (reindex marker present). "
            "Existing vectors are stale; search results are unreliable. "
            "Run 'evoref reindex' to rebuild.",
        )
        mismatch = True

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
                "Search results are unreliable. Run 'evoref reindex' to rebuild.",
                stored_dim, embedder_dim,
            )
            mismatch = True
        else:
            # 同 dim の別モデル切替はマーカーが唯一のガードだったが、マーカーが
            # 誤ってクリアされた後の二重防御としてモデル名も突合する
            # (双方非空時のみ。旧フォーマットの store_info 無しはスキップ)。
            stored_model = str((vs.store_info or {}).get("embedding_model") or "")
            try:
                current_model = str(embedder.model_name() or "")
            except Exception:
                current_model = ""
            if stored_model and current_model and stored_model != current_model:
                logger.warning(
                    "EMBED MODEL MISMATCH: stored vectors were built with '%s' but "
                    "current embedder is '%s'. Search results are unreliable. "
                    "Run 'evoref reindex' to rebuild.",
                    stored_model, current_model,
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
