"""`/api/cartridges` ハンドラから抽出した `CartridgeInfo` シリアライザ

`backend/free/api/cartridges.py` の各ハンドラに散在していた `CartridgeInfo`
→ レスポンス dict のマッピングを集約した純粋関数群。同じ dataclass を異なる
フィールドサブセットで dict 化するパターンが install / list / get / rebuild /
unload / delete の 6 ハンドラに散らばっており、フィールド追加時の漏れリスクが
あった。

レイヤー責務:
- `cartridges.py` (API 層)              — HTTP / FastAPI / state 取得 / async 実行
- `_cartridge_serializers.py` (helper)  — `CartridgeInfo` → dict マッピング
                                          + `lora_impact` 付加

すべて引数のみに依存する純粋関数 (`attach_lora_impact` のみ dict を mutate
するが、副作用は引数 dict に閉じる)。FastAPI / app_state には触れない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.free.rag.cartridge_manager import CartridgeInfo


def cartridge_summary_dict(info: CartridgeInfo) -> dict[str, Any]:
    """カートリッジ一覧表示用の薄い dict (`list_cartridges` 用)。

    含むフィールド: id, name, version, description, status, chunks, size_mb
    """
    return {
        "id": info.id,
        "name": info.name,
        "version": info.version,
        "description": info.description,
        "status": info.status,
        "chunks": info.chunks,
        "size_mb": info.size_mb,
    }


def cartridge_detail_dict(info: CartridgeInfo) -> dict[str, Any]:
    """カートリッジ詳細用の dict (`get_cartridge` 用)。

    一覧用フィールドに author / license / tags / language / doc_count /
    installed_at / compatibility / active_version / content_digest /
    embedding_model_id / embedding_dim を加えた完全形。

    ``priority`` は無い — 順位式の ``store_prior`` へ置き換わり、PC 固有の
    上書きは ``corpus/manifest.json`` が持つ (c_16 §4.3 / §8)。
    """
    return {
        "id": info.id,
        "name": info.name,
        "version": info.version,
        "active_version": info.active_version,
        "author": info.author,
        "license": info.license,
        "description": info.description,
        "tags": info.tags,
        "language": info.language,
        "chunks": info.chunks,
        "doc_count": info.doc_count,
        "size_mb": info.size_mb,
        "status": info.status,
        "installed_at": info.installed_at,
        "compatibility": info.compatibility,
        "content_digest": info.content_digest,
        "embedding_model_id": info.embedding_model_id,
        "embedding_dim": info.embedding_dim,
    }


def cartridge_install_response(
    info: CartridgeInfo,
    install_time_sec: float,
) -> dict[str, Any]:
    """`install_cartridge` のレスポンス dict を構築する純粋関数。

    `install_time_sec` は小数 3 桁に丸める。
    """
    return {
        "id": info.id,
        "name": info.name,
        "version": info.version,
        "active_version": info.active_version,
        "chunks": info.chunks,
        "status": info.status,
        "content_digest": info.content_digest,
        "embedding_model_id": info.embedding_model_id,
        "install_time_sec": round(install_time_sec, 3),
    }


def cartridge_rebuild_response(
    info: CartridgeInfo,
    rebuild_time_sec: float,
    embedder_backend_name: str,
) -> dict[str, Any]:
    """`rebuild_cartridge` のレスポンス dict を構築する純粋関数。

    `rebuild_time_sec` は小数 3 桁に丸める。
    """
    return {
        "id": info.id,
        "name": info.name,
        "version": info.version,
        "chunks": info.chunks,
        "status": info.status,
        "size_mb": info.size_mb,
        "embedding_model_id": info.embedding_model_id,
        "rebuild_time_sec": round(rebuild_time_sec, 3),
        "embedder_used": embedder_backend_name,
    }


def attach_lora_impact(
    result: dict[str, Any],
    handler: object | None,
) -> dict[str, Any]:
    """`cartridge_change_handler.last_impact` が存在すれば `result["lora_impact"]`
    として付加する純粋関数 (引数 dict を破壊的に更新)。

    handler が `None` または `last_impact` 属性なし / `None` の場合は
    何もしない。元 handler の挙動を完全に保持する。
    返り値は引数 result そのもの (chain しやすくするため)。
    """
    if handler is None:
        return result
    last_impact = getattr(handler, "last_impact", None)
    if last_impact is not None:
        result["lora_impact"] = last_impact
    return result
