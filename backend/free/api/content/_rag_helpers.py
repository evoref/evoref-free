"""`/api/rag` ハンドラ用の共通ヘルパー

`backend/free/api/rag.py` の各ハンドラに散在していた以下のロジックを集約:
- `ingest_document` の 6 箇所の inline `HTTPException` 構築
- `get_rag_stats` の sources 集約ロジック (metadata → dict 構築)
- `get_rag_stats` の index size 計算 (file I/O)

レイヤー責務:
- `rag.py` (API 層)         — HTTP / FastAPI / VectorStore / Embedder 取得
- `_rag_helpers.py` (helper) — 検証エラービルダー / 集約 / file I/O ベース計算

検証エラービルダーは `HTTPException` を返却する FastAPI 依存だが、
集約ヘルパー (`aggregate_sources` / `compute_index_size_mb`) は完全な純粋関数
として単体テスト可能。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.error_handlers import ErrorResponse


# ── HTTPException ビルダー ───────────────────────────────────────────


def _rag_error_detail(
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> dict[str, Any]:
    """RAG API 用の `ErrorResponse` を `dict` 形式で返す純粋関数。"""
    return ErrorResponse(
        code=code,
        message=message,
        i18n_key=i18n_key,
        context=context,
    ).to_dict()


def rag_error(
    status_code: int,
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> HTTPException:
    """汎用 `HTTPException` ビルダー。"""
    return HTTPException(
        status_code=status_code,
        detail=_rag_error_detail(code, message, i18n_key, **context),
    )


def rag_filename_required_error() -> HTTPException:
    """400 — アップロードファイル名が空。"""
    return rag_error(
        400, "E0400", "Filename is required", "api.rag_filename_required",
    )


def rag_unsupported_format_error(ext: str) -> HTTPException:
    """400 — サポートされていないファイル拡張子。"""
    return rag_error(
        400, "E0400", f"Unsupported file format: {ext}",
        "api.rag_unsupported_format", ext=ext,
    )


def rag_file_empty_error() -> HTTPException:
    """400 — ファイル内容が空。"""
    return rag_error(
        400, "E0400", "File is empty", "api.rag_file_empty",
    )


def rag_no_chunks_error() -> HTTPException:
    """400 — チャンク分割結果が空。"""
    return rag_error(
        400, "E0400", "No chunks generated from file", "api.rag_no_chunks",
    )


def vector_store_not_initialized_error() -> HTTPException:
    """503 — VectorStore 未初期化。"""
    return rag_error(
        503, "E0503", "Vector store not initialized",
        "api.rag_vector_store_not_initialized",
    )


def embedder_not_initialized_error() -> HTTPException:
    """503 — Embedder 未初期化 (ingest / reindex 共通)。"""
    return rag_error(
        503, "E0503", "Embedder not initialized",
        "api.rag_embedder_not_initialized",
    )


# ── 集約ヘルパー (純粋関数) ────────────────────────────────────────


def aggregate_sources(metadata: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """metadata リストから source ごとの集約 dict を構築する純粋関数。

    各 source キーごとに `{filename, chunks, added_at, category}` を持つ dict
    を返す。同一 source のエントリは `chunks` をインクリメントする。`added_at`
    と `category` は最初に出現した meta の値を保持する。
    """
    sources_map: dict[str, dict[str, Any]] = {}
    for meta in metadata:
        src = meta.get("source", "unknown")
        if src not in sources_map:
            sources_map[src] = {
                "filename": src,
                "chunks": 0,
                "added_at": meta.get("created_at", ""),
                "category": meta.get("category", "document"),
            }
        sources_map[src]["chunks"] += 1
    return sources_map


def compute_index_size_mb(index_path: Path) -> float:
    """インデックスファイルサイズを MB で返す純粋関数。

    存在しない場合は 0.0。`round(3)` で小数 3 桁に丸める (元 handler と同等)。
    """
    if not index_path.exists():
        return 0.0
    try:
        size_bytes = index_path.stat().st_size
    except OSError:
        return 0.0
    return round(size_bytes / (1024 * 1024), 3)
