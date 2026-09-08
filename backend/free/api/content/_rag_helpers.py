"""`/api/rag` ハンドラ用の共通ヘルパー

`backend/free/api/content/rag.py` の各ハンドラに散在していた検証エラー構築を
集約する。

レイヤー責務:
- `rag.py` (API 層)         — HTTP / FastAPI / corpus ストア / Embedder 取得
- `_rag_helpers.py` (helper) — 検証エラービルダー

c_16 で手動投入の書き込み先が corpus パッケージへ移り、統計も corpus から
出すようになったため、旧 `VectorStore` の metadata を集約していた
`aggregate_sources` / `compute_index_size_mb` と
`vector_store_not_initialized_error` は消えた (呼び手が無くなった)。
"""

from __future__ import annotations

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


def corpus_store_not_initialized_error() -> HTTPException:
    """503 — corpus ストア (パッケージの実行時ストア) 未初期化。"""
    return rag_error(
        503, "E0503", "Corpus store not initialized",
        "api.rag_corpus_not_initialized",
    )


def embedder_not_initialized_error() -> HTTPException:
    """503 — Embedder 未初期化 (ingest / reindex 共通)。"""
    return rag_error(
        503, "E0503", "Embedder not initialized",
        "api.rag_embedder_not_initialized",
    )
