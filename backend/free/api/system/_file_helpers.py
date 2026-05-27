"""`/api/files` ハンドラ用の共通ヘルパー

`backend/free/api/files.py` の各ハンドラに散在していた以下のロジックを集約:
- `state.file_manager` が `None` の場合の 503 ガード (5 ハンドラで重複)
- `File not found` 404 (4 ハンドラで重複)
- ファイル不正 (`ValueError` / `FileServiceError`) → 400 変換
- `SessionFile → dict` マッピング (upload / list で異なる field subset)

レイヤー責務:
- `files.py` (API 層)         — HTTP / FastAPI / オーケストレーション
- `_file_helpers.py` (helper) — ガード / エラービルダー / dict 構築

`HTTPException` ビルダーは FastAPI 依存だが、`detail` dict 構築は
`ErrorResponse` を経由して backend 全体のエラースキーマと一貫させる。
dict マッパーは純粋関数として単体テスト可能。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from backend.error_handlers import ErrorResponse

if TYPE_CHECKING:
    from backend.app_state import AppState
    from backend.free.agent.file_manager import SessionFile, SessionFileManager


# ── HTTPException ビルダー ───────────────────────────────────────────


def _file_error_detail(
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> dict[str, Any]:
    """ファイル API 用の `ErrorResponse` を `dict` 形式で返す純粋関数。"""
    return ErrorResponse(
        code=code,
        message=message,
        i18n_key=i18n_key,
        context=context,
    ).to_dict()


def file_error(
    status_code: int,
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> HTTPException:
    """汎用 `HTTPException` ビルダー (`raise file_error(...)` で使用)。"""
    return HTTPException(
        status_code=status_code,
        detail=_file_error_detail(code, message, i18n_key, **context),
    )


def file_manager_not_initialized_error() -> HTTPException:
    """503 — 5 ハンドラで重複していた `state.file_manager is None` ガード用。"""
    return file_error(
        503, "E0503", "File manager not initialized",
        "api.file_manager_not_initialized",
    )


def file_not_found_error(file_id: str = "") -> HTTPException:
    """404 — 4 ハンドラで重複していた "File not found" 用。"""
    return file_error(
        404, "E0404", "File not found",
        "api.file_not_found",
        file_id=file_id,
    )


def file_path_not_found_error(file_path: str) -> HTTPException:
    """404 — `read_and_chunk` のローカルパス未存在用。"""
    return file_error(
        404, "E0404", f"File not found: {file_path}",
        "api.file_path_not_found",
        file_path=file_path,
    )


def file_invalid_error(message: str) -> HTTPException:
    """400 — `ValueError` / `FileServiceError` 由来のクライアントエラー。"""
    return file_error(
        400, "E0400", message,
        "api.file_invalid",
    )


# ── ガードヘルパー ──────────────────────────────────────────────────


def require_file_manager(state: AppState) -> SessionFileManager:
    """`state.file_manager` を取得し、`None` なら 503 を raise する。

    元コードでは 5 ハンドラで個別に if/raise していた処理を 1 関数に集約。
    """
    mgr = state.file_manager
    if mgr is None:
        raise file_manager_not_initialized_error()
    return mgr


# ── SessionFile → dict マッピング ──────────────────────────────────


def file_upload_dict(info: SessionFile) -> dict[str, Any]:
    """`upload_file` 用のレスポンス dict (file_id + filename + size + mime)。"""
    return {
        "file_id": info.file_id,
        "filename": info.filename,
        "size_bytes": info.size_bytes,
        "mime_type": info.mime_type,
    }


def file_summary_dict(info: SessionFile) -> dict[str, Any]:
    """`list_files` 用の薄い dict (file_id + filename + size + uploaded_at)。"""
    return {
        "file_id": info.file_id,
        "filename": info.filename,
        "size_bytes": info.size_bytes,
        "uploaded_at": info.uploaded_at,
    }
