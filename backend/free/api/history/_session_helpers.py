"""`/api/sessions` ハンドラ用の共通ヘルパー

`backend/free/api/sessions.py` の各ハンドラに散在していた以下のロジックを集約:
- `register` 時の重複セッション 409 raise
- `SessionInfo` → `SessionInfoResponse` マッピング
- `SessionListResponse` 構築

レイヤー責務:
- `sessions.py` (API 層)         — HTTP / FastAPI / state 取得
- `_session_helpers.py` (helper) — エラービルダー / dict マッピング

`HTTPException` ビルダーは FastAPI 依存だが、`detail` dict 構築は
`ErrorResponse` を経由して backend 全体のエラースキーマと一貫させる。
マッパーは純粋関数として単体テスト可能。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from backend.error_handlers import ErrorResponse
from backend.free.api.schemas import (
    SessionInfoResponse,
    SessionListResponse,
)

if TYPE_CHECKING:
    from backend.app_state import SessionInfo


# ── HTTPException ビルダー ───────────────────────────────────────────


def _session_error_detail(
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> dict[str, Any]:
    """セッション API 用の `ErrorResponse` を `dict` 形式で返す純粋関数。"""
    return ErrorResponse(
        code=code,
        message=message,
        i18n_key=i18n_key,
        context=context,
    ).to_dict()


def session_error(
    status_code: int,
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> HTTPException:
    """汎用 `HTTPException` ビルダー。"""
    return HTTPException(
        status_code=status_code,
        detail=_session_error_detail(code, message, i18n_key, **context),
    )


def session_already_registered_error(session_id: str) -> HTTPException:
    """409 — `register` で既に同じ `session_id` が登録されている場合。"""
    return session_error(
        409, "E0409",
        f"Session ID already registered: {session_id}",
        "api.session_already_registered",
        session_id=session_id,
    )


# ── SessionInfo → Pydantic マッピング ──────────────────────────────


def session_info_to_response(info: SessionInfo) -> SessionInfoResponse:
    """`SessionInfo` を `SessionInfoResponse` に変換する純粋関数。"""
    return SessionInfoResponse(
        session_id=info.session_id,
        mode=info.mode,
        client_type=info.client_type,
        registered_at=info.registered_at,
    )


def build_session_list_response(
    sessions: list[SessionInfo],
) -> SessionListResponse:
    """`SessionInfo` リストから `SessionListResponse` を構築する純粋関数。

    `count` フィールドは入力リストの長さを採用 (元 handler の挙動を保持)。
    """
    return SessionListResponse(
        sessions=[session_info_to_response(s) for s in sessions],
        count=len(sessions),
    )
