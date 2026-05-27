"""`/api/themes` ハンドラ用の共通エラービルダー

`backend/free/api/themes.py` の各ハンドラに散在していた `HTTPException` 構築
ロジックを集約。元コードでは以下の重複が顕著だった:
- `KeyError → 404 api.theme_not_found` (3 ハンドラ: activate / delete / trust)
- `RuntimeError → 500 api.config_persist_failed` (5 ハンドラ)
- 各種 ValueError → 400 with i18n key

レイヤー責務:
- `themes.py` (API 層)         — HTTP / FastAPI / ThemeManager 取得 / 例外捕捉
- `_theme_errors.py` (helper)  — `HTTPException` 構築 (純粋関数)

`HTTPException` を返すため厳密には FastAPI 依存だが、`detail` dict の構築
ロジックは pure 関数 (`theme_error_detail`) として抽出し単体テスト可能に。
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.error_handlers import ErrorResponse


def theme_error_detail(
    code: str,
    message: str,
    i18n_key: str,
    **context: Any,
) -> dict[str, Any]:
    """テーマ API 用の `ErrorResponse` を `dict` 形式で返す純粋関数。

    `HTTPException` を経由せずに detail dict のみを返すため、単体テストや
    他ハンドラからの再利用が容易。
    """
    return ErrorResponse(
        code=code,
        message=message,
        i18n_key=i18n_key,
        context=context,
    ).to_dict()


def theme_error(
    status_code: int,
    code: str,
    message: str,
    i18n_key: str,
    **context: Any,
) -> HTTPException:
    """汎用 `HTTPException` ビルダー。`raise theme_error(...)` で使用する。"""
    return HTTPException(
        status_code=status_code,
        detail=theme_error_detail(code, message, i18n_key, **context),
    )


# ── 高頻度エラーの named ショートカット ────────────────────────────────


def theme_not_found_error(theme_id: str) -> HTTPException:
    """404 `api.theme_not_found` (activate / delete / trust 等で再利用)。"""
    return theme_error(
        status_code=404,
        code="E0404",
        message=f"Theme '{theme_id}' not found",
        i18n_key="api.theme_not_found",
        theme_id=theme_id,
    )


def config_persist_error(message: str) -> HTTPException:
    """500 `api.config_persist_failed` (activate / delete / trust / untrust 等)。"""
    return theme_error(
        status_code=500,
        code="E0500",
        message=message,
        i18n_key="api.config_persist_failed",
    )


def theme_not_trusted_error() -> HTTPException:
    """403 `api.theme_not_trusted` (component / cli-module ファイル配信用)。"""
    return theme_error(
        status_code=403,
        code="E0403",
        message="Theme is not trusted",
        i18n_key="api.theme_not_trusted",
    )
