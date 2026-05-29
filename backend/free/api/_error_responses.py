"""API 共通エラー応答ビルダー。

各 API ハンドラに散在していた ``HTTPException(detail={"code", "message",
"i18n_key", "context"})`` の手組みを集約する。``_theme_errors`` /
``_prompt_helpers`` 等の feature 別ビルダーと同じ :class:`ErrorResponse`
パターンの汎用版で、``detail`` dict の構築を 1 箇所に閉じる。

``detail`` に追加フィールド (例: バリデーションの ``errors``) が必要な場合は
:func:`api_error_detail` で base dict を作ってから追記する。
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.error_handlers import ErrorResponse


def api_error_detail(
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> dict[str, Any]:
    """共通 ``ErrorResponse`` を ``detail`` dict 形式で返す純粋関数。"""
    return ErrorResponse(
        code=code,
        message=message,
        i18n_key=i18n_key,
        context=context,
    ).to_dict()


def api_error(
    status_code: int,
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> HTTPException:
    """汎用 ``HTTPException`` ビルダー (``raise api_error(...)`` で使用)。"""
    return HTTPException(
        status_code=status_code,
        detail=api_error_detail(code, message, i18n_key, **context),
    )
