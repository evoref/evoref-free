"""`/api/prompts` ハンドラ用の共通ヘルパー

`backend/free/api/prompts.py` の各ハンドラに散在していた以下のロジックを集約:
- `state.prompt_manager` が `None` の場合の 503 ガード (7 ハンドラで重複)
- `ValueError` (Unknown mode) → 404 変換 (6 ハンドラで重複)
- ロケール検証 / 学習実行中チェックの 400 / 409 エラー
- `PromptMeta + content` → dict マッピング (list / detail で異なる subset)

レイヤー責務:
- `prompts.py` (API 層)         — HTTP / FastAPI / オーケストレーション
- `_prompt_helpers.py` (helper) — ガード / エラービルダー / dict 構築

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
    from backend.free.agent.prompt_manager import PromptMeta, SystemPromptManager


# ── HTTPException ビルダー ───────────────────────────────────────────


def _prompt_error_detail(
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> dict[str, Any]:
    """プロンプト API 用の `ErrorResponse` を `dict` 形式で返す純粋関数。"""
    return ErrorResponse(
        code=code,
        message=message,
        i18n_key=i18n_key,
        context=context,
    ).to_dict()


def prompt_error(
    status_code: int,
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> HTTPException:
    """汎用 `HTTPException` ビルダー (`raise prompt_error(...)` で使用)。"""
    return HTTPException(
        status_code=status_code,
        detail=_prompt_error_detail(code, message, i18n_key, **context),
    )


def prompt_manager_not_initialized_error() -> HTTPException:
    """503 — 7 ハンドラで重複していた `state.prompt_manager is None` ガード用。"""
    return prompt_error(
        503, "E0503", "Prompt manager not initialized",
        "api.prompt_manager_not_initialized",
    )


def unknown_mode_error(mode: str) -> HTTPException:
    """404 — 6 ハンドラで重複していた `ValueError` (`Unknown mode`) 用。"""
    return prompt_error(
        404, "E0404", f"Unknown mode: {mode}",
        "api.prompt_mode_unknown",
        mode=mode,
    )


def prompt_file_not_found_error(message: str) -> HTTPException:
    """404 — `reload` / `rollback` で履歴ファイルが見つからない場合。"""
    return prompt_error(
        404, "E0404", message,
        "api.prompt_file_not_found",
    )


def unsupported_locale_error(locale: str) -> HTTPException:
    """400 — `switch_prompt_locale` のロケール検証エラー。"""
    return prompt_error(
        400, "E0400", f"Unsupported prompt locale: {locale}",
        "api.prompt_locale_unsupported",
        locale=locale,
    )


def learning_running_error() -> HTTPException:
    """409 — 学習実行中はロケール切替不可。"""
    return prompt_error(
        409, "E0409", "Cannot switch locale while learning is running",
        "api.prompt_locale_learning_running",
    )


# ── ガードヘルパー ──────────────────────────────────────────────────


def require_prompt_manager(state: AppState) -> SystemPromptManager:
    """`state.prompt_manager` を取得し、`None` なら 503 を raise する。

    元コードでは 7 ハンドラで個別に if/raise していた処理を 1 関数に集約。
    """
    mgr = state.prompt_manager
    if mgr is None:
        raise prompt_manager_not_initialized_error()
    return mgr


# ── PromptMeta + content → dict マッピング ──────────────────────────


_PREVIEW_LIMIT = 100


def _content_preview(content: str) -> str:
    """`list_prompts` 用の content preview を生成する純粋関数。

    `_PREVIEW_LIMIT` 文字を超える場合は末尾に `...` を付加。
    """
    if len(content) > _PREVIEW_LIMIT:
        return content[:_PREVIEW_LIMIT] + "..."
    return content


def prompt_summary_dict(
    mode: str,
    meta: PromptMeta,
    content: str,
) -> dict[str, Any]:
    """`list_prompts` 用の薄い dict (mode + メタ + content preview)。"""
    return {
        "mode": mode,
        "version": meta.version,
        "source": meta.source,
        "updated_at": meta.updated_at,
        "content_preview": _content_preview(content),
    }


def prompt_detail_dict(
    mode: str,
    meta: PromptMeta,
    content: str,
) -> dict[str, Any]:
    """`get_prompt` 用の dict (フル content 込み)。"""
    return {
        "mode": mode,
        "version": meta.version,
        "source": meta.source,
        "updated_at": meta.updated_at,
        "content": content,
    }
