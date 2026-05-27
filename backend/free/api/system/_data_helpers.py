"""`/api/data` ハンドラ用の共通ヘルパー

`backend/free/api/data.py` の各ハンドラに散在していた以下のロジックを集約:
- 5 箇所の inline `HTTPException` 構築 (ValueError / FileNotFoundError /
  invalid mode / invalid categories)
- import カテゴリリスト解析 + 検証
- import レスポンスの dict → Pydantic マッピング

レイヤー責務:
- `data.py` (API 層)         — HTTP / FastAPI / オーケストレーション
- `_data_helpers.py` (helper) — エラービルダー / カテゴリ検証 / レスポンス構築

`HTTPException` ビルダーは FastAPI 依存だが、`detail` dict 構築は
`ErrorResponse` を経由して backend 全体のエラースキーマと一貫させる。
カテゴリ検証 / レスポンス構築は純粋関数として単体テスト可能。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from backend.error_handlers import ErrorResponse

if TYPE_CHECKING:
    from backend.free.api.system.data import (
        ImportResponse,
    )


# ── HTTPException ビルダー ───────────────────────────────────────────


def _data_error_detail(
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> dict[str, Any]:
    """データ API 用の `ErrorResponse` を `dict` 形式で返す純粋関数。"""
    return ErrorResponse(
        code=code,
        message=message,
        i18n_key=i18n_key,
        context=context,
    ).to_dict()


def data_error(
    status_code: int,
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> HTTPException:
    """汎用 `HTTPException` ビルダー (`raise data_error(...)` で使用)。"""
    return HTTPException(
        status_code=status_code,
        detail=_data_error_detail(code, message, i18n_key, **context),
    )


def export_failed_error(message: str) -> HTTPException:
    """400 — export 中の `ValueError` 由来エラー。"""
    return data_error(
        400, "E0400", message,
        "api.data_export_failed",
    )


def invalid_mode_error() -> HTTPException:
    """400 — import の mode が `merge` / `replace` 以外。"""
    return data_error(
        400, "E0400", "mode must be 'merge' or 'replace'",
        "api.data_invalid_mode",
    )


def invalid_categories_error(invalid: set[str]) -> HTTPException:
    """400 — import の categories に未知のカテゴリが含まれている。"""
    return data_error(
        400, "E0400", f"Invalid categories: {invalid}",
        "api.data_invalid_categories",
        invalid=str(invalid),
    )


def import_failed_error(message: str) -> HTTPException:
    """400 — import 中の `ValueError` / `FileNotFoundError` 由来エラー。"""
    return data_error(
        400, "E0400", message,
        "api.data_import_failed",
    )


# ── カテゴリ解析 + 検証 ──────────────────────────────────────────────


def parse_categories(
    categories_str: str | None,
) -> list[str] | None:
    """カンマ区切りカテゴリ文字列を `list[str]` に変換する純粋関数。

    `None` / 空文字列なら `None` を返す (元 handler の挙動を保持)。
    各要素は `strip()` で前後空白を除去する。
    """
    if not categories_str:
        return None
    return [c.strip() for c in categories_str.split(",")]


def find_invalid_categories(
    cat_list: list[str],
    all_categories: list[str] | None,
) -> set[str]:
    """`cat_list` 内の `all_categories` に含まれないカテゴリを返す純粋関数。

    `all_categories` が `None` (Pro plugin 未ロード) の場合は空セット
    (検証スキップ、元 handler の挙動を保持)。
    """
    if all_categories is None:
        return set()
    return set(cat_list) - set(all_categories)


# ── ImportResponse ビルダー ───────────────────────────────────────


def build_import_response(result: dict[str, Any]) -> ImportResponse:
    """`ImportManager.import_from_zip` の result dict から
    `ImportResponse` を構築する。

    `compatibility` と `results` のサブ Pydantic モデル変換を吸収する。
    元 handler の field-by-field 構築をそのまま保持。
    """
    # 循環 import 回避: data.py が本モジュールを import するため遅延 import
    from backend.free.api.system.data import (
        CompatibilityInfo,
        ImportCategoryResult,
        ImportResponse,
    )

    return ImportResponse(
        mode=result["mode"],
        dry_run=result["dry_run"],
        compatibility=CompatibilityInfo(**result["compatibility"]),
        results=[ImportCategoryResult(**r) for r in result["results"]],
    )
