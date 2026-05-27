"""`/api/model` ハンドラから抽出した純粋ヘルパー

`backend/free/api/model.py` の各ハンドラに直書きされていた以下のロジックを
純粋関数群として抽出:
- `ModelDetailResponse` の構築 (client metadata 有無の 2 分岐)
- LoRA アダプタファイルの絶対パス解決
- migration history の `MigrationHistoryItem` マッピング
- `MigrationError` / `MigrationBusyError` → `HTTPException` 変換

レイヤー責務:
- `model.py` (API 層)         — HTTP / FastAPI / ModelMigrator / ModelState 取得
- `_model_helpers.py` (helper) — 純粋構築 / パス解決 / マッピング / エラー変換

`HTTPException` ビルダーは FastAPI に依存するが、`detail` dict 構築自体は
`_theme_errors.theme_error_detail` と同じパターン (純粋関数 + HTTPException
ラッパ) で実装する。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from backend.error_handlers import ErrorResponse
from backend.free.api.schemas import (
    MigrationHistoryItem,
    ModelDetailResponse,
)

if TYPE_CHECKING:
    pass


# ── ModelDetailResponse ビルダー ──────────────────────────────────────


def build_model_detail_response(
    client: object | None,
    llama_cfg: dict[str, Any],
) -> ModelDetailResponse:
    """`/api/model/info` のレスポンス構築。

    `client` が `metadata` を持つ場合は client metadata を優先し、
    なければ `llama_cfg` のみで既定値を返す。`context_size` /
    `gpu_layers` / `flash_attn` は常に `llama_cfg` から取得。
    """
    context_size = int(llama_cfg.get("context_size", 4096))
    gpu_layers = int(llama_cfg.get("gpu_layers", 0))
    flash_attn = bool(llama_cfg.get("flash_attn", False))

    metadata = getattr(client, "metadata", None) if client else None
    if metadata is not None:
        return ModelDetailResponse(
            chat_template=metadata.chat_template,
            has_system_role=metadata.has_system_role,
            context_size=context_size,
            gpu_layers=gpu_layers,
            flash_attn=flash_attn,
        )

    return ModelDetailResponse(
        context_size=context_size,
        gpu_layers=gpu_layers,
        flash_attn=flash_attn,
    )


# ── LoRA パス解決 + Migration History ──────────────────────────────────


def resolve_lora_path(
    local_paths: dict[str, Any],
    project_root: Path,
) -> Path:
    """`config.local_paths.lora_adapter` を絶対パスとして解決する純粋関数。

    既に絶対パスならそのまま `Path` 化、相対パスなら `project_root` 配下に解決。
    既定値は `local/models/adapter.gguf`。
    """
    lora_path = Path(local_paths.get("lora_adapter", "local/models/adapter.gguf"))
    if lora_path.is_absolute():
        return lora_path
    return project_root / lora_path


def map_migration_history_items(
    history: list,
) -> list[MigrationHistoryItem]:
    """`ModelState.migration_history` を `MigrationHistoryItem` リストに
    変換する純粋関数。

    各要素の `lora_archived_to` 属性の真偽で `lora_archived` を決定する。
    """
    return [
        MigrationHistoryItem(
            from_model=h.from_model,
            to_model=h.to_model,
            migrated_at=h.migrated_at,
            lora_archived=bool(h.lora_archived_to),
        )
        for h in history
    ]


# ── HTTPException ビルダー ───────────────────────────────────────────


def _model_error_detail(
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> dict[str, Any]:
    """モデル API 用の `ErrorResponse` を `dict` 形式で返す純粋関数。"""
    return ErrorResponse(
        code=code,
        message=message,
        i18n_key=i18n_key,
        context=context,
    ).to_dict()


def model_error(
    status_code: int,
    code: str,
    message: str,
    i18n_key: str = "",
    **context: Any,
) -> HTTPException:
    """汎用 `HTTPException` ビルダー (`raise model_error(...)` で使用)。"""
    return HTTPException(
        status_code=status_code,
        detail=_model_error_detail(code, message, i18n_key, **context),
    )


def migration_busy_error(message: str) -> HTTPException:
    """409 — モデル移行が既に実行中の場合のエラー。"""
    return model_error(409, "E0409", message)


def migration_error(message: str) -> HTTPException:
    """400 — 通常の移行エラー (`MigrationError`)。"""
    return model_error(400, "E0400", message)


def model_health_check_failed_error() -> HTTPException:
    """503 — `/reload` で llama-server health check に失敗した場合。"""
    return model_error(
        503, "E0503", "llama-server health check failed",
        "api.model_health_check_failed",
    )


def model_reload_failed_error(message: str) -> HTTPException:
    """503 — `/reload` で予期しない例外が発生した場合の汎用エラー。"""
    return model_error(503, "E0503", message)
