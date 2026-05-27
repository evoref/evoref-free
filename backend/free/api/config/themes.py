"""テーマ管理 API"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from urllib.parse import urlparse

from backend.app_state import AppState, get_app_state
from backend.free.api.config._theme_errors import (
    config_persist_error,
    theme_error,
    theme_not_found_error,
    theme_not_trusted_error,
)
from backend.log_config import get_logger
from backend.free.themes.theme_installer import MAX_ZIP_SIZE
from backend.free.themes.theme_service import ThemeManager, _PREVIEW_MIME

logger = get_logger("api.themes")

router = APIRouter(prefix="/api/themes", tags=["themes"])


class InstallUrlRequest(BaseModel):
    """URL 指定テーマインストールリクエスト"""
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """URL の基本バリデーション"""
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https scheme")
        if not parsed.hostname:
            raise ValueError("URL must have a valid hostname")
        return v


class ActivateRequest(BaseModel):
    """テーマアクティベートリクエスト"""
    theme_id: str
    color_mode: str | None = None


# ── エンドポイント ──


def _get_manager(state: AppState) -> ThemeManager:
    """ThemeManager インスタンスを取得"""
    mgr = state.theme_manager
    if mgr is None:
        raise theme_error(
            503, "E0503", "Theme manager not initialized",
            "api.theme_manager_not_initialized",
        )
    return mgr


@router.get("")
async def list_themes(state: AppState = Depends(get_app_state)):
    """テーマ一覧"""
    logger.debug("GET /api/themes")
    mgr = _get_manager(state)
    themes = mgr.list_themes()
    logger.debug("Listed %d themes", len(themes))
    return {
        "themes": themes,
        "active_theme_id": mgr.active_theme_id,
        "color_mode": mgr.color_mode,
    }


@router.post("/install", status_code=201)
async def install_theme(state: AppState = Depends(get_app_state), file: UploadFile = File(...)):
    """ZIP パッケージからテーマをインストール"""
    logger.debug("POST /api/themes/install: filename=%s", file.filename)
    mgr = _get_manager(state)

    # サイズチェック
    content = await file.read()
    if len(content) > MAX_ZIP_SIZE:
        raise theme_error(
            status_code=400,
            code="E0400",
            message=f"ZIP file too large: {len(content)} bytes (max {MAX_ZIP_SIZE})",
            i18n_key="api.theme_zip_too_large",
            size=str(len(content)),
            max_size=str(MAX_ZIP_SIZE),
        )

    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    logger.debug("Saved upload to temp: %s (%d bytes)", tmp_path, len(content))

    try:
        result = mgr.install(Path(tmp_path))
    except ValueError as e:
        raise theme_error(400, "E0400", str(e), "api.theme_install_invalid")
    except FileExistsError as e:
        raise theme_error(409, "E0409", str(e), "api.theme_already_exists")
    except FileNotFoundError as e:
        raise theme_error(400, "E0400", str(e), "api.theme_css_not_found")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@router.post("/install-url", status_code=201)
async def install_theme_from_url(req: InstallUrlRequest, state: AppState = Depends(get_app_state)):
    """URL からテーマ ZIP をダウンロードしてインストール"""
    logger.debug("POST /api/themes/install-url: url=%s", req.url)
    mgr = _get_manager(state)

    try:
        result = await mgr.install_from_url(req.url)
    except ValueError as e:
        raise theme_error(400, "E0400", str(e), "api.theme_install_invalid")
    except FileExistsError as e:
        raise theme_error(409, "E0409", str(e), "api.theme_already_exists")

    return result


@router.get("/active-cli")
async def get_active_cli_theme(state: AppState = Depends(get_app_state)):
    """アクティブテーマの CLI テーマ設定を返す"""
    logger.debug("GET /api/themes/active-cli")
    mgr = _get_manager(state)
    return mgr.get_active_cli_theme()


@router.post("/activate")
async def activate_theme(req: ActivateRequest, state: AppState = Depends(get_app_state)):
    """テーマをアクティベート"""
    logger.debug("POST /api/themes/activate: theme_id=%s, color_mode=%s", req.theme_id, req.color_mode)
    mgr = _get_manager(state)
    try:
        result = mgr.activate(req.theme_id, req.color_mode)
    except KeyError:
        raise theme_not_found_error(req.theme_id)
    except ValueError as e:
        raise theme_error(400, "E0400", str(e), "api.theme_activate_invalid")
    except RuntimeError as e:
        raise config_persist_error(str(e))

    return result


@router.delete("/{theme_id}", status_code=204)
async def delete_theme(theme_id: str, state: AppState = Depends(get_app_state)):
    """テーマをアンインストール"""
    logger.debug("DELETE /api/themes/%s", theme_id)
    mgr = _get_manager(state)
    try:
        mgr.uninstall(theme_id)
    except KeyError:
        raise theme_not_found_error(theme_id)
    except ValueError as e:
        raise theme_error(400, "E0400", str(e), "api.theme_uninstall_invalid")
    except RuntimeError as e:
        raise config_persist_error(str(e))


@router.post("/{theme_id}/trust")
async def trust_theme(theme_id: str, state: AppState = Depends(get_app_state)):
    """テーマを信頼済みとしてマーク"""
    logger.debug("POST /api/themes/%s/trust", theme_id)
    mgr = _get_manager(state)
    try:
        mgr.trust_theme(theme_id)
    except KeyError:
        raise theme_not_found_error(theme_id)
    except RuntimeError as e:
        raise config_persist_error(str(e))
    return {"theme_id": theme_id, "trusted": True}


@router.delete("/{theme_id}/trust", status_code=200)
async def untrust_theme(theme_id: str, state: AppState = Depends(get_app_state)):
    """テーマの信頼を取り消し"""
    logger.debug("DELETE /api/themes/%s/trust", theme_id)
    mgr = _get_manager(state)
    try:
        mgr.untrust_theme(theme_id)
    except RuntimeError as e:
        raise config_persist_error(str(e))
    return {"theme_id": theme_id, "trusted": False}


@router.get("/{theme_id}/preview")
async def get_preview(theme_id: str, state: AppState = Depends(get_app_state)):
    """テーマプレビュー画像を配信"""
    logger.debug("GET /api/themes/%s/preview", theme_id)
    mgr = _get_manager(state)

    preview_path = mgr.get_preview_path(theme_id)
    if preview_path is None:
        raise theme_error(
            404, "E0404", "Preview image not found", "api.theme_preview_not_found",
        )

    suffix = preview_path.suffix.lower()
    media_type = _PREVIEW_MIME.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(preview_path),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{theme_id}/preview-cli")
async def get_cli_preview(theme_id: str, state: AppState = Depends(get_app_state)):
    """CLI プレビュー画像を配信"""
    logger.debug("GET /api/themes/%s/preview-cli", theme_id)
    mgr = _get_manager(state)

    preview_path = mgr.get_cli_preview_path(theme_id)
    if preview_path is None:
        raise theme_error(
            404, "E0404", "CLI preview image not found",
            "api.theme_cli_preview_not_found",
        )

    suffix = preview_path.suffix.lower()
    media_type = _PREVIEW_MIME.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(preview_path),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{theme_id}/components/{filename}")
async def get_component(theme_id: str, filename: str, state: AppState = Depends(get_app_state)):
    """テーマコンポーネントファイルを配信"""
    logger.debug("GET /api/themes/%s/components/%s", theme_id, filename)
    mgr = _get_manager(state)

    # 信頼チェック
    if not mgr.is_trusted(theme_id):
        raise theme_not_trusted_error()

    comp_path = mgr.get_component_path(theme_id, filename)
    if comp_path is None:
        raise theme_error(
            404, "E0404", "Component not found", "api.theme_component_not_found",
        )

    return FileResponse(
        path=str(comp_path),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/{theme_id}/cli-modules/{filename}")
async def get_cli_module(theme_id: str, filename: str, state: AppState = Depends(get_app_state)):
    """CLI モジュール Python ファイルを配信（信頼チェック付き）"""
    logger.debug("GET /api/themes/%s/cli-modules/%s", theme_id, filename)
    mgr = _get_manager(state)

    # 信頼チェック
    if not mgr.is_trusted(theme_id):
        raise theme_not_trusted_error()

    mod_path = mgr.get_cli_module_path(theme_id, filename)
    if mod_path is None:
        raise theme_error(
            404, "E0404", "CLI module not found", "api.theme_cli_module_not_found",
        )

    return FileResponse(
        path=str(mod_path),
        media_type="text/x-python",
        headers={"Cache-Control": "no-cache"},
    )
