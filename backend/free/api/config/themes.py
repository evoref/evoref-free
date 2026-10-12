"""テーマ管理 API"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from urllib.parse import urlparse

from backend.app_state import AppState, get_app_state
from backend.config_normalize import NewerConfigVersionError
from backend.free.api.config._theme_errors import (
    config_newer_version_error,
    config_persist_error,
    theme_error,
    theme_not_found_error,
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


#: アップロードを一時ファイルへ写すときの 1 回の読み取り量
_UPLOAD_CHUNK = 1024 * 1024


def _zip_too_large(size: int) -> HTTPException:
    return theme_error(
        status_code=400,
        code="E0400",
        message=f"ZIP file too large: {size} bytes (max {MAX_ZIP_SIZE})",
        i18n_key="api.theme_zip_too_large",
        size=str(size),
        max_size=str(MAX_ZIP_SIZE),
    )


@router.post("/install", status_code=201)
async def install_theme(state: AppState = Depends(get_app_state), file: UploadFile = File(...)):
    """ZIP パッケージからテーマをインストール"""
    logger.debug("POST /api/themes/install: filename=%s", file.filename)
    mgr = _get_manager(state)

    # サイズチェック: 申告サイズで先に断り、読みながらも上限で止める (全体をメモリに載せない)
    declared = getattr(file, "size", None)
    if declared is not None and declared > MAX_ZIP_SIZE:
        raise _zip_too_large(declared)

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = tmp.name
    try:
        with tmp:
            total = 0
            while chunk := await file.read(_UPLOAD_CHUNK):
                total += len(chunk)
                if total > MAX_ZIP_SIZE:
                    raise _zip_too_large(total)
                tmp.write(chunk)
        logger.debug("Saved upload to temp: %s (%d bytes)", tmp_path, total)
        result = mgr.install(Path(tmp_path))
    except NewerConfigVersionError as e:
        raise config_newer_version_error(e.version)
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
    except NewerConfigVersionError as e:
        raise config_newer_version_error(e.version)
    except ValueError as e:
        raise theme_error(400, "E0400", str(e), "api.theme_install_invalid")
    except FileExistsError as e:
        raise theme_error(409, "E0409", str(e), "api.theme_already_exists")

    return result


@router.post("/activate")
async def activate_theme(req: ActivateRequest, state: AppState = Depends(get_app_state)):
    """テーマをアクティベート"""
    logger.debug("POST /api/themes/activate: theme_id=%s, color_mode=%s", req.theme_id, req.color_mode)
    mgr = _get_manager(state)
    try:
        result = mgr.activate(req.theme_id, req.color_mode)
    except KeyError:
        raise theme_not_found_error(req.theme_id)
    except NewerConfigVersionError as e:
        raise config_newer_version_error(e.version)
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
    except NewerConfigVersionError as e:
        raise config_newer_version_error(e.version)
    except ValueError as e:
        raise theme_error(400, "E0400", str(e), "api.theme_uninstall_invalid")
    except RuntimeError as e:
        raise config_persist_error(str(e))


#: プレビューはテーマ作者の任意ファイル。SVG を直接開かれてもスクリプトを動かさず、
#: 宣言と違う型として解釈させない
_PREVIEW_HEADERS = {
    "Cache-Control": "public, max-age=3600",
    "Content-Security-Policy": "sandbox; default-src 'none'",
    "X-Content-Type-Options": "nosniff",
}


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
        headers=_PREVIEW_HEADERS,
    )
