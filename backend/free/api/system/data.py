"""エクスポート / インポート API（エディション共通エントリーポイント）

Free: 5カテゴリ一括エクスポート + merge インポート
Pro:  カテゴリ選択式エクスポート + merge/replace/dry-run インポート
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.config import get_config, get_path_resolver
from backend.edition import get_pro_handler, is_pro_or_above
from backend.free.api.system._data_helpers import (
    build_import_response,
    export_failed_error,
    find_invalid_categories,
    import_failed_error,
    invalid_categories_error,
    invalid_mode_error,
    parse_categories,
)
from backend.free.export_import import ExportManager, ImportManager
from backend.log_config import get_logger

logger = get_logger("api.data")

router = APIRouter(prefix="/api/data", tags=["data"])


# ── リクエスト / レスポンスモデル ──

class ExportRequest(BaseModel):
    categories: list[str] | None = None


class ImportCategoryResult(BaseModel):
    category: str
    action: str
    added: int = 0
    updated: int = 0
    skipped: int = 0
    warnings: list[str] = Field(default_factory=list)


class CompatibilityInfo(BaseModel):
    embedding_match: bool
    base_model_match: bool
    reembed_required: bool


class ImportResponse(BaseModel):
    mode: str
    dry_run: bool
    compatibility: CompatibilityInfo
    results: list[ImportCategoryResult]


class ImportStatusResponse(BaseModel):
    status: str = "idle"


# ── ヘルパー ──

def _get_export_manager() -> ExportManager:
    """エディションに応じたエクスポートマネージャを返す"""
    resolver = get_path_resolver()
    cfg = get_config()

    if is_pro_or_above():
        pro_cls = get_pro_handler("export_manager_class")
        if pro_cls is not None:
            return pro_cls(resolver, cfg)

    return ExportManager(resolver, cfg)


def _get_import_manager() -> ImportManager:
    """エディションに応じたインポートマネージャを返す"""
    resolver = get_path_resolver()
    cfg = get_config()

    if is_pro_or_above():
        pro_cls = get_pro_handler("import_manager_class")
        if pro_cls is not None:
            return pro_cls(resolver, cfg)

    return ImportManager(resolver, cfg)


# ── エンドポイント ──

@router.post("/export")
async def export_data(req: ExportRequest | None = None):
    """エクスポートパッケージを生成してダウンロード

    Free: 5カテゴリ一括（categories パラメータ無視）
    Pro:  カテゴリ選択式
    """
    categories = req.categories if req else None
    logger.debug("POST /api/data/export: categories=%s, pro=%s", categories, is_pro_or_above())

    try:
        mgr = _get_export_manager()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "export.evoref-export.zip"

            if is_pro_or_above():
                # Pro: カテゴリ選択式
                zip_path = mgr.export_to_zip(output_path, categories)
            else:
                # Free: 5カテゴリ一括（categories 無視）
                zip_path = mgr.export_to_zip(output_path)

            return FileResponse(
                path=str(zip_path),
                media_type="application/zip",
                filename=zip_path.name,
            )
    except ValueError as e:
        raise export_failed_error(str(e))


@router.post("/import", response_model=ImportResponse)
async def import_data(
    file: UploadFile = File(...),
    mode: str = Form("merge"),
    categories: str | None = Form(None),
    dry_run: bool = Form(False),
    skip_reembed: bool = Form(False),
):
    """エクスポートパッケージをインポート

    Free: merge モード固定（mode/categories/dry_run パラメータ無視）
    Pro:  全パラメータ対応
    """
    logger.debug(
        "POST /api/data/import: filename=%s, mode=%s, categories=%s, dry_run=%s, pro=%s",
        file.filename, mode, categories, dry_run, is_pro_or_above(),
    )

    # Free 版制約の適用
    if not is_pro_or_above():
        mode = "merge"
        categories = None
        dry_run = False

    if mode not in ("merge", "replace"):
        raise invalid_mode_error()

    cat_list = parse_categories(categories)
    if cat_list:
        invalid = find_invalid_categories(cat_list, get_pro_handler("all_categories"))
        if invalid:
            raise invalid_categories_error(invalid)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        mgr = _get_import_manager()

        if is_pro_or_above():
            # Pro: 全パラメータ対応
            result = mgr.import_from_zip(
                zip_path=tmp_path,
                mode=mode,
                categories=cat_list,
                dry_run=dry_run,
                skip_reembed=skip_reembed,
            )
        else:
            # Free: merge モード固定
            result = mgr.import_from_zip(zip_path=tmp_path)

    except (ValueError, FileNotFoundError) as e:
        raise import_failed_error(str(e))
    finally:
        if tmp_path:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return build_import_response(result)


@router.get("/import/status", response_model=ImportStatusResponse)
async def import_status():
    """インポート状態（再埋め込み進捗等）を取得"""
    logger.debug("GET /api/data/import/status")
    return ImportStatusResponse(status="idle")
