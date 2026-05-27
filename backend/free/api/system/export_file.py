"""ファイル書出し API エンドポイント

POST /api/export/file — コンテンツを指定形式に変換してダウンロード
GET  /api/export/formats — 対応形式一覧
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.export import get_writer_registry
from backend.export.base import ExportError
from backend.export.content_converter import ContentConverter
from backend.log_config import get_logger

logger = get_logger("api.export_file")

router = APIRouter(prefix="/api/export", tags=["export"])

# MIME タイプマッピング
_MIME_TYPES: dict[str, str] = {
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".tsv": "text/tab-separated-values; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".yaml": "application/x-yaml; charset=utf-8",
    ".yml": "application/x-yaml; charset=utf-8",
    ".tex": "application/x-latex; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".epub": "application/epub+zip",
}


class ExportFileRequest(BaseModel):
    """ファイル書出しリクエスト"""
    content: str = ""  # Markdown テキスト
    format: str  # 出力形式（拡張子: ".docx", ".csv" 等）
    data: list[dict] | list[list] | None = None  # 構造化データ（オプション）
    title: str = ""
    metadata: dict = Field(default_factory=dict)
    filename: str = ""  # ダウンロードファイル名（オプション）


class ExportFormatsResponse(BaseModel):
    """対応形式一覧レスポンス"""
    formats: list[str]
    availability: dict[str, bool]


@router.post("/file")
async def export_file(req: ExportFileRequest) -> Response:
    """コンテンツを指定形式に変換してファイルダウンロード"""
    import backend.free.export  # noqa: F401  Writer 登録を確実に

    registry = get_writer_registry()
    ext = req.format if req.format.startswith(".") else f".{req.format}"

    if not registry.is_supported(ext):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "E0400",
                "message": f"Unsupported format: {ext}. Supported: {sorted(registry.supported_extensions())}",
                "i18n_key": "api.export_unsupported_format",
                "context": {"ext": ext, "supported": str(sorted(registry.supported_extensions()))},
            },
        )

    # ExportContent を構築
    if req.data is not None:
        content = ContentConverter.from_data(req.data, title=req.title, **req.metadata)
        # raw_markdown もあれば保持
        if req.content:
            content.raw_markdown = req.content
    elif req.content:
        content = ContentConverter.from_markdown(req.content, title=req.title, **req.metadata)
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "E0400",
                "message": "Either 'content' or 'data' must be provided",
                "i18n_key": "api.export_missing_content",
                "context": {},
            },
        )

    try:
        data = registry.write_to_bytes(content, ext)
    except ExportError as e:
        if e.code == "missing_library":
            raise HTTPException(
                status_code=501,
                detail={
                    "code": "E0501",
                    "message": str(e),
                    "i18n_key": "",
                    "context": {},
                },
            )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "E0500",
                "message": str(e),
                "i18n_key": "",
                "context": {},
            },
        )

    # ファイル名
    filename = req.filename or f"export{ext}"
    mime = _MIME_TYPES.get(ext, "application/octet-stream")

    logger.debug("Export file: format=%s, size=%d, filename=%s", ext, len(data), filename)

    return Response(
        content=data,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/formats")
async def export_formats() -> ExportFormatsResponse:
    """対応形式一覧を返す"""
    import backend.free.export  # noqa: F401

    registry = get_writer_registry()
    return ExportFormatsResponse(
        formats=sorted(registry.supported_extensions()),
        availability=registry.check_availability(),
    )
