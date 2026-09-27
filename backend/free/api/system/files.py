"""添付ファイルの取り込み API (f_03 §11)

POST /api/files/extract — multipart の 1 ファイルからテキストを抽出し、チャット
要求の ``file_contexts`` にそのまま載せられるチャンクを返す。保存しない。

本文は受信中に上限を数えながらメモリに読み、上限まではメモリに留める multipart
解析器で分ける (Starlette の既定は 1 MB を超えるファイルを OS の一時ディレクトリへ
書き出す)。抽出は ``extract_from_bytes`` なので一時ファイルも作らない。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import PureWindowsPath
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from backend.error_handlers import ErrorResponse
from backend.free.api.chat.chat_constants import (
    MAX_FILE_CONTEXT_TOTAL_CHARS,
    MAX_FILE_CONTEXT_TOTAL_CHUNKS,
)
from backend.free.api.schemas import FileExtractResponse
from backend.free.services.file_service import (
    FileServiceError,
    chunk_text,
    extract_text_from_upload,
    get_supported_extensions,
)
from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context

logger = get_logger("api.files")

router = APIRouter(prefix="/api/files", tags=["files"])

#: 添付 1 件の上限 (フロントの ``FILE_MAX_SIZE_BYTES`` と同じ値)
MAX_EXTRACT_FILE_BYTES = 10 * 1024 * 1024

#: multipart の境界・ヘッダの分として本文の上限に足す余裕
_MULTIPART_OVERHEAD_BYTES = 64 * 1024

_MAX_REQUEST_BYTES = MAX_EXTRACT_FILE_BYTES + _MULTIPART_OVERHEAD_BYTES


class _InMemoryMultiPartParser(MultiPartParser):
    """上限までのファイルを SpooledTemporaryFile のメモリ側に留める解析器"""

    spool_max_size = _MAX_REQUEST_BYTES


def _error(status_code: int, message: str, i18n_key: str, **context: Any) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(
            code=f"E0{status_code}", message=message, i18n_key=i18n_key, context=context,
        ).to_dict(),
    )


def _no_file_error() -> HTTPException:
    return _error(400, "A multipart field 'file' with a filename is required",
                  "api.file_extract_no_file")


def _too_large_error(filename: str) -> HTTPException:
    return _error(
        413, f"File exceeds {MAX_EXTRACT_FILE_BYTES} bytes", "api.file_extract_too_large",
        filename=filename, limit_mb=MAX_EXTRACT_FILE_BYTES // (1024 * 1024),
    )


async def _read_body_capped(request: Request) -> bytes:
    """本文を上限まで読む。超えた時点で 413 (全体を受け取ってから測らない)"""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > _MAX_REQUEST_BYTES:
        raise _too_large_error("")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _MAX_REQUEST_BYTES:
            raise _too_large_error("")
    return bytes(body)


def _extract_chunks(data: bytes, filename: str) -> tuple[list[str], int, bool]:
    """抽出して ``file_contexts`` の上限に収まるチャンクを返す (イベントループの外で呼ぶ)

    Returns:
        (チャンク, 本文全体の文字数, 上限に合わせて切ったか)
    """
    text = extract_text_from_upload(data, filename)
    # 上限を超える本文は分割の前に切る (分割は本文長に比例して遅い)
    truncated = len(text) > MAX_FILE_CONTEXT_TOTAL_CHARS
    chunks: list[str] = []
    used = 0
    for chunk in chunk_text(text[:MAX_FILE_CONTEXT_TOTAL_CHARS]):
        if len(chunks) >= MAX_FILE_CONTEXT_TOTAL_CHUNKS or used + len(chunk) > MAX_FILE_CONTEXT_TOTAL_CHARS:
            truncated = True
            break
        chunks.append(chunk)
        used += len(chunk)
    return chunks, len(text), truncated


@router.post("/extract", response_model=FileExtractResponse)
async def extract_file(request: Request) -> FileExtractResponse:
    """添付 1 件からテキストを抽出してチャンクを返す (保存しない)"""
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise _no_file_error()

    body = await _read_body_capped(request)

    async def _once() -> AsyncGenerator[bytes, None]:
        yield body

    try:
        form = await _InMemoryMultiPartParser(
            request.headers, _once(), max_files=1, max_fields=0,
        ).parse()
    except MultiPartException as exc:
        logger.info("Rejected attachment upload: %s", exc)
        raise _no_file_error() from exc

    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile) or not upload.filename:
            raise _no_file_error()
        # ブラウザ以外はパス付きの名前を送れる。表示とモデルへの注入には名前だけ使う
        filename = PureWindowsPath(upload.filename).name
        data = await upload.read()
    finally:
        await form.close()

    if len(data) > MAX_EXTRACT_FILE_BYTES:
        raise _too_large_error(filename)
    suffix = PureWindowsPath(filename).suffix.lower()
    if suffix not in get_supported_extensions():
        raise _error(415, f"Unsupported file format: {suffix or '(none)'}",
                     "api.file_extract_unsupported", filename=filename, ext=suffix)

    loop = asyncio.get_running_loop()
    try:
        chunks, chars, truncated = await run_in_executor_with_context(
            loop, None, _extract_chunks, data, filename,
        )
    except FileServiceError as exc:
        code = str(exc)
        logger.info("Attachment extraction failed for %s: %s", filename, code)
        if code.startswith(("legacy_format", "unsupported_format")):
            raise _error(415, f"Unsupported file format: {suffix}",
                         "api.file_extract_unsupported", filename=filename, ext=suffix) from exc
        if code in ("empty_content", "scan_only_pdf"):
            raise _error(422, "No text could be extracted", "api.file_extract_empty",
                         filename=filename) from exc
        raise _error(422, f"Extraction failed: {code}", "api.file_extract_failed",
                     filename=filename) from exc

    logger.debug(
        "Extracted attachment %s: %d bytes -> %d chars, %d chunks (truncated=%s)",
        filename, len(data), chars, len(chunks), truncated,
    )
    return FileExtractResponse(filename=filename, chunks=chunks, truncated=truncated, chars=chars)
