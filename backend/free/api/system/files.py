"""ファイル管理 API エンドポイント"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Query
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.free.api.system._file_helpers import (
    file_invalid_error,
    file_not_found_error,
    file_path_not_found_error,
    file_summary_dict,
    file_upload_dict,
    require_file_manager,
)
from backend.log_config import get_logger

logger = get_logger("api.files")


class ReadAndChunkRequest(BaseModel):
    """ファイル読込み・チャンキングリクエスト"""
    file_path: str
    chunk_size: int = 512
    chunk_overlap: int = 128


class ReadAndChunkResponse(BaseModel):
    """ファイル読込み・チャンキングレスポンス"""
    filename: str
    path: str
    chunks: list[str]
    total_chars: int
    chunk_count: int
    file_type: str

router = APIRouter(prefix="/api/files", tags=["files"])


@router.post("/upload")
async def upload_file(
    state: AppState = Depends(get_app_state),
    file: UploadFile = File(...),
    session_id: str = Query(default="default"),
):
    """ファイルをアップロード"""
    logger.debug(
        "POST /api/files/upload: filename=%s, session=%s",
        file.filename, session_id,
    )
    mgr = require_file_manager(state)

    data = await file.read()
    logger.debug("File read: %d bytes", len(data))
    try:
        info = mgr.upload(data, file.filename or "unnamed", session_id)
    except ValueError as e:
        raise file_invalid_error(str(e))

    logger.debug("File uploaded: id=%s, mime=%s", info.file_id, info.mime_type)
    return file_upload_dict(info)


@router.get("")
async def list_files(
    state: AppState = Depends(get_app_state),
    session_id: str = Query(default="default"),
):
    """セッション内のファイル一覧"""
    logger.debug("GET /api/files: session=%s", session_id)
    mgr = require_file_manager(state)

    files = mgr.list_files(session_id)
    logger.debug("Listed %d files for session %s", len(files), session_id)
    return [file_summary_dict(f) for f in files]


@router.get("/{file_id}/content")
async def get_file_content(file_id: str, state: AppState = Depends(get_app_state)):
    """ファイルの内容を取得"""
    logger.debug("GET /api/files/%s/content", file_id)
    mgr = require_file_manager(state)

    content = mgr.get_content(file_id)
    if content is None:
        raise file_not_found_error(file_id)
    return {"content": content}


@router.put("/{file_id}/content")
async def update_file_content(file_id: str, body: dict, state: AppState = Depends(get_app_state)):
    """ファイルの内容を更新"""
    logger.debug("PUT /api/files/%s/content: content_len=%d", file_id, len(body.get("content", "")))
    mgr = require_file_manager(state)

    content = body.get("content", "")
    if not mgr.update_file(file_id, content):
        raise file_not_found_error(file_id)
    return {"status": "updated"}


@router.delete("/{file_id}")
async def delete_file(file_id: str, state: AppState = Depends(get_app_state)):
    """ファイルを削除"""
    logger.debug("DELETE /api/files/%s", file_id)
    mgr = require_file_manager(state)

    if not mgr.delete_file(file_id):
        raise file_not_found_error(file_id)
    logger.debug("File deleted: %s", file_id)
    return {"status": "deleted"}


@router.post("/read-and-chunk", response_model=ReadAndChunkResponse)
async def read_and_chunk_file(body: ReadAndChunkRequest):
    """ローカルファイルを読込みチャンク分割する

    CLI の /file コマンドと同等の機能を API 経由で提供する。
    """
    import asyncio

    from backend.free.services.file_service import (
        FileServiceError,
        read_and_chunk,
    )

    logger.debug(
        "POST /api/files/read-and-chunk: path=%s, chunk_size=%d",
        body.file_path, body.chunk_size,
    )

    p = Path(body.file_path)
    if not p.exists():
        raise file_path_not_found_error(body.file_path)

    try:
        result = await asyncio.to_thread(
            read_and_chunk, p, body.chunk_size, body.chunk_overlap,
        )
    except FileServiceError as e:
        raise file_invalid_error(str(e))

    return ReadAndChunkResponse(
        filename=result.filename,
        path=result.path,
        chunks=result.chunks,
        total_chars=result.total_chars,
        chunk_count=result.chunk_count,
        file_type=result.file_type,
    )
