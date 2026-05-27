"""Diff 適用 API エンドポイント"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.log_config import get_logger

logger = get_logger("api.diffs")

router = APIRouter(prefix="/api/diffs", tags=["diffs"])


class DiffExtractRequest(BaseModel):
    """Diff 抽出リクエスト"""
    response_text: str


class DiffBlockResponse(BaseModel):
    """抽出された diff ブロック"""
    raw: str
    file_path: str | None
    has_hunks: bool


class DiffExtractResponse(BaseModel):
    """Diff 抽出レスポンス"""
    diffs: list[DiffBlockResponse]
    count: int


class DiffApplyRequest(BaseModel):
    """Diff 適用リクエスト"""
    file_path: str
    diff_text: str


class DiffApplyResponse(BaseModel):
    """Diff 適用レスポンス"""
    success: bool
    message: str


@router.post("/extract", response_model=DiffExtractResponse)
async def extract_diffs_endpoint(body: DiffExtractRequest):
    """LLM 応答テキストから diff ブロックを抽出する"""
    from backend.free.services.diff_service import extract_diffs

    logger.debug(
        "POST /api/diffs/extract: text_len=%d", len(body.response_text),
    )

    diffs = extract_diffs(body.response_text)
    return DiffExtractResponse(
        diffs=[
            DiffBlockResponse(
                raw=d.raw,
                file_path=d.file_path,
                has_hunks=d.has_hunks,
            )
            for d in diffs
        ],
        count=len(diffs),
    )


@router.post("/apply", response_model=DiffApplyResponse)
async def apply_diff_endpoint(body: DiffApplyRequest):
    """unified diff をファイルに適用する"""
    from pathlib import Path

    from backend.free.services.diff_service import apply_unified_diff

    logger.debug(
        "POST /api/diffs/apply: file=%s, diff_len=%d",
        body.file_path, len(body.diff_text),
    )

    p = Path(body.file_path)
    if not p.exists():
        raise HTTPException(404, f"File not found: {body.file_path}")

    success, message = await asyncio.to_thread(
        apply_unified_diff, body.file_path, body.diff_text,
    )

    return DiffApplyResponse(success=success, message=message)
