"""会話履歴 API"""

import re

from fastapi import APIRouter, HTTPException, Query

from backend.free.api.history._history_collectors import (
    to_session_detail_response,
    to_session_summaries,
)

# Pydantic スキーマは _history_schemas に集約
# 外部 import 互換性のため re-export する。
from backend.free.api.history._history_schemas import (
    BatchDeleteRequest,
    BatchDeleteResponse,
    CompactResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SessionDetailResponse,
    SessionListResponse,
    SessionSummary,
    StatsResponse,
)
from backend.free.history.history_manager import get_history_manager
from backend.log_config import get_logger

logger = get_logger("api.history")

router = APIRouter(prefix="/api/history", tags=["history"])

_SESSION_ID_RE = re.compile(r"^[0-9a-fA-F]{8,64}$|^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

__all__ = [
    "router",
    # 互換性のため re-export
    "BatchDeleteRequest",
    "BatchDeleteResponse",
    "CompactResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SessionDetailResponse",
    "SessionListResponse",
    "SessionSummary",
    "StatsResponse",
]


# ── エンドポイント ──
# 注意: 固定パス (/stats, /compact, /search) は
# パラメータパス (/{session_id}) より前に定義すること

@router.get("", response_model=SessionListResponse)
async def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    mode: str | None = None,
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
    q: str | None = None,
):
    """会話履歴の一覧を取得"""
    logger.debug(
        "GET /api/history: limit=%d, offset=%d, mode=%s, q=%s",
        limit, offset, mode, q,
    )
    mgr = get_history_manager()
    entries, total = mgr.list_sessions(
        limit=limit, offset=offset, mode=mode,
        date_from=date_from, date_to=date_to, query=q,
    )
    return SessionListResponse(
        total=total,
        sessions=to_session_summaries(entries, q),
    )


@router.get("/stats", response_model=StatsResponse)
async def history_stats():
    """統計情報を取得"""
    mgr = get_history_manager()
    stats = mgr.get_stats()
    return StatsResponse(**stats)


@router.post("/search", response_model=SearchResponse)
async def search_history(req: SearchRequest):
    """セッション検索"""
    logger.debug("POST /api/history/search: query=%r, mode=%s", req.query[:50], req.mode)
    mgr = get_history_manager()
    results = mgr.search_sessions(
        query=req.query,
        mode=req.mode,
        date_from=req.date_from,
        date_to=req.date_to,
        limit=req.limit,
        search_turns=req.search_turns,
    )
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
    )


@router.delete("", response_model=BatchDeleteResponse)
async def batch_delete_history(req: BatchDeleteRequest):
    """複数セッションを一括削除"""
    logger.debug("DELETE /api/history: %d session_ids", len(req.session_ids))
    mgr = get_history_manager()
    deleted = mgr.delete_sessions_batch(req.session_ids)
    return BatchDeleteResponse(deleted=deleted)


@router.post("/compact", response_model=CompactResponse)
async def compact_history():
    """保持ポリシーに基づく圧縮処理を実行"""
    logger.debug("POST /api/history/compact")
    mgr = get_history_manager()
    result = mgr.compact_sessions()
    return CompactResponse(**result)


def _validate_session_id(session_id: str) -> None:
    """session_id の形式チェック（UUID hex）"""
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session_id format")


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_history(session_id: str):
    """特定セッションの詳細を取得"""
    _validate_session_id(session_id)
    mgr = get_history_manager()
    session = mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return to_session_detail_response(session)


@router.delete("/{session_id}")
async def delete_history(session_id: str):
    """セッションを削除"""
    _validate_session_id(session_id)
    logger.debug("DELETE /api/history/%s", session_id)
    mgr = get_history_manager()
    deleted = mgr.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}
