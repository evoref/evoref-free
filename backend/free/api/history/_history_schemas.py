"""`/api/history` のレスポンス Pydantic スキーマ

`backend.free.api.history.history` から Pydantic スキーマを切り出した型モジュール。
循環 import 防止のため、`_history_collectors` (helper) と `history.py`
(handler) の双方から参照される型はここに集約する。

`history.py` は本モジュールから schemas を import + re-export し、外部からの
`from backend.free.api.history.history import SessionListResponse` の互換性を維持する。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionSummary(BaseModel):
    session_id: str
    started_at: str
    duration_sec: int
    mode: str
    turn_count: int
    summary: str | None = None
    topics: list[str] = Field(default_factory=list)
    matched_preview: str | None = None


class SessionListResponse(BaseModel):
    total: int
    sessions: list[SessionSummary]


class SessionDetailResponse(BaseModel):
    session_id: str
    started_at: str
    ended_at: str
    duration_sec: int
    mode: str
    modes_used: list[str] = Field(default_factory=list)
    instance_name: str
    base_model: str
    turns: list[dict] = Field(default_factory=list)
    turn_count: int
    context_files: list[str] = Field(default_factory=list)
    cartridge_ids: list[str] = Field(default_factory=list)
    summary: str | None = None
    topics: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    limit: int = Field(10, ge=1, le=100)
    search_turns: bool = False


class SearchResult(BaseModel):
    session_id: str
    started_at: str
    mode: str
    summary: str | None = None
    relevance_score: float = 0.0
    matched_turns: list[dict] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[SearchResult]


class BatchDeleteRequest(BaseModel):
    session_ids: list[str]


class BatchDeleteResponse(BaseModel):
    deleted: int


class CompactResponse(BaseModel):
    compressed: int
    summarized: int
    deleted: int
    freed_mb: float


class StatsResponse(BaseModel):
    total_sessions: int
    total_turns: int
    total_size_mb: float
    max_storage_mb: float
    mode_counts: dict[str, int] = Field(default_factory=dict)
    summary_generated: int
