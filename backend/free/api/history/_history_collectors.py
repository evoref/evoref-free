"""`/api/history` ハンドラから抽出した純粋な収集ロジック

`backend/free/api/history.py` の `list_history` / `get_history` ハンドラに
直書きされていた以下のマッピングロジックを抽出した純粋関数群:
- `IndexEntry` → `SessionSummary` マッピング (matched preview 計算込み)
- `SessionData` → `SessionDetailResponse` マッピング

レイヤー責務:
- `history.py` (API 層)            — HTTP / FastAPI / HistoryManager 取得
- `_history_collectors.py` (helper) — dataclass → Pydantic マッピング

すべて引数のみに依存する純粋関数。FastAPI / app_state / HistoryManager の
内部状態には触れない (`snippet_around` のみ history.utils を参照、純粋関数)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.free.api.history._history_schemas import (
    SessionDetailResponse,
    SessionSummary,
)
from backend.free.history.utils import snippet_around

if TYPE_CHECKING:
    from backend.free.history.history_manager import IndexEntry, SessionData


def to_session_summary(
    entry: IndexEntry,
    query: str | None = None,
) -> SessionSummary:
    """`IndexEntry` を `SessionSummary` に変換する純粋関数。

    `query` が指定され、かつ `entry.search_text` が存在すれば
    `snippet_around` でマッチ箇所のプレビューを生成する。
    マッチしなければ `matched_preview=None`。
    """
    preview: str | None = None
    if query and entry.search_text:
        preview = snippet_around(entry.search_text, query) or None
    return SessionSummary(
        session_id=entry.session_id,
        started_at=entry.started_at,
        duration_sec=entry.duration_sec,
        mode=entry.mode,
        turn_count=entry.turn_count,
        summary=entry.summary,
        first_user_preview=entry.first_user_preview,
        matched_preview=preview,
    )


def to_session_summaries(
    entries: list[IndexEntry],
    query: str | None = None,
) -> list[SessionSummary]:
    """`IndexEntry` リストを `SessionSummary` リストに変換する純粋関数。

    `to_session_summary` の薄いラッパで、`list_history` ハンドラの
    list comprehension を引き剥がす。
    """
    return [to_session_summary(e, query) for e in entries]


def to_session_detail_response(session: SessionData) -> SessionDetailResponse:
    """`SessionData` を `SessionDetailResponse` に変換する純粋関数。

    元 handler の field-by-field マッピングを保持する。
    """
    return SessionDetailResponse(
        session_id=session.session_id,
        started_at=session.started_at,
        ended_at=session.ended_at,
        duration_sec=session.duration_sec,
        mode=session.mode,
        modes_used=session.modes_used,
        instance_name=session.instance_name,
        base_model=session.base_model,
        turns=session.turns,
        turn_count=session.turn_count,
        context_files=session.context_files,
        cartridge_ids=session.cartridge_ids,
        summary=session.summary,
    )
