"""会話履歴の一覧・検索・詳細・統計・圧縮・削除

対話コマンド /history から呼び出される実装関数群。
"""

from __future__ import annotations

from datetime import datetime

import httpx

from backend.free.cli.renderer import (
    render_error,
    render_info,
    render_table,
)
from backend.free.history.utils import format_datetime, format_duration
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.history")

_DEFAULT_BACKEND = "http://localhost:8000"
_TIMEOUT = 30.0


# ── 内部実装関数 ──


def _history_list(
    backend_url: str, console,
    limit: int = 10,
    date_from: str | None = None,
    date_to: str | None = None,
) -> int:
    """会話履歴一覧を表示"""
    params: dict = {"limit": limit, "offset": 0}
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to

    try:
        resp = httpx.get(
            f"{backend_url}/api/history",
            params=params,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        render_error(console, f"API error: {e.response.status_code} {detail}".rstrip())
        return 1

    data = resp.json()
    sessions = data.get("sessions", [])
    total = data.get("total", 0)

    if not sessions:
        render_info(console, msg("cli.history_empty"))
        return 0

    render_info(console, msg("cli.history_total", total=total, showing=len(sessions)))

    rows = []
    for s in sessions:
        rows.append({
            "ID": s["session_id"][:8],
            msg("cli.history_col_date"): format_datetime(s.get("started_at", "")),
            msg("cli.history_col_turns"): str(s.get("turn_count", 0)),
            msg("cli.history_col_summary"): _truncate(s.get("summary") or "-", 40),
        })

    headers = [
        "ID",
        msg("cli.history_col_date"),
        msg("cli.history_col_turns"),
        msg("cli.history_col_summary"),
    ]
    render_table(console, rows, headers)
    return 0


def _history_search(
    backend_url: str, console,
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
) -> int:
    """会話履歴を検索"""
    payload: dict = {"query": query, "limit": limit, "search_turns": True}
    if date_from:
        payload["date_from"] = date_from
    if date_to:
        payload["date_to"] = date_to

    try:
        resp = httpx.post(
            f"{backend_url}/api/history/search",
            json=payload,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        render_error(console, f"API error: {e.response.status_code} {detail}".rstrip())
        return 1

    results = resp.json().get("results", [])
    if not results:
        render_info(console, msg("cli.history_search_no_results", query=query))
        return 0

    render_info(console, msg("cli.history_search_found", count=len(results), query=query))

    rows = []
    for r in results:
        rows.append({
            "ID": r["session_id"][:8],
            msg("cli.history_col_date"): format_datetime(r.get("started_at", "")),
            msg("cli.history_col_score"): f"{r.get('relevance_score', 0):.1f}",
            msg("cli.history_col_summary"): _truncate(r.get("summary") or "-", 40),
        })

    headers = [
        "ID",
        msg("cli.history_col_date"),
        msg("cli.history_col_score"),
        msg("cli.history_col_summary"),
    ]
    render_table(console, rows, headers)
    return 0


def _history_show(backend_url: str, console, session_id: str) -> int:
    """セッション詳細を表示"""
    try:
        resp = httpx.get(
            f"{backend_url}/api/history/{session_id}",
            timeout=_TIMEOUT,
        )
        if resp.status_code == 404:
            render_error(console, msg("cli.history_not_found", session_id=session_id))
            return 1
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        render_error(console, f"API error: {e.response.status_code} {detail}".rstrip())
        return 1

    s = resp.json()

    # メタ情報
    render_info(console, f"Session: {s['session_id']}")
    render_info(console, f"  {msg('cli.history_col_date')}: {format_datetime(s.get('started_at', ''))}")
    render_info(console, f"  {msg('cli.history_col_turns')}: {s.get('turn_count', 0)}")
    duration = s.get("duration_sec", 0)
    render_info(console, f"  {msg('cli.history_detail_duration')}: {format_duration(duration)}")
    if s.get("summary"):
        render_info(console, f"  {msg('cli.history_col_summary')}: {s['summary']}")
    if s.get("topics"):
        render_info(console, f"  {msg('cli.history_detail_topics')}: {', '.join(s['topics'])}")
    if s.get("context_files"):
        render_info(console, f"  {msg('cli.history_detail_files')}: {', '.join(s['context_files'])}")

    # ターン一覧
    turns = s.get("turns", [])
    if turns:
        render_info(console, "")
        render_info(console, f"--- {msg('cli.history_detail_turns')} ---")
        for i, turn in enumerate(turns):
            role = turn.get("role", "?")
            content = turn.get("content", "")
            compressed = turn.get("compressed", False)
            prefix = "U" if role == "user" else "A"
            suffix = f" ({msg('cli.history_detail_compressed')})" if compressed else ""
            render_info(console, f"  [{prefix}{i + 1}]{suffix} {_truncate(content, 100)}")

    return 0


def _history_stats(backend_url: str, console) -> int:
    """統計情報を表示"""
    try:
        resp = httpx.get(
            f"{backend_url}/api/history/stats",
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        render_error(console, f"API error: {e.response.status_code} {detail}".rstrip())
        return 1

    stats = resp.json()
    render_info(console, f"=== {msg('cli.history_stats_title')} ===")
    render_info(console, f"  {msg('cli.history_stats_sessions')}: {stats.get('total_sessions', 0)}")
    render_info(console, f"  {msg('cli.history_stats_turns')}: {stats.get('total_turns', 0)}")
    render_info(
        console,
        f"  {msg('cli.history_stats_size')}: "
        f"{stats.get('total_size_mb', 0):.2f} MB / {stats.get('max_storage_mb', 0):.0f} MB",
    )
    render_info(console, f"  {msg('cli.history_stats_summaries')}: {stats.get('summary_generated', 0)}")

    return 0


def _history_compact(backend_url: str, console) -> int:
    """圧縮処理を実行"""
    render_info(console, msg("cli.history_compact_running"))

    try:
        resp = httpx.post(
            f"{backend_url}/api/history/compact",
            timeout=60.0,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        render_error(console, f"API error: {e.response.status_code} {detail}".rstrip())
        return 1

    result = resp.json()
    render_info(console, msg(
        "cli.history_compact_done",
        compressed=result.get("compressed", 0),
        summarized=result.get("summarized", 0),
        deleted=result.get("deleted", 0),
        freed=f"{result.get('freed_mb', 0):.2f}",
    ))
    return 0


def _history_delete(backend_url: str, console, session_id: str) -> int:
    """セッションを削除"""
    try:
        resp = httpx.delete(
            f"{backend_url}/api/history/{session_id}",
            timeout=_TIMEOUT,
        )
        if resp.status_code == 404:
            render_error(console, msg("cli.history_not_found", session_id=session_id))
            return 1
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        render_error(console, f"API error: {e.response.status_code} {detail}".rstrip())
        return 1

    render_info(console, msg("cli.history_deleted", session_id=session_id))
    return 0


def _history_prune(backend_url: str, console, before: str) -> int:
    """指定日以前の履歴を削除"""
    # 日付バリデーション
    try:
        datetime.strptime(before, "%Y-%m-%d")
    except ValueError:
        render_error(console, msg("cli.history_invalid_date", date=before))
        return 1

    # 対象セッションを取得
    try:
        resp = httpx.get(
            f"{backend_url}/api/history",
            params={"limit": 1000, "offset": 0, "to": before},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        render_error(console, f"API error: {e.response.status_code} {detail}".rstrip())
        return 1

    sessions = resp.json().get("sessions", [])
    if not sessions:
        render_info(console, msg("cli.history_prune_none", before=before))
        return 0

    render_info(console, msg("cli.history_prune_target", count=len(sessions), before=before))

    session_ids = [s["session_id"] for s in sessions]
    try:
        del_resp = httpx.request(
            "DELETE",
            f"{backend_url}/api/history",
            json={"session_ids": session_ids},
            timeout=_TIMEOUT,
        )
        del_resp.raise_for_status()
        deleted = del_resp.json().get("deleted", 0)
        failed = len(session_ids) - deleted
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except Exception:
        deleted = 0
        failed = len(session_ids)

    render_info(console, msg("cli.history_prune_done", deleted=deleted, failed=failed))
    return 0


# ── ユーティリティ ──

def _truncate(text: str, max_len: int) -> str:
    """文字列を指定長で切り詰め"""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"
