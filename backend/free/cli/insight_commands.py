"""``/memory`` ``/rag`` — 記憶と corpus の統計・ノートを CLI から見る (GUI の統計パネル相当)。"""

from __future__ import annotations

import httpx

from backend.free.cli.backend_headers import backend_headers
from backend.free.cli.command_parser import CommandResult, SessionState
from backend.free.cli.renderer import render_error, render_info
from backend.i18n_helper import msg


async def _get_json(state: SessionState, console, path: str, params: dict | None = None) -> dict | None:
    """GET して JSON を返す。失敗時は表示して ``None``。"""
    try:
        async with httpx.AsyncClient(headers=backend_headers(), timeout=10.0) as client:
            resp = await client.get(f"{state.backend_url}{path}", params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
    except httpx.TimeoutException:
        render_error(console, msg("cli.command_timeout"))
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code}")
    return None


def _flat_lines(data: dict, prefix: str = "") -> list[str]:
    """スカラーと入れ子 dict を ``key: value`` の行にする (list は件数のみ)。"""
    lines: list[str] = []
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            lines.extend(_flat_lines(value, f"{name}."))
        elif isinstance(value, list):
            lines.append(f"{name}: {len(value)}")
        else:
            lines.append(f"{name}: {value}")
    return lines


async def cmd_memory(args: str, state: SessionState, console) -> CommandResult:
    """/memory [stats|notes [N]] — 記憶の統計 / 短期ノート一覧"""
    parts = args.split()
    sub = parts[0] if parts else "stats"
    if sub == "stats":
        data = await _get_json(state, console, "/api/memory/stats")
        if data is not None:
            for line in _flat_lines(data):
                render_info(console, line)
        return CommandResult()
    if sub == "notes":
        limit = 10
        if len(parts) > 1:
            if not parts[1].isdigit() or not 1 <= int(parts[1]) <= 100:
                render_error(console, msg("cli.memory_usage"))
                return CommandResult()
            limit = int(parts[1])
        data = await _get_json(state, console, "/api/memory/notes", {"limit": limit})
        if data is None:
            return CommandResult()
        notes = data.get("notes", [])
        render_info(console, msg("cli.memory_notes_total", total=data.get("total", 0)))
        for note in notes:
            tags = ",".join(note.get("tags", []))
            render_info(console, f"{note.get('id', '')}  [{tags}]  {note.get('content', '')[:80]}")
        return CommandResult()
    render_error(console, msg("cli.memory_usage"))
    return CommandResult()


async def cmd_rag(args: str, state: SessionState, console) -> CommandResult:  # noqa: ARG001
    """/rag — corpus (文書由来チャンク) の統計"""
    data = await _get_json(state, console, "/api/rag/stats")
    if data is not None:
        for line in _flat_lines(data):
            render_info(console, line)
    return CommandResult()
