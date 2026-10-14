"""REPL の ``/feedback``: 直前の応答への明示評価 (docs/f_06 §2.2.2、f_04 §3.2.3)

``/feedback good|bad [理由]`` / ``/feedback clear`` を ``POST /api/learning/feedback`` へ
そのまま送る。宛先は現在のセッションの直前の応答で、その応答を生んだユーザー発話を
``query`` に添える (Web UI の 👍 / 👎 と同じ。直前のターンに経験が無いとき — private など —
に、ひとつ前のターンへ評価が刻まれるのを防ぐ)。

REPL だけのコマンド。非対話 (単発のメッセージ) では評価する応答が無く、サブコマンドに
すると宛先 (どのターンか) を取り違えるので作らない (f_06 §2.9.6)。

👎 だけに読み手があり 👍 には読み手が無い (f_04 §3.2.3) が、CLI は仕様どおり送るだけで
加工しない (不変則 #15)。``recorded=False`` は 200 でも成功として扱わない。
"""

from __future__ import annotations

import httpx

from backend.free.cli.backend_headers import backend_headers
from backend.free.cli.command_parser import CommandResult, SessionState
from backend.free.cli.renderer import render_error, render_info
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.feedback")

_FEEDBACK_TIMEOUT_SEC = 10.0

#: REPL の語 → API の ``verdict``
VERDICTS: dict[str, str] = {"good": "positive", "bad": "negative", "clear": "clear"}


def last_response_query(state: SessionState) -> tuple[bool, str | None]:
    """(直前の応答があるか, その応答を生んだユーザー発話)。

    発話が見つからない (応答だけが在る) ときは ``(True, None)`` で、宛先はセッションの最新。
    """
    for i in range(len(state.turns) - 1, -1, -1):
        if state.turns[i].get("role") != "assistant":
            continue
        for j in range(i - 1, -1, -1):
            if state.turns[j].get("role") == "user":
                return True, state.turns[j].get("content")
        return True, None
    return False, None


def build_feedback_body(state: SessionState, verdict: str, note: str, query: str | None) -> dict:
    """``POST /api/learning/feedback`` の本文。"""
    body: dict = {"session_id": state.session_id, "verdict": verdict, "note": note}
    if query:
        body["query"] = query
    return body


async def cmd_feedback(args: str, state: SessionState, console) -> CommandResult:
    """/feedback good|bad [理由] / /feedback clear"""
    word, _, rest = args.strip().partition(" ")
    word = word.lower()
    note = rest.strip()
    if not word:
        render_info(console, msg("cli.feedback_usage"))
        return CommandResult()
    verdict = VERDICTS.get(word)
    if verdict is None or (verdict == "clear" and note):
        render_error(console, msg("cli.feedback_usage"))
        return CommandResult()

    has_response, query = last_response_query(state)
    if not has_response:
        render_error(console, msg("cli.feedback_no_response"))
        return CommandResult()

    body = build_feedback_body(state, verdict, note, query)
    try:
        async with httpx.AsyncClient(headers=backend_headers(), timeout=_FEEDBACK_TIMEOUT_SEC) as client:
            resp = await client.post(f"{state.backend_url}/api/learning/feedback", json=body)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return CommandResult()
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return CommandResult()

    if data.get("recorded") is not True:
        logger.debug("/feedback not recorded: %s", data.get("message"))
        render_error(console, msg("cli.feedback_not_recorded", reason=data.get("message") or "-"))
        return CommandResult()
    render_info(console, msg(f"cli.feedback_recorded_{verdict}", entry_id=data.get("entry_id") or "-"))
    return CommandResult()
