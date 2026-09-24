"""CLI セッション: 終了時のサマリ・履歴からのセッション復元"""

from __future__ import annotations

import re
import time
from pathlib import Path

from backend.free.cli.command_parser import CommandResult, SessionState
from backend.free.cli.renderer import render_error, render_info
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.session_persistence")


# ────────────────────────────────────────────
# セッション終了
# ────────────────────────────────────────────
#
# 会話履歴は backend が書く (``/api/chat`` の記録経路がターンごとの追記ログへ出し、
# 閉じたとき・起動時に畳む、c_05 §2.1)。CLI は同じセッションを自前で
# ``HistoryManager.save_session`` しない — 以前は CLI と backend の 2 つの書き手が
# 別の形 (``ttft_history`` の有無・ファイル名の session_id の長さ) で同じ会話を
# 書き、CLI 側のチェックポイント (``.checkpoint/``) と backend の追記が二重だった。


def log_session_summary(state: SessionState) -> None:
    """セッション全体の統計サマリを cli.log に INFO レベルで出力

    出力項目:
        - session_id: セッション識別子
        - turns: 総ターン数（user + assistant 合計）
        - user_turns: ユーザー発話ターン数
        - tokens: 総トークン消費量
        - avg_response_ms: 1 ターンあたりの平均応答時間（ミリ秒）
        - duration_sec: セッション継続時間（秒）
        - errors: エラー発生回数

    ターンが存在しないセッション（コマンドのみ・即時終了等）はノイズ削減のため
    INFO 出力をスキップし、debug ログだけ残す。
    """
    duration_sec = max(0, int(time.time() - state.started_at))
    user_turns = sum(1 for t in state.turns if t.get("role") == "user")
    if state.response_times:
        avg_response_ms = int(
            sum(state.response_times) / len(state.response_times) * 1000,
        )
    else:
        avg_response_ms = 0

    if not state.turns:
        logger.debug(
            "session summary skipped (no turns): session_id=%s duration_sec=%d errors=%d",
            state.session_id, duration_sec, state.error_count,
        )
        return

    logger.info(
        "session summary: session_id=%s turns=%d user_turns=%d tokens=%d "
        "avg_response_ms=%d duration_sec=%d errors=%d",
        state.session_id,
        len(state.turns),
        user_turns,
        state.token_used,
        avg_response_ms,
        duration_sec,
        state.error_count,
    )


def finalize_session(state: SessionState) -> None:
    """セッション終了時の共通後処理 (サマリログ出力)。

    履歴の保存は backend の記録経路が行う (このモジュール冒頭の注記)。
    """
    log_session_summary(state)


# ────────────────────────────────────────────
# セッション復元（--history フラグ処理）
# ────────────────────────────────────────────


def _load_from_history(
    parts: list[str], state: SessionState, console,
) -> CommandResult:
    """--history フラグ付き /load の処理"""
    if state.history_dir is None:
        render_error(console, "History directory not configured")
        return CommandResult()

    if not state.history_dir.exists():
        render_error(console, "No history found")
        return CommandResult()

    remaining = [p for p in parts if p != "--history"]

    if "--latest" in remaining:
        return _load_latest_history(state, console)

    session_id = remaining[0] if remaining else ""
    if not session_id:
        render_error(console, "Usage: /load --history <session_id> or /load --history --latest")
        return CommandResult()

    return _load_history_by_id(session_id, state, console)


#: 履歴の月ディレクトリ (``active/`` ``embeddings/`` 等の内部ディレクトリを除く)。
_MONTH_DIR_RE = re.compile(r"\d{4}-\d{2}")


def _month_dirs(history_dir: Path) -> list[Path]:
    return sorted(
        (d for d in history_dir.iterdir() if d.is_dir() and _MONTH_DIR_RE.fullmatch(d.name)),
        reverse=True,
    )


def _load_latest_history(state: SessionState, console) -> CommandResult:
    """最新の自動保存セッションを復元"""
    latest_file: Path | None = None

    for month_dir in _month_dirs(state.history_dir):
        files = sorted(month_dir.glob("*.json"), reverse=True)
        if files:
            latest_file = files[0]
            break

    if latest_file is None:
        render_error(console, "No history sessions found")
        return CommandResult()

    logger.debug("/load --history --latest: found %s", latest_file)
    return _restore_history_session(latest_file, state, console)


def _load_history_by_id(
    session_id: str, state: SessionState, console,
) -> CommandResult:
    """session_id で自動保存セッションを検索して復元"""
    for month_dir in _month_dirs(state.history_dir):
        for f in month_dir.glob(f"*_{session_id}.json"):
            logger.debug("/load --history %s: found %s", session_id, f)
            return _restore_history_session(f, state, console)

    render_error(console, f"History session not found: {session_id}")
    return CommandResult()


def _restore_history_session(
    path: Path, state: SessionState, console,
) -> CommandResult:
    """会話履歴の原本 (``history.session`` の封筒) からステートを復元"""
    from backend.free.history.history_manager import read_session_file

    read = read_session_file(path)
    if not read.ok or not isinstance(read.payload, dict):
        logger.debug("/load: failed to read %s: %s %s", path, read.status, read.detail)
        render_error(console, f"Failed to load session: {read.status} {read.detail}")
        return CommandResult()
    return _apply_session_data(read.payload, path, state, console)


def _apply_session_data(
    data: dict, path: Path, state: SessionState, console,
) -> CommandResult:
    """読んだセッションの dict をステートへ反映する。"""

    # ステート復元 (手動保存の読み手は復元の後で読んだレコードを入れ直す)
    state.loaded_session = None
    state.session_id = data.get("session_id", state.session_id)
    state.turns = data.get("turns", [])
    state.context_files = data.get("context_files", [])
    state.file_chunks.clear()
    token_info = data.get("token_info", {})
    state.token_used = token_info.get("used", 0)
    state.token_limit = token_info.get("limit", 4096)
    # 保存時のモードを復元 (save は mode を書き込むが load 側が捨てていた)。
    # 不正値は現行モードを維持。Free で create を読んだ場合の降格は
    # 起動時 coerce_cli_mode に委ねる (ここでは生値を尊重)。
    saved_mode = data.get("mode")
    if saved_mode in ("chat", "create"):
        state.mode = saved_mode

    # 保存されていたファイルパスを再読込み
    if state.context_files:
        from backend.free.cli.file_reader import FileReaderError, read_and_chunk
        valid_files = []
        for file_path in state.context_files:
            p = Path(file_path)
            if not p.exists():
                logger.debug("/load: context file no longer exists: %s", file_path)
                continue
            try:
                result = read_and_chunk(p)
                state.file_chunks[file_path] = result.chunks
                valid_files.append(file_path)
            except FileReaderError as e:
                logger.debug("/load: failed to re-read context file %s: %s", file_path, e)
        state.context_files = valid_files

    name = data.get("name", path.stem)
    turn_count = len(state.turns)
    source = data.get("source", "unknown")
    logger.debug(
        "/load: restored session %s (name=%s, turns=%d, source=%s, mode=%s)",
        state.session_id, name, turn_count, source, state.mode,
    )
    render_info(console, msg("cli.history_loaded", name=f"{name} ({turn_count} turns)"))
    return CommandResult()
