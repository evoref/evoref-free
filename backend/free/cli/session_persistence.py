"""CLI セッション永続化: 自動保存・チェックポイント・セッション復元"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.free.cli.command_parser import CommandResult, SessionState
from backend.free.cli.renderer import render_error, render_info
from backend.i18n_helper import msg
from backend.io import atomic_write_text
from backend.log_config import get_logger
from backend.utils import utc_now_dt

logger = get_logger("cli.session_persistence")


# ────────────────────────────────────────────
# 自動保存判定・データ構築
# ────────────────────────────────────────────


def _should_auto_save(state: SessionState) -> bool:
    """自動保存の条件を判定

    /save（セッション復元用）と history（アーカイブ）は別系統のため、
    /save 実行済みでも history への自動保存は行う。
    """
    if not state.auto_save_enabled:
        return False
    if not state.turns:
        return False
    # ユーザー発話が 1 ターン未満 → 保存しない
    if not any(t["role"] == "user" for t in state.turns):
        return False
    return True


def _build_history_data(state: SessionState) -> dict:
    """自動保存用のセッションデータを構築"""
    now = utc_now_dt()
    pct = int(state.token_used / state.token_limit * 100) if state.token_limit > 0 else 0
    started = datetime.fromtimestamp(state.started_at, tz=timezone.utc)
    duration = int(now.timestamp() - state.started_at)

    return {
        "session_id": state.session_id,
        "started_at": started.isoformat(),
        "ended_at": now.isoformat(),
        "duration_sec": duration,
        "mode": state.mode,
        "instance_name": state.instance_name,
        "base_model": "",
        "source": "auto",
        "turns": list(state.turns),
        "turn_count": len(state.turns),
        "context_files": list(state.context_files),
        "cartridge_ids": [],
        "token_info": {
            "used": state.token_used,
            "limit": state.token_limit,
            "pct": pct,
        },
        "ttft_history": list(state.ttft_history),
        "summary": None,
        "summary_embedding": None,
        "topics": [],
        "archived_at": now.isoformat(),
    }


# ────────────────────────────────────────────
# 自動保存・チェックポイント
# ────────────────────────────────────────────


def auto_save_session(state: SessionState) -> bool:
    """セッションを local/history/YYYY-MM/ に自動保存

    HistoryManager.save_session() 経由で保存し、index.json も更新する。

    Returns:
        True if saved, False if skipped or failed
    """
    if not _should_auto_save(state):
        return False

    if state.history_dir is None:
        logger.debug("auto_save: history_dir not configured, skipping")
        return False

    data = _build_history_data(state)

    try:
        from backend.free.history.history_manager import (
            HistoryManager, SessionData, get_history_manager,
        )
        try:
            mgr = get_history_manager()
        except RuntimeError:
            # config 未初期化（CLI 単独起動・テスト等）: 直接インスタンス化
            mgr = HistoryManager(state.history_dir)
        session = SessionData(
            session_id=data["session_id"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            duration_sec=data["duration_sec"],
            mode=data["mode"],
            instance_name=data["instance_name"],
            base_model=data["base_model"],
            source=data["source"],
            turns=data["turns"],
            turn_count=data["turn_count"],
            context_files=data["context_files"],
            cartridge_ids=data["cartridge_ids"],
            token_info=data["token_info"],
            summary=data.get("summary"),
            topics=data.get("topics", []),
            archived_at=data["archived_at"],
        )
        path = mgr.save_session(session)
        if path:
            logger.debug("auto_save: wrote %s (%d turns)", path, len(state.turns))
            return True
        logger.debug("auto_save: HistoryManager.save_session returned None")
        return False
    except OSError as e:
        logger.debug("auto_save: failed: %s", e)
        return False


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
    """セッション終了時の共通後処理

    1. サマリログ出力
    2. 自動保存
    3. チェックポイント削除
    """
    log_session_summary(state)
    auto_save_session(state)
    delete_checkpoint(state)


def save_checkpoint(state: SessionState) -> bool:
    """チェックポイントを .checkpoint/ に保存（上書き）"""
    if not state.auto_save_enabled or not state.turns:
        return False

    if state.history_dir is None:
        return False

    checkpoint_dir = state.history_dir / ".checkpoint"
    path = checkpoint_dir / f"{state.session_id}.json"

    data = _build_history_data(state)
    try:
        atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
        logger.debug("checkpoint: wrote %s (%d turns)", path, len(state.turns))
        return True
    except OSError as e:
        logger.debug("checkpoint: failed to write %s: %s", path, e)
        return False


def delete_checkpoint(state: SessionState) -> None:
    """セッションのチェックポイントファイルを削除"""
    if state.history_dir is None:
        return
    path = state.history_dir / ".checkpoint" / f"{state.session_id}.json"
    if path.exists():
        try:
            path.unlink()
            logger.debug("checkpoint: deleted %s", path)
        except OSError as e:
            logger.debug("checkpoint: failed to delete %s: %s", path, e)


def recover_checkpoints(history_dir: Path) -> int:
    """起動時に残存チェックポイントを正式アーカイブに昇格

    Returns:
        復旧したセッション数
    """
    checkpoint_dir = history_dir / ".checkpoint"
    if not checkpoint_dir.exists():
        return 0

    recovered = 0
    for cp_file in checkpoint_dir.glob("*.json"):
        try:
            data = json.loads(cp_file.read_text(encoding="utf-8"))
            started_at = data.get("started_at", "")
            session_id = data.get("session_id", cp_file.stem)

            # 月ディレクトリを started_at から決定
            if started_at:
                try:
                    dt = datetime.fromisoformat(started_at)
                    month_str = dt.strftime("%Y-%m")
                except ValueError:
                    month_str = utc_now_dt().strftime("%Y-%m")
            else:
                month_str = utc_now_dt().strftime("%Y-%m")

            month_dir = history_dir / month_str

            # ファイル名を生成
            ts = utc_now_dt().strftime("%Y%m%d_%H%M%S")
            dest = month_dir / f"{ts}_{session_id}.json"

            # アーカイブとして移動（source を維持）
            atomic_write_text(dest, json.dumps(data, ensure_ascii=False, indent=2))
            cp_file.unlink()
            recovered += 1
            logger.debug("checkpoint recovery: %s -> %s", cp_file.name, dest)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("checkpoint recovery: failed for %s: %s", cp_file.name, e)

    return recovered


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


def _load_latest_history(state: SessionState, console) -> CommandResult:
    """最新の自動保存セッションを復元"""
    latest_file: Path | None = None

    for month_dir in sorted(state.history_dir.iterdir(), reverse=True):
        if not month_dir.is_dir() or month_dir.name.startswith("."):
            continue
        files = sorted(month_dir.glob("*.json"), reverse=True)
        if files:
            latest_file = files[0]
            break

    if latest_file is None:
        render_error(console, "No history sessions found")
        return CommandResult()

    logger.debug("/load --history --latest: found %s", latest_file)
    return _restore_session(latest_file, state, console)


def _load_history_by_id(
    session_id: str, state: SessionState, console,
) -> CommandResult:
    """session_id で自動保存セッションを検索して復元"""
    for month_dir in sorted(state.history_dir.iterdir(), reverse=True):
        if not month_dir.is_dir() or month_dir.name.startswith("."):
            continue
        for f in month_dir.glob(f"*_{session_id}.json"):
            logger.debug("/load --history %s: found %s", session_id, f)
            return _restore_session(f, state, console)

    render_error(console, f"History session not found: {session_id}")
    return CommandResult()


def _restore_session(
    path: Path, state: SessionState, console,
) -> CommandResult:
    """セッションファイルからステートを復元"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("/load: failed to read %s: %s", path, e)
        render_error(console, f"Failed to load session: {e}")
        return CommandResult()

    # ステート復元
    state.session_id = data.get("session_id", state.session_id)
    state.turns = data.get("turns", [])
    state.context_files = data.get("context_files", [])
    state.file_chunks.clear()
    token_info = data.get("token_info", {})
    state.token_used = token_info.get("used", 0)
    state.token_limit = token_info.get("limit", 4096)

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
        "/load: restored session %s (name=%s, turns=%d, source=%s)",
        state.session_id, name, turn_count, source,
    )
    render_info(console, msg("cli.history_loaded", name=f"{name} ({turn_count} turns)"))
    return CommandResult()
