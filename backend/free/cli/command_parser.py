"""CLI コマンド解析とルーティング"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from backend.i18n_helper import msg
from backend.log_config import get_logger
from backend.free.cli.renderer import render_error

logger = get_logger("cli.command_parser")

# 非同期コマンド（handle_command ではなく handle_async_command で処理）
ASYNC_COMMANDS = {"/history", "/learn", "/page", "/status", "/cartridge", "/migrate-model", "/web", "/theme", "/reindex", "/pin", "/unpin", "/pinned", "/loop", "/tasks", "/private"}


@dataclass
class SessionState:
    """CLI セッション状態"""
    instance_name: str = "evoref"
    backend_url: str = "http://localhost:8000"
    context_files: list[str] = field(default_factory=list)
    file_chunks: dict[str, list[str]] = field(default_factory=dict)
    token_used: int = 0
    token_limit: int = 4096
    should_exit: bool = False
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    turns: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    sessions_dir: Path | None = None
    history_dir: Path | None = None
    auto_save_enabled: bool = True
    checkpoint_interval: int = 10
    manually_saved: bool = False
    cli_theme: "CLITheme | None" = None
    ttft_history: list[float] = field(default_factory=list)
    response_times: list[float] = field(default_factory=list)
    error_count: int = 0
    # プライベートセッション (memory_only)
    # ``True`` の間、チャットリクエストに ``private: true`` を付与する。
    private_mode: bool = False
    # `default_cli_mode()` でエディション既定が解決される。Pro=create / Free=chat。
    # `_build_chat_payload` / `_register_session` / `_save_session` 等で参照する。
    mode: str = field(default_factory=lambda: _resolve_default_mode())

    def add_turn(self, role: str, content: str) -> None:
        """対話ターンを追加"""
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "compressed": False,
        })

    def record_ttft(self, ttft_sec: float) -> None:
        """TTFT（Time to First Token）を記録"""
        self.ttft_history.append(round(ttft_sec, 3))

    def record_response_time(self, elapsed_sec: float) -> None:
        """1 ターンの総応答時間（秒）を記録"""
        self.response_times.append(round(elapsed_sec, 3))

    def record_error(self) -> None:
        """セッション中のエラー発生をカウント"""
        self.error_count += 1


def _resolve_default_mode() -> str:
    """Free=chat / Pro=create をエディション判定で解決する。

    `field(default_factory=...)` から呼ぶための関数化。直接 `default_cli_mode` を
    参照すると import 順序の都合で循環するため薄いラッパとして定義。
    """
    from backend.free.cli.cli_mode import default_cli_mode
    return default_cli_mode()


@dataclass
class CommandResult:
    """コマンド実行結果"""
    handled: bool = True  # コマンドとして処理したか
    message: str = ""


def parse_command(text: str) -> tuple[str, str]:
    """入力テキストをコマンド名と引数に分割

    Returns:
        (command, args) — コマンドでなければ ("", "")
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "", ""
    parts = stripped.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return cmd, args


def is_async_command(cmd: str) -> bool:
    """非同期コマンドかどうか"""
    return cmd in ASYNC_COMMANDS


def handle_command(
    cmd: str, args: str, state: SessionState, console,
) -> CommandResult:
    """同期コマンドを実行"""
    from backend.free.cli.command_handlers import (
        _cmd_clear,
        _cmd_exit,
        _cmd_file,
        _cmd_help,
        _cmd_load,
        _cmd_save,
    )
    from backend.free.cli.export_commands import _cmd_export

    logger.debug("handle_command: cmd=%s, args=%r", cmd, args[:50] if args else "")
    handlers = {
        "/help": _cmd_help,
        "/file": _cmd_file,
        "/export": _cmd_export,
        "/clear": _cmd_clear,
        "/save": _cmd_save,
        "/load": _cmd_load,
        "/exit": _cmd_exit,
    }

    handler = handlers.get(cmd)
    if handler is None:
        console.print()
        render_error(console, msg("cli.unknown_command", command=cmd))
        return CommandResult(handled=True)

    console.print()
    return handler(args, state, console)


async def handle_async_command(
    cmd: str, args: str, state: SessionState, console,
) -> CommandResult:
    """非同期コマンドを実行"""
    from backend.free.cli.command_handlers import (
        _cmd_cartridge,
        _cmd_history,
        _cmd_learn,
        _cmd_loop,
        _cmd_migrate_model,
        _cmd_page,
        _cmd_pin,
        _cmd_pinned,
        _cmd_private,
        _cmd_reindex,
        _cmd_status,
        _cmd_tasks,
        _cmd_theme,
        _cmd_unpin,
        _cmd_web,
    )

    console.print()
    if cmd == "/private":
        return await _cmd_private(args, state, console)
    if cmd == "/pin":
        return await _cmd_pin(args, state, console)
    if cmd == "/unpin":
        return await _cmd_unpin(args, state, console)
    if cmd == "/pinned":
        return await _cmd_pinned(args, state, console)
    if cmd == "/history":
        return await _cmd_history(args, state, console)
    if cmd == "/learn":
        return await _cmd_learn(args, state, console)
    if cmd == "/page":
        return await _cmd_page(args, state, console)
    if cmd == "/status":
        return await _cmd_status(args, state, console)
    if cmd == "/cartridge":
        return await _cmd_cartridge(args, state, console)
    if cmd == "/migrate-model":
        return await _cmd_migrate_model(args, state, console)
    if cmd == "/web":
        return await _cmd_web(args, state, console)
    if cmd == "/theme":
        return await _cmd_theme(args, state, console)
    if cmd == "/reindex":
        return await _cmd_reindex(args, state, console)
    if cmd == "/loop":
        return await _cmd_loop(args, state, console)
    if cmd == "/tasks":
        return await _cmd_tasks(args, state, console)
    render_error(console, f"Unknown async command: {cmd}")
    return CommandResult(handled=True)
