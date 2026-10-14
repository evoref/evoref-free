"""CLI コマンド解析とルーティング

REPL のスラッシュコマンドは 1 つの表 :data:`SLASH_COMMANDS` (``SlashSpec``) に並べ、
同期 / 非同期の振り分けと ``/help`` の一覧はそこから作る (docs/f_06 §2.9.5)。
ハンドラは ``"module:func"`` の文字列で持ち、呼ぶときに初めて import する
(``command_handlers`` が本モジュールを import するため)。

Pro のスラッシュコマンドと、Free のコマンドのサブアクション (``/cartridge create``) は
Pro 側が :func:`register_slash_command` で差し込む。登録の読み込みは CLI の登録表
(``registry``) の bootstrap に相乗りする (Pro 以上で動いているときだけ、
``EVOREF_EDITION=free`` では読まない)。
"""

from __future__ import annotations

import importlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.i18n_helper import msg
from backend.log_config import get_logger
from backend.free.cli.renderer import render_error

logger = get_logger("cli.command_parser")


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
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    sessions_dir: Path | None = None
    history_dir: Path | None = None
    auto_save_enabled: bool = True
    manually_saved: bool = False
    cli_theme: "CLITheme | None" = None
    ttft_history: list[float] = field(default_factory=list)
    response_times: list[float] = field(default_factory=list)
    error_count: int = 0
    # `default_cli_mode()` でエディション既定が解決される。Pro=create / Free=chat。
    # `_build_chat_payload` / `_register_session` / `_save_session` 等で参照する。
    mode: str = field(default_factory=lambda: _resolve_default_mode())
    #: ``/load`` で読んだ手動保存セッション。次の ``/save`` がその未知キーを書き戻す。
    loaded_session: "CliSession | None" = None

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


# ── スラッシュコマンドの表 ────────────────────────────────────────────────

#: ``(args, state, console)`` を受け、``CommandResult`` (非同期なら awaitable) を返す
type SlashHandler = Callable[..., Any]


@dataclass(frozen=True)
class SlashSpec:
    """スラッシュコマンド 1 つ (または Free のコマンドのサブアクション 1 つ) の登録内容。"""

    #: ``"/help"`` のようなコマンド名 (小文字)
    name: str
    #: ``"module:func"``。``func(args, state, console)`` が ``CommandResult`` を返す
    handler: str
    is_async: bool
    #: ``/help`` の 1 行目の左欄 (例: ``"/file <path>"``)
    usage: str
    #: ``/help`` の 1 行目の説明の i18n キー
    help_key: str
    edition: Literal["free", "pro"] = "free"
    #: ``/help`` に続けて出す行 (左欄, i18n キー)。1 コマンドに複数の書き方があるとき
    extra_help: tuple[tuple[str, str], ...] = ()
    #: サブアクションの拡張なら、その語 (例: ``/cartridge`` の ``"create"``)。
    #: ``None`` はコマンドそのもの
    action: str | None = None


_H = "backend.free.cli.command_handlers"


def _free(
    name: str, func: str, is_async: bool, usage: str, help_key: str,
    extra_help: tuple[tuple[str, str], ...] = (), *, module: str = _H,
) -> SlashSpec:
    return SlashSpec(
        name=name, handler=f"{module}:{func}", is_async=is_async,
        usage=usage, help_key=help_key, extra_help=extra_help,
    )


#: Free の REPL のスラッシュコマンド。並びが ``/help`` の表示順。
SLASH_COMMANDS: dict[str, SlashSpec] = {s.name: s for s in (
    _free("/help", "_cmd_help", False, "/help", "cli.help_help"),
    _free("/status", "_cmd_status", True, "/status", "cli.help_status"),
    _free("/file", "_cmd_file", False, "/file <path>", "cli.help_file",
          (("/file", "cli.help_file_list"),)),
    _free("/export", "_cmd_export", False, "/export <path> [--all|--last N]", "cli.help_export",
          module="backend.free.cli.export_commands"),
    _free("/history", "_cmd_history", True, "/history [action]", "cli.help_history"),
    _free("/cartridge", "_cmd_cartridge", True, "/cartridge [action]", "cli.help_cartridge"),
    _free("/theme", "_cmd_theme", True, "/theme [action]", "cli.help_theme"),
    _free("/migrate-model", "_cmd_migrate_model", True, "/migrate-model [options]",
          "cli.help_migrate_model"),
    _free("/web", "_cmd_web", True, "/web <url>", "cli.help_web"),
    _free("/learn", "_cmd_learn", True, "/learn [status|level1|full]", "cli.help_learn"),
    _free("/feedback", "cmd_feedback", True, "/feedback good|bad [<reason>] / clear",
          "cli.help_feedback", module="backend.free.cli.feedback_command"),
    _free("/reindex", "_cmd_reindex", True, "/reindex [--dry-run] [--cartridge <id>]",
          "cli.help_reindex"),
    _free("/page", "_cmd_page", True, "/page", "cli.help_page"),
    _free("/clear", "_cmd_clear", False, "/clear", "cli.help_clear"),
    _free("/save", "_cmd_save", False, "/save [name]", "cli.help_save"),
    _free("/load", "_cmd_load", False, "/load [name]", "cli.help_load", (
        ("/load --history <id>", "cli.help_load_history"),
        ("/load --history --latest", "cli.help_load_latest"),
    )),
    _free("/pin", "_cmd_pin", True, "/pin <text>", "cli.help_pin"),
    _free("/unpin", "_cmd_unpin", True, "/unpin <id> [--force]", "cli.help_unpin"),
    _free("/pinned", "_cmd_pinned", True, "/pinned", "cli.help_pinned"),
    _free("/memory", "cmd_memory", True, "/memory [stats|notes [N]]", "cli.help_memory",
          module="backend.free.cli.insight_commands"),
    _free("/rag", "cmd_rag", True, "/rag", "cli.help_rag",
          module="backend.free.cli.insight_commands"),
    _free("/exit", "_cmd_exit", False, "/exit", "cli.help_exit"),
)}

#: Pro が差し込むサブアクション (コマンド名 → 語の集合)。登録が無いとき Free のハンドラが
#: 「Pro 専用」と案内するのに使う。実体の登録は Pro 側 (Free は Pro のモジュールパスを持たない)。
PRO_SLASH_ACTIONS: dict[str, frozenset[str]] = {
    "/cartridge": frozenset({"create"}),
}

_pro_commands: dict[str, SlashSpec] = {}
_pro_actions: dict[tuple[str, str], SlashSpec] = {}


def register_slash_command(spec: SlashSpec) -> None:
    """Pro のスラッシュコマンド / サブアクションを登録する (Pro の自己登録が呼ぶ)。

    - ``spec.action`` が ``None``: Free に無い名前のコマンドを足す (Free の名前は上書きできない)。
    - ``spec.action`` がある: :data:`PRO_SLASH_ACTIONS` に宣言されたサブアクションだけを受ける。

    後勝ち (テストで差し替えられるようにする)。
    """
    if spec.edition != "pro":
        raise ValueError(f"not a Pro slash command: {spec.name!r} (edition={spec.edition})")
    if spec.action is None:
        if spec.name in SLASH_COMMANDS:
            raise ValueError(f"slash command already defined by Free: {spec.name!r}")
        _pro_commands[spec.name] = spec
        return
    if spec.action not in PRO_SLASH_ACTIONS.get(spec.name, frozenset()):
        raise ValueError(f"undeclared Pro slash action: {spec.name} {spec.action}")
    _pro_actions[(spec.name, spec.action)] = spec


def reset_slash_registry() -> None:
    """テスト用: Pro の登録を消す (bootstrap のやり直しは ``registry.reset_cli_registry``)。"""
    _pro_commands.clear()
    _pro_actions.clear()


def _bootstrap_pro() -> None:
    """Pro の登録を読み込む (CLI の登録表の bootstrap に相乗りする)。"""
    from backend.free.cli import registry

    registry.pro_commands()


def get_slash_spec(cmd: str) -> SlashSpec | None:
    """コマンド名の登録内容 (Free → 登録済みの Pro の順)。"""
    spec = SLASH_COMMANDS.get(cmd)
    if spec is not None:
        return spec
    _bootstrap_pro()
    return _pro_commands.get(cmd)


def get_slash_action(cmd: str, action: str) -> SlashSpec | None:
    """Pro が差し込んだサブアクション (無ければ ``None``)。"""
    _bootstrap_pro()
    return _pro_actions.get((cmd, action))


def all_slash_commands() -> list[SlashSpec]:
    """``/help`` に並べるコマンド (Free の表の順、続けて登録済みの Pro)。"""
    _bootstrap_pro()
    return [*SLASH_COMMANDS.values(), *_pro_commands.values()]


def help_entries() -> list[tuple[str, str]]:
    """``/help`` の行 (左欄, 説明) を表から作る。"""
    out: list[tuple[str, str]] = []
    for spec in all_slash_commands():
        out.append((spec.usage, msg(spec.help_key)))
        out += [(usage, msg(key)) for usage, key in spec.extra_help]
    return out


def resolve_slash_handler(spec: SlashSpec) -> SlashHandler:
    """``"module:func"`` を import して関数を返す。"""
    module_name, _, func_name = spec.handler.partition(":")
    return getattr(importlib.import_module(module_name), func_name)


def is_async_command(cmd: str) -> bool:
    """非同期コマンドかどうか"""
    spec = get_slash_spec(cmd)
    return spec is not None and spec.is_async


def handle_command(
    cmd: str, args: str, state: SessionState, console,
) -> CommandResult:
    """同期コマンドを実行"""
    logger.debug("handle_command: cmd=%s, args=%r", cmd, args[:50] if args else "")
    spec = get_slash_spec(cmd)
    if spec is None or spec.is_async:
        console.print()
        render_error(console, msg("cli.unknown_command", command=cmd))
        return CommandResult(handled=True)

    console.print()
    return resolve_slash_handler(spec)(args, state, console)


async def handle_async_command(
    cmd: str, args: str, state: SessionState, console,
) -> CommandResult:
    """非同期コマンドを実行"""
    console.print()
    spec = get_slash_spec(cmd)
    if spec is None or not spec.is_async:
        render_error(console, msg("cli.unknown_async_command", cmd=cmd))
        return CommandResult(handled=True)
    return await resolve_slash_handler(spec)(args, state, console)
