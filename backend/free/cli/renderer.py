"""CLI レンダラー: rich によるレンダリング + プレーンテキストフォールバック

Claude Code 風ミニマルデザイン:
- Panel/Rule 不使用、テキストベース表示
- ❯ プロンプト、パディングなし（横幅最大化）
- ステータスライン（応答後に tokens · elapsed）
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from backend.free.cli.cli_theme import CLITheme

# モジュールレベルの CLITheme 参照（set_cli_theme で差し替え可能）
_cli_theme: CLITheme = CLITheme.default()


def set_cli_theme(theme: CLITheme) -> None:
    """モジュールレベルの CLITheme を差し替える"""
    global _cli_theme
    _cli_theme = theme


def get_cli_theme() -> CLITheme:
    """現在の CLITheme を取得"""
    return _cli_theme


def _token_color(pct: int) -> str:
    """トークン使用率に応じた色"""
    if pct < 60:
        return _cli_theme.token_low
    elif pct < 85:
        return _cli_theme.token_mid
    return _cli_theme.token_high


def create_console(no_color: bool = False) -> Console:
    """環境に応じた Console を作成"""
    if no_color or _is_no_color():
        return Console(width=None, no_color=True, highlight=False)
    return Console(width=None)


def _is_no_color() -> bool:
    """カラー非対応環境を検出

    NO_COLOR 環境変数が設定されている場合、または TERM=dumb の場合に True を返す。
    Windows では TERM が未設定でもカラー対応（Windows Terminal / PowerShell / VSCode）
    のため、TERM 未設定はカラー無効と判定しない。
    """
    if os.environ.get("NO_COLOR"):
        return True
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return True
    # Windows: TERM 未設定でもカラー対応（ConPTY / Windows Terminal）
    # Unix: TERM 未設定は非対話環境（CI/パイプ等）の可能性があるため無効化
    if not term and os.name != "nt":
        return True
    return False


def render_response(console: Console, text: str) -> None:
    """AI 応答をレンダリング（Markdown インライン表示）"""
    if console.no_color:
        console.print(text)
    else:
        md = Markdown(text)
        console.print(md)


def render_user_message(console: Console, text: str, *, clear_echo: bool = False) -> None:
    """ユーザー入力メッセージを枠付きで表示

    clear_echo=True の場合、input() によるエコー行を消去してから Panel を表示する。
    sequential モードで使用。
    """
    if clear_echo and not console.no_color:
        _ensure_windows_vt()
        # input() のエコー行（プロンプト+入力テキスト）を上書き消去
        sys.stdout.write("\033[1A\033[2K\r")
        sys.stdout.flush()

    if console.no_color:
        console.print(f"> {text}")
        console.print()
        return

    panel = Panel(
        text,
        border_style=_cli_theme.border,
        style=f"on {_cli_theme.bg_surface}",
        expand=True,
        padding=(0, 1),
    )
    console.print(panel)


def render_response_start(console: Console) -> None:
    """応答開始（空行のみ）"""
    console.print()


def render_separator(console: Console) -> None:
    """会話・コマンドの区切り横線を表示"""
    if console.no_color:
        console.print("---")
    else:
        console.print()
        console.rule(characters="─", style=_cli_theme.accent)


def format_status_text(
    used: int,
    limit: int,
    elapsed: float,
    context_files: list[str] | None = None,
    *,
    ttft_sec: float | None = None,
) -> str:
    """ステータスラインのプレーンテキストを生成

    split レイアウトのステータスバーと render_status_line の共通ロジック。
    """
    parts = []
    tok_str = f"{used:,} / {limit:,} tokens"
    parts.append(tok_str)
    if elapsed > 0:
        parts.append(f"{elapsed:.1f}s")
    if ttft_sec is not None:
        parts.append(f"TTFT {ttft_sec:.2f}s")
    if context_files:
        names = ", ".join(Path(p).name for p in context_files[:3])
        if len(context_files) > 3:
            names += f" +{len(context_files) - 3}"
        parts.append(f"context: {names}")

    return " \u00b7 ".join(parts)


def render_status_line(
    console: Console,
    used: int,
    limit: int,
    elapsed: float,
    context_files: list[str] | None = None,
    *,
    ttft_sec: float | None = None,
) -> None:
    """応答完了後のステータスラ���ンを表示"""
    line = format_status_text(used, limit, elapsed, context_files, ttft_sec=ttft_sec)
    pct = int(used / limit * 100) if limit > 0 else 0

    if console.no_color:
        console.print(line)
    else:
        tok_color = _token_color(pct)
        console.print(f"[{tok_color}]{line}[/{tok_color}]")
    console.print()


def render_diff(console: Console, diff_text: str) -> None:
    """diff をシンタックスハイライト付きで表示（Panel 枠なし）"""
    if console.no_color:
        console.print(diff_text)
    else:
        syntax = Syntax(diff_text, "diff", theme=_cli_theme.code_theme, line_numbers=False)
        console.print(syntax)


def render_error(
    console: Console,
    message: str,
    code: str | None = None,
    *,
    level: str = "error",
) -> None:
    """レベル付きエラー・警告・ヒントメッセージを表示

    level: "error" (bold red), "warning" (yellow), "hint" (dim cyan)
    """
    prefix = f"[{code}] " if code else ""
    styles = {
        "error": ("bold red", "Error"),
        "warning": ("yellow", "Warning"),
        "hint": ("dim cyan", "Hint"),
    }
    style, label = styles.get(level, styles["error"])
    if console.no_color:
        console.print(f"{label}: {prefix}{message}", style=None)
    else:
        console.print(f"[{style}]{label}:[/{style}] {prefix}{message}")


def render_info(console: Console, message: str) -> None:
    """情報メッセージを表示"""
    if console.no_color:
        console.print(message)
    else:
        console.print(f"[dim]{message}[/dim]")


def render_table(console: Console, data: list[dict], headers: list[str]) -> None:
    """テーブルを表示"""
    from rich import box

    if console.no_color:
        # プレーンテキスト
        header_line = "\t".join(headers)
        console.print(header_line)
        for row in data:
            console.print("\t".join(str(row.get(h, "")) for h in headers))
    else:
        table = Table(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE,
            border_style="dim",
        )
        for h in headers:
            table.add_column(h)
        for row in data:
            table.add_row(*[str(row.get(h, "")) for h in headers])
        console.print(table)


_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_vt_enabled = False


def _ensure_windows_vt() -> None:
    """Windows で ANSI エスケープシーケンスを有効化

    sys.stdout.write() で直接 ANSI コードを使う場合に必要。
    Rich の Console は内部で処理するが、直接書き込みには効かない。
    """
    global _vt_enabled
    if _vt_enabled or os.name != "nt":
        return
    _vt_enabled = True
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode)
    except (AttributeError, OSError):
        pass


def render_step(console: Console, step: dict) -> None:
    """SSE ステップフレーム（完了・失敗）を表示

    running ステップは StepSpinner で処理するため、ここでは done/failed のみ。

    step: {"type": "rag_search"|..., "detail": "...", "status": "running"|"done"|"failed",
           "elapsed_ms": 1200 (optional)}
    """
    from rich.markup import escape

    step_type = step.get("type", "")
    detail = escape(step.get("detail", ""))
    status = step.get("status", "running")
    elapsed_ms = step.get("elapsed_ms")
    time_str = f" ({elapsed_ms / 1000:.1f}s)" if elapsed_ms else ""

    if console.no_color:
        # split レイアウト用: _style_line がステータスで色分けできるよう [status] を付与
        console.print(
            f"[{status}] ● {step_type}: {detail}{time_str}",
            markup=False,
        )
    else:
        if status == "done":
            console.print(
                f"[{_cli_theme.step_done}]  ● {step_type}: {detail}[/{_cli_theme.step_done}][dim]{time_str}[/dim]"
            )
        elif status == "failed":
            console.print(
                f"[{_cli_theme.step_failed}]  ● {step_type}: {detail}[/{_cli_theme.step_failed}][dim]{time_str}[/dim]"
            )
        else:
            console.print(f"[dim]  ● {step_type}: {detail}[/dim]")


class StepSpinner:
    """running ステップのスピナーアニメーション

    asyncio タスクでバックグラウンドにスピナーを回し、
    完了フレーム受信時にタスクを停止して確定表示に置換する。
    """

    def __init__(self, console: Console, interval: float = 0.1):
        self._console = console
        self._interval = interval
        self._task: asyncio.Task | None = None
        self._label = ""
        self._idx = 0
        self._start_time = 0.0

    def start(self, step_type: str, detail: str) -> None:
        """running スピナーを開始"""
        import asyncio
        self._label = f"{step_type}: {detail}"
        self._idx = 0
        self._start_time = time.monotonic()
        self._task = asyncio.create_task(self._animate())

    async def stop(self) -> None:
        """スピナーを停止し、現在行をクリア"""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # スピナー行をクリア
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    async def _animate(self) -> None:
        """スピナーアニメーションループ"""
        if not self._console.no_color:
            _ensure_windows_vt()
        try:
            while True:
                elapsed = time.monotonic() - self._start_time
                ch = _SPINNER_CHARS[self._idx % len(_SPINNER_CHARS)]
                self._idx += 1
                line = f"  {ch} {self._label} ({elapsed:.0f}s)"
                if self._console.no_color:
                    sys.stdout.write(f"\r{line}")
                else:
                    sys.stdout.write(f"\r\033[2m{line}\033[0m")
                sys.stdout.flush()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            raise


def render_startup_result(console: Console, result) -> None:
    """起動前提条件チェックの結果を表示"""
    from backend.free.cli.startup_checks import CheckStatus

    level_str = result.level.value
    code_prefix = f"[{result.code}] " if result.code else ""

    if result.status == CheckStatus.OK:
        if console.no_color:
            console.print(f"[{level_str}] OK  {result.message}")
        else:
            console.print(f"[dim][{level_str}][/dim] [green]OK[/green] {result.message}")
    elif result.status == CheckStatus.WARN:
        if console.no_color:
            console.print(f"[{level_str}] WARN {code_prefix}{result.message}")
        else:
            console.print(
                f"[dim][{level_str}][/dim] [yellow]WARN[/yellow] {code_prefix}{result.message}"
            )
    elif result.status == CheckStatus.FAIL:
        if console.no_color:
            console.print(f"[{level_str}] FAIL {code_prefix}{result.message}")
        else:
            console.print(
                f"[dim][{level_str}][/dim] [bold red]FAIL[/bold red] {code_prefix}{result.message}"
            )


def render_welcome(
    console: Console,
    version: str,
    instance_name: str,
    edition: str = "free",
) -> None:
    """起動時: 画面クリア → タイトル+バージョン → ヒント"""

    # 画面クリア
    console.clear()

    name_part = f" ({instance_name})" if instance_name else ""
    title = f"evoref {edition} v{version}{name_part}"

    if console.no_color:
        console.print(title)
        return

    console.print()
    console.print(f"[bold {_cli_theme.accent}]evoref {edition}  v{version}[/bold {_cli_theme.accent}] [dim]{name_part}[/dim]")


def render_welcome_hint(console: Console) -> None:
    """ヒント行を表示（モデル情報の後に呼び出す）"""
    from backend.i18n_helper import msg

    if console.no_color:
        console.print(msg("cli.welcome_hint"))
        console.print()
        return

    console.print(f"[dim]{msg('cli.welcome_hint')}[/dim]")
    console.print()


@dataclass
class ModelInfoItem:
    """モデル情報の1行分"""
    label: str
    name: str
    connected: bool


def render_model_info(
    console: Console,
    models: list[ModelInfoItem],
    context_size: int | None = None,
) -> None:
    """起動時モデル一覧を表示（ウェルカムバナー直下）"""
    if not models and context_size is None:
        return

    _CLR = _cli_theme.info
    _OK = _cli_theme.success
    _NG = _cli_theme.error

    if console.no_color:
        for m in models:
            status = "ok" if m.connected else "---"
            console.print(f"  {m.label}: {m.name}  \\[{status}]")
        if context_size is not None:
            console.print(f"  context: {context_size:,}")
        console.print()
        return

    for m in models:
        if m.connected:
            status_tag = f"[{_OK}]ok[/{_OK}]"
        else:
            status_tag = f"[{_NG}]---[/{_NG}]"
        console.print(f"  [{_CLR}]{m.label}:[/{_CLR}] [{_CLR}]{m.name}[/{_CLR}]  {status_tag}")
    if context_size is not None:
        console.print(f"  [{_CLR}]context:[/{_CLR}] [{_CLR}]{context_size:,}[/{_CLR}]")
    console.print()


def render_startup_status(
    servers: dict[str, bool],
    elapsed: float,
    spinner_idx: int,
    *,
    no_color: bool = False,
) -> str:
    """起動スピナーの1行文字列を生成（\\r で上書き表示用）

    Args:
        servers: {name: connected} のステータスマップ
        elapsed: 経過秒数
        spinner_idx: スピナー文字インデックス
        no_color: カラー無効フラグ
    """
    if not no_color:
        _ensure_windows_vt()
    ch = _SPINNER_CHARS[spinner_idx % len(_SPINNER_CHARS)]

    parts = []
    for name, ok in servers.items():
        if no_color:
            parts.append(f"{name}:{'ok' if ok else '---'}")
        else:
            if ok:
                parts.append(f"\033[32m{name}:ok\033[0m")
            else:
                parts.append(f"\033[31m{name}:---\033[0m")

    status_str = "  ".join(parts)
    if no_color:
        return f"  {ch} {status_str}  ({elapsed:.0f}s)"
    return f"  \033[2m{ch}\033[0m {status_str}  \033[2m({elapsed:.0f}s)\033[0m"


def clear_spinner_line() -> None:
    """スピナー行をクリア"""
    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.flush()


def render_help(console: Console) -> None:
    """ヘルプをインラインリストで表示"""
    from backend.i18n_helper import msg

    commands = [
        ("/help", msg("cli.help_help")),
        ("/status", msg("cli.help_status")),
        ("/file <path>", msg("cli.help_file")),
        ("/file", msg("cli.help_file_list")),
        ("/export <path> [--all|--last N]", msg("cli.help_export")),
        ("/history [action]", msg("cli.help_history")),
        ("/cartridge [action]", msg("cli.help_cartridge")),
        ("/theme [action]", msg("cli.help_theme")),
        ("/migrate-model [options]", msg("cli.help_migrate_model")),
        ("/web <url>", msg("cli.help_web")),
        ("/learn [status|level1|full]", msg("cli.help_learn")),
        ("/reindex [--dry-run] [--cartridge <id>]", msg("cli.help_reindex")),
        ("/page", msg("cli.help_page")),
        ("/clear", msg("cli.help_clear")),
        ("/save [name]", msg("cli.help_save")),
        ("/load [name]", msg("cli.help_load")),
        ("/load --history <id>", msg("cli.help_load_history")),
        ("/load --history --latest", msg("cli.help_load_latest")),
        ("/pin <text>", msg("cli.help_pin")),
        ("/unpin <id> [--force]", msg("cli.help_unpin")),
        ("/pinned", msg("cli.help_pinned")),
        ("/private on|off|status", msg("cli.help_private")),
        ("/exit", msg("cli.help_exit")),
    ]

    from rich.markup import escape

    max_cmd = max(len(c) for c, _ in commands)
    lines = [f"{c:<{max_cmd}}  {d}" for c, d in commands]
    lines.append("")
    lines.append(msg("cli.help_ctrl_c"))
    help_text = "\n".join(lines)

    if console.no_color:
        console.print(msg("cli.help_header"))
        console.print(help_text)
    else:
        console.print(f"\n[dim]{msg('cli.help_header')}[/dim]")
        console.print(escape(help_text))
        console.print()


def render_shell_out_request(console: Console, cmd: str) -> None:
    """シェルアウト要求をインライン表示"""
    from backend.i18n_helper import msg
    if console.no_color:
        console.print(f"\nShell: {cmd}")
        console.print(f"{msg('cli.shell_out_confirm')}", end="")
    else:
        console.print(f"\n[bold]Shell:[/bold] {cmd}")


def render_shell_out_result(console: Console, result) -> None:
    """シェルアウト実行結果を表示"""
    from backend.i18n_helper import msg
    if not result.executed:
        render_error(console, msg("cli.shell_out_failed", error=result.error))
    elif result.returncode == 0:
        if console.no_color:
            console.print(msg("cli.shell_out_done", code=0))
        else:
            console.print(f"[green]{msg('cli.shell_out_done', code=0)}[/green]")
    else:
        if console.no_color:
            console.print(msg("cli.shell_out_done", code=result.returncode))
        else:
            console.print(
                f"[yellow]{msg('cli.shell_out_done', code=result.returncode)}[/yellow]"
            )


def render_diff_applied(console: Console, file_path: str) -> None:
    """diff 適用成功メッセージを表示"""
    from backend.i18n_helper import msg
    if console.no_color:
        console.print(f"Applied: {file_path}")
    else:
        console.print(f"[green]{msg('cli.diff_applied', path=file_path)}[/green]")


def render_diff_result(console: Console, success: bool, message: str) -> None:
    """diff 適用結果を表示"""
    if success:
        if console.no_color:
            console.print(f"OK: {message}")
        else:
            console.print(f"[green]{message}[/green]")
    else:
        if console.no_color:
            console.print(f"Error: {message}")
        else:
            console.print(f"[bold red]Error:[/bold red] {message}")


def render_diff_prompt_path(console: Console) -> str:  # noqa: ARG001
    """diff のファイルパスが不明な場合にユーザーに尋ねる"""
    from backend.i18n_helper import msg
    try:
        path = input(msg("cli.diff_enter_path")).strip()
        return path
    except (EOFError, KeyboardInterrupt):
        return ""


def format_prompt(*, no_color: bool = False) -> str:  # noqa: ARG001
    """プロンプト文字列を生成"""
    return _cli_theme.prompt_marker
