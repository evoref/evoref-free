"""textual ベースの TUI アプリケーション（Claude Code 風レイアウト）

EvorefApp: 対話モード用のフルスクリーン TUI
Widget 構成: RichLog (出力) + Input (入力) + Static (ステータス)
ヘッダーなし、入力ボーダーなし、ステータスは最小限の1行
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input, RichLog, Static

from rich.text import Text

from backend.free.cli.cli_theme import CLITheme


def build_textual_css(theme: CLITheme) -> str:
    """CLITheme から textual 用 CSS 文字列を生成"""
    return f"""
    Screen {{
        background: {theme.bg_primary};
    }}

    #chat-output {{
        height: 1fr;
        padding: 0 0;
        scrollbar-size-vertical: 1;
        scrollbar-color: {theme.border};
        scrollbar-color-hover: {theme.accent_dark};
        scrollbar-color-active: {theme.accent};
        scrollbar-background: {theme.bg_primary};
        scrollbar-background-hover: {theme.bg_surface};
        scrollbar-background-active: {theme.bg_surface};
        scrollbar-corner-color: {theme.bg_primary};
    }}

    #input-bar {{
        height: 1;
        padding: 0 0;
        margin: 0;
        background: {theme.bg_primary};
    }}

    #prompt-label {{
        width: 2;
        height: 1;
        color: {theme.accent};
        text-style: bold;
    }}

    #user-input {{
        width: 1fr;
        height: 1;
        border: none;
        padding: 0 0;
        background: {theme.bg_primary};
        color: {theme.text_glow};
    }}

    #user-input:focus {{
        border: none;
        background: {theme.bg_primary};
        color: {theme.text_glow};
    }}

    #status-bar {{
        height: 1;
        background: {theme.bg_primary};
        color: {theme.text_muted};
        padding: 0 0;
        text-opacity: 70%;
    }}
    """


class EvorefApp(App):
    """evoref 対話モード用 TUI アプリケーション（Claude Code 風）

    Widget 構成:
    ┌──────────────────────────────────────────┐
    │                                          │
    │  チャット出力エリア                      │  ← RichLog (1fr)
    │                                          │
    │ ❯ ユーザー入力                           │  ← Input (1行)
    │ ⠋ thinking · 1,000 / 4,096 tokens       │  ← ステータス (1行, dim)
    └──────────────────────────────────────────┘
    """

    # デフォルト CSS（テーマ未指定時のフォールバック）
    CSS = build_textual_css(CLITheme.default())

    BINDINGS = [
        # Ctrl 系: Input ウィジェットより優先する必要があるため priority=True
        Binding("ctrl+c", "interrupt", show=False, priority=True),
        Binding("ctrl+l", "clear_output", show=False, priority=True),
        Binding("ctrl+d", "request_exit", show=False, priority=True),
        # スクロール: 将来の拡張を妨げないため priority なし
        Binding("pageup", "page_up", show=False),
        Binding("pagedown", "page_down", show=False),
        Binding("shift+up", "scroll_up", show=False),
        Binding("shift+down", "scroll_down", show=False),
    ]

    def __init__(
        self,
        *,
        instance_name: str = "evoref",
        version: str = "",
        cli_theme: CLITheme | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cli_theme = cli_theme or CLITheme.default()
        self._input_ready = asyncio.Event()
        self._input_text = ""
        self._streaming = False
        self._interrupt_streaming = False
        self._should_exit = False
        self._ctrl_c_count = 0
        self._thinking = False
        self._spinner_frame = 0
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_timer = None
        self._mounted_event = asyncio.Event()
        self._on_layout_mount = None

        label = instance_name
        if version:
            label += f" v{version}"
        self._name_label = label
        self._status_suffix: Text | str = ""
        self._code_spinner_text = ""

        # テーマ指定時は CSS を動的生成
        if cli_theme is not None:
            self.CSS = build_textual_css(cli_theme)

    def compose(self) -> ComposeResult:
        yield RichLog(
            id="chat-output",
            max_lines=10000,
            wrap=True,
            highlight=False,
            markup=False,
        )
        with Horizontal(id="input-bar"):
            yield Static(self._cli_theme.prompt_marker, id="prompt-label")
            yield Input(id="user-input", placeholder="Type a message...")
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        self.query_one("#user-input", Input).focus()
        if self._on_layout_mount:
            self._on_layout_mount()
        self._mounted_event.set()

    def on_unmount(self) -> None:
        """App 終了時のクリーンアップ"""
        if self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._input_text = event.value
        event.input.clear()
        self._ctrl_c_count = 0
        self._input_ready.set()

    # ── Actions ──

    def action_interrupt(self) -> None:
        """Ctrl+C: ストリーミング中断 or 2回連続で終了"""
        if self._streaming:
            self._interrupt_streaming = True
            return
        self._ctrl_c_count += 1
        if self._ctrl_c_count >= 2:
            self._should_exit = True
            self._input_ready.set()
            return
        try:
            richlog = self.query_one("#chat-output", RichLog)
            richlog.write(
                Text("\n(Ctrl+C: press again to exit)\n", style="dim"),
            )
        except Exception:
            pass

    def action_clear_output(self) -> None:
        """Ctrl+L: 出力エリアをクリア"""
        try:
            self.query_one("#chat-output", RichLog).clear()
        except Exception:
            pass

    def action_request_exit(self) -> None:
        """Ctrl+D: 終了要求"""
        self._should_exit = True
        self._input_ready.set()

    def action_page_up(self) -> None:
        try:
            self.query_one("#chat-output", RichLog).scroll_page_up(
                animate=False,
            )
        except Exception:
            pass

    def action_page_down(self) -> None:
        try:
            self.query_one("#chat-output", RichLog).scroll_page_down(
                animate=False,
            )
        except Exception:
            pass

    def action_scroll_up(self) -> None:
        try:
            rl = self.query_one("#chat-output", RichLog)
            rl.scroll_relative(y=-3, animate=False)
        except Exception:
            pass

    def action_scroll_down(self) -> None:
        try:
            rl = self.query_one("#chat-output", RichLog)
            rl.scroll_relative(y=3, animate=False)
        except Exception:
            pass

    # ── Status Bar ──

    def set_status_suffix(self, text: Text | str) -> None:
        """ステータスバーの情報部分（トークン数等）を設定"""
        self._status_suffix = text
        self._refresh_status()

    def set_code_spinner(self, text: str) -> None:
        """コード生成中スピナーのテキストを設定"""
        self._code_spinner_text = text
        self._refresh_status()

    def start_thinking(self) -> None:
        """thinking スピナーを開始"""
        self._thinking = True
        self._spinner_frame = 0
        if not self._spinner_timer:
            self._spinner_timer = self.set_interval(
                0.1, self._tick_spinner,
            )
        self._refresh_status()

    def stop_thinking(self) -> None:
        """thinking スピナーを停止"""
        self._thinking = False
        if self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None
        self._refresh_status()

    def _tick_spinner(self) -> None:
        self._spinner_frame += 1
        self._refresh_status()

    def _refresh_status(self) -> None:
        """ステータスバーを再描画"""
        t = self._cli_theme
        status = Text()

        if self._thinking:
            char = self._spinner_chars[
                self._spinner_frame % len(self._spinner_chars)
            ]
            status.append(f"  {char} ", style=t.accent)
            status.append("thinking", style=t.accent)

        if self._status_suffix:
            if len(status) > 0:
                status.append("  ·  ", style=t.border)
            else:
                status.append("  ", style=t.text_muted)
            if isinstance(self._status_suffix, Text):
                status.append(self._status_suffix)
            else:
                status.append(str(self._status_suffix))

        if self._code_spinner_text:
            if len(status) > 0:
                status.append("  ·  ", style=t.border)
            else:
                status.append("  ", style=t.text_muted)
            status.append(self._code_spinner_text, style=t.text_muted)

        try:
            self.query_one("#status-bar", Static).update(status)
        except Exception:
            pass

    def apply_theme(self, theme: CLITheme) -> None:
        """テーマを動的に再適用（/theme activate 時に使用）"""
        self._cli_theme = theme
        self.stylesheet.set(build_textual_css(theme))
        try:
            self.query_one("#prompt-label", Static).update(theme.prompt_marker)
        except Exception:
            pass
        self.refresh(layout=True)
