"""CLI split レイアウト: textual ベースの TUI

textual App による画面分割アーキテクチャ。
対話モード（evoref create / evoref --mode chat）でのみ使用し、
非対話モードでは従来の逐次出力を維持する。

RichLog で出力エリアの表示と Rich ネイティブレンダリング（Syntax, Markdown 等）
を提供し、_style_line() で行内容に応じた配色を行う。
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.text import Text

from backend.log_config import get_logger

logger = get_logger("cli.split_layout")

# textual は optional — ImportError 時は split レイアウト無効
try:
    from textual.widgets import RichLog

    from backend.free.cli.textual_app import EvorefApp

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


def is_split_available() -> bool:
    """split レイアウトが利用可能か"""
    return HAS_TEXTUAL


# ── 出力エリアの行内容ベース着色 ──


def _style_line(line: str) -> Text | str:
    """出力行を内容パターンでスタイル適用（CLITheme 経由）

    配色ルール:
    - "> ..."          → bold accent_light (ユーザー入力エコー)
    - "Error: ..."     → error (エラー)
    - "Warning: ..."   → warning (警告)
    - "Hint: ..."      → hint (ヒント)
    - "[done] ..."     → step_done (ステップ完了)
    - "[failed] ..."   → step_failed (ステップ失敗)
    - "[running] ..."  → step_running (ステップ実行中)
    - "Applied: ..."   → success (diff 適用)
    - diff ヘッダー    → bold accent_light / accent / step_failed / step_done
    - その他           → デフォルト  (AI 応答テキスト)
    """
    from backend.free.cli.renderer import get_cli_theme
    theme = get_cli_theme()
    stripped = line.lstrip()

    if not stripped:
        return ""

    # ユーザー入力エコー
    if stripped.startswith("> "):
        return Text(line, style=f"bold {theme.accent_light}")

    # エラー・警告・ヒント
    if "Error:" in stripped:
        return Text(line, style=theme.error)
    if "Warning:" in stripped:
        return Text(line, style=theme.warning)
    if "Hint:" in stripped:
        return Text(line, style=theme.hint)

    # ステップ表示（[status] ● マーカー）
    if stripped.startswith("[done]") or stripped.startswith("[failed]") or stripped.startswith("[running]"):
        # [status] プレフィックスを除去して ● 付きテキストのみ表示
        display_line = line
        for prefix in ("[done] ", "[failed] ", "[running] "):
            if stripped.startswith(prefix):
                display_line = "  " + stripped[len(prefix):]
                break
        if "[done]" in stripped:
            return Text(display_line, style=theme.step_done)
        if "[failed]" in stripped:
            return Text(display_line, style=theme.step_failed)
        return Text(display_line, style=theme.step_running)

    # カラーモード時のステップ（Rich markup なしでもマッチ）
    if line.startswith("  ") and ": " in stripped and not stripped.startswith("context:"):
        if len(stripped) < 80 and ("s)" in stripped or stripped.endswith("...")):
            return Text(line, style=theme.step_done)

    # diff 適用成功
    if stripped.startswith("Applied:") or stripped.startswith("OK:"):
        return Text(line, style=theme.success)

    # diff ヘッダー
    if stripped.startswith("---") or stripped.startswith("+++"):
        return Text(line, style=f"bold {theme.accent_light}")
    if stripped.startswith("@@"):
        return Text(line, style=theme.accent)
    if stripped.startswith("-") and not stripped.startswith("---"):
        return Text(line, style=theme.step_failed)
    if stripped.startswith("+") and not stripped.startswith("+++"):
        return Text(line, style=theme.step_done)

    # evoref ウェルカム・システムメッセージ
    if stripped.startswith("evoref") or stripped.startswith("/"):
        return Text(line, style=theme.text_muted)

    return line


class TextualOutputAdapter:
    """Rich Console の出力を RichLog にリダイレクト

    行単位でバッファリングし、完全な行（改行で終わる）のみ RichLog に書き込む。
    """

    def __init__(self, get_richlog) -> None:
        self._get_richlog = get_richlog
        self._line_buffer = ""
        self._pending: list = []

    def write(self, text: str) -> int:
        self._line_buffer += text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            styled = _style_line(line)
            self._write_item(styled)
        return len(text)

    def _write_item(self, item) -> None:
        richlog = self._get_richlog()
        if richlog is not None:
            # 保留中のアイテムを先にフラッシュ
            if self._pending:
                for pending_item in self._pending:
                    richlog.write(pending_item)
                self._pending.clear()
            richlog.write(item)
        else:
            self._pending.append(item)

    def flush(self) -> None:
        """通常の flush は何もしない（改行時のみ書き込み）"""
        pass

    def flush_pending(self) -> None:
        """保留中のアイテムをフラッシュ（マウント後に呼び出す）"""
        richlog = self._get_richlog()
        if richlog and self._pending:
            for item in self._pending:
                richlog.write(item)
            self._pending.clear()

    def flush_all(self) -> None:
        """残りのバッファを強制書き込み（部分行も含む）"""
        if self._line_buffer:
            styled = _style_line(self._line_buffer)
            self._write_item(styled)
            self._line_buffer = ""

    @property
    def encoding(self) -> str:
        return "utf-8"


class SplitLayout:
    """textual ベースの split レイアウトマネージャ（Claude Code 風）

    画面構成:
    ┌──────────────────────────────────────────┐
    │                                          │
    │  応答テキスト（スクロール可能・着色）     │  ← RichLog (1fr)
    │                                          │
    │ ❯ ユーザー入力                           │  ← Input (1行)
    │ ⠋ thinking · tokens                     │  ← ステータス (1行, dim)
    └──────────────────────────────────────────┘
    """

    def __init__(
        self,
        *,
        instance_name: str = "evoref",
        version: str = "",
        cli_theme=None,
    ) -> None:
        if not HAS_TEXTUAL:
            raise RuntimeError("textual is required for split layout")

        self._app = EvorefApp(
            instance_name=instance_name,
            version=version,
            cli_theme=cli_theme,
        )
        self._app._on_layout_mount = self._on_mount
        self._last_status_suffix: Text | str = ""
        self._pre_mount_items: list = []
        self._mounted = False
        self._adapter: TextualOutputAdapter | None = None
        self._pending_status: tuple | None = None

    def _on_mount(self) -> None:
        """App マウント完了時のコールバック"""
        self._mounted = True
        richlog = self._get_richlog()
        if richlog:
            for item in self._pre_mount_items:
                richlog.write(item)
            self._pre_mount_items.clear()
        # Adapter の保留アイテムもフラッシュ
        if self._adapter:
            self._adapter.flush_pending()
        # 保留中のステータス更新を適用
        if self._pending_status:
            used, limit, context_files = self._pending_status
            self._pending_status = None
            self.update_status(used, limit, context_files)

    @property
    def app(self):
        return self._app

    @property
    def _should_exit(self) -> bool:
        return self._app._should_exit

    @_should_exit.setter
    def _should_exit(self, value: bool) -> None:
        self._app._should_exit = value

    # ── RichLog アクセス ──

    def _get_richlog(self):
        """RichLog ウィジェットを取得（マウント前は None）"""
        try:
            return self._app.query_one("#chat-output", RichLog)
        except Exception:
            return None

    # ── 出力エリア ──

    def append_output(self, text: str) -> None:
        """テキストを出力エリアに追記（_style_line で自動着色）"""
        lines = text.split("\n")
        # 末尾の空要素（改行によるもの）を除去
        if text.endswith("\n") and lines and lines[-1] == "":
            lines = lines[:-1]

        for line in lines:
            styled = _style_line(line) if line else ""
            if self._mounted:
                richlog = self._get_richlog()
                if richlog:
                    richlog.write(styled)
            else:
                self._pre_mount_items.append(styled)

    def clear_output(self) -> None:
        """出力エリアをクリア"""
        richlog = self._get_richlog()
        if richlog:
            richlog.clear()

    def write_rich(self, renderable) -> None:
        """Rich renderable を出力エリアに直接書き込み

        Syntax, Markdown, Rule 等の Rich オブジェクトをネイティブ表示。
        """
        if self._mounted:
            richlog = self._get_richlog()
            if richlog:
                richlog.write(renderable)
        else:
            self._pre_mount_items.append(renderable)

    # ── ステータスバー ──

    def update_status(
        self,
        used: int,
        limit: int,
        context_files: list[str] | None = None,
    ) -> None:
        """トークン使用量でステータスバーを更新"""
        if not self._mounted:
            self._pending_status = (used, limit, context_files)
            return

        from backend.free.cli.renderer import get_cli_theme
        theme = get_cli_theme()

        self._stop_spinner()
        pct = int(used / limit * 100) if limit > 0 else 0

        if pct < 60:
            token_style = theme.token_low
        elif pct < 85:
            token_style = theme.token_mid
        else:
            token_style = theme.token_high

        suffix = Text()
        suffix.append(f"{used:,} / {limit:,} tokens", style=token_style)

        if context_files:
            names = ", ".join(Path(p).name for p in context_files[:3])
            if len(context_files) > 3:
                names += f" +{len(context_files) - 3}"
            suffix.append("  ┊  ", style=theme.separator)
            suffix.append(f"context: {names}", style=theme.text_muted)

        self._last_status_suffix = suffix
        self._app.set_status_suffix(suffix)

    def _stop_spinner(self) -> None:
        """スピナーアニメーションを停止"""
        self._app.stop_thinking()
        self._app.set_code_spinner("")

    def update_status_idle(self) -> None:
        """アイドル状態に戻す（トークン情報を保持）"""
        self._app.stop_thinking()
        self._app.set_status_suffix(self._last_status_suffix)

    def update_status_thinking(self) -> None:
        """thinking スピナーを開始"""
        self._app.set_status_suffix(self._last_status_suffix)
        self._app.start_thinking()

    def update_code_spinner(self, lang: str, lines: int) -> None:
        """コード生成中のスピナー表示を更新"""
        lang_label = f" ({lang})" if lang else ""
        lines_info = f" ({lines} lines)" if lines > 0 else ""
        self._app.set_code_spinner(
            f"コード生成中{lang_label}{lines_info}...",
        )

    def stop_code_spinner(self) -> None:
        """コード生成スピナーを停止"""
        self._app.set_code_spinner("")

    # ── ストリーミング制御 ──

    def set_streaming(self, streaming: bool) -> None:
        self._app._streaming = streaming
        self._app._interrupt_streaming = False

    def check_interrupt(self) -> bool:
        if self._app._interrupt_streaming:
            self._app._interrupt_streaming = False
            return True
        return False

    # ── ユーティリティ ──

    def invalidate(self) -> None:
        """UI 再描画をトリガー（textual では自動的に行われるため基本的に不要）"""
        pass

    async def get_input(self) -> str:
        """ユーザー入力を待機"""
        self._app._input_ready.clear()
        self._app._ctrl_c_count = 0
        await self._app._input_ready.wait()
        if self._app._should_exit:
            raise EOFError("User requested exit")
        text = self._app._input_text
        self._app._input_text = ""
        return text

    def create_console(self) -> Console:
        """split レイアウトの出力エリアに書き込む Rich Console を作成

        TextualOutputAdapter が行単位でバッファリングし、
        _style_line() で着色して RichLog に書き込む。
        """
        self._adapter = TextualOutputAdapter(self._get_richlog)
        return Console(
            file=self._adapter,
            no_color=True,
            width=9999,
            highlight=False,
        )
