"""ストリーミング Markdown レンダラー

トークンを逐次受け取り、コードブロック（``` fenced blocks）を検出して
バッファリングし、閉じタグ到着後にハイライト付きで出力する。

split layout モード（textual）:
- コードブロック中はステータスバーにスピナー表示
- コードブロックは RichLog に Syntax オブジェクトとしてネイティブ表示
- 応答完了マーカーは rich.rule.Rule で表示

非 split layout（sequential）モード:
- コードブロック中は sys.stdout.write("\\r") でスピナーを上書き表示
- コードブロックは rich.syntax.Syntax でシンタックスハイライト出力
"""

from __future__ import annotations

import itertools
import re
import sys

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text


# コードブロック開始パターン: ```lang or ``` （インデント許容）
_FENCE_OPEN = re.compile(r"^(\s*)```(\w*)\s*$")
# コードブロック終了パターン: ``` （インデント許容）
_FENCE_CLOSE = re.compile(r"^\s*```\s*$")

# Markdown インライン書式パターン
_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_UNORDERED_LIST = re.compile(r"^(\s*)[-*]\s+(.+)$")
_ORDERED_LIST = re.compile(r"^(\s*)(\d+)\.\s+(.+)$")
_BLOCKQUOTE = re.compile(r"^>\s?(.*)")
# 水平線: ---, ***---, **--- 等（LLM が ** を付けるケースも許容）
_HORIZONTAL_RULE = re.compile(r"^\*{0,3}---+\s*$")


def _md_line_to_text(line: str) -> Text:
    """Markdown 1行を Rich Text オブジェクトに変換する（コードブロック外のテキスト用）

    マークアップ文字列ではなく Text オブジェクトを返すことで、
    テキスト内の ``\\``  ``[`` ``]`` 等が Rich パーサーに干渉する問題を根本的に回避する。
    """
    stripped = line.rstrip("\n\r")

    # 空行
    if not stripped:
        return Text(line)

    # 水平線
    if _HORIZONTAL_RULE.match(stripped):
        return Text("───\n")

    # 見出し
    m = _HEADING.match(stripped)
    if m:
        t = _md_inline_to_text(m.group(2))
        t.stylize("bold")
        t.append("\n")
        return t

    # 引用
    m = _BLOCKQUOTE.match(stripped)
    if m:
        t = Text("│ ", style="dim")
        inner = _md_inline_to_text(m.group(1))
        inner.stylize("dim")
        t.append_text(inner)
        t.append("\n")
        return t

    # 番号なしリスト
    m = _UNORDERED_LIST.match(stripped)
    if m:
        t = Text(f"{m.group(1)}  • ")
        t.append_text(_md_inline_to_text(m.group(2)))
        t.append("\n")
        return t

    # 番号付きリスト
    m = _ORDERED_LIST.match(stripped)
    if m:
        t = Text(f"{m.group(1)}  {m.group(2)}. ")
        t.append_text(_md_inline_to_text(m.group(3)))
        t.append("\n")
        return t

    # 通常テキスト: インライン書式のみ変換
    t = _md_inline_to_text(stripped)
    t.append("\n")
    return t


def _md_inline_to_text(text: str) -> Text:
    """Markdown インライン書式を Rich Text オブジェクトに変換

    Text オブジェクトに直接スタイルを適用するため、
    ``\\[`` や ``\\`` によるマークアップ破壊が原理的に発生しない。
    """
    # インライン書式のスパンを収集（位置順）
    spans: list[tuple[int, int, str, str]] = []  # (start, end, content, style)

    for m in _INLINE_CODE.finditer(text):
        spans.append((m.start(), m.end(), m.group(1), "bold cyan"))

    for m in _BOLD.finditer(text):
        if not any(s[0] <= m.start() < s[1] for s in spans):
            spans.append((m.start(), m.end(), m.group(1), "bold"))

    for m in _ITALIC.finditer(text):
        if not any(s[0] <= m.start() < s[1] for s in spans):
            spans.append((m.start(), m.end(), m.group(1), "italic"))

    if not spans:
        return Text(text)

    spans.sort(key=lambda s: s[0])

    result = Text()
    pos = 0
    for start, end, content, style in spans:
        if start > pos:
            result.append(text[pos:start])
        result.append(content, style=style)
        pos = end
    if pos < len(text):
        result.append(text[pos:])

    return result

# スピナーフレーム（非 split layout 用）
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class StreamingMarkdownRenderer:
    """ストリーミング中のコードブロック検出 + ハイライト出力

    split layout モードでは SplitLayout のメソッドを使用し、
    非 split layout モードでは sys.stdout.write でインラインスピナーを表示する。
    """

    def __init__(self, console: Console, split_layout=None):
        """
        Args:
            console: Rich Console（出力先）
            split_layout: SplitLayout インスタンス（split layout モード時のみ）
        """
        from backend.free.cli.renderer import get_cli_theme
        self._console = console
        self._split = split_layout          # None = sequential モード
        self._theme = get_cli_theme()
        self._full_response = ""
        self._line_buffer = ""
        self._in_code_block = False
        self._code_buffer = ""
        self._code_lang = ""
        self._code_indent = ""              # コードブロック開始行のインデント
        self._code_lines = 0
        self._spinner_iter = itertools.cycle(_SPINNER_FRAMES)
        self._spinner_shown = False
        self._collapse_code = False         # write_file 後のコードブロック折りたたみ

    @property
    def full_response(self) -> str:
        return self._full_response

    def start(self) -> None:
        self._full_response = ""
        self._line_buffer = ""
        self._in_code_block = False
        self._code_buffer = ""
        self._code_lang = ""
        self._code_indent = ""
        self._code_lines = 0
        self._spinner_iter = itertools.cycle(_SPINNER_FRAMES)
        self._spinner_shown = False
        self._collapse_code = False

    def notify_file_written(self) -> None:
        """ファイル書き込み完了を通知し、以降のコードブロックを折りたたみモードにする"""
        self._collapse_code = True

    def feed(self, token: str) -> None:
        self._full_response += token

        for char in token:
            self._line_buffer += char

            if char == "\n":
                self._process_line(self._line_buffer)
                self._line_buffer = ""
            elif self._in_code_block and not self._split:
                # 非 split layout: トークンごとにスピナー更新
                self._update_spinner_sequential()

    def _process_line(self, line: str) -> None:
        stripped = line.rstrip("\n\r")

        if not self._in_code_block:
            m = _FENCE_OPEN.match(stripped)
            if m:
                self._in_code_block = True
                self._code_indent = m.group(1)   # 開始行のインデントを記録
                self._code_lang = m.group(2) or ""
                self._code_buffer = ""
                self._code_lines = 0
                self._start_spinner()
                return
            self._print_text(line)
        else:
            if _FENCE_CLOSE.match(stripped):
                self._stop_spinner()
                self._render_code_block()
                self._in_code_block = False
                self._code_buffer = ""
                self._code_lang = ""
                self._code_indent = ""
                self._code_lines = 0
            else:
                # インデントされたコードブロック: 開始行と同じインデントを除去
                content = line
                if self._code_indent and content.startswith(self._code_indent):
                    content = content[len(self._code_indent):]
                self._code_buffer += content
                self._code_lines += 1
                if self._split:
                    self._update_spinner_split()

    def _print_text(self, text: str) -> None:
        # コードブロック外の孤立した ``` 行は表示しない（LLM の余分な出力）
        if text.rstrip("\n\r").strip() == "```":
            return
        if self._console.no_color:
            self._console.print(text, end="", highlight=False, markup=False)
        else:
            rich_text = _md_line_to_text(text)
            self._console.print(rich_text, end="", highlight=False)

    # ── スピナー ──

    def _start_spinner(self) -> None:
        if self._split and hasattr(self._split, "update_code_spinner"):
            # split layout: ステータスバーに表示
            self._split.update_code_spinner(self._code_lang, 0)
        else:
            frame = next(self._spinner_iter)
            lang_label = f" ({self._code_lang})" if self._code_lang else ""
            text = f"\r  {frame} コード生成中{lang_label}..."
            if self._console.no_color:
                sys.stdout.write(text)
            else:
                sys.stdout.write(f"\r\033[2m  {frame} コード生成中{lang_label}...\033[0m")
            sys.stdout.flush()
        self._spinner_shown = True

    def _update_spinner_split(self) -> None:
        """split layout: ステータスバーのスピナーテキストを更新"""
        if not self._split or not hasattr(self._split, "update_code_spinner"):
            return
        self._split.update_code_spinner(self._code_lang, self._code_lines)

    def _update_spinner_sequential(self) -> None:
        """非 split layout: sys.stdout でスピナー更新"""
        frame = next(self._spinner_iter)
        lang_label = f" ({self._code_lang})" if self._code_lang else ""
        lines_info = f" ({self._code_lines} lines)" if self._code_lines > 0 else ""
        if self._console.no_color:
            sys.stdout.write(f"\r  {frame} コード生成中{lang_label}{lines_info}...")
        else:
            sys.stdout.write(f"\r\033[2m  {frame} コード生成中{lang_label}{lines_info}...\033[0m")
        sys.stdout.flush()

    def _stop_spinner(self) -> None:
        if self._spinner_shown:
            if self._split and hasattr(self._split, "stop_code_spinner"):
                # split layout: コードスピナーを停止
                self._split.stop_code_spinner()
            else:
                sys.stdout.write("\r" + " " * 60 + "\r")
                sys.stdout.flush()
            self._spinner_shown = False

    # ── コードブロック出力 ──

    def _render_code_block(self) -> None:
        code = self._code_buffer.rstrip("\n")
        lang = self._code_lang or "text"

        # write_file 後の折りたたみモード: コード全文ではなくサマリのみ表示
        if self._collapse_code:
            lang_label = f" ({lang})" if lang != "text" else ""
            summary = Text(f"  ... {self._code_lines} lines{lang_label}", style="dim")
            if self._split:
                self._split.write_rich(summary)
            else:
                self._console.print(summary)
            return

        if self._split:
            # split layout: Syntax オブジェクトを RichLog にネイティブ表示
            try:
                syntax = Syntax(
                    code, lang,
                    theme=self._theme.code_theme,
                    line_numbers=True,
                    word_wrap=True,
                )
            except Exception:
                # 未知の言語指定時は text にフォールバック
                syntax = Syntax(
                    code, "text",
                    theme=self._theme.code_theme,
                    line_numbers=True,
                    word_wrap=True,
                )
            self._split.write_rich(syntax)
        elif self._console.no_color:
            self._console.print(f"```{self._code_lang}")
            self._console.print(code)
            self._console.print("```")
        else:
            syntax = Syntax(code, lang, theme="dracula", line_numbers=True, word_wrap=True)
            self._console.print(syntax)

    # ── 終了処理 ──

    def flush(self) -> None:
        if self._line_buffer:
            if self._in_code_block:
                self._code_buffer += self._line_buffer
            else:
                self._print_text(self._line_buffer)
            self._line_buffer = ""

    def finish(self) -> None:
        self._stop_spinner()

        if self._in_code_block:
            self.flush()
            # 中身が空のコードブロック（LLM が余分な ``` を出力したケース）はスキップ
            if self._code_buffer.strip():
                self._render_code_block()
            self._in_code_block = False
            self._code_buffer = ""

        self.flush()

    def render_end_marker(self) -> None:
        if self._split:
            from rich.rule import Rule as RichRule
            # 前の行を確定する改行
            self._console.print(highlight=False)
            # 区切り線を RichLog にネイティブ表示
            self._split.write_rich(
                RichRule(characters="─", style=self._theme.accent),
            )
        else:
            from backend.free.cli.renderer import render_separator
            render_separator(self._console)
