"""Meta-Cognitive ユーティリティ: テキスト正規化・JSON修復・応答パース・表示ヘルパー"""

from __future__ import annotations

import inspect
import json
import re
from typing import Iterator

from backend.free.constants import COMMAND_EXIT_CODE_PREFIX
from backend.log_config import get_logger

logger = get_logger("agent.meta_cognitive.utils")


# ---------------------------------------------------------------------------
# コールバック
# ---------------------------------------------------------------------------

async def call_callback(callback, data) -> None:
    """コールバックを呼び出す（async/sync 両対応）"""
    if inspect.iscoroutinefunction(callback):
        await callback(data)
    else:
        callback(data)


# ---------------------------------------------------------------------------
# ツール実行結果の判定
# ---------------------------------------------------------------------------

#: ツール実行結果が失敗を示す際の先頭マーカー。
TOOL_ERROR_PREFIX = "Error:"


def is_tool_error(text: str) -> bool:
    """ツール実行結果文字列が失敗 (``Error:`` プレフィックス) かを判定する。

    エラーマーカー判定を一元化し、``.startswith("Error:")`` の散在を防ぐ
    (マーカー変更時の修正漏れ耐性)。
    """
    return text.startswith(TOOL_ERROR_PREFIX)


def command_run_failed(text: str) -> bool:
    """``run_command`` の結果文字列が失敗 (非ゼロ終了) を示すかを判定する。

    ``run_command`` (``tools/builtin.py:_run_command_async_impl``) は実行した
    プログラム自身が非ゼロ終了したときのみ末尾へ ``[exit code: N]`` を付与する。
    ``is_tool_error`` (``Error:`` プレフィックス) はツールラッパ自身の失敗
    (例外 / 危険コマンドブロック等) しか拾えず、走ったが失敗したコマンド
    (SyntaxError / 非ゼロ終了) を success と誤判定する。このマーカー検出で補完し、
    誤った成功が SemMem (executable_command) 学習を汚染するのを防ぐ。

    出力切り詰め (200 行超で head+tail) でもマーカーは末尾 tail に残るため、
    切り詰め後の文字列でも検出できる。``run_command`` 以外の結果には付与
    されないため、呼出側は ``run_command`` のときだけ参照すること。
    """
    return COMMAND_EXIT_CODE_PREFIX in text


# ---------------------------------------------------------------------------
# Markdown / パスコメント除去
# ---------------------------------------------------------------------------

def strip_markdown_wrapper(content: str) -> str:
    """LLM が付加した Markdown フェンスと説明テキストを除去する

    LLM はプロンプトで「コンテンツのみ出力」と指示しても、
    説明文やマークダウンフェンスで囲むことがある。
    例:
        インベーダゲームのコードは以下です。
        ```python
        import pygame
        ...
        ```
        このコードを実行してください。

    → ``import pygame\\n...`` のみ抽出する。
    """
    fence_pattern = re.compile(r"```\w*\s*\n(.*?)```", re.DOTALL)
    matches = fence_pattern.findall(content)
    if matches:
        content = max(matches, key=len).strip()

    content = strip_leading_path_comment(content)
    return content


# ファイルパスコメント除去用パターン
# 例: # E:\xxx\invaders.py, // /home/user/app.js
_PATH_COMMENT_RE = re.compile(
    r"^(?:#|//)\s*"                    # コメント開始（# or //）
    r"[A-Za-z]:[/\\]"                  # ドライブレター + パス区切り
    r"[^\n]*$",                        # 行末まで
    re.MULTILINE,
)


def strip_leading_path_comment(content: str) -> str:
    """先頭行がファイルパスコメントなら除去する

    LLM がコード生成時にファイルパスをコメントとして先頭に付加する
    ケースに対応する。例:
        # E:\\xxx\\invaders.py
        import pygame
        ...
    → ``import pygame\\n...`` に修正する。

    ドライブレター付きパス（C:\\, E:\\ 等）のみを対象とし、
    通常のコメント行（# -*- coding: utf-8 -*- 等）は除去しない。
    """
    lines = content.split("\n", 1)
    if len(lines) < 2:
        return content
    first_line = lines[0].strip()
    if _PATH_COMMENT_RE.match(first_line):
        return lines[1].lstrip("\n")
    return content


# ---------------------------------------------------------------------------
# 繰り返し検出 / JSON 修復
# ---------------------------------------------------------------------------

def truncate_repetition(content: str, min_repeat: int = 4) -> str:
    """LLM の繰り返し生成（退化出力）を検出し切除する

    ローカル LLM がトークン生成ループにはまり、同一行を数百回繰り返すケースに
    対応する。連続する同一行が min_repeat 回以上続いた場合、最初の1回だけ残す。
    """
    if not content:
        return content

    lines = content.split("\n")
    result: list[str] = []
    repeat_count = 0
    prev_stripped = None
    truncated = False

    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev_stripped:
            repeat_count += 1
            if repeat_count >= min_repeat:
                truncated = True
                continue
        else:
            repeat_count = 1 if stripped else 0
            prev_stripped = stripped if stripped else prev_stripped
        result.append(line)

    if truncated:
        logger.warning(
            "Repetition detected and truncated in generated content "
            "(original: %d lines → %d lines)",
            len(lines), len(result),
        )

    return "\n".join(result)


def fix_json_backslashes(text: str) -> str:
    """LLM が出力した JSON 内の未エスケープバックスラッシュを修復する

    LLM は Windows パス (e.g. ``e:\\ttte\\tetris.py``) を JSON 文字列内で
    ``e:\ttte\tetris.py`` と出力しがち。``\\t`` がタブ、``\\n`` が改行に
    化けるため、JSON パース前にドライブレター直後のパスを修復する。
    """
    def _escape_path(m: re.Match) -> str:
        path = m.group(0)
        result_chars: list[str] = []
        i = 0
        while i < len(path):
            if path[i] == '\\':
                if i + 1 < len(path) and path[i + 1] == '\\':
                    result_chars.append('\\\\')
                    i += 2
                else:
                    result_chars.append('\\\\')
                    i += 1
            else:
                result_chars.append(path[i])
                i += 1
        return ''.join(result_chars)

    return re.sub(
        r'[A-Za-z]:\\(?:\\?[^"\\,\]\s][^"\\,\]]*)+',
        _escape_path,
        text,
    )


# ---------------------------------------------------------------------------
# コンテンツ判定
# ---------------------------------------------------------------------------

def looks_like_path_not_content(content: str, file_path: str) -> bool:
    """content がファイルパスの誤出力かどうかを判定する"""
    if content == file_path:
        return True
    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return False
    if (
        len(content) < 260
        and "\n" not in content
        and ("/" in content or "\\" in content)
        and "." in content.split("/")[-1].split("\\")[-1]
    ):
        return True
    return False


def text_looks_like_code(text: str) -> bool:
    """テキストがプログラムコードに見えるかを判定する

    LLM がツールコール JSON ではなくコードをそのまま出力した場合に、
    それをファイルに書き込んでよいかを判定するために使用する。
    """
    if not text or len(text) < 20:
        return False
    if "\n" not in text:
        return False
    code_indicators = [
        "import ", "from ", "def ", "class ", "function ",
        "const ", "let ", "var ", "return ", "if __name__",
        "#include", "package ", "public class",
        "#!/", "# -*- coding",
        "pygame", "print(", "console.log",
    ]
    text_lower = text.lower()
    indicator_count = sum(1 for ind in code_indicators if ind.lower() in text_lower)
    return indicator_count >= 1


# ---------------------------------------------------------------------------
# 表示ヘルパー
# ---------------------------------------------------------------------------

def summarize_tool_args(tool_name: str, args: dict) -> str:
    """ツール引数を短い文字列にまとめる（進捗表示用）"""
    if tool_name == "write_file":
        path = args.get("file_path", "")
        content_len = len(args.get("content", ""))
        return f"{path}, {content_len}文字"
    if tool_name == "read_file":
        return args.get("file_path", "")
    if tool_name == "run_command":
        return args.get("command", "")[:60]
    if tool_name == "search_code":
        return args.get("pattern", "")[:40]
    for v in args.values():
        s = str(v)
        return s[:50] if len(s) > 50 else s
    return ""


def summarize_file_content(content: str, max_lines: int = 40) -> str:
    """ファイル内容をサマリ化する（後続タスクのコンテキスト用）

    短いファイルはそのまま返し、長いファイルは先頭部分+構造スケルトンを返す。
    """
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return content
    head = "\n".join(lines[:25])
    skeleton = [
        line for line in lines[25:]
        if re.match(r"\s*(def |class |import |from |#\s)", line)
    ]
    skeleton_text = "\n".join(skeleton) if skeleton else ""
    return (
        f"{head}\n"
        f"... ({len(lines) - 25} more lines) ...\n"
        f"{skeleton_text}"
    )


# ---------------------------------------------------------------------------
# JSON ツール呼び出し抽出ヘルパー
# ---------------------------------------------------------------------------


def try_parse_tool_dict(text: str) -> dict | None:
    """text を JSON としてパースし、'tool' キーを持つ dict ならそれを返す。

    パース失敗・dict 以外・'tool' キー欠落のいずれも None を返す。
    meta_cognitive / agentic_loop の _parse_tool_call から共有される。
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "tool" in data:
        return data
    return None


def _find_matching_close_brace(text: str, start: int) -> int | None:
    """text[start] が '{' のとき、対応する '}' のインデックスを返す（無ければ None）。

    文字列リテラル内の '{' '}' は考慮しない（旧実装と同等の素朴なバランス検出）。
    """
    depth = 0
    for i in range(start, len(text)):
        match text[i]:
            case '{':
                depth += 1
            case '}':
                depth -= 1
                if depth == 0:
                    return i
    return None


def iter_balanced_brace_substrings(text: str) -> Iterator[str]:
    """text 内の各 '{' から始まるバランスの取れた {...} 部分文字列を順に yield する。"""
    for start_match in re.finditer(r'\{', text):
        start = start_match.start()
        end = _find_matching_close_brace(text, start)
        if end is not None:
            yield text[start:end + 1]
