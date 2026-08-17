"""生成テキストの整形・修復と表示ヘルパー

LLM が付ける Markdown フェンス / 説明文 / パスコメントの除去、繰り返しの
打ち切り、壊れた JSON の修復、進捗表示用の要約など、**中身の真偽を問わない**
テキスト操作だけを置く。真偽の判定は ``meta_cognitive_content_gate`` へ。
"""

from __future__ import annotations

import re

from pathlib import Path
from backend.i18n_helper import prose_language_name

from backend.log_config import get_logger

logger = get_logger("agent.meta_cognitive.utils")


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
    content = strip_leading_narration_headings(content)
    content = strip_answer_framing(content)
    return content


#: 「<成果物> の内容は「<本文>」です。」型のチャット回答枠。前置きの末尾が
#: 成果物を指す語 + 助詞で終わるものだけを対象にし、実文書に現れる普通の
#: 引用文 (「〜」です。) を誤って剥がさないようにする。
_ANSWER_FRAMING_LEAD_RE = re.compile(
    r"(?:内容|中身|リスト|一覧|結果|ファイル)\s*(?:は|:|：)\s*$",
)
_ANSWER_FRAMING_RE = re.compile(
    r"\A(?P<lead>[^\n]{1,120}?)[「『]\s*(?P<body>.+?)\s*[」』]"
    r"(?:です|でした|になります|となります)?[。．.]?\s*\Z",
    re.DOTALL,
)


def strip_answer_framing(content: str) -> str:
    """チャット回答の言い回しごと成果物にした出力から本文だけを取り出す。

    生成プロンプトは「本文だけを出せ」と指示しているが、小型モデルは会話の
    癖でファイル本文を回答文に包む (実インシデント 2026-07-28 ライブ検証:
    「E:\\tmp\\audit_round6.txt に登山の持ち物リストを5項目、箇条書きで書いて
    ください。」に対し、ファイルへ ``E:\\tmp\\audit_round6.txt の内容は「登山の
    持ち物リスト：\\n- 登山靴\\n…」です。`` が書き込まれた)。

    前置きが成果物を指す語 (内容 / 中身 / リスト …) + 助詞で終わり、全体が
    鉤括弧 1 組で閉じている場合だけ剥がす (純粋関数)。
    """
    m = _ANSWER_FRAMING_RE.match(content.strip())
    if m is None:
        return content
    if not _ANSWER_FRAMING_LEAD_RE.search(m.group("lead")):
        return content
    body = m.group("body").strip()
    return body or content


#: 「これから何を書くか」を述べているだけの見出しテキスト (成果物ではない)。
#: 見出し全体との完全一致のみを対象にし、実文書の見出し (「# 議事録」等) を
#: 誤って削らないようにする。
_NARRATION_HEADING_TEXT_RE = re.compile(
    r"^(?:writing(?:\s+to)?(?:\s+(?:the\s+)?file)?"
    r"|write(?:\s+to)?(?:\s+(?:the\s+)?file)?"
    r"|file(?:\s+content)?|output|generated\s+content|content"
    r"|ファイル(?:へ)?の?書き(?:込み|出し)|書き(?:込み|出し)"
    r"|ファイル(?:出力|内容|名)?|出力(?:内容)?|生成(?:内容|結果)?)$",
    re.IGNORECASE,
)
#: 見出しがファイルパス/ファイル名そのものを繰り返しているケース
_HEADING_PATH_RE = re.compile(
    r"^(?:[A-Za-z]:[/\\]\S*"           # ドライブレター付きフルパス
    r"|(?:/\S+){2,}"                    # Unix フルパス
    r"|\S+\.[A-Za-z][A-Za-z0-9]{0,9})$" # 拡張子付きファイル名
)


#: 「<literal>」と保存して / 「<literal>」という内容で書き込んで の literal 部分。
#: 閉じ括弧の直後は助詞と定型句だけを許し、書き込み動詞まで到達した場合のみ
#: 「ユーザーが本文そのものを書いた」とみなす。
_LITERAL_WRITE_CONTENT_RE = re.compile(
    r"[「『]([^「」『』]{1,2000})[」』]"
    r"\s*(?:と\s*(?:いう)?\s*(?:内容|文言|テキスト|文字列)?\s*(?:で|を)?|を)?\s*"
    # 限定の副助詞。「『…』とだけ書いて」「『…』のみを保存して」は本文が
    # 確定している最も典型的な言い方なのに、これが無いと literal 抽出が
    # 外れて小型モデルの再生成に倒れる (実インシデント 2026-07-29 ライブ監査:
    # 「…に「監査テスト 1行目」とだけ書いて保存してください。」でファイルに
    # 「タスクの内容: ファイル `…` に「監査テスト 1行目」と…」が書かれた)。
    r"(?:(?:だけ|のみ)\s*(?:を|で)?\s*)?"
    r"(?:そのまま\s*)?"
    r"(?:保存|書き込|書き出|書いて|出力)",
)
#: 引用が本文ではなくタイトル/ファイル名を指しているケースの除外
_LITERAL_WRITE_REJECT_RE = re.compile(
    r"[」』]\s*と?\s*いう?\s*(?:タイトル|題名|件名|ファイル名|名前|名称)",
)
#: literal 書き込みを許可する拡張子 (リッチ文書は構造化生成が必要なので対象外)
_LITERAL_WRITE_EXTENSIONS = frozenset({"", ".txt", ".md", ".log"})
#: 「1行目は alpha、2行目は beta、3行目は gamma です」型の行指定。引用符を
#: 使わずに本文を確定させる言い方で、引用符ベースの literal 抽出では拾えない
#: (実インシデント 2026-07-29 ライブ監査: この依頼が生成経路に落ち、生成物が
#: task_restatement として 2 回とも棄却されて書き込み自体が起きなかった)。
_ENUMERATED_LINE_RE = re.compile(
    r"(?:(\d{1,3})\s*行目|line\s*(\d{1,3}))\s*(?:は|が|:|：)\s*"
    r"(.+?)"
    r"\s*(?:です|だ|である)?"
    # 末尾に続く書込み指示 (「… と書き込んで」) は本文ではないので食わせる。
    r"\s*(?:[とを]\s*(?:書き込|書いて|書き出|保存|出力)\S*)?"
    r"\s*(?=[、,。\n]|$)",
    re.IGNORECASE,
)
#: 行本文を包む引用符 (抽出後に剥がす)
_ENUMERATED_LINE_QUOTES = ("「」", "『』", '""', "''", "``")


def extract_literal_write_content(query: str, file_path: str) -> str:
    """ユーザーが本文そのものを引用符で指定している場合、その literal を返す。

    「E:\\tmp\\memo.txt に『会議メモ: 2026年7月27日』と保存してください」のように
    書く内容が確定しているなら、小型モデルに再生成させる理由が無い。再生成
    させると実況やタスクの言い換えを本文として書き込む退行が起きる
    (実インシデント 2026-07-27 ライブ検証: 見出し形式
    「## Writing to File / ### <path>」および箇条書き形式
    「- ファイル: <path> / - 内容: ...」が本文として書き込まれた)。

    タイトル/ファイル名を指す引用 (「『山の話』というタイトルで作文を書いて」)
    と、構造化生成が必要なリッチ文書 (xlsx/docx/pptx/csv) は対象外。

    Returns:
        literal 本文。抽出できなければ空文字列 (純粋関数)。
    """
    if Path(file_path).suffix.lower() not in _LITERAL_WRITE_EXTENSIONS:
        return ""
    if _LITERAL_WRITE_REJECT_RE.search(query):
        return ""
    match = _LITERAL_WRITE_CONTENT_RE.search(query)
    if match:
        return match.group(1).strip()
    return extract_enumerated_line_content(query)


def extract_enumerated_line_content(query: str) -> str:
    """「1行目は alpha、2行目は beta、3行目は gamma です」型の本文を組み立てる。

    引用符を使わずに行ごとの本文を確定させる言い方。引用符ベースの literal
    抽出では拾えず、生成経路に落ちて棄却される (実インシデント 2026-07-29
    ライブ監査: 「1行目は alpha、2行目は beta、3行目は gamma です」の依頼で
    生成物が 2 回とも ``task_restatement`` 判定になり、書き込みが一度も
    実行されないまま「(書き込みが実行されませんでした)」で終わった)。

    行番号が 1 から連番で欠けなく揃っている場合のみ採用する。部分的な言及
    (「3行目を直して」等) を本文全体と誤認しないための決定論ガード。

    Returns:
        改行連結した本文。組み立てられなければ空文字列 (純粋関数)。
    """
    found: dict[int, str] = {}
    for m in _ENUMERATED_LINE_RE.finditer(query):
        index = int(m.group(1) or m.group(2))
        value = _strip_enclosing_quotes(m.group(3).strip())
        if not value or len(value) > 500:
            return ""
        if index in found:
            return ""
        found[index] = value
    if len(found) < 2:
        return ""
    if sorted(found) != list(range(1, len(found) + 1)):
        return ""
    return "\n".join(found[i] for i in range(1, len(found) + 1))


def _strip_enclosing_quotes(text: str) -> str:
    """本文を包む引用符を 1 組だけ剥がす (純粋関数)。"""
    for pair in _ENUMERATED_LINE_QUOTES:
        if len(text) >= 2 and text[0] == pair[0] and text[-1] == pair[-1]:
            return text[1:-1].strip()
    return text


def strip_leading_narration_headings(content: str) -> str:
    """先頭の「何を書くか」を述べただけの見出しブロックを除去する。

    meta_cognitive の生成プロンプトはタスク文とファイルパスを含むため、
    小型モデルがエージェントの実況をそのまま成果物として出力することがある
    (実インシデント 2026-07-27 ライブ検証: memo.txt に
    「## Writing to File / ### E:\\tmp\\...\\memo.txt / 会議メモ: ...」が
    書き込まれた)。既存の scaffold 検出は literal なエコー用で、見出しに
    言い換えられた形を拾えなかった。

    除去対象は「実況見出し」か「パス/ファイル名の見出し」のみ。実文書の
    見出しを削らないよう、それ以外の見出しに当たった時点で打ち切り、
    全部消える場合は原文を返す (純粋関数)。
    """
    lines = content.split("\n")
    idx = 0
    stripped_any = False
    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            idx = i + 1
            continue
        if not text.startswith("#"):
            break
        heading = text.lstrip("#").strip().rstrip(":：")
        if _NARRATION_HEADING_TEXT_RE.match(heading) or _HEADING_PATH_RE.match(heading):
            idx = i + 1
            stripped_any = True
            continue
        break
    if not stripped_any:
        return content
    remainder = "\n".join(lines[idx:]).strip("\n")
    return remainder or content


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
    通常のコメント行（# -*- create: utf-8 -*- 等）は除去しない。
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

    return _truncate_block_repetition("\n".join(result))


def _truncate_block_repetition(
    content: str, min_cycle_repeats: int = 3, max_period: int = 32,
) -> str:
    """複数行が一塊で周期的に反復する退化出力を検出し切除する

    ``truncate_repetition`` の連続同一行検出は period=1 (直前と全く同じ1行の
    連続) しか捕捉できない。few-shot 例の丸ごと反復生成のように、数行〜
    数十行のブロックがそのまま周期的に繰り返されるケース (period>=2) は
    すり抜けるため、別途ブロック単位で検出する
    (#incident: 無関係な few-shot 例をローカル LLM がそのまま繰り返し、
    要求内容と無関係な生成物がそのまま write_file されていた)。

    周期 (行数) を 2〜``max_period`` の範囲で探索し、同一ブロックが
    ``min_cycle_repeats`` 回以上連続していれば最初の1周期分だけ残す。
    period=1 は ``truncate_repetition`` の担当のためここでは対象外。
    """
    lines = content.split("\n")
    n = len(lines)
    if n < min_cycle_repeats * 2:
        return content

    stripped = [ln.strip() for ln in lines]
    result: list[str] = []
    truncated = False
    i = 0
    while i < n:
        matched: tuple[int, int] | None = None  # (period, resume_index)
        limit = min(max_period, (n - i) // min_cycle_repeats)
        for period in range(2, limit + 1):
            block = stripped[i:i + period]
            if not any(block):
                continue  # 空行だけのブロックは対象外
            j = i + period
            repeats = 1
            while j + period <= n and stripped[j:j + period] == block:
                repeats += 1
                j += period
            if repeats >= min_cycle_repeats:
                matched = (period, j)
                break
        if matched:
            period, resume_index = matched
            result.extend(lines[i:i + period])
            i = resume_index
            truncated = True
        else:
            result.append(lines[i])
            i += 1

    if truncated:
        logger.warning(
            "Block-level repetition detected and truncated in generated "
            "content (original: %d lines → %d lines)",
            n, len(result),
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
# コンテンツ生成の出力言語指示
# ---------------------------------------------------------------------------

def content_language_directive() -> str:
    """write_file コンテンツ生成へ注入する出力言語指示 (locale 追従)。

    英語スキャフォールドプロンプト (CONTENT_GENERATION_PROMPT /
    _CONTENT_GEN_PROMPT) に付加するため英語文面。ユーザーの明示指定と
    既存ファイルの言語を優先し、コード識別子等は原語のまま維持させる。
    生成時点の locale を反映するため呼出毎に組み立てる。
    """
    lang = prose_language_name(english=True)
    return (
        f"Write all natural-language text (prose, headings, descriptions, "
        f"comments) in {lang} unless the user's request explicitly specifies "
        f"another language. Keep code identifiers, commands, file paths, and "
        f"data values as-is. When updating an existing file, follow the "
        f"existing content's language."
    )
