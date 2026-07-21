"""Meta-Cognitive ユーティリティ: テキスト正規化・JSON修復・応答パース・表示ヘルパー"""

from __future__ import annotations

import inspect
import json
import re
from typing import Iterator

from backend.free.constants import COMMAND_EXIT_CODE_PREFIX
from backend.i18n_helper import prose_language_name
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
# コンテンツ判定
# ---------------------------------------------------------------------------

_FEWSHOT_USER_LINE_RE = re.compile(r"^User:\s*(.+)$", re.MULTILINE)


def _char_bigrams(text: str) -> set[str]:
    """文字 bi-gram 集合を返す (テキスト間の粗い関連度判定用)。

    fewshot_pool.py の select_top_k と同じ手法だが、pillar 境界
    (EvorefLoop → EvorefLearn は import 不可) のためここで独立実装する。
    """
    t = text.strip()
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def fewshot_seems_relevant(
    task_text: str, fewshot_block: str, min_overlap: float = 0.03,
) -> bool:
    """few-shot ブロックが現在のタスクとどの程度関連しているかを粗く判定する。

    Level 1 が進化させた few-shot は query 類似度だけでなく fitness (過去の
    成功実績) も加味して選ばれるため、現在のタスクと無関係でも再利用され
    うる。無関係な few-shot 例をファイル内容生成タスクへそのまま注入すると、
    ローカル LLM がその例文をそのまま繰り返す退化を誘発しうる
    (#incident: 「テスト.docx」生成タスクに無関係な「映画ですか？」の
    few-shot が注入され、その例文の反復が延々と write_file されていた)。

    fewshot_block 中の ``User: ...`` 行 (例の質問文) だけを取り出して
    task_text と文字 bi-gram の Jaccard 重なりを見る。Assistant 側の応答文
    (定型の丁寧語などで一般的な bi-gram が多く、関連度判定のノイズになる)
    は比較対象から除く。``User:`` 行が見つからない/どちらかが極端に短い
    等で判定不能な場合は安全側 (注入する) に倒す。
    """
    queries = " ".join(_FEWSHOT_USER_LINE_RE.findall(fewshot_block))
    a = _char_bigrams(task_text)
    b = _char_bigrams(queries)
    if not a or not b:
        return True
    overlap = len(a & b) / len(a | b)
    return overlap >= min_overlap


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


# ---------------------------------------------------------------------------
# 生成コンテンツのエコー検出 (タスクログ / プロンプト scaffold)
# ---------------------------------------------------------------------------

#: エージェントの進捗ノート行 (最終応答フォーマット由来)。few-shot に混入した
#: 「- [done] ... / Written N bytes to ...」形式を小型モデルが成果物として
#: 復唱する退化 (#incident 2026-07-15: 29 ファイル中 10 件が本文なしのログ 1 行)
#: を書込み前に検出するためのパターン。
_TASK_LOG_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\[(?:done|failed|skipped)\]\s"
    r"|^\s*Written\s+\d+\s+bytes\s+to\s+\S"
    r"|^\s*Content of `[^`]+`\s*:?\s*$",
)

#: 生成プロンプトの内部 scaffold マーカー。成果物に現れたら「プロンプトの
#: エコー」であり本文生成に失敗している (例: relocation_notice.txt に
#: 「[現在日時 (UTC基準)] ...」とタスク英文がそのまま書き込まれた事例)。
_PROMPT_SCAFFOLD_MARKERS: tuple[str, ...] = (
    "[現在日時 (UTC基準)]",
    "## 既存ファイル内容 (",
    "【元のコード】",
    "【修正指示】",
)

#: user_prompt の内部構造「タスク: <英語タスク文>」のエコー検出。planner の
#: タスク文は英語動詞で始まるため、日本語文書中の一般語「タスク:」とは
#: 区別できる。
_TASK_SCAFFOLD_LINE_RE = re.compile(
    r"^タスク\s*[:：]\s*(?:Write|Generate|Create|List|Read|Fetch|Revise|Update)\b",
    re.MULTILINE,
)


def strip_task_log_scaffold(content: str) -> str:
    """先頭に混入したタスク進捗ノート行を取り除いた残りを返す。

    「タスクログ + 本文」の連結出力 (部分症状) から本文だけを救済する。
    ログ行が無ければ原文をそのまま返し、全行がログなら空文字列を返す
    (呼出側でエコーとして棄却する)。
    """
    lines = content.split("\n")
    idx = 0
    for i, line in enumerate(lines):
        if not line.strip():
            idx = i + 1
            continue
        if _TASK_LOG_LINE_RE.match(line):
            idx = i + 1
            continue
        break
    if idx == 0:
        return content
    return "\n".join(lines[idx:]).strip("\n")


def fewshot_contains_task_log(fewshot_block: str) -> bool:
    """few-shot 例にタスク進捗ノート形式の応答が含まれるかを判定する。

    「- [done] ... Written N bytes」だけの応答例を参考例として注入すると
    「書いた事実の報告だけ出せば正解」というバイアスを与え、本文なしの
    極小ファイル生成を誘発する (#incident 2026-07-15)。該当例を含む
    few-shot ブロックは注入しない。
    """
    return any(
        _TASK_LOG_LINE_RE.match(ln) for ln in fewshot_block.split("\n")
    )


def looks_like_task_log_echo(content: str) -> bool:
    """content がタスク進捗ノートのエコー (本文なし) かを判定する。"""
    if not any(_TASK_LOG_LINE_RE.match(ln) for ln in content.split("\n")):
        return False
    remainder = strip_task_log_scaffold(content)
    if remainder == content:
        return False
    return len(remainder.strip()) < 40


def looks_like_prompt_echo(content: str) -> bool:
    """content が生成プロンプトの scaffold を含む (= プロンプトエコー) かを判定する。"""
    if any(marker in content for marker in _PROMPT_SCAFFOLD_MARKERS):
        return True
    return bool(_TASK_SCAFFOLD_LINE_RE.search(content))


def csv_content_lacks_rows(content: str, file_path: str) -> bool:
    """.csv 出力先なのに区切り行が実質存在しないかを判定する。

    厳密な CSV 検証ではなく「散文/エコーを CSV として書く」事故の検出が目的。
    カンマ区切り行または GFM パイプ表行が 2 行以上 (ヘッダ+データ) あれば
    合格とする。``.csv`` 以外の出力先では常に False。
    """
    if not file_path.lower().endswith(".csv"):
        return False
    delimited = 0
    for line in content.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.count(",") >= 1 or (s.startswith("|") and s.endswith("|")):
            delimited += 1
        if delimited >= 2:
            return False
    return True


def generated_content_rejection(
    content: str, file_path: str,
) -> str | None:
    """write_file 直前の生成コンテンツ検証。棄却理由を返す (適正なら None)。

    書込みパス (fast path / 自動リカバリー) から共通で呼び、タスクログエコー・
    プロンプトエコー・パス誤出力・CSV 構造欠落を書込み前に弾く。
    """
    if looks_like_path_not_content(content, file_path):
        return "path_only"
    if looks_like_task_log_echo(content):
        return "task_log_echo"
    if looks_like_prompt_echo(content):
        return "prompt_echo"
    if csv_content_lacks_rows(content, file_path):
        return "csv_without_rows"
    return None


_CODE_INDICATORS: tuple[str, ...] = (
    "import ", "from ", "def ", "class ", "function ",
    "const ", "let ", "var ", "return ", "if __name__",
    "#include", "package ", "public class",
    "#!/", "# -*- coding",
    "pygame", "print(", "console.log",
)


def contains_code_indicator(text: str) -> bool:
    """テキストにコードらしさを示すマーカーが1つ以上含まれるかを判定する。

    長さ・改行に依存しないため、1 行の短いコード片でも検出できる。
    散文 (例: "...を設計します。" のみ) は指標を含まず False になる。
    """
    if not text:
        return False
    text_lower = text.lower()
    return any(ind.lower() in text_lower for ind in _CODE_INDICATORS)


def text_looks_like_code(text: str) -> bool:
    """テキストがプログラムコードに見えるかを判定する

    LLM がツールコール JSON ではなくコードをそのまま出力した場合に、
    それをファイルに書き込んでよいかを判定するために使用する。
    """
    if not text or len(text) < 20:
        return False
    if "\n" not in text:
        return False
    return contains_code_indicator(text)


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


# テンプレート由来のツールコール (OAI JSON でない生テキスト) 抽出。
# gemma 系: <|tool_call>call:NAME(k: "v", k2: "v2")<tool_call|>
# マーカーの揺れ (<|tool_call|> / <tool_call> / </tool_call>) を寛容に許容する。
_TEMPLATE_TOOL_CALL_RE = re.compile(
    r"<\|?tool_call\|?>\s*call:\s*(?P<name>\w+)\s*\((?P<args>.*?)\)\s*</?\|?tool_call\|?>",
    re.DOTALL,
)
# タグ内が JSON の Qwen 系: <tool_call>{...}</tool_call>
_TEMPLATE_JSON_CALL_RE = re.compile(
    r"<\|?tool_call\|?>\s*(\{.*?\})\s*</?\|?tool_call\|?>",
    re.DOTALL,
)


def _parse_template_args(args_str: str) -> dict:
    """``call:NAME(...)`` の引数文字列を best-effort で dict 化する。

    ``key: "value"`` / ``key=value`` 形式を抽出する。位置引数や解釈不能な
    断片は無視し、後段の引数正規化 (normalize_*_args) に委ねる。
    """
    args: dict = {}
    if not args_str.strip():
        return args
    for m in re.finditer(
        r'(\w+)\s*[:=]\s*(?:"([^"]*)"|\'([^\']*)\'|([^,]+?))(?:,|$)',
        args_str,
    ):
        key = m.group(1)
        value = m.group(2)
        if value is None:
            value = m.group(3)
        if value is None:
            value = (m.group(4) or "").strip()
        args[key] = value
    return args


def parse_template_tool_call(text: str) -> dict | None:
    """テンプレート形式のツールコール生テキストを ``{"tool", "args"}`` に変換する。

    base モデル (gemma 系等) が OAI JSON ではなくチャットテンプレート由来の
    ``<|tool_call>call:NAME(args)<tool_call|>`` 形式を ``message.content`` に
    そのまま吐くケースを救済する。Qwen 系の ``<tool_call>{json}</tool_call>``
    はタグ内が JSON のため try_parse_tool_dict へ委譲する。
    返り値は OAI JSON パーサ (try_parse_tool_dict) と同形。
    """
    if not text or "tool_call" not in text:
        return None
    # Qwen 系: タグ内が JSON
    m_json = _TEMPLATE_JSON_CALL_RE.search(text)
    if m_json is not None:
        parsed = try_parse_tool_dict(m_json.group(1))
        if parsed is not None:
            return parsed
    # gemma 系: call:NAME(args)
    m = _TEMPLATE_TOOL_CALL_RE.search(text)
    if m is None:
        return None
    return {"tool": m.group("name"), "args": _parse_template_args(m.group("args"))}


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
