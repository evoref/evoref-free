"""Meta-Cognitive ツール推論: タスク記述からのツール決定・引数正規化"""

from __future__ import annotations

import re
from pathlib import Path

from backend.free.agent.meta_cognitive_utils import looks_like_path_not_content
from backend.free.agent.safety_patterns import strip_command_literals
from backend.free.agent.task_write_gate import task_write_verdict
from backend.free.core.intent_vocab import QUOTED_SPAN_RE
from backend.log_config import get_logger

logger = get_logger("agent.meta_cognitive.tools")


# ---------------------------------------------------------------------------
# ツール推論パターン（優先度順）
# ---------------------------------------------------------------------------

# write_file は先頭で判定点 ``task_write_intent`` の ``fire`` だけを見る
# (``infer_tool_from_task``、docs/f_03 §4.3)。
_TOOL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # read_file: 読み取り系
    (re.compile(
        r"読み|読んで|確認|正し[いく]|合って|内容|表示|中身|見せて|見て"
        r"|read|show|display|view|cat|check|inspect|examine|verify|correct",
        re.IGNORECASE,
    ), "read_file"),
    # run_command: コマンド実行系
    (re.compile(
        r"実行|テスト|起動|インストール|ビルド|コンパイル"
        r"|run|execute|test|install|build|compile|lint|npm|pip|pytest",
        re.IGNORECASE,
    ), "run_command"),
    # search_code: 検索系
    (re.compile(
        r"検索|探|grep|find|search|locate",
        re.IGNORECASE,
    ), "search_code"),
]


# ---------------------------------------------------------------------------
# ツール推論
# ---------------------------------------------------------------------------

def infer_tool_from_task(
    description: str,
) -> tuple[str, dict] | None:
    """タスク記述からツールと引数を決定論的に推論する

    Returns:
        (tool_name, args) または None（推論不可の場合）
    """
    from backend.free.agent.tool_call_judge import _extract_file_path
    from backend.free.agent.tool_judge_args import (
        extract_write_target_path,
        without_drive_paths,
    )

    # バッククォート内コマンドの引数パスは読み書きの対象ではない。パス抽出は
    # コマンドを除いた本文に対して行う (コマンド抽出は生の description を見る)。
    path_source = strip_command_literals(description)

    # write_file: 書込みを起こすのは判定点 ``task_write_intent`` の ``fire`` だけ
    # (棄権 = 宛先の標識だけのタスクで書くと「Compare a.txt to b.txt」の b.txt を壊す)
    if task_write_verdict(description).band == "fire":
        # 2 ファイルが登場するタスクでは先頭は常に source (読む側)。
        # 先頭一致だと書き込み先が source に化ける
        # (extract_write_target_path の docstring 参照)。
        file_path = extract_write_target_path(path_source)
        if file_path:
            return ("write_file", {"file_path": file_path})

    # 動詞はドライブ付きパスを除いた本文で見る (パスの ``live_check`` を読みの
    # 動詞に数えない、``ToolCallJudge._infer_tool`` と同じ、f_03 §3.1)
    verb_text = without_drive_paths(description, " ")
    for pattern, tool_name in _TOOL_PATTERNS:
        if not pattern.search(verb_text):
            continue

        if tool_name == "read_file":
            file_path = _extract_file_path(path_source)
            if file_path:
                return ("read_file", {"file_path": file_path})

        elif tool_name == "run_command":
            cmd = extract_command(description)
            if cmd:
                return ("run_command", {"command": cmd})

        elif tool_name == "search_code":
            # 空パターンは全行にマッチし CWD 全域を舐めるので、抽出できた
            # ときだけ確定する (抽出不能ならツールループへ委譲)。探す場所も
            # 依頼文が名指したフォルダに限る — CWD (インストール根) を補うと
            # 本体のソースが答えに混じる (CLAUDE.md §6 #5、f_03 §4.2.1)。
            args = named_folder_search_args(description)
            if args is not None:
                return ("search_code", args)

    return None


def named_folder_search_args(description: str) -> dict | None:
    """名指したフォルダの中の検索の引数 ``{pattern, directory}`` (組めなければ None)。

    パターン (:func:`extract_search_pattern`) と、検索の場所の前置詞
    (``in`` / ``within`` / ``under`` / ``inside``) が導く実在のフォルダの両方が取れたときだけ
    組む。``infer_tool_from_task`` と ``ToolCallJudge._infer_tool`` が共有する
    (f_03 §4.2.1 (a) / §3.1)。

    動詞の直後の複数語 (「Find the files in <フォルダ>」の ``the files``) は探す対象を
    言い表した名詞句であってパターンではないので組まない。複数語のパターンは括るか
    ``for`` で導けば通る (``Search for def main in …``)。括らないパターンが節の境界
    (``and`` / ``then`` / 読点) を飲み込んだら組まない — 「Search for TODO and save the
    results in <フォルダ>」の ``in`` は保存先を導いている。
    """
    from backend.free.core.intent_vocab import CLAUSE_BOUNDARY_RE

    pattern_text, how = _search_pattern_and_introduced(description)
    if how != "quoted" and CLAUSE_BOUNDARY_RE.search(pattern_text):
        return None
    if not how and len(pattern_text.split()) > 1:
        return None
    directory = _named_directory(strip_command_literals(description))
    if pattern_text and directory:
        return {"pattern": pattern_text, "directory": directory}
    return None


#: 検索のパターンを括る引用 (開きと閉じを対で見る)。``intent_vocab.QUOTED_SPAN_RE``
#: (「…」『…』"…") に、コードの括り `…` と、語の外にある '…' を足したもの —
#: 短縮形・所有格の ``'`` (``Let's`` / ``what's``) と、種類の違う引用符を対にしない
#: (2026-10-04 反証レビュー 3 周目 MED-A: 動詞まで括りとして伏せていた)。
_SEARCH_QUOTED_SPAN_RE = re.compile(
    QUOTED_SPAN_RE.pattern + r"|`[^`]*`|(?<!\w)'[^']*'(?!\w)",
)

#: 検索の場所を導く前置詞 (「search for X in <場所>」)
_SEARCH_SCOPE_PREP = r"(?:in|within|under|inside)"
#: 検索の動詞 (「search <場所> for X」では場所を直接の目的語に取る)
_SEARCH_VERB_LEAD = r"(?:search|grep|find|look)"


def _named_directory(description: str) -> str:
    """検索の場所の前置詞が導く実在のフォルダ (ドライブ付き、最初の 1 件)。無ければ空文字。"""
    found = _named_directories(description)
    return found[0] if found else ""


def _named_directories(description: str, *, skip_write_scopes: bool = False) -> list[str]:
    """検索の場所の前置詞が導く実在のフォルダ (ドライブ付き、出現順・重複なし)。

    前置詞はパスと同じ節に要る (節の境界で切った最後の節に在る)。検索の動詞の直後の
    パス (「Search <フォルダ> for TODO」) も探す場所。パスは検索の動詞を持つ節か、それより
    前の節 (「In <フォルダ>, search …」) に在るものだけ — 後ろの節の保存先
    (「… and save the results in <フォルダ>」) は取らない。``skip_write_scopes`` なら、検索の
    動詞より前の名指しのうち、直後の節が書込みの命令形で始まるもの (「In <フォルダ>, write a
    report …」、``clause_head_is_write_verb``) を除く。
    """
    from backend.free.agent.tool_judge_args import drive_path_spans, iter_drive_dir_paths
    from backend.free.core.intent_vocab import CLAUSE_BOUNDARY_RE, clause_head_is_write_verb

    drives = list(zip(
        iter_drive_dir_paths(description), drive_path_spans(description), strict=True,
    ))
    # パスと括りの中身は節の境界・動詞に数えない (位置を保って伏せる)
    masked = description
    for start, end in [span for _, span in drives] + [
        m.span() for m in _SEARCH_QUOTED_SPAN_RE.finditer(description)
    ]:
        masked = masked[:start] + "_" * (end - start) + masked[end:]
    verb = re.search(rf"\b{_SEARCH_VERB_LEAD}\b", masked, re.IGNORECASE)
    if verb is None:
        return []
    after = CLAUSE_BOUNDARY_RE.search(masked, verb.end())
    search_clause_end = after.start() if after else len(masked)

    found: list[str] = []
    prev_end = 0
    for drive_path, (start, end) in drives:
        clause = CLAUSE_BOUNDARY_RE.split(masked[prev_end:start])[-1]
        prev_end = end
        if start >= search_clause_end:
            break
        if skip_write_scopes and start < verb.start() and clause_head_is_write_verb(
            next((c for c in CLAUSE_BOUNDARY_RE.split(masked[end:]) if c.strip()), ""),
        ):
            continue
        if not re.search(
            rf"\b{_SEARCH_SCOPE_PREP}\b|\b{_SEARCH_VERB_LEAD}\W*$", clause, re.IGNORECASE,
        ):
            continue
        try:
            if Path(drive_path.path).is_dir() and drive_path.path not in found:
                found.append(drive_path.path)
        except (OSError, ValueError):
            continue
    return found


def extract_search_pattern(description: str) -> str:
    """タスク記述から search_code の検索パターンを抽出する

    バッククォート / 引用符で囲まれた語を優先し、無ければ
    「search for X in ...」型の X を採る。抽出できなければ空文字。
    ドライブ付きのパス (``iter_drive_dir_paths``) は探す場所であってパターンでは
    ないので、括られていても採らず、括り無しの抽出の前に本文から除く。
    """
    return _search_pattern_and_introduced(description)[0]


def _search_pattern_and_introduced(description: str) -> tuple[str, str]:
    """:func:`extract_search_pattern` の本体。

    ``(パターン, 導き方)`` を返す。導き方は ``"quoted"`` (括り) / ``"for"`` / ``""``。
    """
    from backend.free.agent.tool_judge_args import drive_path_spans, without_drive_paths

    drive_spans = drive_path_spans(description)
    for m in _SEARCH_QUOTED_SPAN_RE.finditer(description):
        inner = m.group(0)[1:-1].strip()
        if inner and not any(m.start() < end and start < m.end() for start, end in drive_spans):
            return inner, "quoted"
    text = without_drive_paths(description)
    m = re.search(
        rf'{_SEARCH_VERB_LEAD}\s+(for\s+)?(.+?)'
        rf'(?:\s+{_SEARCH_SCOPE_PREP}\b.*)?$',
        text.strip(), re.IGNORECASE,
    )
    if m:
        return m.group(2).strip(), "for" if m.group(1) is not None else ""
    return "", ""


def extract_command(description: str) -> str:
    """タスク記述からシェルコマンドを抽出する

    バッククォート内のコマンドや、「Run ...」パターンを認識する。
    """
    m = re.search(r'`([^`]+)`', description)
    if m:
        return m.group(1)
    m = re.search(
        r'(?:run|execute|実行)\s+(.+?)(?:\s*$)', description, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# 引数正規化
# ---------------------------------------------------------------------------

def normalize_read_file_args(args: dict, query: str = "") -> dict:
    """read_file の引数名を正規化する

    LLM が file_path を省略した場合、クエリからパスを抽出して補完する。
    """
    normalized: dict = {}

    path_candidates = ["file_path", "path", "filepath", "filename", "file"]
    for key in path_candidates:
        if key in args and args[key]:
            normalized["file_path"] = args[key]
            return normalized

    if query:
        from backend.free.agent.tool_call_judge import _extract_file_path
        extracted = _extract_file_path(query)
        if extracted:
            normalized["file_path"] = extracted
            return normalized

    return args


def normalize_write_file_args(args: dict) -> dict:
    """write_file の引数名を正規化する

    LLM が file_path / content 以外の引数名を使った場合に救済する。
    例: output_content → content, path → file_path
    """
    normalized: dict = {}

    path_candidates = ["file_path", "path", "filepath", "filename", "file"]
    for key in path_candidates:
        if key in args:
            normalized["file_path"] = args[key]
            break

    content_candidates = [
        "content", "output_content", "text", "data",
        "file_content", "body", "output",
    ]
    for key in content_candidates:
        if key in args:
            normalized["content"] = args[key]
            break

    content = normalized.get("content", "")
    file_path = normalized.get("file_path", "")
    if content and looks_like_path_not_content(content, file_path):
        logger.warning(
            "write_file content looks like a file path, "
            "clearing to trigger content generation: %r",
            content,
        )
        normalized.pop("content")

    return normalized
