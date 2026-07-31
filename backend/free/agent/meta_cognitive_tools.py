"""Meta-Cognitive ツール推論: タスク記述からのツール決定・引数正規化"""

from __future__ import annotations

import re

from backend.free.agent.meta_cognitive_utils import looks_like_path_not_content
from backend.free.agent.safety_patterns import strip_command_literals
from backend.log_config import get_logger
from backend.free.core.intent_vocab import WRITE_VERB_RE

logger = get_logger("agent.meta_cognitive.tools")


# ---------------------------------------------------------------------------
# ツール推論パターン（優先度順）
# ---------------------------------------------------------------------------

_TOOL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # write_file: 作成・修正系 (語彙は core.intent_vocab が SSOT。
    # meta_cognitive_tasks の書込み期待判定と同一定義を持っていた)
    (WRITE_VERB_RE, "write_file"),
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

    # バッククォート内コマンドの引数パスは読み書きの対象ではない。パス抽出は
    # コマンドを除いた本文に対して行う (コマンド抽出は生の description を見る)。
    path_source = strip_command_literals(description)

    for pattern, tool_name in _TOOL_PATTERNS:
        if not pattern.search(description):
            continue

        if tool_name == "write_file":
            file_path = _extract_file_path(path_source)
            if file_path:
                return ("write_file", {"file_path": file_path})

        elif tool_name == "read_file":
            file_path = _extract_file_path(path_source)
            if file_path:
                return ("read_file", {"file_path": file_path})

        elif tool_name == "run_command":
            cmd = extract_command(description)
            if cmd:
                return ("run_command", {"command": cmd})

        elif tool_name == "search_code":
            return ("search_code", {"pattern": "", "directory": "."})

    return None


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
