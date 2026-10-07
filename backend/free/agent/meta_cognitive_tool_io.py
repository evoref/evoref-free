"""ツール呼び出しの抽出と実行結果の判定

LLM 出力から JSON のツール呼び出しを取り出す側と、返ってきた結果が
成功か / 情報を含むかを判定する側をまとめる。どちらもツールとの
境界の解釈で、テキスト整形とは別の責務。
"""

from __future__ import annotations

import json
import re

from typing import Iterator
from backend.free.constants import (
    COMMAND_EXIT_CODE_PREFIX,
    DIRECTORY_NOT_FOUND_PREFIX,
    FETCH_BLOCKED_PREFIX,
    FETCH_ERROR_PREFIX,
    FETCH_UNREACHABLE_ERRORS,
    FETCH_UNRESOLVED_HOST_PREFIX,
    FILE_AMBIGUOUS_PREFIX,
    FILE_NOT_FOUND_PREFIX,
    READ_FILE_META_PREFIX,
    SEARCH_HISTORY_NO_RESULTS_PREFIX,
    SEARCH_NO_LOCATION_ERROR,
)


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


def fetch_error_type(text: str) -> str:
    """``fetch_url`` の取得の例外の型名 (``<先頭> (<例外の型>): <理由>``、tools/web_fetch.py)。違えば空文字。"""
    if not text.startswith(FETCH_ERROR_PREFIX):
        return ""
    return text[len(FETCH_ERROR_PREFIX):].lstrip(" (").split(")", 1)[0]


def read_file_result_path(text: str) -> str | None:
    """``read_file`` の結果ならメタ行のパス、それ以外は ``None`` (純粋関数)。

    ``read_file`` は先頭に ``[file: <path> | lines: N | chars: M]`` を付ける
    (tools/filesystem.py)。自前の形式だけを見る (字句の鍵、不変則 #14)。
    """
    if not (text or "").startswith(READ_FILE_META_PREFIX):
        return None
    return text[len(READ_FILE_META_PREFIX):].split(" | ", 1)[0].strip() or None


def tool_error_kind(text: str) -> str:
    """ツールのエラー文の種別 (利用者向けの注記の鍵、i18n ``agent.retrieval_failed.<種別>``)。

    自前のツールが出す先頭の形式だけを見る (語彙ではなく字句の鍵、不変則 #14)。
    """
    if text.startswith(FETCH_UNRESOLVED_HOST_PREFIX):
        return "unresolved_host"
    if text.startswith(FETCH_BLOCKED_PREFIX):
        return "blocked"
    if text.startswith(SEARCH_NO_LOCATION_ERROR):
        return "no_location"
    if text.startswith((FILE_NOT_FOUND_PREFIX, DIRECTORY_NOT_FOUND_PREFIX)):
        return "not_found"
    if text.startswith(FILE_AMBIGUOUS_PREFIX):
        return "ambiguous"
    if fetch_error_type(text) in FETCH_UNREACHABLE_ERRORS:
        return "unreachable"
    return "error"


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


#: 「ツールは正常終了したが結果が 0 件」を示すツール別の先頭マーカー。
#: 新しいツールで空振り表現を足すときはここへ登録する (判定側は触らない)。
#: 成否判定 (``tool_result_succeeded``) と issue 台帳
#: (``tools_registry._record_tool_issue``) の両方がこの 1 つの表を見る。
_TOOL_EMPTY_RESULT_PREFIXES: dict[str, tuple[str, ...]] = {
    "search_history": (SEARCH_HISTORY_NO_RESULTS_PREFIX,),
    "search_code": ("No matches found",),
}

#: 結果本文のマーカーで「走ったプログラム自身の失敗」を判定するツール。
_EXIT_CODE_TOOLS = frozenset({"run_command", "run_command_readonly"})


def tool_result_lacks_information(tool_name: str | None, text: str) -> bool:
    """ツールは正常終了したが結果が情報ゼロ (空振り) かを判定する。"""
    prefixes = _TOOL_EMPTY_RESULT_PREFIXES.get(tool_name or "")
    if not prefixes:
        return False
    return text.startswith(prefixes)


def tool_result_succeeded(tool_name: str | None, text: str) -> bool:
    """ツール実行が「役に立つ結果を返した」かを判定する (学習シグナルの SSOT)。

    ``is_tool_error`` (``Error:`` プレフィックス) は **ツールラッパ自身が例外を
    出したか** しか見ない。「実行できた」と「役に立った」は別物で、後者でない
    ものを成功として扱うと Level 0/1 の選択圧が反転する:

    - ``run_command`` が非ゼロ終了 → 実行はできたがコマンドは失敗
      (``command_run_failed``)
    - ``search_history`` が 0 件 → 実行はできたが情報は得られていない
      (実インシデント 2026-07-29 ライブ監査: 直近の会話で答えられる質問に
      ``search_history`` が発火し 0 件で終わったターンが ``reward=1.0`` /
      ``outcome=success`` で記録され、「空振りするルーティング」が正例として
      強化されていた)

    ``run_command`` の exit code 判定だけが deliberative 側にローカル実装され、
    meta_cognitive 側には無いという非対称もここへ集約して解消する。

    Args:
        tool_name: 実行したツール名 (不明なら None)。
        text: ツールの戻り値文字列。
    """
    if is_tool_error(text):
        return False
    if tool_name in _EXIT_CODE_TOOLS and command_run_failed(text):
        return False
    return not tool_result_lacks_information(tool_name, text)

# ---------------------------------------------------------------------------
# JSON ツール呼び出し抽出ヘルパー
# ---------------------------------------------------------------------------


def try_parse_tool_dict(text: str) -> dict | None:
    """text を JSON としてパースし、'tool' キーを持つ dict ならそれを返す。

    パース失敗・dict 以外・'tool' キー欠落のいずれも None を返す。
    meta_cognitive の _parse_tool_call から共有される。
    """
    from backend.free.llm.json_extract import escape_windows_path_backslashes

    try:
        data = json.loads(escape_windows_path_backslashes(text))
    except (json.JSONDecodeError, RecursionError):
        return None
    if isinstance(data, dict) and "tool" in data:
        return data
    return None


# テンプレート由来のツールコール (OAI JSON でない生テキスト) 抽出。
# gemma 系: <|tool_call>call:NAME(k: "v", k2: "v2")<tool_call|>
# 引数を波括弧で囲む形 (call:NAME{k: 'v'}) も書く (2026-10-07 audit_replay R10:
# 解析できず、構文のままタスクの結果として利用者に出た)。
# マーカーの揺れ (<|tool_call|> / <tool_call> / </tool_call>) を寛容に許容する。
_TEMPLATE_TOOL_CALL_RE = re.compile(
    r"<\|?tool_call\|?>\s*call:\s*(?P<name>\w+)\s*"
    r"(?:\((?P<args>.*?)\)|\{(?P<brace_args>.*?)\})"
    r"\s*</?\|?tool_call\|?>",
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
    # gemma 系: call:NAME(args) / call:NAME{args}
    m = _TEMPLATE_TOOL_CALL_RE.search(text)
    if m is None:
        return None
    args = m.group("args")
    if args is None:
        args = m.group("brace_args")
    return {"tool": m.group("name"), "args": _parse_template_args(args)}


def iter_balanced_brace_substrings(text: str) -> Iterator[str]:
    """text 内の各 '{' から始まるバランスの取れた {...} 部分文字列を開き位置の順に yield する。

    文字列リテラル内の '{' '}' は考慮しない（素朴なバランス検出）。
    1 回の走査で対応を取るので、閉じない '{' が大量にあっても線形で終わる。
    """
    open_positions: list[int] = []
    close_of: dict[int, int] = {}
    for i, ch in enumerate(text):
        if ch == '{':
            open_positions.append(i)
        elif ch == '}' and open_positions:
            close_of[open_positions.pop()] = i
    for start in sorted(close_of):
        yield text[start:close_of[start] + 1]
