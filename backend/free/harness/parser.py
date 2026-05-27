"""

LLM が生成したテキストから ``Action`` 列を抽出する。

期待する出力フォーマット (優先順):

1. ``<actions>...</actions>`` タグで囲まれた JSON 配列
2. ```` ```actions ... ``` ```` (タグ付き fenced code block) 内の JSON 配列
3. ```` ```json ... ``` ```` 内の JSON 配列 (フォールバック)
4. テキスト中で最初に現れる ``[`` / ``]`` でバランスする JSON 配列

いずれにも該当しない、または JSON が ``Action`` 仕様を満たさない場合は
:class:`ParseError` を送出する。呼び出し側 (Harness) は ``ParseError`` を
捕捉して failure_pattern として記録する想定。
"""

from __future__ import annotations

import json
import re

from backend.free.harness.action import Action, action_from_dict


class ParseError(ValueError):
    """LLM 出力から Action 列を抽出できなかった場合に送出される。"""


_TAG_RE = re.compile(r"<actions>(.*?)</actions>", re.DOTALL | re.IGNORECASE)
_FENCE_ACTIONS_RE = re.compile(
    r"```actions\s*(.*?)```", re.DOTALL | re.IGNORECASE,
)
_FENCE_JSON_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_actions(text: str) -> list[Action]:
    """LLM 出力から ``Action`` 列を抽出する。

    Args:
        text: LLM のレスポンス本文。

    Returns:
        ``Action`` のリスト。空リストは「明示的な空配列」と同義
        (パーサとしては成功扱い)。

    Raises:
        ParseError: 期待フォーマットに合致するブロックが無いか、JSON として
            不正、または ``action_from_dict`` で復元できないエントリを含む。
    """
    payload = _extract_payload(text)
    if payload is None:
        raise ParseError("no <actions>/```actions```/```json``` block found")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ParseError(f"actions block is not valid JSON: {exc}") from exc
    if not isinstance(decoded, list):
        raise ParseError(
            f"actions block must be a JSON array, got {type(decoded).__name__}",
        )
    actions: list[Action] = []
    for i, entry in enumerate(decoded):
        if not isinstance(entry, dict):
            raise ParseError(f"actions[{i}] must be an object")
        try:
            actions.append(action_from_dict(entry))
        except ValueError as exc:
            raise ParseError(f"actions[{i}] invalid: {exc}") from exc
    return actions


def _extract_payload(text: str) -> str | None:
    """テキストから JSON 配列文字列を取り出す。優先順は docstring 参照。"""
    if (m := _TAG_RE.search(text)) is not None:
        return m.group(1).strip()
    if (m := _FENCE_ACTIONS_RE.search(text)) is not None:
        return m.group(1).strip()
    if (m := _FENCE_JSON_RE.search(text)) is not None:
        return m.group(1).strip()
    return _extract_balanced_array(text)


def _extract_balanced_array(text: str) -> str | None:
    """テキスト中で最初に現れる ``[`` から対応する ``]`` までを返す。

    文字列リテラル内の ``[`` ``]`` はバックスラッシュエスケープと
    クォート状態を簡易追跡してスキップする。
    """
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
