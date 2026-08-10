"""ベースモデルのネイティブ tool calling 用の変換層 (docs/c_14 §1.3)

``assist_model.residency: on_demand`` のチャット中はアシストが停止しており、
``purpose="tool_judgment"`` の判定が撃てない。その代わりに **ベースモデル自身の
OAI 互換 tool calling** でツールを選ばせる。本モジュールはその境界だけを担う:

- :func:`build_oai_tools` — ``ToolsRegistry`` → OAI ``tools`` 配列
- :func:`parse_native_tool_call` — 応答 message → ``(tool_name, args)``

**判定結果の安全性検証はここでは行わない**。mode ゲート・引数グラウンディング・
readonly 違反の拒否は呼出側 (``ToolCallJudge``) の既存 funnel (``_finalize``)
が担当する — アシスト由来の判定と同じ経路を通すことで、片方にだけガードが
付く非対称を作らないため (c_14 §8-6)。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.agent.tools_registry import ToolsRegistry

logger = get_logger("agent.native_tools")

#: OAI function schema へ載せる引数の型。``ToolDefinition.parameters`` の
#: ``type`` をそのまま使うが、未知の型は文字列に丸めて schema 不正を避ける。
_ALLOWED_JSON_TYPES = frozenset(
    {"string", "number", "integer", "boolean", "array", "object"},
)


def build_oai_tools(
    tools_registry: "ToolsRegistry", mode: str,
) -> list[dict[str, Any]]:
    """``mode`` で実行可能なツールを OAI ``tools`` 配列へ変換する。

    ``hidden=True`` のツールも含める点が ``get_descriptions_text`` と異なる。
    hidden はプロンプトのツール一覧に出さないための印であって「使わせない」印では
    なく、chat の ``run_command_readonly`` のように **コード側が注入する前提の
    ツール**が該当する。ネイティブ tool calling ではモデル自身に選ばせるので、
    mode で実行可能なものは候補に載せる必要がある。

    Args:
        tools_registry: 登録済みツール。
        mode: ``"chat"`` / ``"create"``。``ToolDefinition.modes`` で絞る。

    Returns:
        OAI 互換の ``tools`` 配列。該当なしなら空リスト。
    """
    tools: list[dict[str, Any]] = []
    for name in tools_registry.list_names():
        if not tools_registry.is_available(name, mode):
            continue
        tool = tools_registry.get(name)
        if tool is None:
            continue
        required = sorted(tools_registry.required_params(name))
        properties: dict[str, Any] = {}
        for key, spec in (tool.parameters or {}).items():
            ptype = str((spec or {}).get("type", "string"))
            if ptype not in _ALLOWED_JSON_TYPES:
                ptype = "string"
            prop: dict[str, Any] = {"type": ptype}
            desc = (spec or {}).get("description")
            if desc:
                prop["description"] = str(desc)
            properties[key] = prop
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    # parameters に無い必須引数 (session_id 等の非公開引数) は
                    # モデルに求めない。コード側が後段で注入する。
                    "required": [r for r in required if r in properties],
                },
            },
        })
    return tools


def parse_native_tool_call(
    message: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """応答 message から最初の tool call を ``(name, args)`` で取り出す。

    llama-server は OAI 互換で ``message.tool_calls[].function.{name,arguments}``
    を返し、``arguments`` は **JSON 文字列**。壊れた JSON / 名前欠落は
    ``None`` を返して呼出側の既存フォールバックへ倒す (例外は投げない)。

    複数 tool call が返っても先頭だけ採用する。既存の実行経路
    (``DeliberativeAgent._judge_and_execute_tool``) が 1 ターン 1 ツール前提の
    ため、ここで無理に並列実行へ広げない。
    """
    if not isinstance(message, dict):
        return None
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    fn = (calls[0] or {}).get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None

    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str) and raw_args.strip():
        try:
            args = json.loads(raw_args)
        except (ValueError, TypeError):
            logger.info(
                "native tool call: unparseable arguments for %s: %r",
                name, raw_args[:120],
            )
            return None
    else:
        args = {}

    if not isinstance(args, dict):
        logger.info(
            "native tool call: arguments for %s is not an object (%s)",
            name, type(args).__name__,
        )
        return None
    # 値は既存ガード (grounding / readonly / 必須引数検証) が見るので、ここでは
    # キーが文字列であることだけ保証する。
    return name, {str(k): v for k, v in args.items()}
