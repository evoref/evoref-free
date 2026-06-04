"""ツール結果の query 連動抽出 (assist モデルへ委譲)。

弱い base モデルがノイズ/長文のツール結果から答えを拾えない問題に対し、
読解・抽出を assist モデルに逸らして base の接地負荷を下げる。
CLAUDE.md §6「抽出/判定は assist で実行」の不変則に整合する。

assist 呼び出しを本ファイルに隔離することで、Deliberative 側に直接
``assist_client.generate`` を持ち込まずに済む (purpose 監査の除外登録を回避)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.free.agent.meta_cognitive_utils import is_tool_error
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.llm.assist_client import AssistModelClient

logger = get_logger("agent.tool_result_digest")

_NO_INFO = "NO_RELEVANT_INFO"

_DIGEST_SYSTEM = (
    "You extract, from a tool execution result, ONLY the information needed to "
    "answer the user's question. Output the key facts (numbers, names, dates, "
    "conditions, short relevant text) concisely, in the SAME language as the "
    "tool result. Do not add analysis, prefaces, or invented data. "
    f"If the result clearly contains no information relevant to the question, or "
    f"is an error/empty, output exactly '{_NO_INFO}' and nothing else."
)


def _content_of(result: dict) -> str:
    """OAI 応答 dict から ``choices[0].message.content`` を安全に取り出す。"""
    try:
        return result["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


async def digest_tool_result(
    assist_client: "AssistModelClient | None",
    *,
    query: str,
    tool_name: str,
    tool_result: str,
) -> str | None:
    """ツール結果から query 連動で要点を抽出する。

    抽出に成功したら digest 文字列を返す。抽出すべき情報が無い (``NO_RELEVANT_INFO``)、
    ツール結果が空/エラー、assist 不在、呼出失敗のときは ``None`` を返す
    (呼出側で raw ツール結果へ安全退避させる)。assist が落ちていても例外を投げない。
    """
    if assist_client is None:
        return None
    if not tool_result or is_tool_error(tool_result):
        return None
    try:
        messages = [
            {"role": "system", "content": _DIGEST_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"User question:\n{query}\n\n"
                    f"Tool ({tool_name}) result:\n{tool_result}\n\n"
                    "Extract the relevant facts now."
                ),
            },
        ]
        result = await assist_client.generate(
            messages,
            stream=False,
            temperature=0.2,
            max_tokens=512,
            purpose="tool_result_digest",
        )
        digest = _content_of(result).strip()
    except Exception as e:  # noqa: BLE001 — degraded 安全 (raw 退避)
        logger.warning("tool_result_digest failed (%r); using raw result", e)
        return None
    if not digest or digest.strip(" .'\"`\n").upper() == _NO_INFO:
        return None
    logger.info(
        "tool_result_digest: tool=%s, %d -> %d chars",
        tool_name, len(tool_result), len(digest),
    )
    return digest
