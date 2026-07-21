"""ツール結果の query 連動抽出 (assist モデルへ委譲)。

弱い base モデルがノイズ/長文のツール結果から答えを拾えない問題に対し、
読解・抽出を assist モデルに逸らして base の接地負荷を下げる。
CLAUDE.md §6「抽出/判定は assist で実行」の不変則に整合する。

assist 呼び出しを本ファイルに隔離することで、Deliberative 側に直接
``assist_client.generate`` を持ち込まずに済む (purpose 監査の除外登録を回避)。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from backend.free.agent.meta_cognitive_utils import is_tool_error
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.llm.assist_client import AssistModelClient

logger = get_logger("agent.tool_result_digest")

_NO_INFO = "NO_RELEVANT_INFO"
_DIGIT_RUN_RE = re.compile(r"\d+")

_DIGEST_SYSTEM = (
    "You extract, from a tool execution result, ONLY the information needed to "
    "answer the user's question. If the result contains information relevant to "
    "the question, output the key facts (numbers, names, dates, conditions, "
    "short relevant text) concisely, in the SAME language as the tool result. "
    "Do not add analysis, prefaces, or invented data. "
    f"If the result clearly contains no information relevant to the question, or "
    f"is an error/empty, this rule OVERRIDES the language-matching rule above: "
    f"output EXACTLY the literal English token {_NO_INFO} — untranslated, "
    f"unparaphrased, no other words, no punctuation — and nothing else."
)


def _numeric_claims_grounded(digest: str, tool_result: str) -> bool:
    """digest 内の数字列がすべて raw tool_result に部分文字列として存在するか。

    digest 抽出は 512 token 上限の小型 assist モデルが担うため、長い raw
    tool_result 中の数値を読み違え・転記ミスすることがある (実インシデント
    2026-07-20: search_history が正しいターン「3の5乗はいくつ？」を返した
    のに、digest が答え「243」を「125」と誤って抽出した)。digest に現れる
    数字列が raw tool_result に一つも見当たらない場合は捏造の疑いが強いため
    grounding 失敗として扱う。digest に数字列が無ければ判定不要 (True)。
    """
    digest_numbers = _DIGIT_RUN_RE.findall(digest)
    if not digest_numbers:
        return True
    return all(n in tool_result for n in digest_numbers)


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

    抽出に成功したら digest 文字列を返す。ツール結果が空/エラー、assist 不在、
    呼出失敗など「抽出そのものができない」場合は ``None`` を返す (呼出側で raw
    ツール結果へ安全退避させる)。一方 assist が抽出に成功した上で質問に関連する
    情報が無いと判定した場合 (``NO_RELEVANT_INFO``) は空文字列 ``""`` を返す。
    ``None`` と区別することで、呼出側が「関連なしと確定した」ケースまで raw
    ツール結果 (無関係な過去セッションの内容等) へフォールバックしないようにする。
    assist が落ちていても例外を投げない。
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
    except Exception as e:
        logger.warning("tool_result_digest failed (%r); using raw result", e)
        return None
    # 完全一致に加え、小型 assist モデルが指示に反して定型句を前後に付けた
    # 場合 (例: "結果: NO_RELEVANT_INFO") も救済する。ただし自然文パラフレーズ
    # (例: "音楽のジャンルについて、具体的な好みについての情報はない。") は
    # トークン自体を含まないため、この部分一致では救済できない
    # (根本対応は上記システムプロンプトの優先順位明確化)。
    # トークン以外の残り文字数に上限を設けるのは、「トークンに加えて実際に
    # 抽出した回答も含む」digest (例: "NO_RELEVANT_INFO ですが部分的に...")
    # まで空文字列に握り潰し、実際に見つかった情報を握り潰さないための保険
    # (レビューで指摘)。
    digest_normalized = digest.strip(" .'\"`\n").upper()
    remainder_len = len(digest_normalized) - len(_NO_INFO)
    if digest_normalized == _NO_INFO or (
        _NO_INFO in digest_normalized and remainder_len <= 20
    ):
        return ""
    if not digest:
        return None
    if not _numeric_claims_grounded(digest, tool_result):
        logger.warning(
            "tool_result_digest: digested number(s) not found in raw tool_result "
            "(tool=%s) - discarding possibly hallucinated digest, falling back to raw",
            tool_name,
        )
        return None
    logger.info(
        "tool_result_digest: tool=%s, %d -> %d chars",
        tool_name, len(tool_result), len(digest),
    )
    return digest
