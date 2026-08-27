"""明示された出力制約の検証と、1 回だけの修復生成。

**なぜ要るか**: 文字数指定 (「20 文字ちょうどで」) や形式指定 (「箇条書きで」
「数値だけ」) は、指定が本文にあり、守れたかは数えれば分かる。ところが実装は
長らく **プロンプトへ注記を足すだけ** で、守れたかを誰も見ていなかった
(``core.inference._char_limit_note`` / ``_output_form_note``)。その後
``LengthDisclosureFilter`` が「破った」と末尾で開示するところまで進んだが、
**ユーザーには依然として違反した回答がそのまま出る**。

ここでは開示の一歩先 — 違反を測って、その測定値をモデルへ返し、**1 回だけ**
書き直させる。閉ループにはしない (1 回で直らなければ元の回答を出して開示する)。

**コストが問題にならない理由**: 発火するのは決定論で検証できる制約が発話に
含まれるターンだけで、そのときの回答は定義上短い。修復生成は元の messages を
接頭辞として再利用するので、llama-server の prefix KV キャッシュがそのまま
効き、再プリフィルは追加した 2 メッセージ分だけで済む。

**なぜ層ではなくここか**: 検証だけならフィルタ (全ストリーミング経路が組む)
に置けるが、修復は「元の messages とクライアント」が要る。呼出は
deliberative / reactive_light の両方から行い、層に依存させない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.free.core.stream_filter import strip_thinking_blocks
from backend.free.core.text_quality import (
    has_verifiable_output_constraint,
    length_disclosure_note,
    match_length_directive,
    violates_enumeration_count,
    violates_length_constraint,
    violates_output_form,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.api.chat.chat_types import ChatMessage, GenerationParams

logger = get_logger("api.chat.constraint_repair")

#: 修復生成の温度。書き直しは創作ではなく指定への当てはめなので低くする
#: (ツール接地ターンで温度を下げているのと同じ理由)。
REPAIR_TEMPERATURE = 0.2

#: 修復生成のトークン上限の下限。指定が「300 文字ちょうど」でも収まるだけの
#: 余裕を持たせる。呼出側の ``max_tokens`` があればそちらを優先する。
REPAIR_MIN_MAX_TOKENS = 512


def violation_reason(query: str, response: str) -> str | None:
    """発話の出力制約に対する違反理由 (英語、ログ用)。守れていれば ``None``。

    長さと形式の両方を見る。純粋関数。
    """
    return (
        violates_length_constraint(query, response)
        or violates_output_form(query, response)
        or violates_enumeration_count(query, response)
    )


def needs_verification(query: str) -> bool:
    """このターンを検証・修復の対象にするか (純粋関数)。"""
    return has_verifiable_output_constraint(query)


def build_repair_note(query: str, response: str, reason: str) -> str:
    """測定値と **目標値・差分** を突き付ける修復指示を作る (純粋関数)。

    「守れ」ではなく「いま何文字で、目標まであと何文字か」を渡す。守れという
    指示は最初のターンで既に渡していて破られているので、繰り返しても意味がない。
    差分を埋める作業に落とすために実測値を書く。

    **目標値を再掲する理由**: 当初は「指定は会話履歴に残っている」として省いて
    いたが、実測 (2026-08-25 ライブ監査) では小型モデルが指定値を拾えず、
    **1 文字も変えない同じ答え**を返した (「ちょうど50文字で」→ 45 文字 →
    修復後も 45 文字)。目標が遠くにあるほど、直前の指示として書く価値が上がる。
    """
    measured = len((response or "").strip())
    directive = match_length_directive(query or "")
    lines = [
        "直前の回答は指定を満たしていない。",
        f"実測: {measured} 文字 / 判定: {reason}。",
    ]
    if directive is not None:
        kind, expected = directive
        diff = expected - measured
        if kind == "limit":
            lines.append(
                f"指定は {expected} 文字以内。{abs(diff)} 文字超過しているので、"
                "語を削って収めること。",
            )
        else:
            if diff > 0:
                lines.append(
                    f"指定は {expected} 文字ちょうど。{diff} 文字足りないので、"
                    "語を足して長さを合わせること。",
                )
            elif diff < 0:
                lines.append(
                    f"指定は {expected} 文字ちょうど。{-diff} 文字多いので、"
                    "語を削って長さを合わせること。",
                )
            else:
                lines.append(f"指定は {expected} 文字ちょうど。")
    lines.append(
        "指定どおりに書き直した**本文だけ**を出力すること。"
        "謝罪・前置き・変更点の説明・コードフェンスは付けない。"
        "**同じ文をそのまま返さないこと。**"
        "書き終えたら自分で数え直し、合っていなければ語を足し引きして合わせること。",
    )
    return chr(10).join(lines)


async def repair_if_violated(
    *,
    query: str,
    response: str,
    messages: list["ChatMessage"],
    client,
    max_tokens: int | None = None,
    generation_params: "GenerationParams | None" = None,
) -> tuple[str, str | None]:
    """制約違反なら 1 回だけ書き直させる。

    Args:
        query: ユーザー発話 (制約の出どころ)。
        response: フィルタ適用後の応答本文。
        messages: 元の生成に使った messages。修復生成の接頭辞として再利用する。
        client: ``generate`` を持つ LLM クライアント。
        max_tokens: 修復生成のトークン上限。
        generation_params: 元のモード別生成パラメータ (温度だけ上書きする)。

    Returns:
        ``(最終本文, 未解決の違反理由)``。修復に成功したか元から違反が無ければ
        第 2 要素は ``None``。修復しても直らなければ **元の本文** を返し、理由を
        添える (呼出側が開示に使う)。悪化させないため、修復版は完全に指定を
        満たしたときだけ採用する。
    """
    reason = violation_reason(query, response)
    if reason is None:
        return response, None
    if client is None or not hasattr(client, "generate"):
        return response, reason

    logger.info("Output constraint violated, attempting one repair (%s)", reason)
    repair_messages = [
        *messages,
        {"role": "assistant", "content": response},
        {"role": "user", "content": build_repair_note(query, response, reason)},
    ]
    gp = dict(generation_params or {})
    kwargs: dict = {
        "stream": False,
        "id_slot": getattr(client, "chat_slot", None),
        "temperature": min(gp.get("temperature", REPAIR_TEMPERATURE), REPAIR_TEMPERATURE),
        "max_tokens": max(max_tokens or 0, REPAIR_MIN_MAX_TOKENS),
    }
    for key in ("top_p", "top_k", "presence_penalty", "frequency_penalty",
                "repetition_penalty"):
        if gp.get(key) is not None:
            kwargs[key] = gp[key]

    try:
        result = await client.generate(repair_messages, **kwargs)
    except Exception as e:
        logger.warning("Constraint repair generation failed (keeping original): %s", e)
        return response, reason

    repaired = _extract_text(result)
    if not repaired:
        return response, reason

    still = violation_reason(query, repaired)
    if still is None:
        logger.info(
            "Output constraint repaired (%d -> %d chars)",
            len(response.strip()), len(repaired.strip()),
        )
        return repaired, None
    logger.info("Repair did not satisfy the constraint either (%s); keeping original", still)
    return response, reason


def _extract_text(result) -> str:
    """``generate(stream=False)`` の戻りから本文を取り出す (思考ブロック除去込み)。"""
    if isinstance(result, str):
        content = result
    elif isinstance(result, dict):
        choices = result.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content") or ""
    else:
        return ""
    return strip_thinking_blocks(content).strip()

async def verify_and_repair_sync(
    *,
    query: str,
    response: str,
    messages: "list[ChatMessage]",
    client,
    max_tokens: int | None = None,
    generation_params: "GenerationParams | None" = None,
) -> str:
    """同期応答用: 検証 → 1 回だけ修復 → 直らなければ開示注記を付けて返す。

    ストリーミング経路の ``_emit_verified_output`` と **同じ結末** を、本文を
    まとめて持っている同期経路にも与える。検証も修復もストリーミング側に
    しか無く、``stream=False`` の API 呼び出しは制約違反がそのまま返っていた
    (``_log_chat_outcome`` が同期 deliberative だけ漏れていたのと同じ非対称)。

    実測 (2026-08-27 ライブ監査、API 経由): 「木の家の良さを50字で説明して
    ください。」に 34 文字で答え、修復も開示も走らなかった。検証器自体は
    違反を正しく検出できている::

        violates_length_constraint(q, a)
          → 'asked for exactly 50 chars but the answer is 34'

    発火するのは ``needs_verification(query)`` が真のターンだけなので、
    通常の会話にコストは乗らない。
    """
    if not response.strip() or not needs_verification(query):
        return response
    final_text, unresolved = await repair_if_violated(
        query=query,
        response=response,
        messages=messages,
        client=client,
        max_tokens=max_tokens,
        generation_params=generation_params,
    )
    if unresolved is not None:
        final_text += length_disclosure_note(query, final_text)
    return final_text
