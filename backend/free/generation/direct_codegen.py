"""staged コーディングパイプライン専用の単発ファイル生成。

``LongFormOrchestrator`` (CogWriterStrategy) はモジュール内部で instruction から
独自に plan JSON / CodeSpec を **再合成**し、CodeUnit (関数/クラス粒度) に分割して
個別生成・連結する。この再合成はアシスト LLM を介した lossy な圧縮であり、
staged executor が組み立てた instruction (spec.md 全文 + flowchart + 契約ブロック)
の大半は base モデルのユニット生成プロンプトに一切届かない
(``build_code_unit_messages`` が見るのは再合成後の CodeSpec render / plan 由来の
unit.spec のみ)。さらに CodeUnit 分割・再連結は import 重複や機能重複などの結合
不整合を生む。

staged は ``synthesize_coding_task_graph`` が既にプログラムをファイル単位へ決定的に
分解済みであり、1 code タスク = 1 ファイルの単発生成で足りる。本モジュールは
その再計画・再合成・分割連結を経由せず、base モデルへの単発呼び出しのみで完結する
軽量パスを提供する。instruction は無劣化のままプロンプトへ渡る。
"""

from __future__ import annotations

import ast
import logging
import re

from backend.free.generation.validators import remove_code_fences
from backend.free.llm.utils import extract_content

logger = logging.getLogger("backend.free.generation.direct_codegen")

_SYSTEM_PROMPT = (
    "You are an expert programmer. Follow the design specification and "
    "instructions in the user message exactly and completely. Output ONLY the "
    "source code for the requested file."
)

# 切断時の再生成で許す max_tokens 上限。
_MAX_TOKENS_CEILING = 16384

# 非ストリーミング呼び出しの per-request タイムアウト算出パラメータ。
# LocalClient の既定タイムアウト (120s) は decode 速度の速い環境向けで、iGPU 等
# 低速環境では max_tokens=4096 の同期生成 1 回すら終わらず必ず timeout する
# (実測: iGPU decode ~7-13 tok/s で 4096 トークンに 300-580 秒必要)。
# 保守的な下限 tok/s で必要時間を見積もり、prefill 等のオーバーヘッド分の
# マージンを足す。無制限に伸びないよう上限でキャップする。
_MIN_TOKENS_PER_SEC = 5.0
_TIMEOUT_MARGIN_SEC = 60.0
_TIMEOUT_CEILING_SEC = 1800.0


def _estimate_timeout(max_tokens: int) -> float:
    """``max_tokens`` から非ストリーミング呼び出しの per-request タイムアウトを見積もる。

    ``LocalClient`` の既定 (120s) を下回らないようにしつつ、低速環境でも
    生成が完了しうる時間を確保する。
    """
    estimated = max_tokens / _MIN_TOKENS_PER_SEC + _TIMEOUT_MARGIN_SEC
    return min(_TIMEOUT_CEILING_SEC, max(120.0, estimated))


def _finish_reason(resp: dict) -> str:
    """base client generate() 応答の finish_reason ('length' は max_tokens 切断)。"""
    try:
        fr = resp["choices"][0].get("finish_reason")
        return fr if isinstance(fr, str) else ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _reasoning_content(resp: dict) -> str:
    """base client generate() 応答の reasoning_content ('<think>' 分離出力)。"""
    try:
        rc = resp["choices"][0]["message"].get("reasoning_content")
        return rc if isinstance(rc, str) else ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


_CODE_FENCE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _parses_to_nonempty_module(code: str) -> bool:
    try:
        return bool(ast.parse(code).body)
    except (SyntaxError, ValueError):
        return False


def _salvage_code_from_reasoning(reasoning_content: str) -> str:
    """打ち切られた reasoning から、最後に書かれた完成コードブロックを救出する。

    coder モデル (Qwopus3.5 等) は推論モデルであり、``<think>`` 内で一度コード
    を書き下してから自己ツッコミ・再検討を続けることがある (2026-07-23 live:
    gcd_calculator/fibonacci_generator の生成で確認 — max_tokens が reasoning
    だけで尽き、可視の ``content`` が空のまま finish_reason='length' で切断
    された。reasoning_content 側には既に完成した正しい関数が書かれていた)。
    reasoning 内の fenced code block を末尾から走査し、構文的に完成している
    (``ast.parse`` が通り body が空でない) 最初の候補を採用する (best-effort、
    無ければ空文字)。
    """
    for block in reversed(_CODE_FENCE_BLOCK_RE.findall(reasoning_content)):
        candidate = block.strip()
        if candidate and _parses_to_nonempty_module(candidate):
            return candidate
    return ""


def _extract_code(resp: dict, file_path: str) -> tuple[str, bool]:
    """応答からコードを取り出す。戻り値は ``(code, from_salvage)``。

    可視 ``content`` が空なら reasoning から救出する。``from_salvage=True`` は
    ``ast.parse`` で完成を検証済みという意味で、``finish_reason == "length"``
    でも不完全な断片ではないと信頼してよい (直接 content から取った切断済み
    テキストとは区別する — 後者は語尾が切れた不完全コードの可能性がある)。
    """
    code = remove_code_fences(extract_content(resp).strip())
    if code:
        return code, False
    salvaged = _salvage_code_from_reasoning(_reasoning_content(resp))
    if salvaged:
        logger.info(
            "direct codegen recovered code from truncated reasoning_content "
            "for %s (visible content was empty)", file_path,
        )
    return salvaged, bool(salvaged)


async def generate_single_file(
    client,
    instruction: str,
    file_path: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> dict[str, str]:
    """instruction から単一ファイルのコードを base モデルへの 1 回の呼び出しで生成する。

    Args:
        client: base LLM client (``LocalClient`` 互換、``generate()``/``chat_slot``
            を持つ)。
        instruction: 呼出側 (staged executor) が組み立てた完全な生成指示
            (spec.md 全文 + flowchart + 契約ブロックを含む)。加工・再合成せず
            そのままユーザーメッセージへ渡す。
        file_path: 生成対象のファイル論理パス。戻り値の辞書キーに使う
            (呼出側は常にこのキーで結果を取得できる)。
        max_tokens: 初回生成の最大トークン。
        temperature: 生成温度。

    Returns:
        ``{file_path: code}``。生成失敗 / 空応答時は空 dict。再生成 (retry) 後も
        ``finish_reason == "length"`` (再切断) の場合も空 dict を返す (不完全
        コードは成果物として返さない)。

    応答が ``finish_reason == "length"`` (max_tokens 切断) の場合のみ、予算を倍に
    広げて 1 回だけ再生成する (``staged.executor._generate_spec_doc`` と同じ方針)。
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    try:
        resp = await client.generate(
            messages, stream=False, max_tokens=max_tokens, temperature=temperature,
            id_slot=client.chat_slot, request_timeout=_estimate_timeout(max_tokens),
        )
    except Exception as exc:
        logger.warning("direct codegen failed for %s: %s", file_path, exc)
        return {}

    code, code_from_salvage = _extract_code(resp, file_path)

    # reasoning から検証済みで救出できた場合は、切断されていても完成コードと
    # して信頼できるため、費用のかかる倍額再生成をスキップする (2026-07-23
    # live: gcd_calculator/fibonacci_generator が reasoning だけで max_tokens
    # を使い切り、直接 content は空だが reasoning 内には既に完成した正しい
    # 関数が書かれていた。再生成を重ねても同じパターンで再度失敗し、
    # 最終的に「コードが生成されませんでした」まで至っていた)。
    if _finish_reason(resp) == "length" and not code_from_salvage:
        retry_tokens = min(max_tokens * 2, _MAX_TOKENS_CEILING)
        if retry_tokens > max_tokens:
            logger.warning(
                "direct codegen truncated at max_tokens=%d for %s; "
                "regenerating at %d", max_tokens, file_path, retry_tokens,
            )
            try:
                resp2 = await client.generate(
                    messages, stream=False, max_tokens=retry_tokens,
                    temperature=temperature, id_slot=client.chat_slot,
                    request_timeout=_estimate_timeout(retry_tokens),
                )
                retry_code, retry_from_salvage = _extract_code(resp2, file_path)
                if _finish_reason(resp2) == "length" and not retry_from_salvage:
                    # 再生成も切断され、reasoning からの検証済み救出も得られな
                    # かった = 完成コードではなく壊れた断片 (直接 content が
                    # 非空でも未検証の切断済みテキストに過ぎない)。長さで
                    # 「マシ」に見えても SyntaxError を伴う不完全ファイルを完了
                    # 扱いで返さない (2026-07-22: test_email_validator.py が
                    # f-string 途中で切れたまま「成果物は配信します」扱いで出荷
                    # された実害の再発防止)。空扱いにして呼出側 (staged executor
                    # の empty-code フォールバック) に委ねる。
                    logger.warning(
                        "direct codegen still truncated after retry at "
                        "max_tokens=%d for %s; discarding incomplete code",
                        retry_tokens, file_path,
                    )
                    return {}
                if retry_code and len(retry_code) > len(code):
                    code = retry_code
            except Exception as exc:
                logger.warning(
                    "direct codegen regeneration failed for %s: %s", file_path, exc,
                )

    if not code.strip():
        return {}
    return {file_path: code}
