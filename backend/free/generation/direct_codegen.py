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

import logging

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
        ``{file_path: code}``。生成失敗 / 空応答時は空 dict。

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

    code = remove_code_fences(extract_content(resp).strip())

    if _finish_reason(resp) == "length":
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
                retry_code = remove_code_fences(extract_content(resp2).strip())
                if retry_code and len(retry_code) > len(code):
                    code = retry_code
            except Exception as exc:
                logger.warning(
                    "direct codegen regeneration failed for %s: %s", file_path, exc,
                )

    if not code.strip():
        return {}
    return {file_path: code}
