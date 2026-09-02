"""モデル能力プローブ + CapabilitySnapshot (設計: docs/c_15_model_capability_adaptation.md)

モデル (GGUF) 接続/切替時に、新モデルの**実挙動**を OAI 互換カナリアで観測し、
処理を自動適応するための capability snapshot を生成する。

静的宣言 (``models/profiles/<arch>.yaml`` の ``reasoning.mode``) は prior (事前推定)、
プローブ実測は posterior (事後観測) とし、**観測が宣言を override** する。プローブ失敗時は
prior にフォールバックして degraded を起こさない (観測フィールドは ``None`` のまま)。

OAI 互換エンドポイント (``/v1/chat/completions``) のみを使う (e_03)。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from backend.free.llm._base_client import (
    GENERATION_RETRYABLE_EXCEPTIONS,
    async_retry_http_call,
    make_retry_logger,
)
from backend.free.llm.model_metadata import TemplateFamily
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("llm.capability")

# プローブのカナリア (固定文字列・temperature=0・最小 max_tokens・履歴非保存)
_JSON_PROBE_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "capability_probe_score",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        },
    },
}
_JSON_PROBE_PROMPT = (
    'Return only a JSON object with a single numeric field "score" equal to 0.7. '
    "No prose, no code fences."
)
_REASONING_PROBE_PROMPT = "What is 2+2? Answer in one short sentence."

# raw OAI ``/v1/chat/completions`` 応答 JSON を返す呼び出し関数の型。
# 実機用は make_llama_chat_fn() が生成。テストはフェイクを注入する。
ChatFn = Callable[[dict], Awaitable[dict]]


@dataclass(frozen=True)
class CapabilitySnapshot:
    """モデルの宣言 (prior) と実機観測 (posterior) を統合した能力スナップショット。

    観測未実施フィールドは ``None`` (prior フォールバックを許容)。client 属性として
    保持され、モデル切替 (クライアント再生成) で作り直される。
    """

    model_id: str
    template_family: TemplateFamily
    # 宣言 (prior)
    declared_reasoning_mode: str | None
    # 観測 (posterior; None = 未プローブ / プローブ失敗)
    reasoning_separated: bool | None = None       # P1: reasoning_content に分離されるか
    emits_think_tags: bool | None = None          # P2a: content に <think> を吐くか
    closes_think_tags: bool | None = None         # P2b: </think> で閉じるか
    json_schema_enforced: bool | None = None      # P3: json_schema grammar が強制されるか
    reasoning_budget_effective: bool | None = None  # P4 (Phase 1.5)
    # 解決済み (観測優先で確定)
    effective_reasoning_mode: str | None = None
    needs_lenient_json: bool = False
    probed_at: str = ""
    probe_divergence: list[str] = field(default_factory=list)

    @property
    def probed(self) -> bool:
        """実機プローブが1つでも成立したか (False=全て prior フォールバック)。"""
        return any(
            v is not None
            for v in (
                self.reasoning_separated,
                self.emits_think_tags,
                self.json_schema_enforced,
            )
        )


# ─────────────────────────────────────────────────────────────────────────
# 純粋な解釈ロジック (サーバ非依存・単体テスト対象)
# ─────────────────────────────────────────────────────────────────────────


def interpret_json_probe(content: str) -> bool:
    """P3: json_schema grammar が強制されているかを raw content から判定する。

    grammar 強制下では必ず JSON オブジェクト (``{"score": ...}``) になる。
    裸スカラ (``"0.7"``) / 非オブジェクト / パース不能なら非強制とみなす。
    """
    text = (content or "").strip()
    if not text.startswith("{"):
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict)


def interpret_reasoning_probe(
    message: dict,
) -> tuple[bool, bool, bool | None]:
    """P1/P2: raw message から (reasoning_separated, emits_think, closes_think) を判定。

    - reasoning_separated: ``message.reasoning_content`` が非空 (サーバが分離済)。
    - emits_think: ``message.content`` に ``<think>`` が出現。
    - closes_think: emits_think 時のみ ``</think>`` で閉じるか (False=未閉じ=暴走危険)。
      emits_think=False のときは ``None``。
    """
    reasoning_content = (message or {}).get("reasoning_content") or ""
    content = (message or {}).get("content") or ""
    reasoning_separated = bool(reasoning_content.strip())
    emits_think = "<think>" in content
    closes_think: bool | None = None
    if emits_think:
        closes_think = "</think>" in content
    return reasoning_separated, emits_think, closes_think


def resolve_effective_reasoning_mode(
    declared: str | None,
    *,
    reasoning_separated: bool | None,
    emits_think: bool | None,
) -> tuple[str | None, list[str]]:
    """宣言 (prior) と観測 (posterior) から effective reasoning mode と乖離を解決する。

    観測が「実際に思考する」(分離 or <think> 出力) ことを示すのに宣言が ``none`` の場合、
    宣言が誤り (実機は思考する) と判断し effective を ``always`` に補正、乖離を記録する。
    観測が未取得 (None) の場合は宣言をそのまま採用 (prior フォールバック)。
    """
    divergence: list[str] = []
    reasons_now = bool(reasoning_separated) or bool(emits_think)

    # 観測未取得 → prior をそのまま。
    if reasoning_separated is None and emits_think is None:
        return declared, divergence

    if reasons_now and declared == "none":
        divergence.append(
            f"reasoning.mode declared=none but model reasons "
            f"(separated={reasoning_separated}, emits_think={emits_think})",
        )
        return "always", divergence

    if not reasons_now and declared in ("always", "toggle"):
        # 宣言は思考可能だが今回は思考しなかった (toggle で OFF など)。乖離としては弱いので
        # 記録のみ・mode は宣言維持 (toggle の OFF 状態は正常系)。
        if declared == "always":
            divergence.append(
                "reasoning.mode declared=always but probe observed no thinking",
            )

    return declared, divergence


# ─────────────────────────────────────────────────────────────────────────
# プローブ実行 (chat_fn 注入で単体テスト可能)
# ─────────────────────────────────────────────────────────────────────────


def _message_of(resp: dict) -> dict:
    """OAI 応答 JSON から ``choices[0].message`` を安全に取り出す。"""
    try:
        return resp["choices"][0].get("message", {}) or {}
    except (KeyError, IndexError, TypeError):
        return {}


async def probe_model_capabilities(
    *,
    model_id: str,
    template_family: TemplateFamily,
    declared_reasoning_mode: str | None,
    chat_fn: ChatFn,
    probe_json: bool = True,
) -> CapabilitySnapshot:
    """カナリアを投げて実挙動を観測し ``CapabilitySnapshot`` を返す。

    ``chat_fn`` は raw OAI ``/v1/chat/completions`` 応答 JSON を返す async 関数。
    ``probe_json=False`` で P3 (json_schema 強制) をスキップする。アシスト撤去以降は
    補助タスクの json purpose もベースモデルが担うため、ベースでも既定で観測する。
    個々のプローブ失敗は握りつぶし観測を ``None`` のままにする (prior フォールバック)。
    全失敗でも例外は投げない (degraded 安全)。
    """
    reasoning_separated: bool | None = None
    emits_think: bool | None = None
    closes_think: bool | None = None
    json_enforced: bool | None = None

    # P1/P2: reasoning 分離・<think> 出力・閉じ
    try:
        resp = await chat_fn(
            {
                "messages": [{"role": "user", "content": _REASONING_PROBE_PROMPT}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": PROBE_REASONING_MAX_TOKENS,
            },
        )
        reasoning_separated, emits_think, closes_think = interpret_reasoning_probe(
            _message_of(resp),
        )
    except Exception as exc:
        logger.warning("capability probe P1/P2 failed (prior fallback): %s", exc)

    # P3: json_schema grammar 強制 (補助タスクの json purpose 用)
    if probe_json:
        try:
            resp = await chat_fn(
                {
                    "messages": [{"role": "user", "content": _JSON_PROBE_PROMPT}],
                    "stream": False,
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "response_format": _JSON_PROBE_SCHEMA,
                },
            )
            content = _message_of(resp).get("content") or ""
            json_enforced = interpret_json_probe(content)
        except Exception as exc:
            logger.warning("capability probe P3 failed (prior fallback): %s", exc)

    effective_mode, divergence = resolve_effective_reasoning_mode(
        declared_reasoning_mode,
        reasoning_separated=reasoning_separated,
        emits_think=emits_think,
    )
    if json_enforced is False:
        divergence.append("json_schema grammar not enforced (lenient parsing enabled)")
    if emits_think and closes_think is False:
        divergence.append("<think> emitted without </think> close (runaway risk)")

    snapshot = CapabilitySnapshot(
        model_id=model_id,
        template_family=template_family,
        declared_reasoning_mode=declared_reasoning_mode,
        reasoning_separated=reasoning_separated,
        emits_think_tags=emits_think,
        closes_think_tags=closes_think,
        json_schema_enforced=json_enforced,
        effective_reasoning_mode=effective_mode,
        needs_lenient_json=(json_enforced is False),
        probed_at=utc_now(),
        probe_divergence=divergence,
    )
    if divergence:
        logger.warning(
            "Capability probe divergence for model=%s: %s",
            model_id, "; ".join(divergence),
        )
    else:
        logger.info(
            "Capability probe ok for model=%s (separated=%s, emits_think=%s, "
            "json_enforced=%s, effective_mode=%s)",
            model_id, reasoning_separated, emits_think, json_enforced, effective_mode,
        )
    return snapshot


#: プローブ 1 本のタイムアウトに含める固定分 (秒)。ロード直後の冷えた KV での
#: prefill とスロット確保の揺れを吸収する。
PROBE_BASE_SEC = 30.0
#: decode 速度の見積り: ``PROBE_DECODE_TPS_NUMERATOR / params_b`` tok/s
#: (4B ≒ 15 tok/s、27B ≒ 2.2 tok/s、iGPU での実測に合わせた粗い比例則)。
#: 下限 ``PROBE_MIN_DECODE_TPS`` で 70B 級でも有限に収める。
PROBE_DECODE_TPS_NUMERATOR = 60.0
PROBE_MIN_DECODE_TPS = 1.5
#: P1/P2 (reasoning) カナリアの max_tokens。
PROBE_REASONING_MAX_TOKENS = 128


def probe_timeout_sec(params_b: float, max_tokens: int) -> float:
    """プローブ 1 本のタイムアウト秒をモデルサイズに追随させる。

    ``PROBE_BASE_SEC + max_tokens / decode_tps(params_b)``。固定 120 秒 / 90 秒
    だと、小型モデルでは長すぎて起動直後の失敗検知が遅れ、大型モデル (27B+) の
    iGPU では 128 トークンの decode だけで足りない (観測が恒久的に ``None`` に
    なり、``enable_thinking`` の自動再解決が宣言値のまま固定される)。
    """
    tps = max(PROBE_MIN_DECODE_TPS, PROBE_DECODE_TPS_NUMERATOR / max(1.0, float(params_b or 1.0)))
    return PROBE_BASE_SEC + float(max_tokens) / tps


def make_llama_chat_fn(
    llama_url: str,
    *,
    debug_logger=None,
    timeout: float = 120.0,
    id_slot: int | None = None,
) -> ChatFn:
    """実機 llama-server ``/v1/chat/completions`` を叩く ``chat_fn`` を生成する。

    ``model_metadata.fetch_model_metadata`` と同じく httpx + ``async_retry_http_call``
    の統一ポリシーに乗せる。OAI 互換エンドポイントのみ (e_03)。

    ``timeout`` はカナリア 1 本あたりの上限。P1/P2 は 128 token 生成するため、
    ロード直後の大型 thinking モデルでは decode だけで容易に 20 秒を超える
    (アシスト撤去前は小型モデルが対象だったので 20 秒で足りていた)。プローブは
    起動をブロックしない背景タスクなので、観測を取り切る方を優先する。短すぎると
    reasoning 系の観測が恒久的に ``None`` のままになり、``enable_thinking`` の
    自動再解決が宣言値のまま固定される。

    ``id_slot`` を渡すとそのスロットへ固定する。プローブは補助タスクと同じく
    背景処理なので専有スロットへ寄せ、起動直後のユーザー初回応答と KV を分離する。
    """
    retry_logger = make_retry_logger(debug_logger, backend="base", purpose="startup/probe")

    async def _chat(payload: dict) -> dict:
        if id_slot is not None:
            payload = {**payload, "id_slot": id_slot}

        async def _post() -> dict:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{llama_url}/v1/chat/completions", json=payload, timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()

        return await async_retry_http_call(
            _post,
            request_label="llama-server /v1/chat/completions (probe)",
            retry_logger=retry_logger,
            retryable_exceptions=GENERATION_RETRYABLE_EXCEPTIONS,
        )

    return _chat


__all__ = [
    "CapabilitySnapshot",
    "ChatFn",
    "interpret_json_probe",
    "interpret_reasoning_probe",
    "resolve_effective_reasoning_mode",
    "probe_model_capabilities",
    "probe_timeout_sec",
    "make_llama_chat_fn",
]
