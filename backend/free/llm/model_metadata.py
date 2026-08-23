"""モデルメタデータ取得（llama-server GET /props）"""

import re
from dataclasses import dataclass
from typing import Literal

import httpx

from backend.free.llm._base_client import async_retry_http_call, make_retry_logger
from backend.log_config import get_logger

logger = get_logger("llm.model_metadata")

# パラメータ数推定の基準値（B単位）
DEFAULT_PARAMS_B = 7.0

# テンプレート系統
TemplateFamily = Literal[
    "harmony",          # gpt-oss / llm-jp-4-thinking: <|channel|>analysis|commentary|final
    "qwen3_thinking",   # Qwen3: <think>...</think> + enable_thinking
    "deepseek_r1",      # DeepSeek-R1: <think>...</think>
    "gemma",            # Gemma 2/3: <start_of_turn> + system ロール非対応
    "llama3",           # Llama 3 系: <|start_header_id|>
    "chatml",           # ChatML: <|im_start|>
    "unknown",
]


@dataclass
class ModelMetadata:
    """llama-server から取得したモデル情報"""
    chat_template: str
    has_system_role: bool = True
    bos_token: str = ""
    eos_token: str = ""
    model_id: str = ""
    template_family: TemplateFamily = "unknown"
    #: llama-server が実際にロードしているコンテキスト長とスロット数。
    #: config の宣言値と食い違いうる (config を書き替えても llama-server を
    #: 再起動しなければ反映されない) ため、**自己構成を答えるときは実測側を使う**。
    #: 取得できなければ 0 (未知)。
    n_ctx: int = 0
    total_slots: int = 0

    @property
    def params_b(self) -> float:
        """モデルのパラメータ数を B（十億）単位で推定"""
        return estimate_params_b(self.model_id)


def detect_template_family(chat_template: str) -> TemplateFamily:
    """chat_template 文字列からテンプレート系統を推定する

    llama-server の /props から取得した Jinja 文字列に含まれる制御トークン
    から系統を判定する。完全識別ではなく、ログ出力と diagnostic を目的とする。
    `_ReasoningFilter` は系統によらず content ストリームを処理するため、
    この判定結果は runtime のフィルタ選択には影響しない。
    """
    if not chat_template:
        return "unknown"
    tmpl = chat_template
    if "<|channel|>" in tmpl or ("<|start|>" in tmpl and "<|message|>" in tmpl):
        return "harmony"
    if "<start_of_turn>" in tmpl:
        return "gemma"
    if "enable_thinking" in tmpl or ("<think>" in tmpl and "qwen" in tmpl.lower()):
        return "qwen3_thinking"
    if "<think>" in tmpl:
        return "deepseek_r1"
    if "<|start_header_id|>" in tmpl:
        return "llama3"
    if "<|im_start|>" in tmpl:
        return "chatml"
    return "unknown"


def estimate_params_b(model_id: str) -> float:
    """モデル名からパラメータ数（B単位）を推定

    よくあるパターン:
    - "Qwen3.5-9B-Q4_K_M.gguf" → 9.0
    - "gemma-2-2b-it" → 2.0
    - "Llama-3.1-8B-Instruct" → 8.0
    - "Phi-3-mini-4k-instruct" → 3.8 (推定不能→デフォルト)
    - "mistral-7b-instruct-v0.3" → 7.0

    Returns:
        パラメータ数（B単位）。推定不能時は DEFAULT_PARAMS_B
    """
    if not model_id:
        logger.debug("estimate_params_b: empty model_id, using default %.1fB", DEFAULT_PARAMS_B)
        return DEFAULT_PARAMS_B

    # パターン: "1.7B", "7b", "14B", "0.5B", "70b" など
    # ファイルパスの場合はファイル名部分のみ使用
    name = model_id.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?:\b|[-_.])", name)
    if match:
        result = float(match.group(1))
        logger.debug("estimate_params_b: model=%s -> %.1fB", name, result)
        return result

    logger.debug("estimate_params_b: no match for '%s', using default %.1fB", name, DEFAULT_PARAMS_B)
    return DEFAULT_PARAMS_B


async def fetch_model_metadata(
    llama_url: str,
    *,
    debug_logger=None,
    purpose: str = "startup/props",
) -> ModelMetadata:
    """llama-server GET /props からチャットテンプレートを取得。

    llama-server がモデルロード中 (特に大きい GGUF) は短時間
    ``503 Service Unavailable`` を返すことがあるため、接続エラー・5xx・
    読み取りタイムアウトについては指数バックオフで再試行する。
    4xx などの永続エラーは即座に再送出する。

    旧来の独自リトライループ (``DEFAULT_PROPS_MAX_RETRIES`` /
    ``DEFAULT_PROPS_INITIAL_DELAY`` / ``DEFAULT_PROPS_MAX_DELAY``) を撤去し、
    ``backend/free/llm/_base_client.py::async_retry_http_call`` の tenacity
    統一ポリシー (``stop_after_attempt(3)`` ×
    ``wait_exponential_jitter(0.5, 4.0)``) に集約。``debug_logger`` を渡すと
    リトライ発火が ``requests.jsonl`` に ``op="retry"`` (``backend="base"``)
    として記録される。

    Args:
        purpose: リトライログに載せる呼出元ラベル。既定は起動時プローブ。
            **起動以外の経路は必ず自分のラベルを渡すこと** — 既定のまま使うと、
            稼働中に発生したリトライが「起動時のプローブ」に見える
            (2026-08-05 ライブ監査: 会話の最中に ``startup/props`` の
            ``RemoteProtocolError`` リトライが 3 回記録され、実体は
            ``/api/status`` ポーリング由来の遅延再接続だった)。
    """
    logger.debug("Fetching model metadata from %s/props", llama_url)
    retry_logger = make_retry_logger(debug_logger, backend="base", purpose=purpose)

    async def _fetch_props() -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{llama_url}/props", timeout=10.0)
            resp.raise_for_status()
            return resp.json()

    props = await async_retry_http_call(
        _fetch_props,
        request_label="llama-server /props",
        retry_logger=retry_logger,
    )

    tmpl = props.get("chat_template", "")
    # systemロール対応判定: テンプレートに "system" が含まれるか
    has_sys = "'system'" in tmpl or '"system"' in tmpl
    family = detect_template_family(tmpl)

    # モデルID: model_alias → model_path → default_generation_settings.model
    # llama-server の /props は model_alias にモデル名を返す
    model_id = props.get("model_alias", "")
    if not model_id:
        model_id = props.get("model_path", "")
    if not model_id:
        gen_settings = props.get("default_generation_settings", {})
        if isinstance(gen_settings, dict):
            model_id = gen_settings.get("model", "")

    logger.info(
        "Model metadata fetched: model_id=%s, template_family=%s, "
        "has_system_role=%s, template_length=%d, bos=%r, eos=%r",
        model_id, family, has_sys, len(tmpl),
        props.get("bos_token", ""), props.get("eos_token", ""),
    )

    gen = props.get("default_generation_settings") or {}
    n_ctx = gen.get("n_ctx") if isinstance(gen, dict) else None
    return ModelMetadata(
        chat_template=tmpl,
        has_system_role=has_sys,
        bos_token=props.get("bos_token", ""),
        eos_token=props.get("eos_token", ""),
        model_id=model_id,
        template_family=family,
        n_ctx=int(n_ctx) if isinstance(n_ctx, (int, float)) else 0,
        total_slots=int(props.get("total_slots") or 0),
    )
