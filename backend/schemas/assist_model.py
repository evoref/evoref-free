"""アシストモデル関連スキーマ"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.schemas.llm import MtpConfig


class AssistModelLocalConfig(BaseModel):
    """アシストモデルローカル接続設定"""

    model_config = ConfigDict(extra="allow")

    host: str = "127.0.0.1"
    port: int = Field(default=8081, ge=1024, le=65535)
    # ``-c`` / アシストモデルのコンテキスト長。``None`` (既定) は arch プロファイル
    # (``models/profiles/<arch>.yaml`` の ``context_size``) を参照し、それも
    # 未宣言なら 8192 にフォールバックする。明示 int でプロファイルを上書き
    # (解決順: config 明示 > profile > 8192)。ベース ``llama.context_size`` とは独立。
    context_size: int | None = Field(default=None, ge=512)
    # llama-server チューニング項目。いずれも省略時は build_assist_cmd で
    # フラグを付与せず、llama-server のデフォルト挙動に任せる。
    # 整数 (-1 / 0 / 999 等) / None (llama 継承) / ``"auto"`` 文字列。
    # ``"auto"`` は scripts/launch_llama.py で base と同じ ratio で
    # 段階縮小される (詳細は LlamaConfig.gpu_layers コメント参照)。
    gpu_layers: int | Literal["auto"] | None = Field(default=None)

    @field_validator("gpu_layers", mode="before")
    @classmethod
    def _validate_gpu_layers(cls, v: object) -> int | str | None:
        """None / 整数 (>=-1) / ``"auto"`` のみ許容。"""
        if v is None:
            return None
        if isinstance(v, str):
            if v == "auto":
                return v
            raise ValueError(
                f"gpu_layers must be an int (>=-1), None, or 'auto', got string {v!r}"
            )
        iv = int(v)
        if iv < -1:
            raise ValueError(f"gpu_layers must be >= -1, got {iv}")
        return iv
    threads: int = Field(default=0, ge=0)
    batch_size: int | None = Field(default=None, ge=1)
    ubatch_size: int | None = Field(default=None, ge=1)
    flash_attn: bool | None = None
    mlock: bool = False
    slots: int = Field(default=1, ge=1, le=16)
    cache_type_k: str | None = Field(
        default=None, pattern=r"^(f16|bf16|q8_0|q5_1|q5_0|q4_1|q4_0)$"
    )
    cache_type_v: str | None = Field(
        default=None, pattern=r"^(f16|bf16|q8_0|q5_1|q5_0|q4_1|q4_0)$"
    )
    cache_reuse: int = Field(default=0, ge=0)
    # idle slot offload (host RAM 退避)。cache_ram_mib: -1=無制限 / 0=disable /
    # >0=MiB 上限。kv-unified とは独立 (RAM 退避の要否のみを制御)。
    cache_ram_mib: int = Field(default=2048, ge=-1)
    cache_idle_slots: bool = True
    # null=auto (slots>1 で ``--kv-unified`` 自動付与。per-seq context が
    # n_ctx/slots に半減するのを防ぐ) / true|false で明示上書き。
    kv_unified: bool | None = None
    # 起動時に ``-rea`` / ``--reasoning-budget`` / ``--reasoning-budget-message``
    # を付与し、リクエストごとに ``chat_template_kwargs.enable_thinking=false``
    # を送信する従来挙動と二重防御で thinking を抑制する。
    #
    # ``reasoning``: ``"on"`` / ``"off"`` / ``"auto"`` のいずれか。``None``
    # (既定) は ``-rea`` フラグを付与しない (上流 default 挙動)。
    # ``reasoning_budget_default``: ``-1`` 無制限 (-rea 制御のみ) / ``0``
    # 即終了 / ``N>0`` で N トークンの token cap。``-1`` (既定) は
    # ``--reasoning-budget`` を付与しない。
    # ``reasoning_budget_message``: budget 超過時に thinking 終了タグの直前へ
    # 挿入される文字列。空文字列 (既定) では ``--reasoning-budget-message``
    # を付与しない。
    reasoning: Literal["on", "off", "auto"] | None = None
    reasoning_budget_default: int = Field(default=-1, ge=-1)
    reasoning_budget_message: str = ""
    # MTP (Multi-Token Prediction) self-speculative decoding。MTP ヘッド内蔵
    # モデルでのみ有効 (非対応モデルは launch 側で warning + 素通り)。``None``
    # (既定) は MTP フラグを付与しない。詳細は backend.schemas.llm.MtpConfig。
    mtp: MtpConfig | None = None
    # true で起動するとサーバ側で reasoning / tool calls の構造解析を完全に
    # スキップし、すべて ``message.content`` に出力される。``<think>...</think>``
    # の post-strip は Python 側 ``_ReasoningFilter`` が肩代わ
    # りする。debug ルート / 学習サイクル用途のみ true にすること
    # (jinja template / tool calling 経路を失うため通常運用は false 推奨)。
    skip_chat_parsing: bool = False
    extra_args: list[str] = Field(default_factory=list)
    # アシストモデルのパラメータ数 (B 単位) を明示指定する。``null`` (既定) で
    # ``estimate_params_b`` (model 名の正規表現) → ``/props`` の順に推定する。
    # モデル名から ``\d+B`` が抽出できない GGUF (例: ``custom-1.7b-q4.gguf``
    # の小文字 b) を使う場合や、量子化版で実効性能が異なる場合に明示指定
    # する。``note_evolver`` / ``conflict_resolver`` の sleep-time インターバル
    # 計算 (``compute_llm_call_interval``) で 7B 基準のスケーリングに使われる。
    params_b: float | None = Field(default=None, gt=0.0, le=1000.0)


class AssistConcurrencyConfig(BaseModel):
    """アシストモデル用途別セマフォスロット数

    チャット応答パス (realtime) / Sleep-time 系 (background) / 学習サイクル
    (learning) の 3 カテゴリに独立したセマフォを割り当て、用途間の
    キュー競合を排除する。purpose → priority の割り付けは
    ``AssistModelClient._PURPOSE_PRIORITY_MAP`` を参照。

    Level 2 CritiqueSynthesizer が並行しても、チャット応答パスで発火する
    ``retrieval_quality_judge`` / ``tool_judgment`` が先行処理されるよう
    slot を独立化する。合計スロット数は従来と同じ 3 を既定とする。
    """

    model_config = ConfigDict(extra="forbid")

    realtime: int = Field(default=1, ge=1)
    background: int = Field(default=1, ge=1)
    learning: int = Field(default=1, ge=1)


class AssistModelConfig(BaseModel):
    """アシストモデル設定"""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    local: AssistModelLocalConfig | None = None
    model_path: str = ""
    timeout: float = Field(default=30.0, ge=0.1)
    # 用途別セマフォスロット数
    # 従来の ``max_concurrent`` を廃し realtime / background / learning に分離。
    concurrency: AssistConcurrencyConfig = Field(
        default_factory=AssistConcurrencyConfig,
    )
    # purpose 別タイムアウト。長文生成プラン/レビュー
    # critique 等の重いタスクは既定 timeout では枯渇するため purpose 文字列
    # をキーに override する。例: {"long_form_planning": 90.0}
    timeouts: dict[str, float] = Field(default_factory=dict)
    # (b8870+) で導入された ``--reasoning-budget`` の per-request override。
    # ``chat_template_kwargs.reasoning_budget`` として送信される。
    # ``-1`` 無制限 / ``0`` 即終了 / ``N>0`` token 上限。
    # 例: {"long_form_planning": 2048, "conflict_resolution": 0}
    reasoning_budgets: dict[str, int] = Field(default_factory=dict)
    # JSON 応答 purpose に GBNF/json_schema 制約サンプリングを適用するか
    #。``true`` で OAI 互換 ``response_format`` を
    # ``/v1/chat/completions`` ペイロードに付与し、Pydantic schema から
    # 自動生成した json_schema で文法外トークンを抑制する。古い llama-server
    # build (``response_format: json_schema`` 未対応) や、パースエラー切り
    # 分け用に ``false`` を指定すると json_extract.py のフォールバックのみ
    # で動作する。詳細は ``backend/free/llm/json_schemas.py``。
    response_format_enabled: bool = True
    # 全 assist リクエストに per-request で注入する chat_template_kwargs。
    # purpose 別の reasoning_budget (`reasoning_budgets`) は本マップに
    # マージされる。空 dict ``{}`` で per-request 注入を完全停止する
    # (非 thinking モデル運用時に指定)。
    # Qwen3 / Gemma-4 / DeepSeek-R1 等の thinking モデル既定:
    # ``{"enable_thinking": False}`` (jinja template が受理した場合に
    # thinking を抑制するための保険として機能)。
    # 非 thinking モデル (Llama 3 / Mistral 等) では空 dict を指定する。
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"enable_thinking": False},
    )
