"""llama.cpp / 推論バックエンド関連スキーマ"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MtpConfig(BaseModel):
    """MTP (Multi-Token Prediction) self-speculative decoding 設定。

    モデル自身に内蔵された MTP ヘッド (NextN 層) を draft に使う自己投機方式。
    外部 draft モデル不要のため追加 VRAM もトークナイザ整合の懸念もなく、Free /
    Pro 共通で利用できる (Pro 限定の :class:`LlamaSpeculativeConfig` とは別系統)。

    **MTP ヘッドを内蔵した GGUF でのみ有効** (Qwen3.5/3.6 等。GGUF メタデータ
    ``<arch>.nextn_predict_layers > 0`` で判定)。非対応モデルに対しては
    ``scripts/launch_llama.py`` 側で warning + フラグ未付与で素通りさせる
    (graceful degrade)。

    起動フラグ (llama-server): ``--spec-type draft-mtp`` +
    ``--spec-draft-n-max <draft_n_max>``。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # 1 サイクルで先読みする draft トークン数 (上流 ``--spec-draft-n-max``)。
    # 上流既定 3。
    draft_n_max: int = Field(default=3, ge=1)


class LlamaSpeculativeConfig(BaseModel):
    """self-speculative decoding 設定

    Pro 限定機能。``evoref create`` モード の base
    モデルでコーディング系ワークロードのスループット改善を狙う。Free では
    ``enabled=true`` でも ``scripts/launch_llama.py`` 側で warning + 無効化
    される (``EVOREF_EDITION`` / ``backend.pro`` 存在で判定)。

    ``--spec-default`` (= ngram-mod, n=24, min=48, max=64) と派生フラグを
    ``mode`` で選択する。

    mode の意味:

    - ``"default"``: ``--spec-default`` 単独。draft モデル不要のプリセット。
    - ``"ngram-mod"`` / ``"ngram-cache"`` / ``"ngram-simple"`` /
      ``"ngram-map-k"`` / ``"ngram-map-k4v"``: self-speculative ngram 系。
      ``draft_max`` / ``draft_min`` / ``ngram_size_n`` 等を明示する。
    - ``"draft-model"``: 外部 draft モデル経路。``draft_model_path`` 必須。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: str = Field(
        default="default",
        pattern=(
            r"^(default|ngram-mod|ngram-cache|ngram-simple|"
            r"ngram-map-k|ngram-map-k4v|draft-model)$"
        ),
    )
    # 以下の 3 つは上流 llama.cpp が 2026-05 に総称フラグを削除し mode 別へ
    # 分割したため、実フラグ名が mode 依存になる。対応表と、対応フラグを持たない
    # 組み合わせの graceful degrade は ``scripts/launch_llama.py`` の
    # ``build_speculative_args`` (と ``_NGRAM_MOD_MODES`` / ``_NGRAM_SIZED_MODES``)
    # が単一情報源。

    # 1 サイクルで採択を試みる draft トークン数の上下限。null で未指定 (= 上流既定)。
    # default / ngram-mod → ``--spec-ngram-mod-n-max`` / ``-n-min``
    # それ以外            → ``--spec-draft-n-max`` / ``-n-min``
    draft_max: int | None = Field(default=64, ge=0)
    draft_min: int | None = Field(default=48, ge=0)
    # draft 確率閾値 (greedy 採択時の足切り、上流 ``--draft-p-min``)。null で未指定。
    draft_p_min: float | None = Field(default=None, ge=0.0, le=1.0)
    # ⚠ 現在は **無視される**。上流が ``-cd`` / ``--ctx-size-draft`` を代替フラグごと
    # 削除したため、>0 を指定しても warning が出るだけでフラグは付与されない。
    ctx_size_draft: int = Field(default=0, ge=0)
    # ngram lookup 長。default / ngram-mod → ``--spec-ngram-mod-n-match``、
    # ngram-simple / ngram-map-k / ngram-map-k4v → ``--spec-<mode>-size-n``、
    # ngram-cache / draft-model → 対応フラグ無し (warning)。
    ngram_size_n: int | None = Field(default=24, ge=0)
    # ngram draft 長。ngram-simple / ngram-map-k / ngram-map-k4v →
    # ``--spec-<mode>-size-m``。それ以外は対応フラグ無し (warning)。
    ngram_size_m: int | None = Field(default=None, ge=0)
    # mode="draft-model" 専用: 外部 draft モデル GGUF 相対 / 絶対パス。
    draft_model_path: str | None = None
    # mode="draft-model" 専用: draft モデルの GPU 層数。null=auto (model 全層)、
    # 整数値、または ``"all"`` (全層) を許容する。
    gpu_layers_draft: int | str | None = None


class LlamaConfig(BaseModel):
    """llama.cpp 設定"""

    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    port: int = Field(default=8080, ge=1024, le=65535)
    # ``-c`` / コンテキスト長。``None`` (既定) は arch プロファイル
    # (``models/profiles/<arch>.yaml`` の ``context_size``) を参照し、それも
    # 未宣言なら 8192 にフォールバックする。明示 int を設定するとプロファイル
    # より優先される (解決順: config 明示 > profile > 8192)。
    context_size: int | None = Field(default=None, ge=512)
    # 整数 (-1 / 0 / 999 等) または ``"auto"`` 文字列。
    # ``"auto"`` 指定時は scripts/launch_llama.py の resolve 関数が
    # GPU 物理容量 + GGUF layer 数 + Vulkan host buffer headroom から
    # 段階的に縮小した値 (100% / 80% / 60% / 40% / 0%) を算出する。
    # iGPU 環境で base の GPU メモリ占有を抑え、embed の
    # Vulkan host buffer 確保失敗 (ErrorOutOfDeviceMemory) を回避する用途。
    gpu_layers: int | Literal["auto"] = Field(default=999)

    @field_validator("gpu_layers", mode="before")
    @classmethod
    def _validate_gpu_layers(cls, v: object) -> int | str:
        """整数値は ``>= -1`` の制約を維持、``"auto"`` 文字列のみ追加で許容。"""
        if isinstance(v, str):
            if v == "auto":
                return v
            raise ValueError(
                f"gpu_layers must be an int (>=-1) or 'auto', got string {v!r}"
            )
        iv = int(v)
        if iv < -1:
            raise ValueError(f"gpu_layers must be >= -1, got {iv}")
        return iv
    threads: int = Field(default=0, ge=0)
    batch_size: int = Field(default=512, ge=1)
    flash_attn: bool = True
    mlock: bool = False
    cache_prompt: bool = True
    # 3 = チャット (0) / バックグラウンド (1: 学習 / sleep-time) / 分類器 (2:
    # ツール分類器 + チャット応答パスの補助判定) をスロット分離する既定。
    # 2 では分類器が背景と相乗りし、1 では全部が直列化して Level 1 変異 1 回ぶん
    # (実測 16〜32 秒) チャットが待たされる。kv_unified が slots>1 で自動付与
    # されるため per-seq context は n_ctx のままで、VRAM は増えない。
    # 起動時に llama-server の実スロット数 (``/props`` の total_slots) と照合し、
    # 少ない方へ丸める (backend/factory/_pillar_wirer.py::_init_llama_server)。
    slots: int = Field(default=3, ge=1, le=16)
    cache_type_k: str = Field(
        default="q8_0", pattern=r"^(f16|bf16|q8_0|q5_1|q5_0|q4_1|q4_0)$"
    )
    cache_type_v: str = Field(
        default="q8_0", pattern=r"^(f16|bf16|q8_0|q5_1|q5_0|q4_1|q4_0)$"
    )
    # 共通 prefix 自動再利用 (上流 ``--cache-reuse``)。0=無効。N>0 で N トークン
    # 以上の連続接頭辞一致を再 prefill せず KV を再利用する。多ターン chat で
    # system/RAG 接頭辞の prefill コストを削減する用途。
    # ⚠️ SWA (sliding window attention) モデル (gemma-4 等) では llama.cpp が
    # ``cache_reuse is not supported by this context`` で自動無効化し、毎ターン
    # full re-prefill するため **no-op** (フラグは無害に無視される)。非 SWA base
    # に差し替えた場合のみ有効。
    cache_reuse: int = Field(default=256, ge=0)
    # 0 = 上限を config で決めない。ただし **無制限では投げない** —
    # ``LocalClient._build_payload`` が context_size からプロンプト推定を引いた
    # 残量 (下限 256) へクランプして送る。
    max_tokens: int = Field(default=1024, ge=0)
    lora_target: str = "auto"
    # None = 未指定 (arch プロファイルの reasoning.enable_thinking / 能力判定で決定。
    # 非対応 arch には送らない)。True/False = ユーザー明示でプロファイルを上書き。
    enable_thinking: bool | None = None
    # SSE ストリーミング受信開始までの read timeout (秒)。
    # llama-server 再起動直後の冷えた KV キャッシュ + 長いプロンプトで prefill が
    # 30s を超える環境 (iGPU 等) では 60〜120s に引き上げる。
    stream_first_token_timeout_sec: float = Field(default=60.0, ge=1.0)
    # idle slot offload
    # cache_ram_mib: -1=無制限 / 0=disable / >0=MiB 上限。上流既定 8192 を
    # 黙従しないよう launch_llama.py から常に明示付与する。
    cache_ram_mib: int = Field(default=0, ge=-1)
    # 上流既定 true (idle slot を退避対象にする)。false で機構自体を OFF。
    #
    # **本プロジェクトの既定は False**。true にしても cache_ram_mib=0 では
    # 退避先が無く llama-server 側で無効化されるため、宣言と実効が食い違う
    # (下の warn_idle_slot_cache_without_ram_budget が毎起動で警告する)。
    # 恒常的に出る警告は読み飛ばされるようになるので、既定は実効に合わせる。
    #
    # 実測 (2026-08-27 ライブ監査 171 ターン) でも >0 にする根拠は無かった —
    # 低キャッシュだった deliberative 7 件のうち 6 件は aux の直後ではなく、
    # プロンプト前方が変わった形だった。idle 退避が救うのはスロットが
    # 追い出されるケースなので、この形には効かない。
    cache_idle_slots: bool = False
    # null=auto (slots>1 で ``--kv-unified`` 自動付与) / true|false で明示上書き。
    # unified KV は multi-slot で VRAM 増なく各シーケンスへ full n_ctx を与え
    # (非 unified だと per-seq context が n_ctx/slots に半減)、かつ cache-ram の
    # 動作前提でもある。cache_ram_mib とは独立。
    kv_unified: bool | None = None
    # self-speculative decoding。Pro 限定機能
    speculative: LlamaSpeculativeConfig = Field(default_factory=LlamaSpeculativeConfig)
    # MTP (Multi-Token Prediction)。Free/Pro 共通。MTP ヘッド内蔵モデルでのみ有効。
    # speculative と同時 enabled の場合は launch 側で MTP 優先 + speculative 無効化。
    mtp: MtpConfig = Field(default_factory=MtpConfig)
    # モデル arch 別の起動フラグ / sampling 自動決定。true で GGUF メタデータ +
    # 同梱 model_profiles から --jinja / --reasoning-format / MoE / sampling 既定を
    # 解決する。false で全機構 OFF (従来挙動)。
    auto_model_flags: bool = True
    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def warn_idle_slot_cache_without_ram_budget(self) -> "LlamaConfig":
        """``cache_idle_slots`` が ``cache_ram_mib=0`` で無効化される構成を通知する。

        ``--cache-idle-slots`` は退避先 (``--cache-ram``) が無いと機能しない。
        llama-server はこれを起動時に自分で無効化するが、警告は **ランチャの
        端末にしか出ずログファイルに残らない**ため、config 上は有効に見えたまま
        効いていない状態が続く。

            W srv init: --cache-idle-slots requires --cache-ram, disabling

        (2026-08-16 ライブ監査時の llama-base.stderr.log で確認。同セッションでは
        窓の押し出しごとに全再 prefill が走っており、idle slot の KV 退避は
        まさに効かせたい機構だった。)

        エラーにはしない: ``cache_idle_slots`` を明示 false にせず ``cache_ram_mib``
        だけで機構ごと切るのは正当な設定なので、片方を触った時に気付ければよい。
        """
        if self.cache_idle_slots and self.cache_ram_mib == 0:
            from backend.log_config import get_logger

            get_logger("config").warning(
                "llama.cache_idle_slots=true has no effect while "
                "llama.cache_ram_mib=0: llama-server disables --cache-idle-slots "
                "when --cache-ram is absent. Set cache_ram_mib > 0 (or -1 for "
                "unlimited) to keep idle slot KV, or set cache_idle_slots=false "
                "to make the intent explicit.",
            )
        return self


class ProfileSamplingConfig(BaseModel):
    """モデルプロファイルの ``sampling`` セクション。

    モデルカードが推奨する生成パラメータの宣言先。宣言したキーだけが
    ``model_dump(exclude_none=True)`` で返り、未宣言キーは ``modes.*`` (base) /
    呼出側既定に委ねられる。プロファイル由来の値は llama-server の
    リクエストへ直接載るため、``extra="forbid"`` + 範囲制約で早期に弾く
    (検証に失敗したセクションは WARNING を出して丸ごと無視する)。
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0)


class ProfileReasoningConfig(BaseModel):
    """``models/profiles/<arch>.yaml`` の ``reasoning`` セクション (docs/c_15)。

    reasoning ハンドリングの SSOT。**起動時のプロファイル読込でのみ反映**され、
    UI/管理画面からの実行時切替は持たない。``server_control`` で 2 経路へ振り分ける:
    - True (llama.cpp が reasoning を認識する Qwen3/DeepSeek 系): 起動フラグ
      (``-rea`` / ``--reasoning-format`` / ``--reasoning-budget``) でサーバ側制御。
    - False (lfm2moe 等 ``thinking=0``): クライアント側ハンドリング
      (``_ReasoningFilter`` のマーカー駆動除去 / watchdog / 履歴除外)。
    """

    model_config = ConfigDict(extra="forbid")

    # 能力宣言
    mode: Literal["toggle", "always", "none"] = "none"
    enable_thinking: bool | None = None  # toggle 型の既定

    # サーバ側制御 (server_control=True; 起動フラグへ)
    # 配線状況の SSOT は docs/c_15 §2.7。reasoning_format は未消費 (--reasoning-format は
    # profile launch_flags 由来)、budget_default/budget_message のみ起動フラグへ反映。
    server_control: bool = False
    reasoning_format: Literal["auto", "deepseek", "deepseek-legacy", "none"] = "auto"
    budget_default: int = Field(default=-1, ge=-1)
    budget_message: str = ""

    # クライアント側ハンドリング (server_control=False)
    # 現状 client_think_budget (watchdog) のみ消費。client_handling は strip 相当のみ・
    # marker_style/think_open/think_close は _ReasoningFilter のハードコードで未消費・
    # on_runaway は値保持のみ (watchdog は値に依らず一律 stream 中断)・exclude_from_history は
    # strip 副作用で達成 (明示配線なし)。詳細は docs/c_15 §2.7。
    client_handling: Literal["strip", "channel", "passthrough"] = "strip"
    marker_style: Literal["think_tags", "harmony", "custom"] = "think_tags"
    think_open: str = "<think>"
    think_close: str = "</think>"
    client_think_budget: int = Field(default=512, ge=0)
    on_runaway: Literal["reask", "truncate", "fallback"] = "fallback"
    exclude_from_history: bool = True
