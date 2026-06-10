"""その他の単発スキーマ群

EvorefConfig 直下の細かいセクションをここに集約する。
専用ファイル (llm / rag / memory / learning / loop / paths /
assist_model) に属さない Config を置く。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InstanceConfig(BaseModel):
    """インスタンス設定"""

    model_config = ConfigDict(extra="forbid")

    name: str = "evoref"


class ServerConfig(BaseModel):
    """サーバー設定"""

    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    frontend_port: int = Field(default=5173, ge=1024, le=65535)
    timeout: int = Field(default=30, ge=1)


class RuntimeConfig(BaseModel):
    """実行環境ランタイム設定

    GPU / CPU 等、llama-server 群を起動するハードウェアの制約と、
    llama-server バイナリのバージョン要件を表現する。

    - ``total_vram_budget_mb``: VRAM 予算検査のソフト上限 (MB)
      ``scripts/launch_llama.py --all`` が本値を参照し、ベース / アシスト /
      埋め込み / リランカーの GPU レイヤ推定 VRAM 合計がこの値を超える場合に
      警告を出して起動を中断する (``--force`` で強制起動可能)。
      None の場合は検査を行わない (従来挙動と同じ)。
    - ``min_llamacpp_build``: 起動時に確認する llama-server の
      最低 build 番号 (例: ``"b8946"``)。CVE-2026-21869 (heap-buffer-overflow)
      および SWA-full ロジック修正取り込み済みのビルドを要件として宣言する。
      None の場合は検査を行わない。
    - ``enforce_min_llamacpp_build``: True の場合、最低 build
      未満を検出すると ``scripts/launch_llama.py`` は exit code 3 でアボート
      する。False (既定) の場合は stderr 警告のみで起動継続。
      バイナリが build 番号を露出しないカスタムビルドの場合は本フラグに
      関わらず警告のみで継続する (検出失敗 ≠ 要件未満)。
    - ``fit_params_enabled``: True (既定) で ``llama-fit-params``
      バイナリを使った Tier 1 VRAM 推定を試みる。バイナリ未存在 / タイム
      アウト時は GGUF サイズベースの Tier 2 ヒューリスティックにフォール
      バックする。False で Tier 1 をスキップ (常に Tier 2)。
    - ``fit_params_timeout_sec``: Tier 1 推定 (1 モデルあたり) の
      タイムアウト秒数。既定 10 秒。
    - ``fit_params_binary``: ``llama-fit-params`` バイナリのパス
      または PATH 上のコマンド名。既定 ``"llama-fit-params"``。
    - ``llama_endpoint_policy``: llama-server エンドポイント利用
      ポリシー宣言。``"oai-only"`` (既定) で OpenAI 互換のみ採用、
      ``"messages-allowed"`` は将来 ``/v1/messages`` 採用判断を行った
      場合の予約値。挙動制御は行わず宣言的フラグ。実体の検証は
      ``backend/free/tests/test_llama_endpoint_policy.py`` の静的検査で
      担保する。詳細 ``docs/e_05_llama_server_endpoint_policy.md``。
    """

    model_config = ConfigDict(extra="forbid")

    # GPU 搭載 VRAM のソフト上限 (MB)。None の場合は検査を行わない。
    total_vram_budget_mb: int | None = Field(default=None, ge=1)

    # llama-server の最低 build 番号 (例: "b8946")。None の場合は検査を行わない。
    # 形式: "b" プレフィックス + 整数、または整数のみ ("8946" でも可)。
    min_llamacpp_build: str | None = Field(
        default=None, pattern=r"^b?\d+$",
    )

    # True で最低 build 未満時に launch_llama.py を exit code 3 でアボートさせる。
    # False (既定) なら警告のみで起動継続。
    enforce_min_llamacpp_build: bool = False

    # llama-fit-params -fitp on ベースの Tier 1 推定を試みるか
    fit_params_enabled: bool = True

    # Tier 1 推定 (1 モデルあたり) のタイムアウト秒数。
    fit_params_timeout_sec: float = Field(default=10.0, gt=0.0)

    # llama-fit-params バイナリのパスまたは PATH 上のコマンド名。
    fit_params_binary: str = "llama-fit-params"

    # llama-server エンドポイント利用ポリシー宣言
    llama_endpoint_policy: Literal["oai-only", "messages-allowed"] = "oai-only"

    # ``llama.gpu_layers`` / ``assist_model.local.gpu_layers`` が ``"auto"``
    # の場合の Vulkan host buffer 予約量 (MiB)。
    # iGPU 環境では embed (CPU モード) でも llama.cpp が Vulkan
    # backend を初期化し model loading 時に host (pinned) buffer を要求する。
    # base/assist が GPU メモリを占有しすぎると後発の embed で
    # ``ggml_vulkan: Failed to allocate pinned memory (ErrorOutOfDeviceMemory)``
    # 警告が出るため、自動段階縮小ロジックはこの分を GPU 容量から差し引いて
    # 予算を計算する。既定 4 GiB は AMD Radeon 890M + Qwen3-Embedding-0.6B
    # 構成での実測 ceiling から決定。
    vulkan_host_buffer_headroom_mib: int = Field(default=4096, ge=0)

    # ``"auto"`` 指定された ``gpu_layers`` を実際に縮小する kill switch。
    # False にすると ``"auto"`` 指定があっても従来の全 offload (999) 挙動に
    # フォールバックする (warning 1 行を残す)。GGUF parse や llama-fit-params
    # に何らかの異常が出た場合の緊急回避用。
    gpu_auto_tune_enabled: bool = True

    # モデル接続/切替時に実挙動プローブ (reasoning 分離 / <think> / json_schema 強制)
    # を実行し、宣言と実機の食い違いを観測して処理を自動適応するか。
    # True (既定) でバックグラウンド実行 (起動遅延なし、完了後に観測値へ切替)。
    # プローブ失敗時は prior (プロファイル宣言) にフォールバックする。
    # 詳細 docs/c_15_model_capability_adaptation.md。
    capability_probe: bool = True


class ThemeConfig(BaseModel):
    """テーマ設定"""

    model_config = ConfigDict(extra="forbid")

    active: str = ""
    color_mode: str = Field(default="dark", pattern=r"^(dark|light)$")
    trusted: list[str] = Field(default_factory=list)
    cli_layout_mode: str = Field(default="auto", pattern=r"^(auto|split|sequential)$")


class CLITimeoutsConfig(BaseModel):
    """CLI サブコマンド別タイムアウト (秒)

    ``backend/free/cli/cartridge_commands.py::CartridgeTimeouts`` の
    既定値と同期させること。カートリッジ install/rebuild/create は
    サーバ側の処理時間が長いため default より大きい値を用いる。
    """

    model_config = ConfigDict(extra="forbid")

    default: float = Field(default=30.0, gt=0.0)
    install: float = Field(default=300.0, gt=0.0)
    rebuild: float = Field(default=300.0, gt=0.0)
    create: float = Field(default=600.0, gt=0.0)


class CLIConfig(BaseModel):
    """CLI 設定

    ``config.yaml`` の ``cli`` セクションは CLI バックエンドクライアント側の
    既定値をまとめる。現状は HTTP タイムアウトのみ。
    """

    model_config = ConfigDict(extra="forbid")

    timeouts: CLITimeoutsConfig = Field(default_factory=CLITimeoutsConfig)


class HistoryConfig(BaseModel):
    """会話履歴設定"""

    model_config = ConfigDict(extra="forbid")

    auto_save: bool = True
    checkpoint_interval: int = Field(default=10, ge=1)
    retention_full_days: int = Field(default=90, ge=1)
    retention_compressed_days: int = Field(default=365, ge=1)
    max_storage_mb: float = Field(default=200, ge=1)
    compress_preview_chars: int = Field(default=100, ge=1)
    summary_batch_size: int = Field(default=5, ge=1)


class StreamingConfig(BaseModel):
    """ストリーミング設定"""

    model_config = ConfigDict(extra="forbid")

    keepalive_interval_sec: float = Field(default=15.0, ge=1.0, le=120.0)


class AgentConfig(BaseModel):
    """エージェント設定"""

    model_config = ConfigDict(extra="forbid")

    step_compaction_enabled: bool = True
    step_compaction_rag_lines: int = Field(default=2, ge=1)
    step_compaction_command_head_tail: int = Field(default=5, ge=1)
    reminders_enabled: bool = True
    max_reminders_per_turn: int = Field(default=2, ge=0)
    dangerous_command_block: bool = True
    # アシスト接続時はモード非依存で常時有効 (False で明示オプトアウト)。
    # アシスト未接続時はこの値に関わらず無効 (決定論ショートカットのみ)。
    tool_judge_enabled: bool = True
    meta_cognitive_enabled: bool = True
    meta_cognitive_min_budget: int = Field(default=512, ge=0)
    # tool_judge_enabled=True (coding mode 等) で 4 層フォールバックが全て
    # no-tool を返した際、assist (executable_command_synth) で環境依存事実
    # クエリを判定し run_command に橋渡しする 5 層目のゲート。
    executable_command_fallback_enabled: bool = True
    executable_command_fallback_min_chars: int = Field(default=5, ge=0, le=80)
    # ── コンテンツ生成タイムアウト (Meta-Cognitive 層) ──
    # ファイル内容生成はトークン間アイドル (無出力) タイムアウトで「停止した
    # ストリーム」を素早く諦めつつ、低速だが進行中の生成は総上限まで継続する。
    # 最初の1トークンまでは別枠の長めタイムアウト (サーバが他生成で busy の間
    # 待機中のリクエストを誤って「停止」扱いしないため) を使う。
    content_gen_first_token_timeout: int = Field(default=120, ge=10, le=600)
    content_gen_idle_timeout: int = Field(default=30, ge=5, le=300)
    content_gen_timeout: int = Field(default=600, ge=30)
    llm_call_timeout: int = Field(default=90, ge=10)
    total_timeout: int = Field(default=900, ge=60)
    # coding モードで editor/chat 出力のコード生成を LongForm 細粒度生成
    # (CodeUnit 計画 → ファイル別生成・検証・修正) へ委譲する。大規模実装は
    # 複数ファイルへ分割出力可能。False で従来の単一ショット生成に戻す。
    delegate_codegen_to_longform: bool = True

    @model_validator(mode="after")
    def _validate_content_gen_timeout_order(self) -> "AgentConfig":
        """content_gen タイムアウト 3 値の整合性チェック。

        意味的には ``idle <= first_token <= total`` でなければ「最初のトークン
        受信前に idle で打ち切る」等の矛盾挙動になる。`Field(ge/le)` の単体
        範囲制約では検出できないため、起動時に明示エラーで止める。
        """
        if self.content_gen_idle_timeout > self.content_gen_first_token_timeout:
            raise ValueError(
                "content_gen_idle_timeout "
                f"({self.content_gen_idle_timeout}) must be <= "
                f"content_gen_first_token_timeout "
                f"({self.content_gen_first_token_timeout})"
            )
        if self.content_gen_first_token_timeout > self.content_gen_timeout:
            raise ValueError(
                "content_gen_first_token_timeout "
                f"({self.content_gen_first_token_timeout}) must be <= "
                f"content_gen_timeout ({self.content_gen_timeout})"
            )
        return self


class ToolsConfig(BaseModel):
    """ツール設定"""

    model_config = ConfigDict(extra="forbid")

    fetch_url_enabled: bool = False
    fetch_url_timeout: int = Field(default=10, ge=1)
    fetch_url_allow_private_ip: bool = False
    # ── URL リコール (Phase 1) ──
    # 過去質問で正しく fetch_url できた URL を新規類似質問で再利用するための設定。
    # 書き込みは sleep-time の url_curator が担当し、引き当ては
    # ToolCallJudge._try_recall_url が決定論的に行う (アシスト同期発火なし)。
    url_recall_enabled: bool = True
    url_recall_topk: int = Field(default=5, ge=1, le=20)
    url_recall_min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    url_recall_min_record_score: float = Field(default=0.6, ge=0.0, le=1.0)
    url_recall_record_history_size: int = Field(default=10, ge=1, le=100)
    # TTL — 最終 fetch から N 日経過した URL は引き当て時に score を半減して
    # min_record_score 閾値判定する (鮮度ペナルティ)。``0`` で無効化。
    url_recall_ttl_days: int = Field(default=30, ge=0, le=365)
    # ── executable command リコール ──
    # 過去成功した run_command を新規類似クエリで再利用するための設定。
    # 書き込みは sleep-time の executable_command_curator が担当し、引き当ては
    # ToolCallJudge._try_recall_executable_command が決定論的に行う。
    executable_command_recall_enabled: bool = True
    executable_command_recall_topk: int = Field(default=5, ge=1, le=20)
    executable_command_recall_min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    executable_command_recall_min_record_score: float = Field(default=0.6, ge=0.0, le=1.0)
    executable_command_recall_record_history_size: int = Field(default=10, ge=1, le=100)
    executable_command_recall_ttl_days: int = Field(default=30, ge=0, le=365)


class WidgetApiConfig(BaseModel):
    """ウィジェットプロキシ個別 API 設定"""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    api_key_env: str = ""
    api_key_header: str = ""
    api_key_param: str = ""
    rate_limit: str = "10/min"
    allowed_paths: list[str] = Field(default_factory=lambda: ["*"])


class WidgetProxyConfig(BaseModel):
    """ウィジェットプロキシ設定"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    global_rate_limit: str = "60/min"
    request_timeout_sec: int = Field(default=10, ge=1)
    max_response_size_kb: int = Field(default=512, ge=1)
    cache_ttl_sec: int = Field(default=60, ge=0)
    apis: list[WidgetApiConfig] = Field(default_factory=list)


class I18nConfig(BaseModel):
    """i18n 設定"""

    model_config = ConfigDict(extra="forbid")

    locale: str = Field(default="ja", pattern=r"^(ja|en)$")
    fallback: str = Field(default="ja", pattern=r"^(ja|en)$")
    prompt_locale: str = Field(default="ja", pattern=r"^(ja|en)$")


class LongFormConfig(BaseModel):
    """長文生成設定"""

    model_config = ConfigDict(extra="forbid")

    max_units: int = Field(default=20, ge=1)
    unit_max_tokens: int = Field(default=2000, ge=100)
    rolling_short_term_chars: int = Field(default=1000, ge=100)
    review_enabled: bool = True
    max_revisions: int = Field(default=3, ge=0)
    # 検証ゲート付きコードリペア: 生成後に assembled を検証し、文法/未定義
    # エラーが残ればアシストモデルで修正→再検証を繰り返す。Python は AST で
    # 厳密検証 (構文 + 未定義名)、他言語は assist による軽い構文自己点検。
    repair_enabled: bool = True
    max_repair_rounds: int = Field(default=2, ge=0)
    rag_per_unit: bool = True
    rag_top_k_per_unit: int = Field(default=3, ge=1)
    # コード生成の事前準備: 設計仕様 (contract) 合成。各ユニットへ注入 +
    # SPEC.md として出力し、ファイル横断の整合性を担保する。
    code_spec_enabled: bool = True
    # 設計仕様から mermaid フローチャートを合成し SPEC.md に埋め込む (既定 OFF)。
    code_flowchart_enabled: bool = False
    # 生成物の import スモークテスト (temp dir でサブプロセス import 検証)。
    # ModuleNotFoundError / dataclass 引数違い等の実行時エラーを捕捉する。
    code_smoke_test_enabled: bool = True
    code_smoke_timeout_sec: float = Field(default=10.0, ge=1.0)
    # 長文生成 1 リクエストの総ウォールクロック上限 (秒)。0 で無効化。
    # 低速ローカル GPU では正当な生成が 10-20 分かかるため既定は余裕を持たせる。
    # 超過時は生成済みユニットで打ち切り、部分結果を返す (無限ハング防止)。
    total_timeout_sec: float = Field(default=1800.0, ge=0.0)


class ChatModeConfig(BaseModel):
    """チャットモード生成パラメータ

    チャットは常にベースモデル (`model_paths.base_model`) を使用するため、
    独立した ``model`` フィールドは持たない
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=1000)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class CodingModeConfig(BaseModel):
    """コーディングモード生成パラメータ

    コーディング用モデルパスは ``model_paths.coding_model`` に保持する
    (未指定時はベースモデルにフォールバック)。本モデルは生成パラメータのみ。
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0, le=1000)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class ModesConfig(BaseModel):
    """モード別設定

    - ``chat``: ベースモデル (= チャット用モデル) を使用。モデル指定不可。
    - ``coding``: 生成パラメータのみ。モデルパスは ``model_paths.coding_model``。
    """

    model_config = ConfigDict(extra="forbid")

    chat: ChatModeConfig = Field(default_factory=ChatModeConfig)
    coding: CodingModeConfig = Field(default_factory=CodingModeConfig)


class EditorSettingsConfig(BaseModel):
    """エディタ設定"""

    model_config = ConfigDict(extra="forbid")

    font_family: str = "monospace"
    font_size: int = Field(default=14, ge=8, le=48)
    line_height: float = Field(default=1.5, ge=1.0, le=3.0)
    tab_size: int = Field(default=4, ge=1, le=8)
    word_wrap: bool = False
    show_line_numbers: bool = True
    show_active_line: bool = True
    show_toolbar: bool = True
    default_encoding: str = Field(
        default="utf-8",
        pattern=r"^(utf-8|shift_jis|euc-jp|iso-2022-jp)$",
    )
    default_line_ending: str = Field(
        default="lf", pattern=r"^(lf|crlf|cr)$"
    )
    highlight_languages: list[str] = Field(
        default_factory=lambda: [
            "markdown", "python", "javascript", "typescript",
            "json", "html", "css", "yaml", "xml", "sql", "php",
        ]
    )


class ProcessManagerConfig(BaseModel):
    """llama-server プロセスマネージャ設定

    `enabled=True` で lifespan 起動時に 3 種 (base/assist/embedding)
    の llama-server を自動 spawn し、`migrate_component` API からの自動
    再起動を有効化する。`False` (デフォルト) は従来通り
    `scripts/launch_llama.py` を別途実行するワークフロー互換。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    health_timeout: int = 60
    stop_timeout: int = 10


class TerminalConfig(BaseModel):
    """Pro Web ターミナル設定

    バックエンドで PTY (pseudo-terminal) を起動し、WebSocket 経由で xterm.js
    とブリッジする。POSIX (``ptyprocess``) と Windows (``pywinpty``) の
    両方に対応する。Free 版には実装しない (Free 観察手段はループ
    イベント観察 UI が代替)。

    セキュリティ 4 重ガード:
        1. ``allowed_origins`` の Origin 検証
        2. Host ヘッダ検証 (DNS rebinding 対策、127.0.0.1 / localhost のみ)
        3. シングルユース token 検証 (60 秒以内、1 回限り)
        4. ``server.host=0.0.0.0`` + ``enabled=true`` は起動時にエラー

    ``shell`` の値:
        - ``auto`` (推奨): POSIX は ``$SHELL`` → ``bash`` → ``zsh`` → ``sh``、
          Windows は ``pwsh.exe`` → ``powershell.exe`` → ``cmd.exe``
        - ``bash`` / ``zsh`` / ``sh``: POSIX 用 shell (Windows でも Git
          Bash / WSL の同名バイナリが PATH にあれば利用可)
        - ``pwsh`` / ``powershell`` / ``cmd``: Windows 用 shell

    エージェント主導の PTY 書込
        - ``agent_write_enabled`` (既定 ``False``): エージェント (Reactive /
          Deliberative / Meta-Cognitive) からの ``terminal_exec`` ツール経由
          の書込を有効化する。既定 OFF。LLM 生成文字列を shell に渡すため
          攻撃面が大きい — 有効化時は allowlist + dangerous pattern + ログ
          記録の 3 重ガードで保護する。
        - ``agent_command_allowlist``: 正規表現の許可リスト。空リストの場合は
          すべてのコマンドを許可 (dangerous pattern は別途常時ブロック)。
          1 件以上のパターンが指定された場合は **少なくとも 1 件にマッチ**
          したコマンドのみを実行する。
        - ``agent_read_timeout_sec`` (既定 ``10``): ``terminal_exec`` の実行
          結果を PTY から読み取る最大秒数。タイムアウトに達しても shell
          プロセスは継続し、累積した出力を返す。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    shell: str = Field(
        default="auto",
        pattern=r"^(auto|bash|pwsh|powershell|zsh|sh|cmd)$",
    )
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
    )
    max_sessions: int = Field(default=4, ge=1, le=64)
    idle_timeout_sec: int = Field(default=1800, ge=60)
    token_ttl_sec: int = Field(default=60, ge=10, le=600)

    # agent → PTY 書込統合
    agent_write_enabled: bool = False
    agent_command_allowlist: list[str] = Field(default_factory=list)
    agent_read_timeout_sec: int = Field(default=10, ge=1, le=300)


class ProUrlRecallConfig(BaseModel):
    """Pro URL リコール拡張設定

    Free 側 ToolCallJudge は個人プロファイル (``profile_id``) のみで URL
    fact を絞るが、Pro ではここに列挙された ``team_profile_ids`` も引き当て
    candidate として許容する。共有プールの物理ストレージは追加せず、既存の
    ``global`` SemanticFactStore に ``profile_id="team:default"`` 等で書き込
    まれた URL fact を読み取り対象にする (Phase 2)。
    """

    model_config = ConfigDict(extra="forbid")

    team_profile_ids: list[str] = Field(default_factory=list)


class ProConfig(BaseModel):
    """Pro 専用機能の設定ルート

    Pro エディション固有の横断機能 (Web ターミナル等) をここに集約する。
    各サブセクションは ``enabled=false`` をデフォルトとし、明示的に有効化
    された場合のみ Pro 起動シーケンスでセットアップされる。
    """

    model_config = ConfigDict(extra="forbid")

    terminal: TerminalConfig = Field(default_factory=TerminalConfig)
    url_recall: ProUrlRecallConfig = Field(default_factory=ProUrlRecallConfig)


class ModelMigrationConfig(BaseModel):
    """ベースモデル移行 / 起動時整合性チェック設定

    `strict_startup_check=True` にすると、起動時に
    `local/model_state.json` の `current_filename` と
    `config.yaml.model_paths.base_model` のファイル名が一致しない場合に
    `RuntimeError` を送出して起動をブロックする。

    `auto_migrate_on_startup=True` にすると、起動時 mismatch を検出した際に
    `ModelMigrator.migrate()` を自動実行し、LoRA アーカイブ / 経験バッファ
    base_model 付与 / プロンプト meta 更新 / eval_core リセット /
    model_state.json 更新を行ってから起動を継続する
    両方 true の場合は auto_migrate を優先し、失敗したときのみ strict 側の
    RuntimeError にフォールバックする。
    """

    model_config = ConfigDict(extra="forbid")

    strict_startup_check: bool = False
    auto_migrate_on_startup: bool = False
