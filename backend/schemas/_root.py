"""evoref 全体設定スキーマのルート

config.yaml の全セクションを集約する ``EvorefConfig`` と、
YAML を辞書として正規化する ``validate_config`` を提供する。
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.config_normalize import CONFIG_VERSION
from backend.log_config import get_logger
from backend.schemas._common import (
    AgentConfig,
    CLIConfig,
    HistoryConfig,
    I18nConfig,
    InstanceConfig,
    LongFormConfig,
    ModelMigrationConfig,
    ModesConfig,
    AutoServeConfig,
    ProcessManagerConfig,
    ProConfig,
    RuntimeConfig,
    ServerConfig,
    StreamingConfig,
    ThemeConfig,
    ToolsConfig,
    WidgetProxyConfig,
)
from backend.schemas.create import CreateConfig
from backend.schemas.learning import LearningConfig, ScheduleConfig
from backend.schemas.llm import LlamaConfig
from backend.schemas.memory import MemoryConfig
from backend.schemas.paths import LocalPathsConfig, ModelPathsConfig
from backend.schemas.prompt import PromptConfig
from backend.schemas.rag import EmbeddingConfig, RAGConfig

logger = get_logger("schemas")


# ── Pro 限定キー警告のデデュープ用キャッシュ ────────────────
#
# Free エディションで Pro 限定キーが explicit に設定されていた場合に、
# ``EvorefConfig.warn_pro_only_keys_in_free`` が一度だけ warning を出す。
# 同一キーが複数回 ``model_validate`` (起動時 + ホットリロード等) されても
# ログが氾濫しないよう、``(edition, key)`` の組で抑止する。
#
# テスト用途で意図的にリセットしたい場合は ``_reset_pro_only_warning_cache``
# を内部 API として公開する。
_WARNED_PRO_KEYS: set[tuple[str, str]] = set()


def _reset_pro_only_warning_cache() -> None:
    """Pro 限定キー警告のデデュープキャッシュをクリアする (テスト専用)"""
    _WARNED_PRO_KEYS.clear()


# ── トップレベル設定モデル ──────────────────────────────────


#: ``memory.working_max_tokens`` が ``llama.context_size`` に占めるべき割合の
#: 下限 / 上限。config.yaml.example の推奨表 (context 8192 → 4096 = 50%) と
#: 「context_size * 0.6 を超えない」という既存の但し書きから採った。
_WORKING_WINDOW_MIN_SHARE = 0.35
_WORKING_WINDOW_MAX_SHARE = 0.60


class EvorefConfig(BaseModel):
    """evoref 全体設定スキーマ

    config.yaml の全セクションを型安全に定義する。
    未知のトップレベルキーは許可（Pro 拡張等に対応）。
    """

    model_config = ConfigDict(extra="allow")

    # 設定ファイルの版 (c_05 §7.6)。版の無い G0 の config と古い版の config は
    # ``evoref config normalize`` が一度だけ直して現行の版を付ける。
    config_version: int = Field(default=CONFIG_VERSION, ge=1)
    instance: InstanceConfig = Field(default_factory=InstanceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    llama: LlamaConfig = Field(default_factory=LlamaConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    model_paths: ModelPathsConfig = Field(default_factory=ModelPathsConfig)
    local_paths: LocalPathsConfig = Field(default_factory=LocalPathsConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    widget_proxy: WidgetProxyConfig = Field(default_factory=WidgetProxyConfig)
    i18n: I18nConfig = Field(default_factory=I18nConfig)
    modes: ModesConfig = Field(default_factory=ModesConfig)
    model_migration: ModelMigrationConfig = Field(default_factory=ModelMigrationConfig)
    process_manager: ProcessManagerConfig = Field(default_factory=ProcessManagerConfig)
    auto_serve: AutoServeConfig = Field(default_factory=AutoServeConfig)
    long_form: LongFormConfig = Field(default_factory=LongFormConfig)
    create: CreateConfig = Field(default_factory=CreateConfig)
    pro: ProConfig = Field(default_factory=ProConfig)

    @model_validator(mode="before")
    @classmethod
    def replace_none_sections(cls, data: dict) -> dict:
        """YAML で値なしセクション（例: 'llama:' のみ）が None になる問題を解決"""
        if isinstance(data, dict):
            for key in data:
                if data[key] is None:
                    data[key] = {}
        return data

    @model_validator(mode="after")
    def validate_ports_are_unique(self) -> "EvorefConfig":
        """常に起動するサーバ (backend / Web UI / base / 埋め込み) のポートが重ならない。

        重なると後から起動した方が bind に失敗し、起動の途中で原因の分かりにくい失敗になる。
        rerank は起動条件があるので :meth:`validate_rerank_port_is_unique` が別に見る。
        """
        ports = {
            "server.port": self.server.port,
            "server.frontend_port": self.server.frontend_port,
            "llama.port": self.llama.port,
            "embedding.llama_port": self.embedding.llama_port,
        }
        by_port: dict[int, list[str]] = {}
        for key, port in ports.items():
            by_port.setdefault(port, []).append(key)
        clash = [f"{' / '.join(keys)} ({port})" for port, keys in by_port.items() if len(keys) > 1]
        if clash:
            raise ValueError(f"ポートが重複しています: {', '.join(clash)}")
        return self

    @model_validator(mode="after")
    def validate_rerank_port_is_unique(self) -> "EvorefConfig":
        """rerank (off 以外・モデル設定済み) のポートが他のサーバのポートと重ならない (c_16 §7.2.1)

        モデル未設定なら起動しないので見ない (既定 on でモデルを置いていない既存の config を拒否しない)。
        """
        if self.rag.rerank.mode == "off" or not self.model_paths.rerank_model:
            return self
        others = {
            "llama.port": self.llama.port,
            "embedding.llama_port": self.embedding.llama_port,
            "server.port": self.server.port,
            "server.frontend_port": self.server.frontend_port,
        }
        clash = [key for key, port in others.items() if port == self.rag.rerank.port]
        if clash:
            raise ValueError(
                f"rag.rerank.port ({self.rag.rerank.port}) は {', '.join(clash)} と重複しています",
            )
        return self

    @model_validator(mode="after")
    def warn_working_memory_vs_context(self) -> "EvorefConfig":
        """会話窓 (memory) とコンテキスト窓 (llama) の食い違いを warning で通知する。

        ``memory.working_max_tokens`` は **プロンプトに載る会話履歴の上限**、
        ``llama.context_size`` は **モデルが受け取れる総量**。前者は後者に対する
        割合で決めるべきなのに、両者は別セクションの独立した定数なので、
        片方だけ動かすと黙って不整合になる。実際に既定同士が食い違っていた
        (context 8192 に対し会話窓 2048 = 25%)。

        - 低すぎる: 窓の半分も使わないうちに押し出しが始まり、そのたびに
          接頭辞 KV キャッシュが無効化されて全履歴が再 prefill される。
        - 高すぎる: 動的ブロック (few-shot / 参考情報 / 添付) と生成予約を
          押し出し、``build_messages`` が毎ターン切り詰めることになる。

        エラーにはしない (用途によっては意図的に狭めたい)。
        """
        from backend.log_config import get_logger

        raw_context = getattr(self.llama, "context_size", 0)
        # ``auto`` / null は起動時に決まる (c_16 §7.2.3) ので、ここでは比べない
        context = raw_context if isinstance(raw_context, int) else 0
        raw_window = self.memory.working_max_tokens
        if context <= 0 or not isinstance(raw_window, int):
            # ``auto`` の窓は context_size から導くので比は常に推奨の内側
            return self
        window = int(raw_window)
        share = window / context
        if share < _WORKING_WINDOW_MIN_SHARE or share > _WORKING_WINDOW_MAX_SHARE:
            get_logger("config").warning(
                "memory.working_max_tokens=%d is %.0f%% of llama.context_size=%d "
                "(recommended %.0f-%.0f%%, i.e. %d-%d). Too small wastes the "
                "window and forces frequent prefix-cache invalidation; too large "
                "squeezes the dynamic blocks and the generation reserve.",
                window, share * 100, context,
                _WORKING_WINDOW_MIN_SHARE * 100, _WORKING_WINDOW_MAX_SHARE * 100,
                int(context * _WORKING_WINDOW_MIN_SHARE),
                int(context * _WORKING_WINDOW_MAX_SHARE),
            )
        return self

    @model_validator(mode="after")
    def warn_create_stage_budget_vs_turn_budget(self) -> "EvorefConfig":
        """予算の 3 層 (f_10 §3) のステージ予算がターン予算を圧迫していないか警告する。

        ``create.turn_timeout_sec`` (ターン予算、最終防衛線) と
        ``create.staged.total_timeout_sec`` (ステージ予算) は独立した設定値で、
        以前は ``agent.total_timeout`` (実設定 1800 秒) と
        ``create.staged.total_timeout_sec`` (実設定 7200 秒) の組で「内側が
        外側より 4 倍大きい」逆転が起きていた。実行時は
        ``_clamp_long_form_timeout`` / ``StagedCreateExecutor`` 側で
        ``create.turn_timeout_sec - 300`` を上限にクランプするため機能は壊れ
        ないが、設定した ``staged.total_timeout_sec`` が黙って縮められている
        ことに気付けるよう起動時に 1 回 warning を出す。エラーにはしない。
        """
        turn_budget = float(self.create.turn_timeout_sec)
        stage_budget = float(self.create.staged.total_timeout_sec)
        ceiling = turn_budget - 300.0
        if stage_budget > ceiling:
            logger.warning(
                "create.staged.total_timeout_sec=%.1f exceeds "
                "create.turn_timeout_sec(%.1f) - 300; it will be clamped to "
                "%.1f at runtime. Lower create.staged.total_timeout_sec or "
                "raise create.turn_timeout_sec to match your intent.",
                stage_budget, turn_budget, ceiling,
            )
        return self

    @model_validator(mode="after")
    def warn_pro_only_keys_in_free(self) -> "EvorefConfig":
        """Free エディションで Pro 限定キーが設定された場合に warning を出す

        Pro でしか効果のないキーが Free 環境で設定されると、ユーザーは設定が
        反映されていると誤解しやすい。本 validator はそれをエラーにせず
        warning ログ (英語固定) として通知する。同一キーは ``(edition, key)``
        単位でデデュープし、起動時の警告氾濫を防ぐ。

        判定対象:

        - ``widget_proxy.enabled = True``  (Pro Widget Proxy / 汎用 Web API プロキシ)
        - ``llama.speculative.enabled = True`` (self-speculative decoding)
        - ``pro.url_recall.team_profile_ids`` / ``pro.policy.pack_path`` が空でない
        - ``pro.hub.*`` を既定から変えた (チーム知識ハブ)
        - ``create.staged.profile == "v1"`` (staged の v1 工程は Pro 限定。v2 は Free でも使う)
        - 未定義トップレベルキー ``mode_models`` の ``enabled`` が真 (Pro ローカルモデル切替)

        過去存在した ``external_api.enabled`` / 補助タスク関連の判定対象は
        削除済。
        """
        # 遅延 import: ``backend.edition`` は ``log_config`` のみ依存するため
        # 循環は発生しないが、トップレベル import で初期化順序の罠を作らない
        # よう関数内 import に留める。
        from backend.edition import Edition, current_edition

        if current_edition() != Edition.FREE:
            return self

        triggers: list[tuple[str, bool]] = [
            ("widget_proxy.enabled", bool(self.widget_proxy.enabled)),
            (
                "pro.url_recall.team_profile_ids",
                bool(self.pro.url_recall.team_profile_ids),
            ),
            ("pro.policy.pack_path", bool(self.pro.policy.pack_path.strip())),
            # self-speculative decoding は Free の起動スクリプトが無効化する
            ("llama.speculative.enabled", bool(self.llama.speculative.enabled)),
            # staged の v1 工程は Pro 限定 (_staged_stage_base_enabled がゲート)。
            # Free は v2 だけを使うので、v1 を指定しても longform のまま無効に
            # なることを警告する (f_10 §11.3)。staged_enabled は intra-staged の
            # キルスイッチ (既定 True) で Pro signal ではないため対象にしない。
            (
                "create.staged.profile",
                self.create.pipeline == "staged" and self.create.staged.profile == "v1",
            ),
        ]
        # チーム知識ハブ (docs/p_10) は Free に実体が無い。雛形は既定値で書いてあるので、
        # 既定から変えたキーだけを明示設定とみなす。
        for name, field in type(self.pro.hub).model_fields.items():
            triggers.append((f"pro.hub.{name}", getattr(self.pro.hub, name) != field.get_default(call_default_factory=True)))

        # extra="allow" で透過する未定義トップレベルキー (Pro 拡張)。
        # 現状は ``mode_models`` のみ明示。新しい Pro 専用トップレベルキーが
        # 増えたらここに追加する。``enabled: false`` (Pro の雛形どおり) は何も効かせていない。
        extras = self.__pydantic_extra__ or {}
        mode_models = extras.get("mode_models")
        if isinstance(mode_models, dict):
            triggers.append(("mode_models", bool(mode_models.get("enabled"))))
        elif "mode_models" in extras:
            triggers.append(("mode_models", bool(mode_models)))

        for key, fired in triggers:
            if not fired:
                continue
            dedup_key = ("free", key)
            if dedup_key in _WARNED_PRO_KEYS:
                continue
            _WARNED_PRO_KEYS.add(dedup_key)
            logger.warning(
                "Pro-only config key '%s' is set under Free edition; "
                "the setting has no effect.",
                key,
            )
        return self


#: 新しい版の config で無視したキーの警告のデデュープ (同じ組は 1 回だけ出す)。
_WARNED_NEWER_KEYS: set[tuple[str, ...]] = set()


def _pop_loc(data: object, loc: tuple) -> bool:
    """``loc`` (検証エラーの位置。リストの添字を含む) のキーを ``data`` から取り除く。"""
    node = data
    for part in loc[:-1]:
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
            node = node[part]
        else:
            return False
    if isinstance(node, dict) and loc and loc[-1] in node:
        node.pop(loc[-1])
        return True
    return False


def _drop_keys_unknown_to_this_version(raw: dict, version: int) -> dict:
    """アプリより新しい版の config から、この版が知らないキーを取り除いた写しを返す (c_05 §7.6)。

    ダウングレードで起動不能にしない。取り除くのは未知キー (``extra_forbidden``) と
    この版の撤去キー (新しい版での意味を知らない) だけで、値・型の誤りは残して通常どおり拒否する。
    """
    import copy

    from pydantic import ValidationError

    from backend.config_normalize import G0_REMOVED_KEYS, G1_RETIRED_KEYS

    data = copy.deepcopy(raw)
    ignored: list[str] = []
    for dotted in (*G0_REMOVED_KEYS, *G1_RETIRED_KEYS):
        if _pop_loc(data, tuple(dotted)):
            ignored.append(".".join(dotted))
    for _ in range(20):
        try:
            EvorefConfig.model_validate(copy.deepcopy(data))
            break
        except ValidationError as e:
            locs = [tuple(err["loc"]) for err in e.errors() if err.get("type") == "extra_forbidden"]
        popped = [loc for loc in locs if _pop_loc(data, loc)]
        if not popped:
            break
        ignored.extend(".".join(str(p) for p in loc) for loc in popped)
    key = tuple(sorted(ignored))
    if ignored and key not in _WARNED_NEWER_KEYS:
        _WARNED_NEWER_KEYS.add(key)
        logger.warning(
            "config_version %d is newer than this evoref (%d); ignoring keys this version does not know: %s",
            version, CONFIG_VERSION, ", ".join(ignored),
        )
    return data


def validate_config(raw: dict) -> dict:
    """config dict を Pydantic スキーマで検証し、デフォルト値を補完した dict を返す

    G0 の撤去キー・旧名キーはここでは扱わない (``evoref config normalize`` が
    設定ファイルから落とす、:mod:`backend.config_normalize`)。``config_version`` が
    アプリより新しいときだけ、未知キーと撤去キーを WARNING を出して無視する (c_05 §7.6)。

    Args:
        raw: YAML から読み込んだ生の設定辞書

    Returns:
        バリデーション済みの正規化された設定辞書

    Raises:
        pydantic.ValidationError: 設定値が不正な場合
    """
    from backend.config_normalize import stored_config_version

    version = stored_config_version(raw) if isinstance(raw, dict) else None
    if version is not None and version > CONFIG_VERSION:
        raw = _drop_keys_unknown_to_this_version(raw, version)
    validated = EvorefConfig.model_validate(raw)
    logger.info("Config validation passed")
    return validated.model_dump()
