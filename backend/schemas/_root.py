"""evoref 全体設定スキーマのルート

config.yaml の全セクションを集約する ``EvorefConfig`` と、
YAML を辞書として正規化する ``validate_config`` を提供する。
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.log_config import get_logger
from backend.schemas._common import (
    AgentConfig,
    CLIConfig,
    EditorSettingsConfig,
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
from backend.schemas.loop import LoopConfig
from backend.schemas.memory import MemoryConfig
from backend.schemas.paths import LocalPathsConfig, ModelPathsConfig
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


class EvorefConfig(BaseModel):
    """evoref 全体設定スキーマ

    config.yaml の全セクションを型安全に定義する。
    未知のトップレベルキーは許可（Pro 拡張等に対応）。
    """

    model_config = ConfigDict(extra="allow")

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
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    widget_proxy: WidgetProxyConfig = Field(default_factory=WidgetProxyConfig)
    i18n: I18nConfig = Field(default_factory=I18nConfig)
    modes: ModesConfig = Field(default_factory=ModesConfig)
    editor: EditorSettingsConfig = Field(default_factory=EditorSettingsConfig)
    model_migration: ModelMigrationConfig = Field(default_factory=ModelMigrationConfig)
    process_manager: ProcessManagerConfig = Field(default_factory=ProcessManagerConfig)
    auto_serve: AutoServeConfig = Field(default_factory=AutoServeConfig)
    long_form: LongFormConfig = Field(default_factory=LongFormConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
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

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_debug_section(cls, data: dict) -> dict:
        """``debug:`` セクションを起動時に明示的に拒否する

        ログ系設定は CLI フラグ ``--develop=<level>`` (``debug`` /
        ``investigate`` / ``evolve``) の SSOT に集約されたため、
        ``config.yaml`` の ``debug:`` セクションは廃止された。

        ``EvorefConfig.model_config = ConfigDict(extra="allow")`` のため
        トップレベル extra フィールドはエラーにならず黙って透過するが、
        ``debug:`` だけは過去設定の残存に気付かせるため明示的に
        ``ValueError`` を上げる。
        """
        if isinstance(data, dict) and "debug" in data:
            raise ValueError(
                "config.yaml must not contain a 'debug:' section anymore "
                ". Develop logging is controlled exclusively via "
                "the --develop=<level> CLI flag (debug | investigate | evolve). "
                "Remove the 'debug:' section from config.yaml.",
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_reranker_section(cls, data: dict) -> dict:
        """``reranker:`` セクションを起動時に明示的に拒否する

        リランカー機能は削除され、検索フローは fetch_multiplier 拡張 +
        score_normalization に一本化された。``EvorefConfig`` は
        ``extra="allow"`` のため残存セクションは黙って透過してしまうが、
        過去設定の残存に気付かせるため明示的に ``ValueError`` を上げる。
        """
        if isinstance(data, dict) and "reranker" in data:
            raise ValueError(
                "config.yaml must not contain a 'reranker:' section anymore. "
                "The reranker feature has been removed; retrieval now uses "
                "rag.fetch_multiplier and rag.score_normalization instead. "
                "Remove the 'reranker:' section (and the "
                "'model_paths.reranker_model' line if present) from "
                "config.yaml.",
            )
        return data

    @model_validator(mode="after")
    def warn_pro_only_keys_in_free(self) -> "EvorefConfig":
        """Free エディションで Pro 限定キーが設定された場合に warning を出す

        Pro でしか効果のないキーが Free 環境で設定されると、ユーザーは設定が
        反映されていると誤解しやすい。本 validator はそれをエラーにせず
        warning ログ (英語固定) として通知する。同一キーは ``(edition, key)``
        単位でデデュープし、起動時の警告氾濫を防ぐ。

        判定対象:

        - ``widget_proxy.enabled = True``  (Pro Widget Proxy / 汎用 Web API プロキシ)
        - ``pro.terminal.enabled = True``  (Pro Web ターミナル)
        - ``learning.optimizer == "full-cma-es"`` (Pro CMA-ES オプティマイザ)
        - ``create.pipeline == "staged"`` (Pro staged クリエイトパイプライン)
        - 未定義トップレベルキー ``mode_models`` (Pro ローカルモデル切替)

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
            ("pro.terminal.enabled", bool(self.pro.terminal.enabled)),
            (
                "learning.optimizer",
                self.learning.optimizer == "full-cma-es",
            ),
            (
                "learning.level2_base_method",
                self.learning.level2_base_method == "cvector",
            ),
            (
                "pro.url_recall.team_profile_ids",
                bool(self.pro.url_recall.team_profile_ids),
            ),
            # staged クリエイトパイプラインは Pro 限定 (_staged_create_enabled が
            # is_pro() でゲート)。Free で pipeline=staged を設定しても longform の
            # まま無効なので警告する。staged_enabled は intra-staged のキルスイッチ
            # (既定 True) で Pro signal ではないため対象にしない。
            ("create.pipeline", self.create.pipeline == "staged"),
        ]

        # extra="allow" で透過する未定義トップレベルキー (Pro 拡張)。
        # 現状は ``mode_models`` のみ明示。新しい Pro 専用トップレベルキーが
        # 増えたらここに追加する。
        extras = self.__pydantic_extra__ or {}
        if "mode_models" in extras:
            triggers.append(("mode_models", True))

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


#: 旧 "coding" モード名時代のキー → 現行キー。``LocalPathsConfig`` /
#: ``ModelPathsConfig`` 等が ``extra="forbid"`` のため、旧キーの残る config.yaml は
#: そのままでは起動時に ValidationError になる。ファイルは書き換えず、読み込んだ
#: dict の上でだけ読み替えて WARNING を出す (更新はユーザーの任意)。
_LEGACY_KEY_RENAMES: dict[str, str] = {
    "coding": "create",
    "coding_task": "create_task",
    "coding_model": "create_model",
    "coding_workspace_dir": "create_workspace_dir",
    "coding_budget_tokens": "create_budget_tokens",
    "coding_code_signal": "create_code_signal",
}


def _rename_legacy_keys(node: object, path: str, renamed: list[str]) -> object:
    """設定ツリーを再帰的に走査し、旧モード名のキーを現行キーへ読み替える。"""
    if isinstance(node, dict):
        out: dict = {}
        for key, value in node.items():
            new_key = _LEGACY_KEY_RENAMES.get(key, key) if isinstance(key, str) else key
            if new_key != key:
                renamed.append(f"{path}{key} -> {new_key}")
            out[new_key] = _rename_legacy_keys(value, f"{path}{new_key}.", renamed)
        return out
    if isinstance(node, list):
        return [_rename_legacy_keys(v, path, renamed) for v in node]
    return node


def validate_config(raw: dict) -> dict:
    """config dict を Pydantic スキーマで検証し、デフォルト値を補完した dict を返す

    旧 "coding" モード名時代のキーは現行キーへ読み替えてから検証する
    (:data:`_LEGACY_KEY_RENAMES`)。

    Args:
        raw: YAML から読み込んだ生の設定辞書

    Returns:
        バリデーション済みの正規化された設定辞書

    Raises:
        pydantic.ValidationError: 設定値が不正な場合
    """
    renamed: list[str] = []
    raw = _rename_legacy_keys(raw, "", renamed)  # type: ignore[assignment]
    if renamed:
        logger.warning(
            "config.yaml uses legacy 'coding' mode keys; reading them as the "
            "current names for this run. Update config.yaml to silence this: %s",
            ", ".join(renamed),
        )
    validated = EvorefConfig.model_validate(raw)
    logger.info("Config validation passed")
    return validated.model_dump()
