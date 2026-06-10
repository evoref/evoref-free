"""config.yaml の Pydantic バリデーションスキーマ群


参照箇所は ``from backend.schemas import <Config>`` の形でアクセスする。
個別のサブモジュール (``backend.schemas.llm`` 等) を直接 import しても
よいが、再エクスポート経由での参照を推奨する。

ファイル配置:

- ``_root.py``  : ``EvorefConfig`` / ``validate_config``
- ``_common.py``: 単発の細かい Config (Instance / Server / Runtime / Theme /
  CLI / History / Streaming / Agent / Tools / WidgetProxy / ExternalApi /
  I18n / Debug / LongForm / Modes / Editor / ProcessManager / Terminal /
  Pro / ModelMigration)
- ``llm.py``    : ``LlamaConfig`` / ``LlamaSpeculativeConfig``
- ``rag.py``    : RAG / Embedding / Self-RAG / Cartridge 系
- ``memory.py`` : EvorefMem 関連 + ``VALID_FACT_TYPES``
- ``learning.py``: 学習サイクル + ``ScheduleConfig``
- ``loop.py``   : 自律ループ driver
- ``assist_model.py``: アシストモデル設定
- ``paths.py``  : ``ModelPathsConfig`` / ``LocalPathsConfig``
"""

from backend.schemas._common import (
    AgentConfig,
    ChatModeConfig,
    CLIConfig,
    CLITimeoutsConfig,
    CodingModeConfig,
    EditorSettingsConfig,
    HistoryConfig,
    I18nConfig,
    InstanceConfig,
    LongFormConfig,
    ModelMigrationConfig,
    ModesConfig,
    ProcessManagerConfig,
    ProConfig,
    RuntimeConfig,
    ServerConfig,
    StreamingConfig,
    TerminalConfig,
    ThemeConfig,
    ToolsConfig,
    WidgetApiConfig,
    WidgetProxyConfig,
)
from backend.schemas._root import EvorefConfig, validate_config
from backend.schemas.assist_model import (
    AssistConcurrencyConfig,
    AssistModelConfig,
    AssistModelLocalConfig,
)
from backend.schemas.learning import (
    FeedbackPipeConfig,
    LearningConfig,
    LearningPolicyConfig,
    ScheduleConfig,
)
from backend.schemas.llm import LlamaConfig, LlamaSpeculativeConfig
from backend.schemas.loop import (
    LoopConfig,
    LoopQualityGatesConfig,
    LoopSandboxConfig,
)
from backend.schemas.memory import (
    VALID_FACT_TYPES,
    ConflictResolverConfig,
    FactsConfig,
    FactsExtractionMaxPerSession,
    InjectionConfig,
    InjectionTierRatios,
    MemoryConfig,
    NoteEvolverConfig,
    PinConfig,
    PrivateSessionConfig,
    SemMemConflictConfig,
    SemMemLimitsConfig,
    SemMemProjectConfig,
    SubjectDictionaryConfig,
)
from backend.schemas.paths import LocalPathsConfig, ModelPathsConfig
from backend.schemas.rag import (
    CartridgeGateConfig,
    ClusterIndexConfig,
    ContextualPrefixConfig,
    EmbeddingConfig,
    RAGConfig,
    SelfRagAssistJudgeConfig,
    SelfRagConfig,
)

__all__ = [
    "VALID_FACT_TYPES",
    "AgentConfig",
    "AssistConcurrencyConfig",
    "AssistModelConfig",
    "AssistModelLocalConfig",
    "CLIConfig",
    "CLITimeoutsConfig",
    "CartridgeGateConfig",
    "ChatModeConfig",
    "ClusterIndexConfig",
    "CodingModeConfig",
    "ConflictResolverConfig",
    "ContextualPrefixConfig",
    "EditorSettingsConfig",
    "EmbeddingConfig",
    "EvorefConfig",
    "FactsConfig",
    "FactsExtractionMaxPerSession",
    "FeedbackPipeConfig",
    "HistoryConfig",
    "I18nConfig",
    "InjectionConfig",
    "InjectionTierRatios",
    "InstanceConfig",
    "LearningConfig",
    "LearningPolicyConfig",
    "LlamaConfig",
    "LlamaSpeculativeConfig",
    "LocalPathsConfig",
    "LongFormConfig",
    "LoopConfig",
    "LoopQualityGatesConfig",
    "LoopSandboxConfig",
    "MemoryConfig",
    "ModelMigrationConfig",
    "ModelPathsConfig",
    "ModesConfig",
    "NoteEvolverConfig",
    "PinConfig",
    "PrivateSessionConfig",
    "ProConfig",
    "ProcessManagerConfig",
    "RAGConfig",
    "RuntimeConfig",
    "ScheduleConfig",
    "SelfRagAssistJudgeConfig",
    "SelfRagConfig",
    "SemMemConflictConfig",
    "SemMemLimitsConfig",
    "SemMemProjectConfig",
    "ServerConfig",
    "StreamingConfig",
    "SubjectDictionaryConfig",
    "TerminalConfig",
    "ThemeConfig",
    "ToolsConfig",
    "WidgetApiConfig",
    "WidgetProxyConfig",
    "validate_config",
]
