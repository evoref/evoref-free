"""モデル・ローカル状態のパス系スキーマ"""

from pydantic import BaseModel, ConfigDict, Field


class ModelPathsConfig(BaseModel):
    """モデルファイルパス

    標準 4 モデル (base/assist/embed/reranker) を明示フィールドとして定義。
    カスタムモデル種を追加できるよう ``extra="allow"`` を維持する。
    """

    model_config = ConfigDict(extra="allow")

    base_model: str = "models/qwen3.5-4b-q4_k_m.gguf"
    assist_model: str = "models/Qwen3.5-4B-Q4_K_M.gguf"
    embed_model: str = "models/Qwen3-Embedding-0.6B-Q8_0.gguf"
    reranker_model: str = "models/Qwen3-Reranker-4B.Q5_K_M.gguf"
    coding_model: str | None = Field(
        default=None,
        description="コーディングモード用 GGUF パス。未指定 (None / 空文字列) の場合は base_model にフォールバック",
    )


class LocalPathsConfig(BaseModel):
    """ローカルパス（読み書き、PC 個別保存）

    PathResolver (``backend/config.py``) が利用するキーは全て本モデルに明示
    定義されているため、未定義キーは typo とみなして ``extra="forbid"`` で
    拒否する。
    """

    model_config = ConfigDict(extra="forbid")

    lora_adapter: str = "local/models/adapter.gguf"
    lora_versions_dir: str = "local/lora_versions/"
    assist_lora_adapter: str = "local/models/assist_adapter.gguf"
    assist_lora_versions_dir: str = "local/models/assist_lora_versions/"
    experience_assist_file: str = "local/experience_assist.json"
    eval_assist_file: str = "local/eval_assist.json"
    lora_archive_dir: str = "local/lora_archive/"
    embed_lora_adapter: str = "local/models/embed_adapter.gguf"
    embed_lora_versions_dir: str = "local/models/embed_lora_versions/"
    vectors_dir: str = "local/vectors/"
    knowledge_dir: str = "local/knowledge/"
    experience_file: str = "local/experience.json"
    eval_core_file: str = "local/eval_core.json"
    model_state_file: str = "local/model_state.json"
    # EvorefMem ローカル状態
    local_state_file: str = "local/state.json"
    memory_dir: str = "local/memory/"
    prompts_dir: str = "local/prompts/"
    cartridges_dir: str = "local/cartridges/"
    history_dir: str = "local/history/"
    learned_patterns_file: str = "local/learned_patterns.json"
    themes_dir: str = "local/themes/"
    # SchemaMigrator バックアップ / 旧 prompts 退避先
    migration_archive_dir: str = "local/migration_archive/"
    # EvorefMem トリガ辞書 (pin / fact / classify) の user override 配置先。
    # 同梱 default は ``backend/free/memory/_defaults/triggers/``。
    triggers_dir: str = "local/triggers/"
