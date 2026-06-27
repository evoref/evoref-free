"""モデル・ローカル状態のパス系スキーマ"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelPathsConfig(BaseModel):
    """モデルファイルパス

    標準 3 モデル (base/assist/embed) を明示フィールドとして定義。
    カスタムモデル種を追加できるよう ``extra="allow"`` を維持する。
    """

    model_config = ConfigDict(extra="allow")

    base_model: str = "models/gemma-4-12b-it-qat-q4_0.gguf"
    assist_model: str = "models/gemma-4-E4B_q4_0-it.gguf"
    embed_model: str = "models/Qwen3-Embedding-0.6B-Q8_0.gguf"
    coding_model: str | None = Field(
        default=None,
        description="コーディングモード用 GGUF パス。未指定 (None / 空文字列) の場合は base_model にフォールバック",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_reranker_model(cls, data: dict) -> dict:
        """旧 ``reranker_model`` キーを明示的に拒否する

        リランカー機能の削除に伴い廃止。``extra="allow"`` のため黙って透過
        してしまうが、過去設定の残存に気付かせるため ``ValueError`` を上げる。
        """
        if isinstance(data, dict) and "reranker_model" in data:
            raise ValueError(
                "model_paths must not contain 'reranker_model' anymore. "
                "The reranker feature has been removed. Remove the "
                "'model_paths.reranker_model' line from config.yaml.",
            )
        return data


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
    # Level 2 base=C: control vector (llama-cvector-generator 生成) の本体 / 版管理 /
    # positive.txt・negative.txt 作業用ディレクトリ。Pro 限定だが PathResolver が
    # 参照するため Free でもフィールドは存在する (extra="forbid" のため明示必須)。
    control_vector_adapter: str = "local/models/control_vector.gguf"
    control_vector_versions_dir: str = "local/models/control_vector_versions/"
    cvector_work_dir: str = "local/cvector/"
    experience_assist_file: str = "local/experience_assist.json"
    eval_assist_file: str = "local/eval_assist.json"
    # アシスト purpose 別 timeout の反応的自己較正値 (model-keyed)。
    # AssistModelClient が ReadTimeout 観測から引き上げた天井を永続化する。
    assist_calibration_file: str = "local/assist_calibration.json"
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
    # staged コーディングパイプラインの一時ワークスペース
    # (spec.md / src / tests / manifest.json の工程間ハンドオフ)。
    coding_workspace_dir: str = "local/coding/"
    # base モデルの自己学習データを (base モデル識別子 × モード) でパーティション化
    # する際のルート。``learning_dir/<base_model_stem>/`` 配下に experience /
    # base prompts / base LoRA・cvector を配置する。PathResolver.resolve_learning が
    # 参照する (assist・共有データは従来どおり flat の local_paths を使う)。
    learning_dir: str = "local/learning/"
