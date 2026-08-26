"""モデル・ローカル状態のパス系スキーマ"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelPathsConfig(BaseModel):
    """モデルファイルパス

    標準 2 モデル (base/embed) を明示フィールドとして定義。
    カスタムモデル種を追加できるよう ``extra="allow"`` を維持する。
    """

    model_config = ConfigDict(extra="allow")

    base_model: str = "models/gemma-4-12b-it-qat-q4_0.gguf"
    embed_model: str = "models/Qwen3-Embedding-0.6B-Q8_0.gguf"
    create_model: str | None = Field(
        default=None,
        description="クリエイトモード用 GGUF パス。未指定 (None / 空文字列) の場合は base_model にフォールバック",
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

    @model_validator(mode="before")
    @classmethod
    def reject_removed_assist_models(cls, data: dict) -> dict:
        """撤去済みの専用アシストモデル系キーを明示的に拒否する

        専用アシストモデル (:8081) は撤去され、補助タスクはベースモデルの
        専有スロットへ集約された (2026-08-14)。``extra="allow"`` なのでこれらの
        キーは黙って透過し、「設定したのに効かない」状態に気付けないため
        ``ValueError`` を上げる。``aux_model`` は撤去時の一括置換で
        ``assist_model`` から機械的に生まれうる綴りなので併せて拒否する
        (補助タスクは専用モデルを持たないので、この名前のキーは常に誤り)。
        """
        if not isinstance(data, dict):
            return data
        removed = [
            k for k in ("assist_model", "assist_create_model", "aux_model")
            if k in data
        ]
        if removed:
            raise ValueError(
                f"model_paths must not contain {removed} anymore. The dedicated "
                "assist model has been removed; auxiliary tasks now run on the "
                "base model's background slot. Remove these lines from "
                "config.yaml (see docs/c_14_aux_client_protocol.md).",
            )
        return data


#: 機能ごと削除された ``local_paths`` キーと、その理由。
_REMOVED_LOCAL_PATH_KEYS: dict[str, str] = {
    "rag_judge_events_file": (
        "RAG necessity/quality decision recall was removed (the LLM judges "
        "that fed it are gone); necessity is decided by rules and quality by "
        "vector thresholds"
    ),
    "aux_experience_file": (
        "the aux prompt evolution loop was removed (its only remaining "
        "prompt had no experience producer, so evolution never ran); aux "
        "prompts are edited manually via /api/aux-prompts"
    ),
}


class LocalPathsConfig(BaseModel):
    """ローカルパス（読み書き、PC 個別保存）

    PathResolver (``backend/config.py``) が利用するキーは全て本モデルに明示
    定義されているため、未定義キーは typo とみなして ``extra="forbid"`` で
    拒否する。
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_removed_keys(cls, data: object) -> object:
        """機能ごと削除されたキーを理由付きで拒否する。

        ``extra="forbid"`` の素のエラーだと「どのキーをなぜ消すのか」が
        伝わらないため、``reject_legacy_reranker_section`` と同じ形で示す。
        """
        if isinstance(data, dict):
            for key, reason in _REMOVED_LOCAL_PATH_KEYS.items():
                if key in data:
                    raise ValueError(
                        f"local_paths.{key} was removed: {reason}. "
                        "Remove the line from config.yaml.",
                    )
        return data

    lora_adapter: str = Field(
        default="local/models/adapter.gguf",
        description="base モデル用 LoRA アダプタ (Level 2 base=lora)。存在時のみ起動時に"
        "付与 (arch 不一致なら scripts/launch_llama.py が fail-closed でスキップする)。",
    )
    lora_versions_dir: str = Field(
        default="local/lora_versions/",
        description="base LoRA のバージョン履歴保存先。",
    )
    lora_spsa_checkpoint: str = Field(
        default="local/models/lora_spsa_checkpoint.json",
        description="base LoRA SPSA 学習の中断/再開チェックポイント。",
    )
    # Level 2 base=C: control vector (llama-cvector-generator 生成) の本体 / 版管理 /
    # positive.txt・negative.txt 作業用ディレクトリ。Pro 限定だが PathResolver が
    # 参照するため Free でもフィールドは存在する (extra="forbid" のため明示必須)。
    control_vector_adapter: str = "local/models/control_vector.gguf"
    control_vector_versions_dir: str = "local/models/control_vector_versions/"
    cvector_work_dir: str = "local/cvector/"
    # 補助タスク (rag_necessity / rag_quality / tool_call / note_evolve) の
    # 経験バッファとプロンプト。Level 1 Phase 2 の進化が読む。partition 有効時は
    # base モデルパーティション配下へ rebase される (PathResolver._LEARNING_SUBPATH)。
    aux_prompts_dir: str = "local/aux_prompts/"
    # 補助タスク purpose 別 timeout の反応的自己較正値 (model-keyed)。
    # AuxClient がタイムアウト観測から引き上げた天井を永続化する。
    aux_calibration_file: str = "local/aux_calibration.json"
    lora_archive_dir: str = "local/lora_archive/"
    embed_lora_adapter: str = Field(
        default="local/models/embed_adapter.gguf",
        description="embed モデル用 LoRA アダプタ。"
        "migrate_component() は新モデルと arch が不一致のときのみ自動アーカイブする。",
    )
    embed_lora_versions_dir: str = Field(
        default="local/models/embed_lora_versions/",
        description="embed LoRA のバージョン履歴保存先。",
    )
    vectors_dir: str = "local/vectors/"
    knowledge_dir: str = "local/knowledge/"
    experience_file: str = "local/experience.json"
    eval_core_file: str = "local/eval_core.json"
    model_state_file: str = "local/model_state.json"
    model_quality_file: str = Field(
        default="local/model_quality.json",
        description=(
            "モデル切替時の出力品質プローブ結果。役割 (base/embedding) ごとに"
            "「どのモデルを最後に検査したか」を記録し、変わったときだけ再検査する。"
        ),
    )
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
    # staged クリエイトパイプラインの一時ワークスペース
    # (spec.md / src / tests / manifest.json の工程間ハンドオフ)。
    create_workspace_dir: str = "local/create/"
    # base モデルの自己学習データを (base モデル識別子 × モード) でパーティション化
    # する際のルート。``learning_dir/<base_model_stem>/`` 配下に experience /
    # base prompts / base LoRA・cvector を配置する。PathResolver.resolve_learning が
    # 参照する (共有データは従来どおり flat の local_paths を使う)。
    learning_dir: str = "local/learning/"
