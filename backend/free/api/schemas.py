"""Pydantic リクエスト/レスポンスモデル"""

from pydantic import BaseModel, Field

from backend.free.__version__ import (
    __schema_version__ as _FREE_SCHEMA_VERSION,
    __version__ as _FREE_VERSION,
)


# ===== Chat =====

class FileContext(BaseModel):
    """チャットコンテキストに注入するファイル情報"""
    filename: str
    chunks: list[str]


class ChatRequest(BaseModel):
    message: str
    mode: str = "chat"
    session_id: str | None = None
    file_contexts: list[FileContext] = Field(default_factory=list)
    stream: bool = True
    # プライベートセッション。``True`` のターンは memory_only
    # で動作し、LTM/SemMem/履歴ディスク永続化に書き込まない。
    private: bool = False


class TokenInfo(BaseModel):
    used: int
    limit: int
    pct: int
    instance_name: str


class ChatResponse(BaseModel):
    response: str
    token_info: TokenInfo
    session_id: str
    agent_layer: str = "reactive"


class CancelRequest(BaseModel):
    session_id: str


class CancelResponse(BaseModel):
    cancelled: bool
    tokens_generated: int = 0


# ===== Status =====

class LlamaServerInfo(BaseModel):
    connected: bool
    host: str
    port: int


class ModelInfo(BaseModel):
    name: str | None = None
    chat_template: str | None = None
    has_system_role: bool = True
    context_size: int = 4096


class MemoryStats(BaseModel):
    working_turns: int = 0
    short_term_notes: int = 0
    long_term_chunks: int = 0


class ComponentStatus(BaseModel):
    """個別コンポーネント（モデル）のステータス"""
    name: str = ""
    connected: bool = False


class LearningBriefStatus(BaseModel):
    """学習サイクルの概要ステータス（デバッグオーバーレイ用）"""
    running: bool = False
    experience_count: int = 0
    conditions_met: bool = False


class DebugStatusInfo(BaseModel):
    """デバッグセクションの詳細情報"""
    enabled: bool = False
    log_dir: str = ""
    disk_usage_mb: float = 0.0
    recent_errors_count: int = 0
    cache_hit_rate: float = 0.0
    last_ttft_ms: float | None = None
    last_tok_per_sec: float | None = None
    learning: LearningBriefStatus = Field(
        default_factory=LearningBriefStatus,
    )


class CapabilityInfo(BaseModel):
    """モデル能力プローブ結果 (docs/c_15_model_capability_adaptation.md)。

    起動/切替時に実機へカナリアを投げて観測した能力。未プローブ / プローブ無効時は
    ``probed=False`` で観測フィールドは ``None``。``probe_divergence`` は宣言と
    実機の食い違い (例: json_schema grammar 非強制 / <think> 未閉じ) の記録。
    """
    slot: str = ""  # "base" | "assist"
    model_id: str = ""
    probed: bool = False
    effective_reasoning_mode: str | None = None
    reasoning_separated: bool | None = None
    emits_think_tags: bool | None = None
    closes_think_tags: bool | None = None
    json_schema_enforced: bool | None = None
    needs_lenient_json: bool = False
    probe_divergence: list[str] = Field(default_factory=list)
    probed_at: str = ""


class StatusResponse(BaseModel):
    status: str = "ok"
    edition: str = "free"
    instance_name: str = "evoref"
    version: str = _FREE_VERSION
    free_version: str = _FREE_VERSION
    pro_version: str | None = None
    schema_version: int = _FREE_SCHEMA_VERSION
    uptime_seconds: float = 0
    llama_server: LlamaServerInfo
    model: ModelInfo | None = None
    components: list[ComponentStatus] = Field(default_factory=list)
    memory: MemoryStats = Field(default_factory=MemoryStats)
    cartridges_loaded: int = 0
    debug: DebugStatusInfo = Field(default_factory=DebugStatusInfo)
    capabilities: list[CapabilityInfo] = Field(default_factory=list)


# ===== Assist Model =====

class AssistModelConcurrency(BaseModel):
    """アシストモデル用途別セマフォスロット数"""
    realtime: int = 0
    background: int = 0
    learning: int = 0


class AssistModelStatusResponse(BaseModel):
    """アシストモデルのステータス（Free版: ローカル専用）"""
    configured: bool = False
    connected: bool = False
    url: str = ""
    host: str = ""
    port: int = 0
    model_params_b: float | None = None
    concurrency: AssistModelConcurrency = Field(default_factory=AssistModelConcurrency)
    timeout_seconds: float = 0


# ===== RAG =====

class RagIngestResponse(BaseModel):
    source: str
    chunks_created: int
    tokens_total: int
    ingest_time_sec: float


class RagSourceInfo(BaseModel):
    filename: str
    chunks: int
    added_at: str
    category: str


class RagStatsResponse(BaseModel):
    total_chunks: int
    total_vectors: int
    total_sources: int
    index_size_mb: float
    embedding_dim: int
    embedding_dim_stored: int | None = None
    embedding_dim_mismatch: bool = False
    chunking_strategy: str
    hybrid_search: bool
    fusion_method: str
    created_at: str | None = None
    last_reindex_at: str | None = None
    embedding_model: str | None = None
    embedding_backend: str | None = None
    sources: list[RagSourceInfo] = Field(default_factory=list)


# ===== Memory =====

class WorkingMemoryStats(BaseModel):
    turns: int
    max_turns: int
    tokens_used: int
    max_tokens: int
    session_id: str


class ShortTermMemoryStats(BaseModel):
    notes: int
    max_notes: int
    pending_embeddings: int
    pending_evolution: int
    avg_lightmem_score: float


class LongTermMemoryStats(BaseModel):
    chunks: int
    index_size_mb: float
    sources: int


class FadeMemStats(BaseModel):
    alpha: float
    beta: float
    gamma: float
    threshold: float


class SemanticMemoryScopeStats(BaseModel):
    """SemanticFactStore 1 スコープあたりの集計"""
    scope: str
    total: int = 0
    active: int = 0
    superseded: int = 0
    pinned: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    by_mode_origin: dict[str, int] = Field(default_factory=dict)


class SemanticMemoryStats(BaseModel):
    """SemMem 全体サマリ

    現在ロード済みの SemanticFactStore (global + 既知の project) を集計する。
    未ロードのプロジェクトは含まれない (lazy 設計)。
    """
    scopes: list[SemanticMemoryScopeStats] = Field(default_factory=list)
    total_facts: int = 0
    total_pinned: int = 0


class MemoryDetailedStats(BaseModel):
    working: WorkingMemoryStats
    short_term: ShortTermMemoryStats
    long_term: LongTermMemoryStats
    fadem: FadeMemStats
    semantic: SemanticMemoryStats = Field(default_factory=SemanticMemoryStats)
    current_mode: str = "chat"


class NoteInfo(BaseModel):
    id: str
    content: str
    keywords: list[str]
    tags: list[str]
    lightmem_score: float
    created_at: float
    accessed_at: float
    access_count: int
    session_id: str
    context_description: str
    evolution_pending: bool
    has_embedding: bool


class MemoryNotesResponse(BaseModel):
    total: int
    notes: list[NoteInfo]


# ===== Pin =====

class PinFactRequest(BaseModel):
    """Pin リクエスト

    既存ファクトを ID で指定して pin する場合は ``fact_id`` を渡す。
    新規ファクトを作って pin する場合は ``content`` (および任意の
    ``predicate`` / ``object_``) を渡し、サーバ側で SemanticFact を生成する。
    """

    scope: str = "global"
    fact_id: str | None = None
    content: str | None = None
    subject: str = "user.pinned"
    predicate: str = "remember"
    object_: str | None = Field(default=None, alias="object")
    type: str = "personal_fact"
    mode_origin: str = "chat"
    lock_duration_s: float | None = None

    model_config = {"populate_by_name": True}


class PinnedFactInfo(BaseModel):
    """Pinned ファクトの一覧表示用 DTO"""

    id: str
    subject: str
    predicate: str
    object: str
    type: str
    scope: str
    confidence: float
    pinned: bool
    pin_locked_until: float | None
    mode_origin: str
    created_at: float
    accessed_at: float
    access_count: int


class PinFactResponse(BaseModel):
    fact: PinnedFactInfo


class UnpinFactRequest(BaseModel):
    fact_id: str
    scope: str = "global"
    force: bool = False


class UnpinFactResponse(BaseModel):
    fact: PinnedFactInfo


class PinnedFactsResponse(BaseModel):
    scope: str
    total: int
    facts: list[PinnedFactInfo]


# ===== Model =====

class ModelDetailResponse(BaseModel):
    chat_template: str | None = None
    has_system_role: bool = True
    context_size: int = 4096
    gpu_layers: int = 0
    flash_attn: bool = False


# ===== Model Migration =====

class MigrateRequest(BaseModel):
    new_model_path: str
    try_lora: bool = False
    regenerate_context: bool = False
    dry_run: bool = False


class MigrateDataSummary(BaseModel):
    memory_notes: int = 0
    experience_entries: int = 0
    perplexity_reset: int = 0
    rag_chunks: int = 0
    cartridges: int = 0
    prompts_modes: list[str] = Field(default_factory=list)


class MigrateCalibration(BaseModel):
    eval_questions: int = 0
    baseline_ppl_avg: float = 0.0


class MigrateResponse(BaseModel):
    dry_run: bool
    old_model: str
    new_model: str
    lora_action: str
    data_summary: MigrateDataSummary
    calibration: MigrateCalibration | None = None
    recommendations: list[str] = Field(default_factory=list)


class MigrationHistoryItem(BaseModel):
    from_model: str
    to_model: str
    migrated_at: str
    lora_archived: bool


class MigrationHistoryResponse(BaseModel):
    current_model: str
    lora_available: bool
    history: list[MigrationHistoryItem] = Field(default_factory=list)


class RollbackRequest(BaseModel):
    target_model: str | None = None


class RollbackResponse(BaseModel):
    rolled_back_to: str
    lora_restored: bool


# --- Component (assist / embedding) migration ---

class ComponentMigrateRequest(BaseModel):
    """assist / embedding モデルの切替リクエスト"""
    new_model_path: str
    dry_run: bool = False
    # L2: 既定で auto_restart=True。LlamaProcessManager が当該
    # コンポーネントを管理している場合のみ実プロセス再起動を試みる。
    auto_restart: bool = True


class ComponentMigrateResponse(BaseModel):
    component: str
    dry_run: bool
    old_model: str
    new_model: str
    # 新モデルと既存 LoRA の arch 整合性判定結果 ("kept" / "archived" / "n/a")。
    # base の MigrateResponse.lora_action に相当。
    lora_action: str = "n/a"
    # L2: 自動再起動 + クライアント差し替えが成功したか
    restarted: bool = False
    recommendations: list[str] = Field(default_factory=list)


class ComponentRollbackRequest(BaseModel):
    target_model: str | None = None


class ComponentRollbackResponse(BaseModel):
    component: str
    rolled_back_to: str
    # base の RollbackResponse.lora_restored に相当。アーカイブされていた
    # LoRA を実際に復元できたか。
    lora_restored: bool = False


class ComponentMigrationHistoryItem(BaseModel):
    from_model: str
    to_model: str
    migrated_at: str


class ComponentMigrationHistoryResponse(BaseModel):
    component: str
    current_model: str
    history: list[ComponentMigrationHistoryItem] = Field(default_factory=list)


class ReloadResponse(BaseModel):
    reloaded: bool
    model_id: str = ""
    chat_template: str = ""
    has_system_role: bool = True


class ModelStateResponse(BaseModel):
    """`GET /api/model/state` レスポンス

    `model_state.json` と `config.yaml.model_paths.base_model` の整合性を
    UI/CLI から検査するためのスナップショット。`config_mismatch=True` の
    場合、`recommendation` を表示して `/migrate-model` 実行を促す。
    """

    current_filename: str = ""
    config_filename: str = ""
    config_base_model: str = ""
    config_mismatch: bool = False
    lora_compatible: bool = True
    strict_startup_check: bool = False
    recommendation: str = ""


# ===== Config =====

class LocaleRequest(BaseModel):
    locale: str


class LocaleResponse(BaseModel):
    locale: str


class LocalesResponse(BaseModel):
    locales: list[str]
    current: str


class ConfigFullResponse(BaseModel):
    """全設定レスポンス"""
    config: dict
    sections: list[str]
    edition: str


class ConfigUpdateRequest(BaseModel):
    """設定セクション更新リクエスト"""
    data: dict


class ConfigUpdateResponse(BaseModel):
    """設定セクション更新レスポンス"""
    section: str
    updated: bool


class ConfigValidateResponse(BaseModel):
    """設定バリデーション結果"""
    section: str
    valid: bool
    errors: list[str] = Field(default_factory=list)


class PresetInfo(BaseModel):
    """パフォーマンスプリセットの 1 件"""
    id: str


class PresetListResponse(BaseModel):
    """プリセット一覧 + 現在一致するプリセット"""
    presets: list[PresetInfo] = Field(default_factory=list)
    current: str | None = None


class PresetApplyResponse(BaseModel):
    """プリセット適用結果

    ``restart_servers`` は起動引数が変わり再起動が必要な llama-server 名
    ("base" / "assist" / "embed") の配列。
    """
    applied: str
    changed_sections: list[str] = Field(default_factory=list)
    restart_servers: list[str] = Field(default_factory=list)


# ===== Sessions =====

class SessionRegisterRequest(BaseModel):
    session_id: str
    mode: str = "coding"
    client_type: str = "cli"


class SessionRegisterResponse(BaseModel):
    registered: bool
    session_id: str


class SessionUnregisterResponse(BaseModel):
    unregistered: bool
    session_id: str


class SessionInfoResponse(BaseModel):
    session_id: str
    mode: str
    client_type: str
    registered_at: float


class SessionListResponse(BaseModel):
    sessions: list[SessionInfoResponse]
    count: int


# ===== Session Persistence (CLI ↔ GUI 共通) =====

class SessionTurn(BaseModel):
    """会話ターン（CLI / GUI 共通）"""
    role: str
    content: str
    timestamp: float = 0
    compressed: bool = False


class SessionTokenInfo(BaseModel):
    """セッション内トークン使用情報"""
    used: int = 0
    limit: int = 4096
    pct: int = 0


class SessionData(BaseModel):
    """セッション永続化データ（CLI / GUI 共通スキーマ）

    CLI の /save, /load および GUI のセッション管理で使用する
    統一フォーマット。local/sessions/{name}.json に保存される。
    """
    session_id: str
    started_at: str
    ended_at: str
    duration_sec: int = 0
    mode: str = "coding"
    instance_name: str = "evoref"
    base_model: str = ""
    source: str = "auto"  # "auto" | "manual" | "gui"
    turns: list[SessionTurn] = Field(default_factory=list)
    turn_count: int = 0
    context_files: list[str] = Field(default_factory=list)
    cartridge_ids: list[str] = Field(default_factory=list)
    token_info: SessionTokenInfo = Field(default_factory=SessionTokenInfo)
    summary: str | None = None
    summary_embedding: list[float] | None = None
    topics: list[str] = Field(default_factory=list)
    archived_at: str | None = None


# ===== Loop / Autonomous Driver =====


class LoopStartRequest(BaseModel):
    """`/api/loop/start` リクエスト"""
    project_id: str


class LoopStateInfo(BaseModel):
    """LoopDriver の状態 DTO"""
    running: bool
    project_id: str | None = None
    started_at: float | None = None
    iteration: int = 0
    stop_requested: bool = False
    last_picked_fact_id: str | None = None
    last_picked_task_id: str | None = None
    # pause/resume 状態
    pause_requested: bool = False
    paused: bool = False


class LoopStateResponse(BaseModel):
    state: LoopStateInfo


class TaskInfo(BaseModel):
    """task ファクトの一覧表示用 DTO"""
    fact_id: str
    task_id: str
    title: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    salience: float = 0.5
    status: str = "open"
    source_path: str | None = None
    project_id: str
    created_at: float = 0.0
    accessed_at: float = 0.0
    stage: str | None = None  # staged コーディング工程 (spec/code/test)。通常タスクは None


class TaskListResponse(BaseModel):
    project_id: str
    total: int
    tasks: list[TaskInfo]
    next_task_id: str | None = None


# ===== Loop Current / Stream =====


class LoopCurrentResponse(BaseModel):
    """`/api/loop/current` レスポンス — 実行中タスク / action の詳細"""
    running: bool
    paused: bool
    project_id: str | None = None
    iteration: int = 0
    started_at: float | None = None
    iteration_started_at: float | None = None
    elapsed_seconds: float | None = None
    iteration_elapsed_seconds: float | None = None
    current_trace_id: str | None = None
    current_task: TaskInfo | None = None
    current_action_kind: str | None = None
    last_outcome: dict | None = None


class TaskImportRequest(BaseModel):
    """`/api/loop/tasks/import` リクエスト

    PRD JSON の与え方は 2 通り:
    - ``prd_path``: バックエンドが読めるファイルパス
    - ``prd_json``: 既にロード済の JSON 文字列または dict
    """
    project_id: str
    prd_path: str | None = None
    prd_json: str | dict | list | None = None


class TaskImportResponse(BaseModel):
    project_id: str
    imported: int
    fact_ids: list[str]


# ===== Loop Report =====


class LoopReportResponse(BaseModel):
    """`/api/loop/report` レスポンス — 自律ループ集計レポート"""
    project_id: str
    total_tasks: int
    open_tasks: int
    in_progress_tasks: int
    done_tasks: int
    failed_tasks: int
    iterations: int
    started_at: float | None = None
    elapsed_seconds: float | None = None
    generated_at: float
    failure_pattern_total: int
    failure_pattern_by_error_type: dict[str, int] = Field(default_factory=dict)
    progress_marker_count: int
    # ラルフループ: 編集ファイルのメタデータ配列
    artifacts: list[dict] = Field(default_factory=list)


# ===== System / VRAM =====


class VramModelInfo(BaseModel):
    """VRAM 使用量レスポンスのモデル単位エントリ

    ``source="actual"`` は ``nvidia-smi`` 実測値で更新できたケース。
    ``source="estimate"`` は GGUF ファイルサイズ + ``gpu_layers`` からの推定値。
    ``present=False`` のモデルは設定未配置 / モデルファイル未配置 / 該当
    backend 無効等のケース。
    """

    name: str  # "base" | "assist" | "embed"
    present: bool = False
    vram_mb: int = 0
    gpu_layers: int = 0
    model_mb: int | None = None
    source: str = "estimate"  # "estimate" | "actual"
    placement: str = "none"  # "GPU" | "CPU" | "none"
    pid: int | None = None


class VramStatusResponse(BaseModel):
    """`GET /api/system/vram_status` レスポンス

    ``source`` (トップレベル) の意味:
        - ``"actual"``: 全 ``present=True`` モデルで ``nvidia-smi`` 実測値が取れた
        - ``"mixed"``: 一部のみ実測値、他は推定値
        - ``"estimate"``: すべて推定値 (``nvidia-smi`` 不在 / PID 不明)

    ``measurement_available=True`` は ``nvidia-smi`` + ``LlamaProcessManager``
    が両方揃っていることを示す UI 向けヒント。実測値が 0 件でもこれが True
    なら「計測手段はある」ことを意味し、GUI は「モデルは GPU に乗っていない」
    と判断できる。False の場合は推定値のみ表示する。

    ``over_budget=True`` は ``budget_mb`` が設定済みかつ
    ``total_mb > budget_mb`` のときだけ。``budget_mb=None`` の場合は常に False。
    """

    source: str = "estimate"  # "actual" | "mixed" | "estimate"
    measurement_available: bool = False
    nvidia_smi_available: bool = False
    process_manager_enabled: bool = False
    models: list[VramModelInfo] = Field(default_factory=list)
    total_mb: int = 0
    budget_mb: int | None = None
    over_budget: bool = False
