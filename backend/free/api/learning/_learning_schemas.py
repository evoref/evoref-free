"""`/api/learning` のレスポンス Pydantic スキーマ

`backend.free.api.learning.learning` から Pydantic スキーマを切り出した型モジュール。
循環 import 防止のため、`_learning_collectors` (helper) と `learning.py`
(handler) の双方から参照される型はここに集約する。

`learning.py` は本モジュールから schemas を import + re-export し、外部からの
`from backend.free.api.learning.learning import SchedulerStatusModel` の互換性を維持する。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCaseInfo(BaseModel):
    id: str
    mode: str
    query: str
    weight: float = 1.0
    description: str = ""


class ExperienceByModeModel(BaseModel):
    """モード別経験数"""
    chat: int = 0
    create: int = 0


class Level1ResultEntry(BaseModel):
    """Level 1 各フェーズの結果"""
    improved: bool = False
    fitness_before: float | None = None
    fitness_after: float | None = None


class PolicyEvolverDomainStatus(BaseModel):
    """ポリシー進化ドメインの状態（Pro）"""
    current_fitness: float | None = None
    best_fitness: float = 0.0
    decline_count: int = 0
    sigma: float = 0.0
    phase: str = ""


class FitnessPoint(BaseModel):
    """fitness 履歴の1ポイント"""
    run: int = 0
    fitness: float = 0.0


class PriorityRequestEntry(BaseModel):
    """優先キュー要求のスナップショット"""
    reason: str
    requested_at: float
    relax_ratio: float = 1.0
    payload: dict | None = None


class ActiveSessionInfo(BaseModel):
    """SUSPENDED な Level1Session の情報"""
    session_id: str
    started_at: float
    reason: str
    completed_phases: list[str] = Field(default_factory=list)
    yield_count: int = 0
    cartridge_snapshot: list[str] = Field(default_factory=list)
    experience_count: int = 0


class Level2TargetStatus(BaseModel):
    """Level 2 の状態 + 発火条件（Pro）"""
    method: str = ""
    bootstrap_enabled: bool = False
    adapter_exists: bool = False
    version: int = 0
    # 蓄積中の発火データ数 (失敗数)
    experiences_current: int = 0
    bootstrap_min: int = 0
    spsa_min: int = 0
    cvector_min: int = 0
    # 発火しない理由コード ("" = 発火可能)。表示ラベル化はフロント側 i18n が行う。
    # 例: noop_base_lora / bootstrap_disabled /
    #     insufficient_data / components_missing
    block_reason: str = ""


class Level2GatesModel(BaseModel):
    """Level 2 自動発火の共通タイミングゲート（Pro）"""
    active_minutes: float = 5.0
    overdue_hours: float = 24.0
    recheck_interval_sec: float = 300.0
    #: target ごとの **実効** クールダウン時間 (h)。連続無改善が
    #: ``stale_streak`` に達した target は延長クールダウンへ落ちるため、
    #: 素の ``overdue_hours`` とは一致しないことがある。
    cooldown_hours: dict[str, float] = Field(default_factory=dict)
    #: target ごとの連続無改善回数 (採用成功で 0 に戻る)。
    no_improve_streak: dict[str, int] = Field(default_factory=dict)
    #: 延長クールダウンへ落ちる連続無改善回数の閾値。
    stale_streak: int = 2
    #: target ごとの「クールダウンを過ぎているか」。データ充足
    #: (``block_reason == ""``) と AND を取って初めて「発火可」になる。
    #: 実行側と同じ単一述語 (``is_level2_overdue``) 由来。
    overdue: dict[str, bool] = Field(default_factory=dict)
    #: target ごとの前回実行からの経過秒。未実行は ``None``。
    seconds_since_run: dict[str, float | None] = Field(default_factory=dict)


class Level2StatusModel(BaseModel):
    """Level 2 (LoRA) の状態（Pro のみ非 None）"""
    running_target: str | None = None
    next_target: str = "base"
    base: Level2TargetStatus = Field(default_factory=Level2TargetStatus)
    gates: Level2GatesModel = Field(default_factory=Level2GatesModel)


class SchedulerStatusModel(BaseModel):
    """学習スケジューラの状態"""
    running: bool = False
    # --no-learning (自己学習 OFF) で起動中かどうか。GUI/CLI が学習無効状態を
    # 観測できるように surface する (scheduler.get_status の is_disabled を伝播)。
    is_disabled: bool = False
    experience_count: int = 0
    new_experience_count: int = 0
    min_experiences: int = 0
    #: **経験件数だけ** を見た表示値。Level 1 の実際の起動ゲートには
    #: アイドル時間 / ユーザー活動 / LLM クライアント配線も含まれるため、
    #: これが True でも走らないことは正常状態として起こる。実際に何で
    #: 止まっているかは ``level1_blocked_reason`` を見る。
    conditions_met: bool = False
    #: Level 1 が「今」走れない理由。走れる状態なら None。
    #: ``learning_disabled`` / ``already_running`` / ``insufficient_experiences``
    #: / ``no_llm_client`` / ``loop_not_started`` / ``user_active`` /
    #: ``waiting_for_idle`` のいずれか。
    level1_blocked_reason: str | None = None
    #: ``waiting_for_idle`` のとき、アイドル成立までの残り秒数。
    level1_seconds_until_idle: float | None = None
    last_level1_run: str | None = None
    last_level2_run: str | None = None
    # 実行中の Level 2 対象 ("base"/None)
    running_target: str | None = None
    # Level 2 (LoRA) の状態 + 発火条件（Pro のみ非 None）
    level2: Level2StatusModel | None = None
    # Level 0 詳細
    last_level0_record: str | None = None
    experience_by_mode: ExperienceByModeModel = Field(
        default_factory=ExperienceByModeModel,
    )
    correction_rate: float = 0.0
    rag_usage_rate: float = 0.0
    prev_correction_rate: float | None = None
    prev_rag_usage_rate: float | None = None
    # phase3 (embed_instruction) / phase4 (token_budget) 部分集合条件の可視化
    # (閾値は phase_subset_min_experiences)
    rag_score_experience_count: int = 0
    long_form_experience_count: int = 0
    phase_subset_min_experiences: int = 0
    # Level 1 詳細
    level1_run_count: int = 0
    last_level1_results: dict[str, Level1ResultEntry] = Field(default_factory=dict)
    executed_phases: list[str] = Field(default_factory=list)
    fitness_history: dict[str, list[FitnessPoint]] = Field(default_factory=dict)
    # 探索/活用フェーズ（Pro のみ値が入る）
    policy_evolver_status: dict[str, PolicyEvolverDomainStatus] = Field(
        default_factory=dict,
    )
    # 優先キューと SUSPENDED session
    priority_queue: list[PriorityRequestEntry] = Field(default_factory=list)
    active_session: ActiveSessionInfo | None = None


class LearningStatusResponse(BaseModel):
    lora_version: int
    lora_adapter_exists: bool
    eval_cases_count: int
    eval_pass_threshold: float
    eval_cases: list[EvalCaseInfo] = Field(default_factory=list)
    scheduler_status: SchedulerStatusModel = Field(default_factory=SchedulerStatusModel)


class TriggerRequest(BaseModel):
    level: str  # "level1" or "full"


class TriggerResponse(BaseModel):
    triggered: bool
    level: str
    experience_count: int
    message: str
    queued: bool = False
    queue_length: int = 0


class ImprovementPoint(BaseModel):
    version: int
    eval_score: float
    created_at: str


class ImprovementCurveResponse(BaseModel):
    lora_scores: list[ImprovementPoint] = Field(default_factory=list)
