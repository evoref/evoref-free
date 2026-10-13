"""`/api/learning` のレスポンス Pydantic スキーマ

`backend.free.api.learning.learning` から Pydantic スキーマを切り出した型モジュール。
循環 import 防止のため、`_learning_collectors` (helper) と `learning.py`
(handler) の双方から参照される型はここに集約する。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.free.api.learning._growth_schemas import GrowthSummaryModel


class ExperienceByModeModel(BaseModel):
    """モード別経験数"""
    chat: int = 0
    create: int = 0


class Level1ResultEntry(BaseModel):
    """Level 1 各フェーズの結果

    ``skipped`` は「対象が無く実行していない」(経験不足 / 選択圧なし /
    outcome シグナル未配線)。``improved=False`` だけでは「実行して改善なし」と
    区別できないため分ける。``measured_*`` は base prompt 採用ゲートの実測値
    (現行 / 候補の judge 平均、f_04 §4.5)。
    """
    improved: bool = False
    fitness_before: float | None = None
    fitness_after: float | None = None
    skipped: bool = False
    reason: str | None = None
    measured_before: float | None = None
    measured_after: float | None = None
    #: 一対比較ゲート (f_04 §4.5) の実測。``gate == "pairwise"`` のとき
    #: ``measured_*`` は常に None なので、勝敗がこちらに載る。
    gate: str | None = None
    wins: int | None = None
    losses: int | None = None
    ties: int | None = None
    #: 現行 vs 現行 (カナリア) の |純勝ち|。候補の純勝ちがこれを超えないと採用しない。
    noise_floor: int | None = None
    #: 一対比較で採用候補になったときだけ測る標本 (成功ターン) の勝敗。
    #: 1 敗でもあれば ``reason == "sample_regression"`` で不採用。
    sample_wins: int | None = None
    sample_losses: int | None = None
    sample_ties: int | None = None
    #: 失敗の証拠があったのに評価ケースにしなかった件数 (除外理由 → 件数)。
    excluded: dict[str, int] | None = None
    #: 標本に足した回帰台帳の guard の合否 (f_04 §6.5)。
    guards_passed: int | None = None
    guards_failed: int | None = None
    #: ゲートを通ったが自動採用が効かず提案にした回の提案 ID (``reason == "proposed"``、f_04 §6.6)。
    proposal_id: str | None = None


class PolicyEvolverDomainStatus(BaseModel):
    """ポリシー進化ドメインの状態 (PolicyParamEvolver は Free でも配線される)"""
    current_fitness: float | None = None
    best_fitness: float = 0.0
    decline_count: int = 0
    sigma: float = 0.0
    phase: str = ""
    #: 恒真ガードで凍結中か (直近窓の fitness が動いていない = 乱歩防止で skip)。
    degenerate: bool = False
    fitness_history_len: int = 0
    #: 評価窓の起点 (この timestamp より新しい経験だけで次を評価する)。
    last_evolved_at: str | None = None


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


class SchedulerStatusModel(BaseModel):
    """学習スケジューラの状態"""
    running: bool = False
    # --no-learning (自己学習 OFF) で起動中かどうか。GUI/CLI が学習無効状態を
    # 観測できるように surface する (scheduler.get_status の is_disabled を伝播)。
    is_disabled: bool = False
    #: ``llama.slots < 2`` で Level 1 を止めているか (f_04 §10.1 #15)。
    single_slot: bool = False
    experience_count: int = 0
    new_experience_count: int = 0
    min_experiences: int = 0
    #: **経験件数だけ** を見た表示値。Level 1 の実際の起動ゲートには
    #: アイドル時間 / ユーザー活動 / LLM クライアント配線も含まれるため、
    #: これが True でも走らないことは正常状態として起こる。実際に何で
    #: 止まっているかは ``level1_blocked_reason`` を見る。
    conditions_met: bool = False
    #: Level 1 が「今」走れない理由。走れる状態なら None。
    #: ``learning_disabled`` / ``single_slot`` / ``already_running`` / ``no_llm_client`` /
    #: ``loop_not_started`` / ``deferred_by_full_cycle`` / ``user_active`` /
    #: ``serving_another_partition`` / ``insufficient_experiences`` /
    #: ``waiting_for_idle`` のいずれか。
    #:
    #: 判定順は常駐ループ (``SleepTimeScheduler._schedule_level1_loop``) と
    #: 同じ。**予約済みの仕事 (``active_session`` / ``priority_queue``) がある
    #: ときは ``insufficient_experiences`` / ``waiting_for_idle`` を返さない**
    #: — ループの resume / 優先キュー経路はどちらの条件も見ないため
    #: (2026-09-15 ライブ監査)。
    level1_blocked_reason: str | None = None
    #: ``waiting_for_idle`` のとき、アイドル成立までの残り秒数。
    level1_seconds_until_idle: float | None = None
    #: Level 1 を延期している理由。``serving_another_partition`` = 載っているモデル
    #: (create_model 等) が束ねた active パーティションのモデルと違う (chat に戻るまで続く)。
    #: このとき ``level1_blocked_reason`` も同じ値になる (ユーザー活動等の手前の理由が無ければ)。
    deferred_reason: str | None = None
    #: 延期が始まった時刻 (ISO 8601 UTC)。延期していなければ None。
    deferred_since: str | None = None
    last_level1_run: str | None = None
    # Level 0 詳細
    last_level0_record: str | None = None
    experience_by_mode: ExperienceByModeModel = Field(
        default_factory=ExperienceByModeModel,
    )
    correction_rate: float = 0.0
    rag_usage_rate: float = 0.0
    prev_correction_rate: float | None = None
    prev_rag_usage_rate: float | None = None
    # 注入ゼロの理由 (f_04 §3.2): 疑似クエリゲートで corpus を引かなかった率と、
    # 注入ありのうち疑似クエリ索引経由が 1 件以上の率。
    rag_gated_rate: float = 0.0
    rag_pseudo_derived_rate: float = 0.0
    # 注入ありのうち「参考情報には記載が無い」と差し控えた率 (検索の取りこぼしの観測)
    rag_abstain_rate: float = 0.0
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
    # 探索/活用フェーズ (PolicyParamEvolver 注入時に値が入る。Free / Pro 共通)
    policy_evolver_status: dict[str, PolicyEvolverDomainStatus] = Field(
        default_factory=dict,
    )
    # 優先キューと SUSPENDED session
    priority_queue: list[PriorityRequestEntry] = Field(default_factory=list)
    active_session: ActiveSessionInfo | None = None
    #: Level 2 (成長) の要約 (自動採用の有効性・未決の提案数・guard / pressure 件数、f_04 §6.8)。
    growth: GrowthSummaryModel | None = None


class LearningStatusResponse(BaseModel):
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


class TurnFeedbackRequest(BaseModel):
    """応答への明示評価 (👎 / 👍 / 取り消し)。``query`` はその応答を生んだユーザー
    発話 (省略時はセッションの最新ターン)。"""

    session_id: str
    verdict: str = Field(pattern="^(negative|positive|clear)$")
    query: str | None = None
    note: str = ""


class TurnFeedbackResponse(BaseModel):
    recorded: bool
    entry_id: str | None = None
    message: str = ""
