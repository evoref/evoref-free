"""学習状態 API"""

import time

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.edition import get_pro_handler, is_pro
from backend.free.api._error_responses import api_error
from backend.i18n_helper import msg
from backend.free.api.learning._learning_collectors import (
    extract_executed_phases,
    latest_level2_run,
    map_active_session,
    map_experience_by_mode,
    map_fitness_history,
    map_level1_results,
    map_level2_status,
    map_policy_evolver_status,
    map_priority_queue,
    ts_to_iso,
)

# Pydantic スキーマは _learning_schemas に集約
# 外部 import 互換性のため re-export する。
from backend.free.api.learning._learning_schemas import (
    ActiveSessionInfo,
    EvalCaseInfo,
    ExperienceByModeModel,
    FitnessPoint,
    ImprovementCurveResponse,
    ImprovementPoint,
    LearningStatusResponse,
    Level1ResultEntry,
    PolicyEvolverDomainStatus,
    PriorityRequestEntry,
    SchedulerStatusModel,
    TriggerRequest,
    TriggerResponse,
    TurnFeedbackRequest,
    TurnFeedbackResponse,
)
from backend.free.learning.level1_session import PriorityRequest
from backend.log_config import get_logger

logger = get_logger("api.learning")

router = APIRouter(prefix="/api/learning", tags=["learning"])

__all__ = [
    "router",
    # 互換性のため re-export (テストや外部から import される)
    "ActiveSessionInfo",
    "EvalCaseInfo",
    "ExperienceByModeModel",
    "FitnessPoint",
    "ImprovementCurveResponse",
    "ImprovementPoint",
    "LearningStatusResponse",
    "Level1ResultEntry",
    "PolicyEvolverDomainStatus",
    "PriorityRequestEntry",
    "SchedulerStatusModel",
    "TriggerRequest",
    "TriggerResponse",
    "TurnFeedbackRequest",
    "TurnFeedbackResponse",
]


@router.post("/trigger", response_model=TriggerResponse)
async def trigger_learning(req: TriggerRequest, state: AppState = Depends(get_app_state)):
    """学習サイクルを手動トリガー

    `level=level1`: 優先キューに `manual` 要求を 1 件積む。LLM 接続を待たず
        永続化されるため、未接続でも次回のループ tick で実行される。
    `level=full`: Full sleep-time update を **Trigger B と同じ 1 本の経路で**
        予約し、完了を待ってから `manual` 要求を優先キューに積む
        (Level 1 が「書き終えた記憶」を見る保証)。既に Full が走っていれば
        2 本目は起こさず、その 1 本の完了を待つ。
    """
    logger.debug("POST /api/learning/trigger: level=%s", req.level)
    if req.level not in ("level1", "full"):
        raise api_error(
            400, "E0400", "level must be 'level1' or 'full'",
            "api.learning_invalid_level",
        )

    scheduler = state.learning_scheduler
    if scheduler is None:
        raise api_error(
            503, "E0503", "Learning scheduler not initialized",
            "api.learning_scheduler_not_initialized",
        )

    # Full モード: まず sleep-time update を通す。
    #
    # **``_worker.run_full()`` を直接呼ばない**。スケジューラの
    # ``run_full_now()`` を通すと WM → STM スナップショット
    # (``_run_pre_full_flush``) とクライアント解決が自動 Trigger B と揃う。
    # private 属性を直接触ってクライアント解決をコピーしていたため 2 経路が
    # 乖離し、**手動トリガーだけスナップショットを飛ばしていた** —
    # 進行中セッションのターンが Step 8 の入力から丸ごと抜ける
    # (2026-08-27 ライブ監査で実測。詳細は ``run_full_now`` の docstring)。
    #
    # ``run_full_now()`` は **予約して Trigger B の 1 本に合流する** だけで、
    # 自分では worker を叩かない。手動と Trigger B が別経路で走り、523 秒の
    # Full が 2 本並走した事故 (2026-09-08 ライブ監査) を構造的に止める。
    full_outcome: str | None = None
    if req.level == "full":
        sleep_sched = state.sleep_scheduler
        if sleep_sched is not None:
            try:
                logger.info("Manual trigger: requesting a Full sleep-time update first")
                full_outcome = await sleep_sched.run_full_now()
                logger.info(
                    "Manual trigger: Full sleep-time outcome=%s", full_outcome,
                )
            except Exception as e:
                logger.error("Full sleep-time update failed during manual trigger: %s", e)
                full_outcome = "failed"
                # Full が失敗しても Level 1 は要求として積む

    # 優先キューに manual 要求を push（LLM 未接続でも OK）
    req_obj = PriorityRequest(
        reason="manual",
        requested_at=time.time(),
        relax_ratio=1.0,
        payload={"level": req.level},
    )
    queue_length = scheduler.push_priority_request(req_obj)
    status = scheduler.get_status()

    message = msg(
        "api.learning_request_queued", level=req.level, position=queue_length,
    )
    if full_outcome is not None:
        # Full の結末を先頭に添える。「押したのに何も起きていない」ように
        # 見える 3 つの状態 (走行中に合流 / 待ち切れず予約のまま / 縮退) を
        # UI 側で区別できるようにする。
        note_key = {
            "completed": "api.learning_full_completed",
            "already_running": "api.learning_full_already_running",
            "deferred": "api.learning_full_deferred",
        }.get(full_outcome, "api.learning_full_skipped")
        message = f"{msg(note_key)} {message}"

    return TriggerResponse(
        triggered=True,
        level=req.level,
        experience_count=status["experience_count"],
        message=message,
        queued=True,
        queue_length=queue_length,
    )


@router.get("/improvement-curve", response_model=ImprovementCurveResponse)
async def improvement_curve():
    """改善カーブ用データを返す（LoRA バージョン別 eval_score 推移）

    base (`lora_scores`) の 1 系列。
    Pro 機能: Free 版では両方空配列を返す。
    """
    logger.debug("GET /api/learning/improvement-curve")

    LoRAVersionManager = get_pro_handler("lora_version_manager")
    pro_path = get_pro_handler("pro_learning_path")
    if not is_pro() or LoRAVersionManager is None or pro_path is None:
        return ImprovementCurveResponse()

    # base は Pro の (モデル×モード) パーティション配下。
    vmgr = LoRAVersionManager(pro_path("lora_versions_dir"), pro_path("lora_adapter"))
    return ImprovementCurveResponse(
        lora_scores=[
            ImprovementPoint(
                version=v.version,
                eval_score=v.eval_score,
                created_at=v.created_at,
            )
            for v in vmgr.list_versions()
        ],
    )


@router.get("/status", response_model=LearningStatusResponse)
async def learning_status(state: AppState = Depends(get_app_state)):
    """学習状態と最新評価情報を取得

    Mixed: LoRA/EvalCore 情報は Pro 時のみ返す。Free ではデフォルト値。
    """
    logger.debug("GET /api/learning/status")

    pro_info = _get_pro_learning_info()
    sched_status = _build_scheduler_status(state.learning_scheduler)
    _annotate_level1_gate(sched_status, state.sleep_scheduler)

    return LearningStatusResponse(
        lora_version=pro_info["lora_version"],
        lora_adapter_exists=pro_info["lora_adapter_exists"],
        eval_cases_count=pro_info["eval_cases_count"],
        eval_pass_threshold=pro_info["eval_pass_threshold"],
        eval_cases=pro_info["eval_cases"],
        scheduler_status=sched_status,
    )


def _annotate_level1_gate(
    status: SchedulerStatusModel, sleep_scheduler: object | None,
) -> None:
    """Level 1 が今走れない理由を ``status`` へ書き込む (in-place)。

    ``conditions_met`` は経験件数だけの表示値で、実ゲート (アイドル /
    ユーザー活動 / LLM クライアント配線 / ループ起動) を含まない。両者を
    突き合わせないと「conditions_met: true なのに level1_run_count が 0 の
    まま」の理由が API からは分からず、ログを読むしかなかった
    (2026-08-14 ライブ監査で実際に切り分けに時間を要した)。

    ``LearningScheduler`` 側で分かる理由を先に見て、残りを
    ``SleepTimeScheduler.level1_gate_status()`` から補う。

    **判定順は常駐ループ (``SleepTimeScheduler._schedule_level1_loop``) の
    実体と同じ順にする。** SUSPENDED session の resume (判定 1) と優先キュー
    (判定 2) は **アイドル待ちも経験件数も見ない** ので、どちらかが待って
    いるときに ``waiting_for_idle`` / ``insufficient_experiences`` を返すのは
    嘘になる。実測 (2026-09-15 ライブ監査): 手動トリガー直後に
    ``waiting_for_idle`` (残り 1246 秒) と表示されたが実際は次 tick (60 秒)
    で走り、2 回目は経験カーソルが進んだ状態で ``insufficient_experiences``
    と表示されたがやはり走った。押したボタンの状態が API から読めないと、
    「効いていない」と誤診してもう一度押すことになる。
    """
    if status.is_disabled:
        status.level1_blocked_reason = "learning_disabled"
        return
    if status.running:
        status.level1_blocked_reason = "already_running"
        return

    gate_fn = getattr(sleep_scheduler, "level1_gate_status", None)
    if gate_fn is None:
        # ゲートが読めない構成では従来どおり経験件数だけで答える。
        if not status.conditions_met:
            status.level1_blocked_reason = "insufficient_experiences"
        return
    gate = gate_fn()
    status.level1_seconds_until_idle = gate.get("seconds_until_idle")
    # 予約済みの仕事 (resume 待ちの session / 優先キュー) はアイドルと経験件数を
    # 迂回する。ループの判定 1 / 2 と対応。
    has_pending_work = bool(status.active_session) or bool(status.priority_queue)
    if not gate.get("llm_client_wired"):
        status.level1_blocked_reason = "no_llm_client"
    elif not gate.get("loop_running"):
        status.level1_blocked_reason = "loop_not_started"
    elif gate.get("full_running"):
        status.level1_blocked_reason = "deferred_by_full_cycle"
    elif gate.get("user_active"):
        status.level1_blocked_reason = "user_active"
    elif has_pending_work:
        # 次 tick で走る。止まってはいない。
        status.level1_blocked_reason = None
    elif not status.conditions_met:
        status.level1_blocked_reason = "insufficient_experiences"
    elif not gate.get("idle"):
        status.level1_blocked_reason = "waiting_for_idle"


def _ts_to_iso(ts: float) -> str | None:
    """float タイムスタンプを ISO 8601 文字列に変換する。0 以下は None。

    互換性のため API 互換シグネチャを保持し、内部実装は
    `_learning_collectors.ts_to_iso` に委譲する。
    """
    return ts_to_iso(ts)


def _get_pro_learning_info() -> dict:
    """Pro 固有の LoRA バージョン・Eval 情報を取得する。Free ではデフォルト値を返す。"""
    result: dict = {
        "lora_version": 0,
        "lora_adapter_exists": False,
        "eval_cases_count": 0,
        "eval_pass_threshold": 0.0,
        "eval_cases": [],
    }

    LoRAVersionManager = get_pro_handler("lora_version_manager")
    EvalCoreManager = get_pro_handler("eval_core_manager")
    pro_path = get_pro_handler("pro_learning_path")
    if (
        not is_pro() or LoRAVersionManager is None or EvalCoreManager is None
        or pro_path is None
    ):
        return result

    # 学習済みアダプタは Pro の (モデル×モード) パーティション配下にある。
    # Level 2 が実際に書き出した版と同じ場所を見ないと、同じ応答の中の
    # level2.base.version と食い違う (実機で version=0/exists=false vs v4)。
    adapter_path = pro_path("lora_adapter")
    vmgr = LoRAVersionManager(pro_path("lora_versions_dir"), adapter_path)
    result["lora_version"] = vmgr.get_latest_version()
    result["lora_adapter_exists"] = adapter_path.exists()

    emgr = EvalCoreManager(pro_path("eval_core_file"))
    eval_set = emgr.load()
    result["eval_cases_count"] = len(eval_set.cases)
    result["eval_pass_threshold"] = eval_set.pass_threshold
    result["eval_cases"] = [
        EvalCaseInfo(
            id=c.id, mode=c.mode, query=c.query,
            weight=c.weight, description=c.description,
        )
        for c in eval_set.cases
    ]

    return result


def _build_scheduler_status(scheduler: object | None) -> SchedulerStatusModel:
    """スケジューラの raw ステータスを SchedulerStatusModel に変換する。

    純粋な dict → Pydantic マッピングは `_learning_collectors` に委譲し、
    本関数は scheduler との対話 (`get_status()` / `get_pro_status()`) と
    Pro/Free ガードに専念する。
    """
    if scheduler is None:
        return SchedulerStatusModel()

    raw = scheduler.get_status()

    # Level 1 結果 + executed_phases (非破壊抽出)
    raw_l1 = raw.get("last_level1_results", {})
    level1_results = map_level1_results(raw_l1)
    executed_phases = extract_executed_phases(raw_l1)

    # PolicyParamEvolver は Free でも配線される (Level 1 Step 12) ので
    # エディションでゲートしない (未注入なら get_pro_status が空を返す)。
    policy_evolver_status = {}
    if hasattr(scheduler, "get_pro_status"):
        policy_evolver_status = map_policy_evolver_status(scheduler.get_pro_status())

    return SchedulerStatusModel(
        running=raw.get("running", False),
        is_disabled=raw.get("is_disabled", False),
        experience_count=raw.get("experience_count", 0),
        new_experience_count=raw.get("new_experience_count", 0),
        min_experiences=raw.get("min_experiences", 0),
        conditions_met=raw.get("conditions_met", False),
        last_level1_run=ts_to_iso(raw.get("last_level1_run", 0.0)),
        last_level2_run=ts_to_iso(latest_level2_run(raw.get("last_level2_run", 0.0))),
        running_target=raw.get("running_target"),
        level2=map_level2_status(raw.get("level2")),
        # Level 0 詳細
        last_level0_record=raw.get("last_level0_record"),
        experience_by_mode=map_experience_by_mode(raw.get("experience_by_mode")),
        correction_rate=raw.get("correction_rate", 0.0),
        rag_usage_rate=raw.get("rag_usage_rate", 0.0),
        prev_correction_rate=raw.get("prev_correction_rate"),
        prev_rag_usage_rate=raw.get("prev_rag_usage_rate"),
        rag_gated_rate=raw.get("rag_gated_rate", 0.0),
        rag_pseudo_derived_rate=raw.get("rag_pseudo_derived_rate", 0.0),
        rag_abstain_rate=raw.get("rag_abstain_rate", 0.0),
        rag_score_experience_count=raw.get("rag_score_experience_count", 0),
        long_form_experience_count=raw.get("long_form_experience_count", 0),
        phase_subset_min_experiences=raw.get("phase_subset_min_experiences", 0),
        # Level 1 詳細
        level1_run_count=raw.get("level1_run_count", 0),
        last_level1_results=level1_results,
        executed_phases=executed_phases,
        fitness_history=map_fitness_history(raw.get("fitness_history")),
        # 探索/活用フェーズ
        policy_evolver_status=policy_evolver_status,
        # priority queue
        priority_queue=map_priority_queue(raw.get("priority_queue")),
        active_session=map_active_session(raw.get("active_session")),
    )


@router.post("/feedback", response_model=TurnFeedbackResponse)
async def record_turn_feedback(
    req: TurnFeedbackRequest, state: AppState = Depends(get_app_state),
):
    """応答への明示評価を経験に刻む (2026-09-14、f_04 §3.2.3)。

    👎 は「本人が失敗と言った」唯一の信号で、Level 1 の採用ゲートのケースに
    最優先で使われ、手本プールには入らない。private ターンなど経験が無い
    応答は ``recorded=False``。
    """
    learn = getattr(state, "learn", None)
    buf = getattr(learn, "experience_buffer", None)
    if buf is None or not hasattr(buf, "mark_user_feedback"):
        return TurnFeedbackResponse(recorded=False, message="experience buffer unavailable")
    negative = {"negative": True, "positive": False, "clear": None}[req.verdict]
    entry = buf.mark_user_feedback(
        req.session_id, negative=negative, note=req.note, query=req.query,
    )
    if entry is None:
        return TurnFeedbackResponse(recorded=False, message="no experience for this turn")
    logger.info(
        "Turn feedback recorded: session=%s verdict=%s entry=%s",
        req.session_id, req.verdict, entry.id,
    )
    return TurnFeedbackResponse(recorded=True, entry_id=entry.id)
