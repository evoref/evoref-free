"""学習状態 API"""

import time

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.config import get_path_resolver
from backend.edition import get_pro_handler, is_pro
from backend.free.api._error_responses import api_error
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
]


@router.post("/trigger", response_model=TriggerResponse)
async def trigger_learning(req: TriggerRequest, state: AppState = Depends(get_app_state)):
    """学習サイクルを手動トリガー

    `level=level1`: 優先キューに `manual` 要求を 1 件積む。LLM 接続を待たず
        永続化されるため、未接続でも次回のループ tick で実行される。
    `level=full`: Full sleep-time update を即座にキックし、追加で `manual`
        要求を優先キューに積む。
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

    # Full モード: まず sleep-time update を実行する。
    #
    # **``_worker.run_full()`` を直接呼ばない**。スケジューラの
    # ``run_full_now()`` を通すと WM → STM スナップショット
    # (``_run_pre_full_flush``) とクライアント解決が自動 Trigger B と揃う。
    # private 属性を直接触ってクライアント解決をコピーしていたため 2 経路が
    # 乖離し、**手動トリガーだけスナップショットを飛ばしていた** —
    # 進行中セッションのターンが Step 8 の入力から丸ごと抜ける
    # (2026-08-27 ライブ監査で実測。詳細は ``run_full_now`` の docstring)。
    #
    # LLM 未接続なら False が返り、Step 5.8-10 はスキップされる (c_14 §6)。
    if req.level == "full":
        sleep_sched = state.sleep_scheduler
        if sleep_sched is not None:
            try:
                logger.info("Manual trigger: running Full sleep-time update first")
                ran = await sleep_sched.run_full_now()
                if not ran:
                    logger.info(
                        "Manual trigger: Full sleep-time skipped "
                        "(worker or sleep client unavailable)",
                    )
            except Exception as e:
                logger.error("Full sleep-time update failed during manual trigger: %s", e)
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

    message = (
        f"Manual {req.level} request queued (position={queue_length}). "
        "Will be processed by Level 1 loop on next tick."
    )

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
    if not is_pro() or LoRAVersionManager is None:
        return ImprovementCurveResponse()

    resolver = get_path_resolver()

    # base は (モデル×モード) パーティション配下。
    # resolve_local (flat) のままだと Level 2 が積んだ版履歴が 1 件も出ない。
    def _scores(resolve, versions_key: str, adapter_key: str) -> list[ImprovementPoint]:
        vmgr = LoRAVersionManager(resolve(versions_key), resolve(adapter_key))
        return [
            ImprovementPoint(
                version=v.version,
                eval_score=v.eval_score,
                created_at=v.created_at,
            )
            for v in vmgr.list_versions()
        ]

    return ImprovementCurveResponse(
        lora_scores=_scores(
            resolver.resolve_learning, "lora_versions_dir", "lora_adapter",
        ),
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
    """
    if status.is_disabled:
        status.level1_blocked_reason = "learning_disabled"
        return
    if status.running:
        status.level1_blocked_reason = "already_running"
        return
    if not status.conditions_met:
        status.level1_blocked_reason = "insufficient_experiences"
        return

    gate_fn = getattr(sleep_scheduler, "level1_gate_status", None)
    if gate_fn is None:
        return
    gate = gate_fn()
    status.level1_seconds_until_idle = gate.get("seconds_until_idle")
    if not gate.get("llm_client_wired"):
        status.level1_blocked_reason = "no_llm_client"
    elif not gate.get("loop_running"):
        status.level1_blocked_reason = "loop_not_started"
    elif gate.get("user_active"):
        status.level1_blocked_reason = "user_active"
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
    if not is_pro() or LoRAVersionManager is None or EvalCoreManager is None:
        return result

    resolver = get_path_resolver()
    # 学習済みアダプタは (モデル×モード) パーティション配下にある。resolve_local
    # (flat) のままだと Level 2 が実際に書き出した版が見えず、同じ応答の中の
    # level2.base.version と食い違う (実機で version=0/exists=false vs v4)。
    versions_dir = resolver.resolve_learning("lora_versions_dir")
    adapter_path = resolver.resolve_learning("lora_adapter")
    vmgr = LoRAVersionManager(versions_dir, adapter_path)
    result["lora_version"] = vmgr.get_latest_version()
    result["lora_adapter_exists"] = adapter_path.exists()

    eval_path = resolver.resolve_local("eval_core_file")
    emgr = EvalCoreManager(eval_path)
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

    # Pro 拡張ステータス (Pro ガードは handler 側に残置)
    policy_evolver_status = {}
    if is_pro() and hasattr(scheduler, "get_pro_status"):
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
