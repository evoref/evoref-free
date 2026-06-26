"""学習状態 API"""

import time

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.config import get_path_resolver
from backend.edition import get_pro_handler, is_pro
from backend.free.api._error_responses import api_error
from backend.free.api.learning._learning_collectors import (
    extract_executed_phases,
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
    # sleep-time の LLM ステージはアシストモデルでのみ実行する (degraded 時は
    # ベースへフォールバックせず run_full(None) が Step 5.8-10 をスキップ)。
    # 通常のスケジュール経路 (scheduler.py の Full) と同じ方針 (c_14 §6.2)。
    if req.level == "full":
        sleep_sched = state.sleep_scheduler
        if sleep_sched is not None and sleep_sched._worker is not None:
            sleep_client = state.assist_client
            if sleep_client:
                try:
                    logger.info(
                        "Manual trigger: running Full sleep-time update first (client=%s)",
                        type(sleep_client).__name__,
                    )
                    await sleep_sched._worker.run_full(sleep_client)
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

    Pro 機能: Free 版では空配列を返す。
    """
    logger.debug("GET /api/learning/improvement-curve")

    LoRAVersionManager = get_pro_handler("lora_version_manager")
    if not is_pro() or LoRAVersionManager is None:
        return ImprovementCurveResponse(lora_scores=[])

    resolver = get_path_resolver()
    versions_dir = resolver.resolve_local("lora_versions_dir")
    adapter_path = resolver.resolve_local("lora_adapter")
    vmgr = LoRAVersionManager(versions_dir, adapter_path)

    versions = vmgr.list_versions()
    lora_scores = [
        ImprovementPoint(
            version=v.version,
            eval_score=v.eval_score,
            created_at=v.created_at,
        )
        for v in versions
    ]

    return ImprovementCurveResponse(lora_scores=lora_scores)


@router.get("/status", response_model=LearningStatusResponse)
async def learning_status(state: AppState = Depends(get_app_state)):
    """学習状態と最新評価情報を取得

    Mixed: LoRA/EvalCore 情報は Pro 時のみ返す。Free ではデフォルト値。
    """
    logger.debug("GET /api/learning/status")

    pro_info = _get_pro_learning_info()
    sched_status = _build_scheduler_status(state.learning_scheduler)

    return LearningStatusResponse(
        lora_version=pro_info["lora_version"],
        lora_adapter_exists=pro_info["lora_adapter_exists"],
        eval_cases_count=pro_info["eval_cases_count"],
        eval_pass_threshold=pro_info["eval_pass_threshold"],
        eval_cases=pro_info["eval_cases"],
        scheduler_status=sched_status,
    )


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
    versions_dir = resolver.resolve_local("lora_versions_dir")
    adapter_path = resolver.resolve_local("lora_adapter")
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
        last_level2_run=ts_to_iso(raw.get("last_level2_run", 0.0)),
        running_target=raw.get("running_target"),
        level2=map_level2_status(raw.get("level2")),
        # Level 0 詳細
        last_level0_record=raw.get("last_level0_record"),
        experience_by_mode=map_experience_by_mode(raw.get("experience_by_mode")),
        correction_rate=raw.get("correction_rate", 0.0),
        rag_usage_rate=raw.get("rag_usage_rate", 0.0),
        prev_correction_rate=raw.get("prev_correction_rate"),
        prev_rag_usage_rate=raw.get("prev_rag_usage_rate"),
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
