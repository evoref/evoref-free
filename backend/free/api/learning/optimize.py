"""最適化制御 API — Level 1 の詳細状態・手動トリガー・履歴"""

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.edition import is_pro
from backend.free.api._error_responses import api_error
from backend.free.core.session_mode import canonicalize_session_mode
from backend.free.api.learning._optimize_collectors import (
    collect_aux_prompt_history,
    collect_aux_task_statuses,
    collect_prompt_history,
    collect_prompt_mode_statuses,
    extract_scheduler_params,
    format_last_level1_run,
)

# Pydantic スキーマは _optimize_schemas に集約
# 外部 import 互換性のため re-export する。
from backend.free.api.learning._optimize_schemas import (
    AuxTaskStatus,
    Level1Status,
    OptimizeHistoryResponse,
    OptimizeStatusResponse,
    OptimizeTriggerRequest,
    OptimizeTriggerResponse,
    PromptHistoryEntry,
    PromptModeStatus,
)
from backend.log_config import get_logger

logger = get_logger("api.optimize")

router = APIRouter(prefix="/api/optimize", tags=["optimize"])

__all__ = [
    "router",
    # 互換性のため re-export (テストや外部から import される)
    "AuxTaskStatus",
    "Level1Status",
    "OptimizeHistoryResponse",
    "OptimizeStatusResponse",
    "OptimizeTriggerRequest",
    "OptimizeTriggerResponse",
    "PromptHistoryEntry",
    "PromptModeStatus",
]

# Level 1 で集計対象とするモード / 補助タスクタスクのリスト
_LEVEL1_MODES = ["create", "chat"]
_AUX_TASKS = ["note_evolve"]


# ── エンドポイント ──


@router.get("/status", response_model=OptimizeStatusResponse)
async def optimize_status(state: AppState = Depends(get_app_state)):
    """最適化状態の詳細を取得"""
    logger.debug("GET /api/optimize/status")

    scheduler = state.learning_scheduler

    # Level 1: モード別 + 補助タスクタスク別プロンプト進化状態
    modes = collect_prompt_mode_statuses(state.prompt_manager, _LEVEL1_MODES)
    if is_pro():
        aux_tasks = collect_aux_task_statuses(
            state.aux_prompt_manager, _AUX_TASKS,
        )
    else:
        aux_tasks = []

    # スケジューラ設定値 (未初期化時は既定値)
    params = extract_scheduler_params(scheduler)

    # 実行タイムスタンプ + running フラグ
    running = False
    last_l1: str | None = None
    if scheduler is not None:
        running = scheduler.running
        last_l1 = format_last_level1_run(scheduler.get_status())

    return OptimizeStatusResponse(
        running=running,
        last_level1_run=last_l1,
        level1=Level1Status(
            modes=modes,
            aux_tasks=aux_tasks,
            generations=params["generations"],
            population_size=params["population_size"],
            min_experiences=params["min_experiences"],
        ),
    )


@router.post("/trigger", response_model=OptimizeTriggerResponse)
async def optimize_trigger(req: OptimizeTriggerRequest, state: AppState = Depends(get_app_state)):
    """最適化の手動トリガー（Level 1）"""
    logger.debug("POST /api/optimize/trigger: level=%s, mode=%s", req.level, req.mode)

    if req.level != "level1":
        raise api_error(
            400, "E0400", "level must be 'level1'",
            "api.optimize_invalid_level",
        )

    if req.mode is not None:
        canonical_mode = canonicalize_session_mode(req.mode)
        if canonical_mode is None:
            raise api_error(
                400, "E0400", "mode must be 'create' or 'chat'",
                "api.optimize_invalid_mode",
            )
        req.mode = canonical_mode

    scheduler = state.learning_scheduler
    if scheduler is None:
        raise api_error(
            503, "E0503", "Learning scheduler not initialized",
            "api.optimize_scheduler_not_initialized",
        )

    if scheduler.running:
        raise api_error(
            409, "E0409", "Optimization already running",
            "api.optimize_already_running",
        )

    # LLMClient を優先、なければ LocalClient にフォールバック
    llm_client = state.llm_client or state.local_client
    if llm_client is None:
        raise api_error(
            503, "E0503", "LLM client not available",
            "api.optimize_llm_unavailable",
        )

    triggered, message = scheduler.trigger_level1(llm_client)
    return OptimizeTriggerResponse(
        triggered=triggered,
        level="level1",
        mode=req.mode,
        message=message,
    )


@router.get("/history", response_model=OptimizeHistoryResponse)
async def optimize_history(state: AppState = Depends(get_app_state)):
    """最適化履歴の取得（プロンプト進化）"""
    logger.debug("GET /api/optimize/history")

    # プロンプト進化履歴 (システム + 補助タスク)
    prompt_history = collect_prompt_history(state.prompt_manager, _LEVEL1_MODES)
    if is_pro():
        prompt_history.update(
            collect_aux_prompt_history(state.aux_prompt_manager, _AUX_TASKS),
        )

    return OptimizeHistoryResponse(prompt_history=prompt_history)
