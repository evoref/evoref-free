"""最適化制御 API — Level 1/2 の詳細状態・手動トリガー・履歴"""

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.config import get_path_resolver
from backend.edition import get_pro_handler, is_pro
from backend.free.api._error_responses import api_error
from backend.free.api.learning._optimize_collectors import (
    collect_assist_prompt_history,
    collect_assist_task_statuses,
    collect_prompt_history,
    collect_prompt_mode_statuses,
    extract_scheduler_params,
    format_run_timestamps,
)

# Pydantic スキーマは _optimize_schemas に集約
# 外部 import 互換性のため re-export する。
from backend.free.api.learning._optimize_schemas import (
    AssistTaskStatus,
    Level1Status,
    Level2Status,
    LoRAHistoryEntry,
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
    "AssistTaskStatus",
    "Level1Status",
    "Level2Status",
    "LoRAHistoryEntry",
    "OptimizeHistoryResponse",
    "OptimizeStatusResponse",
    "OptimizeTriggerRequest",
    "OptimizeTriggerResponse",
    "PromptHistoryEntry",
    "PromptModeStatus",
]

# Level 1 で集計対象とするモード / アシストタスクのリスト
_LEVEL1_MODES = ["coding", "chat"]
_ASSIST_TASKS = ["rag_necessity", "rag_quality", "tool_call", "note_evolve"]


# ── エンドポイント ──


@router.get("/status", response_model=OptimizeStatusResponse)
async def optimize_status(state: AppState = Depends(get_app_state)):
    """最適化状態の詳細を取得"""
    logger.debug("GET /api/optimize/status")

    scheduler = state.learning_scheduler
    resolver = get_path_resolver()

    # Level 1: モード別 + アシストタスク別プロンプト進化状態
    modes = collect_prompt_mode_statuses(state.prompt_manager, _LEVEL1_MODES)
    if is_pro():
        assist_tasks = collect_assist_task_statuses(
            state.assist_prompt_manager, _ASSIST_TASKS,
        )
    else:
        assist_tasks = []

    # スケジューラ設定値 (未初期化時は既定値)
    params = extract_scheduler_params(scheduler)

    # Level 2: LoRA 状態 (Pro のみ)
    lora_adapter_exists = False
    if is_pro():
        try:
            lora_path = resolver.resolve_local("lora_adapter")
            lora_adapter_exists = lora_path.exists()
        except Exception:
            pass

    # 実行タイムスタンプ + running フラグ
    running = False
    last_l1: str | None = None
    last_l2: str | None = None
    if scheduler is not None:
        running = scheduler.running
        last_l1, last_l2 = format_run_timestamps(scheduler.get_status())

    return OptimizeStatusResponse(
        running=running,
        last_level1_run=last_l1,
        last_level2_run=last_l2,
        level1=Level1Status(
            modes=modes,
            assist_tasks=assist_tasks,
            generations=params["generations"],
            population_size=params["population_size"],
            min_experiences=params["min_experiences"],
        ),
        level2=Level2Status(
            spsa_iterations=params["spsa_iterations"],
            sparse_params=params["sparse_params"],
            min_failures=params["min_failures"],
            lora_adapter_exists=lora_adapter_exists,
        ),
    )


@router.post("/trigger", response_model=OptimizeTriggerResponse)
async def optimize_trigger(req: OptimizeTriggerRequest, state: AppState = Depends(get_app_state)):
    """最適化の手動トリガー（Level 1 / Level 2 個別制御）"""
    logger.debug("POST /api/optimize/trigger: level=%s, mode=%s", req.level, req.mode)

    if req.level not in ("level1", "level2"):
        raise api_error(
            400, "E0400", "level must be 'level1' or 'level2'",
            "api.optimize_invalid_level",
        )

    if req.mode is not None and req.mode not in ("coding", "chat"):
        raise api_error(
            400, "E0400", "mode must be 'coding' or 'chat'",
            "api.optimize_invalid_mode",
        )

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

    if req.level == "level1":
        triggered, message = scheduler.trigger_level1(llm_client)
        return OptimizeTriggerResponse(
            triggered=triggered,
            level="level1",
            mode=req.mode,
            message=message,
        )

    # Level 2: SPSA LoRA 微調整（Pro のみ）
    if not is_pro():
        return OptimizeTriggerResponse(
            triggered=False,
            level="level2",
            message="Level 2 optimization requires Pro edition",
        )

    resolver = get_path_resolver()
    try:
        lora_path = resolver.resolve_local("lora_adapter")
    except Exception:
        return OptimizeTriggerResponse(
            triggered=False,
            level="level2",
            message="LoRA adapter path not configured",
        )

    if not lora_path.exists():
        return OptimizeTriggerResponse(
            triggered=False,
            level="level2",
            message="LoRA adapter not found",
        )

    started = scheduler.check_level2(
        is_user_active=False,
        lora_path=lora_path,
    )

    if started:
        message = "Level 2 SPSA optimization triggered"
    else:
        message = "Level 2 trigger conditions not met (check failures count, version manager, eval core)"

    return OptimizeTriggerResponse(
        triggered=started,
        level="level2",
        message=message,
    )


@router.get("/history", response_model=OptimizeHistoryResponse)
async def optimize_history(state: AppState = Depends(get_app_state)):
    """最適化履歴の取得（プロンプト進化 + LoRA バージョン）"""
    logger.debug("GET /api/optimize/history")

    resolver = get_path_resolver()

    # プロンプト進化履歴 (システム + アシスト)
    prompt_history = collect_prompt_history(state.prompt_manager, _LEVEL1_MODES)
    if is_pro():
        prompt_history.update(
            collect_assist_prompt_history(state.assist_prompt_manager, _ASSIST_TASKS),
        )

    # LoRA バージョン履歴（Pro のみ）
    lora_history: list[LoRAHistoryEntry] = []
    LoRAVersionManager = get_pro_handler("lora_version_manager")
    if is_pro() and LoRAVersionManager is not None:
        try:
            versions_dir = resolver.resolve_local("lora_versions_dir")
            adapter_path = resolver.resolve_local("lora_adapter")
            vmgr = LoRAVersionManager(versions_dir, adapter_path)
            for v in vmgr.list_versions():
                lora_history.append(LoRAHistoryEntry(
                    version=v.version,
                    eval_score=v.eval_score,
                    created_at=v.created_at,
                    metadata=v.metadata,
                ))
        except Exception as e:
            logger.debug("LoRA version history unavailable: %s", e)

    return OptimizeHistoryResponse(
        prompt_history=prompt_history,
        lora_history=lora_history,
    )
