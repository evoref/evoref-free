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


def _level2_methods(scheduler) -> dict:
    """Level 2 の有効メソッドと実行可否を返す (#3a)。

    base=lora / assist=none は no-op 経路でトリガ段階 skip されるため、実メソッド
    (base=cvector / assist=spsa-real-eval) が有効かを ``will_run`` で示す。
    """
    base_method = getattr(scheduler, "level2_base_method", "lora") if scheduler else "lora"
    assist_method = getattr(scheduler, "level2_assist_method", "none") if scheduler else "none"
    if base_method == "cvector":
        active = "cvector"
    elif base_method == "spsa-real-eval" or assist_method == "spsa-real-eval":
        active = "spsa-real-eval"
    else:
        active = "none"
    return {
        "base_method": base_method,
        "assist_method": assist_method,
        "active_method": active,
        "will_run": active != "none",
    }


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
            **_level2_methods(scheduler),
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

    # #3a: base=lora / assist=none は no-op 経路でトリガ段階 skip されるため、
    # 実メソッドが無効なら「データ不足」ではなく「メソッド未有効」を明示する。
    methods = _level2_methods(scheduler)
    if not methods["will_run"]:
        return OptimizeTriggerResponse(
            triggered=False,
            level="level2",
            message=(
                "Level 2 real methods not enabled (base=lora/assist=none are no-op "
                "and trigger-skipped). Set level2_base_method=cvector or "
                "level2_assist_method=spsa-real-eval."
            ),
        )

    # 実メソッド有効: 全パス集合を渡して trainer に委譲 (SleepTimeWorker と同形)。
    # cvector は LoRA 不要、base/assist bootstrap は adapter 不在時に走るため、
    # adapter 存在を入口で要求しない (パス未解決は None で degraded に倒す)。
    resolver = get_path_resolver()

    def _safe(resolve, key):
        try:
            return resolve(key)
        except Exception:
            return None

    base_model_path = _safe(resolver.resolve_model, "base_model")
    started = scheduler.check_level2(
        is_user_active=False,
        lora_path=_safe(resolver.resolve_local, "lora_adapter"),
        current_model=base_model_path.name if base_model_path else "",
        base_model_path=base_model_path,
        assist_lora_path=_safe(resolver.resolve_local, "assist_lora_adapter"),
        assist_model_path=_safe(resolver.resolve_model, "assist_model"),
    )

    message = (
        "Level 2 optimization triggered" if started
        else "Level 2 not started (idle/experience/corpus requirements not met)"
    )
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
