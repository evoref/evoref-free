"""補助タスクプロンプト管理 API エンドポイント"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.log_config import get_logger

logger = get_logger("api.aux_prompts")

router = APIRouter(prefix="/api/aux-prompts", tags=["aux-prompts"])


class PromptUpdateRequest(BaseModel):
    content: str


class RollbackRequest(BaseModel):
    version: int


@router.get("")
async def list_aux_prompts(state: AppState = Depends(get_app_state)):
    """全タスクの補助タスクプロンプト一覧"""
    logger.debug("GET /api/aux-prompts")
    mgr = state.aux_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Aux prompt manager not initialized")

    result = []
    for task in mgr.TASKS:
        meta = mgr.get_meta(task)
        content = mgr.get_aux_prompt(task)
        result.append({
            "task": task,
            "version": meta.version,
            "source": meta.source,
            "updated_at": meta.updated_at,
            "fitness_score": meta.fitness_score,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
        })
    return result


@router.get("/{task}")
async def get_aux_prompt(task: str, state: AppState = Depends(get_app_state)):
    """タスク別補助タスクプロンプト詳細"""
    logger.debug("GET /api/aux-prompts/%s", task)
    mgr = state.aux_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Aux prompt manager not initialized")

    try:
        meta = mgr.get_meta(task)
        content = mgr.get_aux_prompt(task)
    except ValueError:
        raise HTTPException(404, f"Unknown task: {task}")

    return {
        "task": task,
        "version": meta.version,
        "source": meta.source,
        "updated_at": meta.updated_at,
        "fitness_score": meta.fitness_score,
        "content": content,
    }


@router.put("/{task}")
async def update_aux_prompt(task: str, body: PromptUpdateRequest, state: AppState = Depends(get_app_state)):
    """補助タスクプロンプト手動更新"""
    logger.debug("PUT /api/aux-prompts/%s: content_len=%d", task, len(body.content))
    mgr = state.aux_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Aux prompt manager not initialized")

    try:
        mgr.update_manual(task, body.content)
    except ValueError:
        raise HTTPException(404, f"Unknown task: {task}")

    return {"status": "updated", "version": mgr.get_meta(task).version}


@router.post("/{task}/reload")
async def reload_aux_prompt(task: str, state: AppState = Depends(get_app_state)):
    """ディスクから再読込み"""
    logger.debug("POST /api/aux-prompts/%s/reload", task)
    mgr = state.aux_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Aux prompt manager not initialized")

    if task not in mgr.TASKS:
        raise HTTPException(404, f"Unknown task: {task}")

    mgr._load_all()
    return {"status": "reloaded"}


@router.get("/{task}/history")
async def get_aux_prompt_history(task: str, state: AppState = Depends(get_app_state)):
    """タスク別履歴一覧"""
    logger.debug("GET /api/aux-prompts/%s/history", task)
    mgr = state.aux_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Aux prompt manager not initialized")

    try:
        return mgr.get_history(task)
    except ValueError:
        raise HTTPException(404, f"Unknown task: {task}")


@router.post("/{task}/rollback")
async def rollback_aux_prompt(task: str, body: RollbackRequest, state: AppState = Depends(get_app_state)):
    """過去バージョンへのロールバック"""
    logger.debug("POST /api/aux-prompts/%s/rollback: version=%d", task, body.version)
    mgr = state.aux_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Aux prompt manager not initialized")

    try:
        mgr.rollback(task, body.version)
    except ValueError:
        raise HTTPException(404, f"Unknown task: {task}")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    return {"status": "rolled_back", "version": mgr.get_meta(task).version}
