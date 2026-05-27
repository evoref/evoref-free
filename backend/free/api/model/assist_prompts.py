"""アシストプロンプト管理 API エンドポイント"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.log_config import get_logger

logger = get_logger("api.assist_prompts")

router = APIRouter(prefix="/api/assist-prompts", tags=["assist-prompts"])


class PromptUpdateRequest(BaseModel):
    content: str


class RollbackRequest(BaseModel):
    version: int


@router.get("")
async def list_assist_prompts(state: AppState = Depends(get_app_state)):
    """全タスクのアシストプロンプト一覧"""
    logger.debug("GET /api/assist-prompts")
    mgr = state.assist_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Assist prompt manager not initialized")

    result = []
    for task in mgr.TASKS:
        meta = mgr.get_meta(task)
        content = mgr.get_assist_prompt(task)
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
async def get_assist_prompt(task: str, state: AppState = Depends(get_app_state)):
    """タスク別アシストプロンプト詳細"""
    logger.debug("GET /api/assist-prompts/%s", task)
    mgr = state.assist_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Assist prompt manager not initialized")

    try:
        meta = mgr.get_meta(task)
        content = mgr.get_assist_prompt(task)
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
async def update_assist_prompt(task: str, body: PromptUpdateRequest, state: AppState = Depends(get_app_state)):
    """アシストプロンプト手動更新"""
    logger.debug("PUT /api/assist-prompts/%s: content_len=%d", task, len(body.content))
    mgr = state.assist_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Assist prompt manager not initialized")

    try:
        mgr.update_manual(task, body.content)
    except ValueError:
        raise HTTPException(404, f"Unknown task: {task}")

    return {"status": "updated", "version": mgr.get_meta(task).version}


@router.post("/{task}/reload")
async def reload_assist_prompt(task: str, state: AppState = Depends(get_app_state)):
    """ディスクから再読込み"""
    logger.debug("POST /api/assist-prompts/%s/reload", task)
    mgr = state.assist_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Assist prompt manager not initialized")

    if task not in mgr.TASKS:
        raise HTTPException(404, f"Unknown task: {task}")

    mgr._load_all()
    return {"status": "reloaded"}


@router.get("/{task}/history")
async def get_assist_prompt_history(task: str, state: AppState = Depends(get_app_state)):
    """タスク別履歴一覧"""
    logger.debug("GET /api/assist-prompts/%s/history", task)
    mgr = state.assist_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Assist prompt manager not initialized")

    try:
        return mgr.get_history(task)
    except ValueError:
        raise HTTPException(404, f"Unknown task: {task}")


@router.post("/{task}/rollback")
async def rollback_assist_prompt(task: str, body: RollbackRequest, state: AppState = Depends(get_app_state)):
    """過去バージョンへのロールバック"""
    logger.debug("POST /api/assist-prompts/%s/rollback: version=%d", task, body.version)
    mgr = state.assist_prompt_manager
    if mgr is None:
        raise HTTPException(503, "Assist prompt manager not initialized")

    try:
        mgr.rollback(task, body.version)
    except ValueError:
        raise HTTPException(404, f"Unknown task: {task}")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    return {"status": "rolled_back", "version": mgr.get_meta(task).version}
