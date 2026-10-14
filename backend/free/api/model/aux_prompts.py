"""補助タスクプロンプト管理 API エンドポイント"""

from fastapi import APIRouter, Depends, HTTPException, Path, Response
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.free.api._pagination import LimitParam, OffsetParam, paginate
from backend.free.agent.aux_prompt_manager import AuxPromptManager
from backend.free.api._error_responses import api_error
from backend.free.api.model._prompt_helpers import HISTORY_VERSION_MAX
from backend.log_config import get_logger

logger = get_logger("api.aux_prompts")

router = APIRouter(prefix="/api/aux-prompts", tags=["aux-prompts"])


class PromptUpdateRequest(BaseModel):
    content: str


class RollbackRequest(BaseModel):
    version: int


def _require_manager(state: AppState) -> AuxPromptManager:
    """``state.aux_prompt_manager`` を返す。``None`` なら 503。"""
    mgr = state.aux_prompt_manager
    if mgr is None:
        raise api_error(
            503, "E0503", "Aux prompt manager not initialized",
            "api.aux_prompt_manager_not_initialized",
        )
    return mgr


def _unknown_task_error(task: str) -> HTTPException:
    return api_error(404, "E0404", f"Unknown task: {task}", "api.aux_prompt_task_unknown", task=task)


def _version_not_found_error(task: str, version: int) -> HTTPException:
    return api_error(
        404, "E0404", f"Aux prompt version not found: {task} v{version}",
        "api.aux_prompt_version_not_found", task=task, version=version,
    )


@router.get("")
async def list_aux_prompts(state: AppState = Depends(get_app_state)):
    """全タスクの補助タスクプロンプト一覧"""
    logger.debug("GET /api/aux-prompts")
    mgr = _require_manager(state)

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
    mgr = _require_manager(state)

    try:
        meta = mgr.get_meta(task)
        content = mgr.get_aux_prompt(task)
    except ValueError:
        raise _unknown_task_error(task)

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
    mgr = _require_manager(state)

    try:
        mgr.update_manual(task, body.content)
    except ValueError:
        raise _unknown_task_error(task)

    return {"status": "updated", "version": mgr.get_meta(task).version}


@router.get("/{task}/history")
async def get_aux_prompt_history(
    task: str,
    response: Response,
    limit: LimitParam = None,
    offset: OffsetParam = 0,
    state: AppState = Depends(get_app_state),
):
    """タスク別履歴一覧 (``limit`` / ``offset`` は任意、絞る前の件数は ``X-Total-Count``)"""
    logger.debug("GET /api/aux-prompts/%s/history", task)
    mgr = _require_manager(state)

    try:
        history = mgr.get_history(task)
    except ValueError:
        raise _unknown_task_error(task)
    return paginate(history, limit, offset, response)


@router.get("/{task}/history/{version}")
async def get_aux_prompt_history_version(
    task: str,
    version: int = Path(ge=1, le=HISTORY_VERSION_MAX),
    state: AppState = Depends(get_app_state),
):
    """履歴の版の本文 (巻き戻す前に旧版を読む。読むだけ)"""
    logger.debug("GET /api/aux-prompts/%s/history/%d", task, version)
    mgr = _require_manager(state)

    try:
        entry = mgr.get_history_version(task, version)
    except ValueError:
        raise _unknown_task_error(task)
    except FileNotFoundError:
        raise _version_not_found_error(task, version)

    return {"task": task, **entry}


@router.post("/{task}/rollback")
async def rollback_aux_prompt(task: str, body: RollbackRequest, state: AppState = Depends(get_app_state)):
    """過去バージョンへのロールバック"""
    logger.debug("POST /api/aux-prompts/%s/rollback: version=%d", task, body.version)
    mgr = _require_manager(state)

    try:
        mgr.rollback(task, body.version)
    except ValueError:
        raise _unknown_task_error(task)
    except FileNotFoundError:
        raise _version_not_found_error(task, body.version)

    return {"status": "rolled_back", "version": mgr.get_meta(task).version}
