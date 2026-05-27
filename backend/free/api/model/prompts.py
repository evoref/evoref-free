"""プロンプト管理 API エンドポイント"""

import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.free.api.model._prompt_helpers import (
    learning_running_error,
    prompt_detail_dict,
    prompt_file_not_found_error,
    prompt_summary_dict,
    require_prompt_manager,
    unknown_mode_error,
    unsupported_locale_error,
)
from backend.free.learning.level1_session import PriorityRequest
from backend.log_config import get_logger

logger = get_logger("api.prompts")

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


class PromptUpdateRequest(BaseModel):
    content: str


class RollbackRequest(BaseModel):
    version: int


class PromptLocaleRequest(BaseModel):
    locale: str  # "ja" | "en"


# --- locale エンドポイント（/{mode} より前に定義して FastAPI のルート優先順位を確保） ---


@router.post("/locale")
async def switch_prompt_locale(
    body: PromptLocaleRequest,
    state: AppState = Depends(get_app_state),
):
    """プロンプト言語を切替え、自動再学習をスケジュール"""
    logger.info("POST /api/prompts/locale: locale=%s", body.locale)
    mgr = require_prompt_manager(state)

    if body.locale not in ("ja", "en"):
        raise unsupported_locale_error(body.locale)

    # 学習実行中は切替不可
    scheduler = state.learning_scheduler
    if scheduler is not None and scheduler.running:
        raise learning_running_error()

    # プロンプト切替
    try:
        versions = mgr.switch_locale(body.locale)
    except ValueError:
        raise unsupported_locale_error(body.locale)

    # config.yaml の prompt_locale を更新
    from backend.config import get_config, save_config_section
    config = get_config()
    i18n_section = dict(config.get("i18n", {}))
    i18n_section["prompt_locale"] = body.locale
    save_config_section("i18n", i18n_section)

    # 優先 Level 1 再学習を優先キューへ push
    # f_04 §5.4 に従い、`prompt_locale_switch` 理由で relax_ratio=0.5 を要求する。
    # キューは永続化されるため LLM 未接続でも次回 tick で実行される。
    triggered = False
    queue_length = 0
    if scheduler is not None:
        req = PriorityRequest(
            reason="prompt_locale_switch",
            requested_at=time.time(),
            relax_ratio=0.5,
            payload={"locale": body.locale},
        )
        try:
            queue_length = scheduler.push_priority_request(req)
            triggered = True
            logger.info(
                "Priority request pushed after locale switch: locale=%s, queue_len=%d",
                body.locale, queue_length,
            )
        except Exception as e:
            logger.error("push_priority_request failed for locale switch: %s", e)

    return {
        "status": "switched",
        "locale": body.locale,
        "versions": versions,
        "relearning_triggered": triggered,
        "queue_length": queue_length,
    }


@router.get("")
async def list_prompts(state: AppState = Depends(get_app_state)):
    """全モードのプロンプト一覧"""
    logger.debug("GET /api/prompts")
    mgr = require_prompt_manager(state)
    return [
        prompt_summary_dict(mode, mgr.get_meta(mode), mgr.get_raw_prompt(mode))
        for mode in mgr.MODES
    ]


@router.get("/{mode}")
async def get_prompt(mode: str, state: AppState = Depends(get_app_state)):
    """モード別プロンプト詳細"""
    logger.debug("GET /api/prompts/%s", mode)
    mgr = require_prompt_manager(state)

    try:
        meta = mgr.get_meta(mode)
        content = mgr.get_raw_prompt(mode)
    except ValueError:
        raise unknown_mode_error(mode)

    return prompt_detail_dict(mode, meta, content)


@router.put("/{mode}")
async def update_prompt(mode: str, body: PromptUpdateRequest, state: AppState = Depends(get_app_state)):
    """プロンプト更新"""
    logger.debug("PUT /api/prompts/%s: content_len=%d", mode, len(body.content))
    mgr = require_prompt_manager(state)

    try:
        mgr.update_manual(mode, body.content)
    except ValueError:
        raise unknown_mode_error(mode)

    return {"status": "updated", "version": mgr.get_meta(mode).version}


@router.post("/{mode}/reload")
async def reload_prompt(mode: str, state: AppState = Depends(get_app_state)):
    """ディスクから再読込み"""
    logger.debug("POST /api/prompts/%s/reload", mode)
    mgr = require_prompt_manager(state)

    try:
        mgr.reload(mode)
    except ValueError:
        raise unknown_mode_error(mode)
    except FileNotFoundError as e:
        raise prompt_file_not_found_error(str(e))

    return {"status": "reloaded"}


@router.get("/{mode}/history")
async def get_history(mode: str, state: AppState = Depends(get_app_state)):
    """履歴一覧"""
    logger.debug("GET /api/prompts/%s/history", mode)
    mgr = require_prompt_manager(state)

    try:
        return mgr.get_history(mode)
    except ValueError:
        raise unknown_mode_error(mode)


@router.post("/{mode}/rollback")
async def rollback(mode: str, body: RollbackRequest, state: AppState = Depends(get_app_state)):
    """過去バージョンへのロールバック"""
    logger.debug("POST /api/prompts/%s/rollback: version=%d", mode, body.version)
    mgr = require_prompt_manager(state)

    try:
        mgr.rollback(mode, body.version)
    except ValueError:
        raise unknown_mode_error(mode)
    except FileNotFoundError as e:
        raise prompt_file_not_found_error(str(e))

    return {"status": "rolled_back", "version": mgr.get_meta(mode).version}
