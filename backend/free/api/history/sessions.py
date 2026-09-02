"""セッション管理 API — CLI セッションの登録・解除・一覧"""

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.free.api.chat.chat_recorder import end_session
from backend.free.api.history._session_helpers import (
    build_session_list_response,
    session_already_registered_error,
)
from backend.free.api.schemas import (
    SessionListResponse,
    SessionRegisterRequest,
    SessionRegisterResponse,
    SessionUnregisterResponse,
)
from backend.log_config import get_logger

logger = get_logger("api.sessions")

router = APIRouter(prefix="/api", tags=["sessions"])


@router.post("/sessions/register", response_model=SessionRegisterResponse)
async def register(req: SessionRegisterRequest, state: AppState = Depends(get_app_state)):
    """セッションを登録（重複は 409 Conflict）"""
    logger.debug(
        "POST /api/sessions/register: session_id=%s, mode=%s, client_type=%s",
        req.session_id, req.mode, req.client_type,
    )
    ok = state.register_session(req.session_id, req.mode, req.client_type)
    if not ok:
        logger.warning("Session ID already registered: %s", req.session_id)
        raise session_already_registered_error(req.session_id)
    logger.debug("Session registered: %s", req.session_id)
    return SessionRegisterResponse(registered=True, session_id=req.session_id)


@router.delete("/sessions/{session_id}", response_model=SessionUnregisterResponse)
async def unregister(session_id: str, state: AppState = Depends(get_app_state)):
    """セッションを登録解除"""
    logger.debug("DELETE /api/sessions/%s", session_id)
    # セッション終了 = 会話終了。経験に conversation_ended を反映 (Level 2 base=C
    # positive 抽出用)。CLI/フロントの session 解除で確実に発火させる。disabled 時は
    # FeedbackCollector 内ガードで no-op。
    if state.feedback_collector is not None:
        state.feedback_collector.mark_conversation_ended()
    # セッション別 WM を STM へ流して台帳から外す (f_02 §1.2 経路 (b))。
    # WM がセッション別になったので、旧セッションの後始末はここが担う。
    if end_session(state, session_id):
        logger.debug("Working memory released for session %s", session_id)
    ok = state.unregister_session(session_id)
    if not ok:
        logger.debug("Session not found for unregister: %s", session_id)
    return SessionUnregisterResponse(unregistered=ok, session_id=session_id)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(state: AppState = Depends(get_app_state)):
    """アクティブセッション一覧"""
    logger.debug("GET /api/sessions")
    sessions = state.get_active_sessions()
    return build_session_list_response(sessions)
