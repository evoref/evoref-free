"""サーバー管理 API (`/api/servers/*`): GUI 用の互換層 (c_06 §2.8)

設定画面 (``frontend/src/lib/free/api/server.ts``) がベース / 埋め込み / リランカーの
llama-server を個別に起動・停止する。プロセスの所有者は
:class:`~backend.free.core.llama_process_manager.LlamaProcessManager` (`/api/model/process/*`、
c_06 §2.6) で、ここは URL と応答形 (:class:`ServerActionResponse`、失敗も HTTP 200 +
``success: false``) を保ったまま manager へ委譲するだけ。CLI は `/api/model/process/*` を使う。

サーバ名は ``base`` / ``embed`` / ``rerank`` (component の ``embedding`` との写像は
``llama_process_manager.component_of`` の 1 か所)。
"""

from __future__ import annotations

from typing import Literal

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.config import get_config
from backend.free.api.model import process_service
from backend.free.core.llama_process_manager import (
    ProcessAlreadyRunningError,
    ProcessManagerError,
    ProcessNotConfiguredError,
    ProcessNotManagedError,
    ProcessPortBusyError,
    ProcessStartTimeoutError,
    component_of,
)
from backend.free.core.llama_process_manager import _resolve_endpoint as _component_endpoint
from backend.log_config import get_logger

logger = get_logger("api.server_control")

router = APIRouter(prefix="/api/servers", tags=["server-control"])

ServerName = Literal["base", "embed", "rerank"]


# ────────────────────────────────────────────
# ヘルスチェック (status.py の rerank 接続表示も使う)
# ────────────────────────────────────────────

async def _check_health(host: str, port: int) -> bool:
    """llama-server の /health エンドポイントに問い合わせ"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://{host}:{port}/health", timeout=3.0,
            )
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _resolve_endpoint(name: ServerName, cfg: dict) -> tuple[str, int] | None:
    """サーバー名から (host, port) を config から直接解決する (enabled / モデルの有無を見ない)"""
    try:
        return _component_endpoint(component_of(name), cfg)
    except ProcessManagerError:
        return None


# ────────────────────────────────────────────
# API スキーマ (frontend の ServerActionResponse と同じ形。変えない)
# ────────────────────────────────────────────

class ServerActionRequest(BaseModel):
    name: ServerName
    force: bool = False


class ServerActionResponse(BaseModel):
    name: str
    action: str
    success: bool
    message: str = ""


def _reply(name: str, action: str, success: bool, message: str) -> ServerActionResponse:
    return ServerActionResponse(name=name, action=action, success=success, message=message)


# ────────────────────────────────────────────
# エンドポイント
# ────────────────────────────────────────────

@router.post("/start", response_model=ServerActionResponse)
async def start_server(
    req: ServerActionRequest,
    state: AppState = Depends(get_app_state),
):
    """指定サーバーの llama-server プロセスを起動 (``force`` は動いているものを止めて起こし直す)"""
    name, force = req.name, req.force
    if state.llama_manager is None:
        return _reply(name, "start", False, "process manager not initialized")
    component = component_of(name)
    cfg = get_config()
    try:
        await process_service.start_component(state, component, cfg, force=force)
    except ProcessAlreadyRunningError:
        return _reply(name, "start", False, "already running")
    except ProcessPortBusyError as e:
        if not e.is_llama:
            return _reply(name, "start", False, str(e))
        # 外部起動 (evoref serve / evoref-ctl) の llama-server — 二重に起こさず接続し直す
        host, port = _component_endpoint(component, cfg)
        if await _check_health(host, port):
            await process_service.reconnect_clients(state, component, cfg)
            return _reply(name, "start", True, "already healthy, reconnected")
        return _reply(name, "start", False, "port in use by an external llama-server that is not healthy yet")
    except ProcessNotConfiguredError:
        return _reply(name, "start", False, "not configured")
    except ProcessStartTimeoutError:
        return _reply(name, "start", False, "health check timed out")
    except ProcessManagerError as e:
        logger.warning("server_control: start %s failed: %s", name, e)
        return _reply(name, "start", False, str(e))
    return _reply(name, "start", True, "force restarted" if force else "started")


@router.post("/stop", response_model=ServerActionResponse)
async def stop_server(
    req: ServerActionRequest,
    state: AppState = Depends(get_app_state),
):
    """指定サーバーの llama-server プロセスを停止 (このバックエンドが起こしたもの、無ければ外部の llama-server)"""
    name = req.name
    if state.llama_manager is None:
        return _reply(name, "stop", False, "process manager not initialized")
    try:
        result = await process_service.stop_component(state, component_of(name), get_config())
    except ProcessNotManagedError:
        return _reply(name, "stop", False, "no process listening on port")
    except ProcessManagerError as e:
        return _reply(name, "stop", False, str(e))
    if result.how == "managed":
        return _reply(name, "stop", True, "stopped")
    return _reply(name, "stop", True, f"stopped external process pid={result.pid}")
