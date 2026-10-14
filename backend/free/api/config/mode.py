"""モード切替 API"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.app_state import AppState, get_app_state
from backend.config import (
    get_config,
    get_mode_generation_params,
    get_path_resolver,
    get_project_root,
)
from backend.free.api._error_responses import api_error
from backend.log_config import get_logger

logger = get_logger("api.mode")

router = APIRouter(prefix="/api/mode", tags=["mode"])


class ModeSwitchRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(chat|create)$")


class ModeSwitchResponse(BaseModel):
    mode: str
    model_changed: bool
    restart_initiated: bool
    message: str = ""


#: モード切替を直列化する (再起動の途中に別の切替が状態とサーバを取り違えないように)。
_switch_lock = asyncio.Lock()


@router.post("/switch", response_model=ModeSwitchResponse)
async def switch_mode(
    req: ModeSwitchRequest,
    state: AppState = Depends(get_app_state),
):
    """モードを切り替える

    旧モードと新モードのモデルパスを比較し、異なる場合は
    ``model_changed=True`` を返してベース llama-server を再起動する。
    再起動に失敗したらモードを旧モードへ戻して 503 ``api.mode_switch_failed`` を返す
    (同じ切替をもう一度要求できる)。
    """
    async with _switch_lock:
        return await _switch_mode_locked(req.mode, state)


async def _switch_mode_locked(new_mode: str, state: AppState) -> ModeSwitchResponse:
    old_mode = state.current_mode

    if old_mode == new_mode:
        return ModeSwitchResponse(
            mode=new_mode,
            model_changed=False,
            restart_initiated=False,
            message="already in requested mode",
        )

    # 旧モードと新モードのモデルパスを比較
    old_params = get_mode_generation_params(old_mode)
    new_params = get_mode_generation_params(new_mode)
    model_changed = old_params["model"] != new_params["model"]

    # モード状態を更新。resolver 側の active_mode も同期する。
    # 学習パーティション全体の束ね直しはしない (f_04 §1.2.0)。経験・プロンプト・
    # aux 較正は書込み・読込みのたびに「載っているモデル」(/props) の model_key で
    # 置き場を決める — チャットと create の並行、Level 1 実行中、書き手スレッドに
    # 積まれた未書出しの経験を取り違えないため。
    state.current_mode = new_mode
    get_path_resolver().set_active_mode(new_mode)

    restart_initiated = False
    message = f"switched from {old_mode} to {new_mode}"

    if model_changed:
        from backend.free.core.llama_process_manager import ProcessPortBusyError

        refused = False
        try:
            restart_initiated = await _restart_base_server(state, new_params["model"])
        except ProcessPortBusyError as e:
            # 何も止めていない (ポートが別アプリ)。旧サーバは動いたまま
            logger.error("Base server restart refused during mode switch: %s", e)
            refused = True
        except Exception as e:
            logger.error("Base server restart raised during mode switch: %s", e)
            restart_initiated = False
        if not restart_initiated:
            state.current_mode = old_mode
            get_path_resolver().set_active_mode(old_mode)
            # 旧サーバは新モデルの起動前に止めている。旧モデルで 1 回だけ起動し直す
            # (失敗しても再試行・再ロールバックはしない。停止中は server_running=false で返す)。
            # 拒否された回は止めていないので起こし直さない
            server_running = refused or await _restore_previous_server(state, old_params["model"])
            logger.error(
                "Mode switch %s -> %s failed (base server restart failed); rolled back to %s "
                "(previous model server running=%s)",
                old_mode, new_mode, old_mode, server_running,
            )
            raise api_error(
                503, "E0503",
                f"Mode switch to {new_mode} failed: base server restart failed. "
                f"Mode was rolled back to {old_mode}; "
                f"the {old_mode} model server is {'running' if server_running else 'stopped'}.",
                "api.mode_switch_failed",
                mode=new_mode, previous=old_mode, server_running=server_running,
            )
        message += ", server restart initiated"

    logger.info(
        "Mode switched: %s -> %s (model_changed=%s, restart=%s)",
        old_mode, new_mode, model_changed, restart_initiated,
    )

    return ModeSwitchResponse(
        mode=new_mode,
        model_changed=model_changed,
        restart_initiated=restart_initiated,
        message=message,
    )


async def _restore_previous_server(state: AppState, model_path: str) -> bool:
    """切替の失敗後に旧モードのモデルで base を起動し直す (1 回だけ。例外は False)。"""
    try:
        return bool(await _restart_base_server(state, model_path))
    except Exception as e:
        logger.error("Restarting the previous base model failed: %s", e)
        return False


async def _restart_base_server(state: AppState, model_path: str) -> bool:
    """ベースサーバーを新モデルで起動し直す (所有者の ``LlamaProcessManager`` 経由、c_06 §2.6)

    動いている base (このバックエンドが起こしたもの、または外部起動の llama-server) を止め、
    ポートが空いてから ``model_override`` で起こす。health 待ちは ``process_manager.health_timeout``
    (整数は明示値、auto は GGUF サイズ連動)。起動前にポートを空けて確かめるので、旧サーバの
    200 を新モデルと取り違えない。

    Returns:
        再起動が成功したかどうか
    Raises:
        ProcessPortBusyError: 何も止めずに拒否した (ポートが別アプリ)。
            旧サーバはそのまま動いている
    """
    from backend.free.api.model.process_service import start_component
    from backend.free.core.llama_process_manager import ProcessManagerError, ProcessPortBusyError

    if state.llama_manager is None:
        logger.error("LlamaProcessManager not initialized; cannot restart the base server")
        return False
    cfg = get_config()
    try:
        await start_component(state, "base", cfg, force=True, model_override=model_path)
    except ProcessPortBusyError:
        raise
    except ProcessManagerError as e:
        logger.error("Base server restart failed after model switch (model=%s): %s", model_path, e)
        return False

    # 実際に載ったモデルをこのモードのモデルと照合する (c_05 §0.5.7)
    from backend.factory._served_model import check_served_model

    check_served_model(state, model_path, get_project_root())

    return True


