"""モード切替 API"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.app_state import AppState, get_app_state
from backend.config import get_mode_generation_params
from backend.log_config import get_logger

logger = get_logger("api.mode")

router = APIRouter(prefix="/api/mode", tags=["mode"])


class ModeSwitchRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(chat|coding)$")


class ModeSwitchResponse(BaseModel):
    mode: str
    model_changed: bool
    restart_initiated: bool
    message: str = ""


@router.post("/switch", response_model=ModeSwitchResponse)
async def switch_mode(
    req: ModeSwitchRequest,
    state: AppState = Depends(get_app_state),
):
    """モードを切り替える

    旧モードと新モードのモデルパスを比較し、
    異なる場合は model_changed=True を返す。
    """
    old_mode = state.current_mode
    new_mode = req.mode

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

    # モード状態を更新
    state.current_mode = new_mode

    restart_initiated = False
    message = f"switched from {old_mode} to {new_mode}"

    # モデル変更がある場合は再起動を実行
    if model_changed:
        restart_initiated = await _restart_base_server(state, new_params["model"])
        if restart_initiated:
            message += ", server restart initiated"
        else:
            message += ", server restart failed"

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


async def _restart_base_server(
    state: AppState, model_path: str,
) -> bool:
    """ベースサーバーを新モデルで再起動する

    Returns:
        再起動が成功したかどうか
    """
    import asyncio
    from pathlib import Path

    from backend.config import get_config
    from backend.free.api.system.server_control import (
        stop_server_process,
        wait_port_released,
        _spawn_server_with_override,
        _try_reconnect,
    )
    from scripts.launch_llama import wait_for_health

    cfg = get_config()

    # 1. 既存プロセスを確実に停止。``_stop_server`` は本バックエンドが spawn して
    #    ``_managed`` に登録したプロセスしか kill しないが、base は通常 CLI /
    #    launch_llama / evoref-ctl 経由の外部プロセスとして起動され ``_managed`` に
    #    入らない。素の ``_stop_server`` だと no-op になり旧サーバが :8080 に残留し、
    #    後続の health/reconnect が旧モデルを掴む (モデル切替が効かない根因)。
    #    ``stop_server_process`` は _managed → LlamaProcessManager → 外部ポート占有
    #    kill の 3 経路を踏むため外部 base も確実に停止する (内部で netstat/taskkill
    #    がブロッキングなので別スレッドへ退避)。
    await asyncio.to_thread(stop_server_process, "base", cfg, state)

    # AppState のクライアント参照をクリア
    if state.local_client:
        await state.local_client.aclose()
    state.local_client = None
    if state.llm_client:
        state.llm_client.local = None

    # 2. 旧サーバが port を解放するまで待ってから再 spawn (graceful shutdown 残響で
    #    旧サーバの 200 を拾う窓を潰す)。
    await asyncio.to_thread(wait_port_released, "base", cfg, 10.0)

    # 3. model_override 付きで新プロセス起動
    managed = _spawn_server_with_override("base", cfg, model_override=model_path)
    if managed is None:
        logger.error("Failed to spawn base server with model override")
        return False

    # 4. ヘルスチェック。``/health`` 200 だけでなく ``/props`` の load 済みモデルが
    #    要求モデルと一致するまで待つ (旧サーバの 200 やロード途中を成功と誤判定
    #    しない)。大きい coding GGUF は load に時間がかかるためタイムアウトは
    #    process_manager.health_timeout (既定 120s) を採用。
    expected_model_id = Path(model_path).name
    health_timeout = int((cfg.get("process_manager") or {}).get("health_timeout", 120))
    healthy = await asyncio.to_thread(
        wait_for_health, managed.host, managed.port, health_timeout, expected_model_id,
    )

    if not healthy:
        logger.error(
            "Base server health check failed/timed out after model switch "
            "(expected model=%s)", expected_model_id,
        )
        await asyncio.to_thread(stop_server_process, "base", cfg, state)
        return False

    # 5. クライアント再接続
    await _try_reconnect("base", state, cfg)

    return True
