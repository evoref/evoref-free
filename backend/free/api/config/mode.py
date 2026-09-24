"""モード切替 API"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from pathlib import Path

from backend.app_state import AppState, get_app_state
from backend.config import (
    get_config,
    get_mode_generation_params,
    get_path_resolver,
    get_project_root,
)
from backend.free.core.launch_adapters import LaunchAdapters, adapters_for_launch
from backend.log_config import get_logger

logger = get_logger("api.mode")

router = APIRouter(prefix="/api/mode", tags=["mode"])


class ModeSwitchRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(chat|create)$")


class ModeSwitchResponse(BaseModel):
    mode: str
    model_changed: bool
    restart_initiated: bool
    lora_changed: bool = False
    message: str = ""


@router.post("/switch", response_model=ModeSwitchResponse)
async def switch_mode(
    req: ModeSwitchRequest,
    state: AppState = Depends(get_app_state),
):
    """モードを切り替える

    旧モードと新モードのモデルパスを比較し、異なる場合は
    ``model_changed=True`` を返してベース llama-server を再起動する。
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

    # 当てるアダプタ (Pro の LoRA / control vector) の比較。Free では常に無し
    # なので差が出ず、再起動トリガーに影響しない。
    cfg = get_config()
    new_adapters = adapters_for_launch(cfg, get_project_root(), new_mode)
    lora_changed = adapters_for_launch(cfg, get_project_root(), old_mode) != new_adapters

    # モード状態を更新。resolver 側の active_mode も同期する (ダッシュボード等の
    # 「現在モードのアダプタ」既定表示に使われる、Level2Runner は mode を明示
    # 引数で受け取るためこれには依存しない)。
    state.current_mode = new_mode
    get_path_resolver().set_active_mode(new_mode)

    restart_initiated = False
    message = f"switched from {old_mode} to {new_mode}"

    if model_changed or lora_changed:
        restart_initiated = await _restart_base_server(
            state, new_params["model"], adapters=new_adapters,
        )
        message += (
            ", server restart initiated" if restart_initiated
            else ", server restart failed"
        )

    logger.info(
        "Mode switched: %s -> %s (model_changed=%s, lora_changed=%s, restart=%s)",
        old_mode, new_mode, model_changed, lora_changed, restart_initiated,
    )

    return ModeSwitchResponse(
        mode=new_mode,
        model_changed=model_changed,
        restart_initiated=restart_initiated,
        lora_changed=lora_changed,
        message=message,
    )


async def _restart_base_server(
    state: AppState, model_path: str, adapters: LaunchAdapters,
) -> bool:
    """ベースサーバーを新モデルで再起動する

    ``adapters`` は新モードの ``adapters_for_launch`` の結果 (存在・互換確認済み)。

    Returns:
        再起動が成功したかどうか
    """
    import asyncio

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

    # 3. model_override / adapters 付きで新プロセス起動
    managed = _spawn_server_with_override(
        "base", cfg, model_override=model_path, adapters=adapters,
    )
    if managed is None:
        logger.error("Failed to spawn base server with model override")
        return False

    # 4. ヘルスチェック。``/health`` 200 だけでなく ``/props`` の load 済みモデルが
    #    要求モデルと一致するまで待つ (旧サーバの 200 やロード途中を成功と誤判定
    #    しない)。大きい create GGUF は load に時間がかかるためタイムアウトは
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

    # 6. 実際に載ったモデルをこのモードのモデルと照合する (c_05 §0.5.7)
    from backend.factory._served_model import check_served_model

    check_served_model(state, model_path, get_project_root())

    return True


