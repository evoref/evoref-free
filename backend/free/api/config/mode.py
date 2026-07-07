"""モード切替 API"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.app_state import AppState, get_app_state
from backend.config import get_mode_assist_model_path, get_mode_generation_params
from backend.log_config import get_logger

logger = get_logger("api.mode")

router = APIRouter(prefix="/api/mode", tags=["mode"])


class ModeSwitchRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(chat|coding)$")


class ModeSwitchResponse(BaseModel):
    mode: str
    model_changed: bool
    restart_initiated: bool
    assist_model_changed: bool = False
    assist_restart_initiated: bool = False
    message: str = ""


@router.post("/switch", response_model=ModeSwitchResponse)
async def switch_mode(
    req: ModeSwitchRequest,
    state: AppState = Depends(get_app_state),
):
    """モードを切り替える

    旧モードと新モードのモデルパスを比較し、
    異なる場合は model_changed=True を返す。アシストモデルも
    ``model_paths.assist_coding_model`` が設定されていれば同様に比較し、
    base と同一リクエスト内で並列に再起動する。
    """
    import asyncio

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

    old_assist_path = get_mode_assist_model_path(old_mode)
    new_assist_path = get_mode_assist_model_path(new_mode)
    assist_model_changed = old_assist_path != new_assist_path

    # モード状態を更新
    state.current_mode = new_mode

    restart_initiated = False
    assist_restart_initiated = False
    message = f"switched from {old_mode} to {new_mode}"

    # base / assist の再起動は同一リクエスト内で並列実行する。逐次だと
    # 最悪 (health_timeout × 2) になり CLI 側 read timeout を超過しうるため、
    # 別ポート・別プロセスで資源競合しない両者を asyncio.gather で並走させる。
    restart_tasks: dict[str, asyncio.Task[bool]] = {}
    if model_changed:
        restart_tasks["base"] = asyncio.create_task(
            _restart_base_server(state, new_params["model"]),
        )
    if assist_model_changed:
        restart_tasks["assist"] = asyncio.create_task(
            _restart_assist_server(state, new_assist_path),
        )
    if restart_tasks:
        results = await asyncio.gather(*restart_tasks.values())
        outcome = dict(zip(restart_tasks.keys(), results, strict=True))
        restart_initiated = outcome.get("base", False)
        assist_restart_initiated = outcome.get("assist", False)

    if model_changed:
        message += ", server restart initiated" if restart_initiated else ", server restart failed"
    if assist_model_changed:
        message += (
            ", assist restart initiated" if assist_restart_initiated
            else ", assist restart failed"
        )

    logger.info(
        "Mode switched: %s -> %s (model_changed=%s, restart=%s, "
        "assist_model_changed=%s, assist_restart=%s)",
        old_mode, new_mode, model_changed, restart_initiated,
        assist_model_changed, assist_restart_initiated,
    )

    return ModeSwitchResponse(
        mode=new_mode,
        model_changed=model_changed,
        restart_initiated=restart_initiated,
        assist_model_changed=assist_model_changed,
        assist_restart_initiated=assist_restart_initiated,
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


async def _restart_assist_server(
    state: AppState, model_path: str,
) -> bool:
    """アシストサーバーを新モデルで再起動する (``_restart_base_server`` と対称)

    ``config.yaml`` / ``local/model_state.json`` は一切変更しない
    (``model_paths.coding_model`` と同じ、モード切替限定のエフェメラル
    override 方式)。既存の ``POST /api/model/assist/migrate``
    (``ModelMigrator.migrate_component``) は使わない —
    ``process_manager.enabled`` (既定 false) に依存し、false だと config
    だけ書き換わってプロセスが再起動されないギャップがあるため。

    失敗時は旧モデルへの再スポーンを試みない (config を元々変更していない
    ため巻き戻す対象が無く、``assist_client=None`` の degraded mode へ
    素直に着地させる方が既存不変則 (呼出元は None 安全) と整合する)。

    Returns:
        再起動が成功したかどうか
    """
    import asyncio
    import copy
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

    # 1. 既存プロセスを確実に停止 (base と同じ 3 経路フォールバック)。
    await asyncio.to_thread(stop_server_process, "assist", cfg, state)

    # クライアント参照を即座に None にする。旧クライアントで接続失敗を
    # 待たせるより、各呼出元の既存 degraded mode フォールバックへ
    # 速やかに乗せた方が切替中のチャット応答パスの待機時間が短い。
    if state.assist_client:
        await state.assist_client.aclose()
    state.set_assist_client(None)

    # 2. 旧サーバが port を解放するまで待ってから再 spawn。
    await asyncio.to_thread(wait_port_released, "assist", cfg, 10.0)

    # 3. model_override 付きで新プロセス起動 (config.yaml は不変)。
    managed = _spawn_server_with_override("assist", cfg, model_override=model_path)
    if managed is None:
        logger.error("Failed to spawn assist server with model override")
        return False

    # 4. ヘルスチェック。
    expected_model_id = Path(model_path).name
    health_timeout = int((cfg.get("process_manager") or {}).get("health_timeout", 120))
    healthy = await asyncio.to_thread(
        wait_for_health, managed.host, managed.port, health_timeout, expected_model_id,
    )

    if not healthy:
        logger.error(
            "Assist server health check failed/timed out after model switch "
            "(expected model=%s)", expected_model_id,
        )
        await asyncio.to_thread(stop_server_process, "assist", cfg, state)
        return False

    # 5. クライアント再接続。``model_paths.assist_model`` を実際にロードした
    #    パスへ差し替えた deep copy を渡す — ``AssistModelClient`` のコンストラクタ
    #    は ``resolve_context_size``/``resolve_reasoning_mode`` をこのキーから
    #    解決するため、実体 (assist_coding_model) と設定 (assist_model) の
    #    不一致で誤った reasoning_budget/enable_thinking を送信しないようにする。
    #    ``config.yaml`` 自体 (get_config() のグローバル state) は変更しない。
    cfg_for_reconnect = copy.deepcopy(cfg)
    cfg_for_reconnect.setdefault("model_paths", {})["assist_model"] = model_path
    await _try_reconnect("assist", state, cfg_for_reconnect)

    return True
