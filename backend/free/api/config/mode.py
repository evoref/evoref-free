"""モード切替 API"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from pathlib import Path

from backend.app_state import AppState, get_app_state
from backend.config import (
    get_mode_assist_lora_path,
    get_mode_assist_model_path,
    get_mode_generation_params,
    get_mode_lora_path,
    get_path_resolver,
    get_project_root,
)
from backend.free.core.session_mode import is_create_mode
from backend.log_config import get_logger

logger = get_logger("api.mode")

router = APIRouter(prefix="/api/mode", tags=["mode"])


class ModeSwitchRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(chat|create)$")


class ModeSwitchResponse(BaseModel):
    mode: str
    model_changed: bool
    restart_initiated: bool
    assist_model_changed: bool = False
    assist_restart_initiated: bool = False
    lora_changed: bool = False
    assist_lora_changed: bool = False
    message: str = ""


def _resolve_lora_override(
    model_path: str, lora_path: Path | None,
) -> tuple[Path | None, bool]:
    """``get_mode_lora_path``/``get_mode_assist_lora_path`` の解決結果を、
    実際に ``--lora`` へ渡してよいか (存在確認 + arch 互換) 検証する。

    ``level2_adapter_partition!="model_mode"`` (既定) または ``lora_path`` が
    None のときは ``(None, True)`` を返し、``build_*_cmd`` 側の従来どおりの
    flat フォールバック分岐に委譲する (既存挙動を完全維持)。

    "model_mode" のときは、対象モードのアダプタが (a) 存在しない、または
    (b) 起動対象モデルと arch 不一致であれば ``(None, False)`` を返し、
    flat フォールバックも踏ませずに明示的に「LoRA なし」で起動させる —
    そのモードがまだ学習されていない状態を安全に表す。
    """
    resolver = get_path_resolver()
    if resolver.adapter_partition_mode != "model_mode" or lora_path is None:
        return None, True
    if not lora_path.exists():
        return None, False

    from scripts.launch_llama import lora_compatible_with_model

    model_abs = Path(model_path)
    if not model_abs.is_absolute():
        model_abs = get_project_root() / model_abs
    ok, reason = lora_compatible_with_model(model_abs, lora_path)
    if not ok:
        logger.warning(
            "Mode switch: LoRA %s incompatible with model %s (%s); starting without LoRA",
            lora_path, model_abs, reason,
        )
        return None, False
    return lora_path, False


@router.post("/switch", response_model=ModeSwitchResponse)
async def switch_mode(
    req: ModeSwitchRequest,
    state: AppState = Depends(get_app_state),
):
    """モードを切り替える

    旧モードと新モードのモデルパスを比較し、
    異なる場合は model_changed=True を返す。アシストモデルも
    ``model_paths.assist_create_model`` が設定されていれば同様に比較し、
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

    # base/assist LoRA パスの比較 (level2_adapter_partition=="model_mode" の
    # ときのみ差が出る。既定 "model" では get_mode_lora_path は常に同一パスを
    # 返すため lora_changed は常に False = 再起動トリガーへの影響なし)。
    new_lora_path = get_mode_lora_path(new_mode)
    lora_changed = get_mode_lora_path(old_mode) != new_lora_path
    new_assist_lora_path = get_mode_assist_lora_path(new_mode)
    assist_lora_changed = get_mode_assist_lora_path(old_mode) != new_assist_lora_path

    # モード状態を更新。resolver 側の active_mode も同期する (ダッシュボード等の
    # 「現在モードのアダプタ」既定表示に使われる、Level2Runner は mode を明示
    # 引数で受け取るためこれには依存しない)。
    state.current_mode = new_mode
    get_path_resolver().set_active_mode(new_mode)

    restart_initiated = False
    assist_restart_initiated = False
    message = f"switched from {old_mode} to {new_mode}"

    # アシストの常駐 (docs/c_14 §1.2)。``residency: on_demand`` では chat の間
    # アシストは停止しているので、create へ入る時点で起動して create の間ずっと
    # 常駐させる (1 ターン目からモデルロード待ちを出さない)。chat へ戻る時に解放。
    # ``assist_create_model`` が設定されていれば下の ``_restart_assist_server`` が
    # model_override 付きで起こすので、ここでは二重に spawn しない。
    residency = getattr(state, "assist_residency", None)
    # アシストを再起動すべきか。``assist_model_changed`` / ``assist_lora_changed``
    # は応答フィールドとして事実をそのまま返すため書き換えず、再起動可否だけを
    # 別変数で持つ。
    restart_assist = assist_model_changed or assist_lora_changed
    if residency is not None and residency.on_demand:
        if is_create_mode(new_mode):
            if restart_assist:
                # 起動は _restart_assist_server が行う。状態だけ引き継ぐ。
                residency.note_external_start("create_mode")
            elif not await residency.acquire("create_mode"):
                message += ", assist start failed (create runs in degraded mode)"
        elif is_create_mode(old_mode):
            # chat へ戻るときは **停止するだけ**。ここで chat 用アシストを
            # 起こし直すと on_demand なのに常駐したままになる (2026-08-08 実機で
            # 観測: residency=stopped なのに :8081 が LISTEN のまま残った)。
            # 次のアイドル窓が chat 用モデルで起こすので、ここで温めておく必要は無い。
            restart_assist = False
            await residency.release("create_mode")

    # base / assist の再起動は同一リクエスト内で並列実行する。逐次だと
    # 最悪 (health_timeout × 2) になり CLI 側 read timeout を超過しうるため、
    # 別ポート・別プロセスで資源競合しない両者を asyncio.gather で並走させる。
    restart_tasks: dict[str, asyncio.Task[bool]] = {}
    if model_changed or lora_changed:
        restart_tasks["base"] = asyncio.create_task(
            _restart_base_server(state, new_params["model"], lora_path=new_lora_path),
        )
    if restart_assist:
        restart_tasks["assist"] = asyncio.create_task(
            _restart_assist_server(
                state, new_assist_path, lora_path=new_assist_lora_path,
            ),
        )
    if restart_tasks:
        results = await asyncio.gather(*restart_tasks.values())
        outcome = dict(zip(restart_tasks.keys(), results, strict=True))
        restart_initiated = outcome.get("base", False)
        assist_restart_initiated = outcome.get("assist", False)

    if model_changed or lora_changed:
        message += ", server restart initiated" if restart_initiated else ", server restart failed"
    if restart_assist:
        message += (
            ", assist restart initiated" if assist_restart_initiated
            else ", assist restart failed"
        )
    elif assist_model_changed and residency is not None and residency.on_demand:
        message += ", assist stopped (starts again on the next idle window)"

    logger.info(
        "Mode switched: %s -> %s (model_changed=%s, lora_changed=%s, restart=%s, "
        "assist_model_changed=%s, assist_lora_changed=%s, assist_restart=%s)",
        old_mode, new_mode, model_changed, lora_changed, restart_initiated,
        assist_model_changed, assist_lora_changed, assist_restart_initiated,
    )

    return ModeSwitchResponse(
        mode=new_mode,
        model_changed=model_changed,
        restart_initiated=restart_initiated,
        assist_model_changed=assist_model_changed,
        assist_restart_initiated=assist_restart_initiated,
        lora_changed=lora_changed,
        assist_lora_changed=assist_lora_changed,
        message=message,
    )


async def _restart_base_server(
    state: AppState, model_path: str, lora_path: Path | None = None,
) -> bool:
    """ベースサーバーを新モデルで再起動する

    ``lora_path`` は ``get_mode_lora_path(new_mode)`` の解決結果 (存在確認前)。
    "model" (既定) スキームでは常に ``None`` 相当 (呼出元の ``_resolve_lora_override``
    が ``adapter_partition_mode!="model_mode"`` を見て ``(None, True)`` に丸める
    ため、legacy flat フォールバックへ委譲され挙動は完全に従来通り)。

    Returns:
        再起動が成功したかどうか
    """
    import asyncio

    from backend.config import get_config
    from backend.free.api.system.server_control import (
        stop_server_process,
        wait_port_released,
        _spawn_server_with_override,
        _try_reconnect,
    )
    from scripts.launch_llama import wait_for_health

    cfg = get_config()
    lora_override, lora_fallback = _resolve_lora_override(model_path, lora_path)

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

    # 3. model_override / lora_override 付きで新プロセス起動
    managed = _spawn_server_with_override(
        "base", cfg, model_override=model_path,
        lora_override=lora_override, lora_fallback=lora_fallback,
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

    return True


async def _restart_assist_server(
    state: AppState, model_path: str, lora_path: Path | None = None,
) -> bool:
    """アシストサーバーを新モデルで再起動する (``_restart_base_server`` と対称)

    ``config.yaml`` / ``local/model_state.json`` は一切変更しない
    (``model_paths.create_model`` と同じ、モード切替限定のエフェメラル
    override 方式)。既存の ``POST /api/model/assist/migrate``
    (``ModelMigrator.migrate_component``) は使わない —
    ``process_manager.enabled`` (既定 false) に依存し、false だと config
    だけ書き換わってプロセスが再起動されないギャップがあるため。

    失敗時は旧モデルへの再スポーンを試みない (config を元々変更していない
    ため巻き戻す対象が無く、``assist_client=None`` の degraded mode へ
    素直に着地させる方が既存不変則 (呼出元は None 安全) と整合する)。

    ``lora_path`` は ``get_mode_assist_lora_path(new_mode)`` の解決結果。
    明示的な ``lora_override`` を渡すことで、既存の
    「``model_override`` 指定時は arch 不明な LoRA を付けない」安全策
    (``launch_llama.build_assist_cmd`` の ``elif model_override is None`` 分岐)
    を安全にバイパスする — 対象モードの LoRA は起動対象モデルとの arch 互換を
    ``_resolve_lora_override`` が事前検証済みのため。

    Returns:
        再起動が成功したかどうか
    """
    import asyncio
    import copy

    from backend.config import get_config
    from backend.free.api.system.server_control import (
        stop_server_process,
        wait_port_released,
        _spawn_server_with_override,
        _try_reconnect,
    )
    from scripts.launch_llama import wait_for_health

    cfg = get_config()
    lora_override, lora_fallback = _resolve_lora_override(model_path, lora_path)

    # 1. 既存プロセスを確実に停止 (base と同じ 3 経路フォールバック)。
    await asyncio.to_thread(stop_server_process, "assist", cfg, state)

    # 切替中は呼出を即拒否させ、各呼出元の既存 degraded フォールバックへ
    # 速やかに乗せる (旧クライアントで接続失敗を待たせるより待機時間が短い)。
    #
    # **クライアントオブジェクトは差し替えない**。``AssistModelClient`` は
    # _pillar_wirer が 20 箇所以上へコンストラクタ注入しており、
    # ``set_assist_client`` が同期するのは 4 つだけなので、None を挟むと残りが
    # 閉じた旧オブジェクトを掴んだままになる (docs/c_14 §1.2)。接続プールだけ
    # 捨て、可用性は residency のゲートで表現する。
    residency = getattr(state, "assist_residency", None)
    if residency is not None:
        residency.suspend("assist_model_swap")
    if state.assist_client:
        await state.assist_client.aclose()
    if residency is None:
        # residency 未配線 (テスト等) では従来どおり None 縮退に倒す。
        state.set_assist_client(None)

    # 2. 旧サーバが port を解放するまで待ってから再 spawn。
    await asyncio.to_thread(wait_port_released, "assist", cfg, 10.0)

    # 3. model_override / lora_override 付きで新プロセス起動 (config.yaml は不変)。
    managed = _spawn_server_with_override(
        "assist", cfg, model_override=model_path,
        lora_override=lora_override, lora_fallback=lora_fallback,
    )
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
        # suspend は解除する — 「差し替え中」ではなく「起動失敗」として
        # residency の状態 (stopped/failed) にそのまま表現させる。
        if residency is not None:
            residency.resume("assist_model_swap_failed")
        return False

    # 5. クライアント再接続。``model_paths.assist_model`` を実際にロードした
    #    パスへ差し替えた deep copy を渡す — ``AssistModelClient`` のコンストラクタ
    #    は ``resolve_context_size``/``resolve_reasoning_mode``/
    #    ``resolve_sampling_params`` をこのキーから解決するため、実体
    #    (assist_create_model) と設定 (assist_model) の不一致で誤った
    #    reasoning_budget/enable_thinking/sampling を送信しないようにする。
    #    モデル別プロファイル (models/profiles/by-model/) もこのキー経由で
    #    解決されるので、chat/create で別のアシストモデルを使う構成でも
    #    それぞれの宣言が正しく効く。
    #    ``config.yaml`` 自体 (get_config() のグローバル state) は変更しない。
    #    既存クライアントがある場合は ``rebind_model_config`` で同一オブジェクトの
    #    上に引き直す (注入済み参照を stale にしない、docs/c_14 §1.2)。
    cfg_for_reconnect = copy.deepcopy(cfg)
    cfg_for_reconnect.setdefault("model_paths", {})["assist_model"] = model_path
    await _try_reconnect("assist", state, cfg_for_reconnect)
    if residency is not None:
        residency.resume("assist_model_swap")

    return True
