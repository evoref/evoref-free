"""アシストモデル ステータス API（Free版）

Free版ではローカルアシストモデルのヘルスチェック・接続状態のみを提供する。
Pro版の /api/assist-model は外部API切替・ハイブリッド設定等の高度な管理機能。
"""

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.config import get_config
from backend.free.api.schemas import (
    AssistModelConcurrency,
    AssistModelStatusResponse,
)
from backend.log_config import get_logger

logger = get_logger("api.assist_model")

router = APIRouter(prefix="/api/assist-model", tags=["assist-model"])


async def _try_lazy_connect_assist(state: AppState, cfg: dict) -> bool:
    """assist_client が未接続の場合、アシストモデルへの遅延接続を試みる"""
    from backend.free.llm.assist_client import AssistModelClient

    assist_model_cfg = cfg.get("assist_model", {})
    if not assist_model_cfg.get("local"):
        return False

    try:
        client = AssistModelClient(cfg, debug_logger=state.debug_logger)
        if await client.health_check():
            state.set_assist_client(client)
            logger.info("Assist model lazy-connected: %s", client.url)
            return True
        else:
            await client.aclose()
    except Exception as e:
        logger.debug("Assist model lazy-connect failed: %s", e)
    return False


@router.get("/status", response_model=AssistModelStatusResponse)
async def get_assist_model_status(
    state: AppState = Depends(get_app_state),
) -> AssistModelStatusResponse:
    """アシストモデルの接続状態・設定情報を返す

    state.assist_client がある場合はリアルタイムでヘルスチェックを実行する。
    未設定・未接続でも 200 を返し、configured / connected フラグで状態を示す。
    """
    logger.debug("GET /api/assist-model/status")
    cfg = get_config()
    assist_cfg = cfg.get("assist_model", {})
    local_cfg = assist_cfg.get("local", {})

    # 設定の有無を判定
    configured = bool(local_cfg)
    if not configured:
        return AssistModelStatusResponse()

    host = local_cfg.get("host", "127.0.0.1")
    port = int(local_cfg.get("port", 8081))
    url = f"http://{host}:{port}"
    timeout = float(assist_cfg.get("timeout", 10))
    # 用途別セマフォ。未指定は 1 スロットを既定
    concurrency_cfg = assist_cfg.get("concurrency") or {}
    concurrency = AssistModelConcurrency(
        realtime=int(concurrency_cfg.get("realtime", 1)),
        background=int(concurrency_cfg.get("background", 1)),
        learning=int(concurrency_cfg.get("learning", 1)),
    )

    # ヘルスチェック
    connected = False
    model_params_b: float | None = None

    if state.assist_client is not None:
        try:
            connected = await state.assist_client.health_check()
            model_params_b = state.assist_client.params_b
        except Exception:
            logger.debug("Assist model health check failed during status request")
    else:
        # assist_client が未初期化 → 遅延接続を試行
        # （起動時にモデルロードが間に合わなかったケース）
        # バックオフ付きで lazy-init を試みることで、未起動時の
        # /api/assist-model/status 応答を 1 秒以内に保つ
        from backend.free.api.system._lazy_connect import guarded_lazy_connect

        lazy_ok = await guarded_lazy_connect(
            "assist",
            lambda: _try_lazy_connect_assist(state, cfg),
        )
        if lazy_ok:
            connected = True
            model_params_b = state.assist_client.params_b
        else:
            # 遅延接続も失敗 — config から情報のみ返す
            model_path = local_cfg.get("model", local_cfg.get("model_path", ""))
            if model_path:
                from backend.free.llm.model_metadata import estimate_params_b
                model_params_b = estimate_params_b(str(model_path))

    # 接続済みなら client 側の実スロット値を優先 (lazy init 時も config と一致)
    if state.assist_client is not None:
        client_cc = state.assist_client.concurrency
        concurrency = AssistModelConcurrency(
            realtime=client_cc["realtime"],
            background=client_cc["background"],
            learning=client_cc["learning"],
        )

    return AssistModelStatusResponse(
        configured=configured,
        connected=connected,
        url=url,
        host=host,
        port=port,
        model_params_b=model_params_b if model_params_b else None,
        concurrency=concurrency,
        timeout_seconds=timeout,
    )
