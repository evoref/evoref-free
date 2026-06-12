"""システム情報 API

VRAM 使用量スナップショットなど、コンポーネント単位ではなくシステム全体を
観測するエンドポイント群を集約する。

``GET /api/system/vram_status``:
    ベース / アシスト / 埋め込み の 3 モデルについて、推定値
    + (条件が揃えば) 実測値を返す。詳細は
    ``backend/free/core/vram_monitor.py`` と
    ``backend/free/api/schemas.py::VramStatusResponse`` を参照。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.config import get_config
from backend.free.api.schemas import VramStatusResponse
from backend.free.core.vram_monitor import collect_vram_status
from backend.log_config import get_logger

logger = get_logger("api.system")

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/vram_status", response_model=VramStatusResponse)
async def get_vram_status(
    state: AppState = Depends(get_app_state),
) -> VramStatusResponse:
    """各 llama-server モデルの VRAM 使用量を返す

    推定値は常に返り、``nvidia-smi`` + ``LlamaProcessManager`` が揃った環境
    では実測値で上書きする。GPU が無い / ``nvidia-smi`` 不在の環境でも
    エラーにはならず推定値で応答する。
    """
    logger.debug("GET /api/system/vram_status")
    cfg = get_config()

    # プロジェクトルート: backend/free/api/system/system.py から 5 階層上
    project_root = Path(__file__).resolve().parents[4]

    snapshot = collect_vram_status(
        cfg,
        project_root,
        process_manager=state.llama_manager,
    )
    return VramStatusResponse(**snapshot)
