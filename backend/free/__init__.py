"""Free エディションプラグイン

setup_free(app) を呼び出すことで Free ルーターを登録する。
backend/free/ はすべてのエディションで必須の基盤機能を提供する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.edition import register_free
from backend.log_config import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger("free")


def setup_free(app: FastAPI) -> None:
    """Free 機能を FastAPI アプリに登録する

    Free API ルーター（23本）を登録し、Free エディションを有効化する。
    """
    from backend.free.api.system.status import router as status_router
    from backend.free.api.chat.chat import router as chat_router
    from backend.free.api.content.rag import router as rag_router
    from backend.free.api.content.memory import router as memory_router
    from backend.free.api.model.model import router as model_router
    from backend.free.api.config.config_api import router as config_router
    from backend.free.api.content.cartridges import router as cartridges_router
    from backend.free.api.system.files import router as files_router
    from backend.free.api.model.prompts import router as prompts_router
    from backend.free.api.model.assist_prompts import router as assist_prompts_router
    from backend.free.api.history.history import router as history_router
    from backend.free.api.system.commands import router as commands_router
    from backend.free.api.history.sessions import router as sessions_router
    from backend.free.api.config.themes import router as themes_router
    from backend.free.api.learning.learning import router as learning_router
    from backend.free.api.learning.optimize import router as optimize_router
    from backend.free.api.system.data import router as data_router
    from backend.free.api.model.assist_model_api import router as assist_model_router
    from backend.free.api.system.export_file import router as export_file_router
    from backend.free.api.system.diffs import router as diffs_router
    from backend.free.api.system.server_control import router as server_control_router
    from backend.free.api.config.mode import router as mode_router
    from backend.free.api.system.loop import router as loop_router
    from backend.free.api.system.system import router as system_router

    app.include_router(status_router)
    app.include_router(chat_router)
    app.include_router(rag_router)
    app.include_router(memory_router)
    app.include_router(model_router)
    app.include_router(config_router)
    app.include_router(cartridges_router)
    app.include_router(files_router)
    app.include_router(prompts_router)
    app.include_router(assist_prompts_router)
    app.include_router(history_router)
    app.include_router(commands_router)
    app.include_router(sessions_router)
    app.include_router(themes_router)
    app.include_router(learning_router)
    app.include_router(optimize_router)
    app.include_router(data_router)
    app.include_router(assist_model_router)
    app.include_router(export_file_router)
    app.include_router(diffs_router)
    app.include_router(server_control_router)
    app.include_router(mode_router)
    app.include_router(loop_router)
    app.include_router(system_router)

    # テキスト抽出 Extractor の登録（Free 4種）
    import backend.free.extraction  # noqa: F401

    # ファイル書出し Writer の登録（Free 5種）
    import backend.free.export  # noqa: F401

    register_free()
    logger.info("Free edition initialized: 24 routers registered")
