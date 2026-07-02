"""コンポーネントクライアント差し替えヘルパー

`migrate_component` API が llama-server を再起動した後、in-memory に
保持しているクライアント (AssistModelClient / EmbeddingBackend) を
新しいモデルメタデータで作り直して `AppState` に再注入する。

`rebind_component()` は失敗しても呼び出し側でロールバック判定できるよう
例外を伝播する。`auto_restart` フローでは migrate_component → restart →
rebind の順で呼び出され、いずれかが失敗すれば旧モデルへロールバックする。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState

logger = get_logger("core.component_rebind")


async def rebind_component(
    component: str, state: "AppState", cfg: dict,
) -> None:
    """指定コンポーネントのクライアントを新インスタンスで差し替える"""
    debug_logger = state.debug_logger
    project_root_owner = state.llama_manager
    project_root = (
        project_root_owner.project_root if project_root_owner else None
    )

    if component == "assist":
        from backend.free.llm.assist_client import AssistModelClient

        old = state.assist_client
        new = AssistModelClient(cfg, debug_logger=debug_logger)
        if not await new.health_check():
            try:
                await new.aclose()
            except Exception:
                pass
            raise RuntimeError("assist health_check failed after restart")
        state.set_assist_client(new)
        if old is not None:
            try:
                await old.aclose()
            except Exception as e:
                logger.warning("old assist_client close failed: %s", e)
        logger.info("assist client rebound")
        return

    if component == "embedding":
        from backend.free.rag.embedding_factory import create_embedding_backend

        old = state.embedder
        new = create_embedding_backend(
            cfg, project_root, debug_logger=debug_logger,
        )
        if not await new.health_check():
            try:
                await new.aclose()
            except Exception:
                pass
            raise RuntimeError("embedding health_check failed after restart")
        state.embedder = new
        if old is not None:
            try:
                await old.aclose()
            except Exception as e:
                logger.warning("old embedder close failed: %s", e)
        logger.info("embedding backend rebound")
        return

    raise ValueError(f"Unknown component: {component}")
