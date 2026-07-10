"""コンポーネントクライアント差し替えヘルパー

`migrate_component` API が llama-server を再起動した後、in-memory に
保持しているクライアント (AssistModelClient / EmbeddingBackend) を
新しいモデルメタデータで作り直して `AppState` に再注入する。

`rebind_component()` は失敗しても呼び出し側でロールバック判定できるよう
例外を伝播する。`auto_restart` フローでは migrate_component → restart →
rebind の順で呼び出され、いずれかが失敗すれば旧モデルへロールバックする。
"""

from __future__ import annotations

from pathlib import Path
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

        # Pro Level2 の sleep_scheduler が保持する assist モデル GGUF パスを
        # 追従させる。起動時 (backend/pro/__init__.py::_wire_assist_sleep_paths)
        # は 1 回しか配線されないため、これを怠ると migrate 後も Level2 が旧
        # モデルのパスから LoRA ターゲット層を解決し続ける (GGUF パスは
        # ターゲット解決に実ファイルが要るため存在時のみ配線する、起動時と
        # 同じ方針)。
        if state.sleep_scheduler is not None:
            assist_model_gguf = (cfg.get("model_paths", {}) or {}).get(
                "assist_model", "",
            )
            if assist_model_gguf:
                p = Path(assist_model_gguf)
                if not p.is_absolute() and project_root is not None:
                    p = project_root / p
                if p.exists():
                    state.sleep_scheduler.set_assist_model_path(p)
                    logger.info("sleep_scheduler assist_model_path refreshed: %s", p)

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
