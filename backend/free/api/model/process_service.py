"""llama-server の起動・停止の API 側の糸 (c_06 §2.6)

プロセスの所有者は :class:`~backend.free.core.llama_process_manager.LlamaProcessManager`
ひとつ。ここは API の各経路 (`/api/model/process/*`、互換層の `/api/servers/*`、
モード切替) が共有する次の 2 つだけを持つ:

- manager の同期呼び出しを executor へ逃がす (数分ブロックしうる。trace_id を伝播する)
- 起動・停止の後に backend のクライアントを付け替える / 手放す (base の LocalClient、
  埋め込みの httpx、リランカーの httpx と遮断器)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from backend.app_state import AppState
from backend.constants import DEFAULT_LLAMA_PORT
from backend.free.core.llama_process_manager import (
    ProcessAlreadyRunningError,
    ProcessEntry,
    ProcessManagerError,
    ProcessPortBusyError,
    StopResult,
)
from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context

logger = get_logger("api.process_service")

_T = TypeVar("_T")


async def run_blocking(func: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """同期の ``func`` を既定の executor で走らせる (trace_id を伝播する)。"""
    loop = asyncio.get_running_loop()
    return await run_in_executor_with_context(loop, None, partial(func, *args, **kwargs))


async def start_component(
    state: AppState, component: str, cfg: dict, *, force: bool = False, model_override: str | None = None,
) -> ProcessEntry:
    """``component`` を起こし、成功したらクライアントを付け替える。

    ``state.llama_manager`` が在ることは呼び出し側が確かめる。起動前の拒否 (自分のプロセスが
    動いている / ポートを外部が占有) 以外の失敗は、旧プロセスを止めた後かもしれないので
    クライアントを手放してから例外を上げ直す。
    """
    manager = state.llama_manager
    assert manager is not None
    try:
        entry = await run_blocking(
            manager.start, component, cfg, force=force, model_override=model_override,
        )
    except (ProcessAlreadyRunningError, ProcessPortBusyError):
        raise
    except ProcessManagerError:
        await release_clients(state, component)
        raise
    await _rebind_after_start(state, component, cfg)
    return entry


async def _rebind_after_start(state: AppState, component: str, cfg: dict) -> None:
    """起こし直した後の付け替え。base は旧サーバへのクライアントを閉じてから繋ぎ直す。"""
    if component == "base":
        await release_clients(state, component)
    await reconnect_clients(state, component, cfg)


async def restart_component(state: AppState, component: str, cfg: dict) -> ProcessEntry:
    """``component`` を再起動し (外部起動なら置き換え)、クライアントを付け替える。"""
    manager = state.llama_manager
    assert manager is not None
    try:
        entry = await run_blocking(manager.restart, component, cfg)
    except ProcessPortBusyError:
        raise
    except ProcessManagerError:
        await release_clients(state, component)
        raise
    await _rebind_after_start(state, component, cfg)
    return entry


async def stop_component(state: AppState, component: str, cfg: dict) -> StopResult:
    """``component`` を止め (管理下か外部の llama-server)、クライアントを手放す。"""
    manager = state.llama_manager
    assert manager is not None
    result = await run_blocking(manager.stop_component, component, cfg)
    await release_clients(state, component)
    return result


async def release_clients(state: AppState, component: str) -> None:
    """止めたサーバへのクライアントを手放す (オブジェクトは残し、死んだ接続だけ捨てる)。"""
    if component == "base":
        if state.local_client:
            await state.local_client.aclose()
        state.local_client = None
        if state.llm_client:
            state.llm_client.local = None
    elif component == "embedding":
        # embedder オブジェクトは残すが httpx クライアントだけ閉じて
        # 死んだサーバへの kept-alive socket を破棄する。次回 embed 呼び出しで
        # _get_http_client が新しい AsyncClient を作る。
        if state.embedder is not None and hasattr(state.embedder, "aclose"):
            try:
                await state.embedder.aclose()
            except Exception as e:
                logger.warning("process_service: failed to close embedder client: %s", e)
    elif component == "rerank":
        await _close_reranker_client(state)


async def reconnect_clients(state: AppState, component: str, cfg: dict) -> None:
    """サーバ起動後にバックエンドのクライアントを再接続

    base だけでなく embed も再構築する。再構築しないと httpx.AsyncClient の kept-alive が
    前世代の死んだサーバの socket を保持したままになり、起動直後の health_check / embed
    呼び出しがフライング失敗する。
    """
    if component == "base":
        from backend.free.api.system.status import _try_lazy_connect

        llama_cfg = cfg.get("llama", {})
        host = llama_cfg.get("host", "127.0.0.1")
        port = llama_cfg.get("port", DEFAULT_LLAMA_PORT)
        await _try_lazy_connect(state, f"http://{host}:{port}", llama_cfg)

    elif component == "embedding":
        from backend.free.api.config.component_reload import reload_embedder

        try:
            await reload_embedder(state)
        except Exception as e:
            logger.warning("process_service: embed reload after start failed: %s", e)

    elif component == "rerank":
        # 死んだサーバへの kept-alive socket を捨てる (次の rerank 呼出で作り直す)。
        # 前のサーバで開いた遮断器も閉じ直す (冷却を待たずに次のターンから使う)
        await _close_reranker_client(state)
        reranker = getattr(getattr(state, "gen", None), "reranker", None)
        if reranker is not None:
            reranker.reset_breaker()


async def _close_reranker_client(state: AppState) -> None:
    """``RerankClient`` の httpx クライアントだけ閉じる (オブジェクトは残す)。"""
    gen = getattr(state, "gen", None)
    reranker = getattr(gen, "reranker", None) if gen is not None else None
    if reranker is None:
        return
    try:
        await reranker.aclose()
    except Exception as e:
        logger.warning("process_service: failed to close rerank client: %s", e)
