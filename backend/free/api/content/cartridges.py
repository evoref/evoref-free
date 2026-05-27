"""カートリッジ API"""

import asyncio
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.error_handlers import ErrorResponse
from backend.free.api.content._cartridge_serializers import (
    attach_lora_impact,
    cartridge_detail_dict,
    cartridge_install_response,
    cartridge_rebuild_response,
    cartridge_summary_dict,
)
from backend.free.core.sse import SSEFrameBuilder
from backend.free.rag.cartridge_manager import CartridgeInstallCancelled
from backend.log_config import get_logger
from backend.trace_context import generate_trace_id, set_trace_id

logger = get_logger("api.cartridges")

router = APIRouter(prefix="/api/cartridges", tags=["cartridges"])

# ストリーミング install 用キャンセルフラグ（session_id → True/False）
_install_cancel_flags: dict[str, bool] = {}

# SSE フレームビルダー（モジュールレベル共有）
_sse = SSEFrameBuilder()


@asynccontextmanager
async def _install_cancel_scope(session_id: str):
    """install/stream 用キャンセルフラグのスコープ管理"""
    _install_cancel_flags[session_id] = False
    try:
        yield
    finally:
        _install_cancel_flags.pop(session_id, None)


class _CartridgeCancelRequest(BaseModel):
    """カートリッジストリーミング処理のキャンセルリクエスト"""
    session_id: str


def _cartridge_error(
    status_code: int,
    code: str,
    message: str,
    i18n_key: str = "",
    **context,
) -> HTTPException:
    """カートリッジ API 用の統一エラーレスポンスを生成"""
    detail = ErrorResponse(
        code=code,
        message=message,
        i18n_key=i18n_key,
        context=context,
    ).to_dict()
    return HTTPException(status_code=status_code, detail=detail)


def _get_manager(state: AppState):
    """CartridgeManager インスタンスを取得"""
    mgr = state.cartridge_manager
    if mgr is None:
        raise _cartridge_error(
            503, "E0501", "Cartridge manager not initialized",
            i18n_key="api.cartridge_manager_not_initialized",
        )
    return mgr


def _get_embedder(state: AppState):
    """EmbeddingBackend インスタンスを取得（必須）"""
    embedder = state.embedder
    if embedder is None:
        raise _cartridge_error(
            503, "E0502", "Embedding backend not initialized",
            i18n_key="api.cartridge_embedder_not_initialized",
        )
    return embedder


@router.post("/install", status_code=201)
async def install_cartridge(
    state: AppState = Depends(get_app_state),
    file: UploadFile = File(...),
):
    """ZIP パッケージからカートリッジをインストール"""
    logger.debug("POST /api/cartridges/install: filename=%s", file.filename)
    mgr = _get_manager(state)
    embedder = _get_embedder(state)

    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    logger.debug("Saved upload to temp: %s (%d bytes)", tmp_path, len(content))

    start = time.time()
    try:
        info = await mgr.install(tmp_path, embedder=embedder)
    except ValueError as e:
        raise _cartridge_error(400, "E0510", str(e))
    except FileNotFoundError as e:
        raise _cartridge_error(400, "E0511", str(e))
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to delete temp file %s: %s", tmp_path, e)

    elapsed = time.time() - start
    logger.debug(
        "Cartridge installed: id=%s, name=%s, chunks=%d, time=%.3fs",
        info.id, info.name, info.chunks, elapsed,
    )
    return cartridge_install_response(info, elapsed)


@router.get("")
async def list_cartridges(state: AppState = Depends(get_app_state)):
    """インストール済みカートリッジ一覧"""
    logger.debug("GET /api/cartridges")
    mgr = _get_manager(state)
    carts = mgr.list_cartridges()
    logger.debug("Listed %d cartridges", len(carts))
    return {"cartridges": [cartridge_summary_dict(c) for c in carts]}


@router.get("/{cartridge_id}")
async def get_cartridge(cartridge_id: str, state: AppState = Depends(get_app_state)):
    """カートリッジ詳細情報"""
    logger.debug("GET /api/cartridges/%s", cartridge_id)
    mgr = _get_manager(state)
    info = mgr.get_cartridge(cartridge_id)
    if info is None:
        raise _cartridge_error(
            404, "E0504", f"Cartridge '{cartridge_id}' not found",
            i18n_key="api.cartridge_not_found_id", cartridge_id=cartridge_id,
        )

    return cartridge_detail_dict(info)


@router.post("/{cartridge_id}/load")
async def load_cartridge(cartridge_id: str, state: AppState = Depends(get_app_state)):
    """カートリッジをロード"""
    logger.debug("POST /api/cartridges/%s/load", cartridge_id)
    mgr = _get_manager(state)
    try:
        start = time.time()
        info = mgr.load(cartridge_id)
        elapsed = time.time() - start
        return {
            "id": info.id,
            "status": "loaded",
            "load_time_ms": round(elapsed * 1000, 1),
        }
    except KeyError:
        raise _cartridge_error(
            404, "E0504", f"Cartridge '{cartridge_id}' not found",
            i18n_key="api.cartridge_not_found_id", cartridge_id=cartridge_id,
        )


@router.post("/{cartridge_id}/unload")
async def unload_cartridge(cartridge_id: str, state: AppState = Depends(get_app_state)):
    """カートリッジをアンロード"""
    logger.debug("POST /api/cartridges/%s/unload", cartridge_id)
    mgr = _get_manager(state)
    try:
        info = mgr.unload(cartridge_id)
        result: dict = {"id": info.id, "status": "installed"}
        # LoRA 影響度情報を付与（CartridgeChangeHandler が登録済みの場合）
        return attach_lora_impact(result, state.cartridge_change_handler)
    except KeyError:
        raise _cartridge_error(
            404, "E0504", f"Cartridge '{cartridge_id}' not found",
            i18n_key="api.cartridge_not_found_id", cartridge_id=cartridge_id,
        )


@router.post("/{cartridge_id}/rebuild")
async def rebuild_cartridge(cartridge_id: str, state: AppState = Depends(get_app_state)):
    """カートリッジのベクトルインデックスを再構築"""
    logger.debug("POST /api/cartridges/%s/rebuild", cartridge_id)
    mgr = _get_manager(state)
    embedder = _get_embedder(state)

    start = time.time()
    try:
        info = await mgr.rebuild(cartridge_id, embedder=embedder)
    except KeyError:
        raise _cartridge_error(
            404, "E0504", f"Cartridge '{cartridge_id}' not found",
            i18n_key="api.cartridge_not_found_id", cartridge_id=cartridge_id,
        )
    except ValueError as e:
        raise _cartridge_error(400, "E0512", str(e))
    elapsed = time.time() - start

    logger.debug(
        "Cartridge rebuilt: id=%s, chunks=%d, time=%.3fs",
        info.id, info.chunks, elapsed,
    )
    return cartridge_rebuild_response(info, elapsed, embedder.backend_type())


@router.delete("/{cartridge_id}")
async def delete_cartridge(cartridge_id: str, state: AppState = Depends(get_app_state)):
    """カートリッジを完全削除"""
    logger.debug("DELETE /api/cartridges/%s", cartridge_id)
    mgr = _get_manager(state)
    try:
        mgr.uninstall(cartridge_id)
    except KeyError:
        raise _cartridge_error(
            404, "E0504", f"Cartridge '{cartridge_id}' not found",
            i18n_key="api.cartridge_not_found_id", cartridge_id=cartridge_id,
        )

    result: dict = {"id": cartridge_id, "status": "uninstalled"}
    # LoRA 影響度情報を付与
    return attach_lora_impact(result, state.cartridge_change_handler)


# ---------------------------------------------------------------------------
# SSE ストリーミング版 install (進捗・キャンセル対応)
# ---------------------------------------------------------------------------


@router.post("/install/stream")
async def install_cartridge_stream(
    state: AppState = Depends(get_app_state),
    file: UploadFile = File(...),
    session_id: str = Form(..., description="クライアントが発行する UUID。キャンセル用キー"),
):
    """SSE ストリーミング版インストール

    フェーズごとに進捗フレームを emit し、最終的に result + [DONE] フレームで終了する。
    クライアントは `POST /install/cancel` に同じ session_id を送ることで処理を中止できる。
    """
    trace_id = generate_trace_id()
    set_trace_id(trace_id)
    logger.debug(
        "POST /api/cartridges/install/stream: session=%s, filename=%s, trace=%s",
        session_id, file.filename, trace_id,
    )
    mgr = _get_manager(state)
    embedder = _get_embedder(state)

    # アップロードを一時ファイルに保存（mgr.install は Path を要求）
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    _SENTINEL: dict = {"__done__": True}

    async def progress_callback(frame: dict) -> None:
        await queue.put(frame)

    def cancel_check() -> bool:
        return _install_cancel_flags.get(session_id, False)

    async def runner() -> None:
        """バックグラウンドで install を実行し、結果またはエラーを queue に投入"""
        try:
            info = await mgr.install(
                tmp_path, embedder=embedder,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            elapsed = 0.0  # ストリーミング版では elapsed は client 側で計測
            payload = cartridge_install_response(info, elapsed)
            await queue.put({"__result__": payload})
        except CartridgeInstallCancelled:
            await queue.put({"__cancelled__": True})
        except (ValueError, FileNotFoundError) as e:
            await queue.put({"__error__": {"code": "E0510", "message": str(e)}})
        except Exception as e:  # noqa: BLE001 — 想定外を SSE で通知
            logger.exception("install/stream failed: %s", e)
            await queue.put({"__error__": {"code": "E0599", "message": str(e)}})
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Failed to delete temp file %s: %s", tmp_path, e)
            await queue.put(_SENTINEL)

    async def event_stream() -> AsyncIterator[str]:
        async with _install_cancel_scope(session_id):
            task = asyncio.create_task(runner())
            try:
                while True:
                    frame = await queue.get()
                    if frame is _SENTINEL:
                        break
                    if "__result__" in frame:
                        yield _sse.result(frame["__result__"])
                        continue
                    if "__cancelled__" in frame:
                        yield _sse.error_with_code(
                            "E0598", "Cartridge install cancelled",
                        )
                        continue
                    if "__error__" in frame:
                        err = frame["__error__"]
                        yield _sse.error_with_code(err["code"], err["message"])
                        continue
                    # 通常の進捗フレーム
                    yield _sse.step(frame)
                yield _sse.done()
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/install/cancel")
async def install_cartridge_cancel(req: _CartridgeCancelRequest):
    """進行中のストリーミング install をキャンセル"""
    logger.debug("POST /api/cartridges/install/cancel: session=%s", req.session_id)
    if req.session_id in _install_cancel_flags:
        _install_cancel_flags[req.session_id] = True
        return {"cancelled": True}
    return {"cancelled": False}
