"""正常停止 API ``POST /api/system/shutdown`` (c_06 §2.7 / c_09 §2.1 / c_05 §0.5.8)

受理を 202 で即返し、裏で次の段を順に進めてから自プロセスを終わらせる:

1. **停止中の印** — 以後の状態を変える要求 (取り消し・停止そのものを除く) を 503 で断る
   (:class:`ShutdownGate`)。readonly の機構は使わない — readonly にすると、取り消した
   ターンの最後の書き込み (履歴・経験) が書き手スレッドへ積めずに落ちる。
2. **協調キャンセル** — 進行中のチャット / 制作のターンに ``/api/chat/cancel`` と同じ
   フラグを立て、終わるのを待つ (staged の run は ``run.json`` を ``cancelled`` で終端する)。
3. **書き手スレッドの drain** — チャット経路の SoT の追記をキューの最後まで書いて fsync する。
4. **終了の合図** — :func:`request_process_exit` が uvicorn に終了を伝える。lifespan の
   shutdown (sleep-time の静止・Level 1 の停止と学習の取り消し・保存・Pro の shutdown hook・
   書き手ロックの解放) はその後に uvicorn が通常どおり走らせる。

各段の失敗は WARNING を出して先へ進む。上限 :data:`SHUTDOWN_DEADLINE_SEC` を過ぎたら
残りを諦めて必ず終了の合図を出す (耐久性は kill 前提で設計している、c_05 §0.5.8)。
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter

from backend.free.api._error_responses import api_error_detail
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("api.shutdown")

router = APIRouter(prefix="/api/system", tags=["system"])

#: 終了の合図を出すまでの上限 (秒)。キャンセル待ちと drain はこの中で打ち切る。
SHUTDOWN_DEADLINE_SEC = 20.0
#: ターンの終わりを確かめる間隔 (秒)。
_POLL_SEC = 0.1

SHUTDOWN_PATH = "/api/system/shutdown"
SHUTTING_DOWN_CODE = "E0503"
SHUTTING_DOWN_STATUS = 503
SHUTTING_DOWN_I18N_KEY = "api.shutting_down"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@dataclass
class ShutdownState:
    """停止の進み具合 (プロセスに 1 つ)。"""

    requested: bool = False
    task: asyncio.Task | None = None


_state = ShutdownState()


def is_shutting_down() -> bool:
    """停止を受理済みか。"""
    return _state.requested


# ──────────────────────────────────────────────────────────────────────────
# 終了の合図
# ──────────────────────────────────────────────────────────────────────────


def request_process_exit() -> None:
    """uvicorn に終了を伝える: 自プロセスへ SIGTERM を上げる。

    uvicorn (``Server.capture_signals``) は SIGINT / SIGTERM に ``signal.signal`` で
    ハンドラを付け、受けると ``should_exit`` を立てて通常の停止 (接続の終わりを待つ →
    lifespan の shutdown) を進める。``signal.raise_signal`` は C の ``raise()`` で、
    Windows でも Python のハンドラ (メインスレッド) を呼ぶ。Ctrl+C / ``kill`` と同じ経路。
    """
    handler = signal.getsignal(signal.SIGTERM)
    if handler in (signal.SIG_DFL, None):
        logger.warning("Graceful shutdown: no SIGTERM handler installed; the process ends without lifespan shutdown")
    elif handler is signal.SIG_IGN:
        logger.warning("Graceful shutdown: SIGTERM is ignored in this process; the process will not exit by itself")
    logger.info("Graceful shutdown: raising SIGTERM to stop the server")
    signal.raise_signal(signal.SIGTERM)


# ──────────────────────────────────────────────────────────────────────────
# 段
# ──────────────────────────────────────────────────────────────────────────


def _cancel_active_turns() -> int:
    from backend.free.api.chat.chat_stream_common import request_cancel_all

    return request_cancel_all()


def _active_turns() -> int:
    from backend.free.api.chat.chat_stream_common import active_request_count

    return active_request_count()


async def _wait_turns_finished(timeout: float) -> bool:
    """取り消したターンが全て終わるまで待つ。``timeout`` 内に終われば ``True``。"""
    deadline = time.monotonic() + timeout
    while _active_turns() > 0:
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_SEC)
    return True


def _drain_writer(timeout: float) -> bool:
    from backend.io.writer_thread import default_writer

    return default_writer().drain(timeout)


async def _stage(name: str, fn: Callable[[], Awaitable[Any]]) -> None:
    try:
        await fn()
    except Exception as e:  # noqa: BLE001 — 停止は続ける (kill 前提、c_05 §0.5.8)
        logger.warning("Graceful shutdown: stage %s failed: %s", name, e)


async def _run_stages(deadline: float) -> None:
    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    async def cancel() -> None:
        count = _cancel_active_turns()
        logger.info("Graceful shutdown: cancel requested for %d in-flight request(s)", count)
        if not await _wait_turns_finished(remaining()):
            logger.warning(
                "Graceful shutdown: %d request(s) still running at the deadline; giving up on them",
                _active_turns(),
            )

    async def drain() -> None:
        from backend.trace_context import run_in_executor_with_context

        loop = asyncio.get_running_loop()
        if not await run_in_executor_with_context(loop, None, _drain_writer, remaining()):
            logger.warning("Graceful shutdown: chat writer queue was not drained before the deadline")

    await _stage("cancel", cancel)
    await _stage("drain", drain)


async def run_shutdown(*, deadline_sec: float = SHUTDOWN_DEADLINE_SEC) -> None:
    """協調キャンセル → drain → 終了の合図。合図は段の成否・上限に関わらず必ず出す。"""
    started = time.monotonic()
    deadline = started + deadline_sec
    try:
        await asyncio.wait_for(_run_stages(deadline), timeout=deadline_sec)
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning("Graceful shutdown: deadline of %.1fs reached; exiting anyway", deadline_sec)
    except Exception as e:  # noqa: BLE001
        logger.warning("Graceful shutdown: stages failed: %s", e)
    finally:
        logger.info(
            "Graceful shutdown: stages finished in %.1fs; signalling the server to exit",
            time.monotonic() - started,
        )
        try:
            request_process_exit()
        except Exception as e:  # noqa: BLE001
            logger.warning("Graceful shutdown: exit trigger failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────
# ルート
# ──────────────────────────────────────────────────────────────────────────


@router.post("/shutdown", status_code=202)
async def shutdown() -> dict[str, Any]:
    """正常停止を受理する (202)。2 回目以降も 202 (``already_requested: true``) で何もしない。"""
    if _state.requested:
        return {"status": "shutting_down", "already_requested": True, "deadline_sec": SHUTDOWN_DEADLINE_SEC}
    _state.requested = True
    logger.info("Graceful shutdown requested (deadline %.1fs)", SHUTDOWN_DEADLINE_SEC)
    _state.task = asyncio.get_running_loop().create_task(run_shutdown())
    return {"status": "shutting_down", "already_requested": False, "deadline_sec": SHUTDOWN_DEADLINE_SEC}


# ──────────────────────────────────────────────────────────────────────────
# 停止中の新規書き込みを断る ASGI ミドルウェア
# ──────────────────────────────────────────────────────────────────────────


def _allowed_while_stopping(path: str) -> bool:
    """停止中も通す状態変更: 停止そのもの (冪等) と取り消し (``.../cancel``)。"""
    return path == SHUTDOWN_PATH or path.rstrip("/").endswith("/cancel")


class ShutdownGate:
    """停止を受理した後の状態を変える要求を 503 (``E0503`` / ``api.shutting_down``) で断る。"""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") == "http"
            and _state.requested
            and str(scope.get("method") or "GET").upper() not in _SAFE_METHODS
            and not _allowed_while_stopping(str(scope.get("path") or ""))
        ):
            body = json.dumps(
                {"detail": api_error_detail(SHUTTING_DOWN_CODE, msg(SHUTTING_DOWN_I18N_KEY), SHUTTING_DOWN_I18N_KEY)},
                ensure_ascii=False,
            ).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": SHUTTING_DOWN_STATUS,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def register_shutdown_gate(app: Any) -> None:
    """``create_app`` から呼ぶ。CORS より内側 (先に add) に置き、503 にも CORS ヘッダを付ける。"""
    app.add_middleware(ShutdownGate)


__all__ = [
    "SHUTDOWN_DEADLINE_SEC",
    "ShutdownGate",
    "is_shutting_down",
    "register_shutdown_gate",
    "request_process_exit",
    "router",
    "run_shutdown",
]
