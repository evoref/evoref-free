"""リクエスト相関ID (trace_id) のコンテキスト管理

contextvars を使用してリクエスト単位の trace_id を伝播する。
asyncio タスク間で自動的に継承されるため、
呼び出し元のシグネチャ変更なしで全デバッグログに trace_id を付与できる。
"""

from __future__ import annotations

import asyncio
import contextvars
import uuid
from concurrent.futures import Executor
from contextvars import ContextVar
from typing import Any, Callable, TypeVar

# リクエストスコープの trace_id（空文字列 = 非リクエストコンテキスト）
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def generate_trace_id() -> str:
    """UUID4 短縮形式の trace_id を生成する（12文字 hex）"""
    return uuid.uuid4().hex[:12]


def set_trace_id(trace_id: str) -> None:
    """現在のコンテキストに trace_id をセットする"""
    trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """現在のコンテキストの trace_id を取得する（未設定時は空文字列）"""
    return trace_id_var.get()


_T = TypeVar("_T")


def run_in_executor_with_context(
    loop: asyncio.AbstractEventLoop,
    executor: Executor | None,
    func: Callable[..., _T],
    *args: Any,
) -> asyncio.Future[_T]:
    """`loop.run_in_executor` のラッパ。呼び出し元の contextvars を保全する

    `asyncio.AbstractEventLoop.run_in_executor` は `concurrent.futures.Executor`
    のワーカースレッドへ関数を投げるが、その際 `contextvars.Context` をコピー
    しないため、スレッド側から `get_trace_id()` を呼ぶと空文字列になる。
    本ヘルパは `contextvars.copy_context()` で現在のコンテキストを束ね、
    ワーカースレッド側で `ctx.run(func, *args)` として実行することで、
    `trace_id` を含む全 `ContextVar` をスレッド境界越しに伝播させる。
    """
    ctx = contextvars.copy_context()
    return loop.run_in_executor(executor, lambda: ctx.run(func, *args))
