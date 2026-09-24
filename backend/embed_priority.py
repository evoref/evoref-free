"""埋め込みの優先度を文脈で運ぶ (c_16 §6.5 の ``EmbedScheduler``)

``trace_context`` と同じく ``contextvars`` で非同期の呼び出しへ伝わる横断基盤。
どの柱からも入口で優先度を指定できるよう、スケジューラ本体
(``backend.free.rag.embed_scheduler``) とは分けて置く。

| 優先度 | 用途 |
|---|---|
| P0 対話 | クエリの埋め込み、tool_gate |
| P1 記憶の鮮度 | tail 索引、snapshot の埋め込み |
| P2 学習 | few-shot の backfill、Level 1 phase3 |
| P3 一括 | 再埋め込み、corpus install、疑似クエリ |

指定が無ければクエリは P0、文書は P1。
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

P0_DIALOG = 0
P1_FRESHNESS = 1
P2_LEARNING = 2
P3_BULK = 3
LEVELS = 4

_priority: ContextVar[int | None] = ContextVar("embed_priority", default=None)

_T = TypeVar("_T")


@contextmanager
def embed_priority(priority: int, *, override: bool = True) -> Iterator[None]:
    """この文脈の埋め込みの優先度を決める。

    Args:
        override: ``False`` なら、外側で既に決まっていればそれを使う (ストアの
            内部から「既定は P1」を示すとき。corpus install の中の版作りは P3 の
            ままにしたい)。
    """
    if not override and _priority.get() is not None:
        yield
        return
    token = _priority.set(priority)
    try:
        yield
    finally:
        _priority.reset(token)


def with_embed_priority(
    priority: int,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """非同期関数の中の埋め込みを ``priority`` にするデコレータ (背景処理の入口用)。"""

    def decorate(fn: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            with embed_priority(priority):
                return await fn(*args, **kwargs)

        return wrapper

    return decorate


def current_priority(is_query: bool) -> int:
    """今の文脈の優先度 (指定が無ければクエリは P0、文書は P1)。"""
    priority = _priority.get()
    if priority is None:
        return P0_DIALOG if is_query else P1_FRESHNESS
    return min(max(int(priority), P0_DIALOG), P3_BULK)
