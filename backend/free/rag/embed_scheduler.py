"""埋め込みサーバの調停 — ``EmbedScheduler`` (c_16 §6.5)

埋め込みサーバ (:8082) を対話・記憶・学習・一括処理が奪い合わないよう、優先度を
付けて順番を決める。

| 優先度 | 用途 |
|---|---|
| P0 対話 | クエリの埋め込み、tool_gate |
| P1 記憶の鮮度 | tail 索引、snapshot の埋め込み |
| P2 学習 | few-shot の backfill、Level 1 phase3 |
| P3 一括 | 再埋め込み、corpus install、疑似クエリ |

- 同時に投げるのは :data:`CAPACITY` 本まで。うち 1 本は P0 のために空けておく
  (P1〜P3 が詰まっていても、チャットのクエリはすぐ送れる)。
- 上位が待っていれば下位は始めない。P2 / P3 は :data:`LOW_BATCH` 件ずつに分けて
  送り、その合間に上位を通す (1 回で数百件送ると、その間クエリが待たされる)。
- 利用者がアクティブな間、P3 は直前のバッチにかかった時間だけ休む (半分まで)。

優先度は ``contextvars`` で運ぶ (``backend.embed_priority``。どの柱からも入口で
指定できるよう横断基盤に置く)。指定が無ければクエリは P0、文書は P1。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import numpy as np

from backend.embed_priority import (
    LEVELS as _LEVELS,
    P0_DIALOG,
    P1_FRESHNESS,
    P2_LEARNING,
    P3_BULK,
    current_priority,
    embed_priority,
    with_embed_priority,
)
from backend.log_config import get_logger

logger = get_logger("rag.embed_scheduler")

#: 同時に埋め込みサーバへ投げる数 (うち 1 本は P0 に予約)。
CAPACITY = 2
#: P2 / P3 の 1 回あたりの件数上限。
LOW_BATCH = 16

_T = TypeVar("_T")


class EmbedScheduler:
    """優先度つきの同時実行数の管理 (1 プロセスに 1 つ、:func:`default_scheduler`)。"""

    def __init__(self, capacity: int = CAPACITY) -> None:
        self.capacity = max(1, int(capacity))
        self._inflight = 0
        self._waiting = [0] * _LEVELS
        self._cond: asyncio.Condition | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._user_active: Callable[[], bool] | None = None

    def set_user_active_probe(self, probe: Callable[[], bool] | None) -> None:
        """利用者がアクティブかを返す関数 (P3 を半分に抑える判定、起動時に配線)。"""
        self._user_active = probe

    def _condition(self) -> asyncio.Condition:
        loop = asyncio.get_running_loop()
        if self._cond is None or self._loop is not loop:
            # 別のイベントループ (テスト / CLI) から使われたら数え直す
            self._cond = asyncio.Condition()
            self._loop = loop
            self._inflight = 0
            self._waiting = [0] * _LEVELS
        return self._cond

    def _may_start(self, priority: int) -> bool:
        if any(self._waiting[level] for level in range(priority)):
            return False  # 上位が待っている
        reserve = 0 if priority == P0_DIALOG or self.capacity <= 1 else 1
        return self._inflight < self.capacity - reserve

    def _user_is_active(self) -> bool:
        probe = self._user_active
        if probe is None:
            return False
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 — 判定できなければ抑えない
            return False

    async def run(self, priority: int, call: Callable[[], Awaitable[_T]]) -> _T:
        """順番が来たら ``call()`` を実行する。"""
        cond = self._condition()
        loop = asyncio.get_running_loop()
        async with cond:
            self._waiting[priority] += 1
            try:
                await cond.wait_for(lambda: self._may_start(priority))
            except BaseException:
                self._waiting[priority] -= 1
                cond.notify_all()  # 待ちから抜けた上位の分、下位が進めるかもしれない
                raise
            self._waiting[priority] -= 1
            self._inflight += 1
        started = loop.time()
        try:
            return await call()
        finally:
            async with cond:
                self._inflight -= 1
                cond.notify_all()
            if priority == P3_BULK and self._user_is_active():
                await asyncio.sleep(loop.time() - started)


_default = EmbedScheduler()


def default_scheduler() -> EmbedScheduler:
    """プロセス既定のスケジューラ。"""
    return _default


class ScheduledEmbeddingBackend:
    """埋め込みバックエンドを :class:`EmbedScheduler` 越しに呼ぶ薄い包み。

    ``embed`` / ``embed_query`` 以外 (``dim`` / ``model_name`` など) はそのまま
    中身へ渡す。
    """

    def __init__(self, inner: Any, scheduler: EmbedScheduler | None = None) -> None:
        self._inner = inner
        self._scheduler = scheduler or default_scheduler()

    async def embed(
        self, texts: list[str], *, is_query: bool = False, mode: str = "chat",
    ) -> np.ndarray:
        priority = current_priority(is_query)
        inner = self._inner
        if priority >= P2_LEARNING and len(texts) > LOW_BATCH:
            parts = []
            for start in range(0, len(texts), LOW_BATCH):
                chunk = texts[start:start + LOW_BATCH]
                parts.append(await self._scheduler.run(
                    priority, lambda chunk=chunk: inner.embed(chunk, is_query=is_query, mode=mode),
                ))
            return np.concatenate([np.asarray(part) for part in parts], axis=0)
        return await self._scheduler.run(
            priority, lambda: inner.embed(texts, is_query=is_query, mode=mode),
        )

    async def embed_query(self, query: str, *, mode: str = "chat") -> np.ndarray:
        inner = self._inner
        return await self._scheduler.run(
            current_priority(True), lambda: inner.embed_query(query, mode=mode),
        )

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":  # 初期化前 (copy / pickle) に再帰しない
            raise AttributeError(name)
        return getattr(self._inner, name)


__all__ = [
    "CAPACITY",
    "LOW_BATCH",
    "P0_DIALOG",
    "P1_FRESHNESS",
    "P2_LEARNING",
    "P3_BULK",
    "EmbedScheduler",
    "ScheduledEmbeddingBackend",
    "current_priority",
    "default_scheduler",
    "embed_priority",
    "with_embed_priority",
]
