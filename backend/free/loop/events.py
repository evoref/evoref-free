"""

LoopDriver の周回実行中に発生する主要イベントを観測するための pub/sub 基盤。
SSE エンドポイント (`/api/loop/stream`) が購読してフロントエンドへ配信する。

設計原則:
- 同一プロセス内の fan-out のみをサポート (asyncio.Queue ベース)
- Publisher (LoopDriver) は **非ブロッキング**。subscriber の消費が遅い場合は
  各キュー単位で最古イベントを捨てる (drop-oldest) ことで LoopDriver の周回を
  止めない
- subscribe/unsubscribe は idempotent。SSE 切断で unsubscribe 漏れが起きても
  publisher の処理に影響しない
- イベントには `trace_id` を含める (CLAUDE.md trace_id 伝播ルール)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from backend.log_config import get_logger
from backend.trace_context import get_trace_id

logger = get_logger("loop.events")


LoopEventKind = Literal[
    "task_picked",
    "action_executed",
    "gate_result",
    "fact_written",
    "iteration_started",
    "iteration_ended",
    "loop_started",
    "loop_paused",
    "loop_resumed",
    "loop_stopped",
]
"""LoopEvent の種別。

- ``task_picked`` : ``pick_next_task`` で次タスクが選ばれた
- ``action_executed`` : Action が 1 件実行された (status / kind / error)
- ``gate_result`` : 品質ゲートの実行結果
- ``fact_written`` : SemMem に書き込まれた (progress_marker / failure_pattern /
  task.status 遷移)
- ``iteration_started`` / ``iteration_ended`` : 周回単位の区切り
- ``loop_started`` / ``loop_paused`` / ``loop_resumed`` / ``loop_stopped`` :
  ループ全体のライフサイクル
"""


@dataclass(frozen=True)
class LoopEvent:
    """LoopDriver から publish される 1 イベント。"""

    event: LoopEventKind
    trace_id: str
    timestamp: float
    iteration: int
    project_id: str | None
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "project_id": self.project_id,
            "data": dict(self.data),
        }


class LoopEventBus:
    """asyncio.Queue ベースの fan-out pub/sub。

    - ``subscribe()`` で新しい購読キューを確保
    - ``publish(event)`` で全購読者のキューへ配信 (キュー満杯なら最古を捨てる)
    - ``unsubscribe(queue)`` で購読解除 (idempotent)

    スレッドセーフではない。`asyncio` イベントループ上からのみ呼ぶこと。
    LoopDriver は単一の周回 task 上で動き、SSE ハンドラは同一イベントループ上
    の別 task なので問題なし。
    """

    def __init__(self, max_queue_size: int = 128) -> None:
        self._max_queue_size = int(max_queue_size)
        self._subscribers: list[asyncio.Queue[LoopEvent]] = []

    def subscribe(self) -> asyncio.Queue[LoopEvent]:
        q: asyncio.Queue[LoopEvent] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers.append(q)
        logger.debug(
            "LoopEventBus: subscribed (subscribers=%d)", len(self._subscribers),
        )
        return q

    def unsubscribe(self, queue: asyncio.Queue[LoopEvent]) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            return
        logger.debug(
            "LoopEventBus: unsubscribed (subscribers=%d)", len(self._subscribers),
        )

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, event: LoopEvent) -> None:
        """全購読者へ non-blocking 配信。キュー満杯なら最古を捨てて追加。"""
        if not self._subscribers:
            return
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "LoopEventBus: failed to publish event=%s after drop",
                        event.event,
                    )

    def emit(
        self,
        event: LoopEventKind,
        *,
        iteration: int,
        project_id: str | None,
        data: dict[str, Any] | None = None,
    ) -> LoopEvent:
        """``LoopEvent`` を構築して ``publish`` する簡易ヘルパ。

        ``trace_id`` は現在の ContextVar から自動取得する。
        """
        evt = LoopEvent(
            event=event,
            trace_id=get_trace_id(),
            timestamp=time.time(),
            iteration=iteration,
            project_id=project_id,
            data=dict(data or {}),
        )
        self.publish(evt)
        return evt
