"""RAG necessity/quality の assist 判定を一時保持するプロセス内リングバッファ

SemMem ではない (``SemanticFactStore`` を一切参照しない)。CLAUDE.md §6 #2 の
「SemMem は読むだけ (チャット応答パス)」を侵さない。sleep-time Step 8.7
(``rag_judge_curator.py``) が drain して ``world_fact`` 化する中間バッファ。

RAG necessity/quality の決定は URL/executable_command のように単一の
MemoryNote フィールドへ 1:1 対応させにくい (STM ノートを跨いだ判定になりうる)
ため、専用リングバッファで chat 応答パス側の変更を最小化する。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RagJudgeAssistEvent:
    """RAG necessity/quality の 1 回の assist 判定を表すイベント。"""

    kind: str
    """"rag_necessity" | "rag_quality" のいずれか。"""
    query: str
    decision: str
    """necessity: "retrieve"/"fetch" (skip は記録しない) / quality: "high"/"medium"/"low"。"""
    ts: float


class RagJudgeAssistLog:
    """chat 応答パスで発火した RAG necessity/quality の assist 判定を蓄積する。

    thread-safe。上限 ``max_size`` を超えると古い順に破棄する
    (``deque(maxlen=...)`` による best-effort、メモリリーク防止)。
    """

    def __init__(self, max_size: int = 500) -> None:
        self._lock = threading.Lock()
        self._events: deque[RagJudgeAssistEvent] = deque(maxlen=max_size)

    def record(self, kind: str, query: str, decision: str) -> None:
        if not query or not decision:
            return
        with self._lock:
            self._events.append(
                RagJudgeAssistEvent(
                    kind=kind, query=query, decision=decision, ts=time.time(),
                ),
            )

    def drain(self) -> list[RagJudgeAssistEvent]:
        """蓄積イベントを取り出してバッファを空にする (冪等消費)。"""
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


__all__ = ["RagJudgeAssistEvent", "RagJudgeAssistLog"]
