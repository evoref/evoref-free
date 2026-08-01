"""RAG necessity/quality の assist 判定を保持するバッファ (再起動耐性つき)

SemMem ではない (``SemanticFactStore`` を一切参照しない)。CLAUDE.md §6 #2 の
「SemMem は読むだけ (チャット応答パス)」を侵さない。sleep-time Step 8.7
(``rag_judge_curator.py``) が drain して ``world_fact`` 化する中間バッファ。

RAG necessity/quality の決定は URL/executable_command のように単一の
MemoryNote フィールドへ 1:1 対応させにくい (STM ノートを跨いだ判定になりうる)
ため、専用バッファで chat 応答パス側の変更を最小化する。

**永続化する理由** (2026-08-01 プロファイリング): 以前は in-memory の deque
だけで、消費側は Full サイクル (実測 22 回 / light 557 回) にしか無かった。
間に再起動が入るとイベントが丸ごと消え、**キュレータ実行 22 回中 16 回が
空の入力**を引いていた (生成 181 件 → 書き込み 3 件)。

**答えを同梱する理由**: 判定時点では応答本文がまだ無いため、キュレータは後から
STM を引き直していた。しかし light サイクル (557 回) の eviction が頻繁で、
Full が回る頃には該当ターンが消えている。ターン確定時に本文を紐付けておけば
STM の寿命に依存しなくなる。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("memory.rag_judge_assist_log")

#: 答えが紐付かないまま滞留したイベントの寿命 (秒)。ターンが中断された等で
#: 本文が付かないイベントを永久に持ち越さない。
_PENDING_TTL_SEC = 7 * 24 * 3600


@dataclass(frozen=True)
class RagJudgeAssistEvent:
    """RAG necessity/quality の 1 回の assist 判定を表すイベント。"""

    kind: str
    """"rag_necessity" | "rag_quality" のいずれか。"""
    query: str
    decision: str
    """necessity: "retrieve"/"fetch" (skip は記録しない) / quality: "high"/"medium"/"low"。"""
    ts: float
    answer: str = ""
    """ターン確定時に紐付けた応答本文。空ならキュレータが STM から引き直す。"""


class RagJudgeAssistLog:
    """chat 応答パスで発火した RAG necessity/quality の assist 判定を蓄積する。

    thread-safe。``path`` を与えると JSONL へ追記して再起動をまたいで保持する
    (``path=None`` は in-memory のみ = 従来挙動、テスト用)。上限 ``max_size``
    を超えると古い順に破棄する (メモリリーク防止)。
    """

    def __init__(self, max_size: int = 500, path: "Path | None" = None) -> None:
        self._lock = threading.Lock()
        self._events: list[RagJudgeAssistEvent] = []
        self._max_size = max_size
        self._path = path
        if path is not None:
            self._events = self._load()

    # -- 永続化 ---------------------------------------------------------
    def _load(self) -> list[RagJudgeAssistEvent]:
        """JSONL からイベントを復元する (壊れた行は捨てて続行)。"""
        if self._path is None or not self._path.exists():
            return []
        events: list[RagJudgeAssistEvent] = []
        try:
            with self._path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        events.append(RagJudgeAssistEvent(
                            kind=d["kind"], query=d["query"],
                            decision=d["decision"], ts=float(d["ts"]),
                            answer=d.get("answer", ""),
                        ))
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError as e:
            logger.warning("rag_judge_assist_log: load failed: %s", e)
            return []
        if events:
            logger.info(
                "rag_judge_assist_log: restored %d pending event(s)", len(events),
            )
        return events

    def _rewrite_locked(self) -> None:
        """現在のイベント列でファイルを置き換える (呼出側でロック済み)。"""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for ev in self._events:
                    f.write(json.dumps({
                        "kind": ev.kind, "query": ev.query,
                        "decision": ev.decision, "ts": ev.ts,
                        "answer": ev.answer,
                    }, ensure_ascii=False) + "\n")
            tmp.replace(self._path)
        except OSError as e:
            logger.warning("rag_judge_assist_log: persist failed: %s", e)

    def _prune_locked(self, now: float) -> None:
        """TTL 超過の未紐付けイベントと、上限超過分を落とす。"""
        self._events = [
            ev for ev in self._events
            if ev.answer or now - ev.ts <= _PENDING_TTL_SEC
        ]
        if len(self._events) > self._max_size:
            self._events = self._events[-self._max_size:]

    # -- 記録 -----------------------------------------------------------
    def record(self, kind: str, query: str, decision: str) -> None:
        if not query or not decision:
            return
        now = time.time()
        with self._lock:
            self._events.append(
                RagJudgeAssistEvent(
                    kind=kind, query=query, decision=decision, ts=now,
                ),
            )
            self._prune_locked(now)
            self._rewrite_locked()

    def attach_answer(self, query: str, answer: str) -> int:
        """確定した応答本文を、同じクエリの未紐付けイベントへ紐付ける。

        判定時点では本文が存在しないため、ターン確定時にここで結びつける。
        これをしないとキュレータが STM を引き直すことになり、eviction 済みの
        ターンでは答えが見つからず学習信号が落ちる。

        Returns:
            紐付けた件数。
        """
        if not query or not answer:
            return 0
        target = query.strip()
        attached = 0
        with self._lock:
            for i, ev in enumerate(self._events):
                if not ev.answer and ev.query.strip() == target:
                    self._events[i] = replace(ev, answer=answer)
                    attached += 1
            if attached:
                self._rewrite_locked()
        return attached

    def drain(self) -> list[RagJudgeAssistEvent]:
        """蓄積イベントを取り出してバッファを空にする (冪等消費)。"""
        with self._lock:
            events = list(self._events)
            self._events = []
            self._rewrite_locked()
        return events

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


__all__ = ["RagJudgeAssistEvent", "RagJudgeAssistLog"]
