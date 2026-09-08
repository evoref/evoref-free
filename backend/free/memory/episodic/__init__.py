"""EvorefMem — エピソード記憶 (会話由来ノート) の統一ストア (c_16 §4.1)。

WM (プロセス内の窓) → STM → LTM の 3 層コピーを廃し、``Evidence``
(``kind="note"``) 1 型 + ``attrs.tier`` (``working`` / ``short`` / ``long``)
へ畳んだ。tier 遷移は ``patch`` だけで、**データは動かないし再チャンクもしない**。

- :mod:`~backend.free.memory.episodic.note` — ``MemoryNote`` ↔ ``Evidence``
- :mod:`~backend.free.memory.episodic.ingest` — 会話ターン → ``short`` ノート
- :mod:`~backend.free.memory.episodic.store` — :class:`EpisodicStore`
- :mod:`~backend.free.memory.episodic.workspace` — sleep-time の作業領域
- :mod:`~backend.free.memory.episodic.progress` — ノート化済みターンの進捗
- :mod:`~backend.free.memory.episodic.turn_source` — 会話履歴からのターン供給

書き手は sleep-time (``SleepTimeWorker``) だけ。チャット応答パスは
:meth:`EpisodicStore.search` で読み、使った id を
:attr:`EpisodicStore.usage` (プロセス内バッファ) へ入れるだけで、
``last_used_at`` の書き込みは sleep-time の ``flush_touch`` が行う。
"""

from __future__ import annotations

from backend.free.memory.episodic.ingest import (
    build_note_from_turn,
    ingest_new_turns,
    ingest_session,
)
from backend.free.memory.episodic.note import (
    NOTE_ATTR_FIELDS,
    MemoryNote,
    evidence_to_note,
    note_to_evidence,
)
from backend.free.memory.episodic.progress import EpisodicProgress
from backend.free.memory.episodic.store import (
    LONG_SHARD_WINDOW_MONTHS,
    EpisodicHit,
    EpisodicStore,
    episodic_shard_key,
)
from backend.free.memory.episodic.turn_source import (
    HistoryTurnSource,
    SessionTurns,
    TurnSource,
)
from backend.free.memory.episodic.workspace import EpisodicWorkspace

__all__ = [
    "LONG_SHARD_WINDOW_MONTHS",
    "NOTE_ATTR_FIELDS",
    "EpisodicHit",
    "EpisodicProgress",
    "EpisodicStore",
    "EpisodicWorkspace",
    "HistoryTurnSource",
    "MemoryNote",
    "SessionTurns",
    "TurnSource",
    "build_note_from_turn",
    "episodic_shard_key",
    "evidence_to_note",
    "ingest_new_turns",
    "ingest_session",
    "note_to_evidence",
]
