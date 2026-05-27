"""分割処理・中断耐性のための I/O 共通基盤

各 pillar (EvorefGen / EvorefMem / EvorefLoop / EvorefLearn) から参照可能な横断
モジュール。``backend.config`` / ``backend.debug_logger`` / ``backend.trace_context``
と同階層の位置付け。

主要 API:

- :class:`AtomicWriter` / :func:`atomic_write_text` / :func:`atomic_write_bytes` —
  tmp ファイル + ``os.replace`` + Windows ``PermissionError`` retry を内蔵した
  原子的書き込み。
- :class:`ChunkedReader` / :class:`Chunk` / :class:`ReadOptions` /
  :func:`csv_row_strategy` — 大ファイルのストリーミング読込基盤。
- :class:`JSONLAppendStore` — append-only JSONL + tombstone + atomic compaction。
- :class:`IncrementalIndexUpdater` — ``{key: value}`` インデックスの差分更新
  (JSONLAppendStore + バッチ flush)。
- :class:`ResumableProcessor` / :class:`CheckpointStore` /
  :class:`CheckpointEntry` / :class:`ResumableResult` — 中断耐性のある
  非同期バッチ処理 (asyncio.Semaphore + checkpoint JSONL)。
"""

from __future__ import annotations

from backend.io.atomic import AtomicWriter, atomic_write_bytes, atomic_write_text
from backend.io.chunked_reader import (
    Chunk,
    ChunkedReader,
    ChunkStrategy,
    ReadOptions,
    csv_row_strategy,
)
from backend.io.incremental_index import IncrementalIndexUpdater
from backend.io.jsonl_store import (
    TOMBSTONE_KEY_FIELD,
    TOMBSTONE_MARKER,
    JSONLAppendStore,
)
from backend.io.resumable import (
    CheckpointEntry,
    CheckpointStore,
    ResumableProcessor,
    ResumableResult,
)

__all__ = [
    "AtomicWriter",
    "CheckpointEntry",
    "CheckpointStore",
    "Chunk",
    "ChunkStrategy",
    "ChunkedReader",
    "IncrementalIndexUpdater",
    "JSONLAppendStore",
    "ReadOptions",
    "ResumableProcessor",
    "ResumableResult",
    "TOMBSTONE_KEY_FIELD",
    "TOMBSTONE_MARKER",
    "atomic_write_bytes",
    "atomic_write_text",
    "csv_row_strategy",
]
