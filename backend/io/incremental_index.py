"""差分インデックス更新

``{key: value}`` 形式のインデックスを :class:`JSONLAppendStore` 上に永続化し、
``upsert`` / ``remove`` を 1 件単位で受け付ける薄いラッパ。SemMem の索引
(``facts_by_subject`` / ``facts_by_type`` / ``facts_by_pillar`` / ``pinned``)
を「ファクト追加ごとに全 dict 再 dump」から「変更されたキーのみ append」に
切り替えるための facade。

設計:

- 内部実装は :class:`JSONLAppendStore` 1 個に集約 (DRY)。upsert 1 件 =
  store.append 1 行、remove 1 件 = store.tombstone 1 行 + auto compaction。
- ``flush_interval_writes`` 件まで in-memory にバッファし、それ以下では
  ファイル I/O を発生させない。閾値超過 or 明示的 ``flush()`` で一括書込。
- :meth:`load` は **常に in-memory 状態** (= flush 済 + pending) を返すので、
  呼出側は flush タイミングを意識せずに dict 同等に使える。

スレッド安全性:

- ``threading.Lock`` で ``upsert`` / ``remove`` / ``flush`` / ``load`` を
  直列化する (内部の :class:`JSONLAppendStore` も自前 lock を持つが、
  ``_state`` と pending 操作列の一貫性を担保するため別途必要)。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, Literal, TypeVar

from backend.io.jsonl_store import JSONLAppendStore
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("io.incremental_index")

__all__ = ["IncrementalIndexUpdater"]

V = TypeVar("V")

_PendingOp = tuple[Literal["upsert", "remove"], str, "object | None"]


class IncrementalIndexUpdater(Generic[V]):
    """JSONLAppendStore 上の ``{key: value}`` インデックスを差分更新する。

    Args:
        path: 索引 JSONL のパス。
        serialize_entry: ``(key, value)`` を 1 行 JSON 文字列に変換 (改行なし)。
        deserialize_entry: 1 行 JSON 文字列から ``(key, value)`` を復元。
        flush_interval_writes: in-memory buffer に溜める変更件数の上限。
            これを **超えた時点** で自動 flush。デフォルト 50。
        compact_threshold_lines: 内部 :class:`JSONLAppendStore` の compaction
            候補閾値 (デフォルト 1000)。
        compact_threshold_ratio: 同 compaction 比率 (デフォルト 2.0)。
        debug_logger: 注入された ``DebugLogger`` があれば flush / compact 時に
            観測情報を流す。

    Usage:
        >>> idx = IncrementalIndexUpdater[set[str]](
        ...     path,
        ...     serialize_entry=lambda k, v: json.dumps({"k": k, "v": sorted(v)}),
        ...     deserialize_entry=lambda line: (
        ...         (lambda d: (d["k"], set(d["v"])))(json.loads(line))
        ...     ),
        ... )
        >>> idx.upsert("opinion", {"f1", "f2"})
        >>> idx.upsert("decision", {"f3"})
        >>> idx.remove("opinion")
        >>> idx.flush()
        >>> idx.load()
        {'decision': {'f3'}}
    """

    def __init__(
        self,
        path: Path | str,
        serialize_entry: Callable[[str, V], str],
        deserialize_entry: Callable[[str], tuple[str, V]],
        *,
        flush_interval_writes: int = 50,
        compact_threshold_lines: int = 1000,
        compact_threshold_ratio: float = 2.0,
        debug_logger: "DebugLogger | None" = None,
    ) -> None:
        if flush_interval_writes < 1:
            raise ValueError("flush_interval_writes must be >= 1")
        self._serialize_entry = serialize_entry
        self._deserialize_entry = deserialize_entry
        self._flush_interval = flush_interval_writes
        self._debug_logger = debug_logger
        self._lock = threading.Lock()
        # 内部 store: 1 item = (key, value) tuple
        self._store: JSONLAppendStore[tuple[str, V]] = JSONLAppendStore(
            path,
            serialize=lambda entry: serialize_entry(entry[0], entry[1]),
            deserialize=deserialize_entry,
            key_of=lambda entry: entry[0],
            compact_threshold_lines=compact_threshold_lines,
            compact_threshold_ratio=compact_threshold_ratio,
            debug_logger=debug_logger,
        )
        # 起動時ロード: 既存ファイル → in-memory state
        loaded = self._store.load_all()
        self._state: dict[str, V] = {key: entry[1] for key, entry in loaded.items()}
        # pending op queue (順序保持で flush 時に再生)
        self._pending_ops: list[_PendingOp] = []

    # ── 書き込み API ──────────────────────────────────────────────────

    def upsert(self, key: str, value: V) -> None:
        """``key`` の value を更新 (なければ追加)。 flush 閾値を超えたら自動 flush。"""
        with self._lock:
            self._state[key] = value
            self._pending_ops.append(("upsert", key, value))
            if len(self._pending_ops) >= self._flush_interval:
                self._flush_locked()

    def remove(self, key: str) -> None:
        """``key`` を削除。未存在キーへの呼出も安全 (no-op 相当の tombstone を書く)。"""
        with self._lock:
            removed = self._state.pop(key, None)
            # 未存在キーへの remove はファイルに tombstone を書かない (no-op)
            # ことで JSONL を不必要に膨らませない。
            if removed is None and not any(
                op == "upsert" and k == key for op, k, _ in self._pending_ops
            ):
                return
            self._pending_ops.append(("remove", key, None))
            if len(self._pending_ops) >= self._flush_interval:
                self._flush_locked()

    def flush(self) -> None:
        """pending 操作を JSONL に書き出す。完了後に auto compaction も走る。"""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._pending_ops:
            return
        for op, key, value in self._pending_ops:
            if op == "upsert":
                # value は V 型として upsert 経由で入っているので cast 不要
                self._store.append((key, value))  # type: ignore[arg-type]
            else:  # "remove"
                self._store.tombstone(key)
        n_ops = len(self._pending_ops)
        self._pending_ops.clear()
        # 閾値超過なら compact (内部で AtomicWriter 経由)
        self._store.maybe_compact()
        if self._debug_logger is not None:
            try:
                self._debug_logger.log_memory_op(
                    "index_flush",
                    {
                        "path": str(self._store.path),
                        "ops": n_ops,
                        "live_keys": len(self._state),
                    },
                )
            except Exception as log_err:
                logger.warning("DebugLogger.log_memory_op failed: %s", log_err)

    def compact(self) -> None:
        """pending を flush した後に強制 compaction (閾値無視)。"""
        with self._lock:
            self._flush_locked()
            self._store.compact()

    # ── 読み込み API ──────────────────────────────────────────────────

    def load(self) -> dict[str, V]:
        """現在の in-memory 状態 (flush 済 + pending) のコピーを返す。"""
        with self._lock:
            return dict(self._state)

    # ── 統計 / 観測 ──────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._state)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._state

    def pending_count(self) -> int:
        """未 flush の pending 操作件数。"""
        with self._lock:
            return len(self._pending_ops)

    @property
    def path(self) -> Path:
        return self._store.path
