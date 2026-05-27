"""中断耐性のあるバッチ処理

長時間かかる items 列を ``asyncio.Semaphore`` で並列処理しつつ、各 item の処理
結果を :class:`CheckpointStore` に append-only で記録する。プロセスが落ちて
再 run しても、checkpoint に "done" として残ったキーは自動的に skip される
ため、最初から全件やり直すことを避けられる。

主要ユースケース: cartridge install (`_chunk_and_embed_docs`) のように
N 件のドキュメントを連続して埋め込みベクトル化する処理。途中で OS kill /
LLM サーバ切断などで中断した場合、再 install で既埋め込み済の N-K 件を
スキップして残りだけ処理できる。

スレッド/トランザクション境界:

- 内部 :class:`CheckpointStore` (=:class:`JSONLAppendStore`) は ``threading.Lock``
  で append を直列化済。
- ``concurrency >= 1`` 時に ``asyncio.Semaphore`` で並列度を制御する。
- ``trace_id`` は :mod:`backend.trace_context` の contextvars 経由で各 worker
  に伝播する (asyncio task 境界では contextvars は自動コピー)。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Literal,
    TypeVar,
)

from backend.io.jsonl_store import JSONLAppendStore
from backend.log_config import get_logger
from backend.trace_context import get_trace_id

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("io.resumable")

__all__ = [
    "CheckpointEntry",
    "CheckpointStore",
    "ResumableProcessor",
    "ResumableResult",
]

ItemKey = str
T = TypeVar("T")
R = TypeVar("R")

CheckpointStatus = Literal["done", "failed"]


# ─────────────────────────────────────────────────────────────────────
# CheckpointEntry
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckpointEntry:
    """1 item の処理結果記録。

    ``status``:

    - ``"done"`` 成功完了。再 run 時に skip 対象。
    - ``"failed"`` 例外発生で失敗。``retry_failed=True`` (デフォルト) なら
      再 run 時に再処理対象、``False`` なら skip 対象。
    """

    key: str
    status: CheckpointStatus
    ts: float  # epoch seconds (UTC)
    trace_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "key": self.key,
                "status": self.status,
                "ts": self.ts,
                "trace_id": self.trace_id,
                "detail": self.detail,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(line: str) -> "CheckpointEntry":
        obj = json.loads(line)
        return CheckpointEntry(
            key=str(obj["key"]),
            status=obj["status"],
            ts=float(obj.get("ts", 0.0)),
            trace_id=str(obj.get("trace_id", "")),
            detail=dict(obj.get("detail", {})),
        )


# ─────────────────────────────────────────────────────────────────────
# ResumableResult
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ResumableResult(Generic[R]):
    """:meth:`ResumableProcessor.run` の戻り値。

    Attributes:
        completed: 成功した ``(key, result)`` 列 (本実行で処理した分のみ)。
        failed: 失敗した ``(key, exception)`` 列 (本実行で発生した分のみ)。
        skipped: checkpoint hit でスキップしたキー列 (前回までに完了済)。
        resumed_from: ``len(skipped)`` と同じ。観測用エイリアス。
    """

    completed: list[tuple[str, R]] = field(default_factory=list)
    failed: list[tuple[str, BaseException]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    resumed_from: int = 0


# ─────────────────────────────────────────────────────────────────────
# CheckpointStore
# ─────────────────────────────────────────────────────────────────────


class CheckpointStore:
    """ResumableProcessor の進捗を JSONL に永続化する store。

    内部実装は :class:`JSONLAppendStore[CheckpointEntry]`。1 行 = 1 entry、
    同一 key の entry は後勝ち (例: 前回 ``"failed"`` のキーが今回 ``"done"``
    なら ``done`` が優先される)。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        compact_threshold_lines: int = 1000,
        compact_threshold_ratio: float = 2.0,
        debug_logger: "DebugLogger | None" = None,
    ) -> None:
        self._store: JSONLAppendStore[CheckpointEntry] = JSONLAppendStore(
            Path(path),
            serialize=lambda entry: entry.to_json(),
            deserialize=CheckpointEntry.from_json,
            key_of=lambda entry: entry.key,
            compact_threshold_lines=compact_threshold_lines,
            compact_threshold_ratio=compact_threshold_ratio,
            debug_logger=debug_logger,
        )

    def load_all_entries(self) -> dict[str, CheckpointEntry]:
        """key → 最新 entry の辞書を返す。"""
        return self._store.load_all()

    def load_done_keys(self) -> set[str]:
        """``status == "done"`` のキー集合 (再 run 時の skip 候補)。"""
        return {
            key for key, e in self._store.load_all().items() if e.status == "done"
        }

    def load_failed_keys(self) -> set[str]:
        """``status == "failed"`` のキー集合 (``retry_failed=False`` 時の skip 候補)。"""
        return {
            key for key, e in self._store.load_all().items() if e.status == "failed"
        }

    def record(self, entry: CheckpointEntry) -> None:
        """1 entry を append する。"""
        self._store.append(entry)

    def compact(self) -> None:
        """強制 compaction (重複 key を物理削除)。"""
        self._store.compact()

    def maybe_compact(self) -> bool:
        """閾値超過時のみ compaction する。戻り値: 実行したか。"""
        return self._store.maybe_compact()

    def unlink(self) -> None:
        """checkpoint ファイルを物理削除する (install 完了後の片付け)。

        ファイル未存在の場合は no-op (例外を上げない)。
        """
        try:
            self._store.path.unlink()
        except FileNotFoundError:
            pass

    @property
    def path(self) -> Path:
        return self._store.path


# ─────────────────────────────────────────────────────────────────────
# ResumableProcessor
# ─────────────────────────────────────────────────────────────────────


class ResumableProcessor(Generic[T, R]):
    """中断耐性のあるバッチ処理。

    Args:
        job_id: 観測用 ID (DebugLogger に流す)。
        items: 処理対象のイテラブル (初期化時に list 化してスナップショット)。
        key_of: ``item → ItemKey`` の関数。**冪等キー** を返すこと
            (再 run 時に同じ item が同じキーを返すこと)。
        process: 各 item を処理する async 関数。
        checkpoint: 進捗永続化用の :class:`CheckpointStore` (``None`` なら
            永続化なし = 中断耐性なしの並列バッチ処理として動く)。
        concurrency: 並列度 (asyncio.Semaphore)。1 で逐次実行。
        on_progress: 各 item 完了時に ``await on_progress(emitted, total)``
            される async コールバック。
        cancel_check: 各 item 開始前に呼ぶ同期判定。``True`` を返した時点で
            残タスクをキャンセルし :class:`asyncio.CancelledError` を伝播する。
        debug_logger: 注入されていれば完了時 / キャンセル時に
            ``log_memory_op`` で観測情報を流す。
        retry_failed: 前回 ``"failed"`` で記録されたキーを再試行するか。
            ``True`` がデフォルト (再試行する)。

    Raises:
        ValueError: ``concurrency < 1``。
        asyncio.CancelledError: ``cancel_check`` が ``True`` を返した場合、または
            外部から task が cancel された場合。
    """

    def __init__(
        self,
        job_id: str,
        items: Iterable[T],
        key_of: Callable[[T], ItemKey],
        process: Callable[[T], Awaitable[R]],
        *,
        checkpoint: CheckpointStore | None = None,
        concurrency: int = 4,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        debug_logger: "DebugLogger | None" = None,
        retry_failed: bool = True,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._job_id = job_id
        self._items: list[T] = list(items)
        self._key_of = key_of
        self._process = process
        self._checkpoint = checkpoint
        self._concurrency = concurrency
        self._on_progress = on_progress
        self._cancel_check = cancel_check
        self._debug_logger = debug_logger
        self._retry_failed = retry_failed

    async def run(self) -> ResumableResult[R]:
        result: ResumableResult[R] = ResumableResult()

        # 1) checkpoint から skip 対象キーを構築
        skip_keys: set[str] = set()
        if self._checkpoint is not None:
            skip_keys |= self._checkpoint.load_done_keys()
            if not self._retry_failed:
                skip_keys |= self._checkpoint.load_failed_keys()

        # 2) skip 振り分け
        to_process: list[tuple[str, T]] = []
        for item in self._items:
            key = self._key_of(item)
            if key in skip_keys:
                result.skipped.append(key)
                continue
            to_process.append((key, item))
        result.resumed_from = len(result.skipped)

        total = len(self._items)

        sem = asyncio.Semaphore(self._concurrency)

        async def _process_one(key: str, item: T) -> None:
            async with sem:
                # cancel_check は item 単位の開始前に判定
                if self._cancel_check is not None and self._cancel_check():
                    raise asyncio.CancelledError(
                        f"ResumableProcessor[{self._job_id}] cancelled before key={key}"
                    )
                try:
                    r = await self._process(item)
                except asyncio.CancelledError:
                    raise
                except BaseException as e:  # noqa: BLE001 — record any failure
                    result.failed.append((key, e))
                    self._record(
                        CheckpointEntry(
                            key=key,
                            status="failed",
                            ts=time.time(),
                            trace_id=get_trace_id(),
                            detail={"error": repr(e)},
                        )
                    )
                else:
                    result.completed.append((key, r))
                    self._record(
                        CheckpointEntry(
                            key=key,
                            status="done",
                            ts=time.time(),
                            trace_id=get_trace_id(),
                        )
                    )
                # progress (skipped + completed + failed) を total に対して通知
                emitted = (
                    len(result.skipped) + len(result.completed) + len(result.failed)
                )
                await self._notify_progress(emitted, total)

        # 初回 progress 通知 (skip 分だけ進捗が進んでいる状態)
        if result.skipped:
            await self._notify_progress(len(result.skipped), total)

        # 3) 並列実行
        tasks = [
            asyncio.create_task(_process_one(key, item))
            for key, item in to_process
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            # 残タスクを cancel して回収
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._log_op(
                "resumable_cancelled",
                {
                    "job_id": self._job_id,
                    "total": total,
                    "completed": len(result.completed),
                    "failed": len(result.failed),
                    "skipped": len(result.skipped),
                },
            )
            raise

        self._log_op(
            "resumable_complete",
            {
                "job_id": self._job_id,
                "total": total,
                "completed": len(result.completed),
                "failed": len(result.failed),
                "skipped": len(result.skipped),
            },
        )
        return result

    # ── 内部: 副作用をまとめる ────────────────────────────────────────

    def _record(self, entry: CheckpointEntry) -> None:
        if self._checkpoint is None:
            return
        try:
            self._checkpoint.record(entry)
        except Exception as cp_err:  # noqa: BLE001 — checkpoint failure must not break batch
            logger.warning(
                "ResumableProcessor[%s]: checkpoint record(%s) failed: %s",
                self._job_id, entry.status, cp_err,
            )

    async def _notify_progress(self, emitted: int, total: int) -> None:
        if self._on_progress is None:
            return
        try:
            await self._on_progress(emitted, total)
        except asyncio.CancelledError:
            raise
        except Exception as prog_err:  # noqa: BLE001
            logger.warning(
                "ResumableProcessor[%s]: on_progress failed: %s",
                self._job_id, prog_err,
            )

    def _log_op(self, op: str, stats: dict[str, Any]) -> None:
        if self._debug_logger is None:
            return
        try:
            self._debug_logger.log_memory_op(op, stats)
        except Exception as log_err:  # noqa: BLE001
            logger.warning("DebugLogger.log_memory_op failed: %s", log_err)
