"""snapshot の畳み込みと版ファイルの書き出しを子プロセスで行う (G1 設計 §11-2 の T1)

``EvidenceStore.create_snapshot`` のうち、入力がファイルだけで済む段 — 前の版の
``records.jsonl`` + 締切位置までの事象を畳み、物理 GC を掛け、``records.jsonl`` /
``offsets.npy`` / ``columns.npz`` / 転置索引を書く — を :func:`build_snapshot_files`
にまとめる。イベントループ上で走らせると 50k 件で十数秒ストリームが止まるので、
常駐の子プロセス 1 本 (spawn、優先度を下げる) へ出す。版の確定 (``COMPLETE``) は
埋め込み版の後に親が書く (c_16 §5.6)。スレッドでもループの停止は
p99 85ms 残る (GIL) — 子プロセスなら無負荷と同じ約 2ms (2026-09-23 実測)。

manifest・書き手のロック・オーバーレイには触らない (親が前後で行う)。ジョブは
pickle できるものだけで組む: 物理 GC は id 集合、シャード鍵はモジュール関数か
:class:`ConstantShardKey`。pickle できないジョブ (レコードを見て決める ``gc_filter``)
と、子プロセスを有効にしていないとき (テスト) はスレッドで走らせる — 結果は同じで、
止まるのがループの一部になるだけ。

子プロセスのログは ``backend.log`` へ直接書かない (回転するファイルを 2 プロセスで
書くと壊れる)。記録を集めて結果に載せ、親が同じロガー名で出し直す。
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import pickle
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.rag.evidence.events import EventPosition, EvidenceEventLog
from backend.free.rag.evidence.snapshot import SnapshotWriter, read_snapshot, snapshot_dir
from backend.free.rag.evidence.types import Evidence
from backend.free.rag.evidence.vectors import EvidenceVectorStore, ids_digest
from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context

if TYPE_CHECKING:
    from backend.free.rag.evidence.lexical_index import LexicalIndexBuilder

logger = get_logger("rag.evidence.snapshot_build")


class ConstantShardKey:
    """全レコードを 1 つのシャードへ入れる鍵 (corpus / 疑似クエリ / ProjectMap)。

    ラムダだと子プロセスへ渡せない (pickle できない) ので呼べるクラスにする。
    """

    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        self.key = key

    def __call__(self, _record: Evidence) -> str:
        return self.key

    def __reduce__(self) -> tuple[Any, ...]:
        return (ConstantShardKey, (self.key,))


@dataclass(frozen=True)
class SnapshotBuildJob:
    """1 版分の入力 (ファイルの場所と、ループ上で固めた値だけ)。"""

    store_dir: str
    #: 畳み込みの土台にする版 (無ければ空文字)。
    prev_version: str
    folded_through: EventPosition
    end: EventPosition
    version: str
    #: 新しい版へ書かない id (物理 GC の対象。親が畳み込みの直前に固める)。
    drop_ids: frozenset[str]
    shard_key_for: Callable[[Evidence], str]
    builder: LexicalIndexBuilder
    lexical_version: int
    #: レコードを見て決める物理 GC (``False`` で落とす)。pickle できないことが
    #: 多いので、あればスレッドで走らせる。
    keep: Callable[[Evidence], bool] | None = None
    #: 埋め込みの準備 (本文・再利用鍵・流用する行) も同じ子プロセスで済ませる。
    embed: EmbedPrepSpec | None = None


@dataclass(frozen=True)
class EmbedPrepSpec:
    """埋め込みの準備に要る値 (前の埋め込み版の場所と、再利用の判定条件)。"""

    #: 前の埋め込み版ディレクトリ (無ければ空文字)。
    previous_dir: str
    declared_model_id: str
    model_id: str
    expected_dim: int
    memmap_threshold: int
    quantization: str


@dataclass
class EmbedPrep:
    """組み立てたばかりの版に対する埋め込みの計画 (行は新しい版の行)。

    ``pending_texts`` は埋め込む行の本文だけ — 全件の本文を親へ運ぶと、50k 件で
    本文の読み直し (0.6 秒) をループ上でやるのと変わらなくなる。
    """

    ids: list[str]
    hashes: list[str]
    sides: list[tuple[bool, str]]
    reuse_dst: list[int]
    reuse_src: list[int]
    pending: list[int]
    pending_texts: list[str]


@dataclass
class SnapshotBuildResult:
    """組み立ての結果。"""

    records: int
    folded: int
    dropped_ids: list[str]
    shard_count: int
    embed_prep: EmbedPrep | None = None
    #: 新しい版の行 id のハッシュ (埋め込み版の刻印と COMPLETE に載せる)。
    id_hash: str = ""
    #: 子プロセスで出たログ ``(ロガー名, レベル, 本文)``。親が出し直す。
    logs: list[tuple[str, int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class IndexBuildJob:
    """埋め込み索引の新しい版を書く入力 (G1 設計 §11-2 の T2)。"""

    directory: str
    previous_dir: str
    memmap_threshold: int
    quantization: str
    ids: list[str]
    hashes: list[str]
    sides: list[tuple[bool, str]]
    reuse_dst: list[int]
    reuse_src: list[int]
    pending: list[int]
    #: ``pending`` の行を埋め込んだベクトル (float32)。無ければ ``None``。
    vectors: np.ndarray | None
    #: 刻印する snapshot の版名・行数・id ハッシュ (c_16 §5.6)。
    source: str
    snapshot_rows: int
    id_hash: str
    model_id: str
    backend_type: str
    cluster: bool
    n_probe_ratio: float


@dataclass
class IndexBuildResult:
    rows: int
    logs: list[tuple[str, int, str]] = field(default_factory=list)


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[tuple[str, int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append((record.name, record.levelno, record.getMessage()))
        except Exception:  # noqa: BLE001 — ログの失敗で版の生成を落とさない
            pass


def build_snapshot_files(job: SnapshotBuildJob) -> SnapshotBuildResult:
    """前の版 + 事象を畳み、新しい版ディレクトリのファイルを全部書く。"""
    return _with_log_capture(_build, job)


def build_index_files(job: IndexBuildJob) -> IndexBuildResult:
    """流用行と新しいベクトルから int8 行列を組み、埋め込みの版を書いてクラスタ索引を作る。"""
    return _with_log_capture(_build_index, job)


def _with_log_capture(fn: Callable[[Any], Any], job: Any) -> Any:
    """``fn(job)`` を走らせる。子プロセスではログを集めて結果の ``logs`` に載せる。

    スレッドで走るときは親のハンドラがそのまま受けるので集めない (二重に出る)。
    """
    collector: _Collect | None = None
    root = logging.getLogger("backend")
    if multiprocessing.parent_process() is not None:
        collector = _Collect()
        root.addHandler(collector)
        root.setLevel(logging.INFO)
    try:
        result = fn(job)
    finally:
        if collector is not None:
            root.removeHandler(collector)
    if collector is not None:
        result.logs = collector.records
    return result


def _build(job: SnapshotBuildJob) -> SnapshotBuildResult:
    # store は本モジュールを import するので関数内で (循環を避ける)
    from backend.free.rag.evidence.store import write_lexical_index

    store_dir = Path(job.store_dir)
    events = list(
        EvidenceEventLog(store_dir / "events").iter_since(job.folded_through, until=job.end),
    )
    prev: list[Evidence] = []
    if job.prev_version:
        prev = list(read_snapshot(store_dir, job.prev_version).iter_records())
    records = SnapshotWriter.fold(prev, events)

    kept: list[Evidence] = []
    dropped: list[str] = []
    for record in records:
        keep = record.id not in job.drop_ids
        if keep and job.keep is not None:
            try:
                keep = bool(job.keep(record))
            except Exception as e:  # noqa: BLE001 — GC の誤りで生きた記録を落とさない
                logger.warning(
                    "gc_filter raised on %s, keeping the record: %s", record.id, e,
                )
                keep = True
        if keep:
            kept.append(record)
        else:
            dropped.append(record.id)

    SnapshotWriter.write_snapshot(store_dir, job.version, kept)
    # 転置索引は ``write_snapshot`` へ渡したのと同じ列から作る (行番号が一対一)。
    # 前の版と中身が同じシャードは作り直さずに写す。
    shard_count = write_lexical_index(
        snapshot_dir(store_dir, job.version), kept,
        shard_key_for=job.shard_key_for, builder=job.builder,
        lexical_version=job.lexical_version,
        previous_dir=snapshot_dir(store_dir, job.prev_version) if job.prev_version else None,
    )
    return SnapshotBuildResult(
        records=len(kept), folded=len(events), dropped_ids=dropped,
        shard_count=shard_count,
        id_hash=ids_digest(record.id for record in kept),
        embed_prep=_prepare_embedding(kept, job.embed, Path(job.store_dir).name)
        if job.embed is not None else None,
    )


def _prepare_embedding(
    records: list[Evidence], spec: EmbedPrepSpec, store_name: str,
) -> EmbedPrep:
    """新しい版の行について、流用する行と埋め込む行を決める。

    ``EvidenceStore._snapshot_id_texts`` + 再利用判定と同じ結果になる (同じ関数を使う)。
    """
    from backend.free.rag.evidence.store import (
        embed_reuse_key,
        embed_side_of,
        plan_reuse,
        reusable_rows,
    )
    ids = [record.id for record in records]
    texts = [str(record.text or "") for record in records]
    sides = [embed_side_of(record.attrs) for record in records]
    hashes = [embed_reuse_key(text, *side) for text, side in zip(texts, sides)]
    reusable: dict[str, tuple[int, str]] = {}
    if spec.previous_dir and Path(spec.previous_dir).is_dir():
        previous = EvidenceVectorStore(
            Path(spec.previous_dir),
            memmap_threshold=spec.memmap_threshold,
            quantization=spec.quantization,
        )
        previous.load()
        reusable = reusable_rows(
            previous,
            declared_model_id=spec.declared_model_id,
            model_id=spec.model_id,
            expected_dim=spec.expected_dim,
            store_name=store_name,
        )
        # memmap を握ったまま返さない (Windows で版を消せなくなる)
        previous.vectors_q8 = None
        previous.scales = None
    reuse_dst, reuse_src, pending = plan_reuse(ids, hashes, reusable)
    return EmbedPrep(
        ids=ids, hashes=hashes, sides=sides,
        reuse_dst=reuse_dst, reuse_src=reuse_src, pending=pending,
        pending_texts=[texts[row] for row in pending],
    )


def _build_index(job: IndexBuildJob) -> IndexBuildResult:
    from backend.free.rag.evidence.store import assemble_embeddings, write_vector_store

    previous: EvidenceVectorStore | None = None
    if job.reuse_src and job.previous_dir and Path(job.previous_dir).is_dir():
        previous = EvidenceVectorStore(
            Path(job.previous_dir),
            memmap_threshold=job.memmap_threshold,
            quantization=job.quantization,
        )
        previous.load()
    q8, scales = assemble_embeddings(
        previous, job.reuse_dst, job.reuse_src, job.pending, job.vectors, len(job.ids),
    )
    store = EvidenceVectorStore(
        Path(job.directory),
        memmap_threshold=job.memmap_threshold,
        quantization=job.quantization,
    )
    write_vector_store(
        store, job.ids, job.hashes, job.sides, q8, scales,
        source=job.source, snapshot_rows=job.snapshot_rows, id_hash=job.id_hash,
        model_id=job.model_id, backend_type=job.backend_type,
        cluster=job.cluster, n_probe_ratio=job.n_probe_ratio,
    )
    store.vectors_q8 = None
    store.scales = None
    return IndexBuildResult(rows=len(job.ids))


# ── 実行先 ──────────────────────────────────────────────────────────

_pool: ProcessPoolExecutor | None = None
_pool_enabled = False
#: 応答の生成中 / 生成前のリクエストがあるか。起動時に配線する (未配線なら待たない)。
_busy_probe: Callable[[], bool] | None = None
#: 差し替え (L2) を延期する上限と確認間隔 (秒)。
L2_MAX_WAIT_SEC = 60.0
L2_POLL_SEC = 0.25
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def _lower_priority() -> None:
    """子プロセスの優先度を下げる (チャットの応答を先に通す)。"""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            # 型を宣言しないと擬似ハンドル (-1) が 32 bit の int で渡り、64 bit では
            # 無効なハンドルになって黙って失敗する (2026-09-23 実機で NORMAL のままだった)。
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), _BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(5)
    except Exception:  # noqa: BLE001 — 優先度を下げられなくても版は作れる
        pass


def set_busy_probe(probe: Callable[[], bool] | None) -> None:
    """応答の生成中 / 生成前のリクエストがあるかを返す関数を配線する (起動時)。"""
    global _busy_probe
    _busy_probe = probe


async def wait_until_idle(
    max_wait: float = L2_MAX_WAIT_SEC, poll: float = L2_POLL_SEC,
) -> float:
    """応答の合間になるまで待つ (最長 ``max_wait`` 秒)。待った秒数を返す。

    版の差し替え (L2) はループ上で数十〜百数十 ms かかるので、ストリーム中や
    生成前のリクエストに重ねない (G1 設計 §17.3)。上限を過ぎたら待たずに進む —
    応答が失敗して「入力はあったが応答が無い」状態が残っても版が止まらないように。
    """
    probe = _busy_probe
    if probe is None:
        return 0.0
    loop = asyncio.get_running_loop()
    started = loop.time()
    while True:
        try:
            busy = bool(probe())
        except Exception:  # noqa: BLE001 — 判定できなければ待たない
            busy = False
        waited = loop.time() - started
        if not busy:
            return waited
        if waited >= max_wait:
            logger.info("Snapshot swap waited %.0fs for a busy chat; swapping anyway", waited)
            return waited
        await asyncio.sleep(poll)


def enable_process_pool() -> None:
    """版の組み立てを子プロセスで行う (起動時に 1 回。テストは既定のスレッドのまま)。

    子プロセスは最初のジョブで起こす (起動を遅らせない)。
    """
    global _pool_enabled
    _pool_enabled = True


def shutdown_process_pool() -> None:
    """子プロセスを止める (シャットダウン時)。"""
    global _pool_enabled
    _pool_enabled = False
    _discard_pool()


def _discard_pool() -> None:
    global _pool
    pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _process_pool_for(job: SnapshotBuildJob | IndexBuildJob) -> ProcessPoolExecutor | None:
    """このジョブを渡せる子プロセス (無効 / 渡せないジョブなら ``None``)。"""
    global _pool
    if not _pool_enabled:
        return None
    if isinstance(job, SnapshotBuildJob):
        if job.keep is not None:
            return None
        try:
            pickle.dumps(job.shard_key_for)
        except (pickle.PicklingError, TypeError, AttributeError):
            return None
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_lower_priority,
        )
    return _pool


async def run_snapshot_build(job: SnapshotBuildJob) -> SnapshotBuildResult:
    """版の組み立てを子プロセス (使えなければスレッド) で走らせる。"""
    return await _run(build_snapshot_files, job, job.version)


async def run_index_build(job: IndexBuildJob) -> IndexBuildResult:
    """埋め込み索引の組み立てを子プロセス (使えなければスレッド) で走らせる。"""
    return await _run(build_index_files, job, Path(job.directory).name)


async def _run(fn: Callable[[Any], Any], job: Any, label: str) -> Any:
    loop = asyncio.get_running_loop()
    pool = _process_pool_for(job)
    if pool is not None:
        try:
            result = await loop.run_in_executor(pool, fn, job)
        except BrokenProcessPool as e:
            # 子プロセスが落ちた。同じジョブをスレッドでやり直す — 版番号は同じなので
            # 書きかけのファイルは原子的に置き換わる。子プロセスは次のジョブで起こし直す。
            logger.warning(
                "Snapshot worker process failed (%s); rebuilding %s in a thread",
                e, label,
            )
            _discard_pool()
        else:
            for name, level, message in result.logs:
                logging.getLogger(name).log(level, "%s", message)
            result.logs = []
            return result
    return await run_in_executor_with_context(loop, None, fn, job)
