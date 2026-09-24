"""チャット応答パスの書き手スレッド (c_05 §0.5.9)。

チャット経路の SoT 書き込み (経験の追記・履歴のターン追記・agent_trace) は全て
ここの **1 本の順序付きキュー** を通す。イベントループがするのは 1 件分の行の
直列化と enqueue だけで、ファイルへの書き込み・fsync・原子的な置き換えは
このスレッドが行う (G0 は ``done`` の前に経験バッファ全体を置き換えており、
上限時に 1 ターン約 60ms ループを止めていた)。

- **順序**: FIFO で処理する。同じファイルへの追記・置き換え・畳み込みは出した順に
  効く (追記の後に出した畳み込みがその追記を取りこぼさない)。
- **fsync**: 追記はファイルごとに印だけ付け、:meth:`ChatWriter.end_turn` (ターンの
  終わり) で 1 回、またはターンの外の追記なら最初の未同期の追記から
  ``sync_interval`` 秒 (既定 1 秒) で 1 回 fsync する。**耐久性の契約は「最後の
  1 秒 (または終わっていないターン) の追記は失ってよい」**。置き換え
  (:meth:`ChatWriter.replace`) は呼出側が指定した ``fsync`` で書く。
- **readonly**: enqueue の時点で :func:`backend.io.readonly.guard_write` が
  :class:`~backend.io.readonly.DataReadonlyError` を送出する (型付きの拒否。
  キューに積んでから黙って落とさない)。
- **失敗**: 書けなかった形式は degraded として :meth:`ChatWriter.degraded` に残し
  (``GET /api/status`` の ``data_health.degraded``)、応答は落とさない
  (c_05 §0.5.8、設計 §7.2)。
- **停止**: :meth:`ChatWriter.stop` がキューを最後まで処理し、未同期のファイルを
  fsync してハンドルを閉じる (lifespan の shutdown)。
- **未起動** (単体テスト・停止中の CLI) の間は、呼出側のスレッドでその場で実行する。
  順序と結果は同じで、ハンドルは操作ごとに閉じる (Windows で一時ディレクトリを
  消せなくしない)。

``backend/io`` は pillar を import しない。何を書くか (行の形・畳み込み) は各ストアが
持ち、ここは「どのファイルへ・どの順で・いつ fsync するか」だけを持つ。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from backend.io.atomic import AtomicWriter
from backend.io.readonly import DataReadonlyError, guard_write
from backend.log_config import get_logger

logger = get_logger("io.writer_thread")

#: 追記の fsync を遅らせてよい上限 (秒)。ターンの外の追記 (agent_trace の
#: バックグラウンド経路等) はこの間隔でまとめて同期する。
DEFAULT_SYNC_INTERVAL = 1.0
#: 保持する追記ハンドルの上限 (超えたら古いものから閉じる)。
_MAX_HANDLES = 32

_APPEND = "append"
_REPLACE = "replace"
_CALL = "call"
_SYNC = "sync"
_BARRIER = "barrier"
_STOP = "stop"


@dataclass(slots=True)
class _Op:
    kind: str
    format_id: str = ""
    path: Path | None = None
    data: Any = None
    fsync: bool = False
    fn: Callable[[], Any] | None = None
    paths: tuple[Path, ...] = ()
    future: Future | None = None
    event: threading.Event | None = None


def _key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


class ChatWriter:
    """チャット経路の書き手 (1 本のスレッド + 順序付きキュー)。"""

    def __init__(self, *, sync_interval: float = DEFAULT_SYNC_INTERVAL, name: str = "chat-writer") -> None:
        self.sync_interval = float(sync_interval)
        self._name = name
        self._queue: queue.SimpleQueue[_Op] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        #: 起動 / 停止と enqueue を直列化する (停止中の enqueue はスレッドが
        #: 終わるまで待ち、その後その場で実行する — 順序を入れ替えない)。
        self._mode_lock = threading.RLock()
        # 以下は書き手スレッド (未起動ならモードロックの下の呼出側) だけが触る。
        self._handles: dict[str, BinaryIO] = {}
        #: 未同期の追記: パスのキー → (形式, 最初の未同期の時刻)。
        self._unsynced: dict[str, tuple[Path, str, float]] = {}
        self._ticks: list[Callable[[bool], None]] = []
        self._degraded: dict[str, str] = {}
        self._degraded_lock = threading.Lock()

    # ── 起動・停止 ──

    @property
    def running(self) -> bool:
        return self._thread is not None

    def is_writer_thread(self) -> bool:
        """呼出側が起動中の書き手スレッドそのものか (その中からは待たずに直接実行する)。"""
        thread = self._thread
        return thread is not None and threading.current_thread() is thread

    def start(self) -> None:
        """書き手スレッドを起こす (二重起動は無視)。"""
        with self._mode_lock:
            if self._thread is not None:
                return
            # その場実行の間に開いたハンドルは無い (操作ごとに閉じている)。
            thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread = thread
            thread.start()

    def stop(self, timeout: float = 30.0) -> bool:
        """キューを最後まで処理して止める。``timeout`` 内に終われば ``True``。

        止めた後の書き込みは呼出側のスレッドでその場で実行する。
        """
        with self._mode_lock:
            thread = self._thread
            if thread is None:
                return True
            done = threading.Event()
            self._queue.put(_Op(_STOP, event=done))
            finished = done.wait(timeout)
            thread.join(max(0.0, timeout) if finished else 0.1)
            if not finished:
                logger.warning(
                    "Chat writer did not drain within %.1fs; pending writes may be lost", timeout,
                )
                return False
            self._thread = None
            return True

    def drain(self, timeout: float = 30.0) -> bool:
        """ここまでに出した操作が全て終わるまで待つ (未同期の追記も fsync する)。"""
        with self._mode_lock:
            if self._thread is None:
                self._sync_all()
                return True
            event = threading.Event()
            self._queue.put(_Op(_BARRIER, event=event))
        return event.wait(timeout)

    def add_tick(self, fn: Callable[[bool], None]) -> None:
        """書き手スレッドの空き時間 (約 1 秒ごと) と停止時 (``final=True``) に呼ぶ関数。

        遅延書き出し (履歴の索引) に使う。未起動の間は呼ばれない。
        """
        with self._mode_lock:
            self._ticks.append(fn)

    def remove_tick(self, fn: Callable[[bool], None]) -> None:
        with self._mode_lock:
            if fn in self._ticks:
                self._ticks.remove(fn)

    # ── enqueue (呼出側のスレッド) ──

    def append(self, path: Path | str, lines: Iterable[str], *, format_id: str) -> None:
        """``lines`` (改行を含まない 1 行ずつの文字列) を ``path`` へ追記する。

        readonly の ``store/`` なら :class:`~backend.io.readonly.DataReadonlyError`。
        """
        target = Path(path)
        payload = list(lines)
        if not payload:
            return
        guard_write(target)
        self._submit(_Op(_APPEND, format_id=format_id, path=target, data=payload))

    def replace(
        self, path: Path | str, data: bytes | str | Callable[[], bytes | str], *,
        format_id: str, fsync: bool,
    ) -> None:
        """``path`` を ``data`` で原子的に置き換える (追記ハンドルは先に閉じる)。

        ``data`` が関数なら書き手スレッドで呼んで本文を得る (コンパクションのように
        ディスクの状態から組み立てる本文。ここまでの追記を読んだ後で作る)。
        """
        target = Path(path)
        guard_write(target)
        self._submit(_Op(_REPLACE, format_id=format_id, path=target, data=data, fsync=fsync))

    def call(
        self, fn: Callable[[], Any], *, format_id: str, paths: Iterable[Path | str] = (),
    ) -> Future:
        """``fn`` を書き手スレッドで実行する (畳み込み・コンパクション・削除)。

        ``paths`` の追記ハンドルは実行前に fsync して閉じる (``fn`` が置き換え・削除
        できるように)。readonly なら ``paths`` の検査で拒否する。結果 / 例外は
        返り値の ``Future`` に載る。例外は degraded にも数える。
        """
        targets = tuple(Path(p) for p in paths)
        for target in targets:
            guard_write(target)
        future: Future = Future()
        self._submit(_Op(_CALL, format_id=format_id, fn=fn, paths=targets, future=future))
        return future

    def end_turn(self) -> None:
        """ターンの終わり: ここまでの追記をファイルごとに 1 回 fsync する。"""
        self._submit(_Op(_SYNC))

    # ── :meth:`call` の関数の中から (書き手スレッド上で同期に) ──

    def write_now(self, path: Path | str, data: bytes | str, *, fsync: bool) -> None:
        """書き手スレッドの上で ``path`` をその場で原子的に置き換える。

        :meth:`call` に渡した関数 (畳み込み・コンパクション) の中からだけ呼ぶ。
        失敗は送出する (呼んだ関数ごと失敗し、その形式が degraded になる)。
        """
        target = Path(path)
        self._check_owner()
        guard_write(target)
        self._release(target)
        self._replace(target, data, fsync)

    def remove_now(self, path: Path | str) -> bool:
        """書き手スレッドの上で ``path`` を消す (無ければ ``False``)。"""
        target = Path(path)
        self._check_owner()
        guard_write(target)
        self._release(target)
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        return True

    def _check_owner(self) -> None:
        thread = self._thread
        if thread is not None and threading.current_thread() is not thread:
            raise RuntimeError("write_now / remove_now must run on the writer thread (inside call())")

    def degraded(self) -> dict[str, str]:
        """書けなかった形式 → 最後の理由 (英語)。"""
        with self._degraded_lock:
            return dict(self._degraded)

    def _submit(self, op: _Op) -> None:
        with self._mode_lock:
            if self._thread is not None:
                self._queue.put(op)
                return
            try:
                self._execute(op)
            finally:
                # その場実行ではハンドルを持ち越さない (fsync の印は残す)。
                self._close_handles()

    # ── 書き手スレッド ──

    def _run(self) -> None:
        while True:
            try:
                op = self._queue.get(timeout=self._wait_seconds())
            except queue.Empty:
                self._idle()
                continue
            if op.kind == _STOP:
                self._finish()
                if op.event is not None:
                    op.event.set()
                return
            self._execute(op)
            if self._queue.empty():
                self._idle()

    def _wait_seconds(self) -> float:
        if self._unsynced:
            oldest = min(since for _, _, since in self._unsynced.values())
            return max(0.0, oldest + self.sync_interval - time.monotonic())
        return 1.0

    def _idle(self) -> None:
        if self._unsynced:
            now = time.monotonic()
            if any(now - since >= self.sync_interval for _, _, since in self._unsynced.values()):
                self._sync_all()
        self._run_ticks(final=False)

    def _finish(self) -> None:
        self._sync_all()
        self._run_ticks(final=True)
        self._sync_all()
        self._close_handles()

    def _run_ticks(self, *, final: bool) -> None:
        for fn in list(self._ticks):
            try:
                fn(final)
            except Exception as exc:  # 遅延書き出しの失敗は次の tick でやり直す
                logger.warning("Chat writer tick failed: %s", exc)

    def _execute(self, op: _Op) -> None:
        try:
            if op.kind == _APPEND:
                self._append(op.path, op.data, op.format_id)
            elif op.kind == _REPLACE:
                self._release(op.path)
                self._replace(op.path, op.data, op.fsync)
            elif op.kind == _CALL:
                for p in op.paths:
                    self._release(p)
                result = op.fn()
                if op.future is not None:
                    op.future.set_result(result)
            elif op.kind == _SYNC:
                self._sync_all()
            elif op.kind == _BARRIER:
                self._sync_all()
                if op.event is not None:
                    op.event.set()
        except Exception as exc:
            if op.path is not None:
                self._drop_handle(_key(op.path))
            # readonly は失敗ではなく拒否 (data_health は readonly で知らせる)。
            if not isinstance(exc, DataReadonlyError):
                self._mark_degraded(op.format_id or "unknown", op.path, exc)
            if op.future is not None and not op.future.done():
                op.future.set_exception(exc)

    def _append(self, path: Path, lines: list[str], format_id: str) -> None:
        key = _key(path)
        handle = self._handles.get(key)
        if handle is None:
            handle = self._open_append(path)
            self._handles[key] = handle
            if len(self._handles) > _MAX_HANDLES:
                oldest = next(iter(self._handles))
                self._close_handle(oldest)
        handle.write(("\n".join(lines) + "\n").encode("utf-8"))
        handle.flush()
        if key not in self._unsynced:
            self._unsynced[key] = (path, format_id, time.monotonic())

    def _open_append(self, path: Path) -> BinaryIO:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 開くたびに末尾を見る (ハンドルを持っている間は自分しか書かない)。
        self._terminate_torn_tail(path)
        return open(path, "ab")  # noqa: SIM115 — ハンドルは書き手が保持する

    @staticmethod
    def _terminate_torn_tail(path: Path) -> None:
        """追記用に初めて開くとき、末尾が改行で終わっていなければ改行で終端する。

        途中で切れた行 (kill・電源断) を切り詰めない — 断片は壊れた行として読み手が
        飛ばし、物理行の位置は保たれる (c_05 §0.5.8)。
        """
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return
        if size == 0:
            return
        with open(path, "rb") as f:
            f.seek(size - 1)
            last = f.read(1)
        if last == b"\n":
            return
        with open(path, "ab") as f:
            f.write(b"\n")
        logger.warning("Terminated a torn last line in %s (%d bytes)", path, size)

    @staticmethod
    def _replace(path: Path, data: Any, fsync: bool) -> None:
        if callable(data):
            data = data()
        raw = data.encode("utf-8") if isinstance(data, str) else data
        with AtomicWriter(path, mode="wb", fsync=fsync) as f:
            f.write(raw)

    def _release(self, path: Path) -> None:
        """``path`` の追記ハンドルを fsync して閉じる (置き換え・削除の前)。"""
        key = _key(path)
        if key in self._unsynced:
            self._sync_one(key)
        self._close_handle(key)

    def _sync_all(self) -> None:
        for key in list(self._unsynced):
            self._sync_one(key)

    def _sync_one(self, key: str) -> None:
        path, format_id, _ = self._unsynced.pop(key)
        handle = self._handles.get(key)
        try:
            if handle is not None:
                handle.flush()
                os.fsync(handle.fileno())
            elif path.exists():
                with open(path, "ab") as f:
                    os.fsync(f.fileno())
        except OSError as exc:
            self._drop_handle(key)
            self._mark_degraded(format_id, path, exc)

    def _close_handle(self, key: str) -> None:
        handle = self._handles.pop(key, None)
        if handle is None:
            return
        try:
            handle.close()
        except OSError as exc:
            logger.warning("Chat writer failed to close %s: %s", key, exc)

    def _drop_handle(self, key: str) -> None:
        handle = self._handles.pop(key, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def _close_handles(self) -> None:
        for key in list(self._handles):
            self._close_handle(key)

    def _mark_degraded(self, format_id: str, path: Path | None, exc: BaseException) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        with self._degraded_lock:
            first = format_id not in self._degraded
            self._degraded[format_id] = reason
        if first:
            logger.warning(
                "Chat-path write failed; format %s is degraded (path=%s): %s",
                format_id, path, reason,
            )
        else:
            logger.debug("Chat-path write failed again (format=%s): %s", format_id, reason)


_default = ChatWriter()


def default_writer() -> ChatWriter:
    """プロセス既定の書き手 (lifespan が起動・停止する)。"""
    return _default


__all__ = ["DEFAULT_SYNC_INTERVAL", "ChatWriter", "default_writer"]
