"""単一書き手ロック ``store/.writer.lock`` (c_05 §0.5.8)。

- ファイルは **切り詰めずに** 開き、EOF の先 (offset 2^20) の 1 バイトをロックする
  (Windows は ``msvcrt.locking(LK_NBLCK)``、POSIX は ``fcntl.lockf``)。
- PID・ホスト名・起動時刻は **ロックしない領域** (先頭) に診断用として書く。
- stale 判定はしない — プロセスが終われば OS が解放する (kill 前提の停止でも残らない)。
- 同じプロセス内で二重に取らない (同じインスタンスを使い回す)。
"""

from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from backend.io.format_registry import FormatSpec, register_format

LOCK_FILENAME = ".writer.lock"

WRITER_LOCK_FORMAT = register_format(FormatSpec(
    format_id="store.writer_lock",
    version=1,
    klass="system",
    writers=frozenset({"free", "pro"}),
    path_key=f"store/{LOCK_FILENAME}",
    retention="one per data root; never deleted by reset",
    encodings=("bin",),
))
_LOCK_OFFSET = 1 << 20
_DIAG_BYTES = 512


class WriterLockHeld(RuntimeError):
    """別のプロセスが書き手ロックを持っている。"""

    def __init__(self, path: Path, holder: dict[str, Any] | None) -> None:
        self.path = path
        self.holder = holder or {}
        who = ", ".join(f"{k}={v}" for k, v in self.holder.items()) or "unknown holder"
        super().__init__(f"{path} is held by another evoref process ({who})")


@dataclass(slots=True)
class WriterLock:
    """保持中のロック。:meth:`release` かプロセス終了で解放される。"""

    path: Path
    _fh: IO[bytes] | None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            _unlock(fh)
        finally:
            fh.close()


def _lock(fh: IO[bytes]) -> None:
    fh.seek(_LOCK_OFFSET)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - 開発環境は Windows
        import fcntl

        fcntl.lockf(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, _LOCK_OFFSET)


def _unlock(fh: IO[bytes]) -> None:
    fh.seek(_LOCK_OFFSET)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover
        import fcntl

        fcntl.lockf(fh.fileno(), fcntl.LOCK_UN, 1, _LOCK_OFFSET)


def read_holder(path: Path) -> dict[str, Any] | None:
    """診断領域に書かれた保持者の情報 (読めなければ ``None``)。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(_DIAG_BYTES).rstrip(b"\x00 \n")
        value = json.loads(head.decode("utf-8")) if head else None
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def lock_held(store_dir: Path) -> bool:
    """別のプロセスが書き手ロックを持っているか (読むだけの CLI が稼働中かを知る)。

    ロックの領域を非ブロッキングで試し、取れたらすぐ放す (保持しない・診断領域に
    書かない)。ロックのファイルが無ければ誰も持っていない。
    """
    path = store_dir / LOCK_FILENAME
    try:
        fh = open(path, "r+b", buffering=0)  # noqa: SIM115 — 下の with で閉じる
    except FileNotFoundError:
        return False
    with fh:
        try:
            _lock(fh)
        except OSError:
            return True
        _unlock(fh)
    return False


def acquire_writer_lock(store_dir: Path, *, started_at: str = "") -> WriterLock:
    """``<store_dir>/.writer.lock`` を取る。取れなければ :class:`WriterLockHeld`。"""
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / LOCK_FILENAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    fh = os.fdopen(fd, "r+b", buffering=0)
    try:
        _lock(fh)
    except OSError:
        fh.close()
        raise WriterLockHeld(path, read_holder(path)) from None
    diag = json.dumps(
        {"pid": os.getpid(), "host": socket.gethostname(), "started_at": started_at},
        ensure_ascii=True,
    ).encode("ascii")[: _DIAG_BYTES - 1]
    fh.seek(0)
    fh.write(diag.ljust(_DIAG_BYTES - 1, b" ") + b"\n")
    return WriterLock(path=path, _fh=fh)


__all__ = ["LOCK_FILENAME", "WriterLock", "WriterLockHeld", "acquire_writer_lock", "lock_held", "read_holder"]
