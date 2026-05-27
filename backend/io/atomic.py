"""原子的ファイル書き込み

``tempfile.mkstemp`` でユニーク tmp ファイル名を発番 → 書き込み完了後に
``os.replace`` (+ Windows ``PermissionError`` retry) で宛先に切り替える。
書き込み途中で例外が発生した場合は tmp ファイルを ``unlink`` し宛先ファイルは
不変のまま残す。

固定名 ``<dst>.tmp`` を共有する素朴な実装は、Windows で別プロセスが open 中
の tmp に対して ``write_text`` / ``os.replace`` が ``ERROR_SHARING_VIOLATION``
を起こすため避ける。``tempfile.mkstemp(prefix=path.name+".", suffix=".tmp",
dir=path.parent)`` で並行プロセス間衝突を回避する。

Windows ``os.replace`` の retry 仕様は :mod:`backend.io._retry` を参照。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import TracebackType
from typing import IO, TYPE_CHECKING, Any, Literal

from backend.io._retry import _replace_with_retry
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("io.atomic")


class AtomicWriter:
    """原子的ファイル書き込みのコンテキストマネージャ。

    Usage:
        >>> with AtomicWriter(path) as f:
        ...     f.write("...")
        >>> # ここまで来れば path に内容が atomic に反映済み

        >>> # バイナリ書き込み
        >>> with AtomicWriter(path, mode="wb") as f:
        ...     f.write(b"...")

    例外が発生した場合は tmp ファイルが unlink され、宛先 ``path`` は変更されない。

    Args:
        path: 書き込み先パス。
        mode: ``"w"`` (テキスト、デフォルト) または ``"wb"`` (バイナリ)。
        encoding: テキストモード時の encoding (デフォルト ``"utf-8"``)。
            バイナリモード時は無視。
        fsync: 書き込み後に ``os.fsync`` を呼ぶか (manifest など crash 耐性を
            必要とする箇所のみ ``True`` 推奨)。
        debug_logger: 注入された ``DebugLogger`` があれば、書き込み完了時に
            ``log_memory_op("atomic_write", ...)`` で観測情報を流す。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        mode: Literal["w", "wb"] = "w",
        encoding: str | None = "utf-8",
        fsync: bool = False,
        debug_logger: "DebugLogger | None" = None,
    ) -> None:
        if mode not in ("w", "wb"):
            raise ValueError(f"AtomicWriter mode must be 'w' or 'wb', got {mode!r}")
        self._path = Path(path)
        self._mode = mode
        self._encoding = encoding if mode == "w" else None
        self._fsync = fsync
        self._debug_logger = debug_logger
        self._tmp_path: Path | None = None
        self._fh: IO[Any] | None = None
        self._fd: int | None = None
        self._bytes_written = 0

    def __enter__(self) -> IO[Any]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_str = tempfile.mkstemp(
            prefix=self._path.name + ".",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        self._tmp_path = Path(tmp_str)
        self._fd = fd
        if self._mode == "w":
            self._fh = os.fdopen(fd, "w", encoding=self._encoding or "utf-8")
        else:
            self._fh = os.fdopen(fd, "wb")
        # mkstemp が返した fd は os.fdopen に所有権が移ったので self._fd は参照用
        self._fd = None
        return self._fh

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        tmp = self._tmp_path
        fh = self._fh
        # 1) ファイルハンドルクローズ (失敗時も含めて必ず close)
        if fh is not None:
            try:
                if exc_type is None and self._fsync:
                    fh.flush()
                    os.fsync(fh.fileno())
                # 書き込みバイト数 (テキストモードでは近似値)
                if exc_type is None:
                    try:
                        self._bytes_written = fh.tell()
                    except (OSError, ValueError):
                        self._bytes_written = 0
            finally:
                try:
                    fh.close()
                except OSError:
                    pass
        # 2) 例外発生時: tmp ファイルを消す
        if exc_type is not None:
            if tmp is not None:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
                except OSError as cleanup_err:
                    logger.warning(
                        "Failed to remove tmp file %s after exception: %s",
                        tmp, cleanup_err,
                    )
            return  # 元例外をそのまま再送出
        # 3) 正常終了時: tmp -> dst を atomic に切り替え (Windows retry 付き)
        assert tmp is not None
        try:
            retry_count = _replace_with_retry(tmp, self._path)
        except OSError:
            # replace 自体失敗時: tmp を unlink して例外を伝播
            try:
                tmp.unlink()
            except (FileNotFoundError, OSError):
                pass
            raise
        # 4) 観測情報
        if self._debug_logger is not None:
            try:
                self._debug_logger.log_memory_op(
                    "atomic_write",
                    {
                        "path": str(self._path),
                        "bytes": self._bytes_written,
                        "retry_count": retry_count,
                        "mode": self._mode,
                    },
                )
            except Exception as log_err:  # noqa: BLE001 — observability must not break writes
                logger.warning("DebugLogger.log_memory_op failed: %s", log_err)


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
    fsync: bool = False,
    debug_logger: "DebugLogger | None" = None,
) -> None:
    """テキスト内容を原子的にファイルに書き込む簡易関数。

    短い文字列を 1 度書くだけのユースケース向け。複数 write を分割したい場合は
    :class:`AtomicWriter` をコンテキストマネージャとして使う。
    """
    with AtomicWriter(
        path,
        mode="w",
        encoding=encoding,
        fsync=fsync,
        debug_logger=debug_logger,
    ) as f:
        f.write(text)


def atomic_write_bytes(
    path: Path | str,
    data: bytes,
    *,
    fsync: bool = False,
    debug_logger: "DebugLogger | None" = None,
) -> None:
    """バイト列を原子的にファイルに書き込む簡易関数。"""
    with AtomicWriter(
        path,
        mode="wb",
        fsync=fsync,
        debug_logger=debug_logger,
    ) as f:
        f.write(data)
