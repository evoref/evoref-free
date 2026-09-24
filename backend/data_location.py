"""データ根の置き場の判定 (c_05 §0.2)。

- ネットワークドライブ / UNC / OneDrive 配下 → 起動拒否 (``--allow-unsafe-data-root``
  で警告付き起動)。同期クライアントや SMB は rename と fsync の順序を保証しない。
- FAT / exFAT → 警告 (ジャーナルが無く、rename と fsync の順序保証が弱い)。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_ONEDRIVE_ENV = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
_WEAK_FILESYSTEMS = frozenset({"FAT", "FAT32", "EXFAT", "VFAT", "MSDOS"})
_DRIVE_REMOTE = 4


@dataclass(frozen=True, slots=True)
class DataLocation:
    """置き場の判定結果。``fs_type`` は取れなければ ``None``。"""

    path: Path
    fs_type: str | None
    remote: bool
    onedrive: bool

    @property
    def unsafe(self) -> bool:
        return self.remote or self.onedrive

    @property
    def weak_fs(self) -> bool:
        return (self.fs_type or "").upper() in _WEAK_FILESYSTEMS


def _within(path: Path, root: str) -> bool:
    try:
        Path(os.path.normcase(os.path.abspath(path))).relative_to(
            os.path.normcase(os.path.abspath(root)),
        )
    except ValueError:
        return False
    return True


def _windows_volume(path: Path) -> tuple[str | None, bool]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.GetVolumePathNameW.restype = wintypes.BOOL
    kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetDriveTypeW.restype = wintypes.UINT
    kernel32.GetVolumeInformationW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    kernel32.GetVolumeInformationW.restype = wintypes.BOOL

    volume = ctypes.create_unicode_buffer(261)
    probe = str(_existing_ancestor(path))
    if not kernel32.GetVolumePathNameW(probe, volume, 261):
        return None, probe.startswith("\\\\")
    remote = kernel32.GetDriveTypeW(volume.value) == _DRIVE_REMOTE
    fs_name = ctypes.create_unicode_buffer(261)
    ok = kernel32.GetVolumeInformationW(volume.value, None, 0, None, None, None, fs_name, 261)
    return (fs_name.value or None) if ok else None, remote or probe.startswith("\\\\")


def _existing_ancestor(path: Path) -> Path:
    current = Path(os.path.abspath(path))
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def inspect_location(path: Path, *, env: dict[str, str] | None = None) -> DataLocation:
    """``path`` (未作成でもよい) の置き場を判定する。"""
    environ = os.environ if env is None else env
    onedrive = any(environ.get(name) and _within(path, environ[name]) for name in _ONEDRIVE_ENV)
    fs_type: str | None = None
    remote = str(path).startswith(("\\\\", "//"))
    if sys.platform == "win32":
        try:
            fs_type, remote_volume = _windows_volume(path)
            remote = remote or remote_volume
        except (OSError, AttributeError):
            pass
    return DataLocation(path=Path(path), fs_type=fs_type, remote=remote, onedrive=onedrive)


__all__ = ["DataLocation", "inspect_location"]
