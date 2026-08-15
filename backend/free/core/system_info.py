"""実行ホストのハードウェア事実 (RAM / CPU / OS) を非シェル経路で取得する。

チャットの ``run_command_readonly`` は ``safety_patterns._READONLY_SAFE_MODULES``
の allow-list (datetime / platform / os / sys / socket / shutil / time / math /
json / locale) しか通さないため、Windows で搭載 RAM を取る手段 (ctypes /
wmic / Get-CimInstance) が **すべて拒否される**。結果として「このPCのメモリ容量
は？」はツールを 1 つも実行できず、`_UNMEASURED_FACT_GUIDANCE` で「今回は調べて
いないので分からない」と答えるしかなかった (2026-08-12 ライブ監査で確認)。

allow-list はチャットから渡される **コマンド文字列** にしか掛からない。backend
自身の Python は制約外で、既に ``free/core/vram_monitor.py`` が nvidia-smi を
subprocess で直接叩いている。本モジュールはその前例に倣い、シェルを介さずに
プロセス内で測る。allow-list は一切広げない。

測れなかった項目は推測せず ``None`` を返す (呼び出し側が「不明」と述べる)。
"""

from __future__ import annotations

import ctypes
import os
import platform
import time

from backend.log_config import get_logger

logger = get_logger("core.system_info")


class _MemoryStatusEx(ctypes.Structure):
    """Win32 ``MEMORYSTATUSEX`` (winbase.h)。"""

    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def _windows_memory_bytes() -> tuple[int, int] | None:
    """Windows の (搭載 RAM, 利用可能 RAM) をバイトで返す。失敗時 None。

    ``ctypes.windll`` は backend 内で既に使用実績がある
    (``free/cli/pid_manager.py`` / ``free/cli/renderer.py``)。
    """
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except (AttributeError, OSError, ValueError) as e:
        logger.debug("GlobalMemoryStatusEx failed: %s", e)
        return None


def _posix_memory_bytes() -> tuple[int, int] | None:
    """POSIX の (搭載 RAM, 利用可能 RAM) をバイトで返す。失敗時 None。"""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page_size
    except (AttributeError, ValueError, OSError) as e:
        logger.debug("sysconf SC_PHYS_PAGES failed: %s", e)
        return None
    try:
        avail = os.sysconf("SC_AVPHYS_PAGES") * page_size
    except (AttributeError, ValueError, OSError):
        avail = 0
    return int(total), int(avail)


def get_memory_bytes() -> tuple[int, int] | None:
    """(搭載 RAM, 利用可能 RAM) をバイトで返す。測れなければ None。"""
    if os.name == "nt":
        return _windows_memory_bytes()
    return _posix_memory_bytes()


class _FileTime(ctypes.Structure):
    """Win32 ``FILETIME`` (minwinbase.h)。100ns 単位の 64bit を 2 ワードで持つ。"""

    _fields_ = (
        ("dwLowDateTime", ctypes.c_ulong),
        ("dwHighDateTime", ctypes.c_ulong),
    )


def _filetime_to_int(ft: "_FileTime") -> int:
    return (int(ft.dwHighDateTime) << 32) | int(ft.dwLowDateTime)


def _windows_cpu_times() -> tuple[int, int] | None:
    """Windows の (idle, total) 累積 CPU 時間を返す。失敗時 None。"""
    try:
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user),
        )
        if not ok:
            return None
        # kernel には idle が含まれる (Win32 仕様)。total = kernel + user。
        return _filetime_to_int(idle), (
            _filetime_to_int(kernel) + _filetime_to_int(user)
        )
    except (AttributeError, OSError, ValueError) as e:
        logger.debug("GetSystemTimes failed: %s", e)
        return None


def _posix_cpu_times() -> tuple[int, int] | None:
    """POSIX の (idle, total) 累積 CPU 時間を ``/proc/stat`` から返す。"""
    try:
        with open("/proc/stat", encoding="ascii") as fh:
            fields = fh.readline().split()
    except OSError as e:
        logger.debug("/proc/stat read failed: %s", e)
        return None
    if len(fields) < 5 or fields[0] != "cpu":
        return None
    try:
        values = [int(v) for v in fields[1:]]
    except ValueError:
        return None
    # idle = idle + iowait (3 番目と 4 番目)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


def get_cpu_usage_percent(interval_sec: float = 0.15) -> float | None:
    """直近 ``interval_sec`` の CPU 使用率 (%) を返す。測れなければ None。

    psutil は入れず、シェルも介さない (``run_command_readonly`` の allow-list は
    コマンド文字列にしか掛からないという本モジュールの前提は維持する)。
    累積 CPU 時間を 2 点サンプリングして差分から求めるため、``interval_sec``
    だけブロックする。ツール 1 回あたりの追加コストなので短く固定する。
    """
    sampler = _windows_cpu_times if os.name == "nt" else _posix_cpu_times
    first = sampler()
    if first is None:
        return None
    time.sleep(max(interval_sec, 0.0))
    second = sampler()
    if second is None:
        return None
    idle_delta = second[0] - first[0]
    total_delta = second[1] - first[1]
    if total_delta <= 0:
        return None
    usage = (1.0 - idle_delta / total_delta) * 100.0
    return max(0.0, min(100.0, usage))


def get_hardware_facts() -> dict[str, object]:
    """実行ホストのハードウェア事実を返す (純粋な読み取り、副作用なし)。

    Returns:
        ``os`` / ``cpu`` / ``cores`` は常に埋まる。``total_ram_mb`` /
        ``available_ram_mb`` / ``cpu_usage_percent`` は測定できなかった場合
        ``None``。
    """
    mem = get_memory_bytes()
    return {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "total_ram_mb": mem[0] // (1024 * 1024) if mem else None,
        "available_ram_mb": mem[1] // (1024 * 1024) if mem else None,
        "cpu_usage_percent": get_cpu_usage_percent(),
    }


def format_hardware_facts() -> str:
    """``get_hardware_facts`` をツール結果向けの 1 行/項目テキストへ整形する。

    測れなかった項目は "unknown (not measurable on this host)" と明示する。
    数値を推測で埋めない — 埋めると base がそれを実測値として述べる。
    """
    f = get_hardware_facts()
    total = f["total_ram_mb"]
    avail = f["available_ram_mb"]
    if total is None:
        ram_line = "RAM: unknown (not measurable on this host)"
    else:
        gb = int(total) / 1024
        ram_line = f"RAM: {gb:.1f} GB total ({total} MB)"
        if avail is not None:
            ram_line += f", {int(avail) / 1024:.1f} GB available"
    usage = f["cpu_usage_percent"]
    if usage is None:
        cpu_usage_line = "CPU usage: unknown (not measurable on this host)"
    else:
        cpu_usage_line = f"CPU usage: {float(usage):.1f}%"
    return "\n".join([
        f"OS: {f['os']}",
        f"CPU: {f['cpu']}",
        f"Cores: {f['cores']}",
        ram_line,
        cpu_usage_line,
    ])
