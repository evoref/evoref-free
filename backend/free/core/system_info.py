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

from backend.free.core.vram_monitor import gpu_memory_snapshot
from backend.log_config import get_logger

logger = get_logger("core.system_info")

#: backend プロセスの起動時刻 (uptime の SSOT)。import 時に確定する。
#: ``free/api/system/status.py`` の ``/api/status`` も同じ値を使う —
#: 別々に持つと「稼働時間は？」の回答が API と食い違う。
PROCESS_START_TIME: float = time.time()


def process_uptime_seconds() -> float:
    """backend プロセスの稼働秒数を返す。"""
    return max(0.0, time.time() - PROCESS_START_TIME)


def format_uptime(seconds: float) -> str:
    """稼働秒数を ``3h 12m 05s`` 形式へ整形する。"""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


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


def get_gpu_lines() -> list[str]:
    """GPU / VRAM の状況を 1 GPU 1 行のテキストで返す。

    測れない環境でも「測れない」と明示する 1 行を返す。行そのものが無いと、
    base は「提供された情報に含まれていない」としか言えず、ユーザーには
    *ツールが答えられなかった* のか *この環境では測れない* のかが区別できない
    (実インシデント 2026-08-19 ライブ監査 ターン8:「この PC の空きメモリと GPU
    の VRAM 使用状況を教えてください。」に RAM だけ答え、VRAM は
    「該当するデータが含まれていない」と返した)。

    推測値は入れない。VRAM の推定値は ``GET /api/system/vram_status`` が別途
    持つが、あれは *モデルサイズからの見積り* であって実測の使用量ではないため、
    ここで実測値として並べてはいけない (本モジュール冒頭の方針と同じ)。
    """
    gpus = gpu_memory_snapshot()
    if not gpus:
        return ["GPU VRAM: unknown (not measurable on this host)"]
    return [
        f"GPU{i} {g['name']}: {g['used_mb']} MB used / {g['total_mb']} MB total"
        for i, g in enumerate(gpus)
    ]


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
        *get_gpu_lines(),
    ])


def format_runtime_facts(cfg: dict, metadata: object | None = None) -> str:
    """evoref 自身の実行構成をツール結果向けテキストへ整形する (準純粋関数)。

    「今動いているモデルは？」「コンテキストサイズは？」「llama-server の
    ポートは？」に答える経路がどこにも無かった。ハードウェアと同じで、
    シェル経由では取れない (readonly allow-list は config も /props も読めない)
    ため backend 内の値をそのまま渡す。実インシデント (2026-08-22 ライブ監査):
    「今動いているモデルの名前を教えてください。」→「私は「Alice」という名前で
    対応しています」(インスタンス名であってモデル名ではない)、「埋め込みモデルは
    何ですか？」→「特定の埋め込みモデル名を保持したり開示したりする仕様では
    ありません」(存在しない方針の捏造)、ポート / n_ctx →「確認できていません」。

    **実測 (llama-server の ``/props`` 由来) と宣言 (config) を区別して書く**。
    config を書き替えても llama-server を再起動しなければ反映されないため、
    両者は食い違いうる (2026-08-13: 62.8 時間ちがうモデルを serve していた)。
    ``metadata`` が無い / 値が 0 の項目は config 側を ``configured`` と明示する。
    """
    from pathlib import Path

    llama = cfg.get("llama", {}) or {}
    embedding = cfg.get("embedding", {}) or {}
    instance = cfg.get("instance", {}) or {}

    configured_base = Path(
        (cfg.get("model_paths", {}) or {}).get("base_model") or "",
    ).name
    served = getattr(metadata, "model_id", "") or ""
    if served:
        base_line = f"Base model (served): {Path(served).name}"
        if configured_base and Path(served).name != configured_base:
            base_line += f"  [config declares: {configured_base}]"
    elif configured_base:
        base_line = f"Base model (configured, not verified): {configured_base}"
    else:
        base_line = "Base model: unknown"

    n_ctx = int(getattr(metadata, "n_ctx", 0) or 0)
    if n_ctx:
        # **モデルカードの公称最大で上書きさせない。** 実測 (2026-09-03 ライブ監査):
        # ここが正しく 8192 を返しているターンで「コンテキスト長は 131072
        # トークンです」と答えた (Qwen3 の公称最大)。served 値が唯一の実効値で
        # あることを値のとなりに書く — 数字だけ置くと、モデルは自分の事前知識の
        # ほうを信用する。
        ctx_line = (
            f"Context size (n_ctx, served): {n_ctx}  "
            f"[this is the effective limit in use; the model architecture's "
            f"advertised maximum is NOT what this deployment runs]"
        )
    else:
        ctx_line = (
            f"Context size (n_ctx, configured): {llama.get('context_size', 'unknown')}"
        )
    slots = int(getattr(metadata, "total_slots", 0) or 0)
    slots_line = (
        f"Slots (served): {slots}" if slots
        else f"Slots (configured): {llama.get('slots', 'unknown')}"
    )

    embed_name = embedding.get("model_name") or "unknown"
    # バージョン / エディション / 稼働時間 — どれも backend 内にしか無く、
    # ``run_command_readonly`` の allow-list では取れない。ここが欠けていたため
    # 「あなたのバージョン番号は？」→「提供されていません」、「稼働時間は？」→
    # 「確認できるツールが利用できない」、「Free 版ですか Pro 版ですか？」→
    # 「該当しません」と、ツールは撃たれているのに答えられなかった
    # (2026-08-22 ライブ監査 2 回目 ターン 6/7/9)。
    from backend.edition import current_edition
    from backend.version import get_runtime_version

    lines = [
        f"Instance name: {instance.get('name') or 'unknown'}"
        "  (this is the assistant's display name, NOT the model name)",
        f"evoref version: {get_runtime_version()}",
        f"Edition: {current_edition().name.lower()}",
        f"Backend uptime: {format_uptime(process_uptime_seconds())}",
        base_line,
        f"Embedding model: {embed_name}",
        ctx_line,
        slots_line,
        f"llama-server (base): {llama.get('host', 'localhost')}:"
        f"{llama.get('port', 'unknown')}",
        f"llama-server (embedding): {embedding.get('llama_host', 'localhost')}:"
        f"{embedding.get('llama_port', 'unknown')}",
    ]
    return "\n".join(lines)
