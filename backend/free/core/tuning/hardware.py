"""この PC のハードウェアの輪郭 (c_16 §7.2.3 ``hardware.py``)。

判定は純関数 (:func:`classify_gpu` / :func:`estimate_physical_cores` / :func:`build_profile`) で、
プローブ (RAM / ``llama-server --list-devices`` / コア数 / OS) は :class:`HardwareProbes` で
注入できる薄い層に分ける。``--list-devices`` は起動スクリプトの既存実装を lazy import し、
無い環境 (backend だけの配布物等) では GPU なしに縮退する。

GPU の種別 (``igpu`` / ``dgpu`` / ``none``) は名前と総メモリから推定する。**外れたときの害が小さい側に
倒す**: iGPU 扱いは後続の項目でヘッドルームを多めに・バッチを控えめにするだけ (遅くなるが OOM しない)
なので、名前で dGPU と言い切れないものは iGPU 側に倒す。判定根拠は ``kind_reason`` に残す。
"""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

GpuKind = Literal["igpu", "dgpu", "none"]

#: GPU の総メモリが RAM の総量のこの割合以上なら共有メモリ (iGPU) とみなす。
#: 実測: Radeon 890M は RAM 63 GB に対して 48923 MiB (約 76%) を総量として報告する。
#: 単体 GPU は 24 GB / RAM 64 GB で 37% 程度。
SHARED_MEMORY_RATIO = 0.5

# 名前で種別を言い切れるもの (大小無視)。上から順に見て最初に当たったものを採る。
_NAME_RULES: tuple[tuple[re.Pattern[str], GpuKind, str], ...] = (
    (re.compile(r"llvmpipe|swiftshader|softpipe|microsoft basic render|lavapipe", re.I), "none", "software_renderer"),
    # Intel Arc の単体 GPU は型番 (A770 / B580 等) を持つ。型番の無い "Arc(TM) Graphics" は Meteor Lake 以降の iGPU。
    (re.compile(r"\barc\b(?:\(tm\))?\s+(?:pro\s+)?[ab]\d{3}", re.I), "dgpu", "name_intel_arc_discrete"),
    # AMD の APU は名前が "Graphics" で終わる ("AMD Radeon(TM) 890M Graphics" / "Radeon(TM) RX Vega 11 Graphics")。
    # 単体の Radeon (RX 7900 XTX / Pro W7900) は "Graphics" で終わらないので、RX より先に見る。
    (re.compile(r"radeon.*graphics\s*$", re.I), "igpu", "name_amd_apu_graphics"),
    (re.compile(r"geforce|quadro|tesla|\brtx\b|\bgtx\b|nvidia\s+[a-z]?\d{2,}", re.I), "dgpu", "name_nvidia_discrete"),
    (re.compile(r"radeon.*\b(?:rx|pro\s+w\d+|vii)\b|\binstinct\b", re.I), "dgpu", "name_amd_discrete"),
    (re.compile(r"intel.*(?:uhd|iris|hd graphics|\(r\) graphics|arc(?:\(tm\))?\s+graphics)", re.I), "igpu", "name_intel_integrated"),
    (re.compile(r"\bapple\b|\bm[1-9]\b", re.I), "igpu", "name_apple_unified"),
)


@dataclass(frozen=True)
class GpuDevice:
    """``--list-devices`` の 1 デバイス。``name`` は指紋と同じ表記 (``"Vulkan0: AMD Radeon(TM) 890M Graphics"``)。"""

    name: str
    total_mib: int
    free_mib: int
    kind: GpuKind
    #: 種別を決めた根拠 (``name_*`` / ``shared_memory_ratio`` / ``unknown_conservative`` / ``software_renderer``)。
    kind_reason: str


@dataclass(frozen=True)
class HardwareProfile:
    """この PC の輪郭 (RAM / GPU / コア / OS)。取れなかった値は 0。"""

    total_ram_mib: int = 0
    free_ram_mib: int = 0
    gpus: tuple[GpuDevice, ...] = ()
    physical_cores: int = 0
    logical_cores: int = 0
    #: 物理コア数を測れず論理コア数から推定した (:func:`estimate_physical_cores`)。
    physical_cores_estimated: bool = False
    os: str = ""
    #: プローブで失敗したもの (``list_devices`` / ``ram`` 等)。縮退した理由の記録。
    degraded: tuple[str, ...] = field(default_factory=tuple)

    @property
    def gpu_names(self) -> list[str]:
        """PC の指紋 (``rerank_selftest.collect_pc_info``) に渡すデバイス名。"""
        return [g.name for g in self.gpus]

    @property
    def best_gpu(self) -> GpuDevice | None:
        """空きの最大の 1 枚 (``launch_llama.gpu_fit`` と同じ選び方)。ソフトウェア描画は除く。"""
        real = [g for g in self.gpus if g.kind != "none"]
        return max(real, key=lambda g: g.free_mib) if real else None


# ── 純関数 ────────────────────────────────────────────────


def classify_gpu(name: str, total_mib: int, total_ram_mib: int) -> tuple[GpuKind, str]:
    """GPU の種別と根拠。名前 → 総メモリの割合 → 保守側 (iGPU) の順に決める。

    名前で単体 GPU と言い切れるもの (GeForce / Radeon RX / Arc A770 等) だけを ``dgpu`` にする。
    言い切れないものは iGPU に倒す — 後続の項目で iGPU はヘッドルームを多めに・バッチを控えめに
    取るので、外れても遅くなるだけで OOM しない (逆に iGPU を dGPU と誤ると共有 RAM を食い潰しうる)。
    """
    product = name.split(":", 1)[1].strip() if ":" in name else name.strip()
    for pattern, kind, reason in _NAME_RULES:
        if pattern.search(product):
            return kind, reason
    if total_ram_mib > 0 and total_mib >= total_ram_mib * SHARED_MEMORY_RATIO:
        return "igpu", "shared_memory_ratio"
    return "igpu", "unknown_conservative"


def estimate_physical_cores(logical: int) -> int:
    """物理コア数を測れないときの推定: 論理の半分 (1 未満にしない)。

    x86 の SMT は 2-way が大半なので半分がほぼ正しく、SMT の無い CPU (Apple Silicon 等) では
    少なめに出る。少なめはスレッドの割り当てを控えめにする側 (過剰なスレッドで base と取り合わない)。
    """
    return max(1, logical // 2) if logical > 0 else 0


def parse_devices(text: str, total_ram_mib: int) -> tuple[GpuDevice, ...]:
    """``--list-devices`` の出力から GPU (``host`` 以外) を取り出す。

    名前は指紋と同じ :func:`scripts.launch_llama._parse_device_names` の表記、容量は
    ``_parse_device_memory`` の値で、どちらも起動スクリプトの実装を使う (表記を二重に持たない)。
    """
    try:
        from scripts.launch_llama import _parse_device_memory, _parse_device_names
    except ImportError:
        return ()
    memory = _parse_device_memory(text)
    devices: list[GpuDevice] = []
    for name in _parse_device_names(text):
        device_id = name.split(":", 1)[0].strip()
        total, free = memory.get(device_id, (0, 0))
        kind, reason = classify_gpu(name, total, total_ram_mib)
        devices.append(GpuDevice(name, total, free, kind, reason))
    return tuple(devices)


# ── プローブ (注入できる薄い層) ────────────────────────────


def _default_list_devices() -> str:
    """``llama-server --list-devices`` の出力 (起動スクリプトの実装を lazy import)。無ければ空。"""
    try:
        from scripts.launch_llama import _list_llama_devices
    except ImportError:
        return ""
    return _list_llama_devices()


def _default_binary_found() -> bool:
    """``llama-server`` を起動できるか (``_list_llama_devices`` と同じく PATH から探す)。"""
    import shutil

    return shutil.which("llama-server") is not None


def list_devices_degraded(text: str, *, binary_found: bool) -> bool:
    """``--list-devices`` の失敗か (純関数)。

    ``_list_llama_devices`` は失敗 (起動できない・タイムアウト・Vulkan の初期化失敗) を例外でなく
    空の出力で返すので、出力だけでは「GPU の無い PC」と区別できない。llama-server がそもそも無い
    (起動できない) なら GPU なし (縮退ではない)。起動できたのに GPU 名もメモリも 1 件も読めないなら
    一時的な失敗とみなす — GPU なしとして保存すると、次に GPU 名が取れた回が「別の PC」になる。
    """
    if not binary_found:
        return False
    try:
        from scripts.launch_llama import _parse_device_memory, _parse_device_names
    except ImportError:
        return not (text or "").strip()
    return not _parse_device_names(text) and not _parse_device_memory(text)


#: 空きが総量のこの割合未満なら、その回の空きは一時的に握られているとみなす (:func:`memory_squeezed`)。
SQUEEZED_FREE_RATIO = 0.10


def memory_squeezed(hw: HardwareProfile) -> str | None:
    """VRAM / RAM の空きが総量に比べて異常に小さいか (純関数)。小さければ理由 (英語)、無ければ ``None``。

    古い llama-server や別のアプリが一時的に握った空きで見積もって保存すると、その小さい値が
    恒久化する (次回も保存値を使う)。総量が分からない値は判定しない。
    """
    gpu = hw.best_gpu
    if gpu is not None and gpu.total_mib > 0 and gpu.free_mib < gpu.total_mib * SQUEEZED_FREE_RATIO:
        return f"gpu_free_low ({gpu.free_mib}/{gpu.total_mib} MiB)"
    if hw.total_ram_mib > 0 and hw.free_ram_mib < hw.total_ram_mib * SQUEEZED_FREE_RATIO:
        return f"ram_free_low ({hw.free_ram_mib}/{hw.total_ram_mib} MiB)"
    return None


def _default_ram() -> tuple[int, int]:
    """(総量, 利用可能量) の MiB。リランカー自己テストのメモリ取得を共有する (取れなければ 0)。"""
    from backend.free.rag.rerank_selftest import _physical_total_bytes, available_physical_memory_mb

    total = _physical_total_bytes()
    return (total // (1024 * 1024) if total else 0), available_physical_memory_mb()


def _default_physical_cores() -> int | None:
    """物理コア数。psutil (任意) → Linux の ``/proc/cpuinfo`` の順。取れなければ ``None``。"""
    try:
        import psutil  # type: ignore[import-untyped]

        count = psutil.cpu_count(logical=False)
        if count:
            return int(count)
    except ImportError:
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    return count_cpuinfo_cores(text)


def count_cpuinfo_cores(text: str) -> int | None:
    """``/proc/cpuinfo`` の (physical id, core id) の組の数。組が無ければ ``None``。"""
    pairs: set[tuple[str, str]] = set()
    physical = "0"
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            physical = value.strip()
        elif key == "core id":
            pairs.add((physical, value.strip()))
    return len(pairs) or None


@dataclass(frozen=True)
class HardwareProbes:
    """プローブ一式 (テストで差し替える)。"""

    list_devices: Callable[[], str] = _default_list_devices
    #: llama-server を起動できるか (``list_devices`` の空の出力が失敗か GPU なしかを分ける)。
    binary_found: Callable[[], bool] = _default_binary_found
    ram_mib: Callable[[], tuple[int, int]] = _default_ram
    logical_cores: Callable[[], int] = lambda: os.cpu_count() or 0
    physical_cores: Callable[[], int | None] = _default_physical_cores
    os_name: Callable[[], str] = lambda: f"{platform.system()} {platform.release()}".strip()


def build_profile(
    *,
    devices_text: str,
    total_ram_mib: int,
    free_ram_mib: int,
    logical_cores: int,
    physical_cores: int | None,
    os_name: str,
    degraded: tuple[str, ...] = (),
) -> HardwareProfile:
    """プローブの生の値から :class:`HardwareProfile` を組む (純関数)。"""
    estimated = not physical_cores
    return HardwareProfile(
        total_ram_mib=max(0, total_ram_mib),
        free_ram_mib=max(0, free_ram_mib),
        gpus=parse_devices(devices_text, total_ram_mib),
        physical_cores=estimate_physical_cores(logical_cores) if estimated else int(physical_cores or 0),
        logical_cores=max(0, logical_cores),
        physical_cores_estimated=estimated,
        os=os_name,
        degraded=degraded,
    )


def probe_hardware(probes: HardwareProbes | None = None, *, devices_text: str | None = None) -> HardwareProfile:
    """プローブを走らせて輪郭を返す。失敗したプローブは 0 / GPU なしに縮退する (起動を止めない)。

    ``devices_text`` を渡すと ``--list-devices`` を呼び直さない (起動スクリプトが既に読んだ出力)。
    llama-server を起動できるのに出力から何も読めない回も ``list_devices`` の縮退とする
    (:func:`list_devices_degraded`。その回の GPU なしを保存しない)。
    """
    p = probes or HardwareProbes()
    degraded: list[str] = []
    if devices_text is None:
        try:
            devices_text = p.list_devices()
        except Exception:  # noqa: BLE001 - プローブの失敗は GPU なしに縮退する
            devices_text = ""
            degraded.append("list_devices")
    if "list_devices" not in degraded:
        try:
            found = bool(p.binary_found())
        except Exception:  # noqa: BLE001 - 分からなければ従来どおり GPU なしとみなす
            found = False
        if list_devices_degraded(devices_text, binary_found=found):
            degraded.append("list_devices")
    try:
        total, free = p.ram_mib()
    except Exception:  # noqa: BLE001 - 取れなければ 0 (判定しない)
        total, free = 0, 0
        degraded.append("ram")
    try:
        logical = int(p.logical_cores())
    except Exception:  # noqa: BLE001
        logical = 0
        degraded.append("logical_cores")
    try:
        physical = p.physical_cores()
    except Exception:  # noqa: BLE001
        physical = None
    return build_profile(
        devices_text=devices_text, total_ram_mib=total, free_ram_mib=free,
        logical_cores=logical, physical_cores=physical, os_name=p.os_name(),
        degraded=tuple(degraded),
    )


__all__ = [
    "SHARED_MEMORY_RATIO",
    "SQUEEZED_FREE_RATIO",
    "GpuDevice",
    "GpuKind",
    "HardwareProbes",
    "HardwareProfile",
    "build_profile",
    "classify_gpu",
    "count_cpuinfo_cores",
    "estimate_physical_cores",
    "list_devices_degraded",
    "memory_squeezed",
    "parse_devices",
    "probe_hardware",
]
