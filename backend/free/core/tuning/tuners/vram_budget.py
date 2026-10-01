"""項目 ``vram_budget`` — VRAM 予算 ``runtime.total_vram_budget_mb`` (c_16 §7.2.3)。

``--list-devices`` の空き (GPU の空きの最大の 1 枚。base / 埋め込み / リランカーを載せる先) に安全率を
掛けて導く。``null`` のとき起動の予算検査 (``launch_llama.check_vram_budget``) がこの値で働く。
GPU が無ければ ``None`` (検査しない) を保存する。
"""

from __future__ import annotations

from backend.free.core.tuning.hardware import HardwareProfile
from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, register

KEY = "vram_budget"
#: 空きに掛ける安全率。予算検査の推奨値 (``suggest_total_vram_budget_mb``) が見積りに 10% の余裕を
#: 足すのと対になる流儀で、空きから 10% を残す (見積りが 10% 外れても空きを越えない)。
#: 例: 8 GB の単体 GPU (空き 7800 MiB) → 7020 MiB で、config の旧コメントの目安 (約 7000) と一致する。
FREE_SAFETY_RATIO = 0.9


def decide_vram_budget(hw: HardwareProfile) -> TuneOutcome:
    """GPU の空きの最大 × 安全率 (純関数)。GPU 無し → ``None``。プローブ失敗は環境起因の失敗。"""
    gpu = hw.best_gpu
    if gpu is None:
        if "list_devices" in hw.degraded:
            return TuneOutcome("failed", reason="list_devices_failed", environmental=True)
        return TuneOutcome("ok", value=None, reason="no_gpu_device")
    if gpu.free_mib <= 0:
        return TuneOutcome("failed", reason="gpu_free_unknown", environmental=True)
    budget = int(gpu.free_mib * FREE_SAFETY_RATIO)
    return TuneOutcome("ok", value=budget, reason=(
        f"{gpu.kind} {gpu.name}: free {gpu.free_mib} MiB x {FREE_SAFETY_RATIO}"
    ))


def run(ctx: TuneContext) -> TuneOutcome:
    """輪郭の GPU の空きから予算を導く。"""
    return decide_vram_budget(ctx.hardware())


SPEC = register(TuneSpec(
    key=KEY,
    config_key="runtime.total_vram_budget_mb",
    method="estimated",
    requires_restart=True,
    run=run,
    auto_values=(None,),
    config_default=None,
    fallback=None,
    description="VRAM budget from the free memory of the largest GPU, c_16 §7.2.3",
))
