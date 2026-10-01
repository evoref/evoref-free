"""項目 ``ctx`` — base の文脈長 ``llama.context_size`` (c_16 §7.2.3)。

候補 (4096 / 8192 / 16384 / 32768 / 65536) のうち、モデル + 文脈メモリ (KV、スロット数と cache 型込み) +
計算バッファが空きに収まる最大を採る。上限はプロファイルの既定 (無ければ 8192) とモデルの学習 ctx。
収まらなくても 4096 を返し、理由に ``warn`` を残す (起動は止めない)。

空きの見方は配置で変わる:

- **単体 GPU** — GPU の空き - ヘッドルーム。4096 でも載らなければ層を RAM へ逃がす (項目 ngl) ので、
  GPU と RAM の空きの和で見る
- **iGPU** — VRAM は共有 RAM なので、GPU の空き - Vulkan host buffer と、RAM の空き - 余白の小さい方
- **CPU** (GPU 無し / ``gpu_layers: 0``) — RAM の空き - 余白
"""

from __future__ import annotations

from typing import Any

from backend.free.core.tuning.base_model import (
    DEFAULT_CTX,
    DGPU_HEADROOM_MIB,
    RAM_RESERVE_MIB,
    BaseModelInfo,
    base_basis,
    gpu_headroom_mib,
    load_base_model_info,
)
from backend.free.core.tuning.hardware import HardwareProfile
from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, register

KEY = "ctx"
CTX_CANDIDATES: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536)
CTX_MIN = 4096


def _pool(
    hw: HardwareProfile, need_min: int, *, gpu_layers: Any, igpu_headroom_mib: int,
) -> tuple[int, str] | None:
    """(使える空き MiB, 見た先)。RAM が要るのに取れなければ ``None``。"""
    ram = hw.free_ram_mib - RAM_RESERVE_MIB if hw.free_ram_mib > 0 else None
    gpu = hw.best_gpu
    if gpu is None or gpu_layers == 0:
        return (ram, "ram") if ram is not None else None
    if gpu.kind == "igpu":
        vram = gpu.free_mib - igpu_headroom_mib
        return (vram, "igpu") if ram is None else (min(vram, ram), "igpu+ram")
    vram = gpu.free_mib - DGPU_HEADROOM_MIB
    if need_min <= vram or ram is None:
        return vram, "dgpu"
    return vram + ram, "dgpu+ram"


def decide_ctx(
    hw: HardwareProfile, model: BaseModelInfo, *, gpu_layers: Any = "auto", igpu_headroom_mib: int = 4096,
) -> TuneOutcome:
    """空きに収まる最大の ctx (純関数)。材料が足りなければ環境起因の失敗 (保存しない)。"""
    upper = model.ctx_upper
    floor = min(CTX_MIN, upper)

    def need(ctx: int) -> int | None:
        kv = model.kv_mb(ctx)
        return None if kv is None else model.model_mb + model.compute_mb + int(kv)

    need_min = need(floor)
    if need_min is None:
        return TuneOutcome("failed", reason="kv_unknown", environmental=True)
    pool = _pool(hw, need_min, gpu_layers=gpu_layers, igpu_headroom_mib=igpu_headroom_mib)
    if pool is None:
        return TuneOutcome("failed", reason="ram_unknown", environmental=True)
    free, where = pool
    candidates = [c for c in CTX_CANDIDATES if c <= upper] or [floor]
    for ctx in reversed(candidates):
        required = need(ctx)
        if required is not None and required <= free:
            return TuneOutcome("ok", value=ctx, reason=(
                f"{where}: need {required} MiB <= free {free} MiB at ctx {ctx} (cap {upper}); {model.basis}"
            ))
    return TuneOutcome("ok", value=floor, reason=(
        f"warn: {where}: need {need_min} MiB > free {free} MiB even at ctx {floor}; {model.basis}"
    ))


def run(ctx: TuneContext) -> TuneOutcome:
    """config の base モデルと輪郭から ctx を見積もる。モデルが読めなければ環境起因の失敗。"""
    model = load_base_model_info(ctx.cfg, ctx.project_root)
    if model is None:
        return TuneOutcome("failed", reason="base_model_unavailable", environmental=True)
    hw = ctx.hardware()
    gpu = hw.best_gpu
    lc = ctx.cfg.get("llama") or {}
    return decide_ctx(
        hw, model, gpu_layers=lc.get("gpu_layers", 999),
        igpu_headroom_mib=gpu_headroom_mib(ctx.cfg, gpu.kind if gpu is not None else "none"),
    )


SPEC = register(TuneSpec(
    key=KEY,
    config_key="llama.context_size",
    method="estimated",
    requires_restart=True,
    run=run,
    # null は以前「プロファイルの既定」だったが、auto と同じ意味にした (c_16 §7.2.3 既知の制約)
    auto_values=("auto", None),
    config_default=None,
    fallback=DEFAULT_CTX,
    basis=base_basis,
    description="base context size that fits the free VRAM / RAM, c_16 §7.2.3",
))
