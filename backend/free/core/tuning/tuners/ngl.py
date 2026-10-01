"""項目 ``ngl`` — base の GPU オフロード層数 ``llama.gpu_layers`` (c_16 §7.2.3)。

GPU の **空き** (総量ではない) からヘッドルームを引いた予算に、モデル + 文脈メモリ (決めた ctx の KV) +
計算バッファが収まる最大の割合を段階 (100% / 80% / 60% / 40% / 0%) で探す。ヘッドルームは GPU 種別で
変える (iGPU: Vulkan host buffer の予約、単体 GPU: 数百 MiB)。複数 GPU は空きの最大の 1 枚で判定する。
計算バッファは項目 ``batch`` が決めた ``-ub`` の見積り (``compute_mb_for_ubatch``) を使う (依存順 ctx → b・ub → ngl)。

段階縮小の式 (:func:`calc_auto_gpu_layers`) は起動スクリプトの旧 ``_resolve_auto_gpu_layers`` から
移したもの (起動スクリプトは :func:`~backend.free.core.tuning.resolve.resolve_tuned` を通してこれを使う)。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from backend.free.core.tuning.base_model import (
    DEFAULT_CTX,
    DGPU_HEADROOM_MIB,
    BaseModelInfo,
    base_basis,
    compute_mb_for_ubatch,
    gpu_headroom_mib,
    load_base_model_info,
    resolve_or_provisional,
)
from backend.free.core.tuning.hardware import HardwareProfile
from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, register

KEY = "ngl"
#: 全層を GPU へ載せる値 (llama.cpp は層数を超える値を「全層」と解釈する)。決められないときの値でもある
#: (従来どおり全層。``runtime.gpu_auto_tune_enabled: false`` と同じ)。ただし環境移行の確認待ちは一時の
#: 見積り、それも取れなければ :func:`conservative_ngl` (``base_model.resolve_or_provisional``)。
ALL_LAYERS = 999

# rationale: 予算内に収まる最大の ratio を採る。ratio を落とすほど CPU 側へ
# 逃げるので応答速度は落ちるが、OOM で起動できないよりは良い。
AUTO_NGL_RATIOS: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.0)


def scaled_vram_mb(full_estimate: dict[str, Any], ratio: float) -> int:
    """ngl 縮小後の VRAM 推定 MiB を返す。

    rationale: ``model_mb`` のみ ratio に線形比例で縮小、``context_mb`` /
    ``compute_mb`` は安全側で据置 (= 多めに見積もる)。実際は ngl=0 で
    ctx/compute も大幅減るが、上振れさせて「縮小不足で起動後 OOM」を回避。
    ratio=0 のときだけ全成分 0 (= CPU 配置)。
    """
    if ratio <= 0.0:
        return 0
    model_mb = int(full_estimate.get("model_mb", 0) or 0)
    ctx_mb = int(full_estimate.get("context_mb", 0) or 0)
    compute_mb = int(full_estimate.get("compute_mb", 0) or 0)
    return int(round(model_mb * ratio)) + ctx_mb + compute_mb


def calc_auto_gpu_layers(
    *,
    base_layers_total: int,
    base_full_estimate: dict[str, Any],
    gpu_free_mib: int,
    headroom_mib: int,
) -> tuple[int, str]:
    """base の ``-ngl`` を段階縮小し、最初に予算内へ収まる値を返す (純関数)。

    Args:
        base_layers_total: base モデルの全 layer 数 (GGUF block_count)
        base_full_estimate: 全 offload 時の見積り ``{"model_mb", "context_mb", "compute_mb"}``
        gpu_free_mib: GPU の空き (MiB)
        headroom_mib: 空きから引くヘッドルーム (MiB)

    Returns:
        ``(base_ngl, reason_str)``。ratio=1.0 採用時は ``(999, ...)``、CPU へ倒すときは ``(0, ...)``。
    """
    budget = max(gpu_free_mib - headroom_mib, 0)
    for ratio in AUTO_NGL_RATIOS:
        base_ngl = ALL_LAYERS if ratio >= 1.0 else int(round(base_layers_total * ratio))
        base_vram = scaled_vram_mb(base_full_estimate, ratio)
        if base_vram <= budget:
            return base_ngl, (
                f"ratio={ratio:.0%} base={base_ngl}/{base_layers_total} "
                f"est={base_vram}MiB budget={budget}MiB "
                f"(gpu_free={gpu_free_mib}MiB - headroom={headroom_mib}MiB)"
            )
    # ratio=0.0 は 0 MiB なので必ず収まる (budget >= 0)。ここへは来ないが CPU へ倒す
    return 0, f"all-CPU fallback (budget={budget}MiB)"


#: 確認待ちで一時の見積りも取れないときに GPU へ載せる割合。:data:`AUTO_NGL_RATIOS` の最小の部分オフロード
#: (40%) で、全層 (999) より OOM しにくく、CPU (0) ほど遅くない。根拠: 8B q4 (約 5 GB) の 40% は約 2 GB で、
#: 4 GB 級の GPU でも KV と計算バッファを足して載る側。一方、載らない層は RAM で回るだけで起動は止まらない。
CONSERVATIVE_RATIO = 0.4


def conservative_ngl(cfg: dict[str, Any], project_root: Any) -> int:
    """確認待ちで見積りも取れないときの ``-ngl`` (全層の :data:`CONSERVATIVE_RATIO`)。

    層数も読めなければ (GGUF が読めない = どのみち起動できない) :data:`ALL_LAYERS`。
    """
    model = load_base_model_info(cfg, project_root)
    if model is None or not model.n_layers:
        return ALL_LAYERS
    return int(round(model.n_layers * CONSERVATIVE_RATIO))


def decide_ngl(
    hw: HardwareProfile, model: BaseModelInfo, ctx: int, *, igpu_headroom_mib: int = 4096,
) -> TuneOutcome:
    """空きに収まる層数 (純関数)。GPU が無ければ 0 (CPU)。材料が足りなければ環境起因の失敗。"""
    gpu = hw.best_gpu
    if gpu is None:
        if "list_devices" in hw.degraded:
            return TuneOutcome("failed", reason="list_devices_failed", environmental=True)
        return TuneOutcome("ok", value=0, reason=f"no_gpu_device; {model.basis}")
    if not model.n_layers:
        return TuneOutcome("failed", reason="layers_unknown", environmental=True)
    kv = model.kv_mb(ctx)
    if kv is None:
        return TuneOutcome("failed", reason="kv_unknown", environmental=True)
    headroom = igpu_headroom_mib if gpu.kind == "igpu" else DGPU_HEADROOM_MIB
    ngl, why = calc_auto_gpu_layers(
        base_layers_total=model.n_layers,
        base_full_estimate={"model_mb": model.model_mb, "context_mb": int(kv), "compute_mb": model.compute_mb},
        gpu_free_mib=gpu.free_mib,
        headroom_mib=headroom,
    )
    return TuneOutcome("ok", value=ngl, reason=f"{gpu.kind} {gpu.name} ctx {ctx}: {why}; {model.basis}")


def run(ctx: TuneContext) -> TuneOutcome:
    """決めた ctx (明示値ならその値) の KV を引いてから層数を決める。

    ctx は確認待ちの間も一時の見積り (:func:`resolve_or_provisional`) を使う (起動の ``-c`` と同じ値)。
    """
    if not (ctx.cfg.get("runtime") or {}).get("gpu_auto_tune_enabled", True):
        return TuneOutcome("skipped", reason="gpu_auto_tune_disabled")
    model = load_base_model_info(ctx.cfg, ctx.project_root)
    if model is None:
        return TuneOutcome("failed", reason="base_model_unavailable", environmental=True)
    hw = ctx.hardware()
    resolved = resolve_or_provisional(ctx.cfg, "ctx", project_root=ctx.project_root, hardware=hw, use_cache=False)
    n_ctx = resolved.value if isinstance(resolved.value, int) and not isinstance(resolved.value, bool) else DEFAULT_CTX
    gpu = hw.best_gpu
    if gpu is not None:
        from backend.free.core.tuning.tuners.batch import resolve_batch

        # 計算バッファは決めた -ub の見積り 1 つ (b・ub の増分を別に足さない)
        model = replace(model, compute_mb=compute_mb_for_ubatch(resolve_batch(ctx.cfg, ctx.project_root, hw).ubatch_size))
    return decide_ngl(
        hw, model, n_ctx,
        igpu_headroom_mib=gpu_headroom_mib(ctx.cfg, gpu.kind if gpu is not None else "none"),
    )


SPEC = register(TuneSpec(
    key=KEY,
    config_key="llama.gpu_layers",
    method="estimated",
    requires_restart=True,
    run=run,
    auto_values=("auto",),
    config_default=ALL_LAYERS,
    fallback=ALL_LAYERS,
    depends_on=("ctx", "batch"),
    basis=base_basis,
    description="base GPU layers from the free VRAM after the context, c_16 §7.2.3",
))
