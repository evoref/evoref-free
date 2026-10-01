"""項目 ``batch`` — base の論理 / 物理バッチ ``llama.batch_size`` (``-b``) / ``llama.ubatch_size`` (``-ub``)
(c_16 §7.2.3「b / ub」)。

GPU 種別と空きから選ぶ:

- **単体 GPU** — prefill が速くなるので大きく (``-b`` :data:`DGPU_BATCH`、``-ub`` は 2048 / 1024 / 512 のうち
  モデル + KV (決めた ctx) + その ub の計算バッファが VRAM の空き - ヘッドルームに収まる最大)。全層が
  載らない (層を RAM へ逃がす) ときは VRAM を層に回すため小さく (:data:`CPU_BATCH`)
- **iGPU** — 計算バッファが共有 RAM を食うので ``-ub`` は既定の 512 のまま、``-b`` だけ 1024
- **CPU** (GPU 無し / ``gpu_layers: 0``) — 小さく (:data:`CPU_BATCH`)

計算バッファは ``base_model.compute_mb_for_ubatch`` の 1 実装で見積もる (ub 512 が ctx / ngl の見積りの
1024 MiB、それより大きい ub は比例)。依存順は ctx → b・ub → ngl: ctx は ub 512 の計算バッファで決め、
b・ub はその ctx の KV に **上乗せして** 収まるときだけ ub を広げ、ngl は解決した ub の計算バッファで層数を
決める (増分を二重に足さない。ub を広げた分で全層が載らなくなることもない)。

反映は **キーごと** (:func:`effective_batch`): ``auto`` / ``null`` / キー無しのキーだけ調整値を使い、明示値は
そのまま使う。調整値の ub は ``-b`` 以下に、調整値の b は ``-ub`` 以上にそろえる (llama-server は ub を
b で切り詰める)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.free.core.tuning.base_model import (
    DEFAULT_CTX,
    DGPU_HEADROOM_MIB,
    BaseModelInfo,
    base_basis,
    compute_mb_for_ubatch,
    load_base_model_info,
    no_nested_recompute,
)
from backend.free.core.tuning.hardware import HardwareProfile
from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, register

KEY = "batch"
#: config のキー (部分名 → config のパス)。``auto`` / ``null`` / キー無しが自動。
CONFIG_KEYS: dict[str, str] = {"batch_size": "llama.batch_size", "ubatch_size": "llama.ubatch_size"}
#: 単体 GPU の ``-b`` (llama-server の既定と同じ。``-b`` は ub ごとに分けて流すだけで計算バッファを増やさない)。
DGPU_BATCH = 2048
#: 単体 GPU で試す ``-ub`` (大きい順)。
DGPU_UBATCH_CANDIDATES: tuple[int, ...] = (2048, 1024, 512)
#: iGPU の (``-b``, ``-ub``)。ub は既定のまま (計算バッファを共有 RAM に増やさない)。
IGPU_BATCH: tuple[int, int] = (1024, 512)
#: CPU のみ (と単体 GPU に全層が載らないとき) の (``-b``, ``-ub``) = 従来の ``-b 512`` と llama-server の既定 ub。
CPU_BATCH: tuple[int, int] = (512, 512)
#: 決められないときの保守側 = 従来の起動 (``-b 512``、``-ub`` は付けず llama-server の既定 512)。
FALLBACK: dict[str, Any] = {"batch_size": CPU_BATCH[0], "ubatch_size": CPU_BATCH[1]}


@dataclass(frozen=True)
class BatchParams:
    """base の実効値 (明示値と調整値を合わせたもの)。``sources`` は部分名 → ``manual`` / ``tuned``。"""

    batch_size: int
    ubatch_size: int
    sources: dict[str, str]


def is_auto_value(value: Any) -> bool:
    """config の値が自動 (``auto`` / ``null``) か。"""
    return value is None or value == "auto"


def manual_fields(cfg: dict[str, Any]) -> list[str]:
    """明示値を持つ部分名 (調整値を反映しないもの)。"""
    lc = cfg.get("llama") or {}
    return [part for part in CONFIG_KEYS if not is_auto_value(lc.get(part))]


def effective_batch(cfg: dict[str, Any], tuned: Any) -> BatchParams:
    """明示値と調整値 (``tuned`` は ``Resolved`` / 値 / ``None``。無ければ :data:`FALLBACK`) を合わせる (純関数)。"""
    value = getattr(tuned, "value", tuned)
    base = dict(FALLBACK)
    if isinstance(value, dict):
        base.update({k: v for k, v in value.items() if k in FALLBACK and isinstance(v, int)})
    lc = cfg.get("llama") or {}
    picked: dict[str, int] = {}
    sources: dict[str, str] = {}
    for part in CONFIG_KEYS:
        raw = lc.get(part)
        if is_auto_value(raw):
            picked[part], sources[part] = int(base[part]), "tuned"
        else:
            picked[part], sources[part] = int(raw), "manual"
    batch, ubatch = picked["batch_size"], picked["ubatch_size"]
    if sources["ubatch_size"] == "tuned":
        ubatch = min(ubatch, batch)
    if sources["batch_size"] == "tuned":
        batch = max(batch, ubatch)
    return BatchParams(batch, ubatch, sources)


def decide_batch(
    hw: HardwareProfile, model: BaseModelInfo, ctx: int, *, gpu_layers: Any = "auto", basis: str = "",
) -> TuneOutcome:
    """GPU 種別と空きから (``-b``, ``-ub``) を選ぶ (純関数)。材料が足りなければ環境起因の失敗。"""
    gpu = hw.best_gpu
    if gpu is None and "list_devices" in hw.degraded:
        return TuneOutcome("failed", reason="list_devices_failed", environmental=True)
    tail = f"; {basis}" if basis else ""
    if gpu is None or gpu_layers == 0:
        (b, ub), why = CPU_BATCH, "cpu: small batch"
    elif gpu.kind == "igpu":
        (b, ub), why = IGPU_BATCH, f"igpu {gpu.name}: ubatch kept at the default (compute buffer is shared RAM)"
    else:
        kv = model.kv_mb(ctx)
        if kv is None:
            return TuneOutcome("failed", reason="kv_unknown", environmental=True)
        budget = gpu.free_mib - DGPU_HEADROOM_MIB
        resident = model.model_mb + int(kv)
        for candidate in DGPU_UBATCH_CANDIDATES:
            need = resident + compute_mb_for_ubatch(candidate)
            if need <= budget:
                b, ub = DGPU_BATCH, candidate
                why = f"dgpu {gpu.name} ctx {ctx}: need {need} MiB <= budget {budget} MiB at ubatch {ub}"
                break
        else:
            (b, ub), why = CPU_BATCH, (
                f"dgpu {gpu.name} ctx {ctx}: the model does not fit fully "
                f"({resident + compute_mb_for_ubatch(CPU_BATCH[1])} MiB > budget {budget} MiB); keep VRAM for layers"
            )
    tier = "cpu" if gpu is None or gpu_layers == 0 else gpu.kind
    return TuneOutcome("ok", value={"batch_size": b, "ubatch_size": ub, "tier": tier}, reason=f"{why}{tail}")


def _ctx_of(cfg: dict[str, Any], project_root: Any, hw: HardwareProfile) -> int:
    from backend.free.core.tuning.resolve import resolve_tuned

    value = resolve_tuned(
        cfg, "ctx", project_root=project_root, hardware=hw, use_cache=False, can_recompute=no_nested_recompute,
    ).value
    return value if isinstance(value, int) and not isinstance(value, bool) else DEFAULT_CTX


def resolve_batch(cfg: dict[str, Any], project_root: Any, hw: HardwareProfile) -> BatchParams:
    """実効値 (明示値 + 調整値)。項目 ngl が計算バッファの ub を知るために使う (全部明示なら調整しない)。"""
    if len(manual_fields(cfg)) == len(CONFIG_KEYS):
        return effective_batch(cfg, None)
    from backend.free.core.tuning.resolve import resolve_tuned

    return effective_batch(cfg, resolve_tuned(
        cfg, KEY, project_root=project_root, hardware=hw, use_cache=False, can_recompute=no_nested_recompute,
    ))


def run(ctx: TuneContext) -> TuneOutcome:
    """config の base モデルと決めた ctx の KV から (``-b``, ``-ub``) を見積もる。"""
    model = load_base_model_info(ctx.cfg, ctx.project_root)
    if model is None:
        return TuneOutcome("failed", reason="base_model_unavailable", environmental=True)
    hw = ctx.hardware()
    outcome = decide_batch(
        hw, model, _ctx_of(ctx.cfg, ctx.project_root, hw),
        gpu_layers=(ctx.cfg.get("llama") or {}).get("gpu_layers", 999), basis=model.basis,
    )
    manual = manual_fields(ctx.cfg)
    if outcome.status == "ok" and manual:
        # 明示値のキーは反映しない (他のキーは反映するので項目は ok のまま)
        outcome = TuneOutcome("ok", value={**outcome.value, "manual": manual}, reason=outcome.reason)
    return outcome


SPEC = register(TuneSpec(
    key=KEY,
    # 2 つのキーをキーごとに反映する (:func:`effective_batch`)。項目としては常に自動
    config_key="",
    method="estimated",
    requires_restart=True,
    run=run,
    fallback=FALLBACK,
    depends_on=("ctx",),
    needs_servers=False,
    safe_while_running=False,  # base に効く (稼働中は予約)
    basis=base_basis,
    description="base -b / -ub from the GPU kind and the free memory after the context, c_16 §7.2.3",
))


__all__ = [
    "CONFIG_KEYS",
    "CPU_BATCH",
    "DGPU_BATCH",
    "DGPU_UBATCH_CANDIDATES",
    "FALLBACK",
    "IGPU_BATCH",
    "KEY",
    "BatchParams",
    "decide_batch",
    "effective_batch",
    "manual_fields",
    "resolve_batch",
]
