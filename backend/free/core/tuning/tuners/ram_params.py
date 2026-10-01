"""項目 ``ram_params`` — base の RAM に載る 2 つ ``llama.ctx_checkpoints`` (``--ctx-checkpoints``) /
``llama.cache_ram_mib`` (``--cache-ram``) (c_16 §7.2.3「RAM 系」)。

どちらも llama-server がホストの RAM に持つ (VRAM ではない)。空き RAM から余白と base 自身が RAM に置く分
(CPU / iGPU 配置はモデル + KV + 計算バッファ、単体 GPU の一部オフロードは RAM に残る層の重み) を引いた
**余裕** を元に、

- ``ctx_checkpoints``: 1 つの checkpoint はスロットごとに再帰状態 (と SWA の窓) のスナップショット 1 つで、
  :data:`CHECKPOINT_STATE_MIB` (約 150 MiB) と見積もる。``スロット数 × checkpoint 数 × 150 MiB`` が余裕の
  :data:`CHECKPOINT_SHARE` に収まる最大を :data:`CHECKPOINT_LADDER` から選ぶ。上限は従来の既定 8 で、
  空きが少ないときだけ 4 / 2 へ **下げる** (150 MiB は実測でない概算なので、従来より増やす側には使わない)
- ``cache_ram_mib``: 常に 0 (無効)。2026-09-11 の A/B (kv_cache_audit) で ``cache_ram_mib: 2048`` は wall
  305 → 326 秒に悪化し効果が無かったため 0 に戻した実測の結論に合わせる (空き RAM から値を出さない)

を決める。反映は **キーごと** (:func:`effective_ram_params`): ``auto`` / ``null`` / キー無しのキーだけ調整値を
使い、明示値はそのまま使う。

**既定の統一**: 以前は ``cache_ram_mib`` の既定が起動スクリプト (キー無しで 4096) と schema・雛形 (0) で
食い違っていた。既定は ``auto`` (``null``) にし、決められないときの値は本モジュールの :data:`FALLBACK`
1 か所 (0 = RAM を食わない側) にした。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.free.core.tuning.base_model import (
    DEFAULT_CTX,
    RAM_RESERVE_MIB,
    BaseModelInfo,
    base_basis,
    base_placement,
    load_base_model_info,
    no_nested_recompute,
    resolve_base_ngl,
    slots_for,
)
from backend.free.core.tuning.hardware import HardwareProfile
from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, register

KEY = "ram_params"
#: config のキー (部分名 → config のパス)。``auto`` / ``null`` / キー無しが自動。
CONFIG_KEYS: dict[str, str] = {
    "ctx_checkpoints": "llama.ctx_checkpoints", "cache_ram_mib": "llama.cache_ram_mib",
}
#: checkpoint 1 つ (1 スロット) が RAM に載せる状態の見積り (MiB)。PLAUSIBLE な概算で実測ではない:
#: hybrid recurrent の 27B 級で再帰状態 1 シーケンスが約 150 MiB (``LlamaConfig.slots`` の注記と同じ値)。
#: 純 attention のモデルは checkpoint を作らないので過大側 (= RAM を多めに見る保守側) の見積りになる。
CHECKPOINT_STATE_MIB = 150
#: 選ぶ checkpoint 数 (大きい順)。先頭は従来の既定 8 (起動スクリプト・schema・雛形で一致していた値) で、
#: これより増やさない: 4 スロット × 16 × 150 MiB ≈ 9.6 GB のように、根拠の無い概算で RAM を大きく取らない。
#: 末尾は余裕が足りなくても使う下限 (0 にすると分岐のたびに全量 re-prefill)。
CHECKPOINT_LADDER: tuple[int, ...] = (8, 4, 2)
#: 余裕のうち checkpoint に使ってよい割合 (残りは backend・埋め込み・OS の揺れ)。
CHECKPOINT_SHARE = 0.25
#: ``--cache-ram`` の自動の値。0 (無効): 2026-09-11 の A/B で 2048 は wall 305 → 326 秒に悪化し効果が無かった。
AUTO_CACHE_RAM_MIB = 0
#: 決められないときの保守側: checkpoint は従来の既定 8 (起動スクリプト・schema・雛形で一致していた値)、
#: cache-ram は 0 (RAM を食わない側。llama.cpp の idle 退避は 2026-08-27 の実測で効果が無かった)。
FALLBACK: dict[str, Any] = {"ctx_checkpoints": 8, "cache_ram_mib": 0}


@dataclass(frozen=True)
class RamParams:
    """base の実効値 (明示値と調整値を合わせたもの)。``sources`` は部分名 → ``manual`` / ``tuned``。"""

    ctx_checkpoints: int
    cache_ram_mib: int
    sources: dict[str, str]


def is_auto_value(value: Any) -> bool:
    """config の値が自動 (``auto`` / ``null``) か。"""
    return value is None or value == "auto"


def manual_fields(cfg: dict[str, Any]) -> list[str]:
    """明示値を持つ部分名 (調整値を反映しないもの)。"""
    lc = cfg.get("llama") or {}
    return [part for part in CONFIG_KEYS if not is_auto_value(lc.get(part))]


def effective_ram_params(cfg: dict[str, Any], tuned: Any) -> RamParams:
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
    return RamParams(picked["ctx_checkpoints"], picked["cache_ram_mib"], sources)


def base_ram_mib(hw: HardwareProfile, model: BaseModelInfo, ctx: int, ngl: int) -> int:
    """base 自身が RAM に置く量の見積り (MiB、純関数)。

    CPU / iGPU (VRAM が共有 RAM) はモデル + KV + 計算バッファ全部。単体 GPU は全層なら 0、一部なら
    RAM に残る層の重み (KV と計算バッファは GPU 側に数える)。
    """
    gpu = hw.best_gpu
    kv = model.kv_mb(ctx) or 0
    placement = base_placement(ngl, model.n_layers, has_gpu=gpu is not None)
    if placement == "cpu" or (gpu is not None and gpu.kind == "igpu"):
        return model.model_mb + int(kv) + model.compute_mb
    if placement == "gpu":
        return 0
    if not model.n_layers:
        return model.model_mb
    return int(model.model_mb * max(0.0, 1.0 - ngl / model.n_layers))


def decide_ram_params(hw: HardwareProfile, *, slots: int, base_ram: int, basis: str = "") -> TuneOutcome:
    """空き RAM の余裕から checkpoint 数と cache-ram を決める (純関数)。RAM が取れなければ環境起因の失敗。"""
    if hw.free_ram_mib <= 0:
        return TuneOutcome("failed", reason="ram_unknown", environmental=True)
    spare = hw.free_ram_mib - RAM_RESERVE_MIB - base_ram
    room = max(0, spare)
    per = max(1, slots) * CHECKPOINT_STATE_MIB
    budget = int(room * CHECKPOINT_SHARE)
    fitting = [n for n in CHECKPOINT_LADDER if n * per <= budget]
    checkpoints = fitting[0] if fitting else CHECKPOINT_LADDER[-1]
    cache = AUTO_CACHE_RAM_MIB
    warn = "" if fitting else "warn: "
    reason = (
        f"{warn}spare {spare} MiB (free {hw.free_ram_mib} - reserve {RAM_RESERVE_MIB} - base {base_ram}); "
        f"checkpoints {checkpoints} x {slots} slots x {CHECKPOINT_STATE_MIB} MiB (budget {budget} MiB); "
        f"cache-ram {cache} MiB (off: no gain in the 2026-09-11 A/B)"
    )
    if basis:
        reason = f"{reason}; {basis}"
    return TuneOutcome("ok", value={
        "ctx_checkpoints": checkpoints, "cache_ram_mib": cache, "spare_ram_mib": spare,
    }, reason=reason)


def ram_basis(cfg: dict[str, Any], project_root: Any) -> str | None:
    """保存結果の前提の印: base モデルと式の版 (``v2`` = cache-ram 0 / checkpoint 上限 8。以前の版の保存値
    (cache-ram 4096 / checkpoint 16) は前提違いとして見積り直す)。"""
    basis = base_basis(cfg, project_root)
    return f"ram_params_v2[{basis}]" if basis is not None else None


def run(ctx: TuneContext) -> TuneOutcome:
    """base モデル・決めた ctx / slots / ngl と空き RAM から見積もる。"""
    from backend.free.core.tuning.resolve import resolve_tuned

    model = load_base_model_info(ctx.cfg, ctx.project_root)
    if model is None:
        return TuneOutcome("failed", reason="base_model_unavailable", environmental=True)
    hw = ctx.hardware()
    resolved = resolve_tuned(
        ctx.cfg, "ctx", project_root=ctx.project_root, hardware=hw, use_cache=False,
        can_recompute=no_nested_recompute,
    ).value
    n_ctx = resolved if isinstance(resolved, int) and not isinstance(resolved, bool) else DEFAULT_CTX
    slots = slots_for((ctx.cfg.get("llama") or {}).get("slots", "auto"), n_ctx)
    ngl = resolve_base_ngl(ctx.cfg, ctx.project_root, hw)
    outcome = decide_ram_params(
        hw, slots=slots, base_ram=base_ram_mib(hw, model, n_ctx, ngl), basis=ram_basis(ctx.cfg, ctx.project_root) or "",
    )
    manual = manual_fields(ctx.cfg)
    if outcome.status == "ok" and manual:
        outcome = TuneOutcome("ok", value={**outcome.value, "manual": manual}, reason=outcome.reason)
    return outcome


SPEC = register(TuneSpec(
    key=KEY,
    # 2 つのキーをキーごとに反映する (:func:`effective_ram_params`)。項目としては常に自動
    config_key="",
    method="estimated",
    requires_restart=True,
    run=run,
    fallback=FALLBACK,
    depends_on=("ctx", "ngl"),
    needs_servers=False,
    safe_while_running=False,  # base に効く (稼働中は予約)
    basis=ram_basis,
    description="base --ctx-checkpoints (up to 8) from the spare RAM; --cache-ram stays 0, c_16 §7.2.3",
))


__all__ = [
    "AUTO_CACHE_RAM_MIB",
    "CHECKPOINT_LADDER",
    "CHECKPOINT_STATE_MIB",
    "CONFIG_KEYS",
    "FALLBACK",
    "KEY",
    "RamParams",
    "base_ram_mib",
    "decide_ram_params",
    "effective_ram_params",
    "manual_fields",
]
