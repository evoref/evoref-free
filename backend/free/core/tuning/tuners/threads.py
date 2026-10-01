"""項目 ``threads`` — 埋め込み / リランカーの ``-t`` (c_16 §7.2.3「threads」)。

**base は配分しない** (``-t`` / ``-tb`` を付けず llama.cpp の既定 = ほぼ物理コア全部。``llama.threads: 0`` の
以前からの意味)。補助のサーバは普段アイドルなので、base のコアを静的に削ると一部オフロード / CPU 推論の PC で
生成が遅くなるだけ (以前の配分は 8 コア・リランカー ON で base を 8 → 4 に削っていた)。補助だけを小さく決める:

- **GPU 配置の埋め込み / リランカー** は CPU を殆ど使わないので :data:`GPU_AUX_THREADS` (リランカーの実測最適
  ``RERANK_GPU_DEFAULT_THREADS`` と同じ 2)
- **CPU 配置の** 埋め込み / リランカーは物理コアの :data:`CPU_AUX_DIVISOR` 分の 1 (下限 1)
- 補助の合計は物理コア数を超えない (超えるときは 1 本ずつまで縮める)
- 物理コア数が分からない / 論理コアからの推定 (``physical_cores_estimated``) なら何も付けない (従来の起動)

配置は読むだけ: base は項目 ngl の解決値、埋め込みは ``cache/embed_placement.json``、リランカーは
``cache/rerank_selftest.json`` (どちらも §7.2.1 / §7.2.2 の保存結果。無ければ CPU を仮定 = 補助に CPU を
多めに回す側)。配置が変われば前提の印 (:func:`threads_basis`) が変わり、見積り直す。

反映は **キーごと** (:func:`effective_threads`): ``0`` / ``null`` のキーだけ調整値を使い、明示値 (正の整数) は
そのまま使う。埋め込み / リランカーは **実際に起こす配置** で値を選ぶ (:meth:`ThreadPlan.aux`): 見積りの
前提と同じ配置ならその値、GPU なら :data:`GPU_AUX_THREADS`、前提が GPU で CPU に起こし直したときは
CPU 配置の値 (物理コアが分からなければ ``-t`` を付けない = llama.cpp の既定)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.free.core.tuning.base_model import base_basis, base_placement, load_base_model_info, resolve_base_ngl
from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, register

KEY = "threads"
#: config のキー (部分名 → config のパス)。``0`` / ``null`` が自動。
CONFIG_KEYS: dict[str, str] = {
    "base": "llama.threads", "embed": "embedding.threads", "rerank": "rag.rerank.threads",
}
#: GPU 配置の補助サーバ (埋め込み / リランカー) の ``-t``。``launch_llama.RERANK_GPU_DEFAULT_THREADS`` と同じ値
#: (890M での実測最適。テストが一致を見る)。
GPU_AUX_THREADS = 2
#: CPU 配置の補助サーバの ``-t`` = 物理コア数 // この値 (下限 1)。8 コアで 2 本。補助は普段アイドルで、
#: 埋め込みのクエリ (数十トークン) は 2 本でも数十 ms に収まる側。
CPU_AUX_DIVISOR = 4
#: 決められないとき (コア数が分からない / 確認待ち) の保守側 = 従来の起動 (0 = ``-t`` を付けず llama.cpp の既定)。
FALLBACK: dict[str, Any] = {"base": 0, "batch": 0, "embed": 0, "rerank": 0}


def is_auto_value(value: Any) -> bool:
    """config の値が自動 (``0`` / ``null``) か (``False`` は 0 と取り違えない)。"""
    return value is None or (type(value) is int and value == 0)


def _section(cfg: dict[str, Any], part: str) -> dict[str, Any]:
    node: Any = cfg
    for name in CONFIG_KEYS[part].split(".")[:-1]:
        node = (node or {}).get(name) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else {}


def explicit_threads(cfg: dict[str, Any]) -> dict[str, int]:
    """明示値を持つ部分名 → 値。"""
    out: dict[str, int] = {}
    for part in CONFIG_KEYS:
        raw = _section(cfg, part).get("threads")
        if not is_auto_value(raw):
            out[part] = int(raw)
    return out


def manual_fields(cfg: dict[str, Any]) -> list[str]:
    """明示値を持つ部分名 (調整値を反映しないもの)。"""
    return list(explicit_threads(cfg))


@dataclass(frozen=True)
class ThreadPlan:
    """実効値。``base`` / ``batch`` は 0 なら ``-t`` / ``-tb`` を付けない。"""

    base: int
    batch: int
    #: 部分名 → (見積りの値, 見積りが前提にした配置 (``gpu`` / ``cpu`` / ``None``))。
    aux_plan: dict[str, tuple[int, str | None]]
    sources: dict[str, str]
    #: 見積りに使った物理コア数 (0 = 分からない / 推定。前提と違う配置の CPU 値を付けない)。
    physical_cores: int = 0

    def aux(self, part: str, placement: str) -> int:
        """補助サーバ ``part`` を ``placement`` (``gpu`` / ``cpu``) で起こすときの ``-t`` (0 は付けない)。"""
        value, assumed = self.aux_plan.get(part, (0, None))
        if self.sources.get(part) == "manual" or assumed is None:
            return value
        if placement == assumed:
            return value
        if placement == "gpu":
            return GPU_AUX_THREADS
        # GPU 前提の 2 本を CPU に持ち込まない (CPU の埋め込みが詰まる)。CPU 配置の値に直す
        return cpu_aux_threads(self.physical_cores) if self.physical_cores > 0 else 0


def cpu_aux_threads(physical_cores: int) -> int:
    """CPU 配置の補助サーバ 1 本の ``-t`` (物理コアの 1/:data:`CPU_AUX_DIVISOR`、下限 1)。"""
    return max(1, physical_cores // CPU_AUX_DIVISOR)


def effective_threads(cfg: dict[str, Any], tuned: Any) -> ThreadPlan:
    """明示値と調整値 (``tuned`` は ``Resolved`` / 値 / ``None``。無ければ :data:`FALLBACK`) を合わせる (純関数)。"""
    value = getattr(tuned, "value", tuned)
    data = value if isinstance(value, dict) else FALLBACK
    placements = data.get("placement") if isinstance(data.get("placement"), dict) else {}
    explicit = explicit_threads(cfg)
    sources = {part: "manual" if part in explicit else "tuned" for part in CONFIG_KEYS}

    def pick(part: str) -> int:
        if part in explicit:
            return explicit[part]
        raw = data.get(part)
        return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else 0

    # base は配分しない: 明示値だけ (``0`` / ``null`` は -t / -tb を付けず llama.cpp の既定)。以前の保存値
    # (base / batch を配分していた版) が残っていても使わない
    base = explicit.get("base", 0)
    batch = 0
    aux_plan = {part: (pick(part), placements.get(part)) for part in ("embed", "rerank")}
    cores = data.get("physical_cores")
    known = isinstance(cores, int) and not isinstance(cores, bool) and not data.get("physical_cores_estimated")
    return ThreadPlan(base, batch, aux_plan, sources, physical_cores=int(cores) if known else 0)


def decide_threads(
    physical_cores: int,
    *,
    base: str,
    embed: str | None,
    rerank: str | None,
    explicit: dict[str, int] | None = None,
    basis: str = "",
    estimated: bool = False,
) -> TuneOutcome:
    """埋め込み / リランカーの ``-t`` を決める (純関数)。base は配分しない (明示値だけ、無ければ 0 = 付けない)。

    ``base`` は ``gpu`` / ``partial`` / ``cpu`` (記録だけ)、``embed`` / ``rerank`` は ``gpu`` / ``cpu`` / ``None``
    (起こさない)。``explicit`` (部分名 → 明示値) はそのまま使い、補助の残りから差し引く。補助の合計は物理コア数を
    超えない (各 1 本までは縮める)。``estimated`` (物理コアが論理コアからの推定) なら補助も付けない (従来の起動)。
    コア数が分からなければ環境起因の失敗。
    """
    if physical_cores <= 0:
        return TuneOutcome("failed", reason="cores_unknown", environmental=True)
    fixed = dict(explicit or {})
    auto_aux = [
        (name, kind) for name, kind in (("embed", embed), ("rerank", rerank)) if kind is not None and name not in fixed
    ]
    alloc: dict[str, int] = {}
    if estimated:
        mode = "estimated_cores"
    else:
        mode = "aux_only"
        left = physical_cores - sum(v for k, v in fixed.items() if k != "base")
        for i, (name, kind) in enumerate(auto_aux):
            want = GPU_AUX_THREADS if kind == "gpu" else cpu_aux_threads(physical_cores)
            # 後に残る補助へ 1 本ずつ残す (合計が物理コア数を超えない。1 コアでも各 1 本は付ける)
            n = max(1, min(want, left - (len(auto_aux) - i - 1)))
            alloc[name] = n
            left -= n
        if left < 0:
            mode = "oversubscribed"
    value: dict[str, Any] = {
        "base": fixed.get("base", 0),
        "batch": 0,
        "embed": fixed.get("embed", alloc.get("embed", 0)) if embed is not None else 0,
        "rerank": fixed.get("rerank", alloc.get("rerank", 0)) if rerank is not None else 0,
        "placement": {"base": base, "embed": embed, "rerank": rerank},
        "physical_cores": physical_cores,
        "physical_cores_estimated": estimated,
    }
    if fixed:
        value["manual"] = sorted(fixed)
    reason = (
        f"{mode}: {physical_cores} physical cores{' (estimated)' if estimated else ''} -> "
        f"base {value['base'] or 'llama.cpp default'} ({base}), "
        f"embed {value['embed']} ({embed or 'off'}), rerank {value['rerank']} ({rerank or 'off'})"
    )
    if basis:
        reason = f"{reason}; {basis}"
    return TuneOutcome("ok", value=value, reason=reason)


# ── 配置 (読むだけ) ─────────────────────────────────────────


def _saved(cfg: dict[str, Any], project_root: Path) -> tuple[Any, Any]:
    """(埋め込みの判別結果, リランカーの自己テスト結果)。読めなければ ``None``。"""
    from backend.free.core.tuning.store import resolve_tune_paths
    from backend.free.rag.embed_placement import load_placement_result
    from backend.free.rag.rerank_selftest import load_selftest_result

    paths = resolve_tune_paths(cfg, project_root)
    return load_placement_result(paths.embed_placement)[0], load_selftest_result(paths.rerank_selftest)[0]


def aux_placements(cfg: dict[str, Any], embed_saved: Any, rerank_saved: Any, digest: str | None) -> tuple[str | None, str | None]:
    """(埋め込みの配置, リランカーの配置) (純関数)。``None`` は起こさない。

    ``auto`` は保存結果の配置 (``digest`` を渡せば指紋が一致するものだけ)、無ければ CPU を仮定する。
    """
    from backend.schemas.rag import rerank_mode_of

    def usable(saved: Any) -> bool:
        return saved is not None and (digest is None or saved.fingerprint == digest)

    emb = cfg.get("embedding") or {}
    embed: str | None = None
    if emb.get("backend", "llama-cpp") == "llama-cpp":
        raw = emb.get("gpu_layers")
        if raw == "auto":
            embed = embed_saved.placement if usable(embed_saved) else "cpu"
        else:
            embed = "gpu" if isinstance(raw, int) and raw != 0 else "cpu"
    rr = (cfg.get("rag") or {}).get("rerank") or {}
    rerank: str | None = None
    if rerank_mode_of(cfg) != "off" and (cfg.get("model_paths") or {}).get("rerank_model"):
        raw = rr.get("gpu_layers", "auto")
        if raw == "auto":
            if usable(rerank_saved):
                rerank = rerank_saved.placement if rerank_saved.enabled else None
            else:
                rerank = "cpu"
        else:
            rerank = "gpu" if isinstance(raw, int) and raw > 0 else "cpu"
    return embed, rerank


def threads_basis(cfg: dict[str, Any], project_root: Path) -> str:
    """保存結果の前提の印 (:attr:`TuneSpec.basis`): base モデルと、埋め込み / リランカーの配置。"""
    embed_saved, rerank_saved = _saved(cfg, project_root)
    embed, rerank = aux_placements(cfg, embed_saved, rerank_saved, None)
    # ``v2`` = base を配分しない版 (base を配分していた以前の保存値は前提違いとして見積り直す)
    return f"threads_basis_v2[{base_basis(cfg, project_root)},embed={embed},rerank={rerank}]"


def run(ctx: TuneContext) -> TuneOutcome:
    """物理コア数と 3 つのサーバの配置から配分を決める。"""
    from backend.free.rag.rerank_selftest import collect_pc_info

    model = load_base_model_info(ctx.cfg, ctx.project_root)
    if model is None:
        return TuneOutcome("failed", reason="base_model_unavailable", environmental=True)
    hw = ctx.hardware()
    ngl = resolve_base_ngl(ctx.cfg, ctx.project_root, hw)
    embed_saved, rerank_saved = _saved(ctx.cfg, ctx.project_root)
    embed, rerank = aux_placements(ctx.cfg, embed_saved, rerank_saved, collect_pc_info(hw.gpu_names).digest)
    return decide_threads(
        hw.physical_cores,
        base=base_placement(ngl, model.n_layers, has_gpu=hw.best_gpu is not None),
        embed=embed, rerank=rerank, explicit=explicit_threads(ctx.cfg),
        basis=threads_basis(ctx.cfg, ctx.project_root), estimated=hw.physical_cores_estimated,
    )


SPEC = register(TuneSpec(
    key=KEY,
    # 3 つのキーをキーごとに反映する (:func:`effective_threads`)。項目としては常に自動
    config_key="",
    method="estimated",
    requires_restart=True,
    run=run,
    fallback=FALLBACK,
    # 配置 (ngl / 埋め込みの判別 / リランカーの自己テスト) が決まってから配分する
    depends_on=("ngl", "embed_placement", "rerank"),
    needs_servers=False,
    safe_while_running=False,  # base に効く (稼働中は予約)
    basis=threads_basis,
    description="-t for the embedding / reranker from the physical cores (base keeps the llama.cpp default), c_16 §7.2.3",
))


__all__ = [
    "CONFIG_KEYS",
    "CPU_AUX_DIVISOR",
    "FALLBACK",
    "GPU_AUX_THREADS",
    "KEY",
    "ThreadPlan",
    "aux_placements",
    "cpu_aux_threads",
    "decide_threads",
    "effective_threads",
    "explicit_threads",
    "manual_fields",
    "threads_basis",
]
