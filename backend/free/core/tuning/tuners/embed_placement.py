"""項目 ``embed_placement`` — 埋め込みサーバの配置 (GPU / CPU)。既存の判別 (c_16 §7.2.2) を呼ぶだけ。

正本は ``cache/embed_placement.json`` のまま。ここは起動スクリプトの ``ensure_embed_placement`` を
呼んで、その結果の要約を ``auto_tune.json`` の項目に載せる。判別の結果を保存しなかった回 (空きの不足・
予算超過・一時サーバの起動失敗など状態に由来するもの) は環境起因の失敗として返す。
"""

from __future__ import annotations

from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, is_auto, register

KEY = "embed_placement"


def run(ctx: TuneContext) -> TuneOutcome:
    """``embedding.gpu_layers: auto`` なら判別 (要るときだけ、``force`` なら必ず) して要約を返す。"""
    emb = ctx.cfg.get("embedding") or {}
    if emb.get("backend", "llama-cpp") != "llama-cpp":
        return TuneOutcome("skipped", reason="embedding_backend_not_llama_cpp")
    if not is_auto(SPEC, ctx.cfg):
        # 明示の整数 / null は判別しない (理由 manual / not_auto は runner が config から付ける)
        return TuneOutcome("skipped")
    try:
        from scripts import launch_llama as ll
    except ImportError:
        return TuneOutcome("failed", reason="launcher_unavailable", environmental=True)
    from backend.free.rag import embed_placement as ep
    from backend.free.rag.rerank_selftest import collect_pc_info

    placement = ll.ensure_embed_placement(ctx.cfg, ctx.project_root, force=ctx.force)
    path = ll.embed_placement_path(ctx.cfg, ctx.project_root)
    saved = ep.load_placement_result(path)[0] if path is not None else None
    digest = collect_pc_info(ctx.hardware().gpu_names).digest
    if saved is None or saved.fingerprint != digest or saved.reason != placement.reason:
        # 保存されなかった判別 (状態に由来する) / 確認待ちで測らなかった
        return TuneOutcome("failed", reason=placement.reason, environmental=True)
    return TuneOutcome(
        "ok",
        value={
            "placement": saved.placement, "gpu_layers": saved.gpu_layers,
            "cpu_p50_ms": saved.cpu_p50_ms, "gpu_p50_ms": saved.gpu_p50_ms,
            "cosine_min": saved.cosine_min,
        },
        reason=saved.reason,
    )


SPEC = register(TuneSpec(
    key=KEY,
    config_key="embedding.gpu_layers",
    method="measured",
    requires_restart=True,
    run=run,
    auto_values=("auto",),
    config_default=None,
    safe_while_running=True,  # 一時ポートで測る (c_16 §7.2.2)
    needs_servers=True,
    description="embedding server placement (GPU / CPU), c_16 §7.2.2",
))
