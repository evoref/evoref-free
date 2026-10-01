"""項目 ``rerank`` — リランカーの自己テスト (c_16 §7.2.1) を呼ぶだけ。

正本は ``cache/rerank_selftest.json`` のまま。測り直しが要らなければ (同じ PC・``force`` でない)
サーバを起こさずに保存済みの結果を要約する。測るときは起動スクリプトの ``start_rerank_server`` で
起こして測り、すぐ止める (rerank のポートが使用中なら測らない — 稼働中の測り直しは停止中の CLI に限る)。
"""

from __future__ import annotations

from typing import Any

from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, register

KEY = "rerank"


def _summary(result: Any) -> dict[str, Any]:
    return {
        "enabled": result.enabled, "placement": result.placement, "gpu_layers": result.gpu_layers,
        "candidates": result.candidates, "ms_per_doc": result.ms_per_doc,
    }


def run(ctx: TuneContext) -> TuneOutcome:
    """``rag.rerank.mode`` が off、またはモデルが未設定 / 未配置なら対象外。

    要るときだけ自己テストを走らせて要約を返す。モデルが無いのは既定 on でモデルを置いて
    いない環境の正常な状態なので ``skipped`` (``no_model`` / ``model_missing``) にする。
    """
    from backend.free.rag import rerank_selftest as st
    from backend.schemas.rag import rerank_mode_of

    if rerank_mode_of(ctx.cfg) == "off":
        return TuneOutcome("skipped", reason="off")
    try:
        from scripts import launch_llama as ll
    except ImportError:
        return TuneOutcome("failed", reason="launcher_unavailable", environmental=True)
    ok, _why = ll.rerank_launchable(ctx.cfg, ctx.project_root)
    if not ok:
        model = ll.resolve_rerank_model_path(ctx.cfg, ctx.project_root)
        reason = st.rerank_model_unavailable_reason(model, model is not None and model.is_file())
        return TuneOutcome("skipped", reason=reason)
    if not ctx.force:
        path = ll.rerank_selftest_path(ctx.cfg, ctx.project_root)
        saved = st.load_selftest_result(path)[0] if path is not None else None
        digest = st.collect_pc_info(ctx.hardware().gpu_names).digest
        needed, _why = st.selftest_needed(
            saved, digest, explicit_gpu_layers=ll._explicit_gpu_layers(ll._rerank_cfg(ctx.cfg)),
        )
        if not needed and saved is not None:
            return TuneOutcome("ok", value=_summary(saved), reason=saved.reason or "saved")
    launched = ll.start_rerank_server(ctx.cfg, ctx.project_root, force_selftest=ctx.force)
    if launched.proc is not None:
        ll._stop_proc(launched.proc)
    result = launched.result
    if result is None or not launched.tested:
        return TuneOutcome("failed", reason=launched.message or "not_started", environmental=True)
    if st.is_environmental_failure(result.reason):
        return TuneOutcome("failed", reason=result.reason, environmental=True)
    return TuneOutcome("ok", value=_summary(result), reason=result.reason)


SPEC = register(TuneSpec(
    key=KEY,
    config_key="rag.rerank.gpu_layers",
    method="measured",
    requires_restart=True,
    run=run,
    auto_values=("auto",),
    config_default="auto",
    # 埋め込みが VRAM を取った後に測る (c_16 §7.2.2「リランカーとの取り合い」)
    depends_on=("embed_placement",),
    safe_while_running=False,  # 本番ポートで測る
    needs_servers=True,
    description="reranker self-test (placement, ms/doc, candidates), c_16 §7.2.1",
))
