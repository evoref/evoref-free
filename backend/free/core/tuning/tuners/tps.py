"""項目 ``tps`` — 生成速度の実測と、それから導いた締切 (c_16 §7.2.3「ストリーム締切」「補助 / エージェントの上限」)。

測るのは backend 自身 (応答の ``timings`` を ``cache/tps_calibration.json`` へ間引いて保存、
:mod:`backend.free.llm.tps_calibration`)。ここはサーバを起こさず、保存済みの実測と、それを使ったときの
締切の要約を返すだけ。実測が無い / 別の PC で測った / 最小サンプル数に届いていないときは
``reason="no_samples"`` で、締切は現行の定数のまま動く。

反映は常に自動 (config のキーは持たない)。ただし ``agent.tool_classifier_timeout_sec`` /
``agent.llm_call_timeout`` を既定から変えていれば、その値は延ばさないので ``manual`` を返す
(ストリーム締切と補助タスクの上限は引き続き実測で延びる)。
"""

from __future__ import annotations

from typing import Any

from backend.free.core.tuning.items import MANUAL, TuneContext, TuneOutcome, TuneSpec, config_value, register

KEY = "tps"
#: 締切の要約に使う代表的な要求 (チャットの冷えた prefill と、補助判定の小さな要求)。
SUMMARY_PROMPT_TOKENS = 2000
SUMMARY_AUX_PROMPT_TOKENS = 600
SUMMARY_AUX_MAX_TOKENS = 256


def _model_key(ctx: TuneContext, resolver: Any) -> str:
    base_model = config_value(ctx.cfg, "model_paths.base_model", "")
    return resolver.model_key_for(base_model) if base_model else ""


def summarize_deadlines(cfg: dict[str, Any], tps: Any) -> dict[str, Any]:
    """``tps`` (``TpsEstimate`` / ``None``) で組んだときの締切の要約 (秒)。"""
    from backend.free.llm.aux_client import (
        CHAT_PATH_PURPOSES,
        PURPOSE_TIMEOUT_CALIBRATION_EXEMPT,
        PURPOSE_TIMEOUT_DEFAULTS,
    )
    from backend.free.llm.local_client import (
        STREAM_FIRST_TOKEN_TIMEOUT,
        stream_first_token_deadline,
        stream_total_deadline,
    )
    from backend.free.llm.tps_calibration import (
        CHAT_PATH_MAX_SCALE,
        chat_path_timeout,
        is_explicit_agent_timeout,
    )

    llama = cfg.get("llama") or {}
    agent = cfg.get("agent") or {}
    base = float(llama.get("stream_first_token_timeout_sec") or STREAM_FIRST_TOKEN_TIMEOUT)
    max_tokens = int(llama.get("max_tokens") or 0) or None

    def _linked(key: str, default: float, out_tokens: int) -> float:
        current = float(agent.get(key, default))
        if is_explicit_agent_timeout(agent, key):
            return current
        return round(chat_path_timeout(
            current, tps, prompt_tokens=SUMMARY_AUX_PROMPT_TOKENS, max_tokens=out_tokens,
        ), 1)

    aux = {
        purpose: round(chat_path_timeout(
            PURPOSE_TIMEOUT_DEFAULTS[purpose], tps,
            prompt_tokens=SUMMARY_AUX_PROMPT_TOKENS, max_tokens=SUMMARY_AUX_MAX_TOKENS,
            # 実行時 (``AuxClient._tps_extended_chat_timeout``) と同じ天井
            ceiling=PURPOSE_TIMEOUT_DEFAULTS[purpose] * CHAT_PATH_MAX_SCALE,
        ), 1)
        for purpose in sorted(CHAT_PATH_PURPOSES - PURPOSE_TIMEOUT_CALIBRATION_EXEMPT)
    }
    return {
        f"stream_first_token_sec_at_{SUMMARY_PROMPT_TOKENS}": round(
            stream_first_token_deadline(SUMMARY_PROMPT_TOKENS, base=base, tps=tps), 1,
        ),
        "stream_total_sec": round(stream_total_deadline(max_tokens, tps=tps), 1),
        "tool_classifier_sec": _linked("tool_classifier_timeout_sec", 60.0, 96),
        "llm_call_sec": _linked("llm_call_timeout", 90.0, SUMMARY_AUX_MAX_TOKENS),
        "aux_chat_path_sec": aux,
    }


def run(ctx: TuneContext) -> TuneOutcome:
    """保存済みの実測 (この PC・base モデル) と、それで組んだ締切の要約を返す。"""
    from backend.config import PathResolver
    from backend.free.llm.tps_calibration import (
        LINKED_AGENT_TIMEOUT_KEYS,
        MIN_SAMPLES,
        estimate_of,
        is_explicit_agent_timeout,
        load_tps_calibration,
    )
    from backend.free.rag.rerank_selftest import collect_pc_info, pc_mismatch

    resolver = PathResolver(ctx.cfg, ctx.project_root)
    model_key = _model_key(ctx, resolver)
    record = load_tps_calibration(resolver.resolve_local("tps_calibration_file"))
    current_pc = collect_pc_info(ctx.hardware().gpu_names)
    entry = None
    if record is not None and not pc_mismatch(record.pc, current_pc):
        entry = record.models.get(model_key)
    tps = estimate_of(entry)
    value: dict[str, Any] = {
        "model_key": model_key,
        "prefill_tps": round(entry.prefill_tps, 2) if entry is not None and entry.prefill_samples else None,
        "decode_tps": round(entry.decode_tps, 2) if entry is not None and entry.decode_samples else None,
        "samples": {
            "prefill": entry.prefill_samples if entry is not None else 0,
            "decode": entry.decode_samples if entry is not None else 0,
            "min": MIN_SAMPLES,
        },
        "deadlines": summarize_deadlines(ctx.cfg, tps),
    }
    agent = ctx.cfg.get("agent") or {}
    manual = [f"agent.{k}" for k in LINKED_AGENT_TIMEOUT_KEYS if is_explicit_agent_timeout(agent, k)]
    if manual:
        value["manual"] = manual
        return TuneOutcome("skipped", value=value, reason=MANUAL)
    return TuneOutcome("ok", value=value, reason="measured" if tps is not None else "no_samples")


SPEC = register(TuneSpec(
    key=KEY,
    config_key="",
    method="estimated",
    requires_restart=False,
    run=run,
    needs_servers=False,
    safe_while_running=True,  # 保存済みの実測を読むだけ
    description="generation speed (prefill / decode tok/s) and the deadlines derived from it, c_16 §7.2.3",
))
