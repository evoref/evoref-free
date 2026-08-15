"""RAG necessity/quality の embedding 決定論的リコール

``ToolCallJudge._try_recall_url`` / ``_try_recall_executable_command`` と
同型のパターン。``mem.world.rag_necessity.*`` / ``mem.world.rag_quality.*``
を query embedding で検索し、類似度 + 過去自己採点平均 + TTL 減衰の閾値を
満たせば aux 呼出を完全スキップして即決定する。

LLM 不使用・``JudgeUsageTracker`` 予算を消費しない。書込は
sleep-time (``backend.free.memory.sleep.rag_judge_curator``) に限定される
(CLAUDE.md §6 #2)。本モジュールは読取のみ。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.views.mem import MemFactView

logger = get_logger("memory.pipeline.rag_judge_recall")

_NECESSITY_SUBJECT_PREFIX = "mem.world.rag_necessity."
_QUALITY_SUBJECT_PREFIX = "mem.world.rag_quality."
_VALID_ACTIONS = {"retrieve", "fetch", "skip"}
_VALID_QUALITIES = {"high", "medium", "low"}


def _resolve_recall_cfg(raw: dict | None) -> dict:
    cfg = raw or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "topk": int(cfg.get("topk", 5)),
        "min_score": float(cfg.get("min_score", 0.75)),
        "min_record_score": float(cfg.get("min_record_score", 0.65)),
        "ttl_days": int(cfg.get("ttl_days", 14)),
    }


async def _try_recall(
    query_vec: np.ndarray,
    mem_view: "MemFactView | None",
    cfg: dict,
    *,
    subject_prefix: str,
    decision_field: str,
    valid_values: set[str],
) -> tuple[str, float] | None:
    resolved = _resolve_recall_cfg(cfg)
    if mem_view is None or not resolved["enabled"]:
        return None
    try:
        candidates = mem_view.search_by_embedding(query_vec, top_k=resolved["topk"])
    except Exception as exc:
        logger.warning("rag_judge recall: search_by_embedding failed: %s", exc)
        return None

    now = time.time()
    ttl_seconds = float(resolved["ttl_days"]) * 86400.0 if resolved["ttl_days"] > 0 else 0.0
    for fact, sim in candidates:
        if fact.type != "world_fact" or not fact.subject.startswith(subject_prefix):
            continue
        if sim < resolved["min_score"]:
            # candidates はスコア降順想定。閾値未満は以降全て無効。
            break
        extra = fact._extra or {}
        decision = extra.get(decision_field)
        if decision not in valid_values:
            continue
        score_avg = float(extra.get("score_avg") or 0.0)
        effective = score_avg
        if ttl_seconds > 0.0:
            last = float(extra.get("last_judged_at") or 0.0)
            if last > 0.0 and (now - last) > ttl_seconds:
                effective = score_avg * 0.5
        if effective >= resolved["min_record_score"]:
            return decision, float(sim)
    return None


async def try_recall_necessity(
    query_vec: np.ndarray, mem_view: "MemFactView | None", cfg: dict,
) -> tuple[str, float] | None:
    """検索必要性 (retrieve/fetch/skip) の embedding 決定論的リコールを試みる。"""
    return await _try_recall(
        query_vec, mem_view, cfg,
        subject_prefix=_NECESSITY_SUBJECT_PREFIX, decision_field="action",
        valid_values=_VALID_ACTIONS,
    )


async def try_recall_quality(
    query_vec: np.ndarray, mem_view: "MemFactView | None", cfg: dict,
) -> tuple[str, float] | None:
    """検索品質 (high/medium/low) の embedding 決定論的リコールを試みる。"""
    return await _try_recall(
        query_vec, mem_view, cfg,
        subject_prefix=_QUALITY_SUBJECT_PREFIX, decision_field="quality",
        valid_values=_VALID_QUALITIES,
    )


__all__ = ["try_recall_necessity", "try_recall_quality"]
