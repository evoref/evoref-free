"""SemMem GC 候補選定

`semmem_limits` 超過時に削除すべきファクトを純粋関数で選定する。
副作用は持たず、`SemanticFactStore.delete_fact` 呼び出しは
sleep-time Step 9 (`SleepTimeWorker._step9_run_semmem_gc`) が行う。

設計原則:
- pinned ファクトは常に削除対象外
- supersession チェーンの旧ファクトはアクティブ集合から除外済 (呼び出し側で
  `include_superseded=False` で渡す前提)
- スコア順位:
    1. ``eval_metric.fitness`` が存在すればこれを使用 (policy ファクト)
    2. それ以外は ``confidence``
- タイブレーカ: ``access_count`` (少ない順) → ``accessed_at`` (古い順) → ``id``
"""

from __future__ import annotations

from collections.abc import Iterable

from backend.free.memory.types import SemanticFact


def _gc_score(fact: SemanticFact) -> float:
    """GC スコアを計算する。値が低いほど削除候補。"""
    if fact.eval_metric:
        fitness = fact.eval_metric.get("fitness")
        if fitness is not None:
            return float(fitness)
    return float(fact.confidence)


def _gc_sort_key(fact: SemanticFact) -> tuple[float, int, float, str]:
    return (
        _gc_score(fact),
        fact.access_count,
        fact.accessed_at,
        fact.id,
    )


def select_gc_candidates(
    facts: Iterable[SemanticFact],
    *,
    max_count: int,
) -> list[str]:
    """``max_count`` を超過した分だけ削除候補 ID を返す。

    pinned は除外する。``max_count <= 0`` は「削除しない」扱い。
    返却順は削除順 (スコア低 → 高) で、呼び出し側はそのまま順次
    ``delete_fact`` してよい。
    """
    if max_count <= 0:
        return []
    active = [f for f in facts if not f.pinned and not f.superseded_by]
    if len(active) <= max_count:
        return []
    active.sort(key=_gc_sort_key)
    overflow = len(active) - max_count
    return [f.id for f in active[:overflow]]
