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

import time
from collections.abc import Callable, Iterable

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


def select_expired_superseded(
    facts: Iterable[SemanticFact],
    *,
    retention_days: float,
    now_provider: Callable[[], float] | None = None,
) -> list[str]:
    """保持期間を過ぎた supersede 済みファクトの ID を返す (純粋関数)。

    type 別上限 (:func:`select_gc_candidates`) は **active ファクトにしか
    効かない** — GC は ``search_by_type()`` の既定
    (``include_superseded=False``) しか見ず、``select_gc_candidates`` も
    さらに ``not f.superseded_by`` で絞る。``compact`` も同一 id の重複行を
    畳むだけで supersede 済みは落とさない。結果、回収経路が 1 つも無く、
    メモリ上の ``_facts`` にも ``facts.jsonl`` にも ``vectors.npy`` にも永久に
    残っていた (実測 2026-09-01: global 25.5% / project 48.0% が supersede
    済み、npy 108 行のうち 31 行が死んだ行として毎回ソート対象)。

    落とさない条件は 2 つ:

    - **pinned** — 他の GC と同じ扱い。
    - **他のファクトから ``superseded_by`` で指されている** (= 鎖の途中や先頭)。
      これは容量の話ではなく **正しさ** の話。``delete_facts`` は消えた
      superseder を指すファクトを live へ戻す (そうしないとスロットの live が
      0 件になる) ので、鎖 ``A -> B -> C`` の **B を消すと A が live に
      復活する** — 訂正で捨てたはずの値が現在値として蘇る。安全に消せるのは
      **誰からも指されていない鎖の末尾 (最古)** だけ。サイクルを重ねるごとに
      鎖は古い側から 1 世代ずつ削れていく。

    ``supersedes`` (由来の記録) に載っていることは削除を妨げない —
    ``delete_facts`` の ``_clear_dangling_supersession`` が残った側のリストから
    その ID を外すので、dangling は生じない。

    経過時間は ``accessed_at`` ではなく **``created_at``** を基準にする。
    supersede そのものが ``update_fact`` を通るため ``accessed_at`` は
    「置き換えられた時刻」に更新されてしまい、古い世代ほど新しく見える。

    Args:
        facts: 判定対象 (``include_superseded=True`` で渡すこと)。
        retention_days: 保持日数。``0`` 以下は「削除しない」。
        now_provider: 現在時刻 (epoch 秒)。テスト用。

    Returns:
        削除候補 ID (``created_at`` 昇順 = 古い順)。
    """
    if retention_days <= 0:
        return []
    now = (now_provider or time.time)()
    cutoff = now - retention_days * 86400.0
    materialized = list(facts)
    # 誰かの ``superseded_by`` が指している ID は残す。消すと
    # ``delete_facts`` がその参照元を live へ戻し、捨てた値が復活する。
    pointed_at: set[str] = {
        f.superseded_by for f in materialized if f.superseded_by
    }
    expired = [
        f for f in materialized
        if f.superseded_by
        and not f.pinned
        and f.id not in pointed_at
        and float(f.created_at or 0.0) < cutoff
    ]
    expired.sort(key=lambda f: (f.created_at, f.id))
    return [f.id for f in expired]


__all__ = ["select_expired_superseded", "select_gc_candidates"]
