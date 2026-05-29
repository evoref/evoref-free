"""

EvorefLoop 配下のハーネス層 (``backend/free/harness/``) が prompt 生成時に
参照する ``policy`` / ``fewshot`` / ``failure_pattern`` の 3 型のみを
**読取専用**で提供する。

## 設計方針

- 本 View は **書込メソッドを一切持たない**。構造的に書込不能。
- owner (``policy``: Learn, ``fewshot``: Learn, ``failure_pattern``: Loop) の
  readers には ``"harness"`` が含まれる (
  :data:`~backend.free.memory.ownership.FACT_OWNERSHIP` 参照)。
- 他 FactType (``task`` / ``decision`` / ``personal_fact`` 等) は readers に
  ``"harness"`` が含まれないため、本 View から読取を試みると
  :class:`~backend.free.memory.views.base.PillarReadAccessError` が送出される。
- 純関数フィルタ (:meth:`filter_by_min_confidence`) は static method として
  提供し、テスト側で mock 不要。
"""

from __future__ import annotations

from collections.abc import Iterable

from backend.free.memory.ownership import Pillar
from backend.free.memory.protocols import SemanticFactStoreProtocol
from backend.free.memory.types import MemoryMode, SemanticFact
from backend.free.memory.views.base import (
    FactViewBase,
    merge_active_facts_across_stores,
)


class HarnessFactView(FactViewBase):
    """ハーネス層の読取専用 Fact View。

    Attributes:
        pillar: ``"harness"`` 固定。
    """

    pillar: Pillar = "harness"

    def __init__(
        self,
        *,
        stores: Iterable[SemanticFactStoreProtocol],
    ) -> None:
        """Args:
            stores: 読取対象の全スコープ (global + project)。
        """
        self._stores: list[SemanticFactStoreProtocol] = list(stores)
        if not self._stores:
            raise ValueError("HarnessFactView requires at least one store")

    # ──────────────────────────────────────────────────────────────────
    # 読取系 (書込メソッドは意図的に非提供)
    # ──────────────────────────────────────────────────────────────────

    def get_active_policies(
        self,
        *,
        mode: MemoryMode | None = None,
        min_confidence: float = 0.7,
    ) -> list[SemanticFact]:
        """``mode`` 付きで有効な policy を返す (superseded 除外)。"""
        self._assert_read("policy")
        return merge_active_facts_across_stores(
            self._stores, "policy", min_confidence=min_confidence, mode=mode,
        )

    def get_active_fewshots(
        self,
        *,
        mode: MemoryMode | None = None,
        min_fitness: float = 0.0,
    ) -> list[SemanticFact]:
        """``mode`` 付きで有効な fewshot を返す (superseded 除外)。"""
        self._assert_read("fewshot")
        return merge_active_facts_across_stores(
            self._stores, "fewshot", min_confidence=min_fitness, mode=mode,
        )

    def get_recent_failures(
        self,
        *,
        mode: MemoryMode | None = None,
        limit: int = 5,
    ) -> list[SemanticFact]:
        """最近の failure_pattern を accessed_at 降順で最大 ``limit`` 件返す。"""
        self._assert_read("failure_pattern")
        collected: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.search_by_type("failure_pattern"):
                if fact.id in seen or fact.superseded_by:
                    continue
                if mode is not None and fact.mode_origin != mode:
                    continue
                collected.append(fact)
                seen.add(fact.id)
        collected.sort(key=lambda f: f.accessed_at, reverse=True)
        return collected[: max(0, int(limit))]

    # ──────────────────────────────────────────────────────────────────
    # 純関数フィルタ
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def filter_by_min_confidence(
        facts: Iterable[SemanticFact],
        threshold: float,
    ) -> list[SemanticFact]:
        """``confidence >= threshold`` のみ残す純関数フィルタ。"""
        return [f for f in facts if f.confidence >= threshold]


__all__ = ["HarnessFactView"]
