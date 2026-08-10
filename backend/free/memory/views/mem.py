"""

EvorefMem pillar の内部で全 :class:`~backend.free.memory.types.FactType` に
対するフルアクセスを提供する Fact View。Mem 所有 FactType に対する
**sanctioned な書込/読取 API** を定義する。

注: Mem 内部の sleep-time worker / conflict resolver / GC / Step 9 promotion は
現状 :class:`~backend.free.memory.semantic.store.SemanticFactStore` を直接
操作している (Mem 内部の store 直参照は pillar boundary test で許容)。本 View の
高レベル API は将来それらを集約する受け皿であり、現時点で全メソッドが production
から呼ばれているわけではない (越境 reader の入口としての利用が主)。

## スコープ

1 View インスタンス = 1 ストア (``global`` または ``project:<id>``)。
複数スコープを横断する呼び出し側 (例: sleep-time worker の bulk add) は
スコープごとに View を生成するか、store 直参照を用いる (Mem 内部の store
直参照は許容、pillar boundary test で検証)

## 設計方針

- owner pillar は ``"mem"``。書込時に :meth:`_assert_write` で検証。
- 読取系は Mem が全 FactType を読めるわけではない (例: ``policy`` は
  Mem readers に含まれない)。従って型別読取メソッドは :meth:`_assert_read`
  を通し、Mem が reader の type のみ返す。しかし本 View の ``search_by_*``
  は生 store メソッドをそのまま返すシンプル実装とし、呼び出し側
  (Mem 内部の conflict_resolver / GC 等) が対象の type を制御する
  (Mem 内部の全検索は mem-owned type + readers に含まれる type に限る)。
- 高レベル操作 (``resolve_conflicts`` / ``gc_facts`` / ``promote_episodic_to_semantic``)
  は既存 Mem 内部モジュールを薄くラップする (同 pillar 内 import)。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.free.memory.ownership import Pillar
from backend.free.memory.protocols import SemanticFactStoreProtocol
from backend.free.memory.types import FactType, SemanticFact
from backend.free.memory.views.base import FactViewBase


class MemFactView(FactViewBase):
    """EvorefMem pillar の共有基盤 Fact View。

    Mem 所有 FactType (``personal_fact`` / ``world_fact`` / ``preference`` /
    ``emotion`` / ``opinion`` / ``belief`` / ``decision`` / ``commitment`` /
    ``project`` / ``create`` / ``create_task`` / ``model``) に対しフルアクセスを
    提供する。

    Attributes:
        pillar: ``"mem"`` 固定。
    """

    pillar: Pillar = "mem"

    def __init__(self, store: SemanticFactStoreProtocol) -> None:
        """1 スコープの :class:`SemanticFactStoreProtocol` を紐付ける。

        Args:
            store: global / project のいずれか 1 スコープに対応するストア。
        """
        self._store: SemanticFactStoreProtocol = store

    # ──────────────────────────────────────────────────────────────────
    # 読取系
    # ──────────────────────────────────────────────────────────────────

    def get_fact(self, fact_id: str) -> SemanticFact | None:
        """ID でファクトを取得する。存在しなければ ``None``。"""
        return self._store.get_fact(fact_id)

    def search_by_subject(
        self,
        subject: str,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """``subject`` 完全一致でファクトを返す。"""
        return self._store.search_by_subject(
            subject, include_superseded=include_superseded,
        )

    def search_by_type(
        self,
        fact_type: FactType,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """``fact_type`` 完全一致でファクトを返す。

        ``fact_type`` の readers に Mem が含まれない場合は
        :class:`~backend.free.memory.views.base.PillarReadAccessError` を送出する。
        """
        self._assert_read(fact_type)
        return self._store.search_by_type(
            fact_type, include_superseded=include_superseded,
        )

    def search_by_pillar_prefix(
        self,
        prefix: str,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """3 pillar namespace (``loop.`` / ``learn.`` / ``mem.``) の subject
        前方一致でファクトを返す

        旧 ``search_by_harness_prefix`` から改名。Mem が読取可能な全
        FactType の subject を対象とするため、caller 側で必要に応じて
        :meth:`search_by_type` との組み合わせで絞り込むこと。
        """
        return self._store.search_by_pillar_prefix(
            prefix, include_superseded=include_superseded,
        )

    def search_by_embedding(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        include_superseded: bool = False,
    ) -> list[tuple[SemanticFact, float]]:
        """埋め込みベクトルで cosine similarity 検索する。"""
        return self._store.search_by_embedding(
            query, top_k=top_k, include_superseded=include_superseded,
        )

    def all_facts(self, *, include_superseded: bool = True) -> list[SemanticFact]:
        """全ファクトを返す (Mem 内部の conflict 検出 / GC 候補列挙で使用)。"""
        return self._store.all_facts(include_superseded=include_superseded)

    def pinned_facts(self) -> list[SemanticFact]:
        """pinned ファクトを返す。"""
        return self._store.pinned_facts()

    def count_by_type(
        self,
        fact_type: FactType,
        *,
        include_superseded: bool = False,
    ) -> int:
        """``fact_type`` に該当するファクト数を返す。"""
        self._assert_read(fact_type)
        return self._store.count_by_type(
            fact_type, include_superseded=include_superseded,
        )

    # ──────────────────────────────────────────────────────────────────
    # 書込系
    # ──────────────────────────────────────────────────────────────────

    def add_fact(self, fact: SemanticFact) -> SemanticFact:
        """1 ファクトを追加する。owner (``fact.type``) が ``mem`` でなければ raise。

        Raises:
            WriteOwnershipError: ``fact.type`` の owner が Mem でない。
            SubjectNamespaceError: ``fact.subject`` が非 ``mem.`` prefix の
                pillar namespace (``loop.*`` / ``learn.*``) を持つ。
        """
        self._assert_write(fact.type)
        self._assert_subject_owner(fact.subject)
        return self._store.add_fact(fact)

    def add_fact_bulk(self, facts: list[SemanticFact]) -> list[SemanticFact]:
        """複数ファクトをまとめて追加する。全ファクトの owner 検証を先に実施する。

        アトミックではない (途中で失敗した場合、先行成功分は残る)。
        """
        for fact in facts:
            self._assert_write(fact.type)
            self._assert_subject_owner(fact.subject)
        return [self._store.add_fact(fact) for fact in facts]

    def update_fact_fields(self, fact_id: str, **changes: Any) -> SemanticFact:
        """既存ファクトのフィールドを差分更新する。

        owner 検証は既存ファクトの ``type`` に対し実施する。``type`` を
        ``changes`` で変更する場合は変更後の ``type`` も owner であること。

        Raises:
            KeyError: ``fact_id`` が存在しない。
            WriteOwnershipError: 既存/変更後の ``type`` owner が Mem でない。
        """
        existing = self._store.get_fact(fact_id)
        if existing is None:
            raise KeyError(fact_id)
        self._assert_write(existing.type)
        if "type" in changes:
            self._assert_write(changes["type"])
        if "subject" in changes:
            self._assert_subject_owner(changes["subject"])
        return self._store.update_fact(fact_id, **changes)

    def supersede_fact(self, old_id: str, new_id: str) -> None:
        """``old_id`` を ``new_id`` で置き換える supersession チェーンを構築する。

        両者の ``type`` が一致し、かつ Mem owner である必要がある。

        Raises:
            KeyError: old / new のいずれかが存在しない。
            ValueError: 両者の ``type`` が異なる。
            WriteOwnershipError: ``type`` の owner が Mem でない。
        """
        old = self._store.get_fact(old_id)
        new = self._store.get_fact(new_id)
        if old is None:
            raise KeyError(f"old fact not found: {old_id}")
        if new is None:
            raise KeyError(f"new fact not found: {new_id}")
        if old.type != new.type:
            raise ValueError(
                f"supersede type mismatch: old={old.type!r} new={new.type!r}",
            )
        self._assert_write(old.type)
        self._store.supersede(old_id, new_id)

    def delete_fact(self, fact_id: str) -> bool:
        """ファクトを物理削除する。

        Returns:
            削除された場合 ``True``、未存在の場合 ``False``。

        Raises:
            WriteOwnershipError: 対象ファクトの ``type`` owner が Mem でない。
        """
        existing = self._store.get_fact(fact_id)
        if existing is None:
            return False
        self._assert_write(existing.type)
        return self._store.delete_fact(fact_id)

    # ──────────────────────────────────────────────────────────────────
    # 高レベル操作 (Mem 内部モジュールへの薄いラッパ)
    # ──────────────────────────────────────────────────────────────────

    def resolve_conflicts(
        self,
        *,
        config: dict[str, Any] | None = None,
        now_provider: Any = None,
    ) -> dict[str, int]:
        """ストア内の競合を検出・解決する (Mem 内部の Step 6 相当)。

        :class:`~backend.free.memory.pipeline.semantic_conflict_resolver.SemanticConflictResolver`
        を薄くラップする。

        Returns:
            ``{"detected", "auto_resolved", "pending", "groups"}`` のサマリ。
        """
        from backend.free.memory.pipeline.semantic_conflict_resolver import (
            SemanticConflictResolver,
        )

        # SemanticConflictResolver は具象 SemanticFactStore を要求するため、
        # 実行時の型安全性は caller の責任 (Mem 内部からの呼出を想定)。
        resolver = SemanticConflictResolver(
            self._store,  # type: ignore[arg-type]
            config=config,
            now_provider=now_provider,
        )
        return resolver.resolve()

    def gc_facts(self, *, fact_type: FactType, max_count: int) -> int:
        """件数超過分のファクトを GC する (削除件数を返す)。

        Mem owner の FactType に対してのみ実行可能。
        :func:`~backend.free.memory.semantic.gc.select_gc_candidates` で削除
        候補を決定し、1 件ずつ :meth:`delete_fact` で消す。

        Args:
            fact_type: 対象 FactType (Mem owner のもの)。
            max_count: 残す最大件数 (これを超えた分が削除候補)。

        Returns:
            削除されたファクト数。
        """
        self._assert_write(fact_type)
        from backend.free.memory.semantic.gc import select_gc_candidates

        target = self._store.search_by_type(fact_type, include_superseded=True)
        candidates = select_gc_candidates(target, max_count=max_count)
        deleted = 0
        for fid in candidates:
            if self._store.delete_fact(fid):
                deleted += 1
        return deleted

    def promote_episodic_to_semantic(
        self,
        *,
        facts: list[SemanticFact],
    ) -> list[SemanticFact]:
        """エピソディック (STM 由来) を意味記憶に昇格する (Step 9 相当の placeholder)。

        シンプル実装として、``decision`` / ``commitment``
        ファクトを ``add_fact`` で追加するだけ
        ``sleep_update.py`` の Step 9 promotion ロジックが専用 module
        (``memory/promotion.py``) に分離され、本メソッドが本実装を呼び出す形に
        置換される予定。

        Args:
            facts: 昇格対象のファクト群 (``decision`` / ``commitment`` のみ許容)。

        Returns:
            永続化された :class:`SemanticFact` のリスト。

        Raises:
            ValueError: 許容外の ``fact.type`` が含まれる。
            WriteOwnershipError: owner が Mem でない type が含まれる。
        """
        allowed: set[str] = {"decision", "commitment"}
        for fact in facts:
            if fact.type not in allowed:
                raise ValueError(
                    "promote_episodic_to_semantic only supports "
                    f"{sorted(allowed)!r}: got {fact.type!r}",
                )
            self._assert_write(fact.type)
            self._assert_subject_owner(fact.subject)
        return [self._store.add_fact(fact) for fact in facts]


__all__ = ["MemFactView"]
