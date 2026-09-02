"""

EvorefMem pillar の内部で全 :class:`~backend.free.memory.types.FactType` に
対するフルアクセスを提供する Fact View。Mem 所有 FactType に対する
**sanctioned な書込/読取 API** を定義する。

## この View は何の検証点か

**Mem 内部の書込はここを通っていない。** sleep-time worker / 3 curator /
conflict resolver / GC / Step 9 promotion はいずれも
:class:`~backend.free.memory.semantic.store.SemanticFactStore` を直接操作する
(Mem 内部の store 直参照は pillar boundary test で許容)。つまり
``FACT_OWNERSHIP`` の強制が実際に効いているのは **loop / learn の書込** と、
**mem pillar の外から来る書込** だけ。

設計書を読んで「ownership が全書込に効いている」と誤解しないこと — それが
この構造の一番のコストだった (2026-09-01 監査 F11)。

現時点で production から呼ばれるのは:

- :meth:`search_by_embedding` — ``ToolCallJudge`` の URL / コマンドリコール。
- :meth:`add_fact` — ``POST /api/memory/pin`` (API 層は mem pillar の外なので
  必ずここを通す。2026-09-01 に store 直呼びから付け替えた)。

残りは Mem 内部が store 直参照で済ませているぶんの受け皿で、**新しく mem の
外から SemMem を触る経路を足すときはここを入口にする**。自己申告の
placeholder だった ``promote_episodic_to_semantic`` は実体
(:mod:`backend.free.memory.sleep.promotion`) が別に作られたまま置換されな
かったため削除した。

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

        # ``select_gc_candidates`` は active 集合に対する上限として働く契約
        # (docstring 参照) なので、supersede 済みは渡さない。渡しても内部で
        # 落とされるが、``sleep.gc.run_semmem_gc`` と条件が食い違っていた。
        target = self._store.search_by_type(fact_type, include_superseded=False)
        candidates = select_gc_candidates(target, max_count=max_count)
        if not candidates:
            return 0
        delete_many = getattr(self._store, "delete_facts", None)
        if callable(delete_many):
            return int(delete_many(candidates))
        # Protocol 実装 (Pro の差し替え等) が一括版を持たない場合の縮退。
        return sum(1 for fid in candidates if self._store.delete_fact(fid))


__all__ = ["MemFactView"]
