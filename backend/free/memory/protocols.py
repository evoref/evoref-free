"""

EvorefMem (memory pillar) が他 pillar (EvorefLoop / EvorefLearn) に公開する
契約を型として固定する。具象クラスへの直接結合を避け、Fact View 層 や
wire_pillars での DI を Protocol 境界に乗せる

現状の Free 実装は `backend.free.memory.semantic.store.SemanticFactStore` が
本 Protocol を **自然に満たす**。Pro 版でも共有クラスを再利用するため、
Protocol は duck typing で十分機能する (別実装を差し込む必要が生じたら
将来的に実装差替を行う)

設計原則 (CLAUDE.md §8 / `docs/f_02_memory_system.md` §6.5):
- 最小 API 原則: 実装が実際に呼び出している API だけを宣言する
- `@runtime_checkable`: isinstance チェックを可能にする
- Protocol ファイルは他 pillar を import しない (最下流 pillar の不変条件)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from backend.free.memory.types import FactType, SemanticFact


@runtime_checkable
class SemanticFactStoreProtocol(Protocol):
    """SemanticFact の永続化抽象。

    1 スコープ (`global` または `project:<id>`) に対応する単一ストアが
    本 Protocol を満たす。Fact View 層はこの Protocol 経由で
    永続化層にアクセスし、`SemanticFactStore` 実装への直接結合を避ける。

    最小 API セット:
    - CRUD: ``add_fact`` / ``get_fact`` / ``update_fact`` / ``delete_fact``
    - 置換: ``supersede``
    - 検索: ``search_by_subject`` / ``search_by_type`` /
      ``search_by_pillar_prefix`` / ``search_by_embedding``
    - 集合: ``all_facts`` / ``pinned_facts`` / ``count_by_type``
    """

    def add_fact(self, fact: SemanticFact) -> SemanticFact:
        """新規ファクトを追加する。既存 ID と衝突したら ``ValueError``。"""
        ...

    def get_fact(self, fact_id: str) -> SemanticFact | None:
        """ID でファクトを取得する。存在しなければ ``None``。"""
        ...

    def update_fact(self, fact_id: str, **changes: Any) -> SemanticFact:
        """既存ファクトのフィールドを差分更新する。"""
        ...

    def delete_fact(self, fact_id: str) -> bool:
        """ファクトを物理削除する。実際に削除できた場合 ``True``。"""
        ...

    def supersede(self, old_id: str, new_id: str) -> None:
        """``old_id`` を ``new_id`` で置き換える supersession チェーンを構築する。"""
        ...

    def search_by_subject(
        self,
        subject: str,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """``subject`` 完全一致でファクトを返す。"""
        ...

    def search_by_type(
        self,
        fact_type: FactType,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """``type`` 完全一致でファクトを返す。"""
        ...

    def search_by_pillar_prefix(
        self,
        prefix: str,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """3 pillar namespace (``loop.`` / ``learn.`` / ``mem.``) の subject
        前方一致でファクトを返す。"""
        ...

    def search_by_embedding(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        include_superseded: bool = False,
    ) -> list[tuple[SemanticFact, float]]:
        """埋め込みベクトルで cosine similarity 検索する。"""
        ...

    def all_facts(self, *, include_superseded: bool = True) -> list[SemanticFact]:
        """全ファクトを返す (観測・テスト用)。"""
        ...

    def pinned_facts(self) -> list[SemanticFact]:
        """pinned ファクトを返す。"""
        ...

    def count_by_type(
        self,
        fact_type: FactType,
        *,
        include_superseded: bool = False,
    ) -> int:
        """``type`` に該当するファクト数を返す。"""
        ...

    def __len__(self) -> int:  # pragma: no cover - trivial
        ...


__all__ = ["SemanticFactStoreProtocol"]
