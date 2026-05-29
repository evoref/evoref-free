"""

EvorefLearn pillar が書き込む ``policy`` / ``fewshot`` の管理、および
Learn が reader として読む ``failure_pattern`` / ``task`` /
``progress_marker`` / ``decision`` / ``commitment`` / ``project`` /
``coding`` / ``coding_task`` の読取 API を提供する。

## 設計方針

- 書込系 (:meth:`write_policy` / :meth:`write_fewshot` /
  :meth:`supersede_old_policy`) は owner ``"learn"`` を前提に設計。
  ``failure_pattern`` の owner は EvorefLoop のため、**現 
  ownership enforcement により常に** :class:`WriteOwnershipError`
  **を送出する構造プレースホルダ**として実装する
  :class:`~backend.free.memory.protocols.LoopWriteAPIProtocol` 経由の
  delegation 実装に置換される予定。
- 読取系は ``failure_pattern`` や ``task`` など Learn readers に含まれる
  型を横断ストアから集める。PolicyParamEvolver / CritiqueSynthesizer /
  FewShotPool が本 View を参照する。

## subject 命名

pillar 命名が一括適用され、Learn pillar が書き込む
subject は必ず ``learn.policy.*`` / ``learn.fewshot.*`` / ``learn.metric.*``
のいずれかで始まる。``loop.*`` / ``mem.*`` の namespace を持つ subject を
書き込もうとすると :class:`~backend.free.memory.notes.subject_ns.SubjectNamespaceError`
が送出される (:meth:`FactViewBase._assert_subject_owner`)。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from backend.free.memory.ownership import Pillar
from backend.free.memory.protocols import SemanticFactStoreProtocol
from backend.free.memory.types import (
    MemoryMode,
    SemanticFact,
    make_fact,
)
from backend.free.memory.views.base import (
    FactViewBase,
    merge_active_facts_across_stores,
    safe_json_loads as _safe_json_loads,
)


class LearnFactView(FactViewBase):
    """EvorefLearn pillar の Fact View。

    Attributes:
        pillar: ``"learn"`` 固定。
    """

    pillar: Pillar = "learn"

    def __init__(
        self,
        *,
        stores: Iterable[SemanticFactStoreProtocol],
        writeback_store: SemanticFactStoreProtocol,
    ) -> None:
        """Args:
            stores: 読取対象の全スコープ (global + project)。
            writeback_store: Learn 所有 fact (policy / fewshot) の書込先。
        """
        self._stores: list[SemanticFactStoreProtocol] = list(stores)
        if not self._stores:
            raise ValueError("LearnFactView requires at least one store")
        self._writeback_store: SemanticFactStoreProtocol = writeback_store

    # ──────────────────────────────────────────────────────────────────
    # 読取系
    # ──────────────────────────────────────────────────────────────────

    def get_failure_patterns_for_critique(
        self,
        *,
        project_id: str,
        min_confidence: float = 0.5,
        min_occurrences: int = 1,
    ) -> list[SemanticFact]:
        """critique (失敗分析) 対象の failure_pattern を返す。

        Args:
            project_id: 対象 project ID。
            min_confidence: 下限 confidence。
            min_occurrences: 下限 occurrences (payload から抽出)。
        """
        self._assert_read("failure_pattern")
        project_scope = SemanticFact.make_project_scope(project_id)
        result: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.search_by_type("failure_pattern"):
                if fact.id in seen or fact.superseded_by:
                    continue
                if fact.scope != project_scope:
                    continue
                if fact.confidence < min_confidence:
                    continue
                payload = _safe_json_loads(fact.object)
                try:
                    occ = int(payload.get("occurrences", 0))
                except (TypeError, ValueError):
                    occ = 0
                if occ < min_occurrences:
                    continue
                result.append(fact)
                seen.add(fact.id)
        return result

    def get_failed_tasks(self, *, project_id: str) -> list[SemanticFact]:
        """``failed`` 状態の task ファクトを返す。"""
        self._assert_read("task")
        project_scope = SemanticFact.make_project_scope(project_id)
        result: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.search_by_type("task"):
                if fact.id in seen or fact.superseded_by:
                    continue
                if fact.scope != project_scope:
                    continue
                payload = _safe_json_loads(fact.object)
                if payload.get("status") == "failed":
                    result.append(fact)
                    seen.add(fact.id)
        return result

    def get_active_policies(
        self,
        *,
        min_confidence: float = 0.7,
        mode: MemoryMode | None = None,
    ) -> list[SemanticFact]:
        """有効 policy を返す (Learn は owner なので readers 自動含有)。"""
        self._assert_read("policy")
        return merge_active_facts_across_stores(
            self._stores, "policy", min_confidence=min_confidence, mode=mode,
        )

    def get_fewshot_examples(
        self,
        *,
        mode: MemoryMode | None = None,
        min_fitness: float = 0.0,
    ) -> list[SemanticFact]:
        """fewshot 例を返す。"""
        self._assert_read("fewshot")
        result: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.search_by_type("fewshot"):
                if fact.id in seen or fact.superseded_by:
                    continue
                if fact.confidence < min_fitness:
                    continue
                if mode is not None and fact.mode_origin != mode:
                    continue
                result.append(fact)
                seen.add(fact.id)
        return result

    # ──────────────────────────────────────────────────────────────────
    # 書込系 — policy
    # ──────────────────────────────────────────────────────────────────

    def write_policy(
        self,
        *,
        subject: str,
        predicate: str,
        object_: str,
        scope: str = "global",
        confidence: float = 0.5,
        mode_origin: MemoryMode = "chat",
        auto_evolved: bool = False,
        eval_metric: dict[str, float] | None = None,
        trace_id: str | None = None,
        now: float | None = None,
    ) -> SemanticFact:
        """policy ファクトを書き込む。

        Raises:
            WriteOwnershipError: caller pillar が ``"learn"`` でない (通常起きない)。
            SubjectNamespaceError: ``subject`` が ``loop.*`` / ``mem.*`` prefix を持つ。
        """
        self._assert_write("policy")
        self._assert_subject_owner(subject)
        fact = make_fact(
            subject=subject,
            predicate=predicate,
            object_=object_,
            type="policy",
            scope=scope,
            confidence=confidence,
            mode_origin=mode_origin,
            now=now,
            trace_id=trace_id,
            auto_evolved=auto_evolved,
            eval_metric=eval_metric,
        )
        return self._writeback_store.add_fact(fact)

    def supersede_old_policy(self, *, old_id: str, new_id: str) -> None:
        """古い policy を新しい policy で置き換える。両者の ``type`` が ``policy`` であること。

        Raises:
            KeyError: old / new のいずれかが存在しない。
            ValueError: old / new のいずれかが ``type='policy'`` でない。
            WriteOwnershipError: owner が Learn でない (通常起きない)。
        """
        self._assert_write("policy")
        old = self._writeback_store.get_fact(old_id)
        new = self._writeback_store.get_fact(new_id)
        if old is None or new is None:
            raise KeyError(f"policy fact not found: old={old_id!r} new={new_id!r}")
        if old.type != "policy" or new.type != "policy":
            raise ValueError(
                "supersede_old_policy requires both facts to be type='policy' "
                f"(old={old.type!r} new={new.type!r})",
            )
        self._writeback_store.supersede(old_id, new_id)

    # ──────────────────────────────────────────────────────────────────
    # 書込系 — fewshot
    # ──────────────────────────────────────────────────────────────────

    def write_fewshot(
        self,
        *,
        subject: str,
        object_: str,
        predicate: str = "example_for",
        scope: str = "global",
        confidence: float = 0.5,
        mode_origin: MemoryMode = "chat",
        trace_id: str | None = None,
        now: float | None = None,
    ) -> SemanticFact:
        """fewshot ファクトを書き込む (confidence は fitness 相当)。

        Raises:
            WriteOwnershipError: owner 違反。
            SubjectNamespaceError: ``subject`` が ``loop.*`` / ``mem.*`` prefix。
        """
        self._assert_write("fewshot")
        self._assert_subject_owner(subject)
        fact = make_fact(
            subject=subject,
            predicate=predicate,
            object_=object_,
            type="fewshot",
            scope=scope,
            confidence=confidence,
            mode_origin=mode_origin,
            now=now,
            trace_id=trace_id,
        )
        return self._writeback_store.add_fact(fact)

    # ──────────────────────────────────────────────────────────────────
    # 書込系 — learned_failure_pattern
    # ──────────────────────────────────────────────────────────────────

    def write_learned_failure_pattern(
        self,
        *,
        subject: str,
        predicate: str,
        object_: str,
        scope: str = "global",
        confidence: float = 0.5,
        mode_origin: MemoryMode = "chat",
        trace_id: str | None = None,
        now: float | None = None,
    ) -> SemanticFact:
        """``learned_failure_pattern`` ファクトを書き込む

        ``decision``/``outcome`` の集約から PolicyAdjuster が生成する失敗
        パターン用ファクト。Loop owned の ``failure_pattern`` (quality_gate
        由来) とは origin / namespace が物理的に分離される (CLAUDE.md §8)。

        ``subject`` は ``learn.failure_pattern.*`` namespace を要求する
        (検証は :meth:`_assert_subject_owner` 経由)。

        Args:
            subject: ``learn.failure_pattern.<decision_point>.<chosen>`` 形式。
            predicate: 関係性表現 (例: ``"indicates_failure"``)。
            object_: JSON 文字列の集約ペイロード (failure_count / total /
                avg_duration_ms / quality_signals 等)。
            confidence: 失敗率 (0.0〜1.0)。
            mode_origin: ``chat`` / ``coding``。
            trace_id: 観測トレース ID (任意)。
            now: テスト用クロック注入。

        Raises:
            WriteOwnershipError: caller pillar が ``"learn"`` でない。
            SubjectNamespaceError: ``subject`` prefix が ``learn.*`` でない。
        """
        self._assert_write("learned_failure_pattern")
        self._assert_subject_owner(subject)
        fact = make_fact(
            subject=subject,
            predicate=predicate,
            object_=object_,
            type="learned_failure_pattern",
            scope=scope,
            confidence=confidence,
            mode_origin=mode_origin,
            now=now,
            trace_id=trace_id,
        )
        return self._writeback_store.add_fact(fact)

    def find_active_learned_failure_pattern_by_subject(
        self,
        subject: str,
    ) -> SemanticFact | None:
        """同一 ``subject`` で active な最新 ``learned_failure_pattern`` を返す。

        supersession チェーンの末端 (``superseded_by`` が空) のうち
        ``created_at`` 最大のもの。見つからなければ ``None``。
        """
        self._assert_read("learned_failure_pattern")
        latest: SemanticFact | None = None
        try:
            facts = self._writeback_store.search_by_subject(subject)
        except (ValueError, AttributeError):
            return None
        for fact in facts:
            if fact.type != "learned_failure_pattern" or fact.superseded_by:
                continue
            if latest is None or fact.created_at > latest.created_at:
                latest = fact
        return latest

    def supersede_learned_failure_pattern(self, *, old_id: str, new_id: str) -> None:
        """``learned_failure_pattern`` の supersedes チェーンを構築する。

        ID 不在 / type 不整合等の軽微なエラーは握り潰す (PolicyAdjuster の
        flush ループを止めないため)。
        """
        self._assert_write("learned_failure_pattern")
        try:
            self._writeback_store.supersede(old_id, new_id)
        except (KeyError, ValueError):
            return

    # ──────────────────────────────────────────────────────────────────
    # 書込系 — failure_pattern
    # ──────────────────────────────────────────────────────────────────

    def merge_failure_patterns(self, *, old_id: str, new_id: str) -> None:
        """Learn 分析結果に基づく failure_pattern の merge

        ``failure_pattern`` の owner は EvorefLoop のため、Learn pillar が直接
        踏襲して定義されているが、:meth:`~FactViewBase._assert_write` が必ず
        :class:`~backend.free.memory.views.base.WriteOwnershipError` を送出する。

        実運用でのクラスタマージは :class:`~backend.free.memory.views.loop.LoopFactView`
        の :meth:`~backend.free.memory.views.loop.LoopFactView.consolidate_failure_patterns`
        を呼ぶこと
        :class:`~backend.free.loop.protocols.LoopWriteAPIProtocol` 経由の
        delegation 実装に差し替える余地を残す。
        """
        self._assert_write("failure_pattern")  # 必ず raise (Learn は owner でない)
        # ``_assert_write`` が raise するため以降は到達不可。
        raise NotImplementedError(  # pragma: no cover - unreachable
            "Learn pillar cannot write failure_pattern directly. Use "
            "LoopFactView.consolidate_failure_patterns from the Loop pillar.",
        )

    # ──────────────────────────────────────────────────────────────────
    # 汎用的読取 / policy / fewshot 操作ヘルパ
    # ──────────────────────────────────────────────────────────────────

    def list_facts_by_type(
        self,
        fact_type: str,
        *,
        limit: int | None = None,
        order: str = "created_at_desc",
    ) -> list[SemanticFact]:
        """読取権のある ``fact_type`` のファクトを全ストア横断で収集する。

        superseded ファクトは除外する。``order`` で並び順を指定
        (``created_at_desc`` / ``created_at_asc`` / ``accessed_at_desc``)。
        ``limit`` 指定時は先頭 N 件に切り詰める。

        Raises:
            PillarReadAccessError: Learn pillar が ``fact_type`` の reader で
                ない。
        """
        self._assert_read(fact_type)  # type: ignore[arg-type]
        seen: set[str] = set()
        out: list[SemanticFact] = []
        for store in self._stores:
            try:
                facts = store.search_by_type(fact_type)  # type: ignore[arg-type]
            except (ValueError, AttributeError):
                continue
            for fact in facts:
                if fact.id in seen or fact.superseded_by:
                    continue
                out.append(fact)
                seen.add(fact.id)
        if order == "created_at_desc":
            out.sort(key=lambda f: getattr(f, "created_at", 0.0), reverse=True)
        elif order == "created_at_asc":
            out.sort(key=lambda f: getattr(f, "created_at", 0.0))
        elif order == "accessed_at_desc":
            out.sort(key=lambda f: getattr(f, "accessed_at", 0.0), reverse=True)
        if limit is not None and limit >= 0:
            out = out[:limit]
        return out

    def count_facts_by_type(self, fact_type: str) -> int:
        """全ストアを横断して ``fact_type`` (superseded 除外) の件数を返す。"""
        self._assert_read(fact_type)  # type: ignore[arg-type]
        total = 0
        for store in self._stores:
            try:
                total += store.count_by_type(fact_type)  # type: ignore[arg-type]
            except (AttributeError, ValueError):
                continue
        return total

    def search_fewshot_by_prefix(
        self,
        prefix: str,
    ) -> list[SemanticFact]:
        """``prefix`` で始まる subject を持つ fewshot (または旧 policy) を全ストアから収集する。

        ``learn.fewshot.*`` prefix に移行済。``type="fewshot"``
        と legacy の ``type="policy"`` を両方受容する
        """
        self._assert_read("fewshot")
        self._assert_read("policy")
        seen: set[str] = set()
        out: list[SemanticFact] = []
        for store in self._stores:
            try:
                facts = store.search_by_pillar_prefix(prefix)
            except (ValueError, AttributeError):
                continue
            for fact in facts:
                if fact.id in seen or fact.superseded_by:
                    continue
                if fact.type not in ("fewshot", "policy"):
                    continue
                out.append(fact)
                seen.add(fact.id)
        return out

    def find_active_policy_by_subject(
        self,
        subject: str,
    ) -> SemanticFact | None:
        """writeback_store 内の同一 ``subject`` で active な最新 policy を返す。

        supersession チェーンの末端 (``superseded_by`` が空) のうち
        ``created_at`` 最大のものを返す。見つからなければ ``None``。
        """
        self._assert_read("policy")
        latest: SemanticFact | None = None
        try:
            facts = self._writeback_store.search_by_subject(subject)
        except (ValueError, AttributeError):
            return None
        for fact in facts:
            if fact.type != "policy" or fact.superseded_by:
                continue
            if latest is None or fact.created_at > latest.created_at:
                latest = fact
        return latest

    def add_policy_fact(self, fact: SemanticFact) -> SemanticFact:
        """既製の policy ``SemanticFact`` を writeback_store に書き込む。

        ``make_fact`` で組み立てた legacy ファクトを投入する経路。ownership
        と subject namespace は呼出側で :meth:`write_policy` 等を使うか、
        呼出前に整合性が保証されていること。本メソッドでは owner 違反のみ
        ガードする。

        Raises:
            WriteOwnershipError: ``fact.type`` が Learn pillar 所有でない。
            ValueError: 既存 ID と衝突 (ストア側 raise)。
        """
        self._assert_write(fact.type)  # type: ignore[arg-type]
        return self._writeback_store.add_fact(fact)

    def add_fewshot_fact(self, fact: SemanticFact) -> SemanticFact:
        """既製の fewshot ``SemanticFact`` を writeback_store に書き込む。

        Raises:
            WriteOwnershipError: ``fact.type`` が Learn pillar 所有でない。
        """
        self._assert_write(fact.type)  # type: ignore[arg-type]
        return self._writeback_store.add_fact(fact)

    def supersede_policy(self, *, old_id: str, new_id: str) -> None:
        """policy ファクトの supersedes チェーンを構築する (存在は呼出側保証)。

        :meth:`supersede_old_policy` との違い: 両 ID の存在や type 整合を
        呼出側で保証している前提で、内部で例外握り潰し (``KeyError`` /
        ``ValueError`` をログに出さず) を行う軽量バージョン。
        """
        self._assert_write("policy")
        try:
            self._writeback_store.supersede(old_id, new_id)
        except (KeyError, ValueError):
            # 既に supersede 済 / ID が存在しない等の軽微なエラーは握り潰す
            return

    @property
    def writeback_store(self) -> SemanticFactStoreProtocol:
        """**テスト / 低レベル検査用** の writeback_store 参照。

        本番コードは書込系メソッド (:meth:`write_policy` /
        :meth:`write_fewshot` / :meth:`add_policy_fact` 等) 経由で使うこと。
        """
        return self._writeback_store

    def iter_stores(self) -> list[SemanticFactStoreProtocol]:
        """**テスト / 低レベル検査用** の読取ストア列。"""
        return list(self._stores)


# ──────────────────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ──────────────────────────────────────────────────────────────────────────


__all__ = ["LearnFactView"]
