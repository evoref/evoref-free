"""

4 pillar アーキテクチャ における Fact View 層の基底クラスと
共通例外を定義する。各 View (:class:`MemFactView` / :class:`LoopFactView` /
:class:`LearnFactView` / :class:`HarnessFactView`) は本基底を継承し、
ownership / read access / subject namespace の強制ロジックを共有する。

## 設計方針 (CLAUDE.md §8 / docs/f_02_memory_system.md §10)

- **書込は owner のみ**: 書込前に :func:`assert_owner` を呼び、違反時は
  :class:`WriteOwnershipError` (alias: :class:`PillarOwnershipError`) を送出する。
- **読取は readers のみ**: 読取前に :func:`can_read` を確認し、違反時は
  :class:`PillarReadAccessError` を送出する。
- **subject namespace 強制**: 書込時に subject が属する pillar namespace が
  View pillar と矛盾する場合は :class:`SubjectNamespaceError` を送出する。
  自然文 subject (pillar prefix を持たないもの) は passthrough される
- **stateless**: View インスタンスは store への薄いラッパ。fixture/mock しやすい。


Issue 本文では書込違反の例外名が ``WriteOwnershipError`` と記載されているが、
導入済みの :class:`PillarOwnershipError` (backend.free.memory.ownership)
と意味的に同一のため、本モジュールでは **エイリアスを公開** して両方の
名前で参照可能にする。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from backend.free.memory.ownership import (
    Pillar,
    PillarOwnershipError,
    assert_owner,
    can_read,
    get_readers,
)
from backend.free.memory.notes.subject_ns import (
    SubjectNamespaceError,
    SubjectPillar,
    validate_subject_namespace,
)
from backend.free.memory.types import FactType

if TYPE_CHECKING:
    from backend.free.memory.protocols import SemanticFactStoreProtocol
    from backend.free.memory.types import MemoryMode, SemanticFact

# ──────────────────────────────────────────────────────────────────────────
# 例外エイリアス
# ──────────────────────────────────────────────────────────────────────────

WriteOwnershipError = PillarOwnershipError


# ──────────────────────────────────────────────────────────────────────────
# 読取違反例外
# ──────────────────────────────────────────────────────────────────────────


class PillarReadAccessError(RuntimeError):
    """reader でない pillar が読取を試みた際に送出される。

    Fact View 層の :meth:`FactViewBase._assert_read` が
    :data:`~backend.free.memory.ownership.FACT_OWNERSHIP` の ``readers`` を
    検証して raise する。

    Attributes:
        caller_pillar: 読取を試みた呼び出し元 pillar。
        fact_type: 対象の :class:`~backend.free.memory.types.FactType`。
        readers: 読取を許可された pillar の集合。
    """

    def __init__(
        self,
        caller_pillar: str,
        fact_type: str,
        readers: frozenset[str],
    ) -> None:
        self.caller_pillar = caller_pillar
        self.fact_type = fact_type
        self.readers = readers
        super().__init__(
            f"pillar read access violation: pillar={caller_pillar!r} "
            f"cannot read fact_type={fact_type!r} "
            f"(readers={sorted(readers)!r})",
        )


# ──────────────────────────────────────────────────────────────────────────
# Fact View 基底
# ──────────────────────────────────────────────────────────────────────────


class FactViewBase:
    """全 Fact View の基底クラス。

    継承する View は ``pillar`` クラス属性で自身の pillar 識別子を固定する
    (``"mem"`` / ``"loop"`` / ``"learn"`` / ``"harness"`` のいずれか)。
    書込/読取時のチェックヘルパを共通化し、ownership / namespace 違反を
    統一的に送出する。

    本クラス自身は store 参照を持たず、サブクラスが自由な形 (単一 store /
    store リスト / writeback_store 分離) を選択できる。
    """

    pillar: Pillar
    """View の pillar 識別子 (サブクラスで上書き)。"""

    # ── ownership 検証 ────────────────────────────────────────────

    def _assert_write(self, fact_type: FactType) -> None:
        """``fact_type`` が本 View pillar の owner かを検証する。

        Raises:
            WriteOwnershipError: ``fact_type`` の owner が本 View pillar でない。
        """
        assert_owner(self.pillar, fact_type)

    def _assert_read(self, fact_type: FactType) -> None:
        """``fact_type`` を本 View pillar が読取可能かを検証する。

        Raises:
            PillarReadAccessError: ``fact_type`` の readers に本 View pillar が
                含まれない。
        """
        if not can_read(self.pillar, fact_type):
            raise PillarReadAccessError(
                caller_pillar=self.pillar,
                fact_type=fact_type,
                readers=get_readers(fact_type),
            )

    # ── subject namespace 検証 ────────────────────────────────────

    def _assert_subject_owner(self, subject: str) -> None:
        """``subject`` の pillar prefix が View pillar の書込範囲と整合するかを検証する。

        ``subject`` が ``loop.*`` / ``learn.*`` / ``mem.*`` のいずれかの
        prefix を持ち、かつ View pillar (``mem`` / ``loop`` / ``learn``) と
        一致しない場合 :class:`SubjectNamespaceError` を送出する。自然文 subject
        (pillar prefix を持たないもの) は passthrough する (旧 ``harness.*``
        prefix は全廃済)。

        Raises:
            SubjectNamespaceError: ``subject`` prefix と View pillar が矛盾する。
        """
        detected: SubjectPillar | None = validate_subject_namespace(subject)
        if detected is None:
            return  # 自然文 subject は passthrough
        # View pillar が "mem"/"loop"/"learn" のいずれかである書込 View のみ
        # subject 所有検証を適用する (HarnessFactView は書込を提供しない)。
        if self.pillar not in ("mem", "loop", "learn"):
            return
        if detected != self.pillar:
            raise SubjectNamespaceError(
                f"pillar {self.pillar!r} view cannot write subject with "
                f"{detected!r} namespace: {subject!r}",
            )


# ──────────────────────────────────────────────────────────────────────────
# View 共通ユーティリティ
# ──────────────────────────────────────────────────────────────────────────


def safe_json_loads(s: str) -> dict[str, Any]:
    """JSON 文字列を dict にパースする。dict 以外 / パース失敗時は空 dict。

    Fact の payload (JSON) を読む際の共通ヘルパ。loop / learn View で
    AST 一致していた ``_safe_json_loads`` を集約したもの。
    """
    try:
        payload = json.loads(s)
        if isinstance(payload, dict):
            return payload
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}


def merge_active_facts_across_stores(
    stores: Iterable[SemanticFactStoreProtocol],
    fact_type: FactType,
    *,
    min_confidence: float = 0.0,
    mode: MemoryMode | None = None,
) -> list[SemanticFact]:
    """複数 store を横断して有効な ``fact_type`` を集約する (superseded / 重複除外)。

    ``get_active_policies`` / ``get_active_fewshots`` 系で harness / loop / learn
    View に AST 一致で重複していた走査ロジックを集約する。

    Args:
        stores: 走査対象の store 群 (global + project スコープ等)。
        fact_type: 対象 FactType ("policy" / "fewshot" 等)。
        min_confidence: この confidence 未満の fact を除外する閾値。
        mode: 指定時は ``mode_origin`` が一致する fact のみ採用する。

    Returns:
        store 走査順・初出順を保った有効 fact のリスト (id 重複は最初の 1 件)。
    """
    result: list[SemanticFact] = []
    seen: set[str] = set()
    for store in stores:
        for fact in store.search_by_type(fact_type):
            if fact.id in seen or fact.superseded_by:
                continue
            if fact.confidence < min_confidence:
                continue
            if mode is not None and fact.mode_origin != mode:
                continue
            result.append(fact)
            seen.add(fact.id)
    return result


__all__ = [
    "FactViewBase",
    "PillarReadAccessError",
    "SubjectNamespaceError",
    "WriteOwnershipError",
    "merge_active_facts_across_stores",
    "safe_json_loads",
]
