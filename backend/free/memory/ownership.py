"""

4 pillar アーキテクチャ における各 :class:`~backend.free.memory.types.FactType`
の **所有 pillar (owner)** と **読取権を持つ pillar (readers)** を宣言する。
実行時に owner 以外が書込を試みた場合は :class:`PillarOwnershipError` を
送出し、pillar 境界違反を検出する。

## 設計方針 (CLAUDE.md §8 / docs/f_02_memory_system.md §7.2 / docs/c_05_data_schemas.md §21.2)

- **owner** は書込権を持つ唯一の pillar。Fact View 層
  (`MemFactView` / `LoopFactView` / `LearnFactView`) は書込前に
  :func:`assert_owner` を呼び、違反時は :class:`PillarOwnershipError` を
  raise する
- **readers** は読取を許可する pillar 集合。``HarnessFactView`` のように
  「読取専用」の pillar も読者として宣言する。
- pillar 識別子は ``"mem"`` / ``"loop"`` / ``"learn"`` / ``"harness"`` の 4 値。
  ``"harness"`` は EvorefLoop 配下のサブレイヤだが、Fact View 層では read-only
  な独立 reader として扱うため個別 pillar として宣言する。

## 導入範囲

本モジュールは **定数と検証ヘルパを定義するのみ** で、既存コードへの組込は
Fact View 層で行う。既存の ``SemanticFactStore.add_fact``
等には直接挿入しない

## FactType 網羅性

:data:`FACT_OWNERSHIP` は :data:`~backend.free.memory.types.FactType` の全
リテラル値を網羅する。新規 FactType を追加した場合は本定数への登録も必須で、
:func:`assert_fact_ownership_complete` がテスト側から呼ばれて検出する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

from backend.free.memory.types import FactType

# ──────────────────────────────────────────────────────────────────────────
# pillar 識別子
# ──────────────────────────────────────────────────────────────────────────

Pillar = Literal["mem", "loop", "learn", "harness"]
"""4 pillar アーキテクチャにおける pillar 識別子。

``"harness"`` は EvorefLoop 配下のサブレイヤだが、Fact View では read-only な
独立 reader として機能するため、pillar 識別子として個別に扱う。
書込権 (owner) を持つのは ``"mem"`` / ``"loop"`` / ``"learn"`` の 3 値のみ。
"""

ALL_PILLARS: frozenset[Pillar] = frozenset(get_args(Pillar))
"""全 pillar 識別子の集合。"""

WRITABLE_PILLARS: frozenset[Pillar] = frozenset({"mem", "loop", "learn"})
"""書込権を持ちうる pillar の集合 (``"harness"`` は含まない)。"""


# ──────────────────────────────────────────────────────────────────────────
# 例外
# ──────────────────────────────────────────────────────────────────────────


class PillarOwnershipError(RuntimeError):
    """owner 以外の pillar が書込を試みた際に送出される。

    Fact View 層および ``SemanticFactStore.save`` フックで
    :func:`assert_owner` が違反を検出して raise する。

    Attributes:
        caller_pillar: 書込を試みた呼び出し元 pillar。
        fact_type: 対象の :class:`~backend.free.memory.types.FactType`。
        owner: 正しい owner pillar。
    """

    def __init__(
        self, caller_pillar: str, fact_type: str, owner: str,
    ) -> None:
        self.caller_pillar = caller_pillar
        self.fact_type = fact_type
        self.owner = owner
        super().__init__(
            f"pillar ownership violation: pillar={caller_pillar!r} "
            f"cannot write fact_type={fact_type!r} (owner={owner!r})",
        )


# ──────────────────────────────────────────────────────────────────────────
# FactOwnership データクラス
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FactOwnership:
    """1 FactType の所有権情報。

    Attributes:
        owner: 書込を許可される唯一の pillar (``"mem"`` / ``"loop"`` /
            ``"learn"`` のいずれか)。
        readers: 読取を許可される pillar の不変集合。owner を含む
            (owner は読める前提) か否かは実装の裁量だが、本定数では
            owner を含む形で統一宣言する (明示性のため)。
    """

    owner: Pillar
    readers: frozenset[Pillar] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # owner は writable pillar に制限
        if self.owner not in WRITABLE_PILLARS:
            raise ValueError(
                f"FactOwnership.owner must be in {sorted(WRITABLE_PILLARS)}: "
                f"got {self.owner!r}",
            )
        # readers は全 pillar 集合に制限
        invalid = self.readers - ALL_PILLARS
        if invalid:
            raise ValueError(
                f"FactOwnership.readers contains invalid pillars: {sorted(invalid)}",
            )

    def can_read(self, pillar: Pillar) -> bool:
        """``pillar`` が読取権を持つか。"""
        return pillar in self.readers

    def can_write(self, pillar: Pillar) -> bool:
        """``pillar`` が書込権を持つか (owner と一致するか)。"""
        return pillar == self.owner


# ──────────────────────────────────────────────────────────────────────────
# FACT_OWNERSHIP 定数
# ──────────────────────────────────────────────────────────────────────────


def _ow(owner: Pillar, readers: set[Pillar]) -> FactOwnership:
    """定義読みやすさのための内部ヘルパ。"""
    return FactOwnership(owner=owner, readers=frozenset(readers))


FACT_OWNERSHIP: dict[FactType, FactOwnership] = {
    # ── EvorefMem owned ──────────────────────────────────────────────────
    # personal_fact / preference の readers に "loop" を追加
    # agent/self_cartridge の @self 仮想カートリッジが LoopFactView 経由で
    # ユーザープロファイルを読み取るため (EvorefLoop pillar からの読取を許可)。
    "personal_fact":   _ow("mem",   {"mem", "loop"}),
    # readers に "loop" を追加: ToolCallJudge (Loop pillar) が
    # ``mem.world.url.*`` を URL リコール (Phase 1) で読み取るため。
    "world_fact":      _ow("mem",   {"mem", "loop"}),
    "preference":      _ow("mem",   {"mem", "loop"}),
    "emotion":         _ow("mem",   {"mem"}),
    "opinion":         _ow("mem",   {"mem"}),
    "belief":          _ow("mem",   {"mem"}),
    "decision":        _ow("mem",   {"mem", "loop"}),
    "commitment":      _ow("mem",   {"mem", "loop"}),
    "project":         _ow("mem",   {"mem", "loop", "learn"}),
    "create":          _ow("mem",   {"mem", "loop", "learn"}),
    "create_task":     _ow("mem",   {"mem", "loop", "learn"}),  # D4: CreateExtractor 由来
    "model":           _ow("mem",   {"mem"}),  # 将来対応

    # ── EvorefLoop owned ─────────────────────────────────────────────────
    "task":            _ow("loop",  {"loop", "learn"}),
    "progress_marker": _ow("loop",  {"loop", "learn"}),
    "failure_pattern": _ow("loop",  {"loop", "learn", "harness"}),
    "artifact":        _ow("loop",  {"loop", "learn", "mem"}),

    # ── EvorefLearn owned ────────────────────────────────────────────────
    "policy":          _ow("learn", {"loop", "learn", "harness"}),
    "fewshot":         _ow("learn", {"loop", "learn", "harness"}),
    # PolicyAdjuster 由来の集約失敗パターン。Loop 所有の
    # `failure_pattern` (quality_gate 由来) とは origin / namespace を物理的に
    # 分離し、二重書き込みを回避する設計 (CLAUDE.md §8 / docs/f_02_memory_system.md §7)。
    "learned_failure_pattern": _ow("learn", {"loop", "learn", "harness"}),
}
"""全 :class:`FactType` についての owner / readers 宣言。

11 mem 所有 + 4 loop 所有 + 3 learn 所有 + 1 将来対応 ``model``)。新規 FactType
を追加した場合は本定数にも対応するエントリを追加すること
(``assert_fact_ownership_complete`` が網羅性違反をテストで検出する)。
"""


# ──────────────────────────────────────────────────────────────────────────
# 検証ヘルパ
# ──────────────────────────────────────────────────────────────────────────


def get_owner(fact_type: FactType) -> Pillar:
    """``fact_type`` の owner pillar を返す。

    Raises:
        KeyError: 未登録の :class:`FactType` (構成ミス検出)。
    """
    return FACT_OWNERSHIP[fact_type].owner


def get_readers(fact_type: FactType) -> frozenset[Pillar]:
    """``fact_type`` の readers 集合を返す (不変)。"""
    return FACT_OWNERSHIP[fact_type].readers


def can_read(caller_pillar: Pillar, fact_type: FactType) -> bool:
    """``caller_pillar`` が ``fact_type`` を読めるか。"""
    return FACT_OWNERSHIP[fact_type].can_read(caller_pillar)


def can_write(caller_pillar: Pillar, fact_type: FactType) -> bool:
    """``caller_pillar`` が ``fact_type`` を書込めるか。"""
    return FACT_OWNERSHIP[fact_type].can_write(caller_pillar)


def assert_owner(caller_pillar: Pillar, fact_type: FactType) -> None:
    """``caller_pillar`` が ``fact_type`` の owner でなければ raise。

    、Fact View 層の書込前チェックとして呼ばれる
    テスト用途および将来組込のための API として公開するのみ。

    Raises:
        PillarOwnershipError: ``caller_pillar`` が owner と一致しない。
        KeyError: ``fact_type`` が :data:`FACT_OWNERSHIP` に未登録。
    """
    ownership = FACT_OWNERSHIP[fact_type]
    if caller_pillar != ownership.owner:
        raise PillarOwnershipError(
            caller_pillar=caller_pillar,
            fact_type=fact_type,
            owner=ownership.owner,
        )


def assert_fact_ownership_complete() -> None:
    """全 :data:`FactType` が :data:`FACT_OWNERSHIP` に登録されているかを検証する。

    テスト側から呼ばれる想定。新規 FactType 追加時の登録漏れを検出する。

    Raises:
        AssertionError: 未登録 / 余分な FactType が存在する。
    """
    declared = set(get_args(FactType))
    registered = set(FACT_OWNERSHIP.keys())
    missing = declared - registered
    extra = registered - declared
    if missing or extra:
        raise AssertionError(
            f"FACT_OWNERSHIP coverage mismatch: missing={sorted(missing)} "
            f"extra={sorted(extra)}",
        )


__all__ = [
    "ALL_PILLARS",
    "FACT_OWNERSHIP",
    "FactOwnership",
    "Pillar",
    "PillarOwnershipError",
    "WRITABLE_PILLARS",
    "assert_fact_ownership_complete",
    "assert_owner",
    "can_read",
    "can_write",
    "get_owner",
    "get_readers",
]
