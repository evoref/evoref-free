"""

:class:`SubjectCategoryRenameMigration` は、`SubjectKey.category` の
in-place 書換を行う Migration の具象雛形。``mem.coding_task.*`` を
``mem.coding_history.*`` に移行するようなケースを将来的に扱えるよう、
本 Issue では枠組みのみ提供する (登録済 Migration には追加しない)。

## スコープ

EvorefMem の semantic スコープ (``global`` / ``projects/<project_id>``) は
各々独立した ``facts.jsonl`` + 索引を持つ。本 Migration は全スコープの
``facts.jsonl`` を走査して対象 fact の subject を書換え、索引ファイルを
削除することで次回ロード時の再生成を促す。

## 設計原則

- **in-place rewrite**: 対象 ``facts.jsonl`` を tempfile + :func:`os.replace`
  でアトミックに差し替える
- **idempotent**: 既に new_category に移行済の fact には触れない
- **rollback**: ``SchemaMigrator`` が退避した backup dir 配下の
  ``facts.jsonl`` / ``.idx`` を復元するだけで状態を戻せる
- **索引は再生成**: 書換後に旧 ``facts_by_*.idx`` / ``pinned.idx`` (v1 索引)
  を削除する。統合 ``index.jsonl`` は次回の :class:`SemanticFactStore` 起動時に
  :meth:`SemanticFactStore._reconcile_index` が書換後の facts.jsonl と突合して
  subject/pillar 属性を自己修復するため、本 Migration では index.jsonl に触れない
  (索引を自前で差し替えるよりバグ面積が小さい)
- **predicate optional**: category 一致に追加して任意の fact 条件を
  掛けられる (例: ``lambda fact: fact.type == "coding_task"``)
- **from_version / to_version 可変**: 本 Migration は「カテゴリ rename」
  という抽象的なパターンを扱うため、実際の版番号ペアは呼び出し側が決める。
  ``Migration`` ABC の ClassVar を instance 属性でシャドーすることで
  :class:`SchemaMigrator` の validator を通過させる

## 本 Issue でのスコープ

SCHEMA_VERSION=1 を据え置き のため、DEFAULT_MIGRATIONS には登録
しない。実使用は将来の「category rename が必要な Issue」で行う。本 Issue
ではクラスとしての正しさ (dry_run / migrate / rollback が round-trip)
を :mod:`.tests.test_subject_migration` で検証する。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.free.memory.migrations.base import (
    Migration,
    MigrationPlan,
    MigrationResult,
)
from backend.free.memory.semantic.store import (
    FACTS_FILENAME,
    PILLAR_IDX_FILENAME,
    PINNED_IDX_FILENAME,
    SUBJECT_IDX_FILENAME,
    TYPE_IDX_FILENAME,
)
from backend.free.memory.semantic.subject_key import (
    SubjectKey,
    SubjectKeyError,
    SubjectPillar,
)
from backend.free.memory.types import (
    SemanticFact,
    deserialize_fact_jsonl,
    serialize_fact_jsonl,
)
from backend.io import AtomicWriter
from backend.log_config import get_logger

logger = get_logger("memory.semantic.subject_migration")


FactPredicate = Callable[[SemanticFact], bool]
"""Migration が対象 fact を追加絞り込みするための条件関数 (optional)。"""


# 削除対象の索引ファイル (いずれも :class:`SemanticFactStore` が
# ``_rebuild_indexes`` で再生成する)。
_INDEX_FILENAMES: tuple[str, ...] = (
    SUBJECT_IDX_FILENAME,
    TYPE_IDX_FILENAME,
    PILLAR_IDX_FILENAME,
    PINNED_IDX_FILENAME,
)


@dataclass
class _ScopeRewriteStats:
    """1 スコープ分の rewrite サマリ (テスト検証用)。"""

    scope_dir: Path
    matched: int = 0
    total: int = 0


class SubjectCategoryRenameMigration(Migration):
    """``pillar.old_category.*`` → ``pillar.new_category.*`` の in-place 書換。

    Args:
        pillar: 対象 pillar (``"loop"`` / ``"learn"`` / ``"mem"``)
        old_category: 置換前の 2 番目セグメント
        new_category: 置換後の 2 番目セグメント
        from_version / to_version: この Migration が繋ぐ版番号
            (``SchemaMigrator`` の内部表現のため instance で保持)
        predicate: 追加条件関数 (``fact -> bool``)。省略時は subject の
            category 一致のみでフィルタする

    Notes:
        ``Migration`` ABC の ``from_version`` / ``to_version`` / ``component``
        は ClassVar だが、本クラスでは **instance 属性でシャドー** して
        呼び出し側が版番号を決定できるようにする。
        :meth:`SchemaMigrator._validate_registry` はインスタンス属性
        ``m.from_version`` を参照するため、この運用で validator を通過する。
    """

    # ClassVar 要件を満たすためのダミー値 (__init__ で instance 属性として
    # 上書きする)。SchemaMigrator._validate_registry は instance 属性を
    # 参照するため、同一値で登録される悪用を避けるためにも未使用値。
    from_version = 0
    to_version = 0
    component = "subject_ns"

    def __init__(
        self,
        *,
        pillar: SubjectPillar,
        old_category: str,
        new_category: str,
        from_version: int,
        to_version: int,
        predicate: FactPredicate | None = None,
    ) -> None:
        if pillar not in {"loop", "learn", "mem"}:
            raise ValueError(f"unknown pillar: {pillar!r}")
        if not old_category or "." in old_category:
            raise ValueError(f"invalid old_category: {old_category!r}")
        if not new_category or "." in new_category:
            raise ValueError(f"invalid new_category: {new_category!r}")
        if old_category == new_category:
            raise ValueError("old_category and new_category must differ")
        if from_version >= to_version:
            raise ValueError(
                f"from_version {from_version} must be < to_version {to_version}",
            )
        # instance 属性で ClassVar をシャドー (dataclass ではないので明示的に)
        self.pillar: SubjectPillar = pillar
        self.old_category: str = old_category
        self.new_category: str = new_category
        self.from_version = from_version
        self.to_version = to_version
        self.component = "subject_ns"
        self._predicate: FactPredicate | None = predicate

    # ──────────────────────────────────────────────────────────────
    # Migration ABC 実装
    # ──────────────────────────────────────────────────────────────

    def dry_run(self, memory_dir: Path) -> MigrationPlan:
        """対象となる ``facts.jsonl`` / 索引ファイルを列挙する (副作用なし)。"""
        scope_dirs = list(_iter_scope_dirs(memory_dir))
        affected: list[Path] = []
        estimated = 0
        for scope_dir in scope_dirs:
            facts_path = scope_dir / FACTS_FILENAME
            if facts_path.exists():
                affected.append(facts_path)
                estimated += self._count_matching(facts_path)
            for idx_name in _INDEX_FILENAMES:
                idx_path = scope_dir / idx_name
                if idx_path.exists():
                    affected.append(idx_path)
        return MigrationPlan(
            from_version=self.from_version,
            to_version=self.to_version,
            component=self.component,
            description=(
                f"rename subject category {self.pillar}.{self.old_category}.* "
                f"-> {self.pillar}.{self.new_category}.* "
                f"(scopes={len(scope_dirs)})"
            ),
            affected_files=tuple(affected),
            estimated_records=estimated,
        )

    def migrate(self, memory_dir: Path) -> MigrationResult:
        """各スコープの ``facts.jsonl`` を atomic rewrite する。"""
        processed = 0
        details: dict[str, Any] = {"per_scope": []}
        for scope_dir in _iter_scope_dirs(memory_dir):
            stats = self._rewrite_scope(scope_dir)
            processed += stats.matched
            details["per_scope"].append({
                "scope_dir": str(scope_dir),
                "matched": stats.matched,
                "total": stats.total,
            })
            # 書換が発生したスコープでは索引を削除して再生成を促す
            # (matched==0 のスコープは索引を残して I/O を節約する)
            if stats.matched > 0:
                for idx_name in _INDEX_FILENAMES:
                    idx_path = scope_dir / idx_name
                    if idx_path.exists():
                        idx_path.unlink()
        logger.info(
            "SubjectCategoryRenameMigration: %s -> %s, processed=%d",
            f"{self.pillar}.{self.old_category}.*",
            f"{self.pillar}.{self.new_category}.*",
            processed,
        )
        return MigrationResult(
            from_version=self.from_version,
            to_version=self.to_version,
            component=self.component,
            processed_records=processed,
            details=details,
        )

    def rollback(self, memory_dir: Path, backup_dir: Path) -> None:
        """``SchemaMigrator`` が退避した backup_dir の内容で memory_dir を復元する。

        :class:`SchemaMigrator._backup_step` は memory_dir からの相対パスを
        保持したまま affected_files をコピーしているため、同じ相対パスで
        書き戻せば rollback が完了する。
        """
        if not backup_dir.exists():
            logger.warning(
                "SubjectCategoryRenameMigration rollback: backup_dir missing: %s",
                backup_dir,
            )
            return
        for src in backup_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(backup_dir)
            dest = memory_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    # ──────────────────────────────────────────────────────────────
    # 内部ヘルパ
    # ──────────────────────────────────────────────────────────────

    def _count_matching(self, facts_path: Path) -> int:
        """``facts_path`` の中で本 Migration が書換対象にする fact 数を数える。"""
        count = 0
        with facts_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    fact = deserialize_fact_jsonl(line)
                except (ValueError, KeyError):
                    continue
                if self._should_rename(fact):
                    count += 1
        return count

    def _rewrite_scope(self, scope_dir: Path) -> _ScopeRewriteStats:
        """1 スコープ分の ``facts.jsonl`` を atomic rewrite する。"""
        facts_path = scope_dir / FACTS_FILENAME
        stats = _ScopeRewriteStats(scope_dir=scope_dir)
        if not facts_path.exists():
            return stats

        # AtomicWriter が tmp 書込 → os.replace (Windows retry 付き) を担い、
        # 途中例外時は tmp を unlink して facts_path を不変のまま残す。
        with AtomicWriter(facts_path) as dst, \
             facts_path.open("r", encoding="utf-8") as src:
            for raw in src:
                line = raw.strip()
                if not line:
                    continue
                stats.total += 1
                try:
                    fact = deserialize_fact_jsonl(line)
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "SubjectCategoryRenameMigration: skip malformed "
                        "fact line in %s: %s", facts_path, exc,
                    )
                    # 破損行は保持しない (writer 出力から除外)
                    continue
                if self._should_rename(fact):
                    new_subject = self._rename_subject(fact.subject)
                    fact.subject = new_subject
                    stats.matched += 1
                dst.write(serialize_fact_jsonl(fact) + "\n")
        return stats

    def _should_rename(self, fact: SemanticFact) -> bool:
        """fact.subject が対象 (pillar + old_category) に該当するか。"""
        key = SubjectKey.try_parse(fact.subject)
        if key is None:
            return False
        if key.pillar != self.pillar or key.category != self.old_category:
            return False
        if self._predicate is not None and not self._predicate(fact):
            return False
        return True

    def _rename_subject(self, subject: str) -> str:
        """old → new の subject 文字列を返す。parse 失敗時は原文を返す。"""
        try:
            key = SubjectKey.parse(subject)
        except SubjectKeyError:
            return subject
        return key.with_category(self.new_category).canonical()


def _iter_scope_dirs(memory_dir: Path) -> list[Path]:
    """``<memory_dir>/semantic/`` 配下の scope ディレクトリを列挙する。

    対象: ``global/`` と ``projects/<id>/`` のそれぞれ (facts.jsonl を持ち得る)。
    存在しない場合は空リスト。
    """
    semantic_root = memory_dir / "semantic"
    if not semantic_root.exists():
        return []
    scopes: list[Path] = []
    global_dir = semantic_root / "global"
    if global_dir.is_dir():
        scopes.append(global_dir)
    projects_root = semantic_root / "projects"
    if projects_root.is_dir():
        for child in sorted(projects_root.iterdir()):
            if child.is_dir():
                scopes.append(child)
    return scopes


def write_jsonl_line(path: Path, payload: dict[str, Any]) -> None:
    """テスト / 外部ツール向けの 1 行 JSONL 追記ヘルパ。

    Migration 本体は使わないが、test fixture から書き込む際の利便性
    のために export する。本モジュールの外部利用者が素の
    ``json.dumps`` を書く必要がなくなる。
    """
    line = json.dumps(payload, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


__all__ = [
    "FactPredicate",
    "SubjectCategoryRenameMigration",
    "write_jsonl_line",
]
