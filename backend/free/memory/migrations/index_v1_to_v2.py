"""SemMem 索引フォーマット v1 → v2 マイグレーション (M5-d)

旧 4 索引ファイル (`facts_by_subject.idx` / `facts_by_type.idx` /
`facts_by_pillar.idx` / `pinned.idx`) を **1 ファイルの fact_id 正規化形
JSONL** (`index.jsonl`) に統合する。

各 scope (`semantic/global` / `semantic/projects/<id>`) の `facts.jsonl` を
**真実のソース** として読み、各 fact から `(fact_id, subject, type, pillar,
pinned)` を抽出して新形式に書き出す。in-memory 索引は無変更で動く
(SemanticFactStore の検索 API は影響を受けない)。

設計判断:

- 旧 4 ファイルは ``SchemaMigrator._backup_step`` が ``backup_dir`` に退避済。
  本 ``migrate()`` 内で **削除する** (新形式 1 本に統一するため)。
  ``rollback()`` で backup から復元する。
- 新 ``index.jsonl`` の書き出しは :func:`backend.io.atomic_write_text` 経由
  (1 度に全 fact を書き出す)。
- 冪等性: 既存の ``index.jsonl`` があれば上書き (新形式生成の確実性優先)。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar, Iterator

from backend.free.memory.migrations.base import (
    ComponentKind,
    Migration,
    MigrationPlan,
    MigrationResult,
)
from backend.free.memory.semantic.store import (
    FACTS_FILENAME,
    INDEX_JSONL_FILENAME,
    PILLAR_IDX_FILENAME,
    PINNED_IDX_FILENAME,
    SUBJECT_IDX_FILENAME,
    TYPE_IDX_FILENAME,
    _fact_to_index_attrs,
    _serialize_index_entry,
)
from backend.free.memory.types import deserialize_fact_jsonl
from backend.io import AtomicWriter
from backend.log_config import get_logger

logger = get_logger("memory.migrations.index_v1_to_v2")

__all__ = ["IndexV1ToV2Migration"]


# 旧 4 索引ファイル名のタプル (backup / 削除 / rollback 復元の対象)
_OLD_INDEX_FILENAMES: tuple[str, ...] = (
    SUBJECT_IDX_FILENAME,
    TYPE_IDX_FILENAME,
    PILLAR_IDX_FILENAME,
    PINNED_IDX_FILENAME,
)


def _iter_scope_dirs(memory_dir: Path) -> Iterator[Path]:
    """``memory_dir/semantic/{global,projects/*}`` の各 scope ディレクトリを yield する。"""
    semantic_dir = memory_dir / "semantic"
    if not semantic_dir.exists():
        return
    global_dir = semantic_dir / "global"
    if global_dir.is_dir():
        yield global_dir
    projects_dir = semantic_dir / "projects"
    if projects_dir.is_dir():
        for child in sorted(projects_dir.iterdir()):
            if child.is_dir():
                yield child


class IndexV1ToV2Migration(Migration):
    """旧 4 索引ファイル形式 → 新 ``index.jsonl`` 形式への 1 度限り変換。"""

    from_version: ClassVar[int] = 1
    to_version: ClassVar[int] = 2
    component: ClassVar[ComponentKind] = "index"

    # ── dry_run ───────────────────────────────────────────────────────

    def dry_run(self, memory_dir: Path) -> MigrationPlan:
        """対象 scope の旧 4 索引と facts.jsonl を ``affected_files`` に列挙する。

        ``SchemaMigrator._backup_step`` がここで挙げたファイルを backup_dir に
        コピーする。``facts.jsonl`` も backup 対象に含めるのは、migration が
        ``deserialize_fact_jsonl`` で読込中にディスクが壊れる可能性に備えるため。
        """
        affected: list[Path] = []
        records = 0
        for scope_dir in _iter_scope_dirs(memory_dir):
            facts_path = scope_dir / FACTS_FILENAME
            if facts_path.exists():
                affected.append(facts_path)
                # fact 数の概算 (空行除外なし、line count)
                with facts_path.open("r", encoding="utf-8") as f:
                    records += sum(1 for _ in f)
            for old_name in _OLD_INDEX_FILENAMES:
                old_path = scope_dir / old_name
                if old_path.exists():
                    affected.append(old_path)
        return MigrationPlan(
            from_version=self.from_version,
            to_version=self.to_version,
            component=self.component,
            description=(
                "Consolidate 4-file SemMem indices "
                "(facts_by_subject/type/pillar.idx, pinned.idx) into a single "
                "fact_id-normalized index.jsonl"
            ),
            affected_files=tuple(affected),
            estimated_records=records,
        )

    # ── migrate ───────────────────────────────────────────────────────

    def migrate(self, memory_dir: Path) -> MigrationResult:
        """各 scope の旧 4 索引を新 ``index.jsonl`` に変換し、旧 4 ファイルを削除する。

        旧 4 ファイルは事前に ``SchemaMigrator._backup_step`` が backup 済の
        ため、本メソッドで削除しても rollback で復元可能。
        """
        total_records = 0
        scope_count = 0
        for scope_dir in _iter_scope_dirs(memory_dir):
            facts_path = scope_dir / FACTS_FILENAME
            if not facts_path.exists():
                # facts.jsonl が無い scope は skip (新規 scope や既に削除済)
                continue

            # facts.jsonl を読んで in-memory に fact 属性辞書を構築
            # last-write-wins on id (現状の SemanticFactStore._load() と同じ規約)
            facts_attrs: dict[str, dict] = {}
            with facts_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        fact = deserialize_fact_jsonl(line)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "Skipping malformed fact line in %s: %s",
                            facts_path, exc,
                        )
                        continue
                    facts_attrs[fact.id] = _fact_to_index_attrs(fact)

            # 新 index.jsonl を AtomicWriter で 1 度に書き出す
            index_path = scope_dir / INDEX_JSONL_FILENAME
            with AtomicWriter(index_path) as out:
                for fact_id in sorted(facts_attrs.keys()):
                    out.write(
                        _serialize_index_entry(fact_id, facts_attrs[fact_id]) + "\n",
                    )

            # 旧 4 ファイルを削除 (SchemaMigrator が backup 済のため安全)
            for old_name in _OLD_INDEX_FILENAMES:
                old_path = scope_dir / old_name
                if old_path.exists():
                    try:
                        old_path.unlink()
                    except OSError as exc:
                        logger.warning(
                            "Failed to remove old index file %s: %s",
                            old_path, exc,
                        )

            total_records += len(facts_attrs)
            scope_count += 1
            logger.info(
                "IndexV1ToV2Migration: scope=%s facts=%d -> index.jsonl",
                scope_dir.name, len(facts_attrs),
            )

        return MigrationResult(
            from_version=self.from_version,
            to_version=self.to_version,
            component=self.component,
            processed_records=total_records,
            details={"scopes_migrated": scope_count},
        )

    # ── rollback ──────────────────────────────────────────────────────

    def rollback(self, memory_dir: Path, backup_dir: Path) -> None:
        """``backup_dir`` から旧 4 ファイルを復元 + 新 ``index.jsonl`` を削除する。

        ``SchemaMigrator._backup_step`` が ``backup_dir`` に memory_dir 相対の
        ディレクトリツリーで旧 4 ファイル + facts.jsonl をコピーしているため、
        本メソッドはそのツリーから旧 4 ファイルのみを memory_dir に戻し
        (``facts.jsonl`` は migrate() で変更していないので復元不要)、新形式
        の ``index.jsonl`` を削除する。冪等。
        """
        # 1) 新 index.jsonl を削除
        for scope_dir in _iter_scope_dirs(memory_dir):
            index_path = scope_dir / INDEX_JSONL_FILENAME
            if index_path.exists():
                try:
                    index_path.unlink()
                except OSError as exc:
                    logger.warning(
                        "Rollback: failed to remove %s: %s", index_path, exc,
                    )

        # 2) backup_dir から旧 4 ファイルを復元
        if not backup_dir.exists():
            return
        for old_name in _OLD_INDEX_FILENAMES:
            for src in backup_dir.rglob(old_name):
                try:
                    rel = src.relative_to(backup_dir)
                except ValueError:
                    continue
                dest = memory_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
