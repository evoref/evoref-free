"""SchemaMigrator — EvorefMem schema migration のランナー

登録された `Migration` 群から (from_version, to_version) の連鎖を解決し、
1. 対象ファイルを `migration_archive/migration_<timestamp>/step_<n>_<component>/` へ退避
2. `Migration.migrate()` を順次呼出
3. 失敗時は退避済 backup を逆順で復元して例外を再送出

という責務を担う。`upgrade()` は idempotent で、既に target 版に到達している
場合は no-op で空リストを返す。

## 置き所

`backend.factory._memory_init._init_memory` の判定ロジックでは以下の順で呼ばれる:

1. **`SchemaMigrator.upgrade()`** — in-place rewrite
2. `needs_initialization` → `initialize_evorefmem` — destructive fallback

v1 のみの現状では登録 Migration は 0 件 (空チェインは no-op)。将来 v2 以降を
導入する際に Migration を追加するだけで連鎖実行が有効化される。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from backend.free.memory.init_evorefmem import write_schema_marker
from backend.log_config import get_logger

from .base import Migration, MigrationPlan, MigrationResult

logger = get_logger("memory.migrations.migrator")

_BACKUP_ROOT_PREFIX: Final[str] = "migration_"


class MigrationError(RuntimeError):
    """Migration 実行中の汎用エラー."""


class MigrationChainNotFoundError(MigrationError):
    """要求された (from, to) を繋ぐ Migration が見つからない."""


@dataclass(frozen=True)
class _Step:
    """内部用: 連鎖内の 1 step."""

    migration: Migration
    backup_dir: Path


def _timestamp(now: float | None = None) -> str:
    """UTC タイムスタンプ (`20261231T235959Z` 形式)."""
    if now is None:
        now = time.time()
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))


class SchemaMigrator:
    """Migration を連鎖実行するランナー.

    Parameters
    ----------
    migrations:
        登録対象の Migration インスタンス一覧。同じ (from, to, component)
        の重複があれば即座に ValueError を送出する。
    migration_archive_dir:
        各 step 開始前にファイルを退避する先の親ディレクトリ。存在しない
        場合は `upgrade()` 実行時に自動作成される。
    """

    def __init__(
        self,
        migrations: list[Migration] | None = None,
        *,
        migration_archive_dir: Path,
    ) -> None:
        self._migration_archive_dir = migration_archive_dir
        self._migrations: list[Migration] = list(migrations or [])
        self._validate_registry()

    # --------------------------------------------------------------
    # 登録・プラン
    # --------------------------------------------------------------

    def _validate_registry(self) -> None:
        seen: set[tuple[int, int, str]] = set()
        for m in self._migrations:
            key = (m.from_version, m.to_version, m.component)
            if key in seen:
                raise ValueError(
                    f"Duplicate migration registered: {key}",
                )
            seen.add(key)
            if m.from_version >= m.to_version:
                raise ValueError(
                    f"Migration {type(m).__name__} has non-increasing versions: "
                    f"{m.from_version} -> {m.to_version}",
                )

    def plan(
        self,
        memory_dir: Path,
        from_version: int,
        to_version: int,
    ) -> list[MigrationPlan]:
        """`upgrade` 実行前の計画を副作用なしで返す.

        同じ `from_version == to_version` の場合は空リスト。連鎖が解決
        できなければ `MigrationChainNotFoundError` を送出する。
        """
        if from_version == to_version:
            return []
        chain = self._resolve_chain(from_version, to_version)
        return [m.dry_run(memory_dir) for m in chain]

    # --------------------------------------------------------------
    # 実行
    # --------------------------------------------------------------

    def upgrade(
        self,
        memory_dir: Path,
        from_version: int,
        to_version: int,
        *,
        dry_run: bool = False,
        now: float | None = None,
    ) -> list[MigrationResult] | list[MigrationPlan]:
        """from_version → to_version へ連鎖実行する.

        - `dry_run=True` の場合は `plan()` と同等 (`MigrationPlan` のリスト)
        - `from_version == to_version` は no-op (空リスト)
        - 連鎖解決に失敗すると `MigrationChainNotFoundError`
        - migrate 中に例外が発生した場合、成功済 step を backup から逆順で
          復元した上で元の例外を再送出する
        """
        if from_version == to_version:
            logger.debug(
                "SchemaMigrator.upgrade no-op: already at version %d",
                to_version,
            )
            return [] if not dry_run else []

        chain = self._resolve_chain(from_version, to_version)

        if dry_run:
            return [m.dry_run(memory_dir) for m in chain]

        backup_root = self._prepare_backup_root(now=now)
        steps: list[_Step] = []
        results: list[MigrationResult] = []

        t_start = time.monotonic()
        try:
            for idx, migration in enumerate(chain):
                step_backup = backup_root / (
                    f"step_{idx:02d}_"
                    f"{migration.from_version}_to_{migration.to_version}_"
                    f"{migration.component}"
                )
                self._backup_step(memory_dir, migration, step_backup)
                steps.append(_Step(migration=migration, backup_dir=step_backup))

                step_t0 = time.monotonic()
                result = migration.migrate(memory_dir)
                result.backup_dir = step_backup
                if result.elapsed_ms == 0.0:
                    result.elapsed_ms = (time.monotonic() - step_t0) * 1000.0
                results.append(result)
                logger.info(
                    "Migration step %d completed: %d -> %d (%s), "
                    "records=%d elapsed=%.1fms",
                    idx,
                    migration.from_version,
                    migration.to_version,
                    migration.component,
                    result.processed_records,
                    result.elapsed_ms,
                )
        except Exception as exc:
            logger.exception(
                "Migration chain failed at step %d; rolling back %d completed step(s)",
                len(results),
                len(steps),
            )
            self._rollback(memory_dir, steps)
            raise MigrationError(
                f"Migration {from_version} -> {to_version} failed: {exc}",
            ) from exc

        # 全 step 成功時のみ SCHEMA_VERSION マーカーを target 版へ書き換える。
        # Migration 側は純粋に「データ変換」の責務だけを負い、マーカー更新は
        # ランナーが集中管理する。rollback 時は書き換えていないため戻す必要はない。
        write_schema_marker(memory_dir, to_version)

        # manifest の監査タイムスタンプ last_migrated_at を更新する
        # (manifest が存在する場合のみ。compact_cmd の last_compacted_at と同様)。
        from backend.free.memory.semantic.manifest import (
            load_manifest,
            update_manifest,
        )

        if load_manifest(memory_dir) is not None:
            update_manifest(
                memory_dir,
                last_migrated_at=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
                ),
            )

        logger.info(
            "SchemaMigrator.upgrade completed: %d -> %d, "
            "steps=%d total_elapsed=%.1fms",
            from_version,
            to_version,
            len(results),
            (time.monotonic() - t_start) * 1000.0,
        )
        return results

    # --------------------------------------------------------------
    # 内部ヘルパ
    # --------------------------------------------------------------

    def _resolve_chain(
        self,
        from_version: int,
        to_version: int,
    ) -> list[Migration]:
        """`from_version` から `to_version` に至る Migration の並びを解決する.

        単純な直列チェインのみサポート (version v に対して (v -> v+1) を
        繋ぐ Migration が 1 つ以上存在すれば、同一 version の複数 component
        を全て通す)。分岐・スキップ版への最短経路探索は現状扱わない。
        """
        if from_version > to_version:
            raise MigrationChainNotFoundError(
                f"Downgrade is not supported: {from_version} -> {to_version}",
            )

        # version -> list[Migration] の索引を作る
        by_from: dict[int, list[Migration]] = {}
        for m in self._migrations:
            by_from.setdefault(m.from_version, []).append(m)

        chain: list[Migration] = []
        current = from_version
        while current < to_version:
            next_migrations = by_from.get(current, [])
            if not next_migrations:
                raise MigrationChainNotFoundError(
                    f"No migration registered for version {current} "
                    f"(goal: {to_version})",
                )
            # 同一 from_version の migration は全て同じ to_version を指すこと
            # (component 別に複数の Migration が並ぶ想定) を検証する
            next_versions = {m.to_version for m in next_migrations}
            if len(next_versions) > 1:
                raise MigrationChainNotFoundError(
                    f"Inconsistent migrations at version {current}: "
                    f"multiple to_versions {next_versions}",
                )
            # component 順で安定ソート
            next_migrations_sorted = sorted(
                next_migrations, key=lambda mig: mig.component,
            )
            chain.extend(next_migrations_sorted)
            current = next_versions.pop()

        if current != to_version:
            raise MigrationChainNotFoundError(
                f"Migration chain overshoots target: reached {current}, "
                f"expected {to_version}",
            )
        return chain

    def _prepare_backup_root(self, now: float | None = None) -> Path:
        root = self._migration_archive_dir / f"{_BACKUP_ROOT_PREFIX}{_timestamp(now)}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _backup_step(
        memory_dir: Path,
        migration: Migration,
        step_backup_dir: Path,
    ) -> None:
        """migration.dry_run の affected_files を step_backup_dir へコピーする.

        `.bak` 拡張子を付与することは行わず、memory_dir からの相対パスを
        維持したままコピーする (rollback 時に相対パスで戻せるため)。
        """
        plan = migration.dry_run(memory_dir)
        step_backup_dir.mkdir(parents=True, exist_ok=True)
        for src in plan.affected_files:
            if not src.exists():
                continue
            try:
                rel = src.relative_to(memory_dir)
            except ValueError:
                # memory_dir 外のファイルは対象外 (誤指定をスキップ)
                logger.warning(
                    "Skip backup of file outside memory_dir: %s", src,
                )
                continue
            dest = step_backup_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)

    @staticmethod
    def _rollback(memory_dir: Path, steps: list[_Step]) -> None:
        """成功済 step を逆順で rollback する."""
        for step in reversed(steps):
            try:
                step.migration.rollback(memory_dir, step.backup_dir)
            except Exception:
                logger.exception(
                    "Rollback failed for step %d -> %d (%s); continuing",
                    step.migration.from_version,
                    step.migration.to_version,
                    step.migration.component,
                )


def run_migrations(
    memory_dir: Path,
    from_version: int,
    to_version: int,
    *,
    migration_archive_dir: Path,
    migrations: list[Migration] | None = None,
    dry_run: bool = False,
    now: float | None = None,
) -> list[MigrationResult] | list[MigrationPlan]:
    """`SchemaMigrator` への薄いパブリック API ラッパ.

    `backend.factory._memory_init._init_memory` からワンショットで呼び出しやすい形に
    整えたもの。マイグレーション登録表は将来的にこのモジュール内の
    `DEFAULT_MIGRATIONS` として集約する想定 (現状は空)。
    """
    migrator = SchemaMigrator(
        migrations=migrations,
        migration_archive_dir=migration_archive_dir,
    )
    return migrator.upgrade(
        memory_dir,
        from_version,
        to_version,
        dry_run=dry_run,
        now=now,
    )


# 登録済 Migration の集約点。
# v1 → v2 は M5-d で導入: SemMem 索引を旧 4 ファイル形式から fact_id 正規化形の
# 統合 ``index.jsonl`` に変換する。
from backend.free.memory.migrations.index_v1_to_v2 import (  # noqa: E402
    IndexV1ToV2Migration,
)

DEFAULT_MIGRATIONS: list[Migration] = [IndexV1ToV2Migration()]


__all__ = [
    "DEFAULT_MIGRATIONS",
    "MigrationChainNotFoundError",
    "MigrationError",
    "SchemaMigrator",
    "run_migrations",
]
