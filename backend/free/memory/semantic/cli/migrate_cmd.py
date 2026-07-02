"""``evorefmem_cli migrate`` 実装

:class:`SchemaMigrator` への薄い CLI ラッパ. ``--list`` で登録 Migration を
列挙し、``--to <version>`` で連鎖実行する.

破壊的なため:
- デフォルトで dry-run (``--apply`` 必須)
- 退避は :class:`SchemaMigrator` 自身が ``migration_archive/migration_<utc_ts>/``
  へ書き出す。本コマンドはランナーに委譲するだけ。
- ``DEFAULT_MIGRATIONS`` が空 (v1 のみの現状) でも no-op で通る。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.free.memory.init_evorefmem import (
    SCHEMA_VERSION,
    read_schema_version,
)
from backend.free.memory.migrations import (
    DEFAULT_MIGRATIONS,
    Migration,
    MigrationChainNotFoundError,
    SchemaMigrator,
)
from backend.free.memory.migrations.base import MigrationPlan, MigrationResult


@dataclass
class MigrationSummary:
    """登録された 1 Migration のサマリ (``--list`` 用)."""

    class_name: str
    from_version: int
    to_version: int
    component: str

    @classmethod
    def from_migration(cls, m: Migration) -> MigrationSummary:
        return cls(
            class_name=type(m).__name__,
            from_version=m.from_version,
            to_version=m.to_version,
            component=m.component,
        )


@dataclass
class MigrateReport:
    memory_dir: str
    current_version: int | None
    target_version: int
    applied: bool
    registered: list[MigrationSummary] = field(default_factory=list)
    plans: list[dict[str, Any]] = field(default_factory=list)
    """dry-run の :class:`MigrationPlan` 列。"""

    results: list[dict[str, Any]] = field(default_factory=list)
    """apply 実行時の :class:`MigrationResult` 列。"""

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_dir": self.memory_dir,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "applied": self.applied,
            "registered": [asdict(r) for r in self.registered],
            "plans": self.plans,
            "results": self.results,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _plan_to_dict(p: MigrationPlan) -> dict[str, Any]:
    return {
        "from_version": p.from_version,
        "to_version": p.to_version,
        "component": p.component,
        "description": p.description,
        "affected_files": [str(x) for x in p.affected_files],
        "estimated_records": p.estimated_records,
    }


def _result_to_dict(r: MigrationResult) -> dict[str, Any]:
    return {
        "from_version": r.from_version,
        "to_version": r.to_version,
        "component": r.component,
        "processed_records": r.processed_records,
        "backup_dir": str(r.backup_dir) if r.backup_dir is not None else None,
        "elapsed_ms": r.elapsed_ms,
        "details": r.details,
    }


def list_registered_migrations(
    migrations: list[Migration] | None = None,
) -> list[MigrationSummary]:
    """登録された Migration のサマリを返す (デフォルトは ``DEFAULT_MIGRATIONS``)."""
    src = DEFAULT_MIGRATIONS if migrations is None else migrations
    return [MigrationSummary.from_migration(m) for m in src]


def run_migrate(
    memory_dir: Path,
    migration_archive_dir: Path,
    *,
    target_version: int = SCHEMA_VERSION,
    apply: bool = False,
    migrations: list[Migration] | None = None,
    now: float | None = None,
) -> MigrateReport:
    """SchemaMigrator を呼び出して dry-run / apply する.

    Args:
        memory_dir: ``local/memory/`` ルート。
        migration_archive_dir: backup 退避先 (``SchemaMigrator`` の責務)。
        target_version: 目標版番号 (デフォルト ``SCHEMA_VERSION``)。
        apply: True なら実行、False なら dry-run。
        migrations: 上書き用 (テスト fixture)。None なら ``DEFAULT_MIGRATIONS``。
        now: タイムスタンプ上書き (テスト)。

    Returns:
        :class:`MigrateReport`。``error`` が設定されていれば失敗。
    """
    current = read_schema_version(memory_dir)
    rep = MigrateReport(
        memory_dir=str(memory_dir),
        current_version=current,
        target_version=target_version,
        applied=apply,
        registered=list_registered_migrations(migrations),
    )
    if current is None:
        rep.error = (
            "SCHEMA_VERSION marker not found; "
            "run `evorefmem_cli init` to initialize"
        )
        return rep

    src = DEFAULT_MIGRATIONS if migrations is None else migrations
    migrator = SchemaMigrator(
        migrations=src,
        migration_archive_dir=migration_archive_dir,
    )
    try:
        if apply:
            results = migrator.upgrade(
                memory_dir, current, target_version, dry_run=False, now=now,
            )
            for r in results:
                if isinstance(r, MigrationResult):
                    rep.results.append(_result_to_dict(r))
        else:
            plans = migrator.upgrade(
                memory_dir, current, target_version, dry_run=True, now=now,
            )
            for p in plans:
                if isinstance(p, MigrationPlan):
                    rep.plans.append(_plan_to_dict(p))
    except MigrationChainNotFoundError as exc:
        rep.error = f"chain not found: {exc}"
    except Exception as exc:
        rep.error = f"{type(exc).__name__}: {exc}"
    return rep


def format_report_text(report: MigrateReport) -> str:
    lines: list[str] = []
    lines.append(f"memory_dir       : {report.memory_dir}")
    lines.append(
        f"version          : current={report.current_version} "
        f"target={report.target_version}",
    )
    mode = "apply" if report.applied else "dry-run"
    lines.append(f"mode             : {mode}")
    if report.registered:
        lines.append("")
        lines.append("registered migrations:")
        for m in report.registered:
            lines.append(
                f"  - {m.class_name:30} {m.from_version} -> {m.to_version} "
                f"({m.component})",
            )
    else:
        lines.append("registered migrations: (none)")

    if report.error:
        lines.append("")
        lines.append(f"ERROR: {report.error}")
        return "\n".join(lines)

    if report.plans:
        lines.append("")
        lines.append("dry-run plans:")
        for p in report.plans:
            lines.append(
                f"  - {p['from_version']} -> {p['to_version']} "
                f"({p['component']}): {p['description']} "
                f"affected={len(p['affected_files'])} "
                f"est_records={p['estimated_records']}",
            )
    if report.results:
        lines.append("")
        lines.append("apply results:")
        for r in report.results:
            lines.append(
                f"  - {r['from_version']} -> {r['to_version']} "
                f"({r['component']}): processed={r['processed_records']} "
                f"elapsed={r['elapsed_ms']:.1f}ms",
            )
            if r["backup_dir"]:
                lines.append(f"      backup: {r['backup_dir']}")
    if not report.plans and not report.results and not report.error:
        if report.current_version == report.target_version:
            lines.append(
                f"\n(no migration needed; already at v{report.target_version})",
            )
    return "\n".join(lines)


__all__ = [
    "MigrateReport",
    "MigrationSummary",
    "format_report_text",
    "list_registered_migrations",
    "run_migrate",
]
