"""EvorefMem schema migration フレームワーク

`SchemaMigrator` は `initialize_evorefmem` (destructive re-init) の前段に
位置する **in-place rewrite** 手段を提供する。v1 のみの現状では登録
Migration は 0 件で `upgrade()` は no-op だが、将来の subject rename /
FactType 統廃合 / 索引再構造化を無痛で取り込む足場として ABC / ランナーを
整備する。現行版以外のマーカーは SchemaMigrator → destructive init の
2 段階で処理される。

Public API:

- :class:`Migration` / :class:`MigrationPlan` / :class:`MigrationResult`
- :class:`SchemaMigrator`
- :func:`run_migrations` (ワンショットヘルパ)
- :exc:`MigrationError` / :exc:`MigrationChainNotFoundError`
- :data:`DEFAULT_MIGRATIONS` (登録済 Migration 集約点 / 現状は index v1→v2 の 1 件)
"""

from __future__ import annotations

from .base import ComponentKind, Migration, MigrationPlan, MigrationResult
from .migrator import (
    DEFAULT_MIGRATIONS,
    MigrationChainNotFoundError,
    MigrationError,
    SchemaMigrator,
    run_migrations,
)

__all__ = [
    "DEFAULT_MIGRATIONS",
    "ComponentKind",
    "Migration",
    "MigrationChainNotFoundError",
    "MigrationError",
    "MigrationPlan",
    "MigrationResult",
    "SchemaMigrator",
    "run_migrations",
]
