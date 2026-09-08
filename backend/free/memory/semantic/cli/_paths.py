"""evorefmem CLI の path / ストア解決ヘルパ

``<memory_dir>/semantic/`` は **1 ストア** になり (c_16 §4.2)、スコープは
ディレクトリではなくレコードのフィールドになった。したがって CLI の共通処理も
「scope ディレクトリを列挙する」から「ストアを 1 つ開いて scope を数える」へ
変わっている。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticStore

logger = get_logger("memory.semantic.cli.paths")


@dataclass(frozen=True)
class CliPaths:
    """CLI が参照する基本パス群."""

    memory_dir: Path
    """``local/memory/`` ルート (PathResolver で解決)。"""

    prompts_dir: Path
    """``local/prompts/`` ルート (init 系で参照)。"""

    migration_archive_dir: Path
    """``local/migration_archive/`` ルート。CLI の破壊的操作はこの配下へ退避する。"""

    @property
    def semantic_dir(self) -> Path:
        """``<memory_dir>/semantic/``。"""
        return self.memory_dir / "semantic"


def resolve_cli_paths() -> CliPaths:
    """``config.yaml`` を読み込んで CLI 用パス群を解決する.

    本関数は遅延 import で ``backend.config`` を読み込む。CLI 単体テストの
    fixture では tmp_path を使うため、本関数を呼ばずに :class:`CliPaths` を
    直接構築するのが原則。
    """
    from backend.config import get_path_resolver, load_config

    load_config()
    resolver = get_path_resolver()
    memory_dir = resolver.resolve_local("memory_dir")
    prompts_dir = resolver.resolve_local("prompts_dir")
    migration_archive_dir = resolver.resolve_local("migration_archive_dir")
    memory_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    migration_archive_dir.mkdir(parents=True, exist_ok=True)
    return CliPaths(
        memory_dir=memory_dir,
        prompts_dir=prompts_dir,
        migration_archive_dir=migration_archive_dir,
    )


def open_semantic_store(memory_dir: Path) -> "SemanticStore":
    """``<memory_dir>/semantic`` のストアを開いて読み込む (埋め込みなし)。

    CLI は検索をしないので ``embedding_backend`` は渡さない。ベクトル索引は
    snapshot 側にあるものをそのまま読む (書き換えない)。
    """
    from backend.free.memory.semantic.store import SemanticStore

    store = SemanticStore(Path(memory_dir))
    store.load()
    return store


def scope_names(store: "SemanticStore") -> list[str]:
    """ストアに実在する scope 名を昇順で返す (``global`` が先頭)。"""
    scopes = {f.scope or "global" for f in store.all_facts(include_superseded=True)}
    scopes.add("global")
    return sorted(scopes, key=lambda s: (s != "global", s))


def cli_backup_root(
    migration_archive_dir: Path,
    subcommand: str,
    *,
    now: float | None = None,
) -> Path:
    """``migration_archive/cli_<utc_ts>/<subcommand>/`` を作って返す.

    破壊的 subcommand が前段で全 affected file をここへコピーしてから rewrite する。
    timestamp 形式は :mod:`backend.free.memory.init_evorefmem` 等の既存規約
    (``YYYYMMDDTHHMMSSZ``) に合わせる
    """
    t = time.time() if now is None else now
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t))
    root = Path(migration_archive_dir) / f"cli_{timestamp}" / subcommand
    root.mkdir(parents=True, exist_ok=True)
    _prune_backup_roots(Path(migration_archive_dir))
    return root


#: 残す ``cli_*`` バックアップ世代数。1 世代が SemMem 埋め込み全量のコピーに
#: なりうるため (再埋め込みのたびに丸ごと複製)、無制限だとコーパス規模 ×
#: 実行回数でディスクを食い潰す (2026-09-05 監査)。
BACKUP_KEEP_GENERATIONS = 5


def _prune_backup_roots(migration_archive_dir: Path, *, keep: int | None = None) -> int:
    """``cli_*`` バックアップを新しい順に ``keep`` 世代残して削除する。"""
    limit = BACKUP_KEEP_GENERATIONS if keep is None else keep
    try:
        roots = sorted(
            (p for p in migration_archive_dir.glob("cli_*") if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
    except OSError:
        return 0
    removed = 0
    for stale in roots[limit:]:
        try:
            shutil.rmtree(stale)
            removed += 1
        except OSError as exc:
            logger.warning("Failed to prune backup %s: %s", stale, exc)
    if removed:
        logger.info("Pruned %d old migration backup(s)", removed)
    return removed


__all__ = [
    "BACKUP_KEEP_GENERATIONS",
    "CliPaths",
    "cli_backup_root",
    "open_semantic_store",
    "resolve_cli_paths",
    "scope_names",
]
