"""evorefmem CLI の path 解決ヘルパ

各 subcommand が ``<memory_dir>/semantic/`` 配下の全 scope を列挙する
共通ロジック。``backend.config.get_path_resolver`` 経由で memory_dir /
prompts_dir / migration_archive_dir を取得し、scope (``global`` /
``projects/<id>``) を順に enumerate する。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from backend.log_config import get_logger

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


@dataclass(frozen=True)
class ScopeInfo:
    """semantic ストア 1 scope 分のメタ情報."""

    name: str
    """``"global"`` または ``"project:<id>"`` 形式 (SemanticFact.scope と同型)。"""

    root_dir: Path
    """``<semantic_dir>/global/`` または ``<semantic_dir>/projects/<id>/``。"""

    @property
    def is_global(self) -> bool:
        return self.name == "global"

    @property
    def is_project(self) -> bool:
        return self.name.startswith("project:")


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


def enumerate_scopes(memory_dir: Path) -> list[ScopeInfo]:
    """``<memory_dir>/semantic/`` 配下の全 scope を列挙する.

    - ``semantic/global/`` が存在すれば ``ScopeInfo(name="global", ...)``
    - ``semantic/projects/<id>/`` の各 subdir を ``project:<id>`` として追加
    - ``archive/`` は管理外 (列挙しない)

    semantic/ 自体が無い (=未初期化) 環境では空リストを返す。
    """
    semantic_dir = Path(memory_dir) / "semantic"
    out: list[ScopeInfo] = []
    if not semantic_dir.exists():
        return out

    global_dir = semantic_dir / "global"
    if global_dir.exists() and global_dir.is_dir():
        out.append(ScopeInfo(name="global", root_dir=global_dir))

    projects_dir = semantic_dir / "projects"
    if projects_dir.exists() and projects_dir.is_dir():
        for entry in sorted(projects_dir.iterdir()):
            if entry.is_dir():
                out.append(
                    ScopeInfo(
                        name=f"project:{entry.name}",
                        root_dir=entry,
                    ),
                )
    return out


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
    "CliPaths",
    "ScopeInfo",
    "cli_backup_root",
    "enumerate_scopes",
    "resolve_cli_paths",
]
