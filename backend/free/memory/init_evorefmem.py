"""EvorefMem 初期化モジュール

旧メモリデータの破棄と新スキーマの初期化を担当する。

責務:
1. 旧 `local/memory/working|short_term|long_term` データの破棄
2. `local/memory/semantic/{global,projects,archive}/` の作成
3. `local/prompts/policy_evolver_state.json` / `local/prompts/fewshot_pool.json` を
   `local/migration_archive/` に退避してから削除 (30 日後 GC)
4. `local/memory/semantic/SCHEMA_VERSION` マーカーの書き出し
5. 起動時の schema_version 検査ヘルパー

設計原則:
- 純粋関数で副作用を最小化、I/O は明示的に外側へ
- 既存 `MemoryConfig.schema_version` (Pydantic) を Single Source of Truth とする
- 後方互換は提供しない (旧データ廃棄を強制)
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.io.atomic import atomic_write_text
from backend.log_config import get_logger

logger = get_logger("memory.init_evorefmem")

SCHEMA_VERSION = 1
SCHEMA_MARKER_FILENAME = "SCHEMA_VERSION"
LEGACY_BACKUP_RETENTION_DAYS = 30

# 破棄対象となる旧メモリデータの相対パス (memory_dir 配下)
LEGACY_MEMORY_PATHS: tuple[str, ...] = (
    "short_term_notes.json",
    "working",
    "short_term",
    "long_term",
)

# 退避対象となる旧 prompts 永続化ファイル (prompts_dir 配下)
LEGACY_PROMPT_FILES: tuple[str, ...] = (
    "policy_evolver_state.json",
    "fewshot_pool.json",
)

# 新スキーマで作成する semantic サブディレクトリ (memory_dir/semantic/ 配下)
SEMANTIC_SUBDIRS: tuple[str, ...] = ("global", "projects", "archive")


@dataclass
class InitResult:
    """初期化結果サマリ"""

    backed_up: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    created: list[Path] = field(default_factory=list)
    gc_removed: list[Path] = field(default_factory=list)
    schema_marker: Path | None = None
    schema_version: int = SCHEMA_VERSION


def _semantic_root(memory_dir: Path) -> Path:
    return memory_dir / "semantic"


def schema_marker_path(memory_dir: Path) -> Path:
    """スキーマバージョンマーカーファイルのパスを返す"""
    return _semantic_root(memory_dir) / SCHEMA_MARKER_FILENAME


def read_schema_version(memory_dir: Path) -> int | None:
    """マーカーファイルからスキーマバージョンを読む。存在しない/壊れていれば None"""
    marker = schema_marker_path(memory_dir)
    if not marker.exists():
        return None
    try:
        text = marker.read_text(encoding="utf-8").strip()
        return int(text)
    except (OSError, ValueError) as exc:
        logger.warning("schema marker unreadable at %s: %s", marker, exc)
        return None


def needs_initialization(memory_dir: Path, expected_version: int = SCHEMA_VERSION) -> bool:
    """初期化が必要かを判定する。

    マーカーが現行版と一致する場合のみ False を返す。それ以外
    (未書き込み / 旧版 / 未知の値) は True を返し、SchemaMigrator
    による in-place migration のフォールバックとして
    `initialize_evorefmem()` (destructive re-init) が呼ばれる。
    """
    current = read_schema_version(memory_dir)
    return current != expected_version


def verify_schema_version(
    memory_dir: Path, expected_version: int = SCHEMA_VERSION,
) -> bool:
    """起動時の schema_version 検査。

    一致すれば True、不一致なら警告ログを出して False を返す。
    呼び出し側で WARN ログ + ユーザー向けメッセージを出すための判定を兼ねる。
    """
    current = read_schema_version(memory_dir)
    if current == expected_version:
        return True
    logger.warning(
        "Memory schema mismatch: expected=%s actual=%s. "
        "Run `python scripts/init_evorefmem.py` to initialize EvorefMem storage.",
        expected_version, current,
    )
    return False


def backup_legacy_prompts(
    prompts_dir: Path, migration_archive_dir: Path, *, now: float | None = None,
) -> tuple[list[Path], Path | None]:
    """旧 prompts 永続化ファイルをタイムスタンプ付きディレクトリへコピー退避する。

    Returns: (退避したファイルパス一覧, 退避先ディレクトリ or None)
    """
    if now is None:
        now = time.time()
    # サーバ実行 tz に依存しないよう UTC で統一し末尾に Z を付与
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    targets: list[Path] = []
    for name in LEGACY_PROMPT_FILES:
        src = prompts_dir / name
        if src.exists() and src.is_file():
            targets.append(src)

    if not targets:
        return [], None

    dest_dir = migration_archive_dir / f"prompts_{timestamp}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    backed_up: list[Path] = []
    for src in targets:
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        backed_up.append(dest)
        logger.info("Backed up legacy prompt: %s -> %s", src, dest)
    return backed_up, dest_dir


def delete_legacy_prompts(prompts_dir: Path) -> list[Path]:
    """旧 prompts 永続化ファイルを削除する (退避済み前提)"""
    deleted: list[Path] = []
    for name in LEGACY_PROMPT_FILES:
        path = prompts_dir / name
        if path.exists() and path.is_file():
            path.unlink()
            deleted.append(path)
            logger.info("Deleted legacy prompt: %s", path)
    return deleted


def delete_legacy_memory(memory_dir: Path) -> list[Path]:
    """旧 memory_dir 配下の WM/STM/LTM 形式データを破棄する。

    semantic/ サブツリーは保護対象 (削除しない)。
    """
    deleted: list[Path] = []
    for rel in LEGACY_MEMORY_PATHS:
        path = memory_dir / rel
        if not path.exists():
            continue
        if path.is_file():
            path.unlink()
            deleted.append(path)
            logger.info("Deleted legacy memory file: %s", path)
        elif path.is_dir():
            shutil.rmtree(path)
            deleted.append(path)
            logger.info("Deleted legacy memory dir: %s", path)
    return deleted


def create_semantic_dirs(memory_dir: Path) -> list[Path]:
    """semantic/{global,projects,archive}/ を作成する"""
    created: list[Path] = []
    root = _semantic_root(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not any(True for _ in [root]):
        created.append(root)
    for sub in SEMANTIC_SUBDIRS:
        d = root / sub
        existed = d.exists()
        d.mkdir(parents=True, exist_ok=True)
        if not existed:
            created.append(d)
    return created


def write_schema_marker(memory_dir: Path, version: int = SCHEMA_VERSION) -> Path:
    """semantic/SCHEMA_VERSION マーカーを書き出す"""
    root = _semantic_root(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / SCHEMA_MARKER_FILENAME
    atomic_write_text(marker, f"{version}\n", fsync=True)
    logger.info("Wrote schema marker: %s = %s", marker, version)
    return marker


def gc_legacy_backup(
    migration_archive_dir: Path,
    *,
    retention_days: int = LEGACY_BACKUP_RETENTION_DAYS,
    now: float | None = None,
) -> list[Path]:
    """retention_days を過ぎた退避ディレクトリを削除する"""
    if not migration_archive_dir.exists():
        return []
    if now is None:
        now = time.time()
    cutoff = now - retention_days * 86400
    removed: list[Path] = []
    for entry in migration_archive_dir.iterdir():
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed.append(entry)
        logger.info("GC removed legacy backup entry: %s (age=%.1fd)",
                    entry, (now - mtime) / 86400)
    return removed


def initialize_evorefmem(
    memory_dir: Path,
    prompts_dir: Path,
    migration_archive_dir: Path,
    *,
    now: float | None = None,
) -> InitResult:
    """初期化のオーケストレータ

    呼び出し順:
    1. 旧 prompts ファイルを migration_archive へ退避
    2. 旧 prompts ファイルを削除
    3. 旧 memory データを削除
    4. semantic/{global,projects,archive}/ を作成
    5. SCHEMA_VERSION マーカーを書き出し
    6. 30 日超過した migration_archive エントリを GC
    """
    result = InitResult()

    backed_up, _backup_dir = backup_legacy_prompts(
        prompts_dir, migration_archive_dir, now=now,
    )
    result.backed_up = backed_up

    result.deleted.extend(delete_legacy_prompts(prompts_dir))
    result.deleted.extend(delete_legacy_memory(memory_dir))
    result.created = create_semantic_dirs(memory_dir)
    result.schema_marker = write_schema_marker(memory_dir, SCHEMA_VERSION)
    result.gc_removed = gc_legacy_backup(migration_archive_dir, now=now)
    result.schema_version = SCHEMA_VERSION

    logger.info(
        "EvorefMem initialized: backed_up=%d deleted=%d created=%d gc=%d",
        len(result.backed_up), len(result.deleted),
        len(result.created), len(result.gc_removed),
    )
    return result
