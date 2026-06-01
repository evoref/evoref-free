"""evoref-dev → evoref-pro 同期スクリプト。

evoref-dev リポジトリの現在の HEAD から **Free + Pro 一式** を ``release/pro/`` に
再生成し、対応する CHANGELOG セクションを自動追記したうえで、
``evoref-pro`` リポジトリへ同期 PR を作成する。

Free 版との違い (= ``release_free_sync.py`` との差分):
    - target: ``evoref/evoref-pro`` (https://github.com/evoref/evoref-pro.git)
    - 出力: ``release/pro/`` (Free + Pro 一式、Pro EULA 配布)
    - include: backend/pro/, frontend/src/lib/pro/, config.pro.yaml.example,
              backend/requirements-pro.txt も同梱
    - exclude: backend/develop/ は引き続き除外 (内部限定)
    - LICENSE: ``LICENSE-PRO`` → ``release/pro/LICENSE`` (主たる EULA)
              ``LICENSE-FREE`` は ``release/pro/LICENSE-FREE`` としてそのまま同梱
    - VERSION SSOT: ``backend/pro/__version__.py`` (Pro 版バージョン)
    - 静的検証: Pro import 検出は無効化、代わりに Develop import を検出

詳細仕様: ``.claude/skills/release-pro-sync/SKILL.md`` を参照。

主な引数:
    --dry-run            release/pro 差分と CHANGELOG プレビューのみ表示
    --no-pr              commit + push まで実行、gh pr create を省略
    --changelog-only     CHANGELOG 追記のみ実行、evoref-pro 同期は行わない
    --skip-changelog     Phase 1.5 (CHANGELOG 追記) を skip
    --changelog-entries  Markdown ファイル経由で entry 本文を手動指定
    --initial-release    commit 履歴を載せず初回リリース用定型セクションを書き込む
    --target-dir         evoref-pro のローカルパス (既定: E:/sources/evoref/evoref-pro)
    --branch-suffix      sync ブランチの suffix (既定: evoref-dev HEAD short SHA)
    --version            バージョン文字列の上書き (既定: SSOT 自動取得)
    --allow-dirty        evoref-dev の working tree dirty でも続行

終了コード:
    0  成功
    10 前提違反 (Phase 0 失敗)
    20 Develop 混入検出 / 必須ファイル不足 (Phase 2 失敗)
    30 同期不要 (差分なし)
    1  その他のエラー
"""

from __future__ import annotations

import argparse
import ast
import filecmp
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Windows のコンソールが cp932 でも日本語ログが文字化けしないよう UTF-8 を強制。
for _stream in (sys.stdout, sys.stderr):
    _enc = getattr(_stream, "encoding", None) or ""
    if _enc.lower() != "utf-8":
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASE_PRO = PROJECT_ROOT / "release" / "pro"
RELEASE_PRO_TMP = PROJECT_ROOT / "release" / "pro.tmp"
LOCK_FILE = PROJECT_ROOT / "release" / ".sync-pro.lock"
VERSION_FILE = PROJECT_ROOT / "backend" / "pro" / "__version__.py"
SOURCE_PYPROJECT = PROJECT_ROOT / "pyproject.toml"
DEFAULT_TARGET = Path("E:/sources/evoref/evoref-pro")
DEFAULT_REMOTE_REPO = "evoref/evoref-pro"
EXPECTED_REMOTE_URL = "https://github.com/evoref/evoref-pro.git"

VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-.+].*)?$")


EXIT_OK = 0
EXIT_PREFLIGHT = 10
EXIT_VALIDATION = 20
EXIT_NO_CHANGES = 30


# ---------------------------------------------------------------------------
# include / exclude ルール (Pro 配布用 — Free + Pro 一式)
# ---------------------------------------------------------------------------

EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".github",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".svelte-kit",
        "node_modules",
        "build",
        "dist",
        "coverage",
        "htmlcov",
        "reports",
        ".idea",
        ".claude",
        "evoref.egg-info",
        "release",
    }
)

# 注: Free と異なり ``backend/pro/`` と ``frontend/src/lib/pro/`` は除外しない。
# Develop 限定コード (``backend/develop/``) は Pro 配布でも内部限定のため除外を継続。
EXCLUDE_PATH_PREFIXES: tuple[str, ...] = (
    "backend/develop/",
    "frontend/src/lib/test-utils/",  # vitest 用の test ヘルパ (production 非参照)
    "frontend/src/__mocks__/",  # vitest 用の $app モジュール mock
    "frontend/coverage/",
    "scripts/bench/",  # 内部ベンチマーク
    "release/",
    ".vscode/",
)

# local/ と models/ は基本除外だが .gitkeep だけはディレクトリ構造維持のため同梱する。
GITKEEP_RETAIN_PREFIXES: tuple[str, ...] = ("local/", "models/")

# 注: Free と異なり ``LICENSE-PRO`` / ``config.pro.yaml.example`` は除外しない。
EXCLUDE_TOP_FILES: frozenset[str] = frozenset(
    {
        "config.yaml",
        ".gitmodules",
        "CLAUDE.md",
        "pytest.ini",  # Pro 利用者はテストを走らせる想定なし
        "requirements-dev.txt",  # 開発時のみ必要
    }
)

# Pro 配布では ``backend/requirements-pro.txt`` も同梱する (Pro 専用依存)。
EXCLUDE_BACKEND_FILES: frozenset[str] = frozenset()

# 個別ファイル単位の除外 (POSIX パス完全一致)。
# EXCLUDE_PATH_PREFIXES では巻き込みすぎる場合に使う。
EXCLUDE_EXACT_PATHS: frozenset[str] = frozenset(
    {
        # 内部ツール (Free/Pro いずれの利用者にも不要、リポジトリ運用専用)
        "scripts/safe_pytest.py",
        "scripts/release_free_sync.py",
        "scripts/release_pro_sync.py",
        "scripts/aggregate_debug_logs.py",
        "scripts/generate_icons.py",  # static/ icon は事前生成済み
        # frontend テスト設定 (production ビルドに不要)
        "frontend/playwright.config.ts",
        "frontend/vitest.config.ts",
        "frontend/scripts/run-e2e.mjs",
        "frontend/src/setupTests.ts",
    }
)

EXCLUDE_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".DS_Store",
        "Thumbs.db",
        ".coverage",
    }
)

EXCLUDE_FILE_SUFFIXES: tuple[str, ...] = (
    ".pyc",
    ".pyo",
    ".log",
    ".bak",
)


def is_changelog_filename(name: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.x_CHANGELOG\.md$", name))


def is_doc_keeper(relative_path: Path) -> bool:
    """``docs/`` 以下で Pro 配布に同梱するファイルか判定。"""
    if relative_path.parts[:1] != ("docs",):
        return False
    if len(relative_path.parts) == 2 and is_changelog_filename(relative_path.parts[1]):
        return True
    if relative_path.parts[-1] == ".gitkeep":
        return True
    return False


def is_doc_excluded(relative_path: Path) -> bool:
    """``docs/`` 配下で除外すべき内部設計書か判定。"""
    if relative_path.parts[:1] != ("docs",):
        return False
    name = relative_path.parts[-1]
    if name.endswith(".md") and re.match(r"^[cefpa]_\d", name):
        return True
    return False


def should_exclude_path(relative_path: Path) -> bool:
    """``relative_path`` (POSIX 風) を再生成対象から外すか判定。"""
    posix = relative_path.as_posix()
    # local/ と models/ 配下の .gitkeep はディレクトリ構造維持のため例外的に keep
    if relative_path.name == ".gitkeep":
        for prefix in GITKEEP_RETAIN_PREFIXES:
            if posix.startswith(prefix):
                return False
    # EXCLUDE_DIRS のディレクトリ名が path のどこかに現れたら除外
    for part in relative_path.parts[:-1]:
        if part in EXCLUDE_DIRS:
            return True
    if posix in EXCLUDE_TOP_FILES:
        return True
    if posix in EXCLUDE_BACKEND_FILES:
        return True
    if posix in EXCLUDE_EXACT_PATHS:
        return True
    if relative_path.name in EXCLUDE_FILE_NAMES:
        return True
    if relative_path.suffix in EXCLUDE_FILE_SUFFIXES:
        return True
    if relative_path.name.startswith(".coverage."):
        return True
    for prefix in EXCLUDE_PATH_PREFIXES:
        if posix.startswith(prefix):
            return True
    # models/profiles/ 配下の arch 別プロファイル同梱ベースは配布対象 (tracked)
    if posix.startswith("models/profiles/"):
        return False
    # local/ と models/ は .gitkeep 以外を除外
    if posix.startswith("local/") or posix.startswith("models/"):
        return True
    if relative_path.parts[:1] == ("docs",):
        return not is_doc_keeper(relative_path) or is_doc_excluded(relative_path)
    # tests/ ディレクトリ・conftest.py は Pro 配布に含めない
    if "tests" in relative_path.parts:
        return True
    if relative_path.name == "conftest.py":
        return True
    if relative_path.parts[:1] == ("frontend",) and relative_path.parts[1:2] == ("e2e",):
        return True
    return False


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int
    raw: str

    @property
    def minor_key(self) -> str:
        return f"{self.major}.{self.minor}"

    @property
    def changelog_filename(self) -> str:
        return f"{self.major}.{self.minor}.x_CHANGELOG.md"

    @property
    def section_header(self) -> str:
        return f"## [{self.major}.{self.minor}.{self.patch}]"


def parse_version(s: str) -> SemVer:
    m = VERSION_RE.match(s.strip())
    if not m:
        raise ValueError(f"version 文字列が semver でない: {s!r}")
    return SemVer(int(m.group(1)), int(m.group(2)), int(m.group(3)), s.strip())


def read_version_from_ssot() -> SemVer:
    text = VERSION_FILE.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"{VERSION_FILE} から __version__ を抽出できない")
    return parse_version(m.group(1))


# ---------------------------------------------------------------------------
# Phase 0: 前提チェック
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_short_sha(repo: Path = PROJECT_ROOT) -> str:
    return _run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"]).stdout.strip()


def _git_full_sha(repo: Path = PROJECT_ROOT) -> str:
    return _run(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()


def _git_current_branch(repo: Path = PROJECT_ROOT) -> str:
    return _run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def _git_working_tree_clean(repo: Path = PROJECT_ROOT) -> bool:
    out = _run(["git", "-C", str(repo), "status", "--porcelain"]).stdout
    return out.strip() == ""


def _git_remote_url(repo: Path) -> str:
    try:
        return _run(["git", "-C", str(repo), "remote", "get-url", "origin"]).stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def acquire_lock() -> None:
    """多重起動防止。``release/.sync-pro.lock`` を atomic に作成し、PID を書き込む。

    Free 版と Pro 版で別ロックを使うため、両者の同時実行は可能 (ただし
    target ディレクトリは別なので衝突しない)。
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder_pid = int(LOCK_FILE.read_text().strip())
        except (OSError, ValueError):
            holder_pid = -1
        if holder_pid > 0 and _pid_alive(holder_pid):
            print(
                f"  [FAIL] Pro sync が既に PID {holder_pid} で実行中です。"
                f" 完了を待つか {LOCK_FILE} を手動削除してください",
                file=sys.stderr,
            )
            sys.exit(EXIT_PREFLIGHT)
        print(f"  [WARN] stale lock を検出 (PID {holder_pid}): 上書きします")
        LOCK_FILE.unlink()
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))


def release_lock() -> None:
    """終了時に lock を解放 (例外を握りつぶす)。"""
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """指定 PID のプロセスが生存しているかを判定。Windows / POSIX 両対応。"""
    if sys.platform == "win32":
        result = _run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], check=False)
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


@dataclass(frozen=True)
class BranchSwitch:
    """main への自動切替情報。``original`` が None なら復帰不要。"""

    original: str | None


def switch_to_main(*, allow_dirty: bool) -> BranchSwitch:
    """skill 起動時に main へ切替。dirty + allow_dirty なら切替しない (ファイル消失防止)。"""
    print("[Pre-phase] main へ切替")
    current = _git_current_branch()
    if current == "main":
        print("  現在のブランチは main。切替不要")
        return BranchSwitch(original=None)
    if allow_dirty and not _git_working_tree_clean():
        print(f"  [WARN] --allow-dirty + dirty のため切替スキップ (現在: {current})")
        return BranchSwitch(original=None)
    print(f"  switching: {current} → main")
    _run(["git", "-C", str(PROJECT_ROOT), "fetch", "origin"])
    _run(["git", "-C", str(PROJECT_ROOT), "checkout", "main"])
    try:
        _run(["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only", "origin", "main"])
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] origin/main の ff-only pull 失敗 (続行): {e.stderr.strip()}")
    return BranchSwitch(original=current)


def restore_target_to_main(target: Path) -> None:
    """skill 終了時に evoref-pro を main に戻す (try/finally から呼ぶ)。"""
    try:
        current = _git_current_branch(target)
    except (subprocess.CalledProcessError, OSError):
        return
    if current == "main":
        return
    print(f"[teardown] evoref-pro 復帰: {current} → main")
    try:
        _run(["git", "-C", str(target), "checkout", "main"])
    except subprocess.CalledProcessError as e:
        print(
            f"  [WARN] evoref-pro main への復帰失敗: {e.stderr.strip()}",
            file=sys.stderr,
        )


def restore_branch(switch: BranchSwitch) -> None:
    """skill 終了時に元ブランチへ復帰 (try/finally から呼ぶ)。"""
    if switch.original is None:
        return
    print(f"[teardown] ブランチ復帰: → {switch.original}")
    try:
        _run(["git", "-C", str(PROJECT_ROOT), "checkout", switch.original])
    except subprocess.CalledProcessError as e:
        print(
            f"  [WARN] {switch.original} への復帰失敗: {e.stderr.strip()}",
            file=sys.stderr,
        )


def phase_0_preflight(args: argparse.Namespace) -> tuple[SemVer, str, str]:
    """前提チェック。失敗時は exit。返り値: (SemVer, short-sha, full-sha)。"""
    print("[Phase 0] 前提チェック")

    if not (PROJECT_ROOT / ".git").is_dir():
        print(f"  [FAIL] {PROJECT_ROOT} が git リポジトリでない", file=sys.stderr)
        sys.exit(EXIT_PREFLIGHT)

    if not _git_working_tree_clean() and not args.allow_dirty:
        print(
            "  [FAIL] evoref-dev の working tree が dirty です。"
            " commit/stash してから再実行するか --allow-dirty を付けてください",
            file=sys.stderr,
        )
        sys.exit(EXIT_PREFLIGHT)

    branch = _git_current_branch()
    if branch != "main":
        print(f"  [WARN] 現在のブランチは {branch!r} (推奨は main)")

    # changelog-only / dry-run は evoref-pro を触らないため target チェックは緩める
    skip_target_check = args.changelog_only or args.dry_run

    target = Path(args.target_dir).resolve()
    if not skip_target_check:
        if not target.is_dir() or not (target / ".git").exists():
            print(f"  [FAIL] target-dir {target} が git リポジトリでない", file=sys.stderr)
            sys.exit(EXIT_PREFLIGHT)

        target_remote = _git_remote_url(target)
        if target_remote != EXPECTED_REMOTE_URL:
            print(
                f"  [FAIL] target の origin が想定と異なる: {target_remote!r} != {EXPECTED_REMOTE_URL!r}",
                file=sys.stderr,
            )
            sys.exit(EXIT_PREFLIGHT)

        if not _git_working_tree_clean(target):
            print(
                f"  [FAIL] target {target} の working tree が dirty です。"
                f" commit/stash してから再実行してください",
                file=sys.stderr,
            )
            sys.exit(EXIT_PREFLIGHT)

    if not args.no_pr and not args.changelog_only and not args.dry_run:
        try:
            auth_user = _run(["gh", "api", "user", "-q", ".login"]).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            stderr = getattr(e, "stderr", "") or ""
            print(f"  [FAIL] gh CLI 認証エラー: {stderr.strip() or e}", file=sys.stderr)
            sys.exit(EXIT_PREFLIGHT)
        print(f"  gh authenticated as: {auth_user}")

    version = parse_version(args.version) if args.version else read_version_from_ssot()
    short_sha = _git_short_sha()
    full_sha = _git_full_sha()

    print(f"  version: {version.raw} (Pro Edition)")
    print(f"  evoref-dev HEAD: {short_sha} ({full_sha})")
    print(f"  target: {target}{' (skipped)' if skip_target_check else ''}")
    return version, short_sha, full_sha


# ---------------------------------------------------------------------------
# Phase 1: release/pro/ 再生成
# ---------------------------------------------------------------------------


def _iter_tracked_files() -> Iterable[Path]:
    """git で追跡されているファイルを ``Path`` (相対) で返す。"""
    out = _run(["git", "-C", str(PROJECT_ROOT), "ls-files", "-z"]).stdout
    for entry in out.split("\0"):
        if not entry:
            continue
        yield Path(entry)


def _preserve_changelogs(dest_docs: Path) -> list[str]:
    """既存 release/pro/docs/ の CHANGELOG と .gitkeep を dest にコピー。"""
    src_docs = RELEASE_PRO / "docs"
    preserved: list[str] = []
    if not src_docs.is_dir():
        return preserved
    dest_docs.mkdir(parents=True, exist_ok=True)
    for f in src_docs.iterdir():
        if f.is_file() and (is_changelog_filename(f.name) or f.name == ".gitkeep"):
            shutil.copy2(f, dest_docs / f.name)
            if is_changelog_filename(f.name):
                preserved.append(f.name)
    return preserved


def _rewrite_pyproject_for_release(content: str) -> str:
    """source pyproject.toml を release/pro 用に整形。

    現状は変更不要 (``dynamic = ["version"]`` のまま、attr 経由で解決)。
    将来 Pro 固有の差分が必要になればここで対応。
    """
    return content


def phase_1_regenerate(version: SemVer, *, dry_run: bool) -> tuple[set[Path], set[Path], set[Path]]:
    """release/pro/ を再生成。返り値: (added, modified, removed) の相対 Path セット。"""
    print("[Phase 1] release/pro/ 再生成")

    if RELEASE_PRO_TMP.exists():
        shutil.rmtree(RELEASE_PRO_TMP)
    RELEASE_PRO_TMP.mkdir(parents=True)

    try:
        # CHANGELOG / .gitkeep を先に退避 (上書きから守る)
        preserved = _preserve_changelogs(RELEASE_PRO_TMP / "docs")
        if preserved:
            print(f"  保全: {len(preserved)} 件の CHANGELOG ({', '.join(preserved)})")

        count = 0
        for rel in _iter_tracked_files():
            if should_exclude_path(rel):
                continue
            src = PROJECT_ROOT / rel
            if not src.is_file():
                continue

            dst_rel = _map_to_release_path(rel)
            dst = RELEASE_PRO_TMP / dst_rel

            # 既に保全コピー済みの CHANGELOG はスキップ
            if dst.exists() and is_changelog_filename(dst.name):
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            if rel.name == "pyproject.toml" and rel.parent == Path("."):
                dst.write_text(_rewrite_pyproject_for_release(src.read_text(encoding="utf-8")), encoding="utf-8")
            elif rel == Path("LICENSE-PRO"):
                # Pro EULA を主たる LICENSE として配置
                shutil.copy2(src, RELEASE_PRO_TMP / "LICENSE")
            else:
                shutil.copy2(src, dst)
            count += 1

        # release/pro/.gitignore は既存を流用 (source 側 .gitignore とは別物)
        existing_gitignore = RELEASE_PRO / ".gitignore"
        if existing_gitignore.is_file():
            shutil.copy2(existing_gitignore, RELEASE_PRO_TMP / ".gitignore")

        # release/pro 直下の README.md は既存 Pro 向けのものを優先
        existing_readme = RELEASE_PRO / "README.md"
        if existing_readme.is_file():
            shutil.copy2(existing_readme, RELEASE_PRO_TMP / "README.md")

        print(f"  コピー: {count} ファイル")

        added, modified, removed = _diff_trees(RELEASE_PRO_TMP, RELEASE_PRO)
        print(f"  diff: +{len(added)} ~{len(modified)} -{len(removed)}")

        if dry_run:
            _print_diff_summary(added, modified, removed)
            return added, modified, removed

        _apply_tree(RELEASE_PRO_TMP, RELEASE_PRO, added, modified, removed)
        return added, modified, removed
    finally:
        if RELEASE_PRO_TMP.exists():
            shutil.rmtree(RELEASE_PRO_TMP)


def _map_to_release_path(source_rel: Path) -> Path:
    """source の相対パスを release/pro 内の相対パスへ変換。

    - ``LICENSE-PRO`` → ``LICENSE`` (主たる EULA)
    - ``LICENSE-FREE`` はそのまま (= Free 部分の Apache-2.0 帰属を維持)
    """
    if source_rel == Path("LICENSE-PRO"):
        return Path("LICENSE")
    return source_rel


def _diff_trees(new_root: Path, old_root: Path) -> tuple[set[Path], set[Path], set[Path]]:
    new_files = {p.relative_to(new_root) for p in new_root.rglob("*") if p.is_file()}
    old_files = (
        {p.relative_to(old_root) for p in old_root.rglob("*") if p.is_file()}
        if old_root.is_dir()
        else set()
    )
    added = new_files - old_files
    removed = old_files - new_files
    modified: set[Path] = set()
    for rel in new_files & old_files:
        if not filecmp.cmp(new_root / rel, old_root / rel, shallow=False):
            modified.add(rel)
    return added, modified, removed


def _print_diff_summary(added: set[Path], modified: set[Path], removed: set[Path]) -> None:
    def _show(label: str, items: set[Path], limit: int = 20) -> None:
        sorted_items = sorted(items)
        for p in sorted_items[:limit]:
            print(f"  [{label}] {p.as_posix()}")
        if len(sorted_items) > limit:
            print(f"  [{label}] ... ({len(sorted_items) - limit} more)")

    _show("ADD", added)
    _show("MOD", modified)
    _show("DEL", removed)


def _apply_tree(new_root: Path, old_root: Path, added: set[Path], modified: set[Path], removed: set[Path]) -> None:
    for rel in removed:
        target = old_root / rel
        if target.exists():
            target.unlink()
    for rel in added | modified:
        dst = old_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(new_root / rel, dst)
    for d in sorted(
        (p for p in old_root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            d.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Phase 1.5: CHANGELOG 自動追記 (Pro Edition)
# ---------------------------------------------------------------------------

NEW_CHANGELOG_TEMPLATE = """# Changelog — {minor_key}.x シリーズ (Pro Edition)

evoref Pro Edition {minor_key}.x シリーズの変更履歴。

フォーマットは [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に準拠し、
バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に従う。

---
"""

PREFIX_CATEGORY: dict[str, str] = {
    "feat": "Added",
    "feature": "Added",
    "fix": "Fixed",
    "refactor": "Changed",
    "perf": "Changed",
    "docs": "Documentation",
}

PREFIX_SKIP: frozenset[str] = frozenset({"chore", "test", "style", "ci", "build"})


@dataclass(frozen=True)
class CommitEntry:
    sha: str
    subject: str
    prefix: str
    body: str


def _classify_prefix(subject: str) -> tuple[str, str]:
    """``feat(scope): body`` から ``("feat", "body")`` を抽出。"""
    m = re.match(r"^([a-zA-Z]+)(?:\([^)]*\))?!?:\s*(.+)$", subject)
    if not m:
        return "", subject
    return m.group(1).lower(), m.group(2)


def _collect_commits(range_spec: str) -> list[CommitEntry]:
    """``git log <range>`` を Pro 配布対象 path のみで実行し CommitEntry を返す。

    Pro 配布は Free + Pro 一式のため、Free 関連 path に加えて Pro 関連 path
    (``backend/pro/`` / ``frontend/src/lib/pro/``) も追跡する。
    """
    paths = [
        "backend/free/",
        "backend/pro/",
        "backend/factory/",
        "backend/schemas/",
        "backend/i18n/",
        "backend/io/",
        "backend/export/",
        "backend/extraction/",
        "frontend/src/lib/free/",
        "frontend/src/lib/pro/",
        "frontend/src/lib/i18n/",
        "frontend/src/lib/edition.ts",
        "scripts/",
        "models/profiles/",
    ]
    cmd = [
        "git",
        "-C",
        str(PROJECT_ROOT),
        "log",
        range_spec,
        "--no-merges",
        "--format=%H%x09%s",
        "--",
        *paths,
    ]
    try:
        out = _run(cmd).stdout
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] git log 失敗: {e.stderr.strip()}")
        return []

    entries: list[CommitEntry] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        sha, subject = line.split("\t", 1)
        prefix, body = _classify_prefix(subject)
        body_clean = re.sub(r"\s*\(#\d+\)\s*$", "", body).strip()
        entries.append(CommitEntry(sha=sha, subject=subject, prefix=prefix, body=body_clean))
    return entries


def _find_prev_bump_commit(prev_version: str) -> str | None:
    pattern = rf"bump version .* → {re.escape(prev_version)}"
    try:
        out = _run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "log",
                "--all",
                "--grep",
                pattern,
                "-E",
                "--format=%H",
                "-1",
            ]
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None
    return out or None


def _extract_prev_section_version(changelog_text: str, current: SemVer) -> str | None:
    """同 CHANGELOG 内で現セクションより新しくない最初の ``## [X.Y.Z]`` を返す。"""
    sections = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog_text, re.MULTILINE)
    current_tuple = (current.major, current.minor, current.patch)
    for s in sections:
        try:
            sv = parse_version(s)
        except ValueError:
            continue
        if (sv.major, sv.minor, sv.patch) < current_tuple:
            return s
    return None


def _determine_commit_range(version: SemVer, changelog_text: str) -> tuple[str, str | None]:
    """git log の range と「採用した前バージョン」のタプルを返す。"""
    prev_version = _extract_prev_section_version(changelog_text, version)
    if prev_version:
        commit = _find_prev_bump_commit(prev_version)
        if commit:
            return f"{commit}..HEAD", prev_version
    return "HEAD", prev_version


def _categorise(entries: list[CommitEntry]) -> dict[str, list[CommitEntry]]:
    buckets: dict[str, list[CommitEntry]] = {}
    for e in entries:
        if e.prefix in PREFIX_SKIP:
            continue
        category = PREFIX_CATEGORY.get(e.prefix, "Other")
        buckets.setdefault(category, []).append(e)
    return buckets


def _render_initial_section(version: SemVer, date_str: str) -> str:
    """``--initial-release`` 指定時のセクション (commit 履歴を載せない定型文)。"""
    lines = [
        f"## [{version.major}.{version.minor}.{version.patch}] - {date_str}",
        "",
        "evoref Pro Edition の初回リリース。",
        "",
        "このバージョン以降の変更履歴は本ファイルに追記していく。",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _render_section(version: SemVer, date_str: str, buckets: dict[str, list[CommitEntry]]) -> str:
    lines: list[str] = []
    lines.append(f"## [{version.major}.{version.minor}.{version.patch}] - {date_str}")
    lines.append("")
    order = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security", "Documentation", "Other"]
    any_content = False
    for cat in order:
        items = buckets.get(cat, [])
        if not items:
            continue
        any_content = True
        lines.append(f"### {cat}")
        lines.append("")
        for e in items:
            lines.append(f"- {e.body}")
        lines.append("")
    if not any_content:
        lines.append("(自動抽出された Pro 配布関連の変更なし。手動で記入してください。)")
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _insert_section(changelog_text: str, section_md: str) -> str:
    """ヘッダ + 説明 + 最初の ``---`` 区切りの直後に section を挿入。"""
    marker = re.search(r"^---\s*$", changelog_text, re.MULTILINE)
    if marker is None:
        return section_md + "\n" + changelog_text
    insert_at = marker.end()
    head = changelog_text[:insert_at]
    tail = changelog_text[insert_at:]
    if not head.endswith("\n"):
        head += "\n"
    if not tail.startswith("\n"):
        tail = "\n" + tail
    return head + "\n" + section_md + tail.lstrip("\n")


def phase_1_5_changelog(
    version: SemVer,
    *,
    dry_run: bool,
    entries_path: Path | None,
    initial_release: bool = False,
) -> bool:
    """CHANGELOG 自動追記。追記された (or 追記予定) なら True、skip なら False。"""
    print("[Phase 1.5] CHANGELOG 自動追記 (Pro Edition)")
    docs_dir = RELEASE_PRO / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / version.changelog_filename
    print(f"  target: docs/{version.changelog_filename}")

    if target.exists():
        text = target.read_text(encoding="utf-8")
    else:
        text = NEW_CHANGELOG_TEMPLATE.format(minor_key=version.minor_key)
        print(f"  [INFO] 新規ファイル作成 ({version.minor_key}.x シリーズ初回)")

    section_re = re.compile(
        rf"^## \[{re.escape(version.raw)}\]", re.MULTILINE
    )
    if section_re.search(text):
        print(f"  [SKIP] {version.section_header} 既に存在")
        return False

    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    if initial_release:
        print("  mode: --initial-release (commit 履歴は載せない)")
        section = _render_initial_section(version, date_str)
    elif entries_path is not None:
        body = entries_path.read_text(encoding="utf-8").strip()
        section = (
            f"## [{version.major}.{version.minor}.{version.patch}] - {date_str}\n\n"
            f"{body}\n\n---\n"
        )
    else:
        range_spec, prev = _determine_commit_range(version, text)
        print(f"  range: {range_spec} (prev section: {prev or 'none'})")
        commits = _collect_commits(range_spec)
        print(f"  commits collected: {len(commits)}")
        buckets = _categorise(commits)
        section = _render_section(version, date_str, buckets)

    new_text = _insert_section(text, section)

    if dry_run:
        print("  --- 追記予定 (dry-run) ---")
        for line in section.splitlines():
            print(f"  | {line}")
        print("  ---------------------------")
        return True

    target.write_text(new_text, encoding="utf-8")
    print(f"  [OK] 追記: {version.section_header}")
    return True


# ---------------------------------------------------------------------------
# Phase 2: 静的検証 (Pro 配布用)
# ---------------------------------------------------------------------------

# Pro 配布では Pro 参照は許可される。代わりに Develop 限定コード参照を検出する。
DEVELOP_IMPORT_PATTERNS_FRONTEND: tuple[re.Pattern[str], ...] = (
    re.compile(r"""from\s+['"]\$lib/develop"""),
    re.compile(r"""import\s+.+from\s+['"]\$lib/develop"""),
)

# 「Pro/Develop の有無によって挙動を切り替える」 ガード識別子の集合。
GUARD_NAMES: frozenset[str] = frozenset(
    {
        "is_pro",
        "pro_available",
        "current_edition",
        "is_develop",
        "develop_available",
    }
)


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _is_typecheck_test(node: ast.expr) -> bool:
    if isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING":
        return True
    return False


def _expression_mentions_guard(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in GUARD_NAMES:
            return True
        if isinstance(n, ast.Attribute):
            base = n.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in {"Edition", "edition"}:
                return True
            if n.attr in GUARD_NAMES:
                return True
    return False


def _try_catches_importerror(node: ast.Try) -> bool:
    for handler in node.handlers:
        ht = handler.type
        if ht is None:
            return True
        candidates = ht.elts if isinstance(ht, ast.Tuple) else [ht]
        for t in candidates:
            name = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else None
            if name in {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}:
                return True
    return False


def _is_guarded(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """``node`` が edition gating ブロックの内側にあるかを判定。"""
    cur: ast.AST = node
    while id(cur) in parents:
        parent = parents[id(cur)]
        if isinstance(parent, ast.If):
            in_body = any(cur is stmt for stmt in parent.body) or any(cur is stmt for stmt in parent.orelse)
            if in_body and (_is_typecheck_test(parent.test) or _expression_mentions_guard(parent.test)):
                return True
        elif isinstance(parent, ast.Try):
            in_body = any(cur is stmt for stmt in parent.body)
            if in_body and _try_catches_importerror(parent):
                return True
        cur = parent
    return False


def _scan_python_develop_imports(content: str, rel_path: str) -> list[str]:
    """Python ファイルから unguarded な backend.develop import を検出。"""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    parents = _build_parent_map(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("backend.develop"):
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("backend.develop"):
                    modules.append(alias.name)
        if modules and not _is_guarded(node, parents):
            hits.append(f"{rel_path}:{node.lineno}: unguarded import {modules[0]}")
    return hits


# Pro 配布で必須のファイル群 (Free + Pro 両方の入口を含む)。
REQUIRED_FILES: tuple[str, ...] = (
    "LICENSE",
    "LICENSE-FREE",
    "NOTICE.md",
    "README.md",
    "pyproject.toml",
    "evoref_cli.py",
    ".gitignore",
    "backend/__init__.py",
    "backend/free/__init__.py",
    "backend/free/__version__.py",
    "backend/pro/__init__.py",
    "backend/pro/__version__.py",
    "backend/requirements.txt",
    "backend/requirements-pro.txt",
    "frontend/package.json",
    "config.pro.yaml.example",
)


def phase_2_static_checks() -> None:
    print("[Phase 2] 静的検証 (Pro 配布)")
    if not RELEASE_PRO.is_dir():
        print(f"  [FAIL] {RELEASE_PRO} が無い", file=sys.stderr)
        sys.exit(EXIT_VALIDATION)

    missing = [f for f in REQUIRED_FILES if not (RELEASE_PRO / f).is_file()]
    if missing:
        print(f"  [FAIL] 必須ファイル不足: {missing}", file=sys.stderr)
        sys.exit(EXIT_VALIDATION)

    develop_hits: list[str] = []
    frontend_suffixes = {".ts", ".tsx", ".js", ".mjs", ".svelte"}
    for f in RELEASE_PRO.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(RELEASE_PRO).as_posix()

        # frontend routes/ は動的 import が許可されているため除外
        if rel.startswith("frontend/src/routes/"):
            continue

        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if f.suffix == ".py":
            develop_hits.extend(_scan_python_develop_imports(content, rel))
        elif f.suffix in frontend_suffixes:
            for pat in DEVELOP_IMPORT_PATTERNS_FRONTEND:
                m = pat.search(content)
                if m:
                    line_no = content[: m.start()].count("\n") + 1
                    develop_hits.append(f"{rel}:{line_no}: {pat.pattern}")
                    break

    if develop_hits:
        print(f"  [FAIL] Develop モジュール参照検出 ({len(develop_hits)} 件):", file=sys.stderr)
        for h in develop_hits[:30]:
            print(f"    {h}", file=sys.stderr)
        sys.exit(EXIT_VALIDATION)

    docs_dir = RELEASE_PRO / "docs"
    if docs_dir.is_dir():
        offenders = [
            p.name
            for p in docs_dir.iterdir()
            if p.is_file() and re.match(r"^[cefpa]_\d", p.name) and p.suffix == ".md"
        ]
        if offenders:
            print(f"  [FAIL] 内部設計書が docs/ に混入: {offenders}", file=sys.stderr)
            sys.exit(EXIT_VALIDATION)

    print("  [OK] 必須ファイル / Develop 混入 / 設計書混入チェック通過")


# ---------------------------------------------------------------------------
# Phase 3: evoref-pro へ rsync
# ---------------------------------------------------------------------------


def _ensure_unique_branch(target: Path, base: str) -> str:
    """ローカル/リモート両方で衝突しない branch 名を返す。"""
    name = base
    suffix = 2
    while True:
        local = _run(
            ["git", "-C", str(target), "rev-parse", "--verify", name],
            check=False,
        )
        remote = _run(
            ["git", "-C", str(target), "ls-remote", "--heads", "origin", name],
            check=False,
        )
        if local.returncode != 0 and not remote.stdout.strip():
            return name
        name = f"{base}-{suffix}"
        suffix += 1


def phase_3_sync_to_pro(target: Path, branch: str) -> str:
    print("[Phase 3] evoref-pro へ rsync")
    _run(["git", "-C", str(target), "fetch", "origin"])
    _run(["git", "-C", str(target), "checkout", "main"])
    _run(["git", "-C", str(target), "pull", "--ff-only", "origin", "main"])

    branch = _ensure_unique_branch(target, branch)
    _run(["git", "-C", str(target), "checkout", "-b", branch])

    # .git 以外を削除
    for child in target.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    # release/pro/ の中身をコピー
    for src in RELEASE_PRO.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(RELEASE_PRO)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    _run(["git", "-C", str(target), "add", "-A"])
    status = _run(["git", "-C", str(target), "status", "--short"]).stdout
    print(f"  branch: {branch}")
    print(f"  staged changes:\n{status}")
    return branch


# ---------------------------------------------------------------------------
# Phase 4: commit + push + PR
# ---------------------------------------------------------------------------


def _has_staged_changes(target: Path) -> bool:
    result = _run(
        ["git", "-C", str(target), "diff", "--cached", "--quiet"],
        check=False,
    )
    return result.returncode != 0


def phase_4_commit_and_pr(
    target: Path,
    branch: str,
    version: SemVer,
    short_sha: str,
    full_sha: str,
    *,
    no_pr: bool,
    initial_release: bool,
) -> str | None:
    print("[Phase 4] commit + push + PR")
    if not _has_staged_changes(target):
        print("  [SKIP] staged 変更なし。同期は不要です。")
        sys.exit(EXIT_NO_CHANGES)

    if initial_release:
        msg = f"chore: Pro Edition v{version.raw} リリース\n"
        title = f"chore: Pro Edition v{version.raw} リリース"
    else:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        msg = (
            f"chore: Pro Edition v{version.raw} リリース\n"
            f"\n"
            f"evoref-dev commit: {full_sha}\n"
            f"同期日時 (UTC): {timestamp}\n"
        )
        title = f"chore: Pro Edition v{version.raw} リリース"

    _run(["git", "-C", str(target), "commit", "-m", msg])
    _run(["git", "-C", str(target), "push", "-u", "origin", branch])

    if no_pr:
        print("  [SKIP] --no-pr 指定により PR 作成をスキップ")
        return None

    if initial_release:
        body = _build_initial_pr_body(version)
    else:
        diff_stat = _run(["git", "-C", str(target), "diff", "--stat", "main...HEAD"]).stdout
        diff_summary = "\n".join(diff_stat.splitlines()[:50])
        body = _build_pr_body(version, short_sha, full_sha, diff_summary)

    out = _run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            DEFAULT_REMOTE_REPO,
            "--base",
            "main",
            "--head",
            branch,
            "--title",
            title,
            "--body",
            body,
        ]
    ).stdout.strip()
    print(f"  [OK] PR 作成: {out}")
    return out


def _build_initial_pr_body(version: SemVer) -> str:
    return f"""## Summary
- Pro Edition v{version.raw} 初回リリース (Free + Pro 一式同梱)
"""


def _build_pr_body(version: SemVer, short_sha: str, full_sha: str, diff_summary: str) -> str:
    return f"""## Summary
- Pro Edition v{version.raw} リリース (Free + Pro 一式同梱)

## 変更ファイル概要
```
{diff_summary}
```
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-pr", action="store_true")
    p.add_argument("--changelog-only", action="store_true")
    p.add_argument("--skip-changelog", action="store_true")
    p.add_argument("--changelog-entries", type=Path, default=None)
    p.add_argument(
        "--initial-release",
        action="store_true",
        help="commit 履歴を載せず初回リリース用の定型セクションを書き込む",
    )
    p.add_argument("--target-dir", type=str, default=str(DEFAULT_TARGET))
    p.add_argument("--branch-suffix", type=str, default=None)
    p.add_argument("--version", type=str, default=None)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.initial_release and args.changelog_entries is not None:
        print(
            "[ERROR] --initial-release と --changelog-entries は同時指定できません",
            file=sys.stderr,
        )
        return EXIT_PREFLIGHT

    acquire_lock()
    try:
        switch = switch_to_main(allow_dirty=args.allow_dirty)
        target_dir = Path(args.target_dir).resolve()
        target_touched = False
        try:
            version, short_sha, full_sha = phase_0_preflight(args)

            if args.changelog_only:
                phase_1_5_changelog(
                    version,
                    dry_run=args.dry_run,
                    entries_path=args.changelog_entries,
                    initial_release=args.initial_release,
                )
                return EXIT_OK

            phase_1_regenerate(version, dry_run=args.dry_run)

            if not args.skip_changelog:
                phase_1_5_changelog(
                    version,
                    dry_run=args.dry_run,
                    entries_path=args.changelog_entries,
                    initial_release=args.initial_release,
                )

            if args.dry_run:
                print("[done] dry-run 完了。実際の変更は加えていません。")
                return EXIT_OK

            phase_2_static_checks()

            branch_base = args.branch_suffix or f"sync/dev-{short_sha}"
            target_touched = True
            branch = phase_3_sync_to_pro(target_dir, branch_base)

            pr_url = phase_4_commit_and_pr(
                target_dir,
                branch,
                version,
                short_sha,
                full_sha,
                no_pr=args.no_pr,
                initial_release=args.initial_release,
            )
            if pr_url:
                print(f"\nPR: {pr_url}")
            return EXIT_OK
        finally:
            if target_touched:
                restore_target_to_main(target_dir)
            restore_branch(switch)
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
