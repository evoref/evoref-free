"""バージョン文字列の一括 bump スクリプト

evoref のバージョン文字列は以下に分散しており、本スクリプトで一括更新する:

SSOT (Python ランタイムから参照される):
  - backend/free/__version__.py     (Free 配布物に必須)
  - backend/pro/__version__.py      (Pro 配布のみ)

派生 (SSOT を import するため自動追随):
  - pyproject.toml                  (dynamic = ["version"] で attr 参照)
  - backend/free/cli/main.py        (get_runtime_version() 経由)
  - backend/free/cli/chat_loop.py   (同上)
  - backend/free/api/schemas.py     (Pydantic デフォルトを __version__ から取る)

frontend は version 管理外。独自の版番号を持たず、実行時に ``/api/status`` の
backend 版 (``version=version_info.free``) を取得して表示する。本スクリプトは
frontend のファイルを一切書き換えない。

使い方:
  python scripts/bump_version.py 0.1.3
  python scripts/bump_version.py 0.2.0 --edition free
  python scripts/bump_version.py 0.2.0 --edition pro
  python scripts/bump_version.py 0.2.0 --edition all      # 既定
  python scripts/bump_version.py --dry-run 0.1.3

エディションを個別に切るとき (例: Pro だけ先行 release) は ``--edition``
で対象を絞る。``--edition all`` のときは Free / Pro を同じ値に揃える。

スキーマバージョン (``__schema_version__``) は本スクリプトでは触らない。
data layout 互換破壊時のみ手動で bump する (ユーザー承認必須)。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[.\-+][\w\.]+)?$")

# (path, regex, replacement-builder)
# 各エントリは「ファイル」「マッチ対象の正規表現 (group 1 が version 部)」
# 「new_version を受け取り group 1 を置き換えた行を返す関数」
# regex は単一行マッチ前提 (multiline 不要)。


def _py_version_module(path: Path):
    return (
        path,
        re.compile(r'^(__version__\s*=\s*)"([^"]+)"', re.MULTILINE),
        lambda new: f'\\g<1>"{new}"',
    )


EDITION_TARGETS: dict[str, list] = {
    "free": [
        _py_version_module(PROJECT_ROOT / "backend" / "free" / "__version__.py"),
    ],
    "pro": [
        _py_version_module(PROJECT_ROOT / "backend" / "pro" / "__version__.py"),
    ],
}


def _apply(target, new_version: str, *, dry_run: bool) -> tuple[bool, str | None]:
    """1 ファイルに対して書き換えを試みる

    Returns:
        (changed, old_version) — マッチが無ければ (False, None)。
    """
    path, pattern, repl_builder = target
    if not path.exists():
        return False, None
    text = path.read_text(encoding="utf-8")
    m = pattern.search(text)
    if not m:
        return False, None
    old = m.group(2)
    if old == new_version:
        return False, old
    new_text = pattern.sub(repl_builder(new_version), text, count=1)
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True, old


def _commit_bump(old_version: str, new_version: str, paths: list[Path]) -> int:
    """bump で変更したファイルだけを stage して bump コミットを作る。

    コミットメッセージは ``chore: bump version <old> → <new>``。
    release_free_sync.py の CHANGELOG 境界検出 (``bump version .* → <ver>``)
    がこのコミットを前バージョンの起点として参照するため、``version`` と
    ``→`` の間に旧バージョン文字列を必ず挟む。
    """
    msg = f"chore: bump version {old_version} → {new_version}"
    rels = [str(p.relative_to(PROJECT_ROOT)) for p in paths]
    try:
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "add", "--", *rels],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "commit", "-m", msg],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"error: git commit に失敗 ({e}). --no-commit で手動運用可", file=sys.stderr)
        return 1
    print(f"committed: {msg}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bump_version",
        description="evoref のバージョン文字列を一括 bump する",
    )
    parser.add_argument(
        "new_version",
        help="新しいバージョン (例: 0.1.3, 0.2.0-rc1)",
    )
    parser.add_argument(
        "--edition",
        choices=["all", "free", "pro"],
        default="all",
        help="bump 対象エディション (既定: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ファイルを書き換えず、対象と差分のみ表示",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="変更後の `chore: bump version → X.Y.Z` 自動コミットを抑制する",
    )
    args = parser.parse_args()

    if not VERSION_RE.match(args.new_version):
        print(
            f"error: invalid version {args.new_version!r} "
            "(expected NN.NN.NN or NN.NN.NN-suffix)",
            file=sys.stderr,
        )
        return 2

    targets: list = []
    editions = ["free", "pro"] if args.edition == "all" else [args.edition]
    for ed in editions:
        targets.extend(EDITION_TARGETS[ed])

    print(
        f"bump_version: new={args.new_version} edition={args.edition} "
        f"dry_run={args.dry_run}",
    )

    changed_count = 0
    changed_paths: list[Path] = []
    prev_version: str | None = None
    missing: list[Path] = []
    for tgt in targets:
        path = tgt[0]
        changed, old = _apply(tgt, args.new_version, dry_run=args.dry_run)
        rel = path.relative_to(PROJECT_ROOT)
        if not path.exists():
            missing.append(path)
            print(f"  [skip] {rel} (not present)")
        elif old is None:
            print(f"  [skip] {rel} (no version field matched)")
        elif not changed:
            print(f"  [skip] {rel} (already {old})")
        else:
            changed_count += 1
            changed_paths.append(path)
            if prev_version is None:
                prev_version = old
            print(f"  [bump] {rel}  {old} -> {args.new_version}")

    if missing:
        print(
            f"note: {len(missing)} target(s) missing (likely Pro "
            "not installed) — skipped without error",
        )

    print(f"done: {changed_count} file(s) changed")

    if not args.dry_run and not args.no_commit and changed_paths:
        return _commit_bump(prev_version or "?", args.new_version, changed_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
