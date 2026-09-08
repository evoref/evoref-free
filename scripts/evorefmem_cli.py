#!/usr/bin/env python
"""EvorefMem 運用 CLI

``<memory_dir>/semantic/`` 配下の運用操作を人手で行うためのツール.

## サブコマンド

    python scripts/evorefmem_cli.py init                             # 初期化 (init_evorefmem 委譲)
    python scripts/evorefmem_cli.py inspect [--scope SCOPE] [--json] # 統計表示
    python scripts/evorefmem_cli.py verify  [--scope SCOPE] [--json] # 整合性検査
    python scripts/evorefmem_cli.py purge-private [--all-curated] [--apply]  # private 由来の索引を掃除
    python scripts/evorefmem_cli.py migrate [--to V] [--apply] [--list]
    python scripts/evorefmem_cli.py export PATH                       # tar.gz バックアップ
    python scripts/evorefmem_cli.py import PATH [--apply]             # リストア

破壊的操作 (``migrate`` / ``import`` / ``purge-private``) はデフォルトで
dry-run。``--apply`` で実行する。

``compact`` / ``rebuild-indices`` / ``migrate-embedding`` / ``reembed-facts``
は撤去した — 事象ログの畳み込み・転置索引・埋め込みはすべて sleep-time の
snapshot 生成が担うようになったため (c_16 §5.3 / §6)。埋め込みモデルを替えた
ときは次の版で全件が作り直される。

多重起動防止のため ``local/.evorefmem_cli.lock`` を PID で占有する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.free.memory.init_evorefmem import (  # noqa: E402
    SCHEMA_VERSION,
    initialize_evorefmem,
    read_schema_version,
)
from backend.free.memory.semantic.cli import (  # noqa: E402
    CliLockError,
    acquire_cli_lock,
    release_cli_lock,
)
from backend.free.memory.semantic.cli._paths import (  # noqa: E402
    resolve_cli_paths,
)
from backend.free.memory.semantic.cli.export_import_cmd import (  # noqa: E402
    format_export_report_text,
    format_import_report_text,
    run_export,
    run_import,
)
from backend.free.memory.semantic.cli.inspect_cmd import (  # noqa: E402
    format_report_text as _inspect_fmt,
    run_inspect,
)
from backend.free.memory.semantic.cli.migrate_cmd import (  # noqa: E402
    format_report_text as _migrate_fmt,
    list_registered_migrations,
    run_migrate,
)
from backend.free.memory.semantic.cli.purge_private_cmd import (  # noqa: E402
    run_purge_private,
)
from backend.free.memory.semantic.cli.verify_cmd import (  # noqa: E402
    format_report_text as _verify_fmt,
    run_verify,
)


# ──────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────


def _cmd_init(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    print(f"Initializing EvorefMem (schema v{SCHEMA_VERSION})...")
    print(f"  memory_dir            : {paths.memory_dir}")
    print(f"  prompts_dir           : {paths.prompts_dir}")
    print(f"  migration_archive_dir : {paths.migration_archive_dir}")
    result = initialize_evorefmem(
        paths.memory_dir, paths.prompts_dir, paths.migration_archive_dir,
    )
    print()
    print("Done.")
    print(f"  backed up : {len(result.backed_up)} files")
    print(f"  deleted   : {len(result.deleted)} entries")
    print(f"  created   : {len(result.created)} dirs")
    print(f"  gc removed: {len(result.gc_removed)} entries (>30d old)")
    print(f"  marker    : {result.schema_marker}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    report = run_inspect(
        paths.memory_dir,
        top_subjects=args.top_subjects,
        scope_filter=args.scope,
    )
    if args.json:
        print(report.to_json())
    else:
        print(_inspect_fmt(report))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    report = run_verify(paths.memory_dir, scope_filter=args.scope)
    if args.json:
        print(report.to_json())
    else:
        print(_verify_fmt(report))
    return report.exit_code()


def _cmd_purge_private(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    report = run_purge_private(
        paths.memory_dir,
        paths.migration_archive_dir,
        apply=args.apply,
        all_curated=args.all_curated,
        scope_filter=args.scope,
        since=args.since,
        until=args.until,
        sessions=args.session or None,
    )
    if args.json:
        print(report.to_json())
    else:
        print(_purge_private_fmt(report))
    return 0


def _purge_private_fmt(report) -> str:
    lines = [
        f"memory_dir: {report.memory_dir}",
        f"mode      : {report.mode}"
        + ("" if report.notes_available else "  (ノート未読込: 厳密照合は無効)"),
        f"candidates: {len(report.candidates)}",
    ]
    by_reason: dict[str, int] = {}
    for c in report.candidates:
        by_reason[c.reason] = by_reason.get(c.reason, 0) + 1
    for reason, count in sorted(by_reason.items()):
        lines.append(f"  - {reason}: {count}")
    for c in report.candidates[:20]:
        lines.append(f"    [{c.reason}] {c.scope} {c.subject}  {c.object_preview!r}")
    if len(report.candidates) > 20:
        lines.append(f"    ... (他 {len(report.candidates) - 20} 件)")
    if report.applied:
        lines.append(f"retracted : {report.deleted}")
        lines.append(f"notes 再生成待ちへ戻した: {report.notes_unmarked}")
        lines.append(f"backup    : {report.backup_path}")
    else:
        lines.append("(dry-run: 削除するには --apply を付ける)")
    return "\n".join(lines)




def _cmd_migrate(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    if args.list:
        registered = list_registered_migrations()
        payload = [
            {
                "class_name": r.class_name,
                "from_version": r.from_version,
                "to_version": r.to_version,
                "component": r.component,
            }
            for r in registered
        ]
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if not registered:
                print("(no migrations registered)")
            else:
                for r in registered:
                    print(
                        f"  - {r.class_name:30} {r.from_version} -> "
                        f"{r.to_version} ({r.component})",
                    )
        return 0

    target = args.to if args.to is not None else SCHEMA_VERSION
    report = run_migrate(
        paths.memory_dir,
        paths.migration_archive_dir,
        target_version=target,
        apply=args.apply,
    )
    if args.json:
        print(report.to_json())
    else:
        print(_migrate_fmt(report))
    return 1 if report.error else 0





def _cmd_export(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    report = run_export(paths.memory_dir, Path(args.path))
    if args.json:
        print(report.to_json())
    else:
        print(format_export_report_text(report))
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    report = run_import(
        paths.memory_dir,
        Path(args.path),
        paths.migration_archive_dir,
        apply=args.apply,
        allow_version_mismatch=args.allow_version_mismatch,
    )
    if args.json:
        print(report.to_json())
    else:
        print(format_import_report_text(report))
    return 1 if report.error else 0


# 後方互換: scripts/init_evorefmem.py は --check を verify にマップする
def _cmd_check_compat(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    current = read_schema_version(paths.memory_dir)
    print(f"memory_dir            : {paths.memory_dir}")
    print(f"prompts_dir           : {paths.prompts_dir}")
    print(f"migration_archive_dir : {paths.migration_archive_dir}")
    print(f"expected version      : {SCHEMA_VERSION}")
    print(f"actual version        : {current}")
    if current == SCHEMA_VERSION:
        print("status             : OK")
        return 0
    print("status             : MISMATCH (run without --check to initialize)")
    return 1


# ──────────────────────────────────────────────────────────────────────────
# argparse
# ──────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evorefmem_cli",
        description="EvorefMem 運用 CLI"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="人間可読形式ではなく JSON で出力する",
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="多重起動防止ロックを取得しない (テスト用; 通常は使わない)",
    )

    sub = parser.add_subparsers(
        dest="subcommand", required=True, metavar="SUBCOMMAND",
    )

    # init
    sp = sub.add_parser(
        "init",
        help="EvorefMem を初期化する (init_evorefmem 委譲)",
    )
    sp.set_defaults(func=_cmd_init)

    # inspect
    sp = sub.add_parser("inspect", help="統計情報を表示する (副作用なし)")
    sp.add_argument(
        "--scope", default=None,
        help='特定 scope に限定 (例: "global" / "project:my_proj")',
    )
    sp.add_argument(
        "--top-subjects", type=int, default=10,
        help="subject 上位件数 (デフォルト 10)",
    )
    sp.set_defaults(func=_cmd_inspect)

    # verify
    sp = sub.add_parser("verify", help="整合性を検査する (副作用なし)")
    sp.add_argument("--scope", default=None, help="特定 scope に限定")
    sp.set_defaults(func=_cmd_verify)

    # purge-private
    sp = sub.add_parser(
        "purge-private",
        help=(
            "private セッション由来のキュレーターファクトを掃除する "
            "(デフォルト dry-run)"
        ),
    )
    sp.add_argument("--scope", default=None, help="特定 scope に限定")
    sp.add_argument(
        "--all-curated", action="store_true",
        help=(
            "mem.world.assertion.* / idx.{url,command}.* を丸ごと候補にする。"
            "取りこぼしゼロだが正当な索引も一度消える (ノートのマーカーを戻すので"
            "次の Full で再生成される。失うのは exec_count 等の統計のみ)"
        ),
    )
    sp.add_argument(
        "--since", type=float, default=None,
        help="created_at がこの epoch 秒以降のものを候補にする",
    )
    sp.add_argument(
        "--until", type=float, default=None,
        help="created_at がこの epoch 秒以前のものを候補にする",
    )
    sp.add_argument(
        "--session", action="append", default=[],
        help="この session_id 由来のものを候補にする (複数指定可)",
    )
    sp.add_argument(
        "--apply", action="store_true",
        help="実際に削除する (未指定時は dry-run)",
    )
    sp.set_defaults(func=_cmd_purge_private)

    # migrate
    sp = sub.add_parser(
        "migrate",
        help="SchemaMigrator を駆動 (デフォルト dry-run / --list で一覧)",
    )
    sp.add_argument(
        "--to", type=int, default=None,
        help=f"目標 schema_version (省略時は現行値 = {SCHEMA_VERSION})",
    )
    sp.add_argument(
        "--apply", action="store_true",
        help="実際に migrate を実行する",
    )
    sp.add_argument(
        "--list", action="store_true",
        help="登録 Migration を列挙するだけ (memory_dir には触れない)",
    )
    sp.set_defaults(func=_cmd_migrate)

    # export
    sp = sub.add_parser(
        "export",
        help="semantic/ 全体を tar アーカイブに出力する",
    )
    sp.add_argument(
        "path",
        help="出力パス (.tar.gz / .tar.zst / .tar)",
    )
    sp.set_defaults(func=_cmd_export)

    # import
    sp = sub.add_parser(
        "import",
        help="tar アーカイブから semantic/ を復元する (デフォルト dry-run)",
    )
    sp.add_argument(
        "path",
        help="入力アーカイブパス",
    )
    sp.add_argument(
        "--apply", action="store_true",
        help="実際に復元する (未指定時は dry-run)",
    )
    sp.add_argument(
        "--allow-version-mismatch", action="store_true",
        help="archive 内 schema_version が現行と異なっても import を許可",
    )
    sp.set_defaults(func=_cmd_import)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # subcommand に応じてロックを取得
    acquired = False
    if not args.no_lock:
        try:
            acquire_cli_lock()
            acquired = True
        except CliLockError as exc:
            print(f"[evorefmem_cli] {exc}", file=sys.stderr)
            return 2
    try:
        return args.func(args)
    finally:
        if acquired:
            release_cli_lock()


if __name__ == "__main__":
    raise SystemExit(main())
