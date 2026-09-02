#!/usr/bin/env python
"""EvorefMem 運用 CLI

``<memory_dir>/semantic/`` 配下の運用操作を人手で行うためのツール.

## サブコマンド

    python scripts/evorefmem_cli.py init                             # 初期化 (init_evorefmem 委譲)
    python scripts/evorefmem_cli.py inspect [--scope SCOPE] [--json] # 統計表示
    python scripts/evorefmem_cli.py verify  [--scope SCOPE] [--json] # 整合性検査
    python scripts/evorefmem_cli.py purge-private [--all-curated] [--apply]  # private 由来の索引を掃除
    python scripts/evorefmem_cli.py compact [--apply] [--scope S]    # facts.jsonl 圧縮
    python scripts/evorefmem_cli.py rebuild-indices [--apply]        # .idx 再生成
    python scripts/evorefmem_cli.py migrate [--to V] [--apply] [--list]
    python scripts/evorefmem_cli.py migrate-embedding --to MODEL_ID --dim N [--apply]
    python scripts/evorefmem_cli.py export PATH                       # tar.gz バックアップ
    python scripts/evorefmem_cli.py import PATH [--apply]             # リストア

破壊的操作 (``migrate`` / ``compact`` / ``rebuild-indices`` / ``import`` /
``migrate-embedding`` / ``purge-private``) はデフォルトで dry-run。``--apply`` で実行する。

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
from backend.free.memory.semantic.cli.compact_cmd import (  # noqa: E402
    format_report_text as _compact_fmt,
    run_compact,
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
from backend.free.memory.semantic.cli.migrate_embedding_cmd import (  # noqa: E402
    format_report_text as _migrate_emb_fmt,
    run_migrate_embedding,
)
from backend.free.memory.semantic.cli.reembed_facts_cmd import (  # noqa: E402
    format_report_text as _reembed_fmt,
    run_reembed_facts,
)
from backend.free.memory.semantic.cli.purge_private_cmd import (  # noqa: E402
    run_purge_private,
)
from backend.free.memory.semantic.cli.rebuild_indices_cmd import (  # noqa: E402
    format_report_text as _rebuild_fmt,
    run_rebuild_indices,
)
from backend.free.memory.semantic.cli.verify_cmd import (  # noqa: E402
    format_report_text as _verify_fmt,
    run_verify,
)
from backend.free.memory.semantic.manifest import (  # noqa: E402
    normalize_embedding_model_id,
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
        + ("" if report.notes_available else "  (STM ノート未読込: 厳密照合は無効)"),
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
        lines.append(f"deleted   : {report.deleted}")
        lines.append(f"notes 再生成待ちへ戻した: {report.notes_unmarked}")
        lines.append(f"backup    : {report.backup_path}")
    else:
        lines.append("(dry-run: 削除するには --apply を付ける)")
    return "\n".join(lines)


def _cmd_compact(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    report = run_compact(
        paths.memory_dir,
        paths.migration_archive_dir,
        apply=args.apply,
        scope_filter=args.scope,
    )
    if args.json:
        print(report.to_json())
    else:
        print(_compact_fmt(report))
    return 0


def _cmd_rebuild_indices(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    report = run_rebuild_indices(
        paths.memory_dir,
        paths.migration_archive_dir,
        apply=args.apply,
        scope_filter=args.scope,
    )
    if args.json:
        print(report.to_json())
    else:
        print(_rebuild_fmt(report))
    return 0


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


def _cmd_migrate_embedding(args: argparse.Namespace) -> int:
    paths = resolve_cli_paths()
    report = run_migrate_embedding(
        paths.memory_dir,
        paths.migration_archive_dir,
        new_model_id=args.to,
        new_dim=args.dim,
        normalized=not args.not_normalized,
        apply=args.apply,
        create_dirs=not args.no_create_dirs,
    )
    if args.json:
        print(report.to_json())
    else:
        print(_migrate_emb_fmt(report))
    return 1 if report.error else 0


def _build_http_embed_fn(
    host: str,
    port: int,
    doc_template: str,
    *,
    batch: int = 32,
    timeout: int = 60,
):
    """live llama-embed サーバ (OAI ``/v1/embeddings``) へ問い合わせる embed_fn.

    生の object テキストに ``doc_template`` (``document: {query}`` 等) を適用し、
    L2 正規化したベクトル列を返す (保存ベクトルは単位長のため正規化を合わせる)。
    ``doc_template`` が空文字列 (Qwen3-Embedding 等の doc 側 prefix なしモデル)
    なら素のテキストをそのまま埋め込む — ``LlamaCppEmbedder`` の fast-path と
    同じ扱い。空テンプレに ``.format()`` を適用すると全文書が空文字列に潰れ、
    embed サーバはエラーを返さないため silent corruption になる。
    """
    import json as _json
    import math
    import urllib.request

    url = f"http://{host}:{port}/v1/embeddings"

    def _embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            inputs = [
                doc_template.format(query=t) if doc_template else t
                for t in chunk
            ]
            body = _json.dumps({"input": inputs}).encode("utf-8")
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
            data = sorted(payload["data"], key=lambda e: e.get("index", 0))
            for entry in data:
                vec = entry["embedding"]
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                out.append([x / norm for x in vec])
        return out

    return _embed


def _cmd_reembed_facts(args: argparse.Namespace) -> int:
    from backend.config import load_config

    paths = resolve_cli_paths()
    cfg = load_config()
    emb = cfg.get("embedding", {}) or {}
    host = args.embed_host or emb.get("llama_host", "localhost")
    port = args.embed_port or int(emb.get("llama_port", 8082))
    doc_template = args.doc_template or emb.get("doc_template", "")
    dim = args.dim if args.dim is not None else int(emb.get("dim", 1024))
    model_name = emb.get("model_name") or "embedding"
    # model_id は API 側 (/api/model/reembed-facts の 409 ガード) と同じ
    # normalize_embedding_model_id で導出する。単純な lower() だと拡張子付き
    # model_name で manifest とガードの期待値が乖離し 409 が解消しない。
    model_id = args.model_id or normalize_embedding_model_id(str(model_name))

    embed_fn = None
    if args.apply:
        embed_fn = _build_http_embed_fn(host, int(port), doc_template)
    report = run_reembed_facts(
        paths.memory_dir,
        paths.migration_archive_dir,
        new_model_id=model_id,
        new_dim=dim,
        embed_fn=embed_fn,
        normalized=not args.not_normalized,
        apply=args.apply,
    )
    if args.json:
        print(report.to_json())
    else:
        print(_reembed_fmt(report))
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

    # compact
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
            "mem.world.{assertion,executable_command,url}.* を丸ごと候補にする。"
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

    # compact
    sp = sub.add_parser(
        "compact",
        help="facts.jsonl の last-write-wins 圧縮 (デフォルト dry-run)",
    )
    sp.add_argument("--scope", default=None, help="特定 scope に限定")
    sp.add_argument(
        "--apply", action="store_true",
        help="実際に書き換える (未指定時は dry-run)",
    )
    sp.set_defaults(func=_cmd_compact)

    # rebuild-indices
    sp = sub.add_parser(
        "rebuild-indices",
        help=".idx 群を facts.jsonl から再生成する (デフォルト dry-run)",
    )
    sp.add_argument("--scope", default=None, help="特定 scope に限定")
    sp.add_argument(
        "--apply", action="store_true",
        help="実際に書き換える (未指定時は dry-run)",
    )
    sp.set_defaults(func=_cmd_rebuild_indices)

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

    # migrate-embedding
    sp = sub.add_parser(
        "migrate-embedding",
        help="埋め込み active model を swap する (デフォルト dry-run)",
    )
    sp.add_argument("--to", required=True, help="新 model_id (例: qwen3-embedding)")
    sp.add_argument("--dim", type=int, required=True, help="新モデルの次元数")
    sp.add_argument(
        "--not-normalized", action="store_true",
        help="埋め込みが L2 正規化されていない場合に付与",
    )
    sp.add_argument(
        "--no-create-dirs", action="store_true",
        help="apply 時に新 model_id 用の subdir を自動作成しない",
    )
    sp.add_argument(
        "--apply", action="store_true",
        help="実際に manifest を書き換える",
    )
    sp.set_defaults(func=_cmd_migrate_embedding)

    # reembed-facts
    sp = sub.add_parser(
        "reembed-facts",
        help="semantic fact の embedding を新モデルで再生成 + manifest swap "
        "(デフォルト dry-run)",
    )
    sp.add_argument(
        "--model-id", default=None,
        help="新 model_id (省略時は config embedding.model_name を小文字化)",
    )
    sp.add_argument(
        "--dim", type=int, default=None,
        help="新モデルの次元数 (省略時は config embedding.dim)",
    )
    sp.add_argument(
        "--not-normalized", action="store_true",
        help="埋め込みが L2 正規化されていない場合に付与",
    )
    sp.add_argument(
        "--embed-host", default=None,
        help="llama-embed ホスト (省略時 config embedding.llama_host)",
    )
    sp.add_argument(
        "--embed-port", type=int, default=None,
        help="llama-embed ポート (省略時 config embedding.llama_port)",
    )
    sp.add_argument(
        "--doc-template", default=None,
        help="doc 側テンプレ (省略時 config embedding.doc_template)",
    )
    sp.add_argument(
        "--apply", action="store_true",
        help="実際に再 embed + manifest swap する (未指定時は dry-run)",
    )
    sp.set_defaults(func=_cmd_reembed_facts)

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
