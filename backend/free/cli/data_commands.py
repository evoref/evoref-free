"""evoref export / import サブコマンド (停止中にデータ根を直接読み書きする)。

どちらも単一書き手ロック (``store/.writer.lock``) を取ってから動くので、serve の
稼働中は拒否する。中身は :mod:`backend.free.export_import`。
"""

from __future__ import annotations

import argparse
import importlib.util
import zipfile
from pathlib import Path

from backend.data_root import DataRootError, resolve_data_root
from backend.free.cli.renderer import create_console, render_error, render_info
from backend.free.export_import import (
    CATEGORIES,
    DataTransferError,
    ExportReport,
    ImportReport,
    export_data,
    import_data,
    resolve_destination,
)
from backend.i18n_helper import init_i18n, msg
from backend.io.writer_lock import WriterLock, WriterLockHeld, acquire_writer_lock
from backend.log_config import get_logger

logger = get_logger("cli.data")


def offline_edition() -> str:
    """停止中の CLI が名乗るエディション (Pro が同梱され ``EVOREF_EDITION`` が free でない)。"""
    from backend.free.cli.cli_mode import is_cli_pro_edition

    return "pro" if is_cli_pro_edition() and importlib.util.find_spec("backend.pro") else "free"


# ────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="Data root (default: EVOREF_DATA_ROOT or <install_root>/userdata)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")


def build_export_parser() -> argparse.ArgumentParser:
    """export サブコマンド用パーサー"""
    parser = argparse.ArgumentParser(prog="evoref export", description=msg("cli.help_data_export"))
    parser.add_argument(
        "-o", "--output",
        help="Output directory or file (default: <data_root>/outputs/). "
             "Inside the data root only outputs/ is allowed",
    )
    parser.add_argument(
        "--categories", help=f"Categories to export (comma-separated: {', '.join(CATEGORIES)})",
    )
    parser.add_argument("--exclude", help="Categories to leave out (comma-separated)")
    parser.add_argument(
        "--include-private", action="store_true",
        help="Also export memory records marked private / secret",
    )
    parser.add_argument(
        "--no-mask", action="store_true",
        help="Keep strings that look like secrets (API keys, tokens, e-mail addresses) as they are",
    )
    _add_common(parser)
    return parser


def build_import_parser() -> argparse.ArgumentParser:
    """import サブコマンド用パーサー"""
    parser = argparse.ArgumentParser(prog="evoref import", description=msg("cli.help_data_import"))
    parser.add_argument("file", help="Path to a .evoref-export.zip package")
    parser.add_argument(
        "--categories", help=f"Categories to import (comma-separated: {', '.join(CATEGORIES)})",
    )
    _add_common(parser)
    return parser


def _parse_categories(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {c.strip() for c in raw.split(",") if c.strip()}


def _resolve_categories(args: argparse.Namespace, console) -> list[str] | None:
    """--categories / --exclude から対象カテゴリを決める (不正なら ``None``)。"""
    chosen = _parse_categories(args.categories)
    excluded = _parse_categories(getattr(args, "exclude", None))
    if chosen and excluded:
        render_error(console, msg("cli.export_categories_conflict"))
        return None
    invalid = (chosen or set()) | (excluded or set())
    invalid -= set(CATEGORIES)
    if invalid:
        render_error(console, msg("cli.export_invalid_category", categories=", ".join(sorted(invalid))))
        return None
    if chosen:
        return [c for c in CATEGORIES if c in chosen]
    return [c for c in CATEGORIES if c not in (excluded or set())]


def _open(argv: list[str], parser: argparse.ArgumentParser):
    """引数・コンソール・データ根を揃える (データ根が不正なら ``data_root=None``)。"""
    from backend.free.cli.config_loader import _find_project_root

    args = parser.parse_args(argv)
    console = create_console(no_color=args.no_color)
    try:
        data_root = resolve_data_root(args.data_root, root=_find_project_root())
    except DataRootError as e:
        render_error(console, msg("cli.data_root_invalid", detail=str(e)))
        return args, console, None
    return args, console, data_root


def _lock(data_root: Path, console, key: str) -> WriterLock | None:
    try:
        return acquire_writer_lock(data_root / "store")
    except WriterLockHeld as e:
        render_error(console, msg(key, detail=str(e)))
        return None


def _render_failure(console, error: DataTransferError) -> int:
    render_error(console, msg(error.key, **error.params))
    return 1


# ────────────────────────────────────────────
# export
# ────────────────────────────────────────────


def run_export(argv: list[str]) -> int:
    """export サブコマンドのエントリーポイント"""
    from backend.formats import load_all_formats

    init_i18n()
    args, console, data_root = _open(argv, build_export_parser())
    if data_root is None:
        return 1
    categories = _resolve_categories(args, console)
    if categories is None:
        return 1
    try:
        destination = resolve_destination(data_root, Path(args.output) if args.output else None)
    except DataTransferError as e:
        return _render_failure(console, e)
    load_all_formats()
    lock = _lock(data_root, console, "cli.export_locked")
    if lock is None:
        return 1
    render_info(console, msg("cli.export_starting"))
    try:
        report = export_data(
            data_root, destination, edition=offline_edition(), categories=categories,
            include_private=args.include_private, mask=not args.no_mask,
        )
    except DataTransferError as e:
        return _render_failure(console, e)
    except OSError as e:
        render_error(console, msg("cli.export_write_error", detail=str(e)))
        return 1
    finally:
        lock.release()
    _render_export(console, report)
    return 0


def _render_export(console, report: ExportReport) -> None:
    for category, stats in report.stats.items():
        render_info(console, msg(
            "cli.export_category_line", category=category, files=stats.files,
            records=stats.records, size=f"{stats.bytes / (1024 * 1024):.1f}",
        ))
    private = sum(s.private_excluded for s in report.stats.values())
    if private:
        render_info(console, msg("cli.export_private_excluded", count=private))
    redacted = sum(s.redacted for s in report.stats.values())
    if redacted:
        key = "cli.export_redacted" if report.masked else "cli.export_redaction_kept"
        render_info(console, msg(key, count=redacted))
    size = report.path.stat().st_size / (1024 * 1024)
    render_info(console, msg("cli.export_completed", path=str(report.path), size=f"{size:.1f}"))


# ────────────────────────────────────────────
# import
# ────────────────────────────────────────────


def run_import(argv: list[str]) -> int:
    """import サブコマンドのエントリーポイント"""
    from backend.formats import load_all_formats

    init_i18n()
    args, console, data_root = _open(argv, build_import_parser())
    if data_root is None:
        return 1
    categories = _resolve_categories(args, console)
    if categories is None:
        return 1
    package = Path(args.file)
    if not package.exists():
        render_error(console, msg("cli.import_file_not_found", path=str(package)))
        return 1
    if not package.is_file():
        render_error(console, msg("cli.import_not_a_file", path=str(package)))
        return 1
    load_all_formats()
    lock = _lock(data_root, console, "cli.import_locked")
    if lock is None:
        return 1
    try:
        report = import_data(data_root, package, edition=offline_edition(), categories=categories)
    except DataTransferError as e:
        return _render_failure(console, e)
    except (OSError, zipfile.BadZipFile) as e:
        render_error(console, msg("cli.import_read_error", detail=str(e)))
        return 1
    finally:
        lock.release()
    _render_import(console, report)
    return 0


def _render_import(console, report: ImportReport) -> None:
    imported = False
    for category, result in report.categories.items():
        if result.status == "not_empty":
            render_info(console, msg("cli.import_category_not_empty", category=category))
            continue
        for partition in result.skipped_partitions:
            render_info(console, msg("cli.import_partition_not_empty", partition=partition))
        if result.files:
            imported = True
            render_info(console, msg(
                "cli.import_category_done", category=category, files=result.files, records=result.records,
            ))
        if result.invalid_records:
            render_info(console, msg("cli.import_invalid_records", count=result.invalid_records))
    if report.skipped_pro:
        render_info(console, msg("cli.import_skipped_pro", count=report.skipped_pro))
    if report.skipped_unknown:
        render_info(console, msg("cli.import_skipped_unknown", count=report.skipped_unknown))
    if not imported:
        render_info(console, msg("cli.import_nothing"))
        return
    render_info(console, msg("cli.import_rebuild_note"))
    render_info(console, msg("cli.import_completed"))
