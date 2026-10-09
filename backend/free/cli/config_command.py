"""evoref config サブコマンド (c_05 §7.6)。

``normalize``: 版 (``config_version``) の無い G0 の ``config.yaml`` を G1 の形へ
一度だけ直す (実体は :mod:`backend.config_normalize`)。単一書き手ロック
(``<data_root>/g1/store/.writer.lock``) の下で動かし、serve の稼働中 (ロックが
取られている) は書き換えずに拒否する。

``evoref serve`` / ``evoref-ctl start`` / setup は起動前に
``config normalize --if-needed`` を呼ぶ (版のある config には何もしない)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console

from backend.config_normalize import NormalizeReport, normalize_config
from backend.data_root import DataRootError, resolve_data_root, store_root
from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import create_console, render_error, render_info
from backend.i18n_helper import init_i18n, msg
from backend.io.writer_lock import WriterLockHeld, acquire_writer_lock
from backend.log_config import get_logger

logger = get_logger("cli.config")


def config_needs_normalize(config_path: Path) -> bool:
    """``config_path`` が版を持たない (G0 の) config か。無ければ ``False``。"""
    if not config_path.is_file():
        return False
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return not (isinstance(raw, dict) and "config_version" in raw)


def normalize_under_lock(config_path: Path, data_root: Path) -> NormalizeReport:
    """書き手ロックを取って正規化する。serve が稼働中なら :class:`WriterLockHeld`。"""
    lock = acquire_writer_lock(store_root(data_root))
    try:
        return normalize_config(config_path)
    finally:
        lock.release()


def render_normalize_report(console: Console, report: NormalizeReport) -> None:
    """正規化の結果 (退避先・落としたキー・値が不正なキー) を表示する。"""
    if not report.changed:
        render_info(console, msg("cli.config_normalize_unchanged"))
        return
    render_info(console, msg("cli.config_normalize_done", backup=str(report.backup)))
    if report.renamed:
        render_info(
            console,
            msg(
                "cli.config_normalize_renamed",
                count=len(report.renamed), keys=", ".join(report.renamed),
            ),
        )
    if report.dropped:
        render_info(
            console,
            msg(
                "cli.config_normalize_dropped",
                count=len(report.dropped), keys=", ".join(report.dropped),
            ),
        )
    if report.invalid:
        render_error(
            console,
            msg(
                "cli.config_normalize_invalid",
                count=len(report.invalid), keys="; ".join(report.invalid),
            ),
            level="warning",
        )


def normalize_before_start(project_root: Path, console: Console) -> int | None:
    """起動前の正規化 (版のある config には何もしない)。

    Returns:
        続行できるなら ``None``。serve の稼働中・書き換えの失敗は終了コード ``1``。
    """
    config_path = project_root / "config.yaml"
    try:
        if not config_needs_normalize(config_path):
            return None
        data_root = resolve_data_root(root=project_root)
        report = normalize_under_lock(config_path, data_root)
    except WriterLockHeld as e:
        render_error(console, msg("cli.config_normalize_locked", detail=str(e)))
        return 1
    except (OSError, ValueError, yaml.YAMLError, DataRootError) as e:
        logger.error("config normalize failed: %s", e)
        render_error(console, msg("cli.config_normalize_failed", detail=str(e)))
        return 1
    render_normalize_report(console, report)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evoref config",
        description=msg("cli.help_config"),
    )
    parser.add_argument("action", choices=["normalize"])
    parser.add_argument(
        "--if-needed", action="store_true",
        help="Do nothing (and take no lock) when config.yaml already has config_version",
    )
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="Data root whose writer lock guards the rewrite "
        "(default: EVOREF_DATA_ROOT or <install_root>/userdata)",
    )
    return parser


def run_config(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref config normalize``)。"""
    if not _setup_encoding():
        return 1
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    root = _find_project_root()
    config_path = root / "config.yaml"
    if not config_path.is_file():
        if args.if_needed:
            return 0
        render_error(console, msg("cli.config_not_found", path=str(config_path)))
        return 1
    try:
        if args.if_needed and not config_needs_normalize(config_path):
            return 0
        data_root = resolve_data_root(args.data_root, root=root)
        report = normalize_under_lock(config_path, data_root)
    except WriterLockHeld as e:
        render_error(console, msg("cli.config_normalize_locked", detail=str(e)))
        return 1
    except DataRootError as e:
        render_error(console, msg("cli.data_root_invalid", detail=str(e)))
        return 1
    except (OSError, ValueError, yaml.YAMLError) as e:
        logger.error("config normalize failed: %s", e)
        render_error(console, msg("cli.config_normalize_failed", detail=str(e)))
        return 1
    render_normalize_report(console, report)
    return 0
