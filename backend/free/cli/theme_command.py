"""`evoref theme install|uninstall` — テーマの導入と削除を端末から操作する (docs/c_11 §1)

GUI と同じ ``/api/themes/*`` を叩く (backend の稼働が必要)。
"""

from __future__ import annotations

import argparse

from backend.free.cli.cli_common import add_common_args, resolve_backend_url
from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import create_console
from backend.i18n_helper import init_i18n
from backend.log_config import setup_cli_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evoref theme",
        description="Install or uninstall a theme",
    )
    parser.add_argument("action", choices=("install", "uninstall"))
    parser.add_argument("theme_id", help="Theme ID (install: a ZIP path or an http(s) URL)")
    add_common_args(parser, json=False)
    return parser


def run_theme(argv: list[str]) -> int:
    """同期エントリーポイント"""
    if not _setup_encoding():
        return 1
    project_root = _find_project_root()
    setup_cli_logging(project_root=project_root, debug=False)
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()

    from backend.free.cli.theme_commands import theme_install, theme_uninstall

    backend_url = resolve_backend_url(args.backend_url, project_root)
    if args.action == "install":
        return theme_install(backend_url, console, args.theme_id)
    return theme_uninstall(backend_url, console, args.theme_id)
