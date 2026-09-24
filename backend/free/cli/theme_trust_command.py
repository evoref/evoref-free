"""`evoref theme trust|untrust <id>` — テーマの信頼を端末から付与する (docs/c_11 §1)

信頼済みテーマの JS はアプリのオリジン内で動く。付与を API に置くとオリジン内の
コードが別のテーマを信頼させられるので、付与は PC の持ち主が端末で実行するこの
コマンドだけにする。``config.yaml`` の ``theme.trusted`` を直接書き換え、動いている
backend には再起動で反映する。取り消し (untrust) は安全側なので API にも残っている。
"""

from __future__ import annotations

import argparse

from backend.config import get_config, get_path_resolver, load_config
from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import create_console, render_error, render_info
from backend.i18n_helper import init_i18n, msg
from backend.log_config import setup_cli_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evoref theme",
        description="Grant or revoke trust for an installed theme",
    )
    parser.add_argument("action", choices=("trust", "untrust"))
    parser.add_argument("theme_id")
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

    load_config(project_root / "config.yaml", project_root)
    from backend.free.themes.theme_service import ThemeManager

    manager = ThemeManager(get_path_resolver().resolve_local("themes_dir"), get_config())
    if not manager.theme_exists(args.theme_id):
        render_error(console, msg("cli.theme_not_found", id=args.theme_id))
        return 1
    try:
        if args.action == "trust":
            manager.trust_theme(args.theme_id)
            render_info(console, msg("cli.theme_trust_granted", id=args.theme_id))
        else:
            manager.untrust_theme(args.theme_id)
            render_info(console, msg("cli.theme_trust_revoked", id=args.theme_id))
    except RuntimeError as exc:
        render_error(console, str(exc))
        return 1
    render_info(console, msg("cli.theme_trust_restart_hint"))
    return 0
