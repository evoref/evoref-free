"""`evoref theme trust|untrust|install|uninstall` — テーマの信頼と導入を端末から操作する (docs/c_11 §1)

信頼済みテーマの JS はアプリのオリジン内で動く。付与を API に置くとオリジン内の
コードが別のテーマを信頼させられるので、付与は PC の持ち主が端末で実行するこの
コマンドだけにする。``config.yaml`` の ``theme.trusted`` を直接書き換え、動いている
backend には再起動で反映する。取り消し (untrust) は安全側なので API にも残っている。

``install`` / ``uninstall`` は GUI と同じ ``/api/themes/*`` を叩く (backend の稼働が必要)。
インストールは信頼を付けない — 入れたテーマを信頼するかは ``trust`` で別に決める。
"""

from __future__ import annotations

import argparse

from backend.config import get_config, get_path_resolver, load_config
from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import create_console, render_error, render_info
from backend.i18n_helper import init_i18n, msg
from backend.log_config import setup_cli_logging

_DEFAULT_BACKEND = "http://localhost:8000"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evoref theme",
        description="Grant or revoke trust for a theme, or install / uninstall a theme",
    )
    parser.add_argument("action", choices=("trust", "untrust", "install", "uninstall"))
    parser.add_argument("theme_id", help="Theme ID (install: a ZIP path or an http(s) URL)")
    parser.add_argument(
        "--backend-url", default=_DEFAULT_BACKEND,
        help="Backend URL for install / uninstall (default: http://localhost:8000)",
    )
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

    if args.action in ("install", "uninstall"):
        from backend.free.cli.theme_commands import theme_install, theme_uninstall

        backend_url = args.backend_url.rstrip("/")
        if args.action == "install":
            return theme_install(backend_url, console, args.theme_id)
        return theme_uninstall(backend_url, console, args.theme_id)

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
