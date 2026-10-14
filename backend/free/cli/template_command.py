"""evoref template サブコマンド — 文書テンプレートの一覧・登録・削除 (docs/f_06 §2.10、c_16 §4.5.2)。

稼働中の backend の API だけを使う (f_06 §2.9.3)。

- ``list``: ``GET /api/templates``
- ``register FILE``: ``POST /api/templates/register`` (multipart の ``file`` / ``doc_type`` /
  ``aliases`` (繰り返し) / ``lang``)。パッケージ ID は本文の sha256 由来 (``tpl-<hex12>``)
  なので、同じ内容・同じ様式名の再登録は何もしない (API が冪等)。
- ``remove ID``: 専用の API は無く、テンプレートのパッケージを ``DELETE /api/cartridges/{id}``
  で消す。他のカートリッジを取り違えて消さないよう、ID は登録の経路が作る
  ``tpl-<hex12>`` だけを受ける。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from rich.console import Console
from rich.markup import escape

from backend.free.cli.cli_common import (
    EXIT_FAIL,
    EXIT_INTERRUPTED,
    EXIT_UNAVAILABLE,
    BackendError,
    add_common_args,
    backend_client,
    diag_console,
    emit,
    request,
    resolve_backend_url,
)
from backend.free.cli.config_loader import _setup_encoding
from backend.free.cli.renderer import create_console, render_error, render_info, render_table
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger

logger = get_logger("cli.template")

_TIMEOUT = 120.0
#: 単一ファイル登録が作るパッケージ ID (``api.content.templates.template_package_id``)
TEMPLATE_PACKAGE_ID = re.compile(r"tpl-[0-9a-f]{12}")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common)
    parser = argparse.ArgumentParser(prog="evoref template", description=msg("cli.help_template"))
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", parents=[common], help=msg("cli.template_help_list"))
    reg = sub.add_parser("register", parents=[common], help=msg("cli.template_help_register"))
    reg.add_argument("file", metavar="FILE", help=".docx / .pptx / .xlsx / .odt / .dotx / .potx / .xltx ...")
    reg.add_argument(
        "--doc-type", default=None, metavar="T",
        help="Document type the template is looked up by (default: the file name without the extension)",
    )
    reg.add_argument("--aliases", default=None, metavar="A,B", help="Other names for the document type (comma separated)")
    reg.add_argument("--lang", default=None, metavar="L", help="Language of the template (default on the backend: ja)")
    rm = sub.add_parser("remove", parents=[common], help=msg("cli.template_help_remove"))
    rm.add_argument("template_id", metavar="ID", help="tpl-<hex12> (the package column of list)")
    for action in ("register", "remove"):
        sub.choices[action].add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation")
    return parser


def run_template(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref template ...``)。"""
    if not _setup_encoding():
        return EXIT_FAIL
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    diag = diag_console(args.json, console)
    if args.action == "remove" and not TEMPLATE_PACKAGE_ID.fullmatch(args.template_id):
        render_error(diag, msg("cli.template_not_template_id", id=args.template_id))
        return EXIT_UNAVAILABLE
    try:
        upload = _read_upload(args, diag) if args.action == "register" else None
        if args.action == "register" and upload is None:
            return EXIT_FAIL
        if args.action != "list" and not args.yes and not _confirmed(diag, args, upload):
            return EXIT_FAIL
        with backend_client(_TIMEOUT) as client:
            out = _call(client, resolve_backend_url(args.backend_url), args, upload)
    except BackendError as e:
        render_error(diag, msg("cli.template_failed", detail=e.detail))
        return EXIT_UNAVAILABLE if e.status in (423, 503) else EXIT_FAIL
    except httpx.HTTPError as e:
        logger.error("template %s failed: %s", args.action, e)
        render_error(diag, msg("cli.template_failed", detail=str(e)))
        return EXIT_FAIL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    if out is None:
        render_error(diag, msg("cli.backend_not_running"))
        return EXIT_UNAVAILABLE
    return emit(out, as_json=args.json, render=lambda: _render(console, args, out))


def _aliases(args: argparse.Namespace) -> list[str]:
    return [a.strip() for a in (args.aliases or "").split(",") if a.strip()]


def _doc_type(args: argparse.Namespace) -> str:
    return (args.doc_type or Path(args.file).stem).strip()


def _read_upload(args: argparse.Namespace, console: Console) -> tuple[str, bytes] | None:
    """登録するファイルの (名前, 本文)。読めない・空なら表示して ``None``。"""
    path = Path(args.file).expanduser()
    try:
        content = path.read_bytes()
    except OSError as e:
        render_error(console, msg("cli.template_file_unreadable", path=args.file, detail=str(e)))
        return None
    if not content:
        render_error(console, msg("cli.template_file_empty", path=args.file))
        return None
    return path.name, content


def _confirmed(console: Console, args: argparse.Namespace, upload: tuple[str, bytes] | None) -> bool:
    """登録・削除の確認。対話端末でなければ確認できないので拒否する。"""
    if sys.stdin is None or not sys.stdin.isatty():
        render_error(console, msg("cli.template_needs_yes"))
        return False
    if args.action == "register":
        question = msg(
            "cli.template_confirm_register", file=(upload or ("", b""))[0], doc_type=_doc_type(args),
            aliases=", ".join(_aliases(args)) or "-",
        )
    else:
        question = msg("cli.template_confirm_remove", id=args.template_id)
    answer = console.input(question + " [y/N] ", markup=False)
    if answer.strip().lower() in ("y", "yes"):
        return True
    render_info(console, msg("cli.template_aborted"))
    return False


# ── backend ──


def _call(client: httpx.Client, base: str, args: argparse.Namespace, upload: tuple[str, bytes] | None) -> Any:
    """backend を呼ぶ。届かなければ ``None``、2xx 以外は :class:`BackendError`。"""
    if args.action == "list":
        return request(client, "GET", f"{base}/api/templates")
    if args.action == "remove":
        return request(client, "DELETE", f"{base}/api/cartridges/{quote(args.template_id, safe='')}")
    assert upload is not None
    data: dict[str, Any] = {"doc_type": _doc_type(args), "aliases": _aliases(args)}
    if args.lang:
        data["lang"] = args.lang
    return request(client, "POST", f"{base}/api/templates/register", data=data, files={"file": upload})


# ── 表示 ──


def _render(console: Console, args: argparse.Namespace, out: Any) -> None:
    if args.action == "register":
        render_info(console, msg(
            "cli.template_registered", package=out.get("package"), doc_type=escape(str(out.get("doc_type"))),
            aliases=escape(", ".join(out.get("aliases") or [])) or "-",
        ))
        return
    if args.action == "remove":
        render_info(console, msg("cli.template_removed", id=out.get("id") or args.template_id))
        return
    templates = out.get("templates") or []
    if not templates:
        render_info(console, msg("cli.template_empty"))
        return
    h = [msg(f"cli.template_col_{c}") for c in ("package", "doc_type", "aliases", "lang", "version")]
    render_table(console, [
        {
            h[0]: t.get("package"), h[1]: escape(str(t.get("doc_type"))),
            h[2]: escape(", ".join(t.get("aliases") or [])) or "-", h[3]: t.get("lang"), h[4]: t.get("version"),
        }
        for t in templates
    ], h)
