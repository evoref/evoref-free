"""evoref fewshot サブコマンド — few-shot 手本の一覧と pin / unpin / archive / restore (docs/f_04 §3.2.2)。

稼働中の backend の API (``/api/learning/fewshot``) だけを使う (f_06 §2.9.3)。操作は
1 件ずつで、ID の glob や一括の指定は持たない。保存できない状態 (readonly) は API が
423 で断る。
"""

from __future__ import annotations

import argparse
from typing import Any
from urllib.parse import quote

import httpx
from rich import box
from rich.cells import cell_len
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from backend.free.cli.cli_common import (
    EXIT_FAIL,
    EXIT_INTERRUPTED,
    EXIT_UNAVAILABLE,
    BackendError,
    add_common_args,
    add_page_args,
    backend_client,
    diag_console,
    emit,
    error_from_response,
    page_params,
    request,
    resolve_backend_url,
)
from backend.free.cli.config_loader import _setup_encoding
from backend.free.cli.renderer import create_console, render_error, render_info, render_table
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger

logger = get_logger("cli.fewshot")

_TIMEOUT = 30.0
_ACTIONS = ("pin", "unpin", "archive", "restore")
_QUERY_PREVIEW = 40


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common)
    parser = argparse.ArgumentParser(prog="evoref fewshot", description=msg("cli.help_fewshot"))
    sub = parser.add_subparsers(dest="action", required=True)
    lst = sub.add_parser("list", parents=[common], help=msg("cli.fewshot_help_list"))
    lst.add_argument("--include-archived", action="store_true", help="Also list archived examples")
    lst.add_argument("--mode", default=None, help="Only this mode (chat / create)")
    add_page_args(lst)
    for action in _ACTIONS:
        text = msg(f"cli.fewshot_help_{action}")
        p = sub.add_parser(action, parents=[common], help=text, description=text)
        p.add_argument("example_id", help="fs_<hex16> (one ID; see list)")
    return parser


def run_fewshot(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref fewshot ...``)。"""
    if not _setup_encoding():
        return EXIT_FAIL
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    diag = diag_console(args.json, console)
    base = f"{resolve_backend_url(args.backend_url)}/api/learning/fewshot"
    try:
        with backend_client(_TIMEOUT) as client:
            if args.action == "list":
                out = _list(client, base, args)
            else:
                out = request(client, "POST", f"{base}/{quote(args.example_id, safe='')}/{args.action}")
    except BackendError as e:
        render_error(diag, msg("cli.fewshot_failed", detail=e.detail))
        return EXIT_UNAVAILABLE if e.status in (423, 503) else EXIT_FAIL
    except httpx.HTTPError as e:
        logger.error("fewshot %s failed: %s", args.action, e)
        render_error(diag, msg("cli.fewshot_failed", detail=str(e)))
        return EXIT_FAIL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    if out is None:
        render_error(diag, msg("cli.backend_not_running"))
        return EXIT_UNAVAILABLE
    return emit(out, as_json=args.json, render=lambda: _render(console, args, out))


#: 退避済みの件数のヘッダ (``backend/free/api/learning/fewshot.py``)
_ARCHIVED_TOTAL_HEADER = "X-Archived-Total-Count"


def _list(client: httpx.Client, base: str, args: argparse.Namespace) -> dict | None:
    """一覧。``--include-archived`` が無ければ退避済みの本文は取らず (``include_archived=false``)、
    件数だけヘッダから残す。ヘッダの無い backend (旧版) は本文の件数で数えて捨てる。"""
    params = {**({"mode": args.mode} if args.mode else {}), **page_params(args)}
    if not args.include_archived:
        params["include_archived"] = "false"
    try:
        resp = client.request("GET", base, params=params or None)
    except httpx.ConnectError:
        return None
    if not 200 <= resp.status_code < 300:
        raise error_from_response(resp)
    data = resp.json()
    archived = data.get("archived") or []
    header = resp.headers.get(_ARCHIVED_TOTAL_HEADER)
    count = int(header) if header is not None and header.isdigit() else len(archived)
    return {**data, "archived": archived if args.include_archived else [], "archived_count": count}


# ── 表示 ──


def _render(console: Console, args: argparse.Namespace, out: Any) -> None:
    if args.action != "list":
        render_info(console, msg(f"cli.fewshot_done_{args.action}", id=out.get("id")))
        return
    examples = out.get("examples") or []
    if examples:
        _render_table(console, examples)
    else:
        render_info(console, msg("cli.fewshot_empty"))
    render_info(console, msg(
        "cli.fewshot_summary", count=len(examples), pool_size=out.get("pool_size"),
        archived=out.get("archived_count", 0),
    ))
    if args.include_archived and out.get("archived"):
        render_info(console, msg("cli.fewshot_archived_title"))
        _render_table(console, out["archived"])
    elif not args.include_archived and out.get("archived_count"):
        render_info(console, msg("cli.fewshot_archived_hint"))


def _render_table(console: Console, rows: list[dict]) -> None:
    h = [msg(f"cli.fewshot_col_{c}") for c in ("id", "mode", "state", "uses", "votes", "query")]

    def _query(text: str) -> str:
        line = " ".join(str(text or "").split())
        return escape(line if len(line) <= _QUERY_PREVIEW else line[:_QUERY_PREVIEW] + "…")

    data = [
        {
            h[0]: r.get("id"), h[1]: r.get("mode"),
            h[2]: str(r.get("state")) + (" (pinned)" if r.get("pinned") else ""),
            h[3]: r.get("use_count"), h[4]: f"+{r.get('helpful', 0)}/-{r.get('harmful', 0)}",
            h[5]: _query(r.get("query")),
        }
        for r in rows
    ]
    if console.no_color:
        render_table(console, data, h)
        return
    # ID は pin などの引数に写すので、端末幅が足りなくても省略しない (縮めるのは問いの列だけ)
    table = Table(show_header=True, header_style="bold", box=box.SIMPLE, border_style="dim")
    widths = [max(cell_len(str(v)) for v in [name, *(d[name] for d in data)]) for name in h[:-1]]
    for name, width in zip(h[:-1], widths, strict=True):
        table.add_column(name, no_wrap=True, min_width=width)
    # 枠と列の間の余白 (box.SIMPLE は列ごとに左右 1、表の左右に 1 ずつ)
    rest = console.width - sum(widths) - 2 * len(h) - 2
    table.add_column(h[-1], no_wrap=True, overflow="ellipsis", max_width=max(rest, 8))
    for d in data:
        table.add_row(*[str(d[name]) for name in h])
    console.print(table)
