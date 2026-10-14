"""evoref memory サブコマンド — 意味記憶のファクトの一覧・検索と詳細 (docs/f_06 §2.10)。

    evoref memory facts [--subject S] [--query Q] [--status ST] [--scope SC] [--limit N] [--offset N] [--json]
    evoref memory fact FACT_ID [--json]

``evoref forget --id`` に渡す ID を見つけるためのもので、稼働中の backend の API
(``GET /api/memory/facts``) だけを使う (読むだけ)。停止中は ``evoref doctor`` と
``evoref forget --id`` の dry-run を案内して終了コード 2。REPL の ``/memory`` (統計と
短期ノート) とは別のコマンド。ID は省略せずに出す (forget --id へ写すため)。
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

logger = get_logger("cli.memory")

_TIMEOUT = 30.0
#: API の ``status`` の値 (``all`` は絞らない)
STATUS_CHOICES = ("all", "live", "disputed", "superseded")
MAX_LIMIT = 200
_DETAIL_KEYS = (
    "id", "subject", "predicate", "type", "scope", "status", "veracity", "origin", "confidence",
    "pinned", "private", "multi_valued", "superseded_by", "created_at", "accessed_at",
    "session_ids", "extractors",
)


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common)
    parser = argparse.ArgumentParser(prog="evoref memory", description=msg("cli.help_memory_command"))
    sub = parser.add_subparsers(dest="action", required=True)
    facts = sub.add_parser("facts", parents=[common], help=msg("cli.memory_help_facts"))
    facts.add_argument("--subject", default=None, metavar="S", help="Subject prefix (e.g. mem.personal)")
    facts.add_argument(
        "--query", default=None, metavar="Q",
        help="Case-insensitive substring of the id, subject, predicate or text",
    )
    facts.add_argument("--status", default="all", choices=STATUS_CHOICES, help="Filter by status (default: all)")
    facts.add_argument("--scope", default="global", metavar="SCOPE", help="global (default) or project:<id>")
    facts.add_argument("--limit", type=int, default=50, metavar="N", help=f"1-{MAX_LIMIT} (default: 50)")
    facts.add_argument("--offset", type=int, default=0, metavar="N", help="Skip N facts (default: 0)")
    fact = sub.add_parser("fact", parents=[common], help=msg("cli.memory_help_fact"))
    fact.add_argument("fact_id", help="Fact id (see `evoref memory facts`)")
    return parser


def run_memory(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref memory ...``)。"""
    if not _setup_encoding():
        return EXIT_FAIL
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    diag = diag_console(args.json, console)
    if args.action == "facts":
        if not 1 <= args.limit <= MAX_LIMIT:
            render_error(diag, msg("cli.memory_bad_limit"))
            return EXIT_UNAVAILABLE
        if args.offset < 0:
            render_error(diag, msg("cli.memory_bad_offset"))
            return EXIT_UNAVAILABLE
    base = f"{resolve_backend_url(args.backend_url)}/api/memory/facts"
    try:
        with backend_client(_TIMEOUT) as client:
            out = _call(client, base, args)
    except BackendError as e:
        render_error(diag, msg("cli.memory_failed", detail=e.detail))
        return EXIT_UNAVAILABLE if e.status in (423, 503) else EXIT_FAIL
    except httpx.HTTPError as e:
        logger.error("memory %s failed: %s", args.action, e)
        render_error(diag, msg("cli.memory_failed", detail=str(e)))
        return EXIT_FAIL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    if out is None:
        render_error(diag, msg("cli.memory_not_running"))
        return EXIT_UNAVAILABLE
    render = _render_facts if args.action == "facts" else _render_fact
    return emit(out, as_json=args.json, render=lambda: render(console, out))


def _call(client: httpx.Client, base: str, args: argparse.Namespace) -> Any:
    """backend を呼ぶ。届かなければ ``None``、2xx 以外は :class:`BackendError`。"""
    if args.action == "fact":
        return request(client, "GET", f"{base}/{quote(args.fact_id, safe='')}")
    params: dict[str, Any] = {
        "scope": args.scope, "status": args.status, "limit": args.limit, "offset": args.offset,
    }
    if args.subject:
        params["subject"] = args.subject
    if args.query:
        params["query"] = args.query
    return request(client, "GET", base, params=params)


# ── 表示 ──


def _value(fact: dict) -> str:
    if fact.get("private"):
        return msg("cli.memory_private_value")
    return " ".join(str(fact.get("value") or "").split())


def _render_facts(console: Console, out: dict) -> None:
    facts = out.get("facts") or []
    if not facts:
        render_info(console, msg("cli.memory_facts_empty"))
        return
    h = [msg(f"cli.memory_col_{c}") for c in ("id", "status", "subject", "created", "value")]
    _render_list_table(console, [
        # 本文に [...] が入りうるので rich のマークアップとして解釈させない
        {
            h[0]: escape(str(f.get("id"))), h[1]: f.get("status"), h[2]: escape(str(f.get("subject"))),
            h[3]: f.get("created_at") or "-", h[4]: escape(_value(f)),
        }
        for f in facts
    ], h)
    render_info(console, msg(
        "cli.memory_facts_total", shown=len(facts), total=out.get("total"), offset=out.get("offset"),
    ))
    render_info(console, escape(msg("cli.memory_forget_hint")))


def _render_list_table(console: Console, data: list[dict], h: list[str]) -> None:
    """ファクトの一覧。ID は forget --id に写すので省略せず、縮めるのは本文の列だけ。"""
    if console.no_color:
        render_table(console, data, h)
        return
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


def _brief(fact: dict) -> str:
    return escape(f"[{fact.get('id')}] {fact.get('status')} {fact.get('subject')} {_value(fact)[:70]}")


def _render_fact(console: Console, out: dict) -> None:
    fact = out.get("fact") or {}
    for key in _DETAIL_KEYS:
        value = fact.get(key)
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        render_info(console, escape(f"{key}: {value if value not in (None, '') else '-'}"))
    render_info(console, escape(f"value: {_value(fact)}"))
    sections = (
        ("cli.memory_siblings", [_brief(f) for f in out.get("siblings") or []]),
        ("cli.memory_revived", [_brief(f) for f in out.get("revived_on_forget") or []]),
        ("cli.memory_notes", [
            escape(f"[{n.get('id')}] {n.get('state')} "
                   f"{msg('cli.memory_private_value') if n.get('private') else (n.get('preview') or '')}")
            for n in out.get("notes") or []
        ]),
    )
    for title, lines in sections:
        render_info(console, msg(title))
        for line in lines or [msg("cli.memory_none")]:
            render_info(console, f"  {line}" if lines else line)
    render_info(console, escape(msg("cli.memory_forget_hint_id", id=fact.get("id"))))
