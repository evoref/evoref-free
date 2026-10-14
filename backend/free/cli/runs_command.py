"""evoref runs サブコマンド — create と long_form の run の参照 (docs/f_06 §2.10)。

``list`` / ``show`` / ``events`` / ``artifacts`` / ``delete``。稼働中の backend の API
(``/api/create/runs*``) だけを使う。

- ``delete RUN_ID [--yes]``: ``DELETE /api/create/runs/{run_id}`` (実行中は 409)。確認を
  取り、対話端末でなければ ``--yes`` が要る。消えるのは run の作業場だけで、出力先へ
  配信したファイルは残る。

- ``events --follow``: ``after=`` の watermark で一定間隔ごとに追記分を読み、run の
  状態が ``working`` でなくなったら最後の追記分を出して止まる。``--json`` は 1 行 1 件の
  JSON (JSON Lines)。
- ``artifacts --out DIR``: API が返す本文を ``DIR`` の下へ書く (書き先は ``--out`` で
  明示させ、CWD へは書かない — CLAUDE.md §6 #5)。``..``・絶対パスを含む名前や、既に
  在るファイルが 1 つでもあれば何も書かずに失敗する。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
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
    EXIT_OK,
    EXIT_UNAVAILABLE,
    BackendError,
    add_common_args,
    add_page_args,
    backend_client,
    diag_console,
    emit,
    page_params,
    request,
    resolve_backend_url,
)
from backend.free.cli.config_loader import _setup_encoding
from backend.free.cli.renderer import create_console, render_error, render_info, render_table
from backend.i18n_helper import init_i18n, msg
from backend.io.atomic import atomic_write_bytes
from backend.io.id_registry import is_valid_id
from backend.log_config import get_logger

logger = get_logger("cli.runs")

_TIMEOUT = 30.0
#: ``events --follow`` のポーリング間隔 (秒)
FOLLOW_INTERVAL_SEC = 2.0
#: run がまだ進む状態 (これ以外は終端か、needs_input = 次のターンを待つ状態で、この run には追記されない)
_ACTIVE_STATUSES = frozenset({"working"})
_QUERY_PREVIEW = 40


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common)
    parser = argparse.ArgumentParser(prog="evoref runs", description=msg("cli.help_runs"))
    sub = parser.add_subparsers(dest="action", required=True)
    lst = sub.add_parser("list", parents=[common], help=msg("cli.runs_help_list"))
    lst.add_argument("--session", default=None, metavar="ID", help="Only runs of this session")
    add_page_args(lst)
    show = sub.add_parser("show", parents=[common], help=msg("cli.runs_help_show"))
    show.add_argument("run_id", help="run_<hex16> (see list)")
    events = sub.add_parser("events", parents=[common], help=msg("cli.runs_help_events"))
    events.add_argument("run_id", help="run_<hex16> (see list)")
    events.add_argument("--after", type=int, default=0, metavar="N", help="Only events with seq > N (default: 0)")
    events.add_argument(
        "--follow", action="store_true",
        help=f"Keep polling every {FOLLOW_INTERVAL_SEC:g}s until the run is no longer working",
    )
    arts = sub.add_parser("artifacts", parents=[common], help=msg("cli.runs_help_artifacts"))
    arts.add_argument("run_id", help="run_<hex16> (see list)")
    arts.add_argument(
        "--out", default=None, metavar="DIR",
        help="Write the artifacts under DIR (existing files are not overwritten)",
    )
    delete = sub.add_parser("delete", parents=[common], help=msg("cli.runs_help_delete"))
    delete.add_argument("run_id", help="run_<hex16> (see list)")
    delete.add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation")
    return parser


def run_runs(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref runs ...``)。"""
    if not _setup_encoding():
        return EXIT_FAIL
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    diag = diag_console(args.json, console)
    if args.action != "list" and not is_valid_id(args.run_id, "run_"):
        render_error(diag, msg("cli.runs_bad_run_id", run_id=args.run_id))
        return EXIT_UNAVAILABLE
    if args.action == "events" and args.after < 0:
        render_error(diag, msg("cli.runs_after_negative"))
        return EXIT_UNAVAILABLE
    if args.action == "delete" and not args.yes and not _confirmed(diag, args):
        return EXIT_FAIL
    base = f"{resolve_backend_url(args.backend_url)}/api/create/runs"
    try:
        with backend_client(_TIMEOUT) as client:
            if args.action == "events" and args.follow:
                return _follow(client, base, args, console, diag)
            out = _call(client, base, args)
    except BackendError as e:
        render_error(diag, msg("cli.runs_failed", detail=e.detail))
        return EXIT_UNAVAILABLE if e.status in (423, 503) else EXIT_FAIL
    except httpx.HTTPError as e:
        logger.error("runs %s failed: %s", args.action, e)
        render_error(diag, msg("cli.runs_failed", detail=str(e)))
        return EXIT_FAIL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    if out is None:
        render_error(diag, msg("cli.backend_not_running"))
        return EXIT_UNAVAILABLE
    if args.action == "artifacts" and args.out:
        return _save_artifacts(console, diag, args, out)
    return emit(out, as_json=args.json, render=lambda: _render(console, args, out))


def _run_url(base: str, run_id: str) -> str:
    return f"{base}/{quote(run_id, safe='')}"


def _call(client: httpx.Client, base: str, args: argparse.Namespace) -> Any:
    """backend を呼ぶ。届かなければ ``None``、2xx 以外は :class:`BackendError`。"""
    if args.action == "list":
        params = {**({"session_id": args.session} if args.session else {}), **page_params(args)}
        return request(client, "GET", base, params=params or None)
    url = _run_url(base, args.run_id)
    if args.action == "show":
        return request(client, "GET", url)
    if args.action == "events":
        return request(client, "GET", f"{url}/events", params={"after": args.after})
    if args.action == "delete":
        return request(client, "DELETE", url)
    return request(client, "GET", f"{url}/artifacts")


def _confirmed(console: Console, args: argparse.Namespace) -> bool:
    """削除の確認。対話端末でなければ確認できないので拒否する。"""
    if sys.stdin is None or not sys.stdin.isatty():
        render_error(console, msg("cli.runs_needs_yes"))
        return False
    answer = console.input(msg("cli.runs_confirm_delete", run_id=args.run_id) + " [y/N] ", markup=False)
    if answer.strip().lower() in ("y", "yes"):
        return True
    render_info(console, msg("cli.runs_aborted"))
    return False


# ── events --follow ──


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _follow(
    client: httpx.Client, base: str, args: argparse.Namespace, console: Console, diag: Console,
) -> int:
    """run が ``working`` でなくなるまで追記分を出し続ける。

    状態を先に読み、その後でイベントを読む — 終端を見た回のイベントの読みには
    終端までの追記が全部入っているので、その回を出して止まれば取りこぼさない。
    """
    url = _run_url(base, args.run_id)
    after = args.after
    while True:
        run = request(client, "GET", url)
        data = request(client, "GET", f"{url}/events", params={"after": after}) if run is not None else None
        if data is None:
            render_error(diag, msg("cli.backend_not_running"))
            return EXIT_UNAVAILABLE
        for event in data.get("events") or []:
            if args.json:
                print(json.dumps(event, ensure_ascii=False), flush=True)
            else:
                _render_event(console, event)
        after = int(data.get("last_seq", after))
        status = str(run.get("status") or "")
        if status not in _ACTIVE_STATUSES:
            render_info(diag, msg("cli.runs_follow_end", run_id=args.run_id, status=status, last_seq=after))
            return EXIT_OK
        _sleep(FOLLOW_INTERVAL_SEC)


# ── artifacts --out ──


def _unsafe_name(name: str) -> bool:
    """``..``・絶対パス・ドライブ指定・空の部分を含む名前か (Windows と POSIX の両方の書式で見る)。"""
    if not name or "\x00" in name:
        return True
    for pure in (PurePosixPath(name), PureWindowsPath(name)):
        if pure.is_absolute() or pure.anchor or any(part in ("..", "") for part in pure.parts):
            return True
    return False


def _save_artifacts(console: Console, diag: Console, args: argparse.Namespace, out: dict) -> int:
    """成果物を ``--out`` の下へ書く。検査に 1 件でも掛かれば何も書かない。"""
    out_dir = Path(args.out).expanduser().resolve()
    artifacts = out.get("artifacts") or []
    unsafe = [str(a.get("path")) for a in artifacts if _unsafe_name(str(a.get("path") or ""))]
    if unsafe:
        render_error(diag, msg("cli.runs_unsafe_paths", paths=", ".join(unsafe)))
        return EXIT_FAIL
    targets = [(out_dir / str(a["path"])).resolve() for a in artifacts]
    escaped = [str(t) for t in targets if not t.is_relative_to(out_dir)]
    if escaped:
        render_error(diag, msg("cli.runs_unsafe_paths", paths=", ".join(escaped)))
        return EXIT_FAIL
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        render_error(diag, msg("cli.runs_out_exists", paths=", ".join(existing)))
        return EXIT_FAIL
    saved: list[str] = []
    try:
        for art, target in zip(artifacts, targets, strict=True):
            target.parent.mkdir(parents=True, exist_ok=True)
            # 改行を変換しない (成果物の本文をそのまま残す)
            atomic_write_bytes(target, str(art.get("content") or "").encode("utf-8"))
            saved.append(str(target))
    except OSError as e:
        render_error(diag, msg("cli.runs_write_failed", detail=str(e), count=len(saved)))
        return EXIT_FAIL
    result = {"run_id": out.get("run_id"), "out": str(out_dir), "saved": saved}

    def _render_saved() -> None:
        for path in saved:
            render_info(console, escape(path))
        render_info(console, msg("cli.runs_saved", count=len(saved), out=str(out_dir)))

    return emit(result, as_json=args.json, render=_render_saved)


# ── 表示 ──


def _preview(text: Any) -> str:
    line = " ".join(str(text or "").split())
    return escape(line if len(line) <= _QUERY_PREVIEW else line[:_QUERY_PREVIEW] + "…")


def _render(console: Console, args: argparse.Namespace, out: Any) -> None:
    if args.action == "list":
        runs = out.get("runs") or []
        if not runs:
            render_info(console, msg("cli.runs_empty"))
            return
        h = [msg(f"cli.runs_col_{c}") for c in ("id", "status", "mode", "started", "query")]
        _render_list_table(console, [
            {
                h[0]: r.get("run_id"), h[1]: r.get("status"), h[2]: r.get("mode"),
                h[3]: r.get("started_at"), h[4]: _preview(r.get("query")),
            }
            for r in runs
        ], h)
    elif args.action == "show":
        for key in (
            "run_id", "status", "mode", "session_id", "started_at", "ended_at", "exit_kind",
            "output_target", "template", "tasks_failed", "last_event_seq",
        ):
            render_info(console, f"{key}: {escape(str(out.get(key) if out.get(key) is not None else '-'))}")
        render_info(console, f"query: {escape(str(out.get('query') or ''))}")
    elif args.action == "delete":
        render_info(console, msg("cli.runs_deleted", run_id=out.get("run_id") or args.run_id))
    elif args.action == "events":
        events = out.get("events") or []
        if not events:
            render_info(console, msg("cli.runs_events_empty", after=out.get("after")))
        for event in events:
            _render_event(console, event)
    else:
        artifacts = out.get("artifacts") or []
        if not artifacts:
            render_info(console, msg("cli.runs_artifacts_empty"))
            return
        h = [msg(f"cli.runs_col_{c}") for c in ("path", "language", "chars")]
        render_table(console, [
            {h[0]: escape(str(a.get("path"))), h[1]: a.get("language"), h[2]: len(str(a.get("content") or ""))}
            for a in artifacts
        ], h)
        render_info(console, msg("cli.runs_artifacts_hint"))


def _render_list_table(console: Console, data: list[dict], h: list[str]) -> None:
    """run の一覧。ID は show などの引数に写すので省略せず、縮めるのは依頼の列だけ。"""
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


def _render_event(console: Console, event: dict) -> None:
    payload = event.get("payload")
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if not detail:
        detail = json.dumps(payload, ensure_ascii=False) if payload else ""
    render_info(console, f"#{event.get('seq')} {event.get('at')} {event.get('kind')} {escape(str(detail))}")
