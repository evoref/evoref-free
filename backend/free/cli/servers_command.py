"""evoref servers サブコマンド — llama-server の状態・起動・停止・再起動 (docs/f_06 §2.10)。

稼働中の backend の API (``/api/model/process/*``、所有者は ``LlamaProcessManager``、
c_06 §2.6) だけを使う。GUI の ``/api/servers/*`` は互換層で、CLI は使わない。

- ``status [NAME]``: ``GET /api/model/process/status``。このバックエンドが起動したもの・
  外部起動 (``evoref serve`` / ``evoref-ctl``) のもの・停止を出す。
- ``start NAME [--force]``: ``POST /api/model/process/{component}/start``。外部起動の
  llama-server が居れば二重に起こさず 409 (``--force`` で置き換え)。
- ``stop NAME`` / ``restart NAME``: 確認を取る。対話端末でなければ ``--yes`` が要る。

``NAME`` は ``base`` / ``embed`` / ``rerank``。API の component (``embedding``) への写像は
``llama_process_manager.component_of`` の 1 か所。backend が居なければ終了コード 2。
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

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
from backend.free.core.llama_process_manager import (
    PROCESS_COMPONENTS,
    SERVER_NAMES,
    component_of,
)
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger

logger = get_logger("cli.servers")

#: status / stop の待ち (停止は stop_timeout + taskkill)
_TIMEOUT = 60.0
#: start / restart の待ち。health 待ちは GGUF サイズ連動で最大 600 秒、GPU の起動失敗から
#: CPU で起こし直すとその倍近くまで延びる
_START_TIMEOUT = 1500.0

_NAMES = tuple(SERVER_NAMES[c] for c in PROCESS_COMPONENTS)


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common)
    parser = argparse.ArgumentParser(prog="evoref servers", description=msg("cli.help_servers"))
    sub = parser.add_subparsers(dest="action", required=True)
    status = sub.add_parser("status", parents=[common], help=msg("cli.servers_help_status"))
    status.add_argument("name", nargs="?", choices=_NAMES, help="Only this server")
    start = sub.add_parser("start", parents=[common], help=msg("cli.servers_help_start"))
    start.add_argument("name", choices=_NAMES)
    start.add_argument(
        "--force", action="store_true",
        help="Stop a running llama-server on the port (this backend's or an external one) and start it again",
    )
    for action in ("stop", "restart"):
        p = sub.add_parser(action, parents=[common], help=msg(f"cli.servers_help_{action}"))
        p.add_argument("name", choices=_NAMES)
        p.add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation")
    return parser


def run_servers(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref servers ...``)。"""
    if not _setup_encoding():
        return EXIT_FAIL
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    diag = diag_console(args.json, console)
    if args.action in ("stop", "restart") and not args.yes and not _confirmed(diag, args):
        return EXIT_FAIL
    base = f"{resolve_backend_url(args.backend_url)}/api/model/process"
    timeout = _START_TIMEOUT if args.action in ("start", "restart") else _TIMEOUT
    try:
        with backend_client(timeout) as client:
            out = _call(client, base, args)
    except BackendError as e:
        render_error(diag, msg("cli.servers_failed", detail=e.detail))
        return EXIT_UNAVAILABLE if e.status == 503 else EXIT_FAIL
    except httpx.HTTPError as e:
        logger.error("servers %s failed: %s", args.action, e)
        render_error(diag, msg("cli.servers_failed", detail=str(e)))
        return EXIT_FAIL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    if out is None:
        render_error(diag, msg("cli.servers_not_running"))
        return EXIT_UNAVAILABLE
    if args.action == "status" and args.name:
        component = component_of(args.name)
        components = out.get("components") or {}
        out = {**out, "components": {component: components[component]} if component in components else {}}
    return emit(out, as_json=args.json, render=lambda: _render(console, args, out))


def _call(client: httpx.Client, base: str, args: argparse.Namespace) -> Any:
    """backend を呼ぶ。届かなければ ``None``、2xx 以外は :class:`BackendError`。"""
    if args.action == "status":
        return request(client, "GET", f"{base}/status")
    url = f"{base}/{component_of(args.name)}/{args.action}"
    if args.action == "start":
        return request(client, "POST", url, params={"force": "true"} if args.force else None)
    return request(client, "POST", url)


def _confirmed(console: Console, args: argparse.Namespace) -> bool:
    """停止・再起動の確認。対話端末でなければ確認できないので拒否する。"""
    if sys.stdin is None or not sys.stdin.isatty():
        render_error(console, msg("cli.servers_needs_yes"))
        return False
    answer = console.input(msg(f"cli.servers_confirm_{args.action}", name=args.name) + " [y/N] ", markup=False)
    if answer.strip().lower() in ("y", "yes"):
        return True
    render_info(console, msg("cli.servers_aborted"))
    return False


def _render(console: Console, args: argparse.Namespace, out: dict) -> None:
    if args.action == "status":
        _render_status(console, out)
        return
    if args.action == "stop":
        render_info(console, msg("cli.servers_stopped", name=args.name, pid=out.get("pid")))
        return
    key = "cli.servers_started" if args.action == "start" else "cli.servers_restarted"
    render_info(console, msg(key, name=args.name, port=out.get("port"), pid=out.get("pid")))


def _state(info: dict) -> str:
    if info.get("managed"):
        return msg("cli.servers_state_managed" if info.get("alive") else "cli.servers_state_managed_dead")
    if info.get("external_pid") is not None:
        return msg("cli.servers_state_external", name=escape(str(info.get("external_name") or "?")))
    return msg("cli.servers_state_stopped")


def _render_status(console: Console, out: dict) -> None:
    h = [msg(f"cli.servers_col_{c}") for c in ("name", "state", "port", "pid")]
    rows = []
    for component, info in (out.get("components") or {}).items():
        pid = info.get("pid") if info.get("managed") else info.get("external_pid")
        rows.append({
            h[0]: SERVER_NAMES.get(component, component), h[1]: _state(info),
            h[2]: info.get("port") or "-", h[3]: pid if pid is not None else "-",
        })
    render_table(console, rows, h)
