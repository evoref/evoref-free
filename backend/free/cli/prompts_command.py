"""evoref prompts サブコマンド — プロンプトと補助タスクのプロンプトの表示・編集・巻き戻し (docs/f_06 §2.10)。

``list`` / ``show`` / ``history`` / ``set`` / ``rollback`` / ``reload`` / ``locale``。
``show`` / ``history`` / ``set`` / ``rollback`` は ``<mode>`` の代わりに ``--aux TASK`` で
補助タスクのプロンプト (``/api/aux-prompts``) を扱う。

稼働中の backend の API (``/api/prompts`` / ``/api/aux-prompts``) だけを使い、停止中の
直接の書き込みは持たない (f_06 §2.9.3)。組織ポリシーパックの固定 (403、docs/p_09 §4) や
Level 1 の実行中 (409)・readonly (423) は API が拒否し、その理由をそのまま表示する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from rich.console import Console

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
    page_params,
    request,
    resolve_backend_url,
)
from backend.free.cli.config_loader import _setup_encoding
from backend.free.cli.renderer import create_console, render_error, render_info, render_table
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger

logger = get_logger("cli.prompts")

_TIMEOUT = 30.0
#: ``<mode>`` か ``--aux TASK`` のどちらかを取る操作
_TARGET_ACTIONS = ("show", "history", "set", "rollback")
#: 確認を取る操作 (本文を置き換える)
_CONFIRM_ACTIONS = ("set", "rollback", "locale")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common)
    parser = argparse.ArgumentParser(prog="evoref prompts", description=msg("cli.help_prompts"))
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", parents=[common], help=msg("cli.prompts_help_list"))
    for action in _TARGET_ACTIONS:
        text = msg(f"cli.prompts_help_{action}")
        p = sub.add_parser(action, parents=[common], help=text, description=text)
        p.add_argument("mode", nargs="?", default=None, help="chat / create")
        p.add_argument("--aux", default=None, metavar="TASK", help="Aux task prompt instead of a mode (e.g. note_evolve)")
        if action == "set":
            p.add_argument(
                "--file", default=None, metavar="PATH",
                help="Read the new prompt from PATH ('-' or omitted: standard input)",
            )
        if action == "history":
            p.add_argument(
                "--version", type=_version_number, default=None, metavar="N",
                help="Print the text of archived version N instead of the list",
            )
            add_page_args(p)
        if action == "rollback":
            p.add_argument(
                "--to", type=int, required=True, metavar="VERSION",
                help="Version to roll back to (see history; read it first with history --version N)",
            )
    reload = sub.add_parser("reload", parents=[common], help=msg("cli.prompts_help_reload"))
    reload.add_argument("mode", help="chat / create")
    locale = sub.add_parser("locale", parents=[common], help=msg("cli.prompts_help_locale"))
    locale.add_argument("locale", choices=["ja", "en"])
    for action in _CONFIRM_ACTIONS:
        sub.choices[action].add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation")
    return parser


def _version_number(text: str) -> int:
    """``history --version`` の版番号 (1 以上の整数)。"""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid version: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"version must be 1 or more: {value}")
    return value


def run_prompts(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref prompts ...``)。"""
    if not _setup_encoding():
        return EXIT_FAIL
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    diag = diag_console(args.json, console)
    if args.action in _TARGET_ACTIONS and (args.mode is None) == (args.aux is None):
        render_error(diag, msg("cli.prompts_target_usage", action=args.action))
        return EXIT_UNAVAILABLE
    try:
        content = _read_content(args, diag) if args.action == "set" else None
        if args.action == "set" and content is None:
            return EXIT_FAIL
        if args.action in _CONFIRM_ACTIONS and not args.yes and not _confirmed(diag, args, content):
            return EXIT_FAIL
        with backend_client(_TIMEOUT) as client:
            out = _call(client, resolve_backend_url(args.backend_url), args, content)
    except BackendError as e:
        render_error(diag, msg("cli.prompts_failed", detail=e.detail))
        return EXIT_UNAVAILABLE if e.status in (423, 503) else EXIT_FAIL
    except httpx.HTTPError as e:
        logger.error("prompts %s failed: %s", args.action, e)
        render_error(diag, msg("cli.prompts_failed", detail=str(e)))
        return EXIT_FAIL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    if out is None:
        render_error(diag, msg("cli.backend_not_running"))
        return EXIT_UNAVAILABLE
    return emit(out, as_json=args.json, render=lambda: _render(console, args, out))


def _target(args: argparse.Namespace) -> str:
    return f"aux:{args.aux}" if args.aux else str(args.mode)


def _read_content(args: argparse.Namespace, console: Console) -> str | None:
    """``--file`` (``-`` / 省略は標準入力) の本文。読めない・空なら表示して ``None``。"""
    path = args.file
    if path in (None, "-"):
        if sys.stdin is None or sys.stdin.isatty():
            render_error(console, msg("cli.prompts_stdin_tty"))
            return None
        content = sys.stdin.read()
    else:
        try:
            content = Path(path).expanduser().read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            render_error(console, msg("cli.prompts_file_unreadable", path=path, detail=str(e)))
            return None
    if not content.strip():
        render_error(console, msg("cli.prompts_empty_content"))
        return None
    return content


def _confirmed(console: Console, args: argparse.Namespace, content: str | None) -> bool:
    """本文を置き換える操作の確認。対話端末でなければ確認できないので拒否する。"""
    if sys.stdin is None or not sys.stdin.isatty():
        render_error(console, msg("cli.prompts_needs_yes"))
        return False
    if args.action == "set":
        question = msg(
            "cli.prompts_confirm_set", target=_target(args), source=args.file or "stdin",
            chars=len(content or ""),
        )
    elif args.action == "rollback":
        question = msg("cli.prompts_confirm_rollback", target=_target(args), version=args.to)
    else:
        question = msg("cli.prompts_confirm_locale", locale=args.locale)
    answer = console.input(question + " [y/N] ", markup=False)
    if answer.strip().lower() in ("y", "yes"):
        return True
    render_info(console, msg("cli.prompts_aborted"))
    return False


# ── backend ──


def _call(client: httpx.Client, base: str, args: argparse.Namespace, content: str | None) -> Any:
    """backend を呼ぶ。届かなければ ``None``、2xx 以外は :class:`BackendError`。"""
    if args.action == "list":
        prompts = request(client, "GET", f"{base}/api/prompts")
        if prompts is None:
            return None
        return {"prompts": prompts, "aux_prompts": request(client, "GET", f"{base}/api/aux-prompts")}
    if args.action == "locale":
        return request(client, "POST", f"{base}/api/prompts/locale", json={"locale": args.locale})
    if args.action == "reload":
        return request(client, "POST", f"{base}/api/prompts/{quote(args.mode, safe='')}/reload")
    if args.aux:
        url = f"{base}/api/aux-prompts/{quote(args.aux, safe='')}"
    else:
        url = f"{base}/api/prompts/{quote(args.mode, safe='')}"
    if args.action == "show":
        return request(client, "GET", url)
    if args.action == "history":
        if args.version is not None:
            return request(client, "GET", f"{url}/history/{args.version}")
        return request(client, "GET", f"{url}/history", params=page_params(args) or None)
    if args.action == "set":
        return request(client, "PUT", url, json={"content": content})
    return request(client, "POST", f"{url}/rollback", json={"version": args.to})


# ── 表示 ──


def _render(console: Console, args: argparse.Namespace, out: Any) -> None:
    target = _target(args) if args.action in _TARGET_ACTIONS else ""
    if args.action == "list":
        _render_list(console, out)
    elif args.action == "show":
        render_info(console, msg(
            "cli.prompts_show_header", target=target, version=out.get("version"),
            source=out.get("source"), updated=out.get("updated_at"),
        ))
        if out.get("model_key"):
            render_info(console, msg(
                "cli.prompts_partition", label=out.get("model_label") or "-", model_key=out.get("model_key"),
            ))
        _write_verbatim(console, out)
    elif args.action == "history" and args.version is not None:
        render_info(console, msg(
            "cli.prompts_history_version_header", target=target, version=out.get("version"), file=out.get("file"),
        ))
        _write_verbatim(console, out)
    elif args.action == "history":
        if not out:
            render_info(console, msg("cli.prompts_history_empty", target=target))
            return
        h = [msg("cli.prompts_col_version"), msg("cli.prompts_col_file")]
        render_table(console, [{h[0]: f"v{r.get('version')}", h[1]: r.get("file")} for r in out], h)
    elif args.action == "set":
        render_info(console, msg("cli.prompts_updated", target=target, version=out.get("version")))
    elif args.action == "rollback":
        render_info(console, msg(
            "cli.prompts_rolled_back", target=target, to=args.to, version=out.get("version"),
        ))
    elif args.action == "reload":
        render_info(console, msg("cli.prompts_reloaded", mode=args.mode))
    else:
        versions = ", ".join(f"{k} v{v}" for k, v in (out.get("versions") or {}).items()) or "-"
        render_info(console, msg(
            "cli.prompts_locale_switched", locale=out.get("locale"), versions=versions,
            triggered=out.get("relearning_triggered"),
        ))


def _write_verbatim(console: Console, out: dict) -> None:
    """本文を端末幅で折り返さず逐語で出す (リダイレクトした本文を set --file へ戻せるように)。"""
    content = str(out.get("content") or "")
    console.file.write(content if content.endswith("\n") else content + "\n")


def _render_list(console: Console, out: dict) -> None:
    h = [msg(f"cli.prompts_col_{c}") for c in ("mode", "version", "source", "updated", "model")]
    render_table(console, [
        {
            h[0]: p.get("mode"), h[1]: f"v{p.get('version')}", h[2]: p.get("source"),
            h[3]: p.get("updated_at"), h[4]: p.get("model_label") or p.get("model_key") or "-",
        }
        for p in out.get("prompts") or []
    ], h)
    render_info(console, msg("cli.prompts_aux_title"))
    h = [msg(f"cli.prompts_col_{c}") for c in ("task", "version", "source", "updated")]
    render_table(console, [
        {h[0]: p.get("task"), h[1]: f"v{p.get('version')}", h[2]: p.get("source"), h[3]: p.get("updated_at")}
        for p in out.get("aux_prompts") or []
    ], h)
