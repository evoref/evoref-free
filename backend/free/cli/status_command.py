"""evoref status サブコマンド — 稼働中の backend の状態 (``GET /api/status``、docs/f_06 §2.10)。

REPL の ``/status`` (版・稼働時間・モデル・記憶の件数) に、データの健全性
(``data_health``)・効果の死活監視 (``liveness``)・自己学習の判定 (``learning_health``) と
リランカーを足した要点を出す。``--json`` は応答の全文。backend が居なければ停止中の
点検 (``evoref doctor``) を案内して終了コード 2。
"""

from __future__ import annotations

import argparse
from pathlib import PureWindowsPath
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
from backend.free.cli.renderer import create_console, render_error, render_info
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger

logger = get_logger("cli.status")

_TIMEOUT = 30.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evoref status", description=msg("cli.help_status_command"))
    add_common_args(parser)
    return parser


def run_status(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref status``)。"""
    if not _setup_encoding():
        return EXIT_FAIL
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    diag = diag_console(args.json, console)
    try:
        with backend_client(_TIMEOUT) as client:
            out = request(client, "GET", f"{resolve_backend_url(args.backend_url)}/api/status")
    except BackendError as e:
        render_error(diag, msg("cli.status_failed", detail=e.detail))
        return EXIT_FAIL
    except httpx.HTTPError as e:
        logger.error("status failed: %s", e)
        render_error(diag, msg("cli.status_failed", detail=str(e)))
        return EXIT_FAIL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    if out is None:
        render_error(diag, msg("cli.status_not_running"))
        return EXIT_UNAVAILABLE
    return emit(out, as_json=args.json, render=lambda: _render(console, out))


def _name(path_or_name: Any) -> str:
    """モデル名が GGUF のパスならファイル名の stem だけにする。"""
    text = str(path_or_name or "")
    return PureWindowsPath(text).stem if text.lower().endswith(".gguf") else text


def _uptime(seconds: Any) -> str:
    h, rem = divmod(int(seconds or 0), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def _render(console: Console, out: dict) -> None:
    render_info(console, msg(
        "cli.status_header", instance=escape(str(out.get("instance_name") or "evoref")),
        version=out.get("version"), edition=out.get("edition"), status=out.get("status"),
        uptime=_uptime(out.get("uptime_seconds")),
    ))
    components = out.get("components") or []
    roles = ("base", "embed")
    for role, comp in zip(roles, components, strict=False):
        render_info(console, msg(
            "cli.status_component", role=role, name=escape(_name(comp.get("name")) or "-"),
            state=msg("cli.status_connected" if comp.get("connected") else "cli.status_disconnected"),
        ))
    rerank = out.get("rerank") or {}
    if rerank.get("enabled"):
        render_info(console, msg(
            "cli.status_component", role="rerank", name=f"{rerank.get('placement') or '-'}",
            state=msg("cli.status_connected" if rerank.get("connected") else "cli.status_disconnected"),
        ))
    memory = out.get("memory") or {}
    render_info(console, msg(
        "cli.status_memory_line", wm=memory.get("working_turns", 0),
        short=memory.get("short_term_notes", 0), long=memory.get("long_term_chunks", 0),
    ))
    _render_data_health(console, out.get("data_health") or {})
    for alert in out.get("liveness") or []:
        render_error(console, msg(
            "cli.status_liveness", stage=alert.get("stage"), kind=alert.get("kind"),
            detail=escape(str(alert.get("detail") or "")),
        ), level="warning")
    learning = out.get("learning_health") or {}
    render_info(console, msg("cli.status_learning", verdict=learning.get("verdict", "-")))
    for reason in learning.get("reasons") or []:
        render_info(console, f"  - {escape(str(reason))}")


def _render_data_health(console: Console, health: dict) -> None:
    if health.get("data_root"):
        render_info(console, msg("cli.status_data_root", path=escape(str(health["data_root"]))))
    if health.get("readonly"):
        render_error(console, msg("cli.status_readonly", reason=escape(str(health.get("reason") or "-"))), level="warning")
    for warning in health.get("warnings") or []:
        render_error(console, escape(str(warning)), level="warning")
    if health.get("reembed_pending"):
        render_error(console, msg(
            "cli.status_reembed_pending", stores=", ".join(health["reembed_pending"]),
        ), level="warning")
    if health.get("served_model_mismatch"):
        render_error(console, msg(
            "cli.status_served_mismatch", served=escape(str(health.get("served_model") or "-")),
            expected=escape(str(health.get("expected_model") or "-")),
        ), level="warning")
    if health.get("degraded"):
        render_error(console, msg("cli.status_degraded", formats=", ".join(health["degraded"])), level="warning")
