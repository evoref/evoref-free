"""evoref projectmap サブコマンド

ProjectMap (c_16 §4.4) の現在版を表示する ``status`` と、手動更新を
トリガする ``update`` を持つ。実体は ``/api/rag/project_map`` を叩くだけの
薄いクライアント (書き手は sleep-time Step 5.87 / API 側と同じ)。
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx
from rich.console import Console

from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import (
    create_console,
    render_error,
    render_info,
    render_table,
)
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger, setup_cli_logging

logger = get_logger("cli.projectmap")

_DEFAULT_BACKEND = "http://localhost:8000"
#: 走査 + tree-sitter 抽出は数分掛かりうる (reindex と同様に同期で待つ)。
_TIMEOUT = 900.0


def _error_detail(resp: httpx.Response) -> str:
    """非 200 応答から人間可読なエラー理由を取り出す (reindex_command と同じ規則)。"""
    try:
        detail = resp.json().get("detail")
        if isinstance(detail, dict):
            message = str(detail.get("message") or "").strip()
            if message:
                return message
        elif isinstance(detail, str) and detail.strip():
            return detail.strip()
    except Exception:
        pass
    return f"HTTP {resp.status_code}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evoref projectmap",
        description=msg("cli.help_projectmap"),
    )
    parser.add_argument(
        "action", nargs="?", default="status", choices=["status", "update"],
        help="status (default): show current versions. update: trigger a manual update",
    )
    parser.add_argument(
        "--backend-url", default=_DEFAULT_BACKEND,
        help="Backend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the raw JSON response instead of a table",
    )
    return parser


def run_projectmap(argv: list[str]) -> int:
    """同期エントリーポイント"""
    if not _setup_encoding():
        return 1
    project_root = _find_project_root()
    setup_cli_logging(project_root=project_root, debug=False)
    init_i18n()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run_projectmap_async(args))
    except KeyboardInterrupt:
        return 130


async def _run_projectmap_async(args: argparse.Namespace) -> int:
    console = create_console()
    backend_url = args.backend_url.rstrip("/")

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        if args.action == "update":
            return await _run_update(client, backend_url, console, as_json=args.json)
        return await _run_status(client, backend_url, console, as_json=args.json)


def _render_status(console: Console, data: dict) -> None:
    render_info(
        console,
        msg("cli.projectmap_enabled_state" if data.get("enabled") else "cli.projectmap_disabled_state"),
    )
    roots = list(data.get("roots") or [])
    if not roots:
        render_info(console, msg("cli.projectmap_no_roots"))
        return
    headers = ["root", "package_id", "version", "update_kind", "languages", "nodes", "edges", "written_at"]
    rows = [
        {
            "root": r.get("root", ""),
            "package_id": r.get("package_id", ""),
            "version": r.get("version") or "-",
            "update_kind": r.get("update_kind") or "-",
            "languages": ", ".join(
                f"{lang}:{count}" for lang, count in (r.get("languages") or {}).items()
            ) or "-",
            "nodes": r.get("nodes", 0),
            "edges": r.get("edges", 0),
            "written_at": r.get("written_at") or "-",
        }
        for r in roots
    ]
    render_table(console, rows, headers)


async def _run_status(
    client: httpx.AsyncClient, backend_url: str, console: Console, *, as_json: bool,
) -> int:
    try:
        resp = await client.get(f"{backend_url}/api/rag/project_map")
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPError as exc:
        render_error(console, msg("cli.projectmap_failed", detail=str(exc)))
        return 1

    if resp.status_code != 200:
        render_error(console, msg("cli.projectmap_failed", detail=_error_detail(resp)))
        return 1

    data = resp.json()
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    _render_status(console, data)
    return 0


async def _run_update(
    client: httpx.AsyncClient, backend_url: str, console: Console, *, as_json: bool,
) -> int:
    render_info(console, msg("cli.projectmap_updating"))
    try:
        resp = await client.post(f"{backend_url}/api/rag/project_map/update")
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPError as exc:
        render_error(console, msg("cli.projectmap_failed", detail=str(exc)))
        return 1

    if resp.status_code == 409:
        render_error(console, msg("cli.projectmap_already_running"))
        return 1
    if resp.status_code != 200:
        render_error(console, msg("cli.projectmap_failed", detail=_error_detail(resp)))
        return 1

    result = resp.json()
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    render_info(
        console,
        msg("cli.projectmap_updated", count=int(result.get("updated_roots", 0))),
    )
    return 0
