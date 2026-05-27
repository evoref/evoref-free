"""evoref reindex サブコマンド

埋め込みモデル変更後にベクトルストア・カートリッジ・メモリの
埋め込みを再計算する CLI コマンド。
"""

from __future__ import annotations

import argparse
import asyncio

import httpx

from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import (
    create_console,
    render_error,
    render_info,
)
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger, setup_cli_logging

logger = get_logger("cli.reindex")

_DEFAULT_BACKEND = "http://localhost:8000"
_TIMEOUT = 600.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evoref reindex",
        description="Rebuild vector indices with the current embedding model",
    )
    parser.add_argument(
        "--backend-url", default=_DEFAULT_BACKEND,
        help="Backend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--cartridge", default=None,
        help="Reindex only the specified cartridge ID",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show targets without rebuilding",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt",
    )
    return parser


def run_reindex(argv: list[str]) -> int:
    """同期エントリーポイント"""
    if not _setup_encoding():
        return 1
    project_root = _find_project_root()
    setup_cli_logging(project_root=project_root, debug=False)
    init_i18n()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run_reindex_async(args))
    except KeyboardInterrupt:
        return 130


async def _run_reindex_async(args: argparse.Namespace) -> int:
    console = create_console()
    backend_url = args.backend_url.rstrip("/")

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        # 1. dry-run でターゲット件数を取得
        try:
            params = {"dry_run": "true"}
            if args.cartridge:
                params["cartridge"] = args.cartridge
            resp = await client.post(
                f"{backend_url}/api/rag/reindex", params=params,
            )
        except httpx.ConnectError:
            render_error(console, msg("cli.backend_not_running"))
            return 1
        except httpx.HTTPError as exc:
            render_error(console, msg("cli.reindex_failed", detail=str(exc)))
            return 1

        if resp.status_code != 200:
            render_error(
                console,
                msg("cli.reindex_failed", detail=f"HTTP {resp.status_code}"),
            )
            return 1

        plan = resp.json()
        rag_chunks = int(plan.get("rag_chunks", 0))
        cart_chunks = int(plan.get("cartridge_chunks", 0))
        carts = list(plan.get("cartridges", []))
        mem_notes = int(plan.get("memory_notes", 0))

        render_info(console, msg(
            "cli.reindex_targets",
            rag_chunks=rag_chunks, cart_chunks=cart_chunks,
            cart_count=len(carts), mem_notes=mem_notes,
        ))

        if rag_chunks == 0 and cart_chunks == 0 and mem_notes == 0:
            render_info(console, msg("cli.reindex_no_data"))
            return 0

        if args.dry_run:
            render_info(console, msg("cli.reindex_dry_run"))
            return 0

        # 2. 確認プロンプト
        if not args.yes:
            try:
                ans = input(msg("cli.reindex_confirm") + " ").strip().lower()
            except EOFError:
                ans = ""
            if ans not in ("y", "yes"):
                render_info(console, msg("cli.reindex_canceled"))
                return 0

        # 3. 実行
        render_info(console, msg("cli.reindex_running"))
        try:
            params = {}
            if args.cartridge:
                params["cartridge"] = args.cartridge
            resp = await client.post(
                f"{backend_url}/api/rag/reindex", params=params,
            )
        except httpx.HTTPError as exc:
            render_error(console, msg("cli.reindex_failed", detail=str(exc)))
            return 1

        if resp.status_code != 200:
            render_error(
                console,
                msg("cli.reindex_failed", detail=f"HTTP {resp.status_code}"),
            )
            return 1

        result = resp.json()
        render_info(console, msg(
            "cli.reindex_complete",
            rag_chunks=int(result.get("rag_chunks", 0)),
            cart_chunks=int(result.get("cartridge_chunks", 0)),
            cart_count=len(result.get("cartridges_rebuilt", [])),
            mem_notes=int(result.get("memory_notes_reset", 0)),
            elapsed=f"{float(result.get('elapsed_sec', 0.0)):.2f}",
        ))
        return 0
