"""evoref reindex サブコマンド

埋め込みモデル変更後にベクトルストア・カートリッジ・メモリの
埋め込みを再計算する CLI コマンド。

``--semantic`` は意味記憶の索引だけを作り直す (``POST /api/model/reembed-facts``、
docs/p_03 §8 (C))。現在の埋め込みモデルの索引ディレクトリを消して全件を埋め直す
破壊的な操作なので、``--yes`` か対話端末での確認が要る (対話端末でなければ拒否)。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx
from rich.console import Console

from backend.free.cli.backend_headers import backend_headers
from backend.free.cli.cli_common import (
    EXIT_FAIL,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_UNAVAILABLE,
    BackendError,
    add_common_args,
    backend_client,
    request,
    resolve_backend_url,
)
from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import (
    create_console,
    render_error,
    render_info,
)
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger, setup_cli_logging

logger = get_logger("cli.reindex")

_TIMEOUT = 600.0
#: reembed-facts は全件の埋め直しを終えるまで応答しない (同期)
_SEMANTIC_TIMEOUT = 1800.0


def _error_detail(resp: httpx.Response) -> str:
    """非 200 応答から人間可読なエラー理由を取り出す。

    backend の構造化エラー (``{"detail": {"message": ...}}``) を優先し、
    素の文字列 detail にも対応する。取り出せない場合のみ HTTP コードへ
    フォールバック (E1007 等のローカライズ済み拒否理由を捨てない)。
    """
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
        prog="evoref reindex",
        description="Rebuild vector indices with the current embedding model",
    )
    add_common_args(parser, json=False)
    parser.add_argument(
        "--cartridge", default=None,
        help="Reindex only the specified cartridge ID",
    )
    parser.add_argument(
        "--semantic", action="store_true",
        help="Rebuild only the semantic memory index (drops and re-embeds every fact)",
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
    if args.semantic:
        return _run_semantic(args)
    try:
        return asyncio.run(_run_reindex_async(args))
    except KeyboardInterrupt:
        return 130


async def _run_reindex_async(args: argparse.Namespace) -> int:
    console = create_console()
    backend_url = resolve_backend_url(args.backend_url)

    async with httpx.AsyncClient(headers=backend_headers(), timeout=_TIMEOUT) as client:
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
                msg("cli.reindex_failed", detail=_error_detail(resp)),
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
                msg("cli.reindex_failed", detail=_error_detail(resp)),
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
        failed = list(result.get("cartridges_failed", []) or [])
        if failed:
            render_error(
                console,
                msg("cli.reindex_cart_failed", ids=", ".join(failed)),
            )
            return 1
        return 0


# ── --semantic (POST /api/model/reembed-facts) ──


def _run_semantic(args: argparse.Namespace) -> int:
    """意味記憶の索引の作り直し。

    ``--dry-run`` は API の ``dry_run=true`` (対象の件数だけを返し、何も消さない)。
    ``--yes`` が無ければ dry-run の結果を見せて対話端末で確認し、対話端末でなければ
    何も送らずに拒否する (終了コード 1)。
    """
    console = create_console()
    if args.cartridge:
        render_error(console, msg("cli.reindex_semantic_cartridge"))
        return EXIT_UNAVAILABLE
    if not args.dry_run and not args.yes and (sys.stdin is None or not sys.stdin.isatty()):
        render_error(console, msg("cli.reindex_semantic_needs_yes"))
        return EXIT_FAIL
    url = f"{resolve_backend_url(args.backend_url)}/api/model/reembed-facts"
    try:
        with backend_client(_SEMANTIC_TIMEOUT) as client:
            if args.dry_run or not args.yes:
                plan = request(client, "POST", url, params={"dry_run": "true"})
                if plan is None:
                    render_error(console, msg("cli.backend_not_running"))
                    return EXIT_UNAVAILABLE
                _render_semantic_plan(console, plan)
                if args.dry_run:
                    render_info(console, msg("cli.reindex_dry_run"))
                    return EXIT_OK
                try:
                    answer = console.input(msg("cli.reindex_semantic_confirm") + " [y/N] ", markup=False)
                except EOFError:
                    answer = ""
                if answer.strip().lower() not in ("y", "yes"):
                    render_info(console, msg("cli.reindex_canceled"))
                    return EXIT_FAIL
            render_info(console, msg("cli.reindex_running"))
            result = request(client, "POST", url)
    except BackendError as e:
        render_error(console, msg("cli.reindex_failed", detail=e.detail))
        return EXIT_UNAVAILABLE if e.status in (423, 503) else EXIT_FAIL
    except httpx.HTTPError as e:
        render_error(console, msg("cli.reindex_failed", detail=str(e)))
        return EXIT_FAIL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    if result is None:
        render_error(console, msg("cli.backend_not_running"))
        return EXIT_UNAVAILABLE
    render_info(console, msg(
        "cli.reindex_semantic_complete",
        facts=int(result.get("reembedded", 0)),
        model=result.get("new_model_id") or "-",
        snapshot=result.get("snapshot") or "-",
        elapsed=f"{float(result.get('elapsed_sec', 0.0)):.2f}",
    ))
    return EXIT_OK


def _render_semantic_plan(console: Console, plan: dict) -> None:
    render_info(console, msg(
        "cli.reindex_semantic_targets",
        facts=int(plan.get("fact_count", 0)),
        model=plan.get("new_model_id") or "-",
    ))
    if plan.get("model_changed"):
        render_info(console, msg("cli.reindex_semantic_model_changed"))
    render_info(console, msg("cli.reindex_semantic_impact"))
