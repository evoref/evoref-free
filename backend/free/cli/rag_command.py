"""evoref rag サブコマンド

- ``calibrate``: 再構築済みベクトルのスコア分布から ``rag.*`` 閾値の推奨値を出す
  (``POST /api/rag/calibrate-thresholds``)。正解ラベルが無いので提案だけで、
  ``--apply`` を付けたときだけ GUI の「適用」と同じく ``PUT /api/config/rag`` で書く。
- ``ingest <file>...``: 文書を 1 件ずつ corpus パッケージにして取り込む
  (``POST /api/rag/ingest``、c_16 §4.3)。書き手は backend 側。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
from rich.console import Console

from backend.free.cli.backend_headers import backend_headers
from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import create_console, render_error, render_info, render_table
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger, setup_cli_logging

logger = get_logger("cli.rag")

_DEFAULT_BACKEND = "http://localhost:8000"
_TIMEOUT = 60.0
#: 取り込みは抽出 + 埋め込みで文書の大きさに比例して掛かる。
_INGEST_TIMEOUT = 900.0
_SUGGESTION_KEYS = ("relevance_threshold", "support_threshold", "confidence_threshold")


def _error_detail(resp: httpx.Response) -> str:
    """非 2xx 応答から人間可読なエラー理由を取り出す。"""
    try:
        detail = resp.json().get("detail")
    except ValueError:
        return f"HTTP {resp.status_code}"
    if isinstance(detail, dict):
        message = str(detail.get("message") or "").strip()
        if message:
            return message
    elif isinstance(detail, str) and detail.strip():
        return detail.strip()
    return f"HTTP {resp.status_code}"


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend-url", default=_DEFAULT_BACKEND)
    common.add_argument("--json", action="store_true", help="Print the raw JSON response")
    parser = argparse.ArgumentParser(prog="evoref rag", description=msg("cli.help_rag_command"))
    sub = parser.add_subparsers(dest="action", required=True)
    cal = sub.add_parser(
        "calibrate", parents=[common], help="Suggest rag.* thresholds from the indexed vectors",
    )
    cal.add_argument(
        "--apply", action="store_true", help="Write the suggested values to the rag section",
    )
    ing = sub.add_parser("ingest", parents=[common], help="Ingest documents as corpus packages")
    ing.add_argument("paths", nargs="+", help="Document files")
    ing.add_argument("--category", default="document", help="Category label (default: document)")
    return parser


def run_rag(argv: list[str]) -> int:
    """同期エントリーポイント"""
    if not _setup_encoding():
        return 1
    setup_cli_logging(project_root=_find_project_root(), debug=False)
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    base = args.backend_url.rstrip("/")
    try:
        with httpx.Client(headers=backend_headers(), timeout=_TIMEOUT) as client:
            if args.action == "calibrate":
                return _run_calibrate(client, base, console, apply=args.apply, as_json=args.json)
            return _run_ingest(
                client, base, console, args.paths, category=args.category, as_json=args.json,
            )
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPError as exc:
        render_error(console, msg("cli.rag_failed", detail=str(exc)))
        return 1
    except KeyboardInterrupt:
        return 130


def _run_calibrate(
    client: httpx.Client, base: str, console: Console, *, apply: bool, as_json: bool,
) -> int:
    if not as_json:
        render_info(console, msg("cli.rag_calibrate_running"))
    resp = client.post(f"{base}/api/rag/calibrate-thresholds")
    if resp.status_code != 200:
        render_error(console, msg("cli.rag_failed", detail=_error_detail(resp)))
        return 1
    data = resp.json()
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    if not data.get("ok"):
        if data.get("reason") == "insufficient_vectors":
            render_error(console, msg("cli.rag_calibrate_insufficient", n=data.get("n_vectors", 0)))
        else:
            render_error(console, msg("cli.rag_calibrate_no_vectors"))
        return 1
    suggestions = {k: data["suggestions"][k] for k in _SUGGESTION_KEYS}
    if not as_json:
        dist = data.get("distribution") or {}
        render_table(
            console,
            [{"key": k, "value": f"{float(v):.3f}"} for k, v in dist.items()],
            ["key", "value"],
        )
        render_table(
            console,
            [{"key": f"rag.{k}", "suggested": f"{float(v):.3f}"} for k, v in suggestions.items()],
            ["key", "suggested"],
        )
    if not apply:
        render_info(console, msg("cli.rag_calibrate_advisory"))
        return 0

    put = client.put(f"{base}/api/config/rag", json={"data": suggestions})
    if put.status_code != 200:
        render_error(console, msg("cli.rag_failed", detail=_error_detail(put)))
        return 1
    logger.info("Applied calibrated rag thresholds: %s", suggestions)
    render_info(console, msg("cli.rag_calibrate_applied"))
    if put.json().get("restart_required", True):
        render_info(console, msg("cli.config_set_restart_required"))
    return 0


def _run_ingest(
    client: httpx.Client, base: str, console: Console, paths: list[str],
    *, category: str, as_json: bool,
) -> int:
    ok = 0
    failed = 0
    results: list[dict] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_file():
            render_error(console, msg("cli.rag_ingest_not_file", path=raw))
            failed += 1
            continue
        if not as_json:
            render_info(console, msg("cli.rag_ingest_running", path=str(path)))
        try:
            with path.open("rb") as fh:
                resp = client.post(
                    f"{base}/api/rag/ingest",
                    files={"file": (path.name, fh, "application/octet-stream")},
                    data={"category": category},
                    timeout=_INGEST_TIMEOUT,
                )
        except OSError as e:
            render_error(console, msg("cli.rag_ingest_unreadable", path=raw, detail=str(e)))
            failed += 1
            continue
        if resp.status_code not in (200, 201):
            render_error(console, msg("cli.rag_ingest_failed", path=raw, detail=_error_detail(resp)))
            failed += 1
            continue
        body = resp.json()
        results.append(body)
        ok += 1
        if not as_json:
            render_info(console, msg(
                "cli.rag_ingest_done",
                source=body.get("source", path.name),
                chunks=body.get("chunks_created", 0),
                tokens=body.get("tokens_total", 0),
                sec=body.get("ingest_time_sec", 0),
            ))
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    if len(paths) > 1:
        render_info(console, msg("cli.rag_ingest_summary", ok=ok, failed=failed))
    return 0 if failed == 0 else 1
