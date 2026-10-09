"""/migrate-model 対話コマンドと ``evoref model`` サブコマンド

ベースモデル移行の CLI インターフェース。
バックエンド API (``/api/model/*``) を呼び出して移行処理を実行する。
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from backend.free.cli.backend_headers import backend_headers
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.model_commands")

_DEFAULT_BACKEND = "http://localhost:8000"


def _handle_migrate(
    base_url: str,
    *,
    new_model_path: str,
    regenerate_context: bool,
    dry_run: bool,
) -> int:
    """移行実行"""
    print(msg("cli.migrate_model_starting"))

    payload = {
        "new_model_path": new_model_path,
        "regenerate_context": regenerate_context,
        "dry_run": dry_run,
    }

    try:
        resp = httpx.post(
            f"{base_url}/api/model/migrate",
            headers=backend_headers(),
            json=payload,
            timeout=120.0,
        )
    except httpx.ConnectError:
        print(msg("cli.backend_not_running"), file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(msg("cli.model_request_failed", detail=str(exc)), file=sys.stderr)
        return 1

    if resp.status_code != 200:
        _print_error(resp)
        return 1

    data = resp.json()
    _print_migrate_result(data)

    # 移行成功後、リロードを提案
    if not dry_run:
        print()
        print(msg("cli.migrate_model_reload_hint"))

    return 0


def _handle_rollback(base_url: str, target_model: str | None) -> int:
    """ロールバック実行"""
    print(msg("cli.migrate_model_rollback_starting"))

    payload: dict = {}
    if target_model:
        payload["target_model"] = target_model

    try:
        resp = httpx.post(
            f"{base_url}/api/model/rollback",
            headers=backend_headers(),
            json=payload,
            timeout=60.0,
        )
    except httpx.ConnectError:
        print(msg("cli.backend_not_running"), file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(msg("cli.model_request_failed", detail=str(exc)), file=sys.stderr)
        return 1

    if resp.status_code != 200:
        _print_error(resp)
        return 1

    data = resp.json()
    print(msg("cli.migrate_model_rollback_done", model=data["rolled_back_to"]))
    return 0


def _print_migrate_result(data: dict) -> None:
    """移行結果の表示"""
    dry_label = " (DRY RUN)" if data.get("dry_run") else ""
    print(f"\n{'=' * 50}")
    print(msg(
        "cli.migrate_model_summary",
        old=data["old_model"],
        new=data["new_model"],
        dry_run=dry_label,
    ))
    print(f"{'=' * 50}")

    print(f"\n  LoRA: {data['lora_action']}")

    summary = data.get("data_summary", {})
    print(f"\n  {msg('cli.migrate_model_data_kept')}:")
    print(f"    {msg('cli.migrate_model_memory_notes')}: {summary.get('memory_notes', 0)}")
    print(f"    {msg('cli.migrate_model_experience')}: {summary.get('experience_entries', 0)}")
    print(f"    {msg('cli.migrate_model_perplexity_reset')}: {summary.get('perplexity_reset', 0)}")
    print(f"    RAG: {summary.get('rag_chunks', 0)} chunks")
    print(f"    {msg('cli.migrate_model_cartridges')}: {summary.get('cartridges', 0)}")
    modes = summary.get("prompts_modes", [])
    if modes:
        print(f"    {msg('cli.migrate_model_prompts')}: {', '.join(modes)}")

    recs = data.get("recommendations", [])
    if recs:
        print(f"\n  {msg('cli.migrate_model_recommendations')}:")
        for i, rec in enumerate(recs, 1):
            print(f"    {i}. {rec}")


def _print_error(resp: httpx.Response) -> None:
    """エラーレスポンスの表示

    移行 API のエラーは ``detail`` が ``{"code", "message", ...}`` の dict なので、
    そのまま出すと dict の repr が表示される。``message`` を取り出す。
    """
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    if isinstance(detail, dict):
        detail = detail.get("message") or detail
    print(f"Error ({resp.status_code}): {detail}", file=sys.stderr)


# ────────────────────────────────────────────
# evoref model サブコマンド
# ────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend-url", default=_DEFAULT_BACKEND)
    as_json = argparse.ArgumentParser(add_help=False)
    as_json.add_argument("--json", action="store_true", help="Print the raw JSON response")
    parser = argparse.ArgumentParser(prog="evoref model", description=msg("cli.help_model"))
    sub = parser.add_subparsers(dest="action")
    sub.add_parser(
        "status", parents=[common, as_json],
        help="Show model_state / config.yaml / served model consistency (default)",
    )
    mg = sub.add_parser("migrate", parents=[common], help="Switch the base model")
    mg.add_argument("new_model_path", help="Path to the new base model GGUF")
    mg.add_argument("--dry-run", action="store_true", help="Preview only")
    mg.add_argument("--regenerate-context", action="store_true")
    sub.add_parser(
        "reload", parents=[common],
        help="Reconnect to llama-server after it was restarted with the new model",
    )
    rb = sub.add_parser(
        "rollback", parents=[common], help="Roll back the last base model migration",
    )
    rb.add_argument(
        "--to", dest="target_model", default=None,
        help="GGUF file name to roll back to, resolved in the current base model's "
             "directory (default: the model before the last migration)",
    )
    sub.add_parser("history", parents=[common, as_json], help="Show the migration history")
    return parser


def run_model(argv: list[str]) -> int:
    """``evoref model`` の同期エントリーポイント"""
    from backend.free.cli.config_loader import _find_project_root, _setup_encoding
    from backend.i18n_helper import init_i18n
    from backend.log_config import setup_cli_logging

    if not _setup_encoding():
        return 1
    setup_cli_logging(project_root=_find_project_root(), debug=False)
    init_i18n()
    # 操作を省くと status (``evoref model --json`` も status として読む)
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["status", *argv]
    return _dispatch(_build_parser().parse_args(argv))


def _dispatch(args: argparse.Namespace) -> int:
    base = args.backend_url.rstrip("/")
    match args.action:
        case "migrate":
            return _handle_migrate(
                base,
                new_model_path=args.new_model_path,
                regenerate_context=args.regenerate_context,
                dry_run=args.dry_run,
            )
        case "rollback":
            return _handle_rollback(base, args.target_model)
        case "reload":
            return _handle_reload(base)
        case "history":
            return _handle_history(base, as_json=args.json)
        case _:
            return _handle_status(base, as_json=args.json)


def _request(method: str, url: str, timeout: float) -> httpx.Response | None:
    """1 回だけ叩く。接続失敗・HTTP 失敗・200 以外はメッセージを出して None。"""
    try:
        resp = httpx.request(method, url, headers=backend_headers(), timeout=timeout)
    except httpx.ConnectError:
        print(msg("cli.backend_not_running"), file=sys.stderr)
        return None
    except httpx.HTTPError as exc:
        print(msg("cli.model_request_failed", detail=str(exc)), file=sys.stderr)
        return None
    if resp.status_code != 200:
        _print_error(resp)
        return None
    return resp


def _handle_status(base_url: str, *, as_json: bool) -> int:
    """model_state.json / config.yaml / llama-server の整合性を表示"""
    resp = _request("GET", f"{base_url}/api/model/state", 30.0)
    if resp is None:
        return 1
    data = resp.json()
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print(msg("cli.model_status_state", name=data.get("current_filename") or "-"))
    print(msg("cli.model_status_config", name=data.get("config_filename") or "-"))
    if data.get("config_mismatch"):
        print(msg("cli.model_status_mismatch", detail=data.get("recommendation") or ""))
    if data.get("served_mismatch"):
        print(msg("cli.model_status_served", name=data.get("served_filename") or "-"))
        print(msg("cli.model_status_mismatch", detail=data.get("served_recommendation") or ""))
    return 0


def _handle_reload(base_url: str) -> int:
    """llama-server へ再接続し、model_state を config の base_model に合わせる"""
    resp = _request("POST", f"{base_url}/api/model/reload", 60.0)
    if resp is None:
        return 1
    print(msg("cli.model_reloaded", model_id=resp.json().get("model_id") or "-"))
    return 0


def _handle_history(base_url: str, *, as_json: bool) -> int:
    """ベースモデルの移行履歴を表示"""
    resp = _request("GET", f"{base_url}/api/model/migration-history", 30.0)
    if resp is None:
        return 1
    data = resp.json()
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print(msg("cli.model_history_current", name=data.get("current_model") or "-"))
    history = data.get("history") or []
    if not history:
        print(msg("cli.model_history_empty"))
        return 0
    for h in history:
        print(f"  {h.get('migrated_at', '')}  {h.get('from_model', '')} -> {h.get('to_model', '')}")
    return 0
