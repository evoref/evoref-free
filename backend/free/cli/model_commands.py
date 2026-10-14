"""/migrate-model 対話コマンドと ``evoref model`` サブコマンド

ベースモデル移行の CLI インターフェース。
バックエンド API (``/api/model/*``) を呼び出して移行処理を実行する。

``migrate`` / ``rollback`` / ``history`` は ``--component embedding`` で埋め込みモデルの
移行 (``/api/model/{component}/*``) を扱う (既定の ``base`` は従来どおり)。
``confirm-reembed`` は埋め込みモデルの変更による再埋め込みの確認待ちを確認する
(``POST /api/model/reembed-confirm``、docs/f_06 §2.10)。埋め込みの側は
``cli_common`` の様式に載せ、backend に繋がらなければ終了コード 2。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx

from backend.free.cli.cli_common import (
    EXIT_FAIL,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_UNAVAILABLE,
    BackendError,
    add_common_args,
    backend_client,
    emit,
    request,
    resolve_backend_url,
)
from backend.free.cli.backend_headers import backend_headers
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.model_commands")

#: ``--component`` が取る値。``base`` は従来の ``/api/model/*``、それ以外は
#: ``/api/model/{component}/*`` (API の ``_VALID_COMPONENTS``)
_COMPONENTS = ("base", "embedding")
#: 移行の概念が無いコンポーネント (使い方の誤りとして案内する)
_NO_MIGRATION_COMPONENTS = ("rerank",)
_COMPONENT_ACTIONS = ("migrate", "rollback", "history")


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
    add_common_args(common, json=False)
    as_json = argparse.ArgumentParser(add_help=False)
    as_json.add_argument("--json", action="store_true", help="Print the raw JSON response")
    parser = argparse.ArgumentParser(prog="evoref model", description=msg("cli.help_model"))
    sub = parser.add_subparsers(dest="action")
    sub.add_parser(
        "status", parents=[common, as_json],
        help="Show model_state / config.yaml / served model consistency (default)",
    )
    component = argparse.ArgumentParser(add_help=False)
    component.add_argument(
        "--component", default="base", metavar="NAME",
        help="Model to operate on: base (default) / embedding",
    )
    mg = sub.add_parser(
        "migrate", parents=[common, component], help="Switch the base (or --component embedding) model",
    )
    mg.add_argument("new_model_path", help="Path to the new GGUF")
    mg.add_argument("--dry-run", action="store_true", help="Preview only")
    mg.add_argument("--regenerate-context", action="store_true", help="Base model only")
    mg.add_argument(
        "--no-restart", action="store_true",
        help="Embedding only: do not restart the managed llama-server (auto_restart=false)",
    )
    sub.add_parser(
        "reload", parents=[common],
        help="Reconnect to llama-server after it was restarted with the new model",
    )
    rb = sub.add_parser(
        "rollback", parents=[common, component],
        help="Roll back the last base (or --component embedding) model migration",
    )
    rb.add_argument(
        "--to", dest="target_model", default=None,
        help="GGUF file name to roll back to, resolved in the current model's "
             "directory (default: the model before the last migration)",
    )
    sub.add_parser(
        "history", parents=[common, as_json, component],
        help="Show the migration history of the base (or --component embedding) model",
    )
    sub.add_parser(
        "confirm-reembed", parents=[common, as_json],
        help="Confirm the embedding model change and re-embed the stores awaiting it (in the background)",
    )
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
    base = resolve_backend_url(args.backend_url)
    if args.action in _COMPONENT_ACTIONS and args.component != "base":
        return _dispatch_component(base, args)
    match args.action:
        case "migrate":
            if args.no_restart:
                print(msg("cli.model_no_restart_component_only"), file=sys.stderr)
                return EXIT_UNAVAILABLE
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
        case "confirm-reembed":
            return _handle_confirm_reembed(base, as_json=args.json)
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
    _print_history(data)
    return 0


# ────────────────────────────────────────────
# 埋め込みモデル (--component embedding) と再埋め込みの確認
# ────────────────────────────────────────────


def _component_usage_error(args: argparse.Namespace) -> str | None:
    """``--component`` の使い方の誤り (表示する文言)。正しければ ``None``。"""
    if args.component in _NO_MIGRATION_COMPONENTS:
        return msg("cli.model_component_no_migration", component=args.component)
    if args.component not in _COMPONENTS:
        return msg("cli.model_component_unknown", component=args.component, choices=" / ".join(_COMPONENTS))
    if args.action == "migrate" and args.regenerate_context:
        return msg("cli.model_regenerate_context_base_only")
    return None


def _dispatch_component(base: str, args: argparse.Namespace) -> int:
    """``--component`` が base 以外の ``migrate`` / ``rollback`` / ``history``。"""
    error = _component_usage_error(args)
    if error is not None:
        print(error, file=sys.stderr)
        return EXIT_UNAVAILABLE
    url = f"{base}/api/model/{args.component}"
    if args.action == "migrate":
        body = {
            "new_model_path": args.new_model_path,
            "dry_run": args.dry_run,
            "auto_restart": not args.no_restart,
        }
        code, out = _call("POST", f"{url}/migrate", timeout=300.0, json=body)
        if out is not None:
            _print_component_migrate(out)
        return code
    if args.action == "rollback":
        body = {"target_model": args.target_model} if args.target_model else {}
        code, out = _call("POST", f"{url}/rollback", timeout=60.0, json=body)
        if out is not None:
            print(msg(
                "cli.model_component_rolled_back",
                component=args.component, model=out.get("rolled_back_to") or "-",
            ))
            print(msg("cli.model_component_rollback_hint", component=args.component))
        return code
    code, out = _call("GET", f"{url}/migration-history", timeout=30.0)
    if out is None:
        return code
    return emit(out, as_json=args.json, render=lambda: _print_history(out))


def _handle_confirm_reembed(base: str, *, as_json: bool) -> int:
    """埋め込みモデルの変更を確認し、確認待ちのストアの再埋め込みを裏で始めさせる"""
    code, out = _call("POST", f"{base}/api/model/reembed-confirm", timeout=60.0)
    if out is None:
        return code

    def render() -> None:
        stores = [str(s) for s in out.get("confirmed") or []]
        if stores:
            print(msg("cli.model_reembed_confirmed", stores=", ".join(stores)))
        else:
            print(msg("cli.model_reembed_none"))

    return emit(out, as_json=as_json, render=render)


def _call(method: str, url: str, *, timeout: float, **kwargs: Any) -> tuple[int, Any]:
    """backend を 1 回呼ぶ。``(終了コード, 本文)``。失敗は stderr に出して本文 ``None``。

    繋がらない・423 / 503 は 2 (実行不能)、それ以外の拒否 (400 / 404 / 409 / 422 / 500) と
    通信の失敗は 1。
    """
    try:
        with backend_client(timeout) as client:
            out = request(client, method, url, **kwargs)
    except BackendError as e:
        detail = f"{e.detail} ({'; '.join(e.errors)})" if e.errors else e.detail
        print(f"Error ({e.status}): {detail}", file=sys.stderr)
        return (EXIT_UNAVAILABLE if e.status in (423, 503) else EXIT_FAIL), None
    except httpx.HTTPError as e:
        print(msg("cli.model_request_failed", detail=str(e)), file=sys.stderr)
        return EXIT_FAIL, None
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED, None
    if out is None:
        print(msg("cli.backend_not_running"), file=sys.stderr)
        return EXIT_UNAVAILABLE, None
    return EXIT_OK, out


def _print_component_migrate(data: dict) -> None:
    """コンポーネントの移行結果の表示"""
    component = data.get("component") or "-"
    print(msg(
        "cli.model_component_migrated",
        component=component,
        old=data.get("old_model") or "-",
        new=data.get("new_model") or "-",
        dry_run=" (DRY RUN)" if data.get("dry_run") else "",
    ))
    if not data.get("dry_run"):
        key = "cli.model_component_restarted" if data.get("restarted") else "cli.model_component_not_restarted"
        print(msg(key, component=component))
    for i, rec in enumerate(data.get("recommendations") or [], 1):
        print(f"  {i}. {rec}")
    if not data.get("dry_run") and data.get("old_model") != data.get("new_model"):
        print(msg("cli.model_embedding_next_steps"))


def _print_history(data: dict) -> None:
    print(msg("cli.model_history_current", name=data.get("current_model") or "-"))
    history = data.get("history") or []
    if not history:
        print(msg("cli.model_history_empty"))
        return
    for h in history:
        print(f"  {h.get('migrated_at', '')}  {h.get('from_model', '')} -> {h.get('to_model', '')}")
