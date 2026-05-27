"""/migrate-model 対話コマンド

ベースモデル移行の CLI インターフェース。
バックエンド API を呼び出して移行処理を実行する。
"""

from __future__ import annotations

import sys

import httpx

from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.model_commands")



def _handle_migrate(
    base_url: str,
    *,
    new_model_path: str,
    try_lora: bool,
    regenerate_context: bool,
    dry_run: bool,
) -> int:
    """移行実行"""
    print(msg("cli.migrate_model_starting"))

    payload = {
        "new_model_path": new_model_path,
        "try_lora": try_lora,
        "regenerate_context": regenerate_context,
        "dry_run": dry_run,
    }

    try:
        resp = httpx.post(
            f"{base_url}/api/model/migrate",
            json=payload,
            timeout=120.0,
        )
    except httpx.ConnectError:
        print(msg("cli.backend_not_running"), file=sys.stderr)
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
            json=payload,
            timeout=60.0,
        )
    except httpx.ConnectError:
        print(msg("cli.backend_not_running"), file=sys.stderr)
        return 1

    if resp.status_code != 200:
        _print_error(resp)
        return 1

    data = resp.json()
    print(msg(
        "cli.migrate_model_rollback_done",
        model=data["rolled_back_to"],
        lora="restored" if data["lora_restored"] else "not available",
    ))
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
    """エラーレスポンスの表示"""
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    print(f"Error ({resp.status_code}): {detail}", file=sys.stderr)
