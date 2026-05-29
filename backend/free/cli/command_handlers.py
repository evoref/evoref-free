"""CLI コマンドハンドラ: 各コマンドの実行ロジック"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from backend.free.cli.command_parser import (
    CommandResult,
    SessionState,
)
from backend.free.cli.renderer import (
    render_error,
    render_help,
    render_info,
)
from backend.free.cli.session_persistence import (
    _load_from_history,
    _restore_session,
    auto_save_session,
    finalize_session,
)
from backend.free.history.utils import parse_iso
from backend.i18n_helper import msg
from backend.log_config import get_logger
from backend.utils import utc_now_dt

logger = get_logger("cli.command_handlers")


def _cmd_help(args: str, state: SessionState, console) -> CommandResult:
    render_help(console)
    return CommandResult()


def _cmd_file(args: str, state: SessionState, console) -> CommandResult:
    if not args:
        if state.context_files:
            render_info(
                console,
                msg("cli.context_files", files=", ".join(
                    Path(p).name for p in state.context_files
                )),
            )
            # 各ファイルのチャンク数を表示
            for p in state.context_files:
                chunks = state.file_chunks.get(p, [])
                render_info(
                    console,
                    msg("cli.file_info", path=Path(p).name, chunks=len(chunks)),
                )
        else:
            render_info(console, msg("cli.no_context_files"))
        return CommandResult()

    from pathlib import Path as _Path
    path = _Path(args.strip()).resolve()
    if not path.exists():
        logger.debug("/file: not found: %s", path)
        render_error(console, msg("cli.file_not_found", path=path))
        return CommandResult()

    if not path.is_file():
        render_error(console, msg("cli.file_not_file", path=path))
        return CommandResult()

    # 既に追加済みチェック
    path_str = str(path)
    if path_str in state.context_files:
        render_info(console, msg("cli.file_already_added", path=path.name))
        return CommandResult()

    # ファイル読込み・チャンク分割
    from backend.free.cli.file_reader import FileReaderError, read_and_chunk
    try:
        result = read_and_chunk(path)
    except FileReaderError as e:
        error_key = str(e)
        if error_key.startswith("legacy_format:"):
            render_error(console, msg("cli.file_legacy_unsupported", path=path.name))
        elif error_key.startswith("unsupported_format:"):
            render_error(console, msg("cli.file_binary_unsupported", path=path.name))
        elif error_key.startswith("missing_library:"):
            lib = error_key.split(":", 1)[1]
            render_error(console, msg("cli.file_missing_library", library=lib))
        elif error_key == "scan_only_pdf":
            render_error(console, msg("cli.file_scan_only_pdf", path=path.name))
        elif error_key == "empty_content":
            render_error(console, msg("cli.file_empty", path=path.name))
        elif error_key.startswith("encoding_error:"):
            render_error(console, msg("cli.file_encoding_error", path=path.name))
        else:
            render_error(console, msg("cli.file_read_error", path=path.name, detail=error_key))
        return CommandResult()

    state.context_files.append(path_str)
    state.file_chunks[path_str] = result.chunks

    logger.debug(
        "/file: added %s (%d chunks, %d chars, type=%s)",
        path.name, result.chunk_count, result.total_chars, result.file_type,
    )
    render_info(
        console,
        msg("cli.file_added", path=path.name, chunks=result.chunk_count),
    )
    return CommandResult()


def _cmd_clear(args: str, state: SessionState, console) -> CommandResult:
    logger.debug(
        "/clear: clearing %d context files, %d turns, resetting tokens",
        len(state.context_files), len(state.turns),
    )
    # /clear 前に自動保存（設計書 23.3.1: リセットされるターンを保存）
    auto_save_session(state)
    state.context_files.clear()
    state.file_chunks.clear()
    state.turns.clear()
    state.token_used = 0
    render_info(console, msg("cli.history_cleared"))
    return CommandResult()


def _cmd_save(args: str, state: SessionState, console) -> CommandResult:
    """セッションを local/sessions/{name}.json に保存"""
    name = args.strip() or "default"

    if state.sessions_dir is None:
        render_error(console, "Sessions directory not configured")
        return CommandResult()

    state.sessions_dir.mkdir(parents=True, exist_ok=True)

    pct = int(state.token_used / state.token_limit * 100) if state.token_limit > 0 else 0
    session_data = {
        "session_id": state.session_id,
        "name": name,
        "saved_at": utc_now_dt().isoformat(),
        "mode": state.mode,
        "source": "manual",
        "instance_name": state.instance_name,
        "turns": state.turns,
        "active_note_ids": [],
        "context_files": list(state.context_files),
        "token_info": {
            "used": state.token_used,
            "limit": state.token_limit,
            "pct": pct,
        },
    }

    path = state.sessions_dir / f"{name}.json"
    try:
        path.write_text(
            json.dumps(session_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("/save: wrote %s (%d turns)", path, len(state.turns))
        state.manually_saved = True
        render_info(console, msg("cli.history_saved", name=name))
    except OSError as e:
        logger.debug("/save: failed to write %s: %s", path, e)
        render_error(console, f"Failed to save session: {e}")
    return CommandResult()


def _cmd_load(args: str, state: SessionState, console) -> CommandResult:
    """セッションを復元。--history <id> / --history --latest にも対応"""
    parts = args.strip().split()

    if "--history" in parts:
        return _load_from_history(parts, state, console)

    name = args.strip() or "default"
    if state.sessions_dir is None:
        render_error(console, "Sessions directory not configured")
        return CommandResult()

    path = state.sessions_dir / f"{name}.json"
    if not path.exists():
        logger.debug("/load: file not found: %s", path)
        render_error(console, f"Session not found: {name}")
        return CommandResult()

    return _restore_session(path, state, console)


async def _cmd_history(args: str, state: SessionState, console) -> CommandResult:
    """/history コマンド: 会話履歴の一覧・検索・詳細・統計・圧縮・削除

    Usage:
        /history                  — 直近10件を一覧表示
        /history list [-n N]      — 件数指定で一覧表示
        /history search <query>   — キーワード検索
        /history show <id>        — セッション詳細表示
        /history stats            — 統計情報
        /history compact          — 圧縮処理を実行
        /history delete <id>      — セッション削除
        /history prune <date>     — 指定日以前を削除 (YYYY-MM-DD)
    """
    from backend.free.cli.history_commands import (
        _history_compact,
        _history_delete,
        _history_list,
        _history_prune,
        _history_search,
        _history_show,
        _history_stats,
    )

    parts = args.strip().split()
    action = parts[0] if parts else ""
    rest = parts[1:] if len(parts) > 1 else []
    backend_url = state.backend_url

    if action == "" or action == "list":
        # /history or /history list [-n N]
        limit = 10
        if "-n" in rest:
            idx = rest.index("-n")
            if idx + 1 < len(rest):
                try:
                    limit = int(rest[idx + 1])
                except ValueError:
                    pass
        _history_list(backend_url, console, limit=limit)
        return CommandResult()

    if action == "search":
        query = " ".join(rest)
        if not query:
            render_error(console, "Usage: /history search <query>")
            return CommandResult()
        _history_search(backend_url, console, query=query)
        return CommandResult()

    if action == "show":
        session_id = rest[0] if rest else ""
        if not session_id:
            render_error(console, "Usage: /history show <session_id>")
            return CommandResult()
        _history_show(backend_url, console, session_id)
        return CommandResult()

    if action == "stats":
        _history_stats(backend_url, console)
        return CommandResult()

    if action == "compact":
        _history_compact(backend_url, console)
        return CommandResult()

    if action == "delete":
        session_id = rest[0] if rest else ""
        if not session_id:
            render_error(console, "Usage: /history delete <session_id>")
            return CommandResult()
        _history_delete(backend_url, console, session_id)
        return CommandResult()

    if action == "prune":
        before = rest[0] if rest else ""
        if not before:
            render_error(console, "Usage: /history prune <YYYY-MM-DD>")
            return CommandResult()
        _history_prune(backend_url, console, before)
        return CommandResult()

    render_error(console, msg("cli.history_unknown_action", action=action))
    return CommandResult()


async def _cmd_page(args: str, state: SessionState, console) -> CommandResult:
    """最後の run_command 全文出力をページャーで表示"""
    logger.debug("/page: fetching last command output from backend")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{state.backend_url}/api/commands/last-output")
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return CommandResult()
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code}")
        return CommandResult()

    if not data.get("has_output"):
        render_info(console, msg("cli.page_no_output"))
        return CommandResult()

    output = data["output"]
    total_lines = data.get("total_lines", 0)
    render_info(console, msg("cli.page_showing", lines=total_lines))

    with console.pager(styles=True):
        console.print(output)

    return CommandResult()


async def _build_model_info_async(
    project_root: Path,
    status_data: dict | None,
) -> tuple[list, int | None]:
    """config.yaml + 各ポートの非同期ヘルスチェックからモデル情報を構築"""
    from backend.free.cli.renderer import ModelInfoItem
    from backend.free.services.model_service import build_model_info_async

    result = await build_model_info_async(project_root, status_data)
    # サービス層の ModelInfoItem を renderer の ModelInfoItem に変換
    models = [
        ModelInfoItem(label=m.label, name=m.name, connected=m.connected)
        for m in result[0]
    ]
    return models, result[1]


async def _cmd_status(args: str, state: SessionState, console) -> CommandResult:
    """サーバー状態を表示"""
    from backend.free.cli.config_loader import _find_project_root
    from backend.free.cli.renderer import render_model_info

    logger.debug("/status: fetching server status from backend")

    # /api/status からバックエンド情報を取得
    status_data: dict | None = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{state.backend_url}/api/status")
            resp.raise_for_status()
            status_data = resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return CommandResult()
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code}")
        return CommandResult()

    from backend.version import get_runtime_version
    instance_name = status_data.get("instance_name", "evoref")
    version = status_data.get("version", get_runtime_version())
    uptime = status_data.get("uptime_seconds", 0)
    debug_section = status_data.get("debug", {})
    debug_on = debug_section.get("enabled", False)

    h, rem = divmod(int(uptime), 3600)
    m, s = divmod(rem, 60)
    uptime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    memory = status_data.get("memory", {})
    wm = memory.get("working_turns", 0)
    stm = memory.get("short_term_notes", 0)
    ltm = memory.get("long_term_chunks", 0)

    # コンポーネント情報（CLI 側で直接ヘルスチェック）
    project_root = _find_project_root()
    models, context_size = await _build_model_info_async(project_root, status_data)

    # ヘッダー → モデル一覧 → 追加情報
    render_info(
        console,
        f"{instance_name} v{version}  "
        f"{msg('cli.status_uptime')}: {uptime_str}  "
        f"{msg('cli.status_debug')}: {'ON' if debug_on else 'OFF'}",
    )
    render_model_info(console, models, context_size)
    render_info(
        console,
        f"  {msg('cli.status_memory')}: WM={wm}, STM={stm}, LTM={ltm}",
    )

    # デバッグ情報の詳細表示（debug モード有効時のみ）
    if debug_on:
        log_dir = debug_section.get("log_dir", "")
        disk_mb = debug_section.get("disk_usage_mb", 0.0)
        errors = debug_section.get("recent_errors_count", 0)
        hit_rate = debug_section.get("cache_hit_rate", 0.0)
        render_info(
            console,
            f"  {msg('cli.status_debug_detail')}:"
            f" log_dir={log_dir},"
            f" {msg('cli.status_disk_usage')}={disk_mb:.1f}MB,"
            f" {msg('cli.status_recent_errors')}={errors},"
            f" {msg('cli.status_cache_hit_rate')}={hit_rate:.1%}",
        )

    return CommandResult()


# ──────────────────────────────────────────────────────────────────────────
# Pin コマンド
# ──────────────────────────────────────────────────────────────────────────


def _format_pinned_fact(info: dict) -> str:
    fid = info.get("id", "?")
    subj = info.get("subject", "")
    obj = info.get("object", "")
    locked = info.get("pin_locked_until")
    suffix = ""
    if locked:
        from datetime import datetime as _dt, timezone as _tz
        suffix = " " + msg(
            "cli.pin_locked_until",
            until=_dt.fromtimestamp(locked, tz=_tz.utc).isoformat(timespec="seconds"),
        )
    return f"  [{fid}] {subj}: {obj}{suffix}"


async def _cmd_pin(args: str, state: SessionState, console) -> CommandResult:
    """/pin <text> — テキストを SemMem に永続 pin する"""
    text = args.strip()
    if not text:
        render_error(console, msg("cli.pin_usage"))
        return CommandResult()
    payload = {
        "scope": "global",
        "content": text,
        "subject": "user.pinned",
        "predicate": "remember",
        "type": "personal_fact",
        "mode_origin": "chat",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{state.backend_url}/api/memory/pin", json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return CommandResult()
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code} {e.response.text}")
        return CommandResult()

    fact = data.get("fact", {})
    render_info(console, msg("cli.pin_added", id=fact.get("id", "?")))
    render_info(console, _format_pinned_fact(fact))
    return CommandResult()


async def _cmd_unpin(args: str, state: SessionState, console) -> CommandResult:
    """/unpin <id> — pin を解除する"""
    parts = args.strip().split()
    if not parts:
        render_error(console, msg("cli.unpin_usage"))
        return CommandResult()
    fact_id = parts[0]
    force = "--force" in parts[1:]
    payload = {"fact_id": fact_id, "scope": "global", "force": force}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{state.backend_url}/api/memory/unpin", json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return CommandResult()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            render_error(console, msg("cli.pin_not_found", id=fact_id))
        elif e.response.status_code == 409:
            render_error(console, msg("cli.pin_locked", id=fact_id))
        else:
            render_error(console, f"API error: {e.response.status_code}")
        return CommandResult()

    fact = data.get("fact", {})
    render_info(console, msg("cli.unpin_done", id=fact.get("id", "?")))
    return CommandResult()


async def _cmd_private(args: str, state: SessionState, console) -> CommandResult:
    """/private on|off — プライベートセッションの ON/OFF を切り替える

    プライベート中は WM/STM までで会話が完結し、LTM/SemMem への昇格と
    ディスク履歴永続化がスキップされる。セッション切替や CLI 終了で揮発する。
    """
    arg = args.strip().lower()
    if arg in ("on", "true", "1", "enable"):
        state.private_mode = True
        render_info(console, msg("cli.private_enabled"))
        return CommandResult()
    if arg in ("off", "false", "0", "disable"):
        state.private_mode = False
        render_info(console, msg("cli.private_disabled"))
        return CommandResult()
    if arg in ("", "status"):
        key = "cli.private_status_on" if state.private_mode else "cli.private_status_off"
        render_info(console, msg(key))
        return CommandResult()
    render_error(console, msg("cli.private_usage"))
    return CommandResult()


async def _cmd_pinned(args: str, state: SessionState, console) -> CommandResult:
    """/pinned — pin 済みファクト一覧を表示する"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{state.backend_url}/api/memory/pinned",
                params={"scope": "global"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return CommandResult()
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code}")
        return CommandResult()

    facts = data.get("facts", [])
    total = data.get("total", 0)
    render_info(console, msg("cli.pinned_total", count=total))
    if not facts:
        render_info(console, msg("cli.pinned_empty"))
        return CommandResult()
    for fact in facts:
        render_info(console, _format_pinned_fact(fact))
    return CommandResult()


def _cmd_exit(args: str, state: SessionState, console) -> CommandResult:
    # 終了前にサマリログ出力＋自動保存
    finalize_session(state)
    state.should_exit = True
    render_info(console, msg("cli.exit_message"))
    return CommandResult()


async def _cmd_cartridge(args: str, state: SessionState, console) -> CommandResult:
    """/cartridge コマンド: カートリッジ管理

    Usage:
        /cartridge                    — インストール済み一覧
        /cartridge list               — インストール済み一覧
        /cartridge install <file>     — ZIP からインストール
        /cartridge show <id>          — 詳細表示
        /cartridge load <id>          — メモリにロード
        /cartridge unload <id>        — メモリからアンロード
        /cartridge uninstall <id>     — 完全削除
        /cartridge rebuild <id>       — ベクトルインデックス再構築
        /cartridge create <dir> ...   — カートリッジ作成 (Pro)
    """
    from backend.free.cli.cartridge_commands import (
        _cartridge_install,
        _cartridge_list,
        _cartridge_load,
        _cartridge_rebuild,
        _cartridge_show,
        _cartridge_uninstall,
        _cartridge_unload,
    )

    parts = args.strip().split()
    action = parts[0] if parts else ""
    rest = parts[1:] if len(parts) > 1 else []
    backend_url = state.backend_url

    if action == "" or action == "list":
        _cartridge_list(backend_url, console)
        return CommandResult()

    if action == "install":
        if not rest:
            render_error(console, "Usage: /cartridge install <file>")
            return CommandResult()
        _cartridge_install(backend_url, console, rest[0])
        return CommandResult()

    if action == "show":
        if not rest:
            render_error(console, "Usage: /cartridge show <id>")
            return CommandResult()
        _cartridge_show(backend_url, console, rest[0])
        return CommandResult()

    if action == "load":
        if not rest:
            render_error(console, "Usage: /cartridge load <id>")
            return CommandResult()
        _cartridge_load(backend_url, console, rest[0])
        return CommandResult()

    if action == "unload":
        if not rest:
            render_error(console, "Usage: /cartridge unload <id>")
            return CommandResult()
        _cartridge_unload(backend_url, console, rest[0])
        return CommandResult()

    if action == "uninstall":
        if not rest:
            render_error(console, "Usage: /cartridge uninstall <id>")
            return CommandResult()
        _cartridge_uninstall(backend_url, console, rest[0])
        return CommandResult()

    if action == "rebuild":
        if not rest:
            render_error(console, "Usage: /cartridge rebuild <id>")
            return CommandResult()
        _cartridge_rebuild(backend_url, console, rest[0])
        return CommandResult()

    if action == "create":
        return _cartridge_create_interactive(rest, state, console)

    render_error(console, msg("cli.cartridge_unknown_action", action=action))
    return CommandResult()


def _cartridge_create_interactive(
    rest: list[str], state: SessionState, console,
) -> CommandResult:
    """/cartridge create のインタラクティブ引数パース"""
    import argparse as _argparse

    from backend.free.cli.cartridge_commands import _cartridge_create

    parser = _argparse.ArgumentParser(
        prog="/cartridge create", exit_on_error=False,
    )
    parser.add_argument("source_dir")
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--author", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-count", type=int, default=10)
    parser.add_argument("--output", "-o", default=None)

    try:
        create_args = parser.parse_args(rest)
    except (SystemExit, _argparse.ArgumentError):
        render_error(
            console,
            "Usage: /cartridge create <source_dir> --id <id> --name <name> [--version V] [--author A] [--tags T]",
        )
        return CommandResult()

    _cartridge_create(
        state.backend_url, console,
        source_dir=create_args.source_dir,
        cart_id=create_args.id,
        name=create_args.name,
        version=create_args.version,
        author=create_args.author,
        description=create_args.description,
        tags=create_args.tags,
        language=create_args.language,
        generate_eval=not create_args.no_eval,
        eval_count=create_args.eval_count,
        output=create_args.output,
    )
    return CommandResult()


async def _cmd_migrate_model(args: str, state: SessionState, console) -> CommandResult:
    """/migrate-model コマンド: ベースモデル移行

    Usage:
        /migrate-model --new-model <path>           — 新モデルに移行
        /migrate-model --new-model <path> --dry-run — プレビューのみ
        /migrate-model --new-model <path> --try-lora — LoRA 互換テスト付き
        /migrate-model --rollback [--to <model>]    — ロールバック
    """
    import argparse as _argparse

    from backend.free.cli.model_commands import _handle_migrate, _handle_rollback

    parser = _argparse.ArgumentParser(
        prog="/migrate-model", exit_on_error=False,
    )
    parser.add_argument("--new-model")
    parser.add_argument("--try-lora", action="store_true")
    parser.add_argument("--regenerate-context", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--to")

    stripped = args.strip()
    if not stripped:
        render_info(
            console,
            "Usage: /migrate-model --new-model <path> [--try-lora] [--dry-run] [--regenerate-context]\n"
            "       /migrate-model --rollback [--to <model>]",
        )
        return CommandResult()

    try:
        parsed = parser.parse_args(stripped.split())
    except (SystemExit, _argparse.ArgumentError):
        render_error(
            console,
            "Usage: /migrate-model --new-model <path> [--try-lora] [--dry-run]",
        )
        return CommandResult()

    backend_url = state.backend_url

    if parsed.rollback:
        _handle_rollback(backend_url, parsed.to)
        return CommandResult()

    if not parsed.new_model:
        render_error(console, msg("cli.migrate_model_no_path"))
        return CommandResult()

    _handle_migrate(
        backend_url,
        new_model_path=parsed.new_model,
        try_lora=parsed.try_lora,
        regenerate_context=parsed.regenerate_context,
        dry_run=parsed.dry_run,
    )
    return CommandResult()


async def _cmd_web(args: str, state: SessionState, console) -> CommandResult:
    """/web <url> — URL のコンテンツを取得してコンテキストに追加

    Usage:
        /web <url>              — URL を取得して表示・コンテキストに追加
        /web <url> --no-add     — 表示のみ（コンテキストに追加しない）
    """
    parts = args.strip().split()
    if not parts:
        render_error(console, msg("cli.web_no_url"))
        return CommandResult()

    url = parts[0]
    no_add = "--no-add" in parts

    # URL バリデーション
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    render_info(console, msg("cli.web_fetching", url=url))

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.web_connect_error", url=url))
        return CommandResult()
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.web_http_error", status=e.response.status_code))
        return CommandResult()
    except Exception as e:
        render_error(console, f"Error: {e}")
        return CommandResult()

    # BeautifulSoup が使えれば使う、なければ正規表現フォールバック
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    except ImportError:
        from backend.free.agent.tools.builtin import _strip_html_fallback
        text = _strip_html_fallback(r.text)

    if not text:
        render_error(console, msg("cli.web_empty"))
        return CommandResult()

    # 20000文字で切り詰め
    truncated = False
    if len(text) > 20000:
        text = text[:20000]
        truncated = True

    # コンソール表示（先頭 2000 文字）
    preview = text[:2000]
    console.print()
    console.print(preview)
    if len(text) > 2000:
        render_info(console, msg("cli.web_preview_truncated", total=len(text)))
    if truncated:
        render_info(console, msg("cli.web_content_truncated"))

    # コンテキストに追加
    if not no_add:
        state.add_turn("user", f"[Web: {url}]\n{text}")
        render_info(console, msg("cli.web_added", url=url, chars=len(text)))
    else:
        render_info(console, msg("cli.web_display_only"))

    logger.debug("/web: fetched %s (%d chars, truncated=%s, added=%s)", url, len(text), truncated, not no_add)
    return CommandResult()


async def _cmd_learn(args: str, state: SessionState, console) -> CommandResult:
    """学習サイクルの手動トリガーまたは状態表示"""
    logger.debug("/learn: subcommand=%r", args.strip())
    sub = args.strip().lower()

    if sub == "status" or not sub:
        return await _learn_status(state, console)
    elif sub == "level1":
        return await _learn_trigger(state, console, "level1")
    elif sub == "full":
        return await _learn_trigger(state, console, "full")
    else:
        render_error(
            console,
            f"Unknown /learn subcommand: {sub}. Use: /learn status, /learn level1, /learn full",
        )
        return CommandResult()


def _handle_httpx_error(e: Exception, console) -> None:
    """httpx 通信エラーの共通ハンドリング"""
    if isinstance(e, httpx.HTTPStatusError):
        render_error(console, f"API error: {e.response.status_code}")
    elif isinstance(e, httpx.ConnectError):
        render_error(console, "Backend connection failed")
    else:
        render_error(console, f"Request failed: {e}")


async def _learn_status(state: SessionState, console) -> CommandResult:
    """学習状態を表示"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{state.backend_url}/api/learning/status")
            resp.raise_for_status()
            data = resp.json()
            sched = data.get("scheduler_status", {})

            def _fmt_time(iso_str: str | None) -> str:
                """UTC ISO文字列をローカル時刻 (YYYY-MM-DD HH:MM:SS) に変換"""
                if not iso_str:
                    return "never"
                dt = parse_iso(iso_str)
                if dt is None:
                    return iso_str
                return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")

            lines = [
                "=== Learning Status ===",
                f"  Experience count:    {sched.get('experience_count', '?')}",
                f"  Min experiences:     {sched.get('min_experiences', '?')}",
                f"  Conditions met:      {sched.get('conditions_met', '?')}",
                f"  Running:             {sched.get('running', '?')}",
                f"  Last Level 1 run:    {_fmt_time(sched.get('last_level1_run'))}",
                f"  Last Level 2 run:    {_fmt_time(sched.get('last_level2_run'))}",
                f"  LoRA version:        {data.get('lora_version', '?')}",
                f"  LoRA adapter exists: {data.get('lora_adapter_exists', '?')}",
                f"  Eval cases:          {data.get('eval_cases_count', '?')}",
            ]
            render_info(console, "\n".join(lines))
    except (httpx.HTTPStatusError, httpx.ConnectError) as e:
        _handle_httpx_error(e, console)
    return CommandResult()


async def _learn_trigger(state: SessionState, console, level: str) -> CommandResult:
    """学習サイクルを手動トリガー"""
    render_info(console, f"Triggering {level}...")
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{state.backend_url}/api/learning/trigger",
                json={"level": level},
            )
            if resp.status_code == 409:
                render_error(console, "Learning cycle already running")
                return CommandResult()
            if resp.status_code == 503:
                render_error(console, f"Service unavailable: {resp.json().get('detail', '')}")
                return CommandResult()
            resp.raise_for_status()
            data = resp.json()

            if data.get("triggered"):
                render_info(console, f"Triggered: {data.get('message', '')}")
                render_info(console, f"Experience count: {data.get('experience_count', '?')}")
            else:
                render_error(console, f"Not triggered: {data.get('message', '')}")
                render_info(console, f"Experience count: {data.get('experience_count', '?')}")
    except (httpx.HTTPStatusError, httpx.ConnectError) as e:
        _handle_httpx_error(e, console)
    return CommandResult()


async def _cmd_theme(args: str, state: SessionState, console) -> CommandResult:
    """/theme コマンド: テーマ管理"""
    from backend.free.cli.theme_commands import handle_theme_command

    await handle_theme_command(args, state.backend_url, console, state)
    return CommandResult()


async def _cmd_reindex(args: str, state: SessionState, console) -> CommandResult:
    """/reindex コマンド: ベクトルインデックスを再構築

    引数:
        --dry-run            : 対象件数のみ表示
        --cartridge <id>     : 特定カートリッジのみ再構築
    """
    tokens = args.split() if args else []
    dry_run = "--dry-run" in tokens
    cartridge: str | None = None
    if "--cartridge" in tokens:
        idx = tokens.index("--cartridge")
        if idx + 1 < len(tokens):
            cartridge = tokens[idx + 1]

    params: dict[str, str] = {"dry_run": "true"} if dry_run else {}
    if cartridge:
        params["cartridge"] = cartridge

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            # まずドライランで対象件数を取得
            preview_params = dict(params)
            preview_params["dry_run"] = "true"
            resp = await client.post(
                f"{state.backend_url}/api/rag/reindex", params=preview_params,
            )
            if resp.status_code != 200:
                render_error(console, msg(
                    "cli.reindex_failed", detail=f"HTTP {resp.status_code}",
                ))
                return CommandResult()
            plan = resp.json()
            render_info(console, msg(
                "cli.reindex_targets",
                rag_chunks=int(plan.get("rag_chunks", 0)),
                cart_chunks=int(plan.get("cartridge_chunks", 0)),
                cart_count=len(plan.get("cartridges", [])),
                mem_notes=int(plan.get("memory_notes", 0)),
            ))

            if dry_run:
                render_info(console, msg("cli.reindex_dry_run"))
                return CommandResult()

            render_info(console, msg("cli.reindex_running"))
            resp = await client.post(
                f"{state.backend_url}/api/rag/reindex", params=params,
            )
            if resp.status_code != 200:
                render_error(console, msg(
                    "cli.reindex_failed", detail=f"HTTP {resp.status_code}",
                ))
                return CommandResult()
            result = resp.json()
            render_info(console, msg(
                "cli.reindex_complete",
                rag_chunks=int(result.get("rag_chunks", 0)),
                cart_chunks=int(result.get("cartridge_chunks", 0)),
                cart_count=len(result.get("cartridges_rebuilt", [])),
                mem_notes=int(result.get("memory_notes_reset", 0)),
                elapsed=f"{float(result.get('elapsed_sec', 0.0)):.2f}",
            ))
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
    except httpx.HTTPError as exc:
        render_error(console, msg("cli.reindex_failed", detail=str(exc)))
    return CommandResult()


# ──────────────────────────────────────────────────────────────────────────
# Loop / Tasks コマンド
# ──────────────────────────────────────────────────────────────────────────


def _format_task_line(task: dict) -> str:
    status = task.get("status", "?")
    tid = task.get("task_id", "?")
    title = task.get("title", "")
    salience = float(task.get("salience", 0.0))
    deps = task.get("depends_on") or []
    dep_str = f" deps={','.join(deps)}" if deps else ""
    return f"  [{status}] {tid} (s={salience:.2f}){dep_str}: {title}"


async def _cmd_loop(args: str, state: SessionState, console) -> CommandResult:
    """/loop start <project_id> | /loop stop | /loop status | /loop report [<project_id>]

    自律ループの起動・停止・状態確認・最終レポート表示。
    """
    parts = args.strip().split()
    if not parts:
        render_error(console, msg("cli.loop_usage"))
        return CommandResult()
    sub = parts[0].lower()
    rest = parts[1:]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if sub == "start":
                if not rest:
                    render_error(console, msg("cli.loop_start_usage"))
                    return CommandResult()
                project_id = rest[0]
                resp = await client.post(
                    f"{state.backend_url}/api/loop/start",
                    json={"project_id": project_id},
                )
            elif sub == "stop":
                resp = await client.post(f"{state.backend_url}/api/loop/stop")
            elif sub == "status":
                resp = await client.get(f"{state.backend_url}/api/loop/status")
            elif sub == "report":
                params: dict[str, str] = {}
                if rest:
                    params["project_id"] = rest[0]
                resp = await client.get(
                    f"{state.backend_url}/api/loop/report",
                    params=params,
                )
                if resp.status_code == 400:
                    render_error(console, msg("cli.loop_report_no_project"))
                    return CommandResult()
            else:
                render_error(console, msg("cli.loop_unknown_sub", sub=sub))
                return CommandResult()
            if resp.status_code == 409:
                render_error(console, msg("cli.loop_already_running"))
                return CommandResult()
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return CommandResult()
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code} {e.response.text}")
        return CommandResult()

    if sub == "report":
        _render_loop_report(console, data)
        return CommandResult()

    info = data.get("state", {})
    if sub == "start":
        render_info(
            console,
            msg("cli.loop_started", project_id=info.get("project_id", "?")),
        )
    elif sub == "stop":
        render_info(console, msg("cli.loop_stopped"))
    render_info(
        console,
        msg(
            "cli.loop_status",
            running="yes" if info.get("running") else "no",
            project_id=info.get("project_id") or "-",
            iteration=info.get("iteration", 0),
        ),
    )
    return CommandResult()


def _render_loop_report(console, data: dict) -> None:
    """LoopReport JSON を CLI に整形表示する"""
    render_info(
        console,
        msg(
            "cli.loop_report_header",
            project_id=data.get("project_id", "?"),
        ),
    )
    render_info(
        console,
        msg(
            "cli.loop_report_tasks",
            total=data.get("total_tasks", 0),
            done=data.get("done_tasks", 0),
            failed=data.get("failed_tasks", 0),
            in_progress=data.get("in_progress_tasks", 0),
            open=data.get("open_tasks", 0),
        ),
    )
    elapsed = data.get("elapsed_seconds")
    elapsed_text = (
        f"{float(elapsed):.1f}s" if isinstance(elapsed, (int, float)) else "-"
    )
    render_info(
        console,
        msg(
            "cli.loop_report_iterations",
            iterations=data.get("iterations", 0),
            elapsed=elapsed_text,
        ),
    )
    render_info(
        console,
        msg(
            "cli.loop_report_failures",
            total=data.get("failure_pattern_total", 0),
            markers=data.get("progress_marker_count", 0),
        ),
    )
    by_type = data.get("failure_pattern_by_error_type") or {}
    if by_type:
        for error_type, count in sorted(
            by_type.items(), key=lambda kv: (-kv[1], kv[0]),
        ):
            render_info(
                console,
                msg(
                    "cli.loop_report_failure_entry",
                    error_type=error_type,
                    count=count,
                ),
            )


async def _cmd_tasks(args: str, state: SessionState, console) -> CommandResult:
    """/tasks [<project_id>] [--status open,in_progress,...]

    プロジェクトの task ファクト一覧を表示する。
    project_id 省略時は LoopDriver で起動中のプロジェクトを使う。
    """
    parts = args.strip().split()
    project_id: str | None = None
    status_filter: str | None = None
    i = 0
    while i < len(parts):
        token = parts[i]
        if token == "--status" and i + 1 < len(parts):
            status_filter = parts[i + 1]
            i += 2
            continue
        if project_id is None:
            project_id = token
        i += 1

    params: dict[str, str] = {}
    if project_id:
        params["project_id"] = project_id
    if status_filter:
        params["status"] = status_filter
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{state.backend_url}/api/loop/tasks", params=params,
            )
            if resp.status_code == 400:
                render_error(console, msg("cli.tasks_no_project"))
                return CommandResult()
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return CommandResult()
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code}")
        return CommandResult()

    tasks = data.get("tasks", [])
    total = data.get("total", 0)
    next_id = data.get("next_task_id")
    render_info(
        console,
        msg(
            "cli.tasks_total",
            project_id=data.get("project_id", "?"),
            count=total,
        ),
    )
    if not tasks:
        render_info(console, msg("cli.tasks_empty"))
        return CommandResult()
    for task in tasks:
        render_info(console, _format_task_line(task))
    if next_id:
        render_info(console, msg("cli.tasks_next", task_id=next_id))
    return CommandResult()
