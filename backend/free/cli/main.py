"""evoref CLI メインエントリーポイント"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import webbrowser
from urllib.parse import urlparse
from pathlib import Path

import httpx

from backend.free.cli.chat_loop import chat_stream, _main_loop
from backend.free.cli.command_parser import SessionState
from backend.free.cli.develop_mode_setup import setup_develop_mode
from backend.free.cli.edition_validator import validate_edition_arg
from backend.free.cli.session_persistence import recover_checkpoints
from backend.free.cli.config_loader import (
    _find_project_root,
    _load_cli_config,
    _load_history_config,
    _setup_encoding,
)
from backend.free.cli.renderer import (
    ModelInfoItem,
    create_console,
    render_error,
    render_info,
    render_model_info,
    render_startup_result,
    render_welcome,
    render_welcome_hint,
)
from backend.free.cli.service_manager import (
    AutoServeState,
    _auto_serve_cleanup,
    _auto_serve_start,
    _build_serve_parser,
    _check_backend,
    _register_session,
    _run_serve,
    _sync_server_mode,
    ensure_extra_servers,
    _unregister_session,
)
from backend.free.cli.startup_checks import (
    CheckStatus,
    run_interactive_checks,
)
from backend.error_handlers import E6001
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger, setup_cli_logging

logger = get_logger("cli.main")

# サブコマンドとして認識する名前
# - "create": 対話モード (Pro はクリエイト、Free は警告 + chat フォールバック)
_SUBCOMMANDS = {"serve", "chat", "create", "gui", "export", "import", "reindex"}

#: 旧サブコマンド名 → 現行名。``code`` はクリエイトモードへの改名に追随して
#: ``create`` になった。既存のスクリプトや手癖を壊さないよう受理し続け、
#: 読み替えたことを 1 行だけ警告する。
_LEGACY_SUBCOMMANDS = {"code": "create"}


def _canonicalize_subcommands(argv: list[str]) -> list[str]:
    """旧サブコマンド名 (``code``) を現行名 (``create``) へ読み替える。"""
    out: list[str] = []
    for arg in argv:
        renamed = _LEGACY_SUBCOMMANDS.get(arg)
        if renamed is None:
            out.append(arg)
            continue
        logger.warning(
            "Subcommand %r was renamed to %r; accepting the old name for now",
            arg, renamed,
        )
        out.append(renamed)
    return out


# ────────────────────────────────────────────
# gui サブコマンド
# ────────────────────────────────────────────


def _run_gui(argv: list[str]) -> int:
    """gui コマンド: Web UI をデフォルトブラウザで開く（同期ラッパー）"""
    try:
        return asyncio.run(_async_run_gui(argv))
    except KeyboardInterrupt:
        return 0


def _build_gui_arg_parser() -> argparse.ArgumentParser:
    """`gui` サブコマンド用 argparse パーサーを構築する。"""
    parser = argparse.ArgumentParser(
        prog="evoref gui",
        description="Open evoref Web UI in the default browser",
    )
    parser.add_argument(
        "--backend-url", default=None,
        help="Backend URL (e.g. http://localhost:8000). Overrides --host/--port",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Frontend host (default: localhost)",
    )
    parser.add_argument(
        "--port", default=8000, type=int,
        help="Backend port (default: 8000)",
    )
    parser.add_argument(
        "--frontend-port", default=None, type=int,
        help="Frontend port (default: from config.yaml or 5173)",
    )
    parser.add_argument(
        "--auto-serve", action="store_true",
        help="Auto-start backend and frontend if not running",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Kill stale processes and port occupants before starting",
    )
    parser.add_argument(
        "--edition", choices=["free", "pro", "develop"], default=None,
        help="Override edition (develop mode only)",
    )
    parser.add_argument(
        "--no-learning", action="store_true",
        help="Disable self-learning when auto-serving the backend "
             "(propagates EVOREF_LEARNING_DISABLED=1).",
    )
    _add_develop_flag(parser)
    return parser


def _resolve_gui_ports(
    args: argparse.Namespace, project_root: Path,
) -> tuple[int, int]:
    """`config.yaml` から frontend / backend ポートを解決する。

    `args.frontend_port` / `args.port` が優先され、未指定または読み込み失敗時は
    config 値またはハードコードのデフォルト (5173 / 8000) へフォールバック。
    """
    from backend.free.cli.service_manager import _load_config
    frontend_port = args.frontend_port
    backend_port = args.port
    try:
        cfg = _load_config(project_root)
        if frontend_port is None:
            frontend_port = cfg.get("server", {}).get("frontend_port", 5173)
        backend_port = cfg.get("server", {}).get("port", args.port)
    except (OSError, ValueError):
        if frontend_port is None:
            frontend_port = 5173
    return frontend_port, backend_port


async def _start_gui_auto_serve(
    args: argparse.Namespace,
    project_root: Path,
    frontend_port: int,
    console,
) -> tuple[AutoServeState, subprocess.Popen | None, object] | None:
    """`--auto-serve` 経路: バックエンド + フロントエンドを自動起動する。

    バックエンドが既に動いていればスキップ、未起動なら `_auto_serve_start` を呼び、
    その後フロントエンドを起動する。バックエンド起動失敗時は ``None``。
    """
    auto_serve_state = AutoServeState()
    backend_alive = await _check_backend(f"http://{args.host}:{args.port}")
    if not backend_alive:
        auto_serve_state = await _auto_serve_start(
            project_root,
            args.host,
            args.port,
            develop=getattr(args, "develop", None),
            edition=getattr(args, "edition", None),
            no_learning=getattr(args, "no_learning", False),
            force=getattr(args, "force", False),
            console=console,
        )
        if not auto_serve_state.active:
            return None

    frontend_proc, frontend_stderr = await _auto_start_frontend(
        project_root, args.host, frontend_port, console,
        edition=getattr(args, "edition", None),
    )
    return auto_serve_state, frontend_proc, frontend_stderr


def _open_gui_in_browser(url: str, console) -> bool:
    """既定ブラウザで `url` を開く。`EVOREF_NO_BROWSER=1` 時はスキップ。

    起動成功または skip 時は ``True``、`webbrowser.open` 失敗時のみ ``False``。
    """
    if os.environ.get("EVOREF_NO_BROWSER") == "1":
        logger.debug("gui: EVOREF_NO_BROWSER=1, skipping webbrowser.open")
        render_info(console, msg("cli.gui_opened", url=url))
        return True
    if not webbrowser.open(url):
        render_error(console, msg("cli.gui_open_failed", url=url))
        return False
    render_info(console, msg("cli.gui_opened", url=url))
    return True


async def _wait_for_gui_processes(
    auto_serve_state: AutoServeState,
    frontend_proc: subprocess.Popen | None,
    frontend_stderr,
    console,
) -> None:
    """auto-serve / frontend 起動時の待機ループ。Ctrl+C で抜けてクリーンアップ。"""
    if not (auto_serve_state.active or frontend_proc is not None):
        return
    render_info(console, msg("cli.gui_waiting"))
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if frontend_proc is not None:
            _kill_frontend(frontend_proc, frontend_stderr)
        _auto_serve_cleanup(auto_serve_state, console)


async def _async_run_gui(argv: list[str]) -> int:
    """gui コマンド: バックエンド/フロントエンドを自動起動し Web UI を開く"""
    init_i18n()
    parser = _build_gui_arg_parser()
    args = parser.parse_args(argv)
    _resolve_backend_url(args)

    console = create_console()

    if (rc := _setup_develop_mode_async(args, console)) is not None:
        return rc

    if (rc := _validate_edition_arg_async(args, console)) is not None:
        return rc

    project_root = _find_project_root()
    frontend_port, backend_port = _resolve_gui_ports(args, project_root)
    args.port = backend_port  # develop モード時は config 基準のポートで接続
    url = f"http://{args.host}:{frontend_port}"

    develop = getattr(args, "develop", None)  # str | None
    setup_cli_logging(project_root=project_root, debug=develop is not None)
    logger.debug(
        "gui: host=%s, port=%d, frontend_port=%d, auto_serve=%s, develop=%s",
        args.host, args.port, frontend_port, args.auto_serve, develop,
    )

    auto_serve_state = AutoServeState()
    frontend_proc: subprocess.Popen | None = None
    frontend_stderr = None
    if args.auto_serve:
        result = await _start_gui_auto_serve(
            args, project_root, frontend_port, console,
        )
        if result is None:
            return 1
        auto_serve_state, frontend_proc, frontend_stderr = result

    render_info(console, msg("cli.gui_opening", url=url))

    if not _open_gui_in_browser(url, console):
        _auto_serve_cleanup(auto_serve_state, console)
        if frontend_proc is not None:
            _kill_frontend(frontend_proc, frontend_stderr)
        return 1

    await _wait_for_gui_processes(
        auto_serve_state, frontend_proc, frontend_stderr, console,
    )
    return 0


async def _auto_start_frontend(
    project_root: Path,
    host: str,
    port: int,
    console,
    edition: str | None = None,
) -> tuple[subprocess.Popen | None, object]:
    """フロントエンド開発サーバーが未起動なら自動起動する

    Args:
        edition: `--edition` で上書きされたエディション (``"free"`` / ``"pro"``)。
            指定時は `VITE_EVOREF_EDITION` を子プロセスへ伝播し、フロント UI も
            同じエディションで動作させる

    Returns:
        (proc, stderr_file) — 既に起動中または失敗時は (None, None)
    """
    # 既に起動中かチェック
    frontend_url = f"http://{host}:{port}"
    logger.debug("Checking frontend at %s", frontend_url)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(frontend_url, timeout=3.0)
            if resp.status_code < 500:
                logger.debug("Frontend already running at %s (status=%d)", frontend_url, resp.status_code)
                render_info(console, msg("cli.gui_frontend_exists"))
                return None, None
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
        logger.debug("Frontend not running: %s", e)

    # npm run dev を起動
    render_info(console, msg("cli.gui_frontend_starting"))
    frontend_dir = project_root / "frontend"
    if not (frontend_dir / "package.json").exists():
        logger.error("frontend/package.json not found at %s", frontend_dir)
        render_error(console, msg("cli.gui_frontend_not_found"))
        return None, None

    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    cmd = [npm_cmd, "run", "dev", "--", "--port", str(port)]
    frontend_env: dict[str, str] | None = None
    if edition is not None:
        frontend_env = {**os.environ, "VITE_EVOREF_EDITION": edition}
        logger.debug("Frontend edition override: VITE_EVOREF_EDITION=%s", edition)
    logger.debug("Starting frontend: %s (cwd=%s)", cmd, frontend_dir)
    try:
        log_dir = project_root / "local" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stderr_f = open(log_dir / "frontend.stderr.log", "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(frontend_dir),
            stdout=subprocess.DEVNULL,
            stderr=stderr_f,
            env=frontend_env,
        )
    except FileNotFoundError as e:
        logger.error("Frontend start failed: %s", e)
        render_error(console, msg("cli.gui_frontend_failed"))
        return None, None

    # フロントエンドが応答するまで待機
    logger.debug("Waiting for frontend to become ready (pid=%d)", proc.pid)
    for i in range(30):
        await asyncio.sleep(1)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(frontend_url, timeout=3.0)
                if resp.status_code < 500:
                    logger.debug("Frontend ready after %ds (pid=%d)", i + 1, proc.pid)
                    render_info(console, msg("cli.gui_frontend_ready"))
                    return proc, stderr_f
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            pass
        if proc.poll() is not None:
            logger.error("Frontend process exited unexpectedly (exit_code=%s)", proc.returncode)
            render_error(console, msg("cli.gui_frontend_failed"))
            stderr_f.close()
            return None, None

    logger.error("Frontend did not start within 30s (pid=%d)", proc.pid)
    render_error(console, msg("cli.gui_frontend_timeout"))
    proc.terminate()
    stderr_f.close()
    return None, None


def _kill_frontend(proc: subprocess.Popen, stderr_f=None) -> None:
    """フロントエンドプロセスを終了"""
    from backend.free.cli.service_manager import _kill_process_tree
    logger.debug("Stopping frontend process (pid=%d)", proc.pid)
    _kill_process_tree(proc.pid)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    if stderr_f is not None:
        try:
            stderr_f.close()
        except OSError:
            pass


# ────────────────────────────────────────────
# サブコマンドルーティング
# ────────────────────────────────────────────


def _run_serve_subcmd(argv: list[str]) -> int:
    """serve サブコマンドのラッパー（パーサー構築 + 実行）"""
    parser = _build_serve_parser()
    args = parser.parse_args(argv)
    return _run_serve(args)


def _get_subcommand_handler(name: str):
    """サブコマンド名に対応するハンドラを返す（遅延 import）

    Returns:
        ハンドラ関数、または未知のサブコマンドなら None
    """
    if name == "serve":
        return _run_serve_subcmd
    if name == "gui":
        return _run_gui
    if name == "export":
        from backend.free.cli.data_commands import run_export
        return run_export
    if name == "import":
        from backend.free.cli.data_commands import run_import
        return run_import
    if name == "reindex":
        from backend.free.cli.reindex_command import run_reindex
        return run_reindex
    return None


# ────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────


def _resolve_backend_url(args: argparse.Namespace) -> None:
    """--backend-url が指定された場合、host/port を上書きする"""
    if args.backend_url:
        parsed = urlparse(args.backend_url)
        args.host = parsed.hostname or "localhost"
        args.port = parsed.port or 8000


def _add_develop_flag(parser: argparse.ArgumentParser) -> None:
    """--develop / --isolate-data フラグをパーサーに追加する

    Pro CLI への直接 import を排除し、`develop_hook` 経由に統一
    Pro が登録されていれば help 付きで追加、未登録なら hidden で追加 (UX 維持)。
    """
    from backend.free.cli.develop_hook import get_develop_hook
    get_develop_hook().extend_parser(parser)


def _build_interactive_parser() -> argparse.ArgumentParser:
    """対話/単発モード用パーサー（create サブコマンド、デフォルト）"""
    parser = argparse.ArgumentParser(
        prog="evoref",
        description="evoref - self-evolving local LLM assistant",
    )
    parser.add_argument(
        "message", nargs="?", default=None,
        help="Single message to send (non-interactive mode)",
    )
    parser.add_argument(
        "--file", "-f", action="append", default=[],
        help="File to add to context (can be repeated)",
    )
    parser.add_argument(
        "--backend-url", default=None,
        help="Backend URL (e.g. http://localhost:8000). Overrides --host/--port",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Backend host (default: localhost)",
    )
    parser.add_argument(
        "--port", default=8000, type=int,
        help="Backend port (default: 8000)",
    )
    parser.add_argument(
        "--frontend-port", default=None, type=int,
        help="Frontend port (default: from config.yaml or 5173)",
    )
    parser.add_argument(
        "--edition", choices=["free", "pro", "develop"], default=None,
        help="Override edition (develop mode only)",
    )
    parser.add_argument(
        "--auto-serve", action="store_true",
        help="Auto-start backend if not running (§9.5.1)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Kill stale processes and port occupants before starting",
    )
    parser.add_argument(
        "--mode", choices=["chat", "create"], default=None,
        help="Chat mode (default: chat in Free / create in Pro). "
             "'create' is Pro-only and falls back to 'chat' with a warning in Free.",
    )
    parser.add_argument(
        "--no-learning", action="store_true",
        help="Disable self-learning when auto-serving the backend "
             "(propagates EVOREF_LEARNING_DISABLED=1).",
    )
    _add_develop_flag(parser)
    return parser


# ────────────────────────────────────────────
# モデル情報ヘルパー（サービス層に委譲）
# ────────────────────────────────────────────


def _build_model_info(
    project_root: Path,
    status_data: dict | None,
) -> tuple[list[ModelInfoItem], int | None]:
    """config.yaml + /api/status からモデル情報一覧を構築"""
    from backend.free.services.model_service import build_model_info_sync

    result = build_model_info_sync(project_root, status_data)
    # サービス層の ModelInfoItem を renderer の ModelInfoItem に変換
    models = [
        ModelInfoItem(label=m.label, name=m.name, connected=m.connected)
        for m in result[0]
    ]
    return models, result[1]


# ────────────────────────────────────────────
# 対話/単発モード
# ────────────────────────────────────────────


def _build_session_state(
    project_root: Path, args: argparse.Namespace,
) -> SessionState:
    """CLI / history config を読み込んで SessionState を構築する。"""
    cli_cfg = _load_cli_config(project_root)
    from backend.free.cli.cartridge_commands import init_cartridge_timeouts
    init_cartridge_timeouts(cli_cfg)

    history_cfg = _load_history_config(project_root)
    history_dir = project_root / "local" / "history"
    return SessionState(
        backend_url=f"http://{args.host}:{args.port}",
        context_files=list(args.file),
        sessions_dir=project_root / "local" / "sessions",
        history_dir=history_dir,
        auto_save_enabled=history_cfg.get("auto_save", True),
        checkpoint_interval=history_cfg.get("checkpoint_interval", 10),
        mode=getattr(args, "_resolved_mode", None) or _resolve_default_mode_value(),
    )


def _resolve_default_mode_value() -> str:
    """SessionState 構築時のフォールバック: argparse 検証前のテスト等で使用。"""
    from backend.free.cli.cli_mode import default_cli_mode
    return default_cli_mode()


def _try_recover_checkpoints(state: SessionState, console) -> None:
    """起動時チェックポイント復旧（設計書 23.3.3）"""
    if not (state.auto_save_enabled and state.history_dir.exists()):
        return
    recovered = recover_checkpoints(state.history_dir)
    if recovered > 0:
        render_info(console, msg("cli.checkpoint_recovered", count=recovered))


def _setup_develop_mode_async(
    args: argparse.Namespace, console,
) -> int | None:
    """`--develop=<level>` / `--isolate-data` の環境セットアップ (共通実装へ委譲)。

    ``debug`` レベルは Free / Pro 共通、``investigate`` / ``evolve`` は Pro 限定
    (Free 環境では起動時に sys.exit で拒否される)。``--isolate-data`` は
    Pro 拡張のみで効果があり、Free では no-op。
    """
    return setup_develop_mode(args, console)


def _apply_develop_port_offset(
    args: argparse.Namespace, state: SessionState, project_root: Path,
) -> None:
    """develop モード時に config のオフセット適用済みポートを反映する。"""
    if getattr(args, "develop", None) is None:
        return
    try:
        from backend.free.cli.service_manager import _load_config
        cfg = _load_config(project_root)
        backend_port = cfg.get("server", {}).get("port", args.port)
        args.port = backend_port
        state.backend_url = f"http://{args.host}:{backend_port}"
    except (OSError, ValueError, ImportError):
        pass


def _resolve_cli_mode(args: argparse.Namespace) -> None:
    """`--mode` を検証し、エディションに応じて補正したモードを `args._resolved_mode`
    に格納する。Free 環境で `--mode create` 指定時は warning を stderr に出して
    `chat` にフォールバック

    `--mode` 未指定時は :func:`default_cli_mode` がエディション既定を返す。
    create モードで起動する場合は :func:`get_create_hook` 経由で Pro 拡張に通知。
    """
    from backend.free.cli.cli_mode import coerce_cli_mode
    from backend.free.cli.create_hook import get_create_hook

    requested = getattr(args, "mode", None)
    resolved, downgraded = coerce_cli_mode(requested)
    if downgraded:
        print(msg("cli.mode_create_pro_only_warning"), file=sys.stderr)
        logger.warning(
            "CLI mode downgraded to chat (requested=create, Free edition)",
        )
    logger.debug("CLI mode resolved: %s (requested=%s)", resolved, requested)
    args._resolved_mode = resolved
    get_create_hook().on_mode_resolved(resolved)


def _validate_edition_arg_async(
    args: argparse.Namespace, console,
) -> int | None:
    """`--edition` 整合性検証 + 環境変数設定 (共通実装へ委譲)。エラー時 1。"""
    return validate_edition_arg(args, console, set_env_var=True)


def _run_interactive_startup_checks(project_root: Path, console) -> int | None:
    """対話モードの起動前提チェック (config.yaml)。FAIL あれば 1。"""
    interactive_results = run_interactive_checks(project_root)
    for result in interactive_results:
        if result.status == CheckStatus.FAIL:
            render_startup_result(console, result)
            return 1
    return None


async def _ensure_backend_running(
    args: argparse.Namespace,
    state: SessionState,
    project_root: Path,
    console,
) -> tuple[AutoServeState, bool]:
    """バックエンド到達確認 + --auto-serve 起動。生存可なら ``(state, True)``。"""
    auto_serve_state = AutoServeState()
    backend_alive = await _check_backend(state.backend_url)
    if backend_alive:
        return auto_serve_state, True

    if getattr(args, "auto_serve", False):
        auto_serve_state = await _auto_serve_start(
            project_root,
            args.host,
            args.port,
            develop=getattr(args, "develop", None),
            edition=args.edition,
            no_learning=getattr(args, "no_learning", False),
            force=getattr(args, "force", False),
            console=console,
        )
        if auto_serve_state.active:
            return auto_serve_state, True
        return auto_serve_state, False

    logger.error("E6001 Backend server not reachable: %s", state.backend_url)
    render_error(console, msg("cli.backend_not_running"), code=E6001)
    return auto_serve_state, False


async def _fetch_status_data(
    state: SessionState, auto_serve_state: AutoServeState,
) -> dict | None:
    """auto-serve キャッシュ または `/api/status` から status を取得する。"""
    status_data: dict | None = getattr(auto_serve_state, "status_data", None)
    if status_data is not None:
        return status_data
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{state.backend_url}/api/status", timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        pass
    return None


async def _run_non_interactive_chat(
    args: argparse.Namespace,
    state: SessionState,
    console,
    auto_serve_state: AutoServeState,
) -> int:
    """非対話 (single-shot) モードのチャット実行と後処理を担う。"""
    message = args.message
    if not message and not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    if not message:
        print("Error: no message provided", file=sys.stderr)
        await _unregister_session(state.backend_url, state.session_id)
        _auto_serve_cleanup(auto_serve_state, console)
        return 1
    try:
        response, _shell_outs, _ttft = await chat_stream(
            message, state, console, non_interactive=True,
        )
        return 0 if response else 1
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        logger.error("Non-interactive chat failed: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        await _unregister_session(state.backend_url, state.session_id)
        _auto_serve_cleanup(auto_serve_state, console)


def _load_context_files(state: SessionState, console) -> None:
    """`--file` 指定ファイルを読み込み state.file_chunks に格納する。"""
    if not state.context_files:
        return
    from backend.free.cli.file_reader import FileReaderError, read_and_chunk
    loaded_files: list[str] = []
    for file_path in list(state.context_files):
        p = Path(file_path)
        if not p.exists():
            render_error(console, msg("cli.file_not_found", path=file_path))
            state.context_files.remove(file_path)
            continue
        try:
            result = read_and_chunk(p)
            state.file_chunks[file_path] = result.chunks
            loaded_files.append(p.name)
        except FileReaderError as e:
            render_error(console, f"{p.name}: {e}")
            state.context_files.remove(file_path)
    if loaded_files:
        render_info(
            console,
            msg("cli.files_loaded_on_start", files=", ".join(loaded_files)),
        )


def _resolve_layout_mode(project_root: Path) -> str:
    """config から CLI レイアウトを解決する（読込失敗時は sequential）。"""
    from backend.free.cli.chat_loop import resolve_layout_mode
    from backend.free.cli.service_manager import _load_config

    try:
        return resolve_layout_mode(_load_config(project_root), project_root)
    except (OSError, ValueError, ImportError) as e:
        logger.warning("Failed to resolve CLI layout mode, using sequential: %s", e)
        return "sequential"


async def _run_interactive_loop(
    state: SessionState,
    console,
    project_root: Path,
    status_data: dict | None,
    auto_serve_state: AutoServeState,
) -> int:
    """対話モード: ウェルカム表示 → メインループ → クリーンアップ。"""
    edition = (status_data.get("edition") if status_data else None) or "free"
    from backend.version import get_runtime_version
    render_welcome(
        console,
        version=get_runtime_version(),
        instance_name=state.instance_name,
        edition=edition,
    )
    render_model_info(console, *_build_model_info(project_root, status_data))
    render_welcome_hint(console)
    try:
        await _main_loop(state, console, layout_mode=_resolve_layout_mode(project_root))
    finally:
        await _unregister_session(state.backend_url, state.session_id)
        _auto_serve_cleanup(auto_serve_state, console)
    return 0


async def async_main(args: argparse.Namespace) -> int:
    """非同期メインエントリーポイント（対話/単発モード）

    フェーズごとに 11 個のヘルパー関数へ責務を分割している。
    """
    project_root = _find_project_root()
    develop = getattr(args, "develop", None)  # str | None
    setup_cli_logging(project_root=project_root, debug=develop is not None)

    # 非対話モード判定（設計書 09 §9.5.6）
    non_interactive = bool(args.message) or not sys.stdout.isatty()
    logger.debug(
        "async_main: host=%s, port=%d, develop=%s, files=%s, non_interactive=%s",
        args.host, args.port, develop, args.file, non_interactive,
    )
    init_i18n()
    console = create_console(no_color=non_interactive)

    # SessionState 構築前に develop / edition を確定し
    # `EVOREF_EDITION` を反映してから `--mode` を解決する。
    if (rc := _setup_develop_mode_async(args, console)) is not None:
        return rc
    if (rc := _validate_edition_arg_async(args, console)) is not None:
        return rc
    _resolve_cli_mode(args)

    state = _build_session_state(project_root, args)
    _try_recover_checkpoints(state, console)

    _apply_develop_port_offset(args, state, project_root)

    if (rc := _run_interactive_startup_checks(project_root, console)) is not None:
        return rc

    auto_serve_state, ok = await _ensure_backend_running(
        args, state, project_root, console,
    )
    if not ok:
        return 1

    status_data = await _fetch_status_data(state, auto_serve_state)
    if status_data:
        state.instance_name = status_data.get("instance_name", "evoref")

    # 設定済みだが未起動の補助サーバー（aux/embed）を自動起動
    await ensure_extra_servers(project_root, auto_serve_state)

    # セッション登録（設計書 09 §9.4.5）
    if not await _register_session(state.backend_url, state.session_id, state.mode):
        render_error(
            console,
            msg("cli.session_duplicate", session_id=state.session_id),
        )
        _auto_serve_cleanup(auto_serve_state, console)
        return 1

    # 解決モードを base サーバへ反映 (create_model が別 GGUF の構成で create
    # モデルをロードさせる)。同一モード / create_model 未設定なら no-op。
    await _sync_server_mode(state.backend_url, state.mode)

    if non_interactive:
        return await _run_non_interactive_chat(args, state, console, auto_serve_state)

    _load_context_files(state, console)
    return await _run_interactive_loop(
        state, console, project_root, status_data, auto_serve_state,
    )


# ────────────────────────────────────────────
# エントリーポイント
# ────────────────────────────────────────────


def main() -> None:
    """同期メインエントリーポイント"""
    if not _setup_encoding():
        sys.exit(1)

    # サブコマンドを位置に依存せず検出（--develop serve 等に対応）
    argv = _canonicalize_subcommands(sys.argv[1:])

    # サブコマンドルーティング
    for subcmd in _SUBCOMMANDS:
        if subcmd in argv:
            handler = _get_subcommand_handler(subcmd)
            if handler is not None:
                sub_argv = [a for a in argv if a != subcmd]
                sys.exit(handler(sub_argv))

    # "chat" / "create" サブコマンド: 対話モードと同等（引数を除去するだけ）。
    # 実モード解決は `--mode` + エディションデフォルト (`cli_mode.coerce_cli_mode`) に委譲する。
    if "chat" in argv:
        argv = [a for a in argv if a != "chat"]
    if "create" in argv:
        argv = [a for a in argv if a != "create"]

    parser = _build_interactive_parser()
    args = parser.parse_args(argv)
    _resolve_backend_url(args)

    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        exit_code = 0

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
