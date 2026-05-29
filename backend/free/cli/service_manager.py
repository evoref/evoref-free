"""CLI サービス管理: バックエンド/llama-server のプロセス管理"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

from backend.free.cli.config_loader import _find_project_root
from backend.free.cli.develop_mode_setup import setup_develop_mode
from backend.free.cli.edition_validator import validate_edition_arg
from backend.free.cli.pid_manager import (
    _run_windows_console_command,
    acquire_pid,
    check_pid,
    collect_configured_ports,
    find_port_occupants,
    force_release_stale_pid,
    kill_port_occupants,
    release_pid,
)
from backend.free.cli.renderer import (
    create_console,
    render_error,
    render_info,
    render_startup_result,
)
from backend.free.cli.startup_checks import (
    has_critical_failure,
    run_serve_checks,
    validate_config,
)
from backend.error_handlers import E6003
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger

logger = get_logger("cli.service_manager")


# ────────────────────────────────────────────
# auto-serve 管理情報
# ────────────────────────────────────────────

@dataclass
class AutoServeState:
    """auto-serve で管理するプロセス群の状態"""
    procs: list[subprocess.Popen] = field(default_factory=list)
    llama_port: int = 0  # llama-server のポート（停止時にポートから kill 用）
    managed_ports: list[int] = field(default_factory=list)  # 全管理対象ポート
    stderr_files: list = field(default_factory=list)  # stderr ログファイルハンドル
    status_data: dict | None = None  # ヘルスチェック完了時の /api/status キャッシュ

    @property
    def active(self) -> bool:
        return bool(self.procs) or self.llama_port > 0 or bool(self.managed_ports)


def _open_stderr_log(project_root: Path, name: str):
    """サブプロセスの stderr をキャプチャするログファイルを開く"""
    log_dir = project_root / "local" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return open(log_dir / f"{name}.stderr.log", "w", encoding="utf-8")


# ────────────────────────────────────────────
# serve コマンド
# ────────────────────────────────────────────

def _add_develop_flag(parser: argparse.ArgumentParser) -> None:
    """--develop / --isolate-data フラグをパーサーに追加する（Pro 版のみ有効）

    Pro CLI への直接 import を排除し、`develop_hook` 経由に統一
    Pro が登録されていれば help 付きで追加、未登録なら hidden で追加。
    """
    from backend.free.cli.develop_hook import get_develop_hook
    get_develop_hook().extend_parser(parser)


def _build_serve_parser() -> argparse.ArgumentParser:
    """serve サブコマンド用パーサー"""
    parser = argparse.ArgumentParser(
        prog="evoref serve",
        description="Start llama-server and FastAPI backend",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Backend bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", default=8000, type=int,
        help="Backend port (default: 8000)",
    )
    parser.add_argument(
        "--no-llama", action="store_true",
        help="Skip llama-server startup (backend only)",
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
        help=(
            "Disable self-learning cycle (Level 0 experience record / "
            "Level 1 prompt evolution / Level 2 LoRA / Pro assist injection). "
            "All side-effecting writes become no-op; reads continue. "
            "Useful with --develop=evolve to observe initial behavior."
        ),
    )
    _add_develop_flag(parser)
    return parser


def _setup_develop_mode(args: argparse.Namespace, console) -> int | None:
    """`--develop=<level>` / `--isolate-data` の環境セットアップ (共通実装へ委譲)。

    Free では ``debug`` のみ受け付け、``investigate`` / ``evolve`` は
    Pro 限定 (起動時 ``apply_develop_overrides`` で sys.exit)。
    ``--isolate-data`` は Pro 拡張のみで効果があり、Free では no-op。
    """
    return setup_develop_mode(args, console)


def _handle_pid_collision(project_root: Path, console, force: bool) -> int | None:
    """既存 PID ファイル検出時の処理。継続可なら ``None``、中止なら終了コード。"""
    existing_pid = check_pid(project_root)
    if existing_pid is None:
        return None
    if force:
        render_info(console, msg("cli.force_killing_stale_pid", pid=existing_pid))
        force_release_stale_pid(project_root)
        return None
    logger.error("E6003 Another evoref process is running: pid=%d", existing_pid)
    render_error(
        console,
        msg("cli.already_running", pid=existing_pid),
        code=E6003,
    )
    render_info(console, msg("cli.hint_use_force"))
    return 1


def _resolve_startup_config(
    project_root: Path,
    args: argparse.Namespace,
    console,
    force: bool,
) -> dict | None:
    """設定読込 + 起動前提チェック + ポート競合解決。失敗時 ``None``。"""
    config, config_result = validate_config(project_root)
    render_startup_result(console, config_result)
    if config is None:
        render_error(console, msg("cli.startup_critical_failed"))
        return None

    # serve コマンド主フロー専用の apply。`validate_config()` が
    # 返した生 config に対して `--isolate-data` を反映する。`_load_config()`
    # 側 (auto-serve / chat / gui 等の独立フロー) との二重 call は意図的な
    # 設計で、両者は別々の cfg dict を扱う。万一同一 dict に複数回適用された
    # 場合も `apply_develop_overrides` の冪等性契約 で安全
    develop_level = getattr(args, "develop", None)
    if develop_level is not None:
        from backend.free.cli.develop_hook import get_develop_hook
        get_develop_hook().apply_develop_overrides(config, develop_level)

    check_results = run_serve_checks(
        project_root,
        config,
        skip_llama_check=not args.no_llama,
        skip_port_check=force,
    )
    for result in check_results:
        render_startup_result(console, result)
    if has_critical_failure(check_results):
        render_error(console, msg("cli.startup_critical_failed"))
        return None

    if not _resolve_port_conflicts(config, console, force):
        return None

    render_info(console, msg("cli.startup_complete"))
    return config


def _resolve_port_conflicts(config: dict, console, force: bool) -> bool:
    """ポート競合を検出し、必要なら kill して再確認する。継続可なら ``True``。"""
    ports = collect_configured_ports(config)
    occupants = find_port_occupants(ports)
    if not occupants:
        return True
    if not force:
        render_info(console, msg("cli.hint_use_force"))
        return True

    for occ in occupants:
        render_info(
            console,
            msg("cli.force_killing_port", port=occ.port, pid=occ.pid, name=occ.process_name or "unknown"),
        )
    kill_port_occupants(occupants)
    time.sleep(1)
    still_occupied = find_port_occupants(ports)
    if still_occupied:
        for occ in still_occupied:
            render_error(
                console,
                msg("cli.port_still_occupied", port=occ.port, pid=occ.pid),
            )
        return False
    render_info(console, msg("cli.force_ports_cleared"))
    return True


def _validate_edition_arg(args: argparse.Namespace, console) -> int | None:
    """`--edition` の整合性検証 (共通実装へ委譲)。継続可なら ``None``、不整合なら 1。"""
    return validate_edition_arg(args, console)


def _make_serve_cleanup(
    console,
    procs: list[subprocess.Popen],
    stderr_files: list,
    project_root: Path,
):
    """serve 用 cleanup クロージャを生成（signal ハンドラから呼ばれる）。"""

    def cleanup(signum=None, frame=None):
        render_info(console, "Shutting down...")
        for p in procs:
            _kill_process_tree(p.pid)
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in stderr_files:
            try:
                f.close()
            except OSError:
                pass
        release_pid(project_root)
        render_info(console, "All processes stopped")

    return cleanup


def _spawn_llama_base(
    project_root: Path,
    config: dict,
    procs: list[subprocess.Popen],
    stderr_files: list,
    console,
) -> tuple[str, int] | None:
    """ベース llama-server を起動。失敗時 ``None`` を返し PID を解放する。"""
    render_info(console, "Starting llama-server...")
    try:
        llama_cmd = _build_llama_cmd(project_root, config)
        stderr_f = _open_stderr_log(project_root, "llama-base")
        stderr_files.append(stderr_f)
        llama_proc = subprocess.Popen(
            llama_cmd,
            cwd=str(project_root),
            stderr=stderr_f,
        )
        procs.append(llama_proc)
    except (FileNotFoundError, ValueError) as e:
        render_error(console, f"Failed to start llama-server: {e}")
        release_pid(project_root)
        return None
    llama_cfg = config.get("llama", {})
    return llama_cfg.get("host", "127.0.0.1"), llama_cfg.get("port", 8080)


def _spawn_and_wait_llama(
    project_root: Path,
    config: dict,
    args: argparse.Namespace,
    procs: list[subprocess.Popen],
    stderr_files: list,
    console,
) -> dict[str, tuple[str, int]] | None:
    """llama-server 群を起動 → 並列ヘルスチェック。失敗時 ``None``。"""
    all_servers: dict[str, tuple[str, int]] = {}
    if args.no_llama:
        return all_servers

    base = _spawn_llama_base(project_root, config, procs, stderr_files, console)
    if base is None:
        return None
    all_servers["base"] = base

    extra_servers = _spawn_extra_servers(
        project_root, config, procs, stderr_files, console,
    )
    all_servers.update(extra_servers)

    render_info(console, f"Waiting for {len(all_servers)} server(s)...")
    if not _wait_for_health_all(
        all_servers, procs, console, project_root, timeout=30,
    ):
        return None
    return all_servers


def _build_backend_env(args: argparse.Namespace) -> dict | None:
    """FastAPI バックエンドプロセスの環境変数を構築（変更不要なら ``None``、
    ``EVOREF_DEVELOP_LEVEL`` / ``EVOREF_LEARNING_DISABLED`` 環境変数経由）。"""
    backend_env: dict | None = None
    develop_level = getattr(args, "develop", None)
    if develop_level is not None:
        backend_env = {**os.environ, "EVOREF_DEVELOP_LEVEL": develop_level}
        if getattr(args, "isolate_data", False):
            backend_env["EVOREF_ISOLATE_DATA"] = "1"
    if args.edition is not None:
        backend_env = {**(backend_env or os.environ), "EVOREF_EDITION": args.edition}
    if getattr(args, "no_learning", False):
        backend_env = {**(backend_env or os.environ), "EVOREF_LEARNING_DISABLED": "1"}
    return backend_env


def _spawn_backend(
    project_root: Path,
    config: dict,
    args: argparse.Namespace,
    procs: list[subprocess.Popen],
    console,
) -> int | None:
    """FastAPI バックエンドを起動。成功時はバインドポート、失敗時 ``None``。"""
    backend_port = config.get("server", {}).get("port", args.port)
    render_info(console, f"Starting FastAPI backend on :{backend_port}...")
    backend_env = _build_backend_env(args)
    try:
        backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app",
             "--host", args.host, "--port", str(backend_port)],
            cwd=str(project_root),
            env=backend_env,
        )
        procs.append(backend_proc)
    except FileNotFoundError as e:
        render_error(console, f"Failed to start backend: {e}")
        return None
    return backend_port


def _print_serve_running_banner(
    console,
    backend_port: int,
    no_llama: bool,
    all_servers: dict[str, tuple[str, int]],
) -> None:
    """`serve` 起動完了バナーを表示する。"""
    render_info(console, "")
    render_info(console, "=== evoref is running ===")
    render_info(console, f"  API:   http://localhost:{backend_port}")
    if not no_llama:
        for name, (host, port) in all_servers.items():
            render_info(console, f"  {name}: http://{host}:{port}")
    render_info(console, "")
    render_info(console, "Press Ctrl+C to stop all services")


def _monitor_serve_processes(
    procs: list[subprocess.Popen],
    project_root: Path,
    console,
    cleanup,
) -> int:
    """全プロセスを監視。子プロセス異常終了 → 1、Ctrl+C → 0。"""
    try:
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    render_error(
                        console,
                        msg("cli.process_exited", pid=p.pid, code=ret),
                    )
                    _show_stderr_tail(console, project_root, "llama-base")
                    _show_stderr_tail(console, project_root, "backend")
                    render_info(
                        console,
                        msg("cli.hint_check_stderr", path=str(project_root / "local" / "logs")),
                    )
                    cleanup()
                    return 1
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()
        return 0


def _run_serve(args: argparse.Namespace) -> int:
    """serve コマンド: llama-server + FastAPI バックエンドを起動

    フェーズごとに 11 個のヘルパー関数へ責務を分割している:
        1. `_setup_develop_mode` (--develop / --isolate-data 検証)
        2. `_handle_pid_collision` (既存 PID チェック / --force kill)
        3. `_resolve_startup_config` (config 読込 + 前提チェック + ポート競合)
        4. `_validate_edition_arg` (--edition + --develop 整合性)
        5. `acquire_pid` + `_make_serve_cleanup` + signal ハンドラ登録
        6. `_spawn_and_wait_llama` (llama-server 群起動 + ヘルスチェック)
        7. `_spawn_backend` (FastAPI 起動)
        8. `_print_serve_running_banner` + `_monitor_serve_processes` (監視ループ)
    """
    init_i18n()
    console = create_console()

    if (rc := _setup_develop_mode(args, console)) is not None:
        return rc

    project_root = _find_project_root()
    force = getattr(args, "force", False)

    if (rc := _handle_pid_collision(project_root, console, force)) is not None:
        return rc

    config = _resolve_startup_config(project_root, args, console, force)
    if config is None:
        return 1

    if (rc := _validate_edition_arg(args, console)) is not None:
        return rc

    if not acquire_pid(project_root):
        render_error(console, msg("cli.already_running", pid=0), code=E6003)
        return 1

    procs: list[subprocess.Popen] = []
    stderr_files: list = []
    cleanup = _make_serve_cleanup(console, procs, stderr_files, project_root)
    signal.signal(signal.SIGINT, lambda s, f: cleanup(s, f) or sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: cleanup(s, f) or sys.exit(0))

    all_servers = _spawn_and_wait_llama(
        project_root, config, args, procs, stderr_files, console,
    )
    if all_servers is None:
        cleanup()
        return 1

    backend_port = _spawn_backend(project_root, config, args, procs, console)
    if backend_port is None:
        cleanup()
        return 1

    _print_serve_running_banner(console, backend_port, args.no_llama, all_servers)
    return _monitor_serve_processes(procs, project_root, console, cleanup)


# ────────────────────────────────────────────
# ヘルパー
# ────────────────────────────────────────────


def _show_stderr_tail(console, project_root: Path, name: str, lines: int = 10) -> None:
    """stderr ログファイルの末尾を表示して障害診断を支援"""
    log_file = project_root / "local" / "logs" / f"{name}.stderr.log"
    if not log_file.exists():
        return
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
        tail = content.strip().splitlines()[-lines:]
        if tail:
            render_info(console, f"── {name} stderr (last {len(tail)} lines) ──")
            for line in tail:
                console.print(f"  {line}")
    except OSError:
        pass


def _load_config(project_root: Path) -> dict:
    """config.yaml を読込み（develop モード時はオーバーライド適用済み）。

    develop モード時の設定オーバーライドは llama コマンド構築の
    ためのデータ分離 (`apply_data_isolation`) のみ。ログ系設定は
    ``setup_logging`` / ``DebugLogger`` が ``develop_level`` から直接導出
    するため、ここで debug セクションを書き換える必要はない。

    env var (`EVOREF_DEVELOP_LEVEL`) 経由の apply。これは
    `_resolve_startup_config()` を経由しない独立フロー (auto-serve / chat /
    gui 等で `_auto_serve_start()` 等から直接呼ばれるケース) を支えるための
    apply。serve コマンド主フローでは別途
    `_resolve_startup_config()` 内でも apply されるが、両者は yaml.safe_load
    で生成した別々の cfg dict を扱うため二重適用には該当しない。
    `apply_develop_overrides` の冪等性契約により、万一同一 dict に複数回
    apply された場合でも結果は変わらない。
    """
    config_path = project_root / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    raw_level = os.environ.get("EVOREF_DEVELOP_LEVEL", "")
    if raw_level in ("debug", "investigate", "evolve"):
        from backend.free.cli.develop_hook import get_develop_hook
        get_develop_hook().apply_develop_overrides(cfg, raw_level)  # type: ignore[arg-type]

    return cfg


def _build_llama_cmd(project_root: Path, cfg: dict | None = None) -> list[str]:
    """config.yaml から llama-server 起動コマンドを構築"""
    from scripts.launch_llama import build_llama_cmd

    if cfg is None:
        cfg = _load_config(project_root)
    return build_llama_cmd(cfg, project_root)


def _build_embed_cmd(project_root: Path, cfg: dict | None = None) -> list[str] | None:
    """config.yaml から埋め込み用 llama-server コマンドを構築（設定時のみ）"""
    from scripts.launch_llama import build_embed_cmd

    if cfg is None:
        cfg = _load_config(project_root)
    return build_embed_cmd(cfg, project_root)


def _build_assist_cmd(project_root: Path, cfg: dict | None = None) -> list[str] | None:
    """config.yaml からアシストモデル用 llama-server コマンドを構築（設定時のみ）"""
    from scripts.launch_llama import build_assist_cmd

    if cfg is None:
        cfg = _load_config(project_root)
    return build_assist_cmd(cfg, project_root)


def _build_reranker_cmd(project_root: Path, cfg: dict | None = None) -> list[str] | None:
    """config.yaml からリランカー用 llama-server コマンドを構築（設定時のみ）"""
    from scripts.launch_llama import build_reranker_cmd

    if cfg is None:
        cfg = _load_config(project_root)
    return build_reranker_cmd(cfg, project_root)


# ────────────────────────────────────────────
# 追加サーバー起動（アシスト・埋め込み・リランカー）
# ────────────────────────────────────────────

# (name, config_key, port_key, default_port, build_cmd_func) の定義
_EXTRA_SERVERS: list[tuple[str, str, str, int]] = [
    # (表示名, config セクションキー, ポートキー, デフォルトポート)
    ("assist", "assist_model.local", "port", 8081),
    ("embed", "embedding", "llama_port", 8082),
    ("reranker", "reranker", "port", 8083),
]

def _get_nested_config(cfg: dict, dotted_key: str) -> dict:
    """ドット区切りキーで config のネストした辞書を取得"""
    result = cfg
    for key in dotted_key.split("."):
        result = result.get(key, {})
        if not isinstance(result, dict):
            return {}
    return result


def _spawn_extra_servers(
    project_root: Path,
    config: dict,
    procs: list[subprocess.Popen],
    stderr_files: list,
    console,
) -> dict[str, tuple[str, int]]:
    """補助サーバー（assist/embed/reranker）を一括スポーン（ヘルスチェックなし）

    Popen はノンブロッキングなので全サーバーを即座にスポーンし、
    ヘルスチェックは呼び出し元の _wait_for_health_all() に委譲する。

    Returns:
        {name: (host, port)} — スポーンしたサーバーのみ
    """
    builders = {
        "assist": _build_assist_cmd,
        "embed": _build_embed_cmd,
        "reranker": _build_reranker_cmd,
    }
    servers: dict[str, tuple[str, int]] = {}

    for name, config_key, port_key, default_port in _EXTRA_SERVERS:
        build_fn = builders[name]
        try:
            cmd = build_fn(project_root, config)
            if cmd is None:
                continue
            section = _get_nested_config(config, config_key)
            port = section.get(port_key, default_port)
            host = section.get("host", "127.0.0.1")
            render_info(console, f"Starting {name} server on :{port}...")
            stderr_f = _open_stderr_log(project_root, f"llama-{name}")
            stderr_files.append(stderr_f)
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.DEVNULL,
                stderr=stderr_f,
            )
            procs.append(proc)
            servers[name] = (host, port)
        except (FileNotFoundError, ValueError, OSError) as e:
            render_error(console, f"Failed to start {name} server: {e}")

    return servers


def _wait_for_health_all(
    servers: dict[str, tuple[str, int]],
    procs: list[subprocess.Popen],
    console,
    project_root: Path,
    *,
    timeout: int = 30,
    base_is_critical: bool = True,
) -> bool:
    """全サーバーの並列ヘルスチェック（同期版）

    全サーバーを同時にポーリングし、最も遅いサーバーに律速される。
    ベースサーバーのヘルスチェック失敗は致命的（return False）。
    補助サーバーの失敗は警告のみ。

    Args:
        servers: {name: (host, port)} — チェック対象サーバー
        procs: 管理下のプロセスリスト（死亡検知用）
        console: 表示用コンソール
        project_root: プロジェクトルート（stderr ログ表示用）
        timeout: ヘルスチェックタイムアウト秒数
        base_is_critical: True の場合、base サーバー失敗で即 return False

    Returns:
        True: 致命的エラーなし, False: ベースサーバー失敗
    """
    if not servers:
        return True

    pending = set(servers.keys())
    deadline = time.time() + timeout

    while pending and time.time() < deadline:
        # プロセス即死チェック
        for p in procs:
            ret = p.poll()
            if ret is not None:
                render_error(
                    console,
                    msg("cli.server_crashed_on_start", name="llama-server", code=ret),
                )
                _show_stderr_tail(console, project_root, "llama-base")
                return False

        for name in list(pending):
            host, port = servers[name]
            try:
                resp = httpx.get(f"http://{host}:{port}/health", timeout=2.0)
                if resp.status_code == 200:
                    pending.discard(name)
                    render_info(console, f"{name} server is ready")
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

        if pending:
            time.sleep(1.0)

    # タイムアウトしたサーバーの処理
    for name in pending:
        host, port = servers[name]
        if name == "base" and base_is_critical:
            # ベースサーバーの失敗は致命的
            render_error(
                console,
                msg("cli.server_health_timeout", name="llama-server", port=port),
            )
            _show_stderr_tail(console, project_root, "llama-base")
            return False
        render_error(console, f"{name} server health check timed out")

    return True


async def ensure_extra_servers(
    project_root: Path,
    auto_serve_state: "AutoServeState",
) -> None:
    """設定済みだが未起動の補助サーバー（assist/embed/reranker）を自動起動

    バックエンドが既に起動している状態で、--auto-serve を使わずに
    evoref code を実行した場合に補助サーバーを補完する。
    """
    cfg = _load_config(project_root)
    builders = {
        "assist": _build_assist_cmd,
        "embed": _build_embed_cmd,
        "reranker": _build_reranker_cmd,
    }

    for name, config_key, port_key, default_port in _EXTRA_SERVERS:
        build_fn = builders[name]
        try:
            cmd = build_fn(project_root, cfg)
            if cmd is None:
                continue
            section = _get_nested_config(cfg, config_key)
            port = section.get(port_key, default_port)
            host = section.get("host", "127.0.0.1")

            # 既に起動中ならスキップ
            if await _check_llama_health(f"http://{host}:{port}"):
                logger.debug("ensure_extra_servers: %s already healthy on :%d", name, port)
                continue

            # スポーン
            stderr_f = _open_stderr_log(project_root, f"llama-{name}")
            auto_serve_state.stderr_files.append(stderr_f)
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.DEVNULL,
                stderr=stderr_f,
            )
            auto_serve_state.procs.append(proc)
            auto_serve_state.managed_ports.append(port)
            logger.info("ensure_extra_servers: spawned %s on :%d (pid=%d)", name, port, proc.pid)
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.warning("ensure_extra_servers: %s start failed: %s", name, e)


def _spawn_all_servers(
    project_root: Path,
    cfg: dict,
    state: AutoServeState,
    *,
    skip_base: bool = False,
) -> dict[str, tuple[str, int]]:
    """全 llama-server を一括スポーン（並列起動、ヘルスチェックなし）

    base + assist + embed + reranker を同時に起動し、
    ヘルスチェックは呼び出し元の _health_check_loop に委譲する。

    Args:
        skip_base: True の場合、ベース llama-server のスポーンをスキップ（既に起動中）

    Returns:
        {name: (host, port)} — スポーンまたは既存検出されたサーバー
    """
    servers: dict[str, tuple[str, int]] = {}
    builders = {
        "assist": _build_assist_cmd,
        "embed": _build_embed_cmd,
        "reranker": _build_reranker_cmd,
    }

    # ── base llama-server ──
    llama_cfg = cfg.get("llama", {})
    llama_host = llama_cfg.get("host", "127.0.0.1")
    llama_port = llama_cfg.get("port", 8080)
    state.llama_port = llama_port

    if not skip_base:
        try:
            llama_cmd = _build_llama_cmd(project_root, cfg)
            stderr_f = _open_stderr_log(project_root, "llama-base")
            state.stderr_files.append(stderr_f)
            proc = subprocess.Popen(
                llama_cmd,
                cwd=str(project_root),
                stdout=subprocess.DEVNULL,
                stderr=stderr_f,
            )
            state.procs.append(proc)
            servers["base"] = (llama_host, llama_port)
            logger.debug("auto-serve: spawned base llama-server on :%d (pid=%d)", llama_port, proc.pid)
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.warning("auto-serve: base llama-server start failed: %s", e)

    # ── extra servers (assist / embed / reranker) ──
    for name, config_key, port_key, default_port in _EXTRA_SERVERS:
        build_fn = builders[name]
        try:
            cmd = build_fn(project_root, cfg)
            if cmd is None:
                continue
            section = _get_nested_config(cfg, config_key)
            port = section.get(port_key, default_port)
            host = section.get("host", "127.0.0.1")
            stderr_f = _open_stderr_log(project_root, f"llama-{name}")
            state.stderr_files.append(stderr_f)
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.DEVNULL,
                stderr=stderr_f,
            )
            state.procs.append(proc)
            state.managed_ports.append(port)
            servers[name] = (host, port)
            logger.debug("auto-serve: spawned %s server on :%d (pid=%d)", name, port, proc.pid)
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.warning("auto-serve: %s server start failed: %s", name, e)

    return servers


async def _check_llama_health(llama_url: str) -> bool:
    """llama-server に直接ヘルスチェック"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{llama_url}/health", timeout=3.0)
            healthy = resp.status_code == 200
            logger.debug("llama-server direct health check: %s → %s", llama_url, healthy)
            return healthy
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.debug("llama-server direct health check failed: %s", e)
        return False


async def _check_backend(url: str) -> bool:
    """バックエンド接続確認"""
    logger.debug("Checking backend at %s/api/health", url)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/api/health", timeout=5.0)
            logger.debug("Backend health check: status=%d", resp.status_code)
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.debug("Backend health check failed: %s", e)
        return False


async def _register_session(
    url: str, session_id: str, mode: str | None = None,
) -> bool:
    """バックエンドにセッションを登録。成功で True、重複で False

    Args:
        mode: チャットモード。``None`` の場合は :func:`default_cli_mode` で
            エディション既定 (Free=chat / Pro=coding) を解決する
    """
    if mode is None:
        from backend.free.cli.cli_mode import default_cli_mode
        mode = default_cli_mode()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{url}/api/sessions/register",
                json={"session_id": session_id, "mode": mode, "client_type": "cli"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                logger.debug("Session registered: %s", session_id)
                return True
            if resp.status_code == 409:
                logger.warning("Session ID duplicate: %s", session_id)
                return False
            logger.warning("Session register unexpected status: %d", resp.status_code)
            return False
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
        logger.debug("Session register failed (ignored): %s", e)
        return True  # 登録失敗は致命的でない — 処理を続行


async def _unregister_session(url: str, session_id: str) -> None:
    """バックエンドからセッションを解除（ベストエフォート）"""
    try:
        async with httpx.AsyncClient() as client:
            await client.delete(
                f"{url}/api/sessions/{session_id}",
                timeout=5.0,
            )
            logger.debug("Session unregistered: %s", session_id)
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
        logger.debug("Session unregister failed (ignored): %s", e)


# ────────────────────────────────────────────
# auto-serve
# ────────────────────────────────────────────

@dataclass
class _HealthCheckSpinner:
    """`_health_check_loop` のスピナー描画状態を集約する。

    `is_tty` が False なら描画をスキップ (パイプ・リダイレクト時)。
    `no_color` が True なら ANSI EL の代わりにスペースパディングでクリア。
    """

    start_time: float
    is_tty: bool
    no_color: bool
    spinner_idx: int = 0

    def render(self, statuses: dict[str, bool]) -> None:
        """現在の statuses でスピナー行を 1 フレーム描画する (TTY のみ)。"""
        if not self.is_tty:
            return
        from backend.free.cli.renderer import render_startup_status
        elapsed = time.time() - self.start_time
        line = render_startup_status(
            statuses, elapsed, self.spinner_idx, no_color=self.no_color,
        )
        if self.no_color:
            sys.stdout.write(f"\r{line: <120}")  # スペースパディングでクリア
        else:
            sys.stdout.write(f"\r{line}\033[K")  # ANSI EL でクリア
        sys.stdout.flush()
        self.spinner_idx += 1

    def clear(self) -> None:
        """スピナー行をクリアする (TTY のみ)。"""
        if not self.is_tty:
            return
        from backend.free.cli.renderer import clear_spinner_line
        clear_spinner_line()


def _check_procs_alive(state: AutoServeState) -> bool:
    """子プロセスのいずれかが死亡していたら ``False``。"""
    for p in state.procs:
        if p.poll() is not None:
            logger.error(
                "auto-serve: process died during startup (pid=%d)", p.pid,
            )
            return False
    return True


async def _wait_backend_phase(
    backend_url: str,
    statuses: dict[str, bool],
    spinner: _HealthCheckSpinner,
    state: AutoServeState,
    console,
    timeout_backend: int,
) -> bool:
    """FastAPI バックエンドの /health が通るまで待機する"""
    for _ in range(timeout_backend):
        spinner.render(statuses)
        await asyncio.sleep(1)
        if not _check_procs_alive(state):
            spinner.clear()
            return False
        if await _check_backend(backend_url):
            statuses["backend"] = True
            logger.debug("auto-serve: backend is up")
            return True
    spinner.clear()
    logger.error(
        "auto-serve: backend health check timed out after %ds", timeout_backend,
    )
    render_error(
        console, msg("cli.auto_serve_timeout_backend", timeout=timeout_backend),
    )
    return False


async def _wait_llama_servers_phase(
    servers: dict[str, tuple[str, int]],
    statuses: dict[str, bool],
    spinner: _HealthCheckSpinner,
    state: AutoServeState,
    console,
    timeout_llama: int,
) -> bool:
    """各 llama-server の /health が通るまで待機する"""
    pending_servers = set(servers.keys())
    for _ in range(timeout_llama):
        spinner.render(statuses)
        await asyncio.sleep(1)
        if not _check_procs_alive(state):
            spinner.clear()
            return False
        for name in list(pending_servers):
            host, port = servers[name]
            if await _check_llama_health(f"http://{host}:{port}"):
                statuses[name] = True
                pending_servers.discard(name)
                logger.debug(
                    "auto-serve: %s server is healthy on :%d", name, port,
                )
        if not pending_servers:
            return True

    spinner.clear()
    still_pending = ", ".join(pending_servers)
    logger.error(
        "auto-serve: llama-server health check timed out after %ds: %s",
        timeout_llama, still_pending,
    )
    render_error(
        console, msg("cli.auto_serve_timeout_llama", timeout=timeout_llama),
    )
    return False


async def _wait_backend_llama_bridge_phase(
    backend_url: str,
    statuses: dict[str, bool],
    spinner: _HealthCheckSpinner,
    state: AutoServeState,
) -> bool:
    """バックエンド経由で llama-server 接続確認 (10 秒)"""
    for _ in range(10):
        spinner.render(statuses)
        await asyncio.sleep(1)
        if not _check_procs_alive(state):
            spinner.clear()
            return False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{backend_url}/api/status", timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("llama_server", {}).get("connected", False):
                        state.status_data = data
                        spinner.clear()
                        return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

    spinner.clear()
    logger.error(
        "auto-serve: backend did not detect llama-server connection after bridge wait",
    )
    return False


async def _health_check_loop(
    backend_url: str,
    servers: dict[str, tuple[str, int]],
    state: AutoServeState,
    console,
    *,
    backend_already_up: bool = False,
    timeout_backend: int = 30,
    timeout_llama: int = 120,
) -> bool:
    """スピナー付きヘルスチェックループ

    FastAPI バックエンド起動待ち (timeout_backend 秒)
    各 llama-server /health 直接チェック (timeout_llama 秒)
    バックエンド経由で llama-server 接続確認 (10 秒)

    Returns:
        True: 全サーバー起動完了, False: タイムアウトまたはプロセス死亡
    """
    spinner = _HealthCheckSpinner(
        start_time=time.time(),
        is_tty=sys.stdout.isatty(),
        no_color=console.no_color if hasattr(console, "no_color") else False,
    )
    statuses: dict[str, bool] = {"backend": backend_already_up}
    for name in servers:
        statuses[name] = False

    if not backend_already_up:
        if not await _wait_backend_phase(
            backend_url, statuses, spinner, state, console, timeout_backend,
        ):
            return False

    if not await _wait_llama_servers_phase(
        servers, statuses, spinner, state, console, timeout_llama,
    ):
        return False

    return await _wait_backend_llama_bridge_phase(
        backend_url, statuses, spinner, state,
    )


def _check_auto_serve_pid_collision(
    project_root: Path, console, force: bool,
) -> bool:
    """auto-serve の PID 重複チェック。継続可なら ``True``、中止なら ``False``。

    `--force` 時は古い PID を kill して継続する。
    """
    existing_pid = check_pid(project_root)
    if existing_pid is None:
        return True
    if force:
        render_info(console, msg("cli.force_killing_stale_pid", pid=existing_pid))
        force_release_stale_pid(project_root)
        return True
    logger.debug(
        "auto-serve: serve already running (pid=%d), skipping", existing_pid,
    )
    render_info(
        console, msg("cli.auto_serve_already_running", pid=existing_pid),
    )
    return False


async def _kill_auto_serve_port_conflicts(
    project_root: Path, console, force: bool,
) -> None:
    """`--force` 時に config 上のポートを占有しているプロセスを kill する。"""
    if not force:
        return
    cfg_pre = _load_config(project_root)
    ports = collect_configured_ports(cfg_pre)
    occupants = find_port_occupants(ports)
    if not occupants:
        return
    for occ in occupants:
        render_info(
            console,
            msg("cli.force_killing_port", port=occ.port, pid=occ.pid,
                name=occ.process_name or "unknown"),
        )
    kill_port_occupants(occupants)
    await asyncio.sleep(1)


async def _maybe_spawn_auto_serve_llama(
    project_root: Path,
    cfg: dict,
    state: AutoServeState,
    no_llama: bool,
    console,
) -> dict[str, tuple[str, int]]:
    """auto-serve 時の llama-server 群スポーン (ベース既存ならスキップ)。

    `no_llama=True` なら何もしない。ベース llama-server が既に動いていれば
    `skip_base=True` で補助サーバーのみ起動。
    """
    if no_llama:
        return {}
    llama_cfg = cfg.get("llama", {})
    llama_host = llama_cfg.get("host", "localhost")
    llama_port_val = llama_cfg.get("port", 8080)
    llama_url = f"http://{llama_host}:{llama_port_val}"
    llama_already_running = await _check_llama_health(llama_url)
    if llama_already_running:
        logger.debug(
            "auto-serve: llama-server already healthy at %s", llama_url,
        )
        render_info(console, msg("cli.auto_serve_llama_exists"))
        state.llama_port = llama_port_val
    # ベースが既存でも、assist/embed/reranker は未起動の可能性があるため常にスポーン
    return _spawn_all_servers(
        project_root, cfg, state,
        skip_base=llama_already_running,
    )


def _build_auto_serve_backend_env(
    develop: str | None, edition: str | None, no_learning: bool = False,
) -> dict | None:
    """FastAPI バックエンド用環境変数を構築 (`gui` の対応関数と同じパターン、
    """
    backend_env: dict | None = None
    if develop is not None:
        backend_env = {**os.environ, "EVOREF_DEVELOP_LEVEL": develop}
    if edition is not None:
        backend_env = {**(backend_env or os.environ), "EVOREF_EDITION": edition}
    if no_learning:
        backend_env = {**(backend_env or os.environ), "EVOREF_LEARNING_DISABLED": "1"}
    return backend_env


def _spawn_auto_serve_backend(
    project_root: Path,
    host: str,
    backend_port: int,
    backend_env: dict | None,
    state: AutoServeState,
    console,
) -> bool:
    """FastAPI バックエンドプロセスを起動する。失敗時 ``False``。"""
    try:
        stderr_f = _open_stderr_log(project_root, "backend")
        state.stderr_files.append(stderr_f)
        backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app",
             "--host", host, "--port", str(backend_port)],
            cwd=str(project_root),
            env=backend_env,
            stdout=subprocess.DEVNULL,
            stderr=stderr_f,
        )
        state.procs.append(backend_proc)
        return True
    except FileNotFoundError as e:
        logger.error("auto-serve: backend start failed: %s", e)
        render_error(console, msg("cli.auto_serve_failed"))
        return False


async def _auto_serve_start(
    project_root: Path,
    host: str,
    port: int,
    *,
    no_llama: bool = False,
    develop: str | None = None,
    edition: str | None = None,
    no_learning: bool = False,
    force: bool = False,
    console,
) -> AutoServeState:
    """バックエンドをバックグラウンドで起動し、ヘルスチェックが通るまで待機

    Returns:
        AutoServeState（/exit 時にクリーンアップ用）
    """
    state = AutoServeState()

    if not _check_auto_serve_pid_collision(project_root, console, force):
        return state

    await _kill_auto_serve_port_conflicts(project_root, console, force)
    render_info(console, msg("cli.auto_serve_starting"))

    cfg = _load_config(project_root)
    auto_serve_cfg = cfg.get("auto_serve", {})
    timeout_backend = auto_serve_cfg.get("timeout_backend", 30)
    timeout_llama = auto_serve_cfg.get("timeout_llama", 120)

    # develop モード時は config のオフセット適用済みポートを使用
    backend_port = cfg.get("server", {}).get("port", port)
    backend_url = f"http://{host}:{backend_port}"

    servers = await _maybe_spawn_auto_serve_llama(
        project_root, cfg, state, no_llama, console,
    )

    # ── FastAPI バックエンド ──
    backend_already_running = await _check_backend(backend_url)
    if backend_already_running:
        logger.debug(
            "auto-serve: backend already healthy at %s", backend_url,
        )
        render_info(console, msg("cli.auto_serve_backend_exists"))
    else:
        backend_env = _build_auto_serve_backend_env(develop, edition, no_learning)
        if not _spawn_auto_serve_backend(
            project_root, host, backend_port, backend_env, state, console,
        ):
            _auto_serve_cleanup(state, console)
            return AutoServeState()

    # ── ヘルスチェックループ（スピナー付き）──
    success = await _health_check_loop(
        backend_url, servers, state, console,
        backend_already_up=backend_already_running,
        timeout_backend=timeout_backend,
        timeout_llama=timeout_llama,
    )

    if success:
        render_info(console, msg("cli.auto_serve_ready"))
        return state

    # 失敗時: stderr ログのパスをヒント表示
    log_dir = project_root / "local" / "logs"
    render_error(console, msg("cli.auto_serve_failed"))
    render_info(console, msg("cli.auto_serve_stderr_hint", path=str(log_dir)))
    _auto_serve_cleanup(state, console)
    return AutoServeState()


# ────────────────────────────────────────────
# プロセス終了
# ────────────────────────────────────────────

def _kill_process_tree(pid: int) -> None:
    """プロセスツリーごと終了（Windows: taskkill /T, Unix: os.killpg）"""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def _kill_by_port(port: int) -> None:
    """指定ポートで LISTEN しているプロセスを終了"""
    if sys.platform == "win32":
        # netstat でポートを使用中の PID を取得。
        # 日本語ロケール (cp932) 環境で UnicodeDecodeError を起こさないよう
        # bytes で受けてから安全にデコードする
        stdout = _run_windows_console_command(["netstat", "-ano"])
        for line in stdout.splitlines():
            # TCP    0.0.0.0:8080    0.0.0.0:0    LISTENING    12345
            # TCP    127.0.0.1:8080  0.0.0.0:0    LISTENING    12345
            if f":{port}" not in line or "LISTENING" not in line:
                continue
            parts = line.split()
            # ポートの正確な一致を確認 (:80 が :8080 に誤マッチしないよう)
            local_addr = parts[1] if len(parts) >= 5 else ""
            if not local_addr.endswith(f":{port}"):
                continue
            pid_str = parts[-1]
            if pid_str.isdigit():
                pid = int(pid_str)
                logger.debug("Killing process on port %d: pid=%d", port, pid)
                _kill_process_tree(pid)
    else:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().split():
                if pid_str.isdigit():
                    pid = int(pid_str)
                    logger.debug("Killing process on port %d: pid=%d", port, pid)
                    _kill_process_tree(pid)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _auto_serve_cleanup(state: AutoServeState, console) -> None:
    """auto-serve で管理するプロセスをすべてクリーンアップ"""
    if not state.active:
        return
    render_info(console, msg("cli.auto_serve_stopping"))

    # 1. 管理下のプロセスをプロセスツリーごと終了
    for p in state.procs:
        _kill_process_tree(p.pid)
    for p in state.procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()

    # 2. llama-server がまだポートで生きていればポート指定で kill
    #    （既に起動済みだったケース・孤児プロセスの回収）
    if state.llama_port > 0:
        _kill_by_port(state.llama_port)

    # 3. アシスト/エンベッド/リランカーのポートも kill
    for port in state.managed_ports:
        _kill_by_port(port)

    # 4. stderr ログファイルをクローズ
    for f in state.stderr_files:
        try:
            f.close()
        except OSError:
            pass

    logger.debug("auto-serve: all processes stopped")
    console.print()
