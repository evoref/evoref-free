"""local/ データを setup.bat 直後の空スケルトンへ初期化するヘルパー (Develop 専用)

backend の Develop ルーター (``POST /api/develop/reset-local-data``) から
**デタッチ起動** される standalone スクリプト。backend 自身を含む全サービスを
停止 → ``local/`` を wipe → サービスを再起動する。停止対象に自身の親 (backend)
が含まれるため、本スクリプトはアプリ context に一切依存しない完全独立プロセス
として動く必要がある (import するのは純粋ヘルパーの ``pid_manager`` のみ)。

工程:

1. **stop**   : llama-server (全インスタンス) + FastAPI backend を停止し、
                ポート解放を待つ。frontend (vite:5173) は停止しない。
2. **wipe**   : ``local/`` を ``.gitkeep`` スケルトンのみへ削除
                (``git clean -xfd local/`` 相当)。
3. **restart**: ``scripts/evoref-ctl.bat start`` をデタッチ起動して再起動する。

``wipe_local_to_skeleton()`` は純関数として分離し、プロセス操作と切り離して
単体テスト可能にしている (``scripts/tests/test_reset_local_data.py``)。

ログは英語固定 (リポジトリ規約)。standalone のため structlog ではなく
``print`` で stdout に出す (呼び出し元が DEVNULL へ捨てるので副作用なし)。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# pid_manager は純粋なプロセス/ポートヘルパー (AppState 非依存) なので
# backend 停止後でも安全に import できる。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.free.cli.pid_manager import (  # noqa: E402
    collect_configured_ports,
    find_port_occupants,
    kill_port_occupants,
)

# evoref-ctl.bat が立てるウィンドウタイトル (start "<title>" ...)。
_WINDOW_TITLES = ("llama-server", "evoref-backend")
# frontend (vite) は再起動対象外なので停止ポートから除外する。
_FRONTEND_PORT = 5173


# ────────────────────────────────────────────
# wipe (純関数 — 単体テスト対象)
# ────────────────────────────────────────────

def _unlink_with_retry(path: Path, retries: int, delay: float) -> bool:
    """ファイルを削除する。Windows のロック残存に備えて軽くリトライする。

    Returns:
        削除できた (もしくは既に存在しない) なら True。
    """
    for attempt in range(retries):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return False
    return False


def wipe_local_to_skeleton(
    local_root: Path,
    *,
    retries: int = 8,
    retry_delay: float = 0.5,
) -> dict:
    """``local/`` を ``.gitkeep`` スケルトンのみの初期状態へ削除する。

    保持するのは ``.gitkeep`` ファイルと、その祖先ディレクトリ群だけ。
    それ以外の全ファイル・全ディレクトリを削除する (``git clean -xfd local/``
    と同一結果: 実体ファイル 0、tracked な ``.gitkeep`` スケルトンは維持)。

    Args:
        local_root: ``local/`` ディレクトリ。
        retries: ロック残存時のファイル削除リトライ回数。
        retry_delay: リトライ間隔 (秒)。

    Returns:
        ``{"deleted_files", "deleted_dirs", "kept_gitkeep", "errors"}`` の集計。
    """
    local_root = Path(local_root)
    if not local_root.exists():
        return {"deleted_files": 0, "deleted_dirs": 0, "kept_gitkeep": 0, "errors": []}

    root_resolved = local_root.resolve()

    # 1. スケルトン (保持対象ディレクトリ) を確定: .gitkeep とその全祖先。
    gitkeeps = list(local_root.rglob(".gitkeep"))
    keep_dirs: set[Path] = {root_resolved}
    for gk in gitkeeps:
        p = gk.parent.resolve()
        while True:
            keep_dirs.add(p)
            if p == root_resolved:
                break
            p = p.parent

    deleted_files = 0
    errors: list[str] = []

    # 2. .gitkeep 以外の全ファイルを削除。
    for path in local_root.rglob("*"):
        if path.is_file() and path.name != ".gitkeep":
            if _unlink_with_retry(path, retries, retry_delay):
                deleted_files += 1
            else:
                errors.append(str(path))

    # 3. スケルトン外ディレクトリを bottom-up (深い順) に削除。
    deleted_dirs = 0
    all_dirs = sorted(
        (p for p in local_root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.resolve().parts),
        reverse=True,
    )
    for d in all_dirs:
        if d.resolve() in keep_dirs:
            continue
        try:
            d.rmdir()
            deleted_dirs += 1
        except OSError as e:
            errors.append(f"{d}: {e}")

    return {
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "kept_gitkeep": len(gitkeeps),
        "errors": errors,
    }


# ────────────────────────────────────────────
# stop / restart (プロセス操作)
# ────────────────────────────────────────────

def _load_config_ports(project_root: Path) -> list[int]:
    """config.yaml から停止対象ポートを収集する。失敗時は既定ポート群へ縮退。"""
    default_ports = [8000, 8080, 8081, 8082, 8083]
    config_path = project_root / "config.yaml"
    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ports = collect_configured_ports(cfg)
        return ports or default_ports
    except Exception as e:  # config 読めなくても停止は続行
        print(f"[reset] WARNING: failed to read config ports ({e}); using defaults")
        return default_ports


def _taskkill_window_titles(titles: tuple[str, ...] = _WINDOW_TITLES) -> None:
    """Windows: evoref-ctl.bat が立てたウィンドウタイトル単位でツリー kill。"""
    if sys.platform != "win32":
        return
    for title in titles:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/FI", f"WINDOWTITLE eq {title}"],
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    # llama-server は実行ファイル名でも確実に落とす (全インスタンス)。
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "llama-server.exe"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _wait_ports_free(ports: list[int], timeout: float, interval: float = 0.5) -> bool:
    """指定ポートが全て解放されるまで待つ。全解放で True、timeout で False。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        occ = find_port_occupants(ports)
        if not occ:
            return True
        time.sleep(interval)
    return not find_port_occupants(ports)


def stop_services(
    project_root: Path,
    *,
    wait_timeout: float = 30.0,
    stop_frontend: bool = False,
) -> dict:
    """全サービス (backend + llama-server) を停止する。

    ``stop_frontend=True`` のとき frontend (vite:5173) も停止する
    (初期化ボタンの「全停止して終了」用)。``False`` のときは従来どおり
    frontend を残す (再起動経路で生きた vite を保つため)。

    ポート占有 kill (backend serve / uvicorn 経路に強い) と
    ウィンドウタイトル kill (evoref-ctl.bat 経路に強い) を併用する。
    """
    config_ports = _load_config_ports(project_root)
    if stop_frontend:
        target_ports = list(config_ports)
        if _FRONTEND_PORT not in target_ports:
            target_ports.append(_FRONTEND_PORT)
    else:
        target_ports = [p for p in config_ports if p != _FRONTEND_PORT]

    # 1. ウィンドウタイトル + 実行ファイル名で kill (evoref-ctl.bat 起動分)。
    #    全停止時は frontend ウィンドウ (evoref-ctl.bat の "evoref-frontend") も対象。
    titles = _WINDOW_TITLES + (("evoref-frontend",) if stop_frontend else ())
    _taskkill_window_titles(titles)
    # 2. ポート占有 PID を kill (evoref serve 起動分 / 取りこぼし)。
    #    自身 (このヘルパー) は対象ポートを LISTEN しないが念のため除外する。
    own_pid = os.getpid()
    occupants = [o for o in find_port_occupants(target_ports) if o.pid != own_pid]
    killed = kill_port_occupants(occupants)
    # 3. backend / llama のポート解放を待つ (ファイルロック解放のため必須)。
    freed = _wait_ports_free(target_ports, wait_timeout)

    return {
        "target_ports": target_ports,
        "killed": [o.summary for o in killed],
        "ports_freed": freed,
    }


def restart_services(project_root: Path) -> int | None:
    """``scripts/evoref-ctl.bat start`` をデタッチ起動して再起動する。

    Develop エディションを維持するため ``EVOREF_EDITION`` /
    ``VITE_EVOREF_EDITION`` を子へ継承させる。

    Returns:
        起動した supervisor の PID。非対応 OS や失敗時は ``None``。
    """
    env = {
        **os.environ,
        "EVOREF_EDITION": "develop",
        "VITE_EVOREF_EDITION": "develop",
    }

    if sys.platform == "win32":
        ctl = project_root / "scripts" / "evoref-ctl.bat"
        creationflags = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )
        try:
            # start-core = llama + backend のみ (frontend:5173 は生かしたまま)。
            # フル "start" だと生きている vite に対し 2 つ目が起動し孤児化する。
            proc = subprocess.Popen(
                ["cmd", "/c", str(ctl), "start-core"],
                cwd=str(project_root),
                env=env,
                creationflags=creationflags,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return proc.pid
        except (OSError, subprocess.SubprocessError) as e:
            print(f"[reset] ERROR: failed to restart via evoref-ctl.bat: {e}")
            return None

    # Unix: evoref serve を新セッションで起動 (開発時の保険)。
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "backend.free.cli.main", "serve"],
            cwd=str(project_root),
            env=env,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc.pid
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[reset] ERROR: failed to restart services: {e}")
        return None


# ────────────────────────────────────────────
# entrypoint
# ────────────────────────────────────────────

def run(
    project_root: Path,
    *,
    initial_delay: float = 2.0,
    do_restart: bool = True,
) -> int:
    """停止 → wipe → (再起動) の全工程を実行する。

    ``do_restart=False`` (初期化ボタンの「全停止して終了」) のときは frontend も
    含めて全サービスを停止し、再起動しない。
    """
    # backend が 202 を返してブラウザへ届くまでの猶予 (親 backend を kill する前)。
    if initial_delay > 0:
        time.sleep(initial_delay)

    shutdown = not do_restart
    label = (
        "all services (backend + llama-server + frontend)"
        if shutdown else "services (backend + llama-server)"
    )
    print(f"[reset] stopping {label}...")
    stop_result = stop_services(project_root, stop_frontend=shutdown)
    print(f"[reset] stop result: {stop_result}")
    if not stop_result["ports_freed"]:
        # ポートが解放されきらなくても wipe は試みる (best-effort)。
        print("[reset] WARNING: some ports still occupied; attempting wipe anyway")

    local_root = project_root / "local"
    print(f"[reset] wiping {local_root} to skeleton...")
    wipe_result = wipe_local_to_skeleton(local_root)
    print(f"[reset] wipe result: {wipe_result}")

    if do_restart:
        print("[reset] restarting services...")
        pid = restart_services(project_root)
        print(f"[reset] restart supervisor pid: {pid}")

    print("[reset] done.")
    return 0 if not wipe_result["errors"] else 1


def main(argv: list[str] | None = None) -> int:
    # pythonw.exe で起動された場合 stdout/stderr が None で print() が失敗する。
    # ウィンドウを残さず終了するため pythonw で起動するので、進捗ログは temp の
    # ファイルへ差し替える (local/logs は wipe 対象なので使わない)。
    if sys.stdout is None or sys.stderr is None:
        import tempfile

        try:
            _logf = open(
                Path(tempfile.gettempdir()) / "evoref_reset.log",
                "a", encoding="utf-8",
            )
            sys.stdout = _logf
            sys.stderr = _logf
        except OSError:
            pass

    parser = argparse.ArgumentParser(
        description="Reset local/ data to fresh setup.bat skeleton (Develop only)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait before stopping services (lets HTTP 202 flush)",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Wipe only; do not restart services",
    )
    args = parser.parse_args(argv)

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parent.parent
    )
    return run(
        project_root,
        initial_delay=args.delay,
        do_restart=not args.no_restart,
    )


if __name__ == "__main__":
    raise SystemExit(main())
