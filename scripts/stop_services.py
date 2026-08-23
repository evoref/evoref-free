"""evoref のサービスを停止し、**本当に止まったかを検証して** 報告する。

``scripts/evoref-ctl.bat stop`` は ``taskkill /fi "WINDOWTITLE eq ..."`` だけで
backend / frontend を止めようとしていた。ウィンドウタイトルは
``evoref-ctl.bat`` が ``start "<title>" ...`` で立てたときにしか付かないため、
別経路 (``evoref serve`` / ``uvicorn`` 直起動 / ラッパ経由の起動) で立ち上がった
プロセスには一致しない。しかも結果を ``>nul 2>&1`` で捨てて無条件に
``FastAPI backend stopped`` と表示していたので、**止まっていないのに止まったと
報告する**。

実インシデント (2026-08-23): コード修正後の再測定のために stop → start した
ところ、``stop`` は 3 行とも "stopped" と表示したが 8000 番を掴んでいたのは
2 時間 44 分前に起動した旧 backend のままだった (llama-server だけは
``/im llama-server.exe`` でも殺しているので入れ替わっていた)。旧コードのまま
計測を続けるところだった。

停止の判定はウィンドウタイトルではなく **ポートの占有** で行う。``pid_manager``
は ``reset_local_data.py`` が同じ目的で使っている純粋ヘルパーで、アプリ context
に依存しない。

ログは英語固定 (リポジトリ規約)。standalone のため print で stdout へ出す。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.free.cli.pid_manager import (  # noqa: E402
    collect_configured_ports,
    find_port_occupants,
    kill_port_occupants,
)

#: evoref-ctl.bat が立てるウィンドウタイトル。
_WINDOW_TITLES = ("llama-server", "evoref-backend", "evoref-frontend")
_FRONTEND_PORT = 5173
_DEFAULT_PORTS = [8000, 8080, 8082]


def _load_ports(project_root: Path, *, include_frontend: bool) -> list[int]:
    ports = list(_DEFAULT_PORTS)
    try:
        import yaml

        with open(project_root / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        configured = collect_configured_ports(cfg)
        if configured:
            ports = list(configured)
    except Exception as e:
        print(f"[stop] WARNING: failed to read config ports ({e}); using defaults")
    if include_frontend:
        if _FRONTEND_PORT not in ports:
            ports.append(_FRONTEND_PORT)
    else:
        ports = [p for p in ports if p != _FRONTEND_PORT]
    return ports


def _taskkill_titles() -> None:
    """従来経路 (ウィンドウタイトル) の kill も併用する。"""
    if sys.platform != "win32":
        return
    for title in _WINDOW_TITLES:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/FI", f"WINDOWTITLE eq {title}"],
                capture_output=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "llama-server.exe"],
            capture_output=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def stop(project_root: Path, *, include_frontend: bool, wait_timeout: float) -> int:
    ports = _load_ports(project_root, include_frontend=include_frontend)
    print(f"[stop] target ports: {ports}")

    _taskkill_titles()

    own_pid = os.getpid()
    occupants = [o for o in find_port_occupants(ports) if o.pid != own_pid]
    if occupants:
        print(f"[stop] port occupants still alive: {[o.summary for o in occupants]}")
        killed = kill_port_occupants(occupants)
        print(f"[stop] killed: {[o.summary for o in killed]}")

    deadline = time.monotonic() + wait_timeout
    remaining = find_port_occupants(ports)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.5)
        remaining = [o for o in find_port_occupants(ports) if o.pid != own_pid]

    if remaining:
        print(
            "[stop] FAILED: still listening after "
            f"{wait_timeout:.0f}s: {[o.summary for o in remaining]}",
        )
        return 1
    print("[stop] verified: no service is listening on the target ports")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stop evoref services and verify the ports are free",
    )
    parser.add_argument(
        "--keep-frontend", action="store_true",
        help="Leave the SvelteKit dev server (port 5173) running",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args(argv)

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parent.parent
    )
    return stop(
        project_root,
        include_frontend=not args.keep_frontend,
        wait_timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
