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
は ``evoref reset --stop-services`` も同じ目的で使っている純粋ヘルパーで、アプリ context
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
    IMAGE_LLAMA,
    expected_images_by_port,
    find_port_occupants,
    kill_port_occupants,
)

#: evoref-ctl.bat が立てるウィンドウタイトル。
_WINDOW_TITLES = ("llama-server", "evoref-backend", "evoref-frontend")
_LLAMA_WINDOW_TITLE = "llama-server"
#: config.yaml が読めないときの停止対象 (既定の 8000 / 8080 / 8082)。
_DEFAULT_CONFIG: dict = {"embedding": {"backend": "llama-cpp", "llama_port": 8082}}


def _load_expected(project_root: Path, *, include_frontend: bool) -> dict[int, tuple[str, ...]]:
    """停止対象ポート → そのポートで kill してよいイメージ名 (config.yaml から)。"""
    cfg = _DEFAULT_CONFIG
    try:
        import yaml

        with open(project_root / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[stop] WARNING: failed to read config ports ({e}); using defaults")
    return expected_images_by_port(cfg, include_frontend=include_frontend)


def _taskkill_titles(titles: tuple[str, ...]) -> None:
    """従来経路 (ウィンドウタイトル) の kill も併用する。

    イメージ名での kill (``/IM llama-server.exe``) はマシン上の全 llama-server を
    巻き添えにするので使わない。
    """
    if sys.platform != "win32":
        return
    for title in titles:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/FI", f"WINDOWTITLE eq {title}"],
                capture_output=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            pass


def stop(
    project_root: Path,
    *,
    include_frontend: bool,
    wait_timeout: float,
    llama_only: bool = False,
) -> int:
    expected = _load_expected(project_root, include_frontend=include_frontend)
    if llama_only:
        expected = {p: imgs for p, imgs in expected.items() if imgs == IMAGE_LLAMA}
    ports = list(expected)
    print(f"[stop] target ports: {ports}")

    _taskkill_titles((_LLAMA_WINDOW_TITLE,) if llama_only else _WINDOW_TITLES)

    own_pid = os.getpid()
    occupants = [o for o in find_port_occupants(ports) if o.pid != own_pid]
    foreign = []
    if occupants:
        print(f"[stop] port occupants still alive: {[o.summary for o in occupants]}")
        killed = kill_port_occupants(occupants, expected=expected)
        print(f"[stop] killed: {[o.summary for o in killed]}")
        foreign = [o for o in occupants if o not in killed]
    if foreign:
        print(
            "[stop] FAILED: not stopping (not an evoref process, image name does not "
            f"match the port's role): {[o.summary for o in foreign]}",
        )
        foreign_ports = {o.port for o in foreign}
        ports = [p for p in ports if p not in foreign_ports]

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
    if foreign:
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
    parser.add_argument(
        "--llama-only", action="store_true",
        help="Stop only the llama-server processes (evoref-ctl start uses this before spawning)",
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
        llama_only=args.llama_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
