"""llama-server プロセスマネージャ

base / assist / embedding の 3 種 llama-server プロセスを
単一のレジストリで管理する。`migrate_component` API がモデル切替成功後に
`restart()` を呼び出し、無停止 (ユーザーが手動シェル操作不要) でモデルを
切り替える。

設計上の制約:
    - **opt-in**: `process_manager.enabled: true` 設定時のみ lifespan で起動する。
      false の場合は `scripts/launch_llama.py` での外部起動と従来通り
      共存する (manager は何も spawn しない)。
    - **管理対象のみ操作可**: 外部起動された llama-server は manager の
      レジストリに含まれないため `restart()` 不可。その場合は API が
      "manually restart" を案内する。
    - **ヘルスチェック**: `httpx` で `GET /health` をポーリングし、最大
      `health_timeout` 秒待機する。
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.log_config import get_logger

logger = get_logger("core.llama_process_manager")


# 3 種のコンポーネント名 (assist/embedding は L1 と一致)
PROCESS_COMPONENTS: tuple[str, ...] = ("base", "assist", "embedding")


class ProcessManagerError(Exception):
    """プロセスマネージャ汎用エラー"""


class ProcessNotManagedError(ProcessManagerError):
    """指定コンポーネントが manager に登録されていない"""


@dataclass
class ProcessEntry:
    component: str
    proc: subprocess.Popen
    host: str
    port: int


def _resolve_endpoint(component: str, cfg: dict) -> tuple[str, int]:
    """config.yaml から component のホスト/ポートを解決"""
    if component == "base":
        lc = cfg.get("llama", {}) or {}
        return lc.get("host", "127.0.0.1"), int(lc.get("port", 8080))
    if component == "assist":
        lcfg = (cfg.get("assist_model", {}) or {}).get("local", {}) or {}
        return lcfg.get("host", "127.0.0.1"), int(lcfg.get("port", 8081))
    if component == "embedding":
        emb = cfg.get("embedding", {}) or {}
        return (
            emb.get("llama_host", "localhost"),
            int(emb.get("llama_port", 8082)),
        )
    raise ProcessManagerError(f"Unknown component: {component}")


def _build_cmd(component: str, cfg: dict, project_root: Path) -> list[str] | None:
    """`scripts/launch_llama.py` の build_*_cmd を流用"""
    # scripts は import path 上に無いことがあるため動的 import
    import importlib.util

    launch_py = project_root / "scripts" / "launch_llama.py"
    if not launch_py.exists():
        raise ProcessManagerError(
            f"launch_llama.py not found: {launch_py}",
        )
    spec = importlib.util.spec_from_file_location("_launch_llama", launch_py)
    if spec is None or spec.loader is None:
        raise ProcessManagerError("Failed to load launch_llama.py spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if component == "base":
        return mod.build_llama_cmd(cfg, project_root)
    if component == "assist":
        return mod.build_assist_cmd(cfg, project_root)
    if component == "embedding":
        return mod.build_embed_cmd(cfg, project_root)
    raise ProcessManagerError(f"Unknown component: {component}")


def _wait_for_health(host: str, port: int, timeout: int) -> bool:
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(1.0)
    return False


class LlamaProcessManager:
    """4 種 llama-server プロセスのライフサイクル管理"""

    def __init__(
        self,
        project_root: Path,
        *,
        health_timeout: int = 60,
        stop_timeout: int = 10,
    ):
        self.project_root = project_root
        self.health_timeout = health_timeout
        self.stop_timeout = stop_timeout
        self._procs: dict[str, ProcessEntry] = {}

    # ── 状態照会 ──

    def is_managed(self, component: str) -> bool:
        return component in self._procs

    def list_managed(self) -> list[str]:
        return list(self._procs.keys())

    def get_entry(self, component: str) -> ProcessEntry | None:
        return self._procs.get(component)

    # ── ライフサイクル ──

    def start(self, component: str, cfg: dict) -> ProcessEntry:
        """指定コンポーネントを spawn してヘルスチェック完了まで待機"""
        if component not in PROCESS_COMPONENTS:
            raise ProcessManagerError(f"Unknown component: {component}")
        if component in self._procs:
            raise ProcessManagerError(f"{component} is already running")

        cmd = _build_cmd(component, cfg, self.project_root)
        if cmd is None:
            raise ProcessManagerError(
                f"{component} is not configured (build_cmd returned None)",
            )
        host, port = _resolve_endpoint(component, cfg)

        logger.info("Starting %s: %s", component, " ".join(cmd))
        proc = subprocess.Popen(cmd)
        entry = ProcessEntry(component, proc, host, port)
        self._procs[component] = entry

        if not _wait_for_health(host, port, self.health_timeout):
            logger.error(
                "%s health check timed out at %s:%s", component, host, port,
            )
            self.stop(component)
            raise ProcessManagerError(
                f"{component} failed to become healthy within "
                f"{self.health_timeout}s",
            )
        logger.info("%s is ready at %s:%s", component, host, port)
        return entry

    def stop(self, component: str) -> None:
        """指定コンポーネントを終了する"""
        entry = self._procs.pop(component, None)
        if entry is None:
            raise ProcessNotManagedError(
                f"{component} is not managed by this manager",
            )
        proc = entry.proc
        if proc.poll() is None:
            logger.info("Terminating %s (pid=%s)", component, proc.pid)
            try:
                proc.terminate()
                proc.wait(timeout=self.stop_timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "%s did not terminate within %ss; killing",
                    component, self.stop_timeout,
                )
                proc.kill()
                proc.wait()
        logger.info("%s stopped", component)

    def restart(self, component: str, cfg: dict) -> ProcessEntry:
        """指定コンポーネントを再起動する"""
        if component not in self._procs:
            raise ProcessNotManagedError(
                f"{component} is not managed by this manager. "
                "Cannot restart externally-launched processes.",
            )
        self.stop(component)
        return self.start(component, cfg)

    def health(self, component: str) -> bool:
        """軽量ヘルスチェック (1 回のみ)"""
        entry = self._procs.get(component)
        if entry is None:
            return False
        try:
            resp = httpx.get(
                f"http://{entry.host}:{entry.port}/health", timeout=2.0,
            )
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def shutdown_all(self) -> None:
        """登録中の全プロセスを順次停止"""
        for component in list(self._procs.keys()):
            try:
                self.stop(component)
            except Exception as e:
                logger.error("Failed to stop %s: %s", component, e)
