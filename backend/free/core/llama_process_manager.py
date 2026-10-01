"""llama-server プロセスマネージャ

base / embedding / rerank (リランカー、c_16 §7.2.1) の llama-server プロセスを
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

from backend.free.llm._base_client import model_path_from_cmd, resolve_health_wait
from backend.log_config import get_logger

logger = get_logger("core.llama_process_manager")


# コンポーネント名 (embedding は L1 と一致)。rerank は保存済みの自己テストの配置で起こす
# (測り直さない。測るのは起動スクリプト)。
PROCESS_COMPONENTS: tuple[str, ...] = ("base", "embedding", "rerank")


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
    if component == "embedding":
        emb = cfg.get("embedding", {}) or {}
        return (
            emb.get("llama_host", "localhost"),
            int(emb.get("llama_port", 8082)),
        )
    if component == "rerank":
        rr = (cfg.get("rag") or {}).get("rerank") or {}
        return "127.0.0.1", int(rr.get("port", 8083))
    raise ProcessManagerError(f"Unknown component: {component}")


def _load_launch_module(project_root: Path):
    """``scripts/launch_llama.py`` を読み込む (scripts は import path 上に無いことがあるため動的 import)。"""
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
    return mod


def _build_cmd(component: str, cfg: dict, project_root: Path) -> list[str] | None:
    """`scripts/launch_llama.py` の build_*_cmd を流用

    環境調整の ``auto`` の項目 (c_16 §7.2.3) の見積りを許すのは base だけ (この管理が base を起こすのは
    base が止まっているとき)。埋め込み / リランカーは base が載ったままなので保存値を読むだけにする。
    """
    from backend.free.core.tuning.resolve import launch_scope

    with launch_scope(component == "base"):
        return _compose_cmd(component, cfg, project_root)


def _compose_cmd(component: str, cfg: dict, project_root: Path) -> list[str] | None:
    """:func:`_build_cmd` の本体。"""
    mod = _load_launch_module(project_root)

    if component == "base":
        # 学習済みアダプタ (Pro) はモード切替経路 (backend.free.api.config.mode)
        # と同じ取得口から、現在のモードの分を当てる (Free では無し)。
        from backend.config import get_path_resolver
        from backend.free.core.launch_adapters import adapters_for_launch

        try:
            mode = get_path_resolver().active_mode
        except RuntimeError:
            # config 未ロード (単体テスト等)。起動時の既定モードへ倒す。
            mode = "chat"
        adapters = adapters_for_launch(cfg, project_root, mode)
        return mod.build_llama_cmd(
            cfg, project_root,
            lora_override=adapters.lora,
            control_vector_override=adapters.control_vector,
        )
    if component == "embedding":
        return mod.build_embed_cmd(cfg, project_root)
    if component == "rerank":
        # 自己テストで有効と記録された配置だけで起こす (無効・未テストなら起動しない)
        placement = mod.saved_rerank_placement(cfg, project_root)
        if placement is None:
            return None
        return mod.build_rerank_cmd(cfg, project_root, placement)
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
        health_timeout: int | str | None = "auto",
        stop_timeout: int = 10,
    ):
        self.project_root = project_root
        #: 整数は明示値 (そのまま)、``auto`` / ``None`` は起動するモデルの GGUF サイズ連動 (c_16 §7.2.3)
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
        model_path = model_path_from_cmd(cmd)
        if model_path and not Path(model_path).is_absolute():
            model_path = str(self.project_root / model_path)
        health_timeout = resolve_health_wait(self.health_timeout, model_path, label=component)

        # 埋め込み / rerank が保存済みの GPU 配置 (auto) なら、起動しなければ CPU で 1 回だけ起こし直す
        # (保存結果は書き換えない、c_16 §7.2.1 / §7.2.2)
        if component in ("embedding", "rerank"):
            try:
                mod = _load_launch_module(self.project_root)
            except ProcessManagerError:
                mod = None  # launch_llama.py が読めない: CPU への起こし直しを諦めるだけ
            fallback = None
            if mod is not None:
                fallback = (
                    mod.embed_cpu_fallback_cmd(cfg, cmd, self.project_root) if component == "embedding"
                    else mod.rerank_cpu_fallback_cmd(cfg, self.project_root, cmd)
                )
            if fallback is not None:
                first = min(health_timeout, resolve_health_wait(
                    "auto", model_path, label=f"{component} GPU start",
                    floor=int(
                        mod.EMBED_GPU_START_TIMEOUT_SEC if component == "embedding"
                        else mod.RERANK_GPU_START_TIMEOUT_SEC,
                    ),
                ))
                if self._spawn_and_wait(component, cmd, host, port, first):
                    logger.info("%s is ready at %s:%s", component, host, port)
                    return self._procs[component]
                logger.warning(
                    "%s on GPU did not become healthy within %ss; restarting it on CPU", component, first,
                )
                self.stop(component)
                cmd = fallback

        if not self._spawn_and_wait(component, cmd, host, port, health_timeout):
            logger.error(
                "%s health check timed out at %s:%s", component, host, port,
            )
            self.stop(component)
            raise ProcessManagerError(
                f"{component} failed to become healthy within "
                f"{health_timeout}s",
            )
        logger.info("%s is ready at %s:%s", component, host, port)
        return self._procs[component]

    def _spawn_and_wait(
        self, component: str, cmd: list[str], host: str, port: int, timeout: int,
    ) -> bool:
        """``cmd`` を起動して登録し、health を ``timeout`` 秒待つ。"""
        logger.info("Starting %s: %s", component, " ".join(cmd))
        proc = subprocess.Popen(cmd)
        self._procs[component] = ProcessEntry(component, proc, host, port)
        return _wait_for_health(host, port, timeout)

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

    def shutdown_all(self) -> None:
        """登録中の全プロセスを順次停止"""
        for component in list(self._procs.keys()):
            try:
                self.stop(component)
            except Exception as e:
                logger.error("Failed to stop %s: %s", component, e)
