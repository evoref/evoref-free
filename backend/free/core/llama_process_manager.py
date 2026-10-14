"""llama-server プロセスマネージャ (backend 発の起動・停止の唯一の所有者)

base / embedding / rerank (リランカー、c_16 §7.2.1) の llama-server プロセスを
単一のレジストリで管理する。backend のプロセス内から llama-server を起こす・止める経路
(`/api/model/process/*`、その互換層の `/api/servers/*`、モード切替、`migrate_component`
の再起動) はすべてここを通る (c_06 §2.6)。

設計上の制約:
    - **外部起動との共存**: `evoref serve` / `evoref-ctl` (`scripts/launch_llama.py`) が
      起こした llama-server はレジストリに無い (adopt はできない — 他プロセスの子の
      Popen を握れないため)。起動前に設定のポートを見て、占有者がいれば **二重に起こさず**
      :class:`ProcessPortBusyError` を上げる。``force`` のときだけ占有者が llama-server
      なら kill して起こし直す (別アプリなら kill しない)。`evoref serve` が起こした
      llama-server も同じ扱い — serve は backend の終了だけを致命的とみなし、llama の
      起こし直しはこの manager に任せる (c_06 §2.7)。起こし直した llama-server は backend の
      子になり、backend の終了 (lifespan の :meth:`LlamaProcessManager.shutdown_all`) で止まる。
    - **ヘルスチェック**: `httpx` で `GET /health` をポーリングし、最大
      `health_timeout` 秒待機する。起動前にポートが空いていることを確かめるので、
      旧サーバの 200 を拾うことは無い。
    - **ブロッキング**: start / stop / restart は同期で数分ブロックしうる。async からは
      executor (``run_in_executor_with_context``) で呼ぶ。操作は 1 本の RLock で直列化する。
"""

from __future__ import annotations

import subprocess
import threading
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

#: component ⇔ サーバ名 (``/api/servers`` / ``evoref servers`` / stderr ログの名前) の写像。
#: 名前の揺れ (``embedding`` / ``embed``) を変換するのはここだけ。
SERVER_NAMES: dict[str, str] = {"base": "base", "embedding": "embed", "rerank": "rerank"}

#: 停止の後にポートが空くまで待つ上限 (秒)
PORT_RELEASE_TIMEOUT_SEC = 10.0

#: llama stderr ログのローテーション閾値 (bytes)。develop JSONL の ``max_log_mb`` と同じ 5MB。
_STDERR_LOG_MAX_BYTES = 5 * 1024 * 1024


def component_of(server_name: str) -> str:
    """サーバ名 (``base`` / ``embed`` / ``rerank``) → component 名。"""
    for component, name in SERVER_NAMES.items():
        if name == server_name:
            return component
    raise ProcessManagerError(f"Unknown server name: {server_name}")


def server_name_of(component: str) -> str:
    """component 名 → サーバ名 (``embedding`` → ``embed``)。"""
    try:
        return SERVER_NAMES[component]
    except KeyError:
        raise ProcessManagerError(f"Unknown component: {component}") from None


class ProcessManagerError(Exception):
    """プロセスマネージャ汎用エラー"""


class ProcessNotManagedError(ProcessManagerError):
    """指定コンポーネントが manager に登録されていない (外部起動も居ない)"""


class ProcessAlreadyRunningError(ProcessManagerError):
    """このマネージャが起こしたプロセスが生きている (``force`` なしの start)"""


class ProcessNotConfiguredError(ProcessManagerError):
    """起動コマンドを組めない (モデル未設定・rerank が off / 未テスト等)"""


class ProcessStartTimeoutError(ProcessManagerError):
    """起動したが health が規定時間内に 200 にならなかった"""


class ProcessPortBusyError(ProcessManagerError):
    """設定のポートを外部のプロセスが占有している (外部の llama-server か、別アプリ)"""

    def __init__(
        self, message: str, *, component: str, port: int,
        pid: int | None = None, process_name: str = "", is_llama: bool = False,
    ) -> None:
        super().__init__(message)
        self.component = component
        self.port = port
        self.pid = pid
        self.process_name = process_name
        self.is_llama = is_llama


@dataclass
class ProcessEntry:
    component: str
    proc: subprocess.Popen
    host: str
    port: int


@dataclass(frozen=True)
class StopResult:
    """:meth:`LlamaProcessManager.stop_component` の結果 (``how`` は ``managed`` / ``external``)。"""

    component: str
    how: str
    pid: int


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
    """``scripts/launch_llama.py`` を読み込む。

    backend の実行時は ``scripts`` が import path 上にある (server_control / モード切替も
    静的 import している) のでそれを使う — 起動スクリプトの解決のキャッシュを共有する。
    無い配布形態では ``project_root`` からファイルを読む。
    """
    try:
        import scripts.launch_llama as mod

        return mod
    except ImportError:
        pass
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


def _build_cmd(
    component: str, cfg: dict, project_root: Path, *, model_override: str | None = None,
) -> list[str] | None:
    """`scripts/launch_llama.py` の build_*_cmd を流用

    環境調整の ``auto`` の項目 (c_16 §7.2.3) の見積りを許すのは base だけ (この管理が base を起こすのは
    base が止まっているとき)。埋め込み / リランカーは base が載ったままなので保存値を読むだけにする。
    """
    from backend.free.core.tuning.resolve import launch_scope

    with launch_scope(component == "base"):
        return _compose_cmd(component, cfg, project_root, model_override=model_override)


def _compose_cmd(
    component: str, cfg: dict, project_root: Path, *, model_override: str | None = None,
) -> list[str] | None:
    """:func:`_build_cmd` の本体。"""
    mod = _load_launch_module(project_root)
    # 起動スクリプトは解決結果 (slots / -ngl / 環境調整の値) を config の ``__*`` キーに覚える。
    # backend の config は長く生きるので、そのままだと再起動が前の解決を使い回し、
    # 予約した環境調整の再計算 (c_16 §7.2.3) が効かない。起動ごとに覚え直させる。
    cfg = {k: v for k, v in cfg.items() if not (isinstance(k, str) and k.startswith("__"))}

    if component == "base":
        return mod.build_llama_cmd(cfg, project_root, model_override=model_override)
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


# ── ポートの占有者 (外部起動との共存) ──


def _port_occupant(port: int):
    """``port`` で LISTEN しているプロセス (``PortOccupant``)。空いていれば ``None``。"""
    from backend.free.cli.pid_manager import find_port_occupant

    return find_port_occupant(port)


def _is_llama(occupant) -> bool:
    from backend.free.cli.pid_manager import IMAGE_LLAMA, occupant_image_matches

    return occupant_image_matches(occupant, IMAGE_LLAMA)


def _kill_llama_occupant(occupant) -> bool:
    """ポートの占有者が llama-server なら kill して ``True``。別のイメージなら何もしない。"""
    from backend.free.cli.pid_manager import IMAGE_LLAMA, kill_port_occupants

    return bool(kill_port_occupants([occupant], expected={occupant.port: IMAGE_LLAMA}))


def _wait_port_released(port: int, timeout: float = PORT_RELEASE_TIMEOUT_SEC) -> bool:
    """``port`` が LISTENING でなくなるまで待つ (停止後の再 spawn 前ガード)。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_occupant(port) is None:
            return True
        time.sleep(0.5)
    return _port_occupant(port) is None


def _busy_error(component: str, port: int, occupant, *, is_llama: bool) -> ProcessPortBusyError:
    pid = getattr(occupant, "pid", None)
    name = getattr(occupant, "process_name", "") or ""
    if occupant is None:
        message = f"port {port} did not become free"
    elif not is_llama:
        message = (
            f"port {port} is held by a non-llama-server process "
            f"(pid={pid}, name={name or 'unknown'}); not killing it"
        )
    else:
        message = (
            f"port {port} is already used by an external llama-server (pid={pid}); "
            "use force to replace it"
        )
    return ProcessPortBusyError(
        message, component=component, port=port, pid=pid, process_name=name, is_llama=is_llama,
    )


# ── stderr ログ ──


def _open_stderr_log(project_root: Path, component: str):
    """``<data_root>/logs/llama-<サーバ名>.stderr.log`` を追記で開く (CLI の service_manager と同じ場所)。

    起動失敗の原因 (port 競合 / モデル不在 / Vulkan 初期化失敗 等) をログに残す。
    上限を超えたら 1 世代だけ ``.1`` へ退避してから開き直す。開けなければ ``None`` (捨てる)。
    """
    from backend.config import resolve_data_path

    try:
        log_dir = resolve_data_path("logs_dir", project_root)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"llama-{server_name_of(component)}.stderr.log"
        _rotate_if_large(path)
        return open(path, "ab")
    except OSError as exc:
        logger.warning("Failed to open the stderr log for %s: %s", component, exc)
        return None


def _rotate_if_large(path: Path) -> None:
    """``path`` が閾値を超えていたら ``.1`` へ退避する (1 世代のみ保持)。"""
    try:
        if not path.exists() or path.stat().st_size < _STDERR_LOG_MAX_BYTES:
            return
        backup = path.with_suffix(path.suffix + ".1")
        backup.unlink(missing_ok=True)
        path.rename(backup)
    except OSError as exc:
        logger.warning("Failed to rotate %s: %s", path, exc)


class LlamaProcessManager:
    """base / embedding / rerank の llama-server プロセスのライフサイクル管理"""

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
        self._lock = threading.RLock()

    # ── 状態照会 ──

    def is_managed(self, component: str) -> bool:
        return component in self._procs

    def list_managed(self) -> list[str]:
        return list(self._procs.keys())

    def get_entry(self, component: str) -> ProcessEntry | None:
        return self._procs.get(component)

    def status(self, cfg: dict) -> dict:
        """component ごとの状態 (管理下なら PID / 生死、管理外ならポートの外部占有者)。ブロッキング。"""
        components: dict[str, dict] = {}
        for component in PROCESS_COMPONENTS:
            host, port = _resolve_endpoint(component, cfg)
            entry = self._procs.get(component)
            if entry is not None:
                components[component] = {
                    "managed": True, "host": entry.host, "port": entry.port,
                    "pid": entry.proc.pid, "alive": entry.proc.poll() is None,
                }
                continue
            occupant = _port_occupant(port)
            components[component] = {
                "managed": False, "host": host, "port": port,
                "external_pid": occupant.pid if occupant is not None else None,
                "external_name": (occupant.process_name or None) if occupant is not None else None,
            }
        return {"managed": self.list_managed(), "components": components}

    # ── ライフサイクル ──

    def start(
        self, component: str, cfg: dict, *, force: bool = False, model_override: str | None = None,
    ) -> ProcessEntry:
        """指定コンポーネントを spawn してヘルスチェック完了まで待機

        設定のポートに外部のプロセスが居れば二重に起こさない (:class:`ProcessPortBusyError`)。
        ``force`` は自分が起こしたプロセス、または外部の llama-server を止めてから起こし直す。
        ``model_override`` は base のモデルの差し替え (モード切替)。
        """
        if component not in PROCESS_COMPONENTS:
            raise ProcessManagerError(f"Unknown component: {component}")
        with self._lock:
            host, port = _resolve_endpoint(component, cfg)
            if self._clear_slot(component, port, force=force) and not _wait_port_released(port):
                raise _busy_error(component, port, _port_occupant(port), is_llama=True)
            return self._launch(component, cfg, host, port, model_override)

    def _clear_slot(self, component: str, port: int, *, force: bool) -> bool:
        """起動前に ``port`` を空ける。何かを止めたら ``True``。止められなければ例外。"""
        stopped = False
        entry = self._procs.get(component)
        if entry is not None:
            if entry.proc.poll() is None and not force:
                raise ProcessAlreadyRunningError(
                    f"{component} is already running (pid={entry.proc.pid})",
                )
            self.stop(component)  # 死んでいれば登録を外すだけ
            stopped = True
        occupant = _port_occupant(port)
        if occupant is None:
            return stopped
        is_llama = _is_llama(occupant)
        if not force or not is_llama:
            raise _busy_error(component, port, occupant, is_llama=is_llama)
        logger.info(
            "Replacing the external %s on port %d (pid=%d)", component, port, occupant.pid,
        )
        if not _kill_llama_occupant(occupant):
            raise _busy_error(component, port, occupant, is_llama=False)
        return True

    def _launch(
        self, component: str, cfg: dict, host: str, port: int, model_override: str | None,
    ) -> ProcessEntry:
        """コマンドを組んで起動し health を待つ (ポートは空いている前提)。"""
        cmd = _build_cmd(component, cfg, self.project_root, model_override=model_override)
        if cmd is None:
            raise ProcessNotConfiguredError(
                f"{component} is not configured (build_cmd returned None)",
            )
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
            raise ProcessStartTimeoutError(
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
        stderr_f = _open_stderr_log(self.project_root, component)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.DEVNULL,
                stderr=stderr_f if stderr_f is not None else subprocess.DEVNULL,
            )
        except OSError as e:
            raise ProcessManagerError(f"{component} failed to spawn: {e}") from e
        finally:
            # 子は複製したハンドルで書き続ける。親の側は閉じてよい
            if stderr_f is not None:
                stderr_f.close()
        self._procs[component] = ProcessEntry(component, proc, host, port)
        return _wait_for_health(host, port, timeout)

    def stop(self, component: str) -> None:
        """このマネージャが起こした指定コンポーネントを終了する"""
        with self._lock:
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

    def stop_component(self, component: str, cfg: dict) -> StopResult:
        """管理下ならそれを、無ければポートを占有する外部の llama-server を止める

        何も居なければ :class:`ProcessNotManagedError`、占有者が別アプリなら
        :class:`ProcessPortBusyError`。
        """
        if component not in PROCESS_COMPONENTS:
            raise ProcessManagerError(f"Unknown component: {component}")
        with self._lock:
            entry = self._procs.get(component)
            if entry is not None:
                pid = entry.proc.pid
                self.stop(component)
                return StopResult(component, "managed", pid)
            _host, port = _resolve_endpoint(component, cfg)
            occupant = _port_occupant(port)
            if occupant is None:
                raise ProcessNotManagedError(
                    f"{component} is not running (nothing listens on port {port})",
                )
            if not _is_llama(occupant):
                raise _busy_error(component, port, occupant, is_llama=False)
            logger.info(
                "Stopping the external %s on port %d (pid=%d)", component, port, occupant.pid,
            )
            if not _kill_llama_occupant(occupant):
                raise _busy_error(component, port, occupant, is_llama=False)
            return StopResult(component, "external", occupant.pid)

    def restart(
        self, component: str, cfg: dict, *, model_override: str | None = None,
    ) -> ProcessEntry:
        """指定コンポーネントを再起動する (外部起動の llama-server は止めてこのマネージャで起こし直す)

        何も動いていなければ :class:`ProcessNotManagedError` (start を使う)。
        """
        if component not in PROCESS_COMPONENTS:
            raise ProcessManagerError(f"Unknown component: {component}")
        with self._lock:
            if component not in self._procs:
                _host, port = _resolve_endpoint(component, cfg)
                if _port_occupant(port) is None:
                    raise ProcessNotManagedError(
                        f"{component} is not running (nothing listens on port {port}); use start",
                    )
            return self.start(component, cfg, force=True, model_override=model_override)

    def shutdown_all(self) -> None:
        """登録中の全プロセスを順次停止"""
        for component in list(self._procs.keys()):
            try:
                self.stop(component)
            except Exception as e:
                logger.error("Failed to stop %s: %s", component, e)
