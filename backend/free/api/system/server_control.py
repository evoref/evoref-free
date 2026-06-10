"""サーバー管理 API: llama-server プロセスの起動・停止

フロントエンドからベース/アシスト/埋め込みの
llama-server プロセスを個別に起動・停止できるようにする。
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.config import get_config
from backend.log_config import get_logger

logger = get_logger("api.server_control")

router = APIRouter(prefix="/api/servers", tags=["server-control"])

# ────────────────────────────────────────────
# プロセス管理コンテナ（モジュールレベルシングルトン）
# ────────────────────────────────────────────

ServerName = Literal["base", "assist", "embed"]


@dataclass
class ManagedProcess:
    """管理対象の llama-server プロセス"""
    name: ServerName
    proc: subprocess.Popen
    host: str
    port: int


_managed: dict[ServerName, ManagedProcess] = {}


def _find_project_root() -> Path:
    """プロジェクトルートを探索"""
    from backend.free.cli.config_loader import _find_project_root
    return _find_project_root()


# ────────────────────────────────────────────
# ヘルスチェック
# ────────────────────────────────────────────

async def _check_health(host: str, port: int) -> bool:
    """llama-server の /health エンドポイントに問い合わせ"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://{host}:{port}/health", timeout=3.0,
            )
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


# ────────────────────────────────────────────
# プロセス起動 / 停止
# ────────────────────────────────────────────

def _kill_process_tree(pid: int) -> None:
    """プロセスツリーごと終了"""
    import os
    import signal as sig
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
            os.killpg(os.getpgid(pid), sig.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def _build_cmd(
    name: ServerName,
    cfg: dict,
    project_root: Path,
    *,
    model_override: str | None = None,
) -> tuple[list[str], str, int] | None:
    """サーバー名から起動コマンドと (host, port) を構築

    Returns:
        (cmd, host, port) or None（設定なし / モデル未指定）
    """
    from scripts.launch_llama import (
        build_llama_cmd,
        build_assist_cmd,
        build_embed_cmd,
    )

    if name == "base":
        llama_cfg = cfg.get("llama", {})
        host = llama_cfg.get("host", "127.0.0.1")
        port = llama_cfg.get("port", 8080)
        cmd = build_llama_cmd(cfg, project_root, model_override=model_override)
        return (cmd, host, port)

    if name == "assist":
        cmd = build_assist_cmd(cfg, project_root)
        if cmd is None:
            return None
        local_cfg = cfg.get("assist_model", {}).get("local", {})
        host = local_cfg.get("host", "127.0.0.1")
        port = local_cfg.get("port", 8081)
        return (cmd, host, port)

    if name == "embed":
        cmd = build_embed_cmd(cfg, project_root)
        if cmd is None:
            return None
        embed_cfg = cfg.get("embedding", {})
        host = embed_cfg.get("llama_host", "127.0.0.1")
        port = embed_cfg.get("llama_port", 8082)
        return (cmd, host, port)

    return None


def _open_stderr_log(project_root: Path, name: ServerName):
    """llama-{name}.stderr.log を開く (CLI service_manager と同じ場所に統一)

    `_spawn_server` が stderr=DEVNULL で完全にエラーを捨てていた問題に対し、
    UI 経由起動時も CLI 起動時と同じファイルへ stderr を書き出す。
    起動失敗の原因 (port 競合 / モデル不在 / Vulkan 初期化失敗 等) が
    `local/logs/llama-{name}.stderr.log` に残るようになる。
    """
    log_dir = project_root / "local" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return open(log_dir / f"llama-{name}.stderr.log", "ab")


def _spawn_server(name: ServerName, cfg: dict) -> ManagedProcess | None:
    """llama-server プロセスを起動"""
    project_root = _find_project_root()
    result = _build_cmd(name, cfg, project_root)
    if result is None:
        logger.warning("server_control: no config for %s", name)
        return None

    cmd, host, port = result
    logger.info("server_control: starting %s on %s:%d", name, host, port)
    logger.debug("server_control: cmd=%s", cmd)

    stderr_f = _open_stderr_log(project_root, name)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=stderr_f,
        )
    except (FileNotFoundError, OSError) as e:
        logger.error("server_control: failed to spawn %s: %s", name, e)
        stderr_f.close()
        return None

    managed = ManagedProcess(name=name, proc=proc, host=host, port=port)
    _managed[name] = managed
    return managed


def _spawn_server_with_override(
    name: ServerName,
    cfg: dict,
    *,
    model_override: str | None = None,
) -> ManagedProcess | None:
    """model_override 対応版の llama-server プロセス起動"""
    project_root = _find_project_root()
    result = _build_cmd(name, cfg, project_root, model_override=model_override)
    if result is None:
        logger.warning("server_control: no config for %s", name)
        return None

    cmd, host, port = result
    logger.info("server_control: starting %s on %s:%d (override=%s)", name, host, port, model_override)
    logger.debug("server_control: cmd=%s", cmd)

    stderr_f = _open_stderr_log(project_root, name)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=stderr_f,
        )
    except (FileNotFoundError, OSError) as e:
        logger.error("server_control: failed to spawn %s: %s", name, e)
        stderr_f.close()
        return None

    managed = ManagedProcess(name=name, proc=proc, host=host, port=port)
    _managed[name] = managed
    return managed


def _stop_server(name: ServerName) -> bool:
    """管理対象プロセスを停止 (このバックエンドが spawn したプロセスのみ)"""
    managed = _managed.pop(name, None)
    if managed is None:
        return False

    if managed.proc.poll() is None:
        logger.info("server_control: stopping %s (pid=%d)", name, managed.proc.pid)
        _kill_process_tree(managed.proc.pid)
        try:
            managed.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            managed.proc.kill()
    return True


def _resolve_endpoint(name: ServerName, cfg: dict) -> tuple[str, int] | None:
    """サーバー名から (host, port) を config から直接解決する (停止経路専用)

    `_build_cmd` と違い `enabled` / モデルパス有無のゲートに依存しない。
    「無効化と同時にプロセスを停止する」経路では config が既に
    `enabled=false` になっており、cmd ビルダーが ``None`` を返して
    停止対象ポートを失う。停止に cmd は不要でポートだけ要るためこちらを使う。
    """
    if name == "base":
        lc = cfg.get("llama", {}) or {}
        return lc.get("host", "127.0.0.1"), int(lc.get("port", 8080))
    if name == "assist":
        local_cfg = (cfg.get("assist_model", {}) or {}).get("local", {}) or {}
        return local_cfg.get("host", "127.0.0.1"), int(local_cfg.get("port", 8081))
    if name == "embed":
        emb = cfg.get("embedding", {}) or {}
        return emb.get("llama_host", "127.0.0.1"), int(emb.get("llama_port", 8082))
    return None


def _stop_external_server(name: ServerName, cfg: dict) -> tuple[bool, str]:
    """外部起動された llama-server をポート占有 PID から特定して停止する

    `_stop_server` が「このバックエンドが起動したプロセス」しか扱えない問題に対し、
    `python scripts/launch_llama.py --all` 等で外部起動されたプロセスにも
    UI の停止ボタンが効くようフォールバックさせる。

    ポートは `_resolve_endpoint` で config から直接引く。`_build_cmd` 経由だと
    enabled=false 等で cmd が ``None`` になりポートを失い、無効化と同時の
    停止が機能しなくなる。

    Returns:
        (success, message)
    """
    from backend.free.cli.pid_manager import find_port_occupant, kill_port_occupants

    endpoint = _resolve_endpoint(name, cfg)
    if endpoint is None:
        return (False, "not configured")

    _host, port = endpoint
    occupant = find_port_occupant(port)
    if occupant is None:
        return (False, "no process listening on port")

    logger.info(
        "server_control: stopping external %s on port %d (pid=%d, name=%s)",
        name, port, occupant.pid, occupant.process_name,
    )
    killed = kill_port_occupants([occupant])
    if not killed:
        return (False, "failed to kill port occupant")
    return (True, f"stopped external process pid={occupant.pid}")


def stop_server_process(
    name: ServerName, cfg: dict, state: AppState,
) -> tuple[bool, str]:
    """llama-server プロセスを停止する (3 つの起動経路を順に試す)

    UI の停止操作と「設定 OFF 時の自動停止」で共有する。VRAM 解放が目的なので、

        1. server_control が spawn した Popen (`_managed`)
        2. LlamaProcessManager 管理下 (`process_manager.enabled=true` 起動経路)
        3. 外部起動 (`launch_llama.py --all` 等) を port 占有 PID から停止

    のいずれかで止められれば ``success=True`` を返す。``AppState`` のクライアント
    参照クリアは呼び出し側の責務。
    """
    if _stop_server(name):
        return (True, "stopped")

    mgr = getattr(state, "llama_manager", None)
    if mgr is not None:
        # LlamaProcessManager の component 名は embed のみ "embedding"
        component = "embedding" if name == "embed" else name
        try:
            if mgr.get_entry(component) is not None:
                mgr.stop(component)
                return (True, "stopped (process manager)")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "server_control: process_manager stop failed for %s: %s", name, e,
            )

    return _stop_external_server(name, cfg)


async def start_server_process(name: ServerName, cfg: dict) -> tuple[bool, str]:
    """llama-server プロセスを起動する (既に起動済みなら no-op)

    設定 ON 時の自動起動で使う。``/health`` が通る or `_managed` に生存プロセスが
    あれば spawn せず ``(True, ...)`` を返す。落ちている場合のみ spawn して
    ヘルスチェック完了まで待機する。クライアント再接続は呼び出し側の責務。
    """
    result = _build_cmd(name, cfg, _find_project_root())
    if result is None:
        return (False, "not configured")
    _, host, port = result

    # 既に起動済み (外部起動 / 別経路) なら spawn しない
    if await _check_health(host, port):
        return (True, "already healthy")
    existing = _managed.get(name)
    if existing and existing.proc.poll() is None:
        return (True, "already running")

    managed = _spawn_server(name, cfg)
    if managed is None:
        return (False, "not configured")

    import asyncio
    from scripts.launch_llama import wait_for_health
    healthy = await asyncio.to_thread(
        wait_for_health, managed.host, managed.port, 30,
    )
    if healthy:
        return (True, "started")
    _stop_server(name)
    return (False, "health check timed out")


# ────────────────────────────────────────────
# API スキーマ
# ────────────────────────────────────────────

class ServerActionRequest(BaseModel):
    name: ServerName
    force: bool = False


class ServerActionResponse(BaseModel):
    name: str
    action: str
    success: bool
    message: str = ""


# ────────────────────────────────────────────
# エンドポイント
# ────────────────────────────────────────────

@router.post("/start", response_model=ServerActionResponse)
async def start_server(
    req: ServerActionRequest,
    state: AppState = Depends(get_app_state),
):
    """指定サーバーの llama-server プロセスを起動"""
    name = req.name
    force = req.force
    cfg = get_config()

    # 既にプロセス管理中ならチェック
    existing = _managed.get(name)
    if existing and existing.proc.poll() is None:
        if force:
            # --force: 既存プロセスを停止してから再起動
            logger.info("server_control: force-stopping %s before restart", name)
            _stop_server(name)
        else:
            return ServerActionResponse(
                name=name, action="start", success=False,
                message="already running",
            )

    # 既にポートで起動中かチェック
    result = _build_cmd(name, cfg, _find_project_root())
    if result is not None:
        _, host, port = result
        if await _check_health(host, port):
            if force:
                # --force: ポート占有プロセスを kill して再起動
                logger.info("server_control: force-killing occupant on port %d", port)
                from backend.free.cli.pid_manager import find_port_occupant, kill_port_occupants
                occ = find_port_occupant(port)
                if occ:
                    kill_port_occupants([occ])
                    import asyncio
                    await asyncio.sleep(1)
            else:
                # 外部起動済み — 遅延接続を試みる
                await _try_reconnect(name, state, cfg)
                return ServerActionResponse(
                    name=name, action="start", success=True,
                    message="already healthy, reconnected",
                )

    managed = _spawn_server(name, cfg)
    if managed is None:
        return ServerActionResponse(
            name=name, action="start", success=False,
            message="not configured",
        )

    # ヘルスチェック待機（最大30秒、ブロッキング関数をスレッドで実行）
    import asyncio
    from scripts.launch_llama import wait_for_health
    healthy = await asyncio.to_thread(
        wait_for_health, managed.host, managed.port, 30,
    )

    if healthy:
        await _try_reconnect(name, state, cfg)
        return ServerActionResponse(
            name=name, action="start", success=True,
            message="force restarted" if force else "started",
        )
    else:
        _stop_server(name)
        return ServerActionResponse(
            name=name, action="start", success=False,
            message="health check timed out",
        )


@router.post("/stop", response_model=ServerActionResponse)
async def stop_server(
    req: ServerActionRequest,
    state: AppState = Depends(get_app_state),
):
    """指定サーバーの llama-server プロセスを停止

    まず本バックエンドが管理しているプロセスを試し、無ければポート占有プロセスを
    特定して停止するフォールバックを行う (外部起動された llama-server にも対応)。
    """
    name = req.name

    if _stop_server(name):
        message = "stopped"
        success = True
    else:
        # 外部起動プロセスへのフォールバック
        cfg = get_config()
        success, message = _stop_external_server(name, cfg)

    # AppState のクライアント参照をクリア (停止に成功した場合のみ)
    if success:
        if name == "base":
            if state.local_client:
                await state.local_client.aclose()
            state.local_client = None
            if state.llm_client:
                state.llm_client.local = None
        elif name == "assist":
            if state.assist_client:
                await state.assist_client.aclose()
            state.set_assist_client(None)
        elif name == "embed":
            # embedder オブジェクトは残すが httpx クライアントだけ閉じて
            # 死んだサーバへの kept-alive socket を破棄する。次回 embed 呼び出しで
            # _get_http_client が新しい AsyncClient を作る。
            if state.embedder is not None and hasattr(state.embedder, "aclose"):
                try:
                    await state.embedder.aclose()
                except Exception as e:
                    logger.warning("server_control: failed to close embedder client: %s", e)

    return ServerActionResponse(
        name=name,
        action="stop",
        success=success,
        message=message,
    )


async def _try_reconnect(
    name: ServerName,
    state: AppState,
    cfg: dict,
) -> None:
    """サーバー起動後にバックエンドのクライアントを再接続

    base/assist だけでなく embed も再構築する。再構築しないと
    httpx.AsyncClient の kept-alive が前世代の死んだサーバの socket を
    保持したままになり、起動直後の health_check / embed 呼び出しが
    フライング失敗する。
    """
    if name == "base":
        from backend.free.api.system.status import _try_lazy_connect
        llama_cfg = cfg.get("llama", {})
        host = llama_cfg.get("host", "127.0.0.1")
        port = llama_cfg.get("port", 8080)
        url = f"http://{host}:{port}"
        await _try_lazy_connect(state, url, llama_cfg)

    elif name == "assist":
        from backend.free.api.model.assist_model_api import _try_lazy_connect_assist
        await _try_lazy_connect_assist(state, cfg)

    elif name == "embed":
        from backend.free.api.config.component_reload import reload_embedder
        try:
            await reload_embedder(state)
        except Exception as e:
            logger.warning("server_control: embed reload after start failed: %s", e)
