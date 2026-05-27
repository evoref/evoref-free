"""PID ファイル管理 — evoref serve の多重起動防止 + ポート占有プロセス検出"""

from __future__ import annotations

import locale
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("cli.pid_manager")

PID_DIR = "local/run"
PID_FILE = "evoref.pid"


# ────────────────────────────────────────────
# Windows コンソール出力のデコード
# ────────────────────────────────────────────
#
# netstat / tasklist などの Windows 標準コマンドは OEM コードページ
# (日本語環境では cp932) で出力する。PYTHONUTF8=1 や `text=True` のまま
# subprocess を起動すると utf-8 でデコードを試みて UnicodeDecodeError で
# クラッシュするため、必ず bytes で受けてから安全にデコードする。

def _decode_windows_console_output(raw: bytes | None) -> str:
    """Windows のコンソールコマンド出力をロケール依存で安全にデコード"""
    if not raw:
        return ""
    # 1) OEM / mbcs コーデック (Windows のみ)。JP 環境では cp932 相当。
    # 2) 現在ロケールの推奨エンコーディング。
    # 3) 最終手段として utf-8 + errors=replace。
    candidates: list[str] = []
    if sys.platform == "win32":
        candidates.extend(["oem", "mbcs"])
    pref = locale.getpreferredencoding(False)
    if pref and pref.lower() not in {c.lower() for c in candidates}:
        candidates.append(pref)
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _run_windows_console_command(cmd: list[str], timeout: float = 5.0) -> str:
    """Windows コンソールコマンドを実行し、出力をロケール安全にデコードして返す

    失敗時 (非ゼロ終了 / タイムアウト / OSError) は空文字列を返す。
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("Windows console command failed: %s (%s)", cmd[0], e)
        return ""
    if result.returncode != 0:
        logger.debug(
            "Windows console command returned %d: %s",
            result.returncode,
            cmd[0],
        )
    return _decode_windows_console_output(result.stdout)


# ────────────────────────────────────────────
# データ構造
# ────────────────────────────────────────────

@dataclass
class PortOccupant:
    """ポートを占有しているプロセスの情報"""
    port: int
    pid: int
    process_name: str = ""

    @property
    def summary(self) -> str:
        name_part = f" ({self.process_name})" if self.process_name else ""
        return f":{self.port} → PID {self.pid}{name_part}"


# ────────────────────────────────────────────
# PID ファイル管理
# ────────────────────────────────────────────

def _pid_path(project_root: Path) -> Path:
    """PID ファイルのパスを返す"""
    return project_root / PID_DIR / PID_FILE


def _is_process_alive(pid: int) -> bool:
    """指定 PID のプロセスが生存しているか確認"""
    if sys.platform == "win32":
        # Windows: OpenProcess で存在確認
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        # Unix: signal 0 で存在確認
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def check_pid(project_root: Path) -> int | None:
    """既存の PID ファイルを確認し、生存中のプロセス PID を返す

    Returns:
        生存中のプロセスの PID、または None（PID ファイルなし or プロセス死亡）
    """
    pid_file = _pid_path(project_root)
    if not pid_file.exists():
        return None

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as e:
        logger.warning("Invalid PID file, removing: %s", e)
        _remove_pid_file(pid_file)
        return None

    if _is_process_alive(pid):
        logger.debug("Process %d is still alive", pid)
        return pid

    logger.debug("Process %d is dead, removing stale PID file", pid)
    _remove_pid_file(pid_file)
    return None


def acquire_pid(project_root: Path) -> bool:
    """PID ファイルを取得（現在のプロセス ID を記録）

    Returns:
        True: 取得成功、False: 既に別プロセスが起動中
    """
    existing_pid = check_pid(project_root)
    if existing_pid is not None:
        return False

    pid_file = _pid_path(project_root)
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
        logger.debug("PID file created: %s (pid=%d)", pid_file, os.getpid())
        return True
    except OSError as e:
        logger.error("Failed to create PID file: %s", e)
        return False


def release_pid(project_root: Path) -> None:
    """PID ファイルを削除"""
    pid_file = _pid_path(project_root)
    _remove_pid_file(pid_file)
    logger.debug("PID file released: %s", pid_file)


def force_release_stale_pid(project_root: Path) -> int | None:
    """stale な PID ファイルを強制削除し、古い PID を返す

    プロセスが生きている場合は kill してから PID ファイルを削除する。

    Returns:
        kill した PID、または None（PID ファイルなし）
    """
    pid_file = _pid_path(project_root)
    if not pid_file.exists():
        return None

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        _remove_pid_file(pid_file)
        return None

    if _is_process_alive(pid):
        logger.info("Force killing stale evoref process: pid=%d", pid)
        _kill_process_tree(pid)
    _remove_pid_file(pid_file)
    return pid


def _remove_pid_file(pid_file: Path) -> None:
    """PID ファイルを安全に削除"""
    try:
        pid_file.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Failed to remove PID file: %s", e)


# ────────────────────────────────────────────
# ポート占有プロセス検出
# ────────────────────────────────────────────

def find_port_occupant(port: int) -> PortOccupant | None:
    """指定ポートで LISTEN しているプロセスを検出

    Returns:
        PortOccupant or None（ポートが空いている場合）
    """
    if sys.platform == "win32":
        return _find_port_occupant_windows(port)
    else:
        return _find_port_occupant_unix(port)


def find_port_occupants(ports: list[int]) -> list[PortOccupant]:
    """複数ポートの占有プロセスを一括検出"""
    occupants: list[PortOccupant] = []
    for port in ports:
        occ = find_port_occupant(port)
        if occ is not None:
            occupants.append(occ)
    return occupants


def kill_port_occupants(occupants: list[PortOccupant]) -> list[PortOccupant]:
    """ポート占有プロセスを kill

    Returns:
        実際に kill したプロセスのリスト
    """
    killed: list[PortOccupant] = []
    seen_pids: set[int] = set()
    for occ in occupants:
        if occ.pid in seen_pids:
            killed.append(occ)
            continue
        seen_pids.add(occ.pid)
        logger.info("Killing port occupant: %s", occ.summary)
        _kill_process_tree(occ.pid)
        killed.append(occ)
    return killed


def _find_port_occupant_windows(port: int) -> PortOccupant | None:
    """Windows: netstat + tasklist でポート占有プロセスを検出

    日本語 Windows (cp932 ロケール) でも安全に動作するよう、subprocess は
    bytes で受け取り `_decode_windows_console_output` で明示的にデコードする。
    """
    stdout = _run_windows_console_command(["netstat", "-ano"])
    for line in stdout.splitlines():
        # TCP    0.0.0.0:8080    0.0.0.0:0    LISTENING    12345
        if f":{port}" not in line or "LISTENING" not in line:
            continue
        parts = line.split()
        # ポートの正確な一致を確認（:80 が :8080 にマッチしないように）
        local_addr = parts[1] if len(parts) >= 5 else ""
        if not local_addr.endswith(f":{port}"):
            continue
        pid_str = parts[-1]
        if pid_str.isdigit():
            pid = int(pid_str)
            name = _get_process_name_windows(pid)
            return PortOccupant(port=port, pid=pid, process_name=name)
    return None


def _find_port_occupant_unix(port: int) -> PortOccupant | None:
    """Unix: lsof でポート占有プロセスを検出"""
    try:
        result = subprocess.run(
            ["lsof", "-iTCP:{port}".format(port=port), "-sTCP:LISTEN", "-nP", "-t"],
            capture_output=True, text=True, timeout=5,
        )
        for pid_str in result.stdout.strip().split():
            if pid_str.isdigit():
                pid = int(pid_str)
                name = _get_process_name_unix(pid)
                return PortOccupant(port=port, pid=pid, process_name=name)
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("Failed to check port %d via lsof", port)
    return None


def _get_process_name_windows(pid: int) -> str:
    """Windows: tasklist で PID からプロセス名を取得

    cp932 ロケール対応のため bytes で受けてから安全にデコードする。
    """
    stdout = _run_windows_console_command(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
    )
    # "llama-server.exe","12345","Console","1","123,456 K"
    line = stdout.strip()
    if line and line.startswith('"'):
        return line.split('"')[1]
    return ""


def _get_process_name_unix(pid: int) -> str:
    """Unix: ps でプロセス名を取得"""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


# ────────────────────────────────────────────
# プロセス kill ヘルパー
# ────────────────────────────────────────────

def _kill_process_tree(pid: int) -> None:
    """プロセスツリーごと終了"""
    import signal as sig
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), sig.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def collect_configured_ports(config: dict) -> list[int]:
    """config.yaml から全サーバーが使用するポート一覧を収集"""
    ports: list[int] = []

    # バックエンド
    server_cfg = config.get("server", {})
    ports.append(server_cfg.get("port", 8000))

    # ベース llama-server
    llama_cfg = config.get("llama", {})
    ports.append(llama_cfg.get("port", 8080))

    # アシストモデル
    assist_local = config.get("assist_model", {}).get("local", {})
    if assist_local.get("port"):
        ports.append(assist_local["port"])

    # 埋め込み (llama-cpp バックエンドの場合)
    embed_cfg = config.get("embedding", {})
    if embed_cfg.get("backend") == "llama-cpp" and embed_cfg.get("llama_port"):
        ports.append(embed_cfg["llama_port"])

    # リランカー
    reranker_cfg = config.get("reranker", {})
    if reranker_cfg.get("enabled") and reranker_cfg.get("backend") == "llama-cpp":
        if reranker_cfg.get("port"):
            ports.append(reranker_cfg["port"])

    return ports
