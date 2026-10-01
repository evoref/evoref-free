"""PID ファイル管理 — evoref serve の多重起動防止 + ポート占有プロセス検出"""

from __future__ import annotations

import json
import locale
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from backend.log_config import get_logger

logger = get_logger("cli.pid_manager")

PID_FILE = "evoref.pid"

# ポートの役割ごとに「kill してよい」イメージ名の接頭辞 (拡張子を除く小文字)。
# ポートの占有者はこれに一致するときだけ kill する (c_09 §2.1)。
IMAGE_LLAMA: tuple[str, ...] = ("llama-server",)
#: backend (uvicorn) と ``evoref serve`` 自身。``python`` は ``pythonw`` / ``python3.12`` も含む。
IMAGE_BACKEND: tuple[str, ...] = ("python", "evoref", "uvicorn")
IMAGE_FRONTEND: tuple[str, ...] = ("node",)
FRONTEND_PORT = 5173


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


@dataclass(frozen=True)
class ProcessIdentity:
    """PID の再利用を見分けるためのプロセスの同一性 (イメージ名と生成時刻)"""
    name: str
    #: 生成時刻 (比較にだけ使う値。Windows は epoch 秒、Linux は起動からの tick)。
    create_time: float | None = None


def _image_matches(name: str, prefixes: Sequence[str]) -> bool:
    """イメージ名 (``python.exe`` / ``llama-server``) が接頭辞のどれかに一致するか"""
    stem = PureWindowsPath(name).name.lower()
    if stem.endswith(".exe"):
        stem = stem[:-4]
    return bool(stem) and any(stem.startswith(p) for p in prefixes)


# ────────────────────────────────────────────
# プロセスの同一性
# ────────────────────────────────────────────

def get_process_identity(pid: int) -> ProcessIdentity | None:
    """``pid`` のプロセスのイメージ名と生成時刻。生きていなければ ``None``"""
    if pid <= 0:
        return None
    if sys.platform == "win32":
        return _process_identity_windows(pid)
    return _process_identity_unix(pid)


def _process_identity_windows(pid: int) -> ProcessIdentity | None:
    """Windows: OpenProcess + QueryFullProcessImageNameW + GetProcessTimes"""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetProcessTimes.argtypes = (
        (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    )

    process_query_limited_information = 0x1000
    still_active = 259
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != still_active:
            return None  # 終了済み (誰かがハンドルを持っているだけ)
        name = ""
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            name = PureWindowsPath(buf.value).name
        create_time: float | None = None
        times = [wintypes.FILETIME() for _ in range(4)]
        if kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            create_time = ticks / 10_000_000 - 11_644_473_600
        return ProcessIdentity(name=name, create_time=create_time)
    finally:
        kernel32.CloseHandle(handle)


def _process_identity_unix(pid: int) -> ProcessIdentity | None:
    """Unix: signal 0 で生存確認、ps でイメージ名、/proc で生成時刻 (無ければ ``None``)"""
    try:
        os.kill(pid, 0)
    except PermissionError:
        pass  # 他ユーザーのプロセス (生きている)
    except OSError:
        return None
    create_time: float | None = None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        # comm は括弧内で空白を含みうるので最後の ")" の後ろから数える (starttime は 22 番目)
        create_time = float(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        pass
    return ProcessIdentity(name=_get_process_name_unix(pid), create_time=create_time)


# ────────────────────────────────────────────
# PID ファイル管理
# ────────────────────────────────────────────

def _pid_path(project_root: Path) -> Path:
    """PID ファイルのパスを返す (データ根の ``run/``)"""
    from backend.config import resolve_data_path

    return resolve_data_path("run_dir", project_root) / PID_FILE


def _read_pid_record(pid_file: Path) -> tuple[int, ProcessIdentity | None] | None:
    """PID ファイルを読む。旧形式 (数字だけ) は同一性 ``None``。読めなければ ``None``"""
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.warning("Unreadable PID file: %s", e)
        return None
    try:
        return int(text), None
    except ValueError:
        pass
    try:
        record = json.loads(text)
        pid = record["pid"]
        name = record.get("name") or ""
        create_time = record.get("create_time")
        if not isinstance(pid, int) or not isinstance(name, str):
            raise TypeError("pid / name")
        if create_time is not None:
            create_time = float(create_time)
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning("Invalid PID file: %s", e)
        return None
    return pid, ProcessIdentity(name=name, create_time=create_time)


def _is_recorded_process(pid: int, recorded: ProcessIdentity | None) -> bool:
    """``pid`` のプロセスが PID ファイルに記録したプロセスそのものか

    PID は再利用されるので生きているだけでは足りない。記録した名前と生成時刻が
    一致するときだけ真。旧形式 (同一性なし) は python / evoref / uvicorn のときだけ信じる。
    """
    current = get_process_identity(pid)
    if current is None:
        return False
    if recorded is None:
        return _image_matches(current.name, IMAGE_BACKEND)
    if not current.name or current.name.lower() != recorded.name.lower():
        return False
    if recorded.create_time is None or current.create_time is None:
        return recorded.create_time is None and current.create_time is None
    return abs(current.create_time - recorded.create_time) < 1e-3


def check_pid(project_root: Path) -> int | None:
    """既存の PID ファイルを確認し、生存中の evoref プロセス PID を返す

    Returns:
        記録したプロセスが生きていればその PID、または None（PID ファイルなし /
        プロセス死亡 / PID が別のプロセスに再利用されている）
    """
    pid_file = _pid_path(project_root)
    if not pid_file.exists():
        return None

    record = _read_pid_record(pid_file)
    if record is None:
        _remove_pid_file(pid_file)
        return None
    pid, recorded = record

    if _is_recorded_process(pid, recorded):
        logger.debug("Process %d is still alive", pid)
        return pid

    logger.debug("Process %d is gone or reused, removing stale PID file", pid)
    _remove_pid_file(pid_file)
    return None


def acquire_pid(project_root: Path) -> bool:
    """PID ファイルを取得（現在のプロセス ID と同一性を記録）

    Returns:
        True: 取得成功、False: 既に別プロセスが起動中
    """
    from backend.io.atomic import atomic_write_text

    existing_pid = check_pid(project_root)
    if existing_pid is not None:
        return False

    pid_file = _pid_path(project_root)
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    me = get_process_identity(pid) or ProcessIdentity(name="")
    record = {"pid": pid, "name": me.name, "create_time": me.create_time}
    try:
        atomic_write_text(pid_file, json.dumps(record))
        logger.debug("PID file created: %s (pid=%d)", pid_file, pid)
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

    記録したプロセスが生きている場合だけ kill してから PID ファイルを削除する。
    PID が別のプロセスに再利用されていれば kill せずファイルだけ消す。

    Returns:
        PID ファイルにあった PID、または None（PID ファイルなし / 読めない）
    """
    pid_file = _pid_path(project_root)
    if not pid_file.exists():
        return None

    record = _read_pid_record(pid_file)
    if record is None:
        _remove_pid_file(pid_file)
        return None
    pid, recorded = record

    if _is_recorded_process(pid, recorded):
        logger.info("Force killing stale evoref process: pid=%d", pid)
        _kill_process_tree(pid)
    else:
        logger.info("PID %d is not the recorded evoref process; removing PID file only", pid)
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


def kill_port_occupants(
    occupants: list[PortOccupant],
    *,
    expected: Mapping[int, Sequence[str]],
) -> list[PortOccupant]:
    """ポート占有プロセスのうち、イメージ名がポートの役割どおりのものだけ kill

    ``expected`` はポート → kill してよいイメージ名の接頭辞 (``IMAGE_*``)。
    載っていないポートや名前が一致しない占有者 (別用途の llama-server や
    無関係なアプリ) は kill せず WARNING を出す。

    Returns:
        実際に kill したプロセスのリスト (kill しなかったものは含まない)
    """
    killed: list[PortOccupant] = []
    seen_pids: set[int] = set()
    for occ in occupants:
        if occ.pid in seen_pids:
            killed.append(occ)
            continue
        name = occ.process_name
        if not name:
            identity = get_process_identity(occ.pid)
            name = identity.name if identity is not None else ""
        allowed = expected.get(occ.port, ())
        if not _image_matches(name, allowed):
            logger.warning(
                "Not killing port occupant %s: image %r is not one of %s",
                occ.summary, name or "unknown", list(allowed),
            )
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
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
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
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
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

    # 補助タスク

    # 埋め込み (llama-cpp バックエンドの場合)
    embed_cfg = config.get("embedding", {})
    if embed_cfg.get("backend") == "llama-cpp" and embed_cfg.get("llama_port"):
        ports.append(embed_cfg["llama_port"])

    # リランカー (rag.rerank.mode が off 以外でモデル設定済みのときだけ。既定 on でもモデル未設定なら
    # 起動しないので見ない。起動時のポート競合検査の対象、c_16 §7.2.1)
    from backend.schemas.rag import rerank_mode_of

    if rerank_mode_of(config) != "off" and (config.get("model_paths") or {}).get("rerank_model"):
        ports.append(_rerank_port(config))

    return ports


def _rerank_port(config: dict) -> int:
    """rerank 用 llama-server のポート (既定 8083)。"""
    rr = (config.get("rag") or {}).get("rerank") or {}
    return int(rr.get("port", 8083))


def expected_images_by_port(
    config: dict, *, include_frontend: bool = False,
) -> dict[int, tuple[str, ...]]:
    """config.yaml のポート → そのポートで kill してよいイメージ名

    範囲は :func:`collect_configured_ports` と同じ (``include_frontend`` で 5173 を足す)。
    """
    expected: dict[int, tuple[str, ...]] = {
        config.get("server", {}).get("port", 8000): IMAGE_BACKEND,
        config.get("llama", {}).get("port", 8080): IMAGE_LLAMA,
    }
    embed_cfg = config.get("embedding", {})
    if embed_cfg.get("backend") == "llama-cpp" and embed_cfg.get("llama_port"):
        expected[embed_cfg["llama_port"]] = IMAGE_LLAMA
    # rerank のポートは mode off でも掃除する (off に変えた後に残った llama-server を止める)。
    # 止めるのは llama-server のイメージだけ (別のプロセスは触らない)。
    expected.setdefault(_rerank_port(config), IMAGE_LLAMA)
    if include_frontend:
        expected[FRONTEND_PORT] = IMAGE_FRONTEND
    return expected
