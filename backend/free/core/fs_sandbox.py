"""生成物を評価する子プロセスの書込み隔離 (f_10 §11.1-4)。

契約テスト・参考テスト・import スモークは生成されたコードを実行する。生成コードが依頼文の絶対パス
(``E:\\tmp\\c0927_01_todo\\todos.json``) やホーム (``Path.home()``) へ書くと、利用者の本物のフォルダを
作る・壊す (2026-09-27 ライブ監査 K01)。ここでは子プロセスの環境を組む:

- 評価フォルダの ``.sandbox/`` に隔離の本体 (:mod:`fs_sandbox_runtime` の写し) と、それを起動時に
  入れる ``sitecustomize.py`` を置き、``PYTHONPATH`` の先頭に足す (孫プロセスにも継承される)
- 書込みを許すのは呼出側が渡すフォルダ (作業フォルダの ``src/`` 等) と ``.sandbox/`` の
  ``home`` / ``tmp`` / ``cwd`` だけ。HOME / USERPROFILE / APPDATA / LOCALAPPDATA / TMP / TEMP 等を
  ``.sandbox/`` の下へ向ける。隔離の本体・違反の記録・作業フォルダの記録 (manifest 等) は許す根の外
- 評価プロセスは **スクリプトとして起動する** (:func:`write_script`)。``-c`` / ``-m`` だと audit hook を
  入れられない (:mod:`fs_sandbox_runtime` の docstring)
- 違反は ``.sandbox/violations.log`` に残り、呼出側は :func:`read_violations` で「未検査」に振り分ける
- 評価プロセスは :func:`run_bounded` で起こす — 出力は一時ファイルで受け、終わったら子孫ごと止める

隔離は多層防御の 1 枚で、書かせない側 (骨組み・本文の絶対パスの書き換え、テストの lint) が第一。
既存の ``sitecustomize`` (環境が持つもの) は連鎖しない — ``PYTHONPATH`` の先頭のこちらが読まれる。
"""

from __future__ import annotations

import os
import shutil
import signal
import site
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from backend.free.core import fs_sandbox_runtime as _runtime
from backend.io.atomic import AtomicWriter

#: 評価フォルダの中の隔離用フォルダ。
SANDBOX_DIR = ".sandbox"
#: 違反の例外メッセージの目印 (runtime と同じ値)。
MARKER = _runtime.MARKER
#: :func:`read_violations` が返す行の上限 (重複を除いた先頭から)。
MAX_VIOLATIONS = 50
#: 実行ごとに空から始める ``.sandbox`` の下のフォルダ (ホーム・一時フォルダ・CWD)。
SCRATCH_DIRS = ("home", "tmp", "cwd")

_RUNTIME_NAME = "_evoref_sandbox.py"
_SITECUSTOMIZE = (
    "# evoref: 作業フォルダの外への書込みを止める (f_10 §11.1-4)\n"
    "import _evoref_sandbox\n"
    "_evoref_sandbox.install()\n"
)
_VIOLATIONS = "violations.log"
#: 子プロセスでホーム・設定・キャッシュの置き場を決める環境変数 (``.sandbox/home`` からの相対)。
_HOME_ENV = {
    "HOME": ".",
    "USERPROFILE": ".",
    "APPDATA": "AppData/Roaming",
    "LOCALAPPDATA": "AppData/Local",
    "XDG_CONFIG_HOME": ".config",
    "XDG_CACHE_HOME": ".cache",
    "XDG_DATA_HOME": ".local/share",
    "MPLCONFIGDIR": ".matplotlib",
}
_TEMP_ENV = ("TMP", "TEMP", "TMPDIR")


def sandbox_dir(root: Path) -> Path:
    """評価フォルダ ``root`` の隔離用フォルダ。"""
    return Path(root) / SANDBOX_DIR


def sandbox_cwd(root: Path) -> Path:
    """評価プロセスの CWD (実行ごとに空から始める)。"""
    return sandbox_dir(root) / "cwd"


def _write_if_changed(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    with AtomicWriter(path) as fh:
        fh.write(text)


def install_runtime(root: Path) -> Path:
    """``root/.sandbox/`` に隔離の本体と sitecustomize を置き、そのフォルダを返す。"""
    sb = sandbox_dir(root)
    sb.mkdir(parents=True, exist_ok=True)
    _write_if_changed(sb / _RUNTIME_NAME, Path(_runtime.__file__).read_text(encoding="utf-8"))
    _write_if_changed(sb / "sitecustomize.py", _SITECUSTOMIZE)
    return sb


def write_script(root: Path, name: str, body: str) -> Path:
    """評価プロセスの入口を ``.sandbox/<name>`` に書く (スクリプトとして起動するため)。"""
    sb = install_runtime(root)
    path = sb / name
    _write_if_changed(path, body)
    return path


def reset_state(root: Path) -> None:
    """前回の実行が ``.sandbox/`` に残したホーム・一時フォルダ・CWD・違反の記録を消す。"""
    sb = sandbox_dir(root)
    for name in SCRATCH_DIRS:
        shutil.rmtree(sb / name, ignore_errors=True)
    try:
        (sb / _VIOLATIONS).unlink()
    except OSError:
        pass


def sandbox_env(
    root: Path, base_env: Mapping[str, str] | None = None, *, write_dirs: Sequence[Path] | None = None,
) -> dict[str, str]:
    """子プロセスの環境変数 (``base_env`` 既定は現在の環境)。

    書込みを許すのは ``write_dirs`` (既定は ``root`` 全体) と ``.sandbox/`` の home / tmp / cwd。
    ``root`` の中の ``.sandbox/`` を用意する (呼出前に :func:`reset_state` で前回の状態を消す)。
    元の user site-packages は ``PYTHONPATH`` の末尾に残す (APPDATA / HOME を差し替えると
    ``site`` がそこを見なくなり、pytest 等の import が壊れる)。
    """
    root = Path(root)
    sb = install_runtime(root)
    home = sb / "home"
    tmp = sb / "tmp"
    cwd = sb / "cwd"
    env = dict(os.environ if base_env is None else base_env)
    for key, rel in _HOME_ENV.items():
        target = home / rel
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    tmp.mkdir(parents=True, exist_ok=True)
    cwd.mkdir(parents=True, exist_ok=True)
    for key in _TEMP_ENV:
        env[key] = str(tmp)
    paths = [str(sb)]
    paths += [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    user_site = site.getusersitepackages() if site.ENABLE_USER_SITE is not False else ""
    if user_site and os.path.isdir(user_site) and user_site not in paths:
        paths.append(user_site)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # 出力は UTF-8 (パイプ・ファイルへの出力が cp932 になり ``print('✅')`` が偽の不合格になった)。
    # ``PYTHONUTF8`` は立てない — ``open()`` の既定が利用者の環境と変わり、評価と実物がずれる (f_10 §11.1-3)
    env["PYTHONIOENCODING"] = "utf-8"
    allowed = [Path(p) for p in (write_dirs if write_dirs is not None else [root])] + [home, tmp, cwd]
    env[_runtime.ROOTS_ENV] = os.pathsep.join(str(p) for p in allowed)
    env[_runtime.LOG_ENV] = str(sb / _VIOLATIONS)
    return env


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
#: 子孫を止めてから消えるまで待つ上限 (秒)。
_STOP_WAIT_SEC = 5.0


def _windows_job(proc: subprocess.Popen):
    """``proc`` を閉じたら中身を止める Job Object に入れる (Windows だけ。入れられなければ ``None``)。"""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class _Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _Basic), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID,
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _Extended()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        kernel32.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info),
        )
        if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):  # type: ignore[attr-defined]
            kernel32.CloseHandle(job)
            return None
        return (kernel32, job)
    except Exception:  # noqa: BLE001 - Job に入れられなければ子と、親の pid で辿った子孫を止める
        return None


def _job_active_processes(kernel32, handle) -> int:
    """Job の中で生きているプロセスの数 (読めなければ 0)。"""
    import ctypes
    from ctypes import wintypes

    class _Accounting(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    info = _Accounting()
    ok = kernel32.QueryInformationJobObject(
        handle, _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION, ctypes.byref(info), ctypes.sizeof(info), None,
    )
    return int(info.ActiveProcesses) if ok else 0


def _windows_descendants(root_pid: int, root_handle: int) -> list[int]:
    """``root_pid`` の子孫 (親の pid の鎖で辿る。親が先に終わった孫も、その親の pid が分かる限り拾う)。

    子より後に作られたプロセスだけを子孫とみなす (終わったプロセスの pid が再利用されたときの取り違えを避ける)。
    """
    import ctypes
    from ctypes import wintypes

    class _Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Entry)]
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Entry)]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(ctypes.c_uint64)] * 4]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    def _times(handle) -> int:
        times = [ctypes.c_uint64() for _ in range(4)]
        ok = kernel32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times])
        return int(times[0].value) if ok else 0

    def _created(pid: int) -> int:
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return 0
        try:
            return _times(handle)
        finally:
            kernel32.CloseHandle(handle)

    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return []
    children: dict[int, list[int]] = {}
    try:
        entry = _Entry()
        entry.dwSize = ctypes.sizeof(_Entry)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            children.setdefault(int(entry.th32ParentProcessID), []).append(int(entry.th32ProcessID))
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    # 子は終わっていても Popen がハンドルを持つので pid は再利用されず、作られた時刻も読める
    root_created = _times(root_handle)
    found: list[int] = []
    stack = [(root_pid, root_created)]
    while stack:
        pid, created = stack.pop()
        for child in children.get(pid, []):
            if child in found or child == root_pid:
                continue
            child_created = _created(child)
            # 作られた時刻が読めないプロセスは子孫と確かめられないので止めない
            if not child_created or (created and child_created < created):
                continue
            found.append(child)
            stack.append((child, child_created or created))
    return found


def _terminate_windows(pids: list[int]) -> None:
    """``pids`` を止めて、消えるまで少し待つ。"""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    for pid in pids:
        handle = kernel32.OpenProcess(0x0001 | 0x00100000, False, pid)  # PROCESS_TERMINATE | SYNCHRONIZE
        if not handle:
            continue
        try:
            kernel32.TerminateProcess(handle, 1)
            kernel32.WaitForSingleObject(handle, int(_STOP_WAIT_SEC * 1000))
        finally:
            kernel32.CloseHandle(handle)


def _stop_tree(proc: subprocess.Popen, job) -> None:
    """``proc`` とその子孫を止める (Windows は Job ごと + 親の pid の鎖で拾った子孫、POSIX はプロセスグループ)。"""
    if os.name == "nt":
        # 先に子孫を拾う (Job から抜けた孫 — venv のリダイレクタが起こしたもの — も親の pid で辿れる)
        try:
            descendants = _windows_descendants(proc.pid, int(proc._handle))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - 拾えなければ Job だけで止める
            descendants = []
        if job is not None:
            kernel32, handle = job
            kernel32.TerminateJobObject(handle, 1)
            # 止め終わるまで待つ (待たないと、消える前の孫が掴んだ一時フォルダを片付けられない)
            deadline = time.monotonic() + _STOP_WAIT_SEC
            while _job_active_processes(kernel32, handle) > 0 and time.monotonic() < deadline:
                time.sleep(0.05)
        elif proc.poll() is None:
            proc.kill()
        _terminate_windows(descendants)
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def _direct_interpreter(argv: list[str], env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """venv のリダイレクタ (Windows の ``.venv\\Scripts\\python.exe``) を介さずに本体の Python を起こす。

    リダイレクタは自分を「子の抜け出しを許す Job」に入れてから本体を起こすので、本体が起こした孫はこちらの
    Job から抜けて止められない (実測: 孫が一時フォルダを掴んだまま残った)。本体を直接起こし、リダイレクタと
    同じく ``__PYVENV_LAUNCHER__`` で venv として振る舞わせる。
    """
    base = getattr(sys, "_base_executable", "") or ""
    if (
        os.name == "nt" and argv and argv[0] == sys.executable and base and base != sys.executable
        and os.path.isfile(base)
    ):
        return [base, *argv[1:]], {**env, "__PYVENV_LAUNCHER__": sys.executable}
    return argv, env


def run_bounded(
    args: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout: float, stdin=None,
) -> subprocess.CompletedProcess:
    """評価の子プロセスを時間上限つきで走らせ、終わったら子孫ごと止める (f_10 §11.1-3)。

    出力はパイプでなく一時ファイルで受ける — 孫が出力を掴んだまま生き残るとパイプの読み取りが
    時間上限を越えて待ち続ける。時間切れなら子孫ごと止めて ``subprocess.TimeoutExpired``
    (``output`` / ``stderr`` にそれまでの出力) を投げる。正常終了でも残った子孫を止める。
    """
    extra = {} if os.name == "nt" else {"start_new_session": True}
    argv, child_env = _direct_interpreter(list(args), dict(env))
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=child_env, stdin=stdin, stdout=out, stderr=err, **extra,
        )
        job = _windows_job(proc)
        try:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _stop_tree(proc, job)
                proc.wait()
                out.seek(0)
                err.seek(0)
                raise subprocess.TimeoutExpired(
                    list(args), timeout, output=out.read(), stderr=err.read(),
                ) from None
        finally:
            _stop_tree(proc, job)
            if job is not None:
                job[0].CloseHandle(job[1])
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(list(args), proc.returncode, out.read(), err.read())


def read_violations(root: Path) -> list[str]:
    """子プロセスが止められた書込み (``<event>\\t<path>``、重複を除いた先頭 :data:`MAX_VIOLATIONS` 行)。"""
    try:
        text = (sandbox_dir(root) / _VIOLATIONS).read_text(encoding="utf-8")
    except OSError:
        return []
    lines = dict.fromkeys(line.strip() for line in text.splitlines() if line.strip())
    return list(lines)[:MAX_VIOLATIONS]


def violation_paths(violations: list[str]) -> list[str]:
    """:func:`read_violations` の行からパスだけを取り出す (重複は除く)。"""
    return list(dict.fromkeys(line.split("\t", 1)[-1] for line in violations))


def is_sandbox_error(text: str) -> bool:
    """例外メッセージが隔離による書込みの拒否か。"""
    return MARKER in (text or "")


#: 作業フォルダの conftest に足す行 (sitecustomize が読まれない起動への備え。二重には入らない)。
CONFTEST_GUARD = (
    "_SANDBOX = Path(__file__).parent / \"" + SANDBOX_DIR + "\"\n"
    "if (_SANDBOX / \"" + _RUNTIME_NAME + "\").is_file():\n"
    "    if str(_SANDBOX) not in sys.path:\n"
    "        sys.path.insert(0, str(_SANDBOX))\n"
    "    import _evoref_sandbox\n"
    "\n"
    "    _evoref_sandbox.install(\n"
    "        default_roots=[str(Path(__file__).parent / \"src\")]\n"
    "        + [str(_SANDBOX / name) for name in " + repr(SCRATCH_DIRS) + "],\n"
    "        default_log=str(_SANDBOX / \"" + _VIOLATIONS + "\"),\n"
    "    )\n"
)


__all__ = [
    "CONFTEST_GUARD",
    "MARKER",
    "MAX_VIOLATIONS",
    "SANDBOX_DIR",
    "SCRATCH_DIRS",
    "install_runtime",
    "is_sandbox_error",
    "read_violations",
    "reset_state",
    "run_bounded",
    "sandbox_cwd",
    "sandbox_dir",
    "sandbox_env",
    "violation_paths",
    "write_script",
]
