"""評価フォルダの外への書込みを止める (生成物を評価する子プロセス用、f_10 §11.1-4)。

**evoref のプロセスでは ``install()`` を呼ばない** — 一度入れた audit hook は外せず、
``os`` / ``open`` の差し替えもプロセス全体に効く。:mod:`backend.free.core.fs_sandbox` が
このファイルを評価フォルダの ``.sandbox/`` へ写し、同じ場所の ``sitecustomize.py`` が
子プロセスの起動時に読み込む。環境変数 (``PYTHONPATH`` / ``EVOREF_SANDBOX_ROOTS`` /
``EVOREF_SANDBOX_LOG``) は子プロセスへ継承されるので、テストが
``subprocess.run([sys.executable, ...])`` で起こした Python にも効く。

止め方は 2 枚:

1. 関数の差し替え (起動の仕方によらず効く): ``builtins.open`` / ``io.open`` / ``_io.open`` /
   ``io.FileIO`` / ``os.*`` と ``nt.*`` (``posix.*``) の書込み系 / ``_winapi.CopyFile2``。
   ``os.makedirs`` / ``shutil`` / ``pathlib`` / ``tempfile`` はこれらを呼ぶ。
2. ``sys.addaudithook`` (C から直接開く経路・``sqlite3.connect``・``shutil`` の監査イベントも拾う)。
   **スクリプトとして起動したプロセス (``python x.py`` / ``python - < x.py``) だけ** に入れる —
   CPython (3.12.0 / 3.13 の python.org・Microsoft Store・conda で確認、2026-09-27) は ``-c`` / ``-m``
   で起動したプロセスに Python の audit hook があると ``marshal.loads`` が
   ``'bytes' object has no attribute 'co_filename'`` で落ち、以後の import がすべて壊れる。
   evoref が起こす評価プロセス (pytest・import / entry スモーク) はすべてスクリプトとして起動するので
   2 枚とも入る。2 枚目が入らないのは、生成コードが ``-c`` / ``-m`` で起こした孫プロセスだけ。

拒否は ``OSError(EROFS)`` で投げる — ``PermissionError`` だと ``tempfile.mkstemp`` が名前の衝突と
みなして ``TMP_MAX`` 回やり直す (違反の記録が 1 万行を超えた)。同じパスの記録は 1 回だけ。

子プロセスの Python は evoref の環境とは限らないので、標準ライブラリだけで書く。
止めるのは書込みだけで、読み取りは通す (依頼が明示した入力ファイルは読める)。
防げないもの: Python 以外のプロセス (``os.system`` / ``subprocess`` で起こした cmd・node 等)、
ctypes / Win32 API の直呼び、環境変数を消して起こした子プロセス、``dir_fd`` 相対の操作、
``-c`` / ``-m`` の孫プロセスでの ``sqlite3``。
"""

import builtins
import errno
import io
import os
import sys

#: 違反の例外メッセージの目印 (呼出側が「不合格」ではなく「未検査」に振り分ける)。
MARKER = "[evoref sandbox]"
#: 書込みを許す根 (``os.pathsep`` 区切り)。
ROOTS_ENV = "EVOREF_SANDBOX_ROOTS"
#: 違反を 1 行ずつ追記するファイル (書込みを許す根の外に置く)。
LOG_ENV = "EVOREF_SANDBOX_LOG"

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
#: 第 1 引数のパスへ書く操作 (最後の引数が dir_fd のもの)。
_PATH_EVENTS_WITH_DIR_FD = frozenset({"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown", "os.utime"})
#: 第 1 引数のパスへ書く操作 (dir_fd を持たないもの)。
_PATH_EVENTS = frozenset({"os.truncate", "shutil.rmtree"})
#: 第 1 / 第 2 引数の両方を変える操作 (移動元が外なら外のファイルを消すのと同じ)。
_BOTH_EVENTS = frozenset({"os.rename", "shutil.move"})
#: 第 2 引数 (宛先) へ書く操作。
_DST_EVENTS = frozenset({
    "os.link", "os.symlink", "shutil.copyfile", "shutil.copytree", "shutil.copymode", "shutil.copystat",
    "_winapi.CopyFile2",
})

_installed = False
_roots: tuple[str, ...] = ()
_log_path = ""
_busy = False
_recorded: set[str] = set()


def _norm(path) -> str | None:
    if path is None or isinstance(path, int):
        return None
    try:
        text = os.fsdecode(os.fspath(path))
    except TypeError:
        return None
    return os.path.normcase(os.path.abspath(text))


def _inside(norm: str) -> bool:
    for root in _roots:
        if norm == root or norm.startswith(root.rstrip(os.sep) + os.sep):
            return True
    return False


def _is_device(norm: str) -> bool:
    # NUL / CON (Windows のデバイス名前空間) と /dev/null 等
    return norm.startswith("\\\\.\\") or norm.startswith("/dev/")


def _allowed(norm: str) -> bool:
    if _inside(norm) or _is_device(norm):
        return True
    try:
        # 8.3 の短い名前・シンボリックリンクで同じ場所を別の綴りで指している場合
        return _inside(os.path.normcase(os.path.realpath(norm)))
    except (OSError, ValueError):
        return False


def _record(event: str, norm: str, path) -> None:
    if not _log_path or norm in _recorded or "__pycache__" in norm.split(os.sep):
        return
    _recorded.add(norm)
    try:
        with _ORIGINAL_OPEN(_log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{event}\t{os.path.abspath(os.fsdecode(os.fspath(path)))}\n")
    except OSError:
        pass


def _check(event: str, path, dir_fd=None) -> None:
    """``path`` への書込みが根の外なら記録して ``OSError(EROFS)``。"""
    global _busy
    if _busy:
        return
    norm = _norm(path)
    if norm is None:
        return
    if isinstance(dir_fd, int) and not os.path.isabs(os.fsdecode(os.fspath(path))):
        return
    _busy = True
    try:
        if _allowed(norm):
            return
        _record(event, norm, path)
    finally:
        _busy = False
    raise OSError(errno.EROFS, f"{MARKER} writing outside the workspace is not allowed ({event})", str(path))


def _mode_writes(mode) -> bool:
    return isinstance(mode, str) and any(c in mode for c in "wax+")


def _sqlite_path(database):
    """``sqlite3.connect`` の database のうち、ディスクへ書きうるファイルのパス (無ければ None)。"""
    if isinstance(database, bytes):
        database = os.fsdecode(database)
    if not isinstance(database, str):
        try:
            database = os.fsdecode(os.fspath(database))
        except TypeError:
            return None
    if database in ("", ":memory:"):
        return None
    if database.startswith("file:"):
        path, _, query = database[5:].partition("?")
        options = dict(p.partition("=")[::2] for p in query.split("&") if p)
        if options.get("mode") in ("ro", "memory") or path in ("", ":memory:"):
            return None
        from urllib.parse import unquote

        path = unquote(path)
        if path.startswith("//"):
            path = path[2:]
            path = path[path.find("/"):] if "/" in path else ""
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]  # file:///C:/x.db
        return path or None
    return database


# ── 1 枚目: 関数の差し替え ────────────────────────────────────────────

_ORIGINAL_OPEN = io.open


def _open(file, mode="r", *args, **kwargs):
    if _mode_writes(mode):
        _check("open", file)
    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)


def _wrap(module, name: str, event: str, *, second: bool = False, both: bool = False) -> None:
    original = getattr(module, name, None)
    if original is None or getattr(original, "_evoref_guard", False):
        return

    def wrapper(*args, **kwargs):
        dir_fd = kwargs.get("dir_fd", kwargs.get("src_dir_fd"))
        if args:
            if both or not second:
                _check(event, args[0], dir_fd)
            if (both or second) and len(args) > 1:
                _check(event, args[1], kwargs.get("dst_dir_fd"))
        return original(*args, **kwargs)

    wrapper.__name__ = getattr(original, "__name__", name)
    wrapper.__doc__ = getattr(original, "__doc__", None)
    wrapper._evoref_guard = True  # type: ignore[attr-defined]
    setattr(module, name, wrapper)


def _wrap_os_open(module) -> None:
    original = getattr(module, "open", None)
    if original is None or getattr(original, "_evoref_guard", False):
        return

    def os_open(path, flags, *args, **kwargs):
        if isinstance(flags, int) and flags & _WRITE_FLAGS:
            _check("open", path, kwargs.get("dir_fd"))
        return original(path, flags, *args, **kwargs)

    os_open._evoref_guard = True  # type: ignore[attr-defined]
    setattr(module, "open", os_open)


def _patch() -> None:
    import _io

    builtins.open = _open
    io.open = _open
    _io.open = _open
    original_fileio = io.FileIO

    class FileIO(original_fileio):  # type: ignore[misc, valid-type]
        def __init__(self, file, mode="r", *args, **kwargs):
            if _mode_writes(mode):
                _check("open", file)
            super().__init__(file, mode, *args, **kwargs)

    io.FileIO = FileIO
    _io.FileIO = FileIO
    # os の関数は nt / posix の同じ関数を指す — 直接 ``nt.open`` を呼ぶ経路も包む
    native = sys.modules.get("nt") or sys.modules.get("posix")
    for module in (os, native):
        if module is None:
            continue
        _wrap_os_open(module)
        for name, event in (
            ("mkdir", "os.mkdir"), ("remove", "os.remove"), ("unlink", "os.remove"), ("rmdir", "os.rmdir"),
            ("truncate", "os.truncate"), ("chmod", "os.chmod"), ("utime", "os.utime"),
        ):
            _wrap(module, name, event)
        for name in ("rename", "replace", "renames"):
            _wrap(module, name, "os.rename", both=True)
        for name in ("link", "symlink"):
            _wrap(module, name, f"os.{name}", second=True)
    try:
        import _winapi
    except ImportError:
        return
    _wrap(_winapi, "CopyFile2", "_winapi.CopyFile2", second=True)


# ── 2 枚目: audit hook ────────────────────────────────────────────────


def _hook(event: str, args: tuple) -> None:
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if _mode_writes(mode) or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS)):
            _check(event, path)
    elif event in _PATH_EVENTS_WITH_DIR_FD and args:
        _check(event, args[0], args[-1] if len(args) > 1 else None)
    elif event in _PATH_EVENTS and args:
        _check(event, args[0])
    elif event in _BOTH_EVENTS and len(args) >= 2:
        _check(event, args[0])
        _check(event, args[1])
    elif event in _DST_EVENTS and len(args) >= 2:
        _check(event, args[1])
    elif event == "sqlite3.connect" and args:
        path = _sqlite_path(args[0])
        if path is not None:
            _check(event, path)


#: 値を 1 つ取るインタプリタのオプション (``-X utf8`` / ``-W ignore``)。
_OPTIONS_WITH_VALUE = frozenset({"X", "W", "Q"})
_LONG_OPTIONS_WITH_VALUE = frozenset({"--check-hash-based-pycs"})


def launched_as_script(argv=None) -> bool:
    """プロセスが ``python <script>`` / ``python - < x.py`` で起動されたか (``-c`` / ``-m`` / 対話ではない)。

    短いオプションはまとめ書き (``-Bc`` / ``-uc`` / ``-Em`` / ``-Xutf8``) を 1 文字ずつ見る。
    """
    args = list(getattr(sys, "orig_argv", []) if argv is None else argv)[1:]
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == "-":
            return True
        if arg.startswith("--"):
            skip = arg in _LONG_OPTIONS_WITH_VALUE
            continue
        if not arg.startswith("-"):
            return True
        for i, flag in enumerate(arg[1:]):
            if flag in "cm":
                return False
            if flag in _OPTIONS_WITH_VALUE:
                # 値が同じ引数に続いていなければ次の引数が値
                skip = i == len(arg) - 2
                break
    return False


def install(default_roots=(), default_log: str = "") -> bool:
    """書込みの制限を入れる (2 回目以降は何もしない)。入れたら True。

    根は ``EVOREF_SANDBOX_ROOTS`` を優先し、無ければ ``default_roots`` (conftest からの備え)。
    """
    global _installed, _roots, _log_path
    if _installed:
        return True
    raw = [r for r in os.environ.get(ROOTS_ENV, "").split(os.pathsep) if r] or [str(r) for r in default_roots]
    if not raw:
        return False
    roots: list[str] = []
    for r in raw:
        for form in (os.path.abspath(r), os.path.realpath(r)):
            norm = os.path.normcase(form)
            if norm not in roots:
                roots.append(norm)
    _roots = tuple(roots)
    _log_path = os.environ.get(LOG_ENV, "") or default_log
    _patch()
    if launched_as_script():
        sys.addaudithook(_hook)
    _installed = True
    return True
