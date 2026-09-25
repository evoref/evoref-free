"""create の実行環境 (``php`` / ``node``) の解決・検証・起動 (f_10 §12.4 / c_06 §1.5)。

staged v2 が生成物の **構文だけ** を確かめるのに使う (``php -l`` / ``node --check``)。生成物を
丸ごと実行はしない。実行ファイルは ``create.runtimes.<name>`` (空なら PATH) で指定する。

設定値は GUI から専用 API で書けるので (c_06 §1.5 の例外)、保存時も使用時も同じ
:func:`validate_runtime_path` を通す — 絶対パス / 実在するファイル / Windows では ``.exe`` /
ファイル名の語幹が ``<name>`` / データ根・出力先・インストール根の外。起動は
``shell=False``・最小の環境変数・時間上限つきで、引数は呼出し側の固定値だけ。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("core.runtimes")

#: 設定できる実行環境 (``create.runtimes.<name>``)。
RUNTIME_NAMES: tuple[str, ...] = ("php", "node")

#: 版の表示に使う引数 (固定)。
_VERSION_ARGS: dict[str, tuple[str, ...]] = {"php": ("--version",), "node": ("--version",)}

#: 子プロセスへ写す環境変数 (存在するものだけ)。サーバの環境を丸ごと継承させない。
_MINIMAL_ENV_VARS: tuple[str, ...] = (
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "HOME",
    "LOCALAPPDATA", "APPDATA", "PATHEXT", "LANG", "LC_ALL",
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def forbidden_roots() -> tuple[Path, ...]:
    """実行ファイルを置いてはいけない場所 (データ根・出力先・インストール根)。"""
    from backend.config import get_path_resolver
    from backend.data_root import install_root

    roots = [install_root()]
    try:
        resolver = get_path_resolver()
        roots += [resolver.data_root, Path(resolver.resolve_local("outputs_dir"))]
    except Exception as exc:  # noqa: BLE001 - config 未ロードでもインストール根だけは守る
        logger.debug("runtimes: path resolver unavailable: %s", exc)
    out = []
    for r in roots:
        try:
            out.append(Path(r).resolve())
        except OSError:
            continue
    return tuple(out)


def validate_runtime_path(
    name: str, raw: str, *, forbidden: tuple[Path, ...] | None = None,
) -> tuple[Path | None, str]:
    """設定されたパスが ``name`` の実行ファイルとして使えるか。戻り値は (解決済みパス, 理由)。"""
    if name not in RUNTIME_NAMES:
        return None, "unknown_runtime"
    text = str(raw or "").strip().strip('"')
    if not text:
        return None, "empty"
    path = Path(text)
    if not path.is_absolute():
        return None, "not_absolute"
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None, "not_found"
    if not resolved.is_file():
        return None, "not_a_file"
    if sys.platform == "win32" and resolved.suffix.lower() != ".exe":
        return None, "not_exe"
    if resolved.stem.lower() != name:
        return None, "name_mismatch"
    for root in forbidden if forbidden is not None else forbidden_roots():
        if _is_within(resolved, root):
            return None, "inside_protected_dir"
    if sys.platform != "win32" and not os.access(resolved, os.X_OK):
        return None, "not_executable"
    return resolved, "ok"


def _search_path(name: str, forbidden: tuple[Path, ...]) -> Path | None:
    """PATH のディレクトリだけを探す (CWD を暗黙に探す ``shutil.which`` は使わない)。"""
    filename = f"{name}.exe" if sys.platform == "win32" else name
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or not Path(entry).is_absolute():
            continue
        found, _ = validate_runtime_path(name, str(Path(entry) / filename), forbidden=forbidden)
        if found is not None:
            return found
    return None


def resolve_runtime(name: str, cfg: dict | None = None) -> tuple[Path | None, str]:
    """``create.runtimes.<name>`` → 無ければ PATH。見つからなければ (None, 理由)。"""
    forbidden = forbidden_roots()
    configured = str((((cfg or {}).get("create") or {}).get("runtimes") or {}).get(name) or "")
    if configured.strip():
        found, reason = validate_runtime_path(name, configured, forbidden=forbidden)
        if found is None:
            logger.warning("runtimes: create.runtimes.%s is not usable (%s)", name, reason)
        return found, reason if found is None else "configured"
    found = _search_path(name, forbidden)
    return (found, "path") if found is not None else (None, "not_found")


def run_runtime(
    executable: Path, args: tuple[str, ...], *, cwd: Path, timeout_sec: float,
) -> tuple[int | None, str]:
    """実行環境を固定の引数で起動する。戻り値は (終了コード, stdout+stderr)。起動できなければ None。"""
    env = {k: os.environ[k] for k in _MINIMAL_ENV_VARS if k in os.environ}
    try:
        proc = subprocess.run(  # noqa: S603 - 解決・検証済みの実行ファイルを shell なしで固定引数で起こす
            [str(executable), *args], cwd=str(cwd), env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout_sec, shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout_sec:.0f}s"
    except OSError as exc:
        return None, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def runtime_version(name: str, executable: Path, *, timeout_sec: float = 10.0) -> str:
    """版の 1 行目 (``PHP 8.4.25 (cli) …`` / ``v24.14.0``)。取れなければ空文字列。"""
    code, out = run_runtime(
        executable, _VERSION_ARGS.get(name, ("--version",)), cwd=executable.parent, timeout_sec=timeout_sec,
    )
    if code != 0:
        return ""
    return next((line.strip() for line in out.splitlines() if line.strip()), "")


__all__ = [
    "RUNTIME_NAMES",
    "forbidden_roots",
    "resolve_runtime",
    "run_runtime",
    "runtime_version",
    "validate_runtime_path",
]
