"""evorefmem CLI 多重起動防止ファイルロック

``scripts/evorefmem_cli.py`` の同時実行 (および ``evoref serve`` プロセスとの
並走) によって ``facts.jsonl`` / ``.idx`` / ``embeddings/`` への
race condition が起こるのを防ぐための軽量な PID ベースロック。

``scripts/safe_pytest.py`` の同パターン (LOCK_PATH + PID 検証) を踏襲する。

## ロックファイルの場所

``local/.evorefmem_cli.lock`` (リポジトリルート相対)。
``local/`` は他の lockfile (``.safe_pytest.lock``) と同居する想定。

## プロセス検出

ロックファイルに記録された PID が **生きているか** を OS 別に確認し、
死んでいたロックは回収する (stale lock の自動奪取)。これにより
クラッシュした CLI が ``release_cli_lock()`` を呼べずに残したロックも
次回起動時に解消される。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[5]
"""リポジトリルート (``backend/free/memory/semantic/cli/lock.py`` の 5 階層上)。"""

CLI_LOCK_PATH: Final[Path] = REPO_ROOT / "local" / ".evorefmem_cli.lock"
"""evorefmem CLI 多重起動防止ファイルのパス。"""


class CliLockError(RuntimeError):
    """別の evorefmem CLI プロセスが既にロックを保持している。"""


def _pid_alive(pid: int) -> bool:
    """``pid`` のプロセスが現在実行中かを OS 別に判定する。

    ``safe_pytest.py::_pid_alive`` と同じ実装。Windows では tasklist、
    POSIX では ``os.kill(pid, 0)`` のシグナル 0 (存在確認のみ) を使う。
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # Windows の tasklist は OS locale (CP932 等) で出力するため、
        # text=True にすると UTF-8 想定でデコード失敗する。bytes で受けて
        # PID 文字列を bytes 比較する。
        try:
            raw = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}"],
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
        return str(pid).encode("ascii") in raw
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_cli_lock(lock_path: Path | None = None) -> Path:
    """CLI ロックを取得する。

    既存の lock ファイルがあり、PID が生きていれば :class:`CliLockError` を送出。
    死んでいるロック (stale) は奪取して上書きする。

    Args:
        lock_path: 上書き用 (テスト fixture 等)。None なら :data:`CLI_LOCK_PATH`。

    Returns:
        実際にロックファイルを書き出したパス。

    Raises:
        CliLockError: 別プロセスが lock を保持中。
    """
    path = lock_path if lock_path is not None else CLI_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old_pid = int(path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            old_pid = -1
        if old_pid > 0 and _pid_alive(old_pid):
            raise CliLockError(
                f"another evorefmem CLI session (pid={old_pid}) is still "
                f"running. lock={path}",
            )
    path.write_text(str(os.getpid()), encoding="utf-8")
    return path


def release_cli_lock(lock_path: Path | None = None) -> bool:
    """CLI ロックを解放する。

    既に削除されていても OK (idempotent)。

    Args:
        lock_path: 上書き用。None なら :data:`CLI_LOCK_PATH`。

    Returns:
        実際に削除した場合 ``True``、元から存在しなければ ``False``。
    """
    path = lock_path if lock_path is not None else CLI_LOCK_PATH
    if not path.exists():
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


__all__ = [
    "CLI_LOCK_PATH",
    "CliLockError",
    "acquire_cli_lock",
    "release_cli_lock",
]
