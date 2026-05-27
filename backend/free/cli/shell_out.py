"""シェルアウトモード: 対話的サブプロセスの TTY パススルー実行

設計書 09 §9.4.4 に基づき、vim・python REPL 等の対話的プロセスを
evoref CLI から起動する際に TTY を子プロセスに明け渡す。
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from typing import NamedTuple

from backend.log_config import get_logger

logger = get_logger("cli.shell_out")

# ────────────────────────────────────────────
# 対話的コマンド判定パターン
# ────────────────────────────────────────────

# コマンド先頭のプログラム名で判定するパターン
_INTERACTIVE_HEAD_PATTERNS: list[re.Pattern[str]] = [
    # エディタ
    re.compile(r"^(vim|vi|nvim|nano|emacs|pico|micro)\b"),
    # REPL（引数なし = 対話モード）
    re.compile(r"^python3?\s*$"),
    re.compile(r"^(ipython|bpython|ptpython)\b"),
    re.compile(r"^node\s*$"),
    re.compile(r"^(irb|pry)\s*$"),
    # ページャ・モニタ
    re.compile(r"^(less|more|man)\b"),
    re.compile(r"^(top|htop|btop|glances)\b"),
    # リモートシェル
    re.compile(r"^(ssh|telnet|ftp|sftp)\b"),
    # DB REPL
    re.compile(r"^(mysql|psql|sqlite3)\s*$"),
]

# コマンド全体にマッチするパターン（引数の組み合わせで判定）
_INTERACTIVE_FULL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bgit\b.*\brebase\b.*-i\b"),
    re.compile(r"\bgit\b.*-i\b.*\brebase\b"),
    re.compile(r"\bgit\b.*\bcommit\b(?!.*--message)(?!.*-m\b)"),
]


def is_interactive_command(cmd: str) -> bool:
    """コマンドが対話的プロセスかを判定する

    vim, nano, python REPL, git rebase -i 等、独自の TTY 制御を行う
    プロセスを検出する。
    """
    stripped = cmd.strip()
    if not stripped:
        return False

    for pattern in _INTERACTIVE_HEAD_PATTERNS:
        if pattern.search(stripped):
            return True

    for pattern in _INTERACTIVE_FULL_PATTERNS:
        if pattern.search(stripped):
            return True

    return False


# ────────────────────────────────────────────
# シェルアウト実行
# ────────────────────────────────────────────


class ShellOutResult(NamedTuple):
    """シェルアウト実行結果"""
    returncode: int
    executed: bool
    error: str


def shell_out(cmd: str) -> ShellOutResult:
    """対話的コマンドをシェルアウト実行する

    CLI の TTY を子プロセスに直接渡し、capture_output=False で実行する。
    textual App は呼び出し元で suspend 済みの前提。
    完了後は _restore_terminal() でターミナル状態を復旧する。
    """
    logger.debug("shell_out: executing cmd=%r", cmd)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            # capture_output=False（デフォルト）: stdin/stdout/stderr を継承
            # → 子プロセスが直接 TTY を使用できる
        )
        logger.debug(
            "shell_out: completed, returncode=%d", result.returncode,
        )
    except Exception as e:
        logger.error("shell_out: execution failed: %s", e)
        return ShellOutResult(returncode=-1, executed=False, error=str(e))

    # ターミナル状態の復旧
    _restore_terminal()

    return ShellOutResult(
        returncode=result.returncode,
        executed=True,
        error="",
    )


def _restore_terminal() -> None:
    """シェルアウト後のターミナル状態を復旧する

    対話的プロセス（特に vim 等）が終了した後、ターミナルの
    入出力設定が変わっている可能性がある。
    """
    if platform.system() == "Windows":
        # Windows: ConPTY が TTY 状態を自動管理するため復旧不要。
        # textual の app.suspend() / resume が画面復旧を担当する。
        logger.debug("shell_out: Windows — ConPTY auto-recovery, no action needed")
    else:
        # Unix/macOS: stty sane でターミナル設定を正常化
        try:
            subprocess.run(
                ["stty", "sane"],
                stdin=sys.stdin,
                check=False,
            )
        except (FileNotFoundError, OSError):
            logger.debug("shell_out: stty sane unavailable")
