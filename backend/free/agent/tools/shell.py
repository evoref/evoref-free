"""``run_command`` ツール — 危険コマンドガード付きの非同期シェル実行

``/page`` 用の全文出力バッファもここが持つ (切り詰めた出力の実体)。
"""

from __future__ import annotations

import asyncio
import locale
import re
import subprocess
import sys

from pathlib import Path
from backend.log_config import get_logger
from backend.free.constants import (
    COMMAND_EXIT_CODE_PREFIX,
    TRUNCATION_MARKER,
)

logger = get_logger("agent.tools.builtin")


# 最後の run_command 全文出力バッファ（/page コマンド用）
_last_full_output: str = ""
_last_full_output_lines: int = 0
async def run_command(command: str, timeout: int = 30, config: dict | None = None) -> str:
    """シェルコマンドを非同期で実行する（危険コマンドガード + 対話的コマンドガード付き）

    config['agent']['dangerous_command_block'] が true（デフォルト）の場合、
    DANGEROUS_PATTERNS にマッチするコマンドの実行をブロックする。

    対話的コマンド（vim, python REPL 等）は TTY パススルーが必要なため、
    バックエンド側では実行せずにシェルアウト要求メッセージを返す（設計書 09 §9.4.4）。

    長時間実行プロセス（GUI アプリ等）はタイムアウト後もプロセスを終了させず
    バックグラウンドで継続させる。
    """
    cfg = config or {}
    if cfg.get("agent", {}).get("dangerous_command_block", True):
        from backend.free.agent.safety_patterns import DANGEROUS_PATTERNS
        from backend.i18n_helper import msg

        if any(re.search(p, command) for p in DANGEROUS_PATTERNS):
            logger.warning("Dangerous command blocked: %s", command[:100])
            return msg("agent.dangerous_command_blocked", command=command[:80])

    # 対話的コマンドガード（設計書 09 §9.4.4）
    from backend.free.cli.shell_out import is_interactive_command

    if is_interactive_command(command):
        logger.info("Interactive command detected, shell-out required: %s", command[:100])
        from backend.i18n_helper import msg
        return msg("agent.interactive_command_blocked", command=command[:80])

    # mkdir コマンドは OS 差異を吸収するため Python で実行
    mkdir_match = re.match(r'^\s*(?:mkdir(?:\s+-p)?)\s+(.+)$', command)
    if mkdir_match:
        return _mkdir_safe(mkdir_match.group(1).strip().strip('"').strip("'"))

    return await _run_command_async_impl(command, timeout)


def _mkdir_safe(dir_path: str) -> str:
    """mkdir を exist_ok=True で安全に実行する（Windows/Unix 差異を吸収）"""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return f"Directory ensured: {dir_path}"
    except Exception as e:
        return f"Error: {e}"


def _decode_subprocess_output(raw: bytes) -> str:
    """子プロセス出力をロケール依存で安全にデコードする。

    Windows の子プロセス (cmd / git / python 等) は OEM コードページ
    (日本語環境では cp932) で出力するため、utf-8 固定 decode では日本語が
    mojibake 化する。OEM/mbcs → ロケール推奨エンコーディング → utf-8(replace)
    の順でフォールバックする (cli/pid_manager._decode_windows_console_output と同方針)。
    """
    if not raw:
        return ""
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


async def _run_command_async_impl(cmd: str, timeout: int = 30) -> str:
    """シェルコマンドの非同期実行本体

    stdlib subprocess.Popen をワーカースレッド経由で呼び出すことでイベントループを
    ブロックしない。タイムアウト時はプロセスを終了させず、バックグラウンドで継続させる。

    asyncio.create_subprocess_shell を使わないのは、Windows の ProactorEventLoop が
    生成するパイプトランスポートが、タイムアウト時に Process オブジェクトを放置する
    と GC 時 (loop 終了後) に `_ProactorBasePipeTransport.__del__` から
    `ValueError: I/O operation on closed pipe` を投げるため
    stdlib のパイプは __del__ で警告を投げないので安全に放置できる。
    """
    global _last_full_output, _last_full_output_lines
    try:
        proc = subprocess.Popen(  # shell=True は設計上必要
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.to_thread(proc.communicate), timeout=timeout,
            )
        except asyncio.TimeoutError:
            # プロセスがタイムアウト内に終了しなかった（GUI/デーモン/長時間処理）
            # プロセスは終了させずバックグラウンドで継続させる
            logger.info(
                "Process still running after %ds, left in background: PID=%s, cmd=%s",
                timeout, proc.pid, cmd[:100],
            )
            _last_full_output = ""
            _last_full_output_lines = 0
            return (
                f"Process (PID {proc.pid}) is still running after {timeout}s. "
                "It was left running in the background."
            )

        output = _decode_subprocess_output(stdout_bytes)
        if stderr_bytes:
            output += f"\n[stderr] {_decode_subprocess_output(stderr_bytes)}"
        if proc.returncode != 0:
            output += f"\n{COMMAND_EXIT_CODE_PREFIX} {proc.returncode}]"
        # 出力を切り詰め（設計書 09 §9.5.3: 200行超で先頭100行+末尾100行）
        lines = output.splitlines()
        if len(lines) > 200:
            # 全文を /page 用バッファに保存
            _last_full_output = output
            _last_full_output_lines = len(lines)
            head = lines[:100]
            tail = lines[-100:]
            skipped = len(lines) - 200
            output = "\n".join(head) + f"\n\n… ({skipped}{TRUNCATION_MARKER}) …\n\n" + "\n".join(tail)
        else:
            _last_full_output = ""
            _last_full_output_lines = 0
        return output or "(no output)"
    except Exception as e:
        return f"Error: {e}"


def get_last_full_output() -> tuple[str, int]:
    """最後の run_command の全文出力を返す（/page コマンド用）

    Returns:
        (output, total_lines): 全文と行数。切り詰めが無かった場合は空文字列。
    """
    return _last_full_output, _last_full_output_lines


def clear_last_full_output() -> None:
    """全文出力バッファをクリア"""
    global _last_full_output, _last_full_output_lines
    _last_full_output = ""
    _last_full_output_lines = 0
