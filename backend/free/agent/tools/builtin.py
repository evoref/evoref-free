"""ビルトインツール群の定義と登録"""

from __future__ import annotations

import asyncio
import ast
import calendar
import locale
import os
import re
import socket
import subprocess
import sys
from ipaddress import ip_address
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from backend.free.llm.utils import extract_content
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.history.history_manager import HistoryManager
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.llm.local_client import LocalClient

logger = get_logger("agent.tools.builtin")

# 最後の run_command 全文出力バッファ（/page コマンド用）
_last_full_output: str = ""
_last_full_output_lines: int = 0

# 出力切り詰めマーカー / exit code マーカー（共通定数モジュールから import）
from backend.free.constants import COMMAND_EXIT_CODE_PREFIX, TRUNCATION_MARKER

# read_file / search_code が読み込むファイルの上限サイズ。models/ (GGUF, 数十GB) 等の
# 巨大バイナリを誤って text として全文読み込みすると、CPython の GIL が長時間手放されず
# 別スレッド実行下でもイベントループを事実上ブロックする (実インシデントで確認済み)。
# 通常のソースファイルはこれより十分小さい。
_TOOL_MAX_FILE_READ_BYTES = 2_000_000

# 安全な計算用に許可するノード
_SAFE_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.USub, ast.UAdd,
}

# 非許可ノードごとの自己修正ヒント。LLM が同一ターン内でエラーを見て
# 書き直せるよう、そのノードが生じがちな典型的な誤記法を指す (実インシデント:
# 「πr²」を "π*5^2" と書いて BitXor に、「GCD」を "gcd(360,504)" と書いて
# Call になり、いずれもエラー後は手計算にフォールバックしていた)。
_DISALLOWED_NODE_HINTS: dict[str, str] = {
    "BitXor": "use ** for exponentiation, not ^ (^ is bitwise XOR here)",
    "Call": (
        "function calls (e.g. gcd(), sqrt()) are not supported; "
        "compute the result manually step-by-step instead"
    ),
    "Name": (
        "symbolic constants (e.g. pi, e) are not supported; "
        "inline the numeric value instead (e.g. 3.14159)"
    ),
}


def calculate(expression: str) -> str:
    """数式を安全に計算する"""
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            node_name = type(node).__name__
            if type(node) not in _SAFE_NODES:
                msg = f"Error: Unsafe expression (disallowed node: {node_name})"
                hint = _DISALLOWED_NODE_HINTS.get(node_name)
                if hint:
                    msg += f" -- {hint}"
                return msg
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def read_file(file_path: str) -> str:
    """ファイルの内容を読み込む"""
    p = Path(file_path)
    if not p.exists():
        return f"Error: File not found: {file_path}"
    if not p.is_file():
        return f"Error: Not a file: {file_path}"
    try:
        size = p.stat().st_size
        if size > _TOOL_MAX_FILE_READ_BYTES:
            return (
                f"Error: file too large to read ({size} bytes, "
                f"limit {_TOOL_MAX_FILE_READ_BYTES}): {file_path}"
            )
        content = p.read_text(encoding="utf-8")
        # 大きすぎるファイルは切り詰め
        if len(content) > 50000:
            return content[:50000] + "\n\n... (truncated, file too large)"
        return content
    except Exception as e:
        return f"Error: {e}"


def verify_syntax(file_path: str) -> str:
    """Python ファイルの構文を検証する（__pycache__ を生成しない）"""
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    if not p.is_file():
        return f"Error: not a file: {file_path}"
    if p.suffix != ".py":
        return f"Error: not a Python file: {file_path}"
    try:
        import py_compile
        import os
        import tempfile
        # cfile を一時ファイルに指定し、__pycache__ の生成を防止
        fd, tmp_pyc = tempfile.mkstemp(suffix=".pyc")
        os.close(fd)
        try:
            py_compile.compile(str(p), cfile=tmp_pyc, doraise=True)
        finally:
            # 一時 .pyc ファイルを削除
            try:
                os.unlink(tmp_pyc)
            except OSError:
                pass
        return f"Syntax OK: {file_path}"
    except py_compile.PyCompileError as e:
        return f"Syntax error: {e}"
    except Exception as e:
        return f"Error: {e}"


# リッチ(バイナリ)文書形式: プレーンテキスト書込みでは壊れたファイルになるため、
# backend.export レジストリ経由で実体 (OOXML / OPF 等) を生成する。対応ライブラリ
# 未導入で Writer が利用不可な場合に「リッチ形式の意図」を検知して明示エラーに
# するための最小定数。
_EXPORT_DOC_EXTS = frozenset(
    {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"},
)


def _block_has_renderable_content(block) -> bool:
    """ContentBlock が文書に描画される実体を持つか (空行/水平線は False)。"""
    if block.type == "table":
        return bool(block.rows)
    if block.type == "list":
        return bool(block.items)
    if block.type == "hr":
        return False
    return bool((block.content or "").strip())


def _reject_unrenderable_rich_content(ext: str, content: str, export_content) -> str | None:
    """空 / コード文字列だけのリッチ文書を success と誤報せず明示エラーにする。

    取得テーブルを使う決定論経路が外れて LLM 生成に落ちた時、モデルが空文字列や
    「文書を作るコード」をそのまま返すと、見た目は valid だが中身の無い .pptx/.docx を
    "Written N bytes" として成功扱いしてしまう (報告された .txt 不具合の .pptx 版)。
    描画可能なブロックが皆無、または table が無くコードに見える本文を弾く。
    問題なければ ``None`` を返す。
    """
    blocks = export_content.blocks
    if not any(_block_has_renderable_content(b) for b in blocks):
        return (
            f"Error: '{ext}' output produced no document content. "
            "Provide the document body as Markdown (headings, paragraphs, tables)."
        )
    has_table = any(b.type == "table" and b.rows for b in blocks)
    if not has_table:
        from backend.free.agent.meta_cognitive_utils import text_looks_like_code
        if text_looks_like_code(content):
            return (
                f"Error: '{ext}' output looks like program code, not document "
                "content. Provide the document body as Markdown (a heading and a "
                "Markdown table), not a script that generates the file."
            )
    return None


# 月ごとの実日数上限 (年に依存しない固定値)。2 月は閏年誤検知を避けるため
# 寛容側に 29 まで許容する (2026 年は平年で 28 日までだが、他年での誤検知を防ぐ)。
_MONTH_MAX_DAYS = {m: calendar.mdays[m] for m in range(1, 13)}
_MONTH_MAX_DAYS[2] = 29


def _check_calendar_table(export_content) -> str | None:
    """月別カレンダー表に実在しない日付が含まれていないか検証する。

    「1月から12月」等の月単位カレンダーは各行の先頭セルが月番号 (1-12)、
    後続セルが日番号という構造で生成されがちだが、ローカル小型モデルは
    「何月は何日まで」という長距離の正確な記憶・計算が不得手で、31日までしか
    無い月に32以上、2月に30日以上の値が混入することがある (実運用で発生:
    2026年2月の行に31日まで埋まっていた)。値の新規作成はせず、実在しない
    日付が見つかった場合のみ再生成を促すエラーを返す。
    """
    for block in export_content.blocks:
        if block.type != "table" or len(block.rows) < 3:
            continue
        month_rows: list[tuple[int, list[str]]] = []
        for row in block.rows[1:]:
            if not row:
                continue
            try:
                month = int(str(row[0]).strip())
            except (ValueError, AttributeError):
                continue
            if 1 <= month <= 12:
                month_rows.append((month, row))
        # 月番号が重複なく昇順に 2 行以上並ぶ表のみ「月別カレンダー」とみなす
        # (無関係な数値表の誤検知を避けるための構造的シグナル)。
        months = [m for m, _ in month_rows]
        if len(months) < 2 or months != sorted(set(months)):
            continue
        bad_months: list[str] = []
        for month, row in month_rows:
            max_day = _MONTH_MAX_DAYS[month]
            for cell in row[1:]:
                try:
                    day = int(str(cell).strip())
                except (ValueError, AttributeError):
                    continue
                # 1-31 の範囲内の値のみ「日番号」候補として扱う (無関係な数値
                # 表 (売上高等) の誤検知を避けるため)。
                if 1 <= day <= 31 and day > max_day:
                    bad_months.append(f"{month}月 ({day}日 > {max_day}日まで)")
                    break
        if bad_months:
            return (
                "Error: calendar table contains dates that do not exist: "
                f"{', '.join(bad_months)}. Each month's day column must stop at "
                "its actual last day (February has at most 29 days; April, June, "
                "September, November have at most 30). Rewrite the table with "
                "the correct number of days for each month."
            )
    return None


def _write_rich_document(p: Path, content: str) -> str:
    """``.docx`` 等のリッチ形式を export フレームワーク経由で実ファイル化する。

    文書 Writer (python-docx 等) がグローバルレジストリに登録・利用可能なら
    OOXML 等を生成する。ライブラリ未導入で利用不可の場合は壊れたファイルを
    書かず、必要パッケージを案内する明示エラーを返す (fail clearly)。
    """
    from backend.export import get_writer_registry
    from backend.export.content_converter import ContentConverter

    ext = p.suffix.lower()
    registry = get_writer_registry()
    writer = registry.get_writer(ext)
    if writer is None or not writer.is_available():
        requires = ", ".join(writer.requires) if writer else "python-docx"
        return (
            f"Error: '{ext}' output requires {requires}. "
            "Install the package, or use a text format such as .txt / .md."
        )
    try:
        # raw_markdown のみでは XLSX/CSV writer が表データを抽出できない
        # (no_table_data になる)。ContentConverter で markdown を構造化ブロック
        # (table 等) に変換して渡す。from_markdown は raw_markdown も保持するため
        # docx/odt 等のプローズ系 writer は従来どおり動作する。
        export_content = ContentConverter.from_markdown(content)
        guard_error = _reject_unrenderable_rich_content(ext, content, export_content)
        if guard_error:
            return guard_error
        calendar_error = _check_calendar_table(export_content)
        if calendar_error:
            return calendar_error
        result = registry.write(export_content, p)
        return f"Written {result.size_bytes} bytes to {p}"
    except Exception as e:
        return f"Error: {e}"


def write_file(file_path: str, content: str) -> str:
    """ファイルに書き込む

    ``file_path`` が既存ディレクトリを指している場合はエラーを返す。
    以前は配下に ``output_<UTC>.txt`` を自動付与していたが、read/verify 指示が
    ツール判定で write と誤認された際に、配下へ生成内容を捏造投棄する原因と
    なっていたため廃止した。長文生成経路は書込み前に
    ``_resolve_long_form_target_path`` でディレクトリ→ファイル名を解決済みのため
    影響しない。ディレクトリ配下の点検は list_directory / read_file を使う。

    ``.docx`` 等のリッチ文書形式 (``_EXPORT_DOC_EXTS``) は export フレームワーク
    経由で実体を生成する。それ以外は UTF-8 テキストとして書き込む。
    """
    p = Path(file_path)
    if p.is_dir():
        return (
            f"Error: '{file_path}' is a directory, not a file. "
            "Provide a file path, or use list_directory/read_file to inspect it."
        )
    if p.suffix.lower() in _EXPORT_DOC_EXTS:
        return _write_rich_document(p, content)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        # len(content) は文字数であり、UTF-8 のマルチバイト文字 (日本語等) を
        # 含むと実バイト数と乖離する。"bytes" と明記するため実測する。
        return f"Written {len(content.encode('utf-8'))} bytes to {file_path}"
    except Exception as e:
        return f"Error: {e}"


def search_code(pattern: str, directory: str = ".", max_results: int = 20) -> str:
    """コードを正規表現パターンで検索する"""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex: {e}"

    results: list[str] = []
    base = Path(directory)
    if not base.exists():
        return f"Error: Directory not found: {directory}"

    for root, dirs, files in os.walk(base):
        # 隠しディレクトリ・一般的な除外 + モデル/ローカルデータ (数十GB級バイナリ/
        # 大量の実行時データを含み、ソースコードではない) をスキップ
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in {"node_modules", "__pycache__", ".git", "models", "local"}
        ]
        for fname in files:
            fpath = Path(root) / fname
            try:
                if fpath.stat().st_size > _TOOL_MAX_FILE_READ_BYTES:
                    continue
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{fpath}:{i}: {line.strip()}")
                        if len(results) >= max_results:
                            return "\n".join(results) + f"\n... (limited to {max_results} results)"
            except (OSError, UnicodeDecodeError):
                continue

    if not results:
        return f"No matches found for pattern: {pattern}"
    return "\n".join(results)


def list_directory(directory: str = ".", max_depth: int = 3) -> str:
    """ディレクトリ構造を表示する"""
    base = Path(directory)
    if not base.exists():
        return f"Error: Directory not found: {directory}"

    lines: list[str] = []
    _walk_tree(base, lines, prefix="", depth=0, max_depth=max_depth)

    if not lines:
        return "(empty directory)"
    return "\n".join(lines[:200])


def _walk_tree(path: Path, lines: list[str], prefix: str, depth: int, max_depth: int) -> None:
    """ディレクトリツリーを再帰的に構築"""
    if depth > max_depth:
        return
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    skip_dirs = {".git", "node_modules", "__pycache__", ".svelte-kit", "dist"}
    for i, entry in enumerate(entries):
        if entry.name.startswith(".") and entry.is_dir():
            continue
        if entry.name in skip_dirs:
            continue
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{prefix}{connector}{entry.name}{suffix}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            _walk_tree(entry, lines, prefix + extension, depth + 1, max_depth)


def apply_diff(file_path: str, diff_text: str) -> str:
    """unified diff をファイルに適用する

    失敗時は原文を保持し、エラーメッセージを返す。
    """
    p = Path(file_path)
    if not p.exists():
        return f"Error: File not found: {file_path}"

    original = p.read_text(encoding="utf-8")

    try:
        result = subprocess.run(
            ["patch", "-p0", "--no-backup-if-mismatch"],
            input=diff_text,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=p.parent,
        )
        if result.returncode == 0:
            return f"Diff applied successfully to {file_path}"
        # 適用失敗: 原文を復元
        p.write_text(original, encoding="utf-8")
        return f"Error: Diff failed: {result.stderr.strip()}"
    except FileNotFoundError:
        return "Error: 'patch' command not found. Install GNU patch."
    except subprocess.TimeoutExpired:
        p.write_text(original, encoding="utf-8")
        return "Error: Diff application timed out"


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


# fetch_url で除去する HTML タグ（ノイズ源）。
# 注: 追加分は void 要素 (input/source/area 等) を避ける。stdlib フォールバック
# (_strip_html_fallback) は終了タグで skip 深度を戻すため、void 要素を入れると
# 深度が戻らず以降が全て欠落する。bs4 経路 (本番) は void を正しく扱う。
_STRIP_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "iframe", "form", "svg", "meta", "link",
    "button", "select", "textarea", "label", "template", "dialog",
    "picture", "video", "audio", "canvas",
]


def _strip_html_fallback(html: str) -> str:
    """BeautifulSoup なしで HTML タグを除去するフォールバック

    stdlib の html.parser を使い、タグ構造を正確にパースする。
    """
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._result: list[str] = []
            self._skip_depth = 0  # スキップ中のタグのネスト深度

        def handle_starttag(self, tag: str, attrs):  # noqa: ARG002
            if tag.lower() in _STRIP_TAGS:
                self._skip_depth += 1

        def handle_endtag(self, tag: str):
            if tag.lower() in _STRIP_TAGS and self._skip_depth > 0:
                self._skip_depth -= 1

        def handle_data(self, data: str):
            if self._skip_depth == 0:
                stripped = data.strip()
                if stripped:
                    self._result.append(stripped)

        def get_text(self) -> str:
            return "\n".join(self._result)

    extractor = _TextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        # パース失敗時は最低限の正規表現フォールバック
        import re as _re
        text = _re.sub(r"<[^>]+>", "", html)
        text = _re.sub(r"\n\s*\n", "\n", text)
        return text.strip()
    return extractor.get_text()


_FETCH_URL_USER_AGENT = "evoref-fetch/1.0"
_FETCH_URL_MAX_BYTES = 5_000_000  # 5 MB (raw body) — DoS 抑止
_FETCH_URL_MAX_REDIRECTS = 5
_FETCH_URL_ALLOWED_SCHEMES = ("http", "https")
_FETCH_URL_TEXT_CONTENT_TYPE_PREFIXES = (
    "text/",
    "application/xhtml",
    "application/xml",
    "application/json",
)
# fetch_url 結果のプロンプト合流時の最大文字数。
# 20_000 ではベース LLM のプリフィルが 30〜50 秒に達して
# フロント側 SSE chunk timeout を引き起こしていたため 8_000 に抑制。
_FETCH_URL_MAX_TEXT_CHARS = 8_000
# fetch_url の戻り値がこの文字数を超えた場合、アシストモデルで
# 要約してからベース LLM に渡す。assist_client が未注入の場合は
# 従来通り truncate のみで返す (degraded mode 安全縮退)。
_FETCH_URL_SUMMARIZE_THRESHOLD = 4_000
# fetch_url 要約プロンプト (英語固定でアシストモデルへ指示)。
_FETCH_URL_SUMMARIZE_SYSTEM = (
    "You are a precise summarizer for retrieved web content. "
    "Summarize the user-provided text in 500-1000 characters while preserving "
    "important facts, numbers, names, and dates. Output the summary directly "
    "without markdown fences, headings, or meta-commentary."
)
# 表を含むページは行データの取りこぼしを避けるため truncate 上限を引き上げる。
# 表はトークンが短く、メタ認知ループのツール結果として消費されるため、ベース LLM の
# プリフィル懸念 (8000 制限の理由) は当てはまりにくい。
_FETCH_URL_MAX_TEXT_CHARS_TABLE = 40_000
# GFM テーブルの区切り行 (``| --- |``)。fetch 結果に表が含まれるかの検出に使う。
_MD_TABLE_SEP_LINE_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$", re.MULTILINE)


def _contains_markdown_table(text: str) -> bool:
    """テキストに GFM テーブル (区切り行付き) が含まれるか判定する。"""
    return bool(_MD_TABLE_SEP_LINE_RE.search(text))


# ── fetch_url 本文抽出ヒューリスティック ──────────────────────────────
# class/id/role/aria-label がボイラープレート (nav/menu/footer/breadcrumb 等) を
# 示す要素を除去するための境界アンカー正規表現。短語 (ad/ads) の誤爆 (address 等)
# を避けるため前後を区切り文字/端でアンカーする。
_BOILERPLATE_ATTR_RE = re.compile(
    r"(?:^|[\s_-])(?:"
    r"nav|navbar|navigation|globalnav|gnav|subnav|menu|"
    r"footer|contentinfo|header|masthead|breadcrumb|breadcrumbs|"
    r"sidebar|widget|banner|advert|advertisement|ads?|adsbygoogle|"
    r"promo|cookie|consent|gdpr|social|share|sns|related|recommend|"
    r"pager|pagination|toc|skiplink|utility|copyright|legal|disclaimer"
    r")(?:$|[\s_-])",
    re.IGNORECASE,
)
# リンク密度判定の対象ブロックタグと閾値 (ナビ/メニュー/リンク一覧の駆除)。
_LINK_DENSE_BLOCK_TAGS = ("ul", "ol", "div", "section")
_LINK_DENSITY_THRESHOLD = 0.6
_LINK_DENSITY_MIN_LINKS = 4
_LINK_DENSITY_MIN_TEXT = 40
# 本文コンテナ候補 (存在すればここへスコープを絞る)。
_MAIN_CONTENT_SELECTORS = ("main", "article", "[role=main]", "#main", "#content", "#main-content")
# ヒューリスティックが naive のこの比率未満しか残さない場合は過剰除去とみなし
# naive へ退避する (本文があるのに空を返さないためのセーフティネット)。
_EXTRACTION_MIN_RETAIN_RATIO = 0.10


def _extract_naive(html: str) -> str:
    """現行どおりの素朴抽出 (タグ名 strip → get_text)。比較・退避用。"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_STRIP_TAGS):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        logger.warning("bs4 not available, falling back to stdlib HTML parser")
        return _strip_html_fallback(html)


def _has_boilerplate_attr(tag) -> bool:
    """tag の class/id/role/aria-label がボイラープレートを示すか。"""
    name = getattr(tag, "name", None)
    if name in (None, "html", "body", "[document]"):
        return False
    parts: list[str] = []
    cls = tag.get("class")
    if cls:
        parts.append(" ".join(cls) if isinstance(cls, list) else str(cls))
    for attr in ("id", "role", "aria-label"):
        val = tag.get(attr)
        if val:
            parts.append(str(val))
    if not parts:
        return False
    return bool(_BOILERPLATE_ATTR_RE.search(" ".join(parts)))


def _select_main_root(soup):
    """本文コンテナ (main/article 等) があればそれを、無ければ body を返す。"""
    for sel in _MAIN_CONTENT_SELECTORS:
        try:
            el = soup.select_one(sel)
        except Exception:
            el = None
        if el is not None and len(el.get_text(strip=True)) >= 200:
            return el
    return soup.body or soup


def _prune_link_dense_blocks(root) -> None:
    """リンク密度の高いブロック (ナビ/メニュー/リンク一覧) を除去する。"""
    for el in root.find_all(_LINK_DENSE_BLOCK_TAGS):
        try:
            if el.parent is None:  # 祖先 decompose 済みで既に分離
                continue
            text = el.get_text(strip=True)
            if len(text) < _LINK_DENSITY_MIN_TEXT:
                continue
            links = el.find_all("a")
            if len(links) < _LINK_DENSITY_MIN_LINKS:
                continue
            link_len = sum(len(a.get_text(strip=True)) for a in links)
            if link_len / max(len(text), 1) >= _LINK_DENSITY_THRESHOLD:
                el.decompose()
        except Exception:
            continue


def _flatten_tables(root) -> None:
    """<table> を GitHub-flavored Markdown 表へ置換する。

    ヘッダ行 + 区切り行 (``| --- |``) + 各データ行を前後パイプ付きで出力する。
    これにより fetch_url 結果中の表を ``ContentConverter.from_markdown`` が table
    ブロックとして解釈でき、取得 → xlsx 出力の経路が成立する。列数が不揃いな行は
    最大列数にパディングし、セル内の ``|`` はエスケープする。
    """
    for table in root.find_all("table"):
        try:
            if table.parent is None:
                continue
            rows: list[list[str]] = []
            for tr in table.find_all("tr"):
                cells = [
                    c.get_text(strip=True).replace("|", "\\|")
                    for c in tr.find_all(["th", "td"])
                ]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            ncol = max(len(r) for r in rows)
            md_lines: list[str] = []
            for idx, r in enumerate(rows):
                padded = r + [""] * (ncol - len(r))
                md_lines.append("| " + " | ".join(padded) + " |")
                if idx == 0:
                    md_lines.append("| " + " | ".join(["---"] * ncol) + " |")
            table.replace_with("\n" + "\n".join(md_lines) + "\n")
        except Exception:
            continue


def _collapse_blank_lines(text: str) -> str:
    """3 連以上の改行を 2 連に畳む。"""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_main_content(html: str) -> str:
    """ヒューリスティックで本文を抽出する (bs4 必須・各段は例外を投げない設計)。"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    for el in soup.find_all(_has_boilerplate_attr):
        try:
            el.decompose()
        except Exception:
            continue
    root = _select_main_root(soup)
    _prune_link_dense_blocks(root)
    _flatten_tables(root)
    text = root.get_text(separator="\n", strip=True)
    return _collapse_blank_lines(text)


def _html_to_text(html: str) -> str:
    """HTML を本文テキストへ変換する。

    ヒューリスティック抽出 (_extract_main_content) を試み、bs4 不在・例外・空・
    過剰除去 (naive 比 _EXTRACTION_MIN_RETAIN_RATIO 未満) の場合は naive 抽出へ
    退避する。本文があるのに空を返さないことを保証する。
    """
    naive = _extract_naive(html)
    try:
        import bs4  # noqa: F401
    except ImportError:
        return naive
    try:
        improved = _extract_main_content(html)
    except Exception as e:
        logger.warning("fetch_url heuristic extraction failed (%r); using naive", e)
        return naive
    if not improved.strip():
        return naive
    if naive and len(improved) < len(naive) * _EXTRACTION_MIN_RETAIN_RATIO:
        logger.info(
            "fetch_url heuristic extraction too aggressive (%d << %d chars); using naive",
            len(improved), len(naive),
        )
        return naive
    return improved


def _validate_fetch_url(url: str, *, allow_private_ip: bool) -> str | None:
    """fetch_url の URL を検証する。エラー時はユーザー向け文字列、安全なら None。

    theme_installer.install_from_url の検証パターンをミラー。
    """
    parsed = urlparse(url)
    if parsed.scheme not in _FETCH_URL_ALLOWED_SCHEMES:
        return f"Error: Unsupported URL scheme: {parsed.scheme!r}"
    if not parsed.hostname:
        return "Error: URL has no hostname"
    if allow_private_ip:
        return None
    try:
        resolved = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        return f"Error: Failed to resolve hostname {parsed.hostname!r}: {e}"
    for _, _, _, _, sockaddr in resolved:
        addr = ip_address(sockaddr[0])
        if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
            return (
                "Error: Access to private/reserved addresses is not allowed "
                "(set tools.fetch_url_allow_private_ip: true to override)"
            )
    return None


def _redact_url_for_log(url: str) -> str:
    """ログ用に URL のクエリ文字列・fragment を除去 (PII / token 漏洩対策)"""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


async def fetch_url(
    url: str,
    timeout: int = 10,
    *,
    allow_private_ip: bool = False,
) -> str:
    """URL を取得してテキスト化する"""
    if not url:
        return "Error: URL is required"

    err = _validate_fetch_url(url, allow_private_ip=allow_private_ip)
    if err:
        return err

    import httpx as _httpx

    body_bytes = bytearray()
    truncated = False
    encoding = "utf-8"
    try:
        async with _httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=_FETCH_URL_MAX_REDIRECTS,
            timeout=timeout,
            headers={"User-Agent": _FETCH_URL_USER_AGENT},
        ) as client:
            async with client.stream("GET", url) as r:
                r.raise_for_status()
                ctype = r.headers.get("content-type", "").lower()
                if not any(ctype.startswith(p) for p in _FETCH_URL_TEXT_CONTENT_TYPE_PREFIXES):
                    return f"Error: Unsupported content-type: {ctype or '(none)'}"
                encoding = r.encoding or "utf-8"
                async for chunk in r.aiter_bytes():
                    body_bytes.extend(chunk)
                    if len(body_bytes) >= _FETCH_URL_MAX_BYTES:
                        truncated = True
                        break
    except Exception as e:
        logger.warning("fetch_url failed: url=%s err=%r", _redact_url_for_log(url), e)
        return f"Error fetching URL ({type(e).__name__}): {e}"

    html_text = bytes(body_bytes).decode(encoding, errors="replace")

    # 本文抽出: ボイラープレート (nav/menu/footer/リンク一覧) を除去して
    # 本文を分離する。bs4 不在・過剰除去時は naive 抽出へ安全に退避。
    text = _html_to_text(html_text)

    # 表を含むページは行の取りこぼしを防ぐため truncate 上限を引き上げる。
    cap = (
        _FETCH_URL_MAX_TEXT_CHARS_TABLE
        if _contains_markdown_table(text)
        else _FETCH_URL_MAX_TEXT_CHARS
    )
    if len(text) > cap:
        text = text[:cap] + "\n... (truncated)"
    if truncated:
        text += f"\n... (response body truncated at {_FETCH_URL_MAX_BYTES} bytes)"
    return text


def _make_fetch_url(cfg: dict, assist_client: "AssistModelClient | None" = None):
    """fetch_url ツールハンドラを生成（config / assist_client をクロージャでバインド）

    ``assist_client`` が与えられている場合、戻り値が
    ``_FETCH_URL_SUMMARIZE_THRESHOLD`` を超えていれば
    アシストモデル (purpose=``summarize``) で要約してから返す。
    要約失敗・assist_client=None の場合は truncate 済み原文で安全縮退する。
    """
    tools_cfg = cfg.get("tools", {})
    default_timeout = int(tools_cfg.get("fetch_url_timeout", 10))
    allow_private_ip = bool(tools_cfg.get("fetch_url_allow_private_ip", False))

    async def _fetch_url(url: str, timeout: int = default_timeout) -> str:
        text = await fetch_url(
            url, timeout=timeout, allow_private_ip=allow_private_ip,
        )
        if (
            assist_client is None
            or text.startswith("Error")
            or len(text) <= _FETCH_URL_SUMMARIZE_THRESHOLD
            # 表を含む結果は要約するとセル/行が失われるため原文のまま返す。
            or _contains_markdown_table(text)
        ):
            return text
        try:
            messages = [
                {"role": "system", "content": _FETCH_URL_SUMMARIZE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"URL: {url}\n\nContent:\n{text}\n\n"
                        "Provide the summary now."
                    ),
                },
            ]
            result = await assist_client.generate(
                messages, stream=False, purpose="summarize",
            )
            summary = extract_content(result).strip()
            if summary:
                logger.info(
                    "fetch_url summarized: url=%s, %d -> %d chars",
                    _redact_url_for_log(url), len(text), len(summary),
                )
                return f"(Summary of {url})\n{summary}"
        except Exception as e:
            logger.warning(
                "fetch_url summarize failed: url=%s err=%r; "
                "falling back to truncated raw text",
                _redact_url_for_log(url), e,
            )
        return text

    return _fetch_url


def _make_summarize(client: LocalClient):
    """summarize ツールハンドラを生成（LocalClient をクロージャでバインド）"""

    async def summarize(text: str) -> str:
        """テキストを要約する"""
        messages = [
            {"role": "system", "content": "You are a summarization assistant. Summarize the given text concisely."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
        ]
        try:
            result = await client.generate(
                messages, stream=False, temperature=0.3,
                id_slot=client.background_slot,
            )
            return extract_content(result)
        except Exception as e:
            logger.error("summarize tool failed: %s", e)
            return f"Error: {e}"

    return summarize


def _make_translate(client: LocalClient):
    """translate ツールハンドラを生成（LocalClient をクロージャでバインド）"""

    async def translate(text: str, target_lang: str) -> str:
        """テキストを指定言語に翻訳する"""
        messages = [
            {"role": "system", "content": "You are a translation assistant. Translate the given text accurately."},
            {"role": "user", "content": f"Translate the following text to {target_lang}:\n\n{text}"},
        ]
        try:
            result = await client.generate(
                messages, stream=False, temperature=0.3,
                id_slot=client.background_slot,
            )
            return extract_content(result)
        except Exception as e:
            logger.error("translate tool failed: %s", e)
            return f"Error: {e}"

    return translate


def _make_draft_document(client: LocalClient):
    """draft_document ツールハンドラを生成（LocalClient をクロージャでバインド）"""

    async def draft_document(instruction: str, format: str = "markdown") -> str:
        """指示に基づいてドキュメントを生成する"""
        messages = [
            {"role": "system", "content": f"You are a document drafting assistant. Generate documents in {format} format."},
            {"role": "user", "content": instruction},
        ]
        try:
            result = await client.generate(
                messages, stream=False, temperature=0.7,
                id_slot=client.background_slot,
            )
            return extract_content(result)
        except Exception as e:
            logger.error("draft_document tool failed: %s", e)
            return f"Error: {e}"

    return draft_document


def _make_run_command(config: dict):
    """run_command ハンドラを生成（config をクロージャでバインド）

    ToolsRegistry 経由の呼び出しで config.agent.dangerous_command_block 設定を反映するため、
    config をクロージャで捕捉する（設計書 §6.8.8）。
    """

    async def wrapped(command: str, timeout: int = 30) -> str:
        return await run_command(command, timeout, config)

    return wrapped


def _make_run_command_readonly(config: dict):
    """run_command_readonly ハンドラを生成（config をクロージャでバインド）

    chat モードの executable query (時刻 / OS / スペック等) 専用。実行前に
    ``reject_readonly_violation`` で読み取り専用 (書込 / 削除 / 導入 /
    ネットワーク送信なし) を検証し、通過したコマンドのみ ``run_command``
    本体へ委譲する (危険コマンドガード / 対話コマンドガードは本体側で適用)。
    判定層のバグや将来コードの誤用があっても、chat から破壊コマンドを実行
    できない構造的保証をこのラッパが担う。
    """

    async def wrapped(command: str, timeout: int = 30) -> str:
        from backend.free.agent.safety_patterns import reject_readonly_violation

        reject = reject_readonly_violation(command)
        if reject is not None:
            logger.warning(
                "Readonly command rejected (%s): %s", reject, command[:100],
            )
            return f"Error: readonly violation: {reject}"
        return await run_command(command, timeout, config)

    return wrapped


def _make_search_history(manager: HistoryManager):
    """search_history ツールハンドラを生成（HistoryManager をクロージャでバインド）"""

    def search_history(query: str, mode: str | None = None, limit: int = 10,
                       date_from: str | None = None, date_to: str | None = None,
                       session_id: str | None = None) -> str:
        """過去の会話履歴を検索する

        ``session_id`` は LLM 向けツールスキーマには公開しない
        (ToolCallJudge がセッション自己参照質問に対してのみ code 側で
        強制注入する。LLM が任意の session_id を指定できると他セッションの
        意図的な絞り込み回避に使われかねないため)。
        """
        try:
            results = manager.search_sessions(
                query=query, mode=mode, limit=limit, search_turns=False,
                date_from=date_from, date_to=date_to, session_id=session_id,
            )
            # 結果が少ない場合のみターン検索で再検索
            if len(results) < limit:
                results = manager.search_sessions(
                    query=query, mode=mode, limit=limit, search_turns=True,
                    date_from=date_from, date_to=date_to, session_id=session_id,
                )
            if not results:
                return f"No results found for: {query}"

            lines: list[str] = []
            for r in results:
                header = f"[{r['started_at']}] mode={r['mode']} score={r['relevance_score']:.1f}"
                if r.get("summary"):
                    header += f" | {r['summary']}"
                lines.append(header)
                for turn in r.get("matched_turns", []):
                    lines.append(f"  turn#{turn['index']} ({turn['role']}): {turn['content_preview']}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("search_history tool failed: %s", e)
            return f"Error: {e}"

    return search_history


def register_builtin_tools(
    registry,
    config: dict | None = None,
    local_client: LocalClient | None = None,
    history_manager: HistoryManager | None = None,
    *,
    assist_client: "AssistModelClient | None" = None,
) -> None:
    """ビルトインツールをレジストリに一括登録

    ``assist_client`` が与えられた場合、fetch_url の戻り値が長文の場合に
    アシストモデル (purpose="summarize") で要約してから返す。
    None の場合は従来通り truncate のみで動作する。
    """
    cfg = config or {}

    registry.register(
        name="calculate",
        func=calculate,
        description=(
            "Evaluate a Python-syntax arithmetic expression safely (numeric "
            "literals + - * / % ** // and parentheses only)"
        ),
        parameters={
            "expression": {
                "type": "string",
                "description": (
                    "Use ** for exponentiation, NOT ^ (which is bitwise XOR in "
                    "this sandbox and will error). Function calls (e.g. gcd(), "
                    "sqrt()) and symbolic constants (e.g. pi, e) are NOT "
                    "supported -- inline the numeric value instead (e.g. 3.14159 "
                    "instead of pi), and compute functions like gcd manually "
                    "step-by-step rather than calling them."
                ),
            },
        },
    )

    registry.register(
        name="read_file",
        func=read_file,
        description="Read the contents of a file",
        parameters={
            "file_path": {"type": "string", "description": "Path to the file to read"},
        },
    )

    registry.register(
        name="write_file",
        func=write_file,
        description="Write content to a file (parent directories are created automatically, no mkdir needed)",
        parameters={
            "file_path": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        modes=["coding"],
    )

    registry.register(
        name="search_code",
        func=search_code,
        description="Search code files using a regex pattern",
        parameters={
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "directory": {"type": "string", "description": "Directory to search in"},
        },
        modes=["coding"],
    )

    registry.register(
        name="list_directory",
        func=list_directory,
        description="List the directory structure",
        parameters={
            "directory": {"type": "string", "description": "Directory to list"},
        },
    )

    registry.register(
        name="apply_diff",
        func=apply_diff,
        description="Apply a unified diff to a file",
        parameters={
            "file_path": {"type": "string", "description": "Path to the file to patch"},
            "diff_text": {"type": "string", "description": "Unified diff content"},
        },
        modes=["coding"],
    )

    registry.register(
        name="run_command",
        func=_make_run_command(cfg),
        description="Execute a shell command (CLI only)",
        parameters={
            "command": {"type": "string", "description": "Shell command to execute"},
        },
        modes=["coding"],
    )

    # chat モードの executable query (時刻 / OS / スペック等) 専用。
    # hidden=True で LLM プロンプトのツール一覧には出さず、tool_call_judge の
    # executable 経路 (_executable_tool_for_mode) がコード側から注入する。
    # mode ゲート (2026-07-18) を変えずに chat のシステム情報クエリを復活
    # させるための登録 (2026-07-21 回帰対策、docs/f_03_agent_engine.md §3.1)。
    registry.register(
        name="run_command_readonly",
        func=_make_run_command_readonly(cfg),
        description=(
            "Execute a read-only shell command for environment facts "
            "(injected by the tool judge; not directly selectable)"
        ),
        parameters={
            "command": {"type": "string", "description": "Read-only shell command to execute"},
        },
        modes=["chat"],
        hidden=True,
    )

    registry.register(
        name="verify_syntax",
        func=verify_syntax,
        description="Verify Python file syntax (checks existence and runs py_compile)",
        parameters={
            "file_path": {"type": "string", "description": "Path to the Python file to verify"},
        },
        modes=["coding"],
    )

    # fetch_url（デフォルト有効、config で無効化可能）
    if cfg.get("tools", {}).get("fetch_url_enabled", True):
        fetch_timeout = cfg.get("tools", {}).get("fetch_url_timeout", 10)
        registry.register(
            name="fetch_url",
            func=_make_fetch_url(cfg, assist_client=assist_client),
            description="Fetch a URL and extract text content",
            parameters={
                "url": {"type": "string", "description": "URL to fetch"},
                "timeout": {"type": "integer", "description": f"Request timeout (default: {fetch_timeout})"},
            },
        )

    # LLM ツール（summarize / translate / draft_document）
    if local_client is not None:
        registry.register(
            name="summarize",
            func=_make_summarize(local_client),
            description="Summarize the given text concisely",
            parameters={
                "text": {"type": "string", "description": "Text to summarize"},
            },
            modes=["chat"],
        )

        registry.register(
            name="translate",
            func=_make_translate(local_client),
            description="Translate text to a target language",
            parameters={
                "text": {"type": "string", "description": "Text to translate"},
                "target_lang": {"type": "string", "description": "Target language (e.g. 'English', 'Japanese')"},
            },
            modes=["chat"],
        )

        registry.register(
            name="draft_document",
            func=_make_draft_document(local_client),
            description="Generate a document based on instructions",
            parameters={
                "instruction": {"type": "string", "description": "Instructions for document generation"},
                "format": {"type": "string", "description": "Output format (e.g. 'markdown', 'plain')"},
            },
            modes=["chat"],
        )

    # 会話履歴検索ツール
    if history_manager is not None:
        registry.register(
            name="search_history",
            func=_make_search_history(history_manager),
            description="Search past conversation history by keyword and/or date range",
            parameters={
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords to search for, NOT the user's question verbatim "
                        "(e.g. use 'Rust' rather than 'what is the user's favorite "
                        "programming language?'). Extract the key noun(s)/proper "
                        "noun(s) from the request."
                    ),
                },
                "mode": {"type": "string", "description": "Filter by mode (chat/coding)"},
                "limit": {"type": "integer", "description": "Maximum number of results (default: 10)"},
                "date_from": {"type": "string", "description": "Start date in ISO 8601 format (e.g. '2026-03-01')"},
                "date_to": {"type": "string", "description": "End date in ISO 8601 format (e.g. '2026-03-31')"},
            },
        )

    logger.info("Registered %d builtin tools", registry.count)
