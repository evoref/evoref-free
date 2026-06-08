"""コード生成物の整合検証 (静的整合ゲート + import スモークテスト)

設計書 f_08 §3.5 の「``validate_python`` は AST 解析のみで実行時 import エラー
(``ModuleNotFoundError``) や dataclass 引数不整合 (``TypeError``) を検出できない」
限界に対し、以下を提供する:

1. :func:`normalize_relative_imports` — flat 出力で解決不能な相対 import の決定的除去
   (除去後に ``import_wirer.wire_imports`` が兄弟定義から正しい絶対 import を再配線)。
2. :func:`check_integrity` — エントリポイント存在 / 残存相対 import の静的検査 (純粋)。
3. :func:`run_import_smoke` — temp dir + サブプロセス import で実行時エラーを捕捉。

(1)(2) は外部リソースを伴わない純粋処理で常時利用可能。(3) はサブプロセスを
起動するため呼出側が config (``long_form.code_smoke_test_enabled``) で gate する。
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.free.llm.json_schemas import CodeSpec

logger = logging.getLogger("backend.free.generation.smoke_validator")


def _stems(paths) -> set[str]:
    """ファイルパス集合からモジュール stem (拡張子・ディレクトリ除去) を返す。"""
    return {os.path.splitext(os.path.basename(p))[0] for p in paths}


def _resolve_entry_code(files: dict[str, str], entry_module: str) -> str | None:
    """エントリポイントモジュールに対応する生成ファイルの内容を返す (無ければ None)。"""
    if entry_module in files:
        return files[entry_module]
    ep_stem = os.path.splitext(os.path.basename(entry_module))[0]
    for path, code in files.items():
        if os.path.splitext(os.path.basename(path))[0] == ep_stem:
            return code
    return None


def normalize_relative_imports(files: dict[str, str]) -> dict[str, str]:
    """flat 構成で解決不能な相対 import (``from .x import`` / ``from . import``) を除去する。

    生成物は同一ディレクトリの flat な ``.py`` 群として配信されるため、パッケージ
    前提の相対 import は実行時に ``ImportError: attempted relative import with no
    known parent package`` を引き起こす (今回の LINE アプリ失敗の主因)。除去後に
    ``import_wirer.wire_imports`` が兄弟ファイルのトップレベル定義から正しい絶対
    import を再配線する。除去は AST の行範囲ベースで決定的に行い、構文エラーの
    ファイルは触らない (AST 検証側が扱う)。
    """
    out: dict[str, str] = {}
    for path, code in files.items():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            out[path] = code
            continue
        drop_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.level or 0) > 0:
                end = getattr(node, "end_lineno", node.lineno) or node.lineno
                drop_lines.update(range(node.lineno, end + 1))
        if not drop_lines:
            out[path] = code
            continue
        kept = [
            line
            for i, line in enumerate(code.splitlines(keepends=True), start=1)
            if i not in drop_lines
        ]
        out[path] = "".join(kept)
    return out


def check_integrity(
    files: dict[str, str],
    spec: CodeSpec | None = None,
) -> list[str]:
    """生成物の残存整合問題を静的に検出する (サブプロセス不要)。

    - flat 構成に残存する相対 import (``normalize_relative_imports`` 後の取りこぼし)。
    - ``spec.entry_point`` が指定されているのに対応ファイルが無い / ``__main__``
      ガードが無い (起動不能なプログラム = 今回の LINE アプリ失敗の一因)。
    """
    errors: list[str] = []

    for path, code in files.items():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue  # AST 検証 (validate_python) 側が扱う
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.level or 0) > 0:
                errors.append(
                    f"{path}: 相対 import が残存 (flat 構成では解決不可)"
                )
                break

    if spec is not None and spec.entry_point and spec.entry_point.module:
        ep = spec.entry_point.module
        entry_code = _resolve_entry_code(files, ep)
        if entry_code is None:
            errors.append(f"エントリポイント {ep} が生成物に存在しない")
        elif "__main__" not in entry_code:
            errors.append(
                f"エントリポイント {ep} に if __name__ == '__main__' ガードが無い"
            )

    return errors


@dataclass
class SmokeResult:
    """import スモークテストの結果。

    - ``errors``: 生成物自身の実行時エラー (cross-file の ModuleNotFoundError /
      相対 import / dataclass 引数違い等)。修正対象。
    - ``warnings``: 外部依存未インストール / タイムアウト等の環境要因。修正不要。
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# サブプロセス内で各モジュールを import し、失敗を JSON で報告する runner。
# __main__ は実行しない (import のみ)。
_SMOKE_RUNNER = (
    "import importlib, json, sys\n"
    "mods = json.loads(sys.argv[1])\n"
    "out = []\n"
    "for m in mods:\n"
    "    try:\n"
    "        importlib.import_module(m)\n"
    "    except BaseException as e:\n"
    "        out.append({'module': m, 'type': type(e).__name__, 'msg': str(e)})\n"
    "print(json.dumps(out))\n"
)


def _classify_failure(failure: dict, stems: set[str]) -> tuple[str, bool]:
    """import 失敗を (メッセージ, is_error) に分類する。

    外部依存の ``ModuleNotFoundError`` (生成物 stem 以外) は環境要因として
    warning 扱い、それ以外 (cross-file 欠落 / 相対 import / 実行時例外) は error。
    """
    module = failure.get("module", "?")
    etype = failure.get("type", "Error")
    msg = failure.get("msg", "")
    if etype == "ModuleNotFoundError":
        missing = ""
        # "No module named 'X'" から X を抽出
        if "'" in msg:
            missing = msg.split("'")[1].split(".")[0]
        if missing and missing not in stems:
            return (f"{module}: 外部依存 '{missing}' が未インストール", False)
    return (f"{module}: {etype}: {msg}", True)


def run_import_smoke(
    files: dict[str, str],
    timeout_sec: float = 10.0,
    python_exe: str | None = None,
) -> SmokeResult:
    """生成ファイルを temp dir に書き出し、各モジュールを import してエラーを収集する。

    ``__main__`` は実行せず import のみ行う。top-level 副作用 (サーバ起動等) は
    ``timeout_sec`` で有界化する。書込/実行不可・タイムアウト時は warning に倒し、
    静的ゲートのみで継続する (生成失敗にはしない)。
    """
    result = SmokeResult()
    py_files = {p: c for p, c in files.items() if p.endswith(".py")}
    if not py_files:
        return result
    stems = _stems(py_files)
    exe = python_exe or sys.executable

    try:
        with tempfile.TemporaryDirectory(prefix="evoref_smoke_") as tmp:
            for path, code in py_files.items():
                # ネストパスは basename へ潰す (flat 配信前提)
                with open(os.path.join(tmp, os.path.basename(path)), "w",
                          encoding="utf-8") as fh:
                    fh.write(code)
            proc = subprocess.run(
                [exe, "-c", _SMOKE_RUNNER, json.dumps(sorted(stems))],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
    except subprocess.TimeoutExpired:
        result.warnings.append(
            f"import スモークテストが {timeout_sec:.0f}s でタイムアウト "
            "(top-level 副作用の可能性)"
        )
        return result
    except Exception as e:  # noqa: BLE001 — 実行不可は warning に倒す
        result.warnings.append(f"import スモークテスト実行不可: {e}")
        return result

    try:
        failures = json.loads(proc.stdout.strip() or "[]")
    except (ValueError, TypeError):
        result.warnings.append(
            f"import スモークテスト出力を解析できません: {proc.stderr[:200]}"
        )
        return result

    seen: set[str] = set()
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        message, is_error = _classify_failure(failure, stems)
        if message in seen:
            continue
        seen.add(message)
        (result.errors if is_error else result.warnings).append(message)
    return result
