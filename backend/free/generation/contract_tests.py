"""契約テスト — 骨組みの入出力例から pytest を組み、LLM が書いたテストを検査する (f_10 §11)。

staged v2 (Pro) のテスト工程の決定論部分。純粋関数だけを置く。

- :func:`build_example_tests`: 骨組みの ``examples`` (Python 式 → 期待値) から、LLM を使わずに
  pytest を組む。**合否を決めるのはこのテストだけ** (正の順序: 依頼 > 骨組み > コード > テスト)。
- :func:`lint_generated_tests`: LLM が書いた参考テストから、壊れやすい検査をするテスト関数を落とす。
  標準出力の差し替え (``sys.stdout = …``)、``__annotations__`` の比較、骨組みに無い文言との完全一致。
  2026-09-25: 生成テストが ``sys.stdout`` を StringIO に差し替えたまま戻さず、2 回目の確認で 2 行が
  溜まって落ちた。文言も骨組みに無い和文 (「引数は…」) と完全一致させていた。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ExampleCase:
    """組めた入出力例 1 件。"""

    module: str  # import 名 (拡張子なしのモジュール名)
    call: str
    expected: str


def _is_expr(text: str) -> bool:
    try:
        ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        return False
    return True


def _is_literal(text: str) -> bool:
    try:
        ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return False
    return True


_FILE_NAME_RE = re.compile(r"[\\/]|\.[A-Za-z0-9]{1,5}$")


def _touches_files(call: str) -> bool:
    """式がファイルを名指すか (``sum_column('data.csv', …)`` / ``open(…)``)。

    骨組みの例は純粋な呼出しに限る約束だが、モデルは実在しないファイルを引数に取る例を
    書く。そのままでは正しいコードが FileNotFoundError で「契約テスト不合格」になる
    (2026-09-25 create ベンチ s3 の偽の失敗)。
    """
    for node in ast.walk(ast.parse(call, mode="eval")):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _FILE_NAME_RE.search(node.value):
            return True
        if isinstance(node, ast.Name) and node.id == "open":
            return True
    return False


def usable_examples(skeleton: dict, module_paths: list[str]) -> list[ExampleCase]:
    """骨組みの例のうち、テストに組めるもの (式と期待値が構文として正しく、モジュールが実在し、
    ファイルを名指さない)。"""
    stems = {PurePosixPath(p).stem for p in module_paths if p.endswith(".py")}
    out: list[ExampleCase] = []
    for ex in skeleton.get("examples") or []:
        call = str(ex.get("call") or "").strip()
        expected = str(ex.get("expected") or "").strip()
        module = PurePosixPath(str(ex.get("module") or "")).stem
        if not call or not expected or module not in stems:
            continue
        if _is_expr(call) and _is_literal(expected) and not _touches_files(call):
            out.append(ExampleCase(module, call, expected))
    return out


def build_example_tests(cases: list[ExampleCase]) -> str:
    """入出力例の pytest ファイル本文 (例が無ければ空文字列)。"""
    if not cases:
        return ""
    lines = [
        '"""骨組み (契約) の入出力例から自動で組んだテスト (staged v2)。"""',
        "",
        "import importlib",
        "",
        "import pytest",
        "",
        "",
        "def _ns(name):",
        "    return dict(vars(importlib.import_module(name)))",
        "",
        "",
        "def _same(got, exp):",
        "    if isinstance(exp, float) or isinstance(got, float):",
        "        return got == pytest.approx(exp)",
        "    return got == exp",
        "",
        "",
        "def _printed(out):",
        "    # 戻り値が None の呼出し (main(['c_to_f', '100']) 等) は表示した最後の行を値として見る",
        "    last = (out.strip().splitlines() or [''])[-1].strip()",
        "    try:",
        "        return ast.literal_eval(last)",
        "    except (SyntaxError, ValueError):",
        "        return last",
    ]
    lines[2:2] = ["import ast", "import contextlib", "import io"]
    for i, case in enumerate(cases, start=1):
        lines += [
            "",
            "",
            f"def test_example_{i}():",
            "    buf = io.StringIO()",
            "    try:",
            "        with contextlib.redirect_stdout(buf):",
            f"            got = eval({case.call!r}, _ns({case.module!r}))",
            "    except OSError as exc:  # 例が環境 (ファイル等) に依存していた — 契約の判定から外す",
            "        pytest.skip(f'example depends on the environment: {exc}')",
            f"    exp = {case.expected}",
            "    if got is None and exp is not None:",
            "        got = _printed(buf.getvalue())",
            "    assert _same(got, exp), (got, exp)",
        ]
    return "\n".join(lines) + "\n"


def _test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            out.append(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            out += [
                m for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test")
            ]
    return out


def _brittle_reason(func: ast.AST, allowed_text: str) -> str:
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute) and target.attr in ("stdout", "stderr")
                    and isinstance(target.value, ast.Name) and target.value.id == "sys"
                ):
                    return "reassigns sys.stdout/sys.stderr"
        if isinstance(node, ast.Attribute) and node.attr == "__annotations__":
            return "compares __annotations__"
        if isinstance(node, ast.Compare) and any(isinstance(op, ast.Eq) for op in node.ops):
            for side in [node.left, *node.comparators]:
                if (
                    isinstance(side, ast.Constant) and isinstance(side.value, str)
                    and len(side.value.strip()) >= 8 and side.value.strip() not in allowed_text
                ):
                    return "asserts exact text that the contract does not specify"
    return ""


def lint_generated_tests(source: str, *, allowed_text: str) -> tuple[str, list[str]]:
    """壊れやすいテスト関数を落とした本文と、落とした理由の一覧を返す。

    ``allowed_text`` は骨組み (と依頼文) の全文。そこに現れる文言との一致は契約の一部として残す。
    構文エラーなら (空文字列, [理由])。残るテスト関数が 0 件なら本文は空文字列。
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return "", [f"syntax error: {exc}"]
    drop: list[tuple[int, int]] = []
    reasons: list[str] = []
    tests = _test_functions(tree)
    for func in tests:
        reason = _brittle_reason(func, allowed_text)
        if reason:
            start = min([func.lineno, *[d.lineno for d in func.decorator_list]])
            drop.append((start, func.end_lineno or func.lineno))
            reasons.append(f"{func.name}: {reason}")
    if len(drop) == len(tests):
        return "", reasons
    lines = source.splitlines()
    for start, end in sorted(drop, reverse=True):
        del lines[start - 1:end]
    return "\n".join(lines).rstrip() + "\n", reasons


__all__ = [
    "ExampleCase",
    "build_example_tests",
    "lint_generated_tests",
    "usable_examples",
]
