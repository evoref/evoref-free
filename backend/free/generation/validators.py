"""コード検証（AST parse, import 整合性）

設計書 f_09_long_form_generation.md §7 準拠。
コード生成完了後にルールベースの静的検証を実施する。
テキスト生成には適用しない。
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass

PYTHON_BUILTINS: frozenset[str] = frozenset(dir(builtins))


@dataclass
class ValidationError:
    """検証エラー"""

    error_type: str
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"[{self.severity}:{self.error_type}] {self.message}"


def _extract_used_names(tree: ast.Module) -> set[str]:
    """AST から使用されている名前を抽出"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
    return names


def _extract_defined_names(tree: ast.Module) -> set[str]:
    """AST から定義されている名前を抽出"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Tuple | ast.List):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    names.add(item.optional_vars.id)
        elif isinstance(node, ast.comprehension):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    # 関数引数
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                names.add(arg.arg)
            if node.args.vararg:
                names.add(node.args.vararg.arg)
            if node.args.kwarg:
                names.add(node.args.kwarg.arg)
    return names


def _extract_imported_names(tree: ast.Module) -> set[str]:
    """AST から import された名前を抽出"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname if alias.asname else alias.name)
    return names


def validate_python(assembled_code: str) -> list[ValidationError]:
    """Python コードの静的検証（LLM不要）"""
    errors: list[ValidationError] = []

    # 1. 構文検証
    try:
        tree = ast.parse(assembled_code)
    except SyntaxError as e:
        errors.append(
            ValidationError("syntax", f"line {e.lineno}: {e.msg}")
        )
        return errors

    # 2. 未定義名の検出
    used = _extract_used_names(tree)
    defined = _extract_defined_names(tree)
    imported = _extract_imported_names(tree)
    undefined = used - defined - imported - PYTHON_BUILTINS
    if undefined:
        errors.append(
            ValidationError("undefined", f"Undefined names: {sorted(undefined)}")
        )

    # 3. 未使用 import の検出（警告レベル）
    unused_imports = imported - used
    if unused_imports:
        errors.append(
            ValidationError(
                "unused_import",
                f"Unused imports: {sorted(unused_imports)}",
                severity="warning",
            )
        )

    return errors
