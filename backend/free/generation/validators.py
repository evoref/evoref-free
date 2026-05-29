"""コード検証（AST parse, import 整合性）

設計書 f_09_long_form_generation.md §7 準拠。
コード生成完了後にルールベースの静的検証を実施する。
テキスト生成には適用しない。
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass

PYTHON_BUILTINS: frozenset[str] = frozenset(dir(builtins))

# バッククォート 3 個以上の markdown コードフェンス区切り。言語タグ (python 等) は任意。
# CODE ユニットは区切り無しに連結されるため、閉じフェンス ``` と次ユニットの開きフェンス
# ```python が結合して ``````python (6 個) になったり、コード行末に ```python が結合する
# ことがある。`{3,}` で連数を問わず捕捉する。
# - 行全体がフェンスのみ → 行ごと削除
# - コード行末に結合したフェンス → 末尾フェンスのみ除去 (本文と改行は残す)
_FENCE_ONLY_LINE_RE = re.compile(r"^[ \t]*`{3,}[A-Za-z0-9_+\-.]*[ \t]*$")
_TRAILING_FENCE_RE = re.compile(r"[ \t]*`{3,}[A-Za-z0-9_+\-.]*[ \t]*$")


def remove_code_fences(code: str) -> str:
    """組み立て済みコードから markdown コードフェンス (```python / ``` / ``````python 等) を除去する。

    行全体がフェンスのみの行は丸ごと削除し、コード行末に結合したフェンス
    (例 ``ALIVE = 1```python``) は末尾フェンスのみ除去する。文字列リテラル内の
    ``` (例 ``x = "```"``) は行末に来ないため保持される。

    ``backend.free.llm.json_extract.strip_code_fences`` (単一ブロックの外側のみ剥がす
    JSON 用) とは別物。
    """
    out: list[str] = []
    for line in code.splitlines():
        if _FENCE_ONLY_LINE_RE.match(line):
            continue
        out.append(_TRAILING_FENCE_RE.sub("", line))
    return "\n".join(out)


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


def _target_names(target: ast.expr) -> set[str]:
    """代入 / ループ対象から束縛される名前を再帰的に抽出する。

    Tuple / List のアンパック (``a, b = ...`` / ``for a, b in ...``)、Starred
    (``a, *rest = ...``)、ネストした分解に対応する。Attribute / Subscript
    (``obj.x`` / ``d[k]``) は新規束縛ではないため無視する。
    """
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, ast.Starred):
        names |= _target_names(target.value)
    elif isinstance(target, ast.Tuple | ast.List):
        for elt in target.elts:
            names |= _target_names(elt)
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
                names |= _target_names(target)
        elif isinstance(node, ast.AnnAssign):
            names |= _target_names(node.target)
        elif isinstance(node, ast.NamedExpr):  # walrus (n := ...)
            names |= _target_names(node.target)
        elif isinstance(node, ast.For | ast.AsyncFor):
            names |= _target_names(node.target)
        elif isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                if item.optional_vars is not None:
                    names |= _target_names(item.optional_vars)
        elif isinstance(node, ast.comprehension):
            names |= _target_names(node.target)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:  # except ... as e
                names.add(node.name)
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
