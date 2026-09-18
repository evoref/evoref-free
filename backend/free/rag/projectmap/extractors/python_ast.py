"""tree-sitter 不在時の Python 専用縮退抽出器 (c_16 §4.4)

標準ライブラリの ``ast`` だけで class / function / method / import / call
を拾う。tree-sitter が使えない環境でも Python だけは構造グラフを作れる
ようにするための最小実装 — 出力形は
:mod:`backend.free.rag.projectmap.extractors.treesitter` と揃える。
"""

from __future__ import annotations

import ast

from backend.free.rag.projectmap.graph import ExtractedFile, Node, RawCall
from backend.free.rag.projectmap.ids import code_node_id
from backend.log_config import get_logger

logger = get_logger("rag.projectmap.python_ast")

_DEF_TYPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _base_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _Extractor(ast.NodeVisitor):
    """1 ファイル分の module を歩いて :class:`ExtractedFile` を組む。"""

    def __init__(self, path: str, source_lines: list[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.stack: list[ast.AST] = []
        self.entries: dict[int, dict[str, str]] = {}
        # 同じ qualname の 2 つ目以降は ``#<n>`` で区別する (treesitter.py と同じ規則)
        self.seen_qualnames: dict[str, int] = {}
        self.nodes: list[Node] = []
        self.imports: list[str] = []
        self.calls: list[RawCall] = []
        self.inherits: list[tuple[str, str]] = []

    def _line(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def _visit_def(self, node: ast.AST, kind_hint: str) -> None:
        parent = self.stack[-1] if self.stack else None
        parent_info = self.entries.get(id(parent)) if parent is not None else None
        if kind_hint == "function" and parent_info is not None and parent_info["node_type"] == "class":
            node_type = "method"
        else:
            node_type = kind_hint

        name = getattr(node, "name", "")
        qualname = f"{parent_info['qualname']}.{name}" if parent_info else name
        occurrence = self.seen_qualnames.get(qualname, 0) + 1
        self.seen_qualnames[qualname] = occurrence
        if occurrence > 1:
            qualname = f"{qualname}#{occurrence}"
        parent_id = parent_info["id"] if parent_info else None
        line_start = int(getattr(node, "lineno", 1))
        line_end = int(getattr(node, "end_lineno", line_start) or line_start)
        signature = self._line(line_start)
        doc = ast.get_docstring(node, clean=True) or ""
        doc_line = next((line.strip() for line in doc.splitlines() if line.strip()), "")
        text = f"{signature}\n{doc_line}" if doc_line else signature
        node_id = code_node_id(self.path, node_type, qualname)

        self.nodes.append(Node(
            id=node_id, node_type=node_type, path=self.path, name=name,
            qualname=qualname, lang="python", line_start=line_start,
            line_end=line_end, parent_id=parent_id, signature=signature, text=text,
        ))
        self.entries[id(node)] = {"id": node_id, "qualname": qualname, "node_type": node_type}

        if node_type == "class":
            for base in getattr(node, "bases", []):
                base_name = _base_name(base)
                if base_name:
                    self.inherits.append((node_id, base_name))

        self.stack.append(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 — ast.NodeVisitor 規約
        self._visit_def(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_def(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_def(node, "function")

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.imports.extend(alias.name for alias in node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        # 相対 import (``level > 0``) はプロジェクト内解決の規則を持たない
        # ため対象外 (tree-sitter 版と同じ割り切り)。
        if node.module and node.level == 0:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        callee = _call_name(node.func)
        if callee and self.stack:
            parent_info = self.entries.get(id(self.stack[-1]))
            if parent_info is not None:
                self.calls.append(RawCall(caller_id=parent_info["id"], callee_name=callee))
        self.generic_visit(node)


def extract_file(path: str, source: str) -> ExtractedFile | None:
    """Python ソースを ``ast`` だけで抽出する。構文エラーは ``None``。"""
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError) as e:
        logger.warning("python ast fallback: failed to parse %s: %s", path, e)
        return None

    extractor = _Extractor(path, source.splitlines())
    extractor.visit(tree)
    extractor.nodes.sort(key=lambda n: n.line_start)

    line_count = source.count("\n") + (0 if source.endswith("\n") else 1)
    return ExtractedFile(
        path=path, lang="python", line_count=max(line_count, 1),
        nodes=extractor.nodes, imports=extractor.imports,
        calls=extractor.calls, inherits=extractor.inherits,
    )


__all__ = ["extract_file"]
