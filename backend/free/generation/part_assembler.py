"""部分生成されたコード片を 1 ファイルへ決定論的に結合する (LLM 不使用)。

staged コーディングの部分ごと生成 (spec の component 単位で base モデルが
生成したコード片群) を、以下の決定論手順で 1 つの Python ファイルへ結合する:

1. コードフェンスの防御的除去 (delegate 側で除去済みだが冪等)
2. ``from __future__ import`` を全部分から収集し、先頭部分の docstring 直後へ
   一意化して 1 回だけ配置 (mid-file の ``__future__`` は SyntaxError になるため
   正当性要件)
3. top-level import を bound-name 単位で初出優先 dedup し、先頭部分へ hoist
   (:mod:`import_wirer` の部品を再利用。相対 import / ``import *`` は据え置き)
4. 部分本体を spec 順のまま連結
5. ``if __name__ == "__main__":`` ガードが複数あれば最後の 1 個のみ残す
6. :func:`smoke_validator.dedup_top_level_defs` で部分間の同名 def/class 重複を
   除去 (LLM が他 component を再掲した場合の吸収)
7. 最終 ``ast.parse`` 検証。失敗なら None (呼出側が単発生成へフォールバック)

旧 CodeUnit 方式の失敗 (lossy 中間 JSON / import・定義重複) の回避策として、
結合は本モジュールの決定論処理のみで行い LLM を一切使わない (f_10 §9 禁則)。
"""

from __future__ import annotations

import ast

from backend.free.generation.import_wirer import (
    _insert_imports,
    _iter_top_level_imports,
    _strip_import_lines,
)
from backend.free.generation.smoke_validator import dedup_top_level_defs
from backend.free.generation.validators import remove_code_fences
from backend.log_config import get_logger

logger = get_logger("generation.part_assembler")


def _is_main_guard(node: ast.stmt) -> bool:
    """トップレベル ``if __name__ == "__main__":`` 判定 (左右反転も許容)。"""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    sides = [test.left, *test.comparators]
    has_name = any(
        isinstance(s, ast.Name) and s.id == "__name__" for s in sides
    )
    has_main = any(
        isinstance(s, ast.Constant) and s.value == "__main__" for s in sides
    )
    return has_name and has_main


def _strip_imports_and_collect(
    code: str,
) -> tuple[str, list[str], list[tuple[str, str]]]:
    """top-level import を除去し (__future__ 機能名, (bound, stmt) 列) を返す。

    ``ast.parse`` 不能な部分は無変換で素通しする (最終検証で弾かれる)。
    相対 import と ``import *`` は意味論を変えないため据え置く。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, [], []

    futures: list[str] = []
    imports: list[tuple[str, str]] = list(_iter_top_level_imports(tree))
    strip_nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            strip_nodes.append(node)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                futures.extend(a.name for a in node.names if a.name != "*")
                strip_nodes.append(node)
            elif (node.level or 0) == 0 and not any(
                a.name == "*" for a in node.names
            ):
                strip_nodes.append(node)
    if strip_nodes:
        code = _strip_import_lines(code, strip_nodes)
    return code, futures, imports


def _drop_duplicate_main_guards(code: str) -> str:
    """複数の ``__main__`` ガードを最後の 1 個だけ残して除去する。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    guards = [n for n in tree.body if _is_main_guard(n)]
    if len(guards) <= 1:
        return code
    drop: set[int] = set()
    for node in guards[:-1]:
        drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    kept = [
        line
        for i, line in enumerate(code.splitlines(keepends=True), start=1)
        if i not in drop
    ]
    return "".join(kept)


def _import_module_token(stmt: str) -> str:
    """単文 import 文字列 (``import X`` / ``from M import N``) のモジュール先頭セグメント。

    ``_iter_top_level_imports`` が生成する正規形のみを想定した決定論パース。
    """
    tokens = stmt.split()
    if not tokens:
        return ""
    module = tokens[1] if len(tokens) > 1 else ""
    return module.split(".")[0]


def assemble_file_parts(
    parts: list[str], *, module_stem: str | None = None,
) -> str | None:
    """部分コード群を 1 ファイルへ決定論結合する。失敗時は None。

    ``module_stem`` (結合先ファイルの stem) を渡すと、**自モジュールからの
    import** (``from breakout import Ball`` を breakout.py 自身に書く) を結合時に
    除去する。部分ごと生成で LLM が他 part の component を import 参照する誤解
    から生じ、実行時に partially-initialized ImportError で起動不能になるため
    (定義は同一ファイル内にあり import は常に不要。2026-07-06 live)。
    """
    cleaned = [remove_code_fences(p.strip()).strip() for p in parts]
    cleaned = [p for p in cleaned if p]
    if not cleaned:
        return None

    stripped: list[str] = []
    futures: list[str] = []
    merged_imports: dict[str, str] = {}  # bound-name -> stmt (初出優先)
    dropped_self_imports = 0
    for part in cleaned:
        body, part_futures, part_imports = _strip_imports_and_collect(part)
        stripped.append(body.strip())
        for name in part_futures:
            if name not in futures:
                futures.append(name)
        for bound, stmt in part_imports:
            if module_stem and _import_module_token(stmt) == module_stem:
                dropped_self_imports += 1
                continue
            merged_imports.setdefault(bound, stmt)
    if dropped_self_imports:
        logger.info(
            "part assembly dropped %d self-import(s) of module '%s'",
            dropped_self_imports, module_stem,
        )

    header_lines: list[str] = []
    if futures:
        header_lines.append(
            "from __future__ import " + ", ".join(sorted(futures))
        )
    header_lines.extend(merged_imports.values())

    first = stripped[0]
    if header_lines:
        try:
            first_tree = ast.parse(first)
        except SyntaxError:
            return None
        first = _insert_imports(first, first_tree, header_lines)

    merged = "\n\n\n".join([first.rstrip(), *[p.rstrip() for p in stripped[1:]]])
    if not merged.endswith("\n"):
        merged += "\n"

    merged = _drop_duplicate_main_guards(merged)
    merged = dedup_top_level_defs(merged)

    try:
        ast.parse(merged)
    except SyntaxError as exc:
        logger.warning("part assembly produced invalid Python: %s", exc)
        return None
    return merged
