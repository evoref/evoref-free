"""複数ファイル生成コードの import 自動補完 (cross-file 配線 + stdlib 伝播)。

複数ファイルへ分割生成された Python コードは、各ファイルが兄弟ファイル定義の
シンボル (クラス/関数/型) や共有 stdlib import を参照しているのに相互 import が
無く NameError で起動できない。本モジュールは AST ベースの決定的変換で各ファイルへ
不足 import を補う:

- cross-file: 兄弟ファイルのトップレベル定義を参照 → ``from <module> import <name>``。
- stdlib/3rd-party 伝播: 集合内のいずれかが import 済みの名前 → その import 文を伝播。

LLM は使わない。``ast.parse`` 失敗ファイルは無変更で据え置く (防御的)。import 解決を
直すのみで実行時ロジックの正否は対象外。フラットな兄弟タブ前提 (module 名 = ファイル
stem)。

``needed`` 集合は「全 Load 参照 − **トップレベル**定義 − import 済 − builtins」で求める
(``validators._extract_defined_names`` の全スコープ減算は使わない)。これにより関数内の
free 参照を確実に拾い、兄弟 export 名が別関数ローカルと衝突しても import を取りこぼさ
ない。誤検知時は未使用 import (警告のみ・無害) で済み、壊れたコードは生成しない。
"""

from __future__ import annotations

import ast
import sys
from pathlib import PurePosixPath

from backend.free.generation.validators import (
    PYTHON_BUILTINS,
    _extract_imported_names,
    _extract_used_names,
)
from backend.log_config import get_logger

logger = get_logger("generation.import_wirer")

# stdlib モジュール名 (py3.10+)。幻覚 import 除去の際に実在 stdlib を誤除去しないため。
_STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ()))


def _top_level_defs(tree: ast.Module) -> set[str]:
    """モジュールのトップレベル定義名 (import 可能な export) を抽出する。"""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Tuple | ast.List):
                    names.update(
                        elt.id for elt in target.elts if isinstance(elt, ast.Name)
                    )
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.TypeAlias):  # PEP 695 (py3.12+)
            if isinstance(node.name, ast.Name):
                names.add(node.name.id)
    return names


def _iter_top_level_imports(tree: ast.Module):
    """トップレベル import を ``(bound_name, 単一名 import 文)`` で yield する。

    多名 import 文は alias ごとに 1 名へ分解する。``__future__`` と相対 import
    (level>0) は対象外 (フラット前提)。
    """
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                stmt = f"import {alias.name}"
                if alias.asname:
                    stmt += f" as {alias.asname}"
                yield bound, stmt
        elif isinstance(node, ast.ImportFrom):
            if (node.level or 0) > 0 or node.module == "__future__":
                continue
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                stmt = f"from {module} import {alias.name}"
                if alias.asname:
                    stmt += f" as {alias.asname}"
                yield bound, stmt


def _insertion_index(tree: ast.Module) -> int:
    """import を挿入する 0-based 行 index を返す。

    先頭のモジュール docstring と ``from __future__`` import を読み飛ばした直後
    (= 最初の実文の手前)。実文がデコレータ付き定義でも、その手前に挿入されるため
    デコレータを割らない。
    """
    insert_line = 0  # = 先頭に残す行数 (1-based lineno 相当)
    for node in tree.body:
        is_docstring = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if is_docstring or is_future:
            insert_line = node.end_lineno or insert_line
        else:
            break
    return insert_line


def _insert_imports(code: str, tree: ast.Module, import_lines: list[str]) -> str:
    """``import_lines`` を docstring/__future__ の直後へ挿入した新コードを返す。"""
    idx = _insertion_index(tree)
    lines = code.splitlines()
    block = list(import_lines)
    if idx < len(lines):
        block.append("")  # 本体との区切り
    new_lines = lines[:idx] + block + lines[idx:]
    result = "\n".join(new_lines)
    if code.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def _is_bogus_sibling_import(
    node: ast.stmt,
    sibling_modules: set[str],
    symbol_modules: dict[str, set[str]],
) -> bool:
    """``from <flat_module> import a, b`` が「実在しない兄弟モジュール参照」かを判定する。

    LLM がフラットな snake_case のローカル風モジュール名 (例 ``game_engine_constants``)
    を幻覚し、その import 名が実際には別の兄弟ファイルで定義されている、というパターン
    を検出する。除去対象とするのは以下を**すべて**満たす場合のみ:

    - ``from <module> import ...`` (相対 / ``__future__`` / dotted は対象外)。
    - ``<module>`` が実在の兄弟 stem でない (= 正しい sibling import ではない)。
    - ``<module>`` が stdlib モジュール名でない (実在 stdlib を誤除去しない)。
    - import される**全名**が他の兄弟ファイルで top-level 定義されている (= 除去後に
      ``wire_imports`` が正しい ``from <real_module> import`` を再配線できる)。

    この保守的条件により、幻覚 import が ``import_map`` を汚染して正しい cross-file 配線を
    阻害する問題を解消しつつ、実在の外部 import の誤除去を避ける。
    """
    if not isinstance(node, ast.ImportFrom):
        return False
    module = node.module
    if (node.level or 0) > 0 or not module or "." in module:
        return False
    if module in sibling_modules or module in _STDLIB_MODULES:
        return False
    names = [alias.name for alias in node.names if alias.name != "*"]
    if not names:
        return False
    return all(symbol_modules.get(name) for name in names)


def _strip_import_lines(code: str, nodes: list[ast.stmt]) -> str:
    """``nodes`` の各文をソース行ごと削除した新コードを返す (1-based lineno 範囲)。"""
    drop: set[int] = set()
    for node in nodes:
        end = getattr(node, "end_lineno", None) or node.lineno
        drop.update(range(node.lineno, end + 1))
    lines = code.splitlines()
    kept = [ln for i, ln in enumerate(lines, start=1) if i not in drop]
    result = "\n".join(kept)
    if code.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def _safe_parse(code: str) -> ast.Module | None:
    """``ast.parse`` 失敗時は ``None`` (防御的)。"""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def wire_imports(files: dict[str, str]) -> dict[str, str]:
    """複数ファイル生成コードへ cross-file / stdlib import を補完する。

    ``len(files) <= 1`` は無変更。``ast.parse`` 失敗ファイルは据え置き、表からも
    除外する。冪等 (補完済みを再適用しても増えない)。

    補完の前に「実在しない兄弟モジュールからの幻覚 import」を除去する pre-pass を
    走らせる。これをしないと幻覚 import が ``import_map`` を汚染し、その名前を必要とする
    他ファイルの正しい cross-file 配線が ``name not in import_map`` ガードで skip されて
    しまう (実測: 1 つの bogus import が複数ファイルの配線を阻害する)。
    """
    if len(files) <= 1:
        return files

    work = dict(files)
    parsed: dict[str, ast.Module | None] = {
        path: _safe_parse(code) for path, code in work.items()
    }
    module_of = {path: PurePosixPath(path).stem for path in files}
    sibling_modules = set(module_of.values())

    # name -> それを top-level 定義する module 集合 (import 可能な export)。
    # prune は import のみ除去し定義は変えないため、prune 前に確定してよい。
    symbol_modules: dict[str, set[str]] = {}
    for path, tree in parsed.items():
        if tree is None:
            continue
        mod = module_of[path]
        for name in _top_level_defs(tree):
            symbol_modules.setdefault(name, set()).add(mod)

    # pre-pass: 幻覚 sibling import を除去 (import_map 構築前)。
    for path, tree in parsed.items():
        if tree is None:
            continue
        bogus = [
            node for node in tree.body
            if _is_bogus_sibling_import(node, sibling_modules, symbol_modules)
        ]
        if bogus:
            work[path] = _strip_import_lines(work[path], bogus)
            parsed[path] = _safe_parse(work[path])
            logger.debug(
                "import_wirer: %s から %d 件の幻覚 import を除去", path, len(bogus),
            )

    # bound_name -> 伝播用 import 文 (stdlib/3rd-party)。prune 後の import から構築し
    # 汚染を回避する。
    import_map: dict[str, str] = {}
    for path, tree in parsed.items():
        if tree is None:
            continue
        for bound, stmt in _iter_top_level_imports(tree):
            import_map.setdefault(bound, stmt)

    out = dict(work)
    for path, tree in parsed.items():
        if tree is None:
            continue
        mod = module_of[path]
        needed = (
            _extract_used_names(tree)
            - _top_level_defs(tree)
            - _extract_imported_names(tree)
            - PYTHON_BUILTINS
        )
        cross: dict[str, set[str]] = {}  # source module -> names
        propagated: set[str] = set()
        for name in needed:
            others = symbol_modules.get(name, set()) - {mod}
            if len(others) == 1 and name not in import_map:
                cross.setdefault(next(iter(others)), set()).add(name)
            elif not others and name in import_map:
                propagated.add(import_map[name])
            # それ以外 (曖昧: len(others)>1 / project+stdlib 両立 / external) は skip

        new_lines: list[str] = [
            f"from {src} import {', '.join(sorted(cross[src]))}"
            for src in cross
        ]
        new_lines.extend(propagated)
        if not new_lines:
            continue
        out[path] = _insert_imports(work[path], tree, sorted(new_lines))
        logger.debug(
            "import_wirer: %s に %d 行の import を補完", path, len(new_lines),
        )
    return out
