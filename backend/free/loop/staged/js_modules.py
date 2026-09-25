"""JavaScript (ES モジュール) の export / import を字句で読む (純粋関数)。

staged の兄弟 API 注入と生成後の照合は Python の AST 専用で、JS のファイルは
要約が空になり、兄弟の export を知らないまま書かれていた。実インシデント
(2026-09-22 実機 K04): ``storage.js`` は ``Storage`` クラスを export したのに、
``app.js`` / ``ui.js`` は ``loadTodos`` / ``saveTodos`` を import し、ES モジュールの
読み込みで起動不能だった。構文の抽出 (宣言名) なので正規表現で足りる
(不変則 #14 の対象外)。
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

#: 兄弟 API を読む対象の拡張子。
JS_MODULE_SUFFIXES = frozenset({".js", ".mjs", ".ts"})

_EXPORT_DECL_RE = re.compile(
    r"^\s*export\s+(?:default\s+)?(?:async\s+)?"
    r"(?:function\s*\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_EXPORT_LIST_RE = re.compile(r"^\s*export\s*\{([^}]*)\}", re.MULTILINE)
_EXPORT_DEFAULT_RE = re.compile(r"^\s*export\s+default\b", re.MULTILINE)
#: ``import {a, b as c} from './x.js'`` / ``import X, {a} from "./x.js"``。
_NAMED_IMPORT_RE = re.compile(
    r"^\s*import\s+(?:[A-Za-z_$][\w$]*\s*,\s*)?\{([^}]*)\}\s*from\s*['\"]\./([^'\"]+)['\"]",
    re.MULTILINE,
)


def is_js_module(path: str) -> bool:
    return PurePosixPath(path.replace("\\", "/")).suffix.lower() in JS_MODULE_SUFFIXES


def js_exports(code: str) -> set[str]:
    """export される名前 (``default`` を含む)。"""
    names = set(_EXPORT_DECL_RE.findall(code or ""))
    for group in _EXPORT_LIST_RE.findall(code or ""):
        for item in group.split(","):
            item = item.strip()
            if item:
                names.add(item.split(" as ")[-1].strip())
    if _EXPORT_DEFAULT_RE.search(code or ""):
        names.add("default")
    return names


_DECL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function\s*\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)[^\n]*$",
    re.MULTILINE,
)
#: クラス本体のメソッド定義行 (``  loadTodos(key) {`` / ``  async save(items) {``)。
_METHOD_RE = re.compile(
    r"^\s+(?:static\s+)?(?:async\s+)?(?!if\b|for\b|while\b|switch\b|catch\b|function\b)"
    r"([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{\s*$",
    re.MULTILINE,
)

#: 末尾の ``export { A, B };`` / ``export default name;`` の行。
_EXPORT_LIST_LINE_RE = re.compile(
    r"^\s*export\s*(?:\{[^}]*\}|default\s+[A-Za-z_$][\w$]*)\s*;?\s*$", re.MULTILINE,
)


def js_export_summary(code: str, max_lines: int = 40) -> str:
    """兄弟 API ブロック用に、export される名前の宣言行とメソッド行を並べる (本文は含めない)。

    ``export { Storage }`` のように末尾の一覧で export する書き方では一覧の行だけ
    では型が分からないので、export 名の宣言行と、クラスのメソッド定義行も含める。
    """
    code = code or ""
    exported = js_exports(code)
    lines: list[str] = []
    for m in _DECL_RE.finditer(code):
        if m.group(1) in exported:
            lines.append(m.group(0).strip().rstrip("{").rstrip())
    lines += [
        "  " + m.group(0).strip().rstrip("{").rstrip() for m in _METHOD_RE.finditer(code)
    ]
    lines += [line.strip() for line in _EXPORT_LIST_LINE_RE.findall(code)]
    return "\n".join(lines[:max_lines])


def relative_js_imports(code: str) -> list[str]:
    """名前付き import の相対指定子 (``./y.js``、document 順)。staged v2 の参照関係 (f_10 §12.3)。"""
    return [f"./{src}" for _group, src in _NAMED_IMPORT_RE.findall(code or "")]


def missing_js_imports(code: str, siblings: dict[str, str]) -> list[str]:
    """``./兄弟`` から名前付き import した名前のうち、兄弟が export していないもの。

    兄弟がまだ生成されていない (``siblings`` に無い) import は判定しない。
    """
    by_name = {PurePosixPath(p.replace("\\", "/")).name: c for p, c in siblings.items()}
    problems: list[str] = []
    for group, src in _NAMED_IMPORT_RE.findall(code or ""):
        target = by_name.get(PurePosixPath(src).name)
        if target is None:
            continue
        exported = js_exports(target)
        for item in group.split(","):
            name = item.strip().split(" as ")[0].strip()
            if name and name not in exported:
                problems.append(
                    f"'{name}' is not exported by {src} "
                    f"(its exports: {', '.join(sorted(exported)) or 'none'})"
                )
    return problems
