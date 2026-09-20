"""HTML / CSS / SCSS の抽出 (c_16 §4.4 マークアップ / SFC)

関数もクラスも持たない言語は **file ノード + ``imports`` 辺だけ**で載せる
(ノード型・辺型は増やさない)。HTML は ``<script src>`` / ``<link href>``、
CSS / SCSS は ``@import`` / ``@use`` / ``@forward`` / ``url()`` を指定子として
読む。呼出側 (``builder.py`` / ``sfc.py``) が辺の解決 (``graph.py``) を行う。
"""

from __future__ import annotations

from typing import Any

from backend.free.rag.projectmap.extractors.html_tree import (
    attr_value,
    child_of_type,
    get_parser,
    iter_nodes,
    node_text,
)
from backend.free.rag.projectmap.graph import ExtractedFile

#: CSS/SCSS の import 系 at-rule (c_16 §4.4)。
_STYLESHEET_STATEMENT_TYPES: frozenset[str] = frozenset({
    "import_statement", "use_statement", "forward_statement",
})


def _line_count(source: bytes) -> int:
    count = source.count(b"\n") + (0 if source.endswith(b"\n") else 1)
    return max(count, 1)


def _string_content(parent: Any, source: bytes) -> str | None:
    """``parent`` 直下の ``string_value`` の中身を読む。

    CSS grammar の ``string_value`` は引用符に挟まれた ``string_content`` を
    子に持つが、SCSS grammar (tree-sitter-language-pack 1.20、実測
    2026-09-20) は ``@use``/``@forward``/``@import`` の引数・``url()`` の
    引数どちらでも引用符 2 個だけを子に持ち、中身を子ノードにしない —
    その場合は ``string_value`` 自身のテキストから引用符を剥がして読む。
    """
    string_value = child_of_type(parent, "string_value")
    if string_value is None:
        return None
    content = child_of_type(string_value, "string_content")
    if content is not None:
        return node_text(content, source)
    text = node_text(string_value, source)
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        return text[1:-1]
    return text


def _plain_value(parent: Any, source: bytes) -> str | None:
    """``parent`` 直下の引用無し値 (``url(foo.png)`` の ``foo.png``) を読む。"""
    plain = child_of_type(parent, "plain_value")
    return node_text(plain, source) if plain is not None else None


def html_import_specs(root: Any, source: bytes) -> list[str]:
    """``<script src>`` / ``<link href>`` の指定子を document 順に返す。"""
    specs: list[str] = []
    for node in iter_nodes(root):
        if node.type == "script_element":
            start_tag = child_of_type(node, "start_tag")
            if start_tag is None:
                continue
            src = attr_value(start_tag, "src", source)
            if src:
                specs.append(src)
        elif node.type == "element":
            start_tag = child_of_type(node, "start_tag")
            if start_tag is None:
                continue
            tag_name = child_of_type(start_tag, "tag_name")
            if tag_name is None or node_text(tag_name, source).lower() != "link":
                continue
            href = attr_value(start_tag, "href", source)
            if href:
                specs.append(href)
    return specs


def css_import_specs(root: Any, source: bytes) -> list[str]:
    """``@import`` / ``@use`` / ``@forward`` / ``url()`` の指定子を document 順に返す。

    ``url(./a/b.png)`` のように引用無しで相対記法 (``.``/``..``) を含む値は
    CSS 文法自体がトークナイズに失敗し ``ERROR`` ノードへ分解される
    (実測 2026-09-20、tree-sitter-language-pack 1.20 の CSS grammar) —
    その場合は指定子を拾えない (誤った指定子を拾うよりは無害)。引用付き
    (``url("./a/b.png")``) は正しく読める。
    """
    specs: list[str] = []
    for node in iter_nodes(root):
        if node.type in _STYLESHEET_STATEMENT_TYPES:
            spec = _string_content(node, source)
            if spec:
                specs.append(spec)
        elif node.type == "call_expression":
            fn = child_of_type(node, "function_name")
            if fn is None or node_text(fn, source).lower() != "url":
                continue
            args = child_of_type(node, "arguments")
            if args is None:
                continue
            spec = _string_content(args, source) or _plain_value(args, source)
            if spec:
                specs.append(spec)
    return specs


def extract_html(path: str, source: bytes) -> ExtractedFile | None:
    """HTML ファイルを file ノード + imports 辺だけで抽出する (c_16 §4.4)。

    tree-sitter / html grammar が読めなければ ``None`` (呼出側が WARNING で
    読み飛ばす)。
    """
    parser = get_parser("html")
    if parser is None:
        return None
    tree = parser.parse(source)
    imports = html_import_specs(tree.root_node, source)
    return ExtractedFile(
        path=path, lang="html", line_count=_line_count(source),
        nodes=[], imports=imports, calls=[],
    )


def extract_style(path: str, lang: str, source: bytes) -> ExtractedFile | None:
    """CSS / SCSS ファイル (``lang`` は ``"css"``/``"scss"``) を抽出する。"""
    parser = get_parser(lang)
    if parser is None:
        return None
    tree = parser.parse(source)
    imports = css_import_specs(tree.root_node, source)
    return ExtractedFile(
        path=path, lang=lang, line_count=_line_count(source),
        nodes=[], imports=imports, calls=[],
    )


__all__ = ["css_import_specs", "extract_html", "extract_style", "html_import_specs"]
