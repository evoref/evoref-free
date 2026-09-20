"""Svelte / Vue の単一ファイルコンポーネント抽出 (c_16 §4.4 マークアップ / SFC)

file 直下に ``node_type=component`` を 1 つ置き (``name`` = ファイル stem)、
``<script>`` ブロック (複数あり得る: Svelte の ``context="module"`` / Vue の
``<script setup>``) の中身を取り出して既存の JS / TS クエリにそのまま通す —
行番号は元ファイルの位置へ戻し、function / class は component の子にする。
``<style>`` の ``@import`` は CSS と同じ扱い。テンプレート部の使用・CSS の
rule ノードからの辺は張らない (c_16 §4.4)。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any

from backend.free.rag.projectmap.extractors import treesitter
from backend.free.rag.projectmap.extractors.html_tree import (
    attr_value,
    child_of_type,
    get_parser,
)
from backend.free.rag.projectmap.extractors.markup import css_import_specs
from backend.free.rag.projectmap.graph import ExtractedFile, Node, RawCall
from backend.free.rag.projectmap.ids import code_node_id
from backend.log_config import get_logger

logger = get_logger("rag.projectmap.sfc")

#: 1 ブロックの抽出結果 (nodes, imports, calls, inherits)。
_BlockResult = tuple[list[Node], list[str], list[RawCall], list[tuple[str, str]]]


def _line_count(source: bytes) -> int:
    count = source.count(b"\n") + (0 if source.endswith(b"\n") else 1)
    return max(count, 1)


def _script_lang(start_tag: Any, source: bytes) -> str:
    lang_attr = attr_value(start_tag, "lang", source)
    if lang_attr and lang_attr.lower() in ("ts", "typescript"):
        return "typescript"
    return "javascript"


def _top_level_sections(root: Any, section_type: str) -> list[Any]:
    """document 直下 (``<template>`` の中は見ない) の指定型ノード。"""
    return [c for c in root.children if c.type == section_type]


def _extract_script_block(
    path: str, start_tag: Any, raw_text: Any, source: bytes,
) -> _BlockResult | None:
    lang = _script_lang(start_tag, source)
    content = source[raw_text.start_byte:raw_text.end_byte]
    extracted = treesitter.extract_file(path, lang, content)
    if extracted is None:
        logger.warning(
            "projectmap: failed to extract <script> block in %s (lang=%s)", path, lang,
        )
        return None
    offset = raw_text.start_point.row
    nodes = [
        replace(n, line_start=offset + n.line_start, line_end=offset + n.line_end)
        for n in extracted.nodes
    ]
    return nodes, extracted.imports, extracted.calls, extracted.inherits


def _dedupe_top_level(
    path: str, nodes_by_block: list[list[Node]],
) -> tuple[list[Node], dict[str, str]]:
    """複数 script ブロックを跨いだ top-level qualname 衝突を ``#<n>`` で分ける。

    Svelte の module/instance context のように、別々の script ブロックが
    たまたま同じ名前のトップレベル定義を持つと、path が同じ (SFC ファイル
    自身) なので id (``path`` + ``node_type`` + ``qualname`` 由来) が衝突する。
    2 つ目以降に ``#<n>`` を付けて id を作り直す (``treesitter.py`` の
    同一ファイル内再定義と同じ規則)。戻り値の ``id_map`` は改名された
    ノードの旧 id → 新 id で、呼出側が ``calls``/``inherits`` の参照先も
    追随させるのに使う。
    """
    seen: dict[str, int] = {}
    id_map: dict[str, str] = {}
    merged: list[Node] = []
    for block_nodes in nodes_by_block:
        for node in block_nodes:
            if node.parent_id is None:
                occurrence = seen.get(node.qualname, 0) + 1
                seen[node.qualname] = occurrence
                if occurrence > 1:
                    new_qualname = f"{node.qualname}#{occurrence}"
                    new_id = code_node_id(path, node.node_type, new_qualname)
                    id_map[node.id] = new_id
                    node = replace(node, qualname=new_qualname, id=new_id)
            merged.append(node)
    remapped = [
        replace(n, parent_id=id_map[n.parent_id]) if n.parent_id in id_map else n
        for n in merged
    ]
    return remapped, id_map


def extract_file(path: str, lang: str, source: bytes) -> ExtractedFile | None:
    """Svelte / Vue ファイルを component + 子ノード + imports で抽出する。

    外殻文法 (``lang``、``svelte``/``vue``) の parser が読めなければ ``None``
    (呼出側が他言語同様 WARNING で読み飛ばす)。script / style の中身は
    既存の JS/TS/CSS 抽出器へそのまま委譲する。
    """
    parser = get_parser(lang)
    if parser is None:
        return None
    tree = parser.parse(source)
    root = tree.root_node

    name = PurePosixPath(path).stem
    component_id = code_node_id(path, "component", name)
    component = Node(
        id=component_id, node_type="component", path=path, name=name,
        qualname=name, lang=lang, line_start=1, line_end=_line_count(source),
        parent_id=None, signature="", text=name,
    )

    nodes_by_block: list[list[Node]] = []
    imports: list[str] = []
    calls: list[RawCall] = []
    inherits: list[tuple[str, str]] = []

    for script_node in _top_level_sections(root, "script_element"):
        start_tag = child_of_type(script_node, "start_tag")
        raw_text = child_of_type(script_node, "raw_text")
        if start_tag is None or raw_text is None:
            continue
        result = _extract_script_block(path, start_tag, raw_text, source)
        if result is None:
            continue
        block_nodes, block_imports, block_calls, block_inherits = result
        nodes_by_block.append(block_nodes)
        imports.extend(block_imports)
        calls.extend(block_calls)
        inherits.extend(block_inherits)

    for style_node in _top_level_sections(root, "style_element"):
        raw_text = child_of_type(style_node, "raw_text")
        if raw_text is None:
            continue
        css_parser = get_parser("css")
        if css_parser is None:
            continue
        style_source = source[raw_text.start_byte:raw_text.end_byte]
        css_tree = css_parser.parse(style_source)
        imports.extend(css_import_specs(css_tree.root_node, style_source))

    merged_nodes, id_map = _dedupe_top_level(path, nodes_by_block)
    child_nodes = [
        replace(n, parent_id=component_id) if n.parent_id is None else n
        for n in merged_nodes
    ]
    calls = [
        RawCall(caller_id=id_map.get(c.caller_id, c.caller_id), callee_name=c.callee_name)
        for c in calls
    ]
    inherits = [
        (id_map.get(class_id, class_id), base_name) for class_id, base_name in inherits
    ]

    all_nodes = [component, *child_nodes]
    all_nodes.sort(key=lambda n: n.line_start)

    return ExtractedFile(
        path=path, lang=lang, line_count=_line_count(source),
        nodes=all_nodes, imports=imports, calls=calls, inherits=inherits,
    )


__all__ = ["extract_file"]
