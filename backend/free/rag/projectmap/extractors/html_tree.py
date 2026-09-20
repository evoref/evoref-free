"""HTML 系文法 (HTML / Svelte / Vue) の共通木走査ユーティリティ (c_16 §4.4)

``<script>`` / ``<style>`` / ``<link>`` のようなタグ構造は HTML・Svelte・Vue の
tree-sitter 文法で同じノード名 (``script_element`` / ``style_element`` /
``start_tag`` / ``attribute`` / ``attribute_name`` / ``quoted_attribute_value``
/ ``attribute_value``) を使う (実測 2026-09-20)。マークアップ抽出
(``markup.py``) と SFC 抽出 (``sfc.py``) の両方から使う。

``treesitter.py`` の ``_get_parser_and_query`` は :data:`queries.LANGUAGE_QUERIES`
に登録された言語 (class/function クエリを持つ言語) 専用。マークアップ系
(imports だけの言語) はクエリを持たないため、ここで parser だけを別枠で
キャッシュする。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from backend.free.rag.projectmap.extractors.treesitter import is_available
from backend.log_config import get_logger

logger = get_logger("rag.projectmap.html_tree")

#: 言語 → parser (取得に失敗した言語は ``None``、再試行/再警告をしない)。
_PARSER_CACHE: dict[str, Any | None] = {}
_WARNED_LANGS: set[str] = set()


def get_parser(lang: str) -> Any | None:
    """``lang`` (``html`` / ``css`` / ``scss`` / ``svelte`` / ``vue``) の parser。"""
    if lang in _PARSER_CACHE:
        return _PARSER_CACHE[lang]
    parser = None
    if is_available():
        try:
            from tree_sitter_language_pack import get_parser as _get_parser

            parser = _get_parser(lang)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 — 1 言語の失敗で全体の走査を止めない
            if lang not in _WARNED_LANGS:
                _WARNED_LANGS.add(lang)
                logger.warning("tree-sitter unavailable for language %s: %s", lang, e)
    _PARSER_CACHE[lang] = parser
    return parser


def node_text(node: Any, source: bytes) -> str:
    """``node`` の範囲を ``source`` から切り出す (UTF-8、不正バイトは置換)。"""
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def iter_nodes(root: Any) -> Iterator[Any]:
    """document 順 (子の並び順) で全ノードを辿る。

    ``.parent`` は使わない — tree-sitter 0.26 の runtime は language-pack
    1.20 の grammar と組むと親を遡る API でアクセス違反を起こす (実測
    2026-09-18、``treesitter.py`` の ``_leading_comment`` と同じ理由)。
    子を積むだけの反復 DFS なら影響を受けない。
    """
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def child_of_type(node: Any, node_type: str) -> Any | None:
    """``node`` の直下の子から最初に一致する型を返す。"""
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def attr_value(start_tag: Any, name: str, source: bytes) -> str | None:
    """``start_tag`` から属性値を読む。

    属性が無ければ ``None``。属性はあるが値を持たない (``<script setup>`` の
    ``setup`` のような真偽属性) 場合は空文字列を返す — 「無い」と「値が空」を
    区別する。
    """
    for child in start_tag.children:
        if child.type != "attribute":
            continue
        name_node = child_of_type(child, "attribute_name")
        if name_node is None or node_text(name_node, source) != name:
            continue
        quoted = child_of_type(child, "quoted_attribute_value")
        if quoted is not None:
            value_node = child_of_type(quoted, "attribute_value")
            return node_text(value_node, source) if value_node is not None else ""
        value_node = child_of_type(child, "attribute_value")
        return node_text(value_node, source) if value_node is not None else ""
    return None


__all__ = [
    "attr_value",
    "child_of_type",
    "get_parser",
    "iter_nodes",
    "node_text",
]
