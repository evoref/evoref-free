"""プレーンテキスト Writer

.txt: Markdown → 書式除去プレーンテキスト
.md: Markdown そのまま書出し
"""

from __future__ import annotations

import json
import re
from typing import override

from backend.export._writer_base import BytesWriterBase
from backend.export.base import (
    ContentBlock,
    ExportContent,
    ListItemNode,
    build_item_tree,
    sibling_runs,
)
from backend.export.markdown_patterns import (
    RE_BOLD as _RE_BOLD,
    RE_INLINE_CODE as _RE_CODE,
    RE_ITALIC as _RE_ITALIC,
)

# inline Markdown 書式除去パターン (link/image は URL を捨てる plaintext 固有)
_RE_LINK = re.compile(r"\[(.+?)\]\(.+?\)")
_RE_IMAGE = re.compile(r"!\[.*?\]\(.+?\)")


def _strip_inline_formatting(text: str) -> str:
    """inline Markdown 書式を除去してプレーンテキストにする"""
    text = _RE_IMAGE.sub("", text)
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    text = _RE_ITALIC.sub(lambda m: m.group(1) or m.group(2), text)
    text = _RE_CODE.sub(r"\1", text)
    return text


def _list_plaintext_lines(nodes: list[ListItemNode], depth: int = 0) -> list[str]:
    """入れ子ツリーを 2 スペース字下げの行列にする。

    番号付きは段ごと・親ごとに 1 から数える。平らな (depth=0 のみ) 入力
    では、従来の ``enumerate(items, 1)`` と同じ行になる。
    """
    lines: list[str] = []
    indent = "  " * depth
    for run in sibling_runs(nodes):
        for counter, node in enumerate(run, 1):
            prefix = f"{counter}." if node.ordered else "-"
            head, *rest = _strip_inline_formatting(node.text).split("\n")
            lines.append(f"{indent}  {prefix} {head}")
            hang = " " * (len(indent) + 2 + len(prefix) + 1)
            lines.extend(f"{hang}{cont}" for cont in rest)
            lines.extend(_list_plaintext_lines(node.children, depth + 1))
    return lines


def _blocks_to_plaintext(blocks: list[ContentBlock]) -> str:
    """ContentBlock リストをプレーンテキストに変換"""
    parts: list[str] = []

    for block in blocks:
        if block.type == "heading":
            text = _strip_inline_formatting(block.content)
            parts.append(text)
            if block.level == 1:
                parts.append("=" * len(text))
            elif block.level == 2:
                parts.append("-" * len(text))
            parts.append("")

        elif block.type == "paragraph":
            parts.append(_strip_inline_formatting(block.content))
            parts.append("")

        elif block.type == "code":
            for line in block.content.split("\n"):
                parts.append(f"    {line}")
            parts.append("")

        elif block.type == "table":
            for row in block.rows:
                parts.append(" | ".join(row))
            parts.append("")

        elif block.type == "list":
            parts.extend(_list_plaintext_lines(build_item_tree(block)))
            parts.append("")

        elif block.type == "quote":
            for line in block.content.split("\n"):
                parts.append(f"  > {line}")
            parts.append("")

        elif block.type == "hr":
            parts.append("---")
            parts.append("")

        elif block.type == "image":
            # .md 出力も兼ねるので Markdown 記法のまま残す。
            parts.append(f"![{block.content}]({block.src})")
            parts.append("")

        elif block.type == "shapes":
            parts.append("```shapes")
            parts.append(json.dumps(block.shapes, ensure_ascii=False))
            parts.append("```")
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


class PlaintextWriter(BytesWriterBase):
    """プレーンテキスト / Markdown Writer"""

    @property
    @override
    def extensions(self) -> frozenset[str]:
        return frozenset({".txt", ".md"})

    @override
    def _render_bytes(self, content: ExportContent, ext: str) -> bytes:
        return self._render(content, ext).encode("utf-8")

    @staticmethod
    def _render(content: ExportContent, ext: str) -> str:
        """拡張子に応じてテキストを生成"""
        if ext == ".md":
            # Markdown はそのまま書出し
            if content.raw_markdown:
                return content.raw_markdown
            # blocks から Markdown 再構成はせず、raw_markdown がなければ空
            return ""
        # .txt: blocks をプレーンテキスト化
        if content.blocks:
            return _blocks_to_plaintext(content.blocks)
        # blocks がない場合は raw_markdown から書式除去
        if content.raw_markdown:
            return _strip_inline_formatting(content.raw_markdown)
        return ""
