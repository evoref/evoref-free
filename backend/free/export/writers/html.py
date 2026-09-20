"""HTML Writer

.html, .htm: ContentBlock → HTML 文書として書出し
"""

from __future__ import annotations

import html
from typing import override

from backend.export._writer_base import BytesWriterBase
from backend.export.base import (
    ContentBlock,
    ExportContent,
    ListItemNode,
    build_item_tree,
    sibling_runs,
)
from backend.log_config import get_logger

logger = get_logger("export.writers.html")
from backend.export.markdown_patterns import (
    RE_BOLD as _RE_BOLD,
    RE_INLINE_CODE as _RE_CODE,
    RE_ITALIC as _RE_ITALIC,
    RE_LINK as _RE_LINK,
)

_MINIMAL_CSS = """\
body { font-family: sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; line-height: 1.6; color: #333; }
pre { background: #f5f5f5; padding: 1em; overflow-x: auto; border-radius: 4px; }
code { font-family: monospace; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background: #f0f0f0; }
blockquote { border-left: 4px solid #ddd; margin: 1em 0; padding: 0.5em 1em; color: #666; }
hr { border: none; border-top: 1px solid #ddd; margin: 2em 0; }
"""


def _inline_html(text: str) -> str:
    """inline Markdown → HTML 変換"""
    text = html.escape(text)
    text = _RE_BOLD.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", text)
    text = _RE_ITALIC.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", text)
    text = _RE_CODE.sub(r"<code>\1</code>", text)
    text = _RE_LINK.sub(r'<a href="\2">\1</a>', text)
    return text


def _item_html(text: str) -> str:
    """リスト項目のテキスト。項目の中の改行 (題名 + 説明) は ``<br>`` にする。"""
    return "<br>".join(_inline_html(part) for part in text.split("\n"))


def _list_html_lines(nodes: list[ListItemNode]) -> list[str]:
    """入れ子ツリーを ``<ul>``/``<ol>`` の行列にする。

    平らな (子を持たない) ノードだけなら、各 ``<li>`` を独立した行として
    積む従来の構造とそのまま一致する。
    """
    lines: list[str] = []
    for run in sibling_runs(nodes):
        tag = "ol" if run[0].ordered else "ul"
        lines.append(f"<{tag}>")
        for node in run:
            child_lines = _list_html_lines(node.children)
            if child_lines:
                lines.append(f"<li>{_item_html(node.text)}")
                lines.extend(child_lines)
                lines.append("</li>")
            else:
                lines.append(f"<li>{_item_html(node.text)}</li>")
        lines.append(f"</{tag}>")
    return lines


def _blocks_to_html(blocks: list[ContentBlock], title: str) -> str:
    """ContentBlock リストを完全な HTML 文書に変換"""
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="ja">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{_MINIMAL_CSS}</style>",
        "</head>",
        "<body>",
    ]

    for block in blocks:
        if block.type == "heading":
            tag = f"h{block.level}"
            parts.append(f"<{tag}>{_inline_html(block.content)}</{tag}>")

        elif block.type == "paragraph":
            parts.append(f"<p>{_inline_html(block.content)}</p>")

        elif block.type == "code":
            lang_attr = f' class="language-{html.escape(block.language)}"' if block.language else ""
            parts.append(f"<pre><code{lang_attr}>{html.escape(block.content)}</code></pre>")

        elif block.type == "table":
            parts.append("<table>")
            for i, row in enumerate(block.rows):
                tag = "th" if i == 0 else "td"
                cells = "".join(f"<{tag}>{_inline_html(c)}</{tag}>" for c in row)
                parts.append(f"<tr>{cells}</tr>")
            parts.append("</table>")

        elif block.type == "list":
            parts.extend(_list_html_lines(build_item_tree(block)))

        elif block.type == "quote":
            parts.append(f"<blockquote><p>{_inline_html(block.content)}</p></blockquote>")

        elif block.type == "hr":
            parts.append("<hr>")

        elif block.type == "image":
            # src はそのまま <img> に載せる (HTML は外部参照が自然)。
            alt = html.escape(block.content or "")
            parts.append(
                f'<img src="{html.escape(block.src)}" alt="{alt}">',
            )

        elif block.type == "shapes":
            logger.warning(
                "shapes blocks are not drawn in .html; %d shape(s) skipped",
                len(block.shapes),
            )

    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


class HtmlWriter(BytesWriterBase):
    """HTML ファイル Writer"""

    @property
    @override
    def extensions(self) -> frozenset[str]:
        return frozenset({".html", ".htm"})

    @override
    def _render_bytes(self, content: ExportContent, ext: str) -> bytes:
        return self._render(content).encode("utf-8")

    @staticmethod
    def _render(content: ExportContent) -> str:
        """ExportContent を HTML 文書に変換"""
        if content.blocks:
            return _blocks_to_html(content.blocks, content.title)
        # blocks がない場合は raw_markdown をそのまま HTML にラップ
        if content.raw_markdown:
            from backend.export.content_converter import ContentConverter
            blocks = ContentConverter().convert(content.raw_markdown)
            return _blocks_to_html(blocks, content.title)
        return _blocks_to_html([], content.title)
