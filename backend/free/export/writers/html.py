"""HTML Writer

.html, .htm: ContentBlock → HTML 文書として書出し
"""

from __future__ import annotations

import html
import re
from typing import override

from backend.export._writer_base import BytesWriterBase
from backend.export.base import ContentBlock, ExportContent

_RE_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_RE_ITALIC = re.compile(r"\*(.+?)\*|_(.+?)_")
_RE_CODE = re.compile(r"`(.+?)`")
_RE_LINK = re.compile(r"\[(.+?)\]\((.+?)\)")

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
            list_tag = "ol" if block.ordered else "ul"
            parts.append(f"<{list_tag}>")
            for item in block.items:
                parts.append(f"<li>{_inline_html(item)}</li>")
            parts.append(f"</{list_tag}>")

        elif block.type == "quote":
            parts.append(f"<blockquote><p>{_inline_html(block.content)}</p></blockquote>")

        elif block.type == "hr":
            parts.append("<hr>")

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
