"""LaTeX Writer

.tex: Markdown → LaTeX 文書として書出し
"""

from __future__ import annotations

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
    RE_INLINE_CODE as _RE_CODE_INLINE,
    RE_ITALIC as _RE_ITALIC,
    RE_LINK as _RE_LINK,
)
from backend.log_config import get_logger

logger = get_logger("export.writers.latex")

# LaTeX 特殊文字のエスケープ
_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


_HEADING_CMDS = {
    1: "section",
    2: "subsection",
    3: "subsubsection",
    4: "paragraph",
    5: "subparagraph",
    6: "subparagraph",
}

_PREAMBLE = r"""\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{CJKutf8}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{longtable}
\usepackage{booktabs}

\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\tiny,
}

"""


def _escape_latex(text: str) -> str:
    """LaTeX 特殊文字をエスケープ"""
    # \ は最初に処理（他のエスケープで \ を使うため）
    text = text.replace("\\", r"\textbackslash{}")
    for ch, repl in _LATEX_SPECIAL.items():
        text = text.replace(ch, repl)
    return text


def _inline_latex(text: str) -> str:
    """inline Markdown → LaTeX 変換"""
    # まず太字・斜体・コード・リンクを処理（エスケープ前）
    text = _RE_BOLD.sub(lambda m: rf"\textbf{{{m.group(1) or m.group(2)}}}", text)
    text = _RE_ITALIC.sub(lambda m: rf"\textit{{{m.group(1) or m.group(2)}}}", text)
    text = _RE_CODE_INLINE.sub(lambda m: rf"\texttt{{{m.group(1)}}}", text)
    text = _RE_LINK.sub(lambda m: rf"\href{{{m.group(2)}}}{{{m.group(1)}}}", text)
    return text


def _list_latex_lines(nodes: list[ListItemNode], indent: str = "") -> list[str]:
    """入れ子ツリーを ``itemize``/``enumerate`` の行列にする。

    平らな (子を持たない) ノードだけなら従来と同じ行列になる。
    """
    lines: list[str] = []
    item_indent = indent + "  "
    for run in sibling_runs(nodes):
        env = "enumerate" if run[0].ordered else "itemize"
        lines.append(rf"{indent}\begin{{{env}}}")
        for node in run:
            parts = [_inline_latex(part) for part in node.text.split("\n")]
            lines.append(rf"{item_indent}\item " + r" \\ ".join(parts))
            lines.extend(_list_latex_lines(node.children, item_indent + "  "))
        lines.append(rf"{indent}\end{{{env}}}")
    return lines


def _blocks_to_latex(blocks: list[ContentBlock], title: str) -> str:
    """ContentBlock リストを完全な LaTeX 文書に変換"""
    parts: list[str] = [_PREAMBLE]

    if title:
        parts.append(rf"\title{{{_escape_latex(title)}}}")
        parts.append(r"\date{}")
        parts.append("")

    parts.append(r"\begin{document}")
    parts.append(r"\begin{CJK}{UTF8}{min}")
    if title:
        parts.append(r"\maketitle")
    parts.append("")

    for block in blocks:
        if block.type == "heading":
            cmd = _HEADING_CMDS.get(block.level, "subparagraph")
            parts.append(rf"\{cmd}{{{_inline_latex(block.content)}}}")
            parts.append("")

        elif block.type == "paragraph":
            parts.append(_inline_latex(block.content))
            parts.append("")

        elif block.type == "code":
            lang_opt = f"[language={block.language}]" if block.language else ""
            parts.append(rf"\begin{{lstlisting}}{lang_opt}")
            parts.append(block.content)
            parts.append(r"\end{lstlisting}")
            parts.append("")

        elif block.type == "table":
            if block.rows:
                cols = len(block.rows[0])
                col_spec = " ".join(["l"] * cols)
                parts.append(rf"\begin{{longtable}}{{{col_spec}}}")
                parts.append(r"\toprule")
                for i, row in enumerate(block.rows):
                    cells = " & ".join(_escape_latex(c) for c in row)
                    parts.append(rf"{cells} \\")
                    if i == 0:
                        parts.append(r"\midrule")
                parts.append(r"\bottomrule")
                parts.append(r"\end{longtable}")
                parts.append("")

        elif block.type == "list":
            parts.extend(_list_latex_lines(build_item_tree(block)))
            parts.append("")

        elif block.type == "quote":
            parts.append(r"\begin{quote}")
            parts.append(_inline_latex(block.content))
            parts.append(r"\end{quote}")
            parts.append("")

        elif block.type == "hr":
            parts.append(r"\bigskip\noindent\rule{\textwidth}{0.4pt}\bigskip")
            parts.append("")

        elif block.type == "image":
            parts.append(r"\begin{figure}[h]")
            parts.append(r"\centering")
            parts.append(rf"\includegraphics[width=0.8\textwidth]{{{block.src}}}")
            if block.content:
                parts.append(rf"\caption{{{_escape_latex(block.content)}}}")
            parts.append(r"\end{figure}")
            parts.append("")

        elif block.type == "shapes":
            logger.warning(
                "shapes blocks are not drawn in .tex; %d shape(s) skipped",
                len(block.shapes),
            )

    parts.append(r"\end{CJK}")
    parts.append(r"\end{document}")
    parts.append("")
    return "\n".join(parts)


class LatexWriter(BytesWriterBase):
    """LaTeX ファイル Writer"""

    @property
    @override
    def extensions(self) -> frozenset[str]:
        return frozenset({".tex"})

    @override
    def _render_bytes(self, content: ExportContent, ext: str) -> bytes:
        return self._render(content).encode("utf-8")

    @staticmethod
    def _render(content: ExportContent) -> str:
        """ExportContent を LaTeX 文書に変換"""
        if content.blocks:
            return _blocks_to_latex(content.blocks, content.title)
        if content.raw_markdown:
            from backend.export.content_converter import ContentConverter
            blocks = ContentConverter().convert(content.raw_markdown)
            return _blocks_to_latex(blocks, content.title)
        return _blocks_to_latex([], content.title)
