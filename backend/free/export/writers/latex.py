"""LaTeX Writer

.tex: Markdown → LaTeX 文書として書出し
"""

from __future__ import annotations

import re
from typing import override

from backend.export._writer_base import BytesWriterBase
from backend.export.base import ContentBlock, ExportContent

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

_RE_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_RE_ITALIC = re.compile(r"\*(.+?)\*|_(.+?)_")
_RE_CODE_INLINE = re.compile(r"`(.+?)`")
_RE_LINK = re.compile(r"\[(.+?)\]\((.+?)\)")

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
            env = "enumerate" if block.ordered else "itemize"
            parts.append(rf"\begin{{{env}}}")
            for item in block.items:
                parts.append(rf"  \item {_inline_latex(item)}")
            parts.append(rf"\end{{{env}}}")
            parts.append("")

        elif block.type == "quote":
            parts.append(r"\begin{quote}")
            parts.append(_inline_latex(block.content))
            parts.append(r"\end{quote}")
            parts.append("")

        elif block.type == "hr":
            parts.append(r"\bigskip\noindent\rule{\textwidth}{0.4pt}\bigskip")
            parts.append("")

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
