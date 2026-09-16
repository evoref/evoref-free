"""DOCX Extractor

.docx ファイルから段落・テーブルのテキストを抽出する。
python-docx を使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from backend.extraction._binary_source_base import BinarySourceExtractorBase
from backend.extraction.base import ExtractionError


class DocxExtractor(BinarySourceExtractorBase):
    """DOCX ファイルからテキストを抽出"""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".docx"})

    @property
    def error_code(self) -> str:
        return "docx_error"

    @property
    def requires(self) -> list[str]:
        return ["python-docx"]

    def is_available(self) -> bool:
        try:
            from docx import Document  # noqa: F401
            return True
        except ImportError:
            return False

    def _extract_from_source(
        self,
        source: Path | BinaryIO,
        source_name: str,  # noqa: ARG002
    ) -> tuple[str, dict[str, Any]]:
        Document = self._import_docx()
        # python-docx は Path / str / file-like を受け付けるが、
        # 既存挙動 (str(path)) を維持するため Path のときのみ str に変換する
        arg = str(source) if isinstance(source, Path) else source
        doc = Document(arg)
        text, para_count, table_count = self._extract_content(doc)
        return text, {"paragraphs": para_count, "tables": table_count}

    @staticmethod
    def _import_docx():
        try:
            from docx import Document
            return Document
        except ImportError:
            raise ExtractionError(
                "missing_library",
                "python-docx is required for DOCX extraction: pip install python-docx",
            )

    @staticmethod
    def _heading_level(style_name: str) -> int:
        """段落スタイル名から見出しレベルを読む (見出しでなければ 0)。"""
        name = (style_name or "").strip()
        if name in ("Title", "表題"):
            return 1
        for prefix in ("Heading ", "見出し "):
            if name.startswith(prefix):
                try:
                    return max(1, min(int(name[len(prefix):].strip()), 6))
                except ValueError:
                    return 0
        return 0

    @classmethod
    def _paragraph_line(cls, para) -> str:
        """1 段落を Markdown 行にする。

        **見出しは ``#`` を付けて出す** (2026-09-16、docs/f_11 §5.1)。抽出は
        「既存文書に 1 節だけ足す」依頼の素材でもあり、素のテキストで返すと
        レベルが失われて書き戻しのたびに構造が崩れる (実測: ``Heading 2``
        の節が追記後に ``Heading 1`` へ落ちた)。
        """
        text = para.text.strip()
        if not text:
            return ""
        try:
            style_name = para.style.name
        except Exception:  # pragma: no cover - スタイル欠損の文書
            style_name = ""
        level = cls._heading_level(style_name)
        if level:
            return f"{'#' * level} {text}"
        if style_name in ("List Bullet", "List Number", "リスト段落"):
            return f"- {text}"
        return text

    @staticmethod
    def _table_lines(table) -> list[str]:
        """表を GFM テーブルにする (先頭行をヘッダとみなす)。"""
        rows = [
            [cell.text.strip().replace("|", r"\|") for cell in row.cells]
            for row in table.rows
        ]
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            return []
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |"]
        lines.append("|" + "---|" * width)
        lines.extend("| " + " | ".join(r) + " |" for r in rows[1:])
        return lines

    @classmethod
    def _extract_content(cls, doc) -> tuple[str, int, int]:
        """本文を **文書順** に Markdown 化する。

        以前は段落を全部出してから表を末尾にまとめていたため、表が本来の
        位置から離れ、読み戻して書き直すと節の順序が入れ替わっていた。
        """
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        lines: list[str] = []
        para_count = 0
        table_count = 0
        for child in doc.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                line = cls._paragraph_line(Paragraph(child, doc))
                if line:
                    lines.append(line)
                    para_count += 1
            elif tag == "tbl":
                table_count += 1
                table_lines = cls._table_lines(Table(child, doc))
                if table_lines:
                    lines.extend(["", *table_lines, ""])
        text = "\n".join(lines).strip("\n")
        return text, para_count, table_count
