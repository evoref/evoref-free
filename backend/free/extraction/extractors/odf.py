"""ODF Extractor

.odt, .ods, .odp (OpenDocument) ファイルからテキストを抽出する。
odfdo を使用 (2026-09-16 に odfpy から移行、docs/f_11_file_export.md §7)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from backend.extraction._binary_source_base import BinarySourceExtractorBase
from backend.extraction.base import ExtractionError


class OdfExtractor(BinarySourceExtractorBase):
    """OpenDocument (ODT/ODS/ODP) ファイルからテキストを抽出"""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".odt", ".ods", ".odp"})

    @property
    def error_code(self) -> str:
        return "odf_error"

    @property
    def requires(self) -> list[str]:
        return ["odfdo"]

    def is_available(self) -> bool:
        try:
            from odfdo import Document  # noqa: F401
            return True
        except ImportError:
            return False

    def _extract_from_source(
        self,
        source: Path | BinaryIO,
        source_name: str,
    ) -> tuple[str, dict[str, Any]]:
        document_cls = self._import_odfdo()
        ext = Path(source_name).suffix.lower()
        # odfdo の Document() は str / Path / BytesIO を受け付ける
        arg = str(source) if isinstance(source, Path) else source
        doc = document_cls(arg)
        text = self._extract_text(doc, ext)
        return text, {"format": ext}

    @staticmethod
    def _import_odfdo():
        try:
            from odfdo import Document
            return Document
        except ImportError:
            raise ExtractionError(
                "missing_library",
                "odfdo is required for ODF extraction: pip install odfdo",
            )

    @staticmethod
    def _cell_text(cell) -> str:
        """セルの表示文字列。

        ``Cell.value`` は ``office:value-type`` を持たないセルで ``None`` になる。
        odfpy が書いた既存ファイルがまさにこれなので、``text_recursive`` へ落ちる
        二段構えにする (f_11 §7.1)。
        """
        value = cell.value
        if value is None:
            return (cell.text_recursive or "").strip()
        return str(value).strip()

    @classmethod
    def _extract_text(cls, doc, ext: str) -> str:
        """ODF ドキュメントからテキストを抽出"""
        body = doc.body
        parts: list[str] = []

        if ext == ".ods":
            for table in body.get_tables():
                table_name = table.name or "Sheet"
                rows: list[str] = []
                for row in table.get_rows():
                    cells = [cls._cell_text(c) for c in row.get_cells()]
                    if any(c for c in cells):
                        rows.append("\t".join(cells))
                if rows:
                    parts.append(f"[Sheet: {table_name}]\n" + "\n".join(rows))
            return "\n\n".join(parts)

        # .odt / .odp: 見出しと段落を出現順に拾う。get_headers() /
        # get_paragraphs() は種別ごとの取得なので、順序を保つために
        # 要素の文書順で走査する。
        for element in body.get_elements("//text:h | //text:p"):
            text = (element.text_recursive or "").strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)
