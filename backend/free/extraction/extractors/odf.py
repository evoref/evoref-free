"""ODF Extractor

.odt, .ods, .odp (OpenDocument) ファイルからテキストを抽出する。
odfpy を使用。
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
        return ["odfpy"]

    def is_available(self) -> bool:
        try:
            from odf.opendocument import load  # noqa: F401
            from odf import teletype  # noqa: F401
            return True
        except ImportError:
            return False

    def _extract_from_source(
        self,
        source: Path | BinaryIO,
        source_name: str,
    ) -> tuple[str, dict[str, Any]]:
        odf_load, odf_teletype = self._import_odfpy()
        ext = Path(source_name).suffix.lower()
        # odfpy の load() は str / file-like を受け付ける
        arg = str(source) if isinstance(source, Path) else source
        doc = odf_load(arg)
        text = self._extract_text(doc, odf_teletype, ext)
        return text, {"format": ext}

    @staticmethod
    def _import_odfpy():
        try:
            from odf.opendocument import load
            from odf import teletype
            return load, teletype
        except ImportError:
            raise ExtractionError(
                "missing_library",
                "odfpy is required for ODF extraction: pip install odfpy",
            )

    @staticmethod
    def _extract_text(doc, teletype_mod, ext: str) -> str:
        """ODF ドキュメントからテキストを抽出"""
        from odf import text as odf_text
        from odf import table as odf_table

        parts: list[str] = []

        if ext in (".odt", ".odp"):
            # テキスト段落を抽出
            for para in doc.getElementsByType(odf_text.P):
                t = teletype_mod.extractText(para)
                if t and t.strip():
                    parts.append(t)

        if ext == ".ods":
            # スプレッドシート: テーブル → 行 → セル
            for table_elem in doc.getElementsByType(odf_table.Table):
                table_name = table_elem.getAttribute("name") or "Sheet"
                rows: list[str] = []
                for row_elem in table_elem.getElementsByType(odf_table.TableRow):
                    cells: list[str] = []
                    for cell_elem in row_elem.getElementsByType(odf_table.TableCell):
                        cell_text = teletype_mod.extractText(cell_elem)
                        cells.append(cell_text if cell_text else "")
                    if any(c.strip() for c in cells):
                        rows.append("\t".join(cells))
                if rows:
                    parts.append(f"[Sheet: {table_name}]\n" + "\n".join(rows))

        return "\n\n".join(parts)
