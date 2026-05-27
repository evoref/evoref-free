"""PDF Extractor

.pdf ファイルからテキストを抽出する。pypdf を使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, override

from backend.extraction._binary_source_base import BinarySourceExtractorBase
from backend.extraction.base import ExtractionError


class PdfExtractor(BinarySourceExtractorBase):
    """PDF ファイルからテキストを抽出"""

    @property
    @override
    def extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    @property
    @override
    def error_code(self) -> str:
        return "pdf_error"

    @property
    @override
    def requires(self) -> list[str]:
        return ["pypdf"]

    @override
    def is_available(self) -> bool:
        try:
            from pypdf import PdfReader  # noqa: F401
            return True
        except ImportError:
            return False

    @override
    def _extract_from_source(
        self,
        source: Path | BinaryIO,
        source_name: str,
    ) -> tuple[str, dict[str, Any]]:
        PdfReader = self._import_pypdf()
        reader = PdfReader(source)
        texts, page_count = self._extract_pages(reader)
        return "\n\n".join(texts), {"pages": page_count}

    @staticmethod
    def _import_pypdf():
        """pypdf を遅延インポート"""
        try:
            from pypdf import PdfReader
            return PdfReader
        except ImportError:
            raise ExtractionError(
                "missing_library",
                "pypdf is required for PDF extraction: pip install pypdf",
            )

    @staticmethod
    def _extract_pages(reader) -> tuple[list[str], int]:
        """全ページからテキストを抽出"""
        texts: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                texts.append(text)
        return texts, len(reader.pages)
