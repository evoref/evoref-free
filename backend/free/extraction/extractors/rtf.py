"""RTF Extractor

.rtf ファイルからテキストを抽出する。
striprtf を使用。
"""

from __future__ import annotations

from backend.extraction._text_source_base import TextSourceExtractorBase
from backend.extraction.base import ExtractionError


class RtfExtractor(TextSourceExtractorBase):
    """RTF ファイルからテキストを抽出"""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".rtf"})

    @property
    def requires(self) -> list[str]:
        return ["striprtf"]

    def is_available(self) -> bool:
        try:
            from striprtf.striprtf import rtf_to_text  # noqa: F401
            return True
        except ImportError:
            return False

    def _process(self, text: str, source_name: str) -> str:
        """RTF を striprtf でプレーンテキストに変換"""
        rtf_to_text = self._import_striprtf()
        return rtf_to_text(text)

    @staticmethod
    def _import_striprtf():
        try:
            from striprtf.striprtf import rtf_to_text
            return rtf_to_text
        except ImportError:
            raise ExtractionError(
                "missing_library",
                "striprtf is required for RTF extraction: pip install striprtf",
            )
