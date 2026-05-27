"""テキスト抽出レジストリ

拡張子 → TextExtractor のマッピングを管理し、
ファイルパスまたはバイトデータからのテキスト抽出を統一的に提供する。
"""

from __future__ import annotations

from pathlib import Path

from backend.extraction.base import ExtractionError, ExtractionResult, TextExtractor
from backend.extraction.encoding import is_likely_text
from backend.log_config import get_logger

logger = get_logger("extraction.registry")


class ExtractorRegistry:
    """拡張子→Extractor のマッピングを管理"""

    def __init__(self) -> None:
        self._extractors: dict[str, TextExtractor] = {}
        self._all_extractors: list[TextExtractor] = []

    def register(self, extractor: TextExtractor) -> None:
        """Extractor を登録（extensions の各拡張子に対応）"""
        self._all_extractors.append(extractor)
        for ext in extractor.extensions:
            ext_lower = ext.lower()
            if ext_lower in self._extractors:
                logger.debug(
                    "Overriding extractor for %s: %s -> %s",
                    ext_lower,
                    type(self._extractors[ext_lower]).__name__,
                    type(extractor).__name__,
                )
            self._extractors[ext_lower] = extractor
            logger.debug("Registered extractor for %s: %s", ext_lower, type(extractor).__name__)

    def extract(self, path: Path) -> ExtractionResult:
        """ファイルパスからテキスト抽出"""
        ext = path.suffix.lower()
        extractor = self._extractors.get(ext)

        if extractor is None:
            # 拡張子なしファイルの判定: テキストの可能性があれば plaintext として処理
            if not ext and is_likely_text(path):
                plaintext = self._extractors.get(".txt")
                if plaintext is not None:
                    return plaintext.extract(path)
            raise ExtractionError(
                "unsupported_format",
                f"Unsupported file format: {ext or '(no extension)'}",
            )

        if not extractor.is_available():
            requires = ", ".join(extractor.requires)
            raise ExtractionError(
                "missing_library",
                f"Required library not installed: {requires}",
            )

        return extractor.extract(path)

    def extract_from_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        """バイトデータからテキスト抽出"""
        ext = Path(filename).suffix.lower()
        extractor = self._extractors.get(ext)

        if extractor is None:
            raise ExtractionError(
                "unsupported_format",
                f"Unsupported file format: {ext or '(no extension)'}",
            )

        if not extractor.is_available():
            requires = ", ".join(extractor.requires)
            raise ExtractionError(
                "missing_library",
                f"Required library not installed: {requires}",
            )

        return extractor.extract_from_bytes(data, filename)

    def supported_extensions(self) -> frozenset[str]:
        """登録済みの全対応拡張子"""
        return frozenset(self._extractors.keys())

    def is_supported(self, ext: str) -> bool:
        """指定拡張子が対応済みか"""
        return ext.lower() in self._extractors

    def check_availability(self) -> dict[str, bool]:
        """全 Extractor のライブラリ可用性を返す"""
        result: dict[str, bool] = {}
        for extractor in self._all_extractors:
            name = type(extractor).__name__
            result[name] = extractor.is_available()
        return result

    def get_extractor(self, ext: str) -> TextExtractor | None:
        """指定拡張子の Extractor を取得"""
        return self._extractors.get(ext.lower())
