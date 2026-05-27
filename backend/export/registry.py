"""ファイル書出しレジストリ

拡張子 → FileWriter のマッピングを管理し、
ExportContent からのファイル書出しを統一的に提供する。
"""

from __future__ import annotations

from pathlib import Path

from backend.export.base import ExportContent, ExportError, FileWriter, WriteResult
from backend.log_config import get_logger

logger = get_logger("export.registry")


class WriterRegistry:
    """拡張子→FileWriter のマッピングを管理"""

    def __init__(self) -> None:
        self._writers: dict[str, FileWriter] = {}
        self._all_writers: list[FileWriter] = []

    def register(self, writer: FileWriter) -> None:
        """Writer を登録（extensions の各拡張子に対応）"""
        self._all_writers.append(writer)
        for ext in writer.extensions:
            ext_lower = ext.lower()
            if ext_lower in self._writers:
                logger.debug(
                    "Overriding writer for %s: %s -> %s",
                    ext_lower,
                    type(self._writers[ext_lower]).__name__,
                    type(writer).__name__,
                )
            self._writers[ext_lower] = writer
            logger.debug("Registered writer for %s: %s", ext_lower, type(writer).__name__)

    def write(self, content: ExportContent, path: Path) -> WriteResult:
        """拡張子から Writer を判定して書出し"""
        ext = path.suffix.lower()
        writer = self._writers.get(ext)

        if writer is None:
            raise ExportError(
                "unsupported_format",
                f"Unsupported export format: {ext or '(no extension)'}",
            )

        if not writer.is_available():
            requires = ", ".join(writer.requires)
            raise ExportError(
                "missing_library",
                f"Required library not installed: {requires}",
            )

        return writer.write(content, path)

    def write_to_bytes(self, content: ExportContent, format_ext: str) -> bytes:
        """指定形式でバイトデータに書出し"""
        ext = format_ext.lower() if format_ext.startswith(".") else f".{format_ext.lower()}"
        writer = self._writers.get(ext)

        if writer is None:
            raise ExportError(
                "unsupported_format",
                f"Unsupported export format: {ext}",
            )

        if not writer.is_available():
            requires = ", ".join(writer.requires)
            raise ExportError(
                "missing_library",
                f"Required library not installed: {requires}",
            )

        return writer.write_to_bytes(content, ext)

    def supported_extensions(self) -> frozenset[str]:
        """登録済みの全対応拡張子"""
        return frozenset(self._writers.keys())

    def is_supported(self, ext: str) -> bool:
        """指定拡張子が対応済みか"""
        return ext.lower() in self._writers

    def check_availability(self) -> dict[str, bool]:
        """全 Writer のライブラリ可用性を返す"""
        result: dict[str, bool] = {}
        for writer in self._all_writers:
            name = type(writer).__name__
            result[name] = writer.is_available()
        return result

    def get_writer(self, ext: str) -> FileWriter | None:
        """指定拡張子の Writer を取得"""
        return self._writers.get(ext.lower())
