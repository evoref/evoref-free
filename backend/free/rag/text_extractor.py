"""ドキュメントテキスト抽出 — extraction モジュールへの委譲ラッパー

RAG インジェストパイプラインで使用するテキスト抽出ユーティリティ。
実際の抽出処理は backend.extraction モジュールに委譲する。

CSV は §4.9.9 に基づき行ごとに1チャンク（ヘッダー付与）で分割する。
この CSV チャンク分割は RAG 固有ロジックとして本モジュールに残す。
"""

import csv
import io
from pathlib import Path

from backend.extraction import get_registry
from backend.extraction.base import ExtractionError
from backend.io import ChunkedReader, csv_row_strategy
from backend.log_config import get_logger

# Extractor がレジストリに登録されていることを保証
import backend.free.extraction  # noqa: F401

logger = get_logger("rag.text_extractor")


def _get_supported_extensions() -> set[str]:
    """レジストリから対応拡張子を取得（RAG で使用する形式のみ）"""
    return {
        ".md", ".txt", ".json", ".html", ".htm", ".pdf", ".csv", ".tsv",
        ".docx", ".xlsx", ".pptx",
        # Office/ODF/メール(eml)/RTF/LaTeX（Free 同梱）
        ".rtf", ".eml", ".odt", ".ods", ".odp", ".tex",
    }


# RAG で対応するドキュメント形式（後方互換のため維持）
SUPPORTED_DOC_EXTENSIONS = _get_supported_extensions()


def extract_text(file_path: Path) -> str:
    """ファイルからテキストを抽出

    extraction モジュールのレジストリに委譲する。
    CSV はこの関数ではなく parse_csv_to_chunks() を使用すること。
    """
    registry = get_registry()
    try:
        result = registry.extract(file_path)
        return result.text
    except ExtractionError as e:
        if e.code == "missing_library":
            logger.error("Library not installed: %s", e)
            raise ImportError(str(e))
        if e.code == "empty_content":
            logger.warning("extract_text: no text extracted from %s", file_path.name)
            return ""
        raise


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """バイトデータからテキストを抽出

    extraction モジュールのレジストリに委譲する。
    CSV はこの関数ではなく parse_csv_bytes_to_chunks() を使用すること。
    """
    registry = get_registry()
    try:
        result = registry.extract_from_bytes(data, filename)
        return result.text
    except ExtractionError as e:
        if e.code == "missing_library":
            logger.error("Library not installed: %s", e)
            raise ImportError(str(e))
        if e.code == "empty_content":
            logger.warning("extract_text_from_bytes: no text extracted from %s", filename)
            return ""
        raise


def parse_csv_to_chunks(file_path: Path) -> list[str]:
    """CSV ファイルを行ごとのチャンクに分割（ヘッダー付与）

    §4.9.9: CSV は「行ごとに1チャンク（ヘッダーを各行に付与）」。
    各チャンクは「ヘッダー行\\nデータ行」の形式。

    内部実装は :class:`backend.io.ChunkedReader` 経由のストリーミング読込。
    旧実装が ``read_text()`` で全件メモリ展開していた分のピークを抑制する。
    戻り値型は後方互換のため ``list[str]`` を維持。
    """
    reader = ChunkedReader(file_path, strategy=csv_row_strategy)
    return [chunk.text for chunk in reader if chunk.text is not None]


def parse_csv_bytes_to_chunks(data: bytes) -> list[str]:
    """CSV バイトデータを行ごとのチャンクに分割（ヘッダー付与）"""
    text = data.decode("utf-8", errors="replace")
    return _parse_csv_text_to_chunks(text)


def _parse_csv_text_to_chunks(text: str) -> list[str]:
    """CSV テキストを行ごとのチャンクに分割（ヘッダー付与）"""
    if not text.strip():
        return []

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if len(rows) < 2:
        # ヘッダーのみ or 空の場合はそのままテキストとして返す
        if rows and any(cell.strip() for cell in rows[0]):
            return [",".join(rows[0])]
        return []

    header = rows[0]
    header_line = ",".join(header)
    chunks: list[str] = []

    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        data_line = ",".join(row)
        chunks.append(f"{header_line}\n{data_line}")

    logger.debug(
        "_parse_csv_text_to_chunks: %d data rows -> %d chunks",
        len(rows) - 1, len(chunks),
    )
    return chunks
