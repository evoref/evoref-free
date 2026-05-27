"""ファイル読込み・テキスト抽出・チャンク分割モジュール

/file コマンドで指定されたファイルを読込み、テキスト抽出・チャンク分割して
LLM コンテキストに注入可能な形式に変換する。

ビジネスロジックは backend.free.services.file_service に委譲する。
"""

from __future__ import annotations

from backend.free.services.file_service import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    LEGACY_FORMATS,
    FileReadResult,
    FileServiceError,
    get_supported_extensions,
    prepare_file_context,
    read_and_chunk,
)

# 後方互換のエイリアス
FileReaderError = FileServiceError

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "LEGACY_FORMATS",
    "FileReadResult",
    "FileReaderError",
    "FileServiceError",
    "get_supported_extensions",
    "prepare_file_context",
    "read_and_chunk",
]
