"""ファイル読込み・テキスト抽出・チャンク分割サービス

CLI の /file コマンドおよび GUI のファイルチャンキング API から利用される
共通ビジネスロジック。テキスト抽出は backend.extraction モジュールに委譲する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.extraction import get_registry
from backend.extraction.base import ExtractionError
from backend.extraction.encoding import is_likely_text
from backend.log_config import get_logger
from backend.utils import estimate_tokens, split_by_scope
from backend.free.rag.chunker import SemanticChunker

# Extractor 登録を保証
import backend.free.extraction  # noqa: F401

logger = get_logger("services.file_service")

# 旧形式（非対応）
LEGACY_FORMATS = {".doc", ".xls", ".ppt"}

# デフォルトのチャンク設定
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128


@dataclass
class FileReadResult:
    """ファイル読込み結果"""
    filename: str
    path: str
    chunks: list[str]
    total_chars: int
    chunk_count: int
    file_type: str  # "text" | "pdf" | "docx" | "xlsx" | "pptx" | etc.


class FileServiceError(Exception):
    """ファイルサービスエラー"""
    pass


def read_and_chunk(
    path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> FileReadResult:
    """ファイルを読込み、テキスト抽出・チャンク分割する

    Args:
        path: ファイルパス
        chunk_size: チャンクサイズ（トークン数）
        chunk_overlap: チャンクオーバーラップ（トークン数）

    Returns:
        FileReadResult: 読込み結果

    Raises:
        FileServiceError: 読込みエラー
    """
    suffix = path.suffix.lower()
    logger.debug("read_and_chunk: path=%s, suffix=%s", path, suffix)

    # 旧形式チェック
    if suffix in LEGACY_FORMATS:
        raise FileServiceError(f"legacy_format:{suffix}")

    # テキスト抽出（extraction モジュールに委譲）
    text = _extract_text(path, suffix)
    file_type = suffix.lstrip(".") if suffix else "text"
    # テキスト系の拡張子は file_type を "text" に統一
    registry = get_registry()
    from backend.free.extraction.extractors.plaintext import PlaintextExtractor
    extractor = registry.get_extractor(suffix)
    if isinstance(extractor, PlaintextExtractor):
        file_type = "text"

    if not text.strip():
        raise FileServiceError("empty_content")

    # チャンク分割
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.chunk(text)

    logger.debug(
        "read_and_chunk: %s -> %d chars, %d chunks",
        path.name, len(text), len(chunks),
    )

    return FileReadResult(
        filename=path.name,
        path=str(path),
        chunks=chunks,
        total_chars=len(text),
        chunk_count=len(chunks),
        file_type=file_type,
    )


def _extract_text(path: Path, suffix: str) -> str:
    """extraction モジュール経由でテキストを抽出"""
    registry = get_registry()

    # レジストリで対応可能か確認
    if registry.is_supported(suffix):
        try:
            result = registry.extract(path)
            return result.text
        except ExtractionError as e:
            _raise_file_service_error(e, path)

    # 拡張子なしファイルの判定
    if not suffix and is_likely_text(path):
        try:
            result = registry.extract(path)
            return result.text
        except ExtractionError as e:
            _raise_file_service_error(e, path)

    raise FileServiceError(f"unsupported_format:{suffix}")


def _raise_file_service_error(e: ExtractionError, path: Path) -> None:
    """ExtractionError を FileServiceError に変換"""
    if e.code == "missing_library":
        msg = str(e)
        lib = msg.split(":")[-1].strip().split(" ")[0] if ":" in msg else msg
        raise FileServiceError(f"missing_library:{lib}")
    if e.code == "empty_content":
        if "scanned PDF" in str(e):
            raise FileServiceError("scan_only_pdf")
        raise FileServiceError("empty_content")
    if e.code == "encoding_error":
        raise FileServiceError(f"encoding_error:{path.name}")
    if e.code == "unsupported_format":
        raise FileServiceError(f"unsupported_format:{path.suffix.lower()}")
    raise FileServiceError(f"{e.code}:{e}")


def prepare_file_context(path: Path, budget: int) -> list[str]:
    """ファイルをコンテキスト予算に収まるチャンクに分割する。

    予算内に収まる場合はそのまま1チャンクとして返す。
    予算超過時はdef/class境界で分割する。

    Args:
        path: ファイルパス
        budget: トークン予算（コンテキスト残余）

    Returns:
        チャンクのリスト

    Raises:
        FileServiceError: ファイル読込みエラー
    """
    suffix = path.suffix.lower()
    logger.debug("prepare_file_context: path=%s, budget=%d", path, budget)

    # 旧形式チェック
    if suffix in LEGACY_FORMATS:
        raise FileServiceError(f"legacy_format:{suffix}")

    # テキスト抽出
    text = _extract_text(path, suffix)

    if not text.strip():
        raise FileServiceError("empty_content")

    # 予算内: そのまま返す
    if estimate_tokens(text) <= budget:
        logger.debug(
            "prepare_file_context: %s fits in budget (%d tokens <= %d)",
            path.name, estimate_tokens(text), budget,
        )
        return [text]

    # 予算超過: def/class境界で分割
    chunks = split_by_scope(text, budget)
    logger.info(
        "File split into %d chunks: %s (budget=%d)",
        len(chunks), path.name, budget,
    )
    return chunks


def get_supported_extensions() -> set[str]:
    """対応する全ファイル拡張子を返す"""
    registry = get_registry()
    return set(registry.supported_extensions())
