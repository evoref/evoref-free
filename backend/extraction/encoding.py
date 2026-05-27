"""マルチエンコーディング検出ユーティリティ

テキストファイルのエンコーディングを自動検出する共通ロジック。
CLI / RAG / Agent の全モジュールで共通使用する。
"""

from __future__ import annotations

from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("extraction.encoding")

# 試行するエンコーディングの順序（日本語環境向け）
ENCODING_CANDIDATES = [
    "utf-8",
    "utf-8-sig",
    "cp932",
    "shift_jis",
    "euc-jp",
    "latin-1",  # フォールバック（ほぼ全バイト列をデコード可能）
]


def read_text_with_encoding(
    path: Path,
    *,
    encodings: list[str] | None = None,
) -> tuple[str, str]:
    """テキストファイルを適切なエンコーディングで読み込む

    Args:
        path: ファイルパス
        encodings: 試行するエンコーディングリスト（省略時はデフォルト）

    Returns:
        (テキスト内容, 検出されたエンコーディング名) のタプル

    Raises:
        UnicodeDecodeError: 全エンコーディングでデコード失敗
    """
    candidates = encodings or ENCODING_CANDIDATES

    for enc in candidates:
        try:
            text = path.read_text(encoding=enc)
            logger.debug(
                "read_text_with_encoding: %s decoded with %s (%d chars)",
                path.name, enc, len(text),
            )
            return text, enc
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise UnicodeDecodeError(
        "multi", b"", 0, 1,
        f"All encoding candidates failed for {path.name}: {candidates}",
    )


def decode_bytes_with_encoding(
    data: bytes,
    *,
    encodings: list[str] | None = None,
) -> tuple[str, str]:
    """バイトデータを適切なエンコーディングでデコードする

    Args:
        data: バイトデータ
        encodings: 試行するエンコーディングリスト（省略時はデフォルト）

    Returns:
        (テキスト内容, 検出されたエンコーディング名) のタプル

    Raises:
        UnicodeDecodeError: 全エンコーディングでデコード失敗
    """
    candidates = encodings or ENCODING_CANDIDATES

    for enc in candidates:
        try:
            text = data.decode(enc)
            return text, enc
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise UnicodeDecodeError(
        "multi", b"", 0, 1,
        f"All encoding candidates failed: {candidates}",
    )


def is_likely_text(path: Path, sample_size: int = 512) -> bool:
    """拡張子がなくてもテキストファイルの可能性があるか判定

    先頭バイトに NULL バイトが含まれなければテキストと判定する。
    """
    try:
        with open(path, "rb") as f:
            sample = f.read(sample_size)
        return b"\x00" not in sample
    except OSError:
        return False
