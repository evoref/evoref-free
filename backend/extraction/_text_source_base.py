"""テキストソースを扱う Extractor の共通テンプレート

`plaintext` / `html` / `csv_tsv` 等、ソースをテキストとして読み込む Extractor は
以下のパターンが完全に重複していた:

1. ``read_text_with_encoding(path)`` または ``decode_bytes_with_encoding(data)``
   でエンコーディング検出付きでテキスト化
2. ``UnicodeDecodeError`` を ``ExtractionError("encoding_error", ...)`` に変換
3. ソース固有の加工 (HTML タグ除去 / CSV パース / そのまま等) を適用
4. 加工結果が空なら ``ExtractionError("empty_content", ...)`` を raise
5. ``ExtractionResult(text=..., metadata={"encoding": ...})`` を構築

本モジュールは ``TextSourceExtractorBase`` 抽象基底クラスを提供し、
サブクラスは ``extensions`` プロパティと ``_process(text, source_name) -> str`` を
実装するだけでよい。エンコーディング以外の metadata を追加したい場合は
``_build_metadata`` をオーバーライドする。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from backend.extraction.base import ExtractionError, ExtractionResult
from backend.extraction.encoding import (
    decode_bytes_with_encoding,
    read_text_with_encoding,
)
from backend.log_config import get_logger

logger = get_logger("extraction._text_source_base")


class TextSourceExtractorBase(ABC):
    """テキストソースから抽出する Extractor の共通テンプレート

    ``TextExtractor`` プロトコルを満たすため、サブクラスは以下を実装する:

    - ``extensions`` プロパティ
    - ``_process(text, source_name) -> str`` メソッド (生テキストの加工)

    ``requires`` と ``is_available`` はデフォルト実装 (依存なし) を提供。
    必要に応じてサブクラスでオーバーライドする。
    """

    @property
    @abstractmethod
    def extensions(self) -> frozenset[str]:
        """対応する拡張子集合"""

    @property
    def requires(self) -> list[str]:
        """必要な pip パッケージ名 (デフォルト: なし)"""
        return []

    def is_available(self) -> bool:
        """必要ライブラリがインストール済みか (デフォルト: True)"""
        return True

    @abstractmethod
    def _process(self, text: str, source_name: str) -> str:
        """生テキストを加工したテキストを返す

        Args:
            text: エンコーディング検出後の生テキスト
            source_name: ファイル名 (path.name または extract_from_bytes の filename)。
                拡張子による分岐や empty_content エラーメッセージに利用する
        """

    def _build_metadata(
        self, encoding: str, processed_text: str,
    ) -> dict[str, Any]:
        """ExtractionResult.metadata を構築する

        デフォルトは ``{"encoding": encoding}`` を返す。
        サブクラスで rows / pages 等を追加したい場合はオーバーライドする。
        """
        return {"encoding": encoding}

    def extract(self, path: Path) -> ExtractionResult:
        """ファイルからテキストを抽出"""
        try:
            raw_text, encoding = read_text_with_encoding(path)
        except UnicodeDecodeError:
            raise ExtractionError(
                "encoding_error",
                f"Failed to decode {path.name} with any supported encoding",
            )
        return self._build_result(raw_text, encoding, path.name)

    def extract_from_bytes(
        self, data: bytes, filename: str,
    ) -> ExtractionResult:
        """バイトデータからテキストを抽出"""
        try:
            raw_text, encoding = decode_bytes_with_encoding(data)
        except UnicodeDecodeError:
            raise ExtractionError(
                "encoding_error",
                f"Failed to decode {filename} with any supported encoding",
            )
        return self._build_result(raw_text, encoding, filename)

    def _build_result(
        self, raw_text: str, encoding: str, source_name: str,
    ) -> ExtractionResult:
        """加工 + 空チェック + ExtractionResult 構築の共通処理"""
        processed = self._process(raw_text, source_name)
        if not processed.strip():
            raise ExtractionError(
                "empty_content", f"No content in {source_name}",
            )
        result = ExtractionResult(
            text=processed,
            metadata=self._build_metadata(encoding, processed),
        )
        logger.debug(
            "%s: %s -> %d chars (encoding=%s)",
            type(self).__name__, source_name, len(processed), encoding,
        )
        return result
