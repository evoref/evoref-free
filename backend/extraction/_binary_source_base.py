"""バイナリソースを扱う Extractor の共通テンプレート

`pdf` / `docx` / `xlsx` / `odf` / `pptx` 等、外部ライブラリで
バイナリ (zip / OLE 等) を処理する Extractor は、以下のパターンが完全に重複していた:

1. ライブラリの遅延 import (失敗時 ``ExtractionError("missing_library", ...)``)
2. ``Lib(str(path))`` または ``Lib(io.BytesIO(data))`` でリソースを開く
3. プライベートヘルパで ``(text, metadata)`` を抽出
4. ``ExtractionError`` を再 raise / その他の ``Exception`` を ``ExtractionError("xxx_error", ...)`` に変換
5. 加工結果が空なら ``ExtractionError("empty_content", ...)`` を raise
6. ``ExtractionResult(text, metadata)`` を構築

本モジュールは ``BinarySourceExtractorBase`` 抽象基底クラスを提供し、サブクラスは
以下のみを実装すればよい:

- ``extensions`` プロパティ
- ``error_code`` プロパティ (``Exception`` 変換時の ``ExtractionError`` code)
- ``_extract_from_source(source, source_name)`` メソッド (実際の抽出ロジック)
- (依存ライブラリがあれば) ``requires`` / ``is_available``

``source`` は ``Path`` (extract 経路) または ``BinaryIO`` (extract_from_bytes 経路) で、
サブクラスは必要に応じて型分岐して扱う。
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO

from backend.extraction.base import ExtractionError, ExtractionResult
from backend.log_config import get_logger

logger = get_logger("extraction._binary_source_base")


class BinarySourceExtractorBase(ABC):
    """バイナリソースから抽出する Extractor の共通テンプレート

    ``TextExtractor`` プロトコルを満たすため、サブクラスは以下を実装する:

    - ``extensions`` プロパティ
    - ``error_code`` プロパティ
    - ``_extract_from_source(source, source_name) -> tuple[str, dict]``

    ``requires`` と ``is_available`` はデフォルト実装 (依存なし) を提供。
    通常は依存ライブラリがあるためサブクラスでオーバーライドする。
    """

    @property
    @abstractmethod
    def extensions(self) -> frozenset[str]:
        """対応する拡張子集合"""

    @property
    @abstractmethod
    def error_code(self) -> str:
        """``Exception`` を ``ExtractionError`` に変換する際の code (例: ``"pdf_error"``)"""

    @property
    def requires(self) -> list[str]:
        """必要な pip パッケージ名 (デフォルト: なし)"""
        return []

    def is_available(self) -> bool:
        """必要ライブラリがインストール済みか (デフォルト: True)"""
        return True

    @abstractmethod
    def _extract_from_source(
        self,
        source: Path | BinaryIO,
        source_name: str,
    ) -> tuple[str, dict[str, Any]]:
        """ソースから ``(text, metadata)`` を抽出する

        Args:
            source: ``Path`` (``extract`` 経路) または ``BinaryIO`` (``extract_from_bytes`` 経路)。
                ライブラリが ``str`` パスを要求する場合はサブクラス側で
                ``str(source)`` に変換する。
            source_name: ファイル名 (``path.name`` または extract_from_bytes の ``filename``)。
                エラーメッセージや拡張子分岐に利用する。

        Returns:
            抽出テキストと metadata の dict。
        """

    def extract(self, path: Path) -> ExtractionResult:
        """ファイルからテキストを抽出"""
        return self._run(path, path.name)

    def extract_from_bytes(
        self, data: bytes, filename: str,
    ) -> ExtractionResult:
        """バイトデータからテキストを抽出"""
        return self._run(io.BytesIO(data), filename)

    def _run(
        self,
        source: Path | BinaryIO,
        source_name: str,
    ) -> ExtractionResult:
        """try/except + empty チェック + ExtractionResult 構築の共通処理"""
        try:
            text, metadata = self._extract_from_source(source, source_name)
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(
                self.error_code,
                f"{type(self).__name__} failed on {source_name}: {e}",
            )

        if not text.strip():
            raise ExtractionError(
                "empty_content", f"No content in {source_name}",
            )

        logger.debug(
            "%s: %s -> %d chars",
            type(self).__name__, source_name, len(text),
        )
        return ExtractionResult(text=text, metadata=metadata)
