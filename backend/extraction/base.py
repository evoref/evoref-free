"""テキスト抽出の基本型定義

ExtractionResult, TextExtractor Protocol, ExtractionError を提供する。
各形式の Extractor はこのモジュールの Protocol を実装する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ExtractionResult:
    """テキスト抽出結果"""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata 例: {"pages": 5}, {"sheets": ["Sheet1", "Sheet2"]},
    #              {"slides": 10}, {"encoding": "utf-8"}


class TextExtractor(Protocol):
    """テキスト抽出プロトコル — 各形式はこれを実装"""

    @property
    def extensions(self) -> frozenset[str]:
        """対応する拡張子の集合（例: frozenset({".txt", ".md"})）"""
        ...

    @property
    def requires(self) -> list[str]:
        """必要な pip パッケージ名"""
        ...

    def extract(self, path: Path) -> ExtractionResult:
        """ファイルパスからテキストを抽出"""
        ...

    def extract_from_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        """バイトデータからテキストを抽出"""
        ...

    def is_available(self) -> bool:
        """必要ライブラリがインストール済みか"""
        ...


class ExtractionError(Exception):
    """テキスト抽出エラー"""
    def __init__(self, code: str, message: str = ""):
        self.code = code  # unsupported_format, missing_library, empty_content, etc.
        super().__init__(message or code)
