"""ファイル書出しの基本型定義

ExportContent, WriteResult, FileWriter Protocol, ExportError を提供する。
各形式の Writer はこのモジュールの Protocol を実装する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ContentBlock:
    """コンテンツの構成要素"""
    type: str       # "heading", "paragraph", "code", "table", "list", "quote", "hr"
    content: str    # テキスト内容
    level: int = 0  # heading レベル (1-6)
    language: str = ""  # code block の言語
    rows: list[list[str]] = field(default_factory=list)  # table の行データ
    ordered: bool = False  # list の順序付き/なし
    items: list[str] = field(default_factory=list)  # list の項目


@dataclass
class ExportContent:
    """書出し対象コンテンツ"""
    title: str = ""
    blocks: list[ContentBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata 例: {"author": "evoref", "date": "2026-03-23", "language": "ja"}
    raw_markdown: str = ""  # 元の Markdown テキスト（フォールバック用）
    raw_data: Any = None    # 構造化データ（csv/xlsx/json 用: list[dict] 等）


@dataclass
class WriteResult:
    """書出し結果"""
    path: Path | None = None
    data: bytes | None = None  # バイトデータ出力（API レスポンス用）
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class FileWriter(Protocol):
    """ファイル書出しプロトコル — 各形式はこれを実装"""

    @property
    def extensions(self) -> frozenset[str]:
        """対応する拡張子の集合（例: frozenset({".txt", ".md"})）"""
        ...

    @property
    def requires(self) -> list[str]:
        """必要な pip パッケージ名"""
        ...

    def write(self, content: ExportContent, path: Path) -> WriteResult:
        """ファイルに書出し"""
        ...

    def write_to_bytes(self, content: ExportContent, ext: str) -> bytes:
        """バイトデータとして書出し（API ダウンロード用）"""
        ...

    def is_available(self) -> bool:
        """必要ライブラリがインストール済みか"""
        ...


class ExportError(Exception):
    """書出しエラー"""
    def __init__(self, code: str, message: str = ""):
        self.code = code  # unsupported_format, missing_library, conversion_error, etc.
        super().__init__(message or code)
