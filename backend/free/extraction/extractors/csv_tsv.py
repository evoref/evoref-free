"""CSV/TSV Extractor

.csv, .tsv ファイルからテキストを抽出する。
stdlib csv モジュールを使用。
RAG 用ヘッダー付与チャンクは RAG 層（text_extractor.py）に残す。
"""

from __future__ import annotations

import csv
import io
from typing import Any, override

from backend.extraction._text_source_base import TextSourceExtractorBase


class CsvTsvExtractor(TextSourceExtractorBase):
    """CSV/TSV ファイルからテキストを抽出"""

    @property
    @override
    def extensions(self) -> frozenset[str]:
        return frozenset({".csv", ".tsv"})

    @override
    def _process(self, text: str, source_name: str) -> str:
        """CSV/TSV を行ごとのタブ区切りテキストに整形"""
        delimiter = "\t" if source_name.lower().endswith(".tsv") else ","
        return self._parse_csv(text, delimiter)

    @override
    def _build_metadata(
        self, encoding: str, processed_text: str,
    ) -> dict[str, Any]:
        """エンコーディングと行数を返す"""
        rows = processed_text.count("\n") + 1 if processed_text else 0
        return {"encoding": encoding, "rows": rows}

    @staticmethod
    def _parse_csv(text: str, delimiter: str) -> str:
        """CSV テキストをタブ区切りテキストに変換"""
        if not text.strip():
            return ""

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        lines: list[str] = []
        for row in reader:
            if any(cell.strip() for cell in row):
                lines.append("\t".join(row))

        return "\n".join(lines)
