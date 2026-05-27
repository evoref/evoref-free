"""CSV / TSV Writer

.csv, .tsv: テーブルデータをCSV/TSV形式で書出し
"""

from __future__ import annotations

import csv
import io
from typing import override

from backend.export._writer_base import BytesWriterBase
from backend.export.base import ExportContent, ExportError

# UTF-8 BOM（Excel 互換）
_UTF8_BOM = b"\xef\xbb\xbf"


def _extract_table_data(content: ExportContent) -> list[list[str]]:
    """ExportContent からテーブルデータを抽出"""
    # raw_data 優先
    if content.raw_data is not None:
        if isinstance(content.raw_data, list) and content.raw_data:
            first = content.raw_data[0]
            if isinstance(first, dict):
                # list[dict] → ヘッダー行 + データ行
                headers = list(first.keys())
                rows = [headers]
                for item in content.raw_data:
                    rows.append([str(item.get(h, "")) for h in headers])
                return rows
            if isinstance(first, (list, tuple)):
                return [list(map(str, row)) for row in content.raw_data]

    # blocks 中のテーブルを検出
    for block in content.blocks:
        if block.type == "table" and block.rows:
            return block.rows

    return []


class CsvTsvWriter(BytesWriterBase):
    """CSV / TSV ファイル Writer"""

    @property
    @override
    def extensions(self) -> frozenset[str]:
        return frozenset({".csv", ".tsv"})

    @override
    def _render_bytes(self, content: ExportContent, ext: str) -> bytes:
        """CSV/TSV バイトデータを生成"""
        rows = _extract_table_data(content)
        if not rows:
            raise ExportError(
                "no_table_data",
                "No table data found in content for CSV/TSV export",
            )

        delimiter = "\t" if ext == ".tsv" else ","
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
        writer.writerows(rows)
        text = buf.getvalue()

        # UTF-8 BOM 付き（Excel 互換）
        return _UTF8_BOM + text.encode("utf-8")
