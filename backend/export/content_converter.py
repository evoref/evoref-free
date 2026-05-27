"""Markdown → 構造化コンテンツ変換

外部ライブラリ不使用（正規表現ベース）。
対応構造: heading, paragraph, code block, table, list (ordered/unordered),
          blockquote, horizontal rule, inline formatting (bold/italic/code)
"""

from __future__ import annotations

import re
from typing import Any

from backend.export.base import ContentBlock, ExportContent

# パターン定義
_RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_RE_CODE_FENCE = re.compile(r"^```(\w*)$")
_RE_HR = re.compile(r"^(?:---|\*\*\*|___)\s*$")
_RE_TABLE_ROW = re.compile(r"^\|(.+)\|$")
_RE_TABLE_SEP = re.compile(r"^\|[\s\-:|]+\|$")
_RE_UL = re.compile(r"^(\s*)[-*+]\s+(.+)$")
_RE_OL = re.compile(r"^(\s*)\d+[.)]\s+(.+)$")
_RE_QUOTE = re.compile(r"^>\s?(.*)")


class ContentConverter:
    """Markdown テキストを ContentBlock リストに変換する"""

    @staticmethod
    def _parse_code_fence(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """``` フェンスコードブロックを 1 件パース。マッチしなければ ``(None, i)``。"""
        m = _RE_CODE_FENCE.match(lines[i])
        if not m:
            return None, i
        lang = m.group(1)
        code_lines: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            code_lines.append(lines[i])
            i += 1
        i += 1  # 閉じ ``` をスキップ
        return ContentBlock(
            type="code",
            content="\n".join(code_lines),
            language=lang,
        ), i

    @staticmethod
    def _parse_heading(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """`# 見出し` をパース。マッチしなければ ``(None, i)``。"""
        m = _RE_HEADING.match(lines[i])
        if not m:
            return None, i
        return ContentBlock(
            type="heading",
            content=m.group(2).strip(),
            level=len(m.group(1)),
        ), i + 1

    @staticmethod
    def _parse_hr(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """水平線をパース。"""
        if not _RE_HR.match(lines[i]):
            return None, i
        return ContentBlock(type="hr", content=""), i + 1

    @staticmethod
    def _parse_table(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """`|...|...|` テーブルをパース。"""
        if not _RE_TABLE_ROW.match(lines[i]):
            return None, i
        table_rows: list[list[str]] = []
        while i < len(lines) and _RE_TABLE_ROW.match(lines[i]):
            if _RE_TABLE_SEP.match(lines[i]):
                i += 1
                continue
            cells = [
                c.strip() for c in lines[i].strip().strip("|").split("|")
            ]
            table_rows.append(cells)
            i += 1
        return ContentBlock(type="table", content="", rows=table_rows), i

    @staticmethod
    def _parse_quote(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """`>` 引用をパース。"""
        if not _RE_QUOTE.match(lines[i]):
            return None, i
        quote_lines: list[str] = []
        while i < len(lines):
            qm = _RE_QUOTE.match(lines[i])
            if qm:
                quote_lines.append(qm.group(1))
                i += 1
            else:
                break
        return ContentBlock(
            type="quote", content="\n".join(quote_lines),
        ), i

    @staticmethod
    def _parse_unordered_list(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """`- item` 順序なしリストをパース。"""
        if not _RE_UL.match(lines[i]):
            return None, i
        items: list[str] = []
        while i < len(lines):
            um = _RE_UL.match(lines[i])
            if um:
                items.append(um.group(2))
                i += 1
            else:
                break
        return ContentBlock(
            type="list", content="", ordered=False, items=items,
        ), i

    @staticmethod
    def _parse_ordered_list(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """`1. item` 順序付きリストをパース。"""
        if not _RE_OL.match(lines[i]):
            return None, i
        items: list[str] = []
        while i < len(lines):
            om = _RE_OL.match(lines[i])
            if om:
                items.append(om.group(2))
                i += 1
            else:
                break
        return ContentBlock(
            type="list", content="", ordered=True, items=items,
        ), i

    @staticmethod
    def _is_block_start(line: str) -> bool:
        """段落終了判定: 次の構造要素の開始行か？"""
        return bool(
            _RE_HEADING.match(line)
            or _RE_CODE_FENCE.match(line)
            or _RE_HR.match(line)
            or _RE_TABLE_ROW.match(line)
            or _RE_QUOTE.match(line)
            or _RE_UL.match(line)
            or _RE_OL.match(line)
        )

    @classmethod
    def _parse_paragraph(
        cls, lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """段落 (連続行) をパース。次の構造要素開始行で停止。"""
        para_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            if cls._is_block_start(lines[i]) and para_lines:
                break
            para_lines.append(lines[i])
            i += 1
        if not para_lines:
            return None, i
        return ContentBlock(
            type="paragraph", content="\n".join(para_lines),
        ), i

    def convert(self, markdown: str) -> list[ContentBlock]:
        """Markdown → ContentBlock リスト"""
        lines = markdown.split("\n")
        blocks: list[ContentBlock] = []
        parsers = (
            self._parse_code_fence,
            self._parse_heading,
            self._parse_hr,
            self._parse_table,
            self._parse_quote,
            self._parse_unordered_list,
            self._parse_ordered_list,
        )
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue

            matched = False
            for parser in parsers:
                block, new_i = parser(lines, i)
                if block is not None:
                    blocks.append(block)
                    i = new_i
                    matched = True
                    break
            if matched:
                continue

            block, i = self._parse_paragraph(lines, i)
            if block is not None:
                blocks.append(block)
        return blocks

    @staticmethod
    def from_markdown(markdown: str, title: str = "", **metadata: Any) -> ExportContent:
        """Markdown テキストから ExportContent を一括生成"""
        converter = ContentConverter()
        blocks = converter.convert(markdown)

        # タイトル自動検出: 最初の heading があればそれを使用
        auto_title = title
        if not auto_title:
            for block in blocks:
                if block.type == "heading":
                    auto_title = block.content
                    break

        return ExportContent(
            title=auto_title,
            blocks=blocks,
            metadata=dict(metadata),
            raw_markdown=markdown,
        )

    @staticmethod
    def from_data(
        data: list[dict] | list[list],
        title: str = "",
        **metadata: Any,
    ) -> ExportContent:
        """構造化データから ExportContent を生成（CSV/XLSX/JSON 用）"""
        return ExportContent(
            title=title,
            blocks=[],
            metadata=dict(metadata),
            raw_data=data,
        )
