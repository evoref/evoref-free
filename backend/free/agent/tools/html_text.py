"""HTML → プレーンテキスト抽出

``fetch_url`` が取得した HTML から本文だけを取り出す層。ナビゲーションや
リンク密度の高いブロックを落とし、見出しは ``#`` 行、表は Markdown へ畳む。
実装は corpus の HTML 抽出器と共有する ``backend.free.extraction.html_structure``
(不変則 #14(a))。取得そのものは ``web_fetch`` の責務。
"""

from __future__ import annotations

from backend.free.extraction.extractors._document_parts import (
    GFM_TABLE_SEP_LINE_RE as _MD_TABLE_SEP_LINE_RE,
)
from backend.free.extraction.html_structure import (
    BOILERPLATE_ATTR_RE as _BOILERPLATE_ATTR_RE,
    EXTRACTION_MIN_RETAIN_RATIO as _EXTRACTION_MIN_RETAIN_RATIO,
    LINK_DENSE_BLOCK_TAGS as _LINK_DENSE_BLOCK_TAGS,
    LINK_DENSITY_MIN_LINKS as _LINK_DENSITY_MIN_LINKS,
    LINK_DENSITY_MIN_TEXT as _LINK_DENSITY_MIN_TEXT,
    LINK_DENSITY_THRESHOLD as _LINK_DENSITY_THRESHOLD,
    MAIN_CONTENT_SELECTORS as _MAIN_CONTENT_SELECTORS,
    STRIP_TAGS as _STRIP_TAGS,
    collapse_blank_lines as _collapse_blank_lines,
    contains_markdown_table as _contains_markdown_table,
    extract_main_content as _extract_main_content,
    extract_naive as _extract_naive,
    flatten_tables as _flatten_tables,
    has_boilerplate_attr as _has_boilerplate_attr,
    html_to_text,
    prune_link_dense_blocks as _prune_link_dense_blocks,
    select_main_root as _select_main_root,
    strip_html_fallback as _strip_html_fallback,
)



def _html_to_text(html: str) -> str:
    """取得した HTML の本文。ナビ・ヘッダ・フッタ等の強い除去は ``fetch_url`` だけに掛ける。"""
    return html_to_text(html, strip_boilerplate=True)


__all__ = [
    "_BOILERPLATE_ATTR_RE",
    "_EXTRACTION_MIN_RETAIN_RATIO",
    "_LINK_DENSE_BLOCK_TAGS",
    "_LINK_DENSITY_MIN_LINKS",
    "_LINK_DENSITY_MIN_TEXT",
    "_LINK_DENSITY_THRESHOLD",
    "_MAIN_CONTENT_SELECTORS",
    "_MD_TABLE_SEP_LINE_RE",
    "_STRIP_TAGS",
    "_collapse_blank_lines",
    "_contains_markdown_table",
    "_extract_main_content",
    "_extract_naive",
    "_flatten_tables",
    "_has_boilerplate_attr",
    "_html_to_text",
    "_prune_link_dense_blocks",
    "_select_main_root",
    "_strip_html_fallback",
]
