"""スライド分割 — ``.pptx`` / ``.odp`` 共通

設計書: [docs/f_11_file_export.md](../../docs/f_11_file_export.md) §6。

以前は ``writers/pptx.py`` と ``writers/odf.py`` に同じロジックが複製されており、
**同じ欠陥を 2 箇所に抱えていた**。分割規則はここが唯一の実装。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.export.base import ContentBlock, ExportContent


@dataclass(frozen=True)
class Slide:
    """1 枚のスライド。"""

    title: str
    blocks: list[ContentBlock] = field(default_factory=list)


@dataclass(frozen=True)
class SlideDeck:
    """表紙 (任意) + 本文スライド列。"""

    cover_title: str
    slides: list[Slide] = field(default_factory=list)


def _resolve_blocks(content: ExportContent) -> list[ContentBlock]:
    """``blocks`` が空なら ``raw_markdown`` から作り直す。"""
    if content.blocks:
        return content.blocks
    if not content.raw_markdown:
        return []
    from backend.export.content_converter import ContentConverter

    return ContentConverter().convert(content.raw_markdown)


def split_into_slides(content: ExportContent) -> SlideDeck:
    """``ExportContent`` を表紙 + スライド列へ分割する。

    規則 (f_11 §6):

    - ``heading`` の ``level <= 2`` が分割点。
    - **中身 (blocks) が空のスライドは作らない**。
    - 中身を持たない先頭の見出しは **表紙** になる。無ければ ``content.title`` を
      使い、最初のスライドの見出しと同じなら表紙を作らない。
    - ``heading`` level>=3 と ``hr`` はスライド本文の一部として残す
      (Writer 側が描く。落とさない)。

    以前は ``current_title`` を ``content.title`` で初期化したまま先頭見出しで
    flush していた。``ContentConverter.from_markdown`` は先頭見出しを title へ
    自動採用するので条件が初回から真になり、**中身の無い同名スライドが 2 枚
    先頭に付いていた** (2026-09-16 実測: 4 枚要求して 6 枚、うち先頭 2 枚が空)。
    """
    blocks = _resolve_blocks(content)

    slides: list[Slide] = []
    cover_title = ""
    current_title = ""
    current_blocks: list[ContentBlock] = []

    def flush() -> None:
        """溜まった分を確定する。中身が無ければスライドにしない。"""
        nonlocal cover_title, current_blocks
        if current_blocks:
            slides.append(Slide(current_title, current_blocks))
        elif current_title and not slides and not cover_title:
            cover_title = current_title
        current_blocks = []

    for block in blocks:
        if block.type == "heading" and block.level <= 2:
            flush()
            current_title = block.content
        else:
            current_blocks.append(block)
    flush()

    # 見出しが 1 つも無い入力ではスライドの題が空になる。文書タイトルを流用する。
    if content.title:
        slides = [
            Slide(s.title or content.title, s.blocks) for s in slides
        ]
        if not cover_title and (not slides or slides[0].title != content.title):
            cover_title = content.title

    if not cover_title and not slides:
        cover_title = content.title or "Untitled"

    return SlideDeck(cover_title, slides)
