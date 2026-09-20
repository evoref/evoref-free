"""ODF Writer

.odt, .ods, .odp: OpenDocument 形式で書出し
odfdo を使用。1 Writer で 3 拡張子を処理（拡張子で分岐）。

設計書: [docs/f_11_file_export.md](../../../../docs/f_11_file_export.md)

2026-09-16 に odfpy から odfdo へ移行した (f_11 §7)。odfpy は 1.4.1 (2020-01) で
更新が止まっており、図形・画像の要素型を持たない。
"""

from __future__ import annotations

import io

from backend.export._writer_base import BytesWriterBase
from backend.export.base import (
    ContentBlock,
    ExportContent,
    ExportError,
    build_item_tree,
    coerce_cell_value,
)
from backend.export.media import (
    export_base_dir,
    resolve_image_path,
    scaled_width_cm,
)
from backend.export.shapes import normalize_shapes
from backend.export.slide_splitter import Slide, split_into_slides
from backend.log_config import get_logger

logger = get_logger("export.writers.odf")

#: コードブロック用の段落スタイル名。
_CODE_STYLE = "EvorefCodeBlock"


def _image_frame(doc, block: ContentBlock, base_dir, **frame_kwargs):
    """``image`` ブロックを ODF の Frame にする。解決できなければ ``None``。

    画像の実体は ``Document.add_file`` でパッケージの ``Pictures/`` へ入る。
    """
    from odfdo import Frame

    path = resolve_image_path(block.src, base_dir)
    if path is None:
        return None
    uri = doc.add_file(str(path))
    width_cm = scaled_width_cm(path)
    size = frame_kwargs.pop("size", (f"{width_cm:.2f}cm", f"{width_cm * 0.75:.2f}cm"))
    return Frame.image_frame(
        uri, text=block.content or None, size=size, **frame_kwargs,
    )


def _resolve_blocks(content: ExportContent) -> list[ContentBlock]:
    if content.blocks:
        return content.blocks
    if not content.raw_markdown:
        return []
    from backend.export.content_converter import ContentConverter

    return ContentConverter().convert(content.raw_markdown)


def _build_list_element(block: ContentBlock):
    """入れ子の ``List`` / ``ListItem`` を組む (odt / odp 共通)。

    odfdo は ``ListItem`` の中に子 ``List`` を置く形で入れ子を表す
    (f_11 §「odt/odp の入れ子リスト」)。平らなリスト (nesting 無し) では
    ``List`` 直下に ``ListItem`` を並べるだけになり、従来と同じ構造になる。
    """
    from odfdo import List, ListItem, Paragraph

    def render(nodes) -> "List":
        lst = List()
        for node in nodes:
            item = ListItem(Paragraph(node.text))
            if node.children:
                item.append(render(node.children))
            lst.append(item)
        return lst

    return render(build_item_tree(block))


def _table_element(rows: list[list[str]], name: str = "Table"):
    """``rows`` から odfdo の Table を組む。"""
    from odfdo import Cell, Row, Table

    table = Table(name)
    for row_data in rows:
        row = Row()
        for cell_text in row_data:
            row.append(Cell(value=coerce_cell_value(cell_text)))
        table.append(row)
    return table


def _build_odt(content: ExportContent) -> bytes:
    """ExportContent を ODT バイトデータに変換"""
    from odfdo import Document, Header, Paragraph, Style

    doc = Document("text")
    doc.insert_style(
        Style(
            "paragraph", name=_CODE_STYLE, area="text",
            font_name="Consolas", font_family="Consolas",
        ),
        automatic=True,
    )
    body = doc.body
    body.clear()
    base_dir = export_base_dir(content)

    for block in _resolve_blocks(content):
        if block.type == "heading":
            body.append(Header(max(1, min(block.level, 6)), block.content))

        elif block.type == "paragraph":
            body.append(Paragraph(block.content))

        elif block.type == "code":
            for line in (block.content or "").split("\n"):
                body.append(Paragraph(line, style=_CODE_STYLE))

        elif block.type == "table":
            if block.rows:
                body.append(_table_element(block.rows))

        elif block.type == "list":
            body.append(_build_list_element(block))

        elif block.type == "quote":
            body.append(Paragraph(f'"{block.content}"'))

        elif block.type == "hr":
            body.append(Paragraph("─" * 24))

        elif block.type == "image":
            frame = _image_frame(doc, block, base_dir, anchor_type="paragraph")
            if frame is not None:
                body.append(Paragraph(frame))

        elif block.type == "shapes":
            # ODT には図形を置かない (f_11 §3.1)。黙って落とさず記録する。
            logger.warning(
                "shapes blocks are not drawn in .odt; %d shape(s) skipped",
                len(block.shapes),
            )

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _extract_ods_tables(content: ExportContent) -> list[tuple[str, list[list[str]]]]:
    """ODS へ書くテーブルを集める。"""
    tables: list[tuple[str, list[list[str]]]] = []
    if content.raw_data is not None:
        if isinstance(content.raw_data, list) and content.raw_data:
            first = content.raw_data[0]
            if isinstance(first, dict):
                headers = list(first.keys())
                rows = [headers]
                for item in content.raw_data:
                    rows.append([str(item.get(h, "")) for h in headers])
                tables.append((content.title or "Sheet1", rows))
            elif isinstance(first, (list, tuple)):
                tables.append(
                    ("Sheet1", [list(map(str, r)) for r in content.raw_data]),
                )
        return tables

    for block in content.blocks:
        if block.type == "table" and block.rows:
            tables.append(("Sheet1", block.rows))
            break
    return tables


def _build_ods(content: ExportContent) -> bytes:
    """ExportContent を ODS バイトデータに変換"""
    from odfdo import Document

    tables = _extract_ods_tables(content)
    if not tables:
        raise ExportError("no_table_data", "No table data for ODS export")

    doc = Document("spreadsheet")
    body = doc.body
    body.clear()
    for sheet_name, rows in tables:
        body.append(_table_element(rows, sheet_name))

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _slide_body_paragraphs(slide: Slide) -> list:
    """スライド本文をテキスト段落へ落とす (table / image / shapes を除く)。

    ``list`` は平らな (入れ子の無い) ときだけ従来どおり ``• `` 接頭の
    段落へ展開する。入れ子があれば odfdo の ``List``/``ListItem`` 構造
    (:func:`_build_list_element`) を段落列に混ぜて返す — ``Frame.text_frame``
    は ``Paragraph`` と ``List`` が混在した列をそのまま受け付ける。
    """
    from odfdo import Paragraph

    paragraphs = []
    for block in slide.blocks:
        if block.type == "list":
            if block.has_nested_items():
                paragraphs.append(_build_list_element(block))
            else:
                paragraphs.extend(Paragraph(f"• {item}") for item in block.items)
        elif block.type == "heading":
            # level<=2 はスライド分割で消費済み。残るのは小見出し。
            paragraphs.append(Paragraph(block.content))
        elif block.type == "hr":
            paragraphs.append(Paragraph("─" * 24))
        elif block.type == "table":
            # ODP のスライドに表要素を置く公開 API が odfdo に無いため、
            # タブ区切りの段落へ落とす。以前は table 分岐が無く、表が
            # 丸ごと消えていた。
            paragraphs.extend(
                Paragraph("\t".join(str(c) for c in row)) for row in block.rows
            )
        elif block.type in ("paragraph", "quote", "code"):
            paragraphs.append(Paragraph(block.content))
    return paragraphs


def _append_shapes(page, block: ContentBlock) -> None:
    """``shapes`` ブロックを ODP のページへ描く (f_11 §4.2)。"""
    from odfdo import EllipseShape, LineShape, RectangleShape

    for shape in normalize_shapes(block.shapes):
        if shape.kind == "line":
            page.append(
                LineShape(
                    p1=(f"{shape.x}cm", f"{shape.y}cm"),
                    p2=(f"{shape.x2}cm", f"{shape.y2}cm"),
                ),
            )
            continue
        cls = RectangleShape if shape.kind == "rect" else EllipseShape
        page.append(
            cls(
                size=(f"{shape.w}cm", f"{shape.h}cm"),
                position=(f"{shape.x}cm", f"{shape.y}cm"),
                text=shape.text or None,
            ),
        )


def _build_odp(content: ExportContent) -> bytes:
    """ExportContent を ODP バイトデータに変換"""
    from odfdo import Document, DrawPage, Frame, Paragraph

    doc = Document("presentation")
    body = doc.body
    body.clear()

    deck = split_into_slides(content)
    base_dir = export_base_dir(content)
    pages: list[tuple[str, Slide | None]] = []
    if deck.cover_title:
        pages.append((deck.cover_title, None))
    pages.extend((s.title, s) for s in deck.slides)

    for index, (title, slide) in enumerate(pages, 1):
        page = DrawPage(f"page{index}", name=title or f"Slide {index}")
        page.append(
            Frame.text_frame(
                Paragraph(title), size=("23cm", "3cm"), position=("1cm", "0.5cm"),
                presentation_class="title",
            ),
        )
        if slide is None:
            body.append(page)
            continue

        paragraphs = _slide_body_paragraphs(slide)
        if paragraphs:
            page.append(
                Frame.text_frame(
                    paragraphs, size=("23cm", "13cm"), position=("1cm", "4cm"),
                    presentation_class="outline",
                ),
            )
        for block in slide.blocks:
            if block.type == "image":
                frame = _image_frame(doc, block, base_dir, position=("2cm", "9cm"))
                if frame is not None:
                    page.append(frame)
            elif block.type == "shapes":
                _append_shapes(page, block)
        body.append(page)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class OdfWriter(BytesWriterBase):
    """ODF ファイル Writer（ODT, ODS, ODP）"""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".odt", ".ods", ".odp"})

    @property
    def requires(self) -> list[str]:
        return ["odfdo"]

    def is_available(self) -> bool:
        try:
            from odfdo import Document  # noqa: F401
            return True
        except ImportError:
            return False

    def _render_bytes(self, content: ExportContent, ext: str) -> bytes:
        """拡張子に応じて ODF を生成"""
        if ext == ".odt":
            return _build_odt(content)
        if ext == ".ods":
            return _build_ods(content)
        if ext == ".odp":
            return _build_odp(content)
        raise ExportError("unsupported_format", f"Unsupported ODF format: {ext}")
