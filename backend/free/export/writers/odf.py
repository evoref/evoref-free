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
    sibling_runs,
)
from backend.export.markdown_patterns import (
    BOLD_KINDS,
    CODE_KINDS,
    ITALIC_KINDS,
    iter_inline_groups,
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


#: 太字 / 斜体の組 → 文字スタイル名 (``_install_inline_styles`` が文書へ入れる)。
_EMPHASIS_STYLE_NAMES = {
    (True, False): "EvorefBold",
    (False, True): "EvorefItalic",
    (True, True): "EvorefBoldItalic",
}
#: inline のコードの文字スタイル。pandoc の odt と同じ名前 (表示名は「Source Text」) で、
#: pandoc 3.8 の odt リーダはこの名前の span だけをコードと読む。共通スタイル
#: (styles.xml) に置く。強調の中のコードは強調の span の中にこの span を入れる。
_CODE_CHAR_STYLE = "Source_Text"
#: 入れ子のリストのスタイル (``text:list-style``)。ordered → 名前。
_LIST_STYLE_NAMES = {False: "EvorefBulletList", True: "EvorefNumberList"}
#: 区切り線 (hr) 用の段落スタイル名。下罫線だけを持つ空段落に当てる。
_HR_STYLE = "EvorefHr"


def _list_style_xml(name: str, ordered: bool) -> str:
    """10 段分の ``text:list-style`` (箇条書きは •、番号付きは ``1.``)。"""
    levels = []
    for level in range(1, 11):
        if ordered:
            head = (
                f'<text:list-level-style-number text:level="{level}" '
                'style:num-suffix="." style:num-format="1">'
            )
            tail = "</text:list-level-style-number>"
        else:
            head = f'<text:list-level-style-bullet text:level="{level}" text:bullet-char="•">'
            tail = "</text:list-level-style-bullet>"
        levels.append(
            head
            + '<style:list-level-properties '
            'text:list-level-position-and-space-mode="label-alignment">'
            '<style:list-level-label-alignment text:label-followed-by="listtab" '
            f'fo:text-indent="-0.635cm" fo:margin-left="{0.635 * level:.3f}cm"/>'
            "</style:list-level-properties>"
            + tail
        )
    return f'<text:list-style style:name="{name}">' + "".join(levels) + "</text:list-style>"


def _install_inline_styles(doc) -> None:
    """inline Markdown 用の文字スタイルとリストのスタイルを文書へ入れる (odt / odp 共通)。"""
    from odfdo import Element, Style

    doc.insert_style(Style("text", name="EvorefBold", bold=True), automatic=True)
    doc.insert_style(Style("text", name="EvorefItalic", italic=True), automatic=True)
    doc.insert_style(
        Style("text", name="EvorefBoldItalic", bold=True, italic=True), automatic=True,
    )
    doc.insert_style(
        Element.from_tag(
            f'<style:style style:name="{_CODE_CHAR_STYLE}" style:display-name="Source Text" '
            'style:family="text"><style:text-properties fo:font-family="Consolas" '
            'style:font-family-generic="modern" style:font-pitch="fixed"/></style:style>',
        ),
    )
    for ordered, name in _LIST_STYLE_NAMES.items():
        doc.insert_style(Element.from_tag(_list_style_xml(name, ordered)), automatic=True)


def _fill_inline(element, text: str, *, bold: bool = False):
    """``element`` (Paragraph / Header) へ inline Markdown を解析して Span で積む。

    解析しないと ``**`` や `` ` `` が生のまま残る (docx writer と同じ原則、パターンは
    ``markdown_patterns`` が SSOT)。リンクは ``text:a``、コードは ``Source_Text``
    の span (強調の中なら強調の span の中) にする。``bold`` は全体を太字にする
    (表の見出し行)。
    """
    from odfdo import Link, Span

    for url, pieces in iter_inline_groups(text):
        target = element
        if url:
            target = Link(url)
            # xlink:type="simple" は text:a の必須属性 (ODF の RelaxNG)。odfdo は付けない。
            target.set_attribute("xlink:type", "simple")
            element.append(target)
        for piece, kind in pieces:
            emphasis = (bold or kind in BOLD_KINDS, kind in ITALIC_KINDS)
            node = Span(piece, style=_CODE_CHAR_STYLE) if kind in CODE_KINDS else piece
            if any(emphasis):
                outer = Span(style=_EMPHASIS_STYLE_NAMES[emphasis])
                outer.append(node)
                node = outer
            target.append(node)
    return element


def _inline_paragraph(text: str, style: str | None = None, *, bold: bool = False):
    """inline Markdown を解析した Paragraph。"""
    from odfdo import Paragraph

    return _fill_inline(Paragraph(style=style), text, bold=bold)


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


def _build_list_elements(block: ContentBlock) -> list:
    """入れ子の ``List`` / ``ListItem`` を組む (odt / odp 共通)。

    odfdo は ``ListItem`` の中に子 ``List`` を置く形で入れ子を表す
    (f_11 §「odt/odp の入れ子リスト」)。箇条書きと番号付きは ``sibling_runs`` の
    区切りごとに別の ``List`` にし、それぞれにリストのスタイル (``text:list-style``)
    を当てる。スタイルが無いと記号の種類が決まらず、読み手 (pandoc) は番号付きと
    読み、表示側も記号を出さない。
    """
    from odfdo import List, ListItem

    def render(nodes) -> list:
        lists = []
        for run in sibling_runs(nodes):
            lst = List(style=_LIST_STYLE_NAMES[run[0].ordered])
            for node in run:
                item = ListItem(_inline_paragraph(node.text))
                for child in render(node.children):
                    item.append(child)
                lst.append(item)
            lists.append(lst)
        return lists

    return render(build_item_tree(block))


def _table_element(rows: list[list[str]], name: str = "Table", *, inline: bool = False):
    """``rows`` から odfdo の Table を組む。

    ``inline`` (odt の表) なら 1 行目を見出し行 (``table:table-header-rows``、太字) にし、
    inline Markdown やリンクを含むセルを Span 付きの段落で組む。ods は表計算のセル
    なので記号を解析せず、値の型推定だけを行う。
    """
    from odfdo import Cell, HeaderRows, Row, Table

    table = Table(name)
    for index, row_data in enumerate(rows):
        header = inline and index == 0
        row = Row()
        for cell_text in row_data:
            groups = (
                list(iter_inline_groups(cell_text))
                if inline and isinstance(cell_text, str)
                else []
            )
            if header or any(
                url or any(kind != "plain" for _piece, kind in pieces)
                for url, pieces in groups
            ):
                cell = Cell()
                cell.append(_inline_paragraph(str(cell_text), bold=header))
                row.append(cell)
            else:
                row.append(Cell(value=coerce_cell_value(cell_text)))
        if header:
            header_rows = HeaderRows()
            header_rows.append(row)
            table.append(header_rows)
        else:
            table.append(row)
    return table


#: 本文の中身とみなす描画物 (文字の無い段落・見出しは数えない)。
_DRAWN_XPATH = "//draw:image | //draw:rect | //draw:ellipse | //draw:line | //table:table"


def _refuse_empty_body(doc, content: ExportContent, ext: str) -> None:
    """中身を渡したのに本文が空なら ``ExportError("empty_output")`` (f_11 §3.3)。

    存在しない画像だけ・壊れた図形だけの本文は、何も描けないまま空の文書になる。
    docx と同じ規則で、成功として書かない。
    """
    if not (content.blocks or (content.raw_markdown or "").strip()):
        return
    body = doc.body
    if body.get_elements(_DRAWN_XPATH):
        return
    if any(
        (e.text_recursive or "").strip()
        for e in body.get_elements("//text:h | //text:p")
    ):
        return
    raise ExportError(
        "empty_output",
        f"Nothing in the content can be drawn in {ext} "
        f"(block types: {sorted({b.type for b in content.blocks})})",
    )


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
    hr_style = Style("paragraph", name=_HR_STYLE)
    hr_style.set_properties(area="paragraph", **{"fo:border-bottom": "0.5pt solid #808080"})
    doc.insert_style(hr_style, automatic=True)
    _install_inline_styles(doc)
    body = doc.body
    body.clear()
    base_dir = export_base_dir(content)

    for block in _resolve_blocks(content):
        if block.type == "heading":
            body.append(
                _fill_inline(Header(max(1, min(block.level, 6))), block.content),
            )

        elif block.type == "paragraph":
            body.append(_inline_paragraph(block.content))

        elif block.type == "code":
            for line in (block.content or "").split("\n"):
                body.append(Paragraph(line, style=_CODE_STYLE))

        elif block.type == "table":
            if block.rows:
                body.append(_table_element(block.rows, inline=True))

        elif block.type == "list":
            for element in _build_list_elements(block):
                body.append(element)

        elif block.type == "quote":
            body.append(_inline_paragraph(f'"{block.content}"'))

        elif block.type == "hr":
            body.append(Paragraph(style=_HR_STYLE))

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

    _refuse_empty_body(doc, content, ".odt")
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
    (:func:`_build_list_elements`) を段落列に混ぜて返す — ``Frame.text_frame``
    は ``Paragraph`` と ``List`` が混在した列をそのまま受け付ける。
    """
    from odfdo import Paragraph

    paragraphs = []
    for block in slide.blocks:
        if block.type == "list":
            if block.has_nested_items():
                paragraphs.extend(_build_list_elements(block))
            else:
                paragraphs.extend(_inline_paragraph(f"• {item}") for item in block.items)
        elif block.type == "heading":
            # level<=2 はスライド分割で消費済み。残るのは小見出し。
            paragraphs.append(_inline_paragraph(block.content))
        elif block.type == "hr":
            paragraphs.append(Paragraph("─" * 24))
        elif block.type == "table":
            # ODP のスライドに表要素を置く公開 API が odfdo に無いため、
            # タブ区切りの段落へ落とす。以前は table 分岐が無く、表が
            # 丸ごと消えていた。
            paragraphs.extend(
                _inline_paragraph("\t".join(str(c) for c in row), bold=index == 0)
                for index, row in enumerate(block.rows)
            )
        elif block.type in ("paragraph", "quote"):
            paragraphs.append(_inline_paragraph(block.content))
        elif block.type == "code":
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
    _install_inline_styles(doc)
    body = doc.body
    body.clear()

    deck = split_into_slides(content)
    base_dir = export_base_dir(content)
    pages: list[tuple[str, Slide | None]] = []
    if deck.cover_title:
        pages.append((deck.cover_title, None))
    pages.extend((s.title, s) for s in deck.slides)

    # draw:master-page-name は draw:page の必須属性で、draw:id は xml:id と組でしか
    # 書けない (ODF 1.2〜1.4 の RelaxNG)。odfdo の既定はどちらも外れるので明示する。
    master = next(iter(doc.get_styles(family="master-page")), None)
    master_name = master.name if master is not None else None
    for index, (title, slide) in enumerate(pages, 1):
        page = DrawPage(None, name=title or f"Slide {index}", master_page=master_name)
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

    _refuse_empty_body(doc, content, ".odp")
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
