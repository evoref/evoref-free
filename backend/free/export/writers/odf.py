"""ODF Writer

.odt, .ods, .odp: OpenDocument 形式で書出し
odfpy を使用。1 Writer で 3 拡張子を処理（拡張子で分岐）。
"""

from __future__ import annotations

import io

from backend.export._writer_base import BytesWriterBase
from backend.export.base import ContentBlock, ExportContent, ExportError


def _build_odt(content: ExportContent) -> bytes:
    """ExportContent を ODT バイトデータに変換"""
    from odf.opendocument import OpenDocumentText
    from odf import table as odf_table
    from odf.style import Style, TextProperties
    from odf.text import H, P, List, ListItem

    doc = OpenDocumentText()

    # コードブロック用スタイル
    code_style = Style(name="CodeBlock", family="paragraph")
    code_style.addElement(TextProperties(fontname="Consolas", fontsize="9pt"))
    doc.automaticstyles.addElement(code_style)

    blocks = content.blocks
    if not blocks and content.raw_markdown:
        from backend.export.content_converter import ContentConverter
        blocks = ContentConverter().convert(content.raw_markdown)

    for block in blocks:
        if block.type == "heading":
            h = H(outlinelevel=block.level, text=block.content)
            doc.text.addElement(h)

        elif block.type == "paragraph":
            p = P(text=block.content)
            doc.text.addElement(p)

        elif block.type == "code":
            p = P(stylename=code_style, text=block.content)
            doc.text.addElement(p)

        elif block.type == "table":
            if block.rows:
                t = odf_table.Table(name="Table")
                for row_data in block.rows:
                    tr = odf_table.TableRow()
                    for cell_text in row_data:
                        tc = odf_table.TableCell()
                        tc.addElement(P(text=cell_text))
                        tr.addElement(tc)
                    t.addElement(tr)
                doc.text.addElement(t)

        elif block.type == "list":
            lst = List()
            for item in block.items:
                li = ListItem()
                li.addElement(P(text=item))
                lst.addElement(li)
            doc.text.addElement(lst)

        elif block.type == "quote":
            p = P(text=block.content)
            doc.text.addElement(p)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _build_ods(content: ExportContent) -> bytes:
    """ExportContent を ODS バイトデータに変換"""
    from odf.opendocument import OpenDocumentSpreadsheet
    from odf import table as odf_table
    from odf.text import P

    doc = OpenDocumentSpreadsheet()

    # テーブルデータを抽出
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
                tables.append(("Sheet1", [list(map(str, r)) for r in content.raw_data]))
    else:
        for block in content.blocks:
            if block.type == "table" and block.rows:
                tables.append(("Sheet1", block.rows))
                break

    if not tables:
        raise ExportError("no_table_data", "No table data for ODS export")

    for sheet_name, rows in tables:
        t = odf_table.Table(name=sheet_name)
        for row_data in rows:
            tr = odf_table.TableRow()
            for cell_text in row_data:
                tc = odf_table.TableCell(valuetype="string")
                tc.addElement(P(text=str(cell_text)))
                tr.addElement(tc)
            t.addElement(tr)
        doc.spreadsheet.addElement(t)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _build_odp(content: ExportContent) -> bytes:
    """ExportContent を ODP バイトデータに変換"""
    from odf.opendocument import OpenDocumentPresentation
    from odf import draw
    from odf.style import MasterPage, PageLayout, PageLayoutProperties
    from odf.text import P

    doc = OpenDocumentPresentation()

    # ページレイアウト
    pl = PageLayout(name="MyLayout")
    pl.addElement(PageLayoutProperties(
        margin="0cm", pagewidth="25.4cm", pageheight="19.05cm",
        printorientation="landscape",
    ))
    doc.automaticstyles.addElement(pl)

    mp = MasterPage(name="MyMaster", pagelayoutname=pl)
    doc.masterstyles.addElement(mp)

    blocks = content.blocks
    if not blocks and content.raw_markdown:
        from backend.export.content_converter import ContentConverter
        blocks = ContentConverter().convert(content.raw_markdown)

    # スライド分割
    slides_data: list[tuple[str, list[ContentBlock]]] = []
    current_title = content.title or ""
    current_blocks: list[ContentBlock] = []

    for block in blocks:
        if block.type == "heading" and block.level <= 2:
            if current_title or current_blocks:
                slides_data.append((current_title, current_blocks))
            current_title = block.content
            current_blocks = []
        else:
            current_blocks.append(block)
    if current_title or current_blocks:
        slides_data.append((current_title, current_blocks))

    if not slides_data:
        slides_data = [(content.title or "Untitled", [])]

    for title, slide_blocks in slides_data:
        page = draw.Page(stylename=None, masterpagename=mp)
        doc.presentation.addElement(page)

        # タイトルフレーム
        title_frame = draw.Frame(
            stylename=None, width="23cm", height="3cm", x="1cm", y="0.5cm",
        )
        tb = draw.TextBox()
        tb.addElement(P(text=title))
        title_frame.addElement(tb)
        page.addElement(title_frame)

        # コンテンツフレーム
        if slide_blocks:
            content_frame = draw.Frame(
                stylename=None, width="23cm", height="13cm", x="1cm", y="4cm",
            )
            cb = draw.TextBox()
            for block in slide_blocks:
                if block.type in ("paragraph", "quote", "code"):
                    cb.addElement(P(text=block.content))
                elif block.type == "list":
                    for item in block.items:
                        cb.addElement(P(text=f"• {item}"))
            content_frame.addElement(cb)
            page.addElement(content_frame)

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
        return ["odfpy"]

    def is_available(self) -> bool:
        try:
            from odf.opendocument import OpenDocumentText  # noqa: F401
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
