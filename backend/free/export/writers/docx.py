"""DOCX Writer

.docx: ContentBlock → Word 文書（見出し・段落・表・コードブロック）
python-docx を使用。
"""

from __future__ import annotations

import io

from backend.export._writer_base import BytesWriterBase
from backend.export.base import ExportContent
from backend.export.media import (
    export_base_dir,
    resolve_image_path,
    scaled_width_cm,
)
from backend.export.markdown_patterns import (
    INLINE_BOLD_GROUPS,
    INLINE_CODE_GROUP,
    INLINE_ITALIC_GROUPS,
    RE_INLINE,
)
from backend.log_config import get_logger

logger = get_logger("export.writers.docx")


def _add_inline_runs(paragraph, text: str) -> None:
    """inline Markdown を解析して Word の Run に変換

    パターンは ``backend.export.markdown_patterns`` が SSOT。以前はここに私有の
    結合正規表現を持っており、``markdown_patterns`` 側と**同じ欠陥を二重に**
    抱えていた (語中のアンダースコアを斜体と誤認して区切り文字を落とす)。
    """
    last_end = 0
    for m in RE_INLINE.finditer(text):
        # マッチ前の通常テキスト
        if m.start() > last_end:
            paragraph.add_run(text[last_end:m.start()])

        bold = m.group(INLINE_BOLD_GROUPS[0]) or m.group(INLINE_BOLD_GROUPS[1])
        italic = m.group(INLINE_ITALIC_GROUPS[0]) or m.group(INLINE_ITALIC_GROUPS[1])
        code = m.group(INLINE_CODE_GROUP)
        if bold is not None:
            run = paragraph.add_run(bold)
            run.bold = True
        elif italic is not None:
            run = paragraph.add_run(italic)
            run.italic = True
        elif code is not None:
            run = paragraph.add_run(code)
            run.font.name = "Consolas"
        last_end = m.end()

    # 残りのテキスト
    if last_end < len(text):
        paragraph.add_run(text[last_end:])


def _build_docx(content: ExportContent) -> bytes:
    """ExportContent を DOCX バイトデータに変換"""
    from docx import Document
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()

    # デフォルトフォント設定
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(11)

    base_dir = export_base_dir(content)
    blocks = content.blocks
    if not blocks and content.raw_markdown:
        from backend.export.content_converter import ContentConverter
        blocks = ContentConverter().convert(content.raw_markdown)

    for block in blocks:
        if block.type == "heading":
            doc.add_heading(block.content, level=block.level)

        elif block.type == "paragraph":
            para = doc.add_paragraph()
            _add_inline_runs(para, block.content)

        elif block.type == "code":
            para = doc.add_paragraph()
            run = para.add_run(block.content)
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
            # グレー背景のシェーディングは python-docx で直接サポートされないため省略

        elif block.type == "table":
            if block.rows:
                table = doc.add_table(
                    rows=len(block.rows),
                    cols=len(block.rows[0]),
                    style="Table Grid",
                )
                for i, row_data in enumerate(block.rows):
                    for j, cell_text in enumerate(row_data):
                        table.rows[i].cells[j].text = cell_text
                    # ヘッダー行を太字に
                    if i == 0:
                        for cell in table.rows[0].cells:
                            for para in cell.paragraphs:
                                for run in para.runs:
                                    run.bold = True

        elif block.type == "list":
            for idx, item in enumerate(block.items):
                style_name = "List Number" if block.ordered else "List Bullet"
                para = doc.add_paragraph(style=style_name)
                _add_inline_runs(para, item)

        elif block.type == "quote":
            para = doc.add_paragraph(block.content)
            para.style = doc.styles["Quote"] if "Quote" in [s.name for s in doc.styles] else None
            if para.style is None:
                para.paragraph_format.left_indent = Pt(36)

        elif block.type == "hr":
            doc.add_paragraph("_" * 50)

        elif block.type == "image":
            path = resolve_image_path(block.src, base_dir)
            if path is not None:
                doc.add_picture(str(path), width=Cm(scaled_width_cm(path)))

        elif block.type == "shapes":
            # .docx に図形は描かない (f_11 §3.1)。黙って落とさず記録する。
            logger.warning(
                "shapes blocks are not drawn in .docx; %d shape(s) skipped",
                len(block.shapes),
            )

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class DocxWriter(BytesWriterBase):
    """DOCX ファイル Writer"""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".docx"})

    @property
    def requires(self) -> list[str]:
        return ["python-docx"]

    def _render_bytes(self, content: ExportContent, ext: str) -> bytes:  # noqa: ARG002
        return _build_docx(content)

    def is_available(self) -> bool:
        try:
            from docx import Document  # noqa: F401
            return True
        except ImportError:
            return False
