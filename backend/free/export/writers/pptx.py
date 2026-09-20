"""PPTX Writer

.pptx: ContentBlock → PowerPoint スライド
python-pptx を使用。設計書: [docs/f_11_file_export.md](../../../../docs/f_11_file_export.md)
"""

from __future__ import annotations

import io
from pathlib import Path

from backend.export._writer_base import BytesWriterBase
from backend.export.base import ContentBlock, ExportContent
from backend.export.media import (
    export_base_dir,
    resolve_image_path,
    scaled_width_cm,
)
from backend.export.shapes import normalize_shapes, rgb_tuple
from backend.export.slide_splitter import Slide, split_into_slides
from backend.log_config import get_logger

logger = get_logger("export.writers.pptx")


def _resolve_pptx_template(content: ExportContent) -> Path | None:
    """``metadata["template_base"]`` を検査して継承元パスを返す (無ければ ``None``)。

    出力の拡張子と ``base`` の拡張子が違えば継承しない (f_11 §9.1)。この
    writer は常に ``.pptx`` を書くので、``base`` も ``.pptx`` の場合だけ使う。
    """
    base = (content.metadata or {}).get("template_base")
    if not base:
        return None
    path = Path(base)
    if path.suffix.lower() != ".pptx":
        return None
    if not path.is_file():
        logger.warning("pptx template base not found: %s", path)
        return None
    return path


def _remove_all_slides(prs) -> None:
    """既存スライドを全て外す (python-pptx に削除の公開 API が無い、f_11 §9.1)。"""
    from pptx.oxml.ns import qn

    xml_slides = prs.slides._sldIdLst
    for sld in list(xml_slides):
        rid = sld.get(qn("r:id"))
        prs.part.drop_rel(rid)
        xml_slides.remove(sld)


def _layout_placeholder_types(layout) -> set:
    types = set()
    for placeholder in layout.placeholders:
        try:
            types.add(placeholder.placeholder_format.type)
        except (AttributeError, ValueError):
            continue
    return types


def _find_cover_layout(prs):
    """title を持つ最初のレイアウト (f_11 §9.1: 名前ではなく placeholder の型で選ぶ)。"""
    from pptx.enum.shapes import PP_PLACEHOLDER

    title_types = {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE}
    for layout in prs.slide_layouts:
        if _layout_placeholder_types(layout) & title_types:
            return layout
    return None


def _find_content_layout(prs):
    """title + body/object を持つ最初のレイアウト。"""
    from pptx.enum.shapes import PP_PLACEHOLDER

    title_types = {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE}
    body_types = {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT}
    for layout in prs.slide_layouts:
        types = _layout_placeholder_types(layout)
        if (types & title_types) and (types & body_types):
            return layout
    return None


class _BodyText:
    """本文プレースホルダへ段落を積む。最初の 1 段落は既存の空段落を使う。"""

    def __init__(self, text_frame) -> None:
        self._tf = text_frame
        self._tf.clear()
        self._used_first = False

    def add(self, text: str, level: int = 0):
        if self._used_first:
            para = self._tf.add_paragraph()
        else:
            para = self._tf.paragraphs[0]
            self._used_first = True
        para.text = text
        para.level = level
        return para


def _add_block_to_body(body: _BodyText, block: ContentBlock) -> None:
    """テキストで表せるブロックを本文へ積む (table / image / shapes は対象外)。"""
    from pptx.util import Pt

    if block.type == "paragraph":
        body.add(block.content)

    elif block.type == "heading":
        # level<=2 はスライド分割で消費済み。ここへ来るのは level>=3 の
        # 小見出し。以前はどの分岐にも当たらず黙って消えていた。
        para = body.add(block.content)
        for run in para.runs:
            run.font.bold = True

    elif block.type == "list":
        for text, level, _ordered in block.iter_items():
            body.add(text, level=min(1 + level, 8))

    elif block.type == "code":
        para = body.add(block.content)
        for run in para.runs:
            run.font.name = "Consolas"
            run.font.size = Pt(10)

    elif block.type == "quote":
        para = body.add(f'"{block.content}"')
        para.font.italic = True

    elif block.type == "hr":
        # 区切り線。以前は落ちていた。
        body.add("─" * 24)


def _add_image(slide, block: ContentBlock, base_dir) -> None:
    """``image`` ブロックをスライドへ貼る (f_11 §4.1)。"""
    from pptx.util import Cm

    path = resolve_image_path(block.src, base_dir)
    if path is None:
        return
    slide.shapes.add_picture(
        str(path), Cm(2.0), Cm(9.0), width=Cm(scaled_width_cm(path)),
    )


def _add_shapes(slide, block: ContentBlock) -> None:
    """``shapes`` ブロックをスライドへ描く (f_11 §4.2)。"""
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.util import Cm, Pt

    auto_shapes = {"rect": MSO_SHAPE.RECTANGLE, "oval": MSO_SHAPE.OVAL}

    for shape in normalize_shapes(block.shapes):
        if shape.kind == "line":
            connector = slide.shapes.add_connector(
                MSO_CONNECTOR.STRAIGHT,
                Cm(shape.x), Cm(shape.y), Cm(shape.x2), Cm(shape.y2),
            )
            connector.line.color.rgb = RGBColor(*rgb_tuple(shape.line))
            connector.line.width = Pt(shape.width_pt)
            continue

        drawn = slide.shapes.add_shape(
            auto_shapes[shape.kind],
            Cm(shape.x), Cm(shape.y), Cm(shape.w), Cm(shape.h),
        )
        if shape.fill is None:
            drawn.fill.background()
        else:
            drawn.fill.solid()
            drawn.fill.fore_color.rgb = RGBColor(*rgb_tuple(shape.fill))
        drawn.line.color.rgb = RGBColor(*rgb_tuple(shape.line))
        drawn.line.width = Pt(shape.width_pt)
        if shape.text:
            drawn.text_frame.text = shape.text


def _render_slide(prs, layout, slide_data: Slide, base_dir=None) -> None:
    """1 枚のスライドを描く。"""
    from pptx.util import Inches

    slide = prs.slides.add_slide(layout)
    if slide.shapes.title is not None:
        slide.shapes.title.text = slide_data.title

    placeholder = slide.placeholders[1] if len(slide.placeholders) > 1 else None
    body = _BodyText(placeholder.text_frame) if placeholder is not None else None

    table_top = Inches(3.5)
    for block in slide_data.blocks:
        if block.type == "table":
            if not block.rows:
                continue
            rows_count = len(block.rows)
            cols_count = len(block.rows[0])
            table = slide.shapes.add_table(
                rows_count, cols_count,
                Inches(0.5), table_top, Inches(9.0), Inches(0.3 * rows_count),
            ).table
            for r_idx, row_data in enumerate(block.rows):
                for c_idx, cell_text in enumerate(row_data):
                    table.cell(r_idx, c_idx).text = cell_text
        elif block.type == "image":
            _add_image(slide, block, base_dir)
        elif block.type == "shapes":
            _add_shapes(slide, block)
        elif body is not None:
            _add_block_to_body(body, block)


def _build_pptx(content: ExportContent) -> bytes:
    """ExportContent を PPTX バイトデータに変換"""
    from pptx import Presentation

    prs = None
    layout_title = layout_content = None
    template_path = _resolve_pptx_template(content)
    if template_path is not None:
        try:
            candidate = Presentation(str(template_path))
        except Exception as e:  # noqa: BLE001 - 壊れた base は白紙へ縮退する
            logger.warning(
                "pptx template base %s could not be opened; writing without "
                "inheritance: %s", template_path, e,
            )
            candidate = None
        if candidate is not None:
            cover = _find_cover_layout(candidate)
            body = _find_content_layout(candidate)
            if cover is None or body is None:
                logger.warning(
                    "pptx template base %s has no usable title/body layout; "
                    "writing without inheritance", template_path,
                )
                content.metadata["template_applied"] = False
            else:
                _remove_all_slides(candidate)
                prs, layout_title, layout_content = candidate, cover, body
                content.metadata["template_applied"] = True
        else:
            content.metadata["template_applied"] = False

    if prs is None:
        prs = Presentation()
        layout_title = prs.slide_layouts[0]    # タイトルスライド
        layout_content = prs.slide_layouts[1]  # タイトル + コンテンツ

    deck = split_into_slides(content)
    base_dir = export_base_dir(content)

    if deck.cover_title:
        cover = prs.slides.add_slide(layout_title)
        if cover.shapes.title is not None:
            cover.shapes.title.text = deck.cover_title

    for slide_data in deck.slides:
        _render_slide(prs, layout_content, slide_data, base_dir)

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


class PptxWriter(BytesWriterBase):
    """PPTX ファイル Writer"""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".pptx"})

    @property
    def requires(self) -> list[str]:
        return ["python-pptx"]

    def _render_bytes(self, content: ExportContent, ext: str) -> bytes:  # noqa: ARG002
        return _build_pptx(content)

    def is_available(self) -> bool:
        try:
            from pptx import Presentation  # noqa: F401
            return True
        except ImportError:
            return False
