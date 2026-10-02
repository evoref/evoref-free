"""PPTX Extractor

.pptx ファイルからスライドのテキストを抽出する。
python-pptx を使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from backend.extraction._binary_source_base import BinarySourceExtractorBase
from backend.extraction.base import ExtractionError
from backend.free.extraction.extractors._document_parts import (
    IMAGE_MARK,
    SHAPE_MARK,
    gfm_table_lines,
)


class PptxExtractor(BinarySourceExtractorBase):
    """PPTX ファイルからテキストを抽出"""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".pptx"})

    @property
    def error_code(self) -> str:
        return "pptx_error"

    @property
    def requires(self) -> list[str]:
        return ["python-pptx"]

    def is_available(self) -> bool:
        try:
            from pptx import Presentation  # noqa: F401
            return True
        except ImportError:
            return False

    def _extract_from_source(
        self,
        source: Path | BinaryIO,
        source_name: str,  # noqa: ARG002
    ) -> tuple[str, dict[str, Any]]:
        Presentation = self._import_pptx()
        # python-pptx は str / file-like を受け付ける。
        # 既存挙動 (str(path)) を維持するため Path のときのみ str に変換する
        arg = str(source) if isinstance(source, Path) else source
        prs = Presentation(arg)
        text, slide_count = self._extract_slides(prs)
        return text, {"slides": slide_count}

    @staticmethod
    def _import_pptx():
        try:
            from pptx import Presentation
            return Presentation
        except ImportError:
            raise ExtractionError(
                "missing_library",
                "python-pptx is required for PPTX extraction: pip install python-pptx",
            )

    @staticmethod
    def _shape_lines(shape) -> list[str]:
        """1 つの図形を行にする。文字枠の段落・表 (GFM)・画像と図形の印。

        画像と文字の無い図形は印 (``[image]`` / ``[shape]``) で在ることだけを示す。
        表と印が無いと、表だけ・図形だけのスライドが empty_content になり、
        追記が既存ファイルを読めずに止まる (f_11 §3.3 / §5.1)。空の
        プレースホルダ (未入力の本文枠) は印を付けない。
        """
        if shape.has_text_frame:
            lines = [p.text for p in shape.text_frame.paragraphs if p.text.strip()]
            if lines or shape.is_placeholder:
                return lines
        if getattr(shape, "has_table", False):
            rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
            table_lines = gfm_table_lines(rows)
            return ["", *table_lines, ""] if table_lines else []
        if shape.is_placeholder:
            return []
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            return [IMAGE_MARK]
        if shape.shape_type in (MSO_SHAPE_TYPE.AUTO_SHAPE, MSO_SHAPE_TYPE.LINE):
            return [SHAPE_MARK]
        return []

    @classmethod
    def _extract_slides(cls, prs) -> tuple[str, int]:
        """全スライドからテキストを抽出"""
        texts: list[str] = []
        for i, slide in enumerate(prs.slides, 1):
            slide_texts: list[str] = []
            for shape in slide.shapes:
                slide_texts.extend(cls._shape_lines(shape))
            body = "\n".join(slide_texts).strip("\n")
            if body:
                texts.append(f"[Slide {i}]\n{body}")
        return "\n\n".join(texts), len(prs.slides)
