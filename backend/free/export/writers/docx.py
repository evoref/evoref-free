"""DOCX Writer

.docx: ContentBlock → Word 文書（見出し・段落・表・コードブロック）
python-docx を使用。
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
from backend.export.markdown_patterns import (
    INLINE_BOLD_GROUPS,
    INLINE_CODE_GROUP,
    INLINE_ITALIC_GROUPS,
    RE_INLINE,
)
from backend.log_config import get_logger

logger = get_logger("export.writers.docx")

#: 体裁の継承 (f_11 §9.1) で参照するスタイル名。python-docx はスタイル名を
#: そのままキーに使う (``doc.add_heading`` / ``doc.add_paragraph(style=...)`` /
#: ``doc.add_table(style=...)``) ので、継承元に無い名前を渡すと ``KeyError``。
#: 書く前に検査し、1 つでも欠けていれば継承せず白紙で書く。
_HEADING_STYLE_NAMES = tuple(f"Heading {n}" for n in range(1, 7))

#: 入れ子リストの段 (0-2) ごとのスタイル名の接尾辞。
#: 2/3 段目は "List Bullet 2/3" 系。継承元に無ければ 1 段目のスタイル +
#: left_indent で縮退する (下位スタイルは「必須スタイル」に含めない —
#: 含めると、そのスタイルを持たない継承元が丸ごと不適用になる)。
_LIST_LEVEL_SUFFIX = ("", " 2", " 3")
#: 縮退時に段ごとに足す左インデント (cm)。
_LIST_FALLBACK_INDENT_CM = 0.63


def _list_style_name(level: int, ordered: bool) -> str:
    """段とordered から docx の組込みスタイル名を返す (段は 0-2 に丸める)。"""
    suffix = _LIST_LEVEL_SUFFIX[max(0, min(level, 2))]
    base = "List Number" if ordered else "List Bullet"
    return f"{base}{suffix}"


class _ListNumbering:
    """番号付きリストごとに番号定義 (``w:num``) を作り、1 から数え直させる。

    ``List Number`` 系の組込みスタイルは段ごとに 1 つの番号定義を共有するので、
    スタイルを当てるだけだと Word 上では文書内の**全ての番号付きリストが
    通し番号**になる (2 つ目のリストが 4 から始まる。別の親の下の
    ``List Number 2`` も続きから数える)。同じ abstractNum を指す ``w:num`` を
    リストごとに作り、``startOverride=1`` を付けて段落から直接参照する。
    見た目 (字下げ・番号の書式) は同じ abstractNum なので変わらない。
    """

    def __init__(self, doc) -> None:
        self._doc = doc
        #: 段 → いま続いている番号付きの (numId, ilvl)
        self._active: dict[int, tuple[int, int]] = {}

    def begin_block(self) -> None:
        """list ブロックの境界。前のリストの番号を引き継がない。"""
        self._active.clear()

    def apply(self, para, style_name: str, level: int, ordered: bool) -> None:
        """``para`` が番号付きなら、その段でいま続いているリストの番号定義を当てる。

        深い段の番号は親が進めば終わる。同じ段に箇条書きが挟まれば、その後の
        番号付きは別のリスト (``sibling_runs`` と同じ区切り)。
        """
        for deeper in [lv for lv in self._active if lv > level]:
            del self._active[deeper]
        if not ordered:
            self._active.pop(level, None)
            return
        ref = self._active.get(level) or self._new_num(style_name)
        if ref is None:
            return
        self._active[level] = ref
        num_pr = para._p.get_or_add_pPr().get_or_add_numPr()
        num_pr.get_or_add_numId().val = ref[0]
        num_pr.get_or_add_ilvl().val = ref[1]

    def _new_num(self, style_name: str) -> tuple[int, int] | None:
        """``style_name`` の番号定義を複製して 1 から始める。番号を持たなければ ``None``。"""
        style = self._doc.styles[style_name]
        source: tuple[int, int] | None = None
        while style is not None and source is None:
            p_pr = style.element.pPr
            num_pr = p_pr.numPr if p_pr is not None else None
            if num_pr is not None and num_pr.numId is not None:
                source = (num_pr.numId.val, num_pr.ilvl.val if num_pr.ilvl is not None else 0)
            style = style.base_style
        if source is None:
            return None
        try:
            numbering = self._doc.part.numbering_part.element
            abstract_id = numbering.num_having_numId(source[0]).abstractNumId.val
        except (KeyError, NotImplementedError):
            # 継承元 (f_11 §9.1) の番号定義が欠けている。スタイルの番号のまま描く。
            return None
        num = numbering.add_num(abstract_id)
        num.add_lvlOverride(source[1]).add_startOverride(1)
        return num.numId, source[1]


def _resolve_docx_template(content: ExportContent) -> Path | None:
    """``metadata["template_base"]`` を検査して継承元パスを返す (無ければ ``None``)。

    出力の拡張子と ``base`` の拡張子が違えば継承しない (f_11 §9.1)。この
    writer は常に ``.docx`` を書くので、``base`` も ``.docx`` の場合だけ使う。
    """
    base = (content.metadata or {}).get("template_base")
    if not base:
        return None
    path = Path(base)
    if path.suffix.lower() != ".docx":
        return None
    if not path.is_file():
        logger.warning("docx template base not found: %s", path)
        return None
    return path


def _required_styles(blocks: list[ContentBlock]) -> set[str]:
    """``blocks`` を書くのに必要なスタイル名 (継承元に無ければ継承を諦める)。

    リストは **0 段目のスタイルだけ**必須にする。``List Bullet 2`` 等の
    下位スタイルを必須に含めると、そのスタイルを持たない継承元が丸ごと
    不適用になる (f_11 §9.1) — 下位スタイルが無ければ描画側で
    left_indent へ縮退するので、事前検査の対象外でよい。
    """
    needed: set[str] = set()
    for block in blocks:
        if block.type == "heading":
            level = min(max(int(block.level or 1), 1), 6)
            needed.add(_HEADING_STYLE_NAMES[level - 1])
        elif block.type == "list":
            for _text, level, ordered in block.iter_items():
                if level == 0:
                    needed.add("List Number" if ordered else "List Bullet")
        elif block.type == "table" and block.rows:
            needed.add("Table Grid")
    return needed


def _clear_docx_body(doc) -> None:
    """``body`` 直下の ``sectPr`` 以外を除く (f_11 §9.1)。

    ``sectPr`` (用紙 / 余白 / セクション設定) を残すことで、ヘッダ / フッタ
    (ロゴ) の定義はそのまま残る (ヘッダ / フッタは別パートで、``sectPr`` が
    参照を持つ)。
    """
    from docx.oxml.ns import qn

    body = doc.element.body
    sect_pr_tag = qn("w:sectPr")
    for child in list(body):
        if child.tag == sect_pr_tag:
            continue
        body.remove(child)


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

    base_dir = export_base_dir(content)
    blocks = content.blocks
    if not blocks and content.raw_markdown:
        from backend.export.content_converter import ContentConverter
        blocks = ContentConverter().convert(content.raw_markdown)

    doc = None
    template_path = _resolve_docx_template(content)
    if template_path is not None:
        try:
            candidate = Document(str(template_path))
        except Exception as e:  # noqa: BLE001 - 壊れた base は白紙へ縮退する
            logger.warning(
                "docx template base %s could not be opened; writing without "
                "inheritance: %s", template_path, e,
            )
            candidate = None
        if candidate is not None:
            missing = sorted(_required_styles(blocks) - {s.name for s in candidate.styles})
            if missing:
                logger.warning(
                    "docx template base %s is missing required style(s) %s; "
                    "writing without inheritance", template_path, missing,
                )
                content.metadata["template_applied"] = False
            else:
                _clear_docx_body(candidate)
                doc = candidate
                content.metadata["template_applied"] = True
        else:
            content.metadata["template_applied"] = False

    if doc is None:
        doc = Document()
        # デフォルトフォント設定 (継承時は base のスタイルをそのまま使う)
        style = doc.styles["Normal"]
        font = style.font
        font.name = "Arial"
        font.size = Pt(11)

    available_style_names = {s.name for s in doc.styles}
    numbering = _ListNumbering(doc)

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
            numbering.begin_block()
            for text, level, ordered in block.iter_items():
                style_name = _list_style_name(level, ordered)
                if style_name not in available_style_names:
                    # 下位スタイル (List Bullet 2/3 等) が継承元に無い縮退
                    # (f_11 §9.1)。1 段目のスタイル + left_indent で描く。
                    style_name = _list_style_name(0, ordered)
                para = doc.add_paragraph(style=style_name)
                numbering.apply(para, style_name, level, ordered)
                if style_name != _list_style_name(level, ordered):
                    para.paragraph_format.left_indent = Cm(
                        _LIST_FALLBACK_INDENT_CM * (level + 1),
                    )
                _add_inline_runs(para, text)

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
