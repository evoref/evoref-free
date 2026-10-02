"""DOCX Writer

.docx: ContentBlock → Word 文書（見出し・段落・表・コードブロック）
python-docx を使用。
"""

from __future__ import annotations

import io
from pathlib import Path

from backend.export._writer_base import BytesWriterBase
from backend.export.base import ContentBlock, ExportContent, ExportError
from backend.export.media import (
    export_base_dir,
    resolve_image_path,
    scaled_width_cm,
)
from backend.export.markdown_patterns import (
    BOLD_KINDS,
    CODE_KINDS,
    ITALIC_KINDS,
    iter_inline_groups,
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
    """リストの段落に多段の番号定義 (``w:numPr`` の ``numId`` + ``ilvl``) を当てる。

    ``List Bullet`` / ``List Bullet 2`` … の組込みスタイルは段ごとに**別の**番号定義
    (1 段だけの abstractNum) を持つ。スタイルを当てるだけでは字下げで入れ子に
    見えるが、構造としては段ごとに別のリストで、読み手 (pandoc など) には平らな
    リストが並んで見える。そこで箇条書き / 番号付きのそれぞれについて、各段の
    スタイルの番号定義 (``w:lvl``) を ``ilvl`` 0-2 に写した多段の abstractNum を
    1 つずつ作り、段落から ``ilvl`` = 段で参照する。見た目 (記号・字下げ) は
    スタイルの定義をそのまま写すので変わらない。

    番号付きはリストごと (段ごと・``sibling_runs`` と同じ区切り) に ``w:num`` を
    作って ``startOverride=1`` を付け、1 から数え直させる (同じ番号定義を共有すると
    文書内の全ての番号付きリストが通し番号になる)。
    """

    def __init__(self, doc) -> None:
        self._doc = doc
        #: ordered → 多段の abstractNumId (作れなければ ``None``)
        self._abstract: dict[bool, int | None] = {}
        self._bullet_num: int | None = None
        #: 段 → いま続いている番号付きの numId
        self._active: dict[int, int] = {}

    def begin_block(self) -> None:
        """list ブロックの境界。前のリストの番号を引き継がない。"""
        self._active.clear()

    def apply(self, para, level: int, ordered: bool) -> None:
        """``para`` に段 ``level`` の番号定義を当てる。作れなければスタイルのまま描く。

        深い段の番号は親が進めば終わる。同じ段に箇条書きが挟まれば、その後の
        番号付きは別のリスト (``sibling_runs`` と同じ区切り)。
        """
        for deeper in [lv for lv in self._active if lv > level]:
            del self._active[deeper]
        if not ordered:
            self._active.pop(level, None)
            if self._bullet_num is None:
                self._bullet_num = self._new_num(False, None)
            num_id = self._bullet_num
        else:
            num_id = self._active.get(level) or self._new_num(True, level)
            if num_id is not None:
                self._active[level] = num_id
        if num_id is None:
            return
        num_pr = para._p.get_or_add_pPr().get_or_add_numPr()
        num_pr.get_or_add_ilvl().val = level
        num_pr.get_or_add_numId().val = num_id

    def _new_num(self, ordered: bool, restart_level: int | None) -> int | None:
        if ordered not in self._abstract:
            self._abstract[ordered] = self._build_abstract(ordered)
        abstract_id = self._abstract[ordered]
        if abstract_id is None:
            return None
        num = self._doc.part.numbering_part.element.add_num(abstract_id)
        if restart_level is not None:
            num.add_lvlOverride(restart_level).add_startOverride(1)
        return num.numId

    def _style_lvl(self, style_name: str):
        """``style_name`` が参照する番号定義の ``w:lvl`` (無ければ ``None``)。"""
        from docx.oxml.ns import qn

        if style_name not in {s.name for s in self._doc.styles}:
            return None
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
        numbering = self._doc.part.numbering_part.element
        abstract_id = str(numbering.num_having_numId(source[0]).abstractNumId.val)
        for abstract in numbering.findall(qn("w:abstractNum")):
            if abstract.get(qn("w:abstractNumId")) == abstract_id:
                for lvl in abstract.findall(qn("w:lvl")):
                    if lvl.get(qn("w:ilvl")) == str(source[1]):
                        return lvl
        return None

    def _build_abstract(self, ordered: bool) -> int | None:
        """段 0-2 のスタイルの ``w:lvl`` を写した多段の abstractNum を作る。

        下位の段のスタイルが無い (継承元に ``List Bullet 2`` が無い) ときは 0 段目の
        定義を写す。字下げは描画側の ``left_indent`` (直接書式) が上書きする。
        """
        import copy
        import re

        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        try:
            numbering = self._doc.part.numbering_part.element
            base = self._style_lvl(_list_style_name(0, ordered))
            if base is None:
                return None
            lvls = []
            for level in range(len(_LIST_LEVEL_SUFFIX)):
                src = self._style_lvl(_list_style_name(level, ordered)) if level else base
                lvl = copy.deepcopy(src if src is not None else base)
                lvl.set(qn("w:ilvl"), str(level))
                # 段落スタイルとの結び付けは元の定義に残す (2 つの定義に結ぶと Word が迷う)
                for p_style in lvl.findall(qn("w:pStyle")):
                    lvl.remove(p_style)
                text = lvl.find(qn("w:lvlText"))
                if text is not None and text.get(qn("w:val")):
                    text.set(qn("w:val"), re.sub(r"%\d", f"%{level + 1}", text.get(qn("w:val"))))
                lvls.append(lvl)
        except (KeyError, NotImplementedError):
            # 継承元 (f_11 §9.1) の番号定義が欠けている。スタイルの番号のまま描く。
            return None
        used = [int(a.get(qn("w:abstractNumId"))) for a in numbering.findall(qn("w:abstractNum"))]
        abstract_id = max(used, default=-1) + 1
        abstract = OxmlElement("w:abstractNum")
        abstract.set(qn("w:abstractNumId"), str(abstract_id))
        multi = OxmlElement("w:multiLevelType")
        multi.set(qn("w:val"), "multilevel")
        abstract.append(multi)
        abstract.extend(lvls)
        # abstractNum は num より前に置く (CT_Numbering の sequence)
        first_num = numbering.find(qn("w:num"))
        if first_num is not None:
            first_num.addprevious(abstract)
        else:
            numbering.append(abstract)
        return abstract_id


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


def _docx_body_has_content(doc) -> bool:
    """本文に文字・表・画像のいずれかがあるか。"""
    from docx.oxml.ns import qn

    body = doc.element.body
    if any((t.text or "").strip() for t in body.iter(qn("w:t"))):
        return True
    return any(True for _ in body.iter(qn("w:tbl"), qn("w:drawing")))


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


#: inline のコードとリンクに当てる文字スタイル。``Verbatim Char`` は pandoc の docx
#: リーダがコードと読む名前 (継承元に同名があればそれを使う)。
_CODE_CHAR_STYLE = "Verbatim Char"
_LINK_CHAR_STYLE = "Hyperlink"


def _ensure_inline_char_styles(doc) -> None:
    """コードとリンクの文字スタイルが無ければ作る。"""
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import RGBColor

    names = {s.name for s in doc.styles}
    if _CODE_CHAR_STYLE not in names:
        doc.styles.add_style(_CODE_CHAR_STYLE, WD_STYLE_TYPE.CHARACTER).font.name = "Consolas"
    if _LINK_CHAR_STYLE not in names:
        font = doc.styles.add_style(_LINK_CHAR_STYLE, WD_STYLE_TYPE.CHARACTER).font
        font.color.rgb = RGBColor(0x05, 0x63, 0xC1)
        font.underline = True


def _add_hyperlink(paragraph, url: str):
    """``paragraph`` の末尾に外部リンク (``w:hyperlink`` + 関係) を足して返す。"""
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    link = OxmlElement("w:hyperlink")
    r_id = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    link.set(qn("r:id"), r_id)
    paragraph._p.append(link)
    return link


def _add_inline_runs(paragraph, text: str) -> None:
    """inline Markdown を解析して Word の Run に変換

    パターンは ``backend.export.markdown_patterns`` が SSOT。以前はここに私有の
    結合正規表現を持っており、``markdown_patterns`` 側と**同じ欠陥を二重に**
    抱えていた (語中のアンダースコアを斜体と誤認して区切り文字を落とす)。
    リンクは ``w:hyperlink`` に、コードは文字スタイル ``Verbatim Char`` にする
    (文字スタイルは ``_ensure_inline_char_styles`` が先に用意する)。
    """
    for url, pieces in iter_inline_groups(text):
        link = _add_hyperlink(paragraph, url) if url else None
        for piece, kind in pieces:
            run = paragraph.add_run(piece)
            if kind in BOLD_KINDS:
                run.bold = True
            if kind in ITALIC_KINDS:
                run.italic = True
            if kind in CODE_KINDS:
                run.style = _CODE_CHAR_STYLE
                run.font.name = "Consolas"
            elif link is not None:
                run.style = _LINK_CHAR_STYLE
            if link is not None:
                link.append(run._r)


def _add_horizontal_rule(paragraph) -> None:
    """段落の下罫線で水平線を描く (ページ幅に追従し、文字数に依存しない)。"""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for key, value in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", "808080")):
        bottom.set(qn(f"w:{key}"), value)
    p_bdr.append(bottom)
    paragraph._p.get_or_add_pPr().append(p_bdr)


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

    _ensure_inline_char_styles(doc)
    available_style_names = {s.name for s in doc.styles}
    numbering = _ListNumbering(doc)

    for block in blocks:
        if block.type == "heading":
            _add_inline_runs(doc.add_heading("", level=block.level), block.content)

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
                        _add_inline_runs(table.rows[i].cells[j].paragraphs[0], cell_text)
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
                numbering.apply(para, level, ordered)
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
            _add_horizontal_rule(doc.add_paragraph())

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

    if blocks and not _docx_body_has_content(doc):
        # 中身を渡したのに本文が空の文書を成功として書かない (f_11 §3.3)。
        # 図形だけの本文 (docx は図形を描かない) が段落 0 の文書になり、次の
        # 追記が既存ファイルを読めずに止まった (2026-10-02 ライブ監査 D07#2/#3)。
        raise ExportError(
            "empty_output",
            f"Nothing in the content can be drawn in .docx "
            f"(block types: {sorted({b.type for b in blocks})})",
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
