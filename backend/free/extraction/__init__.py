"""Free 版テキスト抽出 — Extractor 一括登録

Free 版で対応する全 11 形式をグローバルレジストリに登録する。

- 基本: plaintext, html, pdf, csv_tsv
- Office: docx, xlsx, pptx
- ワープロ/組版: rtf, latex
- ODF: odf (odt/ods/odp)
- メール: email_eml (eml)

使用方法::

    import backend.free.extraction  # 登録が実行される
"""

from backend.extraction import get_registry
from backend.free.extraction.extractors.csv_tsv import CsvTsvExtractor
from backend.free.extraction.extractors.docx import DocxExtractor
from backend.free.extraction.extractors.email_eml import EmailEmlExtractor
from backend.free.extraction.extractors.html import HtmlExtractor
from backend.free.extraction.extractors.latex import LatexExtractor
from backend.free.extraction.extractors.odf import OdfExtractor
from backend.free.extraction.extractors.pdf import PdfExtractor
from backend.free.extraction.extractors.plaintext import PlaintextExtractor
from backend.free.extraction.extractors.pptx import PptxExtractor
from backend.free.extraction.extractors.rtf import RtfExtractor
from backend.free.extraction.extractors.xlsx import XlsxExtractor

_registry = get_registry()
_registry.register(PlaintextExtractor())
_registry.register(HtmlExtractor())
_registry.register(PdfExtractor())
_registry.register(CsvTsvExtractor())
_registry.register(DocxExtractor())
_registry.register(XlsxExtractor())
_registry.register(PptxExtractor())
_registry.register(RtfExtractor())
_registry.register(EmailEmlExtractor())
_registry.register(OdfExtractor())
_registry.register(LatexExtractor())
