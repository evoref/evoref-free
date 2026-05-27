"""Free 版テキスト抽出 — Extractor 一括登録

Free 版で対応する 4 形式（plaintext, html, pdf, csv_tsv）を
グローバルレジストリに登録する。

使用方法::

    import backend.free.extraction  # 登録が実行される
"""

from backend.extraction import get_registry
from backend.free.extraction.extractors.csv_tsv import CsvTsvExtractor
from backend.free.extraction.extractors.html import HtmlExtractor
from backend.free.extraction.extractors.pdf import PdfExtractor
from backend.free.extraction.extractors.plaintext import PlaintextExtractor

_registry = get_registry()
_registry.register(PlaintextExtractor())
_registry.register(HtmlExtractor())
_registry.register(PdfExtractor())
_registry.register(CsvTsvExtractor())
