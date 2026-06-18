"""Free 版ファイル書出し — Writer 一括登録

Free 版で対応する全 9 形式をグローバルレジストリに登録する。

- 基本: plaintext, html, csv_tsv, json_yaml, latex
- Office: docx, xlsx, pptx
- ODF: odf (odt/ods/odp)

使用方法::

    import backend.free.export  # 登録が実行される
"""

from backend.export import get_writer_registry
from backend.free.export.writers.csv_tsv import CsvTsvWriter
from backend.free.export.writers.docx import DocxWriter
from backend.free.export.writers.html import HtmlWriter
from backend.free.export.writers.json_yaml import JsonYamlWriter
from backend.free.export.writers.latex import LatexWriter
from backend.free.export.writers.odf import OdfWriter
from backend.free.export.writers.plaintext import PlaintextWriter
from backend.free.export.writers.pptx import PptxWriter
from backend.free.export.writers.xlsx import XlsxWriter

_registry = get_writer_registry()
_registry.register(PlaintextWriter())
_registry.register(HtmlWriter())
_registry.register(CsvTsvWriter())
_registry.register(JsonYamlWriter())
_registry.register(LatexWriter())
_registry.register(DocxWriter())
_registry.register(XlsxWriter())
_registry.register(PptxWriter())
_registry.register(OdfWriter())
