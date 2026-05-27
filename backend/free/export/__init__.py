"""Free 版ファイル書出し — Writer 一括登録

Free 版で対応する 5 形式（plaintext, html, csv_tsv, json_yaml, latex）を
グローバルレジストリに登録する。

使用方法::

    import backend.free.export  # 登録が実行される
"""

from backend.export import get_writer_registry
from backend.free.export.writers.csv_tsv import CsvTsvWriter
from backend.free.export.writers.html import HtmlWriter
from backend.free.export.writers.json_yaml import JsonYamlWriter
from backend.free.export.writers.latex import LatexWriter
from backend.free.export.writers.plaintext import PlaintextWriter

_registry = get_writer_registry()
_registry.register(PlaintextWriter())
_registry.register(HtmlWriter())
_registry.register(CsvTsvWriter())
_registry.register(JsonYamlWriter())
_registry.register(LatexWriter())
