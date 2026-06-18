"""ファイル書出しフレームワーク

統一的なファイル書出しインターフェースを提供する。
各エディション（Free/Pro）の __init__.py で Writer を登録し、
get_writer_registry() で取得したレジストリ経由で書出しを実行する。

使用例::

    from backend.export import get_writer_registry
    from backend.export.base import ExportContent
    from backend.export.content_converter import ContentConverter

    registry = get_writer_registry()
    content = ContentConverter.from_markdown("# Title\\n\\nHello world")
    result = registry.write(content, Path("output.html"))
"""

from backend.export.registry import WriterRegistry

_registry: WriterRegistry | None = None


def get_writer_registry() -> WriterRegistry:
    """グローバルレジストリを取得（遅延初期化）"""
    global _registry
    if _registry is None:
        _registry = WriterRegistry()
    return _registry


def reset_writer_registry() -> None:
    """レジストリをリセット（テスト用）

    警告: リセット後は Writer が未登録状態になる。
    テスト内でのみ使用し、teardown で _reinitialize_writer_registry() を呼ぶこと。
    """
    global _registry
    _registry = None


def _reinitialize_writer_registry() -> None:
    """レジストリを再初期化して Writer を再登録する（テスト用）

    reset_writer_registry() 後にグローバルレジストリを復元する。
    """
    global _registry
    _registry = WriterRegistry()

    # 全 Writer の再登録（Free 同梱: 基本5 + Office/ODF）
    try:
        from backend.free.export.writers.plaintext import PlaintextWriter
        from backend.free.export.writers.html import HtmlWriter
        from backend.free.export.writers.csv_tsv import CsvTsvWriter
        from backend.free.export.writers.json_yaml import JsonYamlWriter
        from backend.free.export.writers.latex import LatexWriter
        from backend.free.export.writers.docx import DocxWriter
        from backend.free.export.writers.xlsx import XlsxWriter
        from backend.free.export.writers.pptx import PptxWriter
        from backend.free.export.writers.odf import OdfWriter
        _registry.register(PlaintextWriter())
        _registry.register(HtmlWriter())
        _registry.register(CsvTsvWriter())
        _registry.register(JsonYamlWriter())
        _registry.register(LatexWriter())
        _registry.register(DocxWriter())
        _registry.register(XlsxWriter())
        _registry.register(PptxWriter())
        _registry.register(OdfWriter())
    except ImportError:
        pass
