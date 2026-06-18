"""テキスト抽出フレームワーク

統一的なテキスト抽出インターフェースを提供する。
各エディション（Free/Pro）の __init__.py で Extractor を登録し、
get_registry() で取得したレジストリ経由で抽出を実行する。

使用例::

    from backend.extraction import get_registry
    registry = get_registry()
    result = registry.extract(Path("document.pdf"))
    print(result.text)
"""

from backend.extraction.registry import ExtractorRegistry

_registry: ExtractorRegistry | None = None


def get_registry() -> ExtractorRegistry:
    """グローバルレジストリを取得（遅延初期化）"""
    global _registry
    if _registry is None:
        _registry = ExtractorRegistry()
    return _registry


def reset_registry() -> None:
    """レジストリをリセット（テスト用）

    警告: リセット後は Extractor が未登録状態になる。
    テスト内でのみ使用し、teardown で _reinitialize_registry() を呼ぶこと。
    """
    global _registry
    _registry = None


def _reinitialize_registry() -> None:
    """レジストリを再初期化して Extractor を再登録する（テスト用）

    reset_registry() 後にグローバルレジストリを復元する。
    """
    global _registry
    _registry = ExtractorRegistry()

    # 全 Extractor の再登録（Free 同梱: 基本4 + Office/ODF/メール/RTF/LaTeX）
    try:
        from backend.free.extraction.extractors.plaintext import PlaintextExtractor
        from backend.free.extraction.extractors.html import HtmlExtractor
        from backend.free.extraction.extractors.pdf import PdfExtractor
        from backend.free.extraction.extractors.csv_tsv import CsvTsvExtractor
        from backend.free.extraction.extractors.docx import DocxExtractor
        from backend.free.extraction.extractors.xlsx import XlsxExtractor
        from backend.free.extraction.extractors.pptx import PptxExtractor
        from backend.free.extraction.extractors.rtf import RtfExtractor
        from backend.free.extraction.extractors.email_eml import EmailEmlExtractor
        from backend.free.extraction.extractors.odf import OdfExtractor
        from backend.free.extraction.extractors.latex import LatexExtractor
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
    except ImportError:
        pass
