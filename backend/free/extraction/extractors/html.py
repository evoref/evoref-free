"""HTML Extractor

.html, .htm ファイルから本文を構造付きテキストで抽出する。見出しは ``#`` 行、
表は GFM (``<caption>`` を直前の行に)、画像は ``[image] <alt>``、文書題は先頭の
``#`` 行。実装は ``fetch_url`` と共有 (``backend.free.extraction.html_structure``、
f_01 §3.1.4) だが、利用者自身のファイルなので除くのは script / style / noscript
だけ (ナビ・ヘッダ・フッタ・class 名による除去は掛けない)。beautifulsoup4 を使用。
"""

from __future__ import annotations

from typing import override

from backend.extraction._text_source_base import TextSourceExtractorBase
from backend.extraction.base import ExtractionError
from backend.free.extraction.html_structure import html_to_text


class HtmlExtractor(TextSourceExtractorBase):
    """HTML ファイルから構造付きテキストを抽出"""

    @property
    @override
    def extensions(self) -> frozenset[str]:
        return frozenset({".html", ".htm"})

    @property
    @override
    def requires(self) -> list[str]:
        return ["beautifulsoup4"]

    @override
    def is_available(self) -> bool:
        try:
            import bs4  # noqa: F401
            return True
        except ImportError:
            return False

    @override
    def _process(self, text: str, source_name: str) -> str:
        """HTML を構造付きテキスト (見出し ``#`` / GFM 表 / alt / 文書題) にする"""
        self._import_bs4()
        return html_to_text(text)

    @staticmethod
    def _import_bs4():
        """beautifulsoup4 を遅延インポート"""
        try:
            import bs4
            return bs4
        except ImportError:
            raise ExtractionError(
                "missing_library",
                "beautifulsoup4 is required for HTML extraction: pip install beautifulsoup4",
            )
