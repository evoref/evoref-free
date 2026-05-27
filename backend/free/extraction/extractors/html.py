"""HTML Extractor

.html, .htm ファイルからタグを除去してテキストを抽出する。
beautifulsoup4 を使用。
"""

from __future__ import annotations

from typing import override

from backend.extraction._text_source_base import TextSourceExtractorBase
from backend.extraction.base import ExtractionError


class HtmlExtractor(TextSourceExtractorBase):
    """HTML ファイルからテキストを抽出（タグ除去）"""

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
        """HTML タグを除去したテキストを返す"""
        bs4 = self._import_bs4()
        return self._parse_html(bs4, text)

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

    @staticmethod
    def _parse_html(bs4, html_text: str) -> str:
        """HTML からテキストを抽出"""
        soup = bs4.BeautifulSoup(html_text, "html.parser")

        # script, style タグを除去
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # テキストを抽出（改行で区切り）
        text = soup.get_text(separator="\n")

        # 連続する空行を1行に圧縮
        lines = [line.strip() for line in text.splitlines()]
        result_lines: list[str] = []
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    result_lines.append("")
                prev_empty = True
            else:
                result_lines.append(line)
                prev_empty = False

        return "\n".join(result_lines).strip()
