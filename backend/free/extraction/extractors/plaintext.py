"""プレーンテキスト Extractor

.txt, .md, .json, .yaml, .yml, .ini, .conf, .toml, .env,
および各種コードファイルのテキスト抽出を行う。
マルチエンコーディング検出は TextSourceExtractorBase に委譲する。
"""

from __future__ import annotations

from typing import override

from backend.extraction._text_source_base import TextSourceExtractorBase

# 対応する拡張子（テキストデータ + コードファイル）
_EXTENSIONS = frozenset({
    # テキスト・データ
    ".txt", ".md", ".json", ".yaml", ".yml", ".csv",
    ".ini", ".conf", ".toml", ".env",
    # プログラミング言語
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rb", ".sh", ".bash",
    ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rs",
    ".swift", ".kt", ".kts", ".r", ".m",
    ".pl", ".cgi",
    # Web
    ".css", ".scss", ".less", ".sass",
    ".svelte", ".vue",
    ".php", ".asp", ".aspx", ".jsp", ".twig", ".ejs", ".erb", ".cfm",
    # 設定・構成
    ".xml", ".svg", ".htaccess",
    ".gitignore", ".dockerfile",
    ".bat", ".ps1", ".cmd",
    # データ交換
    ".sql", ".graphql", ".proto",
    # その他
    ".log", ".diff", ".patch",
})


class PlaintextExtractor(TextSourceExtractorBase):
    """プレーンテキスト / コードファイル Extractor"""

    @property
    @override
    def extensions(self) -> frozenset[str]:
        return _EXTENSIONS

    @override
    def _process(self, text: str, source_name: str) -> str:
        """生テキストをそのまま返す"""
        return text
