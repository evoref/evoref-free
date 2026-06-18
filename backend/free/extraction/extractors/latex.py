"""LaTeX Extractor

.tex ファイルからテキストを抽出する。
正規表現ベース（外部依存なし）。
"""

from __future__ import annotations

import re

from backend.extraction._text_source_base import TextSourceExtractorBase

# LaTeX コマンドの除去パターン
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # コメント行を除去
    (re.compile(r"(?m)^%.*$"), ""),
    # インラインコメントを除去（エスケープされた \% は保持）
    (re.compile(r"(?<!\\)%.*$", re.MULTILINE), ""),
    # プリアンブル（\documentclass から \begin{document} まで）を除去
    (re.compile(r"\\documentclass.*?\\begin\{document\}", re.DOTALL), ""),
    # \end{document} 以降を除去
    (re.compile(r"\\end\{document\}.*", re.DOTALL), ""),
    # 環境の開始・終了タグを除去（内容は保持）
    (re.compile(r"\\begin\{[^}]+\}"), ""),
    (re.compile(r"\\end\{[^}]+\}"), ""),
    # セクションコマンドを見出しテキストに変換
    (re.compile(r"\\(?:chapter|section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}"), r"\1"),
    # テキスト装飾コマンドを内容に変換
    (re.compile(r"\\(?:textbf|textit|texttt|emph|underline|textrm|textsf)\{([^}]*)\}"), r"\1"),
    # \cite, \ref, \label 等を除去
    (re.compile(r"\\(?:cite|ref|label|eqref|pageref|autoref)\{[^}]*\}"), ""),
    # \usepackage 等の宣言コマンドを除去
    (re.compile(r"\\(?:usepackage|input|include|bibliography|bibliographystyle)(?:\[[^\]]*\])?\{[^}]*\}"), ""),
    # 残りの単純コマンドを除去（引数なし）
    (re.compile(r"\\(?:maketitle|tableofcontents|newpage|clearpage|noindent|bigskip|medskip|smallskip|\\)"), ""),
    # 中括弧を除去
    (re.compile(r"[{}]"), ""),
    # エスケープされた特殊文字を元に戻す
    (re.compile(r"\\([%$&#_])"), r"\1"),
    # 連続する空行を1行に
    (re.compile(r"\n{3,}"), "\n\n"),
]


class LatexExtractor(TextSourceExtractorBase):
    """LaTeX ファイルからテキストを抽出"""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".tex"})

    def _process(self, text: str, source_name: str) -> str:
        """LaTeX コマンドを除去してプレーンテキストを返す"""
        return self._strip_latex(text)

    @staticmethod
    def _strip_latex(text: str) -> str:
        """LaTeX コマンドを除去してプレーンテキストを返す"""
        result = text
        for pattern, replacement in _PATTERNS:
            result = pattern.sub(replacement, result)
        return result.strip()
