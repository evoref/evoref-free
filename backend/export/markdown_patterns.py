"""エクスポート writer 共通の Markdown インライン正規表現。

HTML / LaTeX / plaintext (Free) と docx (Pro) の各 writer で文字単位一致して
いたインライン記法の正規表現定数を集約する。**変換ロジックは各 writer 固有**
(出力フォーマットが異なる) なので定数のみを共有する。

``backend/export/`` は Free/Pro 双方から参照可能な中立レイヤー。
"""

from __future__ import annotations

import re

#: 太字 ``**...**`` / ``__...__`` (group1 または group2 が本文)
RE_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")

#: 斜体 ``*...*`` / ``_..._`` (group1 または group2 が本文)
RE_ITALIC = re.compile(r"\*(.+?)\*|_(.+?)_")

#: インラインコード `` `...` `` (group1 が本文)
RE_INLINE_CODE = re.compile(r"`(.+?)`")

#: リンク ``[text](url)`` (group1=text, group2=url)。
#: URL を捨てる plaintext 系は独自パターンを保持する。
RE_LINK = re.compile(r"\[(.+?)\]\((.+?)\)")
