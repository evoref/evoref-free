"""ファイル名だけで決まる規則 (字句の鍵。判定点ではない — 不変則 #14 の対象外)。

全 pillar から参照するので横断基盤の core に置く (loop の旧 executor も使う)。
"""

from __future__ import annotations

#: 空であるのが慣習のファイル名 (パッケージ印・空ディレクトリ保持・型情報の印)。
#: 中身の無い本文を「生成の失敗」とみなす判定の例外。ファイル名は字句の鍵 (判定点ではない)。
CONVENTIONALLY_EMPTY_NAMES = frozenset({
    "__init__.py", ".gitkeep", ".keep", "py.typed", ".nojekyll",
})


def allows_empty_content(file_path: str) -> bool:
    """空の本文が正当なファイルか (``__init__.py`` 等)。大文字小文字・区切りは問わない。"""
    name = str(file_path or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name in CONVENTIONALLY_EMPTY_NAMES
