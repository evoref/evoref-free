"""エクスポート writer 共通の Markdown インライン正規表現。

HTML / LaTeX / plaintext / docx の各 writer で文字単位一致していたインライン記法の
正規表現定数を集約する。**変換ロジックは各 writer 固有** (出力フォーマットが
異なる) なので定数のみを共有する。

``backend/export/`` は Free/Pro 双方から参照可能な中立レイヤー。

強調の境界規則 (2026-09-16、docs/f_11_file_export.md §3.2):

``_`` / ``__`` は **語中では強調にしない** (CommonMark 準拠)。以前は
``_(.+?)_`` だったため ``snake_case_name`` の中間アンダースコアを区切りと
誤認し、区切り文字が落ちて ``snakecasename`` になっていた。Windows パス
(``E:\\tmp\\office_test\\with_image.docx``) も同じ経路で
``E:\\tmp\\officetest\\withimage.docx`` に壊れる。識別子とパスが黙って
書き換わるので実害が大きい。

``*`` は CommonMark では語中でも強調になるため語境界は課さないが、
``2 * 3 * 4`` を拾わないよう区切りの内側に空白が来ることを禁じる。
"""

from __future__ import annotations

import re

#: 太字の本体パターン。``**...**`` は語境界なし、``__...__`` は語境界あり。
_BOLD_SRC = (
    r"\*\*(?![\s*])(.+?)(?<![\s*])\*\*"
    r"|(?<![\w\\])__(?![\s_])(.+?)(?<![\s_])__(?!\w)"
)

#: 斜体の本体パターン。``*...*`` は語境界なし、``_..._`` は語境界あり。
_ITALIC_SRC = (
    r"\*(?![\s*])(.+?)(?<![\s*])\*"
    r"|(?<![\w\\])_(?![\s_])(.+?)(?<![\s_])_(?!\w)"
)

#: インラインコードの本体パターン。
_INLINE_CODE_SRC = r"`(.+?)`"

#: 太字 ``**...**`` / ``__...__`` (group1 または group2 が本文)
RE_BOLD = re.compile(_BOLD_SRC)

#: 斜体 ``*...*`` / ``_..._`` (group1 または group2 が本文)
RE_ITALIC = re.compile(_ITALIC_SRC)

#: インラインコード `` `...` `` (group1 が本文)
RE_INLINE_CODE = re.compile(_INLINE_CODE_SRC)

#: 太字 / 斜体 / コードを 1 パスで拾う結合パターン。
#: 走査順に依存するので **太字の選択肢を斜体より前に置く** (``**`` が ``*`` に
#: 先に食われないため)。グループ番号: 1,2=太字 / 3,4=斜体 / 5=コード。
RE_INLINE = re.compile(
    f"(?:{_BOLD_SRC})|(?:{_ITALIC_SRC})|(?:{_INLINE_CODE_SRC})",
)

#: ``RE_INLINE`` のグループ番号 (手書きの数え間違いを避けるための名前)。
INLINE_BOLD_GROUPS = (1, 2)
INLINE_ITALIC_GROUPS = (3, 4)
INLINE_CODE_GROUP = 5

#: リンク ``[text](url)`` (group1=text, group2=url)。
#: URL を捨てる plaintext 系は独自パターンを保持する。
RE_LINK = re.compile(r"\[(.+?)\]\((.+?)\)")
