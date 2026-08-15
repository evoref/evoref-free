"""エディタタブ名 (ファイル名) の決定論的導出ヘルパ

クリエイトモードで生成したコード/仕様書を Pro エディタへタブ表示する際、
タブ名 (= ファイル名) が日本語にならないよう、**ASCII snake_case の stem
(拡張子なし)** を決定論的に導出する。long_form 経路
(`api/chat/chat_streaming.py`) と meta_cognitive 経路
(`agent/meta_cognitive.py`) の双方がこのヘルパを共用する。

設計:
- ユーザー指示文の先頭にある ASCII 語をタブ名の材料に使い、拾えなければ
  言語別のフォールバック stem に倒す。**常に非空・ASCII** を保証する。
- 拡張子は付与しない (言語に応じた拡張子は呼出側が付ける)。
"""

from __future__ import annotations

import re

# ASCII slug 化: 英数字 + `_` + `-` 以外を `_` 化 (SPLIT モードの
# `_slug_for_split_file` と同方針)。
_ASCII_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_STEM_LEN = 32

#: ヒントから拾う ASCII 語 (3 文字以上)。日本語指示に混じる英語の固有名詞
#: (``FastAPI`` / ``tetris`` 等) を拾ってタブ名を意味のあるものにする。
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_MAX_HINT_WORDS = 3

#: ヒント語として採らない汎用語 (拾っても情報量がないもの)。
_HINT_STOPWORDS = frozenset({
    "and", "for", "the", "with", "into", "from", "please", "file", "code",
    "python", "typescript", "javascript", "markdown", "html", "css", "json",
    "yaml", "xml", "sql", "bash",
})

# 言語別の決定論的フォールバック stem。
_FALLBACK_STEM_BY_LANGUAGE: dict[str, str] = {
    "markdown": "document",
    "python": "script",
    "typescript": "module",
    "javascript": "script",
    "html": "index",
    "css": "styles",
    "json": "data",
    "yaml": "config",
    "xml": "data",
    "sql": "query",
    "bash": "script",
}
_DEFAULT_FALLBACK_STEM = "output"


def _fallback_stem(language: str) -> str:
    """言語別の決定論的フォールバック stem を返す。"""
    return _FALLBACK_STEM_BY_LANGUAGE.get(
        (language or "").lower(), _DEFAULT_FALLBACK_STEM,
    )


def _to_ascii_slug(raw: str) -> str:
    """任意文字列を ASCII snake_case stem に正規化する (空なら ``""``)。"""
    slug = _ASCII_SAFE_RE.sub("_", raw or "").strip("_-")
    slug = slug[:_MAX_STEM_LEN].rstrip("_-")
    # 数字始まりは識別子として扱いづらいため接頭辞を付ける。
    if slug and slug[0].isdigit():
        slug = f"file_{slug}"
    return slug


def _stem_from_hint(hint: str) -> str:
    """ユーザー指示文の ASCII 語から stem を組み立てる (拾えなければ ``""``)。"""
    words = [
        w.lower() for w in _ASCII_WORD_RE.findall(hint or "")
        if w.lower() not in _HINT_STOPWORDS
    ]
    if not words:
        return ""
    return _to_ascii_slug("_".join(words[:_MAX_HINT_WORDS]))


def derive_editor_filename_stem(*, hint: str, language: str) -> str:
    """エディタタブ用の ASCII snake_case stem (拡張子なし) を導出する。

    Args:
        hint: ユーザー指示文。ASCII 語が含まれていればタブ名の材料に使う。
        language: 言語識別子 (``python`` / ``markdown`` 等)。フォールバック
            stem の選択に使う。

    Returns:
        ASCII snake_case の stem。常に非空。拡張子は含まない。
    """
    return _stem_from_hint(hint) or _fallback_stem(language)
