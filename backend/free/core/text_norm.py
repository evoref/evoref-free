"""空白の畳み込みと省略記号付きの切り詰め (純粋関数)。

空白は ``str.split()`` の定義 (Unicode の空白) で扱う。``re`` の ``\\s+`` と同じ集合。
NFKC などの正規化は含まない (永続キーに使う正規化は呼出側が自分で持つ)。
"""

from __future__ import annotations


def collapse_ws(text: str | None) -> str:
    """連続する空白を半角スペース 1 個にし、前後の空白を落とす。"""
    return " ".join((text or "").split())


def strip_ws(text: str | None) -> str:
    """空白を全て落とす (比較用の形)。"""
    return "".join((text or "").split())


def ellipsize(text: str, limit: int) -> str:
    """``limit`` 文字を超えるとき末尾を ``…`` にして ``limit`` 文字へ切り詰める。"""
    return text if len(text) <= limit else text[: limit - 1] + "…"
