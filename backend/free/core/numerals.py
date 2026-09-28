"""数の綴り (算用数字・全角数字・漢数字) を整数へ読む — 読み取りの SSOT (不変則 #14(a))。

序数 (``intent_vocab`` の「十二番目」) と人数 (``text_quality`` の「四人」) が
それぞれ漢数字の表を書き写して読んでいた (2026-09-28 レビュー M3)。読み取りは
:func:`kanji_number_value` の 1 本にし、範囲 (1〜99 など) は呼出側が決める。
"""

from __future__ import annotations

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
_KANJI_DIGIT_VALUES = {c: i for i, c in enumerate("〇一二三四五六七八九")}
_KANJI_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_KANJI_LARGE_UNITS = {"万": 10**4, "億": 10**8, "兆": 10**12}


def kanji_number_value(token: str) -> int | None:
    """数の綴りを整数へ (純粋関数)。読めなければ ``None``。

    - 算用数字・全角数字 (「12」「１２」)
    - 位取りの漢数字 (「二〇二六」「一二三」) は各字を桁として読む。ただし 〇 を含むか
      3 字以上のときだけ — 隣り合う 2 字 (「二三人」「三四人」) は **範囲** の表現で、
      23 / 34 ではない (2026-09-28 再レビュー 3)
    - 単位付きの漢数字 (「十二」「三千五百」「一億二千万」) は万進の節ごとに足し上げる。
      単位の間に数字が 2 つ並ぶ (「二三十」)・節の中で単位が大きくならない順に
      並ばない (「十十」「百百」) ものは読めない
    """
    t = (token or "").translate(_FULLWIDTH_DIGITS)
    if not t:
        return None
    if t.isascii() and t.isdigit():
        return int(t)
    if all(c in _KANJI_DIGIT_VALUES for c in t):
        if len(t) >= 3 or "〇" in t or len(t) == 1:
            return int("".join(str(_KANJI_DIGIT_VALUES[c]) for c in t))
        return None
    total = section = 0
    digit: int | None = None
    last_small = 10**5
    for c in t:
        if c in _KANJI_DIGIT_VALUES:
            if digit is not None:
                return None
            digit = _KANJI_DIGIT_VALUES[c]
        elif c in _KANJI_SMALL_UNITS:
            unit = _KANJI_SMALL_UNITS[c]
            if unit >= last_small:
                return None
            section += (1 if digit is None else digit) * unit
            digit = None
            last_small = unit
        elif c in _KANJI_LARGE_UNITS:
            section += digit or 0
            total += (section or 1) * _KANJI_LARGE_UNITS[c]
            section = 0
            digit = None
            last_small = 10**5
        else:
            return None
    return total + section + (digit or 0)


__all__ = ["kanji_number_value"]
