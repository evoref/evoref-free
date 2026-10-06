"""``calculate`` の許可関数のうち、時間長・文字数・日付の演算 (純粋関数)。

小型モデルは文字数の数え上げ・時間長の繰り上がり・日付の前後比較を暗算で誤る。
ここに置く関数は ``calc._SAFE_NAMES`` に載り、AST 許可リスト評価の中で呼ばれる
(新しいツールは増やさない、docs/f_03 §3.2)。引数は文字列か数のリテラルで、
解釈できない入力は **もっともらしい値を返さず** ``ValueError`` / ``TypeError`` を
送出する (``calculate`` が ``Error: ...`` の結果に変える)。

文字の数え方 (``chars`` / ``count_kind``):

- Python の ``str`` はコードポイント列なので、UTF-16 のサロゲート対は存在しない
  (😀 は ``len`` でも 1)。``len`` はコードポイント数、``chars`` は **見た目の 1 文字**
  (書記素クラスタの近似) を数える。
- 近似の規則: NFC に正規化したうえで、結合文字 (Mn / Mc / Me)・異体字セレクタ・
  絵文字の肌色修飾・ZWJ・タグ文字は直前の文字に含める。
  地域指示子 (国旗) は 2 つで 1 文字。CR LF は 1 文字。半角の濁点 ``ﾞ`` ``ﾟ`` も直前に
  含める (``ｶﾞ`` は 1 文字)。ZWJ の直後は絵文字のときだけ繋げる。孤立したサロゲート
  (``"\\ud83d"`` のようなエスケープ) は 1 文字と数える。
- 空白・改行も 1 文字と数える (除くなら ``count_kind(s, "nonspace")``)。
"""

from __future__ import annotations

import datetime
import math
import re
import unicodedata

from backend.free.core.response_dates import ISO_DATE_RE, JP_DATE_RE, WEEKDAY_SUFFIX_RE
from backend.free.core.script_ranges import (
    HALFWIDTH_KATAKANA,
    HIRAGANA,
    KANJI,
    KANJI_COMPAT,
    KANJI_EXT_A,
    KANJI_MARKS,
    KATAKANA_WORD,
)

# ── 文字数・語数 ──

_ZWJ = "\u200d"


def _is_extender(ch: str) -> bool:
    """直前の文字に含める (それ自体は 1 文字に数えない) コードポイントか。"""
    cp = ord(ch)
    return (
        unicodedata.category(ch) in ("Mn", "Mc", "Me")
        or 0xFE00 <= cp <= 0xFE0F            # 異体字セレクタ
        or 0xE0100 <= cp <= 0xE01EF          # 異体字セレクタ補助
        or 0x1F3FB <= cp <= 0x1F3FF          # 絵文字の肌色修飾
        or 0xE0020 <= cp <= 0xE007F          # タグ文字 (地域の旗)
        or cp in (0xFF9E, 0xFF9F)            # 半角の濁点・半濁点 (Grapheme_Extend)
        or ch == _ZWJ
    )


def _is_regional_indicator(ch: str) -> bool:
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


#: ZWJ の後で直前の文字に繋げる絵文字のブロック (Extended_Pictographic の近似)。
_PICTOGRAPHIC_BLOCKS = (
    (0x2190, 0x21FF), (0x2300, 0x23FF), (0x2460, 0x27BF), (0x2900, 0x297F),
    (0x2B00, 0x2BFF), (0x1F000, 0x1FAFF),
)


def _is_pictographic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _PICTOGRAPHIC_BLOCKS)


def _graphemes(text: str) -> list[str]:
    """書記素クラスタの近似に分割する (規則はモジュールの docstring)。

    クラスタは文字のリストで組んで最後に連結する (1 つのクラスタに結合文字が
    長く続いても文字列の連結を繰り返さない)。
    """
    clusters: list[list[str]] = []
    for ch in unicodedata.normalize("NFC", text):
        if clusters:
            last = clusters[-1]
            if (
                _is_extender(ch)
                or (last[-1] == _ZWJ and _is_pictographic(ch))
                or (last == ["\r"] and ch == "\n")
                or (
                    _is_regional_indicator(ch) and len(last) == 1
                    and _is_regional_indicator(last[0])
                )
            ):
                last.append(ch)
                continue
        clusters.append([ch])
    return ["".join(c) for c in clusters]


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name}() expects a string literal, got {type(value).__name__}")
    return value


def chars(text: object) -> int:
    """見た目の文字数 (空白・改行を含む)。"""
    return len(_graphemes(_require_str("chars", text)))


def words(text: object) -> int:
    """空白区切りの語数。文字・数字を 1 つも含まない区切り (``-`` ``—``) は数えない。

    日本語のように語を空白で区切らない文は語数にならない (``chars`` で数える)。
    """
    return sum(
        1 for token in _require_str("words", text).split()
        if any(c.isalnum() for c in token)
    )


#: 漢数字のゼロ ``〇`` は漢字、踊り字 ``ゝゞ`` / ``ヽヾ`` はかな・カナに数える。
_KANJI_RE = re.compile(f"[{KANJI}{KANJI_EXT_A}{KANJI_COMPAT}{KANJI_MARKS}〇]")
_HIRAGANA_RE = re.compile(f"[{HIRAGANA}ゝゞ]")
_KATAKANA_RE = re.compile(f"[{KATAKANA_WORD}{HALFWIDTH_KATAKANA}ヽヾ]")

#: ``count_kind`` の種別。``nonspace`` 以外は互いに排他で、全種別の和が ``chars``。
CHAR_KINDS = (
    "kanji", "hiragana", "katakana", "latin", "digit", "space", "punct", "other",
)


def _char_kind(cluster: str) -> str:
    """書記素クラスタの種別 (先頭のコードポイントで決める)。"""
    base = cluster[0]
    if base.isspace():
        return "space"
    if base.isdecimal():
        return "digit"
    if base.isalpha() and unicodedata.name(base, "").startswith(
        ("LATIN ", "FULLWIDTH LATIN "),
    ):
        return "latin"
    if _HIRAGANA_RE.match(base):
        return "hiragana"
    if _KATAKANA_RE.match(base):
        return "katakana"
    if _KANJI_RE.match(base):
        return "kanji"
    if unicodedata.category(base).startswith("P"):
        return "punct"
    return "other"


def count_kind(text: object, kind: object) -> int:
    """文字種ごとの文字数 (``chars`` と同じ数え方)。

    種別は :data:`CHAR_KINDS` と ``nonspace`` (空白・改行以外の全て)。``latin`` は
    アクセント付き・全角の英字を含む。``digit`` は全角数字を含み、漢数字は ``kanji``。``katakana`` は長音記号 ``ー`` と半角カナを含む。
    """
    clusters = _graphemes(_require_str("count_kind", text))
    if kind == "nonspace":
        return sum(1 for c in clusters if not c[0].isspace())
    if kind not in CHAR_KINDS:
        raise ValueError(
            f"unknown character kind: {kind!r} (use one of "
            + ", ".join((*CHAR_KINDS, "nonspace")) + ")"
        )
    return sum(1 for c in clusters if _char_kind(c) == kind)


# ── 時間長 ──

#: 単位 → 秒。長い綴りを先に照合する (``時間`` を ``時`` より、``min`` を ``m`` より先)。
_UNIT_SECONDS: dict[str, int] = {
    "weeks": 604800, "week": 604800, "週間": 604800, "週": 604800, "w": 604800,
    "days": 86400, "day": 86400, "日": 86400, "d": 86400,
    "hours": 3600, "hour": 3600, "hrs": 3600, "hr": 3600, "時間": 3600, "時": 3600,
    "h": 3600,
    "minutes": 60, "minute": 60, "mins": 60, "min": 60, "分": 60, "m": 60,
    "seconds": 1, "second": 1, "secs": 1, "sec": 1, "秒": 1, "s": 1,
}
_UNIT_ALT = "|".join(sorted(_UNIT_SECONDS, key=len, reverse=True))
_UNIT_TERM_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNIT_ALT})(半)?\s*")
_COLON_RE = re.compile(r"(\d+):(\d{1,2})(?::(\d{1,2}(?:\.\d+)?))?")
_DURATION_HINT = (
    "use h:mm:ss, e.g. '2:30:45' (3 min 45 s is '0:03:45'), or number + unit, "
    "e.g. '2時間30分' / '1h 30m' / '90分'"
)
#: 時間長・日付の文字列の長さの上限。照合を短い入力に限り、検査の時間を読めるものにする
#: (数え上げの文字列とは別。``chars`` は式の上限まで受ける)。
_MAX_OPERAND_CHARS = 64


def _number(text: str) -> int | float:
    return float(text) if "." in text else int(text)


def _integral(value: float) -> int | float:
    return int(value) if isinstance(value, float) and value.is_integer() else value


def dur(text: object, unit: object = "s") -> int | float:
    """時間長の文字列を ``unit`` (既定は秒) の数にする。

    ``h:mm:ss`` (分・秒は 59 まで) と、数 + 単位の並び (``2時間30分`` / ``1時間半`` /
    ``1h 30m 15s`` / ``90分``) を読む。先頭の ``-`` は負の時間長。``m`` は分 (月ではない)。
    区切りが 1 つの ``3:45`` は時:分 (時刻・作業時間) か分:秒 (曲の長さ・ペース) か
    決まらず、読み違えると 60 倍ずれるので **読まない** (``ValueError``)。
    読めない入力は ``ValueError``。
    """
    raw = _require_str("dur", text)
    if len(raw) > _MAX_OPERAND_CHARS:
        raise ValueError(
            f"invalid duration: too long ({len(raw)} > {_MAX_OPERAND_CHARS} characters)"
        )
    if unit not in _UNIT_SECONDS:
        raise ValueError(
            f"unknown duration unit: {unit!r} (use s / min / h / d / w or 秒 / 分 / 時間 / 日)"
        )
    s = unicodedata.normalize("NFKC", raw).strip().lower().replace("−", "-")
    sign = 1
    if s.startswith("-"):
        sign, s = -1, s[1:].strip()
    seconds: int | float
    colon = _COLON_RE.fullmatch(s)
    if colon and colon.group(3) is None:
        raise ValueError(
            f"ambiguous duration: {raw!r} (h:mm or m:ss?) -- write '3:45:00' for "
            "3 h 45 min or '0:03:45' for 3 min 45 s"
        )
    if colon:
        hours, minutes = int(colon.group(1)), int(colon.group(2))
        secs = _number(colon.group(3))
        if minutes >= 60 or secs >= 60:
            raise ValueError(f"invalid duration: {raw!r} (minutes and seconds must be < 60)")
        seconds = hours * 3600 + minutes * 60 + secs
    else:
        pos = 0
        total: int | float = 0
        # 先頭から隙間なく並ぶ項だけを読む (``finditer`` は不一致の位置ごとに走査し直す)。
        while (m := _UNIT_TERM_RE.match(s, pos)) is not None:
            per = _UNIT_SECONDS[m.group(2)]
            half = (per // 2 if per % 2 == 0 else per / 2) if m.group(3) else 0
            total += _number(m.group(1)) * per + half
            pos = m.end()
        if pos == 0 or pos != len(s):
            raise ValueError(f"invalid duration: {raw!r} ({_DURATION_HINT})")
        seconds = total
    value = sign * _integral(seconds)
    per = _UNIT_SECONDS[unit]  # type: ignore[index]
    if isinstance(value, int) and value % per == 0:
        return value // per
    return _integral(value / per)


def _require_seconds(name: str, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name}() expects a number of seconds, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name}(): the duration is not finite")
    return value


def _split_hms(seconds: int | float) -> tuple[str, int, int, str]:
    """符号・時・分・秒の文字列 (秒は小数第 3 位まで、末尾ゼロは畳む)。"""
    sign = "-" if seconds < 0 else ""
    if isinstance(seconds, float):
        seconds = round(abs(seconds), 3)
        whole = int(seconds)
        frac = f"{seconds - whole:.3f}".rstrip("0").rstrip(".")[1:]
    else:
        whole, frac = abs(seconds), ""
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return sign, hours, minutes, f"{secs:02d}{frac}"


def hms(seconds: object) -> str:
    """秒を ``H:MM:SS`` にする (時は 24 を超えても繰り上げない、負は ``-`` 付き)。"""
    sign, hours, minutes, secs = _split_hms(_require_seconds("hms", seconds))
    return f"{sign}{hours}:{minutes:02d}:{secs}"


def clock(seconds: object) -> str:
    """0 時からの秒を時刻 ``HH:MM:SS`` にする。日をまたいだ分は ``(+N day)`` で添える。

    ``clock(dur("22:30:00") + dur("3:45:00"))`` → ``02:15:00 (+1 day)``。
    """
    value = _require_seconds("clock", seconds)
    days, rest = divmod(value, 86400)
    _, hours, minutes, secs = _split_hms(rest)
    text = f"{hours:02d}:{minutes:02d}:{secs}"
    days = int(days)
    if days:
        text += f" ({days:+d} day{'s' if abs(days) > 1 else ''})"
    return text


# ── 日付 ──

def date_parts(text: str) -> tuple[int, int, int] | None:
    """年まで揃った日付の綴りから年月日を取り出す (暦として正しいかは見ない、純粋関数)。

    ``2024-2-29`` / ``2024/2/29`` / ``2024年2月29日`` と、末尾に添えた曜日
    (``(火)`` / ``火曜``)。綴りの読み取りは ``core.response_dates`` が SSOT
    (ISO・和文の月日・曜日の添え書き)。式の接地 (数値を値で照合する) も使う。
    """
    if len(text) > _MAX_OPERAND_CHARS:
        return None
    text = unicodedata.normalize("NFKC", text).strip()
    m = ISO_DATE_RE.match(text.replace("/", "-")) or JP_DATE_RE.match(text)
    if m is None or m.group("y") is None:
        return None
    if m.end() < len(text):
        suffix = WEEKDAY_SUFFIX_RE.match(text, m.end())
        # 括弧なしの添え書きは ``火曜`` までを読むので、``火曜日`` の ``日`` が残る。
        rest = text[suffix.end():] if suffix is not None else None
        if rest is None or rest not in ("", "日"):
            return None
    return int(m.group("y")), int(m.group("m")), int(m.group("d"))


def _parse_date(name: str, value: object) -> datetime.date:
    """年まで揃った日付を読む (:func:`date_parts`)。年の無い日付・``today`` は読まない
    (計算機は時計を持たない)。
    """
    raw = _require_str(name, value)
    parts = date_parts(raw)
    if parts is None:
        raise ValueError(
            f"invalid date: {raw!r} (use a full date such as '2024-2-29' or "
            "'2024年2月29日')"
        )
    try:
        return datetime.date(*parts)
    except ValueError as e:
        raise ValueError(f"invalid date: {raw!r} ({e})") from None


def date_diff(start: object, end: object) -> int:
    """``end`` − ``start`` の日数 (符号付き)。正なら ``start`` が先、負なら ``end`` が先。"""
    return (_parse_date("date_diff", end) - _parse_date("date_diff", start)).days


def day_of_year(value: object) -> int:
    """その年の何日目か (1 月 1 日が 1、うるう年の 12 月 31 日が 366)。"""
    return _parse_date("day_of_year", value).timetuple().tm_yday


#: ``calc._SAFE_NAMES`` に載せる関数。
CALC_OPS: dict[str, object] = {
    "chars": chars, "words": words, "count_kind": count_kind,
    "dur": dur, "hms": hms, "clock": clock,
    "date_diff": date_diff, "day_of_year": day_of_year,
}

#: 文字列を返す関数 (``calc._may_be_sized`` が繰り返し・連結の長さを検査する)。
STRING_RESULT_OPS = frozenset({"hms", "clock"})

#: 数える対象の文字列を第 1 引数に取る関数 (式の接地検査が引数を会話と突き合わせる)。
TEXT_COUNT_OPS = frozenset({"chars", "words", "count_kind"})
#: 時間長・日付の文字列を引数に取る関数 (式の接地検査が文字列中の数を値で照合する)。
DURATION_OPS = frozenset({"dur"})
DATE_OPS = frozenset({"date_diff", "day_of_year"})
