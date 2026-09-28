"""ツール実行結果の日付と、応答本文が述べる日付の突合 (2026-09-09 監査 G-06)。

``date_intent`` (`backend.free.agent.tool_judge_commands`) が組んだコマンドは
``target: YYYY-MM-DD (曜日)`` を印字する。ツールが向き (forward/backward) や
除外曜日を正しく踏んでも、モデルが結果を使わず暗算で別の日付を答える経路は
別問題として残る (実インシデント 2026-09-08 T19/3: ツールは 2026-11-02 を
返したが、モデルは暗算で 10/14 と答えた。正しくは向きの修正込みで 10/9)。

``core.text_quality.ignores_calculate_result`` の日付版。calculate と違い
数値の単位換算スケールは無く、代わりに日付表記のゆらぎ (和暦の月日 / ISO /
スラッシュ) を吸収する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from backend.log_config import get_logger

__all__ = [
    "ISO_DATE_RE",
    "JP_DATE_RE",
    "YMD_JA_PATTERN",
    "YMD_NUMERIC_PATTERN",
    "WeekdayClaim",
    "complete_years",
    "extract_tool_target_date",
    "literal_dates",
    "fix_weekday_claims",
    "response_dates",
    "ignores_date_result",
    "mentions_literal_date",
    "nearest_date",
    "weekday_claims",
]

logger = get_logger("core.response_dates")

#: 全角数字 → 半角。応答本文は全角で日付を書くことがある。
_ZENKAKU_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

#: ``date_intent`` の生成コマンドが印字する ``target:`` 行 (``build_date_intent_command``
#: 参照)。``business_days_from`` / ``days_from`` / ``weekday_of`` だけが持つ
#: (``days_between`` は日数を返すのでこの行を持たない)。
_TARGET_LINE_RE = re.compile(r"target:\s*(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})")

#: ISO 形式の日付 (``2026-10-09``)。前後が数字に接していないことを確認し、
#: より長い数値列の部分一致を避ける。
ISO_DATE_RE = re.compile(
    r"(?<!\d)(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?!\d)",
)

#: 和文の月日 (``10月9日`` / ``2026年10月9日`` / ``10 月 15 日``)。年は省略可。
#: 日付の読み取りの SSOT (``core.inference`` の日付注記も使う)。モデルは数字と
#: 単位の間に空白を入れて書く (「10 月 15 日（火）」) ので空白を許す
#: (2026-09-26 監査 C08#2: 空白入りを読めず曜日の照合も長文の日付照合も空振り)。
JP_DATE_RE = re.compile(
    r"(?:(?P<y>\d{4})\s*年\s*)?(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日",
)

#: 月日の直後に添えた曜日 (``（火）`` / ``(火)`` / ``火曜日`` / ``火曜``)。
#: 曜日は 7 字の閉じた集合 (語を足す保守は起きない)。
_WEEKDAY_CHARS = "月火水木金土日"
_WEEKDAY_SUFFIX_RE = re.compile(
    r"\s*(?:[（(]\s*(?P<p>[月火水木金土日])\s*(?:曜日?)?\s*[）)]"
    r"|(?P<w>[月火水木金土日])曜)",
)

#: スラッシュ区切りの月日 (``10/9``)。年は文脈 (``year_hint``) 頼み。区切りや
#: 数字に接する形 (``2026/10/9`` の ``10/9``) は月日として読まない。
_SLASH_DATE_RE = re.compile(r"(?<![\d/-])(?P<m>\d{1,2})/(?P<d>\d{1,2})(?![\d/-])")

#: 年つきの和文日付 / 数字だけの年月日の綴り。他の正規表現へ埋め込む用
#: (``intent_vocab`` の「年つき日付 … 何日間」)。綴りの SSOT はこのモジュール。
YMD_JA_PATTERN = r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
YMD_NUMERIC_PATTERN = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
_YMD_NUMERIC_RE = re.compile(r"(?<!\d)" + YMD_NUMERIC_PATTERN + r"(?!\d)")


def mentions_literal_date(text: str) -> bool:
    """具体日付 (和文の月日 / ``10/15`` / 年月日) を含むか (純粋関数)。"""
    normalized = (text or "").translate(_ZENKAKU_DIGITS)
    return bool(
        JP_DATE_RE.search(normalized) or _SLASH_DATE_RE.search(normalized)
        or _YMD_NUMERIC_RE.search(normalized)
    )


def literal_date_count(text: str) -> int:
    """具体日付 (:func:`mentions_literal_date` と同じ綴り) の個数 (重なりは 1 個、純粋関数)。

    日数の問いの **構造** (「A から B は何日」= 日付 2 つ) を語彙ではなく日付の
    個数で見るのに使う (``core.date_math_cue.asks_day_count``、2026-09-27)。
    """
    return len(literal_dates(text))


def literal_dates(text: str) -> list[tuple[int | None, int, int]]:
    """具体日付を ``(年 | None, 月, 日)`` で出現順に返す (重なりは 1 個、純粋関数)。

    綴りは :func:`mentions_literal_date` と同じ (和文の月日 / 年月日 / ``10/15``)。
    年を書いていない形は年を ``None`` にする (捏造しない)。
    """
    normalized = (text or "").translate(_ZENKAKU_DIGITS)
    found: list[tuple[int, int, tuple[int | None, int, int]]] = []
    for m in _YMD_NUMERIC_RE.finditer(normalized):
        y, mo, d = re.split(r"[-/]", m.group(0))
        found.append((m.start(), m.end(), (int(y), int(mo), int(d))))
    for m in JP_DATE_RE.finditer(normalized):
        year = int(m["y"]) if m["y"] else None
        found.append((m.start(), m.end(), (year, int(m["m"]), int(m["d"]))))
    for m in _SLASH_DATE_RE.finditer(normalized):
        found.append((m.start(), m.end(), (None, int(m["m"]), int(m["d"]))))
    found.sort(key=lambda t: (t[0], t[1]))
    out: list[tuple[int | None, int, int]] = []
    last_end = -1
    for start, end, ymd in found:
        if start < last_end:
            continue
        out.append(ymd)
        last_end = end
    return out


def nearest_date(
    month: int, day: int, anchor: date, *,
    past_days: int | None = None, future_days: int | None = None,
) -> date | None:
    """年の無い月日を、``anchor`` に最も近い巡りの日付にする (純粋関数)。

    「年なしの月日 → 近い年」の 1 実装 (曜日の照合・ツール判定の起点・記憶の
    年の補完が使う)。候補は ``anchor`` の前後 1 年。``past_days`` /
    ``future_days`` を渡すと ``anchor`` から前 N 日〜後 M 日の窓に入る候補だけに
    絞る (窓外なら ``None``)。距離が同じなら早い方。
    """
    best: date | None = None
    for delta_year in (-1, 0, 1):
        candidate = _safe_date(anchor.year + delta_year, month, day)
        if candidate is None:
            continue
        delta = (candidate - anchor).days
        if past_days is not None and delta < -past_days:
            continue
        if future_days is not None and delta > future_days:
            continue
        if best is None or abs(delta) < abs((best - anchor).days):
            best = candidate
    return best


#: 年を足さない文の区切り (:func:`_year_clue_before` の窓はこの後ろから)。
_SENTENCE_ENDS = "。！？!?\n"
#: 半角数字 → 全角 (全角の月日に足す年を揃える)。
_HANKAKU_DIGITS = str.maketrans("0123456789", "０１２３４５６７８９")


def _year_clue_before(normalized: str, start: int) -> bool:
    """月日の前 (同じ文の中、他の日付を除く) に「年」があるか (純粋関数)。

    年の手がかり (「来年度の」「2年後の」「2027年度の」「2027年、」「令和9年」) は
    **語形を列挙せず** 構造で見る — 月日と同じ文の手前に「年」の字があれば、その
    月日の年は西暦に解けない形で指定されている (2026-09-28 独立レビュー P3-3)。
    手前の日付 (「2026年9月27日から10月15日」の前者) の「年」は数えない。
    """
    head = normalized[:start]
    cut = max(head.rfind(ch) for ch in _SENTENCE_ENDS) + 1
    return "年" in JP_DATE_RE.sub("", head[cut:])


def complete_years(text: str, anchor: date | None) -> str:
    """年の無い和文の月日 (``4月25日``) に、``anchor`` に最も近い年を足す (純粋関数)。

    訂正は年を言い直さない (「4月18日ではなく4月25日でした」)。年は置き換える
    旧値の日付 (``anchor``) に最も近い巡りから採る (:func:`nearest_date`)。
    ``anchor`` が無い / 同じ文に年の手がかりがある月日は変えない。全角の月日には
    全角で足す。
    """
    if anchor is None or not text:
        return text
    normalized = text.translate(_ZENKAKU_DIGITS)
    pieces: list[str] = []
    last = 0
    for m in JP_DATE_RE.finditer(normalized):
        year, unresolved = _year_of(normalized, m)
        if year is not None or unresolved:
            continue
        resolved = nearest_date(int(m["m"]), int(m["d"]), anchor)
        if resolved is None:
            continue
        stamp = str(resolved.year)
        if text[m.start("m")] != normalized[m.start("m")]:
            stamp = stamp.translate(_HANKAKU_DIGITS)
        # 月日が数と単位の間に空白を置く書き方 (「4 月 25 日」) なら年も揃える
        month_end = m.end("m")
        gap = normalized[month_end:normalized.index("月", month_end)]
        pieces.append(text[last:m.start()])
        pieces.append(f"{stamp}{gap}年{gap}")
        last = m.start()
    if not pieces:
        return text
    pieces.append(text[last:])
    return "".join(pieces)


def extract_tool_target_date(text: str) -> date | None:
    """コマンド結果ブロックから ``target:`` の日付を取り出す (純粋関数)。

    見つからない / カレンダー上あり得ない日付なら ``None``。
    """
    m = _TARGET_LINE_RE.search(text or "")
    if not m:
        return None
    try:
        return date(int(m["y"]), int(m["m"]), int(m["d"]))
    except ValueError:
        return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def response_dates(text: str, *, year_hint: int | None = None) -> list[date]:
    """応答本文が述べている日付を出現順に取り出す (純粋関数)。

    ISO (``2026-10-09``) / 和文月日 (``10月9日``、``2026年10月9日``) /
    スラッシュ (``10/9``) を拾う。年を書いていない形式 (和文月日の年なし・
    スラッシュ) は ``year_hint`` (通常はツール結果の ``target`` の年) を使う。
    ``year_hint`` が無ければその形式は読み飛ばす (誤った年を捏造しない)。
    """
    normalized = (text or "").translate(_ZENKAKU_DIGITS)
    matches: list[tuple[int, int, date | None]] = []
    for m in ISO_DATE_RE.finditer(normalized):
        matches.append(
            (m.start(), m.end(), _safe_date(int(m["y"]), int(m["m"]), int(m["d"]))),
        )
    for m in JP_DATE_RE.finditer(normalized):
        year = int(m["y"]) if m["y"] else year_hint
        if year is None:
            continue
        matches.append(
            (m.start(), m.end(), _safe_date(year, int(m["m"]), int(m["d"]))),
        )
    for m in _SLASH_DATE_RE.finditer(normalized):
        if year_hint is None:
            continue
        matches.append(
            (m.start(), m.end(), _safe_date(year_hint, int(m["m"]), int(m["d"]))),
        )
    matches.sort(key=lambda t: t[0])
    out: list[date] = []
    last_end = -1
    for start, end, parsed in matches:
        if parsed is None or start < last_end:
            continue
        out.append(parsed)
        last_end = end
    return out


def ignores_date_result(prompt_or_tool_block: str, response: str) -> bool:
    """ツールが接地した日付を回答が使っていないか (純粋関数)。

    ツール結果 (``prompt_or_tool_block``) に ``target:`` があり、回答本文が
    日付を 1 つ以上述べているのに、そのどれもが ``target`` と一致しなければ
    「使っていない」。ツール結果が無い / 回答が日付を述べていないターンは
    判定しない (数と同じく、使ったかどうかを確かめられないため)。
    """
    target = extract_tool_target_date(prompt_or_tool_block)
    if target is None:
        return False
    dates = response_dates(response, year_hint=target.year)
    if not dates:
        return False
    return target not in dates


@dataclass(frozen=True)
class WeekdayClaim:
    """月日に添えた曜日の 1 件 (``10 月 15 日（火）``)。"""

    month: int
    day: int
    #: 本文に書かれた年 (無ければ ``None``)。
    year: int | None
    #: 書かれた曜日 (``月``〜``日`` の 1 字)。
    weekday: str
    #: 曜日の 1 字の本文中の位置 (書き換えに使う)。
    weekday_pos: int
    #: 西暦に解けない年の手がかりが月日の直前にある (「令和7年」「来年の」)。
    year_unresolved: bool = False


def _year_of(normalized: str, m: re.Match[str]) -> tuple[int | None, bool]:
    """月日の一致の年と、「解けない年の手がかり」の有無を返す。

    和暦・相対年を解く SSOT は無いので **解かずに「手がかりあり」とだけ見る**
    (語彙を足さない。docs/f_08 §6.3。手がかりの判定は :func:`_year_clue_before`)。
    """
    if m["y"]:
        return int(m["y"]), False
    return None, _year_clue_before(normalized, m.start())


def weekday_claims(text: str) -> list[WeekdayClaim]:
    """本文から「月日 + 曜日」の組を出現順に取り出す (純粋関数)。"""
    normalized = (text or "").translate(_ZENKAKU_DIGITS)
    out: list[WeekdayClaim] = []
    for m in JP_DATE_RE.finditer(normalized):
        suffix = _WEEKDAY_SUFFIX_RE.match(normalized, m.end())
        if suffix is None:
            continue
        group = "p" if suffix["p"] else "w"
        year, unresolved = _year_of(normalized, m)
        out.append(WeekdayClaim(
            month=int(m["m"]),
            day=int(m["d"]),
            year=year,
            weekday=suffix[group],
            weekday_pos=suffix.start(group),
            year_unresolved=unresolved,
        ))
    return out


def _grounded_years(grounded: str) -> dict[tuple[int, int], set[int | None]]:
    """接地文の月日ごとの年の集合。``None`` は「西暦に解けない年の手がかり」。

    年の手がかりが無い出現は集合に何も足さない (月日のキーだけ作る)。
    """
    normalized = (grounded or "").translate(_ZENKAKU_DIGITS)
    out: dict[tuple[int, int], set[int | None]] = {}
    for m in JP_DATE_RE.finditer(normalized):
        year, unresolved = _year_of(normalized, m)
        years = out.setdefault((int(m["m"]), int(m["d"])), set())
        if year is not None:
            years.add(year)
        elif unresolved:
            years.add(None)
    for m in ISO_DATE_RE.finditer(normalized):
        out.setdefault((int(m["m"]), int(m["d"])), set()).add(int(m["y"]))
    for m in _SLASH_DATE_RE.finditer(normalized):
        out.setdefault((int(m["m"]), int(m["d"])), set())
    return out


def fix_weekday_claims(text: str, *, today: date, grounded: str) -> str:
    """月日に添えた曜日が暦と食い違っていれば、暦の曜日へ書き換える (純粋関数)。

    書き換えるのは **月日が接地文 (ユーザーの発話) にあり、曜日だけをモデルが
    付けた** ときだけ。月日そのものをモデルが作った場合は日付が誤りうるので
    触らず、ユーザー自身がその月日に曜日を書いている場合も触らない
    (docs/f_08 §6.3、2026-09-26 監査 C08#2: 「10 月 15 日（火）」、実際は木曜)。
    年は本文のその月日の西暦 → 接地文の同じ月日の西暦 → (年の手がかりがどこにも
    無いときだけ) ``today`` に最も近い年。**年の手がかりがあるのに西暦に解けない**
    (「令和7年」「来年の」/ 接地文に別々の西暦) ときは書き換えない。
    """
    claims = weekday_claims(text)
    if not claims:
        return text
    grounded_years = _grounded_years(grounded)
    user_weekday_md = {(c.month, c.day) for c in weekday_claims(grounded)}
    chars = list(text)
    fixed = 0
    for claim in claims:
        key = (claim.month, claim.day)
        if key not in grounded_years or key in user_weekday_md:
            continue
        if claim.year is not None:
            actual = _safe_date(claim.year, claim.month, claim.day)
        elif claim.year_unresolved:
            continue
        else:
            years = grounded_years[key]
            if None in years or len(years) > 1:
                continue
            actual = (
                _safe_date(next(iter(years)), claim.month, claim.day)
                if years else nearest_date(claim.month, claim.day, today)
            )
        if actual is None:
            continue
        correct = _WEEKDAY_CHARS[actual.weekday()]
        if correct != claim.weekday:
            chars[claim.weekday_pos] = correct
            fixed += 1
    if not fixed:
        return text
    logger.info("Corrected %d weekday claim(s) against the calendar", fixed)
    return "".join(chars)
