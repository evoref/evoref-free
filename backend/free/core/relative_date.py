"""相対日付表現の決定論的な解決 (横断基盤、純粋関数)。

「来週の金曜日」「明後日」のような発話時点に相対する日付は、**発話時刻と
組で** 初めて意味を持つ。記憶へそのまま残すと翌週には別の日を指す
(2026-09-10 ライブ監査 (h) H-04: 「来週の金曜日に見積書を送る」が
``mem.world.assertion`` / ``mem.personal.schedule`` に相対表現のまま live で
残った。正答できたのは要約由来の commitment に絶対日付があったからで、
運に依っていた)。

ツール判定側 (``agent.tool_judge_commands``) の週相対の解釈と **同じ表** を
使う — 週の起点は月曜 (ISO / 日本の慣行)、``今週`` はその週、``来週`` は
+7 日、``再来週`` は +14 日、``先週`` は -7 日、``先々週`` は -14 日。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# 「今週 / 来週 …」「昨日 / 明日 …」→ オフセットの表 (WEEK_OFFSETS / DAY_OFFSETS) は
# 直示の語彙の SSOT (temporal_deixis) が持つ。ここは解決だけを持つ。
from backend.free.core.script_ranges import KANJI, KATAKANA
from backend.free.core.temporal_deixis import (
    DAY_OFFSETS,
    MONTH_OFFSETS,
    NOW_BARE,
    NOW_NOUNS,
    WEEK_OFFSETS,
    YEAR_OFFSETS,
    alternation,
    kanji_terms,
    present_day_terms,
)

#: 曜日名 → ``datetime.weekday()`` の値 (月曜 = 0)。
WEEKDAY_INDEX: dict[str, int] = {
    "月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6,
}

WEEK_OF_WEEKDAY_RE = re.compile(
    "(" + alternation(WEEK_OFFSETS) + ")"
    r"\s*の?\s*([月火水木金土日])曜日?",
)

_DAY_RELATIVE_RE = re.compile("(" + alternation(DAY_OFFSETS) + ")")

#: 期間の単位 (「3 日」「2 週間」「1 か月」「1 年」)。オフセットの読みと、計算結果の
#: 丸め違いから率の分母の期間を外す判定 (``text_quality.misrounded_result_values``) が共有する。
#: **オフセットの単位と向きの語彙の SSOT** (不変則 #14(a))。規則カスケード
#: (``agent.tool_judge_commands._RELATIVE_OFFSET_RE``)・相対日付の注記
#: (:data:`_OFFSET_RE`)・起点の構造 (:data:`REFERENT_OFFSET_RE`) が同じ語彙から
#: 組む。分かれていた頃は「3 日先」「1 週前」を規則だけが読み、起点の分類から
#: 漏れて今日起点のコマンドに戻った (2026-10-05 監査のレビュー)。裸の「月」は
#: 「12 月後半」「4 月後」と衝突するので単位にしない (「か月」を使う)。
PERIOD_UNIT_ALTERNATION = "日|週間|週|[かヶケヵカ箇]月|年"
#: 向き (「先」は「後」と同じ)。直後の「半」は「後半 / 前半」なので向きではない。
OFFSET_DIRECTION_PATTERN = r"(?:前|後|先)(?!半)"
#: 数量の直前 (「第 3 週後半」の 3 は期間ではない)。
OFFSET_NUMBER_GUARD = r"(?<!第)"

#: 「今日から 2 週間後」「3 日後」「1 か月前」— 起点 (省略時は発話日) からの
#: オフセット。月は暦月で進める (末日超過は末日へ丸める)。
_OFFSET_RE = re.compile(
    r"(?:(?P<base>" + alternation(DAY_OFFSETS) + r")\s*から\s*)?"
    + OFFSET_NUMBER_GUARD + r"(?P<n>\d+)\s*(?P<unit>" + PERIOD_UNIT_ALTERNATION + r")"
    r"\s*(?P<dir>" + OFFSET_DIRECTION_PATTERN + ")"
)
_UNIT_DAYS = {"日": 1, "週間": 7, "週": 7}

#: オフセットの起点の種別。``today`` は発話日 (起点の省略・「今日から」)、
#: ``literal`` は発話に書かれた時間表現 (「9月14日の」「来週の水曜日の」)、
#: ``referent`` は会話で日付が確定した参照名詞 (「リリース日の」「締め切りの」)。
ANCHOR_TODAY = "today"
ANCHOR_LITERAL = "literal"
ANCHOR_REFERENT = "referent"

#: 「<起点>の N 営業日前」「<起点>から 2 週間後」— オフセットの起点となる語句と、
#: その後ろのオフセット。**起点の構造の SSOT** (ツール判定の照応・規則カスケード・
#: 相対日付の注記・日付結果の検証が同じ形を見る、不変則 #14(a))。
REFERENT_OFFSET_RE = re.compile(
    r"(?P<anchor>[^\s、。,！？!?]+?)(?:から|より|の)\s*"
    + OFFSET_NUMBER_GUARD + r"(?P<n>\d+)\s*"
    r"(?:営業日|" + PERIOD_UNIT_ALTERNATION + r")\s*(?:" + OFFSET_DIRECTION_PATTERN + "|以内)"
)
#: 起点が「時間表現そのもの」なら参照ではない (具体日付 / 直示語 / 暦の区切り)。
#: 起点の語句の **最後の区切り** (:data:`_ANCHOR_NOUN_SPLIT_RE`) に掛ける —
#: 「今月の締め切り」の「今月」は修飾で、起点は締め切り。
TEMPORAL_ANCHOR_RE = re.compile(
    alternation(
        kanji_terms(DAY_OFFSETS), kanji_terms(WEEK_OFFSETS), MONTH_OFFSETS,
        YEAR_OFFSETS, [NOW_BARE], NOW_NOUNS,
    )
    + r"|\d+\s*[年月日週]|[月火水木金土日]曜|[週月年](?:末|初|始)"
    r"|[0-9]{4}-[0-9]{2}-[0-9]{2}|\d{1,2}/\d{1,2}"
)
#: 起点の語句が発話日を指す語で終わる (「今から」「締め切りは今日から」)。
_PRESENT_ANCHOR_WORDS = (*present_day_terms(), *NOW_NOUNS, NOW_BARE)
#: 同じ文で先に発話日を **演算の基点として** 置いている (「今日種をまくと、収穫の目安の
#: 80 日後」「今日から見て…」)。「今日、さっきのリリース日の 1 週間前を教えて」の
#: ような文頭の「今日」は基点ではない。
_PRESENT_BASE_IN_CLAUSE_RE = re.compile(
    "(?:" + alternation(present_day_terms(), NOW_NOUNS) + r")(?:から|[^、。]*?と[、,])"
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[。．！？!?\n]")
#: 起点の語句の直前まで含めて具体日付で終わる (「9 月 30 日の」は語句が「日」だけに
#: なる — 語句は空白を含まない)。
_SPACED_LITERAL_TAIL_RE = re.compile(r"\d+\s*[年月日]$")
#: 参照名詞の前の指示詞 (「その締め切り」→「締め切り」)。
_DEMONSTRATIVE_PREFIX_RE = re.compile(r"^(?:その|この|あの|当該の?)")
#: 起点の語句の中の修飾・主題・連体の区切り (「さっきのリリース日」「売却価格は購入価格」
#: 「今日の会議で決まったリリース日」)。最後の区切りの後ろが起点の名詞。
_ANCHOR_NOUN_SPLIT_RE = re.compile(r"[のはがをにでと]|[っし]た")
#: 起点の語句が日単位の直示語で終わる (語の途中の「説明日」等は除く)。
_DAY_WORD_TAIL_RE = re.compile(
    f"(?<![{KANJI}{KATAKANA}])(?:" + alternation(DAY_OFFSETS) + ")$",
)


@dataclass(frozen=True, slots=True)
class OffsetAnchor:
    """オフセット表現 1 件の起点。``offset_start`` は数量 (N) の位置。"""

    kind: str
    #: 起点の語句と助詞 (「さっきのリリース日の」)。
    phrase: str
    #: 参照名詞 (``kind == "referent"`` のときだけ、「リリース日」)。
    referent: str
    offset_start: int
    #: 起点が日単位の直示語 (「明日の 3 日後」) なら今日からの日数。それ以外は ``None``。
    day_offset: int | None = None


def _anchor_head(anchor: str) -> str:
    """起点の語句の最後の区切りの後ろ (「今月の締め切り」→「締め切り」)。

    日単位の直示語で終わるなら (「明日」「あした」「おととい」) その語 — かなの語は
    区切りの字 (「し」「と」) を含むので先に見る。
    """
    day = _DAY_WORD_TAIL_RE.search(anchor)
    if day is not None:
        return day.group(0)
    return _DEMONSTRATIVE_PREFIX_RE.sub("", _ANCHOR_NOUN_SPLIT_RE.split(anchor)[-1])


def offset_anchors(text: str) -> list[OffsetAnchor]:
    """「<起点>の / から / より N <単位> 前 / 後」の起点を出現順に分類する (純粋関数)。

    起点の語句が発話日を指す語 (「今から」) か、同じ文で先に発話日を演算の基点に
    置いていれば ``today``、起点の名詞が時間表現 (「9月14日の」「来週の水曜日の」
    「明日の」「月末の」) なら ``literal``、それ以外 (「リリース日の」「締め切りの」
    「その日の」) は ``referent``。起点の語句を持たない裸のオフセット (「3 日後」) は
    ここに現れない (今日起点)。
    """
    body = text or ""
    out: list[OffsetAnchor] = []
    for m in REFERENT_OFFSET_RE.finditer(body):
        anchor = m.group("anchor")
        head = _anchor_head(anchor)
        clause_start = 0
        for b in _CLAUSE_BOUNDARY_RE.finditer(body, 0, m.start()):
            clause_start = b.end()
        day_offset = DAY_OFFSETS.get(head)
        if anchor.endswith(_PRESENT_ANCHOR_WORDS):
            kind = ANCHOR_TODAY
        elif day_offset is not None or TEMPORAL_ANCHOR_RE.search(head) or (
            _SPACED_LITERAL_TAIL_RE.search(body, 0, m.end("anchor"))
        ):
            kind = ANCHOR_LITERAL
        elif _PRESENT_BASE_IN_CLAUSE_RE.search(body, clause_start, m.start()):
            kind = ANCHOR_TODAY
        else:
            kind = ANCHOR_REFERENT
        out.append(OffsetAnchor(
            kind=kind,
            phrase=body[m.start():m.start("n")],
            referent=head if kind == ANCHOR_REFERENT and len(head) >= 2 else "",
            offset_start=m.start("n"),
            day_offset=day_offset if kind == ANCHOR_LITERAL else None,
        ))
    return out


def has_non_today_offset_anchor(text: str) -> bool:
    """オフセットの起点が今日以外 (具体日付 / 参照名詞) に置かれているか (純粋関数)。"""
    return any(a.kind != ANCHOR_TODAY for a in offset_anchors(text))


def _shift_months(anchor: date, months: int) -> date:
    y, m = divmod(anchor.month - 1 + months, 12)
    year, month = anchor.year + y, m + 1
    last = (date(year + (month // 12), month % 12 + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(anchor.day, last))


def resolve_offset(anchor: date, m: re.Match[str]) -> date:
    """オフセット表現 (``_OFFSET_RE`` の一致) を ``anchor`` から解く。"""
    base = anchor + timedelta(days=DAY_OFFSETS.get(m.group("base") or "今日", 0))
    n = int(m.group("n")) * (-1 if m.group("dir") == "前" else 1)
    unit = m.group("unit")
    if unit in _UNIT_DAYS:
        return base + timedelta(days=n * _UNIT_DAYS[unit])
    if unit == "年":
        return _shift_months(base, 12 * n)
    return _shift_months(base, n)

#: 既に絶対日付が併記されている表現 (「来週の金曜日 (2026-09-18)」) を二重に
#: 注記しないための検査。
_ANNOTATED_RE = re.compile(r"\s*[(（]\d{4}-\d{2}-\d{2}[)）]")


def strip_date_annotation_after(text: str, position: int) -> str:
    """``position`` 直後に ``(YYYY-MM-DD)`` の併記があれば落とす (純粋関数)。

    訂正で相対表現を置換したとき (「来週の金曜日 (2026-09-18)」→「再来週の
    月曜日 (2026-09-18)」)、旧い注記が残ると :func:`annotate_relative_dates` は
    「併記済み」と見て再解決せず、**訂正後の表現に訂正前の日付** が付いたまま
    記憶される (2026-09-12 ライブ監査)。置換した側が注記を外し、再注記に委ねる。
    """
    if not text or position < 0 or position > len(text):
        return text
    m = _ANNOTATED_RE.match(text[position:])
    if m is None:
        return text
    return text[:position] + text[position + m.end():]


def week_of_weekday(anchor: date, week_word: str, weekday_char: str) -> date | None:
    """「来週の金曜日」型を ``anchor`` の週を起点に解く。該当しなければ None。"""
    offset = WEEK_OFFSETS.get(week_word)
    weekday = WEEKDAY_INDEX.get(weekday_char)
    if offset is None or weekday is None:
        return None
    monday = anchor - timedelta(days=anchor.weekday())
    return monday + timedelta(days=offset + weekday)


@dataclass(frozen=True, slots=True)
class RelativeDateSpan:
    """相対日付表現 1 件。起点が今日でなければ ``date`` は ``None`` (今日からは解かない)。"""

    span: str
    date: date | None
    anchor: str
    referent: str = ""


def _non_today_anchor(
    m: re.Match[str], anchors: dict[int, OffsetAnchor],
) -> OffsetAnchor | None:
    """``_OFFSET_RE`` の一致の起点が今日でなければその起点 (今日起点なら ``None``)。"""
    if m.group("base"):
        return None
    found = anchors.get(m.start("n"))
    return None if found is None or found.kind == ANCHOR_TODAY else found


def _resolved_offset(
    m: re.Match[str], anchors: dict[int, OffsetAnchor], today: date,
) -> tuple[date | None, OffsetAnchor | None]:
    """オフセットの解決日と、今日でない起点 (今日起点なら ``None``)。

    起点が日単位の直示語 (「明日の 3 日後」) なら起点日から解く。参照名詞・
    具体日付の起点は会話・発話の日付が要るので解かない (``None``)。
    """
    found = _non_today_anchor(m, anchors)
    if found is None:
        return resolve_offset(today, m), None
    if found.day_offset is not None:
        return resolve_offset(today + timedelta(days=found.day_offset), m), found
    return None, found


def classify_relative_dates(text: str, anchor: date) -> list[RelativeDateSpan]:
    """本文中の相対日付表現を起点の種別つきで返す (出現順、純粋関数)。

    「リリース日の 1 週間前」の起点は会話で確定した日付で、今日ではない。
    今日から解くと誤った絶対日付を確定事実として渡してしまう (2026-10-05
    ライブ監査: 「さっきのリリース日の1週間前」に「1週間前 = 2026-09-28」を注記、
    正 10/13)。起点が参照名詞・具体日付の span は ``date=None`` で返す (起点が
    日単位の直示語 (「明日の 3 日後」) なら起点日から解く)。
    """
    out: list[RelativeDateSpan] = []
    for m in WEEK_OF_WEEKDAY_RE.finditer(text or ""):
        resolved = week_of_weekday(anchor, m.group(1), m.group(2))
        if resolved is not None:
            out.append(RelativeDateSpan(m.group(0), resolved, ANCHOR_TODAY))
    anchors = {a.offset_start: a for a in offset_anchors(text)}
    for m in _OFFSET_RE.finditer(text or ""):
        resolved, found = _resolved_offset(m, anchors, anchor)
        if found is None:
            out.append(RelativeDateSpan(m.group(0), resolved, ANCHOR_TODAY))
        else:
            out.append(RelativeDateSpan(m.group(0), resolved, found.kind, found.referent))
    for m in _DAY_RELATIVE_RE.finditer(text or ""):
        out.append(RelativeDateSpan(
            m.group(0), anchor + timedelta(days=DAY_OFFSETS[m.group(1)]), ANCHOR_TODAY,
        ))
    return out


def resolve_relative_dates(text: str, anchor: date) -> list[tuple[str, date]]:
    """本文中の **今日起点の** 相対日付表現を ``(逐語 span, 解決した日付)`` で返す (出現順)。

    起点が参照名詞・具体日付のオフセット (「リリース日の 1 週間前」) は含めない
    (:func:`classify_relative_dates`)。
    """
    return [
        (s.span, s.date) for s in classify_relative_dates(text, anchor)
        if s.date is not None
    ]


def resolves_today_anchored_span(text: str, target: date, today: date) -> bool:
    """``target`` が本文の **今日起点の** 相対日付表現のどれかの解決と一致するか (純粋関数)。

    今日起点のオフセットと、起点が具体日付・参照名詞のオフセットを並べた問い
    (「今日から3日後と、10月20日の3日前は？」) で、今日起点で数えたツールの
    ``target`` がどちらの span に当たるかを決める。今日起点の span に当たるなら
    ``target`` はその span の正しい答えで、起点の食い違いを理由に検証を外さない。

    見るのはオフセット (「今日から3日後」) と週相対の曜日 (「来週の金曜日」) の span
    だけ。裸の直示語 (「今日は何曜日？」「…は今日から何日後？」の今日) は演算の答えでは
    なく、今日起点の ``target`` が偶然それと一致しても別の span の答えを裏付けない。
    """
    return any(
        s.date == target
        for s in classify_relative_dates(text, today)
        if s.anchor == ANCHOR_TODAY and s.date is not None and s.span not in DAY_OFFSETS
    )


def annotate_relative_dates(text: str, anchor: datetime | date | None) -> str:
    """相対日付表現の直後に ``(YYYY-MM-DD)`` を併記して返す (純粋関数)。

    ``anchor`` は **発話時刻** (現地日付)。無ければそのまま返す。既に絶対日付が
    併記されていれば触らない。「今日」は単独では注記しない (問い・挨拶に
    頻出し、文脈上の日付情報を持たない)。
    """
    if anchor is None or not text:
        return text
    anchor_date = anchor.date() if isinstance(anchor, datetime) else anchor

    def _sub_week(m: re.Match[str]) -> str:
        if _ANNOTATED_RE.match(text[m.end():]):
            return m.group(0)
        resolved = week_of_weekday(anchor_date, m.group(1), m.group(2))
        return m.group(0) if resolved is None else f"{m.group(0)} ({resolved.isoformat()})"

    out = WEEK_OF_WEEKDAY_RE.sub(_sub_week, text)
    anchors = {a.offset_start: a for a in offset_anchors(out)}

    def _sub_offset(m: re.Match[str]) -> str:
        # 起点が今日でない (「契約の 1 か月前」) なら今日から解かない。
        if _ANNOTATED_RE.match(out[m.end():]):
            return m.group(0)
        resolved, _found = _resolved_offset(m, anchors, anchor_date)
        return m.group(0) if resolved is None else f"{m.group(0)} ({resolved.isoformat()})"

    out = _OFFSET_RE.sub(_sub_offset, out)

    def _sub_day(m: re.Match[str]) -> str:
        word = m.group(1)
        # オフセットの起点 (「今日から 2 週間後」の「今日」) は上で解決済み。
        if DAY_OFFSETS[word] == 0 or _ANNOTATED_RE.match(out[m.end():]) or (
            out[m.end():].lstrip().startswith("から")
        ):
            return word
        return f"{word} ({(anchor_date + timedelta(days=DAY_OFFSETS[word])).isoformat()})"

    return _DAY_RELATIVE_RE.sub(_sub_day, out)


#: 併記済みの相対表現「来週の火曜日 (2026-09-15)」/「明日 (2026-09-11)」。
_ANNOTATED_RELATIVE_RE = re.compile(
    r"(?:(?:" + alternation(WEEK_OFFSETS) + r")\s*の?\s*[月火水木金土日]曜日?"
    r"|" + alternation(w for w, off in DAY_OFFSETS.items() if off != 0) + ")"
    r"\s*[(（](?P<date>\d{4}-\d{2}-\d{2})[)）]"
)
_WEEKDAY_JA = ("月", "火", "水", "木", "金", "土", "日")


def absolutize_annotated_dates(text: str) -> str:
    """併記済みの相対表現を絶対日付だけに置き換える (純粋関数)。

    「来週の火曜日 (2026-09-15)」→「2026-09-15 (火)」。相対表現は発話時刻に
    相対で、記憶を **別の日に読む** ときは起点が違う。注入で相対表現が残ると
    モデルはそれを復唱し (実測 2026-09-11 (j) J-09: 「案内文を送る予定日は」に
    「来週の火曜日です」)、読む日によって別の日を指す。
    """
    def _sub(m: re.Match[str]) -> str:
        try:
            d = date.fromisoformat(m.group("date"))
        except ValueError:
            return m.group(0)
        return f"{d.isoformat()} ({_WEEKDAY_JA[d.weekday()]})"

    return _ANNOTATED_RELATIVE_RE.sub(_sub, text or "")

