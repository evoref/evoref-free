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
from datetime import date, datetime, timedelta

#: 「今週 / 来週 / 再来週 / 先週 / 先々週」→ 週オフセット (日数)。
WEEK_OFFSETS: dict[str, int] = {
    "今週": 0, "こんしゅう": 0,
    "来週": 7, "らいしゅう": 7,
    "再来週": 14, "さらいしゅう": 14,
    "先週": -7, "せんしゅう": -7,
    "先々週": -14, "せんせんしゅう": -14,
}

#: 曜日名 → ``datetime.weekday()`` の値 (月曜 = 0)。
WEEKDAY_INDEX: dict[str, int] = {
    "月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6,
}

WEEK_OF_WEEKDAY_RE = re.compile(
    r"(先々週|再来週|今週|来週|先週|こんしゅう|らいしゅう|さらいしゅう"
    r"|せんせんしゅう|せんしゅう)"
    r"\s*の?\s*([月火水木金土日])曜日?",
)

#: 日単位の相対表現 → オフセット (日数)。
DAY_OFFSETS: dict[str, int] = {
    "一昨日": -2, "おととい": -2,
    "昨日": -1, "きのう": -1,
    "今日": 0, "本日": 0,
    "明日": 1, "あす": 1, "あした": 1,
    "明後日": 2, "あさって": 2,
}

_DAY_RELATIVE_RE = re.compile(
    "(" + "|".join(sorted(DAY_OFFSETS, key=len, reverse=True)) + ")",
)

#: 既に絶対日付が併記されている表現 (「来週の金曜日 (2026-09-18)」) を二重に
#: 注記しないための検査。
_ANNOTATED_RE = re.compile(r"\s*[(（]\d{4}-\d{2}-\d{2}[)）]")


def week_of_weekday(anchor: date, week_word: str, weekday_char: str) -> date | None:
    """「来週の金曜日」型を ``anchor`` の週を起点に解く。該当しなければ None。"""
    offset = WEEK_OFFSETS.get(week_word)
    weekday = WEEKDAY_INDEX.get(weekday_char)
    if offset is None or weekday is None:
        return None
    monday = anchor - timedelta(days=anchor.weekday())
    return monday + timedelta(days=offset + weekday)


def resolve_relative_dates(text: str, anchor: date) -> list[tuple[str, date]]:
    """本文中の相対日付表現を ``(逐語 span, 解決した日付)`` で返す (出現順)。"""
    out: list[tuple[str, date]] = []
    for m in WEEK_OF_WEEKDAY_RE.finditer(text or ""):
        resolved = week_of_weekday(anchor, m.group(1), m.group(2))
        if resolved is not None:
            out.append((m.group(0), resolved))
    for m in _DAY_RELATIVE_RE.finditer(text or ""):
        out.append((m.group(0), anchor + timedelta(days=DAY_OFFSETS[m.group(1)])))
    return out


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

    def _sub_day(m: re.Match[str]) -> str:
        word = m.group(1)
        if DAY_OFFSETS[word] == 0 or _ANNOTATED_RE.match(out[m.end():]):
            return word
        return f"{word} ({(anchor_date + timedelta(days=DAY_OFFSETS[word])).isoformat()})"

    return _DAY_RELATIVE_RE.sub(_sub_day, out)
