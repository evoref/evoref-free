"""日本の国民の祝日 (横断基盤、純粋関数)。

営業日の数え上げ (「10 営業日後」「3 営業日前」) で土日と並んで除く対象。
祝日は **閉じた集合** (固定日 / ハッピーマンデー / 春分・秋分の近似式 /
振替休日 / 国民の休日) で、ツール判定側は「質問文に明示された休日だけ」を
抽出器に入れさせる方針 (発明させない) なので、暦の祝日はコードが供給する
(2026-09-11 ライブ監査 (k): 「今日から 10 営業日後」が 9/23 秋分の日を数えて
9/25 と答えた。正 9/28)。

対象は 2020 年以降 (天皇誕生日 2/23、2020〜2021 年の五輪特例を含む)。
春分・秋分は 2099 年まで有効な近似式。
"""

from __future__ import annotations

import datetime
from functools import lru_cache

_FIXED: tuple[tuple[int, int], ...] = (
    (1, 1),    # 元日
    (2, 11),   # 建国記念の日
    (2, 23),   # 天皇誕生日 (2020〜)
    (4, 29),   # 昭和の日
    (5, 3),    # 憲法記念日
    (5, 4),    # みどりの日
    (5, 5),    # こどもの日
    (8, 11),   # 山の日
    (11, 3),   # 文化の日
    (11, 23),  # 勤労感謝の日
)

#: (月, 第 n 月曜) — 成人の日 / 海の日 / 敬老の日 / スポーツの日。
_HAPPY_MONDAYS: tuple[tuple[int, int], ...] = ((1, 2), (7, 3), (9, 3), (10, 2))

#: 東京五輪の特例 (海の日 / スポーツの日 / 山の日の移動)。
_OLYMPIC_OVERRIDES: dict[int, dict[str, datetime.date]] = {
    2020: {
        "sea": datetime.date(2020, 7, 23), "sports": datetime.date(2020, 7, 24),
        "mountain": datetime.date(2020, 8, 10),
    },
    2021: {
        "sea": datetime.date(2021, 7, 22), "sports": datetime.date(2021, 7, 23),
        "mountain": datetime.date(2021, 8, 8),
    },
}


def _nth_monday(year: int, month: int, nth: int) -> datetime.date:
    first = datetime.date(year, month, 1)
    offset = (0 - first.weekday()) % 7
    return first + datetime.timedelta(days=offset + 7 * (nth - 1))


def _equinox(year: int, base: float) -> datetime.date:
    day = int(base + 0.242194 * (year - 1980) - int((year - 1980) / 4))
    month = 3 if base < 22 else 9
    return datetime.date(year, month, day)


@lru_cache(maxsize=64)
def national_holidays(year: int) -> frozenset[datetime.date]:
    """``year`` の国民の祝日 (振替休日・国民の休日を含む) の集合。"""
    base: set[datetime.date] = set()
    override = _OLYMPIC_OVERRIDES.get(year, {})
    for month, day in _FIXED:
        if (month, day) == (8, 11) and "mountain" in override:
            base.add(override["mountain"])
            continue
        base.add(datetime.date(year, month, day))
    for month, nth in _HAPPY_MONDAYS:
        key = {7: "sea", 10: "sports"}.get(month)
        if key and key in override:
            base.add(override[key])
            continue
        base.add(_nth_monday(year, month, nth))
    base.add(_equinox(year, 20.8431))  # 春分の日
    base.add(_equinox(year, 23.2488))  # 秋分の日

    holidays = set(base)
    # 振替休日: 日曜に当たる祝日の直後の「祝日でない日」。
    for d in sorted(base):
        if d.weekday() == 6:
            nxt = d + datetime.timedelta(days=1)
            while nxt in holidays:
                nxt += datetime.timedelta(days=1)
            holidays.add(nxt)
    # 国民の休日: 前後を祝日に挟まれた平日 (日曜・既存祝日を除く)。
    for d in sorted(base):
        mid = d + datetime.timedelta(days=1)
        after = d + datetime.timedelta(days=2)
        if after in base and mid not in holidays and mid.weekday() != 6:
            holidays.add(mid)
    return frozenset(holidays)


def national_holidays_between(
    start: datetime.date, end: datetime.date,
) -> tuple[datetime.date, ...]:
    """``[start, end]`` (両端含む) に入る祝日を昇順で返す。"""
    lo, hi = (start, end) if start <= end else (end, start)
    out: list[datetime.date] = []
    for year in range(lo.year, hi.year + 1):
        out.extend(d for d in national_holidays(year) if lo <= d <= hi)
    return tuple(sorted(out))


def is_national_holiday(day: datetime.date) -> bool:
    return day in national_holidays(day.year)
