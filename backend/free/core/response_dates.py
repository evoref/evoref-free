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
from datetime import date

__all__ = [
    "extract_tool_target_date",
    "response_dates",
    "ignores_date_result",
]

#: 全角数字 → 半角。応答本文は全角で日付を書くことがある。
_ZENKAKU_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

#: ``date_intent`` の生成コマンドが印字する ``target:`` 行 (``build_date_intent_command``
#: 参照)。``business_days_from`` / ``days_from`` / ``weekday_of`` だけが持つ
#: (``days_between`` は日数を返すのでこの行を持たない)。
_TARGET_LINE_RE = re.compile(r"target:\s*(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})")

#: ISO 形式の日付 (``2026-10-09``)。前後が数字に接していないことを確認し、
#: より長い数値列の部分一致を避ける。
_ISO_DATE_RE = re.compile(
    r"(?<!\d)(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?!\d)",
)

#: 和文の月日 (``10月9日`` / ``2026年10月9日``)。年は省略可。
_JP_DATE_RE = re.compile(
    r"(?:(?P<y>\d{4})年)?(?P<m>\d{1,2})月(?P<d>\d{1,2})日",
)

#: スラッシュ区切りの月日 (``10/9``)。年は文脈 (``year_hint``) 頼み。
_SLASH_DATE_RE = re.compile(r"(?<!\d)(?P<m>\d{1,2})/(?P<d>\d{1,2})(?!\d)")


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
    for m in _ISO_DATE_RE.finditer(normalized):
        matches.append(
            (m.start(), m.end(), _safe_date(int(m["y"]), int(m["m"]), int(m["d"]))),
        )
    for m in _JP_DATE_RE.finditer(normalized):
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
