"""日付演算を求めている問いの手掛かり (純粋関数、横断)。

EvorefLoop (ツール判定: 層 5.97 の発火 / 接地の開示) と EvorefLearn (few-shot の
採用拒否) の両方が同じ判定を要る。片方に regex を複製すると「食い違った複製」
になるので、ここ 1 か所に置く。

**単独の ``何日`` / ``何曜日`` は入れない。** 「今日は何日ですか」は現在日時の
問いで、日付演算ではない (now-only コマンドが正解)。
"""

from __future__ import annotations

import re

#: 日付演算をしている手掛かり。カスケードの数字 + 単位パターンから構造的に
#: 漏れる語 (漢数字 / ``営業日`` / ``日目``) をすべて拾う。
DATE_MATH_CUE_RE = re.compile(
    r"営業日|稼働日|平日|祝日"
    r"|日目|週目|[かヶケヵ箇]月目|年目"
    r"|日後|日前|週間後|週間前|[かヶケヵ箇]月後|[かヶケヵ箇]月前|年後|年前"
    r"|逆算|何日間|日間|数えて"
    r"|(?<![A-Za-z])business\s+days?(?![A-Za-z])"
    r"|(?<![A-Za-z])working\s+days?(?![A-Za-z])"
    r"|(?<![A-Za-z])weekdays?(?![A-Za-z])"
    r"|(?<![A-Za-z])days?\s+(?:after|before|from|later|earlier|until)(?![A-Za-z])"
    r"|(?<![A-Za-z])weeks?\s+(?:after|before|from)(?![A-Za-z])",
    re.IGNORECASE,
)


def query_has_date_math_cue(query: str) -> bool:
    """クエリが日付演算を求めているか。"""
    return bool(DATE_MATH_CUE_RE.search(query or ""))


#: 追い質問が「日付の答え」を求めている形。前のターンの日付演算の条件だけを
#: 変える問い (「さらに毎週水曜日は作業できない日として除くと、何月何日に
#: なりますか」) は、それ自身には ``営業日`` も ``逆算`` も無い。
_ASKS_FOR_DATE_RE = re.compile(
    r"何月何日|何日(?:に|で|です|になり|にな|ですか|か)|いつ(?:に|で|です|になり|か)"
    r"|何曜日|どう変わ|どうなり|どうずれ|ずれ(?:ます|る)"
    r"|(?<![A-Za-z])(?:what|which)\s+(?:date|day)(?![A-Za-z])",
    re.IGNORECASE,
)


def query_inherits_date_math(query: str, previous_user_query: str) -> bool:
    """手掛かり語を持たない追い質問が、直前のユーザー発話の日付演算を継ぐか。

    直前の発話に手掛かり語があり、今回の発話が日付の答えを求めている
    (``_ASKS_FOR_DATE_RE``) ときだけ真。「今日は何日ですか」は直前が日付演算で
    なければ従来どおり now-only。実インシデント (2026-09-09 検証 V05/2):
    「さらに毎週水曜日は…除くと、何月何日になりますか」が層 5.97 に届かず、
    モデルが暗算して 10/13 (正 10/8) を返した。
    """
    if query_has_date_math_cue(query):
        return True
    if not query_has_date_math_cue(previous_user_query):
        return False
    return bool(_ASKS_FOR_DATE_RE.search(query or ""))


def last_user_query(conversation: list[dict] | None) -> str:
    """会話履歴の末尾から **直前のユーザー発話** を返す (無ければ空)。"""
    for msg in reversed(conversation or []):
        if str(msg.get("role") or "") == "user":
            return str(msg.get("content") or "")
    return ""


def conversation_has_date_math_cue(query: str, conversation: list[dict] | None) -> bool:
    """今回の発話、または継いだ直前のユーザー発話に日付演算の手掛かりがあるか。"""
    return query_inherits_date_math(query, last_user_query(conversation))


__all__ = [
    "DATE_MATH_CUE_RE",
    "conversation_has_date_math_cue",
    "last_user_query",
    "query_has_date_math_cue",
    "query_inherits_date_math",
]
