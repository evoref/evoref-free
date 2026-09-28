"""日付演算を求めている問いの手掛かり (純粋関数、横断)。

EvorefLoop (ツール判定: 層 5.97 の発火 / 接地の開示) と EvorefLearn (few-shot の
採用拒否) の両方が同じ判定を要る。片方に regex を複製すると「食い違った複製」
になるので、ここ 1 か所に置く。

**単独の ``何日`` / ``何曜日`` は入れない。** 「今日は何日ですか」は現在日時の
問いで、日付演算ではない (now-only コマンドが正解)。
"""

from __future__ import annotations

import re

from backend.free.core.response_dates import literal_date_count

#: 「あと何日」「残り日数」等、**2 点間の日数** を尋ねる語。規則層
#: (``tool_judge_commands._day_count_command``) と層 5.97 の手掛かり
#: (:data:`DATE_MATH_CUE_RE`、下で合成する) の **唯一の語彙** (不変則 #14(a))。
#: 2 本に分かれていた頃は、規則層が日数の問いと認めた「今日から試験日まで
#: あと何日ありますか？」を 5.97 が手掛かり無しとして即 return し、日数が
#: モデルの暗算 (173 日、正 203 日) になった (2026-09-27 ライブ監査 C09#2)。
#:
#: 素の ``何日です`` は入れない。「今日は何日ですか」(now-only が正解) と
#: 「何月何日ですか」(日付を訊く問い) を飲み込む — 後者は年なし日付を読むように
#: なって実害が出た (「9 月 14 日の 3 週間前は何月何日ですか？」が日数カウント側に
#: 取られ ``days: 20`` を返した)。日数を問う形 (あと / 残り / まで + 何日) だけを採る。
DAY_COUNT_ASK_RE = re.compile(
    r"何日間|あと何日|残り\s*(?:の)?\s*日数|日数は|何日ある"
    r"|(?:あと|残り|のこり|まで(?:は|、)?)\s*何日"
    r"|(?<![A-Za-z])how\s+many\s+days(?![A-Za-z])"
    r"|(?<![A-Za-z])days\s+(?:left|remaining|until)(?![A-Za-z])",
    re.IGNORECASE,
)

#: 日付演算の手掛かりのうち **日数を問う語彙以外** の部分 (営業日 / 日後 / 第 N 曜 …)。
#: :data:`DATE_MATH_CUE_RE` はこれと :data:`DAY_COUNT_ASK_RE` の合成。
_NON_DAY_COUNT_CUE_PATTERN = (
    r"営業日|稼働日|平日|祝日"
    r"|日目|週目|[かヶケヵ箇]月目|年目"
    r"|日後|日前|週間後|週間前|[かヶケヵ箇]月後|[かヶケヵ箇]月前|年後|年前"
    r"|逆算|何日間|日間|数えて"
    # 「<月>の第 N X 曜日」も日付演算 (月・序数・曜日で閉じた日付、2026-09-10 F-14)
    r"|第\s*[1-5１-５一二三四五]\s*[月火水木金土日]曜"
    # 「<月>の最終金曜日」も同じく閉じた日付 (2026-09-12 (b))
    r"|(?:最終|最後の)\s*[月火水木金土日]曜"
    r"|(?<![A-Za-z])business\s+days?(?![A-Za-z])"
    r"|(?<![A-Za-z])working\s+days?(?![A-Za-z])"
    r"|(?<![A-Za-z])weekdays?(?![A-Za-z])"
    r"|(?<![A-Za-z])days?\s+(?:after|before|from|later|earlier|until)(?![A-Za-z])"
    r"|(?<![A-Za-z])weeks?\s+(?:after|before|from)(?![A-Za-z])"
)
_NON_DAY_COUNT_CUE_RE = re.compile(_NON_DAY_COUNT_CUE_PATTERN, re.IGNORECASE)

#: 日付演算をしている手掛かり。カスケードの数字 + 単位パターンから構造的に
#: 漏れる語 (漢数字 / ``営業日`` / ``日目``) をすべて拾う。日数を問う語彙は
#: :data:`DAY_COUNT_ASK_RE` を合成する (語を足すのは向こうだけ)。
DATE_MATH_CUE_RE = re.compile(
    DAY_COUNT_ASK_RE.pattern + "|" + _NON_DAY_COUNT_CUE_PATTERN, re.IGNORECASE,
)

#: 素の「何日」(「何月何日」の日は除く)。単独では「今日は何日ですか」と区別
#: できないので、**日付の個数** と組み合わせたときだけ日数の問いとみなす
#: (:func:`asks_day_count`)。
_BARE_HOW_MANY_DAYS_RE = re.compile(r"(?<!何月)何日")
#: 起点が今日であることを示す形 (「今日から 12 月 25 日」「4 月 1 日から今日で」)。
_TODAY_ENDPOINT_RE = re.compile(r"(?:今日|本日|現在)から|から(?:今日|本日|現在)")


def asks_day_count(query: str) -> bool:
    """2 点間の日数を尋ねているか (純粋関数)。

    語彙 (:data:`DAY_COUNT_ASK_RE`) に加えて、**構造** で拾う: 素の「何日」が
    あり、発話に具体日付が 2 つある (「10月1日から12月25日は何日ですか」「4月1日〜
    6月30日は何日ですか」)、または「今日から / から今日」と具体日付が 1 つある
    (「9月14日は今日から何日ですか」「入社日の4月1日から今日で何日ですか」)。
    2026-09-27 に素の「何日です」を語彙から外したとき、これらが規則層から落ちて
    now-only (暗算) に戻った (独立レビュー)。「今日は何日ですか」は日付 0 個なので
    巻き込まない。
    """
    text = query or ""
    if DAY_COUNT_ASK_RE.search(text):
        return True
    if not _BARE_HOW_MANY_DAYS_RE.search(text):
        return False
    dates = literal_date_count(text)
    return dates >= 2 or (dates == 1 and bool(_TODAY_ENDPOINT_RE.search(text)))


def day_count_is_the_only_cue(query: str) -> bool:
    """発話の日付演算の手掛かりが **日数の問いだけ** か (純粋関数)。

    「荷物はあと何日で届きますか？」のように終点がどこにも無い日数の問いで
    抽出器を撃たないための判定 (tool_call_judge の層 5.97)。
    """
    text = query or ""
    return asks_day_count(text) and not _NON_DAY_COUNT_CUE_RE.search(text)


def day_count_closed_in_query(query: str) -> bool:
    """日数の問いの **両端が発話だけで閉じている** か (純粋関数)。

    日付が 2 つ、「今日から / から今日」と日付 1 つ、または「今年の残り / 年末
    まで」。Mem の corpus 省略 (``search_pipeline.corpus_layer_skipped_for_query``)
    はこのときだけ省く — 「設計書 v2 の締切まであと何日？」は終点が文書にある。
    規則層の ``_day_count_command`` が組める形と同じ。
    """
    text = query or ""
    if not asks_day_count(text):
        return False
    # 日数の問いで日付が 1 つ以上あれば、もう一端は日付か今日で閉じる
    # (「9 月 14 日まであと何日」は今日との差)。
    return literal_date_count(text) >= 1 or bool(YEAR_REMAINDER_RE.search(text))


#: 「今年の残り」「年末まで」— 年末までの日数 (規則層も同じ定義を使う)。
YEAR_REMAINDER_RE = re.compile(
    r"今年.{0,6}(?:残り|あと)|年末まで|(?:残り|あと).{0,4}今年"
    r"|(?<![A-Za-z])rest\s+of\s+(?:the\s+)?year(?![A-Za-z])",
    re.IGNORECASE,
)


def query_has_date_math_cue(query: str) -> bool:
    """クエリが日付演算を求めているか (手掛かり語 + 日数の問いの構造)。"""
    text = query or ""
    return bool(DATE_MATH_CUE_RE.search(text)) or asks_day_count(text)


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


def last_user_query(conversation: list[dict] | None, *, before: str = "") -> str:
    """会話履歴の末尾から **直前のユーザー発話** を返す (無ければ空)。

    ``before`` に今回の発話を渡すと、履歴の末尾がその発話自身であれば飛ばす。
    チャット API は今回の user ターンを積んでから履歴を取るので、末尾は
    今回の発話になっている — それを「直前」と読むと継承は決して成立しない
    (2026-09-09 検証 V05/5, V06/2: 「その週の月曜日が休みだとしたら、着手日は
    どうなりますか」が層 5.97 に一度も届かず暗算に落ちた)。
    """
    skipped_self = False
    for msg in reversed(conversation or []):
        if str(msg.get("role") or "") != "user":
            continue
        content = str(msg.get("content") or "")
        if before and not skipped_self and content.strip() == before.strip():
            skipped_self = True
            continue
        return content
    return ""


def conversation_has_date_math_cue(query: str, conversation: list[dict] | None) -> bool:
    """今回の発話、または継いだ直前のユーザー発話に日付演算の手掛かりがあるか。"""
    return query_inherits_date_math(query, last_user_query(conversation, before=query))


__all__ = [
    "DATE_MATH_CUE_RE",
    "DAY_COUNT_ASK_RE",
    "YEAR_REMAINDER_RE",
    "asks_day_count",
    "conversation_has_date_math_cue",
    "day_count_closed_in_query",
    "day_count_is_the_only_cue",
    "last_user_query",
    "query_has_date_math_cue",
    "query_inherits_date_math",
]
