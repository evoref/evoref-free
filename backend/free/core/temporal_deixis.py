"""現在・相対の時間直示 (deixis) の語彙 (横断基盤、純粋関数)。

「今日」「現在」「来週」「去年」「today」のように **発話時点を基準にして初めて
意味の決まる語** の唯一の定義 (CLAUDE.md §6 不変則 #14(a) 同一性)。ツール判定
(現在日時の問い / 照応の起点 / 過去の想起の抑止)、日数の問い、few-shot の採用
拒否、日付注記の発火、相対日付の解決が同じ語を使う。各所が自分で
``今日|本日|…`` を書いていた頃は 6 系統に分裂し、語の集合が少しずつ違った。

ここが持つのは **語彙の断片 (正規表現に埋め込む文字列) と語の集合** だけ。
正規表現の形 (後読み・「から」との組・語境界) は使う側の文脈で違ってよい
(docs/c_17 §1.1: 字句の鍵は正規表現が正しい道具)。語を足す・除くのはここだけ。

語の分類:

- 日: :data:`DAY_OFFSETS` (今日 = 0、かな読みを含む)。
- 週 / 月 / 年: :data:`WEEK_OFFSETS` / :data:`MONTH_OFFSETS` / :data:`YEAR_OFFSETS`。
- 現在 (瞬間): 名詞 :data:`NOW_NOUNS` (現在) / 副詞 :data:`NOW_ADVERBS`
  (只今 / ただいま。ただいまは帰宅の挨拶とも重なる) / 1 字の :data:`NOW_BARE` (今。
  単独では「今後」「今回」に埋もれるので使う側が後続で絞る)。
- かなの「いま」「きょう」は部分文字列で誤爆する (変わって**いま**せん / **きょう**み)
  ので、後続の除外を付けた断片 (:data:`NOW_KANA_PATTERN` / :data:`TODAY_KANA_PATTERN`)
  でだけ渡す。
- 英語: :data:`EN_TODAY` / :data:`EN_RELATIVE_DAYS` / :data:`EN_NOW` /
  :data:`EN_NOW_ADVERBS`。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from backend.free.core.script_ranges import HIRAGANA

#: 日単位の相対表現 → オフセット (日数)。かな読みを含む。
DAY_OFFSETS: dict[str, int] = {
    "一昨日": -2, "おととい": -2,
    "昨日": -1, "きのう": -1,
    "今日": 0, "本日": 0,
    "明日": 1, "あす": 1, "あした": 1,
    "明後日": 2, "あさって": 2,
}

#: 「今週 / 来週 / 再来週 / 先週 / 先々週」→ 週オフセット (日数)。週の起点は月曜。
WEEK_OFFSETS: dict[str, int] = {
    "今週": 0, "こんしゅう": 0,
    "来週": 7, "らいしゅう": 7,
    "再来週": 14, "さらいしゅう": 14,
    "先週": -7, "せんしゅう": -7,
    "先々週": -14, "せんせんしゅう": -14,
}

#: 月単位の相対表現 → オフセット (月数)。
MONTH_OFFSETS: dict[str, int] = {"先月": -1, "今月": 0, "来月": 1, "再来月": 2}

#: 年単位の相対表現 → オフセット (年数)。
YEAR_OFFSETS: dict[str, int] = {"去年": -1, "昨年": -1, "今年": 0, "来年": 1}

#: 比べる相手の「前の期間」を指す語 (前週 / 前月 / 前期 / 前年)。発話時点ではなく、
#: 話題の期間を基準にした 1 つ前。前期比・前年比のラベルに使う。
PREVIOUS_PERIOD_TERMS: tuple[str, ...] = ("前週", "前月", "前期", "前年")

#: 現在 (瞬間) を指す名詞。「現在の」「現在時刻」のように複合語も作る。
NOW_NOUNS: tuple[str, ...] = ("現在",)
#: 現在 (瞬間) を指す副詞。「ただいま」は帰宅の挨拶とも重なる。
NOW_ADVERBS: tuple[str, ...] = ("ただいま", "只今")
#: 1 字の「今」。単独では「今後 / 今回 / 今朝」に埋もれる。
NOW_BARE = "今"

#: かなの「いま」。「〜ています / います」に埋もれる形を後続で除く。
NOW_KANA_PATTERN = r"いま(?![すせしそまん])"
#: かなの「きょう」。「興味 / 教養 / 協力」等に埋もれる形を後続で除く。
TODAY_KANA_PATTERN = r"きょう(?![みりょ])"
#: 「今」が 1 語として立っている形 (「今の / 今は / 今、」/ 文末)。
NOW_BARE_STANDALONE_PATTERN = NOW_BARE + r"[のはがもへ、。 ]|" + NOW_BARE + "$"
#: かなの「いま」を語の前に置いて連体する形 (「いまの」) に使う素の綴り。
NOW_KANA = "いま"

#: 英語の発話日。
EN_TODAY: tuple[str, ...] = ("today",)
#: 英語の発話日の前後。
EN_RELATIVE_DAYS: tuple[str, ...] = ("tomorrow", "yesterday")
#: 英語の現在 (瞬間)。
EN_NOW: tuple[str, ...] = ("now", "current")
#: 英語の現在を表す副詞。
EN_NOW_ADVERBS: tuple[str, ...] = ("currently",)
#: 英語の「今週 / 今月 / 今年」。
EN_THIS_PERIOD_PATTERN = r"this\s+(?:week|month|year)"

#: 「何日」の前に付いて **今からの残り日数** を問う形にする前置き (あと / 残り / まで)。
#: ``date_math_cue.DAY_COUNT_ASK_RE`` が使う。
COUNTDOWN_DAYS_PREFIX = r"(?:あと|残り|のこり|まで(?:は|、)?)\s*"

_HIRAGANA_RE = re.compile(f"[{HIRAGANA}]")


def kanji_terms(offsets: dict[str, int], *, where=None) -> tuple[str, ...]:
    """オフセット表のうち漢字表記の語 (かな読みを除く)。``where`` はオフセットの条件。"""
    return tuple(
        w for w, off in offsets.items()
        if not _HIRAGANA_RE.search(w) and (where is None or where(off))
    )


def present_day_terms() -> tuple[str, ...]:
    """発話日そのもの (今日 / 本日)。"""
    return kanji_terms(DAY_OFFSETS, where=lambda off: off == 0)


def previous_period_terms() -> tuple[str, ...]:
    """1 つ前の期間を指す語 (前週 / 前月 / 前期 / 前年 と 先月 / 昨年)。

    ``response_arithmetic`` の符号の判定が「前期との増減」のラベル (先月比：▲3000円) を
    別の量として除くのに使う。
    """
    return (
        *PREVIOUS_PERIOD_TERMS,
        *kanji_terms(MONTH_OFFSETS, where=lambda off: off == -1),
        "昨年",
    )


def alternation(*groups: Iterable[str]) -> str:
    """語の集合を正規表現の選択 (``a|b|c``) に組む。長い語を先に並べる。

    長い順にするのは、ある語が別の語の接頭辞になっている場合 (仮に足されても)
    最長の語で一致させるため。重複は落とす。括弧は付けない (使う側が組む)。
    """
    seen: dict[str, None] = {}
    for group in groups:
        for word in group:
            seen.setdefault(word, None)
    return "|".join(re.escape(w) for w in sorted(seen, key=len, reverse=True))


def lookbehinds(*groups: Iterable[str], suffix: str = "") -> str:
    """各語 (+ ``suffix``) の直後であることを求める後読みの選択 ``(?<=a)|(?<=b)``。"""
    words = alternation(*groups).split("|")
    return "|".join(f"(?<={w}{suffix})" for w in words)


def ascii_words(*groups: Iterable[str]) -> str:
    """英語の語を英字境界付きで選択に組む ``(?<![A-Za-z])(?:a|b)(?![A-Za-z])``。"""
    return r"(?<![A-Za-z])(?:" + alternation(*groups) + r")(?![A-Za-z])"


def all_terms() -> frozenset[str]:
    """この語彙が持つ語の全体 (同一性検査が「生で書かれた直示語」を探す集合)。"""
    return frozenset(
        [*DAY_OFFSETS, *WEEK_OFFSETS, *MONTH_OFFSETS, *YEAR_OFFSETS,
         *PREVIOUS_PERIOD_TERMS, *NOW_NOUNS, *NOW_ADVERBS, NOW_BARE, NOW_KANA, "きょう",
         *EN_TODAY, *EN_RELATIVE_DAYS, *EN_NOW, *EN_NOW_ADVERBS],
    )
