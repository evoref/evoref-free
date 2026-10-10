"""応答の欠陥を決定論で検出する検証器 (ターン成否の導出用、2026-10-07)。

``FeedbackCollector._derive_turn_outcome_with_reason`` が、既存の検証器がどれも
発火しなかったターンにだけ掛ける。どれも純関数で、入力 → ``None`` か理由の詳細
(英語、ログと ``turn_outcome_reason`` に載る)。例外は外へ出さず、判定できない
ときは検出しない側 (``None``) へ倒す。

**適合率を最優先する**。偽の ``failed`` は学習データを汚すので、別の読み方が
1 つでも成り立つ形は報告しない。語形を足して網を広げる方向では直さない
(CLAUDE.md 不変則 #12 / #14)。出力検査なので判定点の契約 (c_17) の対象外。
"""

from __future__ import annotations

import math
import re

from backend.free.core.response_arithmetic import JaNumber, iter_ja_numbers
from backend.free.core.script_ranges import HIRAGANA
from backend.log_config import get_logger

logger = get_logger("core.response_verifiers")

__all__ = [
    "declared_count_mismatch", "false_tool_unavailability", "mask_item_numbers",
    "misstated_change_rate",
]

#: 走査する本文の上限 (文字)。計算量を入力長に線形で抑え、巨大な応答でも止まらない。
_MAX_SCAN_CHARS = 50_000

#: コード片 (フェンスとインライン)。識別子・引数の数を量として読まない。
_CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)

#: 文の区切り。増減率と元の 2 値は同じ文の中にあるときだけ照合する。
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?\n]")


def _blank_code(text: str) -> str:
    """コード片を同じ長さの空白で潰す (位置を保つ)。"""
    return _CODE_RE.sub(lambda m: " " * len(m.group(0)), text)


def _head_sentences(text: str) -> str:
    """走査の上限までの本文。上限で切れた最後の文は落とす (途中で切れた数を読まない)。"""
    if len(text) <= _MAX_SCAN_CHARS:
        return text
    head = text[:_MAX_SCAN_CHARS]
    cut = max(head.rfind(ch) for ch in "。！？!?\n")
    return head[:cut + 1] if cut >= 0 else ""


# ---------------------------------------------------------------------------
# 増減率の再計算
# ---------------------------------------------------------------------------
#
# 2026-10-05 ライブ監査 (S02): 9 月の前月比 -11.95% を「約11.7%減」と述べた。
# 既存の算術検証 (``A op B = C`` の式・内訳の超過・冒頭の結論・符号) は、式を
# 書かずに 2 つの値と率を並べた文を見ない。同じ文に元の 2 値があれば、率は
# 決定論で再計算できる。

#: 率の直後に続く増減の向き (閉じた類。率が「変化」を述べている印)。
_CHANGE_AFTER_RE = re.compile(
    r"^[\s)）*]*(?:の|も|ほど|程度|近く)?\s*"
    r"(?:増|減|アップ|ダウン|上昇|低下|下落|伸び|成長|プラス|マイナス"
    r"|increase|decrease|growth|drop|rise|up\b|down\b)",
    re.IGNORECASE,
)
#: 率の直前に置かれる「比べた率」のラベル (閉じた類)。「税率」「構成比」は含めない。
_CHANGE_BEFORE_RE = re.compile(
    r"(?:前[年月期週日]比|昨年比|前年同[月期]比|対前[年月期]比"
    r"|伸び率|増加率|減少率|成長率|増減率|変化率|上昇率|下落率)"
    r"[^\d]{0,8}$",
)
#: 率の直前に付く正の符号 (負の符号は ``JaNumber.negative`` が持つ)。
_PLUS_SIGN_CHARS = "+＋"
#: 量でなく順序・日時・ラベルを表す単位の先頭字。「5 月」「2026 年」「第 3 位」を元の値に読まない。
_LABEL_UNIT_HEADS = frozenset("月日年週時分秒位番号回歳才度階期")
#: 単位の読み取り (数の直後の、ひらがな・空白・数字・区切りでない連なり)。
_UNIT_RE = re.compile(
    rf"[ \t　]*([^\s\d{HIRAGANA}%％、。,，.()（）:：;；\-−+＋→⇒~〜/／|*「」\[\]]{{0,4}})",
)
#: 値の直前の概算表現。表記の桁より粗い値として扱う。
_APPROX_BEFORE_RE = re.compile(r"(?:約|およそ|ほぼ|概ね|おおよそ)\s*$")
#: 概算の値に見込む相対の幅。
_APPROX_REL = 0.05
#: 2 値の後ろの値から率までの最大の距離 (文字)。
_RATE_GAP_MAX = 30
#: 主題・主語の助詞 (率が別の量を述べ始めた印)。
_TOPIC_PARTICLE_RE = re.compile(r"[はが]")
#: 率の許容の下限 (百分率のポイント)。表記の最終桁の丸めより粗い揺れを許す。
_RATE_TOLERANCE_FLOOR = 0.1


def _is_percent(sentence: str, num: JaNumber) -> bool:
    return sentence[num.end:].lstrip(" \t　").startswith(("%", "％"))


def _unit_of(sentence: str, num: JaNumber) -> str:
    m = _UNIT_RE.match(sentence, num.end)
    return m.group(1) if m else ""


#: 数どうしをつなぐ記号 (日付・時刻・範囲・識別子の区切り)。
_DIGIT_JOINERS = frozenset("-−/／:：~〜")


def _joined_to_digits(sentence: str, num: JaNumber) -> bool:
    """数が記号で別の数とつながっているか (「2026-08」「10:30」「3〜5万円」)。"""
    before = sentence[max(0, num.start - 2):num.start]
    after = sentence[num.end:num.end + 2]
    return (
        len(before) == 2 and before[1] in _DIGIT_JOINERS and before[0].isdigit()
    ) or (len(after) == 2 and after[0] in _DIGIT_JOINERS and after[1].isdigit())


def _value_interval(sentence: str, num: JaNumber) -> tuple[float, float]:
    """表記の桁 (と概算表現) から、その値が取りうる区間を返す。"""
    half = num.scale / 2.0
    if _APPROX_BEFORE_RE.search(sentence[max(0, num.start - 6):num.start]):
        half = max(half, num.value * _APPROX_REL)
    return num.value - half, num.value + half


def _rate_intervals(
    a: tuple[float, float], b: tuple[float, float],
) -> list[tuple[float, float]]:
    """2 値の区間から、率 (%) として成り立つ値の区間をすべて返す。

    どちらを基準にしたか (前→後 / 後→前) も、変化率か比率か (``B/A - 1`` /
    ``B/A``) も文からは決まらないので、全部を候補にする (報告しない側に倒す)。
    """
    out: list[tuple[float, float]] = []
    for (x1, x2), (y1, y2) in ((a, b), (b, a)):
        lo, hi = y1 / x2 * 100.0, y2 / x1 * 100.0
        out.append((lo, hi))  # 比率 (前月比 88% のような表記)
        c_lo, c_hi = lo - 100.0, hi - 100.0
        if c_lo <= 0.0 <= c_hi:
            out.append((0.0, max(-c_lo, c_hi)))
        else:
            out.append((min(abs(c_lo), abs(c_hi)), max(abs(c_lo), abs(c_hi))))
    return out


def _rate_tolerance(literal: str) -> float:
    decimals = len(literal.split(".")[1]) if "." in literal else 0
    return max(0.75 * 10.0 ** -decimals, _RATE_TOLERANCE_FLOOR)


def _states_change(sentence: str, num: JaNumber) -> bool:
    """率が「変化」として述べられているか (向きの語 / 比べた率のラベル / 符号)。"""
    after = sentence[num.end:].lstrip(" \t　")[1:]
    if _CHANGE_AFTER_RE.match(after):
        return True
    before = sentence[max(0, num.start - 20):num.start]
    if _CHANGE_BEFORE_RE.search(before):
        return True
    return num.negative or before.rstrip(" \t　").endswith(tuple(_PLUS_SIGN_CHARS))


def _change_rate_in_sentence(sentence: str) -> str | None:
    numbers = iter_ja_numbers(sentence)
    percents = [n for n in numbers if _is_percent(sentence, n)]
    if len(percents) != 1:
        return None
    rate = percents[0]
    if not _states_change(sentence, rate):
        return None
    groups: dict[str, list[JaNumber]] = {}
    for n in numbers:
        if n is rate or n.value <= 0:
            continue
        unit = _unit_of(sentence, n)
        # 単位の無い数 (「2026-08」の 2026 と 08、表の素の数) は量と言い切れない
        if not unit or (unit[:1] in _LABEL_UNIT_HEADS and unit != "時間") or unit.endswith("目"):
            continue
        if sentence[max(0, n.start - 1):n.start] == "第" or _joined_to_digits(sentence, n):
            continue
        groups.setdefault(unit, []).append(n)
    if any(len(g) > 2 for g in groups.values()):
        return None
    pairs = [g for g in groups.values() if len(g) == 2]
    if len(pairs) != 1:
        return None
    a, b = pairs[0]
    # 率は 2 値の後ろに近接して続く。あいだに主題・主語の助詞があれば、率は別の量の
    # 話 (「社員は 5 人から 8 人に増え、売上は 20% 増」)。
    gap = sentence[b.end:rate.start]
    if rate.start < b.end or len(gap) > _RATE_GAP_MAX or _TOPIC_PARTICLE_RE.search(gap):
        return None
    a_iv, b_iv = _value_interval(sentence, a), _value_interval(sentence, b)
    if a_iv[0] <= 0 or b_iv[0] <= 0:
        return None
    rate_literal = sentence[rate.start:rate.end].strip()
    tolerance = _rate_tolerance(rate_literal)
    for lo, hi in _rate_intervals(a_iv, b_iv):
        if lo - tolerance <= rate.value <= hi + tolerance:
            return None
    change = (b.value / a.value - 1.0) * 100.0
    if not math.isfinite(change):
        return None
    return (
        f"{rate_literal}% does not follow from "
        f"{sentence[a.start:a.end].strip()} and {sentence[b.start:b.end].strip()} "
        f"(change {change:+.2f}%)"
    )


def misstated_change_rate(response: str) -> str | None:
    """同じ文の 2 つの値から再計算した増減率と、述べた率が合わなければ理由を返す。

    条件をすべて満たしたときだけ報告する:

    1. 文に百分率がちょうど 1 つあり、変化として述べている (直後の向きの語
       「増 / 減 / 上昇」、直前の「前月比 / 伸び率」、または符号 ``+`` / ``-``)
    2. 同じ単位の値がちょうど 2 つ、ほかに同じ単位の値が 3 つ以上並ぶ組が無い。
       単位の無い数・日時や順序のラベル (「5 月」「第 3」)・記号で別の数とつながる数
       (「2026-08」「3〜5万円」) は値に数えない
    3. 率は 2 値の後ろ :data:`_RATE_GAP_MAX` 文字以内にあり、あいだに主題・主語の
       助詞 (は / が) が無い — あれば率は別の量の話
    4. 2 値の表記の桁 (「約」付きは相対 5%) から取りうる区間で、どちらを基準にした
       変化率・比率を計算しても、述べた率 (表記の桁、下限 0.1 ポイント) に届かない

    率と 2 値が別の文にある形 (表の行・前の文の値) は見ない。符号 (増か減か) も
    見ない — 大きさだけを比べる。
    """
    try:
        text = _blank_code(_head_sentences(response or ""))
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            if "%" not in sentence and "％" not in sentence:
                continue
            found = _change_rate_in_sentence(sentence)
            if found is not None:
                return found
    except Exception:  # noqa: BLE001 — 検証器は成否の導出を止めない
        logger.warning("misstated_change_rate failed; treating as not detected", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# 自分で宣言した項目数と、直後の一覧の項目数
# ---------------------------------------------------------------------------
#
# 「以下の 3 つのポイントがあります。」と宣言して 4 項目を並べる (または 2 項目で
# 終わる) 応答は、1 つの応答の中で数が食い違っている。利用者の個数指定の違反
# (``violates_output_form``) は依頼の数を見るので、依頼に数が無いと見ない。

#: 宣言の数 (算用数字・全角・漢数字)。
_COUNT_TOKEN = r"(?P<n>[0-9０-９]{1,2}|[一二三四五六七八九十]{1,2})"
#: 数を受ける助数詞 (閉じた類)。
_COUNTER = r"(?:つ|個|点|項目|ステップ|段階|種類)"
#: 「以下の 3 つ」「次の 3 点」「以下 3 つ」。
_DECLARED_AHEAD_RE = re.compile(rf"(?:以下|次)の?\s*{_COUNT_TOKEN}\s*{_COUNTER}")
#: 「3 つのポイントは以下の通りです」「3 点を挙げます：」(宣言の後ろに一覧が続く形)。
_DECLARED_BEHIND_RE = re.compile(
    rf"(?<![0-9０-９]){_COUNT_TOKEN}\s*{_COUNTER}の[^。\n]{{0,30}}?(?:以下|次)の(?:通り|とおり)",
)
#: 宣言の行の終わり (一覧が直後に続く印)。
_DECLARATION_LINE_END_RE = re.compile(r"[。：:]\s*\**\s*$")
#: 番号付きの項目 (一覧・見出し・太字の番号)。
_NUMBERED_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:#{1,6}\s*)?(?:\*\*)?\s*(?P<num>\d{1,2})\s*[.)．、](?:\s|\*\*|$)",
)
#: 記号の箇条書きの項目。
_BULLET_ITEM_RE = re.compile(r"^(?P<indent>[ \t]*)(?:[-*+•]\s+|・\s*)\S")
#: フェンスの開閉。
_FENCE_LINE_RE = re.compile(r"^\s*(?:```|~~~)")
#: 宣言として見る数の範囲。1 は列挙でなく、20 を超える一覧は数え違いと区別しにくい。
_DECLARED_MIN, _DECLARED_MAX = 2, 20


def _declared_count(token: str) -> int | None:
    from backend.free.core.numerals import kanji_number_value

    n = kanji_number_value(token)
    if n is None or not _DECLARED_MIN <= n <= _DECLARED_MAX:
        return None
    return n


def mask_item_numbers(text: str) -> str:
    """番号付き一覧の番号 (行頭の「1.」「2)」) を空白に置き換えたコピー (純粋関数)。

    一覧の番号は量ではない。消すのは **1 から始まり 2 項目以上続く一覧** の番号だけ —
    行頭に「2.」が 1 つあるだけでは一覧と言えず、量として読む (「合計は 707 万円\\n2. 構成比」)。
    文字位置は保つ (呼出側が元の本文と位置を突き合わせる)。
    """
    lines = (text or "").split("\n")
    by_indent: dict[int, list[tuple[int, re.Match[str]]]] = {}
    for i, line in enumerate(lines):
        m = _NUMBERED_ITEM_RE.match(line)
        if m is not None:
            by_indent.setdefault(_indent_width(m.group("indent")), []).append((i, m))
    for items in by_indent.values():
        nums = [int(m.group("num")) for _, m in items]
        if len(nums) < 2 or nums[:2] != [1, 2]:
            continue
        for i, m in items:
            s, e = m.span("num")
            lines[i] = lines[i][:s] + " " * (e - s) + lines[i][e:]
    return "\n".join(lines)


def _indent_width(indent: str) -> int:
    return len(indent.expandtabs(4))


def _count_numbered(lines: list[str]) -> int | None:
    """先頭から続く番号付き一覧 (1, 2, …) の項目数。途中で途切れて後で続くなら ``None``。"""
    expected = 1
    base_indent: int | None = None
    in_fence = False
    seen_after_end: list[int] = []
    ended = False
    for line in lines:
        if _FENCE_LINE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _NUMBERED_ITEM_RE.match(line)
        if m is None:
            continue
        indent = _indent_width(m.group("indent"))
        if base_indent is None:
            base_indent = indent
        if indent != base_indent:
            continue
        num = int(m.group("num"))
        if ended:
            seen_after_end.append(num)
            continue
        if num == expected:
            expected += 1
            continue
        ended = True
        seen_after_end.append(num)
    count = expected - 1
    if base_indent is None or in_fence or count == 0:
        return None  # 1 から始まらない番号は宣言の一覧と言えない
    # 後ろで同じ一覧の続きの番号が出るなら、途中の別要素で切れただけ
    if any(n == count + 1 for n in seen_after_end):
        return None
    return count


def _count_bullets(lines: list[str]) -> int | None:
    """先頭から続く記号の箇条書きの最上位の項目数。一覧が後でまた続くなら ``None``。"""
    base_indent: int | None = None
    count = 0
    ended = False
    for line in lines:
        if not line.strip():
            continue
        if _FENCE_LINE_RE.match(line):
            if base_indent is not None and not ended:
                return None  # 項目の間のコード片。どこまでが一覧か読めない
            continue
        m = _BULLET_ITEM_RE.match(line)
        indent = _indent_width(line[: len(line) - len(line.lstrip())])
        if base_indent is None:
            if m is None:
                return None
            base_indent = indent
            count = 1
            continue
        if ended:
            if m is not None and indent == base_indent:
                return None  # 一覧が段落を挟んで続く
            continue
        if indent > base_indent:
            continue  # 項目の下の入れ子・続きの行
        if m is not None and indent == base_indent:
            count += 1
            continue
        ended = True
    return count if base_indent is not None else None


def _list_count_after(lines: list[str]) -> int | None:
    """宣言の行の直後の一覧の項目数 (一覧が直後に無い・読めなければ ``None``)。"""
    rest = lines
    while rest and not rest[0].strip():
        rest = rest[1:]
    if not rest:
        return None
    first = rest[0]
    if _NUMBERED_ITEM_RE.match(first):
        return _count_numbered(rest)
    if _BULLET_ITEM_RE.match(first):
        return _count_bullets(rest)
    return None


def declared_count_mismatch(response: str) -> str | None:
    """「以下の N つ」と宣言した直後の一覧の項目数が N でなければ理由を返す。

    条件をすべて満たしたときだけ報告する:

    1. 行が「以下の / 次の N <助数詞>」か「N <助数詞>の … 以下の通り」を含み、
       その行が「。」「：」で終わる (一覧が直後に続く宣言)
    2. 宣言の行の直後 (空行は飛ばす) が番号付きの一覧 (1 から) か記号の箇条書き
    3. 一覧の最上位の項目数 M が N と違う。番号の一覧は 1, 2, … と続く数だけを
       数え、途中で途切れて後ろで続きの番号が出る (コード片や説明を挟んだ) なら
       報告しない。箇条書きは項目の間にコード片があるか、段落を挟んで同じ深さの
       項目が続くなら報告しない

    見出しだけの構成 (「### メリット」) や表は数えない。応答の中に宣言が複数
    あれば最初に照合できたものを見る。
    """
    try:
        text = response or ""
        if len(text) > _MAX_SCAN_CHARS:
            return None  # 切り詰めると一覧の途中で数え終わる
        lines = text.splitlines()
        in_fence = False
        for i, line in enumerate(lines):
            if _FENCE_LINE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence or not _DECLARATION_LINE_END_RE.search(line):
                continue
            m = _DECLARED_AHEAD_RE.search(line) or _DECLARED_BEHIND_RE.search(line)
            if m is None:
                continue
            declared = _declared_count(m.group("n"))
            if declared is None:
                continue
            actual = _list_count_after(lines[i + 1:])
            if actual is None:
                continue
            if actual != declared:
                return f"declared {m.group(0).strip()} but listed {actual} item(s)"
            return None
    except Exception:  # noqa: BLE001 — 検証器は成否の導出を止めない
        logger.warning("declared_count_mismatch failed; treating as not detected", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# 使えるツールを「使えない」と述べる
# ---------------------------------------------------------------------------
#
# 2026-10-03 ライブ監査 D08#4: 「全員の残業を10%減らしたら合計は何時間？」に
# 「計算ツールが利用できないため、10%減らした合計時間を算出できませんでした。」。
# ``calculate`` はビルトインとしてモードを問わず常に登録されている
# (``agent.tools.builtin.register_builtin_tools``)。このターンで実行して失敗して
# いないのに「使えない」と述べるのは、システムの事実と食い違う理由で依頼を断った形。
# なお監査の当該ターンは計算機が ``unknown name: sum`` で失敗しており、「使えなかった」
# は事実に近い — 失敗の記録があるターンは数えない (式を組めずにツールを撃たなかった
# 回の「使えない」だけを見る)。
#
# 対象は **常に登録されているツールの名指し** だけ。書込み・追記はモードで使える
# かが変わる (chat では書けない) ので、「保存ツールが利用できない」は正当な断りの
# ことがある — この検証器では見ない。

#: 常に使えるツールの名指し → ツール名。ツール自身の呼び名だけを持つ (閉じた類)。
_ALWAYS_AVAILABLE_TOOL_NAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"計算ツール|計算機能|calculate\s*(?:ツール|tool)|calculator\s+tool", re.IGNORECASE),
     "calculate"),
)
#: 名指しの直後の「使えない」(閉じた文法の形)。名指しとのあいだは助詞と短い修飾 (6 字まで、
#: 文の区切りを含まない) だけ。修飾の語彙 (現在 / 今 …) は列挙しない (不変則 #14(a): 直示語は
#: temporal_deixis.py が持つ)。
_UNAVAILABLE_TAIL_RE = re.compile(
    r"[`」』\s]*(?:が|は|を|も)?\s*[^。\n、,，]{0,6}?\s*"
    r"(?:(?:利用|使用)(?:でき|が?可能で(?:は)?)(?:ない|ません|ず)"
    r"|使え(?:ない|ません|ず)"
    r"|(?:用意|提供)されて(?:いない|いません|おらず)"
    r"|is\s+(?:not\s+available|unavailable)|isn't\s+available)",
    re.IGNORECASE,
)
#: 同じ文の後ろで依頼を断っている形 (「…ため、算出できませんでした」)。断らずに
#: 手計算で答えた応答は、理由が誤っていても答えが正しいことがあるので数えない。
_REFUSAL_AFTER_RE = re.compile(
    r"(?:ため|ので|から|，|、|,)[^。\n]*?"
    r"(?:でき(?:ません|ない|ず)|出せ(?:ません|ない)|cannot|can't|unable)",
    re.IGNORECASE,
)


def false_tool_unavailability(
    response: str, tool_uses: list[dict] | None,
) -> str | None:
    """常に使えるツールを「使えない」と述べていれば理由を返す。

    条件をすべて満たしたときだけ報告する:

    1. 本文が :data:`_ALWAYS_AVAILABLE_TOOL_NAMES` のツールを名指しし、その直後が
       「利用できない / 使えない / 用意されていない」の形 (:data:`_UNAVAILABLE_TAIL_RE`)
    2. 同じ文の後ろで依頼を断っている (「…ため、算出できませんでした」)。手計算で
       答えた応答は理由が誤っていても答えは正しいことがあるので数えない
    3. 名指しが鉤括弧・コード片の引用の中でない
    4. このターンの根拠台帳 (``tool_uses``) に、そのツールを実行して失敗した記録が
       無い — 失敗したなら「使えなかった」は事実の報告に近い
    """
    try:
        text = _blank_code(_head_sentences(response or ""))
        failed = {
            str(u.get("tool") or "") for u in (tool_uses or ())
            if isinstance(u, dict) and not u.get("success")
        }
        for pattern, tool in _ALWAYS_AVAILABLE_TOOL_NAMES:
            if tool in failed:
                continue
            for m in pattern.finditer(text):
                if text.count("「", 0, m.start()) > text.count("」", 0, m.start()):
                    continue
                tail = _UNAVAILABLE_TAIL_RE.match(text, m.end())
                if tail is None:
                    continue
                sentence_end = _SENTENCE_SPLIT_RE.search(text, tail.end())
                rest = text[tail.end():sentence_end.start() if sentence_end else len(text)]
                if _REFUSAL_AFTER_RE.match(rest) is None:
                    continue
                return f"said {m.group(0)} is unavailable but {tool} is always registered"
    except Exception:  # noqa: BLE001 — 検証器は成否の導出を止めない
        logger.warning("false_tool_unavailability failed; treating as not detected", exc_info=True)
    return None
