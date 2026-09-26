"""応答本文に書かれた算術の主張を決定論で検算する。

few-shot の正例採用ゲート (EvorefLearn) と、ターン成否の決定論判定
(``FeedbackCollector._derive_turn_outcome``、EvorefLoop) の共用。pillar を
またぐので横断基盤 ``backend/free/core/`` に置く (純粋関数・外部依存なし)。

**補助タスクにこの判定を任せてはいけない**:
稼働中の aux (Qwen3.5-4B) に既知の誤答を採点させた実測 (2026-07-31) では、
「42.195 ÷ 1.609 ≈ 26.195」(正しくは 26.2244) に対して
「計算式と結果が正確で、手本として非常に品質が高い」と **満点 1.0** を付けた。
小型モデルは自分が検算できない算術を「正しい」と評価する。

逆に、この種の誤りは応答本文に式が明記されていれば決定論で確実に捕まえられる。
実データ 447 件に対して式 15 個を検算し、**誤答 1 件を検出・誤検出 0** だった。

判定の勘所は許容誤差で、**主張値の表記桁数に連動**させる必要がある。相対誤差の
固定値 (0.5%) では上の実例 (相対誤差 0.11%) を見逃す。3 桁で書かれた値は 3 桁の
精度で主張しているとみなす。
"""

from __future__ import annotations

import re

#: ``A <op> B (= | ≈ | →) C`` 形式の主張。
#:
#: 演算子の前後に **空白を要求** する。要求しないと「シャッターを 1/125 秒」の
#: ような単位表記を除算として拾う (実測で誤検出した)。
_NUM = r"(-?[\d,]+(?:\.\d+)?)"
_CLAIM_RE = re.compile(
    r"(?<![\d.])" + _NUM + r"\s+([÷/×＊\*\+\-−])\s+" + _NUM
    + r"\s*(?:=|＝|≈|≒|→)\s*(?:約\s*)?" + _NUM + r"(?![\d.])",
)

#: 直前が「数値 + 演算子」= 連鎖式の途中。切り出すと誤検算になる
#: (「4800 × 0.75 × 1.1 = 3960」の後ろ 2 項だけを取ると不一致に見える)。
_CHAIN_LEFT_RE = re.compile(r"[\d.]\s*[÷/×＊\*\+\-−]\s*$")

#: 桁数由来の丸め許容に掛ける係数。1.0 だと最終桁の丸めで偽陽性が出る。
_ROUNDING_SLACK = 1.5


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def _tolerance(claim_text: str, expected: float) -> float:
    """主張値の表記桁数から許容誤差を決める (純粋関数)。

    「約 26」なら ±0.75、「26.195」なら ±0.00075 程度。桁数を無視した相対誤差
    では、細かく書かれた値の誤りを取りこぼす。
    """
    decimals = len(claim_text.split(".")[1]) if "." in claim_text else 0
    half_ulp = 0.5 * (10.0 ** -decimals)
    return half_ulp * _ROUNDING_SLACK + abs(expected) * 1e-9


def _evaluate(a: float, op: str, b: float) -> float | None:
    if op in ("÷", "/"):
        return None if b == 0 else a / b
    if op in ("×", "＊", "*"):
        return a * b
    if op == "+":
        return a + b
    if op in ("-", "−"):
        return a - b
    return None


def find_arithmetic_contradictions(text: str) -> list[str]:
    """本文中の算術主張のうち、計算が合わないものを列挙する (純粋関数)。

    式 (``A op B = C``) と、箇条書きの内訳 (:func:`find_breakdown_contradictions`)
    の 2 形を見る。ターン成否と few-shot の足切りが同じ入口を呼ぶので、形を
    足すときはここに足す (片方だけ直る状態を作らない)。

    Returns:
        ``"42.195 ÷ 1.609 ≈ 26.195 (正しくは 26.2244)"`` 形式の説明のリスト。
        検算可能な式が無い / すべて一致する場合は空リスト。
    """
    if not text:
        return []
    found: list[str] = find_breakdown_contradictions(text)
    for m in _CLAIM_RE.finditer(text):
        if _CHAIN_LEFT_RE.search(text[: m.start()]):
            continue
        claim_text = m.group(4)
        expected = _evaluate(_to_float(m.group(1)), m.group(2), _to_float(m.group(3)))
        if expected is None:
            continue
        claimed = _to_float(claim_text)
        if abs(expected - claimed) > _tolerance(claim_text, expected):
            found.append(f"{m.group(0)} (正しくは {expected:.4f})")
    return found


# ---------------------------------------------------------------------------
# 冒頭の結論 vs 本文の計算結果
# ---------------------------------------------------------------------------
#
# ``find_arithmetic_contradictions`` は **1 つの式の中** の不整合しか見ない。
# ところが実インシデント (2026-09-06 監査 F-03) は式ごとには正しく、
# **冒頭で述べた答えだけが本文の計算とまったく別の数** だった:
#
#     借り換え前後の…総支払額の差は、**1,096万4,771円** です。
#     ...
#     差額: 38,168,770.9…円 - 35,373,644.9…円 = **2,795,126円**
#
# 冒頭の値は本文のどこにも現れず、計算過程は一貫して 2,795,126 を出している。
# 読み手が最初に受け取るのは冒頭の値なので、実害は誤った式より大きい。にも
# かかわらず ``turn_outcome`` は success で、few-shot の手本候補にもなった。
#
# 「冒頭で答えとして提示した値が、本文の計算結果のどれとも一致せず、しかも
# 本文の他のどこにも現れない」— これは数えるだけで決まり、推定を含まない。

#: 漢数字単位付きの数値 (「1,096万4,771」「3,200万」「1億2,000万」)。
#: 桁区切りだけを見る素の数値抽出だと ``1,096万4,771`` が ``1,096`` と
#: ``4,771`` の 2 値に割れ、冒頭の主張値を復元できない。万の前の小数
#: (「1.5万」) も受ける — 受けないと 15000 が 1.5 と読まれていた。
_JA_NUMBER_RE = re.compile(
    r"(?<![\d.,])(?=\d)"
    r"(?:(?P<oku>\d[\d,]*)\s*億\s*)?"
    r"(?:(?P<man>\d[\d,]*(?:\.\d+)?)\s*万\s*)?"
    r"(?P<base>\d[\d,]*(?:\.\d+)?)?",
)

#: 「= 2,795,126」形式の計算結果 (右辺)。本文が計算を **実際に行っている**
#: ことの証拠として要求する。式が 1 つも無い応答は照合対象にしない。
#:
#: 右辺は強調表記されることが多い (``= **2,795,126円**``)。装飾を挟んだだけで
#: 「計算していない」と判定すると、実インシデントと同じ形の応答を素通しする。
_EQUATION_RHS_RE = re.compile(r"[=＝]\s*(?:\*{1,3}|`|__?)?\s*(?:約\s*)?(?=\d)")

#: 冒頭とみなす範囲。最初の空行まで、かつこの文字数まで。
_LEAD_CHARS = 400

#: 強調表記 (`**...**`) の中身。
_BOLD_RE = re.compile(r"\*\*([^*\n]{1,60})\*\*")

#: 数値の **直後** に続く断定 (「127,229 円です」「70 営業日となります」)。
#:
#: 数値と述語のあいだに許すのは単位相当の短い語だけ。間隔を広く取ると、
#: 数字を含む散文が丸ごと「結論の断定」に化ける — 実測では
#: ``E0133: use of static mut variable`）になります`` と
#: ``127229.236318）が…単なる数値です`` の 2 件を誤って結論として拾った。
_ASSERTION_TAIL_RE = re.compile(
    r"^\s*[^\s。、，,\n]{0,8}?(?:です|となります|になります|でした)",
)

#: インラインコード。エラーコードや識別子の数字が結論に化けるのを防ぐため、
#: 冒頭の走査前に落とす。
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

#: 概算表現。許容誤差を桁ではなく相対値に切り替える。
_APPROX_RE = re.compile(r"約|およそ|ほぼ|概ね|おおよそ")
_APPROX_REL_TOLERANCE = 0.05

#: 結論値として扱う最小桁数 (桁区切りを除いた数字数)。
#:
#: 「**3 案** です」のような個数・順序の小さな数まで見ると、無関係な式の
#: 右辺と食い違うだけで矛盾を報告してしまう。金額・件数の計算結果が偶然
#: 一致する確率が無視できる桁数に絞る。
_MIN_CONCLUSION_DIGITS = 4


def _parse_ja_number(match: re.Match) -> float | None:
    """``_JA_NUMBER_RE`` のマッチを数値へ (単位が 1 つも無ければ素の数値)。"""
    oku, man, base = match.group("oku"), match.group("man"), match.group("base")
    if oku is None and man is None and base is None:
        return None
    total = 0.0
    if oku is not None:
        total += _to_float(oku) * 100_000_000
    if man is not None:
        total += _to_float(man) * 10_000
    if base is not None:
        total += _to_float(base)
    return total


def _iter_ja_numbers(text: str) -> list[tuple[int, int, float]]:
    """本文中の数値を ``(start, end, value)`` で列挙する (純粋関数)。"""
    out: list[tuple[int, int, float]] = []
    for m in _JA_NUMBER_RE.finditer(text or ""):
        if not m.group(0):
            continue
        value = _parse_ja_number(m)
        if value is not None:
            out.append((m.start(), m.end(), value))
    return out


def _digit_count(text: str) -> int:
    return sum(1 for ch in text if ch.isdigit())


def _conclusion_tolerance(literal: str, value: float) -> float:
    """結論値の照合許容。「約」付きは相対、それ以外は表記桁数由来。"""
    if _APPROX_RE.search(literal):
        return abs(value) * _APPROX_REL_TOLERANCE
    return _tolerance(literal.replace(",", ""), value)


def _find_lead_conclusion(text: str) -> tuple[int, int, float, str] | None:
    """冒頭で答えとして提示された値を返す ``(start, end, value, literal)``。"""
    raw = (text or "").split("\n\n", 1)[0][:_LEAD_CHARS]
    if not raw:
        return None
    # コード片の数字 (エラーコード / 識別子) は結論ではない。位置を保つため
    # 同じ長さの空白で潰す。
    lead = _INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), raw)
    # 強調表記が最優先 (モデルは結論を太字にする)。
    for m in _BOLD_RE.finditer(lead):
        nums = _iter_ja_numbers(m.group(1))
        if nums:
            start, end, value = nums[0]
            return (
                m.start(1) + start, m.start(1) + end, value,
                m.group(1),
            )
    # 強調が無ければ「<数値><単位>です」型の断定を探す。
    for start, end, value in _iter_ja_numbers(lead):
        tail = _ASSERTION_TAIL_RE.match(lead[end:])
        if tail is not None:
            return start, end, value, lead[start:end + tail.end()]
    return None


def find_conclusion_contradiction(text: str) -> str | None:
    """冒頭の結論が本文の計算結果と食い違っていれば理由を返す (純粋関数)。

    条件をすべて満たしたときだけ報告する (誤検出は無駄な再生成を生む):

    1. 冒頭に **答えとして提示された数値** がある (強調 or 断定の文末)
    2. その値の桁数が :data:`_MIN_CONCLUSION_DIGITS` 以上 (個数・順序を除外)
    3. 本文に計算式 (``= 値``) が 1 つ以上ある
    4. 冒頭の値が **本文の他のどこにも現れない**
    """
    body = text or ""
    lead = _find_lead_conclusion(body)
    if lead is None:
        return None
    start, end, value, literal = lead
    if _digit_count(literal) < _MIN_CONCLUSION_DIGITS:
        return None
    if _EQUATION_RHS_RE.search(body) is None:
        return None

    tolerance = _conclusion_tolerance(literal, value)
    others = [
        v for (s, e, v) in _iter_ja_numbers(body)
        if not (s >= start and e <= end)
    ]
    if not others:
        return None
    if any(abs(v - value) <= tolerance for v in others):
        return None
    return (
        f"lead states {literal.strip()} as the answer but the body never "
        f"reaches that value (computed values include "
        f"{', '.join(f'{v:g}' for v in others[:3])})"
    )


# ---------------------------------------------------------------------------
# 箇条書きの内訳 vs 親項目の合計
# ---------------------------------------------------------------------------
#
# 上の 2 判定はどちらも **式** (``A op B = C`` / ``= 値``) を手掛かりにする。
# ところが家計・時間配分・見積もりの応答は、式を書かずに箇条書きの入れ子で
# 合計と内訳を並べるのが普通で、そこでの誤りを 1 件も捕まえていなかった
# (2026-09-21 ライブ監査 C07 turn2、``turn_outcome`` は success のまま):
#
#     *   **固定費**: 11万4000円
#         *   家賃: 9万円
#         *   光熱費: 1.5万円
#         *   通信費: 1.2万円
#
# 内訳の和は 11万7000円。判定は構造と数値だけで決まり、語彙を持たない。

#: 箇条書きの 1 行 (``-`` / ``*`` / ``+`` / ``・`` / ``1.`` / ``1)``)。
_LIST_ITEM_RE = re.compile(r"^(?P<indent>[ \t]*)(?:[-*+・•]|\d{1,2}[.)])\s+(?P<body>\S.*)$")

#: 「<ラベル>: <値>」の区切り。ラベル側は短い語に限る (文を拾わない)。
_LABEL_VALUE_RE = re.compile(r"^(?P<label>[^:：\n]{1,30}?)\s*[:：]\s*(?P<rest>.+)$")

#: 値の直後の単位 (「円」「時間」「kg」)。数値の続き・括弧・区切りは単位でない。
_UNIT_RE = re.compile(r"\s*(?P<unit>[^\s\d（(、。,，/／:：)）〜~\-+×]{1,4})")

#: 強調記号。値が ``**11万円**`` のように装飾されていても同じに読む。
_EMPHASIS_RE = re.compile(r"\*{1,3}|__?")

#: 足し上げの意味を持たない単位。比率の内訳は親の値の和にならない。
_NON_ADDITIVE_UNITS = frozenset({"%", "％", "倍", "割"})


def _literal_scale(match: re.Match) -> float:
    """``_JA_NUMBER_RE`` のマッチが表す値の最小桁 (「11.4万」→ 1000)。"""
    for group, unit in (("base", 1.0), ("man", 10_000.0), ("oku", 100_000_000.0)):
        literal = match.group(group)
        if literal is not None:
            decimals = len(literal.split(".")[1]) if "." in literal else 0
            return unit * 10.0 ** -decimals
    return 1.0


def _parse_amount_item(body: str) -> tuple[str, float, str, float, bool] | None:
    """「<ラベル>: <数値><単位>」の項目を ``(label, value, unit, scale, approx)`` へ。

    値の後ろに別の数値が続く項目 (「1.5h × 5日」「3〜5万円」「(年 120万円)」) は
    内訳の 1 項として読めないので ``None``。
    """
    m = _LABEL_VALUE_RE.match(_EMPHASIS_RE.sub("", body).strip())
    if m is None:
        return None
    rest = m.group("rest").strip()
    approx = _APPROX_RE.match(rest)
    if approx is not None:
        rest = rest[approx.end():].lstrip()
    num = _JA_NUMBER_RE.match(rest)
    if num is None or not num.group(0):
        return None
    value = _parse_ja_number(num)
    unit = _UNIT_RE.match(rest, num.end())
    if value is None or unit is None or unit.group("unit") in _NON_ADDITIVE_UNITS:
        return None
    if re.search(r"\d", rest[unit.end():]):
        return None
    return m.group("label").strip(), value, unit.group("unit"), _literal_scale(num), approx is not None


def find_breakdown_contradictions(text: str) -> list[str]:
    """箇条書きの内訳の和が親項目の値を **超える** ものを列挙する (純粋関数)。

    親項目 (「固定費: 11万4000円」) の直下の子項目がすべて同じ単位の
    「<ラベル>: <値>」で 2 つ以上あるときだけ照合する。

    **超過だけを報告する**。和が足りないのは「主なものだけ挙げた」内訳で
    普通に起こるが、部分の和が全体を超えることは内訳である限り起こらない。
    許容誤差は親の値の表記桁 (「12万円」なら ±0.75万) で、子の値は問いに
    与えられた入力なので丸めを見込まない。「約」がどこかに付けば相対 5%。
    """
    lines = [line.expandtabs(4) for line in (text or "").splitlines()]
    found: list[str] = []
    for i, line in enumerate(lines):
        head = _LIST_ITEM_RE.match(line)
        if head is None:
            continue
        parent = _parse_amount_item(head.group("body"))
        if parent is None:
            continue
        parent_indent = len(head.group("indent"))
        child_indent: int | None = None
        children: list[tuple[str, float, str, float, bool] | None] = []
        for below in lines[i + 1:]:
            if not below.strip():
                continue
            indent = len(below) - len(below.lstrip())
            if indent <= parent_indent:
                break
            item = _LIST_ITEM_RE.match(below)
            if item is None:
                continue
            if child_indent is None:
                child_indent = indent
            if indent == child_indent:
                children.append(_parse_amount_item(item.group("body")))
        if len(children) < 2 or any(c is None for c in children):
            continue
        label, value, unit, scale, approx = parent
        parts = [c for c in children if c is not None]
        if any(c[2] != unit for c in parts):
            continue
        total = sum(c[1] for c in parts)
        if approx or any(c[4] for c in parts):
            tolerance = abs(value) * _APPROX_REL_TOLERANCE
        else:
            tolerance = 0.5 * scale * _ROUNDING_SLACK + abs(value) * 1e-9
        if total - value > tolerance:
            found.append(
                f"breakdown of {label} ({value:g}{unit}) sums to {total:g}{unit}"
            )
    return found
