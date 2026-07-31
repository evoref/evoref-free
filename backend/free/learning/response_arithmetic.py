"""応答本文に書かれた算術の主張を決定論で検算する。

few-shot の正例採用ゲートに使う。**アシストモデルにこの判定を任せてはいけない**:
稼働中の assist (Qwen3.5-4B) に既知の誤答を採点させた実測 (2026-07-31) では、
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

    Returns:
        ``"42.195 ÷ 1.609 ≈ 26.195 (正しくは 26.2244)"`` 形式の説明のリスト。
        検算可能な式が無い / すべて一致する場合は空リスト。
    """
    if not text:
        return []
    found: list[str] = []
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


def has_arithmetic_contradiction(text: str) -> bool:
    """本文中に計算の合わない算術主張が 1 つでもあるか。"""
    return bool(find_arithmetic_contradictions(text))
