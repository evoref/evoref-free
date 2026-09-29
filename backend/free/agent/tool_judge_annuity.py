"""元利均等の月返済額を決定論で組む層 5.8 の部品 (docs/f_03 §3.1、c_17 §3.13)。

2026-09-29 回帰確認 (R27_loan_correction #3): 35 年・金利 1.5% → 1.2% の訂正で、9B の
式合成は理由を添えて作り直させても正しい元利均等の式を組めず、計算なしの概算
(9 万 4,770 円 / 8 万 6,130 円、正しくは 87,510.69 円) になった。元本・年率・年数を
**user の発言**から取り、式はコードが組む。組むのは元利均等の毎月返済額 1 つだけで、
ほかの公式 (元金均等・積立・年金現価 …) へは広げない (公式ごとの型を足す坂道は
不変則 #12 / #14 が戒める「語形を足す」と同じ)。

- 「元利均等の月返済額を尋ねているか」は意味の分類なので判定点 ``annuity_payment_request``
  に載せる (字句段だけのカスケード、``decision.jsonl`` に根拠が残る)。誤発火は別の問いに
  「厳密な計算結果」を付けて返す — 開示付きの概算より悪い — ので、方式の明示を要し、
  決定論で組んではいけない形 (ボーナス・変動・固定期間・繰上げ …) は撃たない。
- 値の取り出し (元本・年率・年数) は構造の抽出で、既存の抽出器を使う (1 実装、#14(a))。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from backend.free.agent.tool_judge_grounding import (
    _PERIOD_YEARS_RE,
    _annual_percents,
    _myriad_pairs,
)
from backend.free.core.correction_target import contrast_pairs
from backend.free.core.intent_vocab import is_plain_statement
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "annuity_payment_request"

#: 元利均等の毎月返済額を尋ねている。
ANNUITY_LABEL = "annuity_monthly_payment"

#: 返済方式の明示 (これが無ければ棄権する — 住宅ローンが元利均等とは限らない)。
_METHOD_RE = re.compile(r"元利均等")
#: 決定論の 1 公式で組んではいけない形。期間・率・元本のどれかが 1 つに決まらないか、
#: 公式が違う。
_EXCLUDED_RE = re.compile(
    r"元金均等|ボーナス|変動|段階|繰り?上げ?|据え?置|頭金|借り?換|残債|残高|手数料|月利"
    r"|\d+\s*年(?:間)?固定"
)
#: 月あたりの返済額を問う形 (「総返済額」は含めない)。
_MONTHLY_PAYMENT_RE = re.compile(
    r"(?:毎月|月々|月額|ひと月|1\s*[かヶケカ]月)\s*の?\s*(?:返済|支払)|月の?返済額"
)


class _AnnuityRequestRule:
    """字句段 (``Predicate`` プロトコル)。根拠を形ごとに書き分けるため ``Verdict`` を直接返す。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        query = text or ""
        user_text = str((ctx or {}).get("user_text") or query)
        if not _MONTHLY_PAYMENT_RE.search(query) or is_plain_statement(query):
            return self._verdict(NEGATIVE_LABEL, "skip", "not_monthly_payment")
        if _EXCLUDED_RE.search(user_text):
            return self._verdict(NEGATIVE_LABEL, "skip", "excluded_form")
        if not _METHOD_RE.search(user_text):
            return self._verdict(None, "abstain", "method_unstated")
        return self._verdict(ANNUITY_LABEL, "fire", "annuity_monthly_payment")

    def _verdict(self, value: str | None, band: str, evidence: str) -> Verdict:
        return Verdict(
            value=value, score=1.0 if band == "fire" else 0.0, band=band,  # type: ignore[arg-type]
            evidence=evidence, predicate=self.name, stage="lexical",
        )


_RULE = _AnnuityRequestRule()

#: プロセス共通の判定点。層 5.8 が候補 (月の返済額を問う形) のあるターンで 1 回だけ引く。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="complement",
        candidates=[ANNUITY_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def annuity_request_verdict(query: str, user_text: str) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する。ターンに 1 回)。"""
    return predicate.evaluate(query or "", {"user_text": user_text or ""})


def annuity_request_rule(query: str, user_text: str) -> Verdict:
    """字句段だけを **記録せずに** 評価する (純粋関数)。"""
    return _RULE.evaluate(query or "", {"user_text": user_text or ""})


def _field_values(text: str) -> dict[str, set[str]]:
    """1 つの発言から (元本, 年率, 年数) の候補を取る (純粋関数)。"""
    return {
        "principal": {expanded for _coef, expanded in _myriad_pairs(text)},
        "rate": set(_annual_percents(text)),
        "years": {y for y in _PERIOD_YEARS_RE.findall(text) if int(y) > 0},
    }


def annuity_slots(user_utterances: Sequence[str]) -> tuple[str, str, int] | None:
    """user の発言 (古い順) から (元本 (円), 年率 (%), 月数) を取る。1 つに決まらなければ ``None``。

    各欄の値は発言ごとに足し、「X ではなく Y」の旧値 X はその欄から外す
    (:func:`contrast_pairs`、訂正の宛先は X の単位 — 万円 / % / 年 — で決まる)。
    走査の後にどれかの欄が 1 値でなければ棄権する (「1.2%だったら？」の仮定形は旧値を
    外さないので 2 値残る)。アシスタントの発言は読まない (前の回答の「9 万 1,855」を元本に
    拾わない)。
    """
    fields: dict[str, set[str]] = {"principal": set(), "rate": set(), "years": set()}
    for utterance in user_utterances:
        for key, values in _field_values(utterance or "").items():
            fields[key] |= values
        for old, _new in contrast_pairs(utterance or ""):
            for key, values in _field_values(old).items():
                fields[key] -= values
    if any(len(values) != 1 for values in fields.values()):
        return None
    (principal,), (rate,), (years,) = fields["principal"], fields["rate"], fields["years"]
    return principal, rate, int(years) * 12


def build_annuity_expression(principal: str, rate_percent: str, months: int) -> str:
    """元利均等の毎月返済額の式 ``P × r / (1 - (1 + r) ** -n)`` (r = 年率/12)。"""
    try:
        annual = Decimal(rate_percent) / Decimal(100)
    except InvalidOperation as exc:  # 抽出器が数字だけを返すので起こらないが、式へ流さない
        raise ValueError(f"not a rate: {rate_percent!r}") from exc
    rate = format(annual.normalize(), "f")
    return f"{principal} * ({rate} / 12) / (1 - (1 + {rate} / 12) ** -{months})"


__all__ = [
    "ANNUITY_LABEL",
    "PREDICATE_NAME",
    "annuity_request_rule",
    "annuity_request_verdict",
    "annuity_slots",
    "bind_debug_logger",
    "build_annuity_expression",
    "predicate",
]
