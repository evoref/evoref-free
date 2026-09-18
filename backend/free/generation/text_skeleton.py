"""TextSkeleton — 正規表現による TEXT 版の文書状態抽出

設計書 f_08_long_form_generation.md §3.3.1 準拠。
``CodeSkeleton`` (code_skeleton.py) の TEXT 版。unit 間で持ち回る固定サイズの
文書状態を、生成済み unit の本文から LLM 不要の決定論で抽出する。

ゲート付き更新 (:meth:`TextSkeleton.update`) が本モジュールの核心:
新しい項目は unit 本文に **逐語で出現するものだけ**受理し、``pinned_values`` は
同じラベルに別の値が来ても上書きしない (最初の値が勝ち、:attr:`contradictions`
に記録する)。ゲート無しの再帰は漂流し、矛盾検出そのものが無効化される
(f_08 §8 禁則 9)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.free.core.response_dates import response_dates
from backend.free.core.text_quality import labeled_numeric_claims
from backend.utils import estimate_tokens, utc_now_dt

if TYPE_CHECKING:
    from backend.free.generation.models import GenerationPlan

__all__ = ["PinnedValue", "TextSkeleton", "extract_pinned_candidates"]


# ── 決定論抽出パターン ──

#: 和文の月日 (年任意)。verbatim 値としてそのまま prompt へ載せる。
_JP_DATE_VERBATIM_RE = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日")

#: ISO 形式の日付。
_ISO_DATE_VERBATIM_RE = re.compile(r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)")

_URL_RE = re.compile(r"https?://[^\s)\]}>、。]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")
_PHONE_RE = re.compile(r"(?<!\d)0\d{1,4}-\d{1,4}-\d{4}(?!\d)")

#: ラベル無しの「N円/N名/N人/N件」。ラベル付きの数量は
#: :func:`labeled_numeric_claims` (core.text_quality) が別途拾う。
_BARE_QUANTITY_RE = re.compile(r"[0-9０-９,，]+(?:万|億)?(?:円|名|人|件)")

_BOLD_TERM_RE = re.compile(r"\*\*([^*\n]{1,20})\*\*")
_ABBR_PAREN_RE = re.compile(r"([^\s（）()、。]{1,20})（略）")
_ASCII_ABBR_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Z]{2,8}(?![A-Za-z0-9_])")

_FORWARD_REF_RE = re.compile(
    r"後述|次章|次節|第[0-9０-９一二三四五六七八九十百]+章|§[0-9０-９]+",
)

_SENTENCE_SPLIT_RE = re.compile(r"[。！？\n]+")
_POLITE_TAIL_RE = re.compile(r"(?:です|ます)$")
_PLAIN_TAIL_RE = re.compile(r"(?:だ|である)$")


def _extract_dates(text: str) -> list[str]:
    """本文に verbatim で出現する日付表記を出現順に返す (純粋関数)。

    :func:`response_dates` でカレンダー妥当性を確認した日付だけを対象にし、
    その日付が本文中でどう書かれているか (和暦月日 / ISO / スラッシュ等) を
    逆引きして verbatim な文字列を採る。年を書いていない月日 (「4月3日」) は
    ``response_dates`` が ``year_hint`` 無しでは読み飛ばすため、暦妥当性の
    確認だけに使う現在年を渡す (表示する値は常に text 中の逐語部分文字列で、
    年を補って表示することはない)。
    """
    out: list[str] = []
    seen: set[str] = set()
    for d in response_dates(text, year_hint=utc_now_dt().year):
        candidates = (
            f"{d.year}年{d.month}月{d.day}日",
            f"{d.month}月{d.day}日",
            d.isoformat(),
            f"{d.month:02d}-{d.day:02d}",
            f"{d.month}/{d.day}",
        )
        for cand in candidates:
            if cand in text and cand not in seen:
                seen.add(cand)
                out.append(cand)
                break
    return out


def extract_pinned_candidates(text: str) -> list[tuple[str, str]]:
    """本文から ``(label, value)`` の固有値候補を出現順に近い形で返す (純粋関数)。

    すべて ``text`` からの正規表現マッチ (または ``response_dates`` が確認した
    日付の verbatim 表記) なので、返り値は常に本文中に逐語で出現する。
    ``labeled_numeric_claims`` はラベルごとに値の集合を返すため、同一ラベルに
    複数値があってもラベルの並びで決定論的に選ぶ (集合の反復順は信用しない)。
    """
    candidates: list[tuple[str, str]] = []
    for value in _extract_dates(text):
        candidates.append(("date", value))
    claims = labeled_numeric_claims(text)
    for label in sorted(claims):
        value = sorted(claims[label])[0]
        # 1 桁の値は「4月3日」の「4」のように日付・数量表記の先頭桁が
        # 数値言明として誤って切り出されたものが大半 (regex は非数字で
        # 止まるため)。2 桁以上だけを固有値候補として扱う。
        if len(value) >= 2:
            candidates.append((label, value))
    for m in _BARE_QUANTITY_RE.finditer(text):
        value = m.group(0)
        candidates.append((value[-1], value))
    for m in _URL_RE.finditer(text):
        candidates.append(("url", m.group(0)))
    for m in _EMAIL_RE.finditer(text):
        candidates.append(("email", m.group(0)))
    for m in _PHONE_RE.finditer(text):
        candidates.append(("phone", m.group(0)))
    return candidates


def _extract_glossary_terms(text: str) -> list[str]:
    """本文中で定義された用語を出現順に返す (純粋関数)。"""
    terms: list[str] = []
    for m in _BOLD_TERM_RE.finditer(text):
        term = m.group(1).strip()
        if term:
            terms.append(term)
    for m in _ABBR_PAREN_RE.finditer(text):
        term = m.group(1).strip()
        if term:
            terms.append(term)
    for m in _ASCII_ABBR_RE.finditer(text):
        terms.append(m.group(0))
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def _extract_commitments(text: str) -> list[str]:
    """前方参照の逐語句を出現順・重複無しで返す (純粋関数)。"""
    seen: set[str] = set()
    out: list[str] = []
    for m in _FORWARD_REF_RE.finditer(text):
        phrase = m.group(0)
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    return out


def _count_tone(text: str) -> tuple[int, int]:
    """文末の です・ます / だ・である を数える (純粋関数)。"""
    polite = plain = 0
    for raw in _SENTENCE_SPLIT_RE.split(text):
        s = raw.strip()
        if not s:
            continue
        if _POLITE_TAIL_RE.search(s):
            polite += 1
        elif _PLAIN_TAIL_RE.search(s):
            plain += 1
    return polite, plain


@dataclass(frozen=True, slots=True)
class PinnedValue:
    """本文で確定した固有値 1 件。"""

    label: str
    value: str
    unit_idx: int


@dataclass
class TextSkeleton:
    """TEXT 生成の unit 間で持ち回る固定サイズの文書状態。"""

    #: 計画の見出し → {"planned", "generated"}。
    outline_status: dict[str, str] = field(default_factory=dict)
    #: 確定済み固有値 (同一 label は最初の値のみ保持)。
    pinned_values: list[PinnedValue] = field(default_factory=list)
    #: 本文中で定義された用語 → 初出 unit_idx。
    glossary: dict[str, int] = field(default_factory=dict)
    #: 前方参照 (unit_idx, 逐語句)。
    commitments: list[tuple[int, str]] = field(default_factory=list)
    #: "polite" (です・ます) | "plain" (だ・である) | "" (未確定)。
    tone: str = ""
    #: 上書きされず記録に回った矛盾 (unit_idx, label, 既存値, 新値)。
    contradictions: list[tuple[int, str, str, str]] = field(default_factory=list)

    # tone の多数決に使う内部カウンタ (公開契約外)。
    _polite_count: int = field(default=0, repr=False, compare=False)
    _plain_count: int = field(default=0, repr=False, compare=False)

    @classmethod
    def from_plan(cls, plan: "GenerationPlan") -> "TextSkeleton":
        """計画の見出しを ``planned`` で初期化する。"""
        outline: dict[str, str] = {}
        for unit in getattr(plan, "units", None) or []:
            heading = getattr(unit, "heading", "") or ""
            if heading and heading not in outline:
                outline[heading] = "planned"
        return cls(outline_status=outline)

    def known_values(self) -> set[str]:
        """既知の固有値・用語の集合 (捏造判定点の ``known`` に使う)。"""
        return {pv.value for pv in self.pinned_values} | set(self.glossary)

    def update(self, unit_idx: int, heading: str, text: str) -> None:
        """unit 確定後に状態を更新する (ゲート付き)。

        新しい項目は ``text`` に逐語で出現するものだけを受理する (抽出関数が
        すべて ``text`` からの正規表現マッチであるため自動的に満たされる)。
        ``pinned_values`` は同じ label に別の値が来ても **上書きしない**
        (最初の値が勝ち、:attr:`contradictions` に積む)。
        """
        if not text:
            return
        if heading and heading in self.outline_status:
            self.outline_status[heading] = "generated"

        for label, value in extract_pinned_candidates(text):
            existing = next(
                (pv for pv in self.pinned_values if pv.label == label), None,
            )
            if existing is None:
                self.pinned_values.append(PinnedValue(label, value, unit_idx))
            elif existing.value != value:
                self.contradictions.append(
                    (unit_idx, label, existing.value, value),
                )

        for term in _extract_glossary_terms(text):
            if term not in self.glossary:
                self.glossary[term] = unit_idx

        for phrase in _extract_commitments(text):
            entry = (unit_idx, phrase)
            if entry not in self.commitments:
                self.commitments.append(entry)

        polite, plain = _count_tone(text)
        self._polite_count += polite
        self._plain_count += plain
        if self._polite_count > self._plain_count:
            self.tone = "polite"
        elif self._plain_count > self._polite_count:
            self.tone = "plain"

    def to_prompt(self, budget_tokens: int) -> str:
        """予算内に収まる状態文字列を返す。

        優先順: ``pinned_values`` > ``outline_status`` > ``glossary`` >
        ``commitments`` > ``tone``。予算超過時はそこで打ち切る
        (:class:`~backend.free.generation.code_skeleton.CodeSkeleton.to_prompt`
        と同じ方式)。同じ入力からは常に同じ出力 (決定論)。
        """
        sections: list[tuple[str, list[str]]] = []
        if self.pinned_values:
            sections.append(("# 確定値", [
                f"- {pv.label}: {pv.value} (§{pv.unit_idx + 1})"
                for pv in self.pinned_values
            ]))
        if self.outline_status:
            sections.append(("# 構成", [
                f"- {heading}: {status}"
                for heading, status in self.outline_status.items()
            ]))
        if self.glossary:
            sections.append(("# 用語", [
                f"- {term} (§{idx + 1})" for term, idx in self.glossary.items()
            ]))
        if self.commitments:
            sections.append(("# 参照済み言及", [
                f"- (§{idx + 1}) {phrase}" for idx, phrase in self.commitments
            ]))
        if self.tone:
            tone_label = "です・ます調" if self.tone == "polite" else "だ・である調"
            sections.append(("# 文体", [f"- {tone_label}"]))

        result_parts: list[str] = []
        remaining = budget_tokens
        for header, items in sections:
            if not items or remaining <= 0:
                continue
            block = header + "\n" + "\n".join(items)
            block_tokens = estimate_tokens(block)
            if block_tokens <= remaining:
                result_parts.append(block)
                remaining -= block_tokens
                continue
            partial_lines = [header]
            partial_tokens = estimate_tokens(header)
            for item in items:
                item_tokens = estimate_tokens(item)
                if partial_tokens + item_tokens + 1 <= remaining:
                    partial_lines.append(item)
                    partial_tokens += item_tokens + 1
                else:
                    break
            if len(partial_lines) > 1:
                result_parts.append("\n".join(partial_lines))
                remaining -= partial_tokens
            break

        return "\n\n".join(result_parts)
