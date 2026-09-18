"""判定点 ``text_unit_fabrication`` — 長文 TEXT unit の固有値の矛盾 / 捏造候補。

設計書 f_08_long_form_generation.md §3.4、c_17_predicate_registry.md §3.7 準拠。

長文 TEXT 生成の unit 後段で、unit 本文から日付 / 金額・数量 / URL / メール /
電話を決定論で抜き (:mod:`backend.free.generation.text_skeleton`)、
``TextSkeleton.pinned_values`` (前 unit までに確定した値) と突き合わせる:

- 同じラベルに別の値が来た → ``fire`` (``contradiction``)。既存の改稿経路
  (``max_revisions`` 内) で直すための :class:`ReviewIssue` を組む。
- ブリーフ / 計画 / 状態 / unit_rag のどこにも無い固有値 → ``abstain``
  (``fabricated``)。**計上のみ** (decision.jsonl + 呼出側の warning)。初版では
  unit を直さない。
- 固有値が無い / 全て既知 → ``skip``。

事例段は持たない (``contentless_social_formula`` と同じ形)。矛盾は「同一
ラベルの値の不一致」という構造的判定で誤発火が出にくく、捏造候補は棄権に
倒しているため、``confirm`` の相手を作る動機が今のところ無い。
"""

from __future__ import annotations

from typing import Any

from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.free.generation.strategy_cogwriter import ReviewIssue
from backend.free.generation.text_skeleton import (
    PinnedValue,
    TextSkeleton,
    extract_pinned_candidates,
)

__all__ = [
    "PREDICATE_NAME",
    "bind_debug_logger",
    "find_unit_issues",
    "predicate",
]

PREDICATE_NAME = "text_unit_fabrication"


def _scan_unit(
    text: str,
    skeleton: TextSkeleton | None,
    known: set[str],
) -> tuple[list[tuple[str, PinnedValue, str]], list[str]]:
    """unit 本文の固有値を skeleton / known と突き合わせる (純粋関数)。

    Returns:
        ``(矛盾のリスト [(label, 既存 PinnedValue, 新値)], 未知値のリスト)``。
        両方とも本文中の出現順。
    """
    contradictions: list[tuple[str, PinnedValue, str]] = []
    fabricated: list[str] = []
    for label, value in extract_pinned_candidates(text):
        existing = None
        if skeleton is not None:
            existing = next(
                (pv for pv in skeleton.pinned_values if pv.label == label), None,
            )
        if existing is not None and existing.value != value:
            contradictions.append((label, existing, value))
            continue
        if value not in known and value not in fabricated:
            fabricated.append(value)
    return contradictions, fabricated


class _FabricationLexicalPredicate:
    """字句段: ``(unit_text, known, skeleton) -> Verdict``。

    ``LexicalPredicate`` (``bool | str | None`` しか運べない) では棄権に
    「どの値が未知だったか」を乗せられないため、:class:`Predicate` Protocol を
    直接実装する。
    """

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: dict[str, Any] | None = None) -> Verdict:
        ctx = ctx or {}
        if not text or not text.strip():
            return Verdict(
                value=NEGATIVE_LABEL, score=0.0, band="skip",
                evidence="empty", predicate=self.name, stage="lexical",
            )
        skeleton = ctx.get("skeleton")
        known: set[str] = ctx.get("known") or set()
        contradictions, fabricated = _scan_unit(text, skeleton, known)
        if contradictions:
            label, existing, new_value = contradictions[0]
            return Verdict(
                value="contradiction", score=1.0, band="fire",
                evidence=f"{label}:{existing.value}->{new_value}",
                predicate=self.name, stage="lexical",
            )
        if fabricated:
            return Verdict(
                value=None, score=0.0, band="abstain",
                evidence=fabricated[0], predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=NEGATIVE_LABEL, score=0.0, band="skip",
            evidence="all_known", predicate=self.name, stage="lexical",
        )


#: プロセス共通の判定点。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_FabricationLexicalPredicate(),
        policy="complement",
        candidates=["contradiction", "fabricated", NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def _known_value_set(known_text: str, skeleton: TextSkeleton | None) -> set[str]:
    known: set[str] = set()
    if skeleton is not None:
        known |= skeleton.known_values()
    for _label, value in extract_pinned_candidates(known_text or ""):
        known.add(value)
    return known


def find_unit_issues(
    unit_idx: int,
    text: str,
    *,
    skeleton: TextSkeleton | None,
    known_text: str,
    is_fiction: bool,
) -> tuple[list[ReviewIssue], list[str]]:
    """unit 本文を判定点へ通し、矛盾 :class:`ReviewIssue` と捏造候補を返す。

    ``known_text`` は ``brief`` / 計画 (``global_context`` / ``title`` /
    ``key_points``) / ``unit_rag`` を連結した文字列 (呼出側で組む)。
    ``is_fiction`` (計画の ``global_context`` に創作の印がある) のときは捏造
    候補の収集を見送る (矛盾は見送らない)。

    判定点の評価は decision.jsonl への記録のためだけに行う (呼出側が既に
    独自に矛盾/捏造を数え上げるため、戻り値の Verdict 自体は使わない —
    単一の Verdict は「最初に見つかった 1 件」しか運べないため)。
    """
    if not text or not text.strip():
        return [], []
    known = _known_value_set(known_text, skeleton)
    ctx = {"skeleton": skeleton, "known": known}
    predicate.evaluate(text, ctx)

    contradictions, fabricated = _scan_unit(text, skeleton, known)
    issues = [
        ReviewIssue(
            unit_idx=unit_idx,
            issue=f"『{new_value}』は §{existing.unit_idx + 1} の『{existing.value}』と矛盾",
            fix=(
                f"『{new_value}』は §{existing.unit_idx + 1} の『{existing.value}』と"
                "矛盾。前節の値に合わせる"
            ),
        )
        for _label, existing, new_value in contradictions
    ]
    return issues, ([] if is_fiction else fabricated)
