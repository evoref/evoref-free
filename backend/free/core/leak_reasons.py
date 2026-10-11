"""他言語トークン混入の失敗理由 (``turn_outcome_reason``) の語彙の SSOT

理由の文字列は経験 (experience.jsonl)・回帰台帳・ログに **そのまま永続化** されて
いるので、文字列を変えない (非対称な ``JA response`` / ``response`` も含む)。
生成側 (``FeedbackCollector`` / ``FewShotPool``) と前方一致で引く側 (チャンネル表・
帰属表・採用ゲート・回帰台帳) が同じ表を読む (不変則 #14 (a))。
判定そのもの (どの種別か) は :func:`~backend.free.core.text_quality.foreign_script_leak`。
"""

from __future__ import annotations

from typing import Literal

LeakKind = Literal["chinese", "hangul"]

#: 前方一致で引く接頭辞 (理由は ``"<接頭辞>…"``)。
CHINESE_PREFIX = "Chinese token leaked"
HANGUL_PREFIX = "Hangul token leaked"
LEAK_REASON_PREFIXES: tuple[str, ...] = (CHINESE_PREFIX, HANGUL_PREFIX)

#: 種別 → 理由の全文。
LEAK_REASONS: dict[str, str] = {
    "chinese": f"{CHINESE_PREFIX} into JA response",
    "hangul": f"{HANGUL_PREFIX} into response",
}


def leak_reason(kind: str | None) -> str | None:
    """``foreign_script_leak`` の種別を理由の全文にする (``None`` は ``None``)。"""
    return LEAK_REASONS.get(kind) if kind else None


def leak_reason_for(text: str, *, query: str = "") -> str | None:
    """応答に他言語トークンの混入があれば、その理由の全文を返す。"""
    from backend.free.core.text_quality import foreign_script_leak

    return leak_reason(foreign_script_leak(text, query=query))


def is_model_origin_reason(reason: object) -> bool:
    """モデル由来 (プロンプトでは直らない) の失敗理由か。"""
    return str(reason or "").startswith(LEAK_REASON_PREFIXES)


__all__ = [
    "CHINESE_PREFIX",
    "HANGUL_PREFIX",
    "LEAK_REASONS",
    "LEAK_REASON_PREFIXES",
    "LeakKind",
    "is_model_origin_reason",
    "leak_reason",
    "leak_reason_for",
]
