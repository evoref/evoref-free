"""検証器の発火を「このターン」に紐づけて集める台帳 (f_03 §3.5.1)。

規則台帳 (``agent.prompt_ledger``) の各規則は ``verifier`` で決定論の検証器
(ストリームフィルタ / 出力制約の検証 / 訂正検出) に対応づく。検証器が
発火したターンは、その規則が **prompt に載っていたのに破られた** ターン
(harmful)。発火しなかったターンは helpful の母数。この計数が
「削ってよい規則」の唯一の根拠になる。

``aux_telemetry`` と同じ contextvar 方式: リクエスト入口で ``open_verifier_scope``
を呼び、検証器は ``record_verifier_hit`` を呼ぶだけ。結末を書く側
(``chat_stream_common._log_chat_outcome``) が ``current_verifier_hits`` で読む。
スコープが無い経路 (単体テスト等) では no-op。

``core`` は pillar に属さない横断層なので、``stream_filter`` / ``text_quality``
(core) と ``chat_stream_common`` (api) の両方から参照できる。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

__all__ = [
    "VERIFIER_IDS",
    "open_verifier_scope",
    "record_verifier_hit",
    "current_verifier_hits",
    "current_verifier_mode",
    "record_turn_outcome",
    "current_turn_outcome",
]

#: 既知の検証器 id。規則台帳の ``verifier`` はこの集合の値を取る。
VERIFIER_IDS: frozenset[str] = frozenset({
    "thinking",          # StreamThinkingFilter: 思考ブロック / ラベルを落とした
    "head_label",        # HeadBufferFilter: 先頭の [応答] 等のラベルを落とした
    "query_echo",        # QueryEchoFilter: ユーザー発話の復唱を落とした
    "repetition",        # RepetitionGuardFilter: 反復を打ち切った
    "internal_frame",    # InternalFrameMentionFilter: 参考枠の名指しを落とした
    "unwritten_file",    # UnwrittenFileClaimFilter: 実体の無い書込み申告
    "constraint.length", # 文字数指定の違反
    "constraint.lines",  # 行数指定の違反
    "constraint.items",  # 個数指定の違反
    "constraint.words",  # 語数指定の違反
    "constraint.banned", # 禁止語 / 禁止文字種の違反
    "constraint.form",   # 出力形式 (箇条書き等) の違反
    "user_correction",   # ユーザーが値を訂正した (陳腐値を答えた)
    # 本文の決定論的な破綻 (FeedbackCollector._derive_turn_outcome)。これらが
    # 無いと、成否の判定は経験 (turn_outcome) にしか残らず、outcome JSONL の
    # verifier_hits は表面の検証器しか映さない (2026-09-05 ライブ監査 F-11:
    # 100 ターン中 3 件しか失敗と記録されず、Level 2 が恒久的にデータ不足)。
    "content.arithmetic",       # 本文の式の検算が合わない
    "content.broken_text",      # 語間空白 / 中国語混入
    "content.self_retraction",  # 1 つの応答に結論が 2 つ (撤回 / 列挙して否定)
    "content.measured",         # 注入した実測値と別の数を述べた
    "content.tool_result",      # calculate の結果を回答に使わなかった
    "content.claimed_change",   # 撃てなかったのに完了を述べた
    "content.user_echo",        # ユーザー発話のオウム返し
})


@dataclass
class _Scope:
    mode: str = "chat"
    hits: list[str] = field(default_factory=list)
    #: 経験記録が導出したターン成否 ("success" / "partial" / "failed") と理由。
    #: 結末 JSONL の ``success`` が「SSE を届けられたか」だけを見て、中身の
    #: 破綻を成功として記録していたのを、同じリクエスト内で持ち上げるための
    #: 通路 (2026-09-05 ライブ監査 F-11)。
    turn_outcome: str | None = None
    turn_outcome_reason: str | None = None


_scope: ContextVar[_Scope | None] = ContextVar("verifier_scope", default=None)


def open_verifier_scope(mode: str = "chat") -> None:
    """このリクエスト用の台帳を開く (リクエスト入口で 1 回)。"""
    _scope.set(_Scope(mode=mode or "chat"))


def record_verifier_hit(verifier_id: str) -> None:
    """検証器の発火を記録する。スコープが無ければ何もしない。"""
    scope = _scope.get()
    if scope is None or not verifier_id:
        return
    scope.hits.append(verifier_id)


def current_verifier_hits() -> list[str]:
    scope = _scope.get()
    return list(scope.hits) if scope else []


def current_verifier_mode() -> str | None:
    scope = _scope.get()
    return scope.mode if scope else None


def record_turn_outcome(outcome: str, reason: str | None = None) -> None:
    """経験記録が導出したターン成否をこのリクエストの台帳へ置く。"""
    scope = _scope.get()
    if scope is None or not outcome:
        return
    scope.turn_outcome = outcome
    scope.turn_outcome_reason = reason or None


def current_turn_outcome() -> tuple[str | None, str | None]:
    """``(outcome, reason)``。スコープが無いか未記録なら ``(None, None)``。"""
    scope = _scope.get()
    if scope is None:
        return None, None
    return scope.turn_outcome, scope.turn_outcome_reason
