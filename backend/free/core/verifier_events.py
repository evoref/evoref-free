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
})


@dataclass
class _Scope:
    mode: str = "chat"
    hits: list[str] = field(default_factory=list)


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
