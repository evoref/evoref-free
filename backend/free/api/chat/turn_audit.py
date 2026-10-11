"""応答 1 回ごとの監査の受け口 (hook)。台帳・連鎖・検証は持たない (docs/p_07)。

Free はここに **Protocol と呼出しだけ** を置く (e_01 §0.1 原則 4)。監査証跡の実体
(ハッシュ連鎖の台帳・検証・報告書) は Pro が持ち、:func:`backend.edition.register_pro_handler`
で :data:`TURN_AUDIT_HANDLER` に :class:`TurnAuditSink` を登録したときだけ呼ばれる。

- 登録が無ければ :func:`emit_turn_audit` は handler の辞書を 1 回引いて返るだけで、
  イベントも作らない (``build`` を呼ばない)。Free の応答パスの挙動・遅延を変えない。
- sink の例外は握り潰して WARNING に出す (監査の失敗でチャットを止めない)。
- 本文 (問い・回答) は sink へ渡すが、Free はどこにも保存しない。sink が保存してよいのは
  ID と digest だけ (p_07 §3)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from backend.edition import get_pro_handler
from backend.log_config import get_logger

logger = get_logger("api.chat.turn_audit")

#: Pro が sink を登録する handler 名。
TURN_AUDIT_HANDLER = "turn_audit_sink"


@dataclass(frozen=True, slots=True)
class TurnAuditEvent:
    """記録の済んだ応答 1 回の素性 (ID と構成だけ。本文は digest 用に一時的に運ぶ)。

    Attributes:
        session_id / turn_id / trace_id: ID 連鎖の連結点 (c_05 §0.6)。
        mode: ``chat`` / ``create``。
        layer: 記録の経路 (``deliberative`` / ``meta-cognitive`` / ``long-form``)。
        private: プライベートセッションのターン。
        cancelled / truncated / generation_failed: 応答の終わり方。
        gen_config: ``GenerationConfigRef`` の直列化 (ID と版だけ、c_05 §5.2)。
        tool_uses: このリクエストで実行したツール (``verifier_events.current_tool_uses``)。
        tools_observed: ツールの実行を数える台帳がこのリクエストで開いていたか
            (``False`` なら ``tool_uses`` が空でも「実行が無かった」とは言えない)。
        query / response: 記録した問いと回答の本文 (sink が digest にする。保存しない)。
    """

    session_id: str
    turn_id: str
    trace_id: str
    mode: str
    layer: str
    private: bool
    cancelled: bool
    truncated: bool
    generation_failed: bool
    gen_config: dict[str, Any] | None = None
    tool_uses: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    tools_observed: bool = False
    query: str = ""
    response: str = ""


@runtime_checkable
class TurnAuditSink(Protocol):
    """応答 1 回の記録の直後に呼ばれる受け口 (Pro が実装する)。

    イベントループの上で同期に呼ばれる。重い処理 (ファイル I/O・大きなハッシュ) は
    書き手スレッドか自前のスレッドへ回すこと。
    """

    def on_turn(self, event: TurnAuditEvent) -> None: ...


def emit_turn_audit(build: Callable[[], TurnAuditEvent]) -> bool:
    """登録された sink があれば ``build()`` のイベントを渡す。渡したら ``True``。

    sink が無ければ ``build`` を呼ばずに ``False``。``build`` / sink の例外は
    握り潰して ``False`` (チャットを止めない)。
    """
    sink = get_pro_handler(TURN_AUDIT_HANDLER)
    if sink is None:
        return False
    try:
        sink.on_turn(build())
    except Exception as exc:  # noqa: BLE001 — 監査の失敗で応答の記録を壊さない
        logger.warning("Turn audit sink failed: %s", exc, exc_info=True)
        return False
    return True


__all__ = ["TURN_AUDIT_HANDLER", "TurnAuditEvent", "TurnAuditSink", "emit_turn_audit"]
