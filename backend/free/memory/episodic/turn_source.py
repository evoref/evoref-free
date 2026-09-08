"""ノート化していない会話ターンの供給元。

ノート生成は sleep-time に移った (c_16 §4.1: ``short`` は sleep-time が
``put`` する)。応答パスは ``WorkingMemory`` にターンを積むだけで、記憶層への
書き込みを一切しない。したがって「何をノートにするか」の入力は **会話履歴**
(``local/history/``) — ターン ID の発行元であり、窓から押し出されても残る
唯一の完全な列 (c_16 §2)。

``TurnSource`` は差し替え可能にしてある。テストは固定のターン列を返す実装を
渡し、本番は :class:`HistoryTurnSource` が ``HistoryManager`` を読む。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from backend.log_config import get_logger

logger = get_logger("memory.episodic.turn_source")

#: 1 サイクルで見に行くセッション数の上限。古いセッションはノート化済みなので、
#: 直近だけ見れば足りる (全件走査は履歴ファイルを全部開くことになる)。
DEFAULT_SESSION_LIMIT = 8


@dataclass
class SessionTurns:
    """1 セッション分のターン列 (ノート生成の入力)。"""

    session_id: str
    turns: list[dict] = field(default_factory=list)
    mode: str = "chat"
    project_id: str | None = None
    #: 履歴側の要約 (あれば要約ノートの本文に使う)。
    summary: str | None = None
    lang: str = ""


@runtime_checkable
class TurnSource(Protocol):
    """``(session_id, turns)`` を供給する契約。"""

    def recent_sessions(self, limit: int = DEFAULT_SESSION_LIMIT) -> list[SessionTurns]:
        """直近セッションのターン列を新しい順に返す。"""
        ...


class HistoryTurnSource:
    """``HistoryManager`` からターン列を読む既定実装。

    Args:
        manager: 既に構築済みの ``HistoryManager``。``None`` なら
            ``get_history_manager()`` で解決する (sleep-time からの呼出
            なので、シングルトンの初回構築が起きても応答を止めない)。
    """

    def __init__(self, manager: Any = None) -> None:
        self._manager = manager

    def _resolve(self) -> Any:
        if self._manager is not None:
            return self._manager
        from backend.free.history.history_manager import get_history_manager

        self._manager = get_history_manager()
        return self._manager

    def recent_sessions(self, limit: int = DEFAULT_SESSION_LIMIT) -> list[SessionTurns]:
        try:
            manager = self._resolve()
            entries, _total = manager.list_sessions(limit=limit)
        except Exception as e:  # noqa: BLE001 — 履歴が読めなくても sleep-time は続ける
            logger.warning("Episodic: failed to list history sessions: %s", e)
            return []
        out: list[SessionTurns] = []
        for entry in entries:
            session_id = getattr(entry, "session_id", "")
            if not session_id:
                continue
            try:
                session = manager.get_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("Episodic: failed to read session %s: %s", session_id, e)
                continue
            if session is None or not session.turns:
                continue
            out.append(
                SessionTurns(
                    session_id=session_id,
                    turns=list(session.turns),
                    mode=str(getattr(session, "mode", "chat") or "chat"),
                    project_id=getattr(session, "project_id", None),
                    summary=getattr(session, "summary", None),
                    lang=str(getattr(session, "lang", "") or ""),
                ),
            )
        return out


__all__ = [
    "DEFAULT_SESSION_LIMIT",
    "HistoryTurnSource",
    "SessionTurns",
    "TurnSource",
]
