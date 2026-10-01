"""ノート化していない会話ターンの供給元。

ノート生成は sleep-time に移った (c_16 §4.1: ``short`` は sleep-time が
``put`` する)。応答パスは ``WorkingMemory`` にターンを積むだけで、記憶層への
書き込みを一切しない。したがって「何をノートにするか」の入力は **会話履歴**
(``<data_root>/g1/store/history/``) — ターン ID の発行元であり、窓から押し出されても残る
唯一の完全な列 (c_16 §2)。

``TurnSource`` は差し替え可能にしてある。テストは固定のターン列を返す実装を
渡し、本番は :class:`HistoryTurnSource` が ``HistoryManager`` を読む。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from backend.log_config import get_logger

logger = get_logger("memory.episodic.turn_source")

#: 1 サイクルで本文を読むセッション数の上限。未処理のセッションだけを数サイクル
#: かけて消化する (索引の件数比較はメモリ上、本文を開くのはこの件数だけ)。
DEFAULT_SESSION_LIMIT = 8

#: 読み出しの例外を何回まで試すか (一時障害かもしれないので、すぐには外さない)。
MAX_READ_ATTEMPTS = 3


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


@dataclass
class PendingSessions:
    """未処理のターンがあるセッションの選択結果。"""

    #: 今回読むセッション (古い backlog → 現セッションの順、上限まで)。
    sessions: list[SessionTurns] = field(default_factory=list)
    #: 上限を超えて次のサイクルへ持ち越すセッション数 / ターン数。
    remaining_sessions: int = 0
    remaining_turns: int = 0
    #: 読めず (原本なし / 空 / 索引より短い / 例外の繰り返し) 候補から外しているセッション数。
    unreadable_sessions: int = 0


@runtime_checkable
class TurnSource(Protocol):
    """``(session_id, turns)`` を供給する契約。"""

    def recent_sessions(self, limit: int = DEFAULT_SESSION_LIMIT) -> list[SessionTurns]:
        """直近セッションのターン列を新しい順に返す。"""
        ...

    # 任意: ``pending_sessions(processed, limit) -> PendingSessions`` を持つ実装は
    # 未処理のセッションだけを返す (無ければ ``recent_sessions`` へ縮退する)。


class HistoryTurnSource:
    """``HistoryManager`` からターン列を読む既定実装。

    Args:
        manager: 既に構築済みの ``HistoryManager``。``None`` なら
            ``get_history_manager()`` で解決する (sleep-time からの呼出
            なので、シングルトンの初回構築が起きても応答を止めない)。
    """

    def __init__(self, manager: Any = None) -> None:
        self._manager = manager
        #: ``{session_id: 諦めた時点の索引の turn_count}`` (プロセス内のみ。progress は
        #: 触らない)。索引の件数がこれを超えれば再び候補になる。
        self._given_up: dict[str, int] = {}
        self._attempts: dict[str, int] = {}

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
        return self._read_sessions(manager, [e.session_id for e in entries if e.session_id])

    def pending_sessions(
        self, processed: Mapping[str, int], limit: int = DEFAULT_SESSION_LIMIT,
    ) -> PendingSessions:
        """索引の件数が処理済み件数を上回るセッションだけを ``limit`` 件読む。

        backlog は古い順に消化する (訂正が元の発話より先にノート化されると、訂正の
        宛先の継承が効かない、f_02 §4.2)。ただし現セッション (追記ログの mtime が最新、無ければ索引で最も新しいもの) は未処理なら先頭枠で必ず取り、残りの枠を古い順の backlog に使う。
        返す順は古い backlog → 現セッション。

        Args:
            processed: ``{session_id: ノート化済みのターン数}`` (progress)。
            limit: 今回本文を読むセッション数の上限。
        """
        try:
            manager = self._resolve()
            entries, _total = manager.list_sessions(limit=sys.maxsize)
        except Exception as e:  # noqa: BLE001
            logger.warning("Episodic: failed to list history sessions: %s", e)
            return PendingSessions()
        pending: list[tuple[str, int, int]] = []  # (id, 未処理ターン数, 索引の件数)
        unreadable = 0
        # 現セッション = 追記ログの mtime が最新のもの。決まらなければ索引で最も新しいもの。
        newest_sid = None
        last_active = getattr(manager, "last_active_session_id", None)
        if last_active is not None:
            try:
                newest_sid = last_active()
            except Exception:  # noqa: BLE001 — 決まらなければ索引の先頭へ
                newest_sid = None
        if newest_sid is None and entries:
            newest_sid = entries[0].session_id
        current: tuple[str, int, int] | None = None
        for e in entries:
            sid, count = e.session_id, int(e.turn_count)
            done = int(processed.get(sid, 0))
            if not sid or count <= done:
                continue
            if count <= self._given_up.get(sid, -1):
                unreadable += 1
                continue
            item = (sid, count - done, count)
            if sid == newest_sid:
                current = item
            else:
                pending.append(item)
        pending.reverse()  # 索引は新しい順 → 古い順
        take = max(limit - (1 if current else 0), 0)
        chosen, rest = pending[:take], pending[take:]
        if current is not None:
            chosen.append(current)
        sessions: list[SessionTurns] = []
        for sid, _n, count in chosen:
            session = self._read_one(manager, sid, count)
            if session is not None:
                sessions.append(session)
            if self._given_up.get(sid, -1) >= count:
                unreadable += 1
        return PendingSessions(
            sessions=sessions,
            remaining_sessions=len(rest),
            remaining_turns=sum(n for _, n, _c in rest),
            unreadable_sessions=unreadable,
        )

    def _read_one(
        self, manager: Any, session_id: str, index_count: int,
    ) -> SessionTurns | None:
        """1 セッションを読む。恒久的に読めない / 短いものは索引の件数に対して諦める。"""
        try:
            session = manager.get_session(session_id)
        except Exception as e:  # noqa: BLE001
            attempts = self._attempts.get(session_id, 0) + 1
            self._attempts[session_id] = attempts
            logger.warning(
                "Episodic: failed to read session %s (attempt %d/%d): %s",
                session_id, attempts, MAX_READ_ATTEMPTS, e,
            )
            if attempts >= MAX_READ_ATTEMPTS:
                self._give_up(session_id, index_count, "read failed repeatedly")
            return None
        self._attempts.pop(session_id, None)
        if session is None or not session.turns:
            self._give_up(session_id, index_count, "missing or empty original")
            return None
        if len(session.turns) < index_count:
            self._give_up(session_id, index_count, "fewer turns than the index says")
        return self._to_session_turns(session_id, session)

    def _give_up(self, session_id: str, index_count: int, reason: str) -> None:
        self._given_up[session_id] = index_count
        logger.warning(
            "Episodic: session %s skipped until its index count grows "
            "(%s, index turn_count=%d)", session_id, reason, index_count,
        )

    def _read_sessions(self, manager: Any, session_ids: list[str]) -> list[SessionTurns]:
        out: list[SessionTurns] = []
        for session_id in session_ids:
            try:
                session = manager.get_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("Episodic: failed to read session %s: %s", session_id, e)
                continue
            if session is None or not session.turns:
                continue
            out.append(self._to_session_turns(session_id, session))
        return out

    @staticmethod
    def _to_session_turns(session_id: str, session: Any) -> SessionTurns:
        return SessionTurns(
            session_id=session_id,
            turns=list(session.turns),
            mode=str(getattr(session, "mode", "chat") or "chat"),
            project_id=getattr(session, "project_id", None),
            summary=getattr(session, "summary", None),
            lang=str(getattr(session, "lang", "") or ""),
        )


__all__ = [
    "DEFAULT_SESSION_LIMIT",
    "HistoryTurnSource",
    "PendingSessions",
    "SessionTurns",
    "TurnSource",
]
