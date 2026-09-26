"""制作中のターン受付 (f_10 §3「制作中のターン受付」、2026-09-26)。

制作 (create) は数十分 GPU を占有する。2 本目の制作は同じスロットの後ろに並び、
待ち時間まで呼出予算・ステージ予算を消費して両方が遅れる (codegen は
``AuxClient`` のロックを通らないので、スロットの外で直列化する仕組みが無い)。
同じセッションで 2 本走ると WM・履歴の順序と ``needs_input`` の再開が二重になる。

そこでターンの入口で走っているターンの台帳を引き、次のどちらかなら断る:

(a) 新しい要求が create で、create のターン (前面 / detached を問わない) が走っている。
(b) 新しい要求と同じ ``session_id`` のターンが走っている (モードを問わない)。

登録は入口、解除はターンの在圏リースの解放 (``ChatTurnLease.on_release``)。
制作の agent タスクを結び付けたターンは、タスクが終わる (detached の履歴・
経験の記録が済む) まで解除を遅らせる。ストリームの後始末は外側 (リース) が
内側 (detached への引き継ぎ) より先に走るので、引き継ぎの側で延長すると
取りこぼす — タスクの生死で判定するのはそのため。``run.json`` は見ない。
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field

from backend.free.core.session_mode import is_create_mode
from backend.log_config import get_logger

logger = get_logger("api.chat.turn_admission")


@dataclass(eq=False)
class TurnAdmission:
    """受け付けた 1 ターン。``release`` で台帳から外れる (冪等)。"""

    session_id: str
    mode: str
    _task: asyncio.Task | None = field(default=None, repr=False)
    _release_requested: bool = field(default=False, repr=False)
    _dropped: bool = field(default=False, repr=False)

    def bind_task(self, task: asyncio.Task) -> None:
        """このターンの agent タスク。生きている間は解除を遅らせる (detached 継続)。"""
        self._task = task

    def release(self) -> None:
        """ターンの終わり。結び付けたタスクが生きていれば、その終了まで待つ。"""
        if self._release_requested:
            return
        self._release_requested = True
        task = self._task
        if task is not None and not task.done():
            # タスクの done callback は登録順に call_soon される。ここからさらに
            # call_soon すると、先に積まれた履歴・経験の記録 (detached の終端処理)
            # より後で外れる。
            task.add_done_callback(
                lambda _t: asyncio.get_running_loop().call_soon(self._drop),
            )
            return
        self._drop()

    def _drop(self) -> None:
        if self._dropped:
            return
        self._dropped = True
        if self in _ACTIVE:
            _ACTIVE.remove(self)


@dataclass(frozen=True)
class TurnBusy:
    """受け付けなかった理由 — 走っているターンの ``session_id`` / ``mode``。"""

    session_id: str
    mode: str
    same_session: bool


_ACTIVE: list[TurnAdmission] = []

_current: ContextVar[TurnAdmission | None] = ContextVar(
    "evoref_turn_admission", default=None,
)


def try_admit(session_id: str | None, mode: str) -> TurnAdmission | TurnBusy:
    """ターンを受け付けるか判定し、受け付けたら台帳に載せる (await を挟まない)。"""
    sid = session_id or ""
    if sid:
        for turn in _ACTIVE:
            if turn.session_id == sid:
                return TurnBusy(turn.session_id, turn.mode, same_session=True)
    if is_create_mode(mode):
        for turn in _ACTIVE:
            if is_create_mode(turn.mode):
                return TurnBusy(turn.session_id, turn.mode, same_session=False)
    admission = TurnAdmission(sid, mode)
    _ACTIVE.append(admission)
    _current.set(admission)
    return admission


def current_admission() -> TurnAdmission | None:
    """このコンテキストで受け付けたターン (入口を通っていなければ ``None``)。"""
    return _current.get()


def active_turns() -> list[TurnAdmission]:
    """走っているターンの一覧 (テスト・診断用の写し)。"""
    return list(_ACTIVE)


__all__ = [
    "TurnAdmission",
    "TurnBusy",
    "active_turns",
    "current_admission",
    "try_admit",
]
