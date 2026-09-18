"""チャット生成中は背景 aux を待たせるプロセス内ゲート。

CLAUDE.md §6 #1 は「**アイドル窓の** sleep-time / 学習はベースモデルで実行する
(専有スロット、チャットと KV を分離)」と定めている。KV の分離は
``LocalClient.chat_slot`` / ``background_slot`` で実現済みだが、**GPU 演算は
スロットで分離されない**。llama.cpp は複数スロットを時分割するため、背景タスクが
走っている間ユーザー応答のデコードが直接遅くなる。本モジュールは
「アイドル窓」の側を実際に強制する。

実測 (2026-09-03 ライブ監査、Qwen3.8-27B Q4_K_M / n_slots=3):

    decode      単独 200-218 ms/tok  →  併走 416-445 ms/tok
    prompt eval 単独  43 ms/tok      →  併走  70 ms/tok
    累積 tg 1.12-1.25 t/s、背景スロットが黙った瞬間だけ tg_3s が 3.2-4.2 t/s へ回復

この 3-4 倍の劣化が二次被害を連鎖させていた — aux が自分の競合で 14 回
タイムアウトし、較正値が 54.6→81.9→122.9→59.0→88.5s と振動し、sleep-time の
要約・競合解決が失敗し続け、最終的にフロントの 60 秒チャンクタイムアウトに
掛かって**完走した応答が捨てられた**。

**チャット側は何も待たない。** 待つのは背景側だけで、ゲートは
「チャットが走っている間、背景の *新規* dispatch を止める」片方向。

加えて **チャット要求の到着** (``chat_request_started``) を背景側へ伝える。
dispatch 後に始まったチャットは、背景の 1 生成 (実測で最長 260 秒:
2026-09-09 検証、conflict_resolution) が終わるまで GPU を分け合い、その間の
チャット側の分類器 / 日付意図の抽出 (40 秒予算) と初トークン (116 秒予算) が
タイムアウトして誤答・空応答になった。deferrable な purpose は
:func:`preempted_by_chat` で要求の到着を待ち受け、生成を打ち切って
「一過性 (contended)」として呼出側に返す — 呼出側は次サイクルで再試行する
(部分状態は書き戻さない: 生成結果を受け取らないだけ)。
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TypeVar
import asyncio
import contextlib

from backend.log_config import get_logger

logger = get_logger("llm.generation_gate")

_T = TypeVar("_T")

__all__ = [
    "ChatTurnLease",
    "begin_chat_turn",
    "current_turn_lease",
    "run_yielding_to_chat",
    "chat_generation",
    "chat_is_active",
    "chat_request_started",
    "activity_token",
    "was_contended_since",
    "wait_for_idle",
    "wait_for_chat_request",
    "gate_stream",
]

#: チャット生成の入れ子カウント。ツール実行→再生成のように 1 ターンで複数回
#: 生成する経路があるため bool ではなく refcount で持つ。
_active: int = 0

#: チャット生成が始まった回数。単調増加。「この aux 呼び出しの最中にチャットが
#: 走ったか」を **開始時と終了時のスナップショット比較** で判定するために使う
#: (終了時点だけ見ると、途中で走って終わったチャットを取りこぼす)。
_activations: int = 0

#: ``_active == 0`` の間セットされているイベント。初期状態はアイドル。
_idle_event: asyncio.Event | None = None

#: チャット要求の到着回数 (単調増加)。``wait_for_chat_request`` は自分の
#: 開始時点より後の到着だけを待つ。
_requests: int = 0
#: 到着ごとに set → 即 clear するイベント (待ち手を起こすためだけのもの)。
#: イベントは最初に待った走行ループに束縛されるので、ループごとに作り直す
#: (テストのようにループが替わる環境で、別ループの待ち手が RuntimeError で
#: 落ちて「到着」と誤認しないため)。
_request_event: asyncio.Event | None = None
_request_event_loop: asyncio.AbstractEventLoop | None = None


def _request_signal() -> asyncio.Event:
    global _request_event, _request_event_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _request_event is None or (loop is not None and _request_event_loop is not loop):
        _request_event = asyncio.Event()
        _request_event_loop = loop
    return _request_event


def chat_request_started() -> None:
    """チャット要求が届いたことを背景側へ知らせる (API の入口で呼ぶ)。

    生成の開始 (``chat_generation``) より前 — 分類器 / 日付意図の抽出 /
    検索 — から GPU を要するので、要求の到着で知らせる。
    """
    global _requests
    _requests += 1
    signal = _request_signal()
    signal.set()
    signal.clear()


async def wait_for_chat_request(since: int | None = None) -> None:
    """``since`` (省略時は今) より後にチャット要求が届くまで待つ。"""
    seen = _requests if since is None else since
    while _requests == seen:
        await _request_signal().wait()


def request_token() -> int:
    """``wait_for_chat_request`` に渡す開始時点のスナップショット。"""
    return _requests


def _event() -> asyncio.Event:
    """アイドルイベントを遅延生成する (import 時に走行ループが無いため)。"""
    global _idle_event
    if _idle_event is None:
        _idle_event = asyncio.Event()
        _idle_event.set()
    return _idle_event


def chat_is_active() -> bool:
    """いまチャット生成が走っているか。"""
    return _active > 0


def activity_token() -> tuple[int, int]:
    """「チャットが走ったか」を後で判定するためのスナップショット。"""
    return (_active, _activations)


def was_contended_since(token: tuple[int, int]) -> bool:
    """``token`` の取得以降にチャット生成と重なったか (純粋な比較)。

    開始時に既に走っていた場合と、途中で新たに始まった場合の両方を拾う。
    タイムアウトの原因が自分の遅さなのか競合なのかを切り分けるのに使う —
    競合由来の所要時間を較正へ食わせると、一過性の混雑が**恒久的な予算膨張**
    として residual に残る (実測でその振動を観測している)。
    """
    was_active, seen = token
    return was_active > 0 or _activations != seen


def _acquire() -> None:
    global _active, _activations
    _active += 1
    _activations += 1
    _event().clear()


def _release() -> None:
    global _active
    _active -= 1
    if _active <= 0:
        _active = 0
        _event().set()


@asynccontextmanager
async def chat_generation() -> AsyncIterator[None]:
    """チャット生成の在圏を宣言する。背景 aux はこの間 dispatch を待つ。"""
    _acquire()
    try:
        yield
    finally:
        _release()


async def wait_for_idle(max_wait: float, *, purpose: str = "") -> float:
    """チャットがアイドルになるまで待ち、実際に待った秒数を返す。

    ``max_wait`` を超えたら **待つのをやめて先へ進む**。背景処理を無期限に
    飢えさせない方が重要 (記憶の統合が永久に走らない方が害が大きい)。
    打ち切った場合はログに残す — 競合したまま走ったことが後から分かるように。
    """
    if not chat_is_active():
        return 0.0
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        await asyncio.wait_for(_event().wait(), timeout=max_wait)
    except TimeoutError:
        waited = loop.time() - started
        logger.info(
            "Aux proceeded without an idle window after %.1fs (purpose=%s); "
            "chat generation is still in flight",
            waited, purpose or "<unspecified>",
        )
        return waited
    return loop.time() - started


#: 応答ストリームへ引き渡したリースを、ストリームが始まらないまま握り続けない
#: 上限。StreamingResponse はハンドラの return 直後に反復を始めるので通常は
#: ミリ秒で始まる。始まらない (応答前に切断) ときだけ効く。
_HANDOVER_GRACE_SEC = 30.0


class ChatTurnLease:
    """1 チャットターン (要求の到着 → 応答の終わり) の在圏を宣言するリース。

    ``chat_generation`` はトークン生成の間しか立たないため、要求の到着から
    初回生成までの前処理 (分類器 / 検索 / ツール実行) は背景から見て
    **アイドル** だった。その窓で dispatch した背景の生成は、到着の通知
    (``chat_request_started``) を既に見逃しているので打ち切られず、ターンの
    生成と最後まで GPU を分け合う (2026-09-17 監査: ``search_history`` を
    撃った想起ターンの間に sleep-time の ``summarize`` が 34.8 秒走り、
    261 トークンの追加 prefill で TTFT 23 秒)。

    リースはターン全体で ``_active`` を立てる。ストリーミング応答では
    ハンドラが先に return するので、:meth:`hand_over` でストリームへ解放の
    責務を移し、ストリームが :meth:`stream_started` → :meth:`release` する。
    ``release`` は冪等。
    """

    def __init__(self) -> None:
        self._released = False
        self._handed_over = False
        self._expiry: asyncio.TimerHandle | None = None
        _acquire()

    @property
    def handed_over(self) -> bool:
        return self._handed_over

    def hand_over(self, grace_sec: float = _HANDOVER_GRACE_SEC) -> None:
        """解放の責務を応答ストリームへ移す (始まらなければ ``grace_sec`` で解放)。"""
        if self._released or self._handed_over:
            return
        self._handed_over = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._expiry = loop.call_later(grace_sec, self._expire)

    def stream_started(self) -> None:
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None

    def _expire(self) -> None:
        self._expiry = None
        if not self._released:
            logger.warning(
                "Chat turn lease released: the response stream never started "
                "within %.0fs", _HANDOVER_GRACE_SEC,
            )
            self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.stream_started()
        _release()


_turn_lease: ContextVar[ChatTurnLease | None] = ContextVar(
    "evoref_chat_turn_lease", default=None,
)


def begin_chat_turn() -> ChatTurnLease:
    """チャット要求の到着を知らせ、ターン全体の在圏リースを取る (API の入口で呼ぶ)。"""
    chat_request_started()
    lease = ChatTurnLease()
    _turn_lease.set(lease)
    return lease


def current_turn_lease() -> ChatTurnLease | None:
    """このコンテキストのターンリース (入口を通っていなければ ``None``)。"""
    return _turn_lease.get()


#: :func:`run_yielding_to_chat` がアイドル窓を待つ上限と、打ち切り後に
#: やり直す回数の上限。どちらも超えたら競合覚悟で最後まで走らせる
#: (背景処理を無期限に飢えさせない、``wait_for_idle`` と同じ方針)。
_YIELD_IDLE_MAX_WAIT_SEC = 120.0
_YIELD_MAX_PREEMPTIONS = 5


async def run_yielding_to_chat(
    call: Callable[[], Awaitable[_T]], *, label: str,
) -> _T:
    """背景の 1 生成を「アイドル窓で出し、チャット要求が来たら打ち切ってやり直す」。

    ``AuxClient`` を通らずに llama-server を直接叩く背景生成 (起動時の能力
    プローブ / 出力品質プローブ) は、ゲートの外にあった。2026-09-17 監査:
    リセット直後の検証ドライブで品質プローブがユーザーの応答と重なり、
    新規トークン 26 の応答の初トークンが 32 秒になった。
    """
    for attempt in range(_YIELD_MAX_PREEMPTIONS + 1):
        await wait_for_idle(_YIELD_IDLE_MAX_WAIT_SEC, purpose=label)
        if attempt == _YIELD_MAX_PREEMPTIONS:
            logger.info(
                "Background generation %s ran to completion after %d preemptions",
                label, attempt,
            )
            return await call()
        since = request_token()
        work = asyncio.ensure_future(call())
        waiter = asyncio.ensure_future(wait_for_chat_request(since))
        try:
            done, _ = await asyncio.wait(
                {work, waiter}, return_when=asyncio.FIRST_COMPLETED,
            )
            if work in done:
                return work.result()
            work.cancel()
            with contextlib.suppress(BaseException):
                await work
            logger.info(
                "Background generation %s preempted by a chat request; retrying "
                "in the next idle window", label,
            )
        finally:
            for task in (work, waiter):
                if not task.done():
                    task.cancel()
    raise AssertionError("unreachable")  # pragma: no cover


def gate_stream(agen: AsyncIterator[str]) -> AsyncIterator[str]:
    """トークンストリームの生存期間だけチャット在圏を立てるラッパ。

    ``aclose()`` / ``GeneratorExit`` でも ``finally`` が走るので、キャンセル
    されたターンでゲートが立ちっぱなしにならない。
    """

    async def _gated() -> AsyncIterator[str]:
        async with chat_generation():
            async for token in agen:
                yield token

    return _gated()


def reset_for_tests() -> None:
    """テスト用にゲート状態を初期化する。"""
    global _active, _activations, _idle_event
    _active = 0
    _activations = 0
    _idle_event = None
