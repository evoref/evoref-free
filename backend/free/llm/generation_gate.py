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
「チャットが走っている間、背景の *新規* dispatch を止める」だけの片方向。
実行中の背景タスクを中断はしない (中断すると部分状態の書き戻しが要る)。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio

from backend.log_config import get_logger

logger = get_logger("llm.generation_gate")

__all__ = [
    "chat_generation",
    "chat_is_active",
    "activity_token",
    "was_contended_since",
    "wait_for_idle",
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


@asynccontextmanager
async def chat_generation() -> AsyncIterator[None]:
    """チャット生成の在圏を宣言する。背景 aux はこの間 dispatch を待つ。"""
    global _active, _activations
    _active += 1
    _activations += 1
    _event().clear()
    try:
        yield
    finally:
        _active -= 1
        if _active <= 0:
            _active = 0
            _event().set()


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
