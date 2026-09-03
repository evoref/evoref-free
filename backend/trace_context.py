"""リクエスト相関ID (trace_id) のコンテキスト管理

contextvars を使用してリクエスト単位の trace_id を伝播する。
asyncio タスク間で自動的に継承されるため、
呼び出し元のシグネチャ変更なしで全デバッグログに trace_id を付与できる。
"""

from __future__ import annotations

import asyncio
import contextvars
import uuid
from collections import OrderedDict, deque
from concurrent.futures import Executor
from contextvars import ContextVar
from typing import Any, Callable, TypeVar

# リクエストスコープの trace_id（空文字列 = 非リクエストコンテキスト）
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def generate_trace_id() -> str:
    """UUID4 短縮形式の trace_id を生成する（12文字 hex）"""
    return uuid.uuid4().hex[:12]


def set_trace_id(trace_id: str) -> None:
    """現在のコンテキストに trace_id をセットする"""
    trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """現在のコンテキストの trace_id を取得する（未設定時は空文字列）"""
    return trace_id_var.get()


# このリクエストがプライベートセッションか。``trace_id`` と同じ理由で
# contextvar に置く — 発話本文をログへ書く地点は多数あり、そのすべてへ
# ``private`` を引数で配ることはできない。
private_var: ContextVar[bool] = ContextVar("private_session", default=False)


def set_private(private: bool) -> None:
    """現在のコンテキストにプライベートセッションかどうかをセットする"""
    private_var.set(bool(private))


def is_private() -> bool:
    """現在のコンテキストがプライベートセッションか（未設定時は False）"""
    return private_var.get()


# このリクエストで伏せるべきユーザー発話。private のときだけ設定し、ログ出力の
# フィルタ (``log_config.PrivateContentFilter``) が「この文字列と共通部分を持つ
# 行」を伏せるために使う。**発話を書く logger 呼び出しは十数箇所あり、その全てを
# 個別に直すのは漏れる** ため、出力の合流点で一括して落とす。
#
# **そのターンの発話だけでは足りない**。private セッションの 2 ターン目
# 「さっきの口座番号を教えて。」には番号そのものが含まれないが、記憶注入で
# 前ターンの値がプロンプトへ載り、``Quantity grounding injected: 口座番号 =
# 7778889990001`` という INFO 行に出た (実機再検証で観測)。セッション単位で
# 積む。
private_text_var: ContextVar[tuple[str, ...]] = ContextVar(
    "private_texts", default=(),
)

#: セッションごとの private 発話。件数・セッション数とも上限を置く
#: (``file_ledger`` と同じ立て付け)。
_MAX_PRIVATE_TEXTS_PER_SESSION = 20
_MAX_PRIVATE_SESSIONS = 16
_private_session_texts: "OrderedDict[str, deque[str]]" = OrderedDict()


def record_private_text(session_id: str, text: str) -> None:
    """private セッションの発話をセッション単位で積む。"""
    cleaned = (text or "").strip()
    if not session_id or not cleaned:
        return
    bucket = _private_session_texts.get(session_id)
    if bucket is None:
        bucket = deque(maxlen=_MAX_PRIVATE_TEXTS_PER_SESSION)
        _private_session_texts[session_id] = bucket
        while len(_private_session_texts) > _MAX_PRIVATE_SESSIONS:
            _private_session_texts.popitem(last=False)
    else:
        _private_session_texts.move_to_end(session_id)
    if cleaned not in bucket:
        bucket.append(cleaned)


def set_private_text(text: str, session_id: str = "") -> None:
    """このリクエストで伏せる発話を確定する（非 private では空にする）。

    ``session_id`` を渡すと、そのセッションで過去に話した private 発話も
    まとめて伏せる対象にする。
    """
    if not text:
        private_text_var.set(())
        return
    record_private_text(session_id, text)
    if session_id:
        private_text_var.set(tuple(_private_session_texts.get(session_id) or (text,)))
    else:
        private_text_var.set((text,))


def get_private_texts() -> tuple[str, ...]:
    """このリクエストで伏せる発話の一覧（未設定時は空タプル）"""
    return private_text_var.get()


def reset_private_texts(session_id: str | None = None) -> None:
    """積んだ private 発話を捨てる（``None`` で全消去）。テスト用。"""
    if session_id is None:
        _private_session_texts.clear()
        return
    _private_session_texts.pop(session_id, None)


_T = TypeVar("_T")


def run_in_executor_with_context(
    loop: asyncio.AbstractEventLoop,
    executor: Executor | None,
    func: Callable[..., _T],
    *args: Any,
) -> asyncio.Future[_T]:
    """`loop.run_in_executor` のラッパ。呼び出し元の contextvars を保全する

    `asyncio.AbstractEventLoop.run_in_executor` は `concurrent.futures.Executor`
    のワーカースレッドへ関数を投げるが、その際 `contextvars.Context` をコピー
    しないため、スレッド側から `get_trace_id()` を呼ぶと空文字列になる。
    本ヘルパは `contextvars.copy_context()` で現在のコンテキストを束ね、
    ワーカースレッド側で `ctx.run(func, *args)` として実行することで、
    `trace_id` を含む全 `ContextVar` をスレッド境界越しに伝播させる。
    """
    ctx = contextvars.copy_context()
    return loop.run_in_executor(executor, lambda: ctx.run(func, *args))
