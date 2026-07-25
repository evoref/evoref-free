"""チャット機能の内部型定義

公開API（schemas.py の Pydantic モデル）は変更せず、
内部で使用する TypedDict・型エイリアスをここに集約する。
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, TypedDict


class ChatMessage(TypedDict):
    """LLM に渡す messages 配列の要素"""

    role: str  # "system" | "user" | "assistant"
    content: str


class GenerationParams(TypedDict, total=False):
    """モード別生成パラメータ（temperature, top_p 等）

    config.yaml の modes.chat / modes.coding から取得される値。
    全フィールドはオプショナル。
    """

    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    frequency_penalty: float
    model: str | None


class StepEvent(TypedDict, total=False):
    """on_step コールバックに渡すステップイベント"""

    type: str       # "plan" | "tool_call" | "task_result" | "long_form_plan" 等
    detail: str     # 表示用テキスト
    status: str     # "running" | "done" | "failed"


# on_step コールバックの型
# 同期版: def on_step(step_data: StepEvent) -> None
# 非同期版: async def on_step(step_data: StepEvent) -> None
SyncStepCallback = Callable[[dict], None]
AsyncStepCallback = Callable[[dict], Coroutine[Any, Any, None]]
StepCallback = SyncStepCallback | AsyncStepCallback | None


class FileContextDict(TypedDict):
    """file_contexts の内部表現（schemas.py の FileContext とは別）"""

    filename: str
    chunks: list[str]
