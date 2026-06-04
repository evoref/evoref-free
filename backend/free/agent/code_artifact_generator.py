"""Agentic 経路がソースコード生成を委譲するための Protocol。

``MetaCognitiveAgent`` (EvorefLoop) は実装を持たず、composition 層が
``LongFormOrchestrator`` (EvorefGen) を包んだ実装を注入する。これにより
agent は generation pillar を直接 import せず、テストではフェイク実装を
差し込める。指示文を細粒度 CodeUnit 計画で生成し、ファイル別の
``EditorArtifact`` 群 (複数ファイル可) を返す。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from backend.free.agent.meta_cognitive_tasks import EditorArtifact

# Meta-Cognitive の on_step コールバック (同期/非同期どちらも可、省略可)。
StepCallback = Callable[[dict], Awaitable[None] | None]


class CodeArtifactGenerator(Protocol):
    """指示文から検証・修正済みのコードファイル群を生成する委譲先。

    返り値が空リストの場合、呼出側 (``_execute_editor_task``) は従来の
    単一ショット生成 (``_generate_content``) にフォールバックする。
    """

    async def __call__(
        self,
        instruction: str,
        on_step: StepCallback | None = None,
    ) -> list[EditorArtifact]:
        ...
