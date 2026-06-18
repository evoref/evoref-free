"""staged コーディングパイプライン (EvorefLoop pillar)。

`evoref code` のリクエストを仕様書(spec)→コーディング(code)→テスト(test) の
タスクグラフに分解し、専用 ``LoopDriver`` インスタンスでインライン駆動する。
各工程は独立した LLM パスとして実行され、中間成果物は temp ワークスペース
(:class:`WorkspaceManager`) で工程間共有される。

公開シンボル:

- :func:`synthesize_coding_task_graph` — request → spec/code/test task ファクト群
- :class:`StagedCodingExecutor` — stage 別 TaskExecutor
- :class:`WorkspaceManager` — temp ワークスペース管理
- :data:`SPEC_TASK_ID` — spec タスクの固定 task_id
"""

from __future__ import annotations

from backend.free.loop.staged.synthesizer import (
    SPEC_TASK_ID,
    synthesize_coding_task_graph,
)
from backend.free.loop.staged.workspace import WorkspaceManager

__all__ = [
    "SPEC_TASK_ID",
    "synthesize_coding_task_graph",
    "WorkspaceManager",
]
