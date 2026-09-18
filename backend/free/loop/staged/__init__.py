"""staged クリエイトパイプライン (EvorefLoop pillar)。

`evoref create` のリクエストを仕様書(spec)→コード(code)→テスト(test) の
タスクグラフに分解し、専用 ``LoopDriver`` インスタンスでインライン駆動する。
各工程は独立した LLM パスとして実行され、中間成果物は temp ワークスペース
(:class:`WorkspaceManager`) で工程間共有される。

公開シンボル:

- :func:`synthesize_create_task_graph` — request → spec/code/test task ファクト群
- :func:`synthesize_create_task_graph_with_plan` — 同上 + planner の
  ``module_deps`` (f_10 §2、Phase 2.5)
- :class:`StagedCreateExecutor` — stage 別 TaskExecutor
- :class:`WorkspaceManager` — temp ワークスペース管理
- :data:`SPEC_TASK_ID` — spec タスクの固定 task_id
- run レコード + 追記イベントログ (f_10 §7、2026-09-18): :class:`RunRecord` /
  :class:`RunRecordStore` / :class:`RunEvent` / :class:`RunEventLog` /
  :func:`derive_run_status` / :func:`list_runs` / :func:`load_run` /
  :func:`read_events` / :func:`gc_old_runs`
- :class:`StagedCodeHarness` — ``create.dispatch=meta`` 用 ``ProductionHarness``
  アダプタ (f_03 §4.4、Phase 3a)
- :class:`StagedRunRecorder` — ``LongFormHarness`` (EvorefGen) 向け ``RunRecorder``
  Protocol 実装 (f_08 §2.3、Phase 4)
"""

from __future__ import annotations

from backend.free.loop.staged.harness import StagedCodeHarness
from backend.free.loop.staged.run_record import (
    RunEvent,
    RunEventLog,
    RunRecord,
    RunRecordStore,
    derive_run_status,
    gc_old_runs,
    list_runs,
    load_run,
    read_events,
)
from backend.free.loop.staged.run_recorder import StagedRunRecorder
from backend.free.loop.staged.synthesizer import (
    SPEC_TASK_ID,
    synthesize_create_task_graph,
    synthesize_create_task_graph_with_plan,
)
from backend.free.loop.staged.workspace import WorkspaceManager

__all__ = [
    "SPEC_TASK_ID",
    "RunEvent",
    "RunEventLog",
    "RunRecord",
    "RunRecordStore",
    "StagedCodeHarness",
    "StagedRunRecorder",
    "WorkspaceManager",
    "derive_run_status",
    "gc_old_runs",
    "list_runs",
    "load_run",
    "read_events",
    "synthesize_create_task_graph",
    "synthesize_create_task_graph_with_plan",
]
