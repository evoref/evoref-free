"""StagedRunRecorder — longform 向け ``RunRecorder`` Protocol 実装 (Phase 4、f_08 §2.3)。

``WorkspaceManager`` + ``RunRecordStore`` + ``RunEventLog`` (いずれも staged と
共有する部品) を束ね、``backend.free.harness.production.RunRecorder`` Protocol
を満たすアダプタ。``run_staged_pipeline`` はこれらの部品を直接使い続ける
(共通化は 3a-2 で検討、本モジュールは変えない)。composition 層 (``chat.py``)
が ``LongFormHarness`` の ``run_recorder_factory`` として注入する。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.free.loop.staged.run_record import RunEventLog, RunRecordStore
from backend.free.loop.staged.workspace import WorkspaceManager
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("loop.staged.run_recorder")

__all__ = ["StagedRunRecorder"]


class StagedRunRecorder:
    """``WorkspaceManager``/``RunRecordStore``/``RunEventLog`` を束ねる ``RunRecorder``。

    longform の生成物は staged と同じ ``src/<path>`` へ書く (``kind="src"``)。
    """

    def __init__(
        self,
        workspace: WorkspaceManager,
        run_store: RunRecordStore,
        event_log: RunEventLog,
    ) -> None:
        self._workspace = workspace
        self._run_store = run_store
        self._event_log = event_log

    @property
    def root(self) -> Path:
        return self._workspace.root

    @classmethod
    def open(
        cls,
        create_workspace_dir: Path | str,
        run_id: str,
        *,
        session_id: str,
        project_id: str = "longform",
        goal: str = "",
        debug_logger: "DebugLogger | None" = None,
    ) -> "StagedRunRecorder":
        """``{create_workspace_dir}/{run_id}`` を生成し空の recorder を返す。

        ``run_id`` == ``workspace_id`` (staged と同じ形、f_10 §7)。
        """
        workspace = WorkspaceManager.open_or_create(
            create_workspace_dir, workspace_id=run_id, session_id=session_id,
            project_id=project_id, goal=goal, debug_logger=debug_logger,
        )
        run_store = RunRecordStore(workspace.root)
        event_log = RunEventLog(workspace.root, debug_logger=debug_logger)
        return cls(workspace, run_store, event_log)

    def start(
        self,
        *,
        session_id: str,
        request_id: str,
        mode: str,
        query: str,
        output_target: str,
        brief_tokens: int = 0,
        resume_of: str | None = None,
    ) -> None:
        """run 開始を記録し即座に永続化する。"""
        self._run_store.start(
            run_id=self._workspace.workspace_id, session_id=session_id,
            request_id=request_id, mode=mode, query=query,
            output_target=output_target, brief_tokens=brief_tokens,
            resume_of=resume_of,
        )

    def append_event(self, kind: str, payload: dict[str, Any]) -> None:
        """1 イベントを ``events.jsonl`` へ追記する。"""
        self._event_log.append(kind, payload)

    def write_src(self, path: str, content: str) -> None:
        """生成物の途中経過 / 確定本文を ``src/<path>`` へ書き戻す (AtomicWriter 経由)。"""
        self._workspace.write_file(
            path, content, kind="src", stage="code", task_id="longform",
        )

    def finish(self, exit_kind: str) -> None:
        """run 終端を記録し即座に永続化する。"""
        self._run_store.finish(exit_kind, last_event_seq=max(0, self._event_log.last_seq))
