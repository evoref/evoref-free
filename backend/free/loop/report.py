"""

``SemanticFactStore`` 直参照を廃止し、集計は
:class:`~backend.free.memory.views.loop.LoopFactView` 経由に統一した。

レポート内容:
- 全タスク数 / open / in_progress / done / failed の内訳
- iterations (反復数) — driver state 由来 or 終端タスク数 (done + failed)
- elapsed_seconds — ``started_at`` からの経過時間
- failure_pattern 統計 — error_type 別 occurrences 合計
- progress_marker count
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from backend.free.loop.driver import list_tasks
from backend.free.loop.executor import ArtifactEntry
from backend.free.loop.progress_marker import list_progress_markers
from backend.free.memory.views.loop import LoopFactView
from backend.log_config import get_logger

logger = get_logger("loop.report")


# ──────────────────────────────────────────────────────────────────────────
# LoopReport DTO
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LoopReport:
    """自律ループの実行スナップショットレポート。"""

    project_id: str
    total_tasks: int
    open_tasks: int
    in_progress_tasks: int
    done_tasks: int
    failed_tasks: int
    iterations: int
    started_at: float | None
    elapsed_seconds: float | None
    generated_at: float
    failure_pattern_total: int
    progress_marker_count: int
    failure_pattern_by_error_type: dict[str, int] = field(default_factory=dict)
    artifacts: list[ArtifactEntry] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "total_tasks": self.total_tasks,
            "open_tasks": self.open_tasks,
            "in_progress_tasks": self.in_progress_tasks,
            "done_tasks": self.done_tasks,
            "failed_tasks": self.failed_tasks,
            "iterations": self.iterations,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
            "generated_at": self.generated_at,
            "failure_pattern_total": self.failure_pattern_total,
            "failure_pattern_by_error_type": dict(
                self.failure_pattern_by_error_type,
            ),
            "progress_marker_count": self.progress_marker_count,
            "artifacts": [a.as_dict() for a in self.artifacts],
        }


# ──────────────────────────────────────────────────────────────────────────
# 集計関数
# ──────────────────────────────────────────────────────────────────────────


def aggregate_failure_patterns(
    view: LoopFactView,
    project_id: str,
) -> tuple[int, dict[str, int]]:
    """指定プロジェクトの ``failure_pattern`` ファクトを集計する。

    Returns:
        ``(total_occurrences, {error_type: occurrences})``
    """
    return view.aggregate_failure_patterns(project_id)


def generate_loop_report(
    view: LoopFactView,
    *,
    project_id: str,
    started_at: float | None = None,
    iterations: int | None = None,
    artifacts: list[ArtifactEntry] | None = None,
    now: float | None = None,
) -> LoopReport:
    """自律ループの ``LoopReport`` を生成する。"""
    if not project_id:
        raise ValueError("project_id must be non-empty")
    ts = float(now) if now is not None else time.time()
    tasks = list_tasks(view, project_id)
    open_count = sum(1 for t in tasks if t.status == "open")
    in_progress_count = sum(1 for t in tasks if t.status == "in_progress")
    done_count = sum(1 for t in tasks if t.status == "done")
    failed_count = sum(1 for t in tasks if t.status == "failed")

    failure_total, failure_by_type = aggregate_failure_patterns(view, project_id)
    progress_marker_count = len(list_progress_markers(view, project_id))

    elapsed: float | None = None
    if started_at is not None:
        elapsed = max(0.0, ts - float(started_at))

    iter_count = (
        int(iterations)
        if iterations is not None
        else done_count + failed_count
    )

    report = LoopReport(
        project_id=project_id,
        total_tasks=len(tasks),
        open_tasks=open_count,
        in_progress_tasks=in_progress_count,
        done_tasks=done_count,
        failed_tasks=failed_count,
        iterations=iter_count,
        started_at=started_at,
        elapsed_seconds=elapsed,
        generated_at=ts,
        failure_pattern_total=failure_total,
        progress_marker_count=progress_marker_count,
        failure_pattern_by_error_type=failure_by_type,
        artifacts=list(artifacts) if artifacts else [],
    )
    logger.info(
        "generate_loop_report: project=%s total=%d done=%d failed=%d "
        "iterations=%d failures=%d markers=%d",
        project_id, report.total_tasks, report.done_tasks, report.failed_tasks,
        report.iterations, report.failure_pattern_total,
        report.progress_marker_count,
    )
    return report


__all__ = [
    "LoopReport",
    "aggregate_failure_patterns",
    "generate_loop_report",
]
