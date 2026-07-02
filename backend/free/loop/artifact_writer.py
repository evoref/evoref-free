"""

``SemanticFactStore`` 直参照を廃止し、書込は
:class:`~backend.free.memory.views.loop.LoopFactView` 経由に統一した。
subject prefix を ``harness.artifact.`` → ``loop.artifact.`` に移行済

- subject: ``loop.artifact.<task_id>.<path_sha1_prefix>``
- predicate: ``produced``
- type: ``artifact``
- scope: ``project:<project_id>``
- mode_origin: ``coding``

冪等性: 同一 ``diff_sha1`` + 同一 subject の artifact が既に存在する場合は
スキップする。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from backend.free.loop.executor import ArtifactEntry
from backend.free.memory.types import SemanticFact
from backend.free.memory.views.loop import (
    ARTIFACT_PREDICATE as _VIEW_ARTIFACT_PREDICATE,
    ARTIFACT_SUBJECT_PREFIX as _VIEW_ARTIFACT_SUBJECT_PREFIX,
    LoopFactView,
    build_artifact_subject as _view_build_artifact_subject,
)
from backend.log_config import get_logger

logger = get_logger("loop.artifact_writer")


# ──────────────────────────────────────────────────────────────────────────
# 定数 (view から再エクスポート)
# ──────────────────────────────────────────────────────────────────────────

ARTIFACT_SUBJECT_PREFIX = _VIEW_ARTIFACT_SUBJECT_PREFIX
ARTIFACT_PREDICATE = _VIEW_ARTIFACT_PREDICATE

MAX_OBJECT_LEN = 2000

PATH_SHA1_PREFIX_LEN = 12


# ──────────────────────────────────────────────────────────────────────────
# データ型
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArtifactPayload:
    """artifact ファクトの ``object`` フィールド JSON 表現 (テスト / 解釈用)。"""

    file_path: str
    diff_sha1: str
    lines_added: int
    lines_removed: int
    gate_passed: bool
    related_task_id: str
    related_progress_marker: str
    iteration: int | None
    created_at: float
    action_kind: str = "edit_file"

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "file_path": self.file_path,
            "diff_sha1": self.diff_sha1,
            "lines_added": int(self.lines_added),
            "lines_removed": int(self.lines_removed),
            "gate_passed": bool(self.gate_passed),
            "related_task_id": self.related_task_id,
            "related_progress_marker": self.related_progress_marker,
            "created_at": float(self.created_at),
            "action_kind": self.action_kind,
        }
        if self.iteration is not None:
            payload["iteration"] = int(self.iteration)
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(text) > MAX_OBJECT_LEN:
            payload["file_path"] = self.file_path[-200:]
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return text


def parse_artifact_object(text: str) -> dict[str, Any]:
    """``artifact.object`` を dict にパースする。"""
    if not text:
        return {}
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw": text}
    if not isinstance(data, dict):
        return {}
    return data


def build_artifact_subject(task_id: str, file_path: str) -> str:
    """``loop.artifact.<task_id>.<path_sha1_prefix>`` を返す (view ヘルパ再エクスポート)。"""
    return _view_build_artifact_subject(task_id, file_path)


# ──────────────────────────────────────────────────────────────────────────
# 即時書き込み (LoopFactView 委譲)
# ──────────────────────────────────────────────────────────────────────────


def write_artifact(
    view: LoopFactView,
    *,
    project_id: str,
    task_id: str,
    entry: ArtifactEntry,
    gate_passed: bool,
    iteration: int | None = None,
    trace_id: str | None = None,
    now: float | None = None,
) -> SemanticFact | None:
    """artifact ファクトを冪等に書き込む (LoopFactView 委譲)。"""
    fact = view.write_artifact(
        project_id=project_id,
        task_id=task_id,
        file_path=entry.path,
        diff_sha1=entry.diff_sha1,
        lines_added=entry.lines_added,
        lines_removed=entry.lines_removed,
        gate_passed=gate_passed,
        action_kind=entry.action_kind,
        iteration=iteration,
        trace_id=trace_id,
        now=now,
    )
    if fact is None:
        logger.debug(
            "write_artifact: skip duplicate task_id=%s path=%s diff_sha1=%s",
            task_id, entry.path, entry.diff_sha1,
        )
        return None
    logger.info(
        "write_artifact (add): task_id=%s path=%s diff_sha1=%s project=%s",
        task_id, entry.path, entry.diff_sha1, project_id,
    )
    return fact


def write_artifacts(
    view: LoopFactView,
    *,
    project_id: str,
    task_id: str,
    entries: Iterable[ArtifactEntry],
    gate_passed: bool,
    iteration: int | None = None,
    trace_id: str | None = None,
    now: float | None = None,
) -> list[SemanticFact]:
    """``ArtifactEntry`` のイテラブルを一括で書き込む。重複はスキップ。"""
    written: list[SemanticFact] = []
    for entry in entries:
        stored = write_artifact(
            view,
            project_id=project_id,
            task_id=task_id,
            entry=entry,
            gate_passed=gate_passed,
            iteration=iteration,
            trace_id=trace_id,
            now=now,
        )
        if stored is not None:
            written.append(stored)
    return written


def list_artifacts(
    view: LoopFactView,
    project_id: str,
    *,
    task_id: str | None = None,
) -> list[SemanticFact]:
    """指定プロジェクトの ``artifact`` ファクト (非 superseded) を返す。"""
    return view.list_artifact_facts(project_id, task_id=task_id)


# ──────────────────────────────────────────────────────────────────────────
# LoopDriver.artifact_hook 用ファクトリ
# ──────────────────────────────────────────────────────────────────────────


LoopViewProvider = Callable[[str], LoopFactView]


def make_loop_artifact_hook(
    view_provider: LoopViewProvider,
    *,
    iteration_getter: Callable[[], int | None] | None = None,
    trace_id_getter: Callable[[], str | None] | None = None,
) -> Callable[[str, Any, list[ArtifactEntry]], None]:
    """``LoopDriver(artifact_hook=...)`` に渡せる closure を作る。

    hook シグネチャは ``(project_id, task, artifacts) -> None`` で、
    ``project_id`` に対応する :class:`LoopFactView` を ``view_provider``
    から取得し、``artifacts`` を順次 ``write_artifact`` で永続化する。
    """

    def _hook(
        project_id: str, task: Any, artifacts: list[ArtifactEntry],
    ) -> None:
        if not artifacts:
            return
        try:
            view = view_provider(project_id)
        except Exception as exc:
            logger.warning(
                "artifact_hook: view lookup failed project=%s: %s",
                project_id, exc,
            )
            return
        task_id = getattr(task, "task_id", None) or ""
        if not task_id:
            logger.warning(
                "artifact_hook: task.task_id is empty, skipping %d artifacts",
                len(artifacts),
            )
            return
        iteration = iteration_getter() if iteration_getter else None
        trace_id = trace_id_getter() if trace_id_getter else None
        write_artifacts(
            view,
            project_id=project_id,
            task_id=task_id,
            entries=artifacts,
            gate_passed=True,
            iteration=iteration,
            trace_id=trace_id,
        )

    return _hook


__all__ = [
    "ARTIFACT_PREDICATE",
    "ARTIFACT_SUBJECT_PREFIX",
    "ArtifactPayload",
    "MAX_OBJECT_LEN",
    "PATH_SHA1_PREFIX_LEN",
    "build_artifact_subject",
    "list_artifacts",
    "make_loop_artifact_hook",
    "parse_artifact_object",
    "write_artifact",
    "write_artifacts",
]
