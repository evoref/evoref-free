"""

``SemanticFactStore`` 直参照を廃止し、書込は
:class:`~backend.free.memory.views.loop.LoopFactView` 経由に統一した。
subject prefix を ``harness.progress.`` → ``loop.progress.`` に移行済

- subject: ``loop.progress.<task_id>``
- predicate: ``reached``
- type: ``progress_marker``
- scope: ``project:<project_id>``
- mode_origin: ``create``

冪等性: 同一 ``task_id`` の progress_marker が既に存在する場合は
``occurrences`` を +1 して ``update_fact`` する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from backend.free.memory.types import SemanticFact
from backend.free.memory.views.loop import (
    PROGRESS_MARKER_PREFIX as _VIEW_PROGRESS_MARKER_PREFIX,
    PROGRESS_PREDICATE as _VIEW_PROGRESS_PREDICATE,
    LoopFactView,
)
from backend.log_config import get_logger

logger = get_logger("loop.progress_marker")


# ──────────────────────────────────────────────────────────────────────────
# 定数 (view から再エクスポート)
# ──────────────────────────────────────────────────────────────────────────

PROGRESS_MARKER_PREFIX = _VIEW_PROGRESS_MARKER_PREFIX
PROGRESS_PREDICATE = _VIEW_PROGRESS_PREDICATE

MAX_OBJECT_LEN = 2000


# ──────────────────────────────────────────────────────────────────────────
# データ型
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProgressPayload:
    """progress_marker ファクトの ``object`` フィールド JSON 表現。

    、書込は :meth:`LoopFactView.write_progress_marker` が担うが
    本 DTO は外部ツール (LoopReport / CLI) の JSON 解釈用に保持する。
    """

    task_id: str
    title: str
    status: str
    completed_at: float
    occurrences: int = 1
    iteration: int | None = None
    trace_id: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status,
            "completed_at": float(self.completed_at),
            "occurrences": int(self.occurrences),
        }
        if self.iteration is not None:
            payload["iteration"] = int(self.iteration)
        if self.trace_id is not None:
            payload["trace_id"] = self.trace_id
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(text) > MAX_OBJECT_LEN:
            payload["title"] = (self.title or "")[:200]
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return text


def parse_progress_object(text: str) -> dict[str, Any]:
    """``progress_marker.object`` を dict にパースする。"""
    if not text:
        return {"occurrences": 1}
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"occurrences": 1, "raw": text}
    if not isinstance(data, dict):
        return {"occurrences": 1}
    data.setdefault("occurrences", 1)
    return data


# ──────────────────────────────────────────────────────────────────────────
# 即時書き込み (LoopFactView 委譲)
# ──────────────────────────────────────────────────────────────────────────


def write_progress_marker(
    view: LoopFactView,
    *,
    project_id: str,
    task_id: str,
    title: str = "",
    status: str = "done",
    iteration: int | None = None,
    trace_id: str | None = None,
    confidence: float = 1.0,
    now: float | None = None,
) -> SemanticFact:
    """progress_marker ファクトを idempotent に書き込む (LoopFactView 委譲)。"""
    fact = view.write_progress_marker(
        project_id=project_id,
        task_id=task_id,
        title=title,
        status=status,
        iteration=iteration,
        trace_id=trace_id,
        confidence=confidence,
        now=now,
    )
    logger.info(
        "write_progress_marker: subject=%s%s task_id=%s project=%s",
        PROGRESS_MARKER_PREFIX, task_id, task_id, project_id,
    )
    return fact


def list_progress_markers(
    view: LoopFactView,
    project_id: str,
) -> list[SemanticFact]:
    """指定プロジェクトの全 progress_marker ファクト (非 superseded) を返す。"""
    return view.list_progress_marker_facts(project_id)


__all__ = [
    "MAX_OBJECT_LEN",
    "PROGRESS_MARKER_PREFIX",
    "PROGRESS_PREDICATE",
    "ProgressPayload",
    "list_progress_markers",
    "parse_progress_object",
    "write_progress_marker",
]
