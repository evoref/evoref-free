"""Meta-Cognitive タスク管理: データクラス・タスク状態判定・タスクマージ"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.agent.credit_assigner import StepCredit

logger = get_logger("agent.meta_cognitive.tasks")


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class TaskItem:
    """タスクリストの1項目"""
    description: str
    status: str = "pending"  # "pending" | "done" | "failed"
    result: str = ""


@dataclass
class EditorArtifact:
    """エディタ出力用の生成コード片（ディスク書込せずフロントのエディタへ流す）"""
    content: str
    language: str = "python"
    filename: str | None = None


@dataclass
class MetaCognitiveResponse:
    """Meta-Cognitive 層の応答"""
    content: str
    tasks: list[TaskItem] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    steps: int = 0
    episode_id: str = ""
    step_credits: list[StepCredit] = field(default_factory=list)
    # 出力先パス未指定時にエディタペインへ流す生成コード（write_file は行わない）
    editor_artifacts: list[EditorArtifact] = field(default_factory=list)


# ---------------------------------------------------------------------------
# タスク状態判定
# ---------------------------------------------------------------------------

# 書き込み期待パターン（日本語・英語）
_WRITE_PATTERN = re.compile(
    r"作成|追加|実装|修正|変更|書き込|生成|更新|書く|書いて"
    r"|create|write|add|implement|modify|update|generate|fix|refactor",
    re.IGNORECASE,
)


def task_expects_write(description: str) -> bool:
    """タスク記述がファイル書き込みを期待しているか判定する

    「作成」「追加」「実装」「修正」などの動詞を含むタスクは
    write_file の実行が期待される。
    読み取り専用の記述（"Read foo.py"）は書き込み動詞を含まないため
    自然に False になる。

    注: ファイルパスの有無は問わない。パスなしでも「書き込み期待」は成立する
    （determine_task_status でのステータス判定で使用）。
    """
    return bool(_WRITE_PATTERN.search(description))


def determine_task_status(
    task: TaskItem, result: str, tool_calls: list[dict],
) -> str:
    """タスク実行結果からステータスを決定する"""
    if result.startswith("Error:") or "Step limit reached" in result:
        return "failed"

    if task_expects_write(task.description) and not any(
        tc.get("tool") == "write_file" and tc.get("success")
        for tc in tool_calls
    ):
        logger.warning(
            "Task marked failed: write expected but not executed: %s",
            task.description[:80],
        )
        return "failed"

    fetch_tools = {"fetch_url", "read_file", "search_code", "list_directory"}
    if any(
        tc.get("tool") in fetch_tools and tc.get("success")
        for tc in tool_calls
    ):
        return "done"

    return "done"


# ---------------------------------------------------------------------------
# タスクマージ
# ---------------------------------------------------------------------------

def merge_same_file_tasks(tasks: list[TaskItem]) -> list[TaskItem]:
    """同一ファイルを対象とする複数タスクを1つにマージする

    PLAN_SYSTEM_PROMPT で「1ファイル=1タスク」を指示しているが、
    ローカル LLM が無視して分割するケースへの防御策。
    """
    from backend.free.agent.tool_call_judge import _extract_file_path

    file_groups: dict[str, tuple[int, list[str]]] = {}
    for i, task in enumerate(tasks):
        path = _extract_file_path(task.description)
        if path:
            if path not in file_groups:
                file_groups[path] = (i, [task.description])
            else:
                file_groups[path][1].append(task.description)

    if all(len(descs) == 1 for _, descs in file_groups.values()):
        return tasks

    merged_indices: set[int] = set()
    result_items: list[tuple[int, TaskItem]] = []

    for path, (first_idx, descriptions) in file_groups.items():
        if len(descriptions) == 1:
            result_items.append((first_idx, tasks[first_idx]))
        else:
            merged_desc = " / ".join(descriptions)
            result_items.append((first_idx, TaskItem(description=merged_desc)))
            for j, task in enumerate(tasks):
                if _extract_file_path(task.description) == path:
                    merged_indices.add(j)

    for i, task in enumerate(tasks):
        if i not in merged_indices and not _extract_file_path(task.description):
            result_items.append((i, task))

    result_items.sort(key=lambda x: x[0])

    merged = [item for _, item in result_items]
    logger.info(
        "Task merging: %d tasks → %d tasks", len(tasks), len(merged),
    )
    return merged
