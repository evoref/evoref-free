"""

互換の最小 PRD JSON フォーマットをパースして ``task`` 型 SemanticFact 群に
変換する。

サポート形式::

    {
      "tasks": [
        {
          "id": "t1",                       # 任意 (省略時は自動採番)
          "title": "DB スキーマ作成",
          "description": "...",             # 任意
          "depends_on": [],                 # 任意 (task_id の文字列リスト)
          "salience": 0.8                   # 任意 (0.0〜1.0、デフォルト 0.5)
        },
        ...
      ]
    }

簡易形式 (トップレベルが配列の場合) もサポートする::

    [
      {"id": "t1", "title": "..."},
      ...
    ]

設計原則:
- 純粋関数。I/O は ``parse_prd_json_file`` のみが行う
- バリデーションエラーは ``PRDParseError`` で集約報告 (1 件目で打ち切り)
- 後方互換不要

呼び出し側 (API / CLI) は本関数で生成された ``SemanticFact`` のリストを
``LoopFactView.writeback_store.add_fact`` 等で順次登録する
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.free.loop.driver import TASK_SUBJECT_PREFIX, make_task_fact
from backend.free.memory.types import SemanticFact
from backend.log_config import get_logger

logger = get_logger("loop.prd_parser")


class PRDParseError(ValueError):
    """PRD JSON の構造 / 値が不正なときに送出"""


def parse_prd_json(
    raw: str | bytes | dict[str, Any] | list[Any],
    *,
    project_id: str,
    source_path: str | None = None,
) -> list[SemanticFact]:
    """PRD JSON 文字列 (または既にパース済の dict/list) を task ファクト群に変換する。

    Args:
        raw: JSON 文字列・bytes または既にデシリアライズ済の Python オブジェクト
        project_id: タスクを所属させる project_id
        source_path: provenance 用 (PRD ファイルパス)

    Returns:
        ``task`` 型 SemanticFact のリスト (まだストアには追加されていない)

    Raises:
        PRDParseError: JSON が不正、または `tasks` 構造が期待形式でない
    """
    if not project_id:
        raise PRDParseError("project_id must be non-empty")

    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PRDParseError(f"PRD encoding error: {exc}") from exc
    if isinstance(raw, str):
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PRDParseError(f"PRD JSON parse error: {exc}") from exc
    else:
        data = raw

    tasks_raw = _extract_tasks_array(data)
    if not tasks_raw:
        raise PRDParseError("PRD contains no tasks")

    seen_ids: set[str] = set()
    facts: list[SemanticFact] = []
    for idx, item in enumerate(tasks_raw):
        if not isinstance(item, dict):
            raise PRDParseError(
                f"task[{idx}] must be an object, got {type(item).__name__}",
            )
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            raise PRDParseError(
                f"task[{idx}] missing required 'title'",
            )
        task_id = item.get("id")
        if task_id is not None:
            if not isinstance(task_id, str) or not task_id.strip():
                raise PRDParseError(
                    f"task[{idx}] 'id' must be a non-empty string",
                )
            task_id = task_id.strip()
            if task_id in seen_ids:
                raise PRDParseError(
                    f"task[{idx}] duplicate id: {task_id!r}",
                )
            seen_ids.add(task_id)

        description = item.get("description", "")
        if not isinstance(description, str):
            raise PRDParseError(
                f"task[{idx}] 'description' must be a string",
            )

        depends_on_raw = item.get("depends_on", [])
        if not isinstance(depends_on_raw, list):
            raise PRDParseError(
                f"task[{idx}] 'depends_on' must be an array",
            )
        depends_on: list[str] = []
        for j, dep in enumerate(depends_on_raw):
            if not isinstance(dep, str) or not dep:
                raise PRDParseError(
                    f"task[{idx}].depends_on[{j}] must be a non-empty string",
                )
            depends_on.append(dep)

        salience_raw = item.get("salience", 0.5)
        try:
            salience = float(salience_raw)
        except (TypeError, ValueError) as exc:
            raise PRDParseError(
                f"task[{idx}] 'salience' must be a number: {salience_raw!r}",
            ) from exc
        if not (0.0 <= salience <= 1.0):
            raise PRDParseError(
                f"task[{idx}] 'salience' must be in [0.0, 1.0]: {salience}",
            )

        fact = make_task_fact(
            project_id=project_id,
            task_id=task_id,
            title=title.strip(),
            description=description,
            depends_on=depends_on,
            salience=salience,
            source_path=source_path,
        )
        facts.append(fact)

    # 依存先が同 PRD 内に存在しない場合は WARN (エラーにはしない: 既に SemMem に
    # 登録済のタスクへの依存が許容されるため)
    declared_ids = {f.subject.removeprefix(TASK_SUBJECT_PREFIX) for f in facts}
    for fact in facts:
        try:
            body = json.loads(fact.object)
        except json.JSONDecodeError:
            continue
        for dep in body.get("depends_on", []):
            if dep not in declared_ids:
                logger.warning(
                    "PRD task %s depends on %s which is not declared in this PRD; "
                    "assuming it exists in SemMem",
                    body.get("task_id"), dep,
                )
    return facts


def parse_prd_json_file(
    path: str | Path,
    *,
    project_id: str,
) -> list[SemanticFact]:
    """ファイルから PRD JSON を読み込んで task ファクト群を返す。

    Raises:
        PRDParseError: ファイルが存在しない、JSON が不正、構造不正
    """
    p = Path(path)
    if not p.exists():
        raise PRDParseError(f"PRD file not found: {p}")
    if not p.is_file():
        raise PRDParseError(f"PRD path is not a file: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise PRDParseError(f"failed to read PRD file {p}: {exc}") from exc
    return parse_prd_json(text, project_id=project_id, source_path=str(p))


def _extract_tasks_array(data: Any) -> list[Any]:
    """PRD ペイロードから tasks 配列を取り出す。

    - dict なら ``data["tasks"]`` を返す (未指定時は ``PRDParseError``)
    - list ならそれ自体を返す (簡易形式)
    - それ以外は ``PRDParseError``
    """
    if isinstance(data, dict):
        tasks = data.get("tasks")
        if not isinstance(tasks, list):
            raise PRDParseError(
                "PRD JSON must contain 'tasks' array at top level",
            )
        return tasks
    if isinstance(data, list):
        return data
    raise PRDParseError(
        f"PRD root must be object or array, got {type(data).__name__}",
    )
