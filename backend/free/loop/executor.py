"""

``LoopDriver.run()`` が周回の各イテレーションでタスクを消化するための
実行インタフェース。`Harness` / `RalphExecutor` 等、実装は
差し替え可能。

責務:

- 1 件の task (``TaskFactView``) を受け取って実行結果 (``ExecutionOutcome``)
  を返す
- **SemMem への書き込み (failure_pattern / progress_marker / task.status 遷移)
  は `LoopDriver` 側で一元的に行う**。executor は outcome を返すだけに徹し、
  副作用の所在を単純化する。
- quality_gate 実行は executor 側に含めてよい (Ralph 実装で gate 結果を
  失敗原因の根拠として使うため)。Gate の結果は ``ExecutionOutcome.gate_outcome``
  に載せる。

本モジュールでは Protocol と ``NoOpExecutor`` のみを提供する。
実行 body (ファイル編集 / コマンド実行 / LLM 駆動) は ``RalphExecutor``
として差し込まれる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from backend.free.harness.action import Action, ActionResult
    from backend.free.loop.driver import TaskFactView
    from backend.free.loop.quality_gate import QualityGateOutcome


ExecutionStatus = Literal["success", "failure", "skipped"]
"""``ExecutionOutcome.status`` の取り得る値。

- ``success`` : すべての Action が成功し、quality_gate も通った (または gate なし)
- ``failure`` : いずれかの Action が失敗、または quality_gate が失敗
- ``skipped`` : executor が実行自体を見送った (Action が空 / noop のみ 等)
"""


@dataclass(frozen=True)
class ArtifactEntry:
    """Action によって生成・変更された成果物メタデータ

    ``RalphExecutor`` / ``ActionRunner`` が編集した実ファイルパス、差分の
    SHA1 (12 桁)、追加 / 削除行数を LoopReport に集計するためのフック。
    SemMem への ``artifact`` ファクト書き込みは別 Issue で実装予定
    """

    path: str
    diff_sha1: str
    lines_added: int
    lines_removed: int
    action_kind: str = "edit_file"

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "diff_sha1": self.diff_sha1,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "action_kind": self.action_kind,
        }


@dataclass(frozen=True)
class ExecutionOutcome:
    """TaskExecutor が 1 タスク実行後に返す DTO。

    - ``status`` : success / failure / skipped
    - ``actions`` : 実行された Action 列 (LLM が生成したもの。実行失敗で
      途中停止した場合はそこまで)
    - ``action_results`` : 各 Action の ActionResult。``actions`` と同じ長さ
    - ``gate_outcome`` : 品質ゲート結果 (スキップ executor では ``None``)
    - ``artifacts`` : 編集・生成されたファイルの成果物メタ
    - ``error`` : 失敗原因の 1 行サマリ (LoopDriver が failure_pattern 記録に使う)
    - ``notes`` : executor 固有の補足情報 (デバッグ表示用)
    """

    status: ExecutionStatus
    actions: tuple["Action", ...] = ()
    action_results: tuple["ActionResult", ...] = ()
    gate_outcome: "QualityGateOutcome | None" = None
    artifacts: tuple[ArtifactEntry, ...] = ()
    error: str | None = None
    notes: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "num_actions": len(self.actions),
            "num_artifacts": len(self.artifacts),
            "gate_ok": (
                self.gate_outcome.ok if self.gate_outcome is not None else None
            ),
            "error": self.error,
            "notes": dict(self.notes),
        }


class TaskExecutor(Protocol):
    """自律ループが 1 件の task を消化するための Protocol。

    実装は副作用 (ファイル編集 / コマンド実行 / LLM 呼び出し) を担ってよいが、
    SemMem への ``failure_pattern`` / ``progress_marker`` / ``task.status``
    書き込みは行わない。これらは ``LoopDriver`` が ``ExecutionOutcome`` を
    見て一元管理する。
    """

    name: str

    async def execute(self, task: "TaskFactView") -> ExecutionOutcome:
        """1 件の task を実行し ``ExecutionOutcome`` を返す。"""
        ...


class NoOpExecutor:
    """何もしない executor。`TaskExecutor` Protocol の参照実装

    周回ロジックを結線する E2E テストや、`config.loop.executor=noop` で
    本番外の dry-run を行う用途に使う。常に ``status="success"`` を返し
    ``task.status`` が ``done`` に遷移する。
    """

    name: str = "noop"

    async def execute(self, task: "TaskFactView") -> ExecutionOutcome:  # noqa: ARG002
        return ExecutionOutcome(
            status="success",
            notes={"executor": "noop"},
        )


__all__ = [
    "ArtifactEntry",
    "ExecutionOutcome",
    "ExecutionStatus",
    "NoOpExecutor",
    "TaskExecutor",
]
