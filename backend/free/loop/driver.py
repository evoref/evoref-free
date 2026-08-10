"""

EvorefMem 統合仕様 + 4 pillar 化 における自律実行ループ
``SemanticFactStore`` 直参照を廃止し、``LoopFactView``
(``backend.free.memory.views.loop.LoopFactView``) 経由に統一した。

提供する機能:

1. ``task`` 型 SemanticFact のエンコード / デコード (``TaskFactView``)
2. ``LoopDriver`` — ``view_provider: Callable[[str], LoopFactView]`` を DI で
   受け取り、周回実行 / pause / resume / bootstrap / orphan 回収を制御する

task の物理表現 (``make_task_fact`` / ``encode_task_object`` で生成):

- ``type=task`` / ``scope=project:<project_id>`` / ``mode_origin=create``
- ``subject=task.<task_id>`` (``task_id`` はユーザ定義 or 自動採番 ``t_<8hex>``)
- ``predicate=defines``
- ``object`` は JSON 文字列で本体 (title / description / depends_on / salience /
  status / source_path) を保持
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
from uuid import uuid4

from backend.free.loop.bootstrap import (
    BootstrapResult,
    MaybeResetReport,
    maybe_reset_and_bootstrap,
)
from backend.free.loop.events import LoopEventBus
from backend.free.loop.executor import (
    ArtifactEntry,
    ExecutionOutcome,
    TaskExecutor,
)
from backend.free.memory.stores.short_term import ShortTermMemory
from backend.free.memory.types import SemanticFact, TaskStatus, make_fact
from backend.free.memory.views.loop import (
    TASK_PREDICATE as _LOOP_TASK_PREDICATE,
    TASK_SUBJECT_PREFIX as _LOOP_TASK_SUBJECT_PREFIX,
    LoopFactView,
)
from backend.free.memory.stores.working import WorkingMemory
from backend.log_config import get_logger
from backend.trace_context import generate_trace_id, set_trace_id

logger = get_logger("loop.driver")


# ──────────────────────────────────────────────────────────────────────────
# 定数 (view の定数を再エクスポート)
# ──────────────────────────────────────────────────────────────────────────

TASK_SUBJECT_PREFIX = _LOOP_TASK_SUBJECT_PREFIX
"""SemanticFact.subject の task ID プレフィックス (``task.``)"""

TASK_PREDICATE = _LOOP_TASK_PREDICATE
"""task ファクトの predicate (``defines``)"""

VALID_TASK_STATUSES: frozenset[str] = frozenset(
    {"open", "in_progress", "done", "failed"},
)
"""許可される task status (TaskStatus Literal と同期)"""

TERMINAL_TASK_STATUSES: frozenset[str] = frozenset({"done", "failed"})
"""ライフサイクル終端ステータス"""


TaskStage = Literal["spec", "code", "test"]
"""staged クリエイトパイプラインの工程種別 (spec→code→test)。

通常の PRD / ralph タスクは ``stage`` を持たず ``None`` になる。``StagedCreateExecutor``
のみが ``stage`` 別に分岐し、それ以外の executor / driver ロジックは ``stage`` を無視
する (後方互換)。
"""

VALID_TASK_STAGES: frozenset[str] = frozenset({"spec", "code", "test"})
"""許可される task stage (TaskStage Literal と同期)"""


LoopViewProvider = Callable[[str], LoopFactView]
"""``project_id`` を受け取り対応する :class:`LoopFactView` を返す callable。"""


# ──────────────────────────────────────────────────────────────────────────
# 例外
# ──────────────────────────────────────────────────────────────────────────


class LoopNotRunningError(RuntimeError):
    """ループが起動していない状態で stop / status 操作を行ったとき"""


class TaskFactDecodeError(ValueError):
    """task ファクトの ``object`` フィールドが期待形式でないとき"""


# ──────────────────────────────────────────────────────────────────────────
# task ファクトのビュー (純粋データ)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class TaskFactView:
    """``task`` 型 SemanticFact の構造化ビュー (純粋データ)。

    SemanticFact のトップレベルを変更せず、``object`` フィールドを JSON として
    パースしたもの。``fact_id`` は SemanticFact の id (``sf_xxx``)、``task_id`` は
    PRD やユーザが指定する人間可読 ID (``t1``, ``setup-db`` など)。
    """

    fact_id: str
    task_id: str
    title: str
    description: str
    depends_on: list[str]
    salience: float
    status: TaskStatus
    source_path: str | None
    project_id: str
    created_at: float
    accessed_at: float
    stage: TaskStage | None = None

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_TASK_STATUSES


# ──────────────────────────────────────────────────────────────────────────
# エンコード / デコード (純粋関数)
# ──────────────────────────────────────────────────────────────────────────


def encode_task_object(
    *,
    task_id: str,
    title: str,
    description: str = "",
    depends_on: list[str] | None = None,
    salience: float = 0.5,
    status: TaskStatus = "open",
    source_path: str | None = None,
    stage: TaskStage | None = None,
) -> str:
    """task ファクトの ``object`` フィールド (JSON 文字列) を生成する。

    Raises:
        ValueError: ``task_id`` が空、``status`` が不正、``salience`` が範囲外、
            ``stage`` が不正
    """
    if not task_id:
        raise ValueError("task_id must be non-empty")
    if status not in VALID_TASK_STATUSES:
        raise ValueError(f"invalid task status: {status!r}")
    if not (0.0 <= float(salience) <= 1.0):
        raise ValueError(
            f"salience must be in [0.0, 1.0]: got {salience!r}",
        )
    if stage is not None and stage not in VALID_TASK_STAGES:
        raise ValueError(f"invalid task stage: {stage!r}")
    payload: dict[str, Any] = {
        "task_id": task_id,
        "title": title,
        "description": description,
        "depends_on": list(depends_on or []),
        "salience": float(salience),
        "status": status,
    }
    if source_path is not None:
        payload["source_path"] = source_path
    if stage is not None:
        payload["stage"] = stage
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def decode_task_fact(fact: SemanticFact) -> TaskFactView:
    """SemanticFact から ``TaskFactView`` を構築する。

    Raises:
        TaskFactDecodeError: type が task でない、scope が project でない、
            object が JSON でない、必須フィールド欠損、status 不正、など
    """
    if fact.type != "task":
        raise TaskFactDecodeError(
            f"fact {fact.id} is not a task (type={fact.type!r})",
        )
    project_id = fact.project_id()
    if not project_id:
        raise TaskFactDecodeError(
            f"task fact {fact.id} must be project-scoped (scope={fact.scope!r})",
        )
    try:
        body = json.loads(fact.object)
    except (TypeError, json.JSONDecodeError) as exc:
        raise TaskFactDecodeError(
            f"task fact {fact.id} object is not valid JSON: {exc}",
        ) from exc
    if not isinstance(body, dict):
        raise TaskFactDecodeError(
            f"task fact {fact.id} object must be a JSON object",
        )
    task_id = body.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise TaskFactDecodeError(
            f"task fact {fact.id} missing task_id",
        )
    status = body.get("status", "open")
    if status not in VALID_TASK_STATUSES:
        raise TaskFactDecodeError(
            f"task fact {fact.id} has invalid status {status!r}",
        )
    depends_raw = body.get("depends_on", [])
    if not isinstance(depends_raw, list):
        raise TaskFactDecodeError(
            f"task fact {fact.id} depends_on must be a list",
        )
    depends_on = [str(d) for d in depends_raw]
    salience_raw = body.get("salience", 0.5)
    try:
        salience = float(salience_raw)
    except (TypeError, ValueError) as exc:
        raise TaskFactDecodeError(
            f"task fact {fact.id} salience invalid: {salience_raw!r}",
        ) from exc
    stage_raw = body.get("stage")
    if stage_raw is not None and stage_raw not in VALID_TASK_STAGES:
        raise TaskFactDecodeError(
            f"task fact {fact.id} has invalid stage {stage_raw!r}",
        )
    return TaskFactView(
        fact_id=fact.id,
        task_id=task_id,
        title=str(body.get("title", "")),
        description=str(body.get("description", "")),
        depends_on=depends_on,
        salience=salience,
        status=status,  # type: ignore[arg-type]
        source_path=body.get("source_path"),
        project_id=project_id,
        created_at=fact.created_at,
        accessed_at=fact.accessed_at,
        stage=stage_raw,
    )


def make_task_fact(
    *,
    project_id: str,
    task_id: str | None = None,
    title: str,
    description: str = "",
    depends_on: list[str] | None = None,
    salience: float = 0.5,
    source_path: str | None = None,
    stage: TaskStage | None = None,
    now: float | None = None,
) -> SemanticFact:
    """``task`` 型 SemanticFact を生成する (ストアには追加しない)。

    ``task_id`` 未指定なら ``t_<8hex>`` を自動採番する。
    """
    if not project_id:
        raise ValueError("project_id must be non-empty")
    tid = task_id or f"t_{uuid4().hex[:8]}"
    object_json = encode_task_object(
        task_id=tid,
        title=title,
        description=description,
        depends_on=depends_on,
        salience=salience,
        status="open",
        source_path=source_path,
        stage=stage,
    )
    return make_fact(
        subject=f"{TASK_SUBJECT_PREFIX}{tid}",
        predicate=TASK_PREDICATE,
        object_=object_json,
        type="task",
        scope=SemanticFact.make_project_scope(project_id),
        mode_origin="create",
        confidence=1.0,
        now=now,
    )


# ──────────────────────────────────────────────────────────────────────────
# View 経由のタスク操作ヘルパ
# ──────────────────────────────────────────────────────────────────────────


def list_tasks(
    view: LoopFactView,
    project_id: str,
    *,
    status_filter: set[str] | None = None,
) -> list[TaskFactView]:
    """対象プロジェクトの task ファクトを ``TaskFactView`` のリストで返す。

    superseded ファクトは除外する (view.list_task_facts が除外済み)。デコードに
    失敗したファクトは WARN ログを出してスキップする。
    """
    out: list[TaskFactView] = []
    for fact in view.list_task_facts(project_id, status_filter=status_filter):
        try:
            out.append(decode_task_fact(fact))
        except TaskFactDecodeError as exc:
            logger.warning("skipping malformed task fact %s: %s", fact.id, exc)
    return out


def pick_next_task(
    view: LoopFactView,
    project_id: str,
) -> TaskFactView | None:
    """次に実行すべき task を依存グラフ考慮で選定する (``LoopFactView`` 委譲)。"""
    fact = view.pick_next_task_with_deps(project_id)
    if fact is None:
        return None
    try:
        return decode_task_fact(fact)
    except TaskFactDecodeError as exc:
        logger.warning(
            "pick_next_task: malformed task fact %s: %s", fact.id, exc,
        )
        return None


def update_task_status(
    view: LoopFactView,
    fact_id: str,
    status: TaskStatus,
) -> TaskFactView:
    """task ファクトの status を更新し、SemMem に即書き込みする。

    ``done`` 遷移時は progress_marker を自動的に追記する (冪等)。

    Args:
        view: 対象スコープの ``LoopFactView``
        fact_id: SemanticFact の id (``sf_xxx``)
        status: 新しい status (``open`` / ``in_progress`` / ``done`` / ``failed``)

    Raises:
        KeyError: fact が存在しない
        TaskFactDecodeError: 既存の object が壊れている
        ValueError: status が不正、または許可されない遷移
    """
    if status not in VALID_TASK_STATUSES:
        raise ValueError(f"invalid task status: {status!r}")
    fact = view.get_fact(fact_id)
    if fact is None:
        raise KeyError(f"task fact not found: {fact_id}")
    current = decode_task_fact(fact)
    if current.status == status:
        return current
    if not _is_allowed_transition(current.status, status):
        raise ValueError(
            f"illegal task status transition: {current.status} -> {status} "
            f"(fact_id={fact_id})",
        )
    updated = view.update_task_status(fact_id, status)
    new_view = decode_task_fact(updated)
    logger.info(
        "update_task_status: fact_id=%s task_id=%s %s -> %s",
        fact_id, current.task_id, current.status, status,
    )
    # タスク成功時に progress_marker を即時書き込み
    if status == "done":
        try:
            view.write_progress_marker(
                project_id=current.project_id,
                task_id=current.task_id,
                title=current.title,
                status="done",
            )
        except Exception as exc:
            logger.warning(
                "update_task_status: progress_marker write failed for "
                "task_id=%s project=%s: %s",
                current.task_id, current.project_id, exc,
            )
    return new_view


def reopen_orphan_in_progress_tasks(
    view: LoopFactView,
    project_id: str,
) -> list[TaskFactView]:
    """``in_progress`` のまま残存している orphan task を ``open`` に戻す。

    強制終了した場合、task ファクトは ``in_progress`` のまま SemMem に残る。
    新 driver の ``pick_next_task`` は ``status=open`` しか拾わないため、本関数
    を bootstrap / driver.start の直後に呼ぶことで orphan を回収する。

    Returns:
        open に戻された task の ``TaskFactView`` リスト (決定的順)。空リスト = 対象なし。
    """
    reopened_facts = view.reopen_orphan_in_progress_tasks(project_id)
    reopened: list[TaskFactView] = []
    for fact in reopened_facts:
        try:
            reopened.append(decode_task_fact(fact))
        except TaskFactDecodeError as exc:
            logger.warning(
                "reopen_orphan_in_progress_tasks: skip fact_id=%s project=%s: %s",
                fact.id, project_id, exc,
            )
            continue
    if reopened:
        logger.info(
            "reopen_orphan_in_progress_tasks: project=%s reopened=%d",
            project_id, len(reopened),
        )
    return reopened


def _is_allowed_transition(old: str, new: str) -> bool:
    """task status の遷移ルール。

    許可される遷移:
    - ``open`` → ``in_progress`` / ``failed``
    - ``in_progress`` → ``done`` / ``failed`` / ``open`` (リトライ復帰)
    - 終端 (``done`` / ``failed``) からは何も許可しない
      (リオープンが必要な場合は明示的に新しい task ファクトを作る)
    """
    allowed: dict[str, set[str]] = {
        "open": {"in_progress", "failed"},
        "in_progress": {"done", "failed", "open"},
        "done": set(),
        "failed": set(),
    }
    return new in allowed.get(old, set())


# ──────────────────────────────────────────────────────────────────────────
# LoopDriver
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class LoopDriverState:
    """LoopDriver の現在状態 (API / CLI から参照する DTO)"""

    running: bool = False
    project_id: str | None = None
    started_at: float | None = None
    iteration: int = 0
    stop_requested: bool = False
    last_picked_fact_id: str | None = None
    last_picked_task_id: str | None = None
    # 直近 bootstrap / reset の結果サマリ
    last_bootstrap: BootstrapResult | None = None
    last_reset_report: MaybeResetReport | None = None
    # 周回実行の進捗統計
    consecutive_failures: int = 0
    success_count: int = 0
    failure_count: int = 0
    skip_count: int = 0
    # artifact 集計
    artifacts: list[ArtifactEntry] = field(default_factory=list)
    last_outcome: dict[str, Any] | None = None
    # pause/resume 制御 + 現タスク詳細
    pause_requested: bool = False
    paused: bool = False
    current_action_kind: str | None = None
    iteration_started_at: float | None = None
    current_trace_id: str | None = None
    # 起動時に in_progress のまま残っていた orphan task を open に戻した task_id 一覧
    orphan_tasks_reopened: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "project_id": self.project_id,
            "started_at": self.started_at,
            "iteration": self.iteration,
            "stop_requested": self.stop_requested,
            "last_picked_fact_id": self.last_picked_fact_id,
            "last_picked_task_id": self.last_picked_task_id,
            "last_bootstrap": (
                self.last_bootstrap.as_dict() if self.last_bootstrap else None
            ),
            "last_reset_report": (
                self.last_reset_report.as_dict()
                if self.last_reset_report
                else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "skip_count": self.skip_count,
            "artifact_count": len(self.artifacts),
            "last_outcome": self.last_outcome,
            "pause_requested": self.pause_requested,
            "paused": self.paused,
            "current_action_kind": self.current_action_kind,
            "iteration_started_at": self.iteration_started_at,
            "current_trace_id": self.current_trace_id,
            "orphan_tasks_reopened": list(self.orphan_tasks_reopened),
        }


class LoopDriver:
    """自律ループ driver。

    ``view_provider`` は ``project_id`` を受け取り、対応する
    :class:`LoopFactView` を返す callable。AppState 由来のファクトリを渡す
    想定。テストでは :class:`LoopFactView` の直接生成関数を渡せる。

    ``SemanticFactStore`` 直参照 (``store_provider`` /
    ``global_store``) を廃止し、:class:`LoopFactView` に統一した。
    """

    def __init__(
        self,
        view_provider: LoopViewProvider,
        *,
        policy_activation_min_confidence: float = 0.7,
        bootstrap_on_start: bool = False,
        executor: TaskExecutor | None = None,
        max_iterations: int = 50,
        max_wall_time_sec: float = 1800.0,
        max_consecutive_failures: int = 3,
        tick_interval_sec: float = 0.0,
        sleep_time_every_n: int = 5,
        sleep_time_hook: Any = None,  # Callable[[], Awaitable[None]]
        on_gate_fail: str = "retry",
        retry_limit_per_task: int = 2,
        artifact_hook: Any = None,  # Callable[[str, TaskFactView, list[ArtifactEntry]], None]
        event_bus: LoopEventBus | None = None,
        pause_poll_interval_sec: float = 0.1,
        debug_logger: "DebugLogger | None" = None,
    ) -> None:
        """LoopDriver を構築する。

        Args:
            view_provider: ``project_id`` を受け取り対応する :class:`LoopFactView`
                を返す callable。view は ``stores=[global, project]`` + ``writeback_store=project``
                の形が想定される。
            policy_activation_min_confidence: ``policy`` ファクトを active と見なす
                最小 confidence (``config.learning.policy.activation_min_confidence``
                と整合させる)
            bootstrap_on_start: ``True`` の場合は ``start(project_id)`` 時に自動的に
                ``bootstrap_context`` を呼び出して結果を ``state.last_bootstrap`` に保存する。
            sleep_time_every_n: ``sleep_time_hook`` を呼ぶ反復間隔
                (``config.loop.sleep_time_every_n``)。
            sleep_time_hook: ``sleep_time_every_n`` 反復ごとに await される
                コルーチン関数。sleep-time (Light) へバトンを渡し、STM 圧縮後の
                メモリで次の反復に入るために使う。``None`` で無効。
        """
        self._view_provider = view_provider
        self._policy_activation_min_confidence = float(
            policy_activation_min_confidence,
        )
        self._bootstrap_on_start = bool(bootstrap_on_start)
        self._executor = executor
        self._max_iterations = int(max_iterations)
        self._max_wall_time_sec = float(max_wall_time_sec)
        self._max_consecutive_failures = int(max_consecutive_failures)
        self._tick_interval_sec = float(tick_interval_sec)
        self._sleep_time_every_n = int(sleep_time_every_n)
        self._sleep_time_hook = sleep_time_hook
        self._on_gate_fail = str(on_gate_fail)
        self._retry_limit_per_task = int(retry_limit_per_task)
        self._artifact_hook = artifact_hook
        self._event_bus = event_bus
        self._pause_poll_interval_sec = float(pause_poll_interval_sec)
        # task_id -> gate 失敗によるリトライ回数
        self._retries: dict[str, int] = {}
        self._state = LoopDriverState()
        # decision.jsonl に記録する (decision_point=``loop_continue_or_abort``,
        # ``quality_gate_action``)。
        self._debug_logger = debug_logger

    @property
    def event_bus(self) -> LoopEventBus | None:
        return self._event_bus

    async def _maybe_run_sleep_time(self) -> None:
        """``sleep_time_every_n`` 反復ごとに sleep-time (Light) へバトンを渡す。

        フックの失敗はループを止めない (warning のみ)。
        """
        if self._sleep_time_hook is None or self._sleep_time_every_n <= 0:
            return
        if self._state.iteration % self._sleep_time_every_n != 0:
            return
        logger.info(
            "LoopDriver: handing off to sleep-time at iteration=%d (every_n=%d)",
            self._state.iteration, self._sleep_time_every_n,
        )
        try:
            await self._sleep_time_hook()
        except Exception as exc:
            logger.warning("LoopDriver sleep-time hook failed: %s", exc)
        else:
            self._emit("sleep_time_ran", {"iteration": self._state.iteration})

    def _emit(
        self,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """``LoopEventBus`` が注入されていればイベントを発行する (null-safe)。"""
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(
                event,  # type: ignore[arg-type]
                iteration=self._state.iteration,
                project_id=self._state.project_id,
                data=data or {},
            )
        except Exception as exc:
            logger.warning("LoopDriver._emit(%s) failed: %s", event, exc)

    @property
    def state(self) -> LoopDriverState:
        return self._state

    def is_running(self) -> bool:
        return self._state.running

    def _resolve_view(self, project_id: str) -> LoopFactView:
        """``view_provider`` から view を取得する (例外は呼び出し側へ委譲)。"""
        return self._view_provider(project_id)

    def start(self, project_id: str) -> LoopDriverState:
        """ループを起動状態にする (実行はしない)。

        Raises:
            RuntimeError: 既に起動中
            ValueError: project_id が空
        """
        if not project_id:
            raise ValueError("project_id must be non-empty")
        if self._state.running:
            raise RuntimeError(
                f"loop already running for project={self._state.project_id!r}",
            )
        self._state = LoopDriverState(
            running=True,
            project_id=project_id,
            started_at=time.time(),
            iteration=0,
            stop_requested=False,
        )
        self._retries = {}
        logger.info("LoopDriver.start: project_id=%s", project_id)
        if self._bootstrap_on_start:
            try:
                view = self._resolve_view(project_id)
            except Exception as exc:
                logger.warning(
                    "LoopDriver.start: view_provider failed for project=%s: %s",
                    project_id, exc,
                )
                view = None
            if view is not None:
                # 前回 driver の kill 等で in_progress のまま残った
                # orphan task を open に戻してから bootstrap する。
                try:
                    reopened = reopen_orphan_in_progress_tasks(view, project_id)
                    self._state.orphan_tasks_reopened = [
                        v.task_id for v in reopened
                    ]
                    if reopened:
                        logger.info(
                            "LoopDriver.start: recovered %d orphan in_progress "
                            "task(s) for project=%s: %s",
                            len(reopened), project_id,
                            self._state.orphan_tasks_reopened,
                        )
                except Exception as exc:
                    logger.warning(
                        "LoopDriver.start: orphan recovery failed for "
                        "project=%s: %s", project_id, exc,
                    )
                try:
                    self._state.last_bootstrap = _view_bootstrap_to_dataclass(
                        view.bootstrap_context(
                            project_id=project_id,
                            policy_activation_min_confidence=(
                                self._policy_activation_min_confidence
                            ),
                        ),
                    )
                except Exception as exc:
                    logger.warning(
                        "LoopDriver.start: bootstrap failed for project=%s: %s",
                        project_id, exc,
                    )
        return self._state

    def stop(self) -> LoopDriverState:
        """ループを停止状態にする (冪等)。"""
        if not self._state.running:
            logger.debug("LoopDriver.stop: already stopped")
            return self._state
        self._state.stop_requested = True
        self._state.running = False
        self._state.paused = False
        self._state.pause_requested = False
        logger.info(
            "LoopDriver.stop: project_id=%s iteration=%d",
            self._state.project_id, self._state.iteration,
        )
        self._emit("loop_stopped", {"iteration": self._state.iteration})
        return self._state

    # ── pause / resume ───────────────────────────────────────────

    def pause(self) -> LoopDriverState:
        """現サイクル完了後に pause する (冪等)。"""
        if not self._state.running:
            logger.debug("LoopDriver.pause: not running, noop")
            return self._state
        if self._state.paused or self._state.pause_requested:
            logger.debug("LoopDriver.pause: already paused/requested")
            return self._state
        self._state.pause_requested = True
        logger.info(
            "LoopDriver.pause: requested project_id=%s iteration=%d",
            self._state.project_id, self._state.iteration,
        )
        return self._state

    def resume(self) -> LoopDriverState:
        """pause からの再開 (冪等)。"""
        if not self._state.running:
            logger.debug("LoopDriver.resume: not running, noop")
            return self._state
        if not self._state.paused and not self._state.pause_requested:
            logger.debug("LoopDriver.resume: not paused, noop")
            return self._state
        was_paused = self._state.paused
        self._state.pause_requested = False
        self._state.paused = False
        logger.info(
            "LoopDriver.resume: project_id=%s iteration=%d was_paused=%s",
            self._state.project_id, self._state.iteration, was_paused,
        )
        if was_paused:
            self._emit("loop_resumed", {"iteration": self._state.iteration})
        return self._state

    def peek_next_task(self, project_id: str | None = None) -> TaskFactView | None:
        """指定プロジェクトの次タスクを (実行せず) 返す。"""
        target = project_id or self._state.project_id
        if not target:
            raise ValueError(
                "project_id required when loop is not running",
            )
        view = self._resolve_view(target)
        task = pick_next_task(view, target)
        if task is not None:
            self._state.last_picked_fact_id = task.fact_id
            self._state.last_picked_task_id = task.task_id
        return task

    def list_tasks(
        self,
        project_id: str | None = None,
        *,
        status_filter: set[str] | None = None,
    ) -> list[TaskFactView]:
        target = project_id or self._state.project_id
        if not target:
            raise ValueError(
                "project_id required when loop is not running",
            )
        view = self._resolve_view(target)
        return list_tasks(view, target, status_filter=status_filter)

    # ── クリーンコンテキスト再起動 + bootstrap ─────────

    def maybe_reset_and_bootstrap(
        self,
        *,
        wm: WorkingMemory | None,
        stm: ShortTermMemory | None,
        threshold_tokens: int,
        project_id: str | None = None,
        force: bool = False,
    ) -> MaybeResetReport:
        """トークン閾値超過時に episodic を破棄し SemMem から再 bootstrap する。"""
        target = project_id or self._state.project_id
        if not target:
            raise ValueError(
                "project_id required when loop is not running",
            )
        view = self._resolve_view(target)
        report = maybe_reset_and_bootstrap(
            view,
            wm=wm,
            stm=stm,
            project_id=target,
            threshold_tokens=threshold_tokens,
            policy_activation_min_confidence=(
                self._policy_activation_min_confidence
            ),
            force=force,
        )
        self._state.last_reset_report = report
        if report.bootstrap is not None:
            self._state.last_bootstrap = report.bootstrap
        return report

    # ── 周回実行 ───────────────────────────────────────────

    async def run(self, project_id: str | None = None) -> LoopDriverState:
        """自律ループ周回を実行する"""
        target = project_id or self._state.project_id
        if not target:
            raise ValueError("project_id required")
        if self._executor is None:
            raise RuntimeError(
                "LoopDriver.run: executor is not configured",
            )
        if not self._state.running:
            raise RuntimeError("LoopDriver.run: loop is not started")

        view = self._resolve_view(target)
        logger.info(
            "LoopDriver.run: start project=%s executor=%s max_iter=%d "
            "max_wall=%.0fs max_consec_fail=%d",
            target, getattr(self._executor, "name", "?"),
            self._max_iterations, self._max_wall_time_sec,
            self._max_consecutive_failures,
        )
        self._emit(
            "loop_started",
            {"executor": getattr(self._executor, "name", "?")},
        )

        while not self._state.stop_requested:
            if self._state.pause_requested:
                self._state.pause_requested = False
                self._state.paused = True
                logger.info(
                    "LoopDriver.run: paused at iteration=%d",
                    self._state.iteration,
                )
                self._emit("loop_paused", {"iteration": self._state.iteration})
            while self._state.paused and not self._state.stop_requested:
                try:
                    await asyncio.sleep(self._pause_poll_interval_sec)
                except asyncio.CancelledError:
                    logger.info("LoopDriver.run: cancelled while paused")
                    self._state.paused = False
                    raise
            if self._state.stop_requested:
                break
            if self._state.iteration >= self._max_iterations:
                logger.info(
                    "LoopDriver.run: max_iterations reached (%d)",
                    self._max_iterations,
                )
                break
            if self._state.started_at is not None:
                elapsed = time.time() - self._state.started_at
                if elapsed >= self._max_wall_time_sec:
                    logger.info(
                        "LoopDriver.run: max_wall_time reached (%.0fs)",
                        elapsed,
                    )
                    break
            if self._state.consecutive_failures >= self._max_consecutive_failures:
                logger.warning(
                    "LoopDriver.run: consecutive_failures >= %d, stopping",
                    self._max_consecutive_failures,
                )
                if self._debug_logger is not None:
                    self._debug_logger.log_decision(
                        decision_point="loop_continue_or_abort",
                        chosen="abort",
                        candidates=["continue", "abort"],
                        reason="max_consecutive_failures_exceeded",
                        context={
                            "consecutive_failures": self._state.consecutive_failures,
                            "max_consecutive_failures": self._max_consecutive_failures,
                            "iteration": self._state.iteration,
                        },
                        scope="loop_iter",
                    )
                break

            task = pick_next_task(view, target)
            if task is None:
                logger.info(
                    "LoopDriver.run: no open tasks for project=%s, stopping",
                    target,
                )
                break

            self._state.last_picked_fact_id = task.fact_id
            self._state.last_picked_task_id = task.task_id
            self._state.iteration += 1
            self._state.iteration_started_at = time.time()
            tid = generate_trace_id()
            self._state.current_trace_id = tid
            set_trace_id(tid)
            self._emit(
                "iteration_started",
                {
                    "task_id": task.task_id,
                    "fact_id": task.fact_id,
                    "title": task.title,
                    "salience": task.salience,
                },
            )
            self._emit(
                "task_picked",
                {
                    "task_id": task.task_id,
                    "fact_id": task.fact_id,
                    "title": task.title,
                    "salience": task.salience,
                    "depends_on": list(task.depends_on),
                },
            )

            await self._execute_iteration(view, target, task)
            self._emit(
                "iteration_ended",
                {
                    "task_id": task.task_id,
                    "success_count": self._state.success_count,
                    "failure_count": self._state.failure_count,
                    "skip_count": self._state.skip_count,
                    "last_outcome": self._state.last_outcome,
                },
            )
            await self._maybe_run_sleep_time()
            # (evolve 限定)。``last_outcome`` の kind フィールドで
            # success / failure / skipped を判定する。
            if self._debug_logger is not None:
                last_outcome = self._state.last_outcome or {}
                outcome_kind_str = str(last_outcome.get("kind", "unknown"))
                duration_ms: float | None = None
                if self._state.iteration_started_at is not None:
                    duration_ms = (
                        time.time() - self._state.iteration_started_at
                    ) * 1000
                self._debug_logger.log_outcome(
                    kind="loop_iter",
                    success=(outcome_kind_str == "success"),
                    duration_ms=duration_ms,
                    quality_signals={
                        "iteration_outcome": outcome_kind_str,
                        "task_id": task.task_id,
                        "iteration": self._state.iteration,
                        "consecutive_failures": self._state.consecutive_failures,
                    },
                )
            self._state.current_action_kind = None

            if self._tick_interval_sec > 0:
                try:
                    await asyncio.sleep(self._tick_interval_sec)
                except asyncio.CancelledError:
                    logger.info("LoopDriver.run: cancelled during tick sleep")
                    break

        self._state.running = False
        logger.info(
            "LoopDriver.run: end project=%s iterations=%d success=%d "
            "failure=%d skip=%d artifacts=%d",
            target, self._state.iteration, self._state.success_count,
            self._state.failure_count, self._state.skip_count,
            len(self._state.artifacts),
        )
        return self._state

    async def _execute_iteration(
        self,
        view: LoopFactView,
        project_id: str,
        task: TaskFactView,
    ) -> None:
        """1 イテレーションを実行する (例外を握って state を維持)。"""
        try:
            update_task_status(view, task.fact_id, "in_progress")
            self._emit(
                "fact_written",
                {
                    "kind": "task_status",
                    "task_id": task.task_id,
                    "fact_id": task.fact_id,
                    "from": "open",
                    "to": "in_progress",
                },
            )
        except (KeyError, ValueError) as exc:
            logger.warning(
                "LoopDriver.run: status transition failed task=%s: %s",
                task.task_id, exc,
            )
            self._state.skip_count += 1
            return

        try:
            outcome = await self._executor.execute(task)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "LoopDriver.run: executor raised for task=%s: %s",
                task.task_id, exc,
            )
            outcome = ExecutionOutcome(
                status="failure",
                error=f"executor_exception: {exc}",
            )

        self._state.last_outcome = outcome.as_dict()
        for i, action_result in enumerate(outcome.action_results):
            kind = getattr(action_result, "kind", None) or getattr(
                getattr(action_result, "action", None), "kind", "",
            )
            self._state.current_action_kind = str(kind) if kind else None
            self._emit(
                "action_executed",
                {
                    "task_id": task.task_id,
                    "index": i,
                    "kind": self._state.current_action_kind,
                    "ok": bool(getattr(action_result, "ok", False)),
                    "error": getattr(action_result, "error", None),
                },
            )
        if outcome.gate_outcome is not None:
            for gr in outcome.gate_outcome.results:
                self._emit(
                    "gate_result",
                    {
                        "task_id": task.task_id,
                        "name": gr.name,
                        "ok": bool(gr.ok),
                        "skipped": bool(getattr(gr, "skipped", False)),
                        "returncode": gr.returncode,
                        "duration_ms": gr.duration_ms,
                        "error": gr.error,
                    },
                )
        self._apply_outcome(view, project_id, task, outcome)

    def _apply_outcome(
        self,
        view: LoopFactView,
        project_id: str,
        task: TaskFactView,
        outcome: ExecutionOutcome,
    ) -> None:
        """``ExecutionOutcome`` を受けて SemMem / driver state を更新する。"""
        if outcome.artifacts:
            self._state.artifacts.extend(outcome.artifacts)
            if self._artifact_hook is not None:
                try:
                    self._artifact_hook(  # type: ignore[misc]
                        project_id, task, list(outcome.artifacts),
                    )
                except Exception as exc:
                    logger.warning(
                        "LoopDriver.run: artifact_hook failed: %s", exc,
                    )

        def _emit_status(from_: str, to: str) -> None:
            self._emit(
                "fact_written",
                {
                    "kind": "task_status",
                    "task_id": task.task_id,
                    "fact_id": task.fact_id,
                    "from": from_,
                    "to": to,
                },
            )

        match outcome.status:
            case "success":
                try:
                    update_task_status(view, task.fact_id, "done")
                    _emit_status("in_progress", "done")
                    self._emit(
                        "fact_written",
                        {"kind": "progress_marker", "task_id": task.task_id},
                    )
                except (KeyError, ValueError) as exc:
                    logger.warning(
                        "LoopDriver.run: done transition failed task=%s: %s",
                        task.task_id, exc,
                    )
                self._state.success_count += 1
                self._state.consecutive_failures = 0
                self._retries.pop(task.task_id, None)

            case "skipped":
                try:
                    update_task_status(view, task.fact_id, "done")
                    _emit_status("in_progress", "done")
                except (KeyError, ValueError) as exc:
                    logger.warning(
                        "LoopDriver.run: skip->done transition failed "
                        "task=%s: %s", task.task_id, exc,
                    )
                self._state.skip_count += 1
                self._state.consecutive_failures = 0
                self._retries.pop(task.task_id, None)

            case "failure":
                self._record_failure(view, project_id, task, outcome)
                self._emit(
                    "fact_written",
                    {"kind": "failure_pattern", "task_id": task.task_id},
                )
                retries_so_far = self._retries.get(task.task_id, 0)
                self._retries[task.task_id] = retries_so_far + 1
                self._state.failure_count += 1
                self._state.consecutive_failures += 1
                if retries_so_far + 1 >= self._retry_limit_per_task:
                    if self._debug_logger is not None:
                        self._debug_logger.log_decision(
                            decision_point="quality_gate_action",
                            chosen="abort",
                            candidates=["retry", "skip", "abort"],
                            reason="retry_limit_exceeded",
                            context={
                                "task_id": task.task_id,
                                "retries_so_far": retries_so_far + 1,
                                "retry_limit": self._retry_limit_per_task,
                                "policy": self._on_gate_fail,
                            },
                            scope="loop_iter",
                        )
                    try:
                        update_task_status(view, task.fact_id, "failed")
                        _emit_status("in_progress", "failed")
                    except (KeyError, ValueError) as exc:
                        logger.warning(
                            "LoopDriver.run: failed transition failed "
                            "task=%s: %s", task.task_id, exc,
                        )
                    logger.info(
                        "LoopDriver.run: task=%s skipped after %d retries",
                        task.task_id, retries_so_far + 1,
                    )
                    self._retries.pop(task.task_id, None)
                else:
                    if self._debug_logger is not None:
                        self._debug_logger.log_decision(
                            decision_point="quality_gate_action",
                            chosen="retry",
                            candidates=["retry", "skip", "abort"],
                            reason="retry_within_limit",
                            context={
                                "task_id": task.task_id,
                                "retries_so_far": retries_so_far + 1,
                                "retry_limit": self._retry_limit_per_task,
                                "policy": self._on_gate_fail,
                            },
                            scope="loop_iter",
                        )
                    try:
                        update_task_status(view, task.fact_id, "open")
                        _emit_status("in_progress", "open")
                    except (KeyError, ValueError) as exc:
                        logger.warning(
                            "LoopDriver.run: retry open transition failed "
                            "task=%s: %s", task.task_id, exc,
                        )

    def _record_failure(
        self,
        view: LoopFactView,
        project_id: str,
        task: TaskFactView,  # noqa: ARG002
        outcome: ExecutionOutcome,
    ) -> None:
        """failure_pattern ファクトを書き込む。"""
        from backend.free.loop.failure_note import write_failure_note
        from backend.free.loop.quality_gate import GateResult

        if outcome.gate_outcome is not None and outcome.gate_outcome.failed:
            for gr in outcome.gate_outcome.results:
                if gr.is_failure():
                    try:
                        write_failure_note(
                            view,
                            project_id=project_id,
                            gate_result=gr,
                        )
                    except Exception as exc:
                        logger.warning(
                            "LoopDriver.run: write_failure_note failed: %s",
                            exc,
                        )
            return
        err_text = outcome.error or "executor_failure"
        action_stderr = "\n".join(
            r.error for r in outcome.action_results if r.error
        )
        # parse 失敗時の LLM 応答先頭を観測材料として stdout_tail に積む。
        # output_tail として failure_pattern.object に乗り、次イテレーションで
        # 同じ malformed JSON を繰り返さないためのヒントになる。
        notes = outcome.notes or {}
        response_head = notes.get("response_head", "")
        response_length = notes.get("response_length", "")
        stdout_parts: list[str] = []
        if response_length:
            stdout_parts.append(f"[response_length={response_length}]")
        if response_head:
            stdout_parts.append(f"[response_head]\n{response_head}")
        synthetic = GateResult(
            name="executor",
            ok=False,
            skipped=False,
            returncode=None,
            duration_ms=0,
            stdout_tail="\n".join(stdout_parts),
            stderr_tail=action_stderr or err_text,
            error=err_text,
        )
        try:
            write_failure_note(
                view,
                project_id=project_id,
                gate_result=synthetic,
            )
        except Exception as exc:
            logger.warning(
                "LoopDriver.run: write_failure_note (synthetic) failed: %s",
                exc,
            )

    # ── LoopReport ─────────────────────────────────────

    def generate_report(self, project_id: str | None = None):
        """指定プロジェクトの ``LoopReport`` を生成する。"""
        from backend.free.loop.report import generate_loop_report

        target = project_id or self._state.project_id
        if not target:
            raise ValueError(
                "project_id required when loop is not running",
            )
        view = self._resolve_view(target)
        if target == self._state.project_id:
            started_at = self._state.started_at
            iterations = self._state.iteration or None
            artifacts = list(self._state.artifacts) or None
        else:
            started_at = None
            iterations = None
            artifacts = None
        return generate_loop_report(
            view,
            project_id=target,
            started_at=started_at,
            iterations=iterations,
            artifacts=artifacts,
        )


# ──────────────────────────────────────────────────────────────────────────
# 内部: LoopFactView の BootstrapResult を loop/bootstrap.py の
# BootstrapResult (互換 dataclass 形式) に変換する (LoopDriverState 互換維持)
# ──────────────────────────────────────────────────────────────────────────


def _view_bootstrap_to_dataclass(
    view_result: Any,
) -> BootstrapResult:
    """``LoopBootstrapResult`` を ``bootstrap.BootstrapResult`` へ変換する。

    LoopFactView.bootstrap_context は View 層で新型 ``LoopBootstrapResult`` を
    返すため、LoopDriverState.last_bootstrap の型 (``bootstrap.BootstrapResult``)
    と橋渡しする。フィールドは同名で 1:1 対応。
    """
    return BootstrapResult(
        project_id=view_result.project_id,
        active_policies=list(view_result.active_policies),
        pinned_global=list(view_result.pinned_global),
        pinned_project=list(view_result.pinned_project),
        candidate_failure_patterns=list(view_result.candidate_failure_patterns),
        skipped_below_confidence=view_result.skipped_below_confidence,
        policy_activation_min_confidence=(
            view_result.policy_activation_min_confidence
        ),
        artifact_count=view_result.artifact_count,
    )
