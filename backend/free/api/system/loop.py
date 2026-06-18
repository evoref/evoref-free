"""

CLI / フロントエンドが叩く REST 入口。LoopDriver が周回実行を
担当し、本 API は観測 / 制御を提供する。

エンドポイント:
- POST /api/loop/start         — ループ起動 + 周回開始
- POST /api/loop/stop          — ループ停止 (冪等)
- POST /api/loop/pause         — 現サイクル完了後に pause
- POST /api/loop/resume        — pause からの再開
- GET  /api/loop/status        — ループ状態 (DTO)
- GET  /api/loop/current       — 実行中 task / action / 経過秒
- GET  /api/loop/stream        — SSE サイクルイベント
- GET  /api/loop/report        — LoopReport
- GET  /api/loop/tasks         — 指定プロジェクトの task 一覧 + 次タスク
- POST /api/loop/tasks/import  — PRD JSON を task ファクトとして登録
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.app_state import AppState, get_app_state
from backend.free.api.schemas import (
    LoopCurrentResponse,
    LoopReportResponse,
    LoopStartRequest,
    LoopStateInfo,
    LoopStateResponse,
    TaskImportRequest,
    TaskImportResponse,
    TaskInfo,
    TaskListResponse,
)
from backend.free.loop.driver import LoopDriver, TaskFactView, list_tasks, pick_next_task
from backend.free.loop.events import LoopEventBus
from backend.free.loop.prd_parser import (
    PRDParseError,
    parse_prd_json,
    parse_prd_json_file,
)
from backend.free.loop.report import generate_loop_report
from backend.free.memory.views.loop import LoopFactView
from backend.log_config import get_logger

logger = get_logger("api.loop")

router = APIRouter(prefix="/api/loop", tags=["loop"])


# ──────────────────────────────────────────────────────────────────────────
# ヘルパ
# ──────────────────────────────────────────────────────────────────────────


def _get_or_create_driver(state: AppState) -> LoopDriver:
    """AppState に LoopDriver をぶら下げる (lazy)。

    lifespan (``app_factory``) が事前に LoopDriver を構築
    していればそれを返す。未構築なら executor=None で空 driver を作って
    互換性を維持する (pick_next_task / list_tasks 用の旧 API は動く)。
    """
    driver = getattr(state, "loop_driver", None)
    if driver is None:
        def _view_provider(project_id: str) -> LoopFactView:
            return _build_loop_view(state, project_id)
        driver = LoopDriver(
            view_provider=_view_provider,
            event_bus=LoopEventBus(),
        )
        state.loop_driver = driver  # type: ignore[attr-defined]
    return driver


def _build_loop_view(state: AppState, project_id: str) -> LoopFactView:
    """``state`` 由来のストアから :class:`LoopFactView` を構築する。

    ``stores=[global, project]`` + ``writeback_store=project`` の標準構成。
    ``global`` ストアが取得できない環境 (テスト等) では project ストア単体で
    フォールバックする。
    """
    project_store = state.get_semantic_store(f"project:{project_id}")
    try:
        global_store = state.get_semantic_store("global")
        stores: list = [global_store, project_store]
    except Exception:
        stores = [project_store]
    return LoopFactView(
        stores=stores,
        writeback_store=project_store,
    )


def _to_state_info(driver: LoopDriver) -> LoopStateInfo:
    s = driver.state
    return LoopStateInfo(
        running=s.running,
        project_id=s.project_id,
        started_at=s.started_at,
        iteration=s.iteration,
        stop_requested=s.stop_requested,
        last_picked_fact_id=s.last_picked_fact_id,
        last_picked_task_id=s.last_picked_task_id,
        pause_requested=s.pause_requested,
        paused=s.paused,
    )


def _to_task_info(view: TaskFactView) -> TaskInfo:
    return TaskInfo(
        fact_id=view.fact_id,
        task_id=view.task_id,
        title=view.title,
        description=view.description,
        depends_on=list(view.depends_on),
        salience=view.salience,
        status=view.status,
        source_path=view.source_path,
        project_id=view.project_id,
        created_at=view.created_at,
        accessed_at=view.accessed_at,
        stage=view.stage,
    )


def _resolve_project_view(state: AppState, project_id: str) -> LoopFactView:
    """``project_id`` に対応する :class:`LoopFactView` を構築して返す。

    400 エラー (project_id 未指定 / ストア解決失敗) を HTTP エラーに変換する。
    """
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id must be non-empty")
    try:
        return _build_loop_view(state, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ──────────────────────────────────────────────────────────────────────────
# /start /stop /status
# ──────────────────────────────────────────────────────────────────────────


@router.post("/start", response_model=LoopStateResponse)
async def start_loop(
    req: LoopStartRequest,
    state: AppState = Depends(get_app_state),
):
    """ループを起動して周回実行を開始する

    executor が lifespan で注入されていれば ``LoopDriver.run()`` を
    ``asyncio.create_task`` でバックグラウンド実行する。executor 未注入
    (degradation mode) の場合は従来通り状態遷移のみ。
    """
    logger.debug("POST /api/loop/start: project_id=%s", req.project_id)
    driver = _get_or_create_driver(state)
    try:
        driver.start(req.project_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 既に動いている周回 Task があれば止める (二重起動防止)
    prev_task = getattr(state, "loop_run_task", None)
    if prev_task is not None and not prev_task.done():
        prev_task.cancel()

    if getattr(driver, "_executor", None) is not None:
        async def _run_background() -> None:
            try:
                await driver.run(req.project_id)
            except asyncio.CancelledError:
                logger.info("loop run task cancelled")
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("loop run task failed: %s", exc)
                driver.stop()

        task = asyncio.create_task(_run_background(), name="loop.run")
        state.loop_run_task = task  # type: ignore[attr-defined]
    return LoopStateResponse(state=_to_state_info(driver))


@router.post("/stop", response_model=LoopStateResponse)
async def stop_loop(state: AppState = Depends(get_app_state)):
    """ループを停止する (冪等)"""
    logger.debug("POST /api/loop/stop")
    driver = _get_or_create_driver(state)
    driver.stop()
    return LoopStateResponse(state=_to_state_info(driver))


@router.get("/status", response_model=LoopStateResponse)
async def get_loop_status(state: AppState = Depends(get_app_state)):
    """LoopDriver の現在状態を返す"""
    driver = _get_or_create_driver(state)
    return LoopStateResponse(state=_to_state_info(driver))


# ──────────────────────────────────────────────────────────────────────────
# /pause /resume /current /stream
# ──────────────────────────────────────────────────────────────────────────


@router.post("/pause", response_model=LoopStateResponse)
async def pause_loop(state: AppState = Depends(get_app_state)):
    """現サイクル完了後に pause 状態へ遷移する (冪等)"""
    logger.debug("POST /api/loop/pause")
    driver = _get_or_create_driver(state)
    if not driver.state.running:
        raise HTTPException(status_code=409, detail="loop is not running")
    driver.pause()
    return LoopStateResponse(state=_to_state_info(driver))


@router.post("/resume", response_model=LoopStateResponse)
async def resume_loop(state: AppState = Depends(get_app_state)):
    """pause からの再開 (冪等)"""
    logger.debug("POST /api/loop/resume")
    driver = _get_or_create_driver(state)
    if not driver.state.running:
        raise HTTPException(status_code=409, detail="loop is not running")
    driver.resume()
    return LoopStateResponse(state=_to_state_info(driver))


@router.get("/current", response_model=LoopCurrentResponse)
async def get_current_task(state: AppState = Depends(get_app_state)):
    """実行中タスクと action / 経過秒を返す

    周回していない場合は running=False で空レスポンス。running 中でも
    task 選定前は current_task=None になる。
    """
    import time as _time

    driver = _get_or_create_driver(state)
    s = driver.state
    current_task: TaskInfo | None = None
    if s.last_picked_fact_id and s.project_id:
        view = _resolve_project_view(state, s.project_id)
        fact = view.get_fact(s.last_picked_fact_id)
        if fact is not None:
            try:
                from backend.free.loop.driver import decode_task_fact
                view = decode_task_fact(fact)
                current_task = _to_task_info(view)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "GET /api/loop/current: failed to decode task fact: %s", exc,
                )
    now = _time.time()
    elapsed = (now - s.started_at) if s.started_at else None
    iter_elapsed = (
        (now - s.iteration_started_at) if s.iteration_started_at else None
    )
    return LoopCurrentResponse(
        running=s.running,
        paused=s.paused,
        project_id=s.project_id,
        iteration=s.iteration,
        started_at=s.started_at,
        iteration_started_at=s.iteration_started_at,
        elapsed_seconds=elapsed,
        iteration_elapsed_seconds=iter_elapsed,
        current_trace_id=s.current_trace_id,
        current_task=current_task,
        current_action_kind=s.current_action_kind,
        last_outcome=s.last_outcome,
    )


async def _sse_generator(
    bus: LoopEventBus,
    request: Request,
    *,
    keepalive_sec: float = 15.0,
) -> AsyncIterator[str]:
    """SSE 用イベントジェネレータ。切断検知で unsubscribe。"""
    queue = bus.subscribe()
    try:
        # 購読確立を即時通知 (クライアントが接続判定に使える)
        yield ": connected\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=keepalive_sec)
            except asyncio.TimeoutError:
                # keep-alive コメント
                yield ": keepalive\n\n"
                continue
            payload = json.dumps(event.as_dict(), ensure_ascii=False)
            yield f"event: {event.event}\ndata: {payload}\n\n"
    finally:
        bus.unsubscribe(queue)


@router.get("/stream")
async def stream_loop_events(
    request: Request,
    state: AppState = Depends(get_app_state),
    keepalive_sec: float = Query(15.0, ge=0.1, le=120.0),
):
    """自律ループのサイクルイベントを SSE で配信

    events: ``task_picked`` / ``action_executed`` / ``gate_result`` /
    ``fact_written`` / ``iteration_started`` / ``iteration_ended`` /
    ``loop_started`` / ``loop_paused`` / ``loop_resumed`` / ``loop_stopped``

    Args:
        keepalive_sec: keep-alive コメントを送る間隔 (秒)。クライアント切断
            検知の最大遅延もこの値になる。テスト向けに短く設定可能。
    """
    driver = _get_or_create_driver(state)
    bus = driver.event_bus
    if bus is None:
        # app_factory で event_bus 未注入 (degradation mode) — 空ストリーム
        bus = LoopEventBus()
        driver._event_bus = bus  # type: ignore[attr-defined]
    return StreamingResponse(
        _sse_generator(bus, request, keepalive_sec=keepalive_sec),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────────────────────
# /tasks
# ──────────────────────────────────────────────────────────────────────────


@router.get("/tasks", response_model=TaskListResponse)
async def list_loop_tasks(
    state: AppState = Depends(get_app_state),
    project_id: str | None = Query(None),
    status: str | None = Query(None, description="フィルタ: open|in_progress|done|failed"),
):
    """プロジェクトの task ファクト一覧を返す。

    ``project_id`` 省略時は LoopDriver で running 中のプロジェクトを使う。
    どちらも未設定なら 400。
    """
    driver = _get_or_create_driver(state)
    target = project_id or driver.state.project_id
    if not target:
        raise HTTPException(
            status_code=400,
            detail="project_id required (no running loop)",
        )
    view = _resolve_project_view(state, target)
    status_filter: set[str] | None = None
    if status:
        status_filter = {s.strip() for s in status.split(",") if s.strip()}
    views = list_tasks(view, target, status_filter=status_filter)
    next_task = pick_next_task(view, target)
    return TaskListResponse(
        project_id=target,
        total=len(views),
        tasks=[_to_task_info(v) for v in views],
        next_task_id=next_task.task_id if next_task else None,
    )


@router.post("/tasks/import", response_model=TaskImportResponse)
async def import_loop_tasks(
    req: TaskImportRequest,
    state: AppState = Depends(get_app_state),
):
    """PRD JSON を task ファクトとして SemMem に登録する"""
    logger.debug(
        "POST /api/loop/tasks/import: project_id=%s path=%s json=%s",
        req.project_id, req.prd_path, "yes" if req.prd_json is not None else "no",
    )
    if not req.project_id:
        raise HTTPException(status_code=400, detail="project_id must be non-empty")
    if req.prd_path is None and req.prd_json is None:
        raise HTTPException(
            status_code=400,
            detail="either prd_path or prd_json must be provided",
        )
    try:
        if req.prd_path is not None:
            facts = parse_prd_json_file(req.prd_path, project_id=req.project_id)
        else:
            facts = parse_prd_json(
                req.prd_json,  # type: ignore[arg-type]
                project_id=req.project_id,
                source_path=None,
            )
    except PRDParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    view = _resolve_project_view(state, req.project_id)
    # task は EvorefLoop owner fact のため、書込先 writeback_store へ直接投入する。
    # view には bulk add API がないので writeback_store を経由する
    # (view 内の書込メソッドは ownership を再検証する個別書込専用のため)。
    fact_ids: list[str] = []
    for fact in facts:
        added = view.writeback_store.add_fact(fact)
        fact_ids.append(added.id)
    logger.info(
        "imported %d task facts into project=%s", len(fact_ids), req.project_id,
    )
    return TaskImportResponse(
        project_id=req.project_id,
        imported=len(fact_ids),
        fact_ids=fact_ids,
    )


# ──────────────────────────────────────────────────────────────────────────
# /report
# ──────────────────────────────────────────────────────────────────────────


@router.get("/report", response_model=LoopReportResponse)
async def get_loop_report(
    state: AppState = Depends(get_app_state),
    project_id: str | None = Query(None),
):
    """指定プロジェクトの ``LoopReport`` を返す。

    ``project_id`` 省略時は LoopDriver で running 中のプロジェクトを使う。
    どちらも未設定なら 400。
    """
    driver = _get_or_create_driver(state)
    target = project_id or driver.state.project_id
    if not target:
        raise HTTPException(
            status_code=400,
            detail="project_id required (no running loop)",
        )
    view = _resolve_project_view(state, target)
    # running 中の同一プロジェクトであれば driver state から started_at /
    # iteration を引き継ぐ。それ以外は省略 (= None / 終端タスク数フォールバック)。
    if target == driver.state.project_id:
        started_at = driver.state.started_at
        iterations = driver.state.iteration or None
    else:
        started_at = None
        iterations = None
    report = generate_loop_report(
        view,
        project_id=target,
        started_at=started_at,
        iterations=iterations,
    )
    return LoopReportResponse(**report.as_dict())
