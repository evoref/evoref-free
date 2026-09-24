"""staged クリエイトの run 一覧 / 詳細 / イベント再接続 API (f_10 §7、c_06)。

書き手は ``chat_stream_staged.run_staged_pipeline`` (staged) と
``generation.harness.LongFormHarness`` (longform、f_08 §2.3) の開始 / 終端
だけ — この router は :mod:`backend.free.loop.staged.run_record` を呼ぶか、
``run.json``/``events.jsonl`` を読むだけの薄い層 (``project_map.py`` の
router 作法に倣う)。
"""

from __future__ import annotations

from pathlib import Path as FsPath
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query

from backend.config import get_path_resolver
from backend.error_handlers import ErrorResponse
from backend.free.loop.staged.run_record import RUN_ID_PATTERN, RunRecord
from backend.log_config import get_logger

logger = get_logger("api.create_runs")

router = APIRouter(prefix="/api/create/runs", tags=["create"])

#: run_id (= workspace_id) の形式 (ID 台帳の ``run_``)。traversal 防止のため FastAPI 側でも検証する。
_RUN_ID_PATTERN = RUN_ID_PATTERN


def _create_runs_error(status_code: int, code: str, message: str, i18n_key: str, **context: Any) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(code=code, message=message, i18n_key=i18n_key, context=context).to_dict(),
    )


def _run_dict(record: RunRecord, status: str) -> dict[str, Any]:
    return {**record.to_dict(), "status": status}


def _stale_after_sec() -> float:
    """``create.turn_timeout_sec`` + 600 秒 (config 未ロードなら既定)。"""
    from backend.free.loop.staged.run_record import stale_after_from_config

    try:
        from backend.config import get_config
        return stale_after_from_config(get_config())
    except Exception:  # noqa: BLE001 - config 未ロード (テスト等) は既定へ
        return stale_after_from_config(None)


@router.get("")
async def list_create_runs(session_id: str | None = Query(default=None)) -> dict[str, Any]:
    """staged クリエイトの run 一覧 (``started_at`` 降順、f_10 §7)。"""
    from backend.free.loop.staged.run_record import list_runs

    create_dir = get_path_resolver().resolve_local("create_workspace_dir")
    runs = list_runs(
        create_dir, session_id=session_id, stale_after_sec=_stale_after_sec(),
    )
    return {"runs": [_run_dict(record, status) for record, status in runs]}


@router.get("/{run_id}")
async def get_create_run(
    run_id: str = Path(pattern=_RUN_ID_PATTERN),
) -> dict[str, Any]:
    """1 run の記録 + 導出状態。``run_id`` の形式検証は FastAPI の ``Path(pattern=)``
    が先に行う (不正形式は 422、正当な形式で未存在は下記 404)。
    """
    from backend.free.loop.staged.run_record import load_run

    create_dir = get_path_resolver().resolve_local("create_workspace_dir")
    loaded = load_run(create_dir, run_id, stale_after_sec=_stale_after_sec())
    if loaded is None:
        raise _create_runs_error(
            404, "E0404", "Run not found", "api.create_run_not_found", run_id=run_id,
        )
    record, status = loaded
    return _run_dict(record, status)


@router.get("/{run_id}/events")
async def get_create_run_events(
    run_id: str = Path(pattern=_RUN_ID_PATTERN),
    after: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """追記イベントログの watermark 付き読み (再接続の口、f_10 §7)。``run_id`` の
    形式検証は FastAPI の ``Path(pattern=)`` が先に行う。
    """
    from backend.free.loop.staged.run_record import read_events

    create_dir = get_path_resolver().resolve_local("create_workspace_dir")
    events = read_events(create_dir, run_id, after=after)
    if events is None:
        raise _create_runs_error(
            404, "E0404", "Run not found", "api.create_run_not_found", run_id=run_id,
        )
    last_seq = events[-1].seq if events else after
    return {
        "run_id": run_id,
        "after": after,
        "events": [e.to_dict() for e in events],
        "last_seq": last_seq,
    }


@router.get("/{run_id}/artifacts")
async def get_create_run_artifacts(
    run_id: str = Path(pattern=_RUN_ID_PATTERN),
) -> dict[str, Any]:
    """run の成果物 (workspace の ``src/**`` + ``SPEC.md`` + ``flowchart.md``、f_05 §4.5)。

    再接続後の editor 復元用の読み口。書き手は変わらず staged パイプライン
    (``chat_stream_staged.py``) だけ — ここは ``WorkspaceManager`` を読むだけの
    薄い層。
    """
    from backend.free.loop.staged.harness import _language_for_path
    from backend.free.loop.staged.workspace import WorkspaceManager

    create_dir = get_path_resolver().resolve_local("create_workspace_dir")
    workspace_root = FsPath(create_dir) / run_id
    if not workspace_root.is_dir():
        raise _create_runs_error(
            404, "E0404", "Run not found", "api.create_run_not_found", run_id=run_id,
        )
    ws = WorkspaceManager(root=workspace_root, workspace_id=run_id)
    artifacts: list[dict[str, Any]] = []
    for wf in ws.list_files(kind="src"):
        content = ws.read_file(wf.logical_path, kind="src")
        if content:
            artifacts.append({
                "path": wf.logical_path,
                "language": _language_for_path(wf.logical_path),
                "content": content,
            })
    spec_md = ws.read_spec()
    if spec_md:
        artifacts.append({"path": "SPEC.md", "language": "markdown", "content": spec_md})
    flowchart_md = ws.read_flowchart()
    if flowchart_md and flowchart_md.strip():
        # StagedCodeHarness (f_03 §4.4) が editor へ送るときと同じ体裁に揃える。
        fc_doc = f"# 設計フローチャート\n\n```mermaid\n{flowchart_md.strip()}\n```\n"
        artifacts.append({"path": "flowchart.md", "language": "markdown", "content": fc_doc})
    return {"run_id": run_id, "artifacts": artifacts}


__all__ = ["router"]
