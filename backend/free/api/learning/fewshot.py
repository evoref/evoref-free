"""few-shot 手本の可視化と手動操作 (f_04 §3.2.2、2026-09-10 (h))。

学習された手本はこれまでユーザーに見えず、誤答が手本になっても消す手段が
無かった (監査で繰り返し「誤答が few-shot 手本になる」が出ている)。自動の
検証より人手の 1 クリック削除が確実なので、一覧 + pin / unpin / archive /
restore を提供する。操作は in-memory プールに効き、その場で
``fewshot_pool.json`` へ保存する。
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.free.api._error_responses import api_error
from backend.free.api._pagination import LimitParam, OffsetParam, paginate
from backend.io.readonly import is_readonly
from backend.log_config import get_logger

logger = get_logger("api")

router = APIRouter(prefix="/api/learning/fewshot", tags=["learning"])


class FewShotExampleModel(BaseModel):
    id: str
    mode: str
    query: str
    response: str
    fitness: float
    quality_score: float | None
    added_at: str
    state: str
    state_since: str
    pinned: bool
    use_count: int
    last_used_at: str
    helpful: int
    harmful: int
    source_experience_id: str
    lang: str


class FewShotListResponse(BaseModel):
    examples: list[FewShotExampleModel]
    archived: list[FewShotExampleModel]
    pool_size: int


class FewShotActionResponse(BaseModel):
    id: str
    action: str
    ok: bool


def _pool(state: AppState):
    scheduler = state.learning_scheduler
    pool = getattr(scheduler, "_fewshot_pool", None) if scheduler else None
    if pool is None:
        raise api_error(
            503, "E0503", "Learning scheduler not initialized",
            "api.learning_scheduler_not_initialized",
        )
    return pool


def _to_model(ex) -> FewShotExampleModel:
    return FewShotExampleModel(
        id=ex.id, mode=ex.mode, query=ex.query, response=ex.response,
        fitness=float(ex.fitness), quality_score=ex.quality_score,
        added_at=ex.added_at, state=ex.state, state_since=ex.state_since,
        pinned=bool(ex.pinned), use_count=int(ex.use_count),
        last_used_at=ex.last_used_at, helpful=int(ex.helpful),
        harmful=int(ex.harmful), source_experience_id=ex.source_experience_id,
        lang=ex.lang,
    )


def _save(state: AppState, pool) -> None:
    scheduler = state.learning_scheduler
    path = scheduler.prompt_manager.prompt_dir / "fewshot_pool.json"
    try:
        pool.save(path)
    except OSError as exc:
        logger.warning("fewshot pool save failed after UI action: %s", exc)


#: ``archived`` の絞る前の件数を返す応答ヘッダ (``examples`` の件数は ``X-Total-Count``)。
ARCHIVED_TOTAL_COUNT_HEADER = "X-Archived-Total-Count"


@router.get("", response_model=FewShotListResponse)
async def list_fewshot(
    response: Response,
    mode: str | None = None,
    include_archived: bool = True,
    limit: LimitParam = None,
    offset: OffsetParam = 0,
    state: AppState = Depends(get_app_state),
) -> FewShotListResponse:
    """プール内 (active / stale) と退避済みの手本を返す。

    ``include_archived=false`` なら ``archived`` は空 (本文を返さない。件数は
    ヘッダで返す)。``limit`` / ``offset`` は任意 (省けば全件) で、``examples`` と
    ``archived`` にそれぞれ同じ窓を掛ける。絞る前の件数は ``X-Total-Count``
    (examples) と ``X-Archived-Total-Count`` (archived、``include_archived`` に関わらず)。
    """
    pool = _pool(state)
    examples = paginate(pool.list_examples(mode), limit, offset, response)
    archived = paginate(
        pool.list_archived(mode), limit, offset, response,
        header=ARCHIVED_TOTAL_COUNT_HEADER,
    )
    if not include_archived:
        archived = []
    return FewShotListResponse(
        examples=[_to_model(ex) for ex in examples],
        archived=[_to_model(ex) for ex in archived],
        pool_size=int(pool.pool_size),
    )


Action = Literal["pin", "unpin", "archive", "restore"]


@router.post("/{example_id}/{action}", response_model=FewShotActionResponse)
async def act_on_fewshot(
    example_id: str, action: Action, state: AppState = Depends(get_app_state),
) -> FewShotActionResponse:
    """pin / unpin / archive / restore。対象が無ければ 404、保存できなければ 423。"""
    pool = _pool(state)
    # 保存できない状態でメモリだけ変えると、画面は変わったのに再起動で戻る。
    # 変更の前に断る (データ根の readonly と、プールのファイル単位の readonly)。
    if pool.readonly or is_readonly():
        raise api_error(
            423, "E0423", "Few-shot pool is read-only; nothing was changed",
            "api.data_readonly",
        )
    if action == "pin":
        ok = pool.set_pinned(example_id, True)
    elif action == "unpin":
        ok = pool.set_pinned(example_id, False)
    elif action == "archive":
        ok = pool.archive_example(example_id)
    else:
        ok = pool.restore_example(example_id)
    if not ok:
        raise api_error(
            404, "E0404", f"few-shot example not found: {example_id}",
            "api.fewshot_not_found", example_id=example_id,
        )
    _save(state, pool)
    logger.info("fewshot %s: id=%s", action, example_id)
    return FewShotActionResponse(id=example_id, action=action, ok=True)
