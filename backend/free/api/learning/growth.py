"""Level 2 (成長) の API — 提案の採用 / 却下・巻き戻し・回帰ケースの除外 (docs/f_04 §6.8)。

すべて Free (学習の資産を見る・直す・巻き戻す・採用理由を知るのはデータ主権、e_01 §0)。
書き込みは束ねた active パーティションの学習ストアへ、イベントループの上で同期に行う
(Level 1 の実行中かを見てから書くまでの間に Level 1 が始まらないように)。
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.free.api._error_responses import api_error
from backend.free.api.learning._growth_schemas import (
    GrowthActionResponse,
    GrowthProposalDetail,
    GrowthResponse,
    GrowthRollbackRequest,
    RegressionRetireResponse,
)
from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context

logger = get_logger("api.learning.growth")

router = APIRouter(prefix="/api/learning/growth", tags=["learning"])

#: 失敗の理由 → i18n キー (``api.*``)。
_REASON_KEYS: dict[str, str] = {
    "proposal_not_found": "api.growth_proposal_not_found",
    "case_not_found": "api.growth_case_not_found",
    "unknown_asset_kind": "api.growth_unknown_asset_kind",
    "unsupported_asset_kind": "api.growth_unknown_asset_kind",
    "stale_parent": "api.growth_stale_parent",
    "learning_running": "api.growth_learning_running",
    "identical_after_normalization": "api.growth_identical",
    "no_rollback_target": "api.growth_no_rollback_target",
    "generation_evolver_unavailable": "api.growth_unavailable",
    "data_readonly": "api.data_readonly",
    # 組織ポリシーパックで固定された資産 (docs/p_09 §4)
    "pinned_by_policy": "api.policy_denied",
}


def _service(state: AppState) -> Any:
    scheduler = state.learning_scheduler
    if scheduler is None:
        raise api_error(
            503, "E0503", "Learning scheduler not initialized",
            "api.learning_scheduler_not_initialized",
        )
    return scheduler.growth


def _raise(exc: Exception) -> None:
    """``GrowthError`` を構造化エラーにする (それ以外は送出し直す)。"""
    from backend.free.learning.growth.service import GrowthError

    if not isinstance(exc, GrowthError):
        raise exc
    key = _REASON_KEYS.get(exc.reason, "")
    context = {**getattr(exc, "context", {}), "reason": exc.reason}
    raise api_error(
        exc.status, f"E0{exc.status}", exc.detail,
        key, **context,
    ) from exc


async def _in_executor(fn, *args):
    return await run_in_executor_with_context(
        asyncio.get_running_loop(), None, functools.partial(fn, *args),
    )


@router.get("", response_model=GrowthResponse)
async def growth_overview(state: AppState = Depends(get_app_state)) -> GrowthResponse:
    """自動採用の状態・未決の提案・直近の台帳・回帰台帳の件数・採用した版の記録。"""
    service = _service(state)
    try:
        # 健康診断 (自動採用の設定が真のときだけ) は経験の全走査なので executor で回す
        overview = await _in_executor(service.overview)
    except Exception as exc:  # noqa: BLE001
        _raise(exc)
        raise
    return GrowthResponse(**overview)


@router.get("/proposals/{proposal_id}", response_model=GrowthProposalDetail)
async def growth_proposal(
    proposal_id: str, state: AppState = Depends(get_app_state),
) -> GrowthProposalDetail:
    """提案の詳細 (候補と現行の本文。差分表示用)。"""
    service = _service(state)
    try:
        proposal = service.get_proposal(proposal_id)
        return GrowthProposalDetail(**service.proposal_summary(proposal, with_text=True))
    except Exception as exc:  # noqa: BLE001
        _raise(exc)
        raise


@router.post("/proposals/{proposal_id}/adopt", response_model=GrowthActionResponse)
async def adopt_proposal(
    proposal_id: str, state: AppState = Depends(get_app_state),
) -> GrowthActionResponse:
    """提案を採用する。``parent_version`` が現行と違えば / Level 1 の実行中は 409。"""
    service = _service(state)
    try:
        out = service.adopt(proposal_id)
    except Exception as exc:  # noqa: BLE001
        _raise(exc)
        raise
    return GrowthActionResponse(**out)


@router.post("/proposals/{proposal_id}/reject", response_model=GrowthActionResponse)
async def reject_proposal(
    proposal_id: str, state: AppState = Depends(get_app_state),
) -> GrowthActionResponse:
    """提案を却下する (台帳に ``reject`` を書いて提案を消す)。"""
    service = _service(state)
    try:
        out = service.reject(proposal_id)
    except Exception as exc:  # noqa: BLE001
        _raise(exc)
        raise
    return GrowthActionResponse(**out)


@router.post("/rollback", response_model=GrowthActionResponse)
async def rollback_asset(
    req: GrowthRollbackRequest, state: AppState = Depends(get_app_state),
) -> GrowthActionResponse:
    """``{asset_kind, mode}`` を直前の版へ戻す (``decided_by=user``)。"""
    service = _service(state)
    if req.mode not in service.s.MODES:
        raise api_error(
            400, "E0400", f"unknown mode: {req.mode}", "api.growth_unknown_mode", mode=req.mode,
        )
    try:
        out = service.rollback(req.asset_kind, req.mode)
    except Exception as exc:  # noqa: BLE001
        _raise(exc)
        raise
    return GrowthActionResponse(
        asset_kind=out["asset_kind"], mode=out["mode"], version=out["version"],
        row_id=out["row_id"], adoption_id=out["adoption_id"],
    )


@router.post("/regression/{case_id}/retire", response_model=RegressionRetireResponse)
async def retire_regression_case(
    case_id: str, state: AppState = Depends(get_app_state),
) -> RegressionRetireResponse:
    """回帰ケースを ``retired`` にする (誤った訂正から収穫したケースを外す)。"""
    service = _service(state)
    try:
        out = service.retire_case(case_id)
    except Exception as exc:  # noqa: BLE001
        _raise(exc)
        raise
    return RegressionRetireResponse(**out)
