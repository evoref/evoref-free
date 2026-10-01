"""環境調整 (auto-tune) API (c_16 §7.2.3 / c_06 §2.7)

測定本体は ``backend.free.core.tuning`` (CLI の ``evoref tune`` と共有の runner)。この router は
実行状態 (``AppState.auto_tune_runner``) を読み書きする薄い層で、稼働中に走らせてよいのは
``safe_while_running`` の項目だけ。書くのは ``cache/`` だけなので readonly でも動く。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app_state import AppState, get_app_state
from backend.free.api._error_responses import api_error
from backend.free.core.tuning.service import ANSWERS, AutoTuneBusy, AutoTuneService
from backend.free.core.tuning.store import TunePaths, resolve_tune_paths
from backend.log_config import get_logger

logger = get_logger("api.auto_tune")

router = APIRouter(prefix="/api/system/auto-tune", tags=["system"])


class AutoTuneRunRequest(BaseModel):
    """``POST /run`` の本文。``only`` は項目のキー (``None`` は全項目)。"""

    only: list[str] | None = None
    force: bool = False


class AutoTuneDecisionRequest(BaseModel):
    """``POST /decision`` の本文。値の検査は handler で行う (i18n キー付きの 422 にするため)。"""

    decision: str


def _context() -> tuple[dict[str, Any], Path, TunePaths]:
    """(config, インストール根, 3 ファイルの置き場)。config 未ロードは 503。"""
    from backend.config import get_config, get_path_resolver

    try:
        cfg = get_config()
        resolver = get_path_resolver()
    except RuntimeError as e:
        raise api_error(503, "E0503", "Config not initialized") from e
    return cfg, resolver.root, resolve_tune_paths(cfg, resolver.root, resolver)


def _service(state: AppState) -> AutoTuneService:
    service = getattr(state, "auto_tune_runner", None)
    if service is None:
        raise api_error(503, "E0503", "Auto-tune runner not initialized")
    return service


def _busy() -> Exception:
    return api_error(409, "E0409", "Auto-tune is already running", "api.auto_tune_running")


def _refresher(state: AppState, cfg: dict[str, Any], root: Path, paths: TunePaths):
    """実行 / 確認の答えの後に ``/api/status.auto_tune`` (起動時の 1 回読み) を読み直す。"""

    def refresh() -> None:
        from backend.free.core.tuning.gate import load_auto_tune_status

        gen = getattr(state, "gen", None)
        if gen is not None:
            service = getattr(state, "auto_tune_runner", None)
            current = getattr(service, "pc", None) if service is not None else None
            gen.auto_tune = load_auto_tune_status(cfg, root, paths=paths, current=current)

    return refresh


@router.get("")
async def get_auto_tune(state: AppState = Depends(get_app_state)) -> dict[str, Any]:
    """状態・進捗・現在の PC・項目ごとの結果・確認状態を返す (測らない)。"""
    cfg, root, paths = _context()
    return _service(state).snapshot(cfg, root, paths)


@router.post("/run", status_code=202)
async def run_auto_tune(
    body: AutoTuneRunRequest, state: AppState = Depends(get_app_state),
) -> dict[str, str]:
    """環境調整を裏で実行する (202)。実行中は 409。

    稼働中に走らせられない項目は実行せず結果に載せる: base に効く見積りは予約 (``scheduled``、次の
    起動 / 再起動で再計算)、本番のポートで測る項目は ``requires_stop`` (停止中の CLI だけ)。
    """
    cfg, root, paths = _context()
    service = _service(state)
    try:
        await service.start(
            cfg, root, paths, only=body.only, force=body.force,
            on_finished=_refresher(state, cfg, root, paths),
        )
    except AutoTuneBusy:
        raise _busy() from None
    except ValueError as e:
        raise api_error(422, "E0422", f"Unknown auto-tune item: {e}", "api.auto_tune_unknown_item", detail=str(e)) from e
    logger.info("Auto-tune run started (only=%s, force=%s)", body.only, body.force)
    return {"state": "running"}


@router.post("/decision")
async def answer_decision(
    body: AutoTuneDecisionRequest, state: AppState = Depends(get_app_state),
) -> dict[str, str]:
    """環境移行の確認に答える。実行は始めない (画面は decision → run の 2 呼び出し)。

    ``unchanged`` は測らずに指紋だけを書き直す (ホスト名だけが変わったときのみ。違えば 409)。
    """
    if body.decision not in ANSWERS:
        raise api_error(
            422, "E0422", f"Invalid decision: {body.decision!r}", "api.auto_tune_invalid_decision",
            allowed=list(ANSWERS),
        )
    cfg, root, paths = _context()
    try:
        ok, reason = await _service(state).answer(paths, body.decision, cfg=cfg)
    except AutoTuneBusy:
        raise _busy() from None
    if not ok:
        # 答え自体は正しい (検査済み)。拒否はホスト名以外の軸が変わった「変更なし」や書き込みの失敗
        raise api_error(409, "E0409", reason, "api.auto_tune_restamp_refused", decision=body.decision, detail=reason)
    _refresher(state, cfg, root, paths)()
    logger.info("Auto-tune decision recorded: %s", body.decision)
    return {"decision": body.decision}


__all__ = ["router"]
