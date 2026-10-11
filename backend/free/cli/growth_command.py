"""evoref growth サブコマンド — Level 2 (成長) の提案・採用・巻き戻し (docs/f_04 §6.8)。

``list`` / ``show ID`` / ``adopt ID`` / ``reject ID`` / ``rollback --kind prompt --mode chat``。

backend が動いていれば ``/api/learning/growth`` 系を使う (Level 1 の実行中の 409 などの判定は
backend 側)。動いていなければ束ねるパーティション (``model_paths.base_model`` の model_key) の
台帳と提案を直接読み、書き込みは単一書き手ロック (``<data_root>/g1/store/.writer.lock``) の下で
同じ :class:`~backend.free.learning.growth.service.GrowthService` を通す。停止中の採用は
採用後監視の基準値を経験から測らない (経験バッファを読まないため。利用者が採用した版は
もともと自動では巻き戻さない)。
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from backend.data_root import DataRootError, resolve_data_root, store_root
from backend.free.cli.backend_headers import backend_headers
from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import create_console, render_diff, render_error, render_info, render_table
from backend.i18n_helper import init_i18n, msg
from backend.io.writer_lock import WriterLockHeld, acquire_writer_lock
from backend.log_config import get_logger

logger = get_logger("cli.growth")

_DEFAULT_BACKEND = "http://localhost:8000"
_TIMEOUT = 60.0
_ACTIONS = ("list", "show", "adopt", "reject", "rollback")


class _BackendError(Exception):
    """backend が非 200 を返した (表示用の理由を持つ)。"""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evoref growth", description=msg("cli.help_growth"))
    parser.add_argument("action", choices=_ACTIONS)
    parser.add_argument("proposal_id", nargs="?", default=None, help="show / adopt / reject: gpp_<hex16>")
    parser.add_argument("--kind", choices=["prompt", "generation_params"], default=None)
    parser.add_argument("--mode", default=None, help="rollback: chat / create")
    parser.add_argument("--json", action="store_true", help="Print the response as JSON")
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="Data root used while the backend is stopped (default: EVOREF_DATA_ROOT or <install_root>/userdata)",
    )
    parser.add_argument(
        "--backend-url", default=_DEFAULT_BACKEND,
        help="Backend URL (default: http://localhost:8000)",
    )
    return parser


def run_growth(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref growth ...``)。"""
    if not _setup_encoding():
        return 1
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    if args.action in ("show", "adopt", "reject") and not args.proposal_id:
        render_error(console, msg("cli.growth_id_required", action=args.action))
        return 1
    if args.action == "rollback" and not (args.kind and args.mode):
        render_error(console, msg("cli.growth_rollback_usage"))
        return 1
    try:
        with httpx.Client(headers=backend_headers(), timeout=_TIMEOUT) as client:
            out = _remote(client, args)
        if out is None:
            out = _offline(args, console)
            if out is None:
                return 1
    except _BackendError as e:
        render_error(console, msg("cli.growth_failed", detail=e.detail))
        return 1
    except (httpx.HTTPError, OSError, ValueError) as e:
        logger.error("growth %s failed: %s", args.action, e)
        render_error(console, msg("cli.growth_failed", detail=str(e)))
        return 1
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    _render(console, args.action, out)
    return 0


# ── backend 稼働中 ──


def _request(client: httpx.Client, method: str, url: str, **kwargs: Any) -> dict | None:
    """backend を呼ぶ。届かなければ ``None`` (= 停止中)、非 200 は :class:`_BackendError`。"""
    try:
        resp = client.request(method, url, **kwargs)
    except httpx.ConnectError:
        return None
    if resp.status_code != 200:
        detail = f"HTTP {resp.status_code}"
        try:
            body = resp.json().get("detail")
            if isinstance(body, dict):
                reason = body.get("context", {}).get("reason")
                key = body.get("i18n_key") or ""
                text = msg(key) if key else str(body.get("message") or "")
                detail = f"{text} ({reason})" if reason else (text or detail)
            elif isinstance(body, str):
                detail = body
        except ValueError:
            pass
        raise _BackendError(resp.status_code, detail)
    return resp.json()


def _remote(client: httpx.Client, args: argparse.Namespace) -> dict | None:
    base = f"{args.backend_url.rstrip('/')}/api/learning/growth"
    if args.action == "list":
        return _request(client, "GET", base)
    if args.action == "show":
        return _request(client, "GET", f"{base}/proposals/{args.proposal_id}")
    if args.action in ("adopt", "reject"):
        return _request(client, "POST", f"{base}/proposals/{args.proposal_id}/{args.action}")
    return _request(client, "POST", f"{base}/rollback", json={"asset_kind": args.kind, "mode": args.mode})


# ── backend 停止中 ──


def _offline_scheduler(root: Path, data_root: Path) -> Any:
    """停止中の読み書きに使う ``LearningScheduler`` (束ねるパーティションの部品だけ)。"""
    from backend.config import PathResolver, read_config_file
    from backend.free.agent.prompt_manager import SystemPromptManager
    from backend.free.learning.generation_param_evolver import GenerationParamEvolver
    from backend.free.learning.level0_instant import ExperienceBuffer
    from backend.free.learning.scheduler import LearningScheduler

    config_path = root / "config.yaml"
    config = read_config_file(config_path, root) if config_path.is_file() else {}
    resolver = PathResolver(config, root, data_root=data_root)
    pm = SystemPromptManager(resolver.resolve_learning("prompts_dir"))
    sched = LearningScheduler(
        config=config, experience_buf=ExperienceBuffer(autosave=False),
        prompt_manager=pm, resolver=resolver,
    )
    sched.set_generation_param_evolver(
        GenerationParamEvolver(delta_file=resolver.resolve_learning("generation_deltas_file")),
    )
    return sched


def _offline_policy(data_root: Path) -> contextlib.AbstractContextManager[Any]:
    """停止中も組織ポリシーパックの固定を効かせる (docs/p_09 §4)。

    backend の稼働中は ``setup_pro`` が判定器を登録するが、この CLI のプロセスでは誰も
    登録しない。Pro が同梱されていれば、この操作の間だけ適用済みの状態から判定器を組んで
    登録する (Free 単独なら何もしない — 関数内の try-import、e_02 §4)。
    """
    try:
        from backend.pro.org_policy import offline_enforcer
    except ImportError:
        return contextlib.nullcontext()
    return offline_enforcer(data_root)


def _offline(args: argparse.Namespace, console: Console) -> dict | None:
    from backend.free.learning.growth.service import GrowthError

    root = _find_project_root()
    try:
        data_root = resolve_data_root(args.data_root, root=root)
    except DataRootError as e:
        render_error(console, msg("cli.data_root_invalid", detail=str(e)))
        return None
    if not args.json:
        render_info(console, msg("cli.growth_source_file", path=str(data_root)))
    lock = None
    if args.action in ("adopt", "reject", "rollback"):
        try:
            lock = acquire_writer_lock(store_root(data_root))
        except WriterLockHeld as e:
            render_error(console, msg("cli.growth_locked", detail=str(e)))
            return None
    try:
        with _offline_policy(data_root):
            service = _offline_scheduler(root, data_root).growth
            if args.action == "list":
                return service.overview()
            if args.action == "show":
                return service.proposal_summary(service.get_proposal(args.proposal_id), with_text=True)
            if args.action == "adopt":
                return service.adopt(args.proposal_id)
            if args.action == "reject":
                return service.reject(args.proposal_id)
            out = service.rollback(args.kind, args.mode)
            return {k: out[k] for k in ("asset_kind", "mode", "version", "row_id", "adoption_id")}
    except GrowthError as e:
        render_error(console, msg("cli.growth_failed", detail=f"{e.detail} ({e.reason})"))
        return None
    finally:
        if lock is not None:
            lock.release()


# ── 表示 ──


def _gate_text(gate: dict | None) -> str:
    if not gate:
        return "-"
    text = f"+{gate.get('wins') or 0}/-{gate.get('losses') or 0}/={gate.get('ties') or 0}"
    if gate.get("guards_passed") is not None or gate.get("guards_failed") is not None:
        text += f" guard {gate.get('guards_passed') or 0}/{gate.get('guards_failed') or 0}"
    return text


def _render(console: Console, action: str, out: dict) -> None:
    if action == "list":
        _render_list(console, out)
    elif action == "show":
        _render_show(console, out)
    elif action == "adopt":
        render_info(console, msg(
            "cli.growth_adopted", id=out.get("proposal_id"), kind=out.get("asset_kind"),
            mode=out.get("mode"), version=out.get("version"),
        ))
    elif action == "reject":
        render_info(console, msg("cli.growth_rejected", id=out.get("proposal_id")))
    else:
        render_info(console, msg(
            "cli.growth_rolled_back", kind=out.get("asset_kind"), mode=out.get("mode"),
            version=out.get("version"),
        ))


def _render_list(console: Console, out: dict) -> None:
    auto = out.get("auto_adopt") or {}
    render_info(console, msg(
        "cli.growth_auto_adopt", configured=auto.get("configured"), effective=auto.get("effective"),
        reason=auto.get("reason") or "-",
    ))
    reg = out.get("regression") or {}
    render_info(console, msg(
        "cli.growth_regression", active=reg.get("active", 0), guards=reg.get("guards", 0),
        pressure=reg.get("pressure", 0), flaky=reg.get("flaky", 0),
        quarantined=reg.get("quarantined", 0),
    ))
    for adoption in out.get("adoptions") or []:
        if adoption.get("rollback_suggested"):
            render_error(console, msg(
                "cli.growth_rollback_suggested", kind=adoption.get("asset_kind"),
                mode=adoption.get("mode"), version=adoption.get("adopted_version"),
                reason=adoption.get("suggestion_reason") or "-",
            ), level="warning")
    proposals = out.get("proposals") or []
    if not proposals:
        render_info(console, msg("cli.growth_no_proposals"))
    else:
        h = [msg(f"cli.growth_col_{c}") for c in ("id", "kind", "mode", "version", "gate", "expires")]
        render_table(console, [
            {
                h[0]: p.get("proposal_id"), h[1]: p.get("asset_kind"), h[2]: p.get("mode"),
                h[3]: f"v{p.get('parent_version')}" + (" (stale)" if p.get("stale") else ""),
                h[4]: _gate_text(p.get("measure")), h[5]: p.get("expires_at"),
            }
            for p in proposals
        ], h)
    recent = out.get("recent") or []
    if recent:
        render_info(console, msg("cli.growth_recent_title", count=len(recent)))
        h = [msg(f"cli.growth_col_{c}") for c in ("time", "op", "kind", "mode", "by", "reason")]
        render_table(console, [
            {
                h[0]: r.get("written_at"), h[1]: r.get("op"), h[2]: r.get("asset_kind"),
                h[3]: r.get("mode"), h[4]: r.get("decided_by") or "-", h[5]: r.get("reason") or "-",
            }
            for r in recent
        ], h)


def _render_show(console: Console, out: dict) -> None:
    render_info(console, msg(
        "cli.growth_show_header", id=out.get("proposal_id"), kind=out.get("asset_kind"),
        mode=out.get("mode"), parent=out.get("parent_version"), current=out.get("current_version"),
        gate=_gate_text(out.get("measure")), expires=out.get("expires_at"),
    ))
    if out.get("stale"):
        render_error(console, msg("cli.growth_stale"), level="warning")
    if out.get("asset_kind") == "prompt":
        diff = "".join(difflib.unified_diff(
            str(out.get("current_text") or "").splitlines(keepends=True),
            str(out.get("candidate_text") or "").splitlines(keepends=True),
            fromfile=f"v{out.get('current_version')}", tofile=str(out.get("proposal_id")),
        ))
        render_diff(console, diff or msg("cli.growth_no_diff"))
    else:
        render_info(console, json.dumps(
            {"current": out.get("current_deltas"), "candidate": out.get("candidate_deltas")},
            ensure_ascii=False,
        ))
