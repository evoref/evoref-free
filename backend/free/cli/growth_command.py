"""evoref growth サブコマンド — Level 2 (成長) の提案・採用・巻き戻し (docs/f_04 §6.8)。

``list`` / ``show ID`` / ``adopt ID`` / ``reject ID`` / ``rollback --kind prompt --mode chat``。
回帰台帳のケースは ``regression list [--status S] [--mode M] [--limit N] [--offset N]`` /
``regression retire ID [--reason TEXT] [--yes]`` / ``regression unretire ID [--yes]``
(retire / unretire は確認を取る。対話端末でなければ ``--yes`` が要る)。

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
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from rich.console import Console
from rich.markup import escape

from backend.data_root import DataRootError, resolve_data_root
from backend.free.cli.cli_common import (
    EXIT_FAIL,
    EXIT_INTERRUPTED,
    EXIT_UNAVAILABLE,
    BackendError,
    add_common_args,
    diag_console,
    emit,
    request,
    resolve_backend_url,
    run_dual,
    writer_locked,
)
from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import create_console, render_diff, render_error, render_info, render_table
from backend.i18n_helper import init_i18n, msg
from backend.io.id_registry import is_valid_id
from backend.io.readonly import DataReadonlyError
from backend.io.writer_lock import WriterLockHeld
from backend.log_config import get_logger

logger = get_logger("cli.growth")

_TIMEOUT = 60.0
_ACTIONS = ("list", "show", "adopt", "reject", "rollback")
_WRITE_ACTIONS = ("adopt", "reject", "rollback")
#: ``regression`` は :func:`_run_regression` が受ける (ここでは --help に出すだけ)。
_REGRESSION = "regression"
_REGRESSION_WRITE_ACTIONS = ("retire", "unretire")
_CASE_PREFIX = "rgc_"
_LIST_LIMIT_MAX = 200


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evoref growth", description=msg("cli.help_growth"))
    parser.add_argument("action", choices=(*_ACTIONS, _REGRESSION))
    parser.add_argument("proposal_id", nargs="?", default=None, help="show / adopt / reject: gpp_<hex16>")
    parser.add_argument("--kind", choices=["prompt", "generation_params"], default=None)
    parser.add_argument("--mode", default=None, help="rollback: chat / create")
    add_common_args(parser, data_root=True)
    return parser


def run_growth(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref growth ...``)。"""
    if not _setup_encoding():
        return 1
    init_i18n()
    if argv and argv[0] == "regression":
        return _run_regression(argv[1:])
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action == _REGRESSION:
        parser.error("put regression first: evoref growth regression list|retire|unretire ...")
    console = create_console()
    diag = diag_console(args.json, console)
    if args.action in ("show", "adopt", "reject") and not args.proposal_id:
        render_error(diag, msg("cli.growth_id_required", action=args.action))
        return EXIT_FAIL
    if args.action == "rollback" and not (args.kind and args.mode):
        render_error(diag, msg("cli.growth_rollback_usage"))
        return EXIT_FAIL
    try:
        out = run_dual(
            lambda client: _remote(client, args), lambda: _offline(args, diag), timeout=_TIMEOUT,
        )
    except BackendError as e:
        render_error(diag, msg("cli.growth_failed", detail=e.detail))
        return EXIT_FAIL
    except (httpx.HTTPError, OSError, ValueError) as e:
        logger.error("growth %s failed: %s", args.action, e)
        render_error(diag, msg("cli.growth_failed", detail=str(e)))
        return EXIT_FAIL
    if out is None:
        return EXIT_FAIL
    return emit(out, as_json=args.json, render=lambda: _render(console, args.action, out))


# ── backend 稼働中 ──


def _remote(client: httpx.Client, args: argparse.Namespace) -> dict | None:
    """backend を呼ぶ。届かなければ ``None`` (= 停止中)、2xx 以外は :class:`BackendError`。"""
    base = f"{resolve_backend_url(args.backend_url)}/api/learning/growth"
    if args.action == "list":
        return request(client, "GET", base)
    if args.action == "show":
        return request(client, "GET", f"{base}/proposals/{args.proposal_id}")
    if args.action in ("adopt", "reject"):
        return request(client, "POST", f"{base}/proposals/{args.proposal_id}/{args.action}")
    return request(client, "POST", f"{base}/rollback", json={"asset_kind": args.kind, "mode": args.mode})


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
    lock = writer_locked(data_root) if args.action in _WRITE_ACTIONS else contextlib.nullcontext()
    try:
        with lock, _offline_policy(data_root):
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
    except WriterLockHeld as e:
        render_error(console, msg("cli.growth_locked", detail=str(e)))
        return None
    except GrowthError as e:
        render_error(console, msg("cli.growth_failed", detail=f"{e.detail} ({e.reason})"))
        return None


# ── regression (回帰台帳のケース) ──


class _OfflineFailed(Exception):
    """停止中の経路の失敗 (表示済み。``code`` は終了コード)。"""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _build_regression_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common, data_root=True)
    parser = argparse.ArgumentParser(prog="evoref growth regression", description=msg("cli.growth_regression_help"))
    sub = parser.add_subparsers(dest="action", required=True)
    lst = sub.add_parser("list", parents=[common], help=msg("cli.growth_regression_help_list"))
    lst.add_argument(
        "--status", default=None, choices=["active", "guard", "pressure", "flaky", "quarantined", "retired"],
    )
    lst.add_argument("--mode", default=None, help="chat / create")
    lst.add_argument("--limit", type=int, default=50, metavar="N", help=f"1-{_LIST_LIMIT_MAX} (default: 50)")
    lst.add_argument("--offset", type=int, default=0, metavar="N", help="Skip the first N cases (default: 0)")
    retire = sub.add_parser("retire", parents=[common], help=msg("cli.growth_regression_help_retire"))
    retire.add_argument("case_id", help="rgc_<hex16> (see regression list)")
    retire.add_argument("--reason", default="", metavar="TEXT", help="Why the case is retired (kept with the case)")
    unretire = sub.add_parser("unretire", parents=[common], help=msg("cli.growth_regression_help_unretire"))
    unretire.add_argument("case_id", help="rgc_<hex16> (see regression list --status retired)")
    for p in (retire, unretire):
        p.add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation")
    return parser


def _run_regression(argv: list[str]) -> int:
    """``evoref growth regression list|retire|unretire``。

    dual: 稼働中は API、停止中は回帰台帳を直接読む / 単一書き手ロックの下で書く
    (retire / unretire は adopt / reject と同じ作法)。
    """
    args = _build_regression_parser().parse_args(argv)
    console = create_console()
    diag = diag_console(args.json, console)
    if args.action == "list" and not (1 <= args.limit <= _LIST_LIMIT_MAX and args.offset >= 0):
        render_error(diag, msg("cli.growth_regression_bad_limit"))
        return EXIT_UNAVAILABLE
    if args.action != "list":
        if not is_valid_id(args.case_id, _CASE_PREFIX):
            render_error(diag, msg("cli.growth_regression_bad_case_id", case_id=args.case_id))
            return EXIT_UNAVAILABLE
        if not args.yes and not _regression_confirmed(diag, args):
            return EXIT_FAIL
    try:
        out = run_dual(
            lambda client: _regression_remote(client, args),
            lambda: _regression_offline(args, diag),
            timeout=_TIMEOUT,
        )
    except _OfflineFailed as e:
        return e.code
    except BackendError as e:
        render_error(diag, msg("cli.growth_failed", detail=e.detail))
        return EXIT_UNAVAILABLE if e.status in (423, 503) else EXIT_FAIL
    except (httpx.HTTPError, OSError, ValueError) as e:
        logger.error("growth regression %s failed: %s", args.action, e)
        render_error(diag, msg("cli.growth_failed", detail=str(e)))
        return EXIT_FAIL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    return emit(out, as_json=args.json, render=lambda: _render_regression(console, args.action, out))


def _regression_confirmed(console: Console, args: argparse.Namespace) -> bool:
    """retire / unretire の確認。対話端末でなければ確認できないので拒否する。"""
    if sys.stdin is None or not sys.stdin.isatty():
        render_error(console, msg("cli.growth_regression_needs_yes"))
        return False
    answer = console.input(
        msg(f"cli.growth_regression_confirm_{args.action}", id=args.case_id) + " [y/N] ", markup=False,
    )
    if answer.strip().lower() in ("y", "yes"):
        return True
    render_info(console, msg("cli.growth_regression_aborted"))
    return False


def _regression_remote(client: httpx.Client, args: argparse.Namespace) -> dict | None:
    """backend を呼ぶ。届かなければ ``None`` (= 停止中)、2xx 以外は :class:`BackendError`。"""
    base = f"{resolve_backend_url(args.backend_url)}/api/learning/growth/regression"
    if args.action == "list":
        params: dict[str, Any] = {"limit": args.limit, "offset": args.offset}
        if args.status:
            params["status"] = args.status
        if args.mode:
            params["mode"] = args.mode
        return request(client, "GET", base, params=params)
    body = {"by": "cli", "reason": getattr(args, "reason", "") or ""}
    return request(client, "POST", f"{base}/{quote(args.case_id, safe='')}/{args.action}", json=body)


def _regression_offline(args: argparse.Namespace, console: Console) -> dict:
    """停止中: 回帰台帳を直接読む。retire / unretire は単一書き手ロックの下で書く。"""
    from backend.free.learning.growth.service import GrowthError, GrowthReadonly

    root = _find_project_root()
    try:
        data_root = resolve_data_root(args.data_root, root=root)
    except DataRootError as e:
        render_error(console, msg("cli.data_root_invalid", detail=str(e)))
        raise _OfflineFailed(EXIT_UNAVAILABLE) from e
    if not args.json:
        render_info(console, msg("cli.growth_source_file", path=str(data_root)))
    write = args.action in _REGRESSION_WRITE_ACTIONS
    lock = writer_locked(data_root) if write else contextlib.nullcontext()
    try:
        with lock, _offline_policy(data_root):
            service = _offline_scheduler(root, data_root).growth
            if args.action == "list":
                return service.regression_cases(
                    status=args.status, mode=args.mode, limit=args.limit, offset=args.offset,
                )
            if args.action == "retire":
                return service.retire_case(args.case_id, by="cli", reason=args.reason)
            return service.unretire_case(args.case_id, by="cli")
    except WriterLockHeld as e:
        render_error(console, msg("cli.growth_locked", detail=str(e)))
        raise _OfflineFailed(EXIT_UNAVAILABLE) from e
    except DataReadonlyError as e:  # 台帳に新しい版の行がある (c_05 §0.4.5)
        render_error(console, msg("cli.growth_failed", detail=str(e)))
        raise _OfflineFailed(EXIT_UNAVAILABLE) from e
    except GrowthError as e:
        render_error(console, msg("cli.growth_failed", detail=f"{e.detail} ({e.reason})"))
        raise _OfflineFailed(EXIT_UNAVAILABLE if isinstance(e, GrowthReadonly) else EXIT_FAIL) from e


def _render_regression(console: Console, action: str, out: dict) -> None:
    if action == "retire":
        if out.get("changed"):
            render_info(console, msg(
                "cli.growth_regression_retired", id=out.get("case_id"), previous=out.get("previous_status") or "-",
            ))
        else:
            render_info(console, msg("cli.growth_regression_unchanged", id=out.get("case_id")))
        return
    if action == "unretire":
        render_info(console, msg("cli.growth_regression_unretired", id=out.get("case_id"), status=out.get("status")))
        return
    cases = out.get("cases") or []
    if not cases:
        render_info(console, msg("cli.growth_regression_empty"))
        return
    cols = ("id", "status", "role", "source", "mode", "written", "query")
    h = [msg(f"cli.growth_regression_col_{c}") for c in cols]
    render_table(console, [
        {
            h[0]: c.get("case_id"), h[1]: c.get("status"), h[2]: c.get("role") or "-",
            h[3]: c.get("source"), h[4]: c.get("mode"), h[5]: c.get("written_at") or "-",
            h[6]: escape(str(c.get("query_preview") or "")),
        }
        for c in cases
    ], h)
    render_info(console, msg(
        "cli.growth_regression_page", shown=len(cases), total=out.get("total", len(cases)),
        offset=out.get("offset", 0),
    ))


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
