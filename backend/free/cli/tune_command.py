"""evoref tune サブコマンド — 環境調整 (PC のスペックに依存する値の測定・見積り・確認)。

設計は docs/c_16_evidence_store.md §7.2.3 / docs/f_06_cli.md §2.7a。判定・実行・保存は
``backend.free.core.tuning`` に任せ、ここは引数・表示・終了コード・確認の質問だけを持つ。

- ``status`` — 現在の PC・各項目の現在値 / 提案値・手動設定の項目・確認状態 (``--json`` は stdout が JSON のみ)
- ``run [--only a,b] [--force]`` — 停止中は ``runner`` を直接、稼働中は backend の API を呼ぶ
- ``--startup-check`` — 環境移行の確認と、新規 / 承認済みのときの見積り項目の自動実行
  (``evoref serve`` / ``evoref-ctl`` が llama-server を起こす前に呼ぶ)。どんな失敗でも起動を止めない (終了コード 0)

終了コードは doctor に揃える: 0 = 変更不要 / 1 = 提案あり・エラー / 2 = 実行不能。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape

from backend.log_config import get_logger

logger = get_logger("cli.tune")

EXIT_OK = 0
EXIT_PROPOSAL = 1
EXIT_CANNOT_RUN = 2

_DEFAULT_BACKEND = "http://localhost:8000"
#: API の進捗を読む間隔と、待つ上限 (秒)。
_POLL_SEC = 1.0
_POLL_MAX_SEC = 3600.0
#: 確認の質問を聞き直す上限 (入力が読めないものが続いたら pending のまま進む)。
_ASK_ATTEMPTS = 3


# ── 共通 ──────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    from backend.i18n_helper import msg

    parser = argparse.ArgumentParser(prog="evoref tune", description=msg("cli.help_tune"))
    parser.add_argument(
        "action", nargs="?", default="status", choices=["status", "run"],
        help="status (default): show the PC and the tuned values. run: measure / estimate and save",
    )
    parser.add_argument("--only", default=None, metavar="A,B", help="Run only these items (comma separated)")
    parser.add_argument("--force", action="store_true", help="Measure again even if saved results are still valid")
    parser.add_argument(
        "--startup-check", action="store_true", dest="startup_check",
        help="Check for an environment move before start (asks on a terminal; never blocks the start)",
    )
    parser.add_argument("--json", action="store_true", help="Print the machine-readable result (stdout is JSON only)")
    parser.add_argument("--yes", "-y", action="store_true", help="Accept the migration question without asking")
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="Data root (default: EVOREF_DATA_ROOT or <install_root>/userdata)",
    )
    parser.add_argument("--backend-url", default=_DEFAULT_BACKEND, help="Backend URL (default: http://localhost:8000)")
    return parser


def _load_cfg(project_root: Path) -> dict[str, Any] | None:
    """config.yaml を dict で読む。無い・読めないときは ``None``。"""
    import yaml

    path = project_root / "config.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        logger.warning("config.yaml unreadable for tune: %s", e)
        return None
    return data if isinstance(data, dict) else {}


def _current_pc() -> Any:
    """この PC の指紋の材料 (``--list-devices`` を読むので GPU 名も入る)。"""
    from backend.free.core.tuning.hardware import probe_hardware
    from backend.free.rag.rerank_selftest import collect_pc_info

    return collect_pc_info(probe_hardware().gpu_names)


def _is_interactive() -> bool:
    """標準入力が端末か (reset の ``_confirmed`` と同じ判定)。"""
    return bool(sys.stdin and sys.stdin.isatty())


def _make_console(as_json: bool) -> Console:
    """``--json`` のときは stdout を JSON だけにするため、補助の表示を stderr へ出す。"""
    if as_json:
        return Console(stderr=True, no_color=True, highlight=False)
    from backend.free.cli.renderer import create_console

    return create_console(no_color=not (sys.stdout and sys.stdout.isatty()))


def _fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _pc_dict(pc: Any) -> dict[str, Any]:
    return {
        "hostname": pc.hostname, "cpu": pc.cpu, "logical_cores": pc.logical_cores,
        "memory_gb": pc.memory_gb, "gpus": list(pc.gpus),
    }


def _item_dict(item: Any) -> dict[str, Any]:
    return {
        "key": item.key, "value": item.value, "source": item.source, "reason": item.reason,
        "config_key": item.config_key, "applied": item.applied,
        "requires_restart": item.requires_restart, "measured_at": item.measured_at,
    }


def _local_view(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """停止中の状態を API (``GET /api/system/auto-tune``) と同じ形で作る。"""
    from backend.free.core.tuning.gate import load_gate
    from backend.free.core.tuning.store import load_auto_tune, resolve_tune_paths

    paths = resolve_tune_paths(cfg, project_root)
    pc = _current_pc()
    gate = load_gate(cfg, project_root, pc, paths=paths)
    record, _ = load_auto_tune(paths.auto_tune)
    items = [_item_dict(i) for i in record.items.values()] if record is not None else []
    return {
        "state": "idle", "progress": None, "pc": _pc_dict(pc), "items": items,
        "decision": gate.decision, "changed_axes": list(gate.changed_axes),
        "error": None, "restart_required": False,
    }


def _current_config_value(cfg: dict[str, Any], config_key: str) -> Any:
    from backend.free.core.tuning.items import config_value

    return config_value(cfg, config_key) if config_key else None


def _item_row(item: dict[str, Any], cfg: dict[str, Any]) -> dict[str, str]:
    from backend.i18n_helper import msg

    proposed = _fmt_value(item.get("value"))
    if item.get("applied"):
        note = msg("cli.tune_note_restart") if item.get("requires_restart") else ""
        status = msg("cli.tune_status_applied")
    else:
        status = msg("cli.tune_status_manual")
        if item.get("reason") == "manual":
            note = msg("cli.tune_note_manual", value=proposed)
        elif item.get("reason") == "not_auto":
            note = msg("cli.tune_note_not_auto", value=proposed)
        else:
            note = str(item.get("reason") or item.get("source") or "")
    return {
        msg("cli.tune_col_item"): str(item.get("key", "")),
        msg("cli.tune_col_config"): str(item.get("config_key") or "-"),
        msg("cli.tune_col_current"): _fmt_value(_current_config_value(cfg, str(item.get("config_key") or ""))),
        msg("cli.tune_col_proposed"): proposed,
        msg("cli.tune_col_status"): status,
        msg("cli.tune_col_note"): note,
    }


def _render_view(console: Console, view: dict[str, Any], cfg: dict[str, Any]) -> None:
    from backend.free.cli.renderer import render_info, render_table
    from backend.i18n_helper import msg

    pc = view.get("pc") or {}
    render_info(console, msg(
        "cli.tune_pc", hostname=pc.get("hostname") or "-", cpu=pc.get("cpu") or "-",
        cores=pc.get("logical_cores", 0), memory_gb=pc.get("memory_gb", 0),
        gpus=", ".join(pc.get("gpus") or []) or "-",
    ))
    decision = str(view.get("decision") or "auto")
    render_info(console, msg("cli.tune_decision_line", decision=msg(f"cli.tune_decision.{decision}")))
    axes = list(view.get("changed_axes") or [])
    if axes:
        render_info(console, msg("cli.tune_changed_axes", axes=", ".join(msg(f"cli.tune_axis.{a}") for a in axes)))
    items = list(view.get("items") or [])
    if not items:
        render_info(console, msg("cli.tune_no_items"))
        return
    headers = [msg(f"cli.tune_col_{c}") for c in ("item", "config", "current", "proposed", "status", "note")]
    render_table(console, [_item_row(i, cfg) for i in items], headers)


def _has_proposal(view: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """提案ありか: 確認待ち、または手動設定の項目で提案値が現在値と違うもの。"""
    if view.get("decision") == "pending":
        return True
    for item in view.get("items") or []:
        if item.get("applied") or item.get("reason") != "manual":
            continue
        if item.get("value") != _current_config_value(cfg, str(item.get("config_key") or "")):
            return True
    return False


# ── 稼働中の API ───────────────────────────────────────────


def _client() -> Any:
    import httpx

    from backend.free.cli.backend_headers import backend_headers

    return httpx.Client(headers=backend_headers(), timeout=30.0)


def _error_detail(resp: Any) -> str:
    try:
        detail = resp.json().get("detail")
        if isinstance(detail, dict) and str(detail.get("message") or "").strip():
            return str(detail["message"]).strip()
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    except Exception:  # noqa: BLE001 - 本文が JSON でなくても HTTP 状態で返す
        pass
    return f"HTTP {resp.status_code}"


class _ApiFailure(Exception):
    """API 呼び出しの失敗。``code`` は終了コード。"""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def _api_get(client: Any, url: str) -> dict[str, Any]:
    import httpx

    try:
        resp = client.get(f"{url}/api/system/auto-tune")
    except httpx.ConnectError as e:
        from backend.i18n_helper import msg

        raise _ApiFailure(msg("cli.backend_not_running"), EXIT_CANNOT_RUN) from e
    except httpx.HTTPError as e:
        from backend.i18n_helper import msg

        raise _ApiFailure(msg("cli.tune_failed", detail=str(e)), EXIT_PROPOSAL) from e
    if resp.status_code != 200:
        from backend.i18n_helper import msg

        raise _ApiFailure(msg("cli.tune_failed", detail=_error_detail(resp)), EXIT_PROPOSAL)
    return resp.json()


def _run_via_api(
    args: argparse.Namespace, console: Console, only: list[str] | None,
) -> tuple[int, dict[str, Any] | None]:
    """稼働中の backend に実行を頼み、done / failed まで進捗を 1 行ずつ出して待つ。"""
    import httpx

    from backend.free.cli.renderer import render_error, render_info
    from backend.i18n_helper import msg

    url = args.backend_url.rstrip("/")
    render_info(console, msg("cli.tune_via_api", url=url))
    try:
        with _client() as client:
            before = _api_get(client, url)
            force = bool(args.force) or before.get("decision") in ("pending", "declined")
            try:
                resp = client.post(f"{url}/api/system/auto-tune/run", json={"only": only, "force": force})
            except httpx.HTTPError as e:
                raise _ApiFailure(msg("cli.tune_failed", detail=str(e)), EXIT_PROPOSAL) from e
            if resp.status_code == 409:
                raise _ApiFailure(msg("cli.tune_already_running"), EXIT_PROPOSAL)
            if resp.status_code not in (200, 202):
                raise _ApiFailure(msg("cli.tune_failed", detail=_error_detail(resp)), EXIT_PROPOSAL)
            seen: tuple[Any, ...] | None = None
            deadline = time.monotonic() + _POLL_MAX_SEC
            while True:
                view = _api_get(client, url)
                progress = view.get("progress") or None
                if progress and (key := (progress.get("phase"), progress.get("current"), progress.get("total"))) != seen:
                    seen = key
                    render_info(console, msg(
                        "cli.tune_progress", phase=progress.get("phase"),
                        current=progress.get("current"), total=progress.get("total"),
                    ))
                if view.get("state") == "failed":
                    render_error(console, msg("cli.tune_run_failed", detail=view.get("error") or "-"))
                    return EXIT_PROPOSAL, view
                if view.get("state") == "done":
                    return EXIT_OK, view
                if time.monotonic() > deadline:
                    raise _ApiFailure(msg("cli.tune_failed", detail="timeout"), EXIT_PROPOSAL)
                time.sleep(_POLL_SEC)
    except _ApiFailure as e:
        render_error(console, str(e))
        return e.code, None


# ── 停止中の直接実行 ───────────────────────────────────────


def _make_progress(console: Console) -> Callable[[str, int, int, str], None]:
    from backend.free.cli.renderer import render_info
    from backend.i18n_helper import msg

    def progress(phase: str, current: int, total: int, key: str) -> None:
        if phase == "item":
            render_info(console, msg("cli.tune_progress", phase=key, current=current + 1, total=total))

    return progress


def _run_direct(
    args: argparse.Namespace, cfg: dict[str, Any], project_root: Path, console: Console,
    only: list[str] | None,
) -> tuple[int, dict[str, Any] | None, dict[str, Any]]:
    """停止中の実行。戻りは (終了コード, API と同じ形の view, 追加の結果)。"""
    from backend.free.cli.renderer import render_error, render_info
    from backend.free.core.tuning import runner as tune_runner
    from backend.free.core.tuning.gate import load_gate, should_measure
    from backend.free.core.tuning.items import load_builtin_tuners
    from backend.free.core.tuning.store import load_auto_tune, resolve_tune_paths
    from backend.i18n_helper import msg

    try:
        load_builtin_tuners().ordered(only)
    except ValueError as e:
        render_error(console, msg("cli.tune_unknown_item", detail=str(e)))
        return EXIT_CANNOT_RUN, None, {}
    paths = resolve_tune_paths(cfg, project_root)
    record = load_auto_tune(paths.auto_tune)[0]
    if only is not None and record is not None:
        # 稼働中の画面の実行が予約した再計算も、停止中のこの実行で果たす
        registry = load_builtin_tuners()
        only = [*only, *(k for k in record.recompute if k not in only and registry.get(k) is not None)]
    # 明示の ``evoref tune run`` は利用者の承認そのもの。確認待ち (pending / declined) の PC でも測る。
    gate = load_gate(cfg, project_root, _current_pc(), paths=paths)
    force = bool(args.force) or not should_measure(gate)[0]
    result = tune_runner.run(cfg, project_root, only=only, force=force, progress=_make_progress(console))
    extra = {
        "measured": result.measured, "reason": result.reason,
        "failed": dict(result.failed), "unsaved": list(result.unsaved),
    }
    view = _local_view(cfg, project_root)
    if not result.measured:
        render_error(console, msg("cli.tune_blocked", decision=result.gate.decision))
        return EXIT_PROPOSAL, view, extra
    for key, reason in result.failed.items():
        render_error(console, msg("cli.tune_item_failed", item=key, reason=reason))
    render_info(console, msg("cli.tune_run_done"))
    if any(i.requires_restart and i.applied for i in result.items.values()):
        render_info(console, msg("cli.tune_restart_required"))
    return (EXIT_PROPOSAL if result.failed or result.unsaved else EXIT_OK), view, extra


# ── サブコマンド ───────────────────────────────────────────


def _writer_running(data_root: Path) -> bool:
    from backend.data_root import store_root
    from backend.io.writer_lock import lock_held

    return lock_held(store_root(data_root))


def run_tune(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref tune``)。"""
    from backend.data_root import DataRootError, export_data_root, resolve_data_root
    from backend.free.cli.config_loader import _find_project_root, _setup_encoding
    from backend.free.cli.renderer import render_error
    from backend.i18n_helper import init_i18n, msg
    from backend.log_config import setup_cli_logging

    if not _setup_encoding():
        return EXIT_CANNOT_RUN
    init_i18n()
    args = _build_parser().parse_args(argv)
    project_root = _find_project_root()
    setup_cli_logging(project_root=project_root, debug=False)
    console = _make_console(args.json)
    try:
        data_root = resolve_data_root(args.data_root, root=project_root)
    except DataRootError as e:
        render_error(console, msg("cli.data_root_invalid", detail=str(e)))
        return EXIT_CANNOT_RUN
    export_data_root(data_root)
    if args.startup_check:
        return run_startup_check(project_root, console, yes=args.yes, data_root=data_root)
    cfg = _load_cfg(project_root)
    if cfg is None:
        render_error(console, msg("cli.tune_failed", detail="config.yaml"))
        return EXIT_CANNOT_RUN
    try:
        if args.action == "run":
            return _cmd_run(args, cfg, project_root, data_root, console)
        return _cmd_status(args, cfg, project_root, data_root, console)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001 - 想定外の失敗は 2 で返す
        logger.warning("tune failed: %s", e, exc_info=True)
        render_error(console, msg("cli.tune_failed", detail=f"{type(e).__name__}: {e}"))
        return EXIT_CANNOT_RUN


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _cmd_status(
    args: argparse.Namespace, cfg: dict[str, Any], project_root: Path, data_root: Path, console: Console,
) -> int:
    from backend.free.cli.renderer import render_error

    if _writer_running(data_root):
        try:
            with _client() as client:
                view = _api_get(client, args.backend_url.rstrip("/"))
        except _ApiFailure as e:
            render_error(console, str(e))
            return e.code
    else:
        view = _local_view(cfg, project_root)
    if args.json:
        _emit_json(view)
    else:
        _render_view(console, view, cfg)
    return EXIT_PROPOSAL if _has_proposal(view, cfg) else EXIT_OK


def _cmd_run(
    args: argparse.Namespace, cfg: dict[str, Any], project_root: Path, data_root: Path, console: Console,
) -> int:
    only = [k.strip() for k in args.only.split(",") if k.strip()] if args.only else None
    extra: dict[str, Any] = {}
    if _writer_running(data_root):
        code, view = _run_via_api(args, console, only)
    else:
        code, view, extra = _run_direct(args, cfg, project_root, console, only)
    if view is not None:
        if args.json:
            _emit_json({**view, **extra})
        else:
            _render_view(console, view, cfg)
            if view.get("restart_required"):
                from backend.free.cli.renderer import render_info
                from backend.i18n_helper import msg

                render_info(console, msg("cli.tune_restart_required"))
    return code


# ── 環境移行の確認 (起動前) ────────────────────────────────


def _ask_migration(console: Console, changed_axes: tuple[str, ...]) -> str | None:
    """確認の質問 (``accepted`` / ``declined`` / ``unchanged``)。EOF・読めない入力が続けば ``None`` (pending のまま)。"""
    from backend.i18n_helper import msg

    offer_unchanged = changed_axes == ("hostname",)
    axes = ", ".join(msg(f"cli.tune_axis.{a}") for a in changed_axes) or "-"
    choices = msg("cli.tune_choices" if offer_unchanged else "cli.tune_choices_no_unchanged")
    prompt = escape(f"{msg('cli.tune_prompt', axes=axes)} {choices} ")
    for _ in range(_ASK_ATTEMPTS):
        try:
            raw = console.input(prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        answer = raw.strip().lower()
        if answer in ("", "y", "yes"):
            return "accepted"
        if answer in ("n", "no"):
            return "declined"
        if offer_unchanged and answer in ("変更なし", "unchanged", "u"):
            return "unchanged"
    return None


def run_startup_check(
    project_root: Path, console: Console | None = None, *, yes: bool = False, data_root: Path | None = None,
) -> int:
    """環境移行の確認と、起動前の自動調整。**どんな失敗でも起動を止めない** (常に 0)。

    - ``fresh`` (保存結果が無い新規) / 確認済み (``accepted``) の移行: 見積りだけで決まる項目
      (``needs_servers`` が偽: ctx / ngl / VRAM 予算 …) を確認なしで実行して保存する (非対話でも)
    - ``migrated`` かつ ``pending``: 端末なら 1 回聞き (``--yes`` なら聞かず承認)、承認したら上と同じ
      項目を実行する。端末でなければ ``--yes`` が無い限り何も聞かず ``pending`` のまま返る (GUI の
      バナーで確認する。``auto`` の項目は保守側の値で起動する)
    """
    try:
        _startup_check(project_root, console or _make_console(False), yes=yes, data_root=data_root)
    except Exception as e:  # noqa: BLE001 - 起動を止めない
        logger.warning("tune startup check failed: %s", e, exc_info=True)
    return EXIT_OK


def _startup_check(project_root: Path, console: Console, *, yes: bool, data_root: Path | None) -> None:
    from backend.data_root import resolve_data_root
    from backend.free.cli.renderer import render_error, render_info
    from backend.free.core.tuning.gate import answer_migration, load_gate, should_measure
    from backend.free.core.tuning.store import load_auto_tune, resolve_tune_paths
    from backend.i18n_helper import msg

    cfg = _load_cfg(project_root)
    if cfg is None:
        return
    if _writer_running(data_root or resolve_data_root(root=project_root)):
        return
    paths = resolve_tune_paths(cfg, project_root)
    pc = _current_pc()
    gate = load_gate(cfg, project_root, pc, paths=paths)
    record = load_auto_tune(paths.auto_tune)[0]
    scheduled = bool(record is not None and record.recompute) and should_measure(gate)[0]
    if gate.state == "fresh" or (gate.state == "migrated" and gate.decision == "accepted") or scheduled:
        # 新規 (移行ではない) / 確認済みの移行 / 稼働中の画面の実行が予約した再計算: 見積りだけで
        # 決まる項目を確認なしで決めて保存する (base が載る前なので空きが正確)
        _run_estimated_items(cfg, project_root, console)
        return
    if gate.state != "migrated" or gate.decision != "pending":
        return
    if yes:
        answer: str | None = "accepted"
    elif _is_interactive():
        answer = _ask_migration(console, gate.changed_axes)
    else:
        answer = None
    if answer is None:
        render_info(console, msg("cli.tune_startup_pending"))
        return
    result = answer_migration(paths, answer, pc, cfg=cfg)  # type: ignore[arg-type]
    if answer == "unchanged":
        if result.ok:
            render_info(console, msg("cli.tune_restamped"))
        else:
            render_error(console, msg("cli.tune_restamp_refused", detail=result.reason), level="warning")
    elif not result.ok:
        render_error(console, msg("cli.tune_failed", detail=result.reason), level="warning")
    elif answer == "declined":
        render_info(console, msg("cli.tune_declined_note"))
    else:
        render_info(console, msg("cli.tune_accepted_running"))
        _run_estimated_items(cfg, project_root, console)


def _base_running(cfg: dict[str, Any]) -> bool:
    """base の llama-server が応答しているか (起動スクリプトの ``_base_ready``。読めなければ偽)。"""
    try:
        from scripts.launch_llama import _base_ready
    except ImportError:
        return False
    return bool(_base_ready(cfg))


def _skip_estimation_reason(cfg: dict[str, Any], hw: Any) -> str | None:
    """起動前の自動の見積りを今は保存しない理由 (英語)。見積ってよければ ``None``。"""
    from backend.free.core.tuning.hardware import memory_squeezed

    if _base_running(cfg):
        return "the base llama-server is already running"
    squeezed = memory_squeezed(hw)
    if squeezed is not None:
        return f"free memory looks held by something else: {squeezed}"
    return None


def _run_estimated_items(cfg: dict[str, Any], project_root: Path, console: Console) -> None:
    """llama-server を起こさずに決まる項目 (``needs_servers`` が偽) を実行して保存する。

    埋め込みの配置とリランカー (サーバを起こして測る項目) は、この後に起動スクリプトが従来どおり
    測る (確認状態は ``accepted`` / ``fresh`` なので測る側に倒れる)。ここで起こすと同じ判別を 2 回する。

    空きで見積もる項目なので、base が既に載っている回と、空きが総量に比べて異常に小さい回
    (別のアプリや古い llama-server が一時的に握っている、:func:`~backend.free.core.tuning.hardware.memory_squeezed`)
    は保存せず警告だけ出す。その回の起動は保存値 / 保守側の値で続く (小さい空きを恒久化しない)。
    """
    from backend.free.cli.renderer import render_error
    from backend.free.core.tuning import runner as tune_runner
    from backend.free.core.tuning.hardware import probe_hardware
    from backend.free.core.tuning.items import load_builtin_tuners
    from backend.i18n_helper import msg

    skip = _skip_estimation_reason(cfg, probe_hardware())
    if skip is not None:
        logger.warning("tune startup check: not estimating now (%s); start continues with saved / safe values", skip)
        render_error(console, msg("cli.tune_failed", detail=skip), level="warning")
        return
    only = [s.key for s in load_builtin_tuners().ordered() if not s.needs_servers]
    outcome = tune_runner.run(cfg, project_root, only=only, progress=_make_progress(console))
    for key, reason in outcome.failed.items():
        render_error(console, msg("cli.tune_item_failed", item=key, reason=reason), level="warning")


if __name__ == "__main__":
    sys.exit(run_tune(sys.argv[1:]))
