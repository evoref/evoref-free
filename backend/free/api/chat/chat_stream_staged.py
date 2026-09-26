"""run_staged_pipeline の置き場 (StagedCodeHarness が消費)

staged クリエイト (仕様書 → コード → テスト) の「合成 → LoopDriver → finalize 検査」
を構造化イベントで駆動する共有実体 (:func:`run_staged_pipeline`)。配信 (SSE 化 /
ディスク書込) は持たない — 消費者は ``create.dispatch=meta`` の
:class:`backend.free.loop.staged.harness.StagedCodeHarness` の 1 本 (3a-2、f_10 §1)。
旧 legacy dispatch (``stream_staged_create`` が SSE を直接組み立てていた経路) は
撤去済み。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path

from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    TYPE_CHECKING,
)
from backend.app_state import AppState
from backend.free.api.chat.chat_constants import DEFAULT_KEEPALIVE_INTERVAL_SEC
from backend.free.generation.key_coherence import find_unmatched_dict_keys
from backend.trace_context import get_trace_id
from backend.utils import estimate_tokens as _estimate_tokens

from backend.free.api.chat.chat_stream_common import (
    cancel_requested,
    logger,
)

if TYPE_CHECKING:
    from backend.free.loop.staged.run_record import RunEventLog, RunRecordStore


def _supports_kwarg(target: Callable, name: str) -> bool:
    """``target`` の呼出しが keyword ``name`` を受け取れるか (``**kwargs`` も可)。

    ``StagedCreateExecutor``/``synthesize_create_task_graph`` へ渡す新設
    キーワード (``deadline_monotonic``/``timeout``) は別エージェントが同時に
    足すため、未着地の間に落ちないよう存在確認してから渡す
    (f_10 §3 予算の 3 層)。
    """
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


#: staged create の run 1 リクエストの壁時計上限の出荷既定
#: (``create.turn_timeout_sec``、f_10 §3 予算の 3 層 — ターン予算)。
_TURN_TIMEOUT_DEFAULT_SEC = 3600.0

#: 計画と spec 本文へ渡す参照設計書の上限 (文字)。見出し単位で均等に縮める
#: (f_10 §2)。6000 字 ≒ 呼出 2 回ぶんの prefill 増で、モジュールごとには乗らない。
_REFERENCE_DOC_MAX_CHARS = 6000

#: ステージ予算 (create.staged.total_timeout_sec) をターン予算から確保する下限。
_STAGE_BUDGET_FLOOR_SEC = 300.0

#: 呼出予算 (task グラフ合成 1 回) の既定上限。
_GRAPH_SYNTHESIS_TIMEOUT_DEFAULT_SEC = 120.0

#: 参照設計書 1 文字あたりに足す合成の呼出予算 (秒)。参照なしでも合成は実測
#: 51〜100 秒で 120 秒に張り付いており、4000 字の設計書を足した 2 回とも 120 秒で
#: 打ち切られて longform へ落ちた (2026-09-22 実機)。落ちると計画からやり直しで
#: 数十分を失うので、prefill の増分 (≒ 0.8 tok/字) に余裕を持たせて延ばす。
_REFERENCE_DOC_SYNTHESIS_SEC_PER_CHAR = 0.05


def _emit_check(
    event_log: "RunEventLog | None", kind: str, payload: dict,
) -> tuple[str, dict]:
    """finalize 検査用の内部 emit: ``events.jsonl`` へ永続化し ``(kind, payload)`` を返す。

    SSE 化はしない — :func:`_finalize_staged_checks_events` が使う。
    """
    if event_log is not None:
        try:
            event_log.append(kind, payload)
        except Exception as exc:  # noqa: BLE001 - イベント記録の失敗で検査を止めない
            logger.warning("staged run event log append failed (kind=%s): %s", kind, exc)
    return kind, payload


def _finish_run_safely(
    run_store: "RunRecordStore | None",
    event_log: "RunEventLog | None",
    exit_kind: str,
    *,
    tasks_failed: int = 0,
) -> None:
    """run.json の終端書込み (失敗しても配信を止めない)。

    ``tasks_failed`` は流れた上で欠けたものの件数 (f_10 §7)。``exit_kind=done``
    のままでも 1 以上なら読み出し時に ``incomplete`` と導出される。
    """
    if run_store is None:
        return
    try:
        last_seq = event_log.last_seq if event_log is not None else -1
        run_store.finish(
            exit_kind, last_event_seq=max(0, last_seq), tasks_failed=tasks_failed,
        )
    except Exception as exc:  # noqa: BLE001 - 後始末の失敗で応答を壊さない
        logger.warning("staged run record finish(%s) failed: %s", exit_kind, exc)


# ---------------------------------------------------------------------------
# Staged クリエイト (仕様書→コード→テスト) ストリーミング
# ---------------------------------------------------------------------------

_STAGE_LABELS = {"spec": "仕様書", "code": "コーディング", "test": "テスト"}


def _stage_label_for_task(task_id: str) -> str:
    if task_id.startswith("spec"):
        return _STAGE_LABELS["spec"]
    if task_id.startswith("code_"):
        return _STAGE_LABELS["code"]
    if task_id.startswith("test_"):
        return _STAGE_LABELS["test"]
    return "タスク"


def _translate_loop_event_payload(
    evt, total_tasks: int = 0, task_indices: dict[str, int] | None = None,
) -> dict | None:
    """LoopEvent を staged 進捗の step payload (dict) へ翻訳する (該当なしは None)。

    :func:`run_staged_pipeline` (構造化イベント版、meta dispatch) が使う (f_10 §1)。

    2 段階表示:
    - 上位 (工程タスク): ``task_picked`` → ``long_form_unit_start`` を
      ``[i/N] {工程}: {title}`` 形式で出し、フロントの ``parseLongFormProgress``
      が進捗バー化する。``iteration_ended`` → ``long_form_unit_done``。
    - 下位 (工程内サブステップ): ``stage_progress`` → ``task_progress`` step
      (フロントは折りたたみリスト、CLI は逐次表示)。

    ``task_indices`` (呼出側所有の可変 dict) を渡すと、ユニット番号を driver の
    iteration ではなく task_id の初出順で採番する。driver リトライで同一タスクが
    再 pick された場合は同じ番号を再利用し「(再試行)」を付ける (旧実装は
    iteration をそのまま使い ``[4/3]`` のように総数を超えて表示されていた)。
    """
    data = getattr(evt, "data", None) or {}
    tid = str(data.get("task_id", ""))
    label = _stage_label_for_task(tid)

    def _unit_index() -> tuple[int, bool]:
        """(表示番号, 再試行か)。task_indices 未指定時は従来の iteration。"""
        if task_indices is None or not tid:
            return getattr(evt, "iteration", 0) or 0, False
        if tid in task_indices:
            return task_indices[tid], True
        task_indices[tid] = len(task_indices) + 1
        return task_indices[tid], False

    if evt.event == "task_picked":
        title = str(data.get("title", ""))
        idx, is_retry = _unit_index()
        prefix = f"[{idx}/{total_tasks}] " if total_tasks else ""
        suffix = " (再試行)" if is_retry else ""
        return {
            "type": "long_form_unit_start",
            "detail": f"{prefix}{label}: {title}{suffix}".strip(),
            "status": "running",
        }
    if evt.event == "iteration_ended":
        outcome = data.get("last_outcome") or {}
        status = str(outcome.get("status", ""))
        ok = status == "success"
        if task_indices is not None and tid in task_indices:
            idx = task_indices[tid]
        else:
            idx = getattr(evt, "iteration", 0) or 0
        prefix = f"[{idx}/{total_tasks}] " if total_tasks else ""
        return {
            "type": "long_form_unit_done",
            "detail": f"{prefix}{label}: {'完了' if ok else (status or '終了')}",
            "status": "done" if ok else "failed",
        }
    if evt.event == "stage_progress":
        detail = str(data.get("detail", "")).strip()
        status = str(data.get("status", "running"))
        if not detail:
            return None
        return {
            "type": "task_progress",
            "detail": detail,
            "status": status,
        }
    if evt.event == "gate_result":
        ok = bool(data.get("ok"))
        # ゲートは工程タスク単位で走るため、同一工程に複数ユニットがあると
        # label だけでは全く同じ行が並ぶ (実測 2026-08-07 ライブ監査: test 工程が
        # 2 ユニットで「テスト: 起動可能性チェック合格 ...」が 2 行、どちらが
        # どのユニットか判別できなかった)。他イベントと同じ [i/N] を付ける。
        if task_indices is not None and tid in task_indices:
            idx = task_indices[tid]
        else:
            idx = getattr(evt, "iteration", 0) or 0
        prefix = f"[{idx}/{total_tasks}] " if total_tasks else ""
        # import スモークゲートは「import 成功＋エントリ静的整合＋OS 互換」までを
        # 静的に検証するもので、プログラムを実行したわけではない。「pass」と書くと
        # 実行検証済みと誤解されるため、起動可能性チェック (静的) と明示する。
        detail = (
            f"{prefix}{label}: 起動可能性チェック合格 "
            "(import/エントリ/整合・静的検証/未実行)"
            if ok else
            f"{prefix}{label}: 起動可能性チェック失敗 (起動不能の可能性)"
        )
        return {
            "type": "task_result",
            "detail": detail,
            "status": "done" if ok else "failed",
        }
    return None


# staged の task グラフは **リクエスト毎の隔離ストア** (workspace 内 .semmem) に持つ。
# 共有 project ストア (state.current_project_id) を使うと ①継続ターンで stale な
# done ファクトが新ターンの spec→code→test 依存ゲートを壊す ②永続ストアの ``task``
# ファクトは create モードのプロンプトへ注入されるので、staged のタスクが次の会話へ
# 漏れる、という不具合になるため、永続プロジェクトストアからは完全に切り離す。
_STAGED_PROJECT_ID = "staged"

#: staged クリエイト 1 リクエストの総時間上限の出荷既定
#: (:class:`backend.schemas.create.CreateStagedConfig` と一致させる)。
#: 打ち切りメッセージで「設定値が既定より低い」ことを示すために参照する。
_STAGED_TOTAL_TIMEOUT_DEFAULT_SEC = 2400.0


def _requested_test_files(ws, requested: list[str]) -> dict[str, str]:
    """依頼されたテストファイルの名前で、test 工程が生成・実行したテストを返す (f_10 §5)。

    計画がテストファイルをコードのモジュールとして組み込み、一度も実行されない
    テストを配信していた (2026-09-22 実機 K06)。同名の検証済みテストがあれば
    それを、無くても検証済みテストが 1 本だけならそれをその名前で配信する。
    """
    if not requested:
        return {}
    tests = {Path(wf.logical_path).name: wf.logical_path for wf in ws.list_files(kind="test")}
    out: dict[str, str] = {}
    for path in requested:
        logical = tests.get(Path(path).name)
        if logical is None and len(tests) == 1:
            logical = next(iter(tests.values()))
        content = ws.read_file(logical, kind="test") if logical else None
        if content:
            out[path] = content
        else:
            logger.info("staged create: no verified test to deliver as %s", path)
    return out


def _staged_remaining_units(ws) -> int:
    """manifest の progress から未完了タスクユニット数を出す (timed_out / cancelled 共有)。"""
    progress = ws.read_manifest().get("progress") or {}
    return max(
        0,
        int(progress.get("tasks_total") or 0)
        - int(progress.get("tasks_done") or 0)
        - int(progress.get("tasks_failed") or 0),
    )


def _staged_postprocess(
    code_map: dict[str, str],
) -> tuple[dict[str, str], list[str], list[str]]:
    """配信前に cross-file import を決定論的に配線し、静的整合 issue を集める。

    test 工程は wall-time で starve され得る (= 工程内スモークゲートが走らない) ため、
    予算非依存のこの終端で必ず検証する。

    - ``wire_imports`` / ``normalize_relative_imports`` は加算的 (不足 import を足し、
      flat 構成で解決不能な相対 import を除くだけ) で機能を削らない = ソースを劣化させない。
    - ``check_coherence`` は重複定義 / どのモジュールにも無い未定義名を検出する (advisory)。

    返り値は (配線済み code_map, issue リスト, 配線で変更したファイル一覧)。issue は
    配信を止めない (advisory)。配線変更一覧は long_form JSONL への可測化に使う。
    """
    from backend.free.generation.import_wirer import wire_imports
    from backend.free.generation.smoke_validator import (
        check_coherence,
        check_cross_module_imports,
        check_entrypoint,
        check_main_invoked,
        normalize_relative_imports,
    )

    out = dict(code_map)
    wired_paths: list[str] = []
    py_map = {p: c for p, c in out.items() if p.endswith(".py")}
    if len(py_map) > 1:
        try:
            wired = wire_imports(normalize_relative_imports(py_map))
            for p, c in wired.items():
                if c and c != out.get(p):
                    out[p] = c
                    wired_paths.append(p)
        except Exception as exc:
            logger.warning("staged finalize wire_imports failed: %s", exc)
    final_py = {p: c for p, c in out.items() if p.endswith(".py")}
    issues: list[str] = []
    # 重複定義/未定義名 (coherence) + 起動経路の未定義メソッド参照 (entrypoint) +
    # 生成物間 from-import の名前欠落 (cross_module_imports) を終端でも必ず検査する
    # (工程内スモークが starve された / 外部依存欠落で import スモークが盲目化した場合の保険)。
    for fn in (check_coherence, check_entrypoint, check_main_invoked, check_cross_module_imports):
        try:
            issues += list(fn(final_py))
        except Exception as exc:
            logger.warning("staged finalize %s failed: %s", fn.__name__, exc)
    return out, issues, sorted(wired_paths)


def _staged_internal_names(ws) -> frozenset[str]:
    """spec が宣言する内部契約名 (幻覚内部 import 判定用、読めなければ空)。

    smoke の「外部依存」分類に渡し、spec の Component / 正準モジュールに由来する
    import 失敗を環境要因 warning へ降格させない (2026-07-07 live: `from game
    import Game` が外部依存扱いになり起動不能コードが偽 success で配信された)。
    """
    try:
        from backend.free.loop.staged.spec_parts import internal_contract_names
        return internal_contract_names(ws.read_spec() or "")
    except Exception as exc:
        logger.debug("staged internal names unavailable: %s", exc)
        return frozenset()


async def _staged_import_smoke(
    code_map: dict[str, str], timeout_sec: float,
    internal_names: frozenset[str] = frozenset(),
) -> list[str]:
    """配信前の code_map を import スモークし error 文字列列を返す (失敗時は空)。

    静的検査 (check_coherence / check_entrypoint) では拾えない cross-file ImportError
    (``from game import GameConfig`` で GameConfig が実在しない等) を終端でも捕捉する。
    ``__main__`` は実行せず、外部依存 (pygame 等) の未インストールは warning に倒れる
    ため error には含まれない (内部契約名 ``internal_names`` に由来する幻覚 import
    は error 側に分類される)。
    """
    py_map = {p: c for p, c in code_map.items() if p.endswith(".py")}
    if not py_map:
        return []
    from backend.free.generation.smoke_validator import run_import_smoke
    try:
        res = await asyncio.to_thread(
            run_import_smoke, py_map, timeout_sec,
            internal_names=internal_names,
        )
    except Exception as exc:
        logger.warning("staged finalize import smoke failed: %s", exc)
        return []
    return [str(e) for e in (getattr(res, "errors", None) or [])]


async def run_staged_pipeline(
    *,
    query: str,
    session_id: str,
    state: AppState,
    cfg: dict,
    output_target: str,
    codegen,
    part_codegen=None,
    prefetched_rag: list[tuple[str, float, str]] | None = None,
    prefetched_rag_top_score: float | None = None,  # noqa: ARG001 - metrics 側は呼出元が使う
    file_context_block: str | None = None,
    brief: str = "",
    keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL_SEC,
    total_timeout_sec: float | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    resume_of: str | None = None,
) -> AsyncIterator[dict]:
    """staged クリエイトの「合成 → LoopDriver → finalize 検査」を構造化イベントで駆動する。

    ``create.dispatch=meta`` (:class:`backend.free.loop.staged.harness.StagedCodeHarness`)
    が消費する共有実体 (f_10 §1)。**配信は行わない** — ``kind="step"`` (進捗) /
    ``kind="keepalive"`` / 終端 ``kind="result"`` の dict を yield する。

    finalize 検査ロジック (postprocess / smoke / coherence / design_drift) は
    :func:`_finalize_staged_checks_events` が持つ。

    ``total_timeout_sec`` 未指定時は既定計算
    (``create.staged.total_timeout_sec`` を ``create.turn_timeout_sec - 300`` に
    クランプ) を使う。``is_cancelled`` 未指定時は ``cancel_requested(session_id)``。

    タスクグラフ合成が空のときは ``exit_kind="error"`` + ``notes["fallback"] =
    "empty_task_graph"`` の終端イベントのみ yield して返す — longform への
    委譲は composition 層 (``chat.py::make_production_stage``) の責務。

    ``resume_of`` (Phase 3b、f_10 §7) は問い返し (needs_input) から再開する
    元 run_id。``run_store.start()`` の ``_extra`` へそのまま渡す。
    """
    from backend.io.id_registry import new_id

    from backend.config import get_path_resolver
    from backend.free.loop.artifact_writer import make_loop_artifact_hook
    from backend.free.loop.driver import LoopDriver, decode_task_fact
    from backend.free.loop.events import LoopEventBus
    from backend.free.loop.staged import (
        RunEventLog,
        RunRecordStore,
        WorkspaceManager,
        synthesize_create_task_graph_with_plan,
    )
    from backend.free.loop.staged.executor import StagedCreateExecutor
    from backend.free.loop.staged.test_runner import StagedTestRunner
    from backend.free.generation.api_contract import check_api_contract
    from backend.free.generation.smoke_validator import (
        check_coherence,
        check_cross_module_imports,
        check_entrypoint,
        check_main_invoked,
        run_entry_smoke,
        run_import_smoke,
    )
    from backend.free.memory.semantic.store import SemanticStore
    from backend.free.memory.views.loop import LoopFactView

    t_start = time.monotonic()
    create_cfg = cfg.get("create", {}) or {}
    staged_cfg = (create_cfg.get("staged", {}) or {})
    _is_cancelled = is_cancelled or (lambda: cancel_requested(session_id))

    if total_timeout_sec is None:
        # 予算の 3 層 (f_10 §3): legacy と同じ既定計算。
        turn_timeout_sec = float(create_cfg.get("turn_timeout_sec", _TURN_TIMEOUT_DEFAULT_SEC))
        staged_total_timeout_sec = float(
            staged_cfg.get("total_timeout_sec", _STAGED_TOTAL_TIMEOUT_DEFAULT_SEC),
        )
        _stage_budget_cap = max(_STAGE_BUDGET_FLOOR_SEC, turn_timeout_sec - 300.0)
        if staged_total_timeout_sec > _stage_budget_cap:
            logger.warning(
                "staged create: create.staged.total_timeout_sec=%.0fs exceeds "
                "turn budget cap %.0fs (create.turn_timeout_sec=%.0fs - 300s); "
                "clamping (f_10 §3)",
                staged_total_timeout_sec, _stage_budget_cap, turn_timeout_sec,
            )
            staged_total_timeout_sec = _stage_budget_cap
    else:
        staged_total_timeout_sec = float(total_timeout_sec)
    deadline_monotonic = t_start + staged_total_timeout_sec

    run_id = new_id("run_")
    workspace_root = get_path_resolver().resolve_local("create_workspace_dir")
    ws = WorkspaceManager.open_or_create(
        workspace_root, workspace_id=run_id, session_id=session_id,
        project_id=_STAGED_PROJECT_ID, goal=query, debug_logger=state.debug_logger,
    )
    _staged_semmem = SemanticStore(ws.root / ".semmem")
    _staged_semmem.load()
    staged_store = _staged_semmem.scoped(f"project:{_STAGED_PROJECT_ID}")

    def _staged_view(_pid: str) -> LoopFactView:
        return LoopFactView(stores=[staged_store], writeback_store=staged_store)

    def _empty_result() -> dict:
        # run_id/workspace_root は composition 層 (chat.py::_ProductionStageSelector)
        # が問い返し (needs_input) を判定・永続化するために notes へ載せる (f_03 §4.4)。
        # 実際の run.json はまだ書かれていない (facts が空なので下の run_store.start
        # に到達しない) — 問い返しに倒すときだけ選択側が run.json を起こす。
        return {
            "exit_kind": "error",
            "notes": {
                "fallback": "empty_task_graph",
                "run_id": run_id, "workspace_root": str(ws.root),
            },
            "code_map": {}, "spec_md": None, "flowchart_md": None,
            "tasks_failed": 0, "runnability_issues": [],
            "design_drift_counts": {"convergence": 0, "divergence": 0, "absence": 0},
            "metrics": {}, "truncated_steps": [], "truncated_max_tokens": None,
            "ws": ws, "event_log": None, "run_store": None, "total_tasks": 0,
        }

    yield {"kind": "step", "payload": {
        "type": "long_form_plan",
        "detail": "タスクグラフ (仕様書/コード/テスト) を合成中…",
        "status": "running",
    }}
    # 依頼文が名指しした設計書の本文 (f_10 §2)。計画と spec 本文にだけ渡す。
    reference_doc = ""
    try:
        from backend.free.api.chat.chat_stream_output import read_reference_design_doc
        from backend.free.generation.strategy_common import condense_design_doc

        reference_doc = condense_design_doc(
            await read_reference_design_doc(query, state), _REFERENCE_DOC_MAX_CHARS,
        )
    except Exception as exc:  # noqa: BLE001 - 読めなければ参照なしで続ける
        logger.warning("staged create: reference design document unavailable: %s", exc)
    if reference_doc:
        logger.info(
            "staged create: passing a reference design document (%d chars) "
            "to planning and spec", len(reference_doc),
        )
    _synth_kwargs: dict[str, Any] = dict(
        request=query, project_id=_STAGED_PROJECT_ID,
        aux_client=state.aux_client,
        include_tests=(
            bool(staged_cfg.get("test_stage_enabled", True))
            or bool(staged_cfg.get("smoke_gate_enabled", True))
        ),
        debug_logger=state.debug_logger,
        brief=brief,
    )
    if reference_doc and _supports_kwarg(synthesize_create_task_graph_with_plan, "reference_doc"):
        _synth_kwargs["reference_doc"] = reference_doc
    # planner が挙げたテストファイルは test 工程の検証済みテストで配信する (f_10 §5)。
    requested_tests: list[str] = []
    if _supports_kwarg(synthesize_create_task_graph_with_plan, "requested_tests_out"):
        _synth_kwargs["requested_tests_out"] = requested_tests
    if _supports_kwarg(synthesize_create_task_graph_with_plan, "timeout"):
        _remaining = max(0.0, deadline_monotonic - time.monotonic())
        _synth_kwargs["timeout"] = min(
            _GRAPH_SYNTHESIS_TIMEOUT_DEFAULT_SEC
            + len(reference_doc) * _REFERENCE_DOC_SYNTHESIS_SEC_PER_CHAR,
            _remaining,
        )
    facts, module_deps = await synthesize_create_task_graph_with_plan(**_synth_kwargs)
    if not facts:
        logger.info("staged create: empty task graph (pipeline)")
        try:
            _staged_semmem.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("staged create: semmem close failed: %s", exc)
        yield {"kind": "result", "payload": _empty_result()}
        return

    if module_deps:
        try:
            ws.write_plan(module_deps)
        except Exception as exc:  # noqa: BLE001 - plan 保存の失敗で staged 実行は止めない
            logger.warning("staged create: write_plan failed: %s", exc)

    run_store: "RunRecordStore | None" = None
    event_log: "RunEventLog | None" = None
    try:
        run_store = RunRecordStore(ws.root)
        event_log = RunEventLog(ws.root, debug_logger=state.debug_logger)
        run_store.start(
            run_id=run_id, session_id=session_id, request_id=get_trace_id(),
            mode="create", query=query, output_target=output_target,
            brief_tokens=_estimate_tokens(brief) if brief else 0,
            resume_of=resume_of,
        )
    except Exception as exc:  # noqa: BLE001 - run レコードの失敗で staged 実行は止めない
        logger.warning("staged run record start failed: %s", exc)
        run_store = None
        event_log = None
    # run_id の通知 (f_10 §7 / f_05 §4.5): StagedCodeHarness → meta_cognitive の
    # on_event → create_run SSE フレームへ写る (chat_stream_meta.py)。
    yield {"kind": "run_started", "payload": {"run_id": run_id, "session_id": session_id}}

    staged_view = _staged_view(_STAGED_PROJECT_ID)
    for f in facts:
        try:
            staged_view.add_facts([f])
            tv = decode_task_fact(f)
            ws.upsert_task(
                task_id=tv.task_id, title=tv.title, stage=tv.stage or "code",
                status="open", depends_on=tv.depends_on,
            )
        except Exception as exc:
            logger.warning("staged create: failed to register task: %s", exc)
    yield {"kind": "step", "payload": {
        "type": "long_form_plan",
        "detail": f"{len(facts)} タスクを生成 (仕様書→コード→テスト)",
        "status": "done",
    }}

    test_runner = (
        StagedTestRunner(
            workspace=ws,
            test_timeout_sec=float(staged_cfg.get("test_timeout_sec", 120.0)),
            debug_logger=state.debug_logger,
        )
        if staged_cfg.get("test_stage_enabled", True) else None
    )
    event_bus = LoopEventBus()
    smoke_timeout = float(staged_cfg.get("test_timeout_sec", 120.0))
    entry_exec_enabled = bool(staged_cfg.get("entry_smoke_exec_enabled", True))
    entry_exec_timeout = float(staged_cfg.get("entry_smoke_timeout_sec", 10.0))

    def _smoke(files: dict[str, str]) -> object:
        result = run_import_smoke(
            files, timeout_sec=smoke_timeout,
            internal_names=_staged_internal_names(ws),
        )
        extra_errors: list[str] = []
        for fn in (check_coherence, check_entrypoint, check_main_invoked, check_cross_module_imports):
            try:
                extra_errors += list(fn(files))
            except Exception as exc:
                logger.debug("staged static gate %s failed: %s", fn.__name__, exc)
        if extra_errors:
            result.errors = list(result.errors) + extra_errors
        if entry_exec_enabled:
            try:
                ent = run_entry_smoke(files, timeout_sec=entry_exec_timeout)
                if getattr(ent, "warnings", None):
                    result.warnings = list(result.warnings) + list(ent.warnings)
            except Exception as exc:
                logger.debug("staged entry exec smoke failed: %s", exc)
        return result

    part_assembler = None
    if part_codegen is not None:
        from backend.free.generation.part_assembler import assemble_file_parts
        part_assembler = assemble_file_parts

    from backend.free.generation.spec_conformance import check_spec_conformance
    from backend.free.generation.test_value_repair import repair_literal_assertions
    from backend.free.loop.staged.language_verify import VerifyCommand as _LoopVerifyCommand

    # 言語パックの検証コマンド (c_16 §4.5.4、段階 C-3、既定 OFF)。corpus 側の
    # 宣言 (LanguageOverlayEntry.verify) を loop pillar 自身の型へ変換する —
    # Loop は corpus を import できないため、この api 層が唯一の変換点
    # (store.py の PROJECT_MAP_PACKAGE_KIND と同じ複製の作法)。
    verify_cfg = staged_cfg.get("verify", {}) or {}
    verify_enabled = bool(verify_cfg.get("enabled", False))
    # 承認単位は argv 全体 (実行ファイル名だけでは無い、2026-09-20 レビュー) —
    # schema (StagedVerifyConfig) が起動時に各 argv の形を検証済み。
    verify_commands = tuple(
        tuple(str(a) for a in argv) for argv in (verify_cfg.get("commands") or [])
    )
    verify_timeout_sec = float(verify_cfg.get("timeout_sec", 60.0))

    language_verify_lookup = None
    if verify_enabled and state.cartridge_manager is not None:
        _cartridge_manager = state.cartridge_manager

        def language_verify_lookup(source_path: str) -> tuple[_LoopVerifyCommand, ...]:
            from pathlib import Path as _Path

            ext = _Path(source_path).suffix.lower()
            entry = _cartridge_manager.language_overlay().by_extension.get(ext)
            if entry is None or not entry.verify:
                return ()
            return tuple(
                _LoopVerifyCommand(
                    id=v.id, executable=v.executable, args=v.args,
                    timeout_sec=v.timeout_sec, success_exit_codes=v.success_exit_codes,
                )
                for v in entry.verify
            )

    _executor_kwargs: dict[str, Any] = dict(
        workspace=ws, aux_client=state.aux_client, codegen=codegen,
        smoke_runner=(_smoke if staged_cfg.get("smoke_gate_enabled", True) else None),
        test_runner=test_runner,
        contract_checker=check_api_contract,
        conformance_checker=check_spec_conformance,
        value_repair=repair_literal_assertions,
        max_test_regen_rounds=int(staged_cfg.get("max_test_regen_rounds", 2)),
        max_repair_rounds=int(staged_cfg.get("max_repair_rounds", 2)),
        spec_max_tokens=int(staged_cfg.get("spec_max_tokens", 6144)),
        spec_timeout_sec=float(staged_cfg.get("spec_timeout_sec", 600.0)),
        flowchart_enabled=bool(staged_cfg.get("flowchart_enabled", True)),
        spec_deepen_enabled=bool(staged_cfg.get("spec_deepen_enabled", True)),
        spec_conformance_enabled=bool(
            staged_cfg.get("spec_conformance_enabled", True),
        ),
        max_spec_revision_rounds=int(staged_cfg.get("max_spec_revision_rounds", 1)),
        part_codegen=part_codegen,
        part_assembler=part_assembler,
        coherence_checker=check_coherence,
        part_max_parts=int(staged_cfg.get("part_max_parts", 4)),
        language_verify_lookup=language_verify_lookup,
        verify_enabled=verify_enabled,
        verify_commands=verify_commands,
        verify_timeout_sec=verify_timeout_sec,
        event_bus=event_bus,
        debug_logger=state.debug_logger,
        brief=brief,
    )
    if _supports_kwarg(StagedCreateExecutor, "deadline_monotonic"):
        _executor_kwargs["deadline_monotonic"] = deadline_monotonic
    if _supports_kwarg(StagedCreateExecutor, "code_stage_min_share"):
        _executor_kwargs["stage_budget_sec"] = staged_total_timeout_sec
        _executor_kwargs["code_stage_min_share"] = float(
            staged_cfg.get("code_stage_min_share", 0.4),
        )
        _executor_kwargs["reference_doc"] = reference_doc
    executor = StagedCreateExecutor(**_executor_kwargs)
    artifact_hook = make_loop_artifact_hook(_staged_view)
    max_iter = int(staged_cfg.get("max_iterations", 60))
    driver = LoopDriver(
        view_provider=_staged_view,
        executor=executor,
        max_iterations=max_iter,
        max_wall_time_sec=staged_total_timeout_sec,
        max_consecutive_failures=max_iter,
        artifact_hook=artifact_hook,
        event_bus=event_bus,
        debug_logger=state.debug_logger,
    )
    driver.start(_STAGED_PROJECT_ID)
    total_tasks = len(facts)
    task_indices: dict[str, int] = {}
    queue = event_bus.subscribe()
    run_task = asyncio.create_task(
        driver.run(_STAGED_PROJECT_ID), name="staged_create_pipeline.run",
    )
    last_ka = time.monotonic()
    timed_out = False
    cancelled = False
    disconnected = False
    try:
        while True:
            # 打ち切るのはドライバがまだ走っている間だけ。ドライバも同じ予算で
            # 止まるので、全タスク成功で終わった同じ秒にここが先に上限を見て
            # 「タイムアウト・失敗」と表示していた (2026-09-21 ライブ監査 K02:
            # success=3 failure=0 の直後に hard cutoff)。終わっていれば残りの
            # イベントを流し切り、未完了の判定はループの後で行う。
            if (
                not run_task.done()
                and time.monotonic() - t_start >= staged_total_timeout_sec
            ):
                logger.warning(
                    "staged create: hard wall-time cutoff reached (%.0fs); "
                    "cancelling run_task (pipeline)",
                    staged_total_timeout_sec,
                )
                timed_out = True
                break
            if _is_cancelled():
                logger.info("staged create: cancel requested; stopping run_task (pipeline)")
                cancelled = True
                break
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if run_task.done() and queue.empty():
                    break
                if time.monotonic() - last_ka >= keepalive_interval:
                    yield {"kind": "keepalive", "payload": {}}
                    last_ka = time.monotonic()
                continue
            payload = _translate_loop_event_payload(
                evt, total_tasks=total_tasks, task_indices=task_indices,
            )
            if payload:
                # legacy と同じく表示に写る LoopEvent だけ events.jsonl へ積む
                # (fact_written / task_status 等の内部事象は積まない、f_10 §7)。
                if event_log is not None:
                    try:
                        event_log.append(str(evt.event), dict(getattr(evt, "data", None) or {}))
                    except Exception as exc:  # noqa: BLE001 - 記録失敗で配信を止めない
                        logger.warning("staged run event log append failed: %s", exc)
                yield {"kind": "step", "payload": payload}
                last_ka = time.monotonic()
    except asyncio.CancelledError:
        disconnected = True
        raise
    finally:
        event_bus.unsubscribe(queue)
        if not run_task.done():
            run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("staged create run task failed (pipeline): %s", exc)
        if disconnected:
            try:
                if event_log is not None:
                    event_log.append("disconnect", {"tasks_total": total_tasks})
            except Exception as exc:  # noqa: BLE001
                logger.warning("staged run event log append failed: %s", exc)
            _finish_run_safely(run_store, event_log, "disconnected")
            try:
                _staged_semmem.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("staged create: semmem close failed: %s", exc)

    if (
        not (timed_out or cancelled)
        and time.monotonic() - t_start >= staged_total_timeout_sec
        and _staged_remaining_units(ws)
    ):
        # ドライバ自身が同じ予算で止まり、未完了のタスクが残った。
        timed_out = True
    if timed_out:
        remaining = _staged_remaining_units(ws)
        suggested = max(
            _STAGED_TOTAL_TIMEOUT_DEFAULT_SEC,
            round(staged_total_timeout_sec * 1.5 / 300.0) * 300,
        )
        hint = (
            f"未完了 {remaining} ユニット。"
            if remaining else ""
        ) + (
            f"時間が足りない場合は config.yaml の "
            f"create.staged.total_timeout_sec を {suggested:.0f} 以上へ"
            f"引き上げてください (出荷既定 "
            f"{_STAGED_TOTAL_TIMEOUT_DEFAULT_SEC:.0f})。"
        )
        _kind, payload = _emit_check(event_log, "timeout", {
            "type": "task_result",
            "detail": (
                f"⏱ タイムアウト ({staged_total_timeout_sec:.0f}秒) のため打ち切りました。"
                f"生成済みの成果物のみ配信します。{hint}"
            ),
            "status": "failed",
        })
        yield {"kind": "step", "payload": payload}
    elif cancelled:
        remaining = _staged_remaining_units(ws)
        _kind, payload = _emit_check(event_log, "cancel", {
            "type": "task_result",
            "detail": (
                f"⏹ キャンセルのため打ち切りました。生成済みの成果物のみ配信します。"
                f"未完了 {remaining} ユニット。"
            ),
            "status": "failed",
        })
        yield {"kind": "step", "payload": payload}

    code_map: dict[str, str] = {}
    for wf in ws.list_files(kind="src"):
        c = ws.read_file(wf.logical_path, kind="src")
        if c:
            code_map[wf.logical_path] = c
    code_map.update(_requested_test_files(ws, requested_tests))
    checks = _StagedFinalizeChecks(code_map=code_map)
    exit_kind = "timeout" if timed_out else "cancelled" if cancelled else "done"
    try:
        if not cancelled:
            async for kind, payload in _finalize_staged_checks_events(
                ws=ws, state=state, event_log=event_log, smoke_timeout=smoke_timeout,
                prefetched_rag=prefetched_rag, file_context_block=file_context_block,
                out=checks,
            ):
                yield {"kind": "step", "payload": payload}
    except Exception:
        _finish_run_safely(run_store, event_log, "error")
        raise
    else:
        # legacy と同じ終端書込み (無いと run.json が running のまま残り、
        # Step 5.89 の GC 対象から外れ続ける — 2026-09-18 実機)。
        _finish_run_safely(
            run_store, event_log, exit_kind, tasks_failed=checks.tasks_failed,
        )

    spec_md = ws.read_spec()
    flowchart_md = ws.read_flowchart() if spec_md else None
    staged_metrics = {
        "units_total": len(checks.code_map),
        "units_completed": len(checks.code_map),
        "validation_errors": checks.tasks_failed + len(checks.runnability_issues),
        "content_type": "code",
        "strategy": "staged",
        "design_drift": (
            checks.design_drift_counts["divergence"]
            + checks.design_drift_counts["absence"]
        ),
    }

    # memmap を握った索引を先に手放す (Windows は掴んだままだと ``.semmem`` を
    # 削除できない、CLAUDE.md §10)。呼出側 (harness) は ``ws``/``ws.root`` を
    # 参照専用 (パス文字列組立) にしか使わないため、close 後でも安全。
    try:
        _staged_semmem.close()
    except Exception as exc:  # noqa: BLE001 - 後始末の失敗で応答を壊さない
        logger.debug("staged create: semmem close failed: %s", exc)
    if staged_cfg.get("cleanup_workspace", False):
        ws.cleanup()

    yield {"kind": "result", "payload": {
        "exit_kind": exit_kind,
        # workspace_root: write_impact 分類 (f_10 §8.1) の呼出元 (meta の
        # _execute_production_task) が RunEventLog を再構築するための鍵
        # (3a-2、以前は legacy finalize がこの関数内で直接 append していた)。
        "notes": {"run_id": run_id, "workspace_root": str(ws.root)},
        "run_id": run_id,
        "code_map": checks.code_map,
        "spec_md": spec_md,
        "flowchart_md": flowchart_md,
        "tasks_failed": checks.tasks_failed,
        "runnability_issues": checks.runnability_issues,
        "design_drift_counts": checks.design_drift_counts,
        "metrics": staged_metrics,
        "truncated_steps": list(getattr(executor, "truncated_steps", ()) or ()),
        "truncated_max_tokens": getattr(executor, "spec_max_tokens", None),
        "ws": ws,
        "event_log": event_log,
        "run_store": run_store,
        "total_tasks": total_tasks,
    }}


def _staged_pytest_counts(manifest: dict) -> tuple[int, int]:
    """staged manifest から生成ユニットテストの (合格数, 未合格数) を返す。

    ``<task_id>.pytest`` エントリは ``executor._run_advisory_pytest`` が
    永続化する。キー無しの ``<task_id>`` エントリは import スモークゲートの
    結果であり pytest 実行ではないため数えない (両者を混ぜると、テストが
    1 つも生成されなかったケースが「合格」に見える)。純粋関数。
    """
    records = [
        rec or {}
        for key, rec in (manifest.get("test_results") or {}).items()
        if key.endswith(".pytest")
    ]
    unpassed = sum(1 for rec in records if not rec.get("passed"))
    return len(records) - unpassed, unpassed


@dataclass
class _StagedFinalizeChecks:
    """finalize 検査の結果 (:func:`_finalize_staged_checks_events` の出力先)。

    async generator は値を ``return`` できない (PEP 525) ため、計算結果は
    この可変な側チャネル経由で書き戻す。呼出側 (SSE / 構造化イベントの
    どちらでも) が検査完了後に参照する。
    """

    code_map: dict[str, str]
    tasks_failed: int = 0
    pytest_passed: int = 0
    pytest_unpassed: int = 0
    runnability_issues: list[str] = field(default_factory=list)
    design_drift_counts: dict[str, int] = field(
        default_factory=lambda: {"convergence": 0, "divergence": 0, "absence": 0},
    )
    unmatched_keys: list[str] = field(default_factory=list)


async def _finalize_staged_checks_events(
    *,
    ws,
    state: AppState,
    event_log: "RunEventLog | None",
    smoke_timeout: float,
    prefetched_rag: list[tuple[str, float, str]] | None,
    file_context_block: str | None,
    out: _StagedFinalizeChecks,
) -> AsyncIterator[tuple[str, dict]]:
    """終端検査 (postprocess / smoke / coherence / design_drift) の本体。

    :func:`run_staged_pipeline` (構造化イベント) から呼ばれる終端検査の実装
    (f_10 §1)。``cancelled=True`` のときは呼出側がこの関数自体を呼ばない
    (``if not cancelled:``)。``events.jsonl`` への永続はここで完結し、
    ``(kind, payload)`` を yield する (ProductionEvent 化は呼出側)。
    ``out.code_map`` は cross-file import 配線後の内容へ書き換わる。
    """
    # 予算非依存の終端検証: cross-file import を決定論的に配線し (加算的 = 非劣化)、
    # 静的整合性 (重複定義 / 未定義名) を必ずチェックする。test 工程が wall-time で
    # starve されスモークゲートが走らなかった場合でも、配信前にここで担保される。
    code_map, coherence_issues, wired_paths = _staged_postprocess(out.code_map)

    # 終端の権威的な起動可能性判定: 静的整合 (coherence/entrypoint) に加え、配線後の
    # code_map へ import スモークを上乗せして cross-file ImportError も拾う。test 工程が
    # wall-time で starve された / test_stage_enabled=false でも、非起動コードを success
    # として学習記録しない (= ゲートをブロッキングにする) ための統合シグナル。
    import_errors = await _staged_import_smoke(
        code_map, smoke_timeout, _staged_internal_names(ws),
    )
    runnability_issues = coherence_issues + [
        e for e in import_errors if e not in coherence_issues
    ]

    manifest = ws.read_manifest() or {}
    progress = manifest.get("progress", {}) or {}
    tasks_failed = int(progress.get("tasks_failed", 0) or 0)
    pytest_passed, pytest_unpassed = _staged_pytest_counts(manifest)

    # 設計↔実装ドリフト検査 (f_10 §8.1、Phase 2.5): planner が意図したグラフ
    # (manifest["plan"]["module_deps"]、§2) と生成物の import 辺を突き合わせる
    # 決定論の観測 (LLM 不使用)。validation_errors には数えない — 観測から始める。
    design_drift_counts = {"convergence": 0, "divergence": 0, "absence": 0}
    module_deps = ws.read_plan()
    if module_deps:
        drift = None
        try:
            from backend.free.loop.staged.design_drift import check_design_drift
            drift = check_design_drift(code_map, module_deps)
        except Exception as exc:  # noqa: BLE001 - 観測の失敗で配信を止めない
            logger.warning("staged finalize design_drift check failed: %s", exc)
        if drift is not None:
            design_drift_counts = {
                "convergence": len(drift.convergence),
                "divergence": len(drift.divergence),
                "absence": len(drift.absence),
            }
            if event_log is not None:
                try:
                    event_log.append("design_drift", {
                        "convergence": [list(p) for p in drift.convergence],
                        "divergence": [list(p) for p in drift.divergence],
                        "absence": [list(p) for p in drift.absence],
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "staged run event log append failed (design_drift): %s", exc,
                    )
            detail = (
                f"設計↔実装: 一致 {design_drift_counts['convergence']} / "
                f"設計外 {design_drift_counts['divergence']} / "
                f"未実装 {design_drift_counts['absence']}"
            )
            if drift.divergence or drift.absence:
                extra: list[str] = []
                if drift.divergence:
                    head = "; ".join(f"{s}→{d}" for s, d in drift.divergence[:5])
                    more = (
                        f" ほか{len(drift.divergence) - 5}件"
                        if len(drift.divergence) > 5 else ""
                    )
                    extra.append(f"設計外: {head}{more}")
                if drift.absence:
                    head = "; ".join(f"{s}→{d}" for s, d in drift.absence[:5])
                    more = (
                        f" ほか{len(drift.absence) - 5}件"
                        if len(drift.absence) > 5 else ""
                    )
                    extra.append(f"未実装: {head}{more}")
                detail += " (" + " / ".join(extra) + ")"
            # 設計外/未実装があっても failed にはしない (観測なので UX を
            # 赤くしない) — 一致していない件数は detail に列挙して示す。
            yield _emit_check(event_log, "finalize_design_drift", {
                "type": "task_result",
                "detail": detail,
                "status": "done",
            })

    # 終端ゲート結果を long_form JSONL に記録し可測化する (develop=investigate/evolve
    # 時のみ出力)。SSE は表示専用で残らないため、配線件数/整合 issue を後から数値で追える。
    if state.debug_logger is not None:
        try:
            state.debug_logger.log_long_form_event({
                "phase": "staged_coherence",
                "strategy": "staged",
                "files": sum(1 for p in code_map if p.endswith(".py")),
                "wired_count": len(wired_paths),
                "wired_files": wired_paths,
                "coherence_issue_count": len(runnability_issues),
                "coherence_issues": runnability_issues[:20],
                "tasks_failed": tasks_failed,
                "pytest_unpassed_count": pytest_unpassed,
                "had_prefetched_rag": bool(prefetched_rag),
                "had_file_context": bool(file_context_block),
                "design_drift": design_drift_counts,
            })
        except Exception as exc:
            logger.debug("staged coherence long_form log failed: %s", exc)
    if tasks_failed:
        yield _emit_check(event_log, "finalize_tasks_failed", {
            "type": "task_result",
            "detail": f"⚠ {tasks_failed} 件のタスクが失敗しました (workspace: {ws.root})",
            "status": "failed",
        })
    if pytest_unpassed:
        yield _emit_check(event_log, "finalize_tests_unpassed", {
            "type": "task_result",
            "detail": f"⚠ テスト未合格: {pytest_unpassed} モジュール — 生成テストが"
                      f"失敗しています (成果物は配信します)",
            "status": "failed",
        })
    # 合格・未実行も明示する。従来は失敗時しか出さなかったため、実際には生成
    # テストが走って合格していても UI 上は起動可能性チェックの「未実行」表記
    # だけが残り、テスト未実行と区別が付かなかった (実インシデント 2026-07-27
    # ライブ検証: pytest が 8 passed で完了したのに合格表示が無かった)。
    if pytest_passed:
        yield _emit_check(event_log, "finalize_tests_passed", {
            "type": "task_result",
            "detail": f"生成ユニットテスト合格: {pytest_passed} モジュール (実行済み)",
            "status": "done",
        })
    elif not pytest_unpassed:
        yield _emit_check(event_log, "finalize_tests_skipped", {
            "type": "task_result",
            "detail": "生成ユニットテストは未実行です (テスト未生成またはスキップ)",
            "status": "done",
        })
    if runnability_issues:
        head = "; ".join(runnability_issues[:5])
        more = (
            f" ほか{len(runnability_issues) - 5}件"
            if len(runnability_issues) > 5 else ""
        )
        yield _emit_check(event_log, "finalize_runnability_issues", {
            "type": "task_result",
            "detail": f"⚠ 起動可能性チェック: {len(runnability_issues)} 件の問題 "
                      f"({head}{more})",
            "status": "failed",
        })

    # モジュール間の辞書キー不一致 (advisory)。import スモークは「起動できるか」
    # しか見ないため、片方が作ったキーをもう片方が別名で読む欠陥は素通りし、
    # 実行して初めて KeyError になる (実インシデント 2026-08-07 ライブ監査:
    # csv_processor が 'mean' を返すのに main.py が stats['average'] を読み、
    # 「起動可能性チェック合格」で配信された)。
    # プロジェクト外由来の辞書 (JSON 入力等) のキーは当然「作られて」いないので
    # **警告に留め、validation_errors には畳み込まない** (正常な生成を
    # 学習上の失敗にしないため)。
    unmatched_keys = find_unmatched_dict_keys(
        {p: c for p, c in code_map.items() if p.endswith(".py")},
    )
    if unmatched_keys:
        shown = "、".join(f"'{k}'" for k in unmatched_keys[:5])
        more_k = (
            f" ほか{len(unmatched_keys) - 5}件"
            if len(unmatched_keys) > 5 else ""
        )
        yield _emit_check(event_log, "finalize_unmatched_keys", {
            "type": "task_result",
            "detail": f"⚠ 参照のみで生成されていない辞書キー: {shown}{more_k} "
                      f"— 実行時 KeyError の可能性 (外部入力由来なら無視可)",
            "status": "failed",
        })

    out.code_map = code_map
    out.tasks_failed = tasks_failed
    out.pytest_passed = pytest_passed
    out.pytest_unpassed = pytest_unpassed
    out.runnability_issues = runnability_issues
    out.design_drift_counts = design_drift_counts
    out.unmatched_keys = unmatched_keys

