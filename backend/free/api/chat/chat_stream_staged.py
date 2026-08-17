"""Staged クリエイト (仕様書 → コード → テスト) のストリーミング"""

from __future__ import annotations

import asyncio
import time

from typing import (
    AsyncIterator,
    TYPE_CHECKING,
)
from pathlib import Path
from backend.app_state import AppState
from backend.free.api.chat.chat_constants import DEFAULT_KEEPALIVE_INTERVAL_SEC
from backend.free.api.chat.chat_recorder import record_long_form_response
from backend.free.api.chat.chat_service import make_token_info
from backend.free.api.chat.chat_types import ChatMessage
from backend.free.agent.tool_call_judge import _extract_file_path
from backend.free.generation.key_coherence import find_unmatched_dict_keys
from backend.free.generation.validators import remove_code_fences
from backend.utils import estimate_tokens as _estimate_tokens

from backend.free.api.chat.chat_stream_common import (
    logger,
    rag_signals_from_chunks,
    sse,
)

from backend.free.api.chat.chat_stream_output import (
    _editor_language_for_extension,
)

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer


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


def _translate_loop_event(
    evt, total_tasks: int = 0, task_indices: dict[str, int] | None = None,
) -> str | None:
    """LoopEvent を staged 進捗の SSE step フレームへ翻訳する (該当なしは None)。

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
        return sse.step({
            "type": "long_form_unit_start",
            "detail": f"{prefix}{label}: {title}{suffix}".strip(),
            "status": "running",
        })
    if evt.event == "iteration_ended":
        outcome = data.get("last_outcome") or {}
        status = str(outcome.get("status", ""))
        ok = status == "success"
        if task_indices is not None and tid in task_indices:
            idx = task_indices[tid]
        else:
            idx = getattr(evt, "iteration", 0) or 0
        prefix = f"[{idx}/{total_tasks}] " if total_tasks else ""
        return sse.step({
            "type": "long_form_unit_done",
            "detail": f"{prefix}{label}: {'完了' if ok else (status or '終了')}",
            "status": "done" if ok else "failed",
        })
    if evt.event == "stage_progress":
        detail = str(data.get("detail", "")).strip()
        status = str(data.get("status", "running"))
        if not detail:
            return None
        return sse.step({
            "type": "task_progress",
            "detail": detail,
            "status": status,
        })
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
        return sse.step({
            "type": "task_result",
            "detail": detail,
            "status": "done" if ok else "failed",
        })
    return None


# staged の task グラフは **リクエスト毎の隔離ストア** (workspace 内 .semmem) に持つ。
# 共有 project ストア (state.current_project_id) を使うと ①継続ターンで stale な
# done ファクトが新ターンの spec→code→test 依存ゲートを壊す ②自律ループ
# (state.loop_driver / RalphExecutor) が stage 付きタスクを誤実行する、という不具合に
# なるため、永続プロジェクトストアからは完全に切り離す。
_STAGED_PROJECT_ID = "staged"

#: staged クリエイト 1 リクエストの総時間上限の出荷既定
#: (:class:`backend.schemas.create.StagedCreateConfig` と一致させる)。
#: 打ち切りメッセージで「設定値が既定より低い」ことを示すために参照する。
_STAGED_TOTAL_TIMEOUT_DEFAULT_SEC = 2400.0


def _staged_output_dir(query: str) -> str:
    """staged 成果物の書き出し先ディレクトリをユーザークエリから解決する。

    従来は logical path (``stats.py`` / ``SPEC.md``) をそのまま write_file に
    渡していたため、バックエンドの CWD (= リポジトリルート) に書き出され、
    ユーザーが指定したディレクトリが無視されていた (実インシデント
    2026-07-27 ライブ検証: 「E:\\tmp\\evoref_test\\stats.py を作って」に対し
    リポジトリ直下へ stats.py / SPEC.md / flowchart.md が生成された)。

    Returns:
        解決したディレクトリ。クエリにパス指定が無ければ空文字列
        (従来どおり logical path をそのまま使う)。純粋関数。
    """
    referenced = _extract_file_path(query)
    if not referenced:
        return ""
    path = Path(referenced)
    parent = path.parent if path.suffix else path
    resolved = str(parent)
    return "" if resolved in ("", ".") else resolved


def _staged_deliverable_path(out_dir: str, logical_path: str) -> str:
    """logical path を出力先ディレクトリ配下へ寄せる (純粋関数)。"""
    return str(Path(out_dir) / logical_path) if out_dir else logical_path


async def _staged_write_file(
    state: AppState, logical_path: str, content: str,
) -> str | None:
    """output_target=="file" 時に生成ファイルを registry.write_file で書き出す。"""
    registry = state.tools_registry
    if registry is None or not registry.has("write_file"):
        return None
    try:
        # markdown (SPEC.md 等) は ```mermaid 等の正当なコードフェンスを含むため
        # 除去しない。コードファイルのみ LLM が付ける外側フェンスを剥がす。
        body = content if logical_path.endswith(".md") else remove_code_fences(content)
        return str(await registry.execute(
            "write_file", file_path=logical_path, content=body,
        ))
    except Exception as exc:
        logger.warning("staged write_file failed for %s: %s", logical_path, exc)
        return None


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
    for fn in (check_coherence, check_entrypoint, check_cross_module_imports):
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


async def stream_staged_create(
    *,
    query: str,
    session_id: str,
    state: AppState,
    cfg: dict,
    instance_name: str,
    context_size: int,
    messages: list[ChatMessage],
    output_target: str,
    codegen,
    fallback_factory,
    part_codegen=None,
    timer: StageTimer | None = None,
    private: bool = False,
    keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL_SEC,
    prefetched_rag: list[tuple[str, float, str]] | None = None,
    prefetched_rag_top_score: float | None = None,
    file_context_block: str | None = None,
) -> AsyncIterator[str]:
    """専用 LoopDriver をインライン駆動し spec→code→test を実行してストリームする。

    タスクグラフ合成が空 (aux degraded 等) のときは ``fallback_factory`` が返す
    従来 longform ストリームへ委譲する。``part_codegen`` (部分ごと生成向けの別予算
    delegate) が渡されたときのみ部分生成→決定論結合経路を有効化する。

    ``prefetched_rag``/``file_context_block`` は spec/code 生成プロンプトへは
    注入しない (f_10 §9 の非可逆な再合成 LLM パスを避ける方針)。Level 0 経験
    記録の rag_used/rag_top1_score シグナルと staged_coherence ログの可観測性
    のみに使う (longform 経路と record 粒度を揃える)。
    """
    from uuid import uuid4

    from backend.config import get_path_resolver
    from backend.free.loop.artifact_writer import make_loop_artifact_hook
    from backend.free.loop.driver import LoopDriver, decode_task_fact
    from backend.free.loop.events import LoopEventBus
    from backend.free.loop.staged import (
        WorkspaceManager,
        synthesize_create_task_graph,
    )
    from backend.free.loop.staged.executor import StagedCreateExecutor
    from backend.free.loop.staged.test_runner import StagedTestRunner
    from backend.free.generation.api_contract import check_api_contract
    from backend.free.generation.smoke_validator import (
        check_coherence,
        check_cross_module_imports,
        check_entrypoint,
        run_entry_smoke,
        run_import_smoke,
    )
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.views.loop import LoopFactView

    t_start = time.monotonic()
    staged_cfg = (cfg.get("create", {}) or {}).get("staged", {}) or {}
    # editor_route は search_error_wrapper (chat.py) が冒頭で 1 度送出するため、
    # ここでは送らない (二重送出回避)。

    # リクエスト毎に隔離されたワークスペース + SemMem ストアを使う (継続ターンの
    # stale ファクト混入・自律ループとの干渉を構造的に排除する)。
    run_id = uuid4().hex[:12]
    workspace_root = get_path_resolver().resolve_local("create_workspace_dir")
    ws = WorkspaceManager.open_or_create(
        workspace_root, workspace_id=run_id, session_id=session_id,
        project_id=_STAGED_PROJECT_ID, goal=query, debug_logger=state.debug_logger,
    )
    # 隔離 SemMem ストア (workspace 内 .semmem)。永続 project ストアには触れない。
    staged_store = SemanticFactStore.for_project(ws.root / ".semmem", _STAGED_PROJECT_ID)

    def _staged_view(_pid: str) -> LoopFactView:
        return LoopFactView(stores=[staged_store], writeback_store=staged_store)

    yield sse.step({
        "type": "long_form_plan",
        "detail": "タスクグラフ (仕様書/コード/テスト) を合成中…",
        "status": "running",
    })
    facts = await synthesize_create_task_graph(
        request=query, project_id=_STAGED_PROJECT_ID,
        aux_client=state.aux_client,
        include_tests=(
            bool(staged_cfg.get("test_stage_enabled", True))
            or bool(staged_cfg.get("smoke_gate_enabled", True))
        ),
        debug_logger=state.debug_logger,
    )
    if not facts:
        logger.info("staged create: empty task graph; falling back to longform")
        async for frame in fallback_factory():
            yield frame
        return

    for f in facts:
        try:
            staged_store.add_fact(f)
            tv = decode_task_fact(f)
            ws.upsert_task(
                task_id=tv.task_id, title=tv.title, stage=tv.stage or "code",
                status="open", depends_on=tv.depends_on,
            )
        except Exception as exc:
            logger.warning("staged create: failed to register task: %s", exc)
    yield sse.step({
        "type": "long_form_plan",
        "detail": f"{len(facts)} タスクを生成 (仕様書→コード→テスト)",
        "status": "done",
    })

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
        # test 工程の決定論的ゲート。外部依存 (pygame 等) の ModuleNotFound は
        # run_import_smoke 内で warning 扱い (=合格)、ただし stdlib の OS 非互換
        # (Windows の curses 等) は error 化して有界リペア対象にする。import only では
        # 拾えない静的整合 (重複定義 / 未定義名) を check_coherence、起動不能 (エントリが
        # 未定義メソッドを呼ぶ) を check_entrypoint、生成物間 from-import の名前欠落
        # (外部依存欠落で import スモークが盲目化しても拾える) を
        # check_cross_module_imports で error に上乗せ。エントリ有界実行は advisory。
        result = run_import_smoke(
            files, timeout_sec=smoke_timeout,
            internal_names=_staged_internal_names(ws),
        )
        extra_errors: list[str] = []
        for fn in (check_coherence, check_entrypoint, check_cross_module_imports):
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
        # 部分結合 (EvorefGen 具象) は有効時のみ lazy import で注入する
        # (smoke_runner / contract_checker と同じ loop→gen 越境回避パターン)。
        from backend.free.generation.part_assembler import assemble_file_parts
        part_assembler = assemble_file_parts

    # spec 宣言契約と生成コードの照合 (EvorefGen 具象) も同パターンで注入。
    from backend.free.generation.spec_conformance import check_spec_conformance
    from backend.free.generation.test_value_repair import repair_literal_assertions

    executor = StagedCreateExecutor(
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
        event_bus=event_bus,
        debug_logger=state.debug_logger,
    )
    artifact_hook = make_loop_artifact_hook(_staged_view)
    max_iter = int(staged_cfg.get("max_iterations", 60))
    staged_total_timeout_sec = float(
        staged_cfg.get("total_timeout_sec", _STAGED_TOTAL_TIMEOUT_DEFAULT_SEC),
    )
    driver = LoopDriver(
        view_provider=_staged_view,
        executor=executor,
        max_iterations=max_iter,
        max_wall_time_sec=staged_total_timeout_sec,
        # モジュールは互いに独立。1 モジュールの失敗で全体を打ち切らないよう
        # 連続失敗での abort を実質無効化する (max_iterations / 総時間で有界)。
        max_consecutive_failures=max_iter,
        artifact_hook=artifact_hook,
        event_bus=event_bus,
        debug_logger=state.debug_logger,
    )
    driver.start(_STAGED_PROJECT_ID)
    total_tasks = len(facts)
    task_indices: dict[str, int] = {}  # task_id 初出順の表示番号 (リトライで再利用)
    queue = event_bus.subscribe()
    run_task = asyncio.create_task(
        driver.run(_STAGED_PROJECT_ID), name="staged_create.run",
    )
    last_ka = time.monotonic()
    timed_out = False
    try:
        while True:
            # LoopDriver.run() 自身の wall-time チェックはイテレーション間の協調的
            # チェックのみで、実行中の単一タスク (await executor.execute(task)) を
            # 打ち切れない。ここで run_task 自体をハード打ち切りすることで、
            # meta_cognitive.py の asyncio.wait_for(total_timeout) 相当の強制締切を
            # staged 側にも持たせる。
            if time.monotonic() - t_start >= staged_total_timeout_sec:
                logger.warning(
                    "staged create: hard wall-time cutoff reached (%.0fs); "
                    "cancelling run_task",
                    staged_total_timeout_sec,
                )
                timed_out = True
                break
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if run_task.done() and queue.empty():
                    break
                if time.monotonic() - last_ka >= keepalive_interval:
                    yield sse.keepalive()
                    last_ka = time.monotonic()
                continue
            frame = _translate_loop_event(
                evt, total_tasks=total_tasks, task_indices=task_indices,
            )
            if frame:
                yield frame
                last_ka = time.monotonic()
    finally:
        event_bus.unsubscribe(queue)
        if not run_task.done():
            run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("staged create run task failed: %s", exc)

    if timed_out:
        # 打ち切りは「設定値が作業量に対して小さい」ことがほとんどなので、
        # どこを変えればよいかまで書く。値だけ出しても利用者は次に何をすれば
        # よいか分からない (実インシデント 2026-08-07 ライブ監査: config.yaml が
        # 出荷既定の 2400 より低い 1800 を明示していたため 5 本中 3 本が打ち切られ、
        # メッセージからはその関係が読み取れなかった)。
        progress = ws.read_manifest().get("progress") or {}
        remaining = max(
            0,
            int(progress.get("tasks_total") or 0)
            - int(progress.get("tasks_done") or 0)
            - int(progress.get("tasks_failed") or 0),
        )
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
        yield sse.step({
            "type": "task_result",
            "detail": (
                f"⏱ タイムアウト ({staged_total_timeout_sec:.0f}秒) のため打ち切りました。"
                f"生成済みの成果物のみ配信します。{hint}"
            ),
            "status": "failed",
        })

    async for frame in _finalize_staged_stream(
        ws=ws, state=state, query=query, messages=messages,
        session_id=session_id, instance_name=instance_name,
        context_size=context_size, output_target=output_target,
        timer=timer, t_start=t_start, private=private,
        smoke_timeout=smoke_timeout,
        prefetched_rag=prefetched_rag,
        prefetched_rag_top_score=prefetched_rag_top_score,
        file_context_block=file_context_block,
    ):
        yield frame

    # 隔離ワークスペース (含 .semmem) のクリーンアップ (config で任意)。
    if staged_cfg.get("cleanup_workspace", False):
        ws.cleanup()


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


async def _finalize_staged_stream(
    *,
    ws,
    state: AppState,
    query: str,
    messages: list[ChatMessage],
    session_id: str,
    instance_name: str,
    context_size: int,
    output_target: str,
    timer: StageTimer | None,
    t_start: float,  # noqa: ARG001
    private: bool,
    smoke_timeout: float = 120.0,
    prefetched_rag: list[tuple[str, float, str]] | None = None,
    prefetched_rag_top_score: float | None = None,
    file_context_block: str | None = None,
) -> AsyncIterator[str]:
    """staged 終端: 生成物を集約し output_target 別に配信 + token_info/done。"""
    if timer:
        timer.stop("llm_total_ms")
    code_map: dict[str, str] = {}
    for wf in ws.list_files(kind="src"):
        c = ws.read_file(wf.logical_path, kind="src")
        if c:
            code_map[wf.logical_path] = c
    # 予算非依存の終端検証: cross-file import を決定論的に配線し (加算的 = 非劣化)、
    # 静的整合性 (重複定義 / 未定義名) を必ずチェックする。test 工程が wall-time で
    # starve されスモークゲートが走らなかった場合でも、配信前にここで担保される。
    code_map, coherence_issues, wired_paths = _staged_postprocess(code_map)

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
            })
        except Exception as exc:
            logger.debug("staged coherence long_form log failed: %s", exc)
    if tasks_failed:
        yield sse.step({
            "type": "task_result",
            "detail": f"⚠ {tasks_failed} 件のタスクが失敗しました (workspace: {ws.root})",
            "status": "failed",
        })
    if pytest_unpassed:
        yield sse.step({
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
        yield sse.step({
            "type": "task_result",
            "detail": f"生成ユニットテスト合格: {pytest_passed} モジュール (実行済み)",
            "status": "done",
        })
    elif not pytest_unpassed:
        yield sse.step({
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
        yield sse.step({
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
        yield sse.step({
            "type": "task_result",
            "detail": f"⚠ 参照のみで生成されていない辞書キー: {shown}{more_k} "
                      f"— 実行時 KeyError の可能性 (外部入力由来なら無視可)",
            "status": "failed",
        })

    assembled = "\n\n".join(
        f"# === {p} ===\n{c}" for p, c in code_map.items()
    )
    # metrics は long_form router の success/false_positive 判定に使われる
    # (success = units_completed>0 ∧ validation_errors==0、false_positive = units==0)。
    # units_completed は生成ファイル数のまま (>0 → routing 自体は妥当で false_positive
    # にしない) だが、validation_errors に失敗タスク + 起動可能性 issue を畳み込み、
    # 非起動コードを long_form_success として学習記録しない (ゲートをブロッキング化)。
    staged_metrics = {
        "units_total": len(code_map),
        "units_completed": len(code_map),
        "validation_errors": tasks_failed + len(runnability_issues),
        "content_type": "code",
        "strategy": "staged",
    }
    try:
        _staged_rag_used, _staged_rag_top1 = rag_signals_from_chunks(
            prefetched_rag, prefetched_rag_top_score,
        )
        record_long_form_response(
            state, assembled, messages, session_id, query, "create",
            _estimate_tokens(assembled), staged_metrics, private=private,
            rag_used=_staged_rag_used, rag_top1_score=_staged_rag_top1,
        )
    except Exception as exc:
        logger.warning("staged: record_long_form_response failed: %s", exc)

    if not code_map:
        yield sse.step({
            "type": "task_result",
            "detail": "コードが生成されませんでした",
            "status": "failed",
        })
    elif output_target == "editor":
        for p, c in code_map.items():
            lang = _editor_language_for_extension(Path(p).suffix) or "python"
            yield sse.editor_code(
                remove_code_fences(c), language=lang, filename=p,
            )
    elif output_target == "chat":
        for p, c in code_map.items():
            lang = _editor_language_for_extension(Path(p).suffix) or ""
            yield sse.token(f"\n\n**{p}**\n```{lang}\n{c}\n```\n")
    else:  # file
        out_dir = _staged_output_dir(query)
        written = [
            res for p, c in code_map.items()
            if (res := await _staged_write_file(
                state, _staged_deliverable_path(out_dir, p), c,
            ))
        ]
        detail = (
            f"{len(written)} ファイルを書き込みました"
            + (f" ({out_dir})" if out_dir and written else "")
            if written else f"生成物は workspace にあります: {ws.root}"
        )
        yield sse.step({
            "type": "task_result", "detail": detail, "status": "done",
        })

    spec_md = ws.read_spec()
    if spec_md:
        # 設計フローチャートは UI 表示せず、ファイル成果物としてのみ出力する
        # (ユーザー要望)。チャットへの mermaid 描画フレームは送らない。
        flowchart = ws.read_flowchart()
        # SPEC.md (flowchart は含まない。flowchart.md は下で別ファイルとして届ける)
        # を output_target 別に成果物として届ける。
        if output_target == "editor":
            yield sse.editor_code(spec_md, language="markdown", filename="SPEC.md")
        elif output_target == "file":
            await _staged_write_file(
                state, _staged_deliverable_path(_staged_output_dir(query), "SPEC.md"),
                spec_md,
            )
        yield sse.step({
            "type": "task_result",
            "detail": f"設計仕様: {ws.path('spec.md')}",
            "status": "done",
        })

        # フローチャートを独立した成果物ファイルとしても届ける (ユーザー要望)。
        if flowchart and flowchart.strip():
            fc_doc = f"# 設計フローチャート\n\n```mermaid\n{flowchart.strip()}\n```\n"
            if output_target == "editor":
                yield sse.editor_code(fc_doc, language="markdown", filename="flowchart.md")
            elif output_target == "file":
                await _staged_write_file(
                    state,
                    _staged_deliverable_path(
                        _staged_output_dir(query), "flowchart.md",
                    ),
                    fc_doc,
                )
            yield sse.step({
                "type": "task_result",
                "detail": f"フローチャート: {ws.path('flowchart.md')}",
                "status": "done",
            })

    ti = make_token_info(
        messages, _estimate_tokens(assembled), context_size, instance_name,
    )
    yield sse.token_info(ti)
    yield sse.done()
