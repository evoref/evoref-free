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
import contextlib
import functools

from typing import (
    TYPE_CHECKING,
)

from backend.free.api.chat.chat_stream_common import (
    logger,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from backend.free.loop.staged.run_record import RunEventLog, RunRecordStore


#: staged create の run 1 リクエストの壁時計上限の出荷既定
#: (``create.turn_timeout_sec``、f_10 §3 予算の 3 層 — ターン予算)。
_TURN_TIMEOUT_DEFAULT_SEC = 3600.0

#: 計画と spec 本文へ渡す参照設計書の上限 (文字)。見出し単位で均等に縮める
#: (f_10 §2)。6000 字 ≒ 呼出 2 回ぶんの prefill 増で、モジュールごとには乗らない。
_REFERENCE_DOC_MAX_CHARS = 6000

#: ステージ予算 (create.staged.total_timeout_sec) をターン予算から確保する下限。
_STAGE_BUDGET_FLOOR_SEC = 300.0

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


class RunTerminal:
    """run.json の終端を 1 回だけ書く (最初の終端が勝つ、f_10 §7)。

    パイプラインは ``run_store.start`` の後で :meth:`bind` し、正常な結末は
    :meth:`finish` で書く。:func:`terminates_run` が取り消し・想定外の例外の
    出口で同じ :meth:`finish` を呼ぶので、終端済みなら何もしない。
    """

    def __init__(self) -> None:
        self.run_store: "RunRecordStore | None" = None
        self.event_log: "RunEventLog | None" = None
        self.finished = False

    def bind(self, run_store: "RunRecordStore | None", event_log: "RunEventLog | None") -> None:
        """開始を書いた run の記録先を覚える (開始に失敗したら ``None`` のまま)。"""
        self.run_store = run_store
        self.event_log = event_log

    def finish(self, exit_kind: str, *, tasks_failed: int = 0) -> None:
        """終端を同期で書く (await しない — 取り消しの最中でも書き切る)。2 回目以降は何もしない。"""
        if self.finished or self.run_store is None:
            return
        self.finished = True
        _finish_run_safely(self.run_store, self.event_log, exit_kind, tasks_failed=tasks_failed)


def terminates_run(body: "Callable[..., AsyncIterator[dict]]") -> "Callable[..., AsyncIterator[dict]]":
    """staged パイプライン (async generator) をどう抜けても run.json を終端する。

    ``body`` は ``run_terminal`` キーワードで :class:`RunTerminal` を受け取る。
    強制取り消し (``agent_task.cancel()`` = ``/api/chat/cancel`` / 正常停止) は
    await の途中に ``CancelledError`` を投げ、段の境界でしか見ない協調フラグの
    終端に届かない — ``running`` のまま残っていた (2026-10-14 実機)。
    ``CancelledError`` / ``GeneratorExit`` は ``cancelled``、それ以外の例外は
    ``error`` で終端してから送出し直す。
    """

    @functools.wraps(body)
    async def _guarded(*args, **kwargs) -> "AsyncIterator[dict]":
        terminal = RunTerminal()
        async with contextlib.aclosing(body(*args, run_terminal=terminal, **kwargs)) as events:
            try:
                async for event in events:
                    yield event
            except (asyncio.CancelledError, GeneratorExit):
                terminal.finish("cancelled")
                raise
            except BaseException:
                terminal.finish("error")
                raise

    return _guarded


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
    unchecked_out: list[str] | None = None,
    not_run_out: list[str] | None = None,
) -> list[str]:
    """配信前の code_map を import スモークし error 文字列列を返す (失敗時は空)。

    ``unchecked_out`` を渡すと、作業フォルダの外への書込みを止めたため検査できなかった
    ものをそこへ足す (不合格ではなく未検査、f_10 §11.1-4)。``not_run_out`` を渡すと、
    スモークが最後まで走らなかった (時間切れ・起動できない) 理由をそこへ足す (空の error を
    合格と取り違えないため、f_10 §11.1-4 の悪化の判定)。

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
        if not_run_out is not None:
            not_run_out.append(str(exc))
        return []
    if unchecked_out is not None:
        unchecked_out.extend(str(u) for u in (getattr(res, "unchecked", None) or []))
    if not_run_out is not None and not getattr(res, "completed", True):
        not_run_out.extend(getattr(res, "warnings", None) or ["import smoke did not complete"])
    return [str(e) for e in (getattr(res, "errors", None) or [])]



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
