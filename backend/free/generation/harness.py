"""LongFormHarness — create の longform 生成を ProductionHarness として動かす (Phase 3a)

[docs/f_03_agent_engine.md](../../../docs/f_03_agent_engine.md) §4.4 /
[docs/f_08_long_form_generation.md](../../../docs/f_08_long_form_generation.md) §2.2 の
``create.dispatch=meta`` アダプタ。``LongFormOrchestrator.generate(brief=,
on_step=)`` を包む。``on_step`` は sync 呼出し (``_call_step``、orchestrator 自体は
変えない) のため ``asyncio.Queue`` を経由して ``on_event`` (async) へ橋渡しする。

Phase 4 (f_08 §2.3): staged と同じ run 記録 (workspace / run.json /
events.jsonl / 途中経過の書き戻し / outline_drift) を持つ。gen pillar は
``backend.free.loop.staged.{workspace,run_record}`` を top-level import できない
ため、``RunRecorder`` Protocol (``backend/free/harness/production.py``) 越しに
composition 層 (``chat.py``) が注入する実装 (``StagedRunRecorder``) を叩く。

pillar 境界: 本モジュールは EvorefGen (``backend/free/generation/``)。gen は
他 pillar への top-level import を持たない (``ALLOWED_CROSS_PILLAR_IMPORTS["gen"]``
が空集合) ため、``EditorArtifact`` / ``ProductionEvent`` / ``RunRecorder`` 等
(EvorefLoop 側) はすべて関数内 / TYPE_CHECKING lazy import で参照する。
``LongFormOrchestrator`` の構築ロジック (pillar 横断の DI) も自前で持たず、
composition 層 (``chat.py``) が渡す ``orchestrator_factory`` に委ねる。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable

from backend.free.llm.editor_filename import derive_editor_filename_stem
from backend.io.id_registry import new_id
from backend.log_config import get_logger
from backend.trace_context import get_trace_id
from backend.utils import estimate_tokens

if TYPE_CHECKING:
    from backend.free.agent.meta_cognitive_tasks import EditorArtifact
    from backend.free.generation.orchestrator import LongFormOrchestrator
    from backend.free.harness.production import (
        ProductionRequest,
        ProductionResult,
        RunRecorder,
    )

logger = get_logger("generation.harness")

#: CODE の生成中 (最終ファイル構成が未確定) の間に途中経過を書き戻す仮ファイル名。
_CODE_PROGRESS_FILE = "output.py"

#: editor タブ表示用の拡張子 → 言語ラベル。``chat.py::_CODE_EXT_LANG`` と同じ
#: 対応表だが、gen pillar から composition 層 (api) を top-level import できない
#: ため独立して持つ (小さな純粋データなので重複のコストは小さい)。
_EXT_LANG: dict[str, str] = {
    "py": "python", "pyi": "python",
    "ts": "typescript", "tsx": "typescript",
    "js": "javascript", "jsx": "javascript", "mjs": "javascript",
    "svelte": "svelte", "vue": "vue",
    "rs": "rust", "go": "go", "java": "java", "kt": "kotlin",
    "c": "c", "h": "c", "cpp": "cpp", "cc": "cpp", "hpp": "cpp",
    "rb": "ruby", "php": "php", "cs": "csharp", "swift": "swift",
    "sh": "bash", "bash": "bash", "sql": "sql",
    "css": "css", "scss": "scss", "html": "html",
    "json": "json", "yaml": "yaml", "yml": "yaml", "md": "markdown",
}


def _artifact_from_file(path: str, code: str) -> "EditorArtifact":
    """``orchestrator.last_code_files`` の 1 エントリを ``EditorArtifact`` に変換する。"""
    from pathlib import PurePosixPath

    from backend.free.agent.meta_cognitive_tasks import EditorArtifact

    pp = PurePosixPath(path) if path else None
    name = pp.name if pp else ""
    ext = pp.suffix.lstrip(".").lower() if pp else ""
    return EditorArtifact(
        content=code, language=_EXT_LANG.get(ext, "python"), filename=name or None,
    )


def _deliverable_path(orchestrator: "LongFormOrchestrator", req: "ProductionRequest") -> str:
    """途中経過 / 確定本文の書き戻し先 (``src/<path>``) を決める (f_08 §2.3)。

    TEXT は ``derive_editor_filename_stem`` + ``orchestrator._target_format``
    (未指定なら ``.md``)。CODE は ``last_code_files`` が確定するまで
    ``_CODE_PROGRESS_FILE`` の仮名を使う (終端で実ファイル群に置き換わる)。
    """
    if getattr(orchestrator, "last_content_type", None) == "code":
        return _CODE_PROGRESS_FILE
    ext = getattr(orchestrator, "_target_format", "") or ".md"
    stem = derive_editor_filename_stem(hint=req.instruction, language="markdown")
    return f"{stem}{ext}"


def _base_notes(recorder: "RunRecorder | None", run_id: str) -> dict[str, str]:
    """``ProductionResult.notes`` の共通部分 (recorder が無ければ空、f_08 §2.3)。"""
    if recorder is None:
        return {}
    return {"run_id": run_id, "workspace_root": str(recorder.root)}


def _finish_recorder(
    recorder: "RunRecorder", exit_kind: str, *, template: str = "", tasks_failed: int = 0,
) -> None:
    """run 終端を記録する (失敗しても生成結果は返す)。

    ``template`` は構成テンプレートで seed した場合の来歴鍵 (c_05 §0.6)。
    """
    try:
        recorder.finish(exit_kind, template=template, tasks_failed=tasks_failed)
    except Exception as exc:  # noqa: BLE001 - 後始末の失敗で応答は壊さない
        logger.warning("longform run recorder finish(%s) failed: %s", exit_kind, exc)


def _write_final_files(recorder: "RunRecorder", files: dict[str, str]) -> None:
    """確定した生成物 (CODE は ``last_code_files``、TEXT は本文 1 件) を書き戻す。"""
    for path, code in files.items():
        if not code or not code.strip():
            continue
        try:
            recorder.write_src(path, code)
        except Exception as exc:  # noqa: BLE001 - 書込み失敗で応答は壊さない
            logger.warning("longform final write_src failed (path=%s): %s", path, exc)


def _validation_issues_artifact(errors: list[str]) -> "EditorArtifact":
    """リペア後も残った検証エラーを提示する markdown artifact (``chat.py`` と同じ形)。"""
    from backend.free.agent.meta_cognitive_tasks import EditorArtifact

    lines = "\n".join(f"- {e}" for e in errors)
    content = (
        f"# ⚠️ 自動検証で未解決のエラーが {len(errors)} 件あります\n\n"
        "生成されたコードには以下の検証エラーが残っています。"
        "実行前に修正してください。\n\n"
        f"{lines}\n"
    )
    return EditorArtifact(content=content, language="markdown", filename="GENERATION_ISSUES.md")


class LongFormHarness:
    """longform 生成 (:class:`LongFormOrchestrator`) の ``ProductionHarness`` アダプタ。"""

    name = "longform"

    def __init__(
        self,
        orchestrator_factory: "Callable[[], LongFormOrchestrator]",
        *,
        session_id: str = "",
        existing_content: str = "",
        existing_content_resolver: "Callable[[str], Awaitable[str]] | None" = None,
        prefetched_rag: list[tuple[str, float, str]] | None = None,
        file_context_block: str | None = None,
        run_recorder_factory: "Callable[[str], RunRecorder] | None" = None,
    ) -> None:
        """``orchestrator_factory`` は呼出のたび新しい ``LongFormOrchestrator`` を返す。

        構築 (main/aux client、memory_wm、policy 等の DI) は composition 層
        (``chat.py::_build_long_form_orchestrator``) の責務のまま — gen pillar
        はそれらを直接組み立てない (他 pillar 非依存の原則、CLAUDE.md §3)。

        ``run_recorder_factory`` は ``run_id`` を受け取り :class:`RunRecorder`
        を返す (composition 層が ``StagedRunRecorder.open`` を渡す、f_08 §2.3)。
        未指定 (``None``) の場合は run 記録を全て skip する (従来挙動、テスト用)。
        """
        self._orchestrator_factory = orchestrator_factory
        self._session_id = session_id
        self._existing_content = existing_content
        # 追記 / 参照依頼の既存ファイル内容は instruction が分かる run() 時にしか
        # 解決できない (composition 層が ``read_existing_for_append`` を注入する)。
        self._existing_content_resolver = existing_content_resolver
        self._prefetched_rag = prefetched_rag
        self._file_context_block = file_context_block
        self._run_recorder_factory = run_recorder_factory

    async def run(
        self,
        req: "ProductionRequest",
        *,
        on_event,
        is_cancelled,
    ) -> "ProductionResult":
        """longform 生成を駆動し、生成物を :class:`ProductionResult` で返す。"""
        from backend.free.harness.production import ProductionEvent, ProductionResult

        orchestrator = self._orchestrator_factory()
        existing_content = self._existing_content
        if self._existing_content_resolver is not None:
            try:
                existing_content = await self._existing_content_resolver(req.instruction)
            except Exception as exc:  # noqa: BLE001 - 参照失敗は空で続行 (legacy と同じ)
                logger.warning("longform harness: existing content resolve failed: %s", exc)
                existing_content = ""
        queue: asyncio.Queue[dict] = asyncio.Queue()

        run_id = new_id("run_")
        recorder: "RunRecorder | None" = None
        if self._run_recorder_factory is not None:
            try:
                recorder = self._run_recorder_factory(run_id)
                recorder.start(
                    session_id=req.session_id or self._session_id,
                    request_id=get_trace_id() or "",
                    mode="create",
                    query=req.instruction,
                    output_target=req.output_target,
                    brief_tokens=estimate_tokens(req.brief) if req.brief else 0,
                    resume_of=req.resume_of,
                )
            except Exception as exc:  # noqa: BLE001 - run 記録の失敗で生成は止めない
                logger.warning("longform run recorder start failed: %s", exc)
                recorder = None
        if recorder is not None:
            # SSE フレーム化 (meta の create_run) 用。events.jsonl への追記は
            # run.json の start() 自体が事実を持つため二重に積まない
            # (staged の run_started フレームと同じ扱い、f_10 §7)。
            await on_event(ProductionEvent(
                kind="run_started",
                payload={"run_id": run_id, "session_id": req.session_id or self._session_id},
            ))

        buffer_parts: list[str] = []

        def _on_step(data: dict) -> None:
            # orchestrator._call_step は sync 呼出し。同一コルーチン内の
            # put_nowait は競合しない (次に drain するまでキューに積むだけ)。
            # 永続化は配信より先に行う (events.jsonl → SSE、f_10 §7)。
            if recorder is not None:
                try:
                    recorder.append_event(str(data.get("type") or "step"), data)
                except Exception as exc:  # noqa: BLE001 - 記録の失敗で生成は止めない
                    logger.warning("longform run event append failed: %s", exc)
            queue.put_nowait(data)

        def _write_progress() -> None:
            if recorder is None:
                return
            try:
                recorder.write_src(
                    _deliverable_path(orchestrator, req), "".join(buffer_parts),
                )
            except Exception as exc:  # noqa: BLE001 - 途中経過の書込み失敗で生成は止めない
                logger.warning("longform progress write_src failed: %s", exc)

        async def _drain() -> None:
            while not queue.empty():
                data = queue.get_nowait()
                kind = str(data.get("type") or "")
                if recorder is not None and kind == "long_form_document_gate":
                    gate = getattr(orchestrator, "last_document_gate", None)
                    if gate is not None:
                        try:
                            recorder.append_event("outline_drift", gate)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("longform outline_drift append failed: %s", exc)
                if kind == "long_form_unit_done":
                    _write_progress()
                await on_event(ProductionEvent(kind="step", payload=data))

        exit_kind = "done"
        try:
            async for token in orchestrator.generate(
                instruction=req.instruction,
                session_id=req.session_id or self._session_id,
                mode="create",
                on_step=_on_step,
                existing_content=existing_content,
                prefetched_rag=self._prefetched_rag,
                file_context_block=self._file_context_block,
                brief=req.brief,
            ):
                buffer_parts.append(token)
                await _drain()
                if is_cancelled():
                    logger.info("longform harness: cancel requested; stopping generate()")
                    exit_kind = "cancelled"
                    break
        except Exception as exc:
            logger.warning("longform harness generation failed: %s", exc)
            await _drain()
            if recorder is not None:
                _finish_recorder(recorder, "error")
            notes = _base_notes(recorder, run_id)
            notes["error"] = str(exc)
            return ProductionResult(exit_kind="error", notes=notes)
        await _drain()
        # 総時間で打ち切ったなら「予算で止まり未完了が残った」= timeout (f_10 §7)。
        # 以前は done と記録し、meta の報告も「完了」だった (2026-09-26)。
        final_metrics = getattr(orchestrator, "last_metrics", None)
        if (
            exit_kind == "done" and isinstance(final_metrics, dict)
            and final_metrics.get("timed_out") is True
        ):
            exit_kind = "timeout"

        content_type = getattr(orchestrator, "last_content_type", None)
        artifacts: list[EditorArtifact] = []
        #: リペア後も残った CODE の検証エラー (構文 / 整合 / import スモーク)。
        #: 配信はするが未完了として数える (f_08 §2.3、2026-09-26)。
        unresolved: list[str] = []
        if content_type == "code":
            files = orchestrator.last_code_files or (
                {"output.py": orchestrator.last_code_output}
                if orchestrator.last_code_output else {}
            )
            artifacts = [
                _artifact_from_file(path, code)
                for path, code in files.items()
                if code and code.strip()
            ]
            # 「壊れたコードを成功として渡さない」: リペア後も残った検証エラーが
            # あれば best-effort で成果物を返しつつ未解決エラーを可視化する。
            if artifacts and orchestrator.last_validation_errors:
                unresolved = list(orchestrator.last_validation_errors)
                artifacts.append(_validation_issues_artifact(unresolved))
            if recorder is not None:
                _write_final_files(recorder, files)
        else:
            from backend.free.agent.meta_cognitive_tasks import EditorArtifact
            text = orchestrator.last_text_output
            if text and text.strip():
                artifacts = [EditorArtifact(content=text, language="text", filename=None)]
                if recorder is not None:
                    _write_final_files(
                        recorder, {_deliverable_path(orchestrator, req): text},
                    )

        if recorder is not None:
            template = str(getattr(orchestrator, "last_metrics", {}).get("template") or "")
            _finish_recorder(
                recorder, exit_kind, template=template, tasks_failed=1 if unresolved else 0,
            )

        notes = _base_notes(recorder, run_id)
        if unresolved:
            notes["validation_errors"] = len(unresolved)
        return ProductionResult(
            artifacts=artifacts,
            exit_kind=exit_kind,
            metrics=dict(getattr(orchestrator, "last_metrics", {}) or {}),
            notes=notes,
        )
