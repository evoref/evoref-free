"""StagedCodeHarness — staged クリエイトを ProductionHarness として動かす (Phase 3a)

[docs/f_03_agent_engine.md](../../../../docs/f_03_agent_engine.md) §4.4 /
[docs/f_10_staged_create_pipeline.md](../../../../docs/f_10_staged_create_pipeline.md) §1 の
``create.dispatch=meta`` アダプタ。合成 → ``LoopDriver`` → finalize 検査の実体
(``run_staged_pipeline``、``backend/free/api/chat/chat_stream_staged.py``) を
駆動し、StagedEvent (``kind``/``payload`` の dict) を :class:`ProductionEvent`
へ写す。配信 (SSE 化 / ディスク書込) は持たない — 生成物は
:class:`EditorArtifact` のリストとして返し、``output_target=="file"`` の実際の
書込みは呼出元 (``MetaCognitiveAgent``) の既存 write 経路に委ねる。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from backend.free.agent.meta_cognitive_tasks import EditorArtifact
from backend.free.harness.production import (
    ProductionEvent,
    ProductionRequest,
    ProductionResult,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState

logger = get_logger("loop.staged.harness")

#: 拡張子 → editor 表示用言語ラベル (best-effort)。
_EXT_LANG: dict[str, str] = {
    "py": "python", "js": "javascript", "ts": "typescript", "jsx": "javascript",
    "tsx": "typescript", "json": "json", "yaml": "yaml", "yml": "yaml",
    "md": "markdown", "html": "html", "css": "css", "sh": "bash",
}


def _language_for_path(path: str) -> str:
    """拡張子から editor 表示用の言語ラベルを引く (best-effort)。"""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return _EXT_LANG.get(ext, "python")


class StagedCodeHarness:
    """staged クリエイトパイプラインの ``ProductionHarness`` アダプタ。"""

    name = "staged"

    def __init__(
        self,
        *,
        state: "AppState",
        cfg: dict,
        codegen,
        part_codegen=None,
        prefetched_rag: list[tuple[str, float, str]] | None = None,
        prefetched_rag_top_score: float | None = None,
        file_context_block: str | None = None,
    ) -> None:
        self._state = state
        self._cfg = cfg
        self._codegen = codegen
        self._part_codegen = part_codegen
        self._prefetched_rag = prefetched_rag
        self._prefetched_rag_top_score = prefetched_rag_top_score
        self._file_context_block = file_context_block

    async def run(
        self,
        req: ProductionRequest,
        *,
        on_event,
        is_cancelled,
    ) -> ProductionResult:
        """staged パイプラインを駆動し、生成物を :class:`ProductionResult` で返す。"""
        # api 層 (composition) の共有実体への参照。loop pillar → api の top-level
        # import は方向が逆になるため lazy import で避ける
        # (backend/free/loop/staged/README 相当の既存パターンと同じ判断)。
        from backend.free.api.chat.chat_stream_staged import run_staged_pipeline

        result_payload: dict | None = None
        async for event in run_staged_pipeline(
            query=req.instruction,
            session_id=req.session_id,
            state=self._state,
            cfg=self._cfg,
            output_target=req.output_target,
            codegen=self._codegen,
            part_codegen=self._part_codegen,
            prefetched_rag=self._prefetched_rag,
            prefetched_rag_top_score=self._prefetched_rag_top_score,
            file_context_block=self._file_context_block,
            brief=req.brief,
            total_timeout_sec=self._stage_budget_sec(req),
            is_cancelled=is_cancelled,
            resume_of=req.resume_of or None,
        ):
            kind = event.get("kind")
            payload = event.get("payload") or {}
            if kind == "keepalive":
                # SSE 専用のキープアライブは on_step へ出さない (無意味な
                # task_progress running フレームを増やさないため)。
                continue
            if kind == "result":
                result_payload = payload
                break
            await on_event(ProductionEvent(kind=str(kind or "step"), payload=payload))

        if result_payload is None:
            # run_staged_pipeline は必ず終端 "result" を yield する契約
            # (タスクグラフ空でも yield して return する)。防御的フォールバック。
            logger.error("staged pipeline ended without a result event")
            return ProductionResult(
                exit_kind="error", notes={"reason": "no_result_event"},
            )

        artifacts = [
            EditorArtifact(
                content=content, language=_language_for_path(path), filename=path,
            )
            for path, content in (result_payload.get("code_map") or {}).items()
        ]
        spec_md = result_payload.get("spec_md")
        if spec_md:
            artifacts.append(
                EditorArtifact(content=spec_md, language="markdown", filename="SPEC.md"),
            )
        flowchart_md = result_payload.get("flowchart_md")
        if flowchart_md and flowchart_md.strip():
            fc_doc = f"# 設計フローチャート\n\n```mermaid\n{flowchart_md.strip()}\n```\n"
            artifacts.append(
                EditorArtifact(content=fc_doc, language="markdown", filename="flowchart.md"),
            )

        return ProductionResult(
            artifacts=artifacts,
            exit_kind=str(result_payload.get("exit_kind", "done")),
            metrics=dict(result_payload.get("metrics") or {}),
            notes=dict(result_payload.get("notes") or {}),
            truncated_steps=tuple(result_payload.get("truncated_steps") or ()),
            truncated_max_tokens=result_payload.get("truncated_max_tokens"),
        )

    @staticmethod
    def _stage_budget_sec(req: ProductionRequest) -> float | None:
        """``req.deadline_monotonic`` をステージ予算の上限にする (f_03 §4.4)。

        ``total_timeout_sec ≤ deadline − now − 300``、床 120s。締切未指定
        (0) のときは ``None`` を返し、パイプライン既定 (legacy と同じ
        ``create.staged.total_timeout_sec`` クランプ計算) に委ねる。
        """
        if not req.deadline_monotonic:
            return None
        remaining = req.deadline_monotonic - time.monotonic() - 300.0
        return max(120.0, remaining)
