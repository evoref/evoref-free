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
    "md": "markdown", "html": "html", "htm": "html", "css": "css", "scss": "scss", "sh": "bash",
    "mjs": "javascript", "sql": "sql", "php": "php",
}


def staged_profile(cfg: dict) -> str:
    """``create.staged.profile`` (``v2`` = 既定 / ``v1`` = 旧経路、Pro だけ、f_10 §11)。"""
    staged_cfg = (cfg.get("create", {}) or {}).get("staged", {}) or {}
    return str(staged_cfg.get("profile", "v2")).strip().lower() or "v2"


def _language_for_path(path: str) -> str:
    """拡張子から editor 表示用の言語ラベルを引く (best-effort)。"""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    # 未知の拡張子は拡張子そのもの (Python と表示しない)。拡張子無しは従来どおり Python
    return _EXT_LANG.get(ext, ext or "python")


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
        client=None,
        prefetched_rag: list[tuple[str, float, str]] | None = None,
        prefetched_rag_top_score: float | None = None,
        file_context_block: str | None = None,
    ) -> None:
        self._state = state
        self._cfg = cfg
        self._codegen = codegen
        self._part_codegen = part_codegen
        self._client = client
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
        from backend.edition import is_pro
        from backend.free.api.chat.chat_stream_staged import run_staged_pipeline

        if staged_profile(self._cfg) == "v2" and self._client is not None:
            # staged v2 (f_10 §11): 骨組み → 同時生成 → smoke → (Pro) 契約テスト → as-built 文書
            from backend.free.api.chat.chat_stream_staged_v2 import (
                UNSUPPORTED_LANGUAGE_FALLBACK,
                run_staged_v2_pipeline,
            )

            staged_cfg = (self._cfg.get("create", {}) or {}).get("staged", {}) or {}
            result_payload = await self._drain(run_staged_v2_pipeline(
                query=req.instruction,
                session_id=req.session_id,
                state=self._state,
                cfg=self._cfg,
                output_target=req.output_target,
                client=self._client,
                brief=req.brief,
                total_timeout_sec=self._stage_budget_sec(req),
                is_cancelled=is_cancelled,
                resume_of=req.resume_of or None,
                # テスト工程は Pro だけ (Free は smoke と as-built 文書まで)
                tests_enabled=is_pro() and bool(staged_cfg.get("test_stage_enabled", True)),
            ), on_event)
            if (
                result_payload is not None
                and (result_payload.get("notes") or {}).get("fallback") == UNSUPPORTED_LANGUAGE_FALLBACK
            ):
                # 第 1 段の対象外の言語・系統は変更前の経路へ (f_10 §12.1):
                # Pro は同じターンで v1、Free は呼出し側 (制作ステージ選択) が longform へ回す
                if not is_pro():
                    return ProductionResult(
                        exit_kind="error", notes=dict(result_payload.get("notes") or {}),
                    )
                result_payload = None
            else:
                return self._to_result(result_payload)
        result_payload = await self._drain(run_staged_pipeline(
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
        ), on_event)
        return self._to_result(result_payload)

    @staticmethod
    async def _drain(events, on_event) -> dict | None:
        """パイプラインのイベントを on_event へ流し、終端 ``result`` の payload を返す。"""
        async for event in events:
            kind = event.get("kind")
            payload = event.get("payload") or {}
            if kind == "keepalive":
                # SSE 専用のキープアライブは on_step へ出さない (無意味な
                # task_progress running フレームを増やさないため)。
                continue
            if kind == "result":
                return payload
            await on_event(ProductionEvent(kind=str(kind or "step"), payload=payload))
        return None

    @staticmethod
    def _to_result(result_payload: dict | None) -> ProductionResult:
        """終端 payload を :class:`ProductionResult` にする。"""
        if result_payload is None:
            # run_staged_pipeline は必ず終端 "result" を yield する契約
            # (タスクグラフ空でも yield して return する)。防御的フォールバック。
            logger.error("staged pipeline ended without a result event")
            return ProductionResult(
                exit_kind="error", notes={"reason": "no_result_event"},
            )

        # v2 は依頼が名指したフォルダ (``todo_app/``) を出力先の接頭辞として返す
        # (作業フォルダでは平置きで import が解決する)。
        folder = str((result_payload.get("notes") or {}).get("output_folder") or "").strip("/")

        def _out(name: str) -> str:
            return f"{folder}/{name}" if folder else name

        artifacts = [
            EditorArtifact(
                content=content, language=_language_for_path(path), filename=_out(path),
            )
            for path, content in (result_payload.get("code_map") or {}).items()
        ]
        spec_md = result_payload.get("spec_md")
        if spec_md:
            artifacts.append(
                EditorArtifact(content=spec_md, language="markdown", filename=_out("SPEC.md")),
            )
        flowchart_md = result_payload.get("flowchart_md")
        if flowchart_md and flowchart_md.strip():
            fc_doc = f"# 設計フローチャート\n\n```mermaid\n{flowchart_md.strip()}\n```\n"
            artifacts.append(
                EditorArtifact(content=fc_doc, language="markdown", filename=_out("flowchart.md")),
            )

        return ProductionResult(
            artifacts=artifacts,
            exit_kind=str(result_payload.get("exit_kind", "done")),
            metrics=dict(result_payload.get("metrics") or {}),
            notes={
                **dict(result_payload.get("notes") or {}),
                "tasks_failed": int(result_payload.get("tasks_failed") or 0),
                "code_files": len(result_payload.get("code_map") or {}),
            },
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
