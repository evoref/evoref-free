"""Meta-Cognitive エージェント: 計画立案 + 多段推論 + ツールループ（10〜30秒）"""

from __future__ import annotations

import asyncio
from pathlib import Path

from backend.free.agent.agent_state import AgentState
from backend.free.agent.event_reminder import EventReminderSystem
from backend.free.agent.self_cartridge import (
    AgentConstants,
    contains_self_reference,
    expand_self_references,
    gather_constants,
)
from backend.free.agent.meta_cognitive_tasks import (
    MetaCognitiveResponse,
    TaskItem,
    determine_task_status,
    merge_same_file_tasks,
    task_expects_write,
)
from backend.free.agent.meta_cognitive_tools import (
    infer_tool_from_task,
    normalize_read_file_args,
    normalize_write_file_args,
)
from backend.free.agent.meta_cognitive_utils import (
    iter_balanced_brace_substrings,
    try_parse_tool_dict,
    call_callback,
    fix_json_backslashes,
    looks_like_path_not_content,
    strip_markdown_wrapper,
    summarize_file_content,
    summarize_tool_args,
    text_looks_like_code,
    truncate_repetition,
)
from backend.free.agent.step_compactor import StepCompactor, StepResult
from backend.free.core.inference import build_messages_for_loop
from backend.utils import estimate_tokens as _estimate_tokens
from backend.log_config import get_logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.agent_tracer import AgentTracer
    from backend.free.agent.tool_call_judge import ToolCallJudge
    from backend.free.core.policy_interpreter import PolicyInterpreter
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.memory.views.loop import LoopFactView

logger = get_logger("agent.meta_cognitive")

# 公開シンボル（後方互換）
__all__ = [
    "MetaCognitiveAgent",
    "MetaCognitiveResponse",
    "TaskItem",
]


# ---------------------------------------------------------------------------
# システムプロンプト
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """\
You are a task planning assistant. Given a user request, break it down into \
a list of concrete steps.

Output a JSON object with a single field "tasks", which is an array of strings — \
each entry being one task description.

IMPORTANT rules:
- Creating or rewriting a SINGLE file is always ONE task, not multiple tasks.
  BAD:  {"tasks": ["Create grid logic", "Add rotation", "Add input handling"]}
  GOOD: {"tasks": ["Create e:\\\\app\\\\tetris.py with full game implementation"]}
- Only split into multiple tasks when genuinely different files or operations are needed.
- Each task should have a SINGLE action type. Do NOT combine "fetch/read" and "write/create" \
in one task.
  BAD:  {"tasks": ["Fetch URL and create script"]}
  GOOD: {"tasks": ["Fetch URL content", "Create script from fetched content"]}
- For information-only requests (explain, summarize, list, show), use fetch_url or read_file \
as the task — do NOT plan to create files.
  BAD:  {"tasks": ["Create script to scrape website"]}
  GOOD: {"tasks": ["Fetch URL and summarize the content"]}
- Each task should be self-contained and produce a concrete result.

Example: {"tasks": ["Read foo.py", "Create bar.py with refactored code", "Run tests"]}
Output the JSON object and nothing else."""

EXECUTE_SYSTEM_PROMPT = """\
You are a coding assistant executing a specific task.
Available tools (call by outputting JSON):
{tool_descriptions}

To call a tool, output ONLY a JSON object: {{"tool": "tool_name", "args": {{...}}}}
Do NOT include any text before or after the JSON.
To provide a final answer (no tool call), output plain text.

Tool argument formats:
- write_file: {{"tool": "write_file", "args": {{"file_path": "path"}}}}
  Do NOT include "content" in args. The system will generate it separately.
  Parent directories are created automatically. Do NOT run mkdir before write_file.
- read_file:  {{"tool": "read_file", "args": {{"file_path": "path"}}}}
  Always include file_path.

Current task: {task}
Context from previous steps:
{context}"""

CONTENT_GENERATION_PROMPT = """\
Generate the requested content below. Output ONLY the content itself, \
no explanations, no markdown fences, no surrounding text. \
Do NOT include the file path as a comment at the top.
"""


class MetaCognitiveAgent:
    """Meta-Cognitive 層: 計画立案 + 多段推論 + ツール実行ループ

    コーディングモードのタスクリスト方式 + diff 駆動を実装。
    StepCompactor によるステップ結果圧縮と EventReminderSystem による
    Instruction Fade-Out 対策を統合。
    目標応答時間: 10〜30秒
    """

    def __init__(
        self,
        max_steps: int = 10,
        config: dict | None = None,
        tool_judge: ToolCallJudge | None = None,
        policy: PolicyInterpreter | None = None,
        agent_tracer: AgentTracer | None = None,
        loop_view: "LoopFactView | None" = None,
        project_id: str | None = None,
        assist_client: "AssistModelClient | None" = None,
        debug_logger: "DebugLogger | None" = None,
        semmem_block: str | None = None,
    ) -> None:
        self.max_steps = max_steps
        cfg = config or {}
        self.config = cfg
        # ツールループ用 system に注入する SemMem メモリブロック
        # (チャット初回ターンと同じ MemoryInjector 出力。reactive 以外の
        #  全ターンで policy / failure_pattern facts を維持するため)
        self._semmem_block = semmem_block
        self.compactor = StepCompactor(cfg, policy=policy)
        self.reminder_system = EventReminderSystem(cfg)
        self._tool_judge = tool_judge
        self._agent_tracer = agent_tracer
        # に従いアシストモデルで実行する。``None`` の場合 (assist_client
        # health_check 失敗による degraded mode) は計画生成をスキップして
        # 単一タスクへフォールバックする。
        self._assist_client = assist_client
        # に記録する (decision_point=``meta_cognitive_llm_route``)。
        self._debug_logger = debug_logger

        agent_cfg = cfg.get("agent", {})
        self.max_tool_iterations = agent_cfg.get("max_tool_iterations", 5)

        ctx_size = cfg.get("llama", {}).get("context_size", 4096)
        history_budget = cfg.get("memory", {}).get("working_max_tokens", 2048)
        self.loop_budget = ctx_size - 512 - 400 - history_budget

        self._execute_max_tokens = max(ctx_size - 512, 1024)
        self._reminder_budget = 100

        self._content_gen_timeout = agent_cfg.get("content_gen_timeout", 60)
        self._llm_call_timeout = agent_cfg.get("llm_call_timeout", 90)
        self._total_timeout = agent_cfg.get("total_timeout", 180)

        # ── EvorefMem: @self 仮想カートリッジ ──
        # SemanticFactStore 直参照を廃止し LoopFactView 経由に統一
        self._loop_view = loop_view
        self._project_id = project_id
        learning_policy_cfg = (cfg.get("learning") or {}).get("policy") or {}
        self._policy_activation_min_confidence = float(
            learning_policy_cfg.get("activation_min_confidence", 0.7),
        )
        self._constants_cache: AgentConstants | None = None

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    async def process(
        self,
        query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        tools_registry=None,
        search_result=None,
        on_step=None,
        generation_params: dict | None = None,
        session_id: str = "",
        mode: str = "coding",
    ) -> MetaCognitiveResponse:
        """Meta-Cognitive 層で計画立案 → タスク実行ループ

        全体をタイムアウトで保護し、失敗時はプレーン LLM にフォールバック。
        """
        try:
            return await asyncio.wait_for(
                self._process_impl(
                    query, system_prompt, conversation, llm_client,
                    tools_registry, search_result, on_step,
                    generation_params=generation_params,
                    session_id=session_id,
                    mode=mode,
                ),
                timeout=self._total_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "MetaCognitive process timed out after %ds, falling back",
                self._total_timeout,
            )
            if on_step:
                await call_callback(on_step, {
                    "type": "task_progress",
                    "detail": "タイムアウト — フォールバック応答を生成中...",
                    "status": "running",
                })
            return await self._fallback_plain_llm(
                query, system_prompt, conversation, llm_client, on_step,
            )
        except Exception as e:
            logger.error(
                "MetaCognitive process failed unexpectedly, falling back: %s", e,
            )
            if on_step:
                await call_callback(on_step, {
                    "type": "task_progress",
                    "detail": "エラー発生 — フォールバック応答を生成中...",
                    "status": "running",
                })
            return await self._fallback_plain_llm(
                query, system_prompt, conversation, llm_client, on_step,
            )

    # ------------------------------------------------------------------
    # @self 仮想カートリッジ
    # ------------------------------------------------------------------

    def get_constants(self, *, refresh: bool = False) -> AgentConstants:
        """SemMem から ``AgentConstants`` を取得する (process 1 回中はキャッシュ)。

        以下を集約する:
        - ユーザープロファイル (``personal_fact`` / ``preference``)
        - active な ``policy`` ファクト (``confidence >= 閾値`` / 非 superseded)
        - pinned ファクト (global + project)

        SemMem 未配線時は :data:`EMPTY_CONSTANTS` を返す。

        Args:
            refresh: ``True`` の場合キャッシュを無視して再収集する
                (テスト・ホットリロード用)
        """
        if not refresh and self._constants_cache is not None:
            return self._constants_cache
        constants = gather_constants(
            view=self._loop_view,
            project_id=self._project_id,
            policy_activation_min_confidence=(
                self._policy_activation_min_confidence
            ),
        )
        self._constants_cache = constants
        return constants

    def _expand_self_in_inputs(
        self, query: str, system_prompt: str,
    ) -> tuple[str, str, AgentConstants | None]:
        """``query`` / ``system_prompt`` の ``@self`` を展開する。

        どちらにも ``@self`` が含まれない場合は SemMem アクセス自体を
        スキップして元の文字列を返す (高速パス)。
        """
        if not (
            contains_self_reference(query)
            or contains_self_reference(system_prompt)
        ):
            return query, system_prompt, None
        constants = self.get_constants()
        new_query = expand_self_references(query, constants)
        new_system_prompt = expand_self_references(system_prompt, constants)
        logger.debug(
            "@self expanded: %s", constants.as_summary(),
        )
        return new_query, new_system_prompt, constants

    # ------------------------------------------------------------------
    # 内部: メインフロー
    # ------------------------------------------------------------------

    async def _process_impl(
        self,
        query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        tools_registry=None,
        search_result=None,
        on_step=None,
        *,
        generation_params: dict | None = None,
        session_id: str = "",
        mode: str = "coding",
    ) -> MetaCognitiveResponse:
        """process() の本体実装"""
        from backend.free.agent.credit_assigner import assign_credit, compute_final_outcome

        all_tool_calls: list[dict] = []

        # MDP トレース: エピソード開始
        tracer = self._agent_tracer
        episode_id = ""
        if tracer is not None:
            episode_id = tracer.begin_episode(session_id, mode)

        # @self 仮想カートリッジ展開
        # constants キャッシュは process 呼び出しごとにリセット
        self._constants_cache = None
        query, system_prompt, _ = self._expand_self_in_inputs(query, system_prompt)

        # Step 1: タスク計画を生成
        tasks = await self._plan(query, conversation, llm_client, on_step)
        if not tasks:
            tasks = [TaskItem(description=query)]

        # 同一ファイル対象のタスクをマージ
        tasks = merge_same_file_tasks(tasks)
        logger.info("Plan generated: %d tasks", len(tasks))

        if on_step:
            task_names = " / ".join(t.description[:50] for t in tasks)
            await call_callback(on_step, {
                "type": "plan",
                "detail": f"{len(tasks)} タスク: {task_names}",
                "status": "done",
            })

        # Step 2: タスクを順番に実行
        context_parts: list[str] = []
        steps = await self._run_tasks(
            tasks, query, system_prompt, conversation,
            llm_client, tools_registry, context_parts,
            all_tool_calls, on_step, generation_params,
            episode_id=episode_id,
        )

        # Step 2.5: 失敗した書き込みタスクを1回リトライ
        retry_enabled = self.config.get("agent", {}).get(
            "retry_failed_writes", True,
        )
        if retry_enabled:
            steps = await self._retry_failed_writes(
                tasks, query, system_prompt, conversation,
                llm_client, tools_registry, context_parts,
                all_tool_calls, steps, len(tasks), on_step,
                generation_params=generation_params,
            )

        # Step 3: 最終応答を組み立て
        content = self._build_final_response(tasks, context_parts)
        logger.info("MetaCognitive completed: %d steps, %d tool calls",
                     steps, len(all_tool_calls))

        # MDP トレース: エピソード終了 + クレジット割当
        step_credits = []
        if tracer is not None and episode_id:
            tasks_done = sum(1 for t in tasks if t.status == "done")
            tasks_failed = sum(1 for t in tasks if t.status == "failed")
            outcome_label = "success" if tasks_failed == 0 else "partial"
            tracer.end_episode(episode_id, outcome_label)

            final_outcome = compute_final_outcome(
                len(tasks), tasks_done, tasks_failed,
            )
            mdp_steps = tracer.get_steps(episode_id)
            step_credits = assign_credit(mdp_steps, final_outcome)

            tracer.cleanup_episode(episode_id)

        return MetaCognitiveResponse(
            content=content,
            tasks=tasks,
            tool_calls=all_tool_calls,
            steps=steps,
            episode_id=episode_id,
            step_credits=step_credits,
        )

    async def _plan(
        self,
        query: str,
        conversation: list[dict],
        llm_client,
        on_step=None,
    ) -> list[TaskItem]:
        """アシストモデルにタスク計画を生成させる

        CLAUDE.md §1 に従い、判定 / 計画系の JSON 応答処理はアシスト
        モデル (``assist_client.generate_json``) で実行する。``llm_client``
        引数はベースモデル (Meta-Cognitive のメイン応答 / ツール呼び出し
        ループ用) の参照を保つため受け取り続けるが、計画生成自体には
        利用しない。

        ``assist_client`` が ``None`` (health_check 失敗で degraded mode)
        の場合は空リストを返し、呼び出し側で単一タスクフォールバックする。
        """
        if on_step:
            await call_callback(on_step, {
                "type": "plan",
                "detail": "タスク計画を生成中...",
                "status": "running",
            })

        if self._assist_client is None:
            logger.warning(
                "Plan generation skipped: assist_client is not configured "
                "(degraded mode); falling back to single task",
            )
            if self._debug_logger is not None:
                self._debug_logger.log_decision(
                    decision_point="meta_cognitive_llm_route",
                    chosen="single_task_fallback",
                    candidates=["base_model", "assist_model", "single_task_fallback"],
                    reason="assist_client_unavailable",
                    scope="request",
                )
            return []

        if self._debug_logger is not None:
            self._debug_logger.log_decision(
                decision_point="meta_cognitive_llm_route",
                chosen="assist_plan",
                candidates=["base_model", "assist_model", "single_task_fallback"],
                reason="assist_client_available",
                scope="request",
            )

        user_content = self._build_plan_user_content(query, conversation)
        prompt = f"{PLAN_SYSTEM_PROMPT}\n\n{user_content}"

        try:
            data = await asyncio.wait_for(
                self._assist_client.generate_json(
                    prompt,
                    max_tokens=256,
                    temperature=0.3,
                    purpose="meta_cognitive_plan",
                    list_key="tasks",
                ),
                timeout=self._llm_call_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Plan generation timed out after %ds", self._llm_call_timeout,
            )
            return []
        except Exception as e:  # noqa: BLE001 — assist 側の任意例外で fallback
            logger.warning("Plan generation failed: %s", e)
            return []

        tasks_raw = data.get("tasks") if isinstance(data, dict) else None
        if isinstance(tasks_raw, list):
            return [TaskItem(description=str(t)) for t in tasks_raw if t]

        logger.warning(
            "Plan response did not contain a 'tasks' list: %r", data,
        )
        return []

    @staticmethod
    def _build_plan_user_content(
        query: str, conversation: list[dict],
    ) -> str:
        """計画生成用のユーザーメッセージを構築する"""
        if not conversation:
            return query
        recent_lines: list[str] = []
        for msg in conversation[-4:]:
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]
            if role in ("user", "assistant"):
                recent_lines.append(f"{role}: {content}")
        if not recent_lines:
            return query
        return (
            "Recent conversation:\n"
            + "\n".join(recent_lines)
            + "\n\nCurrent request: "
            + query
        )

    async def _run_tasks(
        self,
        tasks: list[TaskItem],
        query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        tools_registry,
        context_parts: list[str],
        all_tool_calls: list[dict],
        on_step,
        generation_params: dict | None,
        *,
        episode_id: str = "",
    ) -> int:
        """タスクリストを順次実行し、ステップ数を返す"""
        import time as _time

        from backend.free.agent.agent_tracer import MDPStep

        steps = 0
        total_tasks = len(tasks)

        for i, task in enumerate(tasks):
            if steps >= self.max_steps:
                task.status = "failed"
                task.result = "Step limit reached"
                logger.warning("Step limit reached at task: %s", task.description[:50])
                break

            if on_step:
                await call_callback(on_step, {
                    "type": "task_progress",
                    "detail": f"[{i + 1}/{total_tasks}] {task.description[:150]}",
                    "status": "running",
                })

            result, tool_calls = await self._execute_task(
                task, query, system_prompt, conversation,
                llm_client, tools_registry, context_parts,
                on_step=on_step,
                task_index=i + 1,
                total_tasks=total_tasks,
                generation_params=generation_params,
            )
            task.result = result
            task.status = determine_task_status(task, result, tool_calls)

            all_tool_calls.extend(tool_calls)
            context_parts.append(f"[{task.description}]: {result[:200]}")

            self._append_write_context(tool_calls, context_parts)
            steps += 1

            # MDP トレース: ステップ記録
            tracer = self._agent_tracer
            if tracer is not None and episode_id:
                tool_names = ", ".join(
                    tc.get("tool", "") for tc in tool_calls
                ) if tool_calls else "none"
                reward = 1.0 if task.status == "done" else 0.0
                tracer.record_step(episode_id, MDPStep(
                    step_index=i,
                    state={
                        "task": task.description[:200],
                        "task_index": i + 1,
                        "total_tasks": total_tasks,
                    },
                    action=tool_names,
                    observation=result[:200],
                    reward=reward,
                    timestamp=_time.time(),
                ))

            if on_step:
                await self._notify_task_done(
                    on_step, i, total_tasks, task, tool_calls,
                )

        return steps

    @staticmethod
    def _append_write_context(
        tool_calls: list[dict], context_parts: list[str],
    ) -> None:
        """write_file 成功時にファイル内容サマリをコンテキストに追加"""
        for tc in tool_calls:
            if tc.get("tool") == "write_file" and tc.get("success"):
                content = tc.get("args", {}).get("content", "")
                if content:
                    file_path = tc.get("args", {}).get("file_path", "")
                    summary = summarize_file_content(content)
                    context_parts.append(
                        f"[Content of {file_path}]:\n{summary}"
                    )

    @staticmethod
    async def _notify_task_done(
        on_step, i: int, total_tasks: int,
        task: TaskItem, tool_calls: list[dict],
    ) -> None:
        """タスク完了コールバックを送信"""
        if tool_calls:
            tool_names = ", ".join(tc.get("tool", "") for tc in tool_calls)
            detail = f"[{i + 1}/{total_tasks}] {tool_names}"
        else:
            detail = f"[{i + 1}/{total_tasks}] {task.description[:150]}"
        await call_callback(on_step, {
            "type": "tool_call" if tool_calls else "task_progress",
            "detail": detail,
            "status": "done" if task.status == "done" else "failed",
        })

    async def _retry_failed_writes(
        self,
        tasks: list[TaskItem],
        query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        tools_registry,
        context_parts: list[str],
        all_tool_calls: list[dict],
        steps: int,
        total_tasks: int,
        on_step,
        generation_params: dict | None = None,
    ) -> int:
        """書き込み期待タスクの失敗時リトライ（最大2件、各1回のみ）"""
        failed_writes = [
            (i, task) for i, task in enumerate(tasks)
            if task.status == "failed"
            and task_expects_write(task.description)
        ]
        if not failed_writes:
            return steps

        max_retries = min(2, self.max_steps - steps)
        for idx, (i, task) in enumerate(failed_writes[:max_retries]):
            if task.result and "Content generation failed" in task.result:
                logger.info(
                    "Skipping retry for content generation failure: %s",
                    task.description[:80],
                )
                continue
            logger.info("Retrying failed write task [%d]: %s",
                        idx + 1, task.description[:80])

            enriched_context = self._enrich_context_for_retry(
                task, context_parts,
            )

            if on_step:
                await call_callback(on_step, {
                    "type": "task_progress",
                    "detail": f"[retry] {task.description[:120]}",
                    "status": "running",
                })

            task.status = "pending"
            result, tool_calls = await self._execute_task(
                task, query, system_prompt, conversation,
                llm_client, tools_registry, enriched_context,
                on_step=on_step,
                task_index=i + 1,
                total_tasks=total_tasks,
                generation_params=generation_params,
            )
            task.result = result
            task.status = determine_task_status(task, result, tool_calls)
            all_tool_calls.extend(tool_calls)
            steps += 1

            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call" if tool_calls else "task_progress",
                    "detail": f"[retry] {task.description[:120]}",
                    "status": "done" if task.status == "done" else "failed",
                })

        return steps

    @staticmethod
    def _enrich_context_for_retry(
        task: TaskItem, context_parts: list[str],
    ) -> list[str]:
        """リトライ用にコンテキストを拡充する"""
        enriched = list(context_parts)
        from backend.free.agent.tool_call_judge import _extract_file_path
        file_path = _extract_file_path(task.description)
        if file_path:
            p = Path(file_path)
            if p.exists() and p.is_file():
                try:
                    existing = p.read_text(encoding="utf-8")
                    summary = summarize_file_content(existing)
                    enriched.append(
                        f"[Existing content of {file_path}]:\n{summary}"
                    )
                except Exception:
                    pass
        return enriched

    # ------------------------------------------------------------------
    # タスク実行
    # ------------------------------------------------------------------

    async def _execute_task(
        self,
        task: TaskItem,
        original_query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        tools_registry,
        context_parts: list[str],
        on_step=None,
        task_index: int = 1,
        total_tasks: int = 1,
        generation_params: dict | None = None,
    ) -> tuple[str, list[dict]]:
        """1つのタスクを実行（ツールループ付き）"""
        prefix = f"[{task_index}/{total_tasks}]"

        # ── ファストパス ──
        if tools_registry is not None:
            fast_result = await self._try_fast_path(
                task, original_query, tools_registry, llm_client,
                on_step, prefix,
            )
            if fast_result is not None:
                return fast_result

        # ── 通常パス（ツールループ） ──
        return await self._run_tool_loop(
            task, original_query, system_prompt, conversation,
            llm_client, tools_registry, context_parts,
            on_step, prefix, generation_params,
        )

    async def _try_fast_path(
        self,
        task: TaskItem,
        original_query: str,
        tools_registry,
        llm_client,
        on_step,
        prefix: str,
    ) -> tuple[str, list[dict]] | None:
        """ファストパス判定・実行。適用不可なら None を返す"""
        judgement = await self._judge_tool_for_task(
            task.description, tools_registry,
        )
        if judgement is None or not judgement.tool_needed or not judgement.tool_name:
            return None

        if (
            judgement.tool_name == "write_file"
            and tools_registry.has("write_file")
        ):
            file_path = judgement.tool_args.get("file_path", "")
            if file_path:
                return await self._execute_write_fast(
                    task, original_query, file_path,
                    llm_client, tools_registry,
                    on_step=on_step, prefix=prefix,
                )
        elif tools_registry.has(judgement.tool_name):
            # ツールの必須引数が揃っているか確認（不足時は通常ループに委譲）
            tool_def = tools_registry.get(judgement.tool_name)
            if tool_def and tool_def.parameters and not judgement.tool_args:
                logger.debug(
                    "Fast path skipped: %s requires args but none provided",
                    judgement.tool_name,
                )
                return None
            return await self._execute_tool_fast(
                judgement.tool_name, judgement.tool_args, task,
                tools_registry,
                on_step=on_step, prefix=prefix,
            )

        return None

    def _build_loop_system_prompt(
        self,
        task: TaskItem,
        context_parts: list[str],
        tools_registry,
    ) -> str:
        """ツールループ用 system プロンプトを構築する。

        コンテキストは 3000 文字で truncate し、ツール記述は registry から取得。
        """
        tool_descriptions = ""
        if tools_registry is not None:
            tool_descriptions = tools_registry.get_descriptions_text()
        context_text = "\n".join(context_parts) if context_parts else "(none)"
        if len(context_text) > 3000:
            context_text = context_text[:3000] + "\n... (truncated)"
        prompt = EXECUTE_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions or "(no tools available)",
            task=task.description,
            context=context_text,
        )
        # SemMem メモリをループ全反復の system に維持する
        if self._semmem_block:
            prompt = f"{prompt}\n\n[関連する記憶]\n{self._semmem_block}"
        return prompt

    def _rebuild_loop_messages(
        self,
        prompt: str,
        conversation: list[dict],
        step_results: list[StepResult],
        original_query: str,
        compact_budget: int,
    ) -> list[dict]:
        """直前 step_results を圧縮して loop 用 messages を再構築する。"""
        compacted = self.compactor.compact(step_results, compact_budget)
        messages = build_messages_for_loop(
            prompt, conversation, compacted, self.config,
        )
        last = step_results[-1]
        last_context = (
            f"Your last action: {last.tool_name} → "
            f"{last.output[:200]}\n\n"
        )
        messages.append({
            "role": "user",
            "content": last_context + original_query,
        })
        return messages

    async def _call_llm_in_loop(
        self,
        llm_client,
        injected_messages: list[dict],
        gen_kwargs: dict,
        loop: int,
    ) -> tuple[str, bool]:
        """LLM 呼び出しを timeout 付きで実行。`(text, timed_out)` を返す。"""
        try:
            result = await asyncio.wait_for(
                llm_client.generate(injected_messages, **gen_kwargs),
                timeout=self._llm_call_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out in tool loop iteration %d", loop)
            return "", True
        text = result["choices"][0]["message"]["content"].strip()
        return text, False

    @staticmethod
    def _append_timeout_recovery_messages(messages: list[dict]) -> None:
        """LLM タイムアウト後にループ続行する際のリトライメッセージを追記する。"""
        messages.append({"role": "assistant", "content": "(timeout)"})
        messages.append({
            "role": "user",
            "content": "Previous call timed out. Respond concisely.",
        })

    async def _try_recover_no_tool_call(
        self,
        text: str,
        task: TaskItem,
        original_query: str,
        llm_client,
        tools_registry,
        on_step,
        prefix: str,
        tool_calls: list[dict],
    ) -> tuple[str, list[dict]] | None:
        """`_parse_tool_call` が None のとき write_file 期待タスクならテキスト→ツール変換を試みる。

        成功時は `tool_calls` を mutate して `(result_text, tool_calls)` を返す。
        対象外 / 失敗時は `None` を返し、呼び出し側はテキストのまま終了する。
        """
        if not (
            task_expects_write(task.description)
            and tools_registry is not None
            and tools_registry.has("write_file")
        ):
            return None
        recovery = await self._recover_write_from_text(
            text, task, original_query, llm_client,
            tools_registry, on_step, prefix,
        )
        if recovery is None:
            return None
        tool_calls.append(recovery)
        return recovery.get("result", text), tool_calls

    @staticmethod
    def _next_consecutive_errors(
        step_results: list[StepResult],
        current: int,
    ) -> int:
        """直近 step_results に応じて consecutive_errors を更新する。"""
        last_step = step_results[-1] if step_results else None
        if last_step and last_step.output.startswith("Error:"):
            return current + 1
        return 0

    async def _run_tool_loop(
        self,
        task: TaskItem,
        original_query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        tools_registry,
        context_parts: list[str],
        on_step,
        prefix: str,
        generation_params: dict | None,
    ) -> tuple[str, list[dict]]:
        """ツールループ本体: LLM 推論 → ツール実行を繰り返す"""
        tool_calls: list[dict] = []
        prompt = self._build_loop_system_prompt(task, context_parts, tools_registry)

        state = AgentState(
            agent_layer="meta_cognitive",
            max_iterations=self.max_tool_iterations,
            expected_format="json",
        )

        step_results: list[StepResult] = []
        consecutive_errors = 0
        max_consecutive_errors = 3
        compact_budget = self.loop_budget - self._reminder_budget

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": original_query},
        ]

        for loop in range(self.max_tool_iterations):
            state.current_iteration = loop

            if step_results:
                messages = self._rebuild_loop_messages(
                    prompt, conversation, step_results,
                    original_query, compact_budget,
                )

            if on_step:
                await call_callback(on_step, {
                    "type": "llm",
                    "detail": f"{prefix} LLM 推論中... (ループ {loop + 1})",
                    "status": "running",
                })

            self._update_context_usage(state, messages)
            injected_messages = self.reminder_system.inject(messages, state)
            gen_kwargs = self._build_gen_kwargs(generation_params)

            text, timed_out = await self._call_llm_in_loop(
                llm_client, injected_messages, gen_kwargs, loop,
            )
            if timed_out:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    return "Error: LLM call timed out repeatedly", tool_calls
                self._append_timeout_recovery_messages(messages)
                continue
            state.last_output = text

            tool_call = self._parse_tool_call(text)
            if tool_call is None:
                recovered = await self._try_recover_no_tool_call(
                    text, task, original_query, llm_client,
                    tools_registry, on_step, prefix, tool_calls,
                )
                if recovered is not None:
                    return recovered
                return text, tool_calls

            loop_result = await self._execute_loop_tool_call(
                tool_call, text, task, original_query,
                llm_client, tools_registry, on_step, prefix,
                state, tool_calls, step_results, messages,
                consecutive_errors, max_consecutive_errors,
                loop, generation_params,
            )
            if loop_result is not None:
                return loop_result

            consecutive_errors = self._next_consecutive_errors(
                step_results, consecutive_errors,
            )

        return "Step limit reached during task execution.", tool_calls

    def _update_context_usage(
        self, state: AgentState, messages: list[dict],
    ) -> None:
        """コンテキスト使用率を計算して AgentState に設定"""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        ctx_size = self.config.get("llama", {}).get("context_size", 4096)
        state.context_usage_pct = min(
            100,
            int(_estimate_tokens("x" * total_chars) / ctx_size * 100),
        )

    def _build_gen_kwargs(
        self, generation_params: dict | None,
    ) -> dict:
        """LLM 生成パラメータを組み立てる"""
        gen_kwargs: dict = {
            "stream": False,
            "max_tokens": self._execute_max_tokens,
            "id_slot": -1,
        }
        if generation_params:
            for k in ("temperature", "top_p", "top_k", "presence_penalty"):
                if k in generation_params:
                    gen_kwargs[k] = generation_params[k]
        return gen_kwargs

    @staticmethod
    def _normalize_loop_tool_args(
        tool_name: str, tool_args: dict, original_query: str,
    ) -> dict:
        """`write_file` / `read_file` の args を正規化する。それ以外は素通し。"""
        if tool_name == "write_file":
            return normalize_write_file_args(tool_args)
        if tool_name == "read_file":
            return normalize_read_file_args(tool_args, original_query)
        return tool_args

    async def _emit_loop_tool_running(
        self,
        on_step,
        prefix: str,
        tool_name: str,
        tool_args: dict,
    ) -> None:
        """ツール実行開始 (`status=running`) のコールバック emit。"""
        if on_step is None:
            return
        args_summary = summarize_tool_args(tool_name, tool_args)
        await call_callback(on_step, {
            "type": "tool_call",
            "detail": f"{prefix} {tool_name}({args_summary})",
            "status": "running",
        })

    async def _emit_loop_tool_result(
        self,
        on_step,
        prefix: str,
        tool_name: str,
        tool_result_text: str,
    ) -> None:
        """ツール実行結果 (`status=done|failed`) のコールバック emit。"""
        if on_step is None:
            return
        is_error = tool_result_text.startswith("Error:")
        await call_callback(on_step, {
            "type": "tool_call",
            "detail": f"{prefix} {tool_name}: {tool_result_text[:100]}",
            "status": "failed" if is_error else "done",
        })

    async def _execute_loop_tool_call(
        self,
        tool_call: dict,
        text: str,
        task: TaskItem,
        original_query: str,
        llm_client,
        tools_registry,
        on_step,
        prefix: str,
        state: AgentState,
        tool_calls: list[dict],
        step_results: list[StepResult],
        messages: list[dict],
        consecutive_errors: int,
        max_consecutive_errors: int,
        loop: int,
        generation_params: dict | None,
    ) -> tuple[str, list[dict]] | None:
        """ツールループ内の1回のツール実行

        ループ終了条件に達した場合は結果タプルを返す。
        ループ続行の場合は None を返し、messages を更新する。
        """
        tool_name = tool_call.get("tool", "")
        tool_args = self._normalize_loop_tool_args(
            tool_name, tool_call.get("args", {}), original_query,
        )

        tool_call_entry = {"tool": tool_name, "args": tool_args, "success": False}
        tool_calls.append(tool_call_entry)

        # write_file で content が不足 → 別途コンテンツ生成
        if tool_name == "write_file" and not tool_args.get("content"):
            gen_result = await self._generate_write_content(
                tool_name, tool_args, original_query, task,
                llm_client, on_step, prefix,
                tool_calls, step_results, messages,
                consecutive_errors, max_consecutive_errors, loop,
            )
            if gen_result is not None:
                return gen_result

        await self._emit_loop_tool_running(on_step, prefix, tool_name, tool_args)

        state.pending_tool = tool_name
        state.pending_args = tool_args

        tool_result_text = await self._execute_tool(
            tool_name, tool_args, tools_registry, state,
        )
        tool_call_entry["success"] = not tool_result_text.startswith("Error:")

        state.pending_tool = None
        state.pending_args = {}

        await self._emit_loop_tool_result(on_step, prefix, tool_name, tool_result_text)

        # 連続エラー検出 → ループ打ち切り
        if tool_result_text.startswith("Error:"):
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                logger.warning(
                    "Stopping loop: %d consecutive errors in task '%s'",
                    consecutive_errors, task.description[:50],
                )
                return tool_result_text, tool_calls
        else:
            consecutive_errors = 0

        step_results.append(StepResult(
            tool_name=tool_name,
            output=tool_result_text,
            iteration=loop,
        ))

        # write_file 成功 → タスク完了
        if not tool_result_text.startswith("Error:") and tool_name == "write_file":
            logger.info(
                "Stopping loop: write_file succeeded (%s)",
                tool_args.get("file_path", ""),
            )
            return tool_result_text, tool_calls

        # ツール結果をメッセージに追加して再ループ
        result_msg = self._build_tool_result_message(
            tool_name, tool_result_text, consecutive_errors,
        )
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": result_msg})
        logger.info("Tool call #%d: %s", loop + 1, tool_name)
        return None

    async def _generate_write_content(
        self,
        tool_name: str,
        tool_args: dict,
        original_query: str,
        task: TaskItem,
        llm_client,
        on_step,
        prefix: str,
        tool_calls: list[dict],
        step_results: list[StepResult],
        messages: list[dict],
        consecutive_errors: int,
        max_consecutive_errors: int,
        loop: int,
    ) -> tuple[str, list[dict]] | None:
        """write_file の content が不足時にコンテンツを生成する

        生成失敗時は結果タプルを返す（ループ終了）。
        成功時は tool_args["content"] を更新して None を返す（ループ続行）。
        """
        file_path = tool_args.get("file_path", "")
        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} コンテンツ生成中 → {file_path}",
                "status": "running",
            })
        content = await self._generate_content(
            original_query, task.description, llm_client,
            file_path=file_path,
        )
        if content.startswith("(Content generation failed:"):
            logger.warning(
                "Skipping write_file due to content generation failure: %s",
                file_path,
            )
            tool_result_text = f"Error: {content}"
            consecutive_errors += 1
            step_results.append(StepResult(
                tool_name=tool_name,
                output=tool_result_text,
                iteration=loop,
            ))
            if consecutive_errors >= max_consecutive_errors:
                return tool_result_text, tool_calls
            # ループ続行のためメッセージを追加（呼び出し元の text は使えない）
            return None
        tool_args["content"] = content
        logger.info(
            "Content generated for write_file: %d chars → %s",
            len(content), file_path,
        )
        return None

    @staticmethod
    async def _execute_tool(
        tool_name: str,
        tool_args: dict,
        tools_registry,
        state: AgentState,
    ) -> str:
        """ツールを実行して結果テキストを返す"""
        if tools_registry is not None and tools_registry.has(tool_name):
            try:
                tool_result = await tools_registry.execute(tool_name, **tool_args)
                result_text = str(tool_result)
                state.on_tool_success(tool_name)
            except Exception as e:
                result_text = f"Error: {e}"
                state.on_tool_failure(tool_name, str(e))
                logger.warning("Tool execution failed: %s - %s", tool_name, e)
        else:
            result_text = f"Error: Unknown tool '{tool_name}'"
            state.on_tool_failure(tool_name, result_text)
        return result_text

    @staticmethod
    def _build_tool_result_message(
        tool_name: str, tool_result_text: str, consecutive_errors: int,
    ) -> str:
        """ツール実行結果を次ループ用メッセージに変換する"""
        is_success = not tool_result_text.startswith("Error:")
        if is_success:
            return (
                f"Tool result ({tool_name}): {tool_result_text}"
                "\n\nThe tool executed successfully. "
                "If the current task is complete, respond with a plain text summary. "
                "Only call another tool if additional work is needed."
            )
        if "Missing required argument" in tool_result_text:
            return (
                f"Tool call FAILED: {tool_result_text}\n\n"
                "Fix the arguments and try again, or use a different tool. "
                "If no tool is needed, respond with plain text."
            )
        if consecutive_errors >= 2:
            return (
                f"Tool call FAILED ({consecutive_errors} consecutive errors): "
                f"{tool_result_text}\n\n"
                "IMPORTANT: Multiple failures occurred. "
                "Use a DIFFERENT tool or respond with plain text."
            )
        return (
            f"Tool call FAILED: {tool_result_text}\n"
            "Fix and retry, or use a different approach."
        )

    # ------------------------------------------------------------------
    # ツール判定
    # ------------------------------------------------------------------

    async def _judge_tool_for_task(
        self,
        task_description: str,
        tools_registry,
    ):
        """タスク記述に対してツール判定を行う"""
        if self._tool_judge is not None:
            try:
                judgement = await self._tool_judge.judge(
                    task_description, tools_registry, mode="coding",
                )
                if judgement.tool_needed and judgement.tool_name:
                    return judgement
            except Exception as e:
                logger.warning("ToolCallJudge failed for task, falling back: %s", e)

        inferred = infer_tool_from_task(task_description)
        if inferred is not None:
            from backend.free.agent.tool_call_judge import ToolJudgement
            return ToolJudgement(
                tool_needed=True,
                tool_name=inferred[0],
                tool_args=inferred[1],
                source="rule",
            )
        return None

    def _parse_tool_call(self, text: str) -> dict | None:
        """LLM 応答からツール呼び出し JSON をパース

        まず全文を fix_json_backslashes で前処理してから直接パースし、
        失敗した場合は埋め込みの ``{...}`` 候補を順に試す。
        旧実装はネスト 7 だったが、ヘルパー抽出 + ガード節で平坦化済み。
        """
        # 1. 全文を前処理してから直接パース
        direct = try_parse_tool_dict(fix_json_backslashes(text))
        if direct is not None:
            return direct

        # 2. 埋め込み JSON 候補を順に試す（前処理なし、互換性のため）
        for candidate in iter_balanced_brace_substrings(text):
            parsed = try_parse_tool_dict(candidate)
            if parsed is not None:
                return parsed

        return None

    # ------------------------------------------------------------------
    # ファストパス実行
    # ------------------------------------------------------------------

    async def _execute_tool_fast(
        self,
        tool_name: str,
        tool_args: dict,
        task: TaskItem,
        tools_registry,
        on_step=None,
        prefix: str = "",
    ) -> tuple[str, list[dict]]:
        """read_file / run_command / search_code のファストパス実行"""
        logger.info("Tool fast path: %s(%s)", tool_name, tool_args)

        if on_step:
            args_summary = summarize_tool_args(tool_name, tool_args)
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} {tool_name}({args_summary})",
                "status": "running",
            })

        try:
            result = await tools_registry.execute(tool_name, **tool_args)
            result_text = str(result)
            is_success = not result_text.startswith("Error:")
        except Exception as e:
            result_text = f"Error: {e}"
            is_success = False
            logger.error("Tool fast path failed: %s - %s", tool_name, e)

        tool_entry = {
            "tool": tool_name,
            "args": tool_args,
            "success": is_success,
        }

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} {tool_name}: {result_text[:100]}",
                "status": "done" if is_success else "failed",
            })

        logger.info("Tool fast path completed: %s → %s", tool_name, result_text[:80])
        return result_text, [tool_entry]

    async def _execute_write_fast(
        self,
        task: TaskItem,
        original_query: str,
        file_path: str,
        llm_client,
        tools_registry,
        on_step=None,
        prefix: str = "",
    ) -> tuple[str, list[dict]]:
        """書き込みタスクのファストパス実行"""
        logger.info("Write fast path: %s → %s", task.description[:60], file_path)

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} write_file: コンテンツ生成中 → {file_path}",
                "status": "running",
            })

        content = await self._generate_content(
            original_query, task.description, llm_client,
            file_path=file_path,
        )

        if looks_like_path_not_content(content, file_path):
            logger.warning(
                "Write fast path: content looks like a file path, "
                "retrying content generation: %r",
                content,
            )
            content = await self._generate_content(
                original_query, task.description, llm_client,
                file_path=file_path,
            )

        if content.startswith("(Content generation failed:"):
            logger.warning("Write fast path: content generation failed for %s", file_path)
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成失敗",
                    "status": "failed",
                })
            return f"Error: {content}", []

        if looks_like_path_not_content(content, file_path):
            logger.warning(
                "Write fast path: content still looks like a file path "
                "after retry, aborting: %r",
                content,
            )
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成失敗（パス誤出力）",
                    "status": "failed",
                })
            return "Error: Content generation produced only a file path, not actual content", []

        return await self._write_file(
            file_path, content, tools_registry, on_step, prefix,
        )

    async def _write_file(
        self,
        file_path: str,
        content: str,
        tools_registry,
        on_step,
        prefix: str,
    ) -> tuple[str, list[dict]]:
        """write_file を実行して結果を返す"""
        tool_args = {"file_path": file_path, "content": content}
        try:
            result = await tools_registry.execute("write_file", **tool_args)
            result_text = str(result)
            is_success = not result_text.startswith("Error:")
        except Exception as e:
            result_text = f"Error: {e}"
            is_success = False
            logger.error("write_file failed: %s", e)

        tool_entry = {
            "tool": "write_file",
            "args": tool_args,
            "success": is_success,
        }

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} write_file: {result_text[:100]}",
                "status": "done" if is_success else "failed",
            })

        logger.info("write_file completed: %s → %s", file_path, result_text[:80])
        return result_text, [tool_entry]

    async def _recover_write_from_text(
        self,
        text: str,
        task: TaskItem,
        original_query: str,
        llm_client,
        tools_registry,
        on_step,
        prefix: str,
    ) -> dict | None:
        """LLM がツールコール JSON を出力しなかった場合の自動リカバリー"""
        from backend.free.agent.tool_call_judge import _extract_file_path

        file_path = _extract_file_path(task.description)
        if not file_path:
            logger.warning(
                "Auto-recovery skipped: no file path in task: %s",
                task.description[:80],
            )
            return None

        logger.info(
            "Auto-recovery: LLM returned plain text for write task, "
            "extracting content for %s",
            file_path,
        )

        content = strip_markdown_wrapper(text)
        if not text_looks_like_code(content):
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} コンテンツ生成中（自動リカバリー） → {file_path}",
                    "status": "running",
                })
            content = await self._generate_content(
                original_query, task.description, llm_client,
                file_path=file_path,
            )
            if content.startswith("(Content generation failed:"):
                logger.warning("Auto-recovery content generation failed: %s", file_path)
                return None

        if looks_like_path_not_content(content, file_path):
            logger.warning(
                "Auto-recovery: content looks like a file path, aborting: %r",
                content,
            )
            return None

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} write_file({file_path}, {len(content)}文字) [自動リカバリー]",
                "status": "running",
            })

        tool_args = {"file_path": file_path, "content": content}
        try:
            result = await tools_registry.execute("write_file", **tool_args)
            result_text = str(result)
            is_success = not result_text.startswith("Error:")
            logger.info("Auto-recovery write_file: %s → %s", file_path, result_text[:100])

            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: {result_text[:100]}",
                    "status": "done" if is_success else "failed",
                })

            return {
                "tool": "write_file",
                "args": tool_args,
                "success": is_success,
                "result": result_text,
            }
        except Exception as e:
            logger.error("Auto-recovery write_file failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # コンテンツ生成
    # ------------------------------------------------------------------

    async def _generate_content(
        self,
        original_query: str,
        task_description: str,
        llm_client,
        file_path: str = "",
    ) -> str:
        """write_file 用のコンテンツを LLM に生成させる"""
        ctx_size = self.config.get("llama", {}).get("context_size", 4096)

        existing_content = self._read_existing_file(file_path)
        user_prompt = f"{original_query}\n\nタスク: {task_description}"
        user_prompt = self._inject_existing_content(
            user_prompt, existing_content, file_path, ctx_size,
        )

        messages = [
            {"role": "system", "content": CONTENT_GENERATION_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        gen_max_tokens = self._calc_gen_max_tokens(
            CONTENT_GENERATION_PROMPT + user_prompt, ctx_size,
        )

        return await self._stream_and_clean(
            llm_client, messages, gen_max_tokens,
        )

    @staticmethod
    def _read_existing_file(file_path: str) -> str:
        """既存ファイルの内容を読み込む（存在しなければ空文字列）"""
        if not file_path:
            return ""
        p = Path(file_path)
        if p.exists() and p.is_file():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
        return ""

    @staticmethod
    def _inject_existing_content(
        user_prompt: str,
        existing_content: str,
        file_path: str,
        ctx_size: int,
    ) -> str:
        """既存ファイル内容をプロンプトに注入する"""
        if not existing_content:
            return user_prompt
        base_tokens = _estimate_tokens(
            CONTENT_GENERATION_PROMPT + user_prompt
        )
        existing_tokens = _estimate_tokens(existing_content)
        if base_tokens + existing_tokens < ctx_size // 2:
            user_prompt += (
                f"\n\n## 既存ファイル内容 ({file_path})\n"
                f"以下の既存コードを含め、タスクの内容を統合した完全なファイルを出力してください。\n"
                f"```\n{existing_content}\n```"
            )
        else:
            logger.info(
                "Skipping existing content injection: "
                "base=%d + existing=%d tokens > ctx_size/2=%d",
                base_tokens, existing_tokens, ctx_size // 2,
            )
        return user_prompt

    def _calc_gen_max_tokens(
        self, prompt_text: str, ctx_size: int,
    ) -> int:
        """コンテンツ生成用の max_tokens を計算する"""
        input_tokens = _estimate_tokens(prompt_text)
        available = max(ctx_size - input_tokens - 128, 1024)
        gen_max_tokens = min(self._execute_max_tokens, available)
        logger.debug(
            "Content generation: input_tokens≈%d, ctx_size=%d, max_tokens=%d",
            input_tokens, ctx_size, gen_max_tokens,
        )
        return gen_max_tokens

    async def _stream_and_clean(
        self,
        llm_client,
        messages: list[dict],
        gen_max_tokens: int,
    ) -> str:
        """LLM ストリーミング生成 + 後処理（フェンス除去・繰り返し切除）"""
        async def _consume_stream() -> str:
            stream = await llm_client.generate(
                messages, stream=True,
                max_tokens=gen_max_tokens,
                id_slot=getattr(llm_client, 'background_slot', -1),
            )
            chunks: list[str] = []
            async for token in stream:
                chunks.append(token)
            return "".join(chunks).strip()

        try:
            raw = await asyncio.wait_for(
                _consume_stream(),
                timeout=self._content_gen_timeout,
            )
            content = strip_markdown_wrapper(raw)
            content = truncate_repetition(content)
            if not content:
                logger.warning("Content generation returned empty content")
                return "(Content generation failed: empty output)"
            logger.debug("Content generated: %d chars", len(content))
            return content
        except asyncio.TimeoutError:
            logger.warning(
                "Content generation timed out after %ds",
                self._content_gen_timeout,
            )
            return f"(Content generation failed: timeout after {self._content_gen_timeout}s)"
        except Exception as e:
            logger.error("Content generation failed: %s", e)
            return f"(Content generation failed: {e})"

    # ------------------------------------------------------------------
    # フォールバック / 応答組み立て
    # ------------------------------------------------------------------

    async def _fallback_plain_llm(
        self,
        query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        on_step=None,
    ) -> MetaCognitiveResponse:
        """緊急フォールバック: ツール・計画なしの直接 LLM 応答"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                *conversation[-4:],
                {"role": "user", "content": query},
            ]
            result = await asyncio.wait_for(
                llm_client.generate(
                    messages, stream=False, max_tokens=1024,
                    id_slot=getattr(llm_client, "background_slot", -1),
                ),
                timeout=30,
            )
            content = result["choices"][0]["message"]["content"].strip()
            if on_step:
                await call_callback(on_step, {
                    "type": "task_progress",
                    "detail": "フォールバック応答を生成しました",
                    "status": "done",
                })
            return MetaCognitiveResponse(
                content=content or "リクエストを処理できませんでした。",
                steps=0,
            )
        except Exception as e2:
            logger.error("Fallback plain LLM also failed: %s", e2)
            if on_step:
                await call_callback(on_step, {
                    "type": "task_progress",
                    "detail": "フォールバック応答の生成に失敗しました",
                    "status": "failed",
                })
            return MetaCognitiveResponse(
                content="エラーが発生しました。再度お試しください。",
                steps=0,
            )

    @staticmethod
    def _build_final_response(
        tasks: list[TaskItem],
        context_parts: list[str],
    ) -> str:
        """タスク結果から最終応答を組み立て"""
        if not tasks:
            return "No tasks were generated."

        parts: list[str] = []
        for task in tasks:
            parts.append(f"- [{task.status}] {task.description}")
            if task.result:
                parts.append(f"    {task.result[:500]}")

        return "\n".join(parts)
