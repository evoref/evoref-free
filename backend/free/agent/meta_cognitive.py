"""Meta-Cognitive エージェント: 計画立案 + 多段推論 + ツールループ（10〜30秒）"""

from __future__ import annotations

import asyncio
from pathlib import Path

from backend.config import resolve_context_size_for_mode
from backend.free.agent.context_budget import (
    OUTPUT_RESERVE_TOKENS,
    resolve_meta_cognitive_loop_budget,
)
from backend.free.core.session_mode import is_create_mode
from backend.free.agent.code_artifact_generator import CodeArtifactGenerator
from backend.free.agent.event_reminder import EventReminderSystem
from backend.free.agent.self_cartridge import (
    AgentConstants,
    contains_self_reference,
    expand_self_references,
    gather_constants,
)
from backend.free.agent.meta_cognitive_tasks import (
    EditorArtifact,
    MetaCognitiveResponse,
    TaskItem,
    collapse_document_generation_tasks,
    collapse_editor_write_tasks,
    collapse_fetch_save_tasks,
    determine_task_status,
    merge_same_file_tasks,
    task_expects_write,
)
from backend.free.agent.meta_cognitive_tools import infer_tool_from_task
from backend.free.agent.meta_cognitive_utils import (
    contains_code_indicator,
    iter_balanced_brace_substrings,
    is_tool_error,
    parse_template_tool_call,
    try_parse_tool_dict,
    call_callback,
    fix_json_backslashes,
    summarize_file_content,
)
from backend.free.agent.step_compactor import StepCompactor
from backend.free.llm.editor_filename import derive_editor_filename_stem
from backend.log_config import get_logger

# 責務ごとのメソッド群は mixin へ、プロンプト / 定数は共有モジュールへ分割した。
# mixin 側から本体を import すると循環するため、共有物は meta_cognitive_defs に置く。
from backend.free.agent.meta_cognitive_content import _ContentGenerationMixin
from backend.free.agent.meta_cognitive_fast_path import _FastPathMixin
from backend.free.agent.meta_cognitive_task_exec import _TaskExecutionMixin

# 本体が使う定数。
from backend.free.agent.meta_cognitive_defs import (
    _CODE_LANGUAGES,
    _EXPLICIT_PATH_RE,
    _EXT_LANGUAGE_MAP,
    _LANGUAGE_EXT_MAP,
    _LANGUAGE_KEYWORDS,
    PLAN_SYSTEM_PROMPT,
    _WRITE_REJECTION_RE,
    _WRITE_REJECTION_REASON_JA,
)

# 分割前から ``meta_cognitive.<名前>`` で見えていたプロンプト / 定数。参照元を
# 追い切れないため、見え方は分割前と一致させておく (本モジュールでは未使用)。
from backend.free.agent.meta_cognitive_defs import (  # noqa: F401
    CONTENT_GENERATION_PROMPT,
    CSV_CONTENT_INSTRUCTION,
    _DATA_BEARING_TOOLS,
    EXECUTE_SYSTEM_PROMPT,
    _is_placeholder_write_path,
    MARKDOWN_CONTENT_INSTRUCTION,
    _PLACEHOLDER_WRITE_PATH_NAMES,
    _PRIOR_CONTENT_REFERENCE_RE,
    RICH_DOC_CONTENT_INSTRUCTION,
    TABLE_CONTENT_INSTRUCTION,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.agent_tracer import AgentTracer
    from backend.free.agent.tool_call_judge import ToolCallJudge
    from backend.free.core.policy_interpreter import PolicyInterpreter
    from backend.free.llm.aux_client import AuxClient
    from backend.free.memory.views.loop import LoopFactView

logger = get_logger("agent.meta_cognitive")

# 公開シンボル（後方互換）
__all__ = [
    "MetaCognitiveAgent",
    "MetaCognitiveResponse",
    "TaskItem",
]


class MetaCognitiveAgent(
    _TaskExecutionMixin,
    _FastPathMixin,
    _ContentGenerationMixin,
):
    """Meta-Cognitive 層: 計画立案 + 多段推論 + ツール実行ループ

    クリエイトモードのタスクリスト方式 + diff 駆動を実装。
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
        aux_client: "AuxClient | None" = None,
        debug_logger: "DebugLogger | None" = None,
        semmem_block: str | None = None,
        rag_block: str | None = None,
        file_block: str | None = None,
        fewshot_block: str | None = None,
        code_generator: CodeArtifactGenerator | None = None,
        mode: str = "create",
    ) -> None:
        self.max_steps = max_steps
        cfg = config or {}
        self.config = cfg
        # ツールループ用 system に注入する SemMem メモリブロック
        # (チャット初回ターンと同じ MemoryInjector 出力。reactive 以外の
        #  全ターンで policy / failure_pattern facts を維持するため)
        self._semmem_block = semmem_block
        # search pipeline 取得済み RAG チャンク (STM/LTM/cartridge) を整形した
        # ブロック。long_form の prefetched_rag と同じ取得結果を meta 経路でも
        # ツールループ system に維持する (semmem_block と同じ消費形)。
        self._rag_block = rag_block
        # ユーザー添付ファイルを整形したブロック (deliberative の messages 注入と
        # 等価)。meta 経路は messages を LLM に渡さないためここで維持する。
        self._file_block = file_block
        # Level 1 で進化したモード別 few-shot ブロック。meta 経路は固定の
        # PLAN/EXECUTE/CONTENT scaffold を使うため、deliberative のように最後の
        # user メッセージへ前置できない。ツールループ / コンテンツ生成 / fallback
        # の system に [参考例] として注入し、進化を create 生成へ効かせる。
        self._fewshot_block = fewshot_block
        # create モードで editor/chat 出力のコード生成を LongForm 細粒度生成へ
        # 委譲する (composition が注入)。None なら従来の単一ショット生成。
        self._code_generator = code_generator
        # 構築時モード (内部 loop/token 予算の context_size 解決に使う)。
        # process() でも同値を再設定する (呼出側が同じ req.mode を渡す契約)。
        self._mode = mode
        # ツールループ system 用のツール説明文 (mode 別)。process() 毎にリセット。
        self._tool_descriptions_cache: dict[str, str] = {}
        self.compactor = StepCompactor(cfg, policy=policy)
        self.reminder_system = EventReminderSystem(cfg)
        self._tool_judge = tool_judge
        self._agent_tracer = agent_tracer
        # に従い補助タスクで実行する。``None`` の場合 (aux_client
        # health_check 失敗による degraded mode) は計画生成をスキップして
        # 単一タスクへフォールバックする。
        self._aux_client = aux_client
        # に記録する (decision_point=``meta_cognitive_llm_route``)。
        self._debug_logger = debug_logger

        agent_cfg = cfg.get("agent", {})
        self.max_tool_iterations = agent_cfg.get("max_tool_iterations", 5)

        ctx_size = resolve_context_size_for_mode(cfg, mode)
        self.loop_budget = resolve_meta_cognitive_loop_budget(cfg, mode)

        self._execute_max_tokens = max(ctx_size - OUTPUT_RESERVE_TOKENS, 1024)
        self._reminder_budget = 100

        # content gen の総上限 (低速 GPU でも完走できるよう緩和)。トークン間
        # アイドルタイムアウト (無出力検知) と併用する。
        self._content_gen_timeout = agent_cfg.get("content_gen_timeout", 600)
        self._content_gen_first_token_timeout = agent_cfg.get(
            "content_gen_first_token_timeout", 120,
        )
        self._content_gen_idle_timeout = agent_cfg.get("content_gen_idle_timeout", 30)
        self._llm_call_timeout = agent_cfg.get("llm_call_timeout", 90)
        self._total_timeout = agent_cfg.get("total_timeout", 1800)

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
        on_step=None,
        generation_params: dict | None = None,
        session_id: str = "",
        mode: str = "create",
        output_target: str = "file",
        private: bool = False,
    ) -> MetaCognitiveResponse:
        """Meta-Cognitive 層で計画立案 → タスク実行ループ

        全体をタイムアウトで保護し、失敗時はプレーン LLM にフォールバック。

        ``output_target`` は生成コードの出力先 (create モードのみ意味を持つ):
        ``"file"`` (既定、明示パスへ write_file) / ``"editor"`` (ディスク書込せず
        ``editor_artifacts`` へ) / ``"chat"`` (コードフェンスで ``content`` に返す)。

        ``private`` はリクエストの private フラグ。MDP エピソードの begin
        イベントに刻み、エピソード記憶への昇格を止める (``AgentTracer``)。

        内部の生成が 1 つでも ``finish_reason=length`` で切れていれば、
        どの経路 (通常 / 救出 / フォールバック) の応答にも ``truncated`` を刻む。
        """
        self._truncated_steps: list[str] = []
        self._truncated_tokens = 0
        self._truncated_max_tokens: int | None = None
        resp = await self._process_or_fallback(
            query, system_prompt, conversation, llm_client, tools_registry,
            on_step, generation_params=generation_params,
            session_id=session_id, mode=mode, output_target=output_target,
            private=private,
        )
        if self._truncated_steps:
            resp.truncated = True
            resp.truncated_steps = list(self._truncated_steps)
            resp.truncated_tokens = self._truncated_tokens
            resp.truncated_max_tokens = self._truncated_max_tokens
        return resp

    async def _process_or_fallback(
        self,
        query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        tools_registry=None,
        on_step=None,
        *,
        generation_params: dict | None = None,
        session_id: str = "",
        mode: str = "create",
        output_target: str = "file",
        private: bool = False,
    ) -> MetaCognitiveResponse:
        """``_process_impl`` を総タイムアウトで保護し、失敗時は救出 / フォールバック。"""
        self._private_request = private
        try:
            return await asyncio.wait_for(
                self._process_impl(
                    query, system_prompt, conversation, llm_client,
                    tools_registry, on_step,
                    output_target=output_target,
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
            salvaged = self._salvage_partial_response()
            if salvaged is not None:
                logger.info(
                    "Salvaged %d editor artifact(s) after timeout",
                    len(salvaged.editor_artifacts),
                )
                if on_step:
                    await call_callback(on_step, {
                        "type": "task_progress",
                        "detail": "タイムアウト — 生成済みコードを返します",
                        "status": "done",
                    })
                return salvaged
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
            salvaged = self._salvage_partial_response()
            if salvaged is not None:
                logger.info(
                    "Salvaged %d editor artifact(s) after error",
                    len(salvaged.editor_artifacts),
                )
                return salvaged
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
        on_step=None,
        *,
        output_target: str = "file",
        generation_params: dict | None = None,
        session_id: str = "",
        mode: str = "create",
    ) -> MetaCognitiveResponse:
        """process() の本体実装"""
        from backend.free.agent.credit_assigner import assign_credit, compute_final_outcome

        all_tool_calls: list[dict] = []
        # 出力先パス未指定 (editor/chat 経路) で生成したコードの蓄積先。
        # process 呼び出しごとにリセット (meta_agent はリクエスト毎に生成される)。
        self._output_target = output_target
        self._mode = mode
        self._tool_descriptions_cache = {}
        self._editor_artifacts: list[EditorArtifact] = []
        self._chat_code_parts: list[str] = []
        # データ取得系ツール (fetch_url / read_file 等) の生結果をタスク横断で蓄積。
        # 表/ファイル生成タスクが取得済み実データを直接参照し、転記ハルシネーションを防ぐ。
        self._fetched_tool_outputs: list[str] = []
        # 直近会話。write 全経路 (fast path / tool-loop / auto-recovery) が
        # 合流する _generate_content から参照する。個別に引数を通すと経路ごとの
        # 漏れが出るため、リクエストごとの状態としてここで一括保持する。
        self._conversation: list[dict] = list(conversation or [])

        # MDP トレース: エピソード開始
        tracer = self._agent_tracer
        episode_id = ""
        if tracer is not None:
            episode_id = tracer.begin_episode(
                session_id, mode, private=getattr(self, "_private_request", False),
            )

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
        # 単一 URL → 単一ファイル保存の過分割 (fetch/extract/generate/save) を
        # fetch+write へ集約する。抽出/保存タスクでの小型モデル拒否を防ぐ。
        if self._output_target == "file":
            tasks = collapse_fetch_save_tasks(tasks, query)
            # URL 無しの文書/データ出力で「スクリプト生成 → 実行」と誤分解された
            # プランを単一 write タスクへ正規化する (内容を直接 write_file させ、
            # 実体の無い生成スクリプトの実行や .xlsx へのコード書込みを防ぐ)。
            tasks = collapse_document_generation_tasks(tasks, query)
        # editor/chat 経路 (パス未指定) は merge_same_file_tasks がすり抜けるため、
        # 過分割された書き込みタスクを 1 件へ集約する (1 リクエスト=1 タブ)。
        if self._output_target in ("editor", "chat"):
            tasks = collapse_editor_write_tasks(tasks)
        logger.info(
            "Plan generated: %d tasks (mode=%s, output=%s)",
            len(tasks), self._mode, self._output_target,
        )

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
        if output_target == "chat" and self._chat_code_parts:
            # チャット経路: 生成コードをコードフェンス付きで本文に返す
            content = "\n\n".join(self._chat_code_parts)
        else:
            content = self._build_final_response(tasks, context_parts)
        logger.info(
            "MetaCognitive completed: %d steps, %d tool calls (mode=%s, output=%s)",
            steps, len(all_tool_calls), self._mode, self._output_target,
        )

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
            editor_artifacts=self._editor_artifacts,
        )

    async def _plan(
        self,
        query: str,
        conversation: list[dict],
        llm_client,  # noqa: ARG002
        on_step=None,
    ) -> list[TaskItem]:
        """補助タスクにタスク計画を生成させる

        CLAUDE.md §1 に従い、判定 / 計画系の JSON 応答処理は補助タスク
        モデル (``aux_client.generate_json``) で実行する。``llm_client``
        引数はベースモデル (Meta-Cognitive のメイン応答 / ツール呼び出し
        ループ用) の参照を保つため受け取り続けるが、計画生成自体には
        利用しない。

        ``aux_client`` が ``None`` (health_check 失敗で degraded mode)
        の場合は空リストを返し、呼び出し側で単一タスクフォールバックする。
        """
        if on_step:
            await call_callback(on_step, {
                "type": "plan",
                "detail": "タスク計画を生成中...",
                "status": "running",
            })

        # ベース llama-server 未接続 (degraded mode) では計画を諦め、静かに
        # 単一タスクへ倒す。
        if self._aux_client is None:
            logger.info(
                "Plan generation skipped: aux client is not wired; "
                "falling back to single task",
            )
            if self._debug_logger is not None:
                self._debug_logger.log_decision(
                    decision_point="meta_cognitive_llm_route",
                    chosen="single_task_fallback",
                    candidates=["aux_plan", "single_task_fallback"],
                    reason="aux_client_unavailable",
                    scope="request",
                )
            return []

        if self._debug_logger is not None:
            self._debug_logger.log_decision(
                decision_point="meta_cognitive_llm_route",
                chosen="aux_plan",
                candidates=["aux_plan", "single_task_fallback"],
                reason="aux_client_available",
                scope="request",
            )

        user_content = self._build_plan_user_content(query, conversation)
        prompt = f"{PLAN_SYSTEM_PROMPT}\n\n{user_content}"

        try:
            # 二重 timeout の意図: 内側は AuxClient が purpose 別の実効
            # タイムアウト (較正込み) を HTTP 層へ渡す。外側 wait_for
            # (=_llm_call_timeout, 既定 90s) は、内側がリトライで伸びた場合でも
            # 計画生成の総時間を保証する保険。
            data = await asyncio.wait_for(
                self._aux_client.generate_json(
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
        except Exception as e:
            logger.warning("Plan generation failed: %s", e)
            return []

        tasks_raw = data.get("tasks") if isinstance(data, dict) else None
        if isinstance(tasks_raw, list):
            tasks = [TaskItem(description=str(t)) for t in tasks_raw if t]
            tasks = self._ensure_build_task(query, tasks)
            return self._normalize_planned_paths(query, tasks)

        logger.warning(
            "Plan response did not contain a 'tasks' list: %r", data,
        )
        return []

    @staticmethod
    def _ensure_build_task(
        query: str, tasks: list[TaskItem],
    ) -> list[TaskItem]:
        """build 意図のクエリで plan が write タスクを 1 つも返さなかった場合に補正する。

        ユーザーが「作成 / 生成 / create / implement…」等で明確にプログラム生成を求めて
        いるのに、plan が "Design the game structure…" のような設計・分析タスクのみを返すと、
        codegen 経路 (条件: ``task_expects_write`` が True) に乗らず実ファイルが生成されない
        (status=done でもテキストの設計文書しか出ない)。その退化時に限り、全体を単一の生成
        タスクへ正規化して codegen に乗せる。実際の生成内容は ``original_query`` 駆動なので、
        タスク記述はルーティング用途で十分。
        """
        if not tasks or not task_expects_write(query):
            return tasks
        if any(task_expects_write(t.description) for t in tasks):
            return tasks
        logger.warning(
            "Plan produced no write task for a build request; normalizing to a "
            "single generate task (query=%r)", query[:80],
        )
        return [TaskItem(description="Generate the full program the user requested")]

    @staticmethod
    def _normalize_planned_paths(
        query: str, tasks: list[TaskItem],
    ) -> list[TaskItem]:
        """ユーザー明示パスが plan タスクから脱落した場合に決定論で補完する。

        小型 aux は PLAN_SYSTEM_PROMPT の例文をエコーしてユーザー指定の
        出力パスを落とすことがある (2026-07-15: パス無し "Generate the full
        program..." が 2 ターンで write 不発 → 失敗)。クエリに明示的な
        絶対パスがあるのに、どのタスクにもパスが含まれない場合、最初の
        書込み期待タスクへユーザーのパスを付記して write 経路に載せる。
        """
        if not tasks:
            return tasks
        m = _EXPLICIT_PATH_RE.search(query)
        if not m:
            return tasks
        user_path = m.group(0).rstrip("。、.,;:")
        if any(_EXPLICIT_PATH_RE.search(t.description) for t in tasks):
            return tasks
        for t in tasks:
            if task_expects_write(t.description):
                logger.warning(
                    "Plan dropped the user's explicit output path; "
                    "re-attaching %s to task: %s",
                    user_path, t.description[:80],
                )
                t.description = f"{t.description} and save it to {user_path}"
                break
        return tasks

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

            if (
                self._output_target in ("editor", "chat")
                and task_expects_write(task.description)
            ):
                # 出力先パス未指定: ディスク書込せずコードを生成しエディタ/チャットへ
                result, tool_calls = await self._execute_editor_task(
                    task, query, llm_client, on_step,
                    task_index=i + 1, total_tasks=total_tasks,
                )
                task.result = result
                task.status = "failed" if is_tool_error(result) else "done"
            else:
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
                # タスクが完了扱いでも、使ったツールが軒並み情報ゼロ (0 件検索 /
                # 非ゼロ終了コマンド) なら reward は 0。「実行できた」を成功として
                # credit を配ると、空振りするツール選択が正例に化ける。
                # ツール未使用のタスク (純生成) は従来どおり status のみで判定する。
                produced_info = not tool_calls or any(
                    tc.get("success") for tc in tool_calls
                )
                reward = 1.0 if task.status == "done" and produced_info else 0.0
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

    async def _delegate_code_generation(
        self, original_query: str, on_step,
    ) -> list[EditorArtifact]:
        """code_generator へ委譲。失敗時は空リストでフォールバックさせる。"""
        if self._code_generator is None:
            return []
        try:
            return await self._code_generator(original_query, on_step)
        except Exception as e:
            logger.warning("Code generation delegation failed: %s", e)
            return []

    async def _execute_editor_task(
        self,
        task: TaskItem,
        original_query: str,
        llm_client,
        on_step,
        *,
        task_index: int,
        total_tasks: int,
    ) -> tuple[str, list[dict]]:
        """出力先パス未指定時のタスク実行: ディスク書込せずコードを生成する。

        ``output_target == "editor"`` なら ``editor_artifacts`` に蓄積し、
        ``"chat"`` ならコードフェンス付きで ``_chat_code_parts`` に蓄積する。
        ``write_file`` は一切呼ばない。戻り値の tool_calls は常に空。
        """
        prefix = f"[{task_index}/{total_tasks}]"
        if on_step:
            await call_callback(on_step, {
                "type": "task_progress",
                "detail": f"{prefix} コンテンツ生成中...",
                "status": "running",
            })
        # create モード: LongForm 細粒度生成へ委譲 (複数ファイル可)。空リスト
        # (テキスト判定 / degraded / 失敗) なら従来の単一ショット生成へフォールバック。
        if is_create_mode(self._mode):
            artifacts = await self._delegate_code_generation(original_query, on_step)
            if artifacts:
                if self._output_target == "chat":
                    for art in artifacts:
                        self._chat_code_parts.append(
                            f"```{art.language}\n{art.content}\n```"
                        )
                else:
                    self._editor_artifacts.extend(artifacts)
                return f"Generated {len(artifacts)} file(s)", []
        content = await self._generate_content(
            original_query, task.description, llm_client,
        )
        if content.startswith("(Content generation failed:"):
            return f"Error: {content}", []
        filename, language = self._guess_artifact_meta(
            task.description, original_query,
        )
        # create モードでコード系言語が期待されるのに散文 ("設計します..." のみ)
        # が返った場合は成果物として採用せず failed に倒す。これを done のまま
        # 通すと reward=1.0 で記録され Level 0/1 が誤強化される (false-success)。
        # contains_code_indicator は長さ・改行に依存しないため、1 行の短い
        # 正当コード (print(...) 等) を誤って弾かず、指標皆無の散文だけを拒否する。
        if (
            is_create_mode(self._mode)
            and language in _CODE_LANGUAGES
            and not contains_code_indicator(content)
        ):
            logger.warning(
                "Editor task produced prose, not %s code; marking failed: %s",
                language, task.description[:80],
            )
            return f"Error: generated content is prose, not {language} code", []
        if self._output_target == "chat":
            self._chat_code_parts.append(f"```{language}\n{content}\n```")
            return f"Generated {len(content)} chars (chat)", []
        if filename is None:
            # 明示パス由来のファイル名が無い場合、指示文と言語から
            # ASCII snake_case 名を導出する (日本語タブ名を出さないため)。
            stem = derive_editor_filename_stem(
                hint=original_query, language=language,
            )
            filename = f"{stem}.{_LANGUAGE_EXT_MAP.get(language, 'txt')}"
        self._editor_artifacts.append(
            EditorArtifact(content=content, language=language, filename=filename),
        )
        return f"Generated {len(content)} chars → editor", []

    @staticmethod
    def _guess_artifact_meta(
        task_description: str, query: str,
    ) -> tuple[str | None, str]:
        """エディタ出力片のファイル名と言語を推定する (best-effort)。

        明示パス/ファイル名があれば拡張子から言語を引き、無ければクエリ中の
        言語名キーワードで判定、最終フォールバックは ``"python"``。
        """
        from backend.free.agent.tool_call_judge import _extract_file_path
        path = _extract_file_path(task_description) or _extract_file_path(query)
        if path:
            name = Path(path).name
            if name and "." in name:
                ext = name.rsplit(".", 1)[-1].lower()
                return name, _EXT_LANGUAGE_MAP.get(ext, "python")
        combined = f"{task_description} {query}".lower()
        for keyword, language in _LANGUAGE_KEYWORDS:
            if keyword in combined:
                return None, language
        return None, "python"

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

    def _referential_write_path(self, query_path: str | None = None) -> str:
        """書込み先を直近会話から解決する (解決できなければ空文字列)。

        ``query_path`` に **ディレクトリを伴わない裸のファイル名**
        (``notes.txt``) を渡すと、会話中の同じ basename を持つフルパスに
        解決する。裸の名前をそのまま write_file へ渡すとカレント
        ディレクトリに着地し、ユーザーが指した既存ファイルとは別物を作る
        (2026-08-09 ライブ監査で判明した実害の対処)。``None`` なら従来どおり
        「最後に出たパス」を返す (「同じファイルに保存し直して」型)。
        """
        from backend.free.agent.tool_call_judge import _resolve_referenced_path

        path = _resolve_referenced_path(
            query_path, getattr(self, "_conversation", None),
        )
        if path:
            logger.info(
                "Referential write target resolved from conversation: %s "
                "(query_path=%r)", path, query_path,
            )
            return path
        return ""

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
    # ツール判定
    # ------------------------------------------------------------------

    async def _judge_tool_for_task(
        self,
        task_description: str,
        tools_registry,
    ):
        """タスク記述に対してツール判定を行う

        write 期待タスク (``task_expects_write``) は決定論的な
        ``infer_tool_from_task`` を先に試す。``ToolCallJudge.judge()`` の
        Step 0 (URL recall) は mode に依らず無条件・最優先で発火し、ヒットすると
        以降の全判定層を丸ごとスキップして即確定するため、write 期待タスクの
        クエリが過去の無関係な URL 記憶と埋め込み類似度だけで一致すると
        fetch_url 等へハイジャックされてしまう (例: 「docx を出力して」が
        過去の全く無関係な URL fetch 記憶と誤マッチする)。
        """
        if task_expects_write(task_description):
            inferred = infer_tool_from_task(task_description)
            if inferred is not None and inferred[0] == "write_file":
                from backend.free.agent.tool_call_judge import ToolJudgement
                return ToolJudgement(
                    tool_needed=True,
                    tool_name=inferred[0],
                    tool_args=inferred[1],
                    source="rule",
                )

        if self._tool_judge is not None:
            try:
                # mode は要求のもの (chat.py が req.mode を渡す)。"create" 固定
                # だと chat のタスクで create 専用ツールが選ばれる。
                judgement = await self._tool_judge.judge(
                    task_description, tools_registry, mode=self._mode,
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

        # 3. テンプレート形式 (<|tool_call>call:NAME(...)<tool_call|>) を試す。
        #    base モデルが OAI JSON でなくチャットテンプレート由来の生テキストで
        #    ツールコールを吐く場合 (tool_calls=0 の原因) を救済する。
        template = parse_template_tool_call(text)
        if template is not None:
            return template

        return None


    # ------------------------------------------------------------------
    # フォールバック / 応答組み立て
    # ------------------------------------------------------------------

    def _salvage_partial_response(self) -> MetaCognitiveResponse | None:
        """中断 (タイムアウト/例外) 時に、生成済みコードがあれば破棄せず返す。

        ``_process_impl`` は editor/chat 出力を ``_editor_artifacts`` /
        ``_chat_code_parts`` (インスタンス属性) に逐次蓄積するため、メイン処理が途中で
        打ち切られても既に確定した成果は残る。1 つでもあればそれを返し、無ければ ``None``
        (呼び出し側は通常のフォールバックへ)。``_process_impl`` 開始前の中断にも備えて
        ``getattr`` で防御的に参照する。
        """
        artifacts = list(getattr(self, "_editor_artifacts", None) or [])
        chat_parts = list(getattr(self, "_chat_code_parts", None) or [])
        if not artifacts and not chat_parts:
            return None
        content = (
            "\n\n".join(chat_parts) if chat_parts
            else "生成済みのコードを返しました（処理は時間切れで中断されました）。"
        )
        return MetaCognitiveResponse(
            content=content,
            editor_artifacts=artifacts,
            steps=0,
        )

    async def _fallback_plain_llm(
        self,
        query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        on_step=None,
    ) -> MetaCognitiveResponse:
        """緊急フォールバック: ツール・計画なしの直接 LLM 応答 (ストリーミング)。

        旧実装は ``stream=False`` + 30 秒の総ウォールクロックで、低速モデルではほぼ確実に
        タイムアウトしていた (#1 と同じアンチパターン)。``_stream_text_with_idle_timeout``
        で進行中の生成を殺さず読み取る。
        """
        try:
            # few-shot は従来 system_prompt へ結合して渡されていたが、現在は
            # instance block 化したためここで明示的に注入する (fallback でも維持)。
            system_content = system_prompt
            if self._fewshot_block:
                system_content = f"{system_content}\n\n[参考例]\n{self._fewshot_block}"
            messages = [
                {"role": "system", "content": system_content},
                *conversation[-4:],
                {"role": "user", "content": query},
            ]
            text, timed_out = await self._stream_text_with_idle_timeout(
                llm_client, messages,
                max_tokens=1024,
                id_slot=getattr(llm_client, "background_slot", -1),
                step="fallback",
            )
            content = text.strip()
            if timed_out or not content:
                raise RuntimeError(
                    f"fallback empty or timed out (timed_out={timed_out})",
                )
            if on_step:
                await call_callback(on_step, {
                    "type": "task_progress",
                    "detail": "フォールバック応答を生成しました",
                    "status": "done",
                })
            return MetaCognitiveResponse(content=content, steps=0)
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
    def _write_failure_reason(result: str) -> str:
        """書込み失敗の理由を短い日本語で返す (未知なら空文字列。純粋関数)。"""
        m = _WRITE_REJECTION_RE.search(result or "")
        if m is None:
            return ""
        reason = _WRITE_REJECTION_REASON_JA.get(m.group(1))
        return f": {reason}" if reason else ""

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
            if task.status == "failed" and task_expects_write(task.description):
                # write 期待タスクの失敗時、無関係なツール結果 (誤ってハイジャック
                # された fetch_url の webpage 抽出テキスト等) をそのままユーザーへ
                # 露出させない。ただし **自前の棄却理由コード** は安全に出せるので
                # 添える (理由を伏せるとモデルが次のターンで作り話をする。
                # _WRITE_REJECTION_REASON_JA のコメント参照)。
                reason = MetaCognitiveAgent._write_failure_reason(task.result or "")
                parts.append(f"    (書き込みが実行されませんでした{reason})")
            elif task.result:
                parts.append(f"    {task.result[:500]}")

        return "\n".join(parts)
