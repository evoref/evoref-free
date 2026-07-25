"""Meta-Cognitive エージェント: 計画立案 + 多段推論 + ツールループ（10〜30秒）"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from backend.config import resolve_context_size_for_mode
from backend.free.agent.agent_state import AgentState
from backend.free.agent.context_budget import (
    OUTPUT_RESERVE_TOKENS,
    resolve_meta_cognitive_loop_budget,
)
from backend.free.core.session_mode import is_coding_mode
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
from backend.free.agent.meta_cognitive_tools import (
    infer_tool_from_task,
    normalize_read_file_args,
    normalize_write_file_args,
)
from backend.free.agent.output_format import (
    is_rich_table_output,
    is_table_output,
    resolve_dir_output_path,
    wants_fetched_table,
)
from backend.free.agent.meta_cognitive_utils import (
    contains_code_indicator,
    content_language_directive,
    iter_balanced_brace_substrings,
    is_tool_error,
    parse_template_tool_call,
    try_parse_tool_dict,
    call_callback,
    fewshot_contains_task_log,
    fewshot_seems_relevant,
    fix_json_backslashes,
    generated_content_rejection,
    looks_like_path_not_content,
    strip_markdown_wrapper,
    strip_task_log_scaffold,
    summarize_file_content,
    summarize_tool_args,
    text_looks_like_code,
    truncate_repetition,
)
from backend.free.agent.step_compactor import StepCompactor, StepResult
from backend.free.core.inference import build_messages_for_loop
from backend.free.llm.editor_filename import derive_editor_filename_stem
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
- Implement EXACTLY the program/feature the user requested, using the user's own terms. \
NEVER substitute a different or merely "similar" program. IGNORE any unrelated program \
names that appear in the examples below or in prior conversation context — they are \
illustrations of FORMAT only, not of WHAT to build.
- When the user asks to BUILD, CREATE, MAKE, or WRITE a program, script, app, a document, \
or a data file, the task MUST be an action that PRODUCES that deliverable. Do NOT reduce a \
build request to only a "Design/Analyze/Plan the structure" task — that yields just an \
explanation and no usable deliverable. In the examples below, <...> is a placeholder: \
replace it with the user's actual words. NEVER copy a <...> placeholder or an example \
sentence verbatim into your output.
  BAD  (user asked to build): {"tasks": ["Design the game structure and core logic"]}
  GOOD (user asked to build): {"tasks": ["Generate the full <deliverable the user asked for>"]}
- NEVER invent a file path. Only include a file path in a task when the USER explicitly \
gave one. If the user did not specify an output location, describe the task WITHOUT any path.
  BAD  (user gave no path): {"tasks": ["Create e:\\\\app\\\\solution.py with the full implementation"]}
  GOOD (user gave no path): {"tasks": ["Generate the full <deliverable the user asked for>"]}
  GOOD (user said save to e:\\\\app\\\\solution.py): {"tasks": ["Create e:\\\\app\\\\solution.py with the full implementation"]}
- When the user DID give an explicit output path, EVERY create/write task MUST repeat that \
exact path verbatim. Dropping the user's path from the task loses the write destination.
- Creating or rewriting a SINGLE file is always ONE task, not multiple tasks.
  BAD:  {"tasks": ["Create the core logic", "Add feature A", "Add input handling"]}
  GOOD: {"tasks": ["Generate the full <deliverable the user asked for> in a single file"]}
- When the user asks for a DOCUMENT or DATA FILE (Excel/spreadsheet/CSV, Word, \
PowerPoint, a calendar, a table, a report), the deliverable is the FILE CONTENT itself. \
Plan a SINGLE write task that writes the content to the file. Do NOT plan to "generate a \
Python/openpyxl/VBA script" and do NOT plan to "run/execute" a script — the system renders \
the content into the real .xlsx/.docx/.pptx automatically.
  BAD  (user asked for an Excel calendar): {"tasks": ["Generate a Python script that creates the Excel file", "Execute the generated script"]}
  GOOD (user asked for an Excel calendar): {"tasks": ["Write this month's calendar to the user's specified path"]}
- Only split into multiple tasks when genuinely different files or operations are needed.
- Each task should have a SINGLE action type. Do NOT combine "fetch/read" and "write/create" \
in one task.
  BAD:  {"tasks": ["Fetch URL and create script"]}
  GOOD: {"tasks": ["Fetch URL content", "Generate script from fetched content"]}
- For information-only requests (explain, summarize, list, show), use fetch_url or read_file \
as the task — do NOT plan to create files.
  BAD:  {"tasks": ["Create script to scrape website"]}
  GOOD: {"tasks": ["Fetch URL and summarize the content"]}
- Fetching a URL and saving its data to a file is exactly TWO tasks: fetch, then write. \
Do NOT add separate "extract", "generate the file", or "save" steps — extracting the data \
and creating the file both happen inside the single write task.
  BAD:  {"tasks": ["Fetch the URL", "Extract the results", "Generate the Excel file", "Save it to <path>"]}
  GOOD: {"tasks": ["Fetch the URL content", "Write the results to the user's specified path"]}
- Each task should be self-contained and produce a concrete result.

Example: {"tasks": ["Read foo.py", "Generate refactored code", "Run tests"]}
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

# スプレッドシート/表形式の出力先では本文を GFM マークダウン表として生成させる。
# write_file → ContentConverter.from_markdown → XlsxWriter で実セルに展開される。
TABLE_CONTENT_INSTRUCTION = (
    "The target is a spreadsheet/table file. Output ONLY a GitHub-flavored "
    "Markdown table built from the data gathered in the previous steps: a header "
    "row, a `| --- |` separator row, then one row per record. Every row must "
    "start and end with a pipe `|`. No prose, no code fences, no extra text."
)

# .csv は export 変換を通らず raw テキストとして書き込まれるため、GFM 表ではなく
# CSV 行そのものを出力させる (散文/説明文の混入は書込み前検証で棄却される)。
CSV_CONTENT_INSTRUCTION = (
    "The target is a raw CSV file. Output ONLY comma-separated values: one "
    "header row, then one line per record. Use the exact columns the user "
    "asked for. No prose, no Markdown, no code fences, no extra text."
)

# Word/PowerPoint 等のリッチ文書で、取得済みテーブルが無くモデル生成に落ちる時の
# 保険。export Writer が変換できる GFM を出させ、python-pptx/VBScript 等の「文書を
# 作るコード」をテキスト出力する退行を明示的に禁じる。表は強制しない (散文文書も可)。
RICH_DOC_CONTENT_INSTRUCTION = (
    "The target is a Word/PowerPoint document. Output ONLY GitHub-flavored Markdown "
    "(headings with #, paragraphs, bullet lists, and Markdown tables as appropriate). "
    "Do NOT output python-pptx, python-docx, VBScript, openpyxl, or any program code. "
    "No code fences around the whole document."
)

# ユーザークエリ/タスク記述中の明示的な絶対パス (Windows ドライブレター形式)。
# plan 後のパス脱落補完 (_normalize_planned_paths) で使用する。
_EXPLICIT_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'「」()（）]+")

# 実データを取得するツール。これらの生結果をタスク横断で蓄積し、後続の
# write タスクが取得済みデータを直接参照できるようにする (転記ハルシネーション防止)。
_DATA_BEARING_TOOLS: frozenset[str] = frozenset({
    "fetch_url", "read_file", "search_code", "search_history", "rag_search",
})

# 拡張子 → 言語識別子 (エディタ出力片のシンタックスハイライト用、best-effort)
_EXT_LANGUAGE_MAP: dict[str, str] = {
    "py": "python", "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "mts": "typescript", "cts": "typescript",
    "tsx": "typescript", "jsx": "javascript",
    "json": "json", "html": "html", "htm": "html", "css": "css",
    "xml": "xml", "yaml": "yaml", "yml": "yaml", "sql": "sql",
    "php": "php", "md": "markdown", "sh": "bash", "rb": "ruby",
    "go": "go", "rs": "rust", "java": "java", "c": "c", "cpp": "cpp", "cs": "csharp",
}

# 言語識別子 → 主要拡張子 (エディタ出力片のファイル名生成用、best-effort)。
# `_EXT_LANGUAGE_MAP` の逆引き。未知言語は呼出側で ``txt`` フォールバック。
_LANGUAGE_EXT_MAP: dict[str, str] = {
    "python": "py", "javascript": "js", "typescript": "ts",
    "json": "json", "html": "html", "css": "css", "xml": "xml",
    "yaml": "yaml", "sql": "sql", "php": "php", "markdown": "md",
    "bash": "sh", "ruby": "rb", "go": "go", "rust": "rs",
    "java": "java", "c": "c", "cpp": "cpp", "csharp": "cs",
}

# クエリ中の言語名キーワード → 言語識別子 (拡張子が無い場合のフォールバック、先頭優先)
_LANGUAGE_KEYWORDS: list[tuple[str, str]] = [
    ("typescript", "typescript"), ("javascript", "javascript"),
    ("python", "python"), ("html", "html"), ("css", "css"),
    ("rust", "rust"), ("golang", "go"), ("java", "java"),
    ("ruby", "ruby"), ("bash", "bash"), ("sql", "sql"),
]

# text_looks_like_code の code indicator が信頼できる言語。これら言語の生成物が
# コードに見えない場合は散文 (例: "設計します..." のみ) の false-success とみなす。
# markdown/html/css/json/yaml/xml/text は indicator 不在でも正当なため除外する。
_CODE_LANGUAGES: frozenset[str] = frozenset({
    "python", "javascript", "typescript", "go", "rust",
    "java", "c", "cpp", "csharp", "ruby", "php", "bash",
})


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
        rag_block: str | None = None,
        file_block: str | None = None,
        fewshot_block: str | None = None,
        code_generator: CodeArtifactGenerator | None = None,
        mode: str = "coding",
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
        # の system に [参考例] として注入し、進化を coding 生成へ効かせる。
        self._fewshot_block = fewshot_block
        # coding モードで editor/chat 出力のコード生成を LongForm 細粒度生成へ
        # 委譲する (composition が注入)。None なら従来の単一ショット生成。
        self._code_generator = code_generator
        # 構築時モード (内部 loop/token 予算の context_size 解決に使う)。
        # process() でも同値を再設定する (呼出側が同じ req.mode を渡す契約)。
        self._mode = mode
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
        mode: str = "coding",
        output_target: str = "file",
    ) -> MetaCognitiveResponse:
        """Meta-Cognitive 層で計画立案 → タスク実行ループ

        全体をタイムアウトで保護し、失敗時はプレーン LLM にフォールバック。

        ``output_target`` は生成コードの出力先 (coding モードのみ意味を持つ):
        ``"file"`` (既定、明示パスへ write_file) / ``"editor"`` (ディスク書込せず
        ``editor_artifacts`` へ) / ``"chat"`` (コードフェンスで ``content`` に返す)。
        """
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
        mode: str = "coding",
    ) -> MetaCognitiveResponse:
        """process() の本体実装"""
        from backend.free.agent.credit_assigner import assign_credit, compute_final_outcome

        all_tool_calls: list[dict] = []
        # 出力先パス未指定 (editor/chat 経路) で生成したコードの蓄積先。
        # process 呼び出しごとにリセット (meta_agent はリクエスト毎に生成される)。
        self._output_target = output_target
        self._mode = mode
        self._editor_artifacts: list[EditorArtifact] = []
        self._chat_code_parts: list[str] = []
        # データ取得系ツール (fetch_url / read_file 等) の生結果をタスク横断で蓄積。
        # 表/ファイル生成タスクが取得済み実データを直接参照し、転記ハルシネーションを防ぐ。
        self._fetched_tool_outputs: list[str] = []

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
            # 二重 timeout の意図: 内側の assist realtime 経路が
            # PURPOSE_TIMEOUT_DEFAULTS["meta_cognitive_plan"]=30s を
            # asyncio.timeout で総予算強制するため、realtime 運用では実効上限は
            # 内側 30s 側。外側 wait_for(=_llm_call_timeout, 既定 90s) は
            # (a) config timeouts.meta_cognitive_plan を 90s 超に上書きした場合、
            # または (b) 将来 priority が realtime 以外へ退化し内側 asyncio.timeout
            # が nullcontext 化した場合に、計画生成の総時間を保証する保険。
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

        小型 assist は PLAN_SYSTEM_PROMPT の例文をエコーしてユーザー指定の
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
        # coding モード: LongForm 細粒度生成へ委譲 (複数ファイル可)。空リスト
        # (テキスト判定 / degraded / 失敗) なら従来の単一ショット生成へフォールバック。
        if is_coding_mode(self._mode):
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
        # coding モードでコード系言語が期待されるのに散文 ("設計します..." のみ)
        # が返った場合は成果物として採用せず failed に倒す。これを done のまま
        # 通すと reward=1.0 で記録され Level 0/1 が誤強化される (false-success)。
        # contains_code_indicator は長さ・改行に依存しないため、1 行の短い
        # 正当コード (print(...) 等) を誤って弾かず、指標皆無の散文だけを拒否する。
        if (
            is_coding_mode(self._mode)
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
            # 明示パス由来のファイル名が無い場合、生成内容からアシストモデルで
            # ASCII snake_case 名を導出する (日本語タブ名を出さないため)。
            stem = await derive_editor_filename_stem(
                self._assist_client,
                content=content, hint=original_query, language=language,
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
            tool_descriptions = tools_registry.get_descriptions_text(mode=self._mode)
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
        # search pipeline 取得済み RAG チャンクを参考コンテキストとして維持する
        if self._rag_block:
            prompt = f"{prompt}\n\n[参考コンテキスト]\n{self._rag_block}"
        # ユーザー添付ファイルをループ全反復の system に維持する
        if self._file_block:
            prompt = f"{prompt}\n\n[添付ファイル]\n{self._file_block}"
        # Level 1 で進化した few-shot を参考例として維持する
        if self._fewshot_block:
            prompt = f"{prompt}\n\n[参考例]\n{self._fewshot_block}"
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
        """ツールループの LLM 呼び出しをストリーミングで実行し ``(text, timed_out)`` を返す。

        非ストリーミング + 総ウォールクロック打ち切りだと低速 GPU が長い応答を生成しきる
        前に殺されるため、``_stream_text_with_idle_timeout`` (first-token / idle / total) を
        使う。``max_tokens`` は利用可能コンテキストに収める。後処理はしない (ツールコール
        JSON のパースに生テキストが必要なため)。
        """
        prompt_text = "".join(m.get("content", "") for m in injected_messages)
        ctx_size = resolve_context_size_for_mode(self.config, self._mode)
        sampling = {
            k: v for k, v in gen_kwargs.items()
            if k not in ("max_tokens", "id_slot", "stream")
        }
        text, timed_out = await self._stream_text_with_idle_timeout(
            llm_client, injected_messages,
            max_tokens=self._calc_gen_max_tokens(prompt_text, ctx_size),
            id_slot=gen_kwargs.get("id_slot", -1),
            **sampling,
        )
        if timed_out:
            logger.warning("LLM call timed out in tool loop iteration %d", loop)
        return text.strip(), timed_out

    async def _stream_text_with_idle_timeout(
        self,
        llm_client,
        messages: list[dict],
        *,
        max_tokens: int,
        id_slot: int = -1,
        **sampling,
    ) -> tuple[str, bool]:
        """ストリーミング生成をトークン間アイドルタイムアウトで読み取り ``(text, timed_out)`` を返す。

        最初の1トークンは ``content_gen_first_token_timeout``、以降はトークン間アイドル
        ``content_gen_idle_timeout`` で待ち、総上限 ``content_gen_timeout`` まで継続する。
        総ウォールクロックでは一律に打ち切らない (進行中の生成を殺さない)。無出力で停止した
        時だけ ``timed_out=True``。後処理はしない (生テキストを返す)。``_call_llm_in_loop`` /
        ``_fallback_plain_llm`` が共有する。
        """
        try:
            stream = await llm_client.generate(
                messages, stream=True,
                max_tokens=max_tokens, id_slot=id_slot, **sampling,
            )
        except asyncio.TimeoutError:
            return "", True
        agen = stream.__aiter__()
        chunks: list[str] = []
        start = time.monotonic()
        first_token = True
        try:
            while True:
                wait_timeout = (
                    self._content_gen_first_token_timeout
                    if first_token
                    else self._content_gen_idle_timeout
                )
                try:
                    token = await asyncio.wait_for(
                        agen.__anext__(), timeout=wait_timeout,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    return "", True
                first_token = False
                chunks.append(token)
                if time.monotonic() - start > self._content_gen_timeout:
                    return "", True
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
        return "".join(chunks), False

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
        if last_step and is_tool_error(last_step.output):
            return current + 1
        return 0

    async def _run_tool_loop(
        self,
        task: TaskItem,
        original_query: str,
        system_prompt: str,  # noqa: ARG002
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
        ctx_size = resolve_context_size_for_mode(self.config, self._mode)
        state.context_usage_pct = min(
            100,
            int(_estimate_tokens("x" * total_chars) / ctx_size * 100),
        )

    def _build_gen_kwargs(
        self, generation_params: dict | None,
    ) -> dict:
        """LLM 生成パラメータを組み立てる (stream は呼び出し側で指定)"""
        gen_kwargs: dict = {
            "max_tokens": self._execute_max_tokens,
            "id_slot": -1,
        }
        if generation_params:
            for k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "repetition_penalty"):
                if k in generation_params:
                    gen_kwargs[k] = generation_params[k]
        return gen_kwargs

    @staticmethod
    def _normalize_loop_tool_args(
        tool_name: str, tool_args: dict, original_query: str,
    ) -> dict:
        """`write_file` / `read_file` の args を正規化する。それ以外は素通し。"""
        if tool_name == "write_file":
            args = normalize_write_file_args(tool_args)
            fp = args.get("file_path", "")
            if fp:
                args["file_path"] = MetaCognitiveAgent._resolve_write_path(
                    fp, original_query,
                )
            return args
        if tool_name == "read_file":
            return normalize_read_file_args(tool_args, original_query)
        return tool_args

    @staticmethod
    def _resolve_write_path(file_path: str, query: str) -> str:
        """write_file の出力先を確定する。

        - 既存ディレクトリ指定 (例: C:\\...\\aa) → ``output_<UTC><ext>``
          (write_file はディレクトリをエラーにするため、書込み前にファイル名へ)
        - ディレクトリ成分の無い bare ファイル名で、クエリが出力ディレクトリを
          指定している場合 → そのディレクトリ配下へ寄せる (planner が CWD 相対の
          名前を発明したとき、ユーザー指定の場所へ揃える)
        """
        resolved = resolve_dir_output_path(file_path, query)
        if resolved != file_path:
            return resolved
        p = Path(file_path)
        if str(p.parent) in ("", "."):  # ディレクトリ成分の無い bare ファイル名
            from backend.free.agent.tool_call_judge import _extract_file_path
            qpath = _extract_file_path(query)
            if qpath and ("\\" in qpath or "/" in qpath):
                qp = Path(qpath)
                if qp.is_dir() or not qp.suffix:
                    return str(qp / p.name)
        return file_path

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
        is_error = is_tool_error(tool_result_text)
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
        generation_params: dict | None,  # noqa: ARG002
    ) -> tuple[str, list[dict]] | None:
        """ツールループ内の1回のツール実行

        ループ終了条件に達した場合は結果タプルを返す。
        ループ続行の場合は None を返し、messages を更新する。
        """
        tool_name = tool_call.get("tool", "")
        tool_args = self._normalize_loop_tool_args(
            tool_name, tool_call.get("args", {}), original_query,
        )

        # 取得専任タスク (collapse 済み) では write_file を実行しない。出力は後続の
        # write タスクが取得データを決定論的に書く。小型モデルが取得タスクの途中で
        # 余計な write_file を出してプレースホルダ/重複ファイルを生む退行を防ぐ。
        if getattr(task, "fetch_only", False) and tool_name == "write_file":
            logger.info(
                "fetch-only task: suppressing write_file to %s (delegated to "
                "write task)", tool_args.get("file_path", ""),
            )
            step_results.append(StepResult(
                tool_name="write_file",
                output=(
                    "Skipped: this step only fetches data; the fetched data is "
                    "written to the file by a later step. Do not write files here."
                ),
                iteration=loop,
            ))
            return None

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
        tool_call_entry["success"] = not is_tool_error(tool_result_text)

        state.pending_tool = None
        state.pending_args = {}

        await self._emit_loop_tool_result(on_step, prefix, tool_name, tool_result_text)

        # 連続エラー検出 → ループ打ち切り
        if is_tool_error(tool_result_text):
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
        # データ取得結果をタスク横断アキュムレータへ (write タスクの素材に再利用)。
        if tool_name in _DATA_BEARING_TOOLS and not is_tool_error(tool_result_text):
            self._fetched_tool_outputs.append(tool_result_text)
            # 取得専任タスクは取得成功時点で完了 (後続 write タスクへ委譲)。
            # ここで止めないと小型モデルが次ループで余計な write_file を出す。
            if getattr(task, "fetch_only", False):
                logger.info(
                    "fetch-only task: fetch succeeded, ending step "
                    "(write delegated): %s", tool_name,
                )
                return tool_result_text, tool_calls

        # write_file 成功 → タスク完了
        if not is_tool_error(tool_result_text) and tool_name == "write_file":
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
        messages: list[dict],  # noqa: ARG002
        consecutive_errors: int,
        max_consecutive_errors: int,
        loop: int,
    ) -> tuple[str, list[dict]] | None:
        """write_file の content が不足時にコンテンツを生成する

        生成失敗時は結果タプルを返す（ループ終了）。
        成功時は tool_args["content"] を更新して None を返す（ループ続行）。
        """
        file_path = tool_args.get("file_path", "")
        # 表計算/リッチ文書 (xlsx/csv/docx/pptx) 出力で、タスク横断で取得済みの実
        # テーブルがあれば、モデルに転記させず取得データを直接書き込む
        # (ハルシネーション/行脱落の防止、pptx でコード文字列を吐く退行の防止)。
        if wants_fetched_table(file_path):
            fetched_table = self._extract_fetched_table_markdown()
            if fetched_table:
                tool_args["content"] = fetched_table
                logger.info(
                    "write_file content from fetched table (deterministic): "
                    "%d chars -> %s", len(fetched_table), file_path,
                )
                return None
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
        content, rejection = self._validate_generated_content(content, file_path)
        if rejection and not content.startswith("(Content generation failed:"):
            logger.warning(
                "Tool-loop write: generated content rejected (%s), "
                "retrying content generation: %r",
                rejection, content[:120],
            )
            content = await self._generate_content(
                original_query, task.description, llm_client,
                file_path=file_path,
            )
            content, rejection = self._validate_generated_content(content, file_path)
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
        if rejection:
            logger.warning(
                "Tool-loop write: generated content still rejected (%s) "
                "after retry, aborting: %r",
                rejection, content[:120],
            )
            tool_result_text = (
                f"Error: Content generation produced invalid output ({rejection}), "
                "not actual content"
            )
            consecutive_errors += 1
            step_results.append(StepResult(
                tool_name=tool_name,
                output=tool_result_text,
                iteration=loop,
            ))
            if consecutive_errors >= max_consecutive_errors:
                return tool_result_text, tool_calls
            return None
        tool_args["content"] = content
        logger.info(
            "Content generated for write_file: %d chars → %s",
            len(content), file_path,
        )
        return None

    def _extract_fetched_table_markdown(self) -> str:
        """タスク横断で取得したツール結果から GFM テーブルを抽出・結合する。

        ``fetch_url`` 等が返した本文中の table ブロックを集め、複数テーブル
        (日付ごと等で繰り返されるヘッダ) を 1 ヘッダ + 全データ行へ正規化した
        GFM 文字列を返す。テーブルが無ければ空文字列。
        """
        from backend.export.content_converter import ContentConverter

        outputs = getattr(self, "_fetched_tool_outputs", [])
        header: list[str] | None = None
        data_rows: list[list[str]] = []
        for out in outputs:
            for block in ContentConverter().convert(out):
                if block.type != "table" or not block.rows:
                    continue
                if header is None:
                    header = block.rows[0]
                    data_rows.extend(block.rows[1:])
                else:
                    # 繰り返しヘッダ行はスキップして本文行のみ連結
                    start = 1 if block.rows[0] == header else 0
                    data_rows.extend(block.rows[start:])
        if header is None or not data_rows:
            return ""
        ncol = len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * ncol) + " |",
        ]
        for r in data_rows:
            cells = (list(r) + [""] * ncol)[:ncol]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
        tools_registry,
        state: AgentState,
    ) -> str:
        """ツールを実行して結果テキストを返す"""
        # search_code は coding 専用ツールだが (modes=["coding"])、ここは LLM
        # プランナーが自由選択するループ経路であり ToolDefinition.modes は元々
        # 参照されていない。chat モードの CWD 全域 os.walk が実インシデントの
        # 直接原因になったため、search_code に限り mode ゲートを追加する
        # (write_file は meta_cognitive の長文書き出し機能で chat モードからも
        # 正規に使われるため対象外)。
        if tool_name == "search_code" and not is_coding_mode(self._mode):
            result_text = f"Error: search_code is not available in mode '{self._mode}'"
            state.on_tool_failure(tool_name, result_text)
            logger.warning(
                "Tool not allowed in mode %s: %s", self._mode, tool_name,
            )
            return result_text
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
        is_success = not is_tool_error(tool_result_text)
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

        # 3. テンプレート形式 (<|tool_call>call:NAME(...)<tool_call|>) を試す。
        #    base モデルが OAI JSON でなくチャットテンプレート由来の生テキストで
        #    ツールコールを吐く場合 (tool_calls=0 の原因) を救済する。
        template = parse_template_tool_call(text)
        if template is not None:
            return template

        return None

    # ------------------------------------------------------------------
    # ファストパス実行
    # ------------------------------------------------------------------

    async def _execute_tool_fast(
        self,
        tool_name: str,
        tool_args: dict,
        task: TaskItem,  # noqa: ARG002
        tools_registry,
        on_step=None,
        prefix: str = "",
    ) -> tuple[str, list[dict]]:
        """read_file / run_command / search_code のファストパス実行"""
        logger.info("Tool fast path: %s(%s)", tool_name, tool_args)

        # search_code は coding 専用 (modes=["coding"]) だが、この判定は
        # ToolCallJudge の rule/assist 判定結果をそのまま実行するため
        # ToolDefinition.modes を経由しない。_execute_tool と同じ理由で
        # search_code のみ mode ゲートを追加する (write_file は対象外)。
        if tool_name == "search_code" and not is_coding_mode(self._mode):
            result_text = f"Error: search_code is not available in mode '{self._mode}'"
            logger.warning(
                "Tool fast path not allowed in mode %s: %s", self._mode, tool_name,
            )
            return result_text, [{
                "tool": tool_name,
                "args": tool_args,
                "success": False,
            }]

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
            is_success = not is_tool_error(result_text)
        except Exception as e:
            result_text = f"Error: {e}"
            is_success = False
            logger.error("Tool fast path failed: %s - %s", tool_name, e)

        # データ取得結果をタスク横断アキュムレータへ (後続 write タスクの素材に再利用)。
        # ファストパス経由の fetch_url 等もここで蓄積する (ツールループ経路と対称)。
        if tool_name in _DATA_BEARING_TOOLS and is_success:
            self._fetched_tool_outputs.append(result_text)

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
        # 出力先を確定 (ディレクトリ→output ファイル / bare 名→クエリ指定ディレクトリ配下)。
        # planner/judge が発明した CWD 相対の bare 名をユーザー指定の場所へ寄せる。
        file_path = self._resolve_write_path(file_path, original_query)
        logger.info("Write fast path: %s → %s", task.description[:60], file_path)

        # 表計算/リッチ文書 (xlsx/csv/docx/pptx) 出力で取得済みの実テーブルがあれば、
        # モデルに転記させず取得データを直接書き込む (小型モデルのハルシネーション/
        # 行脱落/拒否、pptx でコード文字列を吐く退行を回避)。
        if wants_fetched_table(file_path):
            fetched_table = self._extract_fetched_table_markdown()
            if fetched_table:
                logger.info(
                    "Write fast path content from fetched table "
                    "(deterministic): %d chars -> %s",
                    len(fetched_table), file_path,
                )
                return await self._write_file(
                    file_path, fetched_table, tools_registry, on_step, prefix,
                )

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
        content, rejection = self._validate_generated_content(content, file_path)

        if rejection and not content.startswith("(Content generation failed:"):
            logger.warning(
                "Write fast path: generated content rejected (%s), "
                "retrying content generation: %r",
                rejection, content[:120],
            )
            content = await self._generate_content(
                original_query, task.description, llm_client,
                file_path=file_path,
            )
            content, rejection = self._validate_generated_content(content, file_path)

        if content.startswith("(Content generation failed:"):
            logger.warning("Write fast path: content generation failed for %s", file_path)
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成失敗",
                    "status": "failed",
                })
            return f"Error: {content}", []

        if rejection:
            logger.warning(
                "Write fast path: generated content still rejected (%s) "
                "after retry, aborting: %r",
                rejection, content[:120],
            )
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成失敗（{rejection}）",
                    "status": "failed",
                })
            return (
                f"Error: Content generation produced invalid output ({rejection}), "
                "not actual content",
                [],
            )

        return await self._write_file(
            file_path, content, tools_registry, on_step, prefix,
        )

    @staticmethod
    def _validate_generated_content(
        content: str, file_path: str,
    ) -> tuple[str, str | None]:
        """生成コンテンツを scaffold 除去してから書込み適性を検証する。

        「タスクログ + 本文」の連結は本文だけに救済し、本文が残らない
        エコー (task_log_echo / prompt_echo / path_only / csv_without_rows)
        は棄却理由を返して呼出側で再生成・中断させる。
        """
        if content.startswith("(Content generation failed:"):
            return content, "generation_failed"
        stripped = strip_task_log_scaffold(content)
        if stripped and stripped != content:
            logger.info(
                "Write fast path: stripped task-log scaffold from generated "
                "content (%d -> %d chars)", len(content), len(stripped),
            )
            content = stripped
        return content, generated_content_rejection(content, file_path)

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
            is_success = not is_tool_error(result_text)
            if is_success:
                verify_error = self._verify_written_file(file_path, content)
                if verify_error:
                    result_text = f"Error: {verify_error}"
                    is_success = False
                    logger.error(
                        "write_file post-verification failed: %s (%s)",
                        file_path, verify_error,
                    )
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

    @staticmethod
    def _verify_written_file(file_path: str, content: str) -> str | None:
        """書込み後にディスク上の実ファイルを読み戻して内容を突合する。

        「Written N bytes」という成功申告と実ファイルの乖離 (書込み経路の
        取り違え・変換事故) を success にしないための最終ガード。リッチ文書
        (xlsx/docx 等) は export 変換で内容が変わるため対象外。改行は
        ``write_text`` のプラットフォーム変換 (LF→CRLF) を正規化して比較する。
        検証自体の失敗 (読み戻し不可等) は書込み失敗と区別できないため
        エラーにせず None (成功維持) を返す。
        """
        from backend.free.agent.tools.builtin import _EXPORT_DOC_EXTS

        try:
            p = Path(file_path)
            if p.suffix.lower() in _EXPORT_DOC_EXTS:
                return None
            if len(content) > 2_000_000:
                return None
            on_disk = p.read_text(encoding="utf-8")
        except Exception:
            return None
        if on_disk.replace("\r\n", "\n") != content.replace("\r\n", "\n"):
            return (
                f"post-write verification failed: on-disk content of "
                f"'{file_path}' does not match the generated content "
                f"({len(on_disk)} vs {len(content)} chars)"
            )
        return None

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

        # 出力先を確定 (ディレクトリ→output / bare→クエリ dir) し、表/CSV 出力で
        # 取得済みの実テーブルがあれば、モデルではなく取得データを直接使う (決定論)。
        file_path = self._resolve_write_path(file_path, original_query)
        fetched_table = (
            self._extract_fetched_table_markdown()
            if wants_fetched_table(file_path) else ""
        )

        logger.info(
            "Auto-recovery: LLM returned plain text for write task, "
            "extracting content for %s",
            file_path,
        )

        if fetched_table:
            content = fetched_table
            logger.info(
                "Auto-recovery content from fetched table (deterministic): "
                "%d chars -> %s", len(content), file_path,
            )
        else:
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

            content, rejection = self._validate_generated_content(content, file_path)
            if rejection:
                logger.warning(
                    "Auto-recovery: generated content rejected (%s), aborting: %r",
                    rejection, content[:120],
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
            is_success = not is_tool_error(result_text)
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
        ctx_size = resolve_context_size_for_mode(self.config, self._mode)

        existing_content = self._read_existing_file(file_path)
        user_prompt = f"{original_query}\n\nタスク: {task_description}"
        # 「今月」「今日」等の相対表現を取り違えないよう現在日付を前置する
        # (カレンダー/予定表など日付依存の成果物で年月・曜日がズレるのを防ぐ)。
        user_prompt = self._inject_current_date(user_prompt)
        # 前ステップで取得した実データ (fetch_url 等) を「使うべき素材」として注入し、
        # データに無い内容の創作 (ハルシネーション) を抑止する。
        user_prompt = self._inject_fetched_data(user_prompt, ctx_size)
        user_prompt = self._inject_existing_content(
            user_prompt, existing_content, file_path, ctx_size,
        )

        # Level 1 で進化した few-shot を参考例として system に注入する
        system_content = CONTENT_GENERATION_PROMPT
        # 出力先がスプレッドシート系なら GFM 表生成を強制する (xlsx 実セル化のため)。
        # .csv は raw 書込みのため CSV 行そのものを出力させる。
        # リッチ文書系 (docx/pptx) は表を強制せず、コード文字列の出力のみ禁じる。
        if file_path.lower().endswith(".csv"):
            system_content = f"{system_content}\n{CSV_CONTENT_INSTRUCTION}"
        elif is_table_output(file_path):
            system_content = f"{system_content}\n{TABLE_CONTENT_INSTRUCTION}"
        elif is_rich_table_output(file_path):
            system_content = f"{system_content}\n{RICH_DOC_CONTENT_INSTRUCTION}"
        # 出力言語指示 (locale 追従)。ここは write 全経路 (ツールループ /
        # ファストパス / auto-recovery / editor タスク) の合流点なので、
        # この 1 箇所で全ファイル出力に効く。
        system_content = f"{system_content}\n{content_language_directive()}"
        # Level 1 few-shot は query 類似度だけでなく fitness も加味して選ばれる
        # ため、現在のタスクと無関係でも再利用されうる。無関係な例文をその
        # まま注入すると、モデルがその例文自体を繰り返す退化を誘発しうる
        # (#incident) ため、タスク文との粗い関連度チェックを通してから注入する。
        # さらに応答例がタスク進捗ノート形式 (- [done] ... Written N bytes) の
        # 場合は「報告だけ出せば正解」バイアスを与えるため注入しない
        # (#incident 2026-07-15: 本文なし極小ファイル 10 件)。
        if (
            self._fewshot_block
            and fewshot_seems_relevant(
                f"{original_query}\n{task_description}", self._fewshot_block,
            )
            and not fewshot_contains_task_log(self._fewshot_block)
        ):
            system_content = f"{system_content}\n\n[参考例]\n{self._fewshot_block}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

        gen_max_tokens = self._calc_gen_max_tokens(
            system_content + user_prompt, ctx_size,
        )

        return await self._stream_and_clean(
            llm_client, messages, gen_max_tokens,
        )

    @staticmethod
    def _inject_current_date(user_prompt: str) -> str:
        """現在日付 (UTC 基準) を user_prompt 先頭に前置する。

        ``_generate_content`` は履歴を持たないため、モデルは現在日付を知らない。
        「今月のカレンダー」のような相対日付依存の生成で年月・曜日を取り違え
        ないよう、現在日付と曜日を明示する。内部時刻不変則 (naive 禁止) に従い
        ``utc_now_dt()`` を使う。
        """
        from backend.utils import utc_now_dt
        now = utc_now_dt()
        weekday = "月火水木金土日"[now.weekday()]
        date_ctx = (
            f"[現在日時 (UTC基準)] {now:%Y-%m-%d} ({weekday}曜)。"
            "「今月」「今日」等の相対表現はこの日付を基準に解釈すること。"
        )
        return f"{date_ctx}\n\n{user_prompt}"

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

    def _inject_fetched_data(self, user_prompt: str, ctx_size: int) -> str:
        """タスク横断で取得したツール結果を「使うべき実データ」として注入する。

        コンテキスト予算 (おおよそ ctx_size/2) に収まる範囲で取得データを付与し、
        モデルにデータ由来の出力を促す。予算不足なら付与しない (安全縮退)。
        """
        outputs = getattr(self, "_fetched_tool_outputs", [])
        if not outputs:
            return user_prompt
        combined = "\n\n".join(outputs)
        base_tokens = _estimate_tokens(CONTENT_GENERATION_PROMPT + user_prompt)
        budget_tokens = ctx_size // 2 - base_tokens
        if budget_tokens < 100:
            return user_prompt
        # token 予算をおおまかに char 予算へ換算 (日本語混在で ~2 char/token 見込み)
        snippet = combined[: budget_tokens * 2]
        return user_prompt + (
            "\n\n## 取得済みデータ (前ステップで取得した実データ)\n"
            "以下のデータのみを根拠に出力を生成し、データに無い情報は創作しないこと。\n"
            f"{snippet}"
        )

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
        """LLM ストリーミング生成 + 後処理（フェンス除去・繰り返し切除）

        最初の1トークンまでは別枠の長めタイムアウト
        (``content_gen_first_token_timeout``) で待つ。これは llama-server が他の
        生成で busy な間にキュー待ちしているリクエストを「停止」と誤判定しないため。
        2トークン目以降はトークン間アイドル (無出力) タイムアウト
        (``content_gen_idle_timeout``) で「停止したストリーム」を素早く諦めつつ、
        低速だが進行中の生成は総上限 (``content_gen_timeout``) まで継続させる。
        総ウォールクロックで一律に打ち切らない。
        """
        stream = await llm_client.generate(
            messages, stream=True,
            max_tokens=gen_max_tokens,
            id_slot=getattr(llm_client, 'background_slot', -1),
        )
        agen = stream.__aiter__()
        chunks: list[str] = []
        start = time.monotonic()
        first_token = True
        try:
            while True:
                wait_timeout = (
                    self._content_gen_first_token_timeout
                    if first_token
                    else self._content_gen_idle_timeout
                )
                try:
                    token = await asyncio.wait_for(
                        agen.__anext__(),
                        timeout=wait_timeout,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    if first_token:
                        logger.warning(
                            "Content generation produced no first token within "
                            "%ds (llama-server likely busy with another generation)",
                            self._content_gen_first_token_timeout,
                        )
                        return (
                            f"(Content generation failed: no output within "
                            f"{self._content_gen_first_token_timeout}s)"
                        )
                    logger.warning(
                        "Content generation stalled (no token for %ds)",
                        self._content_gen_idle_timeout,
                    )
                    return (
                        f"(Content generation failed: stalled after "
                        f"{self._content_gen_idle_timeout}s without output)"
                    )
                first_token = False
                chunks.append(token)
                if time.monotonic() - start > self._content_gen_timeout:
                    logger.warning(
                        "Content generation exceeded total cap %ds",
                        self._content_gen_timeout,
                    )
                    return f"(Content generation failed: timeout after {self._content_gen_timeout}s)"
            raw = "".join(chunks).strip()
            content = strip_markdown_wrapper(raw)
            content = truncate_repetition(content)
            if not content:
                logger.warning("Content generation returned empty content")
                return "(Content generation failed: empty output)"
            logger.debug("Content generated: %d chars", len(content))
            return content
        except Exception as e:
            logger.error("Content generation failed: %s", e)
            return f"(Content generation failed: {e})"
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

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
                # 露出させない。
                parts.append("    (書き込みが実行されませんでした)")
            elif task.result:
                parts.append(f"    {task.result[:500]}")

        return "\n".join(parts)
