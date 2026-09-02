"""TaskExecutionMixin — meta_cognitive_task_exec"""

from __future__ import annotations

import asyncio
import time

from pathlib import Path
from backend.config import resolve_context_size_for_mode
from backend.free.agent.agent_state import AgentState
from backend.free.agent.meta_cognitive_tasks import (
    TaskItem,
    task_expects_write,
)
from backend.free.agent.meta_cognitive_tools import (
    normalize_read_file_args,
    normalize_write_file_args,
)
from backend.free.agent.output_format import resolve_dir_output_path
from backend.free.agent.meta_cognitive_utils import (
    call_callback,
    is_tool_error,
    summarize_tool_args,
    tool_result_succeeded,
)
from backend.free.agent.step_compactor import StepResult
from backend.free.api.chat.chat_constants import (
    TOOL_EXECUTION_TIMEOUT_SEC,
    TOOL_RESULT_HEAD_RATIO,
    TOOL_RESULT_MAX_CHARS,
    TOOL_RESULT_OMISSION_CHARS,
)
from backend.free.core.inference import build_messages_for_loop
from backend.utils import estimate_tokens as _estimate_tokens

from backend.free.agent.meta_cognitive_defs import (
    EXECUTE_SYSTEM_PROMPT,
    _DATA_BEARING_TOOLS,
    _is_placeholder_write_path,
    resolve_read_path,
)

from backend.log_config import get_logger

logger = get_logger("agent.meta_cognitive")


def tool_mode_error(tools_registry, tool_name: str, mode: str) -> str | None:
    """``ToolDefinition.modes`` に基づく実行時の mode ゲート (deliberative と同じ規則)。

    ``modes`` は元々 LLM 向け説明文のフィルタにしか使われず、meta 経路の
    実行時には無視されていた (search_code だけ個別にガードしていた)。
    create 専用ツール (run_command / apply_diff / verify_syntax 等) が chat
    モードのタスクから実行されないよう、登録済み定義があれば必ず照合する。
    例外は ``write_file`` のみ: 選択は create 限定だが chat の書き出し経路から
    正規に実行されるため、``inventory_modes`` に載っているモードでは許可する
    (:attr:`ToolDefinition.inventory_modes` 参照)。

    Returns:
        拒否時は ``Error:`` 文字列、許可時は None。
    """
    tool_def = tools_registry.get(tool_name)
    if tool_def is None or mode in tool_def.modes:
        return None
    if tool_name == "write_file" and tool_def.listed_in(mode):
        return None
    logger.warning(
        "Tool not allowed in mode=%s: %s (allowed modes: %s)",
        mode, tool_name, tool_def.modes,
    )
    return f"Error: {tool_name} is not available in mode '{mode}'"


async def execute_tool_with_timeout(
    tools_registry, tool_name: str, tool_args: dict,
) -> str:
    """ツールを timeout 付きで実行し結果テキストを返す (deliberative と同じ規則)。

    timeout は ``ToolsRegistry.timeout_for`` (ツール宣言 > 既定 30 秒)。
    超過時は ``Error:`` 文字列を返し、``tool_result_succeeded`` が失敗と
    扱えるようにする。timeout 以外の例外は呼出側で扱う (経路ごとに
    state / ログの処理が異なるため)。
    """
    timeout_sec = tools_registry.timeout_for(tool_name, TOOL_EXECUTION_TIMEOUT_SEC)
    try:
        result = await asyncio.wait_for(
            tools_registry.execute(tool_name, **tool_args), timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Tool execution timed out: %s (%.0fs)", tool_name, timeout_sec,
        )
        return f"Error: tool '{tool_name}' timed out after {timeout_sec:g}s"
    return str(result)


def truncate_tool_result(text: str, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    """ツール結果が max_chars を超える場合、先頭と末尾を残して切り詰める。

    deliberative の ``_truncate_tool_result`` と同じ体裁。meta のツールループは
    最新 step の出力を全文で LLM に渡す (StepCompactor は最新反復を圧縮しない)
    ため、巨大な結果がそのままコンテキストを食い潰さないようここで上限を掛ける。
    """
    if len(text) <= max_chars:
        return text
    head_size = int(max_chars * TOOL_RESULT_HEAD_RATIO)
    tail_size = max_chars - head_size - TOOL_RESULT_OMISSION_CHARS
    omitted = len(text) - head_size - tail_size
    return (
        text[:head_size]
        + f"\n\n... ({omitted} chars omitted) ...\n\n"
        + text[-tail_size:]
    )


class _TaskExecutionMixin:
    """タスク 1 件の実行 — ツールループ / LLM 呼び出し / ツール実行。

    計画された ``TaskItem`` を 1 件受け取り、ツールを撃ちながら完了まで
    運ぶ層。``MetaCognitiveAgent`` の一部として mixin される
    (``self`` は同一インスタンスで、他の責務のメソッドも参照できる)。
    """

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
            # パスが無い (「同じファイルに保存し直して」型) か、ディレクトリを
            # 伴わない裸のファイル名 (「notes.txt に追記して」) は保存先が
            # 確定していないので直近会話から引く。ディレクトリ付きパスはその
            # まま返るので無条件に通してよい (実測 2026-07-27: パス不明のまま
            # 生成だけ走りファイルは旧内容のままだった / 2026-08-09: 裸名を
            # そのまま渡すとカレントディレクトリに別物を作る)。
            file_path = judgement.tool_args.get("file_path", "")
            file_path = self._referential_write_path(file_path or None) or file_path
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
                original_query=original_query,
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
            # 1 回の process() 内でタスク/反復ごとに再構築されていたので mode 別に cache
            cache = self._tool_descriptions_cache
            if self._mode not in cache:
                cache[self._mode] = tools_registry.get_descriptions_text(mode=self._mode)
            tool_descriptions = cache[self._mode]
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
        conversation: list[dict] | None = None,
    ) -> dict:
        """`write_file` / `read_file` の args を正規化する。それ以外は素通し。"""
        if tool_name == "write_file":
            args = normalize_write_file_args(tool_args)
            fp = args.get("file_path", "")
            if fp:
                args["file_path"] = _TaskExecutionMixin._resolve_write_path(
                    fp, original_query,
                )
            return args
        if tool_name == "read_file":
            args = normalize_read_file_args(tool_args, original_query)
            fp = args.get("file_path", "")
            if fp:
                args["file_path"] = _TaskExecutionMixin._resolve_read_path(
                    fp, original_query, conversation,
                )
            return args
        return tool_args

    @staticmethod
    def _resolve_read_path(
        file_path: str, original_query: str, conversation: list[dict] | None,
    ) -> str:
        """read_file の対象パスを文脈から解決する (裸のファイル名の救済)。

        書込み側は ``_resolve_write_path`` + ``_referential_write_path`` で
        裸の名前を会話中のフルパスへ寄せているのに、**読取側にはその解決が
        無かった**。``_resolve_referenced_path`` の docstring は最初から
        「書込み/読取の対象パスを会話から解決する」と書いており、読取だけ
        配線が漏れていた。

        実インシデント (2026-08-26 ライブ監査の修正検証): 「mrg_b.txt の中身を
        mrg_a.txt の末尾に追記してください。」でプランナーが裸の名前で
        ``["Read the content of mrg_b.txt", ...]`` を出し、
        ``read_file({'file_path': 'mrg_b.txt'})`` が **File not found** で失敗した
        (ディレクトリはこのターンのクエリに無く、前のターンの会話にしかない)。
        読み取りが失敗すると ``is_tool_error`` で ``_fetched_tool_outputs`` にも
        入らないため、書込み内容の供給元が空のままになる。
        """
        resolved = resolve_read_path(file_path, original_query, conversation)
        if resolved and resolved != file_path:
            logger.info(
                "Read target resolved from conversation: %s (bare=%r)",
                resolved, file_path,
            )
            return resolved
        return file_path

    @staticmethod
    def _resolve_write_path(file_path: str, query: str) -> str:
        """write_file の出力先を確定する。

        - 引数名プレースホルダ (``file_path`` / ``<path>`` 等) → クエリ中の
          明示パスへ差し替える
        - 既存ディレクトリ指定 (例: C:\\...\\aa) → ``output_<UTC><ext>``
          (write_file はディレクトリをエラーにするため、書込み前にファイル名へ)
        - ディレクトリ成分の無い bare ファイル名で、クエリが出力ディレクトリを
          指定している場合 → そのディレクトリ配下へ寄せる (planner が CWD 相対の
          名前を発明したとき、ユーザー指定の場所へ揃える)
        """
        from backend.free.agent.tool_call_judge import _extract_file_path

        if _is_placeholder_write_path(file_path):
            # 小型 aux はパラメータ名をそのまま値として返すことがある
            # (実インシデント 2026-07-28 ライブ検証:
            # `{"tool": "write_file", "args": {"file_path": "file_path", ...}}`
            # がそのまま実行され、リポジトリ直下に `file_path` という名前の
            # ファイルが作られた)。クエリに明示パスがあればそこへ寄せる。
            qpath = _extract_file_path(query)
            if qpath:
                logger.warning(
                    "Write path placeholder %r replaced with query path %s",
                    file_path, qpath,
                )
                return qpath
            logger.warning(
                "Write path is a parameter-name placeholder (%r) and the query "
                "has no explicit path; falling through to normal resolution",
                file_path,
            )
        resolved = resolve_dir_output_path(file_path, query)
        if resolved != file_path:
            return resolved
        p = Path(file_path)
        if str(p.parent) in ("", "."):  # ディレクトリ成分の無い bare ファイル名
            qpath = _extract_file_path(query)
            if qpath and ("\\" in qpath or "/" in qpath):
                qp = Path(qpath)
                if qp.is_dir() or not qp.suffix:
                    return str(qp / p.name)
                if p.name not in query:
                    # クエリが挙げているのは別のファイルで、この bare 名は
                    # planner の発明。そのまま書くとプロセスの CWD
                    # (= リポジトリ直下) にゴミが残る (実インシデント
                    # 2026-07-29 ライブ監査: 「E:\tmp\audit_r4b.md の5番目の
                    # 項目を削除して保存し直してください。」が 2 タスクに割れ、
                    # 2 番目が `document.txt` へタスク文を書き込んだ)。
                    # 少なくともユーザーが作業しているディレクトリへ寄せる。
                    logger.warning(
                        "Invented bare write path %r redirected next to the "
                        "query path %s", file_path, qpath,
                    )
                    return str(qp.parent / p.name)
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
        # deliberative と同じ規則: run_command の非ゼロ終了 (``[exit code: N]``)
        # も失敗として表示する (Error: プレフィックスだけでは ✓ になっていた)。
        succeeded = tool_result_succeeded(tool_name, tool_result_text)
        await call_callback(on_step, {
            "type": "tool_call",
            "detail": f"{prefix} {tool_name}: {tool_result_text[:100]}",
            "status": "done" if succeeded else "failed",
        })

    async def _execute_loop_tool_call(
        self,
        tool_call: dict,
        text: str,  # noqa: ARG002
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
            getattr(self, "_conversation", None),
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
        tool_call_entry["success"] = tool_result_succeeded(
            tool_name, tool_result_text,
        )

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
            output=truncate_tool_result(tool_result_text),
            iteration=loop,
        ))
        # データ取得結果をタスク横断アキュムレータへ (write タスクの素材に再利用)。
        # ここは書込み素材なので切り詰めない (全文を保持する)。
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

        # 再ループ。次反復の messages は step_results から ``_rebuild_loop_messages``
        # が再構築する (ここで messages に追記しても上書きされて LLM に届かない)。
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

        async def _notify_generating() -> None:
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} コンテンツ生成中 → {file_path}",
                    "status": "running",
                })

        content, rejection = await self._resolve_write_content(
            file_path=file_path,
            original_query=original_query,
            task_description=task.description,
            llm_client=llm_client,
            notify_generating=_notify_generating,
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
        if tools_registry is not None and tools_registry.has(tool_name):
            # ToolDefinition.modes は元々 LLM 向け説明文のフィルタにしか使われず、
            # LLM プランナーが自由選択するこのループ経路では実行時に無視されて
            # いた (chat モードの search_code による CWD 全域 os.walk が実
            # インシデント)。deliberative と同じ規則で全ツールを照合する。
            mode_error = tool_mode_error(tools_registry, tool_name, self._mode)
            if mode_error is not None:
                state.on_tool_failure(tool_name, mode_error)
                return mode_error
            try:
                result_text = await execute_tool_with_timeout(
                    tools_registry, tool_name, tool_args,
                )
                if is_tool_error(result_text):
                    state.on_tool_failure(tool_name, result_text)
                else:
                    state.on_tool_success(tool_name)
            except Exception as e:
                result_text = f"Error: {e}"
                state.on_tool_failure(tool_name, str(e))
                logger.warning("Tool execution failed: %s - %s", tool_name, e)
        else:
            result_text = f"Error: Unknown tool '{tool_name}'"
            state.on_tool_failure(tool_name, result_text)
        return result_text
