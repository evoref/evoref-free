"""Deliberative エージェント: LLM 推論 + ツール判定で応答（2〜10秒）"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from backend.free.agent.agent_state import AgentState
from backend.free.agent.event_reminder import EventReminderSystem
from backend.free.agent.meta_cognitive_utils import (
    command_run_failed,
    content_language_directive,
    generated_content_rejection,
    is_tool_error,
    strip_markdown_wrapper,
)
from backend.free.agent.tool_call_judge import ToolCallJudge, ToolJudgement
from backend.free.agent.tool_result_digest import digest_tool_result
from backend.free.core.session_mode import is_coding_mode
from backend.config import resolve_context_size_for_mode
from backend.free.api.chat.chat_constants import (
    CONTENT_MAX_TOKENS_MIN, CONTENT_SYSTEM_RESERVE,
    TOOL_EXECUTION_TIMEOUT_SEC, TOOL_GROUNDED_TEMPERATURE,
    TOOL_RESULT_MAX_CHARS,
    TOOL_RESULT_HEAD_RATIO, TOOL_RESULT_OMISSION_CHARS,
)
from backend.free.api.chat.chat_types import GenerationParams, StepCallback
from backend.log_config import get_logger

logger = get_logger("agent.deliberative")

# write_file でコンテンツ生成が必要な場合のプロンプト
_CONTENT_GEN_PROMPT = """\
Generate the requested content below. Output ONLY the content itself, \
no explanations, no markdown fences, no JSON, no surrounding text.
"""

# digest_tool_result が NO_RELEVANT_INFO と確定した場合に raw の代わりに
# ツール実行結果として渡すプレースホルダ。無関係な内容を「唯一の事実根拠」
# として base に読ませないための安全な代替文言。
_NO_RELEVANT_INFO_MESSAGE = "（ツールを実行しましたが、今回の質問に関連する情報は見つかりませんでした）"

# search_history が空振りした場合専用のグラウンディング文言。通常ツールの
# 「唯一の事実根拠として扱う」枠をそのまま付けると、直前ターンで述べられた
# 情報 (SemMem 未反映のまだ生の会話履歴) まで無視して「見つかりません」と
# 誤答する (実インシデント 2026-07-23: 直前ターンで伝えた氏名・出身地を
# 聞き直され、search_history 空振りを理由に誤って「記録が無い」と回答した)。
# search_history は過去セッションのみを検索するツールであることを明示し、
# 今回進行中の会話履歴の参照を妨げないようにする。
_SEARCH_HISTORY_NO_INFO_GUIDANCE = (
    "search_history は過去の別セッションの会話記録を検索するツールである。"
    "上記の「関連する情報は見つかりませんでした」は、過去の別セッションには"
    "見つからなかったという意味に過ぎず、今回進行中のこの会話で既に述べられた"
    "情報 (直前のユーザー発言を含む会話履歴) を否定するものではない。"
    "会話履歴に該当情報があれば、検索結果とは関係なくそれを使って具体的に"
    "回答すること。会話履歴にも本当に無い場合のみ「わからない」と答えてよい。"
)


def _check_path_traversal(file_path: str, tool_name: str) -> str | None:
    """write_file / read_file のパス検証。違反時はエラーメッセージを返す。

    `..` セグメントを含むパスを拒否することでワークスペース外への
    アクセスを防止する。検査対象外なら ``None``。
    """
    if not file_path or tool_name not in ("write_file", "read_file"):
        return None
    try:
        normalized = file_path.replace("\\", "/")
        if ".." in normalized.split("/"):
            logger.warning(
                "Path traversal detected in tool args: %s", file_path,
            )
            return f"Error: path traversal not allowed: {file_path}"
    except (AttributeError, TypeError):
        pass
    return None


def _emit_tool_running_step(
    on_step: StepCallback, tool_name: str, tool_args: dict,
) -> None:
    """ツール実行開始の step フレームを emit する。"""
    if on_step is None:
        return
    from backend.free.agent.meta_cognitive_utils import summarize_tool_args
    on_step({
        "type": "tool_call",
        "detail": f"{tool_name}({summarize_tool_args(tool_name, tool_args)})",
        "status": "running",
    })


def _emit_tool_result_step(
    on_step: StepCallback, tool_name: str, result_text: str,
) -> None:
    """ツール正常完了 (`task_result`) の step フレームを emit する。"""
    if on_step is None:
        return
    logger.debug(
        "Tool result step: tool=%s, result=%s",
        tool_name, result_text[:120],
    )
    on_step({
        "type": "task_result",
        "detail": f"{tool_name}: {result_text[:100]}",
        "status": "done",
    })


def _emit_tool_failure_step(
    on_step: StepCallback, tool_name: str, error_text: str,
) -> None:
    """ツール失敗 / タイムアウトの step フレームを emit する。"""
    if on_step is None:
        return
    on_step({
        "type": "tool_call",
        "detail": f"{tool_name}: {error_text[:100]}",
        "status": "failed",
    })


@dataclass
class DeliberativeResponse:
    """Deliberative 層の応答"""
    content: str
    rag_used: bool = False
    rag_source: str | None = None
    rag_chunks: list[tuple[str, float, str]] = field(default_factory=list)
    tool_result: str | None = None
    tool_name: str | None = None
    # executable command 学習用 (run_command 実行ターンのみ非 None)
    tool_command: str | None = None
    tool_command_success: bool | None = None


class DeliberativeAgent:
    """Deliberative 層: LLM 推論 + アシストモデルによるツール判定

    ToolCallJudge によるツール呼び出し判定を実行し、
    ツール結果をコンテキストとして注入してから LLM に応答を生成させる。

    write_file でコンテンツ生成が必要な場合は、LLM にプレーンテキストで
    コンテンツを生成させてからツールを実行する（JSON 内にコンテンツを含めない）。
    目標応答時間: 2〜10秒（ツールなし） / 60〜120秒（コンテンツ生成+ツール）
    """

    def __init__(
        self,
        config: dict | None = None,
        tool_judge: ToolCallJudge | None = None,
        tools_registry=None,
        assist_client=None,
        assist_experience_recorder=None,
        agent_tracer=None,
        mode: str = "chat",
    ):
        self.config = config or {}
        self.reminder_system = EventReminderSystem(self.config)
        self._tool_judge = tool_judge
        self._tools_registry = tools_registry
        # _execute_tool の mode ゲートの既定値。process() 呼び出し毎の実際の mode は
        # _judge_and_execute_tool から明示的に渡される (こちらは直接 _execute_tool を
        # 呼ぶ既存テスト等のフォールバック用)。
        self._mode = mode
        # ツール結果の query 連動抽出 (base の接地負荷軽減) に使う。None なら raw を渡す。
        self._assist_client = assist_client
        # assist 由来ツール判定の実行成否を assist 経験へ記録する closure。
        # Pro/Develop 起動時のみ非 None (factory 層が注入)。None なら記録 no-op。
        self._assist_experience_recorder = assist_experience_recorder
        # MDP トレース。develop モード時のみ非 None (factory 層が注入)。
        # deliberative の tool 判定/実行を 1 step エピソードとして記録し、
        # sleep-time Step 7.5 が episodic LTM へ取込 → Level 1 agent ドメインの
        # 学習信号にする (これが無いと agent ドメインは skipped_no_signal)。
        self._agent_tracer = agent_tracer

        # コンテンツ生成用の max_tokens (coding 時は coding_model の実窓に合わせる)
        ctx_size = resolve_context_size_for_mode(self.config, mode)
        self._content_max_tokens = max(ctx_size - CONTENT_SYSTEM_RESERVE, CONTENT_MAX_TOKENS_MIN)

    @staticmethod
    def _init_deliberative_state(mode: str) -> AgentState:
        """`process` 用 AgentState を生成。`coding` モードは unified_diff を期待。"""
        return AgentState(
            agent_layer="deliberative",
            expected_format="unified_diff" if is_coding_mode(mode) else None,
        )

    @staticmethod
    def _append_tool_result_to_last_user(
        messages: list[dict],
        tool_name: str,
        tool_result_text: str,
        query: str | None = None,
    ) -> None:
        """最後の user メッセージにツール実行結果を追記する。

        system ロールを assistant の後に挿入すると Qwen3.5 等の ChatML
        テンプレートで 400 エラーになるため、必ず user に統合する。
        """
        truncated = _truncate_tool_result(tool_result_text, TOOL_RESULT_MAX_CHARS)
        # 話題再フォーカス: 弱いモデルは前ターンの話題に引きずられ、今回の質問
        # (例: ニュース) を取り違える (実機確認: 前ターンが天気だとニュース質問に
        # 天気で誤答)。今回の質問を明示して前話題を無視させる。
        refocus = ""
        if query:
            q = query if len(query) <= 200 else query[:200] + "…"
            refocus = (
                f"今回ユーザーが答えてほしい質問は『{q}』である。"
                f"会話履歴の前の話題は無関係なので無視し、この質問にのみ答えること。\n"
            )
        if tool_name == "search_history" and tool_result_text == _NO_RELEVANT_INFO_MESSAGE:
            # 空振りに「唯一の事実根拠」枠を付けると直前ターンの内容まで
            # 無視されるため、search_history 空振り専用の文言に差し替える
            # (通常ツールの capability assertion は付けない)。
            grounding = _SEARCH_HISTORY_NO_INFO_GUIDANCE
        else:
            # capability assertion: 弱いモデルは「自分はブラウズ/取得できない」という
            # 思い込みでツール結果を無視し拒否することがある (実機確認)。結果が
            # 実際に取得された本物データであると明示して上書きする。
            grounding = (
                f"上記の ## ツール実行結果 は、システムが {tool_name} ツールで実際にアクセスして"
                f"取得した本物のデータである。あなたにはこのツールがあり、取得は既に成功している。"
                f"この結果を唯一の事実根拠として、内容 (数値・名称・日付・条件など) を読み取り、"
                f"それに基づいて具体的に回答すること。"
                f"「ブラウズできない」「取得できない」「アクセスできない」とは言わないこと。"
                f"「取得できない」「データがない」と答えてよいのは、結果が空かエラーの場合のみ。"
                f"結果に該当が無い場合のみ、システムプロンプトの参考コンテキスト (カートリッジ・記憶等) も併用してよい。"
                f"結果に無い数値・事実は創作しないこと。"
            )
        tool_msg = (
            f"\n\n## ツール実行結果\n"
            f"ツール: {tool_name}\n"
            f"結果:\n{truncated}\n\n"
            f"{refocus}"
            f"{grounding}"
        )
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i] = {
                    "role": "user",
                    "content": messages[i]["content"] + tool_msg,
                }
                break

    def _record_tool_call_outcome(
        self, query: str, judgement: ToolJudgement, success: bool, mode: str = "chat",
    ) -> None:
        """assist 由来ツール判定の実行成否を assist 経験へ記録する (best-effort)。

        rule / learned / cartridge 由来は assist モデル出力ではないため記録しない
        (assist=B が学ぶのは assist のツール判定のみ)。recorder 未注入
        (Free / --no-learning) なら no-op。例外は recorder 側で握り潰される。
        ``mode`` は呼び出し元 (``_judge_and_execute_tool`` の明示引数、
        ``self._mode`` ではなく実際の処理対象モード) をそのまま渡す。
        """
        rec = self._assist_experience_recorder
        if rec is None or judgement.source != "assist":
            return
        rec("tool_call", query, judgement.tool_name or "", 1.0 if success else 0.0, mode)

    async def _judge_and_execute_tool(
        self,
        query: str,
        mode: str,
        conversation: list[dict] | None,
        messages: list[dict],
        llm_client,
        state: AgentState,
        on_step: StepCallback,
        tool_judge_task: "asyncio.Task | None" = None,
        session_id: str = "",
    ) -> tuple[str | None, str | None, str | None, bool | None]:
        """ツール判定 → 実行 → messages へのツール結果注入を一括で行う。

        ``tool_judge_task`` が渡された場合は chat() が先行起動した tool 判定
        タスクを await して再利用する (直列待ちの短縮)。タスクが例外で終わった
        場合は直接 judge を再実行してフォールバックする (挙動同等性優先)。
        ``session_id`` は search_history のセッション自己参照スコープ限定用
        (``ToolCallJudge._maybe_scope_session_search`` 参照)。

        Returns:
            ``(tool_result_text, tool_name, command, success)``。
            ツール不要時は ``(None, None, None, None)``。``command`` は
            run_command 系の ``tool_args["command"]`` (それ以外は None)、
            ``success`` は実行成功か (出力が "Error:" prefix でない)。
            executable_command 学習 (sleep-time curator) のデータ源になる。
        """
        if self._tool_judge is None or self._tools_registry is None:
            # 判定経路が無いなら precomputed タスクも使えない。残っていれば破棄。
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            return None, None, None, None

        if tool_judge_task is not None:
            try:
                judgement = await tool_judge_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Precomputed tool judge task failed, re-judging: %r", exc,
                )
                judgement = await self._tool_judge.judge(
                    query, self._tools_registry, mode, conversation or [],
                    session_id=session_id,
                )
        else:
            judgement = await self._tool_judge.judge(
                query, self._tools_registry, mode, conversation or [],
                session_id=session_id,
            )
        if not (judgement.tool_needed and judgement.tool_name):
            return None, None, None, None

        command = None
        if isinstance(judgement.tool_args, dict):
            cmd = judgement.tool_args.get("command")
            command = cmd if isinstance(cmd, str) and cmd else None

        tool_result_text = await self._execute_tool(
            judgement, state, query, llm_client, on_step, mode=mode,
        )
        if tool_result_text is None:
            # 実行されたが結果 None (失敗)。command は penalize 用に返す。
            self._record_tool_call_outcome(query, judgement, False, mode=mode)
            return None, judgement.tool_name, command, False

        # ツール結果を assist で query 連動抽出し、base 文脈には digest を注入して
        # 弱い base の接地負荷を下げる。抽出不能/assist 不在時は raw へ退避 (現挙動)。
        # 戻り値の tool_result_text(raw) は UI 表示用にそのまま保つ。
        digest = await digest_tool_result(
            self._assist_client,
            query=query,
            tool_name=judgement.tool_name,
            tool_result=_truncate_tool_result(tool_result_text, TOOL_RESULT_MAX_CHARS),
        )
        if digest is None:
            prompt_result_text = tool_result_text
        elif digest == "" and judgement.tool_name == "search_history":
            # assist が抽出に成功した上で「関連情報なし」と確定したケース。
            # search_history に限り、raw 結果 (無関係な過去セッションの内容等)
            # をそのまま「唯一の事実根拠」として base に渡すと誤って参照・混同
            # されるため (実インシデント: search_history が別セッションの雑談を
            # ヒットし、base がそれを今回の会話の内容として回答した)、raw へは
            # 退避しない。他のツール (calculate 等) は raw が「今回の呼出し
            # そのものの結果」であり無関係な混同のリスクが無いため、assist の
            # digest 誤判定 (false negative) を安全側に倒せるよう raw へ退避する
            # (元の挙動を維持)。
            prompt_result_text = _NO_RELEVANT_INFO_MESSAGE
        elif digest == "":
            prompt_result_text = tool_result_text
        else:
            prompt_result_text = digest
        self._append_tool_result_to_last_user(
            messages, judgement.tool_name, prompt_result_text, query=query,
        )
        success = not is_tool_error(tool_result_text)
        # run_command は走ったが非ゼロ終了したケース (SyntaxError 等) を
        # is_tool_error が拾えない。exit code マーカーで失敗を反映し、
        # 誤った success が executable_command の SemMem 学習を汚染するのを防ぐ。
        if success and judgement.tool_name in ("run_command", "run_command_readonly"):
            success = not command_run_failed(tool_result_text)
        logger.info(
            "Tool executed: %s, result_length=%d, source=%s, success=%s",
            judgement.tool_name, len(tool_result_text), judgement.source,
            success,
        )
        self._record_tool_call_outcome(query, judgement, success, mode=mode)
        return tool_result_text, judgement.tool_name, command, success

    async def process(
        self,
        query: str,
        messages: list[dict],
        llm_client,
        *,
        mode: str = "chat",
        stream: bool = True,
        conversation: list[dict] | None = None,
        max_tokens: int | None = None,
        on_step: StepCallback = None,
        generation_params: GenerationParams | None = None,
        tool_capture: dict | None = None,
        tool_judge_task: "asyncio.Task | None" = None,
        session_id: str = "",
    ) -> DeliberativeResponse | AsyncIterator[str]:
        """Deliberative 層で LLM 推論を実行

        Args:
            query: ユーザーのクエリ
            messages: build_messages() で組み立て済みのメッセージ配列
            llm_client: LocalClient インスタンス
            mode: 動作モード ('chat' | 'coding')
            stream: ストリーミング応答を返すか
            conversation: 直近の会話履歴（ツール判定の精度向上用）
            max_tokens: 最大生成トークン数
            on_step: ステップ進行コールバック (step_dict) -> None
            generation_params: モード別生成パラメータ（temperature, top_p 等）
            tool_judge_task: chat() が先行起動した tool 判定タスク (並列化時)。
                None なら判定をここで直列実行する。

        Returns:
            stream=False: DeliberativeResponse
            stream=True: AsyncIterator[str]（生トークンのイテレータ）
        """
        logger.debug(
            "process: query=%r, messages=%d, stream=%s, mode=%s",
            query[:50], len(messages), stream, mode,
        )

        state = self._init_deliberative_state(mode)
        (
            tool_result_text, tool_name_used, tool_command, tool_success,
        ) = await self._judge_and_execute_tool(
            query, mode, conversation, messages, llm_client, state, on_step,
            tool_judge_task=tool_judge_task, session_id=session_id,
        )

        # MDP トレース: tool 判定/実行を 1 step エピソードとして記録する。
        # ``_judge_and_execute_tool`` は stream 返却前に完了済みのため、ここで
        # begin→step→end を同期完結できる (生成は応答であり agent action ではない)。
        self._trace_tool_episode(
            session_id, mode, query, tool_name_used, tool_result_text, tool_success,
        )

        # streaming 経路は DeliberativeResponse を返さないため、command を
        # 呼出側へ渡す唯一の経路として tool_capture dict に書き出す。
        # ``_judge_and_execute_tool`` は iterator 返却前に完了するので、
        # ``await process(...)`` 完了時点で dict は確定している。
        if tool_capture is not None:
            tool_capture["command"] = tool_command
            tool_capture["command_name"] = tool_name_used if tool_command else None
            tool_capture["success"] = tool_success

        # ツール結果に基づく接地回答は創作不要。chat 既定 0.7 のままだと weak base が
        # 非決定的に拒否/話題混同しやすい (実機: ニュースで 0.7→~25%拒否、0.2→安定)。
        # ツール使用ターンのみ温度を下げて決定性を上げる (既に低ければ据え置く)。
        if tool_result_text is not None:
            gp = dict(generation_params or {})
            gp["temperature"] = min(
                gp.get("temperature", TOOL_GROUNDED_TEMPERATURE),
                TOOL_GROUNDED_TEMPERATURE,
            )
            generation_params = gp

        # リマインダー注入
        messages = self.reminder_system.inject(messages, state)
        logger.debug(
            "Messages finalized: %d messages, total_chars=%d",
            len(messages),
            sum(len(m.get("content", "")) for m in messages),
        )

        if stream:
            return self._stream_response(
                messages, llm_client, max_tokens,
                tool_result=tool_result_text, tool_name=tool_name_used,
                generation_params=generation_params,
            )
        return await self._sync_response(
            messages, llm_client, max_tokens,
            tool_result=tool_result_text, tool_name=tool_name_used,
            tool_command=tool_command, tool_command_success=tool_success,
            generation_params=generation_params,
        )

    def _trace_tool_episode(
        self,
        session_id: str,
        mode: str,
        query: str,
        tool_name: str | None,
        tool_result: str | None,
        tool_success: bool | None,
    ) -> None:
        """deliberative の tool 実行を 1 step MDP エピソードとして記録する。

        tracer 未注入 (通常起動) / session_id 空 / tool 未実行なら no-op。
        tool を実行したターンのみを記録対象とし、no_tool ルーティング signal は
        record_response の tool_routing_success (経験記録) 側に委ねて episodic LTM
        の膨張を避ける。reward は tool 実行成否。
        """
        tracer = self._agent_tracer
        if tracer is None or not session_id or tool_result is None:
            return
        from backend.free.agent.agent_tracer import MDPStep

        reward = 1.0 if tool_success else 0.0
        try:
            episode_id = tracer.begin_episode(session_id, mode)
            tracer.record_step(episode_id, MDPStep(
                step_index=0,
                state={"query": query[:200], "agent_layer": "deliberative"},
                action=tool_name or "tool",
                observation=tool_result[:200],
                reward=reward,
            ))
            tracer.end_episode(episode_id, "success" if tool_success else "partial")
            tracer.cleanup_episode(episode_id)
        except Exception as exc:
            logger.warning("deliberative MDP trace failed (continuing): %s", exc)

    async def _sync_response(
        self,
        messages: list[dict],
        llm_client,
        max_tokens: int | None = None,
        *,
        tool_result: str | None = None,
        tool_name: str | None = None,
        tool_command: str | None = None,
        tool_command_success: bool | None = None,
        generation_params: GenerationParams | None = None,
    ) -> DeliberativeResponse:
        """非ストリーミング応答"""
        kwargs: dict = {"stream": False, "id_slot": llm_client.chat_slot}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # モード別生成パラメータを適用
        if generation_params:
            for k in ("temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty"):
                if k in generation_params:
                    kwargs[k] = generation_params[k]
        result = await llm_client.generate(messages, **kwargs)
        content = result["choices"][0]["message"]["content"]
        logger.info("Deliberative sync response: %d chars", len(content))
        return DeliberativeResponse(
            content=content,
            tool_result=tool_result,
            tool_name=tool_name,
            tool_command=tool_command,
            tool_command_success=tool_command_success,
        )

    async def _stream_response(
        self,
        messages: list[dict],
        llm_client,
        max_tokens: int | None = None,
        *,
        tool_result: str | None = None,  # noqa: ARG002
        tool_name: str | None = None,  # noqa: ARG002
        generation_params: GenerationParams | None = None,
    ) -> AsyncIterator[str]:
        """ストリーミング応答（生トークンのイテレータを返す）"""
        kwargs: dict = {"stream": True, "id_slot": llm_client.chat_slot}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # モード別生成パラメータを適用
        if generation_params:
            for k in ("temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty"):
                if k in generation_params:
                    kwargs[k] = generation_params[k]
        token_gen = await llm_client.generate(messages, **kwargs)
        tokens_generated = 0
        async for token in token_gen:
            tokens_generated += 1
            yield token
        logger.debug(
            "Deliberative stream complete: tokens_generated=%d",
            tokens_generated,
        )

    async def _ensure_write_file_content(
        self,
        tool_name: str,
        tool_args: dict,
        query: str,
        llm_client,
        on_step: StepCallback,
    ) -> None:
        """`write_file` の `content` が空なら LLM で生成して `tool_args` に注入する。

        ``_generate_content`` がエラーセンチネル文字列を返した場合は
        ``tool_args["content"]`` に注入せず、呼び出し元 ``_execute_tool``
        で実行スキップさせる。
        """
        if tool_name != "write_file" or tool_args.get("content"):
            return
        file_path = tool_args.get("file_path", "")
        if on_step:
            on_step({
                "type": "tool_call",
                "detail": f"コンテンツ生成中 → {file_path}",
                "status": "running",
            })
        content = await self._generate_content(query, llm_client)
        if content.startswith("(Content generation failed:"):
            logger.warning(
                "Deliberative: content generation failed for %s; "
                "skipping write_file injection",
                file_path,
            )
            if on_step:
                on_step({
                    "type": "tool_call",
                    "detail": f"write_file: コンテンツ生成失敗 → {file_path}",
                    "status": "failed",
                })
            return
        rejection = generated_content_rejection(content, file_path)
        if rejection:
            logger.warning(
                "Deliberative: generated content rejected (%s); skipping "
                "write_file injection: %r",
                rejection, content[:120],
            )
            if on_step:
                on_step({
                    "type": "tool_call",
                    "detail": f"write_file: コンテンツ生成失敗（{rejection}） → {file_path}",
                    "status": "failed",
                })
            return
        tool_args["content"] = content
        logger.info(
            "Content generated for write_file: %d chars → %s",
            len(content), file_path,
        )

    async def _run_tool_with_handling(
        self,
        tool_name: str,
        tool_args: dict,
        state: AgentState,
        on_step: StepCallback,
    ) -> str:
        """登録済みツールを timeout 付きで実行し、結果テキスト or エラーを返す。

        正常終了 / TimeoutError / 一般例外をそれぞれ handling し、
        対応する step フレームを emit する。`finally` で `state.pending_*` をクリア。
        """
        state.pending_tool = tool_name
        state.pending_args = tool_args
        try:
            result = await asyncio.wait_for(
                self._tools_registry.execute(tool_name, **tool_args),
                timeout=TOOL_EXECUTION_TIMEOUT_SEC,
            )
            result_text = str(result)
            state.on_tool_success(tool_name)
            logger.info("Tool executed successfully: %s", tool_name)
            _emit_tool_result_step(on_step, tool_name, result_text)
            return result_text
        except asyncio.TimeoutError:
            error_text = (
                f"Error: tool execution timed out after {TOOL_EXECUTION_TIMEOUT_SEC}s"
            )
            state.on_tool_failure(tool_name, error_text)
            logger.warning(
                "Tool execution timed out: %s (%.0fs)",
                tool_name, TOOL_EXECUTION_TIMEOUT_SEC,
            )
            _emit_tool_failure_step(on_step, tool_name, error_text)
            return error_text
        except Exception as e:
            error_text = f"Error: {e}"
            state.on_tool_failure(tool_name, str(e))
            logger.warning("Tool execution failed: %s - %s", tool_name, e)
            _emit_tool_failure_step(on_step, tool_name, error_text)
            return error_text
        finally:
            state.pending_tool = None
            state.pending_args = {}

    async def _execute_tool(
        self,
        judgement: ToolJudgement,
        state: AgentState,
        query: str,
        llm_client,
        on_step: StepCallback = None,
        mode: str | None = None,
    ) -> str | None:
        """ToolJudgement に基づいてツールを実行

        write_file でコンテンツが不足している場合は、LLM にプレーンテキストで
        コンテンツを生成させてから実行する。

        Returns:
            ツール実行結果のテキスト。ツールが見つからない/実行失敗時は None。
        """
        if self._tools_registry is None:
            return None

        tool_name = judgement.tool_name
        # tool_args は dict 契約だが、アシスト応答の機械修復経路で非 dict が
        # 紛れ込むことがあるため防御的にガードする (cf. _judge_and_execute_tool)。
        raw_args = judgement.tool_args
        tool_args = dict(raw_args) if isinstance(raw_args, dict) else {}  # コピー

        if not self._tools_registry.has(tool_name):
            logger.warning("Tool not found: %s", tool_name)
            return None

        # ToolDefinition.modes は元々 get_descriptions_text() (LLM 向け説明文) の
        # フィルタ用にしか参照されておらず、実行時には無視されていた。ルールベース
        # 判定 (tool_call_judge) が誤トリガーで coding 専用ツールを選んでも、ここで
        # 弾かなければ chat モードでも実行されてしまう (search_code の CWD 全域
        # os.walk がイベントループを長時間ブロックした実インシデントの直接原因)。
        tool_def = self._tools_registry.get(tool_name)
        effective_mode = mode if mode is not None else self._mode
        if tool_def is not None and effective_mode not in tool_def.modes:
            logger.warning(
                "Tool not allowed in mode=%s: %s (allowed modes: %s)",
                effective_mode, tool_name, tool_def.modes,
            )
            return None

        path_error = _check_path_traversal(
            tool_args.get("file_path", ""), tool_name,
        )
        if path_error:
            return path_error

        await self._ensure_write_file_content(
            tool_name, tool_args, query, llm_client, on_step,
        )

        # write_file で content が依然空 → LLM 生成失敗。誤実行を防ぐため
        # tool_args をそのまま流さずエラー文字列を返してスキップする。
        if tool_name == "write_file" and not tool_args.get("content"):
            error_text = "Error: content generation failed"
            state.on_tool_failure(tool_name, error_text)
            return error_text

        # 必須引数チェック（必須パラメータが空の場合を防止）。running フレーム
        # の emit より前に行う — emit 後に skip すると完了フレームが出ず
        # UI に空ステップ (running のまま) が残る (2026-07-21 ライブ検証
        # ターン35 の引数なし calculate で実発生)。
        if tool_def and tool_def.parameters and not tool_args:
            logger.warning(
                "Tool %s requires args but none provided, skipping", tool_name,
            )
            return None

        _emit_tool_running_step(on_step, tool_name, tool_args)

        return await self._run_tool_with_handling(
            tool_name, tool_args, state, on_step,
        )

    async def _generate_content(
        self,
        query: str,
        llm_client,
    ) -> str:
        """write_file 用のコンテンツを LLM にプレーンテキストで生成させる"""
        messages = [
            # 出力言語指示 (locale 追従) を毎回組み立てて付加する
            {
                "role": "system",
                "content": f"{_CONTENT_GEN_PROMPT}{content_language_directive()}",
            },
            {"role": "user", "content": query},
        ]
        try:
            result = await llm_client.generate(
                messages, stream=False,
                max_tokens=self._content_max_tokens,
                id_slot=llm_client.chat_slot,
            )
            content = result["choices"][0]["message"]["content"].strip()
            content = strip_markdown_wrapper(content)
            logger.debug("Content generated: %d chars", len(content))
            return content
        except Exception as e:
            logger.error("Content generation failed: %s", e)
            return f"(Content generation failed: {e})"


def _truncate_tool_result(text: str, max_chars: int) -> str:
    """ツール結果が max_chars を超える場合、先頭と末尾を残して切り詰める"""
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
