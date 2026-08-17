"""Deliberative / Reactive 軽量パスのストリーミング・同期応答"""

from __future__ import annotations

import asyncio
import time

from dataclasses import dataclass
from typing import (
    AsyncIterator,
    TYPE_CHECKING,
)
from backend.app_state import AppState
from backend.free.api.chat.chat_constants import DEFAULT_KEEPALIVE_INTERVAL_SEC
from backend.free.api.chat.chat_recorder import record_response
from backend.free.api.chat.chat_service import make_token_info
from backend.free.api.chat.chat_types import (
    ChatMessage,
    GenerationParams,
)
from backend.free.api.schemas import ChatResponse
from backend.free.agent.deliberative import DeliberativeAgent
from backend.free.core.stream_filter import (
    HeadBufferFilter,
    InternalFrameMentionFilter,
    QueryEchoFilter,
    RepetitionGuardFilter,
    StreamThinkingFilter,
)
from backend.free.core.stream_pipeline import StreamPipeline
from backend.free.llm.local_client import LocalClient
from backend.utils import estimate_tokens as _estimate_tokens
from fastapi import HTTPException

from backend.free.api.chat.chat_stream_common import (
    _cancel_flags,
    _emit_timing,
    _log_chat_outcome,
    _make_step_queue_callback,
    _sync_chat_response,
    cancel_scope,
    logger,
    sse,
)

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer


# ---------------------------------------------------------------------------
# Deliberative ストリーミング / 同期
# ---------------------------------------------------------------------------


@dataclass
class _DeliberativeStreamState:
    """`stream_deliberative` のループ mutable 状態を集約。"""

    tokens_generated: int = 0
    #: フィルタ通過後に実際にユーザーへ出た文字数。``tokens_generated`` は
    #: tok/s 指標用に raw トークンを数えるため、フィルタが全部落としても 0 に
    #: ならず、ゼロトークン再試行が発火しない。復唱だけの応答を落とすと画面が
    #: 空になるため、再試行判定はこちらを見る。
    emitted_chars: int = 0
    full_response: str = ""
    first_token_recorded: bool = False
    # executable command 学習用 (run_command 実行ターンのみ非 None)
    tool_command: str | None = None
    tool_command_name: str | None = None
    tool_command_success: bool | None = None
    tool_command_source: str | None = None




async def _drain_deliberative_step_queue(
    step_queue: list[dict],
) -> AsyncIterator[str]:
    """ツール実行で蓄積された step フレームを順次 yield する。"""
    if step_queue:
        logger.debug("Deliberative: sending %d step frames", len(step_queue))
    for step_data in step_queue:
        logger.debug(
            "Deliberative step: type=%s, status=%s, detail=%s",
            step_data.get("type"), step_data.get("status"),
            step_data.get("detail", "")[:120],
        )
        yield sse.step(step_data)
    step_queue.clear()


async def _stream_filtered_token_pipeline(
    token_stream: AsyncIterator[str],
    state: _DeliberativeStreamState,
    session_id: str,
    timer: StageTimer | None,
    query: str = "",
) -> AsyncIterator[str]:
    """フィルタパイプライン (思考ブロック除去 + 先頭ラベル除去) でトークンを yield する。

    LLM の prefill が長引いて keepalive_interval を超える間トークンが届かない場合は、
    SSE keepalive コメントを送出してフロントエンドの chunk timeout を防ぐ。

    ``query`` は復唱除去用。渡さない場合は復唱フィルタが素通しになる。
    """
    pipeline = StreamPipeline([
        StreamThinkingFilter(),
        HeadBufferFilter(),
        # 先頭ラベル除去のあとに復唱を判定する (「**回答:** 今日は何曜日ですか。」
        # のようにラベルが前置されている形でも冒頭一致で拾えるようにする)。
        QueryEchoFilter(query),
        # 思考ブロック除去・先頭ラベル除去を通したあとの「実際にユーザーへ出る
        # テキスト」に対して反復を判定する (前段が落とす行を数えないため)。
        RepetitionGuardFilter(query),
        # 内部の根拠枠 (「（参考情報1に基づく）」) の名指しを最後に落とす。
        # system プロンプトと動的ブロックの区切り文の両方で禁じているのに
        # 実機では破られた (2026-08-16 ライブ監査 ターン25)。
        InternalFrameMentionFilter(),
    ])
    aiter = token_stream.__aiter__()
    pending: asyncio.Task[str] | None = None
    # keepalive を「LLM からトークンが来ない時間」ではなく「SSE フレームを
    # 送っていない時間」で測る。フィルタはトークンをバッファするので、LLM が
    # 途切れなく生成していてもフロントには何も届かない時間が生じる
    # (実インシデント 2026-08-07 ライブ監査: E:\tmp のファイル一覧が 1 行の長い
    # 列挙になり、バックエンドは全文を生成し終えていたのにフロントが 60 秒の
    # chunk timeout で切って `Error: stream_timeout` を出し、応答が失われた)。
    # 長文経路 (_stream_long_form) は既にこの測り方をしており、こちらだけが
    # トークン到着ベースだった。
    last_frame_at = time.monotonic()
    while True:
        if _cancel_flags.get(session_id):
            if pending is not None and not pending.done():
                pending.cancel()
            break
        if pending is None:
            pending = asyncio.create_task(aiter.__anext__())
        # `asyncio.wait` はタイムアウト時にタスクをキャンセルしないため、
        # keepalive 送出後も同じ `__anext__()` 呼び出しを継続できる。
        done, _ = await asyncio.wait(
            {pending}, timeout=DEFAULT_KEEPALIVE_INTERVAL_SEC,
        )
        if pending not in done:
            yield sse.keepalive()
            last_frame_at = time.monotonic()
            continue
        try:
            token = pending.result()
        except StopAsyncIteration:
            pending = None
            break
        pending = None
        state.full_response += token
        # raw トークン単位でカウント (tok/s 指標用)。
        # HeadBufferFilter によるバッファリングで SSE フレーム数と乖離するため、
        # フィルタ出力の有無にかかわらず受信トークンをそのまま数える。
        state.tokens_generated += 1
        # TTFT は **モデルから最初のトークンが届いた時刻** で止める。
        # 以前は filtered (フィルタが実際に吐いた瞬間) で止めていたため、
        # HeadBufferFilter が最後までバッファする短い応答では
        # llm_first_token_ms が記録されないまま終わっていた
        # (2026-08-16 の監査では 40 ターン中 25 ターンしか値が無く、
        #  この系で最も重要な指標が短い応答ほど欠落していた)。
        if not state.first_token_recorded and timer:
            timer.stop("llm_first_token_ms")
            state.first_token_recorded = True
        filtered = pipeline.process(token)
        if filtered:
            state.emitted_chars += len(filtered)
            yield sse.token(filtered)
            last_frame_at = time.monotonic()
        elif time.monotonic() - last_frame_at >= DEFAULT_KEEPALIVE_INTERVAL_SEC:
            # フィルタがバッファしている間もフロントの chunk timeout を防ぐ
            # (上の last_frame_at のコメント参照)。
            yield sse.keepalive()
            last_frame_at = time.monotonic()

    remaining = pipeline.flush()
    if remaining:
        state.emitted_chars += len(remaining)
        yield sse.token(remaining)


async def _retry_zero_tokens_deliberative(
    state: _DeliberativeStreamState,
    messages: list[ChatMessage],
    client: LocalClient,
    max_tokens: int | None,
    session_id: str,
    query: str = "",
) -> AsyncIterator[str]:
    """ユーザーへ 1 文字も出なかった場合に再試行する。

    判定は raw トークン数ではなく **フィルタ通過後に出た文字数**。復唱だけの
    応答はフィルタが全部落とすため、raw では 0 にならないのに画面は空になる
    (実インシデント 2026-08-04 ライブ監査: 質問文の復唱しか生成せず、
    QueryEchoFilter がそれを落とした結果 5 回中 2 回が無応答)。

    再試行トークンも **本流と同じフィルタへ通す**。素通しにすると、初回を
    フィルタが落としたケースで再試行だけが無防備になり、結果として最悪の
    出力がそのまま画面へ出る (実インシデント 2026-08-04: 初回の復唱を落とした
    直後の再試行が同じ復唱を 170 回出力して max_tokens まで走った)。

    既にキャンセル済みか出力済みなら何も yield しない。
    リトライ後もゼロなら error フレームを yield。
    """
    if state.emitted_chars > 0 or _cancel_flags.get(session_id):
        return
    logger.warning(
        "No content tokens from llama-server, "
        "retrying with fresh request (reasoning-only or stale cache)",
    )
    retry_stream = await client.generate(
        messages, stream=True, id_slot=client.chat_slot,
        max_tokens=max_tokens,
    )
    pipeline = StreamPipeline([
        StreamThinkingFilter(),
        HeadBufferFilter(),
        QueryEchoFilter(query),
        RepetitionGuardFilter(query),
        InternalFrameMentionFilter(),
    ])
    async for token in retry_stream:
        if _cancel_flags.get(session_id):
            break
        state.full_response += token
        state.tokens_generated += 1
        filtered = pipeline.process(token)
        if filtered:
            state.emitted_chars += len(filtered)
            yield sse.token(filtered)
    remaining = pipeline.flush()
    if remaining:
        state.emitted_chars += len(remaining)
        yield sse.token(remaining)

    if state.emitted_chars > 0:
        logger.info("Retry succeeded: tokens=%d", state.tokens_generated)
        return
    logger.error("Retry also returned 0 content tokens")
    yield sse.error(
        "No content generated after retry. "
        "The model may be stuck in a reasoning loop."
    )


def _maybe_cache_reactive_response(
    sess_state: AppState,
    query: str,
    response: str,
    *,
    private: bool,
    tool_command: str | None,
    session_id: str,
) -> None:
    """deliberative / 軽量パス応答を ReactiveAgent キャッシュへ蓄積する。

    再訪クエリ (5 分以内・同一文) を reactive 層が即応答できるようにする。
    除外: private ターン / ツール使用応答 (時刻・環境依存で再利用不可) /
    キャンセル済み (部分テキスト) / 空応答。
    """
    agent = getattr(sess_state, "reactive_agent", None)
    if agent is None:
        return
    if private or tool_command is not None:
        return
    if _cancel_flags.get(session_id):
        return
    if not response or not response.strip():
        return
    agent.cache_response(query, response)


async def _finalize_deliberative_stream(
    state: _DeliberativeStreamState,
    sess_state: AppState,
    query: str,
    messages: list[ChatMessage],
    session_id: str,
    mode: str,
    instance_name: str,
    context_size: int,
    timer: StageTimer | None,
    t_start: float,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> AsyncIterator[str]:
    """Deliberative ストリーム終端処理: timer 停止 + record + token_info + done."""
    if timer:
        timer.stop("llm_total_ms")
    elapsed = time.monotonic() - t_start
    tok_per_sec = state.tokens_generated / elapsed if elapsed > 0 else 0
    logger.info(
        "Deliberative stream complete: tokens=%d, elapsed=%.2fs, tok/s=%.1f, session=%s",
        state.tokens_generated, elapsed, tok_per_sec, session_id,
    )
    record_response(
        sess_state, state.full_response, messages, session_id,
        query, mode, state.tokens_generated,
        private=private,
        tool_command=state.tool_command,
        tool_command_name=state.tool_command_name,
        tool_command_success=state.tool_command_success,
        tool_command_source=state.tool_command_source,
        tool_routing_success=state.tool_command_success is True,
        rag_used=rag_used,
        rag_top1_score=rag_top1_score,
    )
    _maybe_cache_reactive_response(
        sess_state, query, state.full_response,
        private=private, tool_command=state.tool_command, session_id=session_id,
    )
    _emit_timing(sess_state, timer, "deliberative", state.tokens_generated, mode=mode)
    ti = make_token_info(
        messages, state.tokens_generated, context_size, instance_name,
    )
    yield sse.token_info(ti)
    yield sse.done()


async def stream_deliberative(
    agent: DeliberativeAgent, query: str, messages: list[ChatMessage],
    client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    *, mode: str = "chat", max_tokens: int | None = None,
    conversation: list[ChatMessage] | None = None,
    generation_params: GenerationParams | None = None,
    timer: StageTimer | None = None,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
    tool_judge_task: "asyncio.Task | None" = None,
    escalated_from: str | None = None,
    evicted_turns: int = 0,
    session_head: str = "",
):
    """Deliberative 層の SSE ストリーミング

    DeliberativeAgent が返す生トークンストリームを SSE フレームに変換し、
    キャンセル・0トークンリトライ・token_info・record_response を処理する。
    ツール実行のステップフレームもリアルタイムで送信する。
    StreamPipeline でフィルタチェーン（思考ブロック除去 + 先頭ラベル除去）を適用。
    """
    async with cancel_scope(session_id):
        stream_state = _DeliberativeStreamState()
        t_start = time.monotonic()
        step_queue: list[dict] = []
        on_step = _make_step_queue_callback(step_queue)
        # (asyncio.CancelledError) 時も finally で確実に emit するため、
        # finalize 完了まで到達した場合のみ True にする。
        outcome_success = False
        # genuine error (except Exception) と client cancel
        # (CancelledError/GeneratorExit; except を素通り) を区別する。
        errored = False
        # keepalive ループで使う process() タスク (finally の衛生処理が参照)。
        process_task: asyncio.Task | None = None
        # executable command 学習用に command/success を受け取る dict。
        # process() は iterator 返却前に _judge_and_execute_tool を完了
        # するため、await 完了時点で値が確定している。
        # try の外で束縛する — finally の outcome ログが参照するため、
        # try 内の早期例外で未定義になると NameError で握り潰される。
        tool_capture: dict = {}

        try:
            yield sse.agent_layer("deliberative")

            # Trigger A: LLM 生成開始直後に sleep-time Light を並列実行（§8.1）
            scheduler = state.sleep_scheduler
            if scheduler:
                scheduler.on_llm_start()

            if timer:
                timer.start("llm_total_ms")
                timer.start("llm_first_token_ms")

            process_task = asyncio.create_task(agent.process(
                query=query,
                messages=list(messages),
                llm_client=client,
                mode=mode,
                stream=True,
                conversation=conversation,
                max_tokens=max_tokens,
                on_step=on_step,
                generation_params=generation_params,
                tool_capture=tool_capture,
                tool_judge_task=tool_judge_task,
                session_id=session_id,
                evicted_turns=evicted_turns,
                session_head=session_head,
            ))
            # process() はトークンを返す前にツール判定と実行を完了させる。
            # ここを素の await にすると、その間 SSE フレームが 1 つも流れず、
            # フロントの chunk timeout (60 秒) に掛かって "stream_timeout" で
            # 落ちる (実測 2026-07-27: draft_document がベースモデルの
            # 非ストリーミング生成に入り、iGPU で 60 秒を超えて失敗した)。
            # 待機中も keepalive と step フレームを流し続ける。
            while True:
                done, _ = await asyncio.wait(
                    {process_task}, timeout=DEFAULT_KEEPALIVE_INTERVAL_SEC,
                )
                # ツール開始の step フレームを実行完了まで溜めない
                # (UI に「実行中」が即座に出る副次効果もある)。
                async for frame in _drain_deliberative_step_queue(step_queue):
                    yield frame
                if process_task in done:
                    break
                yield sse.keepalive()
            token_stream = process_task.result()
            stream_state.tool_command = tool_capture.get("command")
            stream_state.tool_command_name = tool_capture.get("command_name")
            stream_state.tool_command_success = tool_capture.get("success")
            stream_state.tool_command_source = tool_capture.get("command_source")

            async for frame in _drain_deliberative_step_queue(step_queue):
                yield frame

            async for frame in _stream_filtered_token_pipeline(
                token_stream, stream_state, session_id, timer, query,
            ):
                yield frame

            async for frame in _retry_zero_tokens_deliberative(
                stream_state, messages, client, max_tokens, session_id, query,
            ):
                yield frame

            async for frame in _finalize_deliberative_stream(
                stream_state, state, query, messages, session_id,
                mode, instance_name, context_size, timer, t_start,
                private=private,
                rag_used=rag_used,
                rag_top1_score=rag_top1_score,
            ):
                yield frame
            outcome_success = True

        except Exception as e:
            errored = True
            # 例外がメッセージを持たないケース (httpx.ReadError('') 等) では
            # %s だと原因不明のログ行しか残らないため repr で型を残す。
            logger.error("Deliberative stream error: %r", e)
            if timer:
                timer.stop("llm_total_ms")
            _emit_timing(state, timer, "deliberative", 0, mode=mode)
            yield sse.error(str(e))
            yield sse.done()
        finally:
            # クライアント切断等でジェネレータが中断された場合、未完了の
            # precomputed tool 判定タスクが残らないよう cancel する (衛生)。
            # 正常経路では process() 内で既に await 済み (done) のため no-op。
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            # 同様に、keepalive ループ中にジェネレータが閉じられた場合は
            # process() タスクを孤児にしない。
            if process_task is not None and not process_task.done():
                process_task.cancel()
            # CancelledError 経路でも finally に入るため client cancel
            # 検知 (success=False) が確実に行える。
            signals: dict = {"agent_layer": "deliberative"}
            if escalated_from:
                signals["escalated_from"] = escalated_from
            # 「どのツールを撃って、役に立つ結果が出たか」を残す。
            # これが無いと outcome JSONL には agent_layer しか載らず、
            # 空振りツールに頼ったターンと接地できたターンが事後に
            # 区別できない (2026-08-05 ライブ監査: chat_response 40/40 が
            # success=true で、捏造したターンを含めて全て同じ見え方だった)。
            if tool_capture.get("tool_name"):
                signals["tool_name"] = tool_capture["tool_name"]
                signals["tool_success"] = bool(tool_capture.get("tool_success"))
            _log_chat_outcome(
                state,
                started_at=t_start,
                success=outcome_success,
                tokens_out=stream_state.tokens_generated,
                signals=signals,
                cancelled=not outcome_success and not errored,
            )


async def sync_deliberative(
    agent: DeliberativeAgent, query: str, messages: list[ChatMessage],
    client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    *, mode: str = "chat", max_tokens: int | None = None,
    conversation: list[ChatMessage] | None = None,
    generation_params: GenerationParams | None = None,
    timer: StageTimer | None = None,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
    tool_judge_task: "asyncio.Task | None" = None,
    escalated_from: str | None = None,  # noqa: ARG001
    evicted_turns: int = 0,
    session_head: str = "",
) -> ChatResponse:
    """Deliberative 層の非ストリーミング応答 (escalated_from は API 一貫性用、未使用)"""
    logger.debug("Sync deliberative: session=%s, messages=%d", session_id, len(messages))
    try:
        # Trigger A: LLM 生成開始直後に sleep-time Light を並列実行（§8.1）
        scheduler = state.sleep_scheduler
        if scheduler:
            scheduler.on_llm_start()

        if timer:
            timer.start("llm_total_ms")
        resp = await agent.process(
            query=query,
            messages=list(messages),
            llm_client=client,
            mode=mode,
            stream=False,
            conversation=conversation,
            max_tokens=max_tokens,
            generation_params=generation_params,
            tool_judge_task=tool_judge_task,
            session_id=session_id,
            evicted_turns=evicted_turns,
            session_head=session_head,
        )

        if timer:
            timer.stop("llm_total_ms")

        estimated_tokens = max(1, _estimate_tokens(resp.content))
        record_response(
            state, resp.content, messages, session_id,
            query, mode, estimated_tokens,
            private=private,
            tool_command=resp.tool_command,
            tool_command_name=resp.tool_name if resp.tool_command else None,
            tool_command_success=resp.tool_command_success,
            tool_command_source=resp.tool_command_source,
            tool_routing_success=resp.tool_command_success is True,
            rag_used=rag_used,
            rag_top1_score=rag_top1_score,
        )
        _maybe_cache_reactive_response(
            state, query, resp.content,
            private=private, tool_command=resp.tool_command, session_id=session_id,
        )

        return _sync_chat_response(
            state, timer,
            agent_layer="deliberative",
            text=resp.content,
            tokens=estimated_tokens,
            messages=messages,
            session_id=session_id,
            instance_name=instance_name,
            context_size=context_size,
            mode=mode,
        )
    except Exception as e:
        logger.error("Deliberative error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        # process() が判定タスクを await する前に例外で抜けた場合の衛生。
        # 正常経路では process() 内で await 済み (done) のため no-op。
        if tool_judge_task is not None and not tool_judge_task.done():
            tool_judge_task.cancel()


def _apply_generation_params(gen_kwargs: dict, generation_params: "GenerationParams | None") -> None:
    """モード別生成パラメータを client.generate の kwargs へ転写する。"""
    if not generation_params:
        return
    for k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "repetition_penalty"):
        if k in generation_params:
            gen_kwargs[k] = generation_params[k]


async def stream_reactive_light(
    query: str,
    messages: list[ChatMessage],
    client: LocalClient,
    state: AppState,
    session_id: str,
    instance_name: str,
    context_size: int,
    *,
    mode: str = "chat",
    max_tokens: int | None = None,
    generation_params: "GenerationParams | None" = None,
    timer: "StageTimer | None" = None,
    private: bool = False,
):
    """Reactive 軽量パス: few-shot/RAG/semmem/tool なしの最小プロンプトで base 1 ターン。

    agent.process を介さず client.generate を直接叩き、deliberative のストリーミング
    ヘルパー (フィルタ / 0トークンリトライ / timing / cache) を再利用する。
    SSE 上の agent_layer は "reactive"。
    """
    async with cancel_scope(session_id):
        stream_state = _DeliberativeStreamState()
        t_start = time.monotonic()
        outcome_success = False
        errored = False
        try:
            yield sse.agent_layer("reactive")

            scheduler = state.sleep_scheduler
            if scheduler:
                scheduler.on_llm_start()

            if timer:
                timer.start("llm_total_ms")
                timer.start("llm_first_token_ms")

            gen_kwargs: dict = {"stream": True, "id_slot": client.chat_slot}
            if max_tokens is not None:
                gen_kwargs["max_tokens"] = max_tokens
            _apply_generation_params(gen_kwargs, generation_params)
            token_stream = await client.generate(list(messages), **gen_kwargs)

            async for frame in _stream_filtered_token_pipeline(
                token_stream, stream_state, session_id, timer, query,
            ):
                yield frame

            async for frame in _retry_zero_tokens_deliberative(
                stream_state, messages, client, max_tokens, session_id, query,
            ):
                yield frame

            if timer:
                timer.stop("llm_total_ms")
            record_response(
                state, stream_state.full_response, messages, session_id,
                query, mode, stream_state.tokens_generated,
                private=private,
                rag_used=False,
            )
            _maybe_cache_reactive_response(
                state, query, stream_state.full_response,
                private=private, tool_command=None, session_id=session_id,
            )
            _emit_timing(state, timer, "reactive", stream_state.tokens_generated, mode=mode)
            ti = make_token_info(
                messages, stream_state.tokens_generated, context_size, instance_name,
            )
            yield sse.token_info(ti)
            yield sse.done()
            outcome_success = True

        except Exception as e:
            errored = True
            logger.error("Reactive-light stream error: %s", e)
            if timer:
                timer.stop("llm_total_ms")
            _emit_timing(state, timer, "reactive", 0, mode=mode)
            yield sse.error(str(e))
            yield sse.done()
        finally:
            _log_chat_outcome(
                state,
                started_at=t_start,
                success=outcome_success,
                tokens_out=stream_state.tokens_generated,
                signals={"agent_layer": "reactive", "reactive_light": True},
                # success=False かつ genuine error でない = client cancel。
                cancelled=not outcome_success and not errored,
            )


async def sync_reactive_light(
    query: str,
    messages: list[ChatMessage],
    client: LocalClient,
    state: AppState,
    session_id: str,
    instance_name: str,
    context_size: int,
    *,
    mode: str = "chat",
    max_tokens: int | None = None,
    generation_params: "GenerationParams | None" = None,
    timer: "StageTimer | None" = None,
    private: bool = False,
) -> ChatResponse:
    """Reactive 軽量パスの非ストリーミング応答。"""
    from backend.free.llm.utils import extract_content

    t_start = time.monotonic()
    try:
        scheduler = state.sleep_scheduler
        if scheduler:
            scheduler.on_llm_start()

        if timer:
            timer.start("llm_total_ms")
        gen_kwargs: dict = {"stream": False, "id_slot": client.chat_slot}
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
        _apply_generation_params(gen_kwargs, generation_params)
        data = await client.generate(list(messages), **gen_kwargs)
        content = extract_content(data) if isinstance(data, dict) else str(data)
        if timer:
            timer.stop("llm_total_ms")

        estimated_tokens = max(1, _estimate_tokens(content))
        record_response(
            state, content, messages, session_id,
            query, mode, estimated_tokens,
            private=private,
            rag_used=False,
        )
        _maybe_cache_reactive_response(
            state, query, content,
            private=private, tool_command=None, session_id=session_id,
        )
        _log_chat_outcome(
            state,
            started_at=t_start,
            success=True,
            tokens_out=estimated_tokens,
            signals={"agent_layer": "reactive", "reactive_light": True},
        )
        return _sync_chat_response(
            state, timer,
            agent_layer="reactive",
            text=content,
            tokens=estimated_tokens,
            messages=messages,
            session_id=session_id,
            instance_name=instance_name,
            context_size=context_size,
            mode=mode,
        )
    except Exception as e:
        logger.error("Reactive-light sync error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
