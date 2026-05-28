"""チャット API（SSE ストリーミング + 3層エージェントディスパッチ）"""

import re
from collections.abc import AsyncIterator, Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.app_state import AppState, get_app_state
from backend.config import get_config, get_mode_generation_params
from backend.free.api.chat.chat_constants import (
    DEFAULT_CONTEXT_SIZE, DEFAULT_KEEPALIVE_INTERVAL_SEC, DEFAULT_MAX_TOKENS,
    MAX_FILE_CONTEXT_TOTAL_CHARS, MAX_FILE_CONTEXT_TOTAL_CHUNKS,
    MAX_MESSAGE_LENGTH,
    SESSION_ID_MAX_LENGTH, SESSION_ID_MIN_LENGTH,
)
from backend.free.api.schemas import (
    CancelRequest, CancelResponse, ChatRequest, ChatResponse, TokenInfo,
)
from backend.free.api.chat._editor_routing import detect_editor_route
from backend.free.agent.tool_call_judge import _extract_file_path
from backend.free.api.chat.chat_recorder import record_response
from backend.free.api.chat.chat_service import (
    build_chat_messages, build_semmem_injection, convert_file_contexts,
    ensure_base_model_health,
    ensure_llm_client, prepare_memory_context,
    run_search_pipeline,
)
from backend.free.api.chat.chat_streaming import (
    _cancel_flags,
    read_existing_for_append,
    stream_deliberative, stream_long_form, stream_meta_cognitive, stream_reactive,
    sync_deliberative, sync_long_form, sync_meta_cognitive,
)
from backend.free.core.sse import SSEFrameBuilder
from backend.free.agent.deliberative import DeliberativeAgent
from backend.free.agent.meta_cognitive import MetaCognitiveAgent
from backend.free.agent.reactive import ReactiveAgent
from backend.free.agent.router import ComplexityClassifier
from backend.free.core.stage_timer import StageTimer
from backend.free.generation.orchestrator import LongFormOrchestrator
from backend.log_config import get_logger
from backend.trace_context import generate_trace_id, set_trace_id

logger = get_logger("api.chat")

router = APIRouter(prefix="/api", tags=["chat"])


# PEP 695 type alias: SSE フレームジェネレータをラップする中間関数
type StreamWrapper = Callable[[AsyncIterator[str]], AsyncIterator[str]]


async def _with_chat_in_flight(client, inner_gen):
    """ストリーミングジェネレータを ``chat_in_flight()`` でラップする。

    LLMClient にユーザー応答進行中であることを通知し、バックグラウンド
    処理（Level 1 進化、sleep-time）が協調的に yield できるようにする
    （f_04_self_learning.md §8.2）。
    """
    async with client.chat_in_flight():
        async for frame in inner_gen:
            yield frame


# session_id のフォーマット: 英数字・ハイフンのみ、8-64文字
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9\-]{%d,%d}$" % (SESSION_ID_MIN_LENGTH, SESSION_ID_MAX_LENGTH))


def _resolve_loop_view_for_agent(state: AppState):
    """MetaCognitiveAgent 用の LoopFactView を生成する

    ``@self`` 仮想カートリッジが SemMem を読み取る際の read-only 入口。
    ``SemanticFactStore`` 直参照を廃止し LoopFactView 経由に統一
    ``current_project_id`` 未解決の場合は writeback_store も global に向ける
    (self_cartridge は読取のみのため実害なし)。ストアアクセス失敗時は
    ``None`` で graceful degrade する (チャット応答を阻害しない)。
    """
    from backend.free.memory.views.loop import LoopFactView

    try:
        global_store = state.get_semantic_store("global")
    except Exception as exc:
        logger.warning("@self: global SemMem store unavailable: %s", exc)
        return None
    stores: list = [global_store]
    writeback = global_store
    pid = state.current_project_id
    if pid:
        try:
            project_store = state.get_semantic_store(f"project:{pid}")
            stores.append(project_store)
            writeback = project_store
        except Exception as exc:
            logger.warning(
                "@self: project SemMem store unavailable for %s: %s", pid, exc,
            )
    try:
        return LoopFactView(stores=stores, writeback_store=writeback)
    except Exception as exc:  # noqa: BLE001
        logger.warning("@self: LoopFactView construction failed: %s", exc)
        return None


def _validate_chat_request(req: ChatRequest) -> None:
    """ChatRequest の入力バリデーション（不正なら HTTPException を送出）"""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty")

    if len(req.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long: {len(req.message)} chars (max {MAX_MESSAGE_LENGTH})",
        )

    if req.mode not in ("chat", "coding"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {req.mode}")

    if req.session_id is not None and not _SESSION_ID_RE.match(req.session_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session_id format: must be {SESSION_ID_MIN_LENGTH}-{SESSION_ID_MAX_LENGTH} alphanumeric/hyphen chars",
        )

    if not req.file_contexts:
        return

    total_chunks = sum(len(fc.chunks) for fc in req.file_contexts)
    if total_chunks > MAX_FILE_CONTEXT_TOTAL_CHUNKS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many file context chunks: {total_chunks} (max {MAX_FILE_CONTEXT_TOTAL_CHUNKS})",
        )
    total_chars = sum(sum(len(c) for c in fc.chunks) for fc in req.file_contexts)
    if total_chars > MAX_FILE_CONTEXT_TOTAL_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"File contexts too large: {total_chars} chars (max {MAX_FILE_CONTEXT_TOTAL_CHARS})",
        )


def _llm_unavailable_response(stream: bool) -> StreamingResponse:
    """LLM クライアント未接続時のレスポンス（stream=True 専用）"""
    sse = SSEFrameBuilder()

    async def _gen():
        from backend.i18n_helper import msg
        yield sse.error(msg('cli.llm_not_connected'))
        yield sse.done()

    return StreamingResponse(_gen(), media_type="text/event-stream")


def _resolve_system_prompt(state: AppState, mode: str, instance_name: str) -> str:
    """システムプロンプトを取得（PromptManager 未設定時はフォールバック）"""
    prompt_mgr = state.prompt_manager
    if prompt_mgr:
        return prompt_mgr.get_prompt(mode)
    return f"You are {instance_name}, a helpful AI assistant."


def _try_reactive_layer(
    req: ChatRequest,
    state: AppState,
    session_id: str,
    instance_name: str,
    context_size: int,
) -> StreamingResponse | ChatResponse | None:
    """Reactive 層でのパターンマッチ即応答。マッチしなければ None。"""
    reactive_agent = ReactiveAgent()
    reactive_resp = reactive_agent.process(req.message)
    if reactive_resp is None:
        return None

    logger.info("Reactive response: source=%s", reactive_resp.source)
    record_response(
        state, reactive_resp.content, [], session_id,
        req.message, req.mode, 0,
        private=req.private,
    )
    if req.stream:
        return StreamingResponse(
            stream_reactive(reactive_resp.content, instance_name, context_size),
            media_type="text/event-stream",
        )
    return ChatResponse(
        response=reactive_resp.content,
        token_info=TokenInfo(used=0, limit=context_size, pct=0,
                             instance_name=instance_name),
        session_id=session_id,
        agent_layer="reactive",
    )


async def _build_messages_with_search(
    req: ChatRequest,
    state: AppState,
    cfg: dict,
    system_prompt: str,
    history: list,
    file_contexts: list,
    context_size: int,
    max_tokens: int | None,
    timer: StageTimer,
    editor_route: str | None = None,
) -> tuple[list, StreamWrapper, str | None]:
    """統合検索を実行し ``messages`` / SSE 通知ラッパ / semmem ブロックを構築する。"""
    timer.start("search_ms")
    search_result = await run_search_pipeline(
        req.message, state, cfg, mode=req.mode, timer=timer,
    )
    timer.stop("search_ms")
    rag_chunks = search_result.chunks
    search_error = search_result.error
    scored_chunks = search_result.scored_chunks

    salience_ranker = None
    if scored_chunks:
        from backend.free.core.salience_ranker import SalienceRanker
        salience_ranker = SalienceRanker(
            policy=state.policy_interpreter, mode=req.mode,
        )

    # SemMem facts + STM notes を MemoryInjector で tier 整形して注入
    # (RAG とは独立、読み取りのみ)
    semmem_block = build_semmem_injection(state, cfg, mode=req.mode)

    messages = build_chat_messages(
        system_prompt, history, rag_chunks, file_contexts,
        context_size, max_tokens,
        rag_scored_chunks=scored_chunks,
        salience_ranker=salience_ranker,
        semmem_block=semmem_block,
    )

    sse_notify = SSEFrameBuilder()
    rag_debug_frame = _build_rag_debug_frame(state, scored_chunks, sse_notify, timer)

    async def _wrapper(inner_gen: AsyncIterator[str]) -> AsyncIterator[str]:
        """エディタ振り分け / 検索エラー通知 + RAG デバッグ情報をストリームの冒頭に挿入"""
        if editor_route is not None:
            yield sse_notify.editor_route(editor_route)
        if search_error:
            yield sse_notify.step({
                "type": "search_error",
                "detail": f"RAG search failed: {search_error}",
                "status": "failed",
            })
        if rag_debug_frame:
            yield rag_debug_frame
        async for frame in inner_gen:
            yield frame

    return messages, _wrapper, semmem_block


def _build_rag_debug_frame(
    state: AppState,
    scored_chunks: list,
    sse_notify: SSEFrameBuilder,
    timer: StageTimer,
) -> str | None:
    """デバッグモード時の RAG チャンク可視化フレームを構築"""
    dl = state.debug_logger
    if not (dl and dl.enabled and scored_chunks):
        return None
    rag_debug_chunks = [
        {
            "source": chunk_id,
            "score": round(score, 4),
            "preview": content[:100],
        }
        for chunk_id, score, content in scored_chunks
    ]
    search_time_ms = timer.to_dict().get("search_ms", 0.0)
    return sse_notify.rag_debug(rag_debug_chunks, search_time_ms)


async def _dispatch_long_form(
    req: ChatRequest,
    client,
    state: AppState,
    cfg: dict,
    gen_params: dict,
    session_id: str,
    instance_name: str,
    context_size: int,
    messages: list,
    search_error_wrapper: StreamWrapper,
    timer: StageTimer,
) -> StreamingResponse | ChatResponse:
    """Meta-Cognitive (long_form) 経路: 長文生成オーケストレータを起動する。"""
    base_ok, client = await ensure_base_model_health(client, state, cfg)
    if not base_ok:
        if req.stream:
            return _llm_unavailable_response(req.stream)
        raise HTTPException(status_code=503, detail="llama-server not connected")

    mem_sys = state.get_memory_system()
    orchestrator = LongFormOrchestrator(
        main_client=client,
        assist_client=state.assist_client,
        memory_wm=mem_sys[0] if mem_sys else None,
        memory_stm=mem_sys[1] if mem_sys else None,
        config=cfg,
        debug_logger=state.debug_logger,
        generation_params=gen_params,
        policy=state.policy_interpreter,
    )
    existing_content = await read_existing_for_append(req.message, state)

    if req.stream:
        return StreamingResponse(
            _with_chat_in_flight(client, search_error_wrapper(stream_long_form(
                orchestrator, req.message, session_id,
                req.mode, state, instance_name, context_size,
                messages, existing_content,
                timer=timer,
                private=req.private,
            ))),
            media_type="text/event-stream",
        )
    async with client.chat_in_flight():
        return await sync_long_form(
            orchestrator, req.message, session_id,
            req.mode, state, instance_name, context_size,
            messages, existing_content,
            timer=timer,
            private=req.private,
        )


async def _dispatch_meta_cognitive(
    req: ChatRequest,
    client,
    state: AppState,
    cfg: dict,
    gen_params: dict,
    system_prompt: str,
    history: list,
    session_id: str,
    instance_name: str,
    context_size: int,
    messages: list,
    search_error_wrapper: StreamWrapper,
    timer: StageTimer,
    semmem_block: str | None = None,
    output_target: str = "file",
) -> StreamingResponse | ChatResponse:
    """Meta-Cognitive (通常) 経路: 計画 + ツールループ。"""
    # @self 仮想カートリッジ用 LoopFactView 配線
    loop_view = _resolve_loop_view_for_agent(state)
    meta_agent = MetaCognitiveAgent(
        config=cfg,
        tool_judge=state.tool_call_judge,
        policy=state.policy_interpreter,
        agent_tracer=state.agent_tracer,
        loop_view=loop_view,
        project_id=state.current_project_id,
        # 計画立案 (`_plan`) は CLAUDE.md §1 に従いアシスト
        # モデルで実行する。``state.assist_client`` は health_check 失敗時
        # ``None`` (degraded mode) になるが、その場合 ``_plan`` は空リスト
        # を返し単一タスクへフォールバックする。
        assist_client=state.assist_client,
        # に記録 (decision_point=``meta_cognitive_llm_route``)
        debug_logger=state.debug_logger,
        # ツールループ全反復で SemMem メモリを維持 (初回ターンと同じ block)
        semmem_block=semmem_block,
    )
    keepalive_sec = cfg.get("streaming", {}).get(
        "keepalive_interval_sec", DEFAULT_KEEPALIVE_INTERVAL_SEC,
    )
    if req.stream:
        return StreamingResponse(
            _with_chat_in_flight(client, search_error_wrapper(stream_meta_cognitive(
                meta_agent, req.message, system_prompt, history,
                client, state, session_id, instance_name, context_size,
                messages, req.mode,
                generation_params=gen_params,
                keepalive_interval=keepalive_sec,
                timer=timer,
                private=req.private,
                output_target=output_target,
            ))),
            media_type="text/event-stream",
        )
    async with client.chat_in_flight():
        return await sync_meta_cognitive(
            meta_agent, req.message, system_prompt, history,
            client, state, session_id, instance_name, context_size,
            messages, req.mode,
            generation_params=gen_params,
            timer=timer,
            private=req.private,
            output_target=output_target,
        )


async def _dispatch_deliberative(
    req: ChatRequest,
    client,
    state: AppState,
    cfg: dict,
    gen_params: dict,
    history: list,
    session_id: str,
    instance_name: str,
    context_size: int,
    max_tokens: int | None,
    messages: list,
    search_error_wrapper: StreamWrapper,
    timer: StageTimer,
) -> StreamingResponse | ChatResponse:
    """Deliberative 経路: ツール判定 + LLM 推論。"""
    delib_agent = DeliberativeAgent(
        config=cfg,
        tool_judge=state.tool_call_judge,
        tools_registry=state.tools_registry,
    )

    if req.stream:
        return StreamingResponse(
            _with_chat_in_flight(client, search_error_wrapper(stream_deliberative(
                delib_agent, req.message, messages, client, state,
                session_id, instance_name, context_size,
                mode=req.mode, max_tokens=max_tokens,
                conversation=history,
                generation_params=gen_params,
                timer=timer,
                private=req.private,
            ))),
            media_type="text/event-stream",
        )
    async with client.chat_in_flight():
        return await sync_deliberative(
            delib_agent, req.message, messages, client, state,
            session_id, instance_name, context_size,
            mode=req.mode, max_tokens=max_tokens,
            conversation=history,
            generation_params=gen_params,
            timer=timer,
            private=req.private,
        )


@router.post("/chat")
async def chat(req: ChatRequest, state: AppState = Depends(get_app_state)):
    """SSE ストリーミングチャット応答（3層エージェントディスパッチ）"""
    trace_id = generate_trace_id()
    set_trace_id(trace_id)

    logger.debug(
        "POST /api/chat: mode=%s, stream=%s, message_len=%d, session=%s, trace_id=%s",
        req.mode, req.stream, len(req.message), req.session_id, trace_id,
    )
    _validate_chat_request(req)

    cfg = get_config()
    client = await ensure_llm_client(state, cfg)
    if client is None:
        if req.stream:
            return _llm_unavailable_response(req.stream)
        raise HTTPException(status_code=503, detail="llama-server not connected")

    instance_name = cfg.get("instance", {}).get("name", "evoref")
    context_size = cfg.get("llama", {}).get("context_size", DEFAULT_CONTEXT_SIZE)
    max_tokens = cfg.get("llama", {}).get("max_tokens", DEFAULT_MAX_TOKENS) or None
    gen_params = get_mode_generation_params(req.mode)

    if state.sleep_scheduler:
        state.sleep_scheduler.on_user_input()

    history, session_id = await prepare_memory_context(req, state)
    file_contexts = convert_file_contexts(req)
    system_prompt = _resolve_system_prompt(state, req.mode, instance_name)

    classifier = ComplexityClassifier(
        config=cfg,
        learned_patterns=getattr(state, "learned_patterns_store", None),
        policy=state.policy_interpreter,
    )
    agent_layer = classifier.classify(req.message, mode=req.mode)
    logger.info("Agent layer: %s for query: %s", agent_layer, req.message[:80])

    if agent_layer == "reactive":
        # URL recall プリチェック: 過去会話で fetch 済みの URL fact が
        # クエリに意味的にヒットする場合、reactive で前知識のみ応答せず
        # deliberative にエスカレートして fetch_url を実行させる。
        # ``recall_url_judgement`` は閾値・TTL・profile match まで判定
        # 済みのため、ヒット時のみ True/None を返す軽量チェック。
        if (
            state.tool_call_judge is not None
            and state.tools_registry is not None
        ):
            try:
                recall_judgement = await state.tool_call_judge.recall_url_judgement(
                    req.message, state.tools_registry,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "URL recall pre-check failed (continuing as reactive): %s", exc,
                )
                recall_judgement = None
            if recall_judgement is not None and recall_judgement.tool_needed:
                agent_layer = "deliberative"
                logger.info(
                    "Reactive escalated to deliberative due to URL recall hit: %s",
                    req.message[:80],
                )

    if agent_layer == "reactive":
        reactive_response = _try_reactive_layer(
            req, state, session_id, instance_name, context_size,
        )
        if reactive_response is not None:
            return reactive_response
        agent_layer = "deliberative"
        logger.info("Reactive returned None, escalating to deliberative")

    # coding モードのみ生成コードの出力先を決定する。
    # - 出力先パス明示 → "file" (従来どおり write_file でディスクへ)
    # - 否定指示 ("エディタに出さず…") → "chat" (チャット本文にコードブロック)
    # - 既定 → "editor" (ディスク書込せず editor_code チャネルでエディタペインへ)
    # editor_route SSE フレームはフロント表示制御 (suppressCode) 用に併せて通知する。
    if req.mode == "coding":
        if _extract_file_path(req.message):
            output_target = "file"
        elif detect_editor_route(req.message) == "chat":
            output_target = "chat"
        else:
            output_target = "editor"
        editor_route = "editor" if output_target == "editor" else "chat"
    else:
        output_target = "file"
        editor_route = None

    timer = StageTimer()
    messages, search_error_wrapper, semmem_block = await _build_messages_with_search(
        req, state, cfg, system_prompt, history, file_contexts,
        context_size, max_tokens, timer,
        editor_route=editor_route,
    )

    match agent_layer:
        case "meta_cognitive" if classifier.is_long_form:
            return await _dispatch_long_form(
                req, client, state, cfg, gen_params, session_id,
                instance_name, context_size, messages, search_error_wrapper, timer,
            )
        case "meta_cognitive":
            return await _dispatch_meta_cognitive(
                req, client, state, cfg, gen_params, system_prompt, history,
                session_id, instance_name, context_size,
                messages, search_error_wrapper, timer,
                semmem_block=semmem_block,
                output_target=output_target,
            )
        case _:
            return await _dispatch_deliberative(
                req, client, state, cfg, gen_params, history,
                session_id, instance_name, context_size, max_tokens,
                messages, search_error_wrapper, timer,
            )


@router.post("/chat/cancel", response_model=CancelResponse)
async def cancel_chat(req: CancelRequest):
    """ストリーミング生成を中断"""
    logger.debug("POST /api/chat/cancel: session=%s", req.session_id)
    if req.session_id in _cancel_flags:
        _cancel_flags[req.session_id] = True
        logger.debug("Cancel flag set for session %s", req.session_id)
        return CancelResponse(cancelled=True)
    return CancelResponse(cancelled=False, tokens_generated=0)
