"""チャット API（SSE ストリーミング + 3層エージェントディスパッチ）"""

import asyncio
import re
from collections.abc import AsyncIterator, Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.app_state import AppState, get_app_state
from backend.config import (
    get_config,
    get_mode_generation_params,
    resolve_context_size_for_mode,
)
from backend.free.api.chat.chat_constants import (
    DEFAULT_KEEPALIVE_INTERVAL_SEC, DEFAULT_MAX_TOKENS,
    MAX_FILE_CONTEXT_TOTAL_CHARS, MAX_FILE_CONTEXT_TOTAL_CHUNKS,
    MAX_MESSAGE_LENGTH,
    REACTIVE_LIGHT_HISTORY_TURNS, REACTIVE_LIGHT_MAX_TOKENS,
    SESSION_ID_MAX_LENGTH, SESSION_ID_MIN_LENGTH,
)
from backend.free.api.schemas import (
    CancelRequest, CancelResponse, ChatRequest, ChatResponse, TokenInfo,
)
from backend.free.api.chat._editor_routing import detect_editor_route
from backend.free.agent.tool_call_judge import _extract_file_path
from backend.free.api.chat.chat_recorder import record_response
from backend.free.api.chat.chat_service import (
    ConflictTurnContext,
    SearchPipelineResult,
    _render_conflict_section,
    build_chat_messages, build_semmem_injection, convert_file_contexts,
    ensure_base_model_health,
    ensure_llm_client, maybe_resolve_pending_conflicts, prepare_memory_context,
    run_search_pipeline,
)
from backend.free.api.chat.chat_streaming import (
    _cancel_flags,
    rag_signals_from_chunks,
    read_existing_for_append,
    stream_deliberative, stream_long_form, stream_meta_cognitive, stream_reactive,
    stream_reactive_light, stream_staged_coding,
    sync_deliberative, sync_long_form, sync_meta_cognitive, sync_reactive_light,
)
from backend.edition import is_pro
from backend.free.core.sse import SSEFrameBuilder
from backend.free.agent.deliberative import DeliberativeAgent
from backend.free.agent.meta_cognitive import MetaCognitiveAgent
from backend.free.agent.reactive import ReactiveAgent
from backend.free.agent.router import ComplexityClassifier
from backend.free.core.stage_timer import StageTimer
from backend.free.generation.orchestrator import LongFormOrchestrator
from backend.free.generation.content_detector import detect_content_type
from backend.free.generation.direct_codegen import generate_single_file
from backend.free.generation.models import ContentType
from backend.free.agent.meta_cognitive_tasks import EditorArtifact
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
    except Exception as exc:
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


def _llm_unavailable_response(stream: bool) -> StreamingResponse:  # noqa: ARG001
    """LLM クライアント未接続時のレスポンス（stream=True 専用）"""
    sse = SSEFrameBuilder()

    async def _gen():
        from backend.i18n_helper import msg
        yield sse.error(msg('cli.llm_not_connected'))
        yield sse.done()

    return StreamingResponse(_gen(), media_type="text/event-stream")


def _resolve_system_prompt(
    state: AppState, mode: str, instance_name: str,
) -> str:
    """静的システムプロンプトを取得（PromptManager 未設定時はフォールバック）。

    query 非依存 (few-shot を含まない) なので連続リクエスト間で安定し、
    llama-server の prefix KV キャッシュが効く。few-shot は
    ``_resolve_fewshot_block`` で別途取得し最後の user メッセージへ前置する。
    """
    prompt_mgr = state.prompt_manager
    if prompt_mgr:
        get_static = getattr(prompt_mgr, "get_prompt_static", None)
        if get_static is not None:
            return get_static(mode)
        # 後方互換: get_prompt_static 未実装の Mock 等は query なし get_prompt へ縮退
        return prompt_mgr.get_prompt(mode)
    return f"You are {instance_name}, a helpful AI assistant."


def _resolve_fewshot_block(state: AppState, mode: str, query: str | None) -> str:
    """query 依存の few-shot ブロックを取得 ("" = 無し / PromptManager 未設定)。"""
    prompt_mgr = state.prompt_manager
    if prompt_mgr is None:
        return ""
    get_block = getattr(prompt_mgr, "get_fewshot_block", None)
    if get_block is None:
        return ""
    return get_block(mode, query)


def _try_reactive_layer(
    req: ChatRequest,
    state: AppState,
    session_id: str,
    instance_name: str,
    context_size: int,
) -> StreamingResponse | ChatResponse | None:
    """Reactive 層でのパターンマッチ即応答。マッチしなければ None。"""
    # 常駐インスタンスを使う (LRU キャッシュをセッション跨ぎで温める)。
    # 未配線環境 (テスト等) では新規生成にフォールバック。
    reactive_agent = state.reactive_agent or ReactiveAgent()
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


async def _completed(value):
    """既知の値を返す done タスク化用コルーチン (judge 結果を deliberative へ流用)。"""
    return value


def _log_layer_escalation(state: AppState, *, chosen: str, reason: str) -> None:
    """reactive→light/deliberative の分岐を decision.jsonl へ記録 (evolve レベル限定)。"""
    dl = getattr(state, "debug_logger", None)
    if dl is None:
        return
    dl.log_decision(
        decision_point="layer_escalation",
        chosen=chosen,
        candidates=["reactive_rule", "reactive_light", "deliberative"],
        reason=reason,
        scope="request",
    )


async def _gate_reactive_light(
    req: ChatRequest,
    state: AppState,
    cfg: dict,
    history: list,
    judge_task: "asyncio.Task | None",
) -> tuple[str, "asyncio.Task | None", str]:
    """reactive ルール miss 後、軽量パス採否を判定する。

    Returns ``(decision, judge_task, reason)``:
      - decision: ``"light"`` (base 1 ターン軽量パス) | ``"deliberative"`` (エスカレート)
      - judge_task: deliberative へ流用する done 済み判定タスク (light / None もありうる)
      - reason: log_decision / ログ用の英語識別子
    """
    if not cfg.get("agent", {}).get("reactive_light_enabled", True):
        return "deliberative", judge_task, "light_disabled"
    # 添付ファイルを無視した軽量応答は品質事故 → deliberative
    if getattr(req, "file_contexts", None):
        return "deliberative", judge_task, "file_context"
    if state.tool_call_judge is None or state.tools_registry is None:
        return "deliberative", judge_task, "judge_unavailable"

    try:
        if judge_task is not None:
            judgement = await judge_task  # 投機起動済み (並列構成)、残り時間のみ待つ
        else:
            judgement = await state.tool_call_judge.judge(  # 直列構成で直接実行
                req.message, state.tools_registry, req.mode, history,
            )
    except Exception as exc:
        logger.warning("reactive-light judge failed, escalating: %s", exc)
        return "deliberative", judge_task, "judge_error"

    if judgement is not None and judgement.tool_needed:
        # deliberative へ流用 (直列構成なら done タスク化して再 judge を防ぐ)
        if judge_task is None:
            judge_task = asyncio.create_task(_completed(judgement))
        return "deliberative", judge_task, "tool_needed"
    return "light", judge_task, "judge_no_tool"


async def _dispatch_reactive_light(
    req: ChatRequest,
    client,
    state: AppState,
    gen_params: dict,
    history: list,
    session_id: str,
    instance_name: str,
    context_size: int,
    max_tokens: int | None,
    timer: StageTimer,
    conflict_notice: str | None = None,
) -> "StreamingResponse | ChatResponse":
    """Reactive 軽量パス dispatch: 静的 system + 履歴末尾で base 1 ターン (few-shot/RAG/semmem なし)。

    ``conflict_notice`` が指定された場合のみ、記憶の競合セクション
    (解決通知・確認指示) を system プロンプトに連結して surface する。
    deliberative の build_semmem_injection と同じ注入機構だが、軽量パスを
    維持したまま通知を出すための最小注入。
    """
    system_prompt = _resolve_system_prompt(state, req.mode, instance_name)
    if conflict_notice:
        system_prompt = f"{system_prompt}\n\n{conflict_notice}"
    light_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    light_messages.extend(history[-REACTIVE_LIGHT_HISTORY_TURNS:])
    light_max = min(max_tokens or REACTIVE_LIGHT_MAX_TOKENS, REACTIVE_LIGHT_MAX_TOKENS)
    if req.stream:
        return StreamingResponse(
            _with_chat_in_flight(client, stream_reactive_light(
                req.message, light_messages, client, state, session_id,
                instance_name, context_size,
                mode=req.mode, max_tokens=light_max,
                generation_params=gen_params, timer=timer, private=req.private,
            )),
            media_type="text/event-stream",
        )
    async with client.chat_in_flight():
        return await sync_reactive_light(
            req.message, light_messages, client, state, session_id,
            instance_name, context_size,
            mode=req.mode, max_tokens=light_max,
            generation_params=gen_params, timer=timer, private=req.private,
        )


def _realtime_parallel_enabled(state: AppState) -> bool:
    """チャット応答パスの assist 判定を並列化してよいか。

    assist の realtime セマフォが 2 以上 (= サーバ側 slots とセットで並列度が
    確保された構成) のときだけ True。1 (既定) では現行の直列フローを維持する。
    """
    client = state.assist_client
    if client is None:
        return False
    return getattr(client, "realtime_concurrency", 1) >= 2


async def _run_search_timed(
    req: ChatRequest, state: AppState, cfg: dict, timer: StageTimer,
) -> SearchPipelineResult:
    """検索パイプラインを ``search_ms`` 計測付きで実行する。

    chat() で ``asyncio.create_task`` 化して conflict 判定 / tool 判定と並走
    させる入口。``run_search_pipeline`` は内部で例外を握って
    ``SearchPipelineResult(error=...)`` を返すため、ここでは計測のみ担う。
    """
    timer.start("search_ms")
    try:
        return await run_search_pipeline(
            req.message, state, cfg, mode=req.mode, timer=timer,
        )
    finally:
        timer.stop("search_ms")


def _cancel_pending_task(task: "asyncio.Task | None") -> None:
    """投機タスクが未完了なら cancel する (reactive 早期 return / 経路不一致時)。"""
    if task is not None and not task.done():
        task.cancel()


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
    conflict_ctx: ConflictTurnContext | None = None,
    search_task: "asyncio.Task | None" = None,
    fewshot_block: str | None = None,
) -> tuple[list, StreamWrapper, str | None, list[tuple[str, float, str]] | None]:
    """統合検索を実行し ``messages`` / SSE 通知ラッパ / semmem ブロック / 取得済み
    scored_chunks を構築する。``scored_chunks`` は long_form 経路が orchestrator に
    再利用注入するために返す (非 long_form 経路は ``messages`` 側で消費するため未使用)。

    ``search_task`` が渡された場合は chat() が先行起動した検索タスクを await して
    回収する (conflict 判定 / tool 判定との並走)。None の場合はここで直列実行する。

    ``system_prompt`` は静的 (query 非依存)、``fewshot_block`` 等の query 依存部は
    build_messages 内で最後の user メッセージへ前置される (KV キャッシュ対応)。"""
    if search_task is not None:
        try:
            search_result = await search_task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Search task failed, continuing without RAG: %s", exc)
            search_result = SearchPipelineResult(error=str(exc))
    else:
        search_result = await _run_search_timed(req, state, cfg, timer)
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
    # (RAG とは独立、読み取りのみ)。pending 競合セクションも併せて連結する。
    semmem_block = build_semmem_injection(
        state, cfg, mode=req.mode, conflict_ctx=conflict_ctx,
    )

    messages = build_chat_messages(
        system_prompt, history, rag_chunks, file_contexts,
        context_size, max_tokens,
        rag_scored_chunks=scored_chunks,
        salience_ranker=salience_ranker,
        semmem_block=semmem_block,
        fewshot_block=fewshot_block,
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

    return messages, _wrapper, semmem_block, scored_chunks


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


# meta_cognitive ループ system へ渡す RAG 参考ブロックの整形上限。
# long_form の prefetched_rag と同じ取得結果を、salience 順 (search pipeline 順)
# のまま上位数件・各チャンク要約で連結する。
_META_RAG_MAX_CHUNKS = 5
_META_RAG_CHAR_CAP = 1200


def _format_rag_block_for_meta(
    scored_chunks: list[tuple[str, float, str]] | None,
) -> str | None:
    """search pipeline 取得済み ``scored_chunks`` を meta ループ用ブロックに整形。

    deliberative 経路は ``messages`` 側で RAG を消費するが、meta_cognitive 経路は
    ``messages`` を LLM に渡さないため、取得済みチャンクをループ system に注入する
    (semmem_block と同じ消費形)。整形できる内容が無ければ ``None``。
    """
    if not scored_chunks:
        return None
    parts: list[str] = []
    for i, (_chunk_id, score, text) in enumerate(
        scored_chunks[:_META_RAG_MAX_CHUNKS]
    ):
        snippet = (text or "")[:_META_RAG_CHAR_CAP]
        if not snippet:
            continue
        parts.append(f"[参考情報 {i + 1}] (score={score:.2f})\n{snippet}")
    if not parts:
        return None
    return "\n\n".join(parts)


# meta / long_form 経路へ渡す添付ファイルブロックの整形上限。
_FILE_BLOCK_CHAR_CAP = 4000


def _format_file_block(
    file_contexts: list | None,
) -> str | None:
    """``convert_file_contexts`` 出力 (``{filename, chunks}`` のリスト) を、
    meta / long_form 経路の system へ注入するブロック文字列に整形する。

    deliberative 経路は ``messages`` 側でファイルを消費するが、meta_cognitive /
    long_form 経路は ``messages`` を LLM に渡さないため、添付内容を別途注入する。
    整形できる内容が無ければ ``None``。
    """
    if not file_contexts:
        return None
    sections: list[str] = []
    used = 0
    for fc in file_contexts:
        filename = fc.get("filename", "unknown")
        chunks = fc.get("chunks", []) or []
        body = "\n\n".join(chunks)
        section = f"[ファイル: {filename}]\n{body}" if body else f"[ファイル: {filename}]"
        section = section[:_FILE_BLOCK_CHAR_CAP]
        if used + len(section) > _FILE_BLOCK_CHAR_CAP and sections:
            break
        sections.append(section)
        used += len(section)
    if not sections:
        return None
    return "\n\n---\n\n".join(sections)


def _build_long_form_orchestrator(
    client, state: AppState, cfg: dict, gen_params: dict,
) -> LongFormOrchestrator:
    """LongFormOrchestrator を構築する (long_form ディスパッチ / コード委譲で共用)。"""
    mem_sys = state.get_memory_system()
    return LongFormOrchestrator(
        main_client=client,
        assist_client=state.assist_client,
        memory_wm=mem_sys[0] if mem_sys else None,
        config=cfg,
        debug_logger=state.debug_logger,
        generation_params=gen_params,
        policy=state.policy_interpreter,
    )


# editor タブ表示用の拡張子 → 言語ラベル (フロントのシンタックスハイライト向け)。
_CODE_EXT_LANG: dict[str, str] = {
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


def _artifact_from_file(path: str, code: str) -> EditorArtifact:
    """orchestrator の last_code_files エントリを EditorArtifact に変換する。"""
    from pathlib import PurePosixPath

    pp = PurePosixPath(path) if path else None
    name = pp.name if pp else ""
    ext = pp.suffix.lstrip(".").lower() if pp else ""
    return EditorArtifact(
        content=code,
        language=_CODE_EXT_LANG.get(ext, "python"),
        filename=name or None,
    )


def _validation_issues_artifact(errors: list[str]) -> EditorArtifact:
    """リペア後も残った検証エラーを提示する markdown artifact を生成する。

    生成コードは best-effort で配信しつつ、未解決エラーをユーザーに明示し、
    壊れたコードを無言で「成功」扱いしないための可視化。
    """
    lines = "\n".join(f"- {e}" for e in errors)
    content = (
        f"# ⚠️ 自動検証で未解決のエラーが {len(errors)} 件あります\n\n"
        "生成されたコードには以下の検証エラーが残っています。"
        "実行前に修正してください。\n\n"
        f"{lines}\n"
    )
    return EditorArtifact(
        content=content,
        language="markdown",
        filename="GENERATION_ISSUES.md",
    )


def _clamp_long_form_timeout(cfg: dict) -> dict:
    """coding 委譲時、orchestrator の total_timeout_sec を agent 上限未満にする。

    agent (`_total_timeout` 既定 900s、超過時に artifacts 破棄) より短く打ち切り、
    orchestrator 側でユニット境界の部分結果 + repair を確定させる (artifacts 喪失回避)。
    """
    agent_total = float((cfg.get("agent") or {}).get("total_timeout", 900) or 900)
    clamped = max(300.0, agent_total - 90.0)
    lf_cfg = dict(cfg.get("long_form") or {})
    existing = float(lf_cfg.get("total_timeout_sec", 1800.0) or 0.0)
    lf_cfg["total_timeout_sec"] = clamped if existing <= 0 else min(existing, clamped)
    return {**cfg, "long_form": lf_cfg}


def make_code_artifact_generator(
    client, state: AppState, cfg: dict, gen_params: dict, session_id: str,
):
    """coding の editor/chat 出力向け code_generator を返す (MetaCognitiveAgent に注入)。

    指示文を LongFormOrchestrator の細粒度 CodeUnit 計画で生成し、ファイル別の
    検証・修正済みコードを EditorArtifact 群 (複数ファイル可) として返す。テキスト
    判定 / 生成失敗時は空リストを返し、agent の単一ショット生成にフォールバックさせる。
    """
    gen_cfg = _clamp_long_form_timeout(cfg)

    async def _generate(instruction: str, on_step=None) -> list[EditorArtifact]:  # noqa: ARG001
        if detect_content_type(instruction, "coding") != ContentType.CODE:
            return []
        orchestrator = _build_long_form_orchestrator(
            client, state, gen_cfg, gen_params,
        )
        try:
            # orchestrator の _call_step は sync 呼出 (on_step(data)) だが、
            # MetaCognitive ランナーの on_step は async。委譲時に転送すると
            # 「coroutine never awaited」で進捗フレームを取りこぼすため転送しない
            # (進捗は agent 側の task_progress が表示する)。
            async for _token in orchestrator.generate(
                instruction=instruction, session_id=session_id,
                mode="coding", on_step=None,
            ):
                pass
        except Exception as e:
            logger.warning("Delegated code generation failed: %s", e)
            return []
        files = orchestrator.last_code_files or (
            {"output.py": orchestrator.last_code_output}
            if orchestrator.last_code_output else {}
        )
        artifacts = [
            _artifact_from_file(path, code)
            for path, code in files.items()
            if code and code.strip()
        ]
        # 「壊れたコードを成功として渡さない」: リペア後も残った検証エラーがあれば、
        # 生成物は best-effort で返しつつ、未解決エラーを可視化する artifact を添える。
        if artifacts and orchestrator.last_validation_errors:
            artifacts.append(_validation_issues_artifact(
                orchestrator.last_validation_errors,
            ))
        return artifacts

    return _generate


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
    output_target: str = "file",
    prefetched_rag: list[tuple[str, float, str]] | None = None,
    file_context_block: str | None = None,
) -> StreamingResponse | ChatResponse:
    """Meta-Cognitive (long_form) 経路: 長文生成オーケストレータを起動する。

    ``output_target`` は coding モード時の出力先 (``"file"`` / ``"editor"`` /
    ``"chat"``) を ``stream_long_form`` / ``sync_long_form`` に伝播する。
    既定 ``"file"`` (チャット応答パス互換)。
    """
    base_ok, client = await ensure_base_model_health(client, state, cfg)
    if not base_ok:
        if req.stream:
            return _llm_unavailable_response(req.stream)
        raise HTTPException(status_code=503, detail="llama-server not connected")

    orchestrator = _build_long_form_orchestrator(client, state, cfg, gen_params)
    existing_content = await read_existing_for_append(req.message, state)

    if req.stream:
        return StreamingResponse(
            _with_chat_in_flight(client, search_error_wrapper(stream_long_form(
                orchestrator, req.message, session_id,
                req.mode, state, instance_name, context_size,
                messages, existing_content,
                timer=timer,
                private=req.private,
                output_target=output_target,
                prefetched_rag=prefetched_rag,
                file_context_block=file_context_block,
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
            output_target=output_target,
            prefetched_rag=prefetched_rag,
            file_context_block=file_context_block,
        )


def make_staged_codegen_delegate(client, cfg: dict):
    """base コーディングモデル経由の codegen 委譲を作る
    ((instruction, file_path) -> {path: code})。

    ``StagedCodingExecutor`` に注入する。以前は ``make_code_artifact_generator`` と
    同じ LongFormOrchestrator 経路 (plan/CodeSpec 再合成 + CodeUnit 細粒度分割生成)
    を経由していたが、これは instruction (spec.md 全文 + flowchart + 契約ブロック)
    の大半をアシストの再合成・トークン予算切り詰めで失い、生成コードが仕様と乖離
    する原因になっていた (副作用として ``detect_content_type`` の TEXT 誤判定対策
    も必要だった)。staged は ``synthesize_coding_task_graph`` が既にプログラムを
    ファイル単位へ決定的に分解済みのため、1 code タスク = 1 ファイルの単発生成で
    足りる。``direct_codegen.generate_single_file`` で base モデルへの単発呼び出し
    のみに委譲し、instruction を無劣化のまま渡す (再計画・content_type 判定は
    どちらも不要になる)。
    """
    staged_cfg = (cfg.get("coding", {}) or {}).get("staged", {}) or {}
    max_tokens = int(staged_cfg.get("code_max_tokens", 4096))

    async def _generate(instruction: str, file_path: str) -> dict[str, str]:
        return await generate_single_file(
            client, instruction, file_path, max_tokens=max_tokens,
        )

    return _generate


def _staged_coding_enabled(req: ChatRequest, cfg: dict, state: AppState) -> bool:
    """staged コーディングパイプラインを起動すべきか判定する。

    全条件を満たすときのみ True。いずれか欠ければ従来 longform 経路へ倒す。
    """
    if req.mode != "coding":
        return False
    coding_cfg = cfg.get("coding", {}) or {}
    if coding_cfg.get("pipeline") != "staged":
        return False
    if not coding_cfg.get("staged_enabled", True):
        return False
    if getattr(state, "assist_client", None) is None:
        return False
    if not is_pro():
        return False
    try:
        if detect_content_type(req.message, "coding") != ContentType.CODE:
            return False
    except Exception:
        return False
    return True


async def _dispatch_staged_coding(
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
    output_target: str = "file",
    prefetched_rag: list[tuple[str, float, str]] | None = None,
    file_context_block: str | None = None,
) -> StreamingResponse | ChatResponse:
    """staged コーディング: 専用 LoopDriver をインライン駆動し spec→code→test を実行。

    非ストリーミング要求 / base 不健全時は従来 longform 経路へフォールバックする。
    タスクグラフ合成が空 (assist degraded 等) のときも stream 内で longform へ委譲。
    """
    base_ok, client = await ensure_base_model_health(client, state, cfg)
    if not base_ok:
        if req.stream:
            return _llm_unavailable_response(req.stream)
        raise HTTPException(status_code=503, detail="llama-server not connected")

    orchestrator = _build_long_form_orchestrator(client, state, cfg, gen_params)
    existing_content = await read_existing_for_append(req.message, state)

    # staged はストリーミング前提。非ストリーム要求は従来 longform に委譲する。
    if not req.stream:
        async with client.chat_in_flight():
            return await sync_long_form(
                orchestrator, req.message, session_id,
                req.mode, state, instance_name, context_size,
                messages, existing_content,
                timer=timer, private=req.private, output_target=output_target,
                prefetched_rag=prefetched_rag, file_context_block=file_context_block,
            )

    codegen = make_staged_codegen_delegate(client, cfg)

    def _fallback_factory():
        # 合成失敗時のフォールバック (外側で search_error_wrapper 済みのため raw)
        return stream_long_form(
            orchestrator, req.message, session_id,
            req.mode, state, instance_name, context_size,
            messages, existing_content,
            timer=timer, private=req.private, output_target=output_target,
            prefetched_rag=prefetched_rag, file_context_block=file_context_block,
        )

    return StreamingResponse(
        _with_chat_in_flight(client, search_error_wrapper(stream_staged_coding(
            query=req.message, session_id=session_id, state=state, cfg=cfg,
            instance_name=instance_name, context_size=context_size,
            messages=messages, output_target=output_target,
            codegen=codegen, fallback_factory=_fallback_factory,
            timer=timer, private=req.private,
        ))),
        media_type="text/event-stream",
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
    rag_block: str | None = None,
    file_block: str | None = None,
    fewshot_block: str | None = None,
    output_target: str = "file",
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> StreamingResponse | ChatResponse:
    """Meta-Cognitive (通常) 経路: 計画 + ツールループ。"""
    # @self 仮想カートリッジ用 LoopFactView 配線
    loop_view = _resolve_loop_view_for_agent(state)
    # coding モード: editor/chat 出力のコード生成を LongForm 細粒度生成へ委譲する
    # generator を注入する (複数ファイル可)。非 coding / 無効時は None。
    code_generator = None
    if (
        req.mode == "coding"
        and cfg.get("agent", {}).get("delegate_codegen_to_longform", True)
    ):
        code_generator = make_code_artifact_generator(
            client, state, cfg, gen_params, session_id,
        )
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
        # search pipeline 取得済み RAG を維持 (long_form の prefetched_rag と同型)
        rag_block=rag_block,
        # 添付ファイル内容を維持 (deliberative の messages 注入と等価)
        file_block=file_block,
        # Level 1 進化 few-shot を維持 (固定 scaffold の [参考例] に注入)
        fewshot_block=fewshot_block,
        code_generator=code_generator,
        # 内部 loop/token 予算を coding_model の実窓に合わせる
        mode=req.mode,
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
                rag_used=rag_used,
                rag_top1_score=rag_top1_score,
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
            rag_used=rag_used,
            rag_top1_score=rag_top1_score,
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
    rag_used: bool = False,
    rag_top1_score: float | None = None,
    tool_judge_task: "asyncio.Task | None" = None,
    escalated_from: str | None = None,
) -> StreamingResponse | ChatResponse:
    """Deliberative 経路: ツール判定 + LLM 推論。

    ``tool_judge_task`` が渡された場合は chat() が先行起動した tool 判定タスクを
    再利用する (process() 内で await)。None の場合は process() が判定を直列実行。
    ``escalated_from`` は reactive からエスカレートした場合の出自 (outcome 観測用)。"""
    delib_agent = DeliberativeAgent(
        config=cfg,
        tool_judge=state.tool_call_judge,
        tools_registry=state.tools_registry,
        assist_client=state.assist_client,
        assist_experience_recorder=state.assist_experience_recorder,
        agent_tracer=state.agent_tracer,
        # コンテンツ生成 max_tokens を coding_model の実窓に合わせる
        mode=req.mode,
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
                rag_used=rag_used,
                rag_top1_score=rag_top1_score,
                tool_judge_task=tool_judge_task,
                escalated_from=escalated_from,
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
            rag_used=rag_used,
            rag_top1_score=rag_top1_score,
            tool_judge_task=tool_judge_task,
            escalated_from=escalated_from,
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
    context_size = resolve_context_size_for_mode(cfg, req.mode)
    max_tokens = cfg.get("llama", {}).get("max_tokens", DEFAULT_MAX_TOKENS) or None
    gen_params = get_mode_generation_params(req.mode)

    if state.sleep_scheduler:
        state.sleep_scheduler.on_user_input()

    history, session_id = await prepare_memory_context(req, state)
    file_contexts = convert_file_contexts(req)
    # system は静的 (query 非依存) に保ち KV キャッシュを効かせる。query 依存の
    # few-shot は動的ブロックとして最後の user メッセージへ前置する (build_messages)。
    system_prompt = _resolve_system_prompt(state, req.mode, instance_name)
    fewshot_block = _resolve_fewshot_block(state, req.mode, req.message)

    # classify は conflict 結果に依存しない (req.message のみ) ため先に確定し、
    # 並列モードでの投機タスク (tool 判定 / 検索) 起動のゲートに使う。
    classifier = ComplexityClassifier(
        config=cfg,
        learned_patterns=getattr(state, "learned_patterns_store", None),
        policy=state.policy_interpreter,
    )
    agent_layer = classifier.classify(req.message, mode=req.mode)
    logger.info(
        "Agent layer: %s (mode=%s) for query: %s",
        agent_layer, req.mode, req.message[:80],
    )
    # primary routing を decision.jsonl に記録 (evolve 限定)。後続の reactive→
    # light/deliberative escalation (_log_layer_escalation) は別 decision_point。
    # context={"mode"} は policy_adjuster が mode 別 routing 学習に使う (load-bearing)。
    dl = getattr(state, "debug_logger", None)
    if dl is not None:
        dl.log_decision(
            decision_point="layer_classification",
            chosen=agent_layer,
            candidates=["reactive", "deliberative", "meta_cognitive"],
            reason=getattr(classifier, "_last_classify_reason", "default"),
            context={"mode": req.mode},
            scope="request",
        )

    timer = StageTimer()
    # pending 競合のユーザー回答判定 + 即時反映 (不変則例外 (b)、解決結果は
    # 同ターンの semmem 注入へ反映)。private ターンは SemMem へ書かない契約のため
    # allow_write=False (注入のみ継続)。
    #
    # realtime 並列化が有効なときは conflict 判定 / 検索パイプライン / tool 判定を
    # 同時起動して直列待ちを畳む。依存順は守る:
    #   - conflict_ctx は build_semmem_injection より前に await 済みとし、解決
    #     通知の同ターン注入契約を維持する。検索パイプラインは SemMem facts を
    #     注入に使わない (読取は injection 側) ため conflict 書込と競合しない。
    #   - tool 判定 (judge) は preliminary layer が meta_cognitive 以外のときのみ
    #     投機する (meta は task 記述単位で judge するため query 単位の流用不可、
    #     long_form は meta_cognitive 分類配下なので自動的に除外される)。
    #   - 検索は preliminary layer が reactive 以外のときのみ投機する (reactive
    #     即応答は検索結果を使わない)。reactive→deliberative にエスカレートした
    #     場合は search_task=None で _build_messages_with_search が直列実行する。
    judge_task: asyncio.Task | None = None
    search_task: asyncio.Task | None = None
    try:
        if _realtime_parallel_enabled(state):
            conflict_task = asyncio.create_task(
                maybe_resolve_pending_conflicts(
                    state, cfg, history, req.message,
                    allow_write=not req.private, session_id=session_id,
                )
            )
            if (
                agent_layer != "meta_cognitive"
                and state.tool_call_judge is not None
                and state.tools_registry is not None
            ):
                judge_task = asyncio.create_task(
                    state.tool_call_judge.judge(
                        req.message, state.tools_registry, req.mode, history,
                    )
                )
            if agent_layer != "reactive":
                search_task = asyncio.create_task(
                    _run_search_timed(req, state, cfg, timer),
                )
            try:
                conflict_ctx = await conflict_task
            except Exception as exc:
                logger.warning("Conflict review task failed (degrading): %s", exc)
                conflict_ctx = ConflictTurnContext()
        else:
            conflict_ctx = await maybe_resolve_pending_conflicts(
                state, cfg, history, req.message,
                allow_write=not req.private, session_id=session_id,
            )

        # reactive→deliberative エスカレート時に Level 0 経験記録の出自を残す。
        escalated_from: str | None = None

        # pending / 解決済み競合がある場合、競合セクション (解決通知・確認指示) を
        # 必ず surface する必要がある。従来は full deliberative へ昇格していたが、
        # 雑談返信でも 32-140s かかる。通知ブロックを軽量パス (reactive_light) の
        # プロンプトへ差し込めば、同じ base 生成機構で surface しつつ重経路を回避
        # できる (注入機構は deliberative と同一: プロンプト連結 + base 生成。
        # むしろ短いプロンプトで通知が目立つ)。rule-instant の canned 応答だけは
        # 通知を運べないため、競合時はスキップして必ず LLM ターンを経由させる。
        conflict_notice: str | None = None
        if agent_layer == "reactive" and (
            conflict_ctx.pending_groups or conflict_ctx.resolved is not None
        ):
            conflict_notice = _render_conflict_section(state, cfg, conflict_ctx)

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
                        req.message, state.tools_registry, mode=req.mode,
                    )
                except Exception as exc:
                    logger.warning(
                        "URL recall pre-check failed (continuing as reactive): %s", exc,
                    )
                    recall_judgement = None
                if recall_judgement is not None and recall_judgement.tool_needed:
                    agent_layer = "deliberative"
                    escalated_from = "reactive"
                    _log_layer_escalation(state, chosen="deliberative", reason="url_recall_hit")
                    logger.info(
                        "Reactive escalated to deliberative due to URL recall hit: %s",
                        req.message[:80],
                    )

        if agent_layer == "reactive":
            # 競合通知がある場合は canned 即応答 (挨拶/日時/キャッシュ) では通知を
            # 運べないため rule-instant をスキップし、必ず LLM ターン経由で surface する。
            reactive_response = (
                None if conflict_notice
                else _try_reactive_layer(
                    req, state, session_id, instance_name, context_size,
                )
            )
            if reactive_response is not None:
                # reactive 即応答 (挨拶/日時/キャッシュ) は検索/tool 判定結果を
                # 使わない。投機タスクを破棄。
                _cancel_pending_task(judge_task)
                _cancel_pending_task(search_task)
                return reactive_response

            # ルールベース miss → 軽量パス gating。tool 判定 (judge) で tool 不要
            # なら base 1 ターンの軽量パス、tool 必要なら deliberative へエスカレート。
            decision, judge_task, gate_reason = await _gate_reactive_light(
                req, state, cfg, history, judge_task,
            )
            if decision == "light":
                # 軽量パスは検索を使わない。judge_task も tool 不要なので破棄。
                _cancel_pending_task(judge_task)
                _cancel_pending_task(search_task)
                _log_layer_escalation(
                    state, chosen="reactive_light",
                    reason="conflict_notice" if conflict_notice else gate_reason,
                )
                return await _dispatch_reactive_light(
                    req, client, state, gen_params, history,
                    session_id, instance_name, context_size, max_tokens, timer,
                    conflict_notice=conflict_notice,
                )
            # deliberative へエスカレート (judge_task は tool 実行用に流用される)。
            # 競合通知は conflict_ctx 経由で build_semmem_injection が surface する。
            agent_layer = "deliberative"
            escalated_from = "reactive"
            _log_layer_escalation(state, chosen="deliberative", reason=gate_reason)
            logger.info("Reactive escalated to deliberative (%s)", gate_reason)

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

        messages, search_error_wrapper, semmem_block, scored_chunks = (
            await _build_messages_with_search(
                req, state, cfg, system_prompt, history, file_contexts,
                context_size, max_tokens, timer,
                editor_route=editor_route,
                conflict_ctx=conflict_ctx,
                search_task=search_task,
                fewshot_block=fewshot_block,
            )
        )

        # 添付ファイルは deliberative 経路では messages 側で消費されるが、
        # meta_cognitive / long_form 経路は messages を LLM に渡さないため別途注入する。
        file_block = _format_file_block(file_contexts)

        # Level 0 経験記録用 RAG シグナル。long_form 経路は prefetched_rag から
        # 自前で導出するため、ここでは meta_cognitive / deliberative へ伝播する。
        rag_used, rag_top1_score = rag_signals_from_chunks(scored_chunks)

        match agent_layer:
            case "meta_cognitive" if classifier.is_long_form:
                # long_form は precomputed tool 判定を使わない (judge_task は通常 None)。
                _cancel_pending_task(judge_task)
                # coding mode + pipeline=staged + Pro + assist 健全 のときは
                # 仕様書→コード→テストの staged パイプライン (専用 LoopDriver) へ。
                # それ以外は従来 longform 経路 (無改変フォールバック)。
                if _staged_coding_enabled(req, cfg, state):
                    return await _dispatch_staged_coding(
                        req, client, state, cfg, gen_params, session_id,
                        instance_name, context_size, messages,
                        search_error_wrapper, timer,
                        output_target=output_target,
                        prefetched_rag=scored_chunks,
                        file_context_block=file_block,
                    )
                return await _dispatch_long_form(
                    req, client, state, cfg, gen_params, session_id,
                    instance_name, context_size, messages, search_error_wrapper, timer,
                    output_target=output_target,
                    prefetched_rag=scored_chunks,
                    file_context_block=file_block,
                )
            case "meta_cognitive":
                # meta は task 記述単位で judge するため query 単位の precomputed は
                # 流用不可 (judge_task は通常 None)。念のため破棄する。
                # meta 経路は固定の PLAN/EXECUTE/CONTENT scaffold を使うため、
                # few-shot は system へ結合せず instance block (fewshot_block) として
                # 渡し、ツールループ / コンテンツ生成 / fallback の system に
                # [参考例] として注入する (Level 1 進化を coding 生成へ反映)。
                _cancel_pending_task(judge_task)
                # coding mode + pipeline=staged + Pro + assist 健全 のときは、
                # is_long_form でない coding 要求 (「テトリスを作成して」級) も
                # staged パイプラインへ。is_long_form ブランチと同一ゲート。
                if _staged_coding_enabled(req, cfg, state):
                    return await _dispatch_staged_coding(
                        req, client, state, cfg, gen_params, session_id,
                        instance_name, context_size, messages,
                        search_error_wrapper, timer,
                        output_target=output_target,
                        prefetched_rag=scored_chunks,
                        file_context_block=file_block,
                    )
                return await _dispatch_meta_cognitive(
                    req, client, state, cfg, gen_params,
                    system_prompt, history,
                    session_id, instance_name, context_size,
                    messages, search_error_wrapper, timer,
                    semmem_block=semmem_block,
                    rag_block=_format_rag_block_for_meta(scored_chunks),
                    file_block=file_block,
                    fewshot_block=fewshot_block,
                    output_target=output_target,
                    rag_used=rag_used,
                    rag_top1_score=rag_top1_score,
                )
            case _:
                return await _dispatch_deliberative(
                    req, client, state, cfg, gen_params, history,
                    session_id, instance_name, context_size, max_tokens,
                    messages, search_error_wrapper, timer,
                    rag_used=rag_used,
                    rag_top1_score=rag_top1_score,
                    tool_judge_task=judge_task,
                    escalated_from=escalated_from,
                )
    except BaseException:
        # 例外が伝播する経路 (build/dispatch 等) で未消費の投機タスクが残らない
        # よう破棄する。reactive 即応答や streaming dispatch の正常 return では
        # 発火しない (return は except を通らない)。
        _cancel_pending_task(judge_task)
        _cancel_pending_task(search_task)
        raise


@router.post("/chat/cancel", response_model=CancelResponse)
async def cancel_chat(req: CancelRequest):
    """ストリーミング生成を中断"""
    logger.debug("POST /api/chat/cancel: session=%s", req.session_id)
    if req.session_id in _cancel_flags:
        _cancel_flags[req.session_id] = True
        logger.debug("Cancel flag set for session %s", req.session_id)
        return CancelResponse(cancelled=True)
    return CancelResponse(cancelled=False, tokens_generated=0)
