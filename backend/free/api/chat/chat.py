"""チャット API（SSE ストリーミング + 3層エージェントディスパッチ）"""

import asyncio
import re
from dataclasses import dataclass, field, replace
from collections.abc import AsyncIterator, Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.app_state import AppState, get_app_state
from backend.aux_telemetry import open_aux_failure_ledger
from backend.free.core.verifier_events import open_verifier_scope
from backend.config import (
    get_config,
    get_mode_generation_params,
    get_path_resolver,
    resolve_context_size_for_mode,
)
from backend.free.api.chat.chat_constants import (
    CONTEXT_GROUNDED_TEMPERATURE,
    DEFAULT_HISTORY_MIN_TOKENS,
    DEFAULT_KEEPALIVE_INTERVAL_SEC, DEFAULT_MAX_TOKENS,
    MAX_FILE_CONTEXT_TOTAL_CHARS, MAX_FILE_CONTEXT_TOTAL_CHUNKS,
    MAX_MESSAGE_LENGTH,
    REACTIVE_LIGHT_HISTORY_TURNS, REACTIVE_LIGHT_MAX_TOKENS,
    SESSION_ID_MAX_LENGTH, SESSION_ID_MIN_LENGTH,
    DEFAULT_WORKING_MAX_TOKENS,
)
from backend.free.api.schemas import (
    CancelRequest, CancelResponse, ChatRequest, ChatResponse, TokenInfo,
)
from backend.free.api.chat._editor_routing import detect_editor_route
from backend.free.agent.router import indicates_write_destination
from backend.free.agent.tool_call_judge import (
    _extract_file_path,
    _recent_dialogue_text,
)
from backend.free.api.chat.chat_recorder import (
    record_response,
    record_turn_sources,
    set_turn_evidence_ids,
    set_turn_rag_meta,
    set_turn_fewshot_ids,
)
from backend.free.api.chat.chat_types import ChatMessage
from backend.free.api.chat.chat_service import (
    ConflictTurnContext,
    SearchPipelineResult,
    build_chat_messages, build_semmem_injection, convert_file_contexts,
    ensure_base_model_health,
    collect_pending_conflicts, deliberative_post_append_reserve_tokens,
    notes_post_append_reserve_tokens,
    ensure_llm_client, prepare_memory_context,
    run_search_pipeline, session_evicted_turns, session_first_user_message,
)
from backend.free.api.chat.chat_streaming import (
    collect_chat_response,
    request_cancel,
    rag_signals_from_chunks,
    read_existing_for_append,
    stream_deliberative, stream_long_form, stream_meta_cognitive, stream_reactive,
    stream_reactive_light,
)
from backend.free.api.chat._artifact import (
    LastArtifact,
    artifact_reference_verdict,
    peek_artifact,
    render_artifact_block,
)
from backend.free.api.chat._continuation import (
    TruncatedResponse,
    build_continuation_query,
    disarm_continuation,
    resume_from_last_response,
    take_continuation,
)
from backend.edition import is_pro
from backend.free.core.inference import latest_turn_truncation
from backend.free.core.intent_vocab import is_today_scope_query, is_whole_session_scope_query
from backend.free.core.session_mode import (
    is_create_mode,
    canonicalize_session_mode,
    normalize_session_mode,
)
from backend.free.core.sse import SSEFrameBuilder
from backend.free.agent.deliberative import (
    DeliberativeAgent,
    query_short_circuits_tool_judge,
)
from backend.free.agent.meta_cognitive import MetaCognitiveAgent
from backend.free.agent.prompt_manager import ensure_static_directives
from backend.free.agent.prompt_utils import format_fewshot_section
from backend.free.agent.reactive import ReactiveAgent
from backend.free.agent.router import (
    ComplexityClassifier,
    needs_write_intent_hint,
)
from backend.free.agent.issue_ledger import issue_ledger_scope
from backend.free.agent.file_ledger import file_ledger_scope
from backend.free.agent.tool_ledger import set_ledger_target
from backend.free.core.stage_timer import StageTimer
from backend.free.generation.harness import LongFormHarness
from backend.free.generation.orchestrator import LongFormOrchestrator
from backend.free.llm.aux_client import AuxClient
from backend.free.generation.content_detector import detect_content_type
from backend.free.generation.direct_codegen import generate_single_file
from backend.free.generation.models import ContentType
from backend.free.loop.staged.harness import StagedCodeHarness
from backend.free.loop.staged.run_recorder import StagedRunRecorder
from backend.free.generation.production_brief import (
    BriefLimits,
    build_code_map_block,
    build_production_brief,
)
from backend.free.agent.create_target_gate import names_creation_target
from backend.utils import estimate_tokens
from backend.free.llm.generation_gate import begin_chat_turn, current_turn_lease
from backend.log_config import get_logger
from backend.trace_context import (
    generate_trace_id,
    set_private,
    set_private_text,
    set_trace_id,
)

logger = get_logger("api.chat")

router = APIRouter(prefix="/api", tags=["chat"])


# PEP 695 type alias: SSE フレームジェネレータをラップする中間関数
type StreamWrapper = Callable[[AsyncIterator[str]], AsyncIterator[str]]


async def _with_chat_in_flight(client, inner_gen, lease=None):
    """ストリーミングジェネレータを ``chat_in_flight()`` でラップする。

    LLMClient にユーザー応答進行中であることを通知し、バックグラウンド
    処理（Level 1 進化、sleep-time）が協調的に yield できるようにする
    （`is_user_active` の定義は f_02_memory_system.md §4.3）。
    ``lease`` はハンドラから引き継いだターンの在圏リースで、ストリームの
    終わり (切断を含む) で解放する (``generation_gate.ChatTurnLease``)。
    """
    if lease is not None:
        lease.stream_started()
    try:
        async with client.chat_in_flight():
            async for frame in inner_gen:
                yield frame
    finally:
        if lease is not None:
            lease.release()


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

    # 旧名 ("coding") は現行名へ正規化してから通す。``canonicalize_session_mode``
    # の docstring が「API 受信など入口で使う」と定めている互換で、
    # ``LEGACY_SESSION_MODES`` も「旧クライアントからの受信を読めるように
    # するための入口互換」と書いているのに、**どの入口でも呼ばれていなかった**
    # (2026-09-03 ライブ監査: mode="coding" が 10 ターンすべて
    # ``HTTP 400 Invalid mode: coding``)。req.mode を現行名へ書き換えるので
    # 下流の ``is_create_mode`` 等はそのまま効く。
    canonical_mode = canonicalize_session_mode(req.mode)
    if canonical_mode is None:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {req.mode}")
    req.mode = canonical_mode

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


def _llm_unavailable(req: ChatRequest) -> StreamingResponse:
    """llama-server 未接続を要求の形式に合わせて返す (stream は error フレーム、同期は 503)。"""
    if req.stream:
        return _llm_unavailable_response(req.stream)
    raise HTTPException(status_code=503, detail="llama-server not connected")


def _history_budget(cfg: dict) -> tuple[int, int]:
    """``build_chat_messages`` へ渡す履歴予算 ``(history_min_tokens, working_max_tokens)``。

    3 つの組み立て経路 (主経路 / 軽量パス / 継続生成) が同じ値を使う。
    軽量パスは「履歴を削らない」立て付けなので、履歴の床も主経路と同じ値にする
    (0 のままだと動的ブロック (記憶) が履歴を押し出しうる)。
    """
    mem = cfg.get("memory") or {}
    return (
        int(mem.get("history_min_tokens", DEFAULT_HISTORY_MIN_TOKENS)),
        int(mem.get("working_max_tokens", DEFAULT_WORKING_MAX_TOKENS)),
    )


async def _respond(
    req: ChatRequest,
    client,
    session_id: str,
    stream_factory: Callable[[], AsyncIterator[str]],
    wrapper: StreamWrapper | None = None,
    *,
    state: AppState | None = None,
) -> StreamingResponse | ChatResponse:
    """層のストリーム実装を要求の形式で返す共通の出口。

    実装はストリーム 1 本で、非ストリーミング要求はそのフレーム列を
    ``collect_chat_response`` で ``ChatResponse`` に畳む (層ごとの ``sync_*``
    実装は廃止)。どちらも ``chat_in_flight()`` の中で走らせる (LLMClient に
    ユーザー応答進行中であることを通知し、背景処理が協調的に yield できる
    ようにする)。``wrapper`` は検索エラー通知等をストリーム冒頭へ挿す中間関数。

    Trigger A (sleep-time Light、f_02 §4.3) もここで 1 回だけ撃つ。Light は
    LLM を呼ばず埋め込みサーバだけを使うので、どの層の生成とも並走できる。
    以前は deliberative / 軽量パスのストリームだけが呼び、meta_cognitive /
    long_form / staged の長い生成中は Light が走らなかった。
    """
    scheduler = getattr(state, "sleep_scheduler", None) if state is not None else None
    if scheduler is not None:
        scheduler.on_llm_start()
    gen = stream_factory()
    if wrapper is not None:
        gen = wrapper(gen)
    if req.stream:
        lease = current_turn_lease()
        if lease is not None:
            lease.hand_over()
        return StreamingResponse(
            _with_chat_in_flight(client, gen, lease), media_type="text/event-stream",
        )
    async with client.chat_in_flight():
        return await collect_chat_response(gen, session_id=session_id)


def _resolve_system_prompt(
    state: AppState, mode: str, instance_name: str,
) -> str:
    """静的システムプロンプトを取得（PromptManager 未設定時はフォールバック）。

    query 非依存 (few-shot を含まない) なので連続リクエスト間で安定し、
    llama-server の prefix KV キャッシュが効く (応答言語指示・参考枠の扱いも
    config 固定文字列のため安定)。few-shot は ``_resolve_fewshot_block`` で
    別途取得し最後の user メッセージへ前置する。

    末尾の固定指示 (応答言語 / 参考枠の扱い) は ``SystemPromptManager.
    get_prompt_static`` が付けて返す (system 文の出所は PromptManager に閉じる、
    docs/f_03 §12 #5)。フォールバック文と ``get_prompt_static`` を持たない代替
    実装に対してだけ ``ensure_static_directives`` が補う (冪等)。
    """
    prompt_mgr = state.prompt_manager
    if prompt_mgr:
        get_static = getattr(prompt_mgr, "get_prompt_static", None)
        if get_static is not None:
            return ensure_static_directives(get_static(mode))
        # 後方互換: get_prompt_static 未実装の Mock 等は query なし get_prompt へ縮退
        return ensure_static_directives(prompt_mgr.get_prompt(mode))
    return ensure_static_directives(
        f"You are {instance_name}, a helpful AI assistant.",
    )


def _fact_slate_text(state: AppState, session_id: str | None, budget: int) -> str:
    """押し出したターンの要点表 (f_02 §1.2) を生テキストで返す (無ければ "")。

    ``_append_fact_slate`` (静的 system 末尾への合成) と ProductionBrief
    (f_08 §2.2) の Facts 節が、同じスレートを別々の予算で読む共通材料。
    """
    get = getattr(state, "get_memory_system", None)
    if get is None:
        return ""
    try:
        mem_sys = get(session_id)
    except Exception:
        return ""
    slate = getattr(mem_sys[0], "fact_slate", None) if mem_sys else None
    if slate is None or not len(slate):
        return ""
    from backend.i18n_helper import prompt_locale

    return slate.render(budget, prompt_locale())


def _append_fact_slate(state: AppState, session_id: str | None, system_prompt: str) -> str:
    """押し出したターンの要点表 (f_02 §1.2) を静的 system の末尾に足す。

    スレートは押し出しのあったターンでしか変わらないので、system は同一
    セッション内で押し出しの間隔だけ安定し、接頭辞 KV は静的部分まで共通のまま。
    予算は ``prompt.fact_slate_max_tokens`` (0 で無効)。
    """
    cfg = getattr(state, "config", None) or {}
    budget = int((cfg.get("prompt") or {}).get("fact_slate_max_tokens", 200))
    text = _fact_slate_text(state, session_id, budget)
    return f"{system_prompt}\n\n{text}" if text else system_prompt


def _resolve_fewshot_block(
    state: AppState, mode: str, query: str | None, query_vec=None,
    session_id: str | None = None,
) -> str:
    """query 依存の few-shot ブロックを取得 ("" = 無し / PromptManager 未設定)。

    ``query_vec`` を渡すと手本の選択が密ベクトル (記憶検索と同じ尺度) になる。
    渡さない経路 (meta_cognitive の scaffold 用) は従来の文字 bi-gram のまま。
    ``session_id`` があれば選択した例の id を ``set_turn_fewshot_ids`` へ置き、
    経験の ``gen_config.fewshot_ids`` (f_04 §3.2.2) の源にする。
    """
    prompt_mgr = state.prompt_manager
    if prompt_mgr is None:
        return ""
    get_examples = getattr(prompt_mgr, "get_fewshot_examples", None)
    if get_examples is not None and session_id:
        try:
            examples = get_examples(mode, query, query_vec)
        except TypeError:
            examples = get_examples(mode, query)
        set_turn_fewshot_ids(session_id, [ex.id for ex in examples])
        return format_fewshot_section(examples) if examples else ""
    get_block = getattr(prompt_mgr, "get_fewshot_block", None)
    if get_block is None:
        return ""
    try:
        return get_block(mode, query, query_vec)
    except TypeError:
        # query_vec を受けない旧シグネチャ (Mock 等) への後方互換
        return get_block(mode, query)


def _try_reactive_layer(
    req: ChatRequest,
    state: AppState,
    session_id: str,
    instance_name: str,
    context_size: int,
) -> StreamingResponse | ChatResponse | None:
    """Reactive 層でのパターンマッチ即応答。マッチしなければ None。"""
    # 常駐インスタンス (未配線環境 = テスト等では新規生成にフォールバック)。
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


async def _url_recall_hit(req: ChatRequest, state: AppState) -> bool:
    """過去に fetch 済みの URL ファクトがクエリに意味的に当たるか (reactive 昇格用)。

    判定本体は ``ToolCallJudge.recall_url_judgement`` (閾値 / TTL / profile まで
    見る)。埋め込みの HTTP 往復を伴うので、``idx.url.*`` が索引に 0 件なら
    埋め込まずに False を返す。判定器が未配線 / 例外時も False (reactive のまま)。
    """
    judge = state.tool_call_judge
    if judge is None or state.tools_registry is None:
        return False
    try:
        if not judge.has_url_recall_candidates():
            return False
        judgement = await judge.recall_url_judgement(
            req.message, state.tools_registry, mode=req.mode,
        )
    except Exception as exc:
        logger.warning("URL recall pre-check failed (continuing as reactive): %s", exc)
        return False
    return judgement is not None and judgement.tool_needed


async def _gate_reactive_light(
    req: ChatRequest,
    state: AppState,
    cfg: dict,
    history: list,
    judge_task: "asyncio.Task | None",
    timer: "StageTimer | None" = None,
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
    # judge_task は chat() が分類直後に投機起動する (前処理は常に並列)。None は
    # 判定器が配線されていない構成だけ。
    if (
        state.tool_call_judge is None
        or state.tools_registry is None
        or judge_task is None
    ):
        return "deliberative", judge_task, "judge_unavailable"
    # 会話全体を見ないと答えられない質問は、STM/SemMem/RAG 注入なしの視界で
    # 答えさせない (実インシデント 2026-08-12 ライブ監査 ターン21:「ここまでの
    # 会話を 5 行以内で要約して。」が 21 文字で short_query → 軽量パスに落ち、
    # 直近 3 往復だけを要約した)。
    #
    # 軽量パスの履歴窓は build_chat_messages へ委ねたので、視界の差は履歴では
    # なく **記憶と検索の有無** になった。それでも上げる価値はある — 窓外へ
    # 押し出されたターンは STM/search_history からしか辿れない。閾値は
    # 「会話が単発でない」ことを見るだけの目安。
    # 「今日の会話」は現在セッションの長さと無関係に他セッションを要する
    # (2026-09-05 F-12)。閾値ゲートを通さない。
    if is_today_scope_query(req.message):
        return "deliberative", judge_task, "today_scope"
    if (
        len(history) > REACTIVE_LIGHT_HISTORY_TURNS
        and is_whole_session_scope_query(req.message)
    ):
        return "deliberative", judge_task, "whole_session_scope"

    # 軽量パスは「ツール判定の結論」を待たないと採否を決められない。判定の
    # 最終層 (層5.9) はベースモデルの文法制約分類で、**推論 1 往復ぶんの待ち**
    # になりうる。決定論プリゲートが大半を落とすとはいえ、落ちなかったターンは
    # 「軽量」と呼びながら推論を 1 回払っている。実測が無いと調整もできないので
    # 待ち時間を計測して requests JSONL に載せる。
    if timer is not None:
        timer.start("light_gate_judge_ms")
    try:
        judgement = await judge_task  # 投機起動済み、残り時間のみ待つ
    except Exception as exc:
        logger.warning("reactive-light judge failed, escalating: %s", exc)
        return "deliberative", judge_task, "judge_error"
    finally:
        if timer is not None:
            timer.stop("light_gate_judge_ms")

    if judgement is not None and judgement.tool_needed:
        # deliberative へ流用 (done 済みタスクをそのまま渡す)
        return "deliberative", judge_task, "tool_needed"

    # 「撃てなかった」と分かっているターンは軽量パスに落とさない。
    #
    # ``action_blocked`` / ``measurement_blocked`` は「状態を変える / 実測する
    # 依頼なのに、それを実行できるツールが無い」という判定結果で、これが立った
    # ターンは deliberative が ``_UNPERFORMED_ACTION_GUIDANCE`` /
    # ``_UNMEASURED_FACT_GUIDANCE`` を最後の user メッセージへ足して完了・断定の
    # 捏造を止める。ところが軽量パスは few-shot/RAG/semmem/tool を全て外すので
    # 注記も付かず、判定だけログに出て **プロンプトには何も伝わらない**。
    #
    # 実インシデント 2026-08-22 ライブ監査 (修正の実機検証): 書込み直後の
    # 「そのファイルを削除してください。」(16 文字) が short_query で
    # reactive へ落ち、``Action blocked: file deletion requested but no tool can
    # delete`` がログに出ていながら回答は「削除しました。」。ファイルは残存。
    # 判定結果から読む。共有インスタンスの属性を後から読むと、チャットが
    # 2 本重なったときに他方の judge() がリセット済みでガードが消える
    # (ToolJudgement.action_blocked のコメント参照)。
    blocked = judgement is not None and (
        judgement.action_blocked or judgement.measurement_blocked
    )
    if blocked:
        return "deliberative", judge_task, "blocked_action"
    return "light", judge_task, "judge_no_tool"


async def _light_semmem_block(
    req: ChatRequest, state: AppState, cfg: dict, timer: StageTimer,
    session_id: str | None = None,
) -> str | None:
    """軽量パス用の ``[関連する記憶]`` ブロック (検索は走らせない)。

    **層の切り替えを「崖」にしない**ための経路。軽量パスは長らく RAG・SemMem・
    few-shot・ツール判定を **同時に全部** 落としていたため、``short_query`` で
    ここへ落ちた瞬間に記憶へ一度も到達しなくなり、「覚えているのに『情報が
    ありません』と答える」事故が繰り返し起きた。救済のたびに ``short_query``
    の手前へルールを積む対処を重ねてきたが、語彙の列挙は必ず漏れる。

    重いのは **検索パイプライン** (STM/LTM/カートリッジ + 各ゲート) であって、
    記憶の注入そのものではない。注入に必要なのはクエリ埋め込み 1 回だけなので、
    検索は落としたまま注入だけ残す — これで軽量パスは「速いが記憶が無い」から
    「速いが検索をしない」に変わる。

    埋め込みが取れなければ ``None`` を返す (``build_semmem_injection`` は
    ``query_vec=None`` を全店注入に読み替えるため、ここで止めるのが安全側)。
    """
    if not cfg.get("agent", {}).get("reactive_light_memory_enabled", True):
        return None
    if state.embedder is None:
        return None
    timer.start("light_embedding_ms")
    try:
        query_vec = await state.embedder.embed_query(req.message, mode=req.mode)
    except Exception as exc:
        logger.warning("reactive-light embedding failed (no memory): %s", exc)
        return None
    finally:
        timer.stop("light_embedding_ms")
    timer.start("semmem_ms")
    injected_evidence_ids: list[str] = []
    try:
        return build_semmem_injection(
            state, cfg, mode=req.mode, query_vec=query_vec,
            query_text=req.message, session_id=session_id,
            evidence_ids=injected_evidence_ids,
        )
    except Exception as exc:
        logger.warning("reactive-light semmem injection failed: %s", exc)
        return None
    finally:
        timer.stop("semmem_ms")
        # 軽量パスは検索を落とすので corpus 由来は出ない (c_16 §5.5)。
        if session_id:
            set_turn_evidence_ids(session_id, injected_evidence_ids)


async def _dispatch_continuation(
    req: ChatRequest,
    client,
    state: AppState,
    cfg: dict,
    gen_params: dict,
    system_prompt: str,
    history: list,
    pending: TruncatedResponse,
    session_id: str,
    instance_name: str,
    context_size: int,
    max_tokens: int | None,
    timer: StageTimer,
) -> "StreamingResponse | ChatResponse":
    """継続生成 dispatch: 直前の切断応答の **続きだけ** を書かせる。

    分類器を通さない。「続けて」は 3 文字なので必ず ``short_query`` →
    ``reactive_light`` へ落ち、切れた履歴を見たモデルが最善の推測として
    直前ブロックを再掲する (2026-08-25 実測: 2 回試して 2 回とも同一の
    末尾ブロックが返り、履歴に残っていた切断注記まで逐語コピーされた)。
    切断ケースの発火条件は ``take_continuation`` が握る観測事実 (直前ターンの
    ``finish_reason="length"``)。切断していない応答への「続けて」は
    ``resume_from_last_response`` が同じ材料を組み、指示文だけが変わる。
    どちらも **直前に assistant の応答がある** ことが前提なので、会話の
    1 ターン目の「続けて」はこの経路に入らない。

    生成そのものは軽量パスを使う (RAG / SemMem / ツール判定は続きの執筆に
    寄与しない)。ただし:

    - ``max_tokens`` は軽量パスの上限 (512) ではなく通常のチャット既定を使う。
      512 で切ったのがそもそもの原因なので、続きまで同じ幅で切らない。
    """
    # 履歴の最後 (= 今回の「続けて」) を継続指示へ差し替える。WM 側は
    # ユーザーの実発話のまま残すので、記録と表示は「続けて」で一貫する。
    cont_history = list(history)
    instruction = build_continuation_query(pending)
    if cont_history and cont_history[-1].get("role") == "user":
        cont_history[-1] = {**cont_history[-1], "content": instruction}
    else:
        cont_history.append({"role": "user", "content": instruction})
    history_min_tokens, working_max_tokens = _history_budget(cfg)
    cont_messages = build_chat_messages(
        system_prompt, cont_history,
        rag_chunks=None, file_contexts=None,
        semmem_block=None,
        context_size=context_size, max_tokens=max_tokens,
        # 履歴の床は主経路と同じ (続きを書くには直前の文脈が要る)。
        history_min_tokens=history_min_tokens,
        working_max_tokens=working_max_tokens,
        evicted_turns=session_evicted_turns(state, session_id),
        session_id=session_id,
        # ツール結果は積まれないが、接地注記は全経路で積まれる。
        post_append_reserve_tokens=notes_post_append_reserve_tokens(),
    )
    logger.info(
        "Continuation dispatch: resuming %s response (tail=%d chars)",
        "truncated" if pending.truncated else "completed",
        len(pending.tail),
    )
    # layer_escalation とは別の decision_point にする。既存キーの候補集合
    # (reactive_rule / reactive_light / deliberative) に無い chosen を混ぜると
    # policy_adjuster の (decision_point, chosen) 集計が汚れる。
    dl = getattr(state, "debug_logger", None)
    if dl is not None:
        dl.log_decision(
            decision_point="continuation_resume",
            chosen="continuation",
            candidates=["continuation", "normal"],
            reason=(
                "prev_turn_finish_reason_length"
                if pending.truncated
                else "prev_turn_completed_followup"
            ),
            scope="request",
        )
    args = (
        req.message, cont_messages, client, state, session_id,
        instance_name, context_size,
    )
    kwargs = dict(
        mode=req.mode, max_tokens=max_tokens,
        generation_params=gen_params, timer=timer, private=req.private,
        continuation_tail=pending.tail,
    )
    return await _respond(
        req, client, session_id,
        lambda: stream_reactive_light(*args, **kwargs),
        state=state,
    )


async def _dispatch_reactive_light(
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
    max_tokens: int | None,
    timer: StageTimer,
) -> "StreamingResponse | ChatResponse":
    """Reactive 軽量パス dispatch: 静的 system + 履歴で base 1 ターン (few-shot/RAG なし、SemMem は ``_light_semmem_block`` のみ)。

    ``system`` は静的なまま保つ。以前は記憶の競合セクションをここで連結して
    いたが、system の書き換えは接頭辞 KV キャッシュの境界そのものを動かす
    (競合の出現 / 解消 / 採番替えのたびに全損する)。競合の提示は関連度ゲートを
    掛けられる経路 (build_semmem_injection) だけが担う。

    履歴の切り出しは ``build_chat_messages`` (= ``_trim_history``) に委ねる。
    以前は ``history[-REACTIVE_LIGHT_HISTORY_TURNS:]`` の **末尾スライド窓**を
    自前で切っていたが、これは接頭辞キャッシュと最悪の相性で、会話が 1 ターン
    伸びるたびに ``system`` の直後が別物になり **窓の全体が再プリフィル**される。
    実測 (2026-08-19): 軽量パス 21 ターンの ``prompt_n`` 中央値 311 に対し、
    ユーザー本文は 6〜10 文字しかなかった。``_trim_history`` は
    ``_quantize_history_drop`` で先頭をブロック境界に止めるため、窓はブロックを
    跨ぐまで不変になる。未キャッシュのトークンはキャッシュ済みの 6〜12 倍高い
    ので、視界を広げてなお速くなる (2026-08-05 の捏造 2 件はどちらもこの経路の
    視界の狭さが原因でもあった)。

    軽量さは「RAG / SemMem / few-shot / ツール判定を通さない」ことと
    ``max_tokens`` の上限で担保しており、履歴を削ることではない。
    """
    semmem_block = await _light_semmem_block(req, state, cfg, timer, session_id)
    history_min_tokens, working_max_tokens = _history_budget(cfg)
    light_messages = build_chat_messages(
        system_prompt, history,
        rag_chunks=None, file_contexts=None,
        semmem_block=semmem_block,
        context_size=context_size, max_tokens=max_tokens,
        # 「履歴を削らない」が軽量パスの立て付けなので、履歴の床も主経路と同じ値。
        history_min_tokens=history_min_tokens,
        working_max_tokens=working_max_tokens,
        # 軽量パスも切り詰め注記 / 自己出力の計量を通す (build_chat_messages 内)。
        evicted_turns=session_evicted_turns(state, session_id),
        # 会話全体の計量 (「何ターン目?」) は session_id が無いと no-op になる。
        session_id=session_id,
        # ツール結果は積まれないが、接地注記は全経路で積まれる。
        post_append_reserve_tokens=notes_post_append_reserve_tokens(),
    )
    light_max = min(max_tokens or REACTIVE_LIGHT_MAX_TOKENS, REACTIVE_LIGHT_MAX_TOKENS)
    args = (
        req.message, light_messages, client, state, session_id,
        instance_name, context_size,
    )
    kwargs = dict(
        mode=req.mode, max_tokens=light_max,
        generation_params=gen_params, timer=timer, private=req.private,
    )
    return await _respond(
        req, client, session_id,
        lambda: stream_reactive_light(*args, **kwargs),
        state=state,
    )


async def _run_search_timed(
    req: ChatRequest, state: AppState, cfg: dict, timer: StageTimer,
    session_id: str | None = None,
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
            session_id=session_id,
            corpus_mode=getattr(req, "corpus_mode", "auto") or "auto",
        )
    finally:
        timer.stop("search_ms")


async def _collect_conflicts_timed(
    state: AppState, cfg: dict, mode: str, timer: StageTimer,
) -> ConflictTurnContext:
    """競合収集を ``conflict_ms`` 計測付きで実行する。

    ``collect_review_groups`` は各スコープの ``all_facts()`` 全ロードと属性
    類似度クラスタリングを **イベントループ上で同期に** 回す。ストアが育った
    ときに効いてくる場所なのに区間が無く、``search_ms`` / ``semmem_ms`` の
    どちらにも入っていなかった (``semmem_ms`` を足したのと同じ理由)。
    内部に await が無いので、ここの実測値はそのままイベントループの占有時間。
    """
    timer.start("conflict_ms")
    try:
        return await collect_pending_conflicts(state, cfg, mode=mode)
    finally:
        timer.stop("conflict_ms")


def _cancel_pending_task(task: "asyncio.Task | None") -> None:
    """投機タスクが未完了なら cancel する (reactive 早期 return / 経路不一致時)。"""
    if task is not None and not task.done():
        task.cancel()


def _answered_attributes(
    query: str, mode: str, covered: set[str],
) -> frozenset[str]:
    """クエリが尋ねている属性のうち、**今回注入済み** のものを返す。

    ``search_history`` の抑止条件。過去の監査で「答えは今の窓の中にある」を
    前提にしたスキップが、WorkingMemory が 1 件でも押し出した瞬間から永久に
    偽になった (2026-08-23)。ここは会話窓ではなく **このターンのプロンプトに
    実際に載ったファクト** を根拠にするので、その失敗にはならない —
    載っていなければ空集合になり、抑止は起きない。

    尋ねている属性が解決できないクエリ (自由な話題) は空集合。
    """
    if not covered:
        return frozenset()
    from backend.free.memory.pipeline.injector import MemoryInjector

    asked = MemoryInjector._asked_attributes(query, normalize_session_mode(mode))
    return frozenset(asked & covered)


#: 成果物ブロックへ割り当てる文字数の上限。動的ブロック全体の予算は
#: ``build_messages`` が決めるので、ここは「渡す前に常識的な大きさへ畳む」
#: ための上限にすぎない (入り切らなければ build_messages 側が更に切る)。
_ARTIFACT_BLOCK_MAX_CHARS = 6000


def _resolve_artifact_block(
    state: AppState, session_id: str, query: str,
) -> str | None:
    """この発話が直前の成果物を指していれば、その参照ブロックを返す。"""
    return _resolve_artifact_reference(state, session_id, query)[0]


def _resolve_artifact_reference(
    state: AppState, session_id: str, query: str,
) -> "tuple[str | None, LastArtifact | None]":
    """この発話が直前の成果物を指していれば ``(参照ブロック, 成果物)`` を返す。

    成果物そのものも返すのは、計量の注記 (「この案内文は何文字?」) が
    直前の返答ではなく成果物を測る材料にするため (2026-09-17 監査)。

    2 条件の AND:

    1. **観測事実** — 直前ターンで長文成果物を作った (レジストリに在る)
    2. この発話がそれを指している (:func:`references_artifact`)

    1 を先に置くのが要点。「その」「全体」のような語だけで判定すると、
    成果物が無いターンでも拾ってしまう。逆に成果物の **種類** を表す語
    (計画書 / レポート / 仕様書 …) を列挙する方式は取らない — 属性語の
    列挙は 2026-07 以降 4 回破れている。
    """
    if not session_id:
        return None, None
    artifact = peek_artifact(state, session_id)
    if artifact is None:
        return None, None
    verdict = artifact_reference_verdict(query)
    dl = getattr(state, "debug_logger", None)
    if dl is not None:
        dl.log_decision(
            decision_point="artifact_reference",
            chosen=str(verdict.value) if verdict.band == "fire" else "none",
            candidates=["section_ref", "demonstrative", "artifact_operation", "none"],
            reason=verdict.evidence,
            scope="request",
        )
    if verdict.band != "fire":
        return None, None
    block = render_artifact_block(
        artifact, budget_chars=_ARTIFACT_BLOCK_MAX_CHARS, query=query,
    )
    logger.info(
        "Artifact reference: injecting the previous long-form output "
        "(%d chars stored, %d chars injected)",
        len(artifact.text), len(block),
    )
    return block, artifact


@dataclass
class TurnContext:
    """1 ターンで組み立てた文脈。層はここから必要な形を引く。

    以前は ``_build_messages_with_search`` が 6 要素タプルを返し、層ごとに
    別名 (``prefetched_rag`` / ``rag_block`` / ``file_context_block`` …) で
    受け取っていた。同じ材料を層ごとに違う形で渡す構造が「軽量パスだけ記憶が
    無い」型の崖 (2026-08 に 3 回) を生んだ。

    **描画順の契約**: ``messages`` の中身 (静的 system → 履歴 → 最後の user に
    前置する動的ブロックの順) は ``build_chat_messages`` が決める。ここは順序を
    持たない — 順序を変えると接頭辞 KV キャッシュが 0 になる (2026-09-11 実測)。
    """

    #: deliberative が LLM に渡す messages 配列 (動的ブロック込み)。
    messages: list
    #: 検索エラー通知 / 出典 / 切り詰め通知をストリーム冒頭へ挿す中間関数。
    wrapper: StreamWrapper
    #: ``[関連する記憶]`` ブロック (meta が system へ再注入する)。
    semmem_block: str | None
    #: 検索で採用した ``(chunk_id, salience, content)`` (long_form / meta が再利用)。
    scored_chunks: list[tuple[str, float, str]] | None
    #: 採用チャンクの生スコア最大値 (cosine スケール、Level 0 の ``rag_top1_score``)。
    rag_top_raw: float | None
    #: query 依存の few-shot ブロック ("" = 無し)。
    fewshot_block: str
    #: 添付ファイルのブロック (meta / long_form は messages を LLM に渡さないため別途注入)。
    file_block: str | None
    #: 実際に注入されたファクトの属性スロット (``search_history`` の抑止に使う)。
    covered_attributes: set[str] = field(default_factory=set)

    @property
    def rag_used(self) -> bool:
        return rag_signals_from_chunks(self.scored_chunks, self.rag_top_raw)[0]

    @property
    def rag_top1_score(self) -> float | None:
        return rag_signals_from_chunks(self.scored_chunks, self.rag_top_raw)[1]

    @property
    def rag_block_for_meta(self) -> str | None:
        return _format_rag_block_for_meta(self.scored_chunks)


@dataclass
class RoutePlan:
    """このターンをどの層で、何を投機し、どこへ出すか — 分岐の根拠を 1 箇所に持つ。

    ``chat()`` は以前、分類結果を見て「judge を投機するか」「検索を投機するか」
    「output_target をどう決めるか」を手書きで分岐していた。ここに畳むことで、
    投機とディスパッチが同じ根拠から出て、``layer_escalation`` の理由も
    1 系統になる。
    """

    layer: str
    reason: str
    is_long_form: bool = False
    escalated_from: str | None = None
    #: create: file / editor / chat、chat: file / chat。
    output_target: str = "chat"
    #: create モードで UI へ通知する出力先 (``editor_route`` フレーム)。
    editor_route: str | None = None

    @property
    def speculate_judge(self) -> bool:
        """query 単位のツール判定を先行起動するか (meta は task 単位で判定する)。"""
        return self.layer != "meta_cognitive"

    @property
    def speculate_search(self) -> bool:
        """検索パイプラインを先行起動するか (reactive 即応答は使わない)。"""
        return self.layer != "reactive"

    def escalate(self, state: AppState, layer: str, reason: str) -> None:
        """reactive から上位層へ上げる (decision.jsonl に理由を残す)。"""
        _log_layer_escalation(state, chosen=layer, reason=reason)
        self.escalated_from = self.layer if self.layer == "reactive" else self.escalated_from
        self.layer = layer
        self.reason = reason


def _plan_output_target(req: ChatRequest) -> tuple[str, str | None]:
    """``(output_target, editor_route)`` を発話とモードから決める。

    create モード:
    - 出力先パス明示 → "file" (write_file でディスクへ)
    - 否定指示 ("エディタに出さず…") → "chat" (チャット本文にコードブロック)
    - 既定 → "editor" (ディスク書込せず editor_code チャネルでエディタペインへ)

    chat モードは **書込み先が特定できるときだけ** file。従来は無条件に
    "file" だったため、パスを一切含まない依頼でも「書き込む」プランが組まれ、
    write_file を撃ちようがないまま failed になり成果物が捨てられていた
    (2026-09-06 監査 F-02)。
    """
    if is_create_mode(req.mode):
        if _extract_file_path(req.message):
            target = "file"
        elif detect_editor_route(req.message) == "chat":
            target = "chat"
        else:
            target = "editor"
        return target, ("editor" if target == "editor" else "chat")
    return ("file" if indicates_write_destination(req.message) else "chat"), None


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
    covered_attributes: set[str] | None = None,
    session_id: str = "",
) -> TurnContext:
    """統合検索を実行し、このターンの文脈 (:class:`TurnContext`) を組む。

    ``messages`` は deliberative が消費し、``scored_chunks`` / ``file_block`` /
    ``semmem_block`` / ``fewshot_block`` は meta / long_form が messages を LLM に
    渡さないため別形で消費する。``rag_top_raw`` は Level 0 の ``rag_top1_score``
    用で、``scored_chunks`` 側のスコアが順位式 (c_16 §7.2) 適用後の salience
    なのに対し、こちらは cosine スケールの生スコア (``SearchResult.top_raw_score``)。

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
        search_result = await _run_search_timed(
            req, state, cfg, timer, session_id,
        )
    rag_chunks = search_result.chunks
    search_error = search_result.error
    scored_chunks = search_result.scored_chunks
    rag_top_raw_score = search_result.rag_top_score

    # 手本の選択も密ベクトルへ揃える。検索で算出済みのクエリ埋め込みを再利用
    # するので追加の埋め込み呼出は無い。記憶検索と手本選択で別々の「関連性」を
    # 使っていると、言い換えただけで手本が外れる (文字 bi-gram の弱点)。
    if fewshot_block is None:
        fewshot_block = _resolve_fewshot_block(
            state, req.mode, req.message, search_result.query_vec,
            session_id=session_id,
        )

    salience_ranker = None
    if scored_chunks:
        from backend.free.core.salience_ranker import SalienceRanker
        salience_ranker = SalienceRanker(
            policy=state.policy_interpreter, mode=req.mode,
        )

    # SemMem facts + STM notes を MemoryInjector で tier 整形して注入
    # (RAG とは独立、読み取りのみ)。pending 競合セクションも併せて連結する。
    #
    # ``semmem_ms`` を計測するのは、この経路が全件ロード (``all_facts()``) +
    # ファクトごとの numpy 演算 + レンダリングを**イベントループ上で同期**に
    # 回すため。``search_ms`` は RAG しか覆っておらず、ここは実測の空白だった
    # (2026-08-18 の requests.jsonl に区間が無い)。ストアが育ったときに
    # 最初に効いてくる場所なので、先に見えるようにしておく。
    timer.start("semmem_ms")
    # このターンで実際に注入した Evidence の id (c_16 §5.5)。``[参考情報]`` 枠
    # (corpus / episodic) は検索側が、``[関連する記憶]`` 枠 (semantic /
    # episodic) は注入側が埋める。学習帰属 (``GenerationConfigRef``) の材料。
    injected_evidence_ids: list[str] = list(search_result.evidence_ids)
    try:
        semmem_block = build_semmem_injection(
            state, cfg, mode=req.mode, conflict_ctx=conflict_ctx,
            # 検索で算出済みのクエリ埋め込みを再利用し、無関係な記憶の注入を防ぐ
            query_vec=search_result.query_vec,
            query_text=req.message,
            covered_attributes=covered_attributes,
            session_id=session_id,
            evidence_ids=injected_evidence_ids,
        )
    finally:
        timer.stop("semmem_ms")
    if session_id:
        set_turn_evidence_ids(session_id, injected_evidence_ids)
        set_turn_rag_meta(
            session_id,
            corpus_gated=search_result.corpus_gated,
            pseudo_derived=search_result.pseudo_derived,
            lexical_candidate_ids=search_result.lexical_candidate_ids,
            corpus_starved=search_result.corpus_starved,
        )

    # 直前ターンで作った長文成果物が、この発話の対象になっているか。
    # 長文は履歴予算に入らず次ターンで消えるため、これが無いとモデルは
    # 「履歴に含まれていない」としか言えない (_artifact の説明を参照)。
    artifact_block, referenced_artifact = _resolve_artifact_reference(
        state, session_id, req.message,
    )

    history_min_tokens, working_max_tokens = _history_budget(cfg)
    messages = build_chat_messages(
        system_prompt, history, rag_chunks, file_contexts,
        context_size, max_tokens,
        artifact_block=artifact_block,
        referenced_artifact=referenced_artifact,
        rag_scored_chunks=scored_chunks,
        salience_ranker=salience_ranker,
        semmem_block=semmem_block,
        fewshot_block=fewshot_block,
        history_min_tokens=history_min_tokens,
        working_max_tokens=working_max_tokens,
        # 会話の前半が窓外へ落ちている状態を「全体を走査する質問」にだけ伝える。
        evicted_turns=session_evicted_turns(state, session_id),
        # 「何ターン目?」「「横浜」は何回?」は窓の中だけでは数えられない。
        # 蓄積バッファを引くために session_id を渡す。
        session_id=session_id,
        # この経路は deliberative へ流れ、ツール結果 (最大 TOOL_RESULT_MAX_CHARS)
        # と各種注記が組み立て後に最後の user へ積まれる。その分を先に予約する。
        post_append_reserve_tokens=deliberative_post_append_reserve_tokens(),
    )

    sse_notify = SSEFrameBuilder()
    rag_debug_frame = _build_rag_debug_frame(state, scored_chunks, sse_notify, timer)
    sources_frame = _build_sources_frame(
        state, scored_chunks, sse_notify, session_id=session_id,
    )
    # ユーザー発言が長さ制限で切られた場合は UI へも伝える。system 注記だけでは
    # ベースモデルが従わず全体を見た前提で断定する実測があるため、モデルの遵守に
    # 依存せずユーザー自身が気づけるようにする (2026-07-26)。
    truncation = latest_turn_truncation(history, messages)

    async def _wrapper(inner_gen: AsyncIterator[str]) -> AsyncIterator[str]:
        """エディタ振り分け / 検索エラー通知 + RAG デバッグ情報をストリームの冒頭に挿入"""
        if editor_route is not None:
            yield sse_notify.editor_route(editor_route)
        if truncation is not None:
            yield sse_notify.input_truncated(*truncation)
        if search_error:
            yield sse_notify.step({
                "type": "search_error",
                "detail": f"RAG search failed: {search_error}",
                "status": "failed",
            })
        if rag_debug_frame:
            yield rag_debug_frame
        if sources_frame:
            yield sources_frame
        async for frame in inner_gen:
            yield frame

    return TurnContext(
        messages=messages,
        wrapper=_wrapper,
        semmem_block=semmem_block,
        scored_chunks=scored_chunks,
        rag_top_raw=rag_top_raw_score,
        fewshot_block=fewshot_block,
        # 添付ファイルは deliberative 経路では messages 側で消費されるが、
        # meta_cognitive / long_form 経路は messages を LLM に渡さないため別途注入する。
        file_block=_format_file_block(file_contexts),
        covered_attributes=covered_attributes if covered_attributes is not None else set(),
    )


#: 出典フレームに載せる本文プレビューの長さ。
_SOURCE_PREVIEW_CHARS = 160


def _build_sources_frame(
    state: AppState,
    scored_chunks: list,
    sse_notify: SSEFrameBuilder,
    *,
    session_id: str | None = None,
) -> str | None:
    """出典フレーム (c_06 §2.1 ``sources``) を組む。注入が無ければ ``None``。

    本文には出典を書かせない (``InternalFrameMentionFilter``) ので、UI は
    このフレームで根拠を出す。corpus チャンクは所在 (パッケージ / 文書 /
    見出し) を ``CartridgeManager.describe_chunk`` で引き、会話ノートは
    ``episodic`` として本文プレビューだけを付ける。
    """
    if not scored_chunks:
        return None
    manager = getattr(state, "cartridge_manager", None)
    items: list[dict] = []
    for chunk_id, score, content in scored_chunks:
        is_corpus = ":" in chunk_id
        evidence_id = chunk_id.split(":", 1)[1] if is_corpus else chunk_id
        item: dict = {
            "id": f"{'corpus' if is_corpus else 'episodic'}:{evidence_id}",
            "store": "corpus" if is_corpus else "episodic",
            "package_id": "",
            "package_name": "",
            "doc_id": "",
            "heading": "",
            "score": round(float(score), 4),
            "preview": (content or "")[:_SOURCE_PREVIEW_CHARS],
        }
        if is_corpus and manager is not None and hasattr(manager, "describe_chunk"):
            try:
                described = manager.describe_chunk(chunk_id)
            except Exception:  # noqa: BLE001 — 出典の装飾で応答を止めない
                described = None
            if described:
                item.update({k: described.get(k, "") for k in (
                    "package_id", "package_name", "doc_id", "heading",
                )})
        items.append(item)
    if session_id:
        # 「この会話で参照した資料は」に答える台帳 (chat_service の会話計量)。
        record_turn_sources(session_id, items)
    return sse_notify.sources(items)


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
    session_id: str | None = None,
) -> LongFormOrchestrator:
    """LongFormOrchestrator を構築する (long_form ディスパッチ / コード委譲で共用)。

    プラン生成 / レビュー / 設計仕様合成は ``AuxClient`` 越しに
    **ベースモデルの専有スロット** で実行する。長文生成はユーザーが明示的に
    起動する一次生成タスクで、本文生成そのものも同じベースモデルが担うため、
    補助段だけを別モデルへ逃がす理由がない (docs/c_14 §1.1 の例外)。
    """
    mem_sys = state.get_memory_system(session_id)
    local = state.local_client
    planner = (
        AuxClient(local, config=cfg, debug_logger=state.debug_logger)
        if local is not None and hasattr(local, "generate_constrained")
        else None
    )
    return LongFormOrchestrator(
        main_client=client,
        aux_client=planner,
        memory_wm=mem_sys[0] if mem_sys else None,
        config=cfg,
        debug_logger=state.debug_logger,
        generation_params=gen_params,
        policy=state.policy_interpreter,
    )


def _clamp_long_form_timeout(cfg: dict, mode: str) -> dict:
    """long_form 委譲時、orchestrator の total_timeout_sec をターン予算未満にする。

    予算の 3 層 (f_10 §3): 内側 (orchestrator の呼出予算) は常に外側 (ターン予算)
    より小さくする。ターン予算は mode で分岐する — create は
    ``create.turn_timeout_sec`` (既定 3600s)、chat は ``agent.total_timeout``
    (既定 1800s、超過時に artifacts 破棄)。90s のマージンを引いた上限より短く
    打ち切り、orchestrator 側でユニット境界の部分結果 + repair を確定させる
    (artifacts 喪失回避)。
    """
    if is_create_mode(mode):
        turn_budget = float((cfg.get("create") or {}).get("turn_timeout_sec", 3600.0) or 3600.0)
    else:
        turn_budget = float((cfg.get("agent") or {}).get("total_timeout", 1800) or 1800)
    clamped = max(300.0, turn_budget - 90.0)
    lf_cfg = dict(cfg.get("long_form") or {})
    existing = float(lf_cfg.get("total_timeout_sec", 1800.0) or 0.0)
    lf_cfg["total_timeout_sec"] = clamped if existing <= 0 else min(existing, clamped)
    return {**cfg, "long_form": lf_cfg}


def _brief_limits_from_cfg(cfg: dict) -> BriefLimits:
    """``create.brief`` (config.yaml) から :class:`BriefLimits` を組む。

    未設定キーは :class:`BriefLimits` の既定値に倒す (config.yaml.example の
    値と揃えてある)。
    """
    brief_cfg = ((cfg.get("create") or {}).get("brief") or {})
    return BriefLimits(**{
        field: brief_cfg[field] for field in (
            "max_tokens", "facts", "memory", "prior_work",
            "references", "attachments", "code_map",
        )
        if field in brief_cfg
    })


def _build_create_production_brief(
    state: AppState, cfg: dict, session_id: str, query: str, ctx: TurnContext,
) -> str:
    """create 経路の入口で 1 回だけ組む ProductionBrief (f_08 §2.2)。

    材料はこのターンで既に計算済みのもの (fact slate / SemMem 注入 / 直前
    成果物 / RAG チャンク / 添付ブロック)。ProjectMap の neighborhood だけは
    ここで決定論的に引く (新しい検索・LLM 呼出はしない、ms 級)。
    """
    limits = _brief_limits_from_cfg(cfg)
    fact_slate = _fact_slate_text(state, session_id, limits.facts)
    prior_work = _resolve_artifact_block(state, session_id, query) or ""
    reader_getter = getattr(state, "project_map_reader_getter", None)
    reader = reader_getter() if reader_getter is not None else None
    code_map = build_code_map_block(query, reader)
    brief = build_production_brief(
        fact_slate=fact_slate,
        semmem_block=ctx.semmem_block or "",
        prior_work=prior_work,
        rag_chunks=ctx.scored_chunks or [],
        file_block=ctx.file_block or "",
        code_map=code_map,
        limits=limits,
    )
    dl = getattr(state, "debug_logger", None)
    if dl is not None:
        references_text = "\n\n".join(
            c for _cid, _score, c in (ctx.scored_chunks or [])
        )
        dl.log_long_form_event({
            "phase": "production_brief",
            "facts_tokens": estimate_tokens(fact_slate),
            "memory_tokens": estimate_tokens(ctx.semmem_block or ""),
            "prior_work_tokens": estimate_tokens(prior_work),
            "references_tokens": estimate_tokens(references_text),
            "attachments_tokens": estimate_tokens(ctx.file_block or ""),
            "code_map_tokens": estimate_tokens(code_map),
            "total_tokens": estimate_tokens(brief),
        })
    return brief


def _find_needs_input_run(session_id: str):
    """このセッションの直前 staged create ターンが問い返し (``needs_input``、
    Phase 3b) のまま終わっていれば、その run を返す (無ければ ``None``)。

    ``list_runs`` は ``started_at`` 降順なので、最初に見つかったものが最新。
    """
    from backend.free.loop.staged.run_record import list_runs

    try:
        create_dir = get_path_resolver().resolve_local("create_workspace_dir")
        runs = list_runs(create_dir, session_id=session_id)
    except Exception as exc:  # noqa: BLE001 - 台帳が読めなければ「保留 run は無い」に倒す
        logger.debug("needs_input lookup skipped (session=%s): %s", session_id, exc)
        return None
    for record, status in runs:
        if status == "needs_input":
            return record
    return None


def _resume_brief_prefix(question: str, answer: str) -> str:
    """blocked run 再開時に ProductionBrief 先頭へ足す前回の問い/回答 (f_10 §7)。"""
    return f"# Resume\n前回の問い: {question}\n回答: {answer}\n---\n"


async def _dispatch_long_form(
    req: ChatRequest,
    client,
    state: AppState,
    cfg: dict,
    gen_params: dict,
    session_id: str,
    instance_name: str,
    context_size: int,
    ctx: TurnContext,
    timer: StageTimer,
    output_target: str = "file",
    brief: str = "",
) -> StreamingResponse | ChatResponse:
    """Meta-Cognitive (long_form) 経路: 長文生成オーケストレータを起動する。

    ``output_target`` は create モード時の出力先 (``"file"`` / ``"editor"`` /
    ``"chat"``) を ``stream_long_form`` / ``sync_long_form`` に伝播する。
    既定 ``"file"`` (チャット応答パス互換)。

    create の制作 (staged/longform) は 3a-2 でディスパッチが meta 経路の 1 本に
    なったため (f_03 §4.4)、この関数は **chat モードの is_long_form** 専用
    (「企画書を書いて」等、create ではない通常の長文生成) — chat は base を
    動かさないため production_stage を経由しない。

    ``brief`` は ProductionBrief (f_08 §2.2)。create モード限定で呼出側
    (``_dispatch``) が組み、空文字なら create 以外 (通常の長文生成) を意味する。
    """
    base_ok, client = await ensure_base_model_health(client, state, cfg)
    if not base_ok:
        return _llm_unavailable(req)

    # 予算の 3 層 (f_10 §3): orchestrator の total_timeout_sec をターン予算
    # (create.turn_timeout_sec / agent.total_timeout) 未満にクランプする。
    cfg = _clamp_long_form_timeout(cfg, req.mode)
    orchestrator = _build_long_form_orchestrator(
        client, state, cfg, gen_params, session_id,
    )
    existing_content = await read_existing_for_append(req.message, state)
    args = (
        orchestrator, req.message, session_id,
        req.mode, state, instance_name, context_size,
        ctx.messages, existing_content,
    )
    kwargs = dict(
        timer=timer,
        private=req.private,
        output_target=output_target,
        prefetched_rag=ctx.scored_chunks,
        prefetched_rag_top_score=ctx.rag_top_raw,
        file_context_block=ctx.file_block,
        brief=brief,
    )
    return await _respond(
        req, client, session_id,
        lambda: stream_long_form(*args, **kwargs),
        ctx.wrapper, state=state,
    )


def make_staged_codegen_delegate(client, cfg: dict, *, max_tokens: int | None = None):
    """base クリエイトモデル経由の codegen 委譲を作る
    ((instruction, file_path) -> {path: code})。

    ``StagedCreateExecutor`` に注入する。以前は ``make_code_artifact_generator`` と
    同じ LongFormOrchestrator 経路 (plan/CodeSpec 再合成 + CodeUnit 細粒度分割生成)
    を経由していたが、これは instruction (spec.md 全文 + flowchart + 契約ブロック)
    の大半を補助タスクの再合成・トークン予算切り詰めで失い、生成コードが仕様と乖離
    する原因になっていた (副作用として ``detect_content_type`` の TEXT 誤判定対策
    も必要だった)。staged は ``synthesize_create_task_graph`` が既にプログラムを
    ファイル単位へ決定的に分解済みのため、1 code タスク = 1 ファイルの単発生成で
    足りる。``direct_codegen.generate_single_file`` で base モデルへの単発呼び出し
    のみに委譲し、instruction を無劣化のまま渡す (再計画・content_type 判定は
    どちらも不要になる)。

    ``max_tokens`` 指定時は config (``code_max_tokens``) より優先する
    (部分ごと生成向けの ``part_max_tokens`` 予算で別 delegate を作る用途)。

    返す delegate は ``request_timeout`` (kw-only、既定 None) を受け取る —
    呼出予算 (f_10 §3)。``StagedCreateExecutor`` が残りステージ予算を渡し、
    未指定 (``None``) なら ``generate_single_file`` 自身の既定
    (``sync_request_timeout``) に委ねる。
    """
    staged_cfg = (cfg.get("create", {}) or {}).get("staged", {}) or {}
    resolved_max_tokens = (
        int(max_tokens) if max_tokens is not None
        else int(staged_cfg.get("code_max_tokens", 4096))
    )

    async def _generate(
        instruction: str, file_path: str, *, request_timeout: float | None = None,
    ) -> dict[str, str]:
        return await generate_single_file(
            client, instruction, file_path, max_tokens=resolved_max_tokens,
            request_timeout=request_timeout,
        )

    return _generate


def _staged_stage_base_enabled(mode: str, cfg: dict, state: AppState) -> bool:
    """staged パイプラインを起動しうる、instruction 非依存の前提条件。

    ``make_production_stage`` (instruction の content_type 判定は ``run()`` 時に
    遅延するため事前条件だけをここで判定する) が使う。
    """
    if not is_create_mode(mode):
        return False
    create_cfg = cfg.get("create", {}) or {}
    if create_cfg.get("pipeline") != "staged":
        return False
    if not create_cfg.get("staged_enabled", True):
        return False
    # 補助クライアント未配線 (ベース llama-server 未接続) なら従来 longform へ倒す。
    if getattr(state, "aux_client", None) is None:
        return False
    return is_pro()


class _ProductionStageSelector:
    """``create.dispatch=meta`` の制作ステージ選択ハーネス (f_03 §4.4)。

    instruction の content_type は ``run()`` 呼出し時にしか分からないため、
    Staged/LongForm の選択自体を遅延する ``ProductionHarness``。staged が
    タスクグラフ合成空 (``exit_kind="error"`` + ``notes["fallback"] ==
    "empty_task_graph"``) を返したとき: 要求に作成対象 (パス/言語/成果物の
    種類) が無く、かつ再開ターンでもなければ問い返し (``blocked``、Phase 3b)
    へ倒す。対象がある / 再開ターンなら従来どおり longform へ委譲する
    (legacy の ``fallback_factory`` と同じ役目)。
    """

    name = "production"

    def __init__(
        self, *, staged_factory, longform_factory, staged_base_enabled: bool,
        resume_of: str | None = None,
    ) -> None:
        self._staged_factory = staged_factory
        self._longform_factory = longform_factory
        self._staged_base_enabled = staged_base_enabled
        self._resume_of = resume_of
        self._resume_finished = False

    async def precheck(self, req):
        """計画の前に決定論で問い返すか (f_03 §4.4、2026-09-19)。

        対象 (パス / 言語 / 成果物名詞、``names_creation_target``) が無く再開でも
        なければ、LLM を 1 回も呼ばずに ``blocked`` を返す。run は
        ``project_id="needs_input"`` の workspace を起こしてそこに置く (f_10 §7)。
        """
        from pathlib import Path
        from types import SimpleNamespace
        from uuid import uuid4

        from backend.free.loop.staged import WorkspaceManager

        if req.resume_of or self._resume_of or names_creation_target(req.instruction):
            return None
        run_id = uuid4().hex[:12]
        create_dir = Path(get_path_resolver().resolve_local("create_workspace_dir"))
        ws = WorkspaceManager.open_or_create(
            create_dir, workspace_id=run_id, session_id=req.session_id,
            project_id="needs_input", goal=req.instruction,
        )
        placeholder = SimpleNamespace(
            notes={"run_id": run_id, "workspace_root": str(ws.root)},
        )
        logger.info(
            "Production stage: request names no creation target; asking before plan "
            "(run=%s)", run_id,
        )
        return self._block_for_input(placeholder, req)

    async def run(self, req, *, on_event, is_cancelled):
        if self._resume_of and not req.resume_of:
            req = replace(req, resume_of=self._resume_of)
        if req.resume_of and not self._resume_finished:
            # 制作ステージが走り出したら旧 (blocked) run を終端する (1 回だけ、f_10 §7)。
            self._resume_finished = True
            self._finish_resumed_run(req.resume_of)
        use_staged = self._staged_base_enabled
        if use_staged:
            try:
                use_staged = (
                    detect_content_type(req.instruction, "create") == ContentType.CODE
                )
            except Exception:
                use_staged = False
        if use_staged:
            result = await self._staged_factory().run(
                req, on_event=on_event, is_cancelled=is_cancelled,
            )
            if not (
                result.exit_kind == "error"
                and result.notes.get("fallback") == "empty_task_graph"
            ):
                return result
            if req.resume_of is None and not names_creation_target(req.instruction):
                return self._block_for_input(result, req)
            logger.info(
                "Production stage: staged task graph empty; falling back to longform",
            )
        return await self._longform_factory().run(
            req, on_event=on_event, is_cancelled=is_cancelled,
        )

    @staticmethod
    def _block_for_input(result, req):
        """作成対象が無い空タスクグラフを問い返し (``blocked``) へ倒す (Phase 3b)。

        staged が既に起こしたワークスペース (``result.notes`` 経由) へ run.json
        を新規に立て、即座に ``blocked`` にする (このターンでは staged 実行に
        入らなかったため run.json はまだ無い、f_10 §7)。
        """
        from pathlib import Path

        from backend.free.harness.production import ProductionResult
        from backend.free.loop.staged.run_record import RunRecordStore
        from backend.i18n_helper import msg
        from backend.trace_context import get_trace_id

        question = msg("create.needs_input_question")
        run_id = str(result.notes.get("run_id") or "")
        workspace_root = str(result.notes.get("workspace_root") or "")
        if workspace_root:
            try:
                store = RunRecordStore(Path(workspace_root))
                store.start(
                    run_id=run_id, session_id=req.session_id,
                    request_id=get_trace_id() or "", mode="create",
                    query=req.instruction, output_target=req.output_target,
                )
                store.block(question)
            except Exception as exc:  # noqa: BLE001 - 永続化の失敗で問い返し自体は止めない
                logger.warning(
                    "Production stage: failed to persist blocked run: %s", exc,
                )
        return ProductionResult(
            exit_kind="blocked", question=question, artifacts=[], metrics={},
            notes={"run_id": run_id, "workspace_root": workspace_root},
        )

    @staticmethod
    def _finish_resumed_run(old_run_id: str) -> None:
        from pathlib import Path

        from backend.free.loop.staged.run_record import RunRecordStore

        create_dir = Path(get_path_resolver().resolve_local("create_workspace_dir"))
        try:
            store = RunRecordStore(create_dir / old_run_id)
            if store.load() and store.record is not None:
                store.finish("resumed")
        except Exception as exc:  # noqa: BLE001 - 後始末の失敗で再開自体は止めない
            logger.warning(
                "Production stage: failed to finish resumed run %s: %s",
                old_run_id, exc,
            )


def make_production_stage(
    client, state: AppState, cfg: dict, gen_params: dict, session_id: str,
    *, brief: str, mode: str, output_target: str,  # noqa: ARG001 - 現状は ProductionRequest 側が持つため未使用 (署名は f_03 §4.4 と一致させる)
    resume_of: str | None = None,
) -> _ProductionStageSelector:
    """create.dispatch=meta 用の制作ステージを組み立てる (composition 層、f_03 §4.4)。

    Staged/LongForm の実選択は instruction の content_type に依存するため
    ``run()`` 時まで遅延する (:class:`_ProductionStageSelector`)。両ハーネスの
    構築 (codegen delegate / LongFormOrchestrator の DI) も呼出しの都度
    factory 経由で遅延させ、使わない側のコストを払わない。

    ``resume_of`` は問い返し (``needs_input``、Phase 3b) から再開する元
    run_id — 呼出側 (``_dispatch``) が直前ターンの ``blocked`` run を見つけた
    ときだけ渡す。
    """
    staged_base_enabled = _staged_stage_base_enabled(mode, cfg, state)

    def _staged_factory() -> StagedCodeHarness:
        codegen = make_staged_codegen_delegate(client, cfg)
        staged_cfg = (cfg.get("create", {}) or {}).get("staged", {}) or {}
        part_codegen = (
            make_staged_codegen_delegate(
                client, cfg, max_tokens=int(staged_cfg.get("part_max_tokens", 1536)),
            )
            if staged_cfg.get("part_generation_enabled", False) else None
        )
        return StagedCodeHarness(
            state=state, cfg=cfg, codegen=codegen, part_codegen=part_codegen,
        )

    def _longform_factory() -> LongFormHarness:
        gen_cfg = _clamp_long_form_timeout(cfg, mode)

        def _run_recorder_factory(run_id: str) -> StagedRunRecorder:
            # longform も staged と同じ run.json/events.jsonl/workspace を持つ
            # (Phase 4、f_08 §2.3)。create_workspace_dir は staged と共有 — GC
            # (sleep-time Step 5.89) / /api/create/runs* が run_id 単位で扱う。
            create_dir = get_path_resolver().resolve_local("create_workspace_dir")
            return StagedRunRecorder.open(
                create_dir, run_id, session_id=session_id,
                project_id="longform", debug_logger=state.debug_logger,
            )

        return LongFormHarness(
            lambda: _build_long_form_orchestrator(
                client, state, gen_cfg, gen_params, session_id,
            ),
            session_id=session_id,
            # 追記 / 参照依頼の既存ファイル内容 (chat の長文と同じ解決器)。
            existing_content_resolver=lambda q: read_existing_for_append(q, state),
            run_recorder_factory=_run_recorder_factory,
        )

    return _ProductionStageSelector(
        staged_factory=_staged_factory, longform_factory=_longform_factory,
        staged_base_enabled=staged_base_enabled, resume_of=resume_of,
    )


def _with_artifact_material(
    state: AppState, session_id: str, history: list,
) -> list:
    """直前の成果物を ``history`` の末尾へ合成 assistant 発話として足す。

    ``MetaCognitiveAgent`` は ``conversation`` を
    ``meta_cognitive_content._inject_recent_conversation`` へ渡し、書くべき
    本文の素材にする。仕組みは既にあるが、長文成果物は履歴予算 (実測 1612
    トークン) に入らず落ちているため **素材が空のまま生成が走る**。

    素材が無いと小型モデルは別物を作る。実インシデントは 2 件:

    - 2026-08-10: 「先ほどの JSON Schema の内容で上書きして」→ draft-07 の
      別スキーマを新規作成
    - 2026-08-27: 6696 文字の計画書を ``plan.md`` へ保存させたら、構成も
      文面も違う 3867 文字の文書が書かれた (再生成した事実は非開示)

    既に履歴へ載っている場合は足さない (同じ本文の二重掲載で素材予算を食う)。
    """
    artifact = peek_artifact(state, session_id)
    if artifact is None:
        return history
    body = artifact.text
    for msg in reversed(history[-6:]):
        if isinstance(msg, dict) and body[:200] in (msg.get("content") or ""):
            return history
    logger.info(
        "Artifact material: handing the previous long-form output to the "
        "content generator (%d chars)", len(body),
    )
    return [*history, {"role": "assistant", "content": body}]


def _make_write_impact_classifier(
    state: AppState,
) -> "Callable[[list[str]], list[dict]] | None":
    """meta の制作タスク file 出力向け write_impact 分類器を組む (f_10 §8.1)。

    composition 層 (ここ) が ``state.project_map_reader_getter`` を束ね、
    実体 (:func:`backend.free.loop.staged.write_impact.staged_write_impact_payload`)
    は EvorefLoop 側の純粋関数。agent (EvorefLoop) から api.chat (composition) は
    import できない (pillar 境界) ため、呼出可能オブジェクトとして注入する。
    reader 未配線 (ProjectMap 無効) なら ``None`` (呼ばれない)。
    """
    reader_getter = getattr(state, "project_map_reader_getter", None)
    if reader_getter is None:
        return None

    def _classify(written_paths: list[str]) -> list[dict]:
        from backend.free.loop.staged.write_impact import staged_write_impact_payload

        reader = reader_getter()
        return staged_write_impact_payload(reader, written_paths)

    return _classify


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
    ctx: TurnContext,
    timer: StageTimer,
    output_target: str = "file",
    brief: str = "",
    production_stage: "_ProductionStageSelector | None" = None,
) -> StreamingResponse | ChatResponse:
    """Meta-Cognitive (通常) 経路: 計画 + ツールループ。

    meta 経路は固定の PLAN/EXECUTE/CONTENT scaffold を使うため、few-shot は
    system へ結合せず instance block として渡し、ツールループ / コンテンツ生成 /
    fallback の system に [参考例] として注入する (Level 1 進化を create 生成へ反映)。

    ``brief`` は ProductionBrief (f_08 §2.2、create 限定)。``production_stage``
    (create 限定、f_03 §4.4) への委譲時に使う。``production_stage`` が None
    (chat モード) のときは、単発ショット生成 (``_execute_editor_task`` の
    フォールバック経路、3a-2 で legacy の code_generator 委譲を撤去) にそのまま落ちる。
    """
    # 「その計画書を保存して」型の依頼に、直前の成果物を **素材** として渡す。
    # ``_generate_content`` は既に「直近の会話」を素材にする仕組みを持つが、
    # 長文成果物は履歴予算に入らず落ちているので素材が空になり、**別物を
    # 新規生成して書き込む** (2026-08-27 ライブ監査 T10-8: 6696 文字の計画書を
    # 保存させたら 3867 文字の別文書が書かれ、再生成した事実は開示されなかった)。
    history = _with_artifact_material(state, session_id, history)
    # @self 仮想カートリッジ用 LoopFactView 配線
    loop_view = _resolve_loop_view_for_agent(state)
    meta_agent = MetaCognitiveAgent(
        config=cfg,
        tool_judge=state.tool_call_judge,
        policy=state.policy_interpreter,
        agent_tracer=state.agent_tracer,
        loop_view=loop_view,
        project_id=state.current_project_id,
        # 計画立案 (`_plan`) は CLAUDE.md §1 に従い補助タスク
        # モデルで実行する。``state.aux_client`` は health_check 失敗時
        # ``None`` (degraded mode) になるが、その場合 ``_plan`` は空リスト
        # を返し単一タスクへフォールバックする。
        aux_client=state.aux_client,
        # に記録 (decision_point=``meta_cognitive_llm_route``)
        debug_logger=state.debug_logger,
        # ツールループ全反復で SemMem メモリを維持 (初回ターンと同じ block)
        semmem_block=ctx.semmem_block,
        # search pipeline 取得済み RAG を維持 (long_form の prefetched_rag と同型)
        rag_block=ctx.rag_block_for_meta,
        # 添付ファイル内容を維持 (deliberative の messages 注入と等価)
        file_block=ctx.file_block,
        # Level 1 進化 few-shot を維持 (固定 scaffold の [参考例] に注入)
        fewshot_block=ctx.fewshot_block,
        # 内部 loop/token 予算を create_model の実窓に合わせる
        mode=req.mode,
        production_stage=production_stage,
        brief=brief,
        write_impact_classifier=_make_write_impact_classifier(state),
    )
    keepalive_sec = cfg.get("streaming", {}).get(
        "keepalive_interval_sec", DEFAULT_KEEPALIVE_INTERVAL_SEC,
    )
    args = (
        meta_agent, req.message, system_prompt, history,
        client, state, session_id, instance_name, context_size,
        ctx.messages, req.mode,
    )
    kwargs = dict(
        generation_params=gen_params,
        timer=timer,
        private=req.private,
        output_target=output_target,
        rag_used=ctx.rag_used,
        rag_top1_score=ctx.rag_top1_score,
    )
    return await _respond(
        req, client, session_id,
        lambda: stream_meta_cognitive(
            *args, keepalive_interval=keepalive_sec, **kwargs,
        ),
        ctx.wrapper, state=state,
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
    ctx: TurnContext,
    timer: StageTimer,
    tool_judge_task: "asyncio.Task | None" = None,
    escalated_from: str | None = None,
) -> StreamingResponse | ChatResponse:
    """Deliberative 経路: ツール判定 + LLM 推論。

    ``tool_judge_task`` が渡された場合は chat() が先行起動した tool 判定タスクを
    再利用する (process() 内で await)。None の場合は process() が判定を直列実行。
    ``escalated_from`` は reactive からエスカレートした場合の出自 (outcome 観測用)。"""
    # 参考情報が付いたターンは接地回答なので温度を下げる (ツール接地と同じ
    # 理屈、ただし記憶は実測値ではないので 0.2 まで下げない。
    # CONTEXT_GROUNDED_TEMPERATURE のコメント参照)。既に低ければ据え置く。
    rag_used, rag_top1_score = ctx.rag_used, ctx.rag_top1_score
    if rag_used:
        gen_params = {
            **gen_params,
            "temperature": min(
                gen_params.get("temperature", CONTEXT_GROUNDED_TEMPERATURE),
                CONTEXT_GROUNDED_TEMPERATURE,
            ),
        }
    delib_agent = DeliberativeAgent(
        config=cfg,
        tool_judge=state.tool_call_judge,
        tools_registry=state.tools_registry,
        agent_tracer=state.agent_tracer,
        # コンテンツ生成 max_tokens を create_model の実窓に合わせる
        mode=req.mode,
    )
    args = (
        delib_agent, req.message, ctx.messages, client, state,
        session_id, instance_name, context_size,
    )
    kwargs = dict(
        mode=req.mode, max_tokens=max_tokens,
        conversation=history,
        generation_params=gen_params,
        timer=timer,
        private=req.private,
        rag_used=rag_used,
        rag_top1_score=rag_top1_score,
        tool_judge_task=tool_judge_task,
        escalated_from=escalated_from,
        # 窓の先頭が会話の先頭かを deliberative 側で判定するために渡す
        # (``_append_session_position_fact`` 参照)。
        evicted_turns=session_evicted_turns(state, session_id),
        session_head=session_first_user_message(state, session_id),
        # 実際に注入されたファクトの属性スロット。「この属性の現在値はもう
        # プロンプトに載っている」を判定して search_history を抑止する。
        answered_attributes=_answered_attributes(
            req.message, req.mode, ctx.covered_attributes,
        ),
    )
    return await _respond(
        req, client, session_id,
        lambda: stream_deliberative(*args, **kwargs),
        ctx.wrapper, state=state,
    )


@router.post("/chat")
async def chat(req: ChatRequest, state: AppState = Depends(get_app_state)):
    """SSE ストリーミングチャット応答（3層エージェントディスパッチ）

    要求の到着から応答の終わりまでをターンの在圏リースで包む。背景 aux は
    この間 dispatch を待つ (前処理の窓も含む、``generation_gate.ChatTurnLease``)。
    ストリーミング応答では解放の責務を ``_respond`` がストリームへ移す。
    """
    lease = begin_chat_turn()
    try:
        return await _chat_turn(req, state)
    finally:
        if not lease.handed_over:
            lease.release()


async def _chat_turn(req: ChatRequest, state: AppState):
    """``chat`` の本体 (ターンリースの内側)。"""
    trace_id = generate_trace_id()
    set_trace_id(trace_id)
    # private セッションではユーザー発話をログへ書かない。書く地点は多数
    # あるので contextvar で伝播し、redaction processor 側で伏せる
    # (structlog_config._PRIVATE_CONTENT_KEYS_LOWER)。
    set_private(req.private)
    set_private_text(
        req.message if req.private else "", req.session_id or "",
    )

    logger.debug(
        "POST /api/chat: mode=%s, stream=%s, message_len=%d, session=%s, trace_id=%s",
        req.mode, req.stream, len(req.message), req.session_id, trace_id,
    )
    _validate_chat_request(req)

    cfg = get_config()
    client = await ensure_llm_client(state, cfg)
    if client is None:
        return _llm_unavailable(req)

    instance_name = cfg.get("instance", {}).get("name", "evoref")
    context_size = resolve_context_size_for_mode(cfg, req.mode)
    max_tokens = cfg.get("llama", {}).get("max_tokens", DEFAULT_MAX_TOKENS) or None
    gen_params = get_mode_generation_params(req.mode)

    if state.sleep_scheduler:
        state.sleep_scheduler.on_user_input()
    history, session_id = await prepare_memory_context(req, state)
    file_contexts = convert_file_contexts(req)
    # このリクエストのツール実行の記録先を確定する。記録自体は実行の合流点
    # (``ToolsRegistry.execute``) が行い、ここは宛先を渡すだけ
    # (tool_ledger._current_target のコメント参照)。
    set_ledger_target(session_id, req.message)
    # 不首尾の台帳も同じ宛先に向ける (tool_ledger と対)。自己申告の問いに
    # 「システムが観測した不首尾」を決定論的に渡すための材料。
    issue_ledger_scope(session_id, req.message)
    # 補助判定の縮退もこのターンに紐づけて数える。結末 (outcome) と自己申告の
    # 両方が「中身が縮退したターン」を見分けられるようにするための台帳。
    open_aux_failure_ledger()
    # 検証器の発火をこのターンに紐づける (規則台帳の計数、f_03 §3.5.1)。
    open_verifier_scope(req.mode)
    # ファイル台帳も同じ宛先へ (「保存したファイルを読んで」の解決材料)。
    file_ledger_scope(session_id)
    # system は静的 (query 非依存) に保ち KV キャッシュを効かせる。query 依存の
    # few-shot は動的ブロックとして最後の user メッセージへ前置する (build_messages)。
    system_prompt = _append_fact_slate(
        state, session_id, _resolve_system_prompt(state, req.mode, instance_name),
    )

    # 直前の応答が max_tokens で切れていて、今回の発話が「続けて」だけなら
    # 分類器を通さず継続生成へ。分類器を通すと必ず short_query →
    # reactive_light に落ち、切れた履歴からモデルが直前ブロックを再掲する
    # (_dispatch_continuation の docstring 参照)。
    #
    # 切断が観測されていない場合も、直前に assistant の応答があれば同じ経路へ
    # 流す (resume_from_last_response)。層分類を deliberative へ上げるだけでは
    # 足りず、検索意図を持たない 3 文字に SemMem ブロックが噛み合って
    # 「あなたについて、現在確認できる情報はありません。」を返した (2026-08-25)。
    pending_continuation = take_continuation(
        state, session_id, req.message, req.mode,
    ) or resume_from_last_response(history, req.message, req.mode)
    if pending_continuation is not None:
        return await _dispatch_continuation(
            req, client, state, cfg, gen_params, system_prompt, history,
            pending_continuation, session_id, instance_name, context_size,
            max_tokens, StageTimer(),
        )
    # 「続けて」は **直前のターン** の続き。継続でない通常ターンが始まった
    # 時点で古い切断待ちは意味を失うので、ここで解除する。deliberative 系の
    # ストリームは終端で解除 / 再武装するが、meta_cognitive / long_form /
    # 同期経路は解除しないため、切断 → 別経路のターン → 「続けて」で
    # 1 時間 (TTL) は古い切断末尾を継ぎ足していた。
    disarm_continuation(state, session_id)

    # few-shot は _build_messages_with_search が 1 度だけ選ぶ (検索のクエリ埋め込み
    # を再利用)。以前はここで bi-gram 版を先に選んでいたが、deliberative では
    # 密ベクトル版に置き換わって捨てられ、meta には古い方が渡り、軽量パスでは
    # 注入していない手本の id が経験に刻まれていた (2026-09-14)。
    fewshot_block: str | None = None

    # classify は conflict 結果に依存しない (req.message のみ) ため先に確定し、
    # 並列モードでの投機タスク (tool 判定 / 検索) 起動のゲートに使う。
    classifier = ComplexityClassifier(
        config=cfg,
        learned_patterns=getattr(state, "learned_patterns_store", None),
        policy=state.policy_interpreter,
    )
    # 直近会話を渡す。被演算子が前ターンにしか無い計算 (「その差を月あたりに
    # 直すと何分？」) は数値ゼロのクエリになり、context 無しでは short_query →
    # reactive に落ちてツール判定へ一度も到達しない (2026-08-10 ライブ監査)。
    # context は遅延評価 — 消費するのは numeric_question ルールだけで、
    # 大半のターンはそこへ到達する前に分類が確定する。
    # 書込み意図の事例ゲート (c_17 / write_intent_gate)。**宛先は既に立って
    # いるのに書込み動詞だけが無い** ターンだけ埋め込みを 1 回引く。
    # `needs_write_intent_hint` が偽のターン (ファイル名を含まない通常の会話)
    # では 1 度も呼ばれないので TTFT に載らない。棄権 / 未 warmup は None で、
    # 規則の判定がそのまま通る。
    write_intent_hint: bool | None = None
    write_intent_gate = getattr(state, "write_intent_gate", None)
    if write_intent_gate is not None and needs_write_intent_hint(
        req.message, req.mode,
    ):
        try:
            write_intent_hint = await write_intent_gate.decide(req.message)
        except Exception as e:  # pragma: no cover - 縮退で吸収する
            logger.warning("Write intent gate failed, falling back: %s", e)

    output_target, editor_route = _plan_output_target(req)
    layer = classifier.classify(
        req.message, mode=req.mode,
        context=lambda: _recent_dialogue_text(history),
        write_intent_hint=write_intent_hint,
    )
    reason = getattr(classifier, "_last_classify_reason", "default")
    if (
        is_create_mode(req.mode)
        and layer != "meta_cognitive"
        and _find_needs_input_run(session_id) is not None
    ):
        # 問い返し (needs_input) 中のセッションの次の発話は回答なので、router
        # の層に関わらず meta (再開) へ回す。短い「…に保存して」が deliberative
        # に振られて再開経路を通らず、旧 run が blocked のまま残った (2026-09-19)。
        layer, reason = "meta_cognitive", "needs_input_resume"
    plan = RoutePlan(
        layer=layer,
        reason=reason,
        is_long_form=bool(classifier.is_long_form),
        output_target=output_target,
        editor_route=editor_route,
    )
    logger.info(
        "Agent layer: %s (mode=%s) for query: %s",
        plan.layer, req.mode,
        "[PRIVATE]" if req.private else req.message[:80],
    )
    # primary routing を decision.jsonl に記録 (evolve 限定)。後続の reactive→
    # light/deliberative escalation (_log_layer_escalation) は別 decision_point。
    # context={"mode"} は policy_adjuster が mode 別 routing 学習に使う (load-bearing)。
    dl = getattr(state, "debug_logger", None)
    if dl is not None:
        dl.log_decision(
            decision_point="layer_classification",
            chosen=plan.layer,
            candidates=["reactive", "deliberative", "meta_cognitive"],
            reason=plan.reason,
            context={"mode": req.mode},
            scope="request",
        )

    # 事例ゲートとの shadow 比較。**挙動は変えない** — 不一致だけを
    # decision.jsonl に貯め、切り替えるかどうかは人が判断する
    # (router は EVOLVABLE_DOMAINS から意図的に凍結されている)。
    # 投げっぱなしにして TTFT を 1ms も増やさない。private ターンは渡さない。
    layer_shadow = getattr(state, "layer_shadow", None)
    if layer_shadow is not None and not req.private:
        try:
            _shadow_task = asyncio.create_task(
                layer_shadow.observe(req.message, plan.layer),
                name="layer_shadow",
            )
            # 参照を握らないと GC されうる。結果は見ないので握り潰す。
            _shadow_task.add_done_callback(lambda t: t.exception())
        except RuntimeError:  # イベントループ外 (同期テスト等)
            pass

    timer = StageTimer()
    # pending 競合のユーザー回答判定 + 即時反映 (不変則例外 (b)、解決結果は
    # 同ターンの semmem 注入へ反映)。private ターンは SemMem へ書かない契約のため
    # allow_write=False (注入のみ継続)。
    #
    # conflict 判定 / 検索パイプライン / tool 判定は同時起動して直列待ちを畳む
    # (3 つとも互いに独立。以前は補助タスクの realtime セマフォ数で直列構成へ
    # 切り替えていたが、チャット応答パスから補助タスク呼出が無くなり分岐自体が
    # 恒真になったので撤去した)。依存順は守る:
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
        conflict_task = asyncio.create_task(
            _collect_conflicts_timed(state, cfg, req.mode, timer),
        )
        # 発話だけで決定論の短絡 (自己構成 / ツール目録 / 台帳の問い等) に落ちる
        # と分かるターンは投機判定を起動しない — 結果は使われず cancel されるだけ
        # で、分類器往復 (10〜24 秒) を空撃ちしていた (2026-09-14)。
        if (
            plan.speculate_judge
            and state.tool_call_judge is not None
            and state.tools_registry is not None
            and not query_short_circuits_tool_judge(req.message)
        ):
            judge_task = asyncio.create_task(
                state.tool_call_judge.judge(
                    req.message, state.tools_registry, req.mode, history,
                    session_id=session_id,
                    window_complete=session_evicted_turns(state, session_id) == 0,
                )
            )
        if plan.speculate_search:
            search_task = asyncio.create_task(
                _run_search_timed(req, state, cfg, timer, session_id),
            )
        try:
            conflict_ctx = await conflict_task
        except Exception as exc:
            logger.warning("Conflict review task failed (degrading): %s", exc)
            conflict_ctx = ConflictTurnContext()

        # 競合セクションは reactive / reactive_light には載せない。この 2 経路は
        # 検索もクエリ埋め込みも走らせないため、注入本体と同じ関連度ゲートを
        # 掛けられず、無関係な矛盾をそのまま出すことになる。
        #
        # かつては「canned 応答では通知を運べない」を理由に rule-instant を
        # スキップし、通知を軽量パスの **system プロンプトへ連結** していた。
        # その前提だったチャット内解決 (ユーザーの回答を判定して
        # apply_resolution へ流す経路) は撤去済みで、ブロックは情報提示のみに
        # なっている。実測 (2026-08-19): reactive 21 ターン中 14 ターンで
        # system が書き換わり、挨拶や短文の即答 (ReactiveAgent.process) が
        # 丸ごと LLM ターンに化けていた。解決は sleep-time と TTL が担うので、
        # 即答経路を潰す理由は無い。
        #
        # 注: 「今は何時ですか？」「今日の日付を教えてください。」は
        # ``executable_query`` で deliberative に分類されるため、そもそも
        # rule-instant には到達しない (2026-08-21 に実機で確認)。影響を受ける
        # のは ``greeting`` / ``short_query`` で reactive に落ちたターン。

        if plan.layer == "reactive":
            reactive_response = _try_reactive_layer(
                req, state, session_id, instance_name, context_size,
            )
            if reactive_response is not None:
                # reactive 即応答 (挨拶/日時/キャッシュ) は検索/tool 判定結果を
                # 使わない。投機タスクを破棄。
                _cancel_pending_task(judge_task)
                _cancel_pending_task(search_task)
                return reactive_response

            # URL recall プリチェック: 過去会話で fetch 済みの URL fact が
            # クエリに意味的にヒットする場合、軽量パスで前知識のみ応答せず
            # deliberative にエスカレートして fetch_url を実行させる。
            # ``recall_url_judgement`` は閾値・TTL・profile match まで判定
            # 済みのため、ヒット時のみ judgement を返す。
            #
            # ルール即応 (挨拶 / キャッシュ命中) の **後** に置く。クエリ埋め込みの
            # HTTP 往復を伴うため、以前の位置 (即応の手前) では挨拶 1 つごとに
            # 埋め込みを払っていた。URL ファクトが索引に 1 件も無い環境では
            # 何を埋め込んでも当たらないので、件数で先に短絡する
            # (2026-09-02 監査 C1)。
            if await _url_recall_hit(req, state):
                plan.escalate(state, "deliberative", "url_recall_hit")
                logger.info(
                    "Reactive escalated to deliberative due to URL recall hit: %s",
                    req.message[:80],
                )

        if plan.layer == "reactive":
            # ルールベース miss → 軽量パス gating。tool 判定 (judge) で tool 不要
            # なら base 1 ターンの軽量パス、tool 必要なら deliberative へエスカレート。
            decision, judge_task, gate_reason = await _gate_reactive_light(
                req, state, cfg, history, judge_task, timer,
            )
            if decision == "light":
                # 軽量パスは検索を使わない。judge_task も tool 不要なので破棄。
                _cancel_pending_task(judge_task)
                _cancel_pending_task(search_task)
                _log_layer_escalation(
                    state, chosen="reactive_light", reason=gate_reason,
                )
                return await _dispatch_reactive_light(
                    req, client, state, cfg, gen_params, system_prompt, history,
                    session_id, instance_name, context_size, max_tokens, timer,
                )
            # deliberative へエスカレート (judge_task は tool 実行用に流用される)。
            # 競合は conflict_ctx 経由で build_semmem_injection が (関連度ゲート
            # を通ったときだけ) surface する。
            plan.escalate(state, "deliberative", gate_reason)
            logger.info("Reactive escalated to deliberative (%s)", gate_reason)

        ctx = await _build_messages_with_search(
            req, state, cfg, system_prompt, history, file_contexts,
            context_size, max_tokens, timer,
            editor_route=plan.editor_route,
            conflict_ctx=conflict_ctx,
            covered_attributes=set(),
            search_task=search_task,
            fewshot_block=fewshot_block,
            session_id=session_id,
        )

        match plan.layer:
            case "meta_cognitive":
                # meta / long_form は precomputed tool 判定を使わない (meta は
                # task 記述単位で judge するため query 単位の流用不可、judge_task
                # は通常 None)。念のため破棄する。
                _cancel_pending_task(judge_task)
                # ProductionBrief (f_08 §2.2): create モードのターン入口で
                # 1 回だけ決定論で組み、全 LLM 呼出のプロンプト先頭へ同じ bytes
                # を渡す (create 限定)。
                brief = (
                    _build_create_production_brief(
                        state, cfg, session_id, req.message, ctx,
                    )
                    if is_create_mode(req.mode) else ""
                )
                # ディスパッチは 1 本 (3a-2、f_03 §4.4): create は常に meta の
                # production_stage 経由 (staged/longform の選択は
                # _ProductionStageSelector.run() が instruction の content_type
                # から遅延判定する)。chat モードの is_long_form (「企画書を
                # 書いて」等) だけが _dispatch_long_form を使う (chat = base を
                # 動かさない)。
                production_stage = None
                if is_create_mode(req.mode):
                    # 問い返し (needs_input、Phase 3b) からの再開: 直前ターンが
                    # blocked のまま終わっていれば、その問いと今回の回答を
                    # brief 先頭へ足し、resume_of として新 run へ引き継ぐ
                    # (f_10 §7)。
                    resume_of: str | None = None
                    resume_run = _find_needs_input_run(session_id)
                    if resume_run is not None:
                        resume_of = resume_run.run_id
                        question = str(resume_run.to_dict().get("question", ""))
                        brief = _resume_brief_prefix(question, req.message) + brief
                    production_stage = make_production_stage(
                        client, state, cfg, gen_params, session_id,
                        brief=brief, mode=req.mode, output_target=plan.output_target,
                        resume_of=resume_of,
                    )
                elif plan.is_long_form:
                    return await _dispatch_long_form(
                        req, client, state, cfg, gen_params, session_id,
                        instance_name, context_size, ctx, timer,
                        output_target=plan.output_target,
                        brief=brief,
                    )
                return await _dispatch_meta_cognitive(
                    req, client, state, cfg, gen_params,
                    system_prompt, history,
                    session_id, instance_name, context_size, ctx, timer,
                    output_target=plan.output_target,
                    brief=brief,
                    production_stage=production_stage,
                )
            case _:
                return await _dispatch_deliberative(
                    req, client, state, cfg, gen_params, history,
                    session_id, instance_name, context_size, max_tokens,
                    ctx, timer,
                    tool_judge_task=judge_task,
                    escalated_from=plan.escalated_from,
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
    logger.debug(
        "POST /api/chat/cancel: session=%s request=%s",
        req.session_id, req.request_id,
    )
    if request_cancel(req.session_id, req.request_id):
        logger.debug("Cancel flag set for session %s", req.session_id)
        return CancelResponse(cancelled=True)
    return CancelResponse(cancelled=False, tokens_generated=0)
