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
from backend.free.agent.tool_call_judge import (
    _extract_file_path,
    _recent_dialogue_text,
)
from backend.free.api.chat.chat_recorder import record_response
from backend.free.api.chat.chat_types import ChatMessage
from backend.free.api.chat.chat_service import (
    ConflictTurnContext,
    SearchPipelineResult,
    build_chat_messages, build_semmem_injection, convert_file_contexts,
    ensure_base_model_health,
    collect_pending_conflicts, ensure_llm_client, prepare_memory_context,
    run_search_pipeline, session_evicted_turns, session_first_user_message,
)
from backend.free.api.chat.chat_streaming import (
    _cancel_flags,
    rag_signals_from_chunks,
    read_existing_for_append,
    stream_deliberative, stream_long_form, stream_meta_cognitive, stream_reactive,
    stream_reactive_light, stream_staged_create,
    sync_deliberative, sync_long_form, sync_meta_cognitive, sync_reactive_light,
)
from backend.free.api.chat._artifact import (
    peek_artifact,
    references_artifact,
    render_artifact_block,
)
from backend.free.api.chat._continuation import (
    TruncatedResponse,
    build_continuation_query,
    resume_from_last_response,
    take_continuation,
)
from backend.edition import is_pro
from backend.free.core.inference import latest_turn_truncation
from backend.free.core.intent_vocab import is_whole_session_scope_query
from backend.free.core.session_mode import (
    is_create_mode,
    is_valid_session_mode,
    normalize_session_mode,
)
from backend.free.core.sse import SSEFrameBuilder
from backend.free.agent.deliberative import DeliberativeAgent
from backend.free.agent.meta_cognitive import MetaCognitiveAgent
from backend.free.agent.reactive import ReactiveAgent
from backend.free.agent.router import ComplexityClassifier
from backend.free.agent.issue_ledger import issue_ledger_scope
from backend.free.agent.file_ledger import file_ledger_scope
from backend.free.agent.tool_ledger import set_ledger_target
from backend.free.core.stage_timer import StageTimer
from backend.free.generation.orchestrator import LongFormOrchestrator
from backend.free.llm.aux_client import AuxClient
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
    （`is_user_active` の定義は f_02_memory_system.md §4.3）。
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

    if not is_valid_session_mode(req.mode):
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


# 応答言語のランタイム指示 (i18n.prompt_locale 追従)。プロンプト本文 (.md) は
# 初回生成 / ロケール切替時にしか再導出されないため、既存インストールの本文が
# 古い locale のまま・create 本文に言語制約が無い場合でも設定に追従させる保険。
# PromptManager の外で付加することで、Level 1 進化が本文へ焼き込む汚染を防ぐ
# (名前プレフィックスの前例: prompt_manager._strip_name_prefix)。
_RESPONSE_LANGUAGE_DIRECTIVES: dict[str, str] = {
    "ja": "（ユーザーが使用言語を明示的に指定した場合を除き、応答は日本語で行うこと）",
    "en": "(Respond in English unless the user explicitly requests another language.)",
}


def _response_language_directive() -> str:
    """prompt_locale に応じた応答言語指示行を返す (未知 locale / 取得失敗は ja)。"""
    try:
        locale = get_config().get("i18n", {}).get("prompt_locale", "ja")
    except Exception:
        locale = "ja"
    return _RESPONSE_LANGUAGE_DIRECTIVES.get(
        locale, _RESPONSE_LANGUAGE_DIRECTIVES["ja"],
    )


# 参考枠 ([参考情報] / [関連する記憶] / [参考例] / [添付ファイル]) の扱いを述べる
# **静的**な指示。動的ブロック側のマーカーから本文を移してきたもの。
#
# なぜ system 側なのか: 動的ブロックは最後の user メッセージへ前置されるため
# 接頭辞 KV キャッシュの外にあり、内容が定数でも **毎ターン再プリフィル**される。
# 実測 (2026-08-18/19、chat 232 ターン): 区切り文 105 tok × 57% のターン /
# 記憶ラベル 105 tok × 40% / RAG ヘッダ 20 tok × 38% で、定数の指示文だけに
# 1 ターン平均 90 トークン前後を払っていた。未キャッシュ 1 トークンは 21〜37ms
# なので、これだけで数秒の TTFT になる。system へ移せば同じ文字が接頭辞
# キャッシュに乗り、2 ターン目以降の再プリフィルはゼロになる。
#
# PromptManager の外で付加する理由は ``_RESPONSE_LANGUAGE_DIRECTIVES`` と同じ:
# Level 1 進化がプロンプト本文へ焼き込んで劣化させるのを防ぐ。
#
# 内容は既存の指示の移設で、**新しい制約は足していない**:
#   - 「無関係なら言及せず自分の知識で答える」  ← 旧 _DYNAMIC_CONTEXT_DELIMITER
#     (実測 2026-07-25: PC が重い相談の最中に「ご提示いただいた参考情報には…
#      含まれていません」と述べ、空き 548GB あるのに空き容量不足の対処を回答)
#   - 「今回の質問に関係しなければ無視する / 無い予定・日付・数値を創作しない」
#     ← 旧 _SEMMEM_BLOCK_LABEL (実測 2026-07-27: 過去 note を根拠に、この会話に
#        存在しない歯科の予約と健康診断を捏造)
_REFERENCE_BLOCK_DIRECTIVES: dict[str, str] = {
    "ja": (
        "（[参考情報]・[関連する記憶]・[参考例]・[添付ファイル] は"
        "システムが用意した参考枠であり、ユーザーの発言ではない。"
        "今回の質問に関係しない場合は、そのことに言及せず、"
        "参考枠の話題に引きずられずに自分の知識で普通に答えること。"
        "参考枠に無い予定・日付・数値を創作しないこと。）"
    ),
    "en": (
        "([参考情報] / [関連する記憶] / [参考例] / [添付ファイル] blocks are "
        "reference material supplied by the system, not user input. "
        "If they are unrelated to the question, do not mention that fact and "
        "do not let them steer the topic - just answer from your own knowledge. "
        "Never invent schedules, dates, or numbers that are not in them.)"
    ),
}


def _reference_block_directive() -> str:
    """参考枠の扱いを述べる静的指示行を返す (未知 locale / 取得失敗は ja)。"""
    try:
        locale = get_config().get("i18n", {}).get("prompt_locale", "ja")
    except Exception:
        locale = "ja"
    return _REFERENCE_BLOCK_DIRECTIVES.get(
        locale, _REFERENCE_BLOCK_DIRECTIVES["ja"],
    )


def _resolve_system_prompt(
    state: AppState, mode: str, instance_name: str,
) -> str:
    """静的システムプロンプトを取得（PromptManager 未設定時はフォールバック）。

    query 非依存 (few-shot を含まない) なので連続リクエスト間で安定し、
    llama-server の prefix KV キャッシュが効く (応答言語指示・参考枠の扱いも
    config 固定文字列のため安定)。few-shot は ``_resolve_fewshot_block`` で
    別途取得し最後の user メッセージへ前置する。

    末尾に付ける 2 つの指示は **ここに置くこと自体が要点**。どちらも内容が
    ターンに依らない定数なので、動的ブロック側 (最後の user へ前置される =
    毎ターン再プリフィルされる) に置くと、同じ文字を毎回払うことになる。
    """
    directives = "\n\n".join(
        (_response_language_directive(), _reference_block_directive()),
    )
    prompt_mgr = state.prompt_manager
    if prompt_mgr:
        get_static = getattr(prompt_mgr, "get_prompt_static", None)
        if get_static is not None:
            return f"{get_static(mode)}\n\n{directives}"
        # 後方互換: get_prompt_static 未実装の Mock 等は query なし get_prompt へ縮退
        return f"{prompt_mgr.get_prompt(mode)}\n\n{directives}"
    return f"You are {instance_name}, a helpful AI assistant.\n\n{directives}"


def _resolve_fewshot_block(
    state: AppState, mode: str, query: str | None, query_vec=None,
) -> str:
    """query 依存の few-shot ブロックを取得 ("" = 無し / PromptManager 未設定)。

    ``query_vec`` を渡すと手本の選択が密ベクトル (記憶検索と同じ尺度) になる。
    渡さない経路 (meta_cognitive の scaffold 用) は従来の文字 bi-gram のまま。
    """
    prompt_mgr = state.prompt_manager
    if prompt_mgr is None:
        return ""
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
    見る)。埋め込みの HTTP 往復を伴うので、``mem.world.url.*`` が索引に 0 件なら
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
    try:
        return build_semmem_injection(
            state, cfg, mode=req.mode, query_vec=query_vec,
            query_text=req.message,
        )
    except Exception as exc:
        logger.warning("reactive-light semmem injection failed: %s", exc)
        return None
    finally:
        timer.stop("semmem_ms")


async def _dispatch_continuation(
    req: ChatRequest,
    client,
    state: AppState,
    cfg: dict,
    gen_params: dict,
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
    - 応答を ReactiveAgent キャッシュへ入れない (``cacheable=False``)。
      「続けて」をキーに入れると、後日の無関係な「続けて」へ 5 分以内に
      同じ続きが再生される。
    """
    system_prompt = _resolve_system_prompt(state, req.mode, instance_name)
    # 履歴の最後 (= 今回の「続けて」) を継続指示へ差し替える。WM 側は
    # ユーザーの実発話のまま残すので、記録と表示は「続けて」で一貫する。
    cont_history = list(history)
    instruction = build_continuation_query(pending)
    if cont_history and cont_history[-1].get("role") == "user":
        cont_history[-1] = {**cont_history[-1], "content": instruction}
    else:
        cont_history.append({"role": "user", "content": instruction})
    cont_messages = build_chat_messages(
        system_prompt, cont_history,
        rag_chunks=None, file_contexts=None,
        semmem_block=None,
        context_size=context_size, max_tokens=max_tokens,
        working_max_tokens=int(
            (cfg.get("memory") or {}).get(
                "working_max_tokens", DEFAULT_WORKING_MAX_TOKENS,
            ),
        ),
        evicted_turns=session_evicted_turns(state),
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
    if req.stream:
        return StreamingResponse(
            _with_chat_in_flight(client, stream_reactive_light(
                req.message, cont_messages, client, state, session_id,
                instance_name, context_size,
                mode=req.mode, max_tokens=max_tokens,
                generation_params=gen_params, timer=timer, private=req.private,
                cacheable=False,
                continuation_tail=pending.tail,
            )),
            media_type="text/event-stream",
        )
    async with client.chat_in_flight():
        return await sync_reactive_light(
            req.message, cont_messages, client, state, session_id,
            instance_name, context_size,
            mode=req.mode, max_tokens=max_tokens,
            generation_params=gen_params, timer=timer, private=req.private,
            cacheable=False,
            continuation_tail=pending.tail,
        )


async def _dispatch_reactive_light(
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
    system_prompt = _resolve_system_prompt(state, req.mode, instance_name)
    semmem_block = await _light_semmem_block(req, state, cfg, timer)
    light_messages = build_chat_messages(
        system_prompt, history,
        rag_chunks=None, file_contexts=None,
        semmem_block=semmem_block,
        context_size=context_size, max_tokens=max_tokens,
        working_max_tokens=int(
            (cfg.get("memory") or {}).get(
                "working_max_tokens", DEFAULT_WORKING_MAX_TOKENS,
            ),
        ),
        # 軽量パスも切り詰め注記 / 自己出力の計量を通す (build_chat_messages 内)。
        evicted_turns=session_evicted_turns(state),
    )
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
    """この発話が直前の成果物を指していれば、その参照ブロックを返す。

    2 条件の AND:

    1. **観測事実** — 直前ターンで長文成果物を作った (レジストリに在る)
    2. この発話がそれを指している (:func:`references_artifact`)

    1 を先に置くのが要点。「その」「全体」のような語だけで判定すると、
    成果物が無いターンでも拾ってしまう。逆に成果物の **種類** を表す語
    (計画書 / レポート / 仕様書 …) を列挙する方式は取らない — 属性語の
    列挙は 2026-07 以降 4 回破れている。
    """
    if not session_id:
        return None
    artifact = peek_artifact(state, session_id)
    if artifact is None or not references_artifact(query):
        return None
    block = render_artifact_block(
        artifact, budget_chars=_ARTIFACT_BLOCK_MAX_CHARS, query=query,
    )
    logger.info(
        "Artifact reference: injecting the previous long-form output "
        "(%d chars stored, %d chars injected)",
        len(artifact.text), len(block),
    )
    return block


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
) -> tuple[
    list, StreamWrapper, str | None,
    list[tuple[str, float, str]] | None, float | None,
]:
    """統合検索を実行し ``messages`` / SSE 通知ラッパ / semmem ブロック / 取得済み
    scored_chunks / 採用チャンクの生スコア最大値を構築する。``scored_chunks`` は
    long_form 経路が orchestrator に再利用注入するために返す (非 long_form 経路は
    ``messages`` 側で消費するため未使用)。最後の要素は Level 0 の
    ``rag_top1_score`` 用で、``scored_chunks`` 側のスコアが
    ``rag.score_normalization`` 適用後 (minmax なら先頭は定義上 1.0) なのに対し、
    こちらは cosine スケールの生スコア (``SearchResult.top_raw_score``)。

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
    rag_top_raw_score = search_result.rag_top_score

    # 手本の選択も密ベクトルへ揃える。検索で算出済みのクエリ埋め込みを再利用
    # するので追加の埋め込み呼出は無い。記憶検索と手本選択で別々の「関連性」を
    # 使っていると、言い換えただけで手本が外れる (文字 bi-gram の弱点)。
    if search_result.query_vec is not None:
        fewshot_block = _resolve_fewshot_block(
            state, req.mode, req.message, search_result.query_vec,
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
    try:
        semmem_block = build_semmem_injection(
            state, cfg, mode=req.mode, conflict_ctx=conflict_ctx,
            # 検索で算出済みのクエリ埋め込みを再利用し、無関係な記憶の注入を防ぐ
            query_vec=search_result.query_vec,
            query_text=req.message,
            covered_attributes=covered_attributes,
        )
    finally:
        timer.stop("semmem_ms")

    # 直前ターンで作った長文成果物が、この発話の対象になっているか。
    # 長文は履歴予算に入らず次ターンで消えるため、これが無いとモデルは
    # 「履歴に含まれていない」としか言えない (_artifact の説明を参照)。
    artifact_block = _resolve_artifact_block(state, session_id, req.message)

    messages = build_chat_messages(
        system_prompt, history, rag_chunks, file_contexts,
        context_size, max_tokens,
        artifact_block=artifact_block,
        rag_scored_chunks=scored_chunks,
        salience_ranker=salience_ranker,
        semmem_block=semmem_block,
        fewshot_block=fewshot_block,
        history_min_tokens=int(
            (cfg.get("memory") or {}).get(
                "history_min_tokens", DEFAULT_HISTORY_MIN_TOKENS,
            ),
        ),
        working_max_tokens=int(
            (cfg.get("memory") or {}).get(
                "working_max_tokens", DEFAULT_WORKING_MAX_TOKENS,
            ),
        ),
        # 会話の前半が窓外へ落ちている状態を「全体を走査する質問」にだけ伝える。
        evicted_turns=session_evicted_turns(state),
        # 「何ターン目?」「「横浜」は何回?」は窓の中だけでは数えられない。
        # 蓄積バッファを引くために session_id を渡す。
        session_id=session_id,
    )

    sse_notify = SSEFrameBuilder()
    rag_debug_frame = _build_rag_debug_frame(state, scored_chunks, sse_notify, timer)
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
        async for frame in inner_gen:
            yield frame

    return messages, _wrapper, semmem_block, scored_chunks, rag_top_raw_score


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
    """LongFormOrchestrator を構築する (long_form ディスパッチ / コード委譲で共用)。

    プラン生成 / レビュー / 設計仕様合成は ``AuxClient`` 越しに
    **ベースモデルの専有スロット** で実行する。長文生成はユーザーが明示的に
    起動する一次生成タスクで、本文生成そのものも同じベースモデルが担うため、
    補助段だけを別モデルへ逃がす理由がない (docs/c_14 §1.1 の例外)。
    """
    mem_sys = state.get_memory_system()
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
    """create 委譲時、orchestrator の total_timeout_sec を agent 上限未満にする。

    agent (`_total_timeout` 既定 900s、超過時に artifacts 破棄) より短く打ち切り、
    orchestrator 側でユニット境界の部分結果 + repair を確定させる (artifacts 喪失回避)。
    """
    agent_total = float((cfg.get("agent") or {}).get("total_timeout", 1800) or 1800)
    clamped = max(300.0, agent_total - 90.0)
    lf_cfg = dict(cfg.get("long_form") or {})
    existing = float(lf_cfg.get("total_timeout_sec", 1800.0) or 0.0)
    lf_cfg["total_timeout_sec"] = clamped if existing <= 0 else min(existing, clamped)
    return {**cfg, "long_form": lf_cfg}


def make_code_artifact_generator(
    client, state: AppState, cfg: dict, gen_params: dict, session_id: str,
):
    """create の editor/chat 出力向け code_generator を返す (MetaCognitiveAgent に注入)。

    指示文を LongFormOrchestrator の細粒度 CodeUnit 計画で生成し、ファイル別の
    検証・修正済みコードを EditorArtifact 群 (複数ファイル可) として返す。テキスト
    判定 / 生成失敗時は空リストを返し、agent の単一ショット生成にフォールバックさせる。
    """
    gen_cfg = _clamp_long_form_timeout(cfg)

    async def _generate(instruction: str, on_step=None) -> list[EditorArtifact]:  # noqa: ARG001
        if detect_content_type(instruction, "create") != ContentType.CODE:
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
                mode="create", on_step=None,
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
    prefetched_rag_top_score: float | None = None,
    file_context_block: str | None = None,
) -> StreamingResponse | ChatResponse:
    """Meta-Cognitive (long_form) 経路: 長文生成オーケストレータを起動する。

    ``output_target`` は create モード時の出力先 (``"file"`` / ``"editor"`` /
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
                prefetched_rag_top_score=prefetched_rag_top_score,
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
            prefetched_rag_top_score=prefetched_rag_top_score,
            file_context_block=file_context_block,
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
    """
    staged_cfg = (cfg.get("create", {}) or {}).get("staged", {}) or {}
    resolved_max_tokens = (
        int(max_tokens) if max_tokens is not None
        else int(staged_cfg.get("code_max_tokens", 4096))
    )

    async def _generate(instruction: str, file_path: str) -> dict[str, str]:
        return await generate_single_file(
            client, instruction, file_path, max_tokens=resolved_max_tokens,
        )

    return _generate


def _staged_create_enabled(req: ChatRequest, cfg: dict, state: AppState) -> bool:
    """staged クリエイトパイプラインを起動すべきか判定する。

    全条件を満たすときのみ True。いずれか欠ければ従来 longform 経路へ倒す。
    """
    if not is_create_mode(req.mode):
        return False
    create_cfg = cfg.get("create", {}) or {}
    if create_cfg.get("pipeline") != "staged":
        return False
    if not create_cfg.get("staged_enabled", True):
        return False
    # 補助クライアント未配線 (ベース llama-server 未接続) なら従来 longform へ倒す。
    if getattr(state, "aux_client", None) is None:
        return False
    if not is_pro():
        return False
    try:
        if detect_content_type(req.message, "create") != ContentType.CODE:
            return False
    except Exception:
        return False
    return True


async def _dispatch_staged_create(
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
    prefetched_rag_top_score: float | None = None,
    file_context_block: str | None = None,
) -> StreamingResponse | ChatResponse:
    """staged クリエイト: 専用 LoopDriver をインライン駆動し spec→code→test を実行。

    非ストリーミング要求 / base 不健全時は従来 longform 経路へフォールバックする。
    タスクグラフ合成が空 (aux degraded 等) のときも stream 内で longform へ委譲。
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
                prefetched_rag=prefetched_rag,
                prefetched_rag_top_score=prefetched_rag_top_score,
                file_context_block=file_context_block,
            )

    codegen = make_staged_codegen_delegate(client, cfg)
    staged_cfg = (cfg.get("create", {}) or {}).get("staged", {}) or {}
    part_codegen = (
        make_staged_codegen_delegate(
            client, cfg, max_tokens=int(staged_cfg.get("part_max_tokens", 1536)),
        )
        if staged_cfg.get("part_generation_enabled", False) else None
    )

    def _fallback_factory():
        # 合成失敗時のフォールバック (外側で search_error_wrapper 済みのため raw)
        return stream_long_form(
            orchestrator, req.message, session_id,
            req.mode, state, instance_name, context_size,
            messages, existing_content,
            timer=timer, private=req.private, output_target=output_target,
            prefetched_rag=prefetched_rag,
            prefetched_rag_top_score=prefetched_rag_top_score,
            file_context_block=file_context_block,
        )

    return StreamingResponse(
        _with_chat_in_flight(client, search_error_wrapper(stream_staged_create(
            query=req.message, session_id=session_id, state=state, cfg=cfg,
            instance_name=instance_name, context_size=context_size,
            messages=messages, output_target=output_target,
            codegen=codegen, part_codegen=part_codegen,
            fallback_factory=_fallback_factory,
            timer=timer, private=req.private,
            prefetched_rag=prefetched_rag,
            prefetched_rag_top_score=prefetched_rag_top_score,
            file_context_block=file_context_block,
        ))),
        media_type="text/event-stream",
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
    # 「その計画書を保存して」型の依頼に、直前の成果物を **素材** として渡す。
    # ``_generate_content`` は既に「直近の会話」を素材にする仕組みを持つが、
    # 長文成果物は履歴予算に入らず落ちているので素材が空になり、**別物を
    # 新規生成して書き込む** (2026-08-27 ライブ監査 T10-8: 6696 文字の計画書を
    # 保存させたら 3867 文字の別文書が書かれ、再生成した事実は開示されなかった)。
    history = _with_artifact_material(state, session_id, history)
    # @self 仮想カートリッジ用 LoopFactView 配線
    loop_view = _resolve_loop_view_for_agent(state)
    # create モード: editor/chat 出力のコード生成を LongForm 細粒度生成へ委譲する
    # generator を注入する (複数ファイル可)。非 create / 無効時は None。
    code_generator = None
    if (
        is_create_mode(req.mode)
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
        # 計画立案 (`_plan`) は CLAUDE.md §1 に従い補助タスク
        # モデルで実行する。``state.aux_client`` は health_check 失敗時
        # ``None`` (degraded mode) になるが、その場合 ``_plan`` は空リスト
        # を返し単一タスクへフォールバックする。
        aux_client=state.aux_client,
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
        # 内部 loop/token 予算を create_model の実窓に合わせる
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
    answered_attributes: frozenset[str] = frozenset(),
) -> StreamingResponse | ChatResponse:
    """Deliberative 経路: ツール判定 + LLM 推論。

    ``tool_judge_task`` が渡された場合は chat() が先行起動した tool 判定タスクを
    再利用する (process() 内で await)。None の場合は process() が判定を直列実行。
    ``escalated_from`` は reactive からエスカレートした場合の出自 (outcome 観測用)。"""
    # 参考情報が付いたターンは接地回答なので温度を下げる (ツール接地と同じ
    # 理屈、ただし記憶は実測値ではないので 0.2 まで下げない。
    # CONTEXT_GROUNDED_TEMPERATURE のコメント参照)。既に低ければ据え置く。
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
                # 窓の先頭が会話の先頭かを deliberative 側で判定するために渡す
                # (``_append_session_position_fact`` 参照)。
                evicted_turns=session_evicted_turns(state),
                session_head=session_first_user_message(state),
                answered_attributes=answered_attributes,
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
            evicted_turns=session_evicted_turns(state),
            session_head=session_first_user_message(state),
            answered_attributes=answered_attributes,
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
    # このリクエストのツール実行の記録先を確定する。記録自体は実行の合流点
    # (``ToolsRegistry.execute``) が行い、ここは宛先を渡すだけ
    # (tool_ledger._current_target のコメント参照)。
    set_ledger_target(session_id, req.message)
    # 不首尾の台帳も同じ宛先に向ける (tool_ledger と対)。自己申告の問いに
    # 「システムが観測した不首尾」を決定論的に渡すための材料。
    issue_ledger_scope(session_id, req.message)
    # ファイル台帳も同じ宛先へ (「保存したファイルを読んで」の解決材料)。
    file_ledger_scope(session_id)
    # system は静的 (query 非依存) に保ち KV キャッシュを効かせる。query 依存の
    # few-shot は動的ブロックとして最後の user メッセージへ前置する (build_messages)。
    system_prompt = _resolve_system_prompt(state, req.mode, instance_name)

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
            req, client, state, cfg, gen_params, history, pending_continuation,
            session_id, instance_name, context_size, max_tokens, StageTimer(),
        )

    fewshot_block = _resolve_fewshot_block(state, req.mode, req.message)

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
    agent_layer = classifier.classify(
        req.message, mode=req.mode,
        context=lambda: _recent_dialogue_text(history),
    )
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
        if (
            agent_layer != "meta_cognitive"
            and state.tool_call_judge is not None
            and state.tools_registry is not None
        ):
            judge_task = asyncio.create_task(
                state.tool_call_judge.judge(
                    req.message, state.tools_registry, req.mode, history,
                    session_id=session_id,
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

        # reactive→deliberative エスカレート時に Level 0 経験記録の出自を残す。
        escalated_from: str | None = None

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

        if agent_layer == "reactive":
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
                agent_layer = "deliberative"
                escalated_from = "reactive"
                _log_layer_escalation(state, chosen="deliberative", reason="url_recall_hit")
                logger.info(
                    "Reactive escalated to deliberative due to URL recall hit: %s",
                    req.message[:80],
                )

        if agent_layer == "reactive":
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
                    req, client, state, cfg, gen_params, history,
                    session_id, instance_name, context_size, max_tokens, timer,
                )
            # deliberative へエスカレート (judge_task は tool 実行用に流用される)。
            # 競合は conflict_ctx 経由で build_semmem_injection が (関連度ゲート
            # を通ったときだけ) surface する。
            agent_layer = "deliberative"
            escalated_from = "reactive"
            _log_layer_escalation(state, chosen="deliberative", reason=gate_reason)
            logger.info("Reactive escalated to deliberative (%s)", gate_reason)

        # create モードのみ生成コードの出力先を決定する。
        # - 出力先パス明示 → "file" (従来どおり write_file でディスクへ)
        # - 否定指示 ("エディタに出さず…") → "chat" (チャット本文にコードブロック)
        # - 既定 → "editor" (ディスク書込せず editor_code チャネルでエディタペインへ)
        # editor_route SSE フレームはフロント表示制御 (suppressCode) 用に併せて通知する。
        if is_create_mode(req.mode):
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

        # 実際に注入されたファクトの属性スロット。「この属性の現在値はもう
        # プロンプトに載っている」を判定して search_history を抑止する
        # (_dispatch_deliberative → process の answered_attributes)。
        covered_attributes: set[str] = set()
        messages, search_error_wrapper, semmem_block, scored_chunks, rag_top_raw = (
            await _build_messages_with_search(
                req, state, cfg, system_prompt, history, file_contexts,
                context_size, max_tokens, timer,
                editor_route=editor_route,
                conflict_ctx=conflict_ctx,
                covered_attributes=covered_attributes,
                search_task=search_task,
                fewshot_block=fewshot_block,
                session_id=session_id,
            )
        )

        # 添付ファイルは deliberative 経路では messages 側で消費されるが、
        # meta_cognitive / long_form 経路は messages を LLM に渡さないため別途注入する。
        file_block = _format_file_block(file_contexts)

        # Level 0 経験記録用 RAG シグナル。long_form 経路は prefetched_rag から
        # 自前で導出するため、ここでは meta_cognitive / deliberative へ伝播する。
        # スコアは正規化前の生スコア (rag_top_raw) を渡す — scored_chunks 側は
        # score_normalization 適用後で minmax なら先頭が定義上 1.0 になる。
        rag_used, rag_top1_score = rag_signals_from_chunks(scored_chunks, rag_top_raw)

        match agent_layer:
            case "meta_cognitive" if classifier.is_long_form:
                # long_form は precomputed tool 判定を使わない (judge_task は通常 None)。
                _cancel_pending_task(judge_task)
                # create mode + pipeline=staged + Pro + aux 健全 のときは
                # 仕様書→コード→テストの staged パイプライン (専用 LoopDriver) へ。
                # それ以外は従来 longform 経路 (無改変フォールバック)。
                if _staged_create_enabled(req, cfg, state):
                    return await _dispatch_staged_create(
                        req, client, state, cfg, gen_params, session_id,
                        instance_name, context_size, messages,
                        search_error_wrapper, timer,
                        output_target=output_target,
                        prefetched_rag=scored_chunks,
                        prefetched_rag_top_score=rag_top_raw,
                        file_context_block=file_block,
                    )
                return await _dispatch_long_form(
                    req, client, state, cfg, gen_params, session_id,
                    instance_name, context_size, messages, search_error_wrapper, timer,
                    output_target=output_target,
                    prefetched_rag=scored_chunks,
                    prefetched_rag_top_score=rag_top_raw,
                    file_context_block=file_block,
                )
            case "meta_cognitive":
                # meta は task 記述単位で judge するため query 単位の precomputed は
                # 流用不可 (judge_task は通常 None)。念のため破棄する。
                # meta 経路は固定の PLAN/EXECUTE/CONTENT scaffold を使うため、
                # few-shot は system へ結合せず instance block (fewshot_block) として
                # 渡し、ツールループ / コンテンツ生成 / fallback の system に
                # [参考例] として注入する (Level 1 進化を create 生成へ反映)。
                _cancel_pending_task(judge_task)
                # create mode + pipeline=staged + Pro + aux 健全 のときは、
                # is_long_form でない create 要求 (「テトリスを作成して」級) も
                # staged パイプラインへ。is_long_form ブランチと同一ゲート。
                if _staged_create_enabled(req, cfg, state):
                    return await _dispatch_staged_create(
                        req, client, state, cfg, gen_params, session_id,
                        instance_name, context_size, messages,
                        search_error_wrapper, timer,
                        output_target=output_target,
                        prefetched_rag=scored_chunks,
                        prefetched_rag_top_score=rag_top_raw,
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
                    answered_attributes=_answered_attributes(
                        req.message, req.mode, covered_attributes,
                    ),
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
