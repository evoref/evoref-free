"""セッション管理・ルーティング・メモリ統合ロジック"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from backend.app_state import AppState
from backend.free.api.chat.chat_recorder import clear_session_data, drain_evicted_to_stm
from backend.free.api.chat.chat_types import ChatMessage, FileContextDict
from backend.free.api.schemas import ChatRequest
from backend.free.core.inference import build_messages
from backend.free.llm.llm_client import LLMClient
from backend.free.memory.pipeline.search_pipeline import unified_search
from backend.utils import estimate_tokens as _estimate_tokens
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer

logger = get_logger("api.chat.service")

# セッション切替の排他制御ロック（並列リクエストでの WM 不整合を防止）
_session_switch_lock = asyncio.Lock()


def make_token_info(
    messages: list[ChatMessage], tokens_generated: int,
    context_size: int, instance_name: str,
) -> dict:
    """トークン使用量情報を構築"""
    used = sum(
        max(1, _estimate_tokens(m.get("content", "")))
        for m in messages
    ) + tokens_generated
    pct = int(used / context_size * 100) if context_size > 0 else 0
    return {
        "used": used,
        "limit": context_size,
        "pct": min(100, pct),
        "instance_name": instance_name,
    }


async def ensure_llm_client(state: AppState, cfg: dict) -> LLMClient | None:
    """LLM クライアントを取得する。未接続なら遅延接続を試行する。

    Returns:
        接続済みの LLMClient ラッパー。接続失敗時は None。
    """
    if state.local_client is not None:
        # 通常は set_local_client 経由で llm_client が同期生成されるが、
        # 直接 local_client を代入したケース（テスト等）に備えて遅延ラップする
        if state.llm_client is None:
            state.llm_client = LLMClient(local=state.local_client)
        return state.llm_client

    from backend.free.api.system.status import _try_lazy_connect
    llama_cfg = cfg.get("llama", {})
    llama_host = llama_cfg.get("host", "127.0.0.1")
    llama_port = llama_cfg.get("port", 8080)
    llama_url = f"http://{llama_host}:{llama_port}"
    connected = await _try_lazy_connect(state, llama_url, llama_cfg)
    if connected:
        logger.info("llama-server lazy-connected via chat endpoint")
        return state.llm_client
    return None


async def prepare_memory_context(
    req: ChatRequest, state: AppState,
) -> tuple[list[dict], str]:
    """メモリからコンテキストを取得し、セッション切替を処理する

    セッション切替ブロックは asyncio.Lock で排他制御し、
    並列リクエストでの WorkingMemory 不整合を防止する。

    Returns:
        (history, session_id) のタプル
    """
    mem_sys = state.get_memory_system()
    if not mem_sys:
        logger.debug("Memory not initialized, using single-turn context")
        session_id = req.session_id or "default"
        return [{"role": "user", "content": req.message}], session_id

    wm, stm, ltm = mem_sys

    # セッション切替検出: フロントエンドの session_id が変わったら WM をリセット
    # asyncio.Lock で排他制御し、並列リクエストでの競合を防止
    async with _session_switch_lock:
        if req.session_id and wm.session_id != req.session_id:
            old_session_id = wm.session_id
            logger.info(
                "Session switch detected: %s -> %s, clearing WorkingMemory",
                old_session_id, req.session_id,
            )
            drain_evicted_to_stm(wm, stm, old_session_id)
            wm.clear()
            wm.session_id = req.session_id
            # 旧セッションの蓄積データをクリーンアップ（メモリ解放）
            clear_session_data(old_session_id)
            # assist_judge のセッション単位カウンタもリセット
            # 旧セッションの残存カウントで新セッションが session_cap に
            # 張り付くのを防ぐ。
            if state.assist_judge_tracker is not None:
                state.assist_judge_tracker.reset_session(old_session_id)
            # 旧会話の経験に conversation_ended を反映 (Level 2 base=C positive 抽出用)。
            # disabled 時は FeedbackCollector 内ガードで no-op。
            if state.feedback_collector is not None:
                state.feedback_collector.mark_conversation_ended()

        wm.add_turn(
            "user",
            req.message,
            private=req.private,
            mode=req.mode,
            source="user",
        )
        history = wm.get_messages()

    logger.debug("Memory context: %d history turns from WorkingMemory", len(history))

    session_id = req.session_id or wm.session_id
    return history, session_id


def convert_file_contexts(req: ChatRequest) -> list[FileContextDict] | None:
    """file_contexts を辞書リストに変換"""
    if not req.file_contexts:
        return None
    file_contexts = [
        {"filename": fc.filename, "chunks": fc.chunks}
        for fc in req.file_contexts
    ]
    logger.debug("File contexts: %d files", len(file_contexts))
    return file_contexts


class SearchPipelineResult:
    """検索パイプラインの結果（BUG-9: 成功/失敗/スキップの区別を明確化）"""

    __slots__ = ("chunks", "scored_chunks", "error")

    def __init__(
        self,
        chunks: list[str] | None = None,
        scored_chunks: list[tuple[str, float, str]] | None = None,
        error: str | None = None,
    ):
        self.chunks = chunks
        self.scored_chunks = scored_chunks
        self.error = error

    @property
    def failed(self) -> bool:
        return self.error is not None


def _collect_semmem_stats(state: AppState) -> dict | None:
    """SemMem ストアから memory.jsonl 用の統計を収集する

    chat 応答パスでは SemMem は読み取りのみ (EvorefMem 設計原則 7) のため、
    軽量な type 別件数 + pinned 件数のみを返す。失敗時は None。
    """
    debug_logger = state.debug_logger
    if debug_logger is None or not getattr(debug_logger, "log_memory", False):
        return None
    try:
        global_store = state.get_semantic_store("global")
    except Exception:
        return None
    relevant_types = (
        "policy", "failure_pattern", "progress_marker", "task", "artifact",
    )
    stats: dict = {
        "global_total": len(global_store),
        "global_pinned": len(global_store.pinned_facts()),
    }
    by_type: dict[str, int] = {}
    for ft in relevant_types:
        try:
            by_type[ft] = global_store.count_by_type(ft)  # type: ignore[arg-type]
        except Exception:
            by_type[ft] = 0
    stats["global_by_type"] = by_type
    pid = state.current_project_id
    if pid:
        try:
            project_store = state.get_semantic_store(f"project:{pid}")
            stats["project_id"] = pid
            stats["project_total"] = len(project_store)
            stats["project_pinned"] = len(project_store.pinned_facts())
            proj_by_type: dict[str, int] = {}
            for ft in relevant_types:
                try:
                    proj_by_type[ft] = project_store.count_by_type(ft)  # type: ignore[arg-type]
                except Exception:
                    proj_by_type[ft] = 0
            stats["project_by_type"] = proj_by_type
        except Exception:
            pass
    return stats


async def run_search_pipeline(
    query: str, state: AppState, cfg: dict, mode: str = "chat",
    timer: StageTimer | None = None,
) -> SearchPipelineResult:
    """統合検索パイプライン: 3層メモリ + Self-RAG

    Returns:
        SearchPipelineResult — チャンクリスト + エラー情報。
        BUG-9 対策: 失敗時はエラー情報を保持し、呼び出し元で
        フロントエンドに通知可能にする。
    """
    mem_sys = state.get_memory_system()
    if not mem_sys or state.embedder is None:
        return SearchPipelineResult()

    try:
        if timer:
            timer.start("embedding_ms")
        # mode 別 instruction を埋め込みへ伝搬
        query_vec = await state.embedder.embed_query(query, mode=mode)
        if timer:
            timer.stop("embedding_ms")

        wm, stm, ltm = mem_sys
        # assist_judge のセッション上限判定に使う session_id は
        # WorkingMemory 側で常に同期されている (セッション切替時に
        # prepare_memory_context が wm.session_id を更新する)。
        session_id = getattr(wm, "session_id", None) or "default"
        # アシスト判定プロンプト (AssistPromptManager 由来) を plain str で解決し注入する。
        # search_pipeline (EvorefMem) / self_rag_judge (EvorefGen) から agent ピラーへ
        # 越境 import しないため、composition 層 (api/chat) で文字列に解決して渡す。
        # manager 未初期化や task 欠落時は None → judge 側の既定指示にフォールバック。
        mgr = state.assist_prompt_manager
        necessity_prompt = quality_prompt = None
        if mgr is not None:
            try:
                necessity_prompt = mgr.get_assist_prompt("rag_necessity")
                quality_prompt = mgr.get_assist_prompt("rag_quality")
            except (ValueError, KeyError):
                necessity_prompt = quality_prompt = None
        search_result = await unified_search(
            query=query,
            query_vec=query_vec,
            working_mem=wm,
            short_term=stm,
            long_term=ltm,
            cartridge_mgr=state.cartridge_manager,
            config=cfg,
            assist_client=state.assist_client,
            debug_logger=state.debug_logger,
            mode=mode,
            policy=state.policy_interpreter,
            reranker=state.reranker,
            timer=timer,
            semmem_stats=_collect_semmem_stats(state),
            lazy_contextual=state.lazy_contextual,
            session_id=session_id,
            assist_judge_tracker=state.assist_judge_tracker,
            necessity_prompt=necessity_prompt,
            quality_prompt=quality_prompt,
            assist_experience_recorder=state.assist_experience_recorder,
        )
        if not search_result.skipped and search_result.sources:
            rag_chunks = [content for _, _, content in search_result.sources]
            logger.info(
                "Search pipeline: %d chunks, quality=%s",
                len(rag_chunks), search_result.quality,
            )
            return SearchPipelineResult(
                chunks=rag_chunks,
                scored_chunks=search_result.sources,
            )
    except Exception as e:
        logger.warning("Search pipeline failed, continuing without RAG: %s", e)
        return SearchPipelineResult(error=str(e))

    return SearchPipelineResult()


def build_semmem_injection(
    state: AppState, cfg: dict, mode: str = "chat",
) -> str | None:
    """SemMem facts + STM notes を MemoryInjector で tier 整形し、
    プロンプト注入用テキストを返す。

    chat 応答パスでは SemMem は読み取りのみ (EvorefMem 設計原則 2/7)。
    RAG とは独立して呼び出し、検索ヒットの有無に関わらずメモリを注入する。
    失敗時は None を返してチャットを止めない。
    """
    mem_sys = state.get_memory_system()
    if not mem_sys:
        return None
    inj_mode = mode if mode in ("chat", "coding") else "chat"
    try:
        from backend.free.memory.pipeline.injector import MemoryInjector

        _, stm, _ = mem_sys
        facts: list = []
        pid = state.current_project_id
        for scope in ("global", f"project:{pid}" if pid else None):
            if scope is None:
                continue
            try:
                facts.extend(
                    state.get_semantic_store(scope).all_facts(
                        include_superseded=False,
                    ),
                )
            except Exception:
                continue
        stm_notes = list(getattr(stm, "notes", {}).values())
        if not facts and not stm_notes:
            return None
        plan = MemoryInjector(cfg).inject(
            mode=inj_mode,
            facts=facts,
            stm_notes=stm_notes,
            current_project_id=pid,
            failure_signatures=(),
        )
        return plan.render() or None
    except Exception as e:
        logger.warning("semmem injection skipped: %s", e)
        return None


def build_chat_messages(
    system_prompt: str, history: list[dict],
    rag_chunks: list[str] | None,
    file_contexts: list[dict] | None,
    context_size: int, max_tokens: int | None,
    rag_scored_chunks: list[tuple[str, float, str]] | None = None,
    salience_ranker=None,
    semmem_block: str | None = None,
) -> list[dict]:
    """messages 組み立て（build_messages で file_contexts・メモリ・RAG・履歴を統合）"""
    messages = build_messages(
        system_prompt, history,
        rag_chunks=rag_chunks,
        file_contexts=file_contexts,
        context_size=context_size,
        max_tokens=max_tokens,
        rag_scored_chunks=rag_scored_chunks,
        salience_ranker=salience_ranker,
        semmem_block=semmem_block,
    )
    logger.debug("Messages assembled: %d messages for LLM", len(messages))
    return messages


async def ensure_base_model_health(
    client: LLMClient, state: AppState, cfg: dict,
) -> tuple[bool, LLMClient | None]:
    """ベースモデルの接続確認 — 未接続なら遅延接続を試行

    Returns:
        (ok, client) のタプル。ok=False の場合は接続失敗。
    """
    base_ok = await client.health_check()
    if base_ok:
        return True, client

    from backend.free.api.system.status import _try_lazy_connect
    llama_cfg = cfg.get("llama", {})
    llama_host = llama_cfg.get("host", "127.0.0.1")
    llama_port = llama_cfg.get("port", 8080)
    llama_url = f"http://{llama_host}:{llama_port}"
    reconnected = await _try_lazy_connect(state, llama_url, llama_cfg)
    if reconnected:
        logger.info("llama-server reconnected during long-form fallback")
        return True, state.llm_client

    logger.warning("Base model unreachable for long-form, returning error")
    return False, client
