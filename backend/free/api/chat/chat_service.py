"""セッション管理・ルーティング・メモリ統合ロジック"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from backend.app_state import AppState
from backend.free.api.chat.chat_constants import DEFAULT_WORKING_MAX_TOKENS
from backend.free.api.chat.chat_recorder import clear_session_data, drain_evicted_to_stm
from backend.free.api.chat.chat_types import ChatMessage, FileContextDict
from backend.free.api.schemas import ChatRequest
from backend.free.core.intent_vocab import (
    is_whole_session_scope_query,
    self_output_measure_kinds,
)
from backend.free.core.session_mode import is_chat_mode, normalize_session_mode
from backend.free.core.inference import build_messages
from backend.free.core.turn_text import append_to_last_user
from backend.free.llm.assist_client import assist_ready
from backend.free.llm.llm_client import LLMClient
from backend.free.memory.pipeline.search_pipeline import unified_search
from backend.utils import estimate_tokens as _estimate_tokens
from backend.log_config import get_logger
from backend.trace_context import get_trace_id

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer
    from backend.free.memory.pipeline.conflict_review import (
        PendingConflictGroup,
        ResolutionResult,
    )

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
) -> tuple[list[ChatMessage], str]:
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
            # conflict_chat_judge のセッション内発火カウンタも同様にリセット。
            if state.conflict_judge_tracker is not None:
                state.conflict_judge_tracker.reset_session(old_session_id)
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

    # 単位はメッセージ数 (user/assistant を各 1 と数える)。往復数ではない —
    # 「N history turns」と書いていたため、上限 30 を 30 往復と読み違えやすく
    # なっていた (実際は 15 往復)。
    logger.debug(
        "Memory context: %d history messages from WorkingMemory", len(history),
    )

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

    __slots__ = ("chunks", "scored_chunks", "error", "query_vec")

    def __init__(
        self,
        chunks: list[str] | None = None,
        scored_chunks: list[tuple[str, float, str]] | None = None,
        error: str | None = None,
        query_vec=None,
    ):
        self.chunks = chunks
        self.scored_chunks = scored_chunks
        self.error = error
        # クエリ埋め込み。MemoryInjector の関連度ゲートが再利用する
        # (検索が necessity judge で skip されても埋め込み自体は計算済みなので、
        #  ゲートを効かせるために持ち回る)。
        self.query_vec = query_vec

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
            timer=timer,
            semmem_stats=_collect_semmem_stats(state),
            lazy_contextual=state.lazy_contextual,
            session_id=session_id,
            assist_judge_tracker=state.assist_judge_tracker,
            necessity_prompt=necessity_prompt,
            quality_prompt=quality_prompt,
            assist_experience_recorder=state.assist_experience_recorder,
            mem_view=state.mem_view,
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
                query_vec=query_vec,
            )
    except Exception as e:
        logger.warning("Search pipeline failed, continuing without RAG: %s", e)
        return SearchPipelineResult(error=str(e))

    # necessity judge が retrieve を skip した経路。チャンクは無いが
    # query_vec は算出済みなので、関連度ゲート用に返す。
    return SearchPipelineResult(query_vec=locals().get("query_vec"))


@dataclass
class ConflictTurnContext:
    """1 ターン分の pending 競合コンテキスト。

    ``maybe_resolve_pending_conflicts`` が構築し、同ターンの SemMem 注入
    (``build_semmem_injection``) に渡される。``resolved`` が非 None の場合、
    直前のユーザー発話で解決が反映済み (注入には確認通知を含める)。
    """

    pending_groups: list["PendingConflictGroup"] = field(default_factory=list)
    resolved: "ResolutionResult | None" = None
    # 解決した競合グループ (解決通知を番号でなく内容で示すため保持する)。
    resolved_group: "PendingConflictGroup | None" = None


def _iter_scopes(state: AppState):
    """(scope 名, store) を global → project の順で yield する。"""
    pid = state.current_project_id
    for scope in ("global", f"project:{pid}" if pid else None):
        if scope is None:
            continue
        try:
            yield scope, state.get_semantic_store(scope)
        except Exception:
            continue


#: chat モードで注入しない競合グループの FactType。
#:
#: project スコープには create_task / create (ソースコード本文を含む) の
#: pending が溜まる。これを chat の競合セクションへ載せると、ユーザーに
#: 「todo_item.py の 2 版のどちらが正しいか」を毎ターン尋ねることになり、
#: Tier 予算の外で数百トークンを消費する (実測 2026-07-25: 16 件 ≒ 840 tokens。
#: 弱い base モデルがこれを回答対象と誤解し、本題と無関係な応答を返していた)。
_CREATE_ONLY_CONFLICT_TYPES = frozenset({"create", "create_task"})


def _collect_all_pending_groups(state: AppState, mode: str = "chat") -> list:
    """global + project ストアの pending 競合グループを集約する (読取のみ)。

    chat モードではクリエイト専用型 (``create`` / ``create_task``) の
    グループを除外する。type は混在時 ``"a/b"`` 形式なので、構成型がすべて
    クリエイト専用のときだけ落とす (混在は残す)。
    """
    # 表示用グループを使う (collect_pending_groups ではない)。pending だけを
    # 並べるとスロットの最新値が欠け、古い値に「新」ラベルが付く
    # (``collect_review_groups`` の docstring 参照)。解決対象は変わらない。
    from backend.free.memory.pipeline.conflict_review import collect_review_groups

    groups: list = []
    for scope, store in _iter_scopes(state):
        try:
            groups.extend(collect_review_groups(store, scope))
        except Exception as exc:
            logger.warning("collect pending conflicts failed (%s): %s", scope, exc)
    if is_chat_mode(mode):
        before = len(groups)
        groups = [
            g for g in groups
            if not set((g.type or "").split("/")).issubset(
                _CREATE_ONLY_CONFLICT_TYPES,
            )
        ]
        if before != len(groups):
            logger.debug(
                "conflict groups: dropped %d create-only group(s) in chat mode",
                before - len(groups),
            )
    return groups


def _chat_review_cfg(cfg: dict) -> dict:
    return ((cfg.get("memory") or {}).get("conflict") or {}).get(
        "chat_review",
    ) or {}


async def maybe_resolve_pending_conflicts(
    state: AppState, cfg: dict, history: list[ChatMessage], user_message: str,
    *, allow_write: bool = True, session_id: str = "default",
    mode: str = "chat",
) -> ConflictTurnContext:
    """pending 競合のユーザー回答を assist で判定し、有効なら即時反映する。

    SemMem 書込は sleep-time に閉じる不変則の例外 2 例目
    (``conflict_review.apply_resolution``、CLAUDE.md §6.2)。

    ゲート:
    - ``memory.conflict.chat_review.enabled=false`` → 全体スキップ
    - pending 無し → 即 return
    - ``allow_write=False`` (private ターン等) → 判定/書込スキップ (注入は継続)
    - アシストが無い / 非常駐 (``residency: on_demand`` のチャット中) →
      判定スキップ (注入は情報提示のみで継続)。pending は sleep-time の
      ``conflict_resolution`` と TTL 自動解決に委ねる
    - history に assistant 発話なし (= まだ確認を出していない) → 判定スキップ
    - ``chat_review.max_judge_per_session`` 到達 → 判定**と注入**を停止
      (回答されないまま毎ターン assist スロットを専有するのを避ける。pending は
      次セッション or sleep-time TTL で解消する)

    タイムアウト (インフラ的失敗) は「ユーザーが回答しなかった」判定とは
    区別し、セッション cap を消費しない (呼出前に予約したカウントを
    ``tracker.refund`` で払い戻す)。予約自体は並行リクエストでの cap
    超過防止のため呼出前に行う。

    全例外は warning + 素通しでチャットを止めない。
    """
    ctx = ConflictTurnContext()
    review_cfg = _chat_review_cfg(cfg)
    if not review_cfg.get("enabled", True):
        return ctx
    try:
        ctx.pending_groups = _collect_all_pending_groups(state, mode)
        if not ctx.pending_groups:
            return ctx
        # セッション内発火上限: 到達済みなら判定も注入も止める。allow_write
        # チェックより前に置き、private ターンでも注入を止める。
        cap = int(review_cfg.get("max_judge_per_session", 3) or 0)
        tracker = state.conflict_judge_tracker
        if (
            cap > 0
            and tracker is not None
            and tracker.get_session_count(session_id, namespace="conflict_chat_judge") >= cap
        ):
            logger.debug(
                "conflict_chat_judge session cap reached (%d), "
                "suppressing pending injection for session=%s",
                cap, session_id,
            )
            ctx.pending_groups = []
            return ctx
        if not allow_write:
            # private ターン: pending の提示 (注入) はするが、ユーザー回答判定に
            # よる SemMem 書込 (apply_resolution) は行わない。private 契約
            # (LTM/SemMem/履歴へ書かない) を競合解決経路でも守る。
            return ctx
        if not assist_ready(state.assist_client, "conflict_chat_judge"):
            return ctx
        last_assistant = next(
            (
                m.get("content", "")
                for m in reversed(history)
                if m.get("role") == "assistant"
            ),
            "",
        )
        if not last_assistant:
            return ctx

        from backend.free.memory.pipeline.conflict_review import (
            apply_resolution,
            judge_user_reply,
        )

        # assist を呼びに行く時点でカウント (タイムアウトでも realtime スロットは
        # 消費されるため、呼出前に数える)。pending 無し / private / assist 未接続 /
        # assistant 履歴なしの早期 return はここに到達せずカウントしない。
        if tracker is not None:
            count = tracker.record(session_id, namespace="conflict_chat_judge")
            # 並行リクエストが上の cap チェックと record の間に割り込んで cap を
            # 超えた場合、record の atomic な戻り値で判定し、このターンは judge を
            # 打たず注入も止める (発火数が cap を超えないようにする)。
            if cap > 0 and count > cap:
                logger.debug(
                    "conflict_chat_judge over session cap (%d/%d) for "
                    "session=%s (concurrent); skipping judge this turn",
                    count, cap, session_id,
                )
                ctx.pending_groups = []
                return ctx
            if cap > 0 and count >= cap:
                logger.info(
                    "conflict_chat_judge reached session cap (%d/%d) for "
                    "session=%s; further turns skip judging and injection",
                    count, cap, session_id,
                )

        try:
            judgement = await judge_user_reply(
                state.assist_client,
                groups=ctx.pending_groups,
                user_message=user_message,
                last_assistant_message=last_assistant,
            )
        except (httpx.TimeoutException, TimeoutError):
            if tracker is not None:
                tracker.refund(session_id, namespace="conflict_chat_judge")
            logger.warning(
                "conflict_chat_judge timed out; not counted against "
                "session cap (session=%s)", session_id,
            )
            return ctx
        if judgement is None:
            return ctx

        index = judgement["group_index"]
        group = ctx.pending_groups[index - 1]
        action = judgement["action"]
        winner = group.oldest if action == "keep_old" else group.newest
        losers = [f.id for f in group.facts if f.id != winner.id]
        store = state.get_semantic_store(group.scope)
        result = apply_resolution(
            store,
            scope=group.scope,
            action=action,
            winner_id=winner.id,
            loser_ids=losers,
            merged_object=judgement["merged_object"] or None,
            decision_source="user_chat",
            trace_id=get_trace_id(),
        )
        ctx.resolved = result
        ctx.resolved_group = group
        logger.info(
            "SemMem conflict resolved via chat: scope=%s action=%s winner=%s",
            group.scope, action, winner.id,
        )
        if state.debug_logger is not None:
            # 監査ログ失敗で後続の pending 再収集 (同ターン注入の整合) を
            # 落とさないよう、debug ログ単体を握る。
            try:
                state.debug_logger.log_memory_op(
                    "semmem_conflict_user_resolve",
                    {
                        "scope": group.scope,
                        "action": action,
                        "winner_id": result.winner_id,
                        "superseded": len(result.superseded_ids),
                        "new_fact_id": result.new_fact_id or "",
                    },
                )
            except Exception as exc:
                logger.warning("conflict resolve audit log failed: %s", exc)
        # 解決を同ターンの注入へ反映するため pending を再収集
        ctx.pending_groups = _collect_all_pending_groups(state, mode)
    except Exception as exc:
        logger.warning("pending conflict chat review failed: %s", exc)
    return ctx


def build_semmem_injection(
    state: AppState, cfg: dict, mode: str = "chat",
    conflict_ctx: ConflictTurnContext | None = None,
    query_vec=None,
) -> str | None:
    """SemMem facts + STM notes を MemoryInjector で tier 整形し、
    プロンプト注入用テキストを返す。

    chat 応答パスでは SemMem は読み取りのみ (EvorefMem 設計原則 2/7)。
    RAG とは独立して呼び出し、検索ヒットの有無に関わらずメモリを注入する。
    失敗時は None を返してチャットを止めない。

    ``conflict_ctx`` に pending 競合がある場合、Tier パッキングとは独立した
    「記憶の競合」セクションを末尾に連結する (Tier 予算の drop 対象に
    しないことで毎ターンの注入を保証する)。
    """
    mem_sys = state.get_memory_system()
    if not mem_sys:
        return None
    inj_mode = normalize_session_mode(mode)
    rendered: str | None = None
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
        if facts or stm_notes:
            plan = MemoryInjector(cfg).inject(
                mode=inj_mode,
                facts=facts,
                stm_notes=stm_notes,
                current_project_id=pid,
                failure_signatures=(),
                query_embedding=query_vec,
            )
            rendered = plan.render() or None
    except Exception as e:
        logger.warning("semmem injection skipped: %s", e)
        rendered = None

    conflict_block = _render_conflict_section(state, cfg, conflict_ctx)
    if conflict_block:
        rendered = f"{rendered}\n\n{conflict_block}" if rendered else conflict_block
    return rendered


def _render_conflict_section(
    state: AppState, cfg: dict, conflict_ctx: ConflictTurnContext | None,
) -> str | None:
    """pending 競合セクション (+ 解決済み通知) を組み立てる。失敗時は None。"""
    if conflict_ctx is None:
        return None
    if not conflict_ctx.pending_groups and conflict_ctx.resolved is None:
        return None
    try:
        from backend.free.memory.pipeline.conflict_review import (
            render_pending_conflicts_block,
            render_resolved_notice,
        )

        review_cfg = _chat_review_cfg(cfg)
        parts: list[str] = []
        block = render_pending_conflicts_block(
            conflict_ctx.pending_groups,
            # 回答を促す文言は「その回答を判定できる」ときだけ出す。
            # アシスト非常駐では judge が動かないため、指示なしの情報提示に留める。
            instruct=assist_ready(state.assist_client, "conflict_chat_judge"),
            max_groups=int(review_cfg.get("max_groups", 3) or 0),
            max_tokens=int(review_cfg.get("max_tokens", 400) or 0),
        )
        if block:
            parts.append(block)
        if conflict_ctx.resolved is not None and conflict_ctx.resolved_group is not None:
            parts.append(
                render_resolved_notice(
                    conflict_ctx.resolved, conflict_ctx.resolved_group,
                ),
            )
        return "\n".join(parts) or None
    except Exception as exc:
        logger.warning("conflict section render failed: %s", exc)
        return None


def session_evicted_turns(state: AppState) -> int:
    """現在セッションでワーキングメモリから押し出したターン数を安全に取る。

    pillar 未構築 (degraded / テストの部分モック) では 0 を返す。
    """
    working = getattr(getattr(state, "mem", None), "working_memory", None)
    return int(getattr(working, "session_evicted_turns", 0) or 0)


def build_chat_messages(
    system_prompt: str, history: list[ChatMessage],
    rag_chunks: list[str] | None,
    file_contexts: list[dict] | None,
    context_size: int, max_tokens: int | None,
    rag_scored_chunks: list[tuple[str, float, str]] | None = None,
    salience_ranker=None,
    semmem_block: str | None = None,
    fewshot_block: str | None = None,
    history_min_tokens: int = 0,
    working_max_tokens: int = DEFAULT_WORKING_MAX_TOKENS,
    evicted_turns: int = 0,
) -> list[ChatMessage]:
    """messages 組み立て（build_messages で few-shot・file・メモリ・RAG・履歴を統合）。

    ``system_prompt`` は静的 (query 非依存)、``fewshot_block`` 等の query 依存部は
    build_messages 内で最後の user メッセージへ前置される (KV キャッシュ対応)。

    Args:
        evicted_turns: 現在セッションでワーキングメモリから押し出したターン数
            (``WorkingMemory.session_evicted_turns``)。0 より大きい場合、会話
            全体を走査しないと答えられない質問には切り詰め注記を付ける。
    """
    messages = build_messages(
        system_prompt, history,
        rag_chunks=rag_chunks,
        file_contexts=file_contexts,
        context_size=context_size,
        max_tokens=max_tokens,
        rag_scored_chunks=rag_scored_chunks,
        salience_ranker=salience_ranker,
        semmem_block=semmem_block,
        fewshot_block=fewshot_block,
        history_min_tokens=history_min_tokens,
        working_max_tokens=working_max_tokens,
    )
    apply_grounding_notes(messages, history, evicted_turns)
    logger.debug("Messages assembled: %d messages for LLM", len(messages))
    return messages


def apply_grounding_notes(
    messages: list[ChatMessage],
    history: list[ChatMessage],
    evicted_turns: int,
) -> None:
    """視界の欠落と自己出力の計量に関する注記をまとめて付ける (in-place)。

    **messages を組み立てる全経路がここを通ること**。reactive 軽量パス
    (``chat._dispatch_reactive_light``) は ``build_chat_messages`` を通らず
    独自に messages を組むため、片方にだけ入れると軽量パスが素通しになる。
    実際、2026-08-05 ライブ監査で捏造が起きた 2 ターンはどちらも軽量パス
    だった (「今の回答は実際に何字ありましたか？」が short_query →
    reactive_light に落ち、直前の出力を数えずに「488 文字」と答えた)。

    ``history`` は**切り詰め前の全履歴**を渡すこと。計量対象の直前 assistant
    発言は軽量パスの窓から外れていることがある。
    """
    _append_truncated_history_note(messages, history, evicted_turns)
    _append_self_output_measurement(messages, history)


# 会話の前半がワーキングメモリから押し出された状態で、会話全体を走査しないと
# 答えられない質問が来たときの注記。
#
# モデルには「見えている履歴」と「会話の全体」の区別が付かないため、切り詰めを
# 伝えないと部分的な視界を全体だと思い込んで断定する (2026-08-05 ライブ監査:
# ターン19 のファイル書き込みが 30 メッセージ窓から落ちた状態で「この会話で
# 依頼したファイル操作を全部リストアップして」→「ファイル操作はありません」、
# ターン7 で読んだ README が窓外の状態で「最初に読ませたファイルは」→ 窓内で
# 最後に読んだ別ファイルを回答)。
#
# 「答えるな」ではなく「見えている範囲を答え、見えていない範囲を断定するな」
# と書く。否定形の禁止だけを書くと退行する実測がある (2026-07-28)。
_TRUNCATED_HISTORY_GUIDANCE = (
    "\n\nこの質問は会話全体を見ないと正確に答えられないが、"
    "この会話の前半 {n} 件のやり取りは文脈の上限を超えたため、"
    "今のあなたには見えていない。"
    "見えている範囲について答えたうえで、"
    "「会話の前半は参照できないため、これより前にもあった可能性がある」"
    "と明示すること。"
    "見えていない範囲について「無い」「一度も〜していない」と断定しないこと。"
)


def _append_truncated_history_note(
    messages: list[ChatMessage],
    history: list[ChatMessage],
    evicted_turns: int,
) -> None:
    """履歴が切り詰められている状態の全体走査質問へ注記を付ける (in-place)。

    切り詰めが起きていない (``evicted_turns == 0``) 場合は何もしない。全体走査
    質問の判定が多少広くても、切り詰めが無ければ注記は出ないためコストは無い。
    """
    if evicted_turns <= 0:
        return
    query = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            query = str(msg.get("content") or "")
            break
    if not is_whole_session_scope_query(query):
        return
    if append_to_last_user(
        messages, _TRUNCATED_HISTORY_GUIDANCE.format(n=evicted_turns),
        separator="",
    ):
        logger.debug(
            "Truncated-history note appended (%d turns evicted): %s",
            evicted_turns, query[:60],
        )


# 直前の自分の出力の計量結果を「実測値」として渡す注記。
#
# 自分が出した文章の文字数はモデルには数えられない (2026-08-05 ライブ監査
# ターン33: 実測 633 文字の回答を「488 文字」と回答)。しかもクエリが短いため
# router が reactive に落とし、ツール判定すら走らない経路だった。ファイルの
# 文字数を read_file のメタ行で決定論化したのと同じ扱いにする。
_SELF_OUTPUT_MEASURE_LABELS: dict[str, str] = {
    "chars": "文字数",
    "lines": "行数",
    "words": "単語数",
}
_SELF_OUTPUT_MEASUREMENT_GUIDANCE = (
    "\n\n[システム計測] 直前のあなたの回答を機械的に数えた結果: {values}。"
    "この数値をそのまま使って答えること。自分で数え直したり概算したりしないこと。"
)


def _measure_text(text: str, kinds: tuple[str, ...]) -> list[str]:
    """直前出力の計量結果を人間可読な文字列リストにする (純粋関数)。"""
    parts: list[str] = []
    for kind in kinds:
        label = _SELF_OUTPUT_MEASURE_LABELS[kind]
        if kind == "chars":
            stripped = "".join(text.split())
            parts.append(
                f"{label} {len(text)} 文字"
                f" (空白・改行を除くと {len(stripped)} 文字)",
            )
        elif kind == "lines":
            parts.append(f"{label} {len(text.splitlines())} 行")
        else:
            parts.append(f"{label} {len(text.split())} 語")
    return parts


def _append_self_output_measurement(
    messages: list[ChatMessage], history: list[ChatMessage],
) -> None:
    """「今の回答は何文字?」に実測値を添える (in-place)。

    直前の assistant 発言が無い / 計量質問でない場合は何もしない。
    """
    query = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            query = str(msg.get("content") or "")
            break
    kinds = self_output_measure_kinds(query)
    if not kinds:
        return
    previous = ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            previous = str(msg.get("content") or "")
            break
    if not previous.strip():
        return
    values = "、".join(_measure_text(previous, kinds))
    if append_to_last_user(
        messages, _SELF_OUTPUT_MEASUREMENT_GUIDANCE.format(values=values),
        separator="",
    ):
        logger.debug("Self-output measurement injected: %s", values)


# モデル入替中 (mode 切替が llama-server を再起動する) の待機上限とポーリング間隔。
# 9B Q4_K_M の実測ロード時間は iGPU で 5-15 秒、寒いページキャッシュだと数十秒。
BASE_MODEL_LOADING_WAIT_SEC = 90.0
BASE_MODEL_LOADING_POLL_SEC = 2.0


async def _wait_base_model_loading(client: LLMClient, cfg: dict) -> bool:
    """ベースモデルのロード完了を上限付きで待つ。

    mode 切替 (chat ↔ create) は llama-server を停止 → 再起動するため、
    その最中に届いたチャット要求は ``/health`` 503 に当たる。従来は 1 回の
    health_check 失敗で「LLM サーバーに接続できません。llama-server が起動
    しているか確認してください。」を返しており、実際にはロード中なだけなのに
    ユーザーへ誤った対処を促していた (実インシデント 2026-07-27 ライブ検証:
    モードを create へ切替えた直後の 1 通目が失敗)。

    ポートが LISTEN されていなければ本当に起動していないので待たない
    (llama-server 未起動という本来のケースを遅延させない)。
    """
    from backend.free.cli.pid_manager import find_port_occupant

    port = (cfg.get("llama") or {}).get("port", 8080)
    if await asyncio.to_thread(find_port_occupant, port) is None:
        return False

    logger.info(
        "Base model not ready but port %d is occupied; "
        "waiting up to %.0fs for model load to finish",
        port, BASE_MODEL_LOADING_WAIT_SEC,
    )
    deadline = time.monotonic() + BASE_MODEL_LOADING_WAIT_SEC
    while time.monotonic() < deadline:
        await asyncio.sleep(BASE_MODEL_LOADING_POLL_SEC)
        if await client.health_check():
            logger.info("Base model became healthy after model load")
            return True
        if await asyncio.to_thread(find_port_occupant, port) is None:
            logger.warning("llama-server disappeared while waiting for model load")
            return False
    logger.warning(
        "Base model still not healthy after %.0fs wait", BASE_MODEL_LOADING_WAIT_SEC,
    )
    return False


async def ensure_base_model_health(
    client: LLMClient, state: AppState, cfg: dict,
) -> tuple[bool, LLMClient | None]:
    """ベースモデルの接続確認 — ロード中は待機し、なお駄目なら遅延接続を試行

    Returns:
        (ok, client) のタプル。ok=False の場合は接続失敗。
    """
    base_ok = await client.health_check()
    if base_ok:
        return True, client

    if await _wait_base_model_loading(client, cfg):
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
