"""セッション管理・ルーティング・メモリ統合ロジック"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


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
from backend.free.llm.llm_client import LLMClient
from backend.free.memory.pipeline.search_pipeline import unified_search
from backend.utils import estimate_tokens as _estimate_tokens
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer
    from backend.free.memory.pipeline.conflict_review import (
        PendingConflictGroup,
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

    # ``session_id`` 未指定は **新規セッションの要求** として扱う。
    #
    # 以前は None が下の分岐に入らず、WorkingMemory の現行セッションをそのまま
    # 継続していた。しかも応答には **古い session_id** が載るため、呼出側は
    # 「新しい会話を始めたつもりが前の会話に足されている」ことを検知できない
    # (2026-08-27 ライブ監査: 想起テストのつもりで送った 6 ターンが直前の
    # 108 ターンの続きになり、窓の中を読むだけで答えられてしまった)。
    # API に「新しい会話を始める」手段が無い状態でもあった。
    #
    # 既存 UI は常に session_id を送るので影響を受けない。
    requested_session = req.session_id
    if not requested_session:
        requested_session = str(uuid.uuid4())
        logger.info(
            "Chat request without session_id: starting a new session %s",
            requested_session,
        )

    # セッション切替検出: フロントエンドの session_id が変わったら WM をリセット
    # asyncio.Lock で排他制御し、並列リクエストでの競合を防止
    async with _session_switch_lock:
        if requested_session and wm.session_id != requested_session:
            old_session_id = wm.session_id
            logger.info(
                "Session switch detected: %s -> %s, clearing WorkingMemory",
                old_session_id, requested_session,
            )
            # ``clear()`` を **先に** 実行してから drain する (f_02 §1.2 経路 (b))。
            # 逆順だと drain が拾えるのは「窓超過で既に押し出された分」だけで、
            # ``clear()`` が積んだ会話本体は次ターンの
            # ``drain_evicted_to_stm`` まで滞留する。そちらは現在の
            # ``session_id`` を渡すため、旧セッションのターンが **新しい**
            # セッション ID で STM に吸収され帰属がずれていた。
            # 先に clear すれば、窓超過分と会話本体を 1 回の drain で、
            # かつ正しい旧セッション ID で吸収できる。
            # プロセス終了時の flush (factory/_lifespan.py `_shutdown_wm_flush`)
            # も clear → flush の順で、これで両経路が揃う。
            wm.clear()
            drain_evicted_to_stm(wm, stm, old_session_id)
            wm.session_id = requested_session
            # 旧セッションの蓄積データをクリーンアップ（メモリ解放）
            clear_session_data(old_session_id)
            # quality_judge のセッション単位カウンタもリセット
            # 旧セッションの残存カウントで新セッションが session_cap に
            # 張り付くのを防ぐ。
            if state.judge_tracker is not None:
                state.judge_tracker.reset_session(old_session_id)
            # conflict_chat_judge のセッション内発火カウンタも同様にリセット。
            if state.conflict_judge_tracker is not None:
                state.conflict_judge_tracker.reset_session(old_session_id)
            # 旧会話の経験に conversation_ended を反映 (Level 2 base=C positive 抽出用)。
            # disabled 時は FeedbackCollector 内ガードで no-op。
            if state.feedback_collector is not None:
                state.feedback_collector.mark_conversation_ended()

        # 値の言い直しの印。ここで立てておくと (a) 抽出器が直前の名前付き属性を
        # 継承して訂正が対象と同じスロットへ入り、(b) sleep-time の競合解決が
        # 「同一セッションだから微妙ケース」として pending へ落とすのを免除する。
        # アシスタントの誤りの指摘だけでなく、ユーザー自身の申告訂正
        # (「すみません、登山ではなく写真の間違いでした」) も含む — 記憶側は
        # 「現在値が何か」を持つので、後者も正当な値更新として扱う
        # (restates_a_value / SemanticFact.from_correction 参照)。
        # 判定は純粋関数で、失敗しても訂正の印が付かないだけなので握って続ける。
        try:
            from backend.free.agent.feedback import restates_a_value

            correction = restates_a_value(req.message)
        except Exception as exc:
            logger.warning("correction detection failed (continuing): %s", exc)
            correction = False
        wm.add_turn(
            "user",
            req.message,
            private=req.private,
            mode=req.mode,
            source="user",
            correction=correction,
        )
        history = wm.get_messages()

    # 単位はメッセージ数 (user/assistant を各 1 と数える)。往復数ではない —
    # 「N history turns」と書いていたため、上限 30 を 30 往復と読み違えやすく
    # なっていた (実際は 15 往復)。
    logger.debug(
        "Memory context: %d history messages from WorkingMemory", len(history),
    )

    session_id = requested_session or wm.session_id
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

    __slots__ = ("chunks", "scored_chunks", "error", "query_vec", "rag_top_score")

    def __init__(
        self,
        chunks: list[str] | None = None,
        scored_chunks: list[tuple[str, float, str]] | None = None,
        error: str | None = None,
        query_vec=None,
        rag_top_score: float | None = None,
    ):
        self.chunks = chunks
        self.scored_chunks = scored_chunks
        self.error = error
        # 採用チャンクの生スコア (cosine) 最大値。``scored_chunks`` 側のスコアは
        # 正規化後で Level 0 シグナルには使えない (SearchResult.top_raw_score 参照)。
        self.rag_top_score = rag_top_score
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
        # content gate のセッション上限判定に使う session_id は
        # WorkingMemory 側で常に同期されている (セッション切替時に
        # prepare_memory_context が wm.session_id を更新する)。
        session_id = getattr(wm, "session_id", None) or "default"
        search_result = await unified_search(
            query=query,
            query_vec=query_vec,
            working_mem=wm,
            short_term=stm,
            long_term=ltm,
            cartridge_mgr=state.cartridge_manager,
            config=cfg,
            aux_client=state.aux_client,
            debug_logger=state.debug_logger,
            mode=mode,
            policy=state.policy_interpreter,
            timer=timer,
            semmem_stats=_collect_semmem_stats(state),
            lazy_contextual=state.lazy_contextual,
            session_id=session_id,
            judge_tracker=state.judge_tracker,
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
                rag_top_score=search_result.top_raw_score,
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

    ``collect_pending_conflicts`` が構築し、同ターンの SemMem 注入
    (``build_semmem_injection``) に渡される。

    かつては ``resolved`` / ``resolved_group`` を持ち、直前のユーザー回答で
    解決した競合の確認通知を同ターンに載せていた。回答を判定して
    ``apply_resolution`` へ流す経路が撤去された後もフィールドだけが残り、
    **どこからも代入されない恒偽の分岐**になっていたので落とした
    (解決は sleep-time の ``SemanticConflictResolver`` と TTL が担う)。
    """

    pending_groups: list["PendingConflictGroup"] = field(default_factory=list)


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
    from backend.free.memory.pipeline.conflict_review import (
        collect_review_groups, dedupe_equivalent_groups,
    )

    groups: list = []
    for scope, store in _iter_scopes(state):
        try:
            groups.extend(collect_review_groups(store, scope))
        except Exception as exc:
            logger.warning("collect pending conflicts failed (%s): %s", scope, exc)
    # 1 つの言い直しが複数の型へ書き出されると、同じ「旧…/新…」が
    # グループ数だけ並ぶ。スコープを跨いで畳めるのはここだけ。
    groups = dedupe_equivalent_groups(groups)
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


async def collect_pending_conflicts(
    state: AppState, cfg: dict, *, mode: str = "chat",
) -> ConflictTurnContext:
    """このターンで提示する pending 競合を集める。

    ユーザー回答の LLM 判定は撤去済み。pending は sleep-time の
    ``conflict_resolution`` と TTL 自動解決に委ねる (docs/c_14 §1.2)。
    本関数は注入用の pending グループを収集するだけで、SemMem へは書かない。

    ゲート:
    - ``memory.conflict.chat_review.enabled=false`` → 全体スキップ
    - pending 無し → 即 return

    全例外は warning + 素通しでチャットを止めない。
    """
    ctx = ConflictTurnContext()
    review_cfg = _chat_review_cfg(cfg)
    if not review_cfg.get("enabled", True):
        return ctx
    try:
        ctx.pending_groups = _collect_all_pending_groups(state, mode)
    except Exception as exc:
        logger.warning("pending conflict collection failed: %s", exc)
    return ctx


def _attribute_slots(text: str) -> frozenset[tuple[str, str]]:
    """本文が述べているユーザー属性スロット ``(fact_type, attribute)`` の集合。

    抽出側が subject を決めるのと同じ決定論辞書 (``fact_attributes.yaml``) を
    使う。``core.inference`` から EvorefMem を直接引かせないため、解決器は
    呼出側 (ここ) から渡す。LLM 呼び出しは無い。
    """
    from backend.free.memory.notes.note_builder import resolve_fact_attribute
    from backend.free.memory.pipeline.injector import _USER_ATTRIBUTE_FACT_TYPES

    if not text:
        return frozenset()
    slots: set[tuple[str, str]] = set()
    for fact_type in _USER_ATTRIBUTE_FACT_TYPES:
        attr = resolve_fact_attribute(text, fact_type, mode="chat")
        if attr:
            slots.add((fact_type, attr))
    return frozenset(slots)


def _session_user_texts(state: AppState) -> list[str]:
    """今回の会話でユーザーが述べた本文を返す (属性スロットの述べ直し検出用)。

    窓に残っているものだけで足りる — 押し出された発話は「今回の会話の方が
    新しい」という主張の根拠として使えるほど近くない。
    """
    working = getattr(getattr(state, "mem", None), "working_memory", None)
    if working is None:
        return []
    try:
        return [
            str(t.get("content") or "")
            for t in working.get_context()
            if t.get("role") == "user"
        ]
    except Exception:
        return []


def build_semmem_injection(
    state: AppState, cfg: dict, mode: str = "chat",
    conflict_ctx: ConflictTurnContext | None = None,
    query_vec=None,
    query_text: str = "",
    covered_attributes: set[str] | None = None,
) -> str | None:
    """SemMem facts + STM notes を MemoryInjector で tier 整形し、
    プロンプト注入用テキストを返す。

    chat 応答パスでは SemMem は読み取りのみ (EvorefMem 設計原則 2/7)。
    RAG とは独立して呼び出し、検索ヒットの有無に関わらずメモリを注入する。
    失敗時は None を返してチャットを止めない。

    ``conflict_ctx`` に pending 競合がある場合、Tier パッキングとは独立した
    「記憶の競合」セクションを末尾に連結する (Tier 予算の drop 対象に
    しないことで毎ターンの注入を保証する)。

    **埋め込みが取れなかったターンは facts / notes を注入しない。**
    ``MemoryInjector`` は ``query_embedding=None`` を「関連度ゲート無効」と
    解釈して全候補を通す (``_is_relevant`` 冒頭)。これは embedder 自体が無い
    構成のための後方互換だが、embedder はあるのに埋め込みが失敗 / デッドライン
    超過したターンでも同じ経路に落ちるため、**ゲートが最も要る場面で全店注入に
    切り替わる**。予算 800 トークンが無関係な記憶で埋まり、弱い base モデルが
    それを回答対象と誤解する — 関連度ゲートを入れた元の事象そのもの。
    embedder があるのに ``query_vec`` が無い = そのターンの埋め込みが失敗した、
    と判定して注入を見送る (競合セクションは関連度と無関係なので出す)。

    ``covered_attributes`` を渡すと、**実際に注入されたファクト** の属性スロット名を
    その場で書き込む (``InjectionPlan.covered_attributes``)。呼出側が
    「この属性の現在値はもうプロンプトに載っている」を判定するための出力で、
    ``search_history`` の抑止に使う。``dropped`` になった候補は含めない。
    """
    mem_sys = state.get_memory_system()
    if not mem_sys:
        return None
    inj_mode = normalize_session_mode(mode)
    rendered: str | None = None
    if state.embedder is not None and query_vec is None:
        logger.info(
            "semmem injection skipped: query embedding unavailable this turn "
            "(the relevance gate would be bypassed and inject the whole store)",
        )
        # 競合セクションも同じ理由で見送る。ゲートを掛けられないターンに
        # 出すと、クエリと無関係な矛盾がプロンプトへ入る。
        return None
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
                session_user_texts=_session_user_texts(state),
                query_text=query_text,
            )
            rendered = plan.render() or None
            if covered_attributes is not None:
                covered_attributes.update(plan.covered_attributes)
    except Exception as e:
        logger.warning("semmem injection skipped: %s", e)
        rendered = None

    conflict_block = _render_conflict_section(
        cfg, conflict_ctx, query_vec=query_vec,
    )
    if conflict_block:
        rendered = f"{rendered}\n\n{conflict_block}" if rendered else conflict_block
    return rendered


def _gate_groups_by_relevance(
    cfg: dict, groups: list, query_vec,
) -> list:
    """競合グループをクエリとの関連度で絞る (ゲート無効なら素通し)。

    競合セクションは Tier 予算の外で毎ターン連結される。予算に載せないのは
    「drop されない」保証のためだが、その代わり **関連度の棒も掛かって
    いなかった**。実測 (2026-08-19、chat 29 ターン): 飲み物の競合 2 件が
    28 ターンへ注入され、そのうち飲み物についての質問は **0 件**だった
    (話題は担当プロジェクト / 空きディスク容量 / HTTP と HTTPS / JWT / 日付)。
    中央値 155 トークンを毎ターン再プリフィルしたうえ、無関係な矛盾を
    弱い base に見せ続けることになる。

    棒は注入本体と同じ ``MemoryInjector`` の較正済み閾値を使う
    (:meth:`MemoryInjector.relevant_fact_ids`)。グループは member の
    いずれかが棒を越えれば残す — 競合は「同じスロットの複数の値」なので、
    片方だけがクエリに近いのが普通。
    """
    if not groups:
        return groups
    try:
        from backend.free.memory.pipeline.injector import MemoryInjector

        facts = [f for g in groups for f in g.facts]
        relevant = MemoryInjector(cfg).relevant_fact_ids(facts, query_vec)
    except Exception as exc:
        logger.warning("conflict relevance gate failed (passing through): %s", exc)
        return groups
    if relevant is None:
        return groups
    kept = [g for g in groups if any(f.id in relevant for f in g.facts)]
    if len(kept) != len(groups):
        logger.debug(
            "conflict groups: %d/%d passed the relevance gate",
            len(kept), len(groups),
        )
    return kept


def _render_conflict_section(
    cfg: dict, conflict_ctx: ConflictTurnContext | None, *, query_vec=None,
) -> str | None:
    """pending 競合セクションを組み立てる。失敗時 / 該当なしは None。

    ``query_vec`` を渡すと関連度ゲートが掛かる
    (:func:`_gate_groups_by_relevance`)。``None`` はゲート無効 = 素通しで、
    embedder 自体が無い構成のための後方互換。
    """
    if conflict_ctx is None or not conflict_ctx.pending_groups:
        return None
    try:
        from backend.free.memory.pipeline.conflict_review import (
            render_pending_conflicts_block,
        )

        groups = _gate_groups_by_relevance(
            cfg, conflict_ctx.pending_groups, query_vec,
        )
        if not groups:
            return None
        review_cfg = _chat_review_cfg(cfg)
        return render_pending_conflicts_block(
            groups,
            max_groups=int(review_cfg.get("max_groups", 3) or 0),
            max_tokens=int(review_cfg.get("max_tokens", 400) or 0),
        )
    except Exception as exc:
        logger.warning("conflict section render failed: %s", exc)
        return None


def session_evicted_turns(state: AppState) -> int:
    """現在セッションでワーキングメモリから押し出したターン数を安全に取る。

    pillar 未構築 (degraded / テストの部分モック) では 0 を返す。
    """
    working = getattr(getattr(state, "mem", None), "working_memory", None)
    return int(getattr(working, "session_evicted_turns", 0) or 0)


def session_first_user_message(state: AppState) -> str:
    """現在セッションで最初に届いた user 発話を安全に取る。

    窓から押し出されても ``WorkingMemory`` が 1 件だけ保持している
    (``session_first_user_turn``)。「この会話で最初に何を言ったか」を
    押し出し後も決定論で答えるための材料。

    pillar 未構築 (degraded / テストの部分モック) では空文字列を返す。
    """
    working = getattr(getattr(state, "mem", None), "working_memory", None)
    return str(getattr(working, "session_first_user_turn", "") or "")


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
        slot_resolver=_attribute_slots,
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
