"""セッション管理・ルーティング・メモリ統合ロジック"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


from backend.app_state import AppState
from backend.free.api.chat.chat_constants import DEFAULT_WORKING_MAX_TOKENS
from backend.free.api.chat.chat_recorder import accumulate_user_turn
from backend.free.api.chat.chat_types import ChatMessage, FileContextDict
from backend.free.api.schemas import ChatRequest
from backend.free.core.intent_vocab import (
    split_user_text_measurement,
    _PRIOR_OUTPUT_CODE_RE,
    conversation_turn_count_question,
    is_whole_session_scope_query,
    referenced_quantity,
    occurrence_count_term,
    self_output_measure_kinds,
)
from backend.free.core.session_mode import is_chat_mode, normalize_session_mode
from backend.free.core.text_quality import (
    SYSTEM_MEASUREMENT_MARKER,
    conversational_numeric_claims,
    count_response_lines,
    strip_system_notes,
)
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
    """メモリからコンテキストを取得する (セッション別 WM)

    WM はセッション別 (``WorkingMemoryRegistry``) なので「切替」は無い —
    要求されたセッションの窓を引いて user 発話を積むだけ。旧セッションの
    後始末 (drain / カウンタ reset) は明示のセッション終了
    (``chat_recorder.end_session``) か台帳の LRU 押し出しが担う。

    Returns:
        (history, session_id) のタプル
    """
    if not state.get_memory_system():
        logger.debug("Memory not initialized, using single-turn context")
        session_id = req.session_id or "default"
        accumulate_user_turn(session_id, req.message, private=req.private)
        return [{"role": "user", "content": req.message}], session_id

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

    # このセッションの窓 (無ければ台帳が作る)。別セッションの窓には触らない。
    wm, _stm, _ltm = state.get_memory_system(requested_session)

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
    if correction:
        # 「この会話で私が訂正した回数は何回ですか。」に答える材料。
        # 監査では「1回です」(実際は 4 回) と答え、しかも挙げた 1 件は
        # **モデルがユーザーを訂正した** ケースで主体が逆だった。
        from backend.free.agent.issue_ledger import record_correction

        record_correction(requested_session, req.message)
    wm.add_turn(
        "user",
        req.message,
        private=req.private,
        mode=req.mode,
        source="user",
        correction=correction,
    )
    # 履歴の蓄積バッファにも **ここで** 積む。record_* の末尾だけだと、
    # 生成が失敗 / タイムアウトしたターンは WM に居るのに履歴には無い。
    # 冪等なので record_* 側の保険と二重には数えない。
    accumulate_user_turn(requested_session, req.message, private=req.private)
    history = wm.get_messages()

    # 単位はメッセージ数 (user/assistant を各 1 と数える)。往復数ではない —
    # 「N history turns」と書いていたため、上限 30 を 30 往復と読み違えやすく
    # なっていた (実際は 15 往復)。
    logger.debug(
        "Memory context: %d history messages from WorkingMemory (session=%s)",
        len(history), requested_session,
    )

    return history, requested_session


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
    timer: StageTimer | None = None, *, session_id: str | None = None,
) -> SearchPipelineResult:
    """統合検索パイプライン: 3層メモリ + Self-RAG

    ``session_id`` はこのターンのセッション。WM (窓の完全性判定 / セッション
    自己参照枝) と content gate のセッション上限はその窓で判定する。

    Returns:
        SearchPipelineResult — チャンクリスト + エラー情報。
        BUG-9 対策: 失敗時はエラー情報を保持し、呼び出し元で
        フロントエンドに通知可能にする。
    """
    mem_sys = state.get_memory_system(session_id)
    if not mem_sys or state.embedder is None:
        return SearchPipelineResult()

    query_vec = None
    try:
        if timer:
            timer.start("embedding_ms")
        # mode 別 instruction を埋め込みへ伝搬
        query_vec = await state.embedder.embed_query(query, mode=mode)
        if timer:
            timer.stop("embedding_ms")

        wm, stm, ltm = mem_sys
        # content gate のセッション上限判定に使う session_id。呼出側が渡さない
        # legacy 経路では WM の session_id で代用する。
        session_id = session_id or getattr(wm, "session_id", None) or "default"
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
            # STM は [関連する記憶] と [参考情報] の 2 経路で注入される。
            # 訂正で退役した値は **両方**で止める (片方だけだともう片方から
            # 訂正前の値が返る — 2026-08-27 実機検証)。
            retired_note_ids=_collect_retired_note_ids(
                state, state.current_project_id,
            ),
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
        # 埋め込みまでは済んでいるなら関連度ゲート用に返す。ここで落とすと
        # ``build_semmem_injection`` が「埋め込み失敗」と判定して SemMem の
        # 注入ごと見送り、RAG の失敗 1 件でそのターンの記憶が全部消える
        # (2026-09-02 監査 S-A1: 次元不一致のカートリッジ 1 つで再現)。
        return SearchPipelineResult(error=str(e), query_vec=query_vec)

    # necessity judge が retrieve を skip した経路。チャンクは無いが
    # query_vec は算出済みなので、関連度ゲート用に返す。
    return SearchPipelineResult(query_vec=query_vec)


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


def _session_wm(state: AppState, session_id: str | None):
    """``session_id`` の WorkingMemory を安全に取る (pillar 未構築なら ``None``)。"""
    get = getattr(state, "get_memory_system", None)
    if not callable(get):
        return None
    try:
        mem_sys = get(session_id)
    except Exception:
        return None
    return mem_sys[0] if mem_sys else None


def _session_user_texts(state: AppState, session_id: str | None) -> list[str]:
    """今回の会話でユーザーが述べた本文を返す (属性スロットの述べ直し検出用)。

    窓に残っているものだけで足りる — 押し出された発話は「今回の会話の方が
    新しい」という主張の根拠として使えるほど近くない。
    """
    working = _session_wm(state, session_id)
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


def _collect_retired_note_ids(state, project_id: str | None) -> set[str]:
    """値が supersede された発話ノートの ID を集める。

    SemMem 側で世代を閉じても (``sleep/extraction._supersede_corrected_slots``)、
    その値を述べた **STM ノートの原文** はそのまま残る。STM は
    ``[関連する記憶]`` の Tier 2 として注入されるので、訂正前の発話が
    「現在値」として読まれ続ける。

    実機検証 (2026-08-27、クリーンなストア):
    ``mem.personal.occupation`` は「データベース管理者」が supersede され
    「ネットワークエンジニア」が live になっていたにもかかわらず、新規
    セッションの「私の職業と住んでいる場所を教えてください。」が
    **「データベース管理者」** を返した。供給元は STM に残った
    「職業はデータベース管理者で、名古屋に住んでいます。」のノートだった。

    ノート単位で見て、そこから生まれたファクトが **すべて** supersede
    済みのときだけ落とす。1 つでも live が残っていれば、そのノートは
    まだ現在値を運んでいる。

    履歴そのものは消さない — ``search_history`` や会話履歴ファイルからは
    従来どおり辿れる。ここで止めるのは「現在値として黙って提示する」経路
    だけ。

    **ストアの世代番号でキャッシュする。** この走査は全ファクト × 全
    provenance で、チャット 1 ターンごとに scope の数だけ走る。ストアは
    プロセス常駐なので、書込 (= sleep-time) が無い間は結果が変わらない。
    ``SemanticFactStore.revision`` が動いたときだけ作り直す。
    """
    retired: dict[str, bool] = {}
    for scope in ("global", f"project:{project_id}" if project_id else None):
        if scope is None:
            continue
        try:
            store = state.get_semantic_store(scope)
        except Exception:
            continue
        cached = _retired_note_ids_for_store(store)
        if cached is None:
            continue
        for note_id, only_retired in cached.items():
            retired[note_id] = retired.get(note_id, True) and only_retired
    return {note_id for note_id, only_retired in retired.items() if only_retired}


#: ``_collect_retired_note_ids`` のストア別キャッシュ。
#: ``store -> (revision, {note_id: そのノート由来のファクトが全部 supersede 済みか})``
#: 弱参照キー — ストアが差し替わったら (再ロード / テストの使い捨て) 古い
#: エントリは一緒に消える。``id(store)`` をキーにすると、解放されたアドレスを
#: 新しいストアが再利用したときに **別ストアの結果を返す** うえ、寿命が無い。
_RETIRED_NOTE_CACHE: "weakref.WeakKeyDictionary[Any, tuple[int, dict[str, bool]]]" = (
    weakref.WeakKeyDictionary()
)


def _retired_note_ids_for_store(store) -> dict[str, bool] | None:
    """1 ストア分の ``{note_id: 全部 supersede 済みか}`` を返す (revision キャッシュ)。"""
    try:
        revision = int(store.revision)
    except Exception:
        revision = -1
    try:
        hit = _RETIRED_NOTE_CACHE.get(store)
    except TypeError:
        # 弱参照できない型 (テストの部分モック等) はキャッシュしない。
        hit = None
        revision = -1
    if hit is not None and hit[0] == revision and revision >= 0:
        return hit[1]
    try:
        all_facts = store.all_facts(include_superseded=True)
    except Exception:
        return None
    computed: dict[str, bool] = {}
    for fact in all_facts:
        is_retired = bool(getattr(fact, "superseded_by", None))
        for prov in getattr(fact, "provenances", ()) or ():
            note_id = getattr(prov, "note_id", None)
            if not note_id:
                continue
            # live が 1 つでもあれば False で確定させる。
            computed[note_id] = computed.get(note_id, True) and is_retired
    if revision >= 0:
        _RETIRED_NOTE_CACHE[store] = (revision, computed)
    return computed


def _semmem_embeddings_are_stale() -> bool:
    """SemMem ファクトの埋め込みが embed モデル切替で stale になっているか。

    ``stale_guard`` のマーカー (``.reembed_facts_required``) を見る。判定に
    失敗したら ``False`` (= 従来どおり注入する) — マーカーの読み取り失敗で
    記憶を止める方が害が大きい。
    """
    try:
        from backend.free.memory.semantic.stale_guard import (
            is_semmem_reembed_required,
        )

        return is_semmem_reembed_required() is not None
    except Exception:
        return False


def build_semmem_injection(
    state: AppState, cfg: dict, mode: str = "chat",
    conflict_ctx: ConflictTurnContext | None = None,
    query_vec=None,
    query_text: str = "",
    covered_attributes: set[str] | None = None,
    session_id: str | None = None,
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
    if _semmem_embeddings_are_stale():
        # embed モデルを替えた直後、ファクトのベクトルは旧モデル空間のまま
        # 残る。関連度ゲートはこれを 2 通りに壊す — 次元が違えば形の不一致で
        # 素通し (= 全店注入)、次元が同じで空間だけ違えばコサインが雑音に
        # なって全件却下 (実測 2026-09-01: 通過率 0.00%)。閾値の較正
        # (threshold_mode: auto) はスケールずれしか救えず、空間のずれには
        # 効かない。どちらへ倒れるか予測できない以上、再 embed が済むまでは
        # **注入しない** のが唯一の安全側 (履歴検索や RAG は従来どおり動く)。
        logger.warning(
            "semmem injection skipped: fact embeddings are stale after an "
            "embed model change; the relevance gate cannot be trusted. "
            "Run 'reembed-facts --apply' (or POST /api/model/reembed-facts).",
        )
        return None
    # このターンの MemoryInjector は 1 個。競合セクションのゲートも同じ
    # インスタンス・同じ事前計算スコアで判定する (別インスタンスで
    # スコア無しに判定し直すと、埋め込み無しのファクトが素通りしていた)。
    injector = None
    fact_scores: dict[str, float] = {}
    try:
        from backend.free.memory.pipeline.injector import MemoryInjector

        injector = MemoryInjector(cfg)
        _, stm, _ = mem_sys
        facts: list = []
        # 関連度スコアは埋め込み行列を持つストア側で 1 回の積として求める。
        # 候補ごとに正規化し直すと N=10000 で 52.5ms、常駐行列なら 1.5ms
        # (MemoryInjector._relevance_scores の実測)。
        pid = state.current_project_id
        for scope in ("global", f"project:{pid}" if pid else None):
            if scope is None:
                continue
            try:
                store = state.get_semantic_store(scope)
                facts.extend(store.all_facts(include_superseded=False))
            except Exception:
                continue
            if query_vec is not None:
                try:
                    fact_scores.update(store.embedding_scores(query_vec))
                except Exception:
                    # スコアが取れなければ候補ごとの判定へ縮退するだけ。
                    pass
        stm_notes = list(getattr(stm, "notes", {}).values())
        retired_note_ids = _collect_retired_note_ids(state, pid)
        if facts or stm_notes:
            plan = injector.inject(
                mode=inj_mode,
                facts=facts,
                stm_notes=stm_notes,
                current_project_id=pid,
                failure_signatures=(),
                query_embedding=query_vec,
                session_user_texts=_session_user_texts(state, session_id),
                query_text=query_text,
                retired_note_ids=retired_note_ids,
                fact_relevance_scores=fact_scores or None,
            )
            rendered = plan.render() or None
            if covered_attributes is not None:
                covered_attributes.update(plan.covered_attributes)
    except Exception as e:
        logger.warning("semmem injection skipped: %s", e)
        rendered = None

    conflict_block = _render_conflict_section(
        cfg, conflict_ctx, query_vec=query_vec,
        fact_scores=fact_scores or None, injector=injector,
    )
    if conflict_block:
        rendered = f"{rendered}\n\n{conflict_block}" if rendered else conflict_block
    return rendered


def _gate_groups_by_relevance(
    cfg: dict, groups: list, query_vec,
    fact_scores: dict[str, float] | None = None,
    injector=None,
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

    ``fact_scores`` は注入本体が使った事前計算スコア (``embedding_scores``)、
    ``injector`` は同ターンの ``MemoryInjector`` インスタンス。どちらも
    省略可 (テスト経路 / 旧呼出) だが、本番では両方渡す。埋め込みもスコアも
    無いファクトは **落とす** (``require_embedding=True``) — 予算の外で毎ターン
    連結される枠で「判定不能なので通す」を許すと、無関係な矛盾が全ターンに載る。
    """
    if not groups:
        return groups
    try:
        if injector is None:
            from backend.free.memory.pipeline.injector import MemoryInjector

            injector = MemoryInjector(cfg)
        facts = [f for g in groups for f in g.facts]
        relevant = injector.relevant_fact_ids(
            facts, query_vec, fact_scores, require_embedding=True,
        )
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
    fact_scores: dict[str, float] | None = None, injector=None,
) -> str | None:
    """pending 競合セクションを組み立てる。失敗時 / 該当なしは None。

    ``query_vec`` を渡すと関連度ゲートが掛かる
    (:func:`_gate_groups_by_relevance`)。``None`` はゲート無効 = 素通しで、
    embedder 自体が無い構成のための後方互換。``fact_scores`` / ``injector``
    は注入本体と同じ判定材料をゲートへ渡す。
    """
    if conflict_ctx is None or not conflict_ctx.pending_groups:
        return None
    try:
        from backend.free.memory.pipeline.conflict_review import (
            render_pending_conflicts_block,
        )

        groups = _gate_groups_by_relevance(
            cfg, conflict_ctx.pending_groups, query_vec,
            fact_scores, injector,
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


def session_evicted_turns(state: AppState, session_id: str | None = None) -> int:
    """``session_id`` のセッションでワーキングメモリから押し出したターン数を安全に取る。

    pillar 未構築 (degraded / テストの部分モック) では 0 を返す。
    """
    working = _session_wm(state, session_id)
    try:
        return int(getattr(working, "session_evicted_turns", 0) or 0)
    except (TypeError, ValueError):
        return 0


def session_first_user_message(
    state: AppState, session_id: str | None = None,
) -> str:
    """``session_id`` のセッションで最初に届いた user 発話を安全に取る。

    窓から押し出されても ``WorkingMemory`` が 1 件だけ保持している
    (``session_first_user_turn``)。「この会話で最初に何を言ったか」を
    押し出し後も決定論で答えるための材料。

    pillar 未構築 (degraded / テストの部分モック) では空文字列を返す。
    """
    working = _session_wm(state, session_id)
    head = getattr(working, "session_first_user_turn", "")
    return head if isinstance(head, str) else ""


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
    artifact_block: str | None = None,
    session_id: str = "",
    post_append_reserve_tokens: int = 0,
) -> list[ChatMessage]:
    """messages 組み立て（build_messages で few-shot・file・メモリ・RAG・履歴を統合）。

    ``system_prompt`` は静的 (query 非依存)、``fewshot_block`` 等の query 依存部は
    build_messages 内で最後の user メッセージへ前置される (KV キャッシュ対応)。

    Args:
        evicted_turns: 現在セッションでワーキングメモリから押し出したターン数
            (``WorkingMemory.session_evicted_turns``)。0 より大きい場合、会話
            全体を走査しないと答えられない質問には切り詰め注記を付ける。
        post_append_reserve_tokens: 組み立て後に最後の user へ積まれる分
            (接地注記 / deliberative のツール結果) の予約。主経路は
            :func:`deliberative_post_append_reserve_tokens`、ツール結果を持たない
            軽量 / 継続パスは :func:`notes_post_append_reserve_tokens`。
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
        artifact_block=artifact_block,
        post_append_reserve_tokens=post_append_reserve_tokens,
    )
    apply_grounding_notes(messages, history, evicted_turns, session_id)
    logger.debug("Messages assembled: %d messages for LLM", len(messages))
    return messages


#: 組み立て後に積まれる接地注記 (``apply_grounding_notes`` 4 種 + deliberative の
#: 決定論注記) の見込み (トークン)。1 ターンに載るのは高々数種で、各 100〜200
#: トークン。
_POST_APPEND_NOTES_ALLOWANCE_TOKENS = 256


def notes_post_append_reserve_tokens() -> int:
    """ツール結果を持たない経路 (軽量 / 継続) の予約 (トークン)。

    ``apply_grounding_notes`` は **全経路** で最後の user へ注記を積むので、
    ツール結果の分だけを落とした定額を予約する。
    """
    return _POST_APPEND_NOTES_ALLOWANCE_TOKENS


def deliberative_post_append_reserve_tokens() -> int:
    """deliberative 経路が組み立て後に最後の user へ積む分の見込み (トークン)。

    ツール結果は ``TOOL_RESULT_MAX_CHARS`` で切り詰められる。トークン換算は
    本文の構成で 4 倍違う (ASCII 4 文字 ≒ 1 tok / CJK 1 文字 ≒ 1 tok) ため、
    その中間 (文字数の 1/2) を取り、接地指示・話題再フォーカスと各種注記の
    定額を足す。実際の残予算に対する上限は ``build_messages`` 側
    (``_POST_APPEND_MAX_SHARE``) が掛ける。
    """
    from backend.free.api.chat.chat_constants import TOOL_RESULT_MAX_CHARS

    return TOOL_RESULT_MAX_CHARS // 2 + _POST_APPEND_NOTES_ALLOWANCE_TOKENS


# ── 注記のロケール辞書 (i18n.prompt_locale 追従) ─────────────────────
#
# chat.py の ``_REFERENCE_BLOCK_DIRECTIVES`` と同じ形。以前は日本語固定で、
# ``prompt_locale: en`` でも計測注記だけ日本語が混ざっていた。マーカー
# ``[システム計測]`` (``SYSTEM_MEASUREMENT_MARKER``) は両ロケール共通 —
# ``text_quality.extract_measured_values`` がプロンプトから実測値を読み戻す
# 契約になっているため、ロケールで変えない。


def _prompt_locale() -> str:
    """``i18n.prompt_locale`` を返す (``backend.i18n_helper.prompt_locale`` に委譲)。"""
    from backend.i18n_helper import prompt_locale

    return prompt_locale()


def _localized(table: dict[str, str]) -> str:
    return table.get(_prompt_locale(), table["ja"])


_QUANTITY_GROUNDING_GUIDANCE: dict[str, str] = {
    "ja": (
        "\n\n" + SYSTEM_MEASUREMENT_MARKER + " この会話で確定している値: {values}。"
        "「{quantity}」を使う計算では **この値** を式に入れること。"
        "直前の回答に出た別の数値で代用しないこと。"
    ),
    "en": (
        "\n\n" + SYSTEM_MEASUREMENT_MARKER + " Value established in this "
        "conversation: {values}. Use **this value** in any calculation "
        "involving \"{quantity}\"; do not substitute another number from the "
        "previous answer."
    ),
}


def _append_quantity_grounding(
    messages: list[ChatMessage], history: list[ChatMessage],
) -> None:
    """「同じ<量>を」が指す値を確定事実として渡す (in-place)。

    実インシデント (2026-08-27 ライブ監査 T12-4)::

        T12-1 「東京駅と横浜駅の直線距離はおよそ何kmですか。」→ 15km
        T12-2 「それを自転車で時速18kmで走ると何時間かかりますか。」
              → calculate(15 / 18) = 0.83 時間  ✓
        T12-4 「同じ距離を時速4.5kmで歩くとどうなりますか。」
              → calculate(0.83 * 4.5) = 3.735   ✗  (正: 15 / 4.5 = 3.33)

    「**同じ距離**を」と言っているのに、距離 (15) ではなく直前の時間 (0.83)
    を掴んだ。calculate は渡された式を正しく計算しており、誤りは **モデルが
    立てた式** の側。誤値は 4 ターン伝播して最終的な表にも残った。

    次元解析は入れない (単位の追跡は重く、扱えない単位で必ず穴が空く)。
    代わりに「その量の値はこれだ」を先に渡す — 式を立てる前に正しい値が
    目の前にあれば、直前の別の数値を掴む余地が減る。値が会話で確定して
    いなければ何もしない (**観測事実が無ければ注入しない**)。
    """
    query = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            query = str(msg.get("content") or "")
            break
    quantity = referenced_quantity(query)
    if not quantity:
        return
    # ``build_messages`` が supersede 判定のために同じ値を計算済み
    # (``BuiltMessages.numeric_claims``)。素の list なら再計算する。
    claims = getattr(messages, "numeric_claims", None)
    if claims is None:
        claims = conversational_numeric_claims(
            [(str(m.get("role") or ""), str(m.get("content") or "")) for m in history],
        )
    # ラベルは「東京駅と横浜駅の直線距離」のように修飾が付く。参照側は裸の
    # 「距離」なので、**ラベルが参照語を含む** ものを拾う。
    matched = {
        label: values for label, values in claims.items()
        if quantity in label and len(values) == 1
    }
    if not matched:
        return
    # 同じ値を指す複数のラベル (接尾辞違い) は 1 件に畳む。
    values = {next(iter(v)) for v in matched.values()}
    if len(values) != 1:
        # 候補が割れているなら黙る — どれを使うべきか決められない。
        return
    label = min(matched, key=len)
    value = next(iter(values))
    if append_to_last_user(
        messages,
        _localized(_QUANTITY_GROUNDING_GUIDANCE).format(
            values=f"{label} = {value}", quantity=quantity,
        ),
        separator="",
    ):
        logger.info(
            "Quantity grounding injected: %s = %s (referenced as %s)",
            label, value, quantity,
        )


_CONVERSATION_MEASUREMENT_GUIDANCE: dict[str, str] = {
    "ja": (
        "\n\n" + SYSTEM_MEASUREMENT_MARKER + " {values}。"
        "この数値をそのまま使って答えること。自分で数え直したり概算したりしないこと。"
    ),
    "en": (
        "\n\n" + SYSTEM_MEASUREMENT_MARKER + " {values}. "
        "Use these numbers as they are; do not recount or estimate yourself."
    ),
}
_CONVERSATION_TURN_COUNT_FACT: dict[str, str] = {
    "ja": "この会話の累計ターン数 (user + assistant) は {n} です",
    "en": "the total number of turns in this conversation (user + assistant) is {n}",
}
_CONVERSATION_TERM_COUNT_FACT: dict[str, str] = {
    "ja": "この会話の全ターン本文に「{term}」が現れた回数は {n} 回です",
    "en": "\"{term}\" appears {n} time(s) across every turn of this conversation",
}
_MEASUREMENT_JOINER: dict[str, str] = {"ja": "、", "en": "; "}


def _append_conversation_measurement(
    messages: list[ChatMessage], history: list[ChatMessage], session_id: str,
) -> None:
    """会話全体を対象にした計量の問いへ実測値を添える (in-place)。

    ``_append_self_output_measurement`` は **直前の自分の出力** を測る。
    こちらは **会話全体** で、窓に入っている分しか見えないモデルには原理的に
    数えられない:

    - 「この会話はいま何ターン目ですか。」→ 148 ターン目に「50ターン目です」
      (2026-08-27 ライブ監査 T19-4。窓の中だけを数えていた)
    - 「これまでの会話に「横浜」は何回出てきましたか。」→「5回」(実際 4 回。
      同 T08-7。ツールを使わず数を断定した)

    真実は ``chat_recorder`` の蓄積バッファにある。数えるのはコードの仕事。

    契約: 蓄積バッファには **進行中の user 発話がすでに積まれている**
    (``prepare_memory_context`` → ``accumulate_user_turn`` が messages を組む
    前に走る)。したがってターン数も語の出現回数も現在のターンを含めて数え、
    ここで ``+1`` などの補正はしない (以前はターン数だけ +1 し、出現回数は
    現在の発話を含まない、と契約が食い違っていた)。
    """
    if not session_id:
        return
    query = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            query = str(msg.get("content") or "")
            break
    if not query:
        return

    from backend.free.api.chat.chat_recorder import (
        count_term_in_session,
        session_turn_count,
    )

    parts: list[str] = []
    if conversation_turn_count_question(query):
        parts.append(
            _localized(_CONVERSATION_TURN_COUNT_FACT).format(
                n=session_turn_count(session_id),
            ),
        )
    term = occurrence_count_term(query)
    if term:
        parts.append(
            _localized(_CONVERSATION_TERM_COUNT_FACT).format(
                term=term, n=count_term_in_session(session_id, term),
            ),
        )
    if not parts:
        return
    if append_to_last_user(
        messages,
        _localized(_CONVERSATION_MEASUREMENT_GUIDANCE).format(
            values=_localized(_MEASUREMENT_JOINER).join(parts),
        ),
        separator="",
    ):
        logger.debug("Conversation measurement injected: %s", parts)


def apply_grounding_notes(
    messages: list[ChatMessage],
    history: list[ChatMessage],
    evicted_turns: int,
    session_id: str = "",
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
    _append_quantity_grounding(messages, history)
    _append_conversation_measurement(messages, history, session_id)


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
_TRUNCATED_HISTORY_GUIDANCE: dict[str, str] = {
    "ja": (
        "\n\nこの質問は会話全体を見ないと正確に答えられないが、"
        "この会話の前半 {n} 件のやり取りは文脈の上限を超えたため、"
        "今のあなたには見えていない。"
        "見えている範囲について答えたうえで、"
        "「会話の前半は参照できないため、これより前にもあった可能性がある」"
        "と明示すること。"
        "見えていない範囲について「無い」「一度も〜していない」と断定しないこと。"
    ),
    "en": (
        "\n\nThis question cannot be answered accurately without the whole "
        "conversation, but the first {n} exchanges of this conversation exceeded "
        "the context limit and are not visible to you now. Answer for the part "
        "you can see, then state explicitly that the earlier part of the "
        "conversation is unavailable and there may have been more before it. "
        "Do not assert that something \"never happened\" or \"does not exist\" "
        "in the part you cannot see."
    ),
}


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
        messages, _localized(_TRUNCATED_HISTORY_GUIDANCE).format(n=evicted_turns),
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
#: 計量結果の文面 ({kind: {locale: template}})。単位語 (文字 / 行 / 語) は
#: ``text_quality._MEASURED_VALUE_RE`` が読み戻す契約なので ja は変えない。
_SELF_OUTPUT_MEASURE_FORMATS: dict[str, dict[str, str]] = {
    "chars": {
        "ja": "文字数 {n} 文字 (空白・改行を除くと {stripped} 文字)",
        "en": "{n} characters ({stripped} excluding whitespace and line breaks)",
    },
    "lines": {"ja": "行数 {n} 行", "en": "{n} lines"},
    "words": {"ja": "単語数 {n} 語", "en": "{n} words"},
}
_SELF_OUTPUT_MEASUREMENT_GUIDANCE: dict[str, str] = {
    "ja": (
        "\n\n" + SYSTEM_MEASUREMENT_MARKER + " 直前のあなたの回答を機械的に数えた結果: {values}。"
        "この数値をそのまま使って答えること。自分で数え直したり概算したりしないこと。"
    ),
    "en": (
        "\n\n" + SYSTEM_MEASUREMENT_MARKER + " Your previous answer, counted "
        "mechanically: {values}. Use these numbers as they are; do not recount "
        "or estimate yourself."
    ),
}


def _measure_text(text: str, kinds: tuple[str, ...]) -> list[str]:
    """直前出力の計量結果を人間可読な文字列リストにする (純粋関数)。"""
    parts: list[str] = []
    for kind in kinds:
        template = _localized(_SELF_OUTPUT_MEASURE_FORMATS[kind])
        if kind == "chars":
            stripped = "".join(text.split())
            parts.append(template.format(n=len(text), stripped=len(stripped)))
        elif kind == "lines":
            # 「行」の定義は開示側 (text_quality.count_response_lines) と揃える。
            # splitlines() は空行とシステムの開示注記まで数えるため、両者が
            # 食い違っていた。実測 (2026-08-27 の WS6 検証): 1 行の回答 +
            # 空行 + 注記 を 3 行と数えて渡し、モデルは忠実に「3行でした」と
            # 答えた (ユーザーから見た本文は 1 行)。
            parts.append(template.format(n=count_response_lines(text)))
        else:
            parts.append(template.format(n=len(text.split())))
    return parts


#: コードフェンス。計量対象を「直近のコード成果物」に絞るときの目印。
_CODE_FENCE_BLOCK_RE = re.compile(r"```")


def _measurement_target(history: list[ChatMessage], query: str) -> str:
    """計量の対象になる自分の出力を選ぶ (純粋関数)。

    従来は **直前の assistant メッセージ全文** に固定していた。そのため
    対象を取り違える:

        T13-7 「保存したファイルを読んで」→「ファイルの内容を確認できません。」
        T13-8 「いま書いたコードは何行ですか。」→「1行です」

    実ファイルは 26 行。機構は直前の失敗通知 (1 行) を **忠実に測って**
    答えていた。「いま書いたコード」が指すのは直前の発話ではなく、直近の
    コード成果物。

    クエリがコードを名指ししていれば、コードフェンスを含む直近の assistant
    発話を選ぶ。無ければ従来どおり直前の発話に落ちる (退行しない)。
    どちらの場合もシステムの開示注記は除く (``text_quality.strip_system_notes``、
    記録側 ``chat_recorder._recorded_body`` と同じ関数)。
    """
    assistants = [
        str(m.get("content") or "")
        for m in history if m.get("role") == "assistant"
    ]
    if not assistants:
        return ""
    if _PRIOR_OUTPUT_CODE_RE.search(query or ""):
        for text in reversed(assistants):
            if _CODE_FENCE_BLOCK_RE.search(text):
                return strip_system_notes(text).strip()
    return strip_system_notes(assistants[-1]).strip()


_USER_TEXT_MEASUREMENT_GUIDANCE: dict[str, str] = {
    "ja": (
        "\n\n" + SYSTEM_MEASUREMENT_MARKER + " あなたが示した文章を機械的に数えた結果: {values}。"
        "この数値をそのまま使って答えること。自分で数え直したり概算したりしないこと。"
    ),
    "en": (
        "\n\n" + SYSTEM_MEASUREMENT_MARKER + " The text you provided, counted "
        "mechanically: {values}. Use these numbers as they are; do not recount "
        "or estimate yourself."
    ),
}


def _append_self_output_measurement(
    messages: list[ChatMessage], history: list[ChatMessage],
) -> None:
    """「今の回答は何文字?」に実測値を添える (in-place)。

    計量対象が無い / 計量質問でない場合は何もしない。

    ユーザーが **同じ発話の中で示した文章** についての計量
    (「<本文> ← これは何文字ですか？」) はそちらを優先する。指示対象が
    「自分の直前の出力」ではなく「いま渡された文章」で別物だから。実インシデント
    (2026-08-31 ライブ監査 t18#9): 1213 文字の入力に **「500文字です」**
    (入力は欠損なく届いており、単に数え違えていた)。
    """
    query = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            query = str(msg.get("content") or "")
            break
    payload, payload_kinds = split_user_text_measurement(query)
    joiner = _localized(_MEASUREMENT_JOINER)
    if payload and payload_kinds:
        values = joiner.join(_measure_text(payload, payload_kinds))
        if append_to_last_user(
            messages,
            _localized(_USER_TEXT_MEASUREMENT_GUIDANCE).format(values=values),
            separator="",
        ):
            logger.debug("User-text measurement injected: %s", values)
        return
    kinds = self_output_measure_kinds(query)
    if not kinds:
        return
    previous = _measurement_target(history, query)
    if not previous.strip():
        return
    values = joiner.join(_measure_text(previous, kinds))
    if append_to_last_user(
        messages,
        _localized(_SELF_OUTPUT_MEASUREMENT_GUIDANCE).format(values=values),
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
