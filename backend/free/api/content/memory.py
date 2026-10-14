"""メモリ API"""

from typing import get_args

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app_state import AppState, get_app_state
from backend.free.api._error_responses import api_error
from backend.free.api.schemas import (
    LongTermMemoryStats,
    MemoryDetailedStats,
    MemoryFactDetailResponse,
    MemoryFactInfo,
    MemoryFactNoteInfo,
    MemoryFactsResponse,
    MemoryNotesResponse,
    NoteInfo,
    PinFactRequest,
    PinFactResponse,
    PinnedFactInfo,
    PinnedFactsResponse,
    SemanticMemoryScopeStats,
    SemanticMemoryStats,
    ShortTermMemoryStats,
    UnpinFactRequest,
    UnpinFactResponse,
    WorkingMemoryStats,
)
from backend.config import get_config
from backend.free.api.chat.chat_constants import (
    DEFAULT_WORKING_MAX_TURNS,
)
from backend.free.core.prompt_budget import resolve_working_max_tokens
from backend.free.core.session_mode import canonicalize_session_mode
from backend.free.memory.semantic.pin_manager import (
    PinLockedError,
    list_pinned,
    pin_fact,
    unpin_fact,
)
from backend.free.memory.semantic.store import ScopedSemanticStore
from backend.free.memory.pipeline.stats_calculator import (
    compute_ltm_stats,
    compute_stm_stats,
)
from backend.free.memory.ownership import FACT_OWNERSHIP
from backend.free.memory.views.base import (
    SubjectNamespaceError,
    WriteOwnershipError,
)
from backend.free.memory.views.mem import MemFactView
from backend.free.memory.types import FactType, SemanticFact, make_fact
from backend.log_config import get_logger
from backend.structlog_config import redact_string
from backend.utils import epoch_to_utc

logger = get_logger("api.memory")

router = APIRouter(prefix="/api/memory", tags=["memory"])


# 受け入れ可能な FactType (SemanticFact.type に渡せる文字列)。
# FactType Literal と同期
# FactType Literal から直接導出して set のドリフトを防ぐ
# (旧ハードコードは learned_failure_pattern を取りこぼしていた)。
_KNOWN_FACT_TYPES: frozenset[str] = frozenset(get_args(FactType))

#: 本 API から **新規作成** できる FactType。mem pillar が owner のものだけ。
#:
#: 以前は ``_KNOWN_FACT_TYPES`` (= 全 FactType) をそのまま受け、しかも
#: ``MemFactView`` を経由せず ``store.add_fact()`` を直接呼んでいたため、
#: ``type: "policy"`` (learn 所有) や ``subject: "loop.task.x"`` が素通りした
#: (2026-09-01 監査で再現)。docs/f_02_memory_system.md §12 の禁則 3 / 4 が
#: この 1 経路だけ効いていなかった。混入した learn 相当のファクトは
#: PolicyInterpreter / HarnessFactView の読み出しに乗りうる。
#:
#: 検証の本体は :class:`MemFactView` (``_assert_write`` / ``_assert_subject_owner``)
#: に委ねる。ここは 400 と 403 を区別するための事前チェックにすぎない。
_ALLOWED_FACT_TYPES: frozenset[str] = frozenset(
    t for t in _KNOWN_FACT_TYPES if FACT_OWNERSHIP[t].owner == "mem"  # type: ignore[index]
)


def _resolve_store(state: AppState, scope: str) -> ScopedSemanticStore:
    """scope 文字列を検証し、スコープ束縛の SemMem ビューを返す"""
    if not scope or (scope != "global" and not scope.startswith("project:")):
        raise HTTPException(
            status_code=400,
            detail=f"invalid scope: {scope!r}",
        )
    if scope.startswith("project:") and len(scope) <= len("project:"):
        raise HTTPException(status_code=400, detail="empty project id in scope")
    try:
        return state.get_semantic_store(scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _to_pinned_info(fact: SemanticFact) -> PinnedFactInfo:
    return PinnedFactInfo(
        id=fact.id,
        subject=fact.subject,
        predicate=fact.predicate,
        object=fact.object,
        type=fact.type,
        scope=fact.scope,
        confidence=fact.confidence,
        pinned=fact.pinned,
        pin_locked_until=fact.pin_locked_until,
        mode_origin=fact.mode_origin,
        created_at=fact.created_at,
        accessed_at=fact.accessed_at,
    )


def _compute_semantic_stats(state: AppState) -> SemanticMemoryStats:
    """スコープ別に SemMem を集計する。

    実体は 1 ストアなので (c_16 §4.2)、ここに現れるのは **これまでに参照された
    スコープ** のビューだけ。Pin API / facts API がアクセスした時点で増える。
    """
    scopes: list[SemanticMemoryScopeStats] = []
    total_facts = 0
    total_pinned = 0
    for scope, store in sorted(state._semantic_stores.items()):
        all_facts = store.all_facts(include_superseded=True)
        active = [f for f in all_facts if not f.superseded_by]
        pinned = [f for f in all_facts if f.pinned]
        by_type: dict[str, int] = {}
        by_mode: dict[str, int] = {}
        for f in active:
            by_type[f.type] = by_type.get(f.type, 0) + 1
            by_mode[f.mode_origin] = by_mode.get(f.mode_origin, 0) + 1
        scope_stats = SemanticMemoryScopeStats(
            scope=scope,
            total=len(all_facts),
            active=len(active),
            superseded=len(all_facts) - len(active),
            pinned=len(pinned),
            by_type=by_type,
            by_mode_origin=by_mode,
        )
        scopes.append(scope_stats)
        total_facts += len(all_facts)
        total_pinned += len(pinned)
    return SemanticMemoryStats(
        scopes=scopes,
        total_facts=total_facts,
        total_pinned=total_pinned,
    )


@router.get("/stats", response_model=MemoryDetailedStats)
async def get_memory_stats(state: AppState = Depends(get_app_state)):
    """メモリシステム全体の統計"""
    logger.debug("GET /api/memory/stats")
    cfg = get_config()
    mem_cfg = cfg.get("memory", {})
    mem_sys = state.get_memory_system()
    semantic_stats = _compute_semantic_stats(state)
    current_mode = getattr(state, "current_mode", "chat") or "chat"

    if mem_sys is None:
        return MemoryDetailedStats(
            working=WorkingMemoryStats(
                turns=0,
                max_turns=mem_cfg.get("working_max_turns", DEFAULT_WORKING_MAX_TURNS),
                tokens_used=0,
                max_tokens=resolve_working_max_tokens(cfg),
                session_id="",
            ),
            short_term=ShortTermMemoryStats(
                notes=0, max_notes=mem_cfg.get("short_term_max_notes", 100),
                pending_embeddings=0, pending_evolution=0,
            ),
            long_term=LongTermMemoryStats(chunks=0, index_size_mb=0.0, sources=0),
            semantic=semantic_stats,
            current_mode=current_mode,
        )

    wm, episodic = mem_sys

    # ``short`` / ``long`` は同じストアの tier (c_16 §4.1)。レスポンスの形は
    # 既存 UI と揃えたまま、件数の出所だけをエピソード記憶へ移した。
    short_notes = episodic.short_notes() if episodic is not None else []
    long_notes = episodic.iter_notes(tier="long") if episodic is not None else []
    stm_stats = compute_stm_stats(short_notes)
    index_path = None
    if episodic is not None:
        vectors = episodic.evidence.vector_store()
        index_path = getattr(vectors, "index_path", None)
    ltm_stats = compute_ltm_stats(
        chunks=len(long_notes),
        index_path=index_path,
        sources=len({n.session_id for n in long_notes if n.session_id}),
    )

    return MemoryDetailedStats(
        working=WorkingMemoryStats(
            turns=len(wm.turns),
            max_turns=wm.max_turns,
            tokens_used=wm._total_tokens(),
            max_tokens=wm.max_tokens,
            session_id=wm.session_id,
        ),
        short_term=ShortTermMemoryStats(
            notes=len(short_notes),
            max_notes=mem_cfg.get("short_term_max_notes", 100),
            **stm_stats,
        ),
        long_term=LongTermMemoryStats(**ltm_stats),
        semantic=semantic_stats,
        current_mode=current_mode,
    )


@router.get("/notes", response_model=MemoryNotesResponse)
async def get_memory_notes(
    state: AppState = Depends(get_app_state),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort: str = Query("score"),
    tag: str | None = Query(None),
):
    """Layer 2 短期記憶のノート一覧"""
    logger.debug(
        "GET /api/memory/notes: limit=%d, offset=%d, sort=%s, tag=%s",
        limit, offset, sort, tag,
    )
    episodic = state.episodic_memory
    if episodic is None:
        return MemoryNotesResponse(total=0, notes=[])
    notes = list(episodic.short_notes())

    # タグフィルタ
    if tag:
        notes = [n for n in notes if tag in n.tags]

    # ソート (``score`` は廃止した LightMem スコアの名残なので新しい順に倒す)
    if sort in ("score", "created"):
        notes.sort(key=lambda n: -n.created_at)
    elif sort == "accessed":
        notes.sort(key=lambda n: -n.accessed_at)

    total = len(notes)
    notes = notes[offset:offset + limit]

    return MemoryNotesResponse(
        total=total,
        notes=[
            NoteInfo(
                id=n.id,
                content=n.content,
                keywords=n.keywords,
                tags=n.tags,
                created_at=n.created_at,
                accessed_at=n.accessed_at,
                session_id=n.session_id,
                context_description=n.context_description,
                evolution_pending=n.evolution_pending,
                has_embedding=n.embedding is not None,
            )
            for n in notes
        ],
    )


# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────


@router.post("/pin", response_model=PinFactResponse)
async def pin_memory(
    req: PinFactRequest,
    state: AppState = Depends(get_app_state),
):
    """SemanticFact を pin する。

    - ``fact_id`` を渡すと既存ファクトを pin する
    - ``content`` (または ``object``) を渡すと新規ファクトを生成して pin する
    - ``scope`` は ``"global"`` または ``"project:<id>"``
    - ``lock_duration_s`` を指定すると ``pin_locked_until`` が設定される
    """
    logger.debug("POST /api/memory/pin: scope=%s id=%s", req.scope, req.fact_id)
    store = _resolve_store(state, req.scope)

    if req.type not in _KNOWN_FACT_TYPES:
        raise HTTPException(status_code=400, detail=f"unknown type: {req.type}")
    if req.type not in _ALLOWED_FACT_TYPES:
        # loop / learn 所有の型はこの API から作れない (owner pillar が書く)。
        raise HTTPException(
            status_code=403,
            detail=(
                f"fact type {req.type!r} is owned by pillar "
                f"{FACT_OWNERSHIP[req.type].owner!r}; "  # type: ignore[index]
                "the memory API can only create mem-owned facts"
            ),
        )
    canonical_origin = canonicalize_session_mode(req.mode_origin)
    if canonical_origin is None:
        raise HTTPException(
            status_code=400, detail=f"unknown mode_origin: {req.mode_origin}",
        )
    req.mode_origin = canonical_origin

    if req.fact_id:
        try:
            updated = pin_fact(
                store, req.fact_id, lock_duration_s=req.lock_duration_s,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return PinFactResponse(fact=_to_pinned_info(updated))

    obj = req.object_ if req.object_ is not None else req.content
    if obj is None:
        raise HTTPException(
            status_code=400,
            detail="either fact_id or content/object must be provided",
        )
    # subject はストア境界で正規化する。SubjectKey.parse は前後空白を strip して
    # pillar を判定するため、" mem.user" のような raw subject をそのまま保存すると
    # by_pillar バケットには入るが search_by_subject / search_by_pillar_prefix の
    # raw 文字列一致では永久に不可視になる。抽出経路 (SubjectCanonicalizer) と
    # 違い API 直書きは正規化を欠くため、ここで最低限 strip して整合させる。
    subject = req.subject.strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subject must be non-empty")
    new_fact = make_fact(
        subject=subject,
        predicate=req.predicate,
        object_=obj,
        type=req.type,  # type: ignore[arg-type]
        scope=req.scope,
        mode_origin=req.mode_origin,  # type: ignore[arg-type]
        confidence=1.0,
    )
    # 書込は MemFactView 経由 — ownership と subject namespace の検証点は
    # View にしかない。以前はここが store 直呼びで、``type: "policy"`` +
    # ``subject: "loop.task.x"`` が検証なしに書き込めた (2026-09-01 監査)。
    try:
        added = MemFactView(store).add_fact(new_fact)
    except WriteOwnershipError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except SubjectNamespaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pinned = pin_fact(store, added.id, lock_duration_s=req.lock_duration_s)
    return PinFactResponse(fact=_to_pinned_info(pinned))


@router.post("/unpin", response_model=UnpinFactResponse)
async def unpin_memory(
    req: UnpinFactRequest,
    state: AppState = Depends(get_app_state),
):
    """SemanticFact の pin を解除する。

    ``pin_locked_until`` が未来の場合は ``409 Conflict`` を返す
    (``force=True`` で上書き可)。
    """
    logger.debug("POST /api/memory/unpin: scope=%s id=%s", req.scope, req.fact_id)
    store = _resolve_store(state, req.scope)
    try:
        updated = unpin_fact(store, req.fact_id, force=req.force)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PinLockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return UnpinFactResponse(fact=_to_pinned_info(updated))


@router.get("/pinned", response_model=PinnedFactsResponse)
async def list_pinned_memory(
    state: AppState = Depends(get_app_state),
    scope: str = Query("global"),
):
    """pinned ファクトの一覧を返す"""
    logger.debug("GET /api/memory/pinned: scope=%s", scope)
    store = _resolve_store(state, scope)
    facts = list_pinned(store)
    return PinnedFactsResponse(
        scope=scope,
        total=len(facts),
        facts=[_to_pinned_info(f) for f in facts],
    )


# ──────────────────────────────────────────────────────────────────────────
# ファクトの一覧・詳細 (evoref forget --id の ID を探す、docs/f_02 §9)
# ──────────────────────────────────────────────────────────────────────────

#: ``status`` の絞り込みに使える値 (``all`` は絞らない)。取り下げ済みはストアが読まないので載らない。
FACT_STATUSES: tuple[str, ...] = ("live", "disputed", "superseded")
_FACT_STATUS_FILTERS: tuple[str, ...] = (*FACT_STATUSES, "all")
#: 根拠ノートの要約の長さ (字)
_NOTE_PREVIEW_CHARS = 80


def _facts_view(state: AppState, scope: str) -> MemFactView:
    """scope を検証し、そのスコープの SemMem を読む Fact View を返す (読むだけ)。"""
    if scope != "global" and not (scope.startswith("project:") and len(scope) > len("project:")):
        raise api_error(
            400, "E0400", f"invalid scope: {scope!r}", "api.memory_invalid_scope", scope=scope,
        )
    try:
        return MemFactView(state.get_semantic_store(scope))
    except RuntimeError as exc:
        raise api_error(
            503, "E0503", "Semantic memory is not initialized", "api.memory_unavailable",
        ) from exc


def _fact_status(fact: SemanticFact) -> str:
    if fact.superseded_by:
        return "superseded"
    return "disputed" if fact.veracity == "disputed" else "live"


def _masked(text: str) -> tuple[str, bool]:
    """秘密らしき文字列を伏せる (``evoref export`` と同じ ``redact_string``)。"""
    masked = redact_string(text)
    return masked, masked != text


def _is_multi_valued(subject: str) -> bool:
    # forget の dry-run (forget_command.plan_forget) と同じ判定
    from backend.free.memory.notes.note_builder import (
        is_multi_valued_subject,
        is_span_only_fold_subject,
    )

    return is_multi_valued_subject(subject) or is_span_only_fold_subject(subject)


def _unique(values) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def _to_fact_info(fact: SemanticFact) -> MemoryFactInfo:
    value: str | None = None
    redacted = False
    if not fact.private:
        value, redacted = _masked(fact.text or fact.object or "")
    provenances = list(fact.provenances or ())
    return MemoryFactInfo(
        id=fact.id,
        subject=fact.subject,
        predicate=fact.predicate,
        value=value,
        type=str(fact.type),
        scope=fact.scope,
        status=_fact_status(fact),
        veracity=str(fact.veracity),
        origin=str(fact.origin),
        confidence=float(fact.confidence),
        pinned=bool(fact.pinned),
        private=bool(fact.private),
        redacted=redacted,
        multi_valued=_is_multi_valued(fact.subject),
        superseded_by=fact.superseded_by,
        created_at=epoch_to_utc(fact.created_at),
        accessed_at=epoch_to_utc(fact.accessed_at),
        note_ids=_unique(p.note_id for p in provenances),
        session_ids=_unique([*(p.session_id for p in provenances), *sorted(fact.session_ids or ())]),
        extractors=_unique(p.extractor for p in provenances),
    )


def _matches_query(fact: SemanticFact, needle: str) -> bool:
    """語彙の部分一致 (大小文字を区別しない)。private の本文は照合しない (本文を返さないのと同じ理由)。"""
    fields = [fact.id, fact.subject, fact.predicate]
    if not fact.private:
        fields.append(_masked(fact.text or fact.object or "")[0])
    return any(needle in f.casefold() for f in fields if f)


@router.get("/facts", response_model=MemoryFactsResponse)
async def list_memory_facts(
    state: AppState = Depends(get_app_state),
    scope: str = Query("global"),
    subject: str | None = Query(None),
    query: str | None = Query(None),
    status: str = Query("all"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """意味記憶のファクトの一覧・検索 (読むだけ)。

    ``subject`` は前方一致、``query`` は id・subject・predicate・本文の部分一致
    (大小文字を区別しない、埋め込みは使わない)。新しい順 (発話時刻)。
    """
    logger.debug(
        "GET /api/memory/facts: scope=%s status=%s limit=%d offset=%d", scope, status, limit, offset,
    )
    if status not in _FACT_STATUS_FILTERS:
        raise api_error(
            400, "E0400", f"unknown status: {status}", "api.memory_unknown_fact_status",
            status=status, allowed=", ".join(_FACT_STATUS_FILTERS),
        )
    view = _facts_view(state, scope)
    facts = view.all_facts(include_superseded=True)
    if status != "all":
        facts = [f for f in facts if _fact_status(f) == status]
    if subject:
        facts = [f for f in facts if f.subject.startswith(subject)]
    needle = (query or "").strip().casefold()
    if needle:
        facts = [f for f in facts if _matches_query(f, needle)]
    facts.sort(key=lambda f: (-f.created_at, f.id))
    return MemoryFactsResponse(
        scope=scope,
        status=status,
        total=len(facts),
        limit=limit,
        offset=offset,
        facts=[_to_fact_info(f) for f in facts[offset:offset + limit]],
    )


def _note_info(episodic, note_id: str) -> MemoryFactNoteInfo:
    record = episodic.get(note_id) if episodic is not None else None
    if record is None:
        return MemoryFactNoteInfo(id=note_id, state="missing", private=False, preview=None)
    private = bool(record.private)
    preview = None
    if not private:
        line = " ".join(str(record.text or "").split())
        preview = _masked(line[:_NOTE_PREVIEW_CHARS])[0]
    state = "retracted" if record.veracity == "retracted" else "active"
    return MemoryFactNoteInfo(id=note_id, state=state, private=private, preview=preview)


@router.get("/facts/{fact_id}", response_model=MemoryFactDetailResponse)
async def get_memory_fact(fact_id: str, state: AppState = Depends(get_app_state)):
    """ファクト 1 件の詳細 (読むだけ)。``evoref forget`` の dry-run と同じ材料を返す。

    - ``siblings``: 同じ subject・同じ scope の live の別ファクト (多値スロットの別の値)
    - ``revived_on_forget``: このファクトが置き換えた旧値 (取り下げると live に戻る)
    - ``notes``: 根拠ノート (provenance の ``note_id``) の状態と要約
    """
    logger.debug("GET /api/memory/facts/%s", fact_id)
    # id 引きはスコープで絞らない (ScopedSemanticStore.get_fact と同じ)
    fact = _facts_view(state, "global").get_fact(fact_id)
    if fact is None:
        raise api_error(
            404, "E0404", f"fact not found: {fact_id}", "api.memory_fact_not_found", fact_id=fact_id,
        )
    view = _facts_view(state, fact.scope)
    siblings = [
        f for f in view.search_by_subject(fact.subject, include_superseded=False) if f.id != fact.id
    ]
    revived = [f for f in view.all_facts(include_superseded=True) if f.superseded_by == fact.id]
    info = _to_fact_info(fact)
    return MemoryFactDetailResponse(
        fact=info,
        siblings=[_to_fact_info(f) for f in sorted(siblings, key=lambda f: (-f.created_at, f.id))],
        revived_on_forget=[_to_fact_info(f) for f in sorted(revived, key=lambda f: f.id)],
        notes=[_note_info(state.episodic_memory, note_id) for note_id in info.note_ids],
    )
