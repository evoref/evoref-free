"""メモリ API"""

from typing import get_args

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app_state import AppState, get_app_state
from backend.free.api.schemas import (
    FadeMemStats,
    LongTermMemoryStats,
    MemoryDetailedStats,
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
from backend.free.core.session_mode import is_valid_session_mode
from backend.free.memory.semantic.pin_manager import (
    PinLockedError,
    list_pinned,
    pin_fact,
    unpin_fact,
)
from backend.free.memory.semantic.store import SemanticFactStore
from backend.free.memory.pipeline.stats_calculator import (
    compute_ltm_stats,
    compute_stm_stats,
)
from backend.free.memory.types import FactType, SemanticFact, make_fact
from backend.log_config import get_logger

logger = get_logger("api.memory")

router = APIRouter(prefix="/api/memory", tags=["memory"])


# 受け入れ可能な FactType (SemanticFact.type に渡せる文字列)。
# FactType Literal と同期
# FactType Literal から直接導出して set のドリフトを防ぐ
# (旧ハードコードは learned_failure_pattern を取りこぼしていた)。
_ALLOWED_FACT_TYPES: frozenset[str] = frozenset(get_args(FactType))


def _resolve_store(state: AppState, scope: str) -> SemanticFactStore:
    """scope 文字列を検証し、対応する SemanticFactStore を返す"""
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
        access_count=fact.access_count,
    )


def _compute_semantic_stats(state: AppState) -> SemanticMemoryStats:
    """ロード済みの SemanticFactStore を集計する

    `AppState._semantic_stores` は lazy 初期化されるため、まだ参照されていない
    プロジェクトはここに含まれない (= "現在ロード中のもののみ" の集計)。
    Pin API / facts API がアクセスされたタイミングで自動的にロードされる。
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
                turns=0, max_turns=mem_cfg.get("working_max_turns", 10),
                tokens_used=0, max_tokens=mem_cfg.get("working_max_tokens", 2048),
                session_id="",
            ),
            short_term=ShortTermMemoryStats(
                notes=0, max_notes=mem_cfg.get("short_term_max_notes", 100),
                pending_embeddings=0, pending_evolution=0, avg_lightmem_score=0.0,
            ),
            long_term=LongTermMemoryStats(chunks=0, index_size_mb=0.0, sources=0),
            fadem=FadeMemStats(
                alpha=mem_cfg.get("fade_alpha", 0.4),
                beta=mem_cfg.get("fade_beta", 0.3),
                gamma=mem_cfg.get("fade_gamma", 0.3),
                threshold=mem_cfg.get("fade_threshold", 0.15),
            ),
            semantic=semantic_stats,
            current_mode=current_mode,
        )

    wm, stm, ltm = mem_sys

    # STM 統計 (純粋関数に委譲)
    stm_stats = compute_stm_stats(stm.notes.values())

    # LTM 統計 (純粋関数に委譲)
    ltm_stats = compute_ltm_stats(
        chunks=ltm.vectors.count if ltm else 0,
        index_path=ltm.vectors.index_path if ltm else None,
        metadata=ltm.vectors.metadata if ltm else None,
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
            notes=len(stm.notes),
            max_notes=stm.max_notes,
            **stm_stats,
        ),
        long_term=LongTermMemoryStats(**ltm_stats),
        fadem=FadeMemStats(
            alpha=mem_cfg.get("fade_alpha", 0.4),
            beta=mem_cfg.get("fade_beta", 0.3),
            gamma=mem_cfg.get("fade_gamma", 0.3),
            threshold=mem_cfg.get("fade_threshold", 0.15),
        ),
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
    mem_sys = state.get_memory_system()
    if mem_sys is None:
        return MemoryNotesResponse(total=0, notes=[])

    _, stm, _ = mem_sys
    notes = list(stm.notes.values())

    # タグフィルタ
    if tag:
        notes = [n for n in notes if tag in n.tags]

    # ソート
    if sort == "score":
        notes.sort(key=lambda n: -n.lightmem_score)
    elif sort == "created":
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
                lightmem_score=n.lightmem_score,
                created_at=n.created_at,
                accessed_at=n.accessed_at,
                access_count=n.access_count,
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

    if req.type not in _ALLOWED_FACT_TYPES:
        raise HTTPException(status_code=400, detail=f"unknown type: {req.type}")
    if not is_valid_session_mode(req.mode_origin):
        raise HTTPException(
            status_code=400, detail=f"unknown mode_origin: {req.mode_origin}",
        )

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
    added = store.add_fact(new_fact)
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
