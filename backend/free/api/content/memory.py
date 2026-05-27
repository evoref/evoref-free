"""メモリ API"""

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app_state import AppState, get_app_state
from backend.free.api.schemas import (
    ConflictGroupInfo,
    ConflictsResponse,
    FactInfo,
    FactsResponse,
    FadeMemStats,
    LongTermMemoryStats,
    MemoryDetailedStats,
    MemoryNotesResponse,
    NoteInfo,
    PinFactRequest,
    PinFactResponse,
    PinnedFactInfo,
    PinnedFactsResponse,
    ResolveConflictRequest,
    ResolveConflictResponse,
    SemanticMemoryScopeStats,
    SemanticMemoryStats,
    ShortTermMemoryStats,
    TaskFactInfo,
    TaskFactsResponse,
    UnpinFactRequest,
    UnpinFactResponse,
    WorkingMemoryStats,
)
from backend.config import get_config
from backend.free.memory.semantic.pin_manager import (
    PinLockedError,
    list_pinned,
    pin_fact,
    unpin_fact,
)
from backend.free.memory.semantic.store import SemanticFactStore
from backend.free.memory.pipeline.semantic_conflict_resolver import (
    CONFLICTS_PENDING_FILENAME,
    CONFLICTS_RESOLVED_FILENAME,
)
from backend.free.memory.pipeline.stats_calculator import (
    compute_ltm_stats,
    compute_stm_stats,
)
from backend.free.memory.types import SemanticFact, make_fact
from backend.log_config import get_logger

logger = get_logger("api.memory")

router = APIRouter(prefix="/api/memory", tags=["memory"])


# 受け入れ可能な FactType (SemanticFact.type に渡せる文字列)。
# FactType Literal と同期
_ALLOWED_FACT_TYPES: frozenset[str] = frozenset({
    "personal_fact", "world_fact", "preference", "emotion", "opinion",
    "belief", "decision", "commitment", "project", "policy", "fewshot",
    "failure_pattern", "progress_marker", "task", "coding_task",
    "artifact", "coding", "model",
})


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


def _to_fact_info(fact: SemanticFact) -> FactInfo:
    """SemanticFact → FactInfo"""
    return FactInfo(
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
        superseded_by=fact.superseded_by,
        supersedes=list(fact.supersedes),
        auto_evolved=fact.auto_evolved,
        failure_signature=fact.failure_signature,
        eval_metric=dict(fact.eval_metric) if fact.eval_metric is not None else None,
        trace_id=fact.trace_id,
        private=fact.private,
        requires_user_review=fact.requires_user_review,
        review_status=fact.review_status,
    )


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
    if req.mode_origin not in ("chat", "coding"):
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
    new_fact = make_fact(
        subject=req.subject,
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


# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────


def _filter_facts(
    facts: list[SemanticFact],
    *,
    type_: str | None,
    mode: str | None,
    pinned: bool | None,
) -> list[SemanticFact]:
    """インスペクタ用の post-fetch フィルタ。"""
    out = facts
    if type_ is not None:
        out = [f for f in out if f.type == type_]
    if mode is not None:
        out = [f for f in out if f.mode_origin == mode]
    if pinned is not None:
        out = [f for f in out if f.pinned == pinned]
    return out


@router.get("/facts", response_model=FactsResponse)
async def list_facts(
    state: AppState = Depends(get_app_state),
    scope: str = Query("global"),
    type: str | None = Query(None, description="fact type filter"),
    mode: str | None = Query(None, description="mode_origin filter (chat|coding)"),
    pinned: bool | None = Query(None),
    include_superseded: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort: str = Query("created", description="created|accessed|confidence"),
):
    """SemanticFact を多条件で一覧する

    Args:
        scope: ``global`` または ``project:<id>``
        type: ``personal_fact`` / ``policy`` / ``task`` などタイプ絞り込み
        mode: ``chat`` または ``coding`` (モード別タブ用)
        pinned: True なら pin 済みのみ、False なら pin 解除済みのみ
        include_superseded: True なら supersession 履歴も含める
        limit / offset: ページング (最大 500)
        sort: 並び替えキー
    """
    logger.debug(
        "GET /api/memory/facts: scope=%s type=%s mode=%s pinned=%s",
        scope, type, mode, pinned,
    )
    if type is not None and type not in _ALLOWED_FACT_TYPES:
        raise HTTPException(status_code=400, detail=f"unknown type: {type}")
    if mode is not None and mode not in ("chat", "coding"):
        raise HTTPException(
            status_code=400, detail=f"unknown mode: {mode}",
        )
    store = _resolve_store(state, scope)

    if type is not None:
        facts = store.search_by_type(
            type,  # type: ignore[arg-type]
            include_superseded=include_superseded,
        )
    else:
        facts = store.all_facts(include_superseded=include_superseded)
        if not include_superseded:
            facts = [f for f in facts if not f.superseded_by]

    facts = _filter_facts(facts, type_=None, mode=mode, pinned=pinned)

    if sort == "accessed":
        facts.sort(key=lambda f: -f.accessed_at)
    elif sort == "confidence":
        facts.sort(key=lambda f: -f.confidence)
    else:  # "created" / default
        facts.sort(key=lambda f: -f.created_at)

    total = len(facts)
    page = facts[offset:offset + limit]
    return FactsResponse(
        scope=scope,
        total=total,
        facts=[_to_fact_info(f) for f in page],
    )


@router.get("/policy", response_model=FactsResponse)
async def list_policy_facts(
    state: AppState = Depends(get_app_state),
    scope: str = Query("global"),
    include_superseded: bool = Query(True, description="進化履歴を含めるか"),
    only_active: bool = Query(False, description="active (confidence>=0.7) のみ"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """``policy`` 型ファクトの一覧

    PolicyEvolver / Few-shot プールが SemMem に書き戻す ``policy`` 型ファクトを
    一覧する。``include_superseded`` のデフォルトを True にしてあるのは、
    インスペクタで supersedes チェーン (進化履歴) を確認する用途を想定するため。
    """
    logger.debug(
        "GET /api/memory/policy: scope=%s only_active=%s", scope, only_active,
    )
    store = _resolve_store(state, scope)
    facts = store.search_by_type(
        "policy", include_superseded=include_superseded,
    )
    if only_active:
        facts = [f for f in facts if f.confidence >= 0.7 and not f.superseded_by]
    # supersession チェーンの根 (新しい順) を見やすくするため confidence 降順 +
    # created_at 降順
    facts.sort(key=lambda f: (-f.confidence, -f.created_at))
    total = len(facts)
    page = facts[offset:offset + limit]
    return FactsResponse(
        scope=scope,
        total=total,
        facts=[_to_fact_info(f) for f in page],
    )


@router.get("/tasks", response_model=TaskFactsResponse)
async def list_task_facts(
    state: AppState = Depends(get_app_state),
    scope: str = Query(..., description="project:<id> 必須 (task は project スコープ専用)"),
    status: str | None = Query(
        None, description="open|in_progress|done|failed",
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """``task`` 型ファクトの一覧

    自律ループの駆動源となる task ファクトを `TaskFactView` 互換の構造化形式で
    返す。``scope`` は ``project:<id>`` のみ受け付ける (グローバルタスクは
    EvorefMem 仕様で禁止されている)。
    """
    logger.debug(
        "GET /api/memory/tasks: scope=%s status=%s", scope, status,
    )
    if not scope.startswith("project:"):
        raise HTTPException(
            status_code=400,
            detail="task facts require a project: scope",
        )
    if status is not None and status not in (
        "open", "in_progress", "done", "failed",
    ):
        raise HTTPException(status_code=400, detail=f"unknown status: {status}")
    store = _resolve_store(state, scope)
    project_id = scope.split(":", 1)[1]

    # 遅延 import (loop モジュールは API 層から見ると周辺レイヤ扱い)
    from backend.free.loop.driver import (
        TaskFactDecodeError,
        decode_task_fact,
    )

    out: list[TaskFactInfo] = []
    for fact in store.search_by_type("task"):
        if fact.scope != scope:
            continue
        try:
            view = decode_task_fact(fact)
        except TaskFactDecodeError as exc:
            logger.warning(
                "list_task_facts: skipping malformed task fact %s: %s",
                fact.id, exc,
            )
            continue
        if status is not None and view.status != status:
            continue
        out.append(
            TaskFactInfo(
                fact_id=view.fact_id,
                task_id=view.task_id,
                project_id=view.project_id,
                title=view.title,
                description=view.description,
                depends_on=list(view.depends_on),
                salience=view.salience,
                status=view.status,
                source_path=view.source_path,
                created_at=view.created_at,
                accessed_at=view.accessed_at,
                access_count=fact.access_count,
            ),
        )

    # salience 降順 + created 昇順 (pick_next_task と同じ順)
    out.sort(key=lambda t: (-t.salience, t.created_at, t.task_id))
    total = len(out)
    page = out[offset:offset + limit]
    # project_id を返却に活用 (ログ用)
    logger.debug(
        "list_task_facts: project=%s total=%d", project_id, total,
    )
    return TaskFactsResponse(scope=scope, total=total, tasks=page)


# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────


_RESOLVED_ACTION_TO_STATUS: dict[str, str] = {
    "keep_old": "resolved_keep_old",
    "keep_new": "resolved_keep_new",
    "merge": "resolved_merged",
}


def _group_pending_conflicts(
    store: SemanticFactStore, scope: str,
) -> list[ConflictGroupInfo]:
    """``review_status="pending"`` の active ファクトを (subject, predicate, type)
    でグルーピングして返す。"""
    pending = [
        f for f in store.all_facts(include_superseded=False)
        if f.review_status == "pending"
    ]
    buckets: dict[tuple[str, str, str], list[SemanticFact]] = {}
    for f in pending:
        buckets.setdefault((f.subject, f.predicate, f.type), []).append(f)

    groups: list[ConflictGroupInfo] = []
    for (subject, predicate, type_), facts in buckets.items():
        if len(facts) < 2:
            # 単独 pending は表示しない (グループとして競合になっていない)
            continue
        facts.sort(key=lambda f: f.created_at)
        winner = facts[-1]
        losers = facts[:-1]
        groups.append(ConflictGroupInfo(
            scope=scope,
            subject=subject,
            predicate=predicate,
            type=type_,
            facts=[_to_fact_info(f) for f in facts],
            decision="pending",
            winner_id=winner.id,
            loser_ids=[f.id for f in losers],
        ))
    # 新しい競合 (winner.created_at が新しい順) を上に
    groups.sort(
        key=lambda g: -(max(f.created_at for f in g.facts) if g.facts else 0.0),
    )
    return groups


def _read_resolved_history(
    store: SemanticFactStore, scope: str, *, limit: int,
) -> list[ConflictGroupInfo]:
    """``conflicts_resolved.jsonl`` を末尾から最大 ``limit`` 件読み込んで
    `ConflictGroupInfo` 一覧を返す。

    ファイル各行は :func:`SemanticConflictResolver._write_jsonl` で書かれた
    スキーマに準拠する。読み込み時、対応する SemanticFact が現存していれば
    `facts` フィールドに展開する。
    """
    path = store.root_dir / CONFLICTS_RESOLVED_FILENAME
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("read conflicts_resolved failed: %s", exc)
        return []

    history: list[ConflictGroupInfo] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        winner_id = entry.get("winner_id")
        loser_ids = list(entry.get("loser_ids") or [])
        ids = [winner_id, *loser_ids] if winner_id else list(loser_ids)
        facts: list[FactInfo] = []
        for fid in ids:
            if not fid:
                continue
            f = store.get_fact(fid)
            if f is not None:
                facts.append(_to_fact_info(f))
        history.append(ConflictGroupInfo(
            scope=scope,
            subject=str(entry.get("subject", "")),
            predicate=str(entry.get("predicate", "")),
            type=str(entry.get("type", "")),
            facts=facts,
            detected_at=entry.get("ts"),
            decision=entry.get("decision"),
            reason=entry.get("reason"),
            winner_id=winner_id,
            loser_ids=loser_ids,
        ))
    history.reverse()  # 新しい順
    return history


def _append_resolution_log(
    store: SemanticFactStore,
    *,
    scope: str,
    winner: SemanticFact,
    loser_ids: list[str],
    action: str,
    new_fact_id: str | None,
) -> None:
    """ユーザー解消を ``conflicts_resolved.jsonl`` に追記する (audit trail)。"""
    path = store.root_dir / CONFLICTS_RESOLVED_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "scope": scope,
        "subject": winner.subject,
        "predicate": winner.predicate,
        "type": winner.type,
        "winner_id": new_fact_id or winner.id,
        "loser_ids": loser_ids if not new_fact_id else [winner.id, *loser_ids],
        "decision": "user",
        "reason": f"user_{action}",
    }
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _remove_pending_lines(
    store: SemanticFactStore, *, fact_ids: set[str],
) -> None:
    """``conflicts.jsonl`` から ``fact_ids`` を含むエントリを削除する。

    ファイル全体を書き換える (規模的に十分実用範囲)。
    """
    path = store.root_dir / CONFLICTS_PENDING_FILENAME
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("read conflicts.jsonl failed: %s", exc)
        return
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            entry = json.loads(s)
        except json.JSONDecodeError:
            kept.append(s)
            continue
        ids = {entry.get("winner_id"), *(entry.get("loser_ids") or [])}
        if ids & fact_ids:
            continue
        kept.append(s)
    path.write_text(
        ("\n".join(kept) + "\n") if kept else "",
        encoding="utf-8",
    )


@router.get("/conflicts", response_model=ConflictsResponse)
async def list_conflicts(
    state: AppState = Depends(get_app_state),
    scope: str = Query("global"),
    history_limit: int = Query(50, ge=1, le=500),
):
    """SemMem 競合一覧を返す

    - ``pending``: ``review_status="pending"`` の active ファクトを
      (subject, predicate, type) でグループ化して返す
    - ``auto_resolved_history``: ``conflicts_resolved.jsonl`` から最新
      ``history_limit`` 件を読み込み、``auto_evolved=True`` policy の
      自動マージを timeline 表示するために使う
    """
    logger.debug("GET /api/memory/conflicts: scope=%s", scope)
    store = _resolve_store(state, scope)
    pending = _group_pending_conflicts(store, scope)
    history = _read_resolved_history(store, scope, limit=history_limit)
    return ConflictsResponse(
        scope=scope,
        pending=pending,
        auto_resolved_history=history,
    )


@router.post("/conflicts/resolve", response_model=ResolveConflictResponse)
async def resolve_conflict(
    req: ResolveConflictRequest,
    state: AppState = Depends(get_app_state),
):
    """競合を手動解消する

    - ``keep_old`` / ``keep_new``: ``winner_id`` を残し、``loser_ids`` を
      supersede する。``review_status`` を resolved_keep_old / resolved_keep_new
      に更新。
    - ``merge``: ``merged_object`` を持つ新規ファクトを作成し、winner と
      losers をまとめて新規ファクトに supersede する。新規ファクトの
      ``review_status`` は resolved_merged。
    """
    logger.debug(
        "POST /api/memory/conflicts/resolve: scope=%s action=%s winner=%s",
        req.scope, req.action, req.winner_id,
    )
    if req.action not in _RESOLVED_ACTION_TO_STATUS:
        raise HTTPException(
            status_code=400, detail=f"unknown action: {req.action}",
        )
    store = _resolve_store(state, req.scope)

    winner = store.get_fact(req.winner_id)
    if winner is None:
        raise HTTPException(
            status_code=404, detail=f"winner fact not found: {req.winner_id}",
        )
    if winner.superseded_by is not None:
        raise HTTPException(
            status_code=409,
            detail=f"winner fact already superseded: {req.winner_id}",
        )
    losers: list[SemanticFact] = []
    for lid in req.loser_ids:
        if lid == req.winner_id:
            raise HTTPException(
                status_code=400,
                detail="winner_id must not appear in loser_ids",
            )
        loser = store.get_fact(lid)
        if loser is None:
            raise HTTPException(
                status_code=404, detail=f"loser fact not found: {lid}",
            )
        if loser.superseded_by is not None:
            raise HTTPException(
                status_code=409,
                detail=f"loser fact already superseded: {lid}",
            )
        losers.append(loser)

    superseded_ids: list[str] = []
    new_fact_id: str | None = None
    review_status = _RESOLVED_ACTION_TO_STATUS[req.action]

    if req.action == "merge":
        if not req.merged_object:
            raise HTTPException(
                status_code=400,
                detail="merged_object is required for action=merge",
            )
        # winner / losers の supersedes チェーンを集約しつつ新ファクトを作成
        merged_supersedes: list[str] = []
        for f in (winner, *losers):
            for sid in f.supersedes:
                if sid not in merged_supersedes:
                    merged_supersedes.append(sid)
        new_fact = make_fact(
            subject=winner.subject,
            predicate=winner.predicate,
            object_=req.merged_object,
            type=winner.type,  # type: ignore[arg-type]
            scope=winner.scope,
            mode_origin=winner.mode_origin,
            confidence=max(winner.confidence, *(l.confidence for l in losers)),
        )
        added = store.add_fact(new_fact)
        new_fact_id = added.id
        # winner も含めて全件 supersede
        for f in (winner, *losers):
            try:
                store.supersede(f.id, new_fact_id)
                superseded_ids.append(f.id)
            except (KeyError, ValueError) as exc:
                logger.warning("merge supersede failed for %s: %s", f.id, exc)
        store.update_fact(
            new_fact_id,
            requires_user_review=False,
            review_status=review_status,
            supersedes=list({*merged_supersedes, *(f.id for f in (winner, *losers))}),
        )
    else:
        # keep_old / keep_new: losers のみ supersede
        for loser in losers:
            try:
                store.supersede(loser.id, winner.id)
                superseded_ids.append(loser.id)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "supersede failed for %s -> %s: %s",
                    loser.id, winner.id, exc,
                )
        store.update_fact(
            winner.id,
            requires_user_review=False,
            review_status=review_status,
        )

    # pending エントリも掃除
    cleaned_ids = {req.winner_id, *req.loser_ids}
    _remove_pending_lines(store, fact_ids=cleaned_ids)
    _append_resolution_log(
        store,
        scope=req.scope,
        winner=winner,
        loser_ids=[l.id for l in losers],
        action=req.action,
        new_fact_id=new_fact_id,
    )

    return ResolveConflictResponse(
        scope=req.scope,
        action=req.action,
        winner_id=req.winner_id,
        superseded_ids=superseded_ids,
        new_fact_id=new_fact_id,
    )


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
