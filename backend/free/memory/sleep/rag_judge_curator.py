"""Step 8.7: RAG necessity/quality リコール用のキュレーター

チャット応答パスで発火した RAG necessity/quality の assist 判定
(``RagJudgeAssistLog`` 経由でリングバッファに蓄積されたもの) を、STM の
直近 Q&A ペアと突き合わせて「この判定は最終応答の観点で妥当だったか」を
アシストで自己採点し、score が閾値以上のものを ``world_fact``
(subject = ``mem.world.rag_necessity.*`` / ``mem.world.rag_quality.*``) と
して永続化する。``url_curator.py`` と対称の設計 (プロンプト差し替えのみ)。

CLAUDE.md §6 不変則 #2 より、SemMem への書込は sleep-time に限定される。
本モジュールは ``SleepTimeWorker.run_full`` の Step 8.7 として呼び出され、
チャット応答パスでは利用しない (引き当ては
``backend.free.memory.pipeline.rag_judge_recall`` が読み取りのみで完結する)。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.memory.sleep._curator_common import (
    build_scoring_prompt,
    coerce_bare_score,
    subject_digest,
    truncate_for_prompt,
)
from backend.free.llm.json_schemas import UrlRelevanceJudgement
from backend.free.memory.types import SemanticFact, make_fact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.memory.pipeline.rag_judge_assist_log import RagJudgeAssistEvent
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.stores.short_term import MemoryNote
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.sleep.rag_judge_curator")

_SUBJECT_PREFIXES = {
    "rag_necessity": "mem.world.rag_necessity.",
    "rag_quality": "mem.world.rag_quality.",
}
_DECISION_FIELDS = {"rag_necessity": "action", "rag_quality": "quality"}
_WS_RE = re.compile(r"\s+")

_PROMPT_SYSTEM = {
    "rag_necessity": (
        "あなたは検索要否判定の評価者です。ユーザの質問と、その質問に対する"
        "アシスタントの最終応答を見て、示された判定 (retrieve/fetch/skip) が"
        "妥当だったかを 0..1 のスコアで評価してください。"
        "0.0 = 明らかに誤った判定、1.0 = 完全に妥当、相応の中間値で出力してください。"
    ),
    "rag_quality": (
        "あなたは検索結果品質判定の評価者です。ユーザの質問と、その質問に対する"
        "アシスタントの最終応答を見て、示された品質ラベル (high/medium/low) が"
        "妥当だったかを 0..1 のスコアで評価してください。"
        "0.0 = 明らかに誤った判定、1.0 = 完全に妥当、相応の中間値で出力してください。"
    ),
}


def _normalize_query(query: str) -> str:
    return _WS_RE.sub(" ", query or "").strip()


def _make_subject(kind: str, query_normalized: str) -> str:
    digest = subject_digest(query_normalized)
    return f"{_SUBJECT_PREFIXES[kind]}{digest}"


#: 体裁は _curator_common が SSOT (3 兄弟で byte 一致の定義を各々持っていた)。
_truncate = truncate_for_prompt


def _build_user_prompt(query: str, answer: str, decision: str) -> str:
    return build_scoring_prompt(query, answer, JUDGEMENT=decision)


_coerce_bare_score = coerce_bare_score


async def _score_decision(
    assist_client: "AssistModelClient",
    *,
    kind: str,
    query: str,
    answer: str,
    decision: str,
) -> float | None:
    """アシストモデルで判定の妥当性を採点する。失敗時は ``None``。"""
    try:
        result = await assist_client.generate(
            messages=[
                {"role": "system", "content": _PROMPT_SYSTEM[kind]},
                {"role": "user", "content": _build_user_prompt(query, answer, decision)},
            ],
            # 768: tool_judgment と同根拠 (2026-07-18 実インシデント)。gemma4 系
            # アシストで reasoning_budget=0 が稀に実効せず、256 では reasoning
            # 出力だけで max_tokens を使い切り空応答になる事例が確認された。
            max_tokens=768,
            temperature=0.1,
            purpose="rag_judge_relevance_score",
            response_schema=UrlRelevanceJudgement,
        )
    except Exception as exc:
        logger.warning("rag_judge_curator: assist scoring failed: %s", exc)
        return None
    from backend.free.llm.json_extract import extract_json_object
    from backend.free.llm.utils import extract_content

    content = extract_content(result)
    parsed = extract_json_object(content)
    if isinstance(parsed, dict):
        raw = parsed.get("score")
    else:
        raw = _coerce_bare_score(content)
        if raw is None:
            logger.debug(
                "rag_judge_curator: failed to parse score JSON: %s", content[:120],
            )
            return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def _find_answer_for_query(notes: list["MemoryNote"], query: str) -> str | None:
    """STM から ``query`` と完全一致する user note の直後 assistant note を探す。

    ``RagJudgeAssistEvent`` は session_id を持たないため、全 session を対象に
    ``created_at`` 昇順で直近一致 (末尾から探索) を優先する。
    """
    ordered = sorted(notes, key=lambda n: n.created_at)
    target = query.strip()
    for idx in range(len(ordered) - 1, -1, -1):
        note = ordered[idx]
        if note.source == "user" and note.content.strip() == target:
            for after in ordered[idx + 1:]:
                if after.source == "assistant":
                    return after.content
            return None
    return None


def _existing_fact(store: "SemanticFactStore", subject: str) -> SemanticFact | None:
    for fact in store.search_by_subject(subject):
        if fact.type == "world_fact":
            return fact
    return None


def _record_score(
    fact: SemanticFact,
    score: float,
    query: str,
    decision: str,
    decision_field: str,
    *,
    history_size: int,
    now: float,
) -> dict[str, Any]:
    extra = dict(fact._extra) if fact._extra else {}
    history = list(extra.get("score_history") or [])
    history.append(round(float(score), 4))
    if len(history) > history_size:
        history = history[-history_size:]
    score_avg = sum(history) / len(history) if history else 0.0
    extra.update(
        {
            decision_field: decision,
            "score_history": history,
            "score_avg": round(score_avg, 4),
            "score_count": len(history),
            "last_judged_at": now,
            "last_query": _truncate(query or "", 200),
        },
    )
    return extra


async def curate_rag_judge_facts(
    events: list["RagJudgeAssistEvent"],
    notes: list["MemoryNote"],
    *,
    config: dict | None,
    store_provider: Callable[[str], "SemanticFactStore | None"] | None,
    assist_client: "AssistModelClient | None",
    embedder: "EmbeddingBackend | None",
    profile_id: str = "default",
    debug_logger=None,
    now_provider: Callable[[], float] | None = None,
) -> int:
    """RAG necessity/quality リコール用の ``world_fact`` を sleep-time で書き込む。

    Args:
        events: ``RagJudgeAssistLog.drain()`` から得たイベント列。
        notes: 直近の ``MemoryNote`` 群 (通常 ``ShortTermMemory.notes.values()``)。
        config: 全体 config。``rag.self_rag.necessity_recall`` /
            ``quality_recall`` セクションを参照する。
        store_provider: ``scope -> SemanticFactStore | None`` のコールバック。
        assist_client: ``None`` の場合は no-op (degraded mode)。
        embedder: subject 用 embedding 生成用。``None`` で no-op。
        profile_id: 書込先 fact の profile_id。
        debug_logger: 任意の DebugLogger。
        now_provider: 時刻供給。テスト用。

    Returns:
        新規に書き込まれた / 更新された fact 件数。
    """
    if assist_client is None:
        logger.debug("rag_judge_curator: assist_client is None (degraded mode), skipping")
        return 0
    if embedder is None:
        logger.debug("rag_judge_curator: embedder is None, skipping")
        return 0
    if store_provider is None:
        logger.debug("rag_judge_curator: store_provider is None, skipping")
        return 0
    if not events:
        return 0

    rag_cfg = ((config or {}).get("rag") or {}).get("self_rag") or {}
    store = store_provider("global")
    if store is None:
        logger.debug("rag_judge_curator: global store not available, skipping")
        return 0

    now_fn = now_provider or time.time
    written = 0

    for ev in events:
        recall_cfg = rag_cfg.get(
            "necessity_recall" if ev.kind == "rag_necessity" else "quality_recall",
        ) or {}
        if not bool(recall_cfg.get("enabled", True)):
            continue
        min_record = float(recall_cfg.get("min_record_score", 0.65))
        history_size = int(recall_cfg.get("record_history_size", 10))

        # ターン確定時に紐付けた本文を優先する。STM 引き直しは、紐付け導入前の
        # 残存イベントと、紐付けに失敗したケースのフォールバック
        # (light サイクルの eviction で見つからないことが多い)。
        answer = ev.answer or _find_answer_for_query(notes, ev.query)
        if not answer:
            continue

        score = await _score_decision(
            assist_client, kind=ev.kind, query=ev.query, answer=answer,
            decision=ev.decision,
        )
        if score is None or score < min_record:
            continue

        normalized = _normalize_query(ev.query)
        if not normalized:
            continue
        subject = _make_subject(ev.kind, normalized)
        decision_field = _DECISION_FIELDS[ev.kind]
        now = now_fn()
        try:
            existing = _existing_fact(store, subject)
            if existing is not None:
                new_extra = _record_score(
                    existing, score, ev.query, ev.decision, decision_field,
                    history_size=history_size, now=now,
                )
                new_avg = float(new_extra.get("score_avg") or 0.0)
                store.update_fact(existing.id, _extra=new_extra, confidence=new_avg)
                written += 1
            else:
                embedding = None
                try:
                    emb = await embedder.embed([normalized], is_query=False)
                    if emb is not None and len(emb) > 0:
                        embedding = np.asarray(emb[0], dtype=np.float32)
                except Exception as exc:
                    logger.warning("rag_judge_curator: embed failed: %s", exc)

                extra = {
                    decision_field: ev.decision,
                    "query": normalized,
                    "score_history": [round(float(score), 4)],
                    "score_avg": round(float(score), 4),
                    "score_count": 1,
                    "last_judged_at": now,
                    "last_query": _truncate(ev.query or "", 200),
                }
                fact = make_fact(
                    subject=subject,
                    predicate="judged_as",
                    object_=ev.decision,
                    type="world_fact",
                    scope="global",
                    confidence=float(score),
                    now=now,
                    profile_id=profile_id,
                    embedding=embedding,
                    _extra=extra,
                )
                store.add_fact(fact)
                written += 1
        except Exception as exc:
            logger.warning(
                "rag_judge_curator: persist failed for %s: %s", subject, exc,
            )

    if written and debug_logger is not None:
        try:
            debug_logger.log_memory_op(
                op="rag_judge_curator",
                stats={"facts_curated": written},
            )
        except Exception:
            pass
    if written:
        logger.info("rag_judge_curator: curated %d fact(s)", written)
    return written


__all__ = ["curate_rag_judge_facts"]
