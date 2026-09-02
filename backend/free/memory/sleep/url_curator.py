"""Step 8.5: URL リコール用のキュレーター

ユーザの直近セッションで参照された URL について、補助タスクが
「その URL が回答に正しく寄与したか」を 0..1 で自己採点し、
score >= 閾値の URL を ``world_fact`` (subject = ``mem.world.url.*``) として
SemMem に永続化する。

CLAUDE.md §6 不変則 #2 より、SemMem への書込は sleep-time に限定される。
本モジュールは ``SleepTimeWorker.run_full`` の Step 8.5 として呼び出され、
チャット応答パスでは利用しない (引き当ては
``ToolCallJudge._try_recall_url`` が読み取りのみで完結する)。

設計ポリシー:

- 新 FactType を追加せず ``world_fact`` を流用する (CLAUDE.md §3 / §6 #2)。
- subject = ``mem.world.url.<host>.<sha1_12(url_normalized)>``。同一 URL の
  決定論的キーで既存 fact を引き当て可能。
- ``_extra`` に URL 専用メタ (url / fetch_count / score_history /
  score_avg / last_query 等) を載せる。SemanticFact の round-trip 機能で
  そのまま JSONL に保持される。
- ``scorer_client is None`` (ベース未接続) では何もせず ``0`` を返す。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import numpy as np

from backend.free.memory.sleep._curator_common import (
    build_scoring_prompt,
    coerce_bare_score,
    public_notes,
    subject_digest,
    truncate_for_prompt,
)
from backend.free.llm.json_schemas import UrlRelevanceJudgement
from backend.free.memory.note_facts import fact_from_note
from backend.free.memory.types import SemanticFact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.stores.short_term import MemoryNote
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.sleep.url_curator")

# 非 ASCII (CJK 等) も除外し、「URL + 日本語」入力で末尾テキストを
# URL に取り込まないようにする (例: https://news.yahoo.co.jp/で取得して...)。
_URL_RE = re.compile(r"(?i:https?)://[^\s\]\)」』、,\u0080-\U0010ffff]+")
_SUBJECT_PREFIX = "mem.world.url."

# fetch_url が失敗した時にユーザ応答に残る代表的なシグナル文字列。
# 1 つでもマッチすれば「URL は今回有効でなかった」とみなし、
# 既存 fact の score_history に 0 を追加して penalize する。
_FETCH_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Error fetching URL", re.IGNORECASE),
    re.compile(r"Error: Unsupported content-type", re.IGNORECASE),
    re.compile(r"\bHTTP\s*[45]\d{2}\b", re.IGNORECASE),
    re.compile(r"\b(?:404|403|410|500|502|503|504)\s*(?:Not\s*Found|Forbidden|Error)?", re.IGNORECASE),
)


def _has_fetch_failure(answer: str | None) -> bool:
    """assistant 応答が fetch_url 失敗を示唆しているか。"""
    if not answer:
        return False
    return any(p.search(answer) for p in _FETCH_FAILURE_PATTERNS)

_PROMPT_SYSTEM = (
    "あなたは URL の関連性評価者です。"
    "ユーザの質問とアシスタントの最終応答を見て、提示された URL が"
    "その質問の回答に使われた / 使うべきだったかを 0..1 のスコアで評価してください。"
    "0.0 = 全く関係ない、1.0 = 完全に関連している、相応の中間値で出力してください。"
)


def _normalize_url(url: str) -> tuple[str, str]:
    """URL を ``(host_lower, normalized_url)`` に正規化する。

    - scheme + host + path のみ残し、fragment / query を削除
    - host は lowercase、先頭の ``www.`` を削る
    - path 末尾の ``/`` を削る (ルートのみは残す)
    - host が解決できない場合は ``("", "")``
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return "", ""
    if parsed.scheme not in ("http", "https"):
        return "", ""
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return "", ""
    path = parsed.path or ""
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    normalized = urlunparse((parsed.scheme, host, path, "", "", ""))
    return host, normalized


def _make_subject(host: str, normalized_url: str) -> str:
    """URL リコール用 subject を構築する。"""
    digest = subject_digest(normalized_url)
    return f"{_SUBJECT_PREFIX}{host}.{digest}"


def _iter_qa_pairs(
    notes: list["MemoryNote"],
) -> list[tuple["MemoryNote", "MemoryNote", list[str]]]:
    """user note とその直後の assistant note を ``(q, a, urls)`` の組として返す。

    URL を含む user note のみ対象とする。同一 session 内で連続する user
    と assistant note を ``created_at`` でソートして対にする。
    """
    by_session: dict[str, list["MemoryNote"]] = {}
    for note in notes:
        by_session.setdefault(note.session_id or "_", []).append(note)
    pairs: list[tuple["MemoryNote", "MemoryNote", list[str]]] = []
    for sess_notes in by_session.values():
        ordered = sorted(sess_notes, key=lambda n: n.created_at)
        for idx, note in enumerate(ordered):
            if note.source != "user":
                continue
            urls = _URL_RE.findall(note.content or "")
            if not urls:
                continue
            assistant_note: "MemoryNote | None" = None
            for after in ordered[idx + 1:]:
                if after.source == "assistant":
                    assistant_note = after
                    break
            if assistant_note is None:
                continue
            pairs.append((note, assistant_note, urls))
    return pairs


#: 体裁は _curator_common が SSOT (3 兄弟で byte 一致の定義を各々持っていた)。
_truncate = truncate_for_prompt


def _build_user_prompt(query: str, answer: str, url: str) -> str:
    return build_scoring_prompt(query, answer, URL=url)


def _topic_text(query: str) -> str:
    """質問テキストから保存用 topic を作る (200 chars 以内に圧縮)。"""
    cleaned = re.sub(_URL_RE, "", query or "").strip()
    if not cleaned:
        cleaned = (query or "").strip()
    return _truncate(cleaned, 200)


def _existing_url_fact(
    store: "SemanticFactStore", subject: str, topic: str,
) -> SemanticFact | None:
    """同一 subject + 同一 topic の既存 fact を探す。"""
    for fact in store.search_by_subject(subject):
        if fact.type != "world_fact":
            continue
        if fact.object == topic:
            return fact
    return None


def _record_score(
    fact: SemanticFact,
    score: float,
    query: str,
    *,
    history_size: int,
    now: float,
) -> dict[str, Any]:
    """既存 fact の ``_extra`` を採点 1 件分だけ更新した dict を返す。"""
    extra = dict(fact._extra) if fact._extra else {}
    history = list(extra.get("score_history") or [])
    history.append(round(float(score), 4))
    if len(history) > history_size:
        history = history[-history_size:]
    fetch_count = int(extra.get("fetch_count") or 0) + 1
    score_avg = sum(history) / len(history) if history else 0.0
    extra.update(
        {
            "fetch_count": fetch_count,
            "score_history": history,
            "score_avg": round(score_avg, 4),
            "score_count": len(history),
            "last_fetched_at": now,
            "last_query": _truncate(query or "", 200),
        },
    )
    return extra


_coerce_bare_score = coerce_bare_score


async def _score_url(
    scorer_client,
    *,
    query: str,
    answer: str,
    url: str,
) -> float | None:
    """sleep-time の LLM で URL 関連性を採点する。失敗時は ``None``。"""
    try:
        result = await scorer_client.generate(
            messages=[
                {"role": "system", "content": _PROMPT_SYSTEM},
                {"role": "user", "content": _build_user_prompt(query, answer, url)},
            ],
            # 128 だと score/relevant の後ろで reason(最大200字) を出力中に
            # finish_reason=length で頻繁に切れ json_repair 依存になるため 256。
            max_tokens=256,
            temperature=0.1,
            purpose="url_relevance_score",
            response_schema=UrlRelevanceJudgement,
        )
    except Exception as exc:
        logger.warning("url_curator: relevance scoring failed: %s", exc)
        return None
    from backend.free.llm.json_extract import extract_json_object
    from backend.free.llm.utils import extract_content

    content = extract_content(result)
    parsed = extract_json_object(content)
    if isinstance(parsed, dict):
        raw = parsed.get("score")
    else:
        # LFM2 系 hybrid モデルは json_schema grammar が強制されず、裸の数値
        # ("0.7") を返すことがある。その数値をそのまま score として採用する。
        raw = _coerce_bare_score(content)
        if raw is None:
            logger.debug("url_curator: failed to parse score JSON: %s", content[:120])
            return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


async def curate_url_facts(
    notes: list["MemoryNote"],
    *,
    config: dict | None,
    store_provider: Callable[[str], "SemanticFactStore | None"] | None,
    scorer_client,
    embedder: "EmbeddingBackend | None",
    profile_id: str = "default",
    debug_logger=None,
    now_provider: Callable[[], float] | None = None,
) -> int:
    """URL リコール用の world_fact を sleep-time で書き込む。

    Args:
        notes: 直近の MemoryNote 群 (通常 ``ShortTermMemory.notes.values()``)。
        config: 全体 config。``tools.url_recall_*`` を参照する。
        store_provider: ``scope -> SemanticFactStore | None`` のコールバック。
        scorer_client: 採点に使う LLM クライアント (sleep-time のベース
            モデルアダプタ)。``None`` の場合は no-op。
        embedder: topic embedding 生成用。``None`` で no-op。
        profile_id: 書込先 fact の profile_id。
        debug_logger: 任意の DebugLogger。
        now_provider: 時刻供給。テスト用。

    Returns:
        新規に書き込まれた / 更新された URL fact 件数。
    """
    if scorer_client is None:
        logger.debug("url_curator: scorer_client is None, skipping")
        return 0
    if embedder is None:
        logger.debug("url_curator: embedder is None, skipping")
        return 0
    if store_provider is None:
        logger.debug("url_curator: store_provider is None, skipping")
        return 0
    cfg = (config or {}).get("tools") or {}
    if not bool(cfg.get("url_recall_enabled", True)):
        logger.debug("url_curator: tools.url_recall_enabled=false, skipping")
        return 0
    min_record = float(cfg.get("url_recall_min_record_score", 0.6))
    history_size = int(cfg.get("url_recall_record_history_size", 10))

    # private セッション由来のノートは SemMem へ昇格させない。
    # (``_curator_common.public_notes`` の docstring に実害と経緯)
    notes = public_notes(notes)
    pairs = _iter_qa_pairs(notes)
    if not pairs:
        return 0
    store = store_provider("global")
    if store is None:
        logger.debug("url_curator: global store not available, skipping")
        return 0

    now_fn = now_provider or time.time
    written = 0
    for user_note, assistant_note, urls in pairs:
        # 処理済みアンカー (user note) はスキップ。STM に残り続ける同一 QA ペアを
        # Full サイクルごとに再採点 / fetch_count 水増しするのを防ぐ。
        if user_note.url_curated_at is not None:
            continue
        seen_urls: set[str] = set()
        fetch_failed = _has_fetch_failure(assistant_note.content)
        for raw_url in urls:
            host, normalized = _normalize_url(raw_url)
            if not host or not normalized:
                continue
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)

            if fetch_failed:
                # 失敗シグナル検出 — 補助タスクには採点させず score=0 を強制。
                # 既存 fact のみ更新する (新規 fact の作成はしない)。
                subject = _make_subject(host, normalized)
                topic = _topic_text(user_note.content)
                existing = _existing_url_fact(store, subject, topic)
                if existing is None:
                    logger.warning(
                        "url_curator: fetch failure for %s with no existing "
                        "fact, skipping (no fact to penalize)",
                        host,
                    )
                    continue
                logger.warning(
                    "url_curator: fetch failure for %s, recording score=0",
                    host,
                )
                now = now_fn()
                try:
                    new_extra = _record_score(
                        existing, 0.0, user_note.content,
                        history_size=history_size, now=now,
                    )
                    new_avg = float(new_extra.get("score_avg") or 0.0)
                    store.update_fact(
                        existing.id,
                        _extra=new_extra,
                        confidence=new_avg,
                    )
                    written += 1
                except Exception as exc:
                    logger.warning(
                        "url_curator: penalize-update failed for %s: %s",
                        host, exc,
                    )
                continue

            score = await _score_url(
                scorer_client,
                query=user_note.content,
                answer=assistant_note.content,
                url=raw_url,
            )
            if score is None:
                continue
            if score < min_record:
                logger.debug(
                    "url_curator: score=%.2f < min_record=%.2f for %s, skip",
                    score, min_record, host,
                )
                continue

            subject = _make_subject(host, normalized)
            topic = _topic_text(user_note.content)
            existing = _existing_url_fact(store, subject, topic)
            now = now_fn()
            try:
                if existing is not None:
                    new_extra = _record_score(
                        existing, score, user_note.content,
                        history_size=history_size, now=now,
                    )
                    new_avg = float(new_extra.get("score_avg") or 0.0)
                    store.update_fact(
                        existing.id,
                        _extra=new_extra,
                        confidence=new_avg,
                    )
                    written += 1
                else:
                    embedding = None
                    try:
                        emb = await embedder.embed([topic], is_query=False)
                        if emb is not None and len(emb) > 0:
                            embedding = np.asarray(emb[0], dtype=np.float32)
                    except Exception as exc:
                        logger.warning("url_curator: embed failed: %s", exc)

                    extra = {
                        "url": raw_url,
                        "url_normalized": normalized,
                        "host": host,
                        "fetch_count": 1,
                        "score_history": [round(float(score), 4)],
                        "score_avg": round(float(score), 4),
                        "score_count": 1,
                        "last_fetched_at": now,
                        "last_query": _truncate(user_note.content or "", 200),
                    }
                    fact = fact_from_note(
                        user_note,
                        subject=subject,
                        predicate="answers_topic",
                        object_=topic,
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
                logger.warning("url_curator: persist failed for %s: %s", host, exc)

        # ペア処理完了 — score < min_record / fetch 失敗を含む全分岐で必ず
        # マークし、次サイクルで同じペアを再採点させない。
        user_note.url_curated_at = now_fn()

    if written and debug_logger is not None:
        try:
            debug_logger.log_memory_op(
                op="url_curator",
                stats={"facts_curated": written},
            )
        except Exception:
            pass
    if written:
        logger.info("url_curator: curated %d URL fact(s)", written)
    return written


__all__ = ["curate_url_facts"]
