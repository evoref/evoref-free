"""``search_history`` の再順位 (c_16 §7.2.1 の第 5 段階 A)。

``HistoryManager.search_candidates`` の字句スコア順の候補のうち上位を、リランカー
(cross-encoder、``RerankClient``) のスコアで並べ替える。**並べ替えるだけ** で、候補を
落とさない — 再順位の経路 (``session_id`` 無し) の候補は ``list_sessions`` の絞り込みを
通っているので、字句スコアの下駄 (``max(score, 0.1)``) で拾われた中身の無いヒットは
来ない (``session_id`` 指定の全ターン走査の経路は再順位しない)。落とす判定は較正が
要るので持たない。代わりにプールのスコアを rag JSONL (``op="history_rerank"``) に残し、
較正の材料にする。

縮退 (候補 2 件未満 / 締切超過 / 失敗 / 退化) は ``None`` を返し、呼出側は再順位の無い
経路 (現行) をそのまま使う。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.history.history_manager import IndexEntry

logger = get_logger("agent.tools.history_rerank")

#: 1 候補あたりリランカーへ渡す本文の上限 (字)。自己テストの 1 件あたり ms は実際の
#: チャンク長 (300 トークン前後、c_16 §7.2.1) で測っているので、それに揃えて締切と
#: 候補数の前提を崩さない。索引の検索テキスト (最大 5000 字) をそのまま渡すと 1 件が
#: 10 倍以上重くなる。
RERANK_PASSAGE_CHARS = 400
#: プールは ``limit`` のこの倍まで (上限は ``RerankClient.candidates``)。
POOL_PER_LIMIT = 3
#: プールがこれ未満なら呼ばない (並べ替える相手が無い)。
MIN_POOL = 2


class HistoryReranker(Protocol):
    """再順位のクライアント (EvorefGen の ``RerankClient`` が満たす面)。"""

    candidates: int

    async def rerank(
        self, query: str, documents: Sequence[str], *, ids: Sequence[str] | None = None,
    ) -> list[float] | None: ...


def rerank_passage(entry: "IndexEntry", query: str, *, max_chars: int = RERANK_PASSAGE_CHARS) -> str:
    """候補セッションからリランカーへ渡す本文 (最大 ``max_chars`` 字)。

    索引の検索テキスト (要約 + 各ターンの先頭 5000 字) のうち、クエリ (無ければ
    空白区切りの 2 字以上の語) が最初に現れる位置の周りを切り出す。どれも現れない
    (トークン重なりだけで一致した) なら先頭。
    """
    text = entry.search_text or entry.summary or ""
    if len(text) <= max_chars:
        return text
    low = text.lower()
    q = (query or "").lower().strip()
    pos = low.find(q) if q else -1
    if pos < 0:
        hits = [low.find(t) for t in q.split() if len(t) >= 2]
        hits = [h for h in hits if h >= 0]
        pos = min(hits) if hits else -1
    if pos < 0:
        return text[:max_chars]
    start = max(0, min(pos - max_chars // 4, len(text) - max_chars))
    return text[start:start + max_chars]


def pool_size(limit: int, reranker: HistoryReranker) -> int:
    """1 回に並べ替える候補数 (``limit × POOL_PER_LIMIT`` とリランカーの候補数の小さい方)。"""
    return max(0, min(max(0, int(limit)) * POOL_PER_LIMIT, int(reranker.candidates)))


async def rerank_history_candidates(
    query: str,
    candidates: Sequence[tuple["IndexEntry", float]],
    reranker: HistoryReranker,
    *,
    limit: int,
    debug_logger: Any = None,
) -> list[tuple["IndexEntry", float]] | None:
    """字句スコア順の ``candidates`` の上位を再順位で並べた全候補。縮退は ``None``。

    プール (上位 :func:`pool_size` 件) を再順位の降順 (同点は字句の順) に並べ、プール外は
    後ろに字句の順で続ける。字句スコア (``relevance_score`` の元) は書き換えない。
    """
    n_pool = pool_size(limit, reranker)
    pool = list(candidates[:n_pool])
    started = time.perf_counter()
    reason = ""
    scores: list[float] | None = None
    if len(pool) < MIN_POOL:
        reason = "few_candidates"
    else:
        ids = [entry.session_id for entry, _ in pool]
        try:
            raw = await reranker.rerank(query, [rerank_passage(e, query) for e, _ in pool], ids=ids)
        except Exception as e:  # noqa: BLE001 — 再順位の失敗で検索を止めない
            logger.warning("History rerank failed, keeping the lexical order: %s: %s", type(e).__name__, e)
            raw = None
        if raw is None or len(raw) != len(pool):
            reason = "rerank_unavailable"
        else:
            scores = [float(s) for s in raw]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ordered: list[tuple["IndexEntry", float]] | None = None
    if scores is not None:
        order = sorted(range(len(pool)), key=lambda i: -scores[i])
        ordered = [pool[i] for i in order] + list(candidates[len(pool):])
    if debug_logger is not None:
        debug_logger.log_history_rerank(
            n_candidates=len(candidates),
            limit=int(limit),
            ids=[e.session_id for e, _ in pool],
            lexical_scores=[round(float(s), 4) for _, s in pool],
            rerank_scores=None if scores is None else [round(s, 4) for s in scores],
            ids_after=None if ordered is None else [e.session_id for e, _ in ordered[:len(pool)]],
            elapsed_ms=elapsed_ms,
            reason=reason,
        )
    if ordered is None:
        logger.info(
            "History rerank skipped (%s) for %d candidate(s); keeping the lexical order",
            reason, len(candidates),
        )
    return ordered


__all__ = [
    "MIN_POOL",
    "POOL_PER_LIMIT",
    "RERANK_PASSAGE_CHARS",
    "HistoryReranker",
    "pool_size",
    "rerank_history_candidates",
    "rerank_passage",
]
