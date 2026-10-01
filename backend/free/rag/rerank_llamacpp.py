"""llama-server ``/v1/rerank`` 経由の再順位スコア (c_16 §7.2.1)。

cross-encoder のリランカー (例: japanese-bge-reranker-v2-m3) を ``--reranking`` で
起動した専用 llama-server (既定 :8083) に、クエリ 1 本と候補文書を渡してスコアを得る。

``rag.rerank.mode: on`` のとき、検索パイプライン (``unified_search`` の Step 6.8) が floor を
通った corpus の候補を渡して並べ替える。

**チャット経路の待ち時間に直接乗る** ので、CLAUDE.md §6 #10 のリトライ (3 回) の例外として
試行 1 回 + 締切 (``deadline_ms``) で打ち切る (docs/e_03 §4.4)。失敗・締切超過・同値の
スコア (旧リランカーを撤去させた退化) は ``None`` を返し、呼び手は cosine 順のまま続ける。
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from backend.free.llm._base_client import BaseHTTPClient
from backend.log_config import get_logger

logger = get_logger("rag.rerank_llamacpp")

RERANK_PATH = "/v1/rerank"
HEALTH_PATH = "/health"

#: スコアの幅 (最大 − 最小) がこれ未満なら「全候補同値」とみなす (退化)。
FLAT_SCORE_EPSILON = 1e-3
#: 全スコアの絶対値がこれ未満なら「全部 0」とみなす (旧リランカーの 0.0 / ~1e-22)。
ZERO_SCORE_EPSILON = 1e-6
#: rag JSONL に残す上位スコアの件数。
TOP_SCORES_LOGGED = 5


def build_rerank_payload(query: str, documents: Sequence[str]) -> dict[str, Any]:
    """``/v1/rerank`` の要求本文 (全件のスコアを返させる)。"""
    return {"query": query, "documents": list(documents), "top_n": len(documents)}


def parse_rerank_response(data: Any, n_docs: int) -> tuple[list[float], int | None]:
    """``/v1/rerank`` の応答を **入力順** のスコア列にする。

    llama-server は ``results`` をスコア降順に並べて返すことがあるので ``index`` で戻す。
    件数・index・スコアの型が合わなければ ``ValueError`` (呼び手は失敗として扱う)。

    Returns:
        (入力順のスコア, ``usage.prompt_tokens`` — 無ければ ``None``)
    """
    if not isinstance(data, Mapping):
        raise ValueError("rerank response is not an object")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("rerank response has no results list")
    scores: list[float | None] = [None] * n_docs
    for item in results:
        if not isinstance(item, Mapping):
            raise ValueError("rerank result is not an object")
        index = item.get("index")
        score = item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n_docs:
            raise ValueError(f"rerank result index out of range: {index!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"rerank result score is not a finite number: {score!r}")
        if scores[index] is not None:
            raise ValueError(f"rerank result index duplicated: {index}")
        scores[index] = float(score)
    missing = [i for i, s in enumerate(scores) if s is None]
    if missing:
        raise ValueError(f"rerank response lacks scores for {len(missing)} of {n_docs} documents")
    usage = data.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        prompt_tokens = None
    return [float(s) for s in scores if s is not None], prompt_tokens


def degenerate_reason(scores: Sequence[float]) -> str | None:
    """スコアが退化していれば理由 (``all_zero`` / ``flat``)、健全なら ``None``。

    ``all_zero`` は 1 件でも判定する — 健全な cross-encoder の logit が絶対値 1e-6 未満に
    なることはまず無く、旧リランカーの退化 (0.0 / ~1e-22) はそのまま 1 件でも現れる
    (sleep-time の疑似クエリの採点は 1 件ずつ送る、c_16 §7.2.1 の第 5 段階 B)。
    ``flat`` は比べる相手が要るので 2 件以上のときだけ。
    """
    if not scores:
        return None
    if all(abs(s) < ZERO_SCORE_EPSILON for s in scores):
        return "all_zero"
    if len(scores) < 2:
        return None
    if max(scores) - min(scores) < FLAT_SCORE_EPSILON:
        return "flat"
    return None


class RerankClient(BaseHTTPClient):
    """rerank 用 llama-server のクライアント (EvorefGen)。

    自己テストの結果 (配置・1 件あたり ms・候補数) を属性に持つ — 呼び手 (検索経路の
    Step 6.8) は ``candidates`` 件までを渡す。
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8083,
        deadline_ms: int = 1000,
        candidates: int = 20,
        placement: str = "",
        ms_per_doc: float | None = None,
        debug_logger=None,
        pseudo_query_calibration: Any = None,
    ) -> None:
        # httpx 自身の timeout は締切より少し長く取り、打ち切りは asyncio.wait_for に一本化する。
        super().__init__(timeout=deadline_ms / 1000.0 + 1.0)
        self.base_url = f"http://{host}:{port}"
        self.deadline_ms = int(deadline_ms)
        self.candidates = int(candidates)
        self.placement = placement
        self.ms_per_doc = ms_per_doc
        self.debug_logger = debug_logger
        #: 疑似クエリの品質検査の較正 (``pseudo_query_gate.PseudoQueryGateCalibration``、
        #: c_17 §3.15)。起動時にこのモデルの model_key の項目を読む。``None`` なら捨てない。
        self.pseudo_query_calibration = pseudo_query_calibration

    async def health_check(self) -> bool:
        """``/health`` が 200 か (リトライしない、docs/e_03 §4.1)。"""
        try:
            resp = await self._get_http_client().get(f"{self.base_url}{HEALTH_PATH}", timeout=2.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def rerank(
        self, query: str, documents: Sequence[str], *, ids: Sequence[str] | None = None,
    ) -> list[float] | None:
        """``documents`` の入力順のスコア。失敗・締切超過・退化は ``None`` (試行 1 回)。

        ``ids`` (``documents`` と同じ並び) を渡すと、並べ替えの前後の id 順と上位スコアを
        rag JSONL に残す (``op="rerank"``)。
        """
        if not documents:
            return []
        started = time.perf_counter()
        reason = ""
        prompt_tokens: int | None = None
        scores: list[float] | None = None
        try:
            resp = await asyncio.wait_for(
                self._get_http_client().post(
                    f"{self.base_url}{RERANK_PATH}", json=build_rerank_payload(query, documents),
                ),
                timeout=self.deadline_ms / 1000.0,
            )
            resp.raise_for_status()
            parsed, prompt_tokens = parse_rerank_response(resp.json(), len(documents))
            bad = degenerate_reason(parsed)
            if bad is None:
                scores = parsed
            else:
                reason = f"degenerate_{bad}"
        except TimeoutError:
            reason = "deadline"
        except httpx.HTTPStatusError as e:
            reason = f"http_{e.response.status_code}"
        except httpx.HTTPError as e:
            reason = f"http_error:{type(e).__name__}"
        except ValueError as e:
            reason = "bad_response"
            logger.debug("rerank response rejected: %s", e)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if scores is None:
            logger.info(
                "Rerank skipped (%s) after %.0f ms for %d documents; keeping cosine order",
                reason, elapsed_ms, len(documents),
            )
        if self.debug_logger is not None:
            ids_before = list(ids) if ids is not None else None
            ids_after = top_scores = None
            if ids_before is not None and scores is not None and len(ids_before) == len(scores):
                order = sorted(range(len(scores)), key=lambda i: -scores[i])
                ids_after = [ids_before[i] for i in order]
                top_scores = [round(scores[i], 4) for i in order[:TOP_SCORES_LOGGED]]
            self.debug_logger.log_rerank(
                n_in=len(documents),
                elapsed_ms=elapsed_ms,
                ok=scores is not None,
                reason=reason,
                prompt_tokens=prompt_tokens,
                deadline_ms=self.deadline_ms,
                ids_before=ids_before,
                ids_after=ids_after,
                top_scores=top_scores,
            )
        return scores


__all__ = [
    "FLAT_SCORE_EPSILON",
    "RERANK_PATH",
    "ZERO_SCORE_EPSILON",
    "RerankClient",
    "build_rerank_payload",
    "degenerate_reason",
    "parse_rerank_response",
]
