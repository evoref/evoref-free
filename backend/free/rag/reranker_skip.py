"""Reranker quality-aware skip 判定

hybrid 融合後のスコア分布から reranker を通す価値が薄いケースを検出し、
cross-encoder forward pass を短絡するための共通ユーティリティ。

``HybridRetriever.search`` (benchmark / cartridge 評価経路) と
``search_pipeline._maybe_rerank`` (チャット応答経路) の双方から利用される。
skip 条件はいずれかを満たせば skip する OR 判定。
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.log_config import get_logger

logger = get_logger("rag.reranker_skip")


@dataclass(frozen=True)
class RerankerSkipDecision:
    """Skip 判定結果

    Attributes:
        should_skip: reranker 実行を skip すべきなら True
        reason: skip 理由 (``high_score`` / ``large_gap`` / ``few_candidates``)。
            skip しない場合は空文字列。複数条件を同時に満たす場合は
            最初にヒットした理由を記録する (評価順: few_candidates →
            high_score → large_gap)。
        top_score: 融合後 top-1 スコア (未算出時は 0.0)
        gap: top-1 と top-2 のスコア差 (候補 1 件以下なら 0.0)
        candidates_count: 評価時の候補数
    """

    should_skip: bool
    reason: str
    top_score: float
    gap: float
    candidates_count: int


def _read_skip_cfg(cfg: dict | None) -> dict:
    """``reranker.skip`` セクションを dict で取り出す。

    未指定・不正型のいずれの場合も空 dict を返し、呼び出し側で
    ``get(..., 0)`` による既定値解決に委ねる。
    """
    if not isinstance(cfg, dict):
        return {}
    reranker_cfg = cfg.get("reranker")
    if not isinstance(reranker_cfg, dict):
        return {}
    skip_cfg = reranker_cfg.get("skip")
    if not isinstance(skip_cfg, dict):
        return {}
    return skip_cfg


def is_skip_config_active(cfg: dict | None) -> bool:
    """skip しきい値が 1 つでも有効化されていれば True。

    すべて既定値 (0) の場合は False を返し、呼び出し側は evaluate を
    スキップしてオーバヘッドを避けられる (既存挙動維持)。
    """
    skip_cfg = _read_skip_cfg(cfg)
    return (
        float(skip_cfg.get("hybrid_top_score_threshold", 0.0) or 0.0) > 0.0
        or float(skip_cfg.get("score_gap_threshold", 0.0) or 0.0) > 0.0
        or int(skip_cfg.get("min_candidates", 0) or 0) > 0
    )


def evaluate_reranker_skip(
    candidates: list[tuple[str, float, str]],
    cfg: dict | None,
) -> RerankerSkipDecision:
    """候補リストから reranker を skip すべきか判定する。

    Args:
        candidates: 融合後の候補リスト ``[(chunk_id, score, text), ...]``。
            score 降順にソート済みであることを前提とする。
        cfg: config.yaml 全体 dict。``reranker.skip`` セクションを読む。

    Returns:
        :class:`RerankerSkipDecision`。しきい値がすべて未設定なら
        ``should_skip=False`` の no-op 決定を返す (従来挙動)。

    Notes:
        評価順:
        1. ``min_candidates``: 候補数 < しきい値 → skip (few_candidates)
        2. ``hybrid_top_score_threshold``: top-1 >= しきい値 → skip (high_score)
        3. ``score_gap_threshold``: top-1 と top-2 の差 >= しきい値 → skip (large_gap)

        しきい値 0.0 / 0 は「無効」を意味し、該当条件は評価しない。
    """
    skip_cfg = _read_skip_cfg(cfg)
    top_score_threshold = float(
        skip_cfg.get("hybrid_top_score_threshold", 0.0) or 0.0,
    )
    gap_threshold = float(skip_cfg.get("score_gap_threshold", 0.0) or 0.0)
    min_candidates = int(skip_cfg.get("min_candidates", 0) or 0)

    count = len(candidates)
    top_score = float(candidates[0][1]) if count >= 1 else 0.0
    gap = (
        float(candidates[0][1]) - float(candidates[1][1])
        if count >= 2
        else 0.0
    )

    # min_candidates: 候補が少なすぎる場合は reranker を通さない
    if min_candidates > 0 and count < min_candidates:
        return RerankerSkipDecision(
            should_skip=True,
            reason="few_candidates",
            top_score=top_score,
            gap=gap,
            candidates_count=count,
        )

    # hybrid_top_score_threshold: top-1 の confidence が十分高い場合 skip
    if top_score_threshold > 0.0 and count >= 1 and top_score >= top_score_threshold:
        return RerankerSkipDecision(
            should_skip=True,
            reason="high_score",
            top_score=top_score,
            gap=gap,
            candidates_count=count,
        )

    # score_gap_threshold: top-1 が top-2 を大きく引き離している場合 skip
    if gap_threshold > 0.0 and count >= 2 and gap >= gap_threshold:
        return RerankerSkipDecision(
            should_skip=True,
            reason="large_gap",
            top_score=top_score,
            gap=gap,
            candidates_count=count,
        )

    return RerankerSkipDecision(
        should_skip=False,
        reason="",
        top_score=top_score,
        gap=gap,
        candidates_count=count,
    )
