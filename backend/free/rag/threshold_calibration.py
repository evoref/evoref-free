"""埋め込みモデル切替後の検索閾値ヒューリスティック較正。

再構築済み (reindex 後) のメイン RAG VectorStore の格納ベクトルから、
チャンク間コサイン類似度の分布をサンプリングし、``rag.*`` の絶対閾値
(relevance / support / confidence) の **推奨値** を導出する。

**重要 (advisory のみ)**: これは正解ラベル (golden set) を用いた recall 最適化
ではなく、新しい埋め込みモデルのスコア**スケール**に閾値を合わせるための分布
ヒューリスティックである。チャンク↔チャンク類似度は実クエリ↔文書より高めに出る
ため、提案値はやや過大ゲート寄りになりうる。よって API は自動適用せず提案のみを
返し、UI でユーザーがレビューして適用する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.rag.vector_store import dequantize_int8
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState

logger = get_logger("rag.threshold_calibration")

# 較正に必要な最小ベクトル数 (これ未満は分布が不安定なため拒否)。
_MIN_VECTORS = 20
# サンプリングするクエリ (チャンク) 数の上限。
_MAX_SAMPLE = 256
# 背景 (無関係ペア) 分布のサンプル数。
_BACKGROUND_PAIRS = 2000


def _clamp01(x: float) -> float:
    return round(float(max(0.0, min(1.0, x))), 3)


def calibrate_thresholds(
    state: "AppState", *, seed: int = 42,
) -> dict[str, Any]:
    """メイン RAG VectorStore のスコア分布から ``rag.*`` 閾値の推奨値を返す。

    Returns:
        ``{"ok": bool, "reason"?: str, "n_vectors": int, "sampled": int,
           "distribution": {...}, "suggestions": {relevance_threshold,
           support_threshold, confidence_threshold}}``
        ``ok=False`` の場合は ``reason`` に理由 (ベクトル不足等)。
    """
    vs = state.vector_store
    if vs is None or vs.vectors_q8 is None or vs.scales is None:
        return {"ok": False, "reason": "no_vectors", "n_vectors": 0}

    vectors = dequantize_int8(np.asarray(vs.vectors_q8), np.asarray(vs.scales))
    n = int(vectors.shape[0])
    if n < _MIN_VECTORS:
        return {"ok": False, "reason": "insufficient_vectors", "n_vectors": n}

    # コサイン用に L2 正規化 (量子化で厳密な単位長から僅かにずれるため再正規化)。
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = (vectors / norms).astype(np.float32)

    rng = np.random.default_rng(seed)

    # ── マッチ分布: 各サンプルチャンクを query として最近傍 (自分自身を除く) ──
    sample_n = min(_MAX_SAMPLE, n)
    q_idx = rng.choice(n, size=sample_n, replace=False)
    match_top1: list[float] = []
    match_top3: list[float] = []
    for i in q_idx:
        sims = unit @ unit[i]
        sims[i] = -1.0  # 自分自身 (≈1.0) を除外
        # top-3 (降順)
        k = min(3, n - 1)
        top = np.partition(sims, -k)[-k:]
        top.sort()
        match_top1.append(float(top[-1]))
        match_top3.append(float(top.mean()))

    # ── 背景分布: ランダムな無関係ペア ──
    a = rng.integers(0, n, size=_BACKGROUND_PAIRS)
    b = rng.integers(0, n, size=_BACKGROUND_PAIRS)
    mask = a != b
    bg = np.einsum("ij,ij->i", unit[a[mask]], unit[b[mask]])

    m1 = np.asarray(match_top1)
    m3 = np.asarray(match_top3)

    dist = {
        "match_top1_p25": _clamp01(np.percentile(m1, 25)),
        "match_top1_p50": _clamp01(np.percentile(m1, 50)),
        "match_top1_p75": _clamp01(np.percentile(m1, 75)),
        "match_top3_p25": _clamp01(np.percentile(m3, 25)),
        "match_top3_p50": _clamp01(np.percentile(m3, 50)),
        "background_p50": _clamp01(np.percentile(bg, 50)),
        "background_p95": _clamp01(np.percentile(bg, 95)),
    }

    # ── 推奨値の導出 ──
    # 背景 (無関係) の上端 p95 と マッチ p25 の間に floor を置く (両者の中点)。
    bg95 = float(np.percentile(bg, 95))
    m1_p25 = float(np.percentile(m1, 25))
    m1_p50 = float(np.percentile(m1, 50))
    m3_p25 = float(np.percentile(m3, 25))
    sep = bg95 + 0.5 * max(0.0, m1_p25 - bg95)

    suggestions = {
        # relevance: 無関係/マッチ分離点。support はその少し下 (補助ゲート)。
        "relevance_threshold": _clamp01(sep),
        "support_threshold": _clamp01(min(sep, m3_p25) - 0.05),
        # confidence: マッチ中央値 = 「確信できる」上側帯。
        "confidence_threshold": _clamp01(m1_p50),
    }

    logger.info(
        "Threshold calibration: n=%d sampled=%d dist=%s suggestions=%s",
        n, sample_n, dist, suggestions,
    )
    return {
        "ok": True,
        "n_vectors": n,
        "sampled": sample_n,
        "distribution": dist,
        "suggestions": suggestions,
    }
