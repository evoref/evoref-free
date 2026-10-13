"""上位 k の選択 — スコア降順、同点は ``tiebreak`` 昇順で **境界の同点も** 決定論にする。"""

from __future__ import annotations

import numpy as np


def top_k_desc(scores: np.ndarray, tiebreak: np.ndarray, k: int) -> np.ndarray:
    """``scores`` の上位 ``k`` の **位置** を、スコア降順・同点は ``tiebreak`` 昇順で返す。

    ``np.argpartition`` は k 番目と同点の行のどれを採るかが不定なので、
    境界のスコアに並んだ行は ``tiebreak`` の小さい順に採る。
    """
    n = scores.shape[0]
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if n <= k:
        return np.lexsort((tiebreak, -scores))
    neg = -scores
    kth = np.partition(neg, k - 1)[k - 1]
    chosen = np.flatnonzero(neg < kth)
    ties = np.flatnonzero(neg == kth)
    need = k - chosen.shape[0]
    if ties.shape[0] > need:
        ties = ties[np.argsort(tiebreak[ties], kind="stable")[:need]]
    sel = np.concatenate([chosen, ties])
    return sel[np.lexsort((tiebreak[sel], -scores[sel]))]
