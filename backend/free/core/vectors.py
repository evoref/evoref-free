"""ベクトル行列の小さな共通処理 (純粋関数)。"""

from __future__ import annotations

import numpy as np


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """行ごとに L2 正規化した float32 行列を返す (ゼロベクトルの行はそのまま)。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)
