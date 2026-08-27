"""記憶検索の品質ゲート閾値を、実ストアの分布から較正する。

``rag.relevance_threshold`` / ``support_threshold`` / ``confidence_threshold``
は **cosine スケール前提**の絶対閾値だが、到達可能なスコア域は埋め込みモデルに
よって大きく変わる。出荷時の既定 (0.65 / 0.50 / 0.80) は Qwen3-Embedding の
分布を前提にした値で、別モデルへ切り替えるとゲートが恒久的に閉じたままになる
(実測 2026-08-12、LFM2.5-Embedding-350M: 記憶が要るクエリの top cosine が
0.338〜0.636 に対し relevance=0.65 は到達不能で、31 ターンの実会話で採用 0 件)。

そこで **埋め込みモデル指紋** (model_id + 次元) をキーに、実ストアのスコア分布
から閾値を導出してキャッシュする。

導出式:

- ``relevance`` = 背景 (無関係ペア) コサインの **p95**。
  「無関係な内容がここまでしか届かない」水位。実測で背景 p95=0.288 に対し、
  ラベル付きプローブの no-need 上限 0.243 / need 下限 0.338 のちょうど間に落ちた。
- ``support`` = ``relevance - 0.05`` (top3 平均に掛かる補助ゲートなので少し下)。
- ``confidence`` = マッチ top1 コサインの **p50**。「確信できる」上側帯。

**代理クエリ**: 背景・マッチ両方の分布は「クエリ↔ノート」で測る必要がある。
質問文と平叙文では埋め込みが非対称で、ノート↔ノートで測ると系統的に高く出る
(既存の :mod:`backend.free.rag.threshold_calibration` はチャンク↔チャンクを
測っており、docstring 自身が過大ゲート寄りになると認めている)。ここでは
**保存済みのユーザー発話**をクエリ整形テンプレート経由で再埋め込みし、実際の
検索と同じ非対称性を持つ代理クエリとして使う。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from backend.log_config import get_logger

logger = get_logger("rag.memory_threshold_calibration")

#: 較正に必要な最小ノート数 / 最小代理クエリ数。これ未満は分布が不安定。
MIN_NOTES = 30
MIN_QUERIES = 10
#: 背景分布のサンプルペア数。
BACKGROUND_PAIRS = 4000
#: support を relevance からどれだけ下げるか。
SUPPORT_MARGIN = 0.05
#: 較正キャッシュのファイル名 (``local_paths.memory_dir`` 配下)。
CALIBRATION_FILENAME = "threshold_calibration.json"
#: 保存フォーマットのバージョン。式を変えたら上げてキャッシュを無効化する。
SCHEMA_VERSION = 1


#: プロセス共通のアクティブ較正値。起動時に一度だけ解決し、
#: :meth:`QualityThresholds.from_config` が参照する。``backend.config._config``
#: と同じく「モデル構成から導かれるプロセス全体の設定」の位置づけ。
_ACTIVE_CALIBRATION: dict[str, float] | None = None


def set_active_calibration(thresholds: dict[str, float] | None) -> None:
    """アクティブ較正値を差し替える (起動時 / モデル切替時 / テスト)。"""
    global _ACTIVE_CALIBRATION
    _ACTIVE_CALIBRATION = dict(thresholds) if thresholds else None


def get_active_calibration() -> dict[str, float] | None:
    """アクティブ較正値。未較正なら ``None`` (呼出側は config 値へ縮退)。"""
    return _ACTIVE_CALIBRATION


#: 起動時に較正できなかったときの再試行口。起動時のバインド済み引数
#: (memory_dir / fingerprint / short_term / embedder) を閉じ込めた 0 引数の
#: コルーチンファクトリを ``_pillar_wirer`` が登録する。
_PENDING_RECALIBRATION: "Callable[[], Any] | None" = None


def set_pending_recalibration(factory: "Callable[[], Any] | None") -> None:
    """較正の再試行口を登録する (起動時に較正できなかった場合のみ)。"""
    global _PENDING_RECALIBRATION
    _PENDING_RECALIBRATION = factory


async def retry_pending_calibration() -> bool:
    """較正がまだ確定していなければ 1 回だけ再試行する。

    **なぜ要るか**: 較正は起動時に 1 回しか走らない。``MIN_NOTES`` (=30) に
    満たない状態で起動すると skip され、そのプロセスの生涯にわたって config の
    静的閾値が使われ続ける。出荷時の既定 (relevance 0.65 等) は別の埋め込み
    モデルの分布を前提にした値なので、**品質ゲートが恒久的に閉じたまま**になる。

    実測 (2026-08-27 ライブ監査、LFM2.5-Embedding-350M): 起動時ノート 0 件で
    skip → その後 STM は 105 件まで増えたが再較正されず、``Quality: low`` が
    12/12。観測された top_score の最大は 0.428 で、``relevance_threshold``
    0.65 には到達しようがなかった。

    sleep-time Full の末尾から呼ぶ (ノートが増えるのは Step 1-8 の後なので、
    同じサイクル内で増えた分をそのまま使える)。

    Returns:
        このコールで較正が確定したか。既に確定済み / 未登録 / 条件未達なら
        ``False``。
    """
    if _ACTIVE_CALIBRATION is not None:
        return False
    factory = _PENDING_RECALIBRATION
    if factory is None:
        return False
    await factory()
    return _ACTIVE_CALIBRATION is not None


def embedder_fingerprint(model_id: str, dim: int) -> str:
    """埋め込みモデル指紋。これが変わったらキャッシュを捨てる。"""
    return f"{model_id}::{int(dim)}"


def _clamp01(x: float) -> float:
    return round(float(max(0.0, min(1.0, x))), 3)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """行ごとに L2 正規化する (ゼロベクトルはそのまま)。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def compute_calibration(
    note_vecs: np.ndarray,
    query_vecs: np.ndarray,
    *,
    self_index: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """ノート集合と代理クエリ集合から閾値を導出する。

    Args:
        note_vecs: ``(N, D)`` の L2 正規化済みノートベクトル。
        query_vecs: ``(Q, D)`` の L2 正規化済み代理クエリベクトル。
        self_index: ``query_vecs[i]`` が由来するノートの行番号。マッチ分布の
            計算で自分自身 (コサイン≒1) を除外するために使う。
        seed: 背景ペアサンプリングの乱数種。

    Returns:
        ``{"ok": bool, "reason"?: str, "n_notes": int, "n_queries": int,
           "distribution": {...}, "thresholds": {...}}``
    """
    n = int(note_vecs.shape[0]) if note_vecs.size else 0
    q = int(query_vecs.shape[0]) if query_vecs.size else 0
    if n < MIN_NOTES:
        return {"ok": False, "reason": "insufficient_notes", "n_notes": n, "n_queries": q}
    if q < MIN_QUERIES:
        return {"ok": False, "reason": "insufficient_queries", "n_notes": n, "n_queries": q}

    # dot 積をコサインとして扱うため、両側を L2 正規化しておく
    # (量子化・キャッシュ経路によっては厳密な単位長でないことがある)。
    note_vecs = _l2_normalize(np.asarray(note_vecs, dtype=np.float32))
    query_vecs = _l2_normalize(np.asarray(query_vecs, dtype=np.float32))

    rng = np.random.default_rng(seed)

    # ── マッチ分布: 各代理クエリの top1 (自分自身を除く) ──
    match_top1: list[float] = []
    for i in range(q):
        sims = note_vecs @ query_vecs[i]
        if self_index is not None and 0 <= self_index[i] < n:
            sims[self_index[i]] = -1.0
        match_top1.append(float(sims.max()))
    m1 = np.asarray(match_top1)

    # ── 背景分布: ランダムなクエリ×ノートのペア (自分自身を除く) ──
    qa = rng.integers(0, q, size=BACKGROUND_PAIRS)
    nb = rng.integers(0, n, size=BACKGROUND_PAIRS)
    if self_index is not None:
        owner = np.asarray(self_index)[qa]
        keep = owner != nb
    else:
        keep = np.ones(BACKGROUND_PAIRS, dtype=bool)
    bg = np.einsum("ij,ij->i", query_vecs[qa[keep]], note_vecs[nb[keep]])

    bg95 = float(np.percentile(bg, 95))
    relevance = _clamp01(bg95)
    support = _clamp01(relevance - SUPPORT_MARGIN)
    confidence = _clamp01(float(np.percentile(m1, 50)))

    # confidence が relevance を下回ると high/medium の順序が壊れる。
    if confidence <= relevance:
        confidence = _clamp01(relevance + 0.10)

    distribution = {
        "background_p50": _clamp01(np.percentile(bg, 50)),
        "background_p95": _clamp01(bg95),
        "match_top1_p25": _clamp01(np.percentile(m1, 25)),
        "match_top1_p50": _clamp01(np.percentile(m1, 50)),
        "match_top1_p75": _clamp01(np.percentile(m1, 75)),
    }
    thresholds = {
        "relevance_threshold": relevance,
        "support_threshold": support,
        "confidence_threshold": confidence,
    }
    logger.info(
        "Memory threshold calibration: notes=%d queries=%d dist=%s thresholds=%s",
        n, q, distribution, thresholds,
    )
    return {
        "ok": True,
        "n_notes": n,
        "n_queries": q,
        "distribution": distribution,
        "thresholds": thresholds,
    }


def calibration_path(memory_dir: Path | str) -> Path:
    """較正キャッシュのファイルパス。"""
    return Path(memory_dir) / CALIBRATION_FILENAME


def load_calibration(
    memory_dir: Path | str, fingerprint: str,
) -> dict[str, float] | None:
    """指紋が一致する較正済み閾値を返す。無い / 不一致 / 壊れていれば ``None``。"""
    path = calibration_path(memory_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Threshold calibration cache unreadable (%s): %s", path, e)
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        logger.info(
            "Threshold calibration cache schema %s != %s; ignoring",
            data.get("schema_version"), SCHEMA_VERSION,
        )
        return None
    if data.get("fingerprint") != fingerprint:
        logger.info(
            "Threshold calibration cache is for a different embedder "
            "(cached=%s, current=%s); recalibration required",
            data.get("fingerprint"), fingerprint,
        )
        return None
    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict):
        return None
    try:
        return {k: float(v) for k, v in thresholds.items()}
    except (TypeError, ValueError):
        logger.warning("Threshold calibration cache has non-numeric values; ignoring")
        return None


def save_calibration(
    memory_dir: Path | str, fingerprint: str, result: dict[str, Any],
) -> None:
    """較正結果を保存する。``ok=False`` の結果は保存しない。"""
    if not result.get("ok"):
        return
    path = calibration_path(memory_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "n_notes": result.get("n_notes"),
        "n_queries": result.get("n_queries"),
        "distribution": result.get("distribution"),
        "thresholds": result.get("thresholds"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from backend.io.atomic import AtomicWriter

        with AtomicWriter(path) as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    except OSError as e:
        logger.warning("Failed to persist threshold calibration to %s: %s", path, e)
