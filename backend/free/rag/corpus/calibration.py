"""corpus 側の関連度較正 (f_01 §4.4 / §6.6)。

記憶側の較正 (:mod:`backend.free.rag.memory_threshold_calibration`) は
**クエリ ↔ 発話ノート** の分布から棒を作る。corpus の文書チャンクは分布が
違い (bge-m3 で正解チャンクの cosine 中央値 0.607 に対し記憶側の棒 0.606 +
弱い集合の上乗せ = 0.64)、その棒を流用すると正解の大半が落ちる (2026-09-12
実測: golden 100 問中 72 問が注入ゼロ)。一方で棒を外すと無関係な問いに資料が
混入する (静的 floor で 25 問中 24 問)。本モジュールは corpus 専用の分布から
2 種類の棒を導く:

1. **チャンク本体の棒** (`relevance` / `support` / `confidence`):
   疑似クエリ (f_01 §6) を代理クエリにした :func:`compute_calibration`。
   代理クエリの由来チャンクを ``self_index`` で除くのは記憶側と同じ。
2. **疑似クエリの棒 兼 関連性ゲート** (`pq_gate`): 問い ↔ 問いの cosine は
   本体より高いスケールで、無関係な問い (top1 p50 0.51) と正解 (p05 0.62) が
   分離する唯一の信号。**ラベル無しで** 両側を推定する —
   正側 = 疑似クエリ自身を検索側に回した leave-one-out top1 の p05
   (実測 0.628、golden の p05 0.62 と一致)、null 側 = 同梱の中立カナリア発話の
   top1 の p95 (実測 0.59)。棒はその中点 (null 側が上回れば null 側)。

較正値は ``CartridgeManager`` が保持し、疑似クエリの充足率が
``rag.pseudo_query.gate_min_coverage`` 以上のときだけ有効になる。それまでは
記憶側の較正 (従来動作、混入ゼロだが厳しい) に倒す。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.rag.memory_threshold_calibration import (
    MIN_NOTES,
    MIN_QUERIES,
    _l2_normalize,
    compute_calibration,
)
from backend.io import AtomicWriter
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("rag.corpus.calibration")

#: 保存ファイル (``local/memory/corpus/`` 配下)。
CALIBRATION_FILENAME = "calibration.json"
SCHEMA_VERSION = 1

#: 正側 (leave-one-out top1) から採る分位。5% の正当な問いを落とす側に倒す。
POSITIVE_QUANTILE = 0.05
#: null 側 (カナリア top1) から採る分位。
NULL_QUANTILE = 0.95

#: 中立カナリア発話。corpus の話題と無関係な「普通の発話」で、疑似クエリ索引の
#: top1 がここまでしか届かない水位 (null 分布) を測る。言語を跨いで散らす。
#: **特定の corpus に寄せないこと** — 寄せると null 側が上がり棒が過大になる。
CANARY_UTTERANCES: tuple[str, ...] = (
    "今日の夕飯は何にしようかな",
    "東京から大阪まで新幹線で何分かかりますか？",
    "おすすめの映画を教えて",
    "猫の名前を考えてください",
    "来週の月曜日は何日ですか？",
    "英語で自己紹介を書いて",
    "りんごとバナナどちらが好き？",
    "筋トレのメニューを組んで",
    "私の名前は田中です",
    "天気が悪いので気分が沈みます",
    "メールの件名を考えて",
    "5+7は？",
    "コーヒーの淹れ方を教えて",
    "週末は家族と温泉に行きました",
    "会議の議事録をまとめて",
    "確定申告の期限はいつ？",
    "おはようございます",
    "この文章を英訳して",
    "出張の経費精算はどうすればいい？",
    "子どもの誕生日プレゼントを考えて",
    "電車が遅れていて会議に間に合いそうにない",
    "健康診断の結果が返ってきた",
    "引っ越しの荷造りのコツは？",
    "今年の夏休みはどこに行こう",
    "住宅ローンの繰り上げ返済は得ですか？",
    "What should I cook for dinner tonight?",
    "Recommend a good sci-fi novel.",
    "How do I get from the airport to downtown?",
    "Write a short birthday message for my colleague.",
    "What's the capital of Australia?",
    "I feel tired today.",
    "Convert 5 miles to kilometers.",
    "Help me plan a weekend hike.",
    "My flight got cancelled, what are my options?",
    "Summarize the plot of Romeo and Juliet.",
)


def compute_corpus_calibration(
    chunk_vecs: np.ndarray,
    pq_vecs: np.ndarray,
    pq_target_rows: list[int],
    canary_vecs: np.ndarray,
) -> dict[str, Any]:
    """corpus の 2 種類の棒を導く。

    Args:
        chunk_vecs: ``(N, D)`` チャンク本体の埋め込み。
        pq_vecs: ``(Q, D)`` 疑似クエリの埋め込み (query 側)。
        pq_target_rows: ``pq_vecs[i]`` の対象チャンク行 (``chunk_vecs`` の行番号)。
        canary_vecs: ``(C, D)`` カナリア発話の埋め込み (query 側)。

    Returns:
        ``{"ok", "reason"?, "n_chunks", "n_pq", "distribution", "thresholds"}``。
        ``thresholds`` は ``relevance_threshold`` / ``support_threshold`` /
        ``confidence_threshold`` (チャンク本体) と ``pq_gate`` (疑似クエリ)。
    """
    base = compute_calibration(chunk_vecs, pq_vecs, self_index=list(pq_target_rows))
    if not base.get("ok"):
        return {
            "ok": False, "reason": base.get("reason"),
            "n_chunks": base.get("n_notes", 0), "n_pq": base.get("n_queries", 0),
        }
    pq = _l2_normalize(np.asarray(pq_vecs, dtype=np.float32))
    canary = _l2_normalize(np.asarray(canary_vecs, dtype=np.float32))

    # 正側: 各疑似クエリを検索側に回し、自分自身を除いた top1。
    sims = pq @ pq.T
    np.fill_diagonal(sims, -1.0)
    positive_top1 = sims.max(axis=1)
    positive_p05 = float(np.quantile(positive_top1, POSITIVE_QUANTILE))

    # null 側: カナリア発話の top1。
    if canary.shape[0] > 0:
        null_top1 = (canary @ pq.T).max(axis=1)
        null_p95 = float(np.quantile(null_top1, NULL_QUANTILE))
    else:
        null_p95 = positive_p05
    pq_gate = null_p95 if null_p95 >= positive_p05 else (null_p95 + positive_p05) / 2.0
    pq_gate = float(min(1.0, max(0.0, pq_gate)))

    distribution = dict(base["distribution"])
    distribution.update({
        "pq_positive_top1_p05": round(positive_p05, 4),
        "pq_null_top1_p95": round(null_p95, 4),
        "pq_overlap": bool(null_p95 >= positive_p05),
    })
    thresholds = dict(base["thresholds"])
    thresholds["pq_gate"] = round(pq_gate, 4)
    # 充足率が低い間の関連性ゲートで本体側に使う棒 = 正解 top1 の p25
    # (f_01 §6.6)。p50 (confidence) は定義上正しい問いの半分を落とす。
    on_topic = distribution.get("match_top1_p25")
    if isinstance(on_topic, (int, float)):
        thresholds["on_topic_threshold"] = round(float(on_topic), 4)
    logger.info(
        "Corpus calibration: chunks=%d pq=%d canaries=%d dist=%s thresholds=%s",
        base["n_notes"], base["n_queries"], canary.shape[0], distribution, thresholds,
    )
    return {
        "ok": True,
        "n_chunks": base["n_notes"],
        "n_pq": base["n_queries"],
        "distribution": distribution,
        "thresholds": thresholds,
    }


def calibration_signature(fingerprint: str, n_chunks: int, n_pq: int) -> str:
    """キャッシュの有効性を決める署名 (埋め込み指紋 + 件数)。"""
    return f"{fingerprint}::chunks={n_chunks}::pq={n_pq}"


def calibration_path(corpus_dir: Path | str) -> Path:
    return Path(corpus_dir) / CALIBRATION_FILENAME


def load_corpus_calibration(
    corpus_dir: Path | str, signature: str,
) -> dict[str, Any] | None:
    """署名が一致する較正結果を返す。無い / 不一致 / 壊れていれば ``None``。"""
    path = calibration_path(corpus_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Corpus calibration cache unreadable (%s): %s", path, e)
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    payload = data.get("payload") or {}
    if payload.get("signature") != signature:
        return None
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        return None
    try:
        payload["thresholds"] = {k: float(v) for k, v in thresholds.items()}
    except (TypeError, ValueError):
        return None
    return payload


def save_corpus_calibration(
    corpus_dir: Path | str, signature: str, result: dict[str, Any],
) -> None:
    """較正結果を c_05 §0.5 の封筒で保存する (``ok=False`` は保存しない)。"""
    if not result.get("ok"):
        return
    path = calibration_path(corpus_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "written_at": utc_now(),
        "producer": "corpus_calibration",
        "payload": {
            "signature": signature,
            "n_chunks": result.get("n_chunks"),
            "n_pq": result.get("n_pq"),
            "distribution": result.get("distribution"),
            "thresholds": result.get("thresholds"),
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with AtomicWriter(path) as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    except OSError as e:
        logger.warning("Failed to persist corpus calibration to %s: %s", path, e)


__all__ = [
    "CANARY_UTTERANCES",
    "MIN_NOTES",
    "MIN_QUERIES",
    "calibration_signature",
    "compute_corpus_calibration",
    "load_corpus_calibration",
    "save_corpus_calibration",
]
