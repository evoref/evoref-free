"""自己学習の信号の量と、学習側の追加開発を続けてよいかの判定 (読むだけ)。

経験 (``experience.jsonl``) だけを読み、データ根へは何も書かない。判定の基準 (2026-10-07 決定) と
経緯は ``scripts/learning_health_report.py`` の docstring を参照。``/api/status`` の
``learning_health`` と ``evoref doctor`` も同じ集計を使う (判定の実装を 1 つに保つ)。
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from backend.constants import SECONDS_PER_DAY
from backend.data_root import store_root
from backend.free.learning.level0_instant import fold_experience_file
from backend.utils import parse_utc, utc_now_dt

#: 判定の基準 (学習側の追加開発を止める条件)
#: 経験の始まりから今までの日数の下限、かつ「直近の窓」の日数
NATURAL_USE_DAYS = 30
#: 直近 ``NATURAL_USE_DAYS`` 日の窓に要る自然な利用日の数
MIN_NATURAL_DAYS = 15
#: 自然な利用日とみなす 1 日のターン数の下限 (UTC の日)
MIN_TURNS_PER_ACTIVE_DAY = 3
#: その日の任意の 60 分 (滑る窓) にこの件数以上のターンがあればバーストの日 (監査・試験)
BURST_TURNS_PER_HOUR = 30
MIN_DETECTED_FAILURES = 30
MIN_VERIFIED_CORRECTIONS = 5
#: ターン間隔がこれ以上のものを「アイドルの窓の候補」とみなす (分)
IDLE_GAP_MINUTES = 30
#: この日数以下に全体が収まる経験は、監査・試験のトラフィックの疑いがある
BURST_DAYS = 3


def load_records(data_root: Path) -> dict[str, list[dict]]:
    """``<data_root>`` の学習パーティションごとの経験 (model_key → 記録の dict)。"""
    learning = store_root(data_root) / "learning"
    out: dict[str, list[dict]] = {}
    if not learning.is_dir():
        return out
    for part in sorted(learning.iterdir()):
        path = part / "experience.jsonl"
        if part.is_dir() and path.is_file():
            records, _stats = fold_experience_file(path)
            out[part.name] = records
    return out


def _signals(rec: dict) -> dict:
    sig = rec.get("signals")
    return sig if isinstance(sig, dict) else {}


def _is_burst(stamps: list[datetime]) -> bool:
    """時刻順の ``stamps`` のうち、任意の 60 分に ``BURST_TURNS_PER_HOUR`` 件以上が入るか。"""
    hour = timedelta(minutes=60)
    lo = 0
    for hi, ts in enumerate(stamps):
        while ts - stamps[lo] > hour:
            lo += 1
        if hi - lo + 1 >= BURST_TURNS_PER_HOUR:
            return True
    return False


def _natural_use(stamped: list[tuple[datetime, dict]], now: datetime) -> dict[str, Any]:
    """直近 ``NATURAL_USE_DAYS`` 日の窓の自然な利用日と、その日のターンだけの信号の件数。

    自然な利用日 = UTC の日で、``MIN_TURNS_PER_ACTIVE_DAY`` ターン以上あり、バーストでない日。
    監査・試験は欠陥を狙って失敗を膨らませるので、失敗・検証済み訂正も自然な利用日のターン
    だけで数える。
    """
    window_start = now - timedelta(days=NATURAL_USE_DAYS)
    by_day: dict[str, list[tuple[datetime, dict]]] = {}
    for ts, rec in stamped:
        if ts >= window_start:
            by_day.setdefault(ts.date().isoformat(), []).append((ts, rec))
    natural_days = burst_days = failures = corrections = 0
    for turns in by_day.values():
        if _is_burst([ts for ts, _ in turns]):
            burst_days += 1
            continue
        if len(turns) < MIN_TURNS_PER_ACTIVE_DAY:
            continue
        natural_days += 1
        for _ts, rec in turns:
            sig = _signals(rec)
            if sig.get("turn_outcome") == "failed":
                failures += 1
            if sig.get("user_correction"):
                corrections += 1
    return {
        "window_turns": sum(len(t) for t in by_day.values()),
        "natural_days": natural_days,
        "burst_days": burst_days,
        "natural_failures": failures,
        "natural_corrections": corrections,
    }


def _unlearned_partitions(
    records_by_partition: dict[str, list[dict]], active_model_key: str | None,
) -> dict[str, dict[str, Any]]:
    """Level 1 が学習しない (active 以外の) パーティションの経験の件数。active が分からなければ空。"""
    if not active_model_key:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, recs in records_by_partition.items():
        if key == active_model_key or not recs:
            continue
        modes = Counter(str(rec.get("mode") or "?") for rec in recs)
        out[key] = {"experiences": len(recs), "modes": dict(modes)}
    return out


def summarize(
    records_by_partition: dict[str, list[dict]],
    *,
    now: datetime | None = None,
    since: datetime | None = None,
    active_model_key: str | None = None,
) -> dict[str, Any]:
    """経験から信号の件数・利用の日数・アイドルの窓の候補を集計する (純関数)。

    Level 1 は束ねた active パーティション (``active_model_key``) の経験だけを学習する
    (``level1_scope``)。それ以外のパーティションの経験 (create_model が生成した分など) は
    ``unlearned_partitions`` に件数を出す。
    """
    now = now or utc_now_dt()
    stamped: list[tuple[datetime, dict]] = []
    for recs in records_by_partition.values():
        for rec in recs:
            ts = parse_utc(rec.get("timestamp"))
            if ts is None or (since is not None and ts < since):
                continue
            stamped.append((ts, rec))
    stamped.sort(key=lambda x: x[0])

    outcomes: Counter[str] = Counter()
    failed_reasons: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    days: Counter[str] = Counter()
    for ts, rec in stamped:
        sig = _signals(rec)
        days[ts.date().isoformat()] += 1
        modes[str(rec.get("mode") or "?")] += 1
        outcome = str(sig.get("turn_outcome") or "unlabeled")
        outcomes[outcome] += 1
        if outcome == "failed":
            failed_reasons[str(sig.get("turn_outcome_reason") or "(理由なし)")] += 1
        if sig.get("user_correction"):
            counts["verified_corrections"] += 1
        if sig.get("correction_candidate"):
            counts["correction_candidates"] += 1
        if sig.get("rephrased_query"):
            counts["rephrased_queries"] += 1
        if sig.get("user_negative"):
            counts["thumbs_down"] += 1
        if sig.get("tool_routing_false_negative"):
            counts["tool_routing_false_negatives"] += 1

    gaps = [
        (b[0] - a[0]).total_seconds() / 60.0 for a, b in zip(stamped, stamped[1:], strict=False)
    ]
    idle = [g for g in gaps if g >= IDLE_GAP_MINUTES]
    span_days = (stamped[-1][0] - stamped[0][0]).total_seconds() / SECONDS_PER_DAY if len(stamped) > 1 else 0.0
    history_days = (now - stamped[0][0]).total_seconds() / SECONDS_PER_DAY if stamped else 0.0
    per_day = sorted(days.values())
    return {
        "generated_at": now.isoformat(),
        "partitions": {k: len(v) for k, v in records_by_partition.items()},
        "turns": len(stamped),
        "first_turn": stamped[0][0].isoformat() if stamped else None,
        "last_turn": stamped[-1][0].isoformat() if stamped else None,
        "span_days": round(span_days, 2),
        "active_days": len(days),
        "turns_per_active_day_median": statistics.median(per_day) if per_day else 0,
        "turns_per_active_day_max": per_day[-1] if per_day else 0,
        "modes": dict(modes),
        "outcomes": dict(outcomes),
        "failed_reasons": dict(failed_reasons.most_common(10)),
        "detected_failures": outcomes.get("failed", 0),
        "verified_corrections": counts.get("verified_corrections", 0),
        "correction_candidates": counts.get("correction_candidates", 0),
        "rephrased_queries": counts.get("rephrased_queries", 0),
        "thumbs_down": counts.get("thumbs_down", 0),
        "tool_routing_false_negatives": counts.get("tool_routing_false_negatives", 0),
        "idle_gap_minutes": IDLE_GAP_MINUTES,
        "idle_gaps": len(idle),
        "idle_gap_hours_total": round(sum(idle) / 60.0, 2),
        "idle_gap_hours_max": round(max(idle) / 60.0, 2) if idle else 0.0,
        "history_days": round(history_days, 2),
        **_natural_use(stamped, now),
        "level1_scope": "active_only",
        "unlearned_partitions": _unlearned_partitions(records_by_partition, active_model_key),
    }


def evaluate(summary: dict[str, Any]) -> dict[str, Any]:
    """学習側の追加開発を続けてよいかを判定し、Level 1 の学習範囲を添える (純関数)。

    ``level1_scope`` / ``unlearned_partitions`` は集計をそのまま写し、学習されない
    パーティションがあれば警告に出す (判定そのものは変えない)。
    """
    result = _judge(summary)
    unlearned = dict(summary.get("unlearned_partitions") or {})
    for key, part in unlearned.items():
        result["warnings"].append(
            f"Level 1 は束ねた active パーティションだけを学習する。{key} の経験 "
            f"{part.get('experiences', 0)} 件は学習されない"
        )
    return {
        **result,
        "level1_scope": summary.get("level1_scope", "active_only"),
        "unlearned_partitions": unlearned,
    }


def _judge(summary: dict[str, Any]) -> dict[str, Any]:
    """学習側の追加開発を続けてよいかを判定する。

    「経験の始まりから今まで ``NATURAL_USE_DAYS`` 日以上」かつ「直近 ``NATURAL_USE_DAYS``
    日の窓に自然な利用日が ``MIN_NATURAL_DAYS`` 日以上」で初めて判定する。暦の幅
    (``span_days``) だけでは、2 日分の監査バーストと 30 日後の 1 ターンで基準を満たして
    しまう。失敗・検証済み訂正は自然な利用日のターンだけの件数 (``natural_*``) で見る
    (全体の件数は理由に参考として出す)。

    Returns:
        ``verdict`` は ``hold`` (判定保留) / ``continue`` (続けてよい) / ``stop`` (止める)。
    """
    warnings: list[str] = []
    turns = int(summary.get("turns") or 0)
    span = float(summary.get("span_days") or 0.0)
    if turns == 0:
        return {"verdict": "hold", "reasons": ["経験が 1 件も無い"], "warnings": warnings}
    if span <= BURST_DAYS:
        warnings.append(
            f"全体が {span:.1f} 日に収まっている。監査・試験のトラフィックの疑いがあり、"
            "自然な個人利用とは言えない"
        )
    failures = int(summary.get("detected_failures") or 0)
    corrections = int(summary.get("verified_corrections") or 0)
    history = float(summary.get("history_days") or 0.0)
    natural = int(summary.get("natural_days") or 0)
    bursts = int(summary.get("burst_days") or 0)
    natural_failures = int(summary.get("natural_failures") or 0)
    natural_corrections = int(summary.get("natural_corrections") or 0)
    if history < NATURAL_USE_DAYS or natural < MIN_NATURAL_DAYS:
        return {
            "verdict": "hold",
            "reasons": [
                f"自然な利用が足りない: 経験の始まりから {history:.1f} 日 (基準 {NATURAL_USE_DAYS} 日以上)、"
                f"直近 {NATURAL_USE_DAYS} 日の自然な利用日 {natural} 日 (基準 {MIN_NATURAL_DAYS} 日以上、"
                f"バーストの日 {bursts} 日を除外)。判定保留 (参考: 全体の失敗 {failures} 件 / "
                f"検証済み訂正 {corrections} 件、自然な利用日では {natural_failures} 件 / "
                f"{natural_corrections} 件)"
            ],
            "warnings": warnings,
        }
    reasons = []
    if natural_failures < MIN_DETECTED_FAILURES:
        reasons.append(
            f"自然な利用日の検出された失敗が {natural_failures} 件 (基準 {MIN_DETECTED_FAILURES} 件以上)"
        )
    if natural_corrections < MIN_VERIFIED_CORRECTIONS:
        reasons.append(
            f"自然な利用日の検証済み訂正が {natural_corrections} 件 (基準 {MIN_VERIFIED_CORRECTIONS} 件以上)"
        )
    if reasons:
        return {"verdict": "stop", "reasons": reasons + ["学習側の追加開発を止める"], "warnings": warnings}
    return {"verdict": "continue", "reasons": ["信号が基準を満たしている"], "warnings": warnings}


#: ``/api/status`` が全走査を毎回走らせないためのキャッシュ寿命 (秒)。
SNAPSHOT_TTL_SECONDS = 300.0
_SNAPSHOT_KEYS = (
    "detected_failures", "verified_corrections", "active_days", "span_days", "turns",
    "history_days", "window_turns", "natural_days", "burst_days",
    "natural_failures", "natural_corrections", "level1_scope", "unlearned_partitions",
)
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def health_snapshot(
    data_root: Path, *, now: datetime | None = None, active_model_key: str | None = None,
) -> dict[str, Any]:
    """経験を全走査して ``{verdict, reasons, detected_failures, ...}`` を返す。

    ``active_model_key`` は Level 1 が学習するパーティション。分からなければ
    ``unlearned_partitions`` は空。
    """
    summary = summarize(load_records(data_root), now=now, active_model_key=active_model_key)
    verdict = evaluate(summary)
    return {
        "verdict": verdict["verdict"],
        "reasons": list(verdict["reasons"]),
        **{key: summary[key] for key in _SNAPSHOT_KEYS},
    }


def cached_health_snapshot(
    data_root: Path, *, active_model_key: str | None = None,
    ttl: float = SNAPSHOT_TTL_SECONDS, clock=time.monotonic,
) -> dict[str, Any]:
    """:func:`health_snapshot` を ``ttl`` 秒メモ化する (データ根と active ごと)。同期処理なので executor から呼ぶ。"""
    key = f"{data_root}|{active_model_key or ''}"
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and clock() - hit[0] < ttl:
            return dict(hit[1])
    snap = health_snapshot(data_root, active_model_key=active_model_key)
    with _cache_lock:
        _cache[key] = (clock(), snap)
    return dict(snap)
