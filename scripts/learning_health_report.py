"""自己学習の信号の量と、学習側の追加開発を続けてよいかの判定を出す (読むだけ)。

自然な個人利用で、学習に使える信号が実際にどれだけ貯まっているかを数える。経験 (``experience.jsonl``)
だけを読み、データ根へは何も書かない。判定の基準 (2026-10-07 決定):

- 自然な利用が 30 日以上続いたうえで、検出された失敗が 30 件未満、または検証済み訂正が 5 件未満なら、
  学習側の追加開発を止める。
- 利用が 30 日に満たない間は「判定保留」。短い期間に集中した経験 (監査・試験のトラフィック) は
  自然な利用として数えない旨を警告する。

使い方::

    python scripts/learning_health_report.py                # 既定のデータ根
    python scripts/learning_health_report.py --data-root userdata --json
    python scripts/learning_health_report.py --since 2026-10-07
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_root import resolve_data_root, store_root  # noqa: E402
from backend.free.learning.level0_instant import fold_experience_file  # noqa: E402
from backend.utils import parse_utc, utc_now_dt  # noqa: E402

#: 判定の基準 (学習側の追加開発を止める条件)
NATURAL_USE_DAYS = 30
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


def summarize(
    records_by_partition: dict[str, list[dict]],
    *,
    now: datetime | None = None,
    since: datetime | None = None,
) -> dict[str, Any]:
    """経験から信号の件数・利用の日数・アイドルの窓の候補を集計する (純関数)。"""
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
    span_days = (stamped[-1][0] - stamped[0][0]).total_seconds() / 86400.0 if len(stamped) > 1 else 0.0
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
    }


def evaluate(summary: dict[str, Any]) -> dict[str, Any]:
    """学習側の追加開発を続けてよいかを判定する (純関数)。

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
    if span < NATURAL_USE_DAYS:
        return {
            "verdict": "hold",
            "reasons": [
                f"利用が {NATURAL_USE_DAYS} 日に満たない ({span:.1f} 日)。判定保留 "
                f"(現時点: 失敗 {failures} 件 / 検証済み訂正 {corrections} 件)"
            ],
            "warnings": warnings,
        }
    reasons = []
    if failures < MIN_DETECTED_FAILURES:
        reasons.append(f"検出された失敗が {failures} 件 (基準 {MIN_DETECTED_FAILURES} 件以上)")
    if corrections < MIN_VERIFIED_CORRECTIONS:
        reasons.append(f"検証済み訂正が {corrections} 件 (基準 {MIN_VERIFIED_CORRECTIONS} 件以上)")
    if reasons:
        return {"verdict": "stop", "reasons": reasons + ["学習側の追加開発を止める"], "warnings": warnings}
    return {"verdict": "continue", "reasons": ["信号が基準を満たしている"], "warnings": warnings}


def render(summary: dict[str, Any], verdict: dict[str, Any]) -> str:
    """人が読む形に整形する。"""
    lines = [
        "== 自己学習の信号の量 ==",
        f"期間: {summary['first_turn']} → {summary['last_turn']} ({summary['span_days']} 日、"
        f"利用のあった日 {summary['active_days']} 日)",
        f"ターン: {summary['turns']} 件 (1 日あたり 中央値 {summary['turns_per_active_day_median']} / "
        f"最大 {summary['turns_per_active_day_max']})  モード: {summary['modes']}",
        f"結果: {summary['outcomes']}",
        f"検出された失敗: {summary['detected_failures']} 件  理由 (上位): {summary['failed_reasons']}",
        f"検証済み訂正: {summary['verified_corrections']} 件 / 訂正の候補: {summary['correction_candidates']} 件 / "
        f"言い直し: {summary['rephrased_queries']} 件 / 👎: {summary['thumbs_down']} 件",
        f"アイドルの窓の候補 (ターン間隔 {summary['idle_gap_minutes']} 分以上): {summary['idle_gaps']} 回、"
        f"合計 {summary['idle_gap_hours_total']} 時間、最長 {summary['idle_gap_hours_max']} 時間 "
        "(PC が起動していた保証はない)",
        "",
        f"== 判定: {verdict['verdict']} ==",
        *[f"- {r}" for r in verdict["reasons"]],
    ]
    if verdict["warnings"]:
        lines += ["", "== 注意 =="] + [f"- {w}" for w in verdict["warnings"]]
    lines += [
        "",
        "別途の基準 (この報告では測らない): 現行同士の雑音の幅が狙う効果量 (3pt) を上回るなら影の比較を止める /",
        "整合性検査の再現率が 20% 未満なら止める。",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="自己学習の信号の量と、追加開発を続けてよいかの判定")
    parser.add_argument("--data-root", default=None, help="データ根 (既定は解決規則どおり)")
    parser.add_argument("--since", default=None, help="この日付 (UTC、YYYY-MM-DD) 以降の経験だけを数える")
    parser.add_argument("--json", action="store_true", help="JSON で出す")
    args = parser.parse_args(argv)

    data_root = resolve_data_root(args.data_root)
    since = parse_utc(args.since + "T00:00:00Z") if args.since else None
    if args.since and since is None:
        parser.error(f"--since を解釈できない: {args.since!r}")
    summary = summarize(load_records(data_root), since=since)
    verdict = evaluate(summary)
    if args.json:
        print(json.dumps({"summary": summary, "verdict": verdict}, ensure_ascii=False, indent=2))
    else:
        print(render(summary, verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
