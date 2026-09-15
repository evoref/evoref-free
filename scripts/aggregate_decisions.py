"""判定点の記録を集計する — 帯域の分布 / shadow の不一致 / 結末との相関。

``decision_*.jsonl`` は判定点が「何を根拠に何を選んだか」の生ログで、1 監査で
数百行になる。これを読める形にするのが本スクリプト。3 つの見方を出す:

1. **判定点ごとの分布** — ``chosen`` と ``band`` の内訳、``score`` の分位。
   棄権率が想定とずれていたら較正か ``k`` が合っていない。
2. **shadow の不一致** — ``policy=shadow`` の判定点で、規則と事例が食い違った
   ターンを列挙する。層振り分けを切り替えるかどうかはこれを見て人が決める
   (docs/c_17 §3.3)。
3. **結末との相関** — ``outcome_*.jsonl`` と ``trace_id`` で突き合わせ、
   ``(decision_point, chosen)`` ごとの失敗率を出す。「脆い判定点」の手作業
   ランキングを毎回の監査で自動更新するための材料 (c_17 §5)。

``decision`` は ``--develop=investigate`` 以上で出る (c_07 §5.2)。``outcome`` は
``evolve`` 限定なので、3 の相関は evolve で採ったログにしか出ない。

使い方::

    python scripts/aggregate_decisions.py
    python scripts/aggregate_decisions.py --point layer_classification_shadow
    python scripts/aggregate_decisions.py --dir local/logs/debug --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):  # pragma: no cover - 実行環境依存
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_LOG_DIR = REPO_ROOT / "local" / "logs" / "debug"


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def load_category(log_dir: Path, category: str) -> list[dict]:
    """``<category>_YYYY-MM-DD.jsonl`` を日付順に読む。"""
    files = sorted(log_dir.glob(f"{category}_*.jsonl"))
    records: list[dict] = []
    for f in files:
        records.extend(_read_jsonl(f))
    return records


def summarize_points(decisions: list[dict]) -> dict[str, dict[str, Any]]:
    """判定点ごとの ``chosen`` / ``band`` / ``score`` の分布。"""
    by_point: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in decisions:
        point = rec.get("decision_point") or "?"
        grouped[point].append(rec)
    for point, rows in sorted(grouped.items()):
        chosen = Counter(str(r.get("chosen")) for r in rows)
        reasons = Counter(str(r.get("reason") or "") for r in rows)
        bands = Counter(
            str((r.get("context") or {}).get("band") or "-") for r in rows
        )
        scores = [
            float((r.get("context") or {}).get("score"))
            for r in rows
            if isinstance((r.get("context") or {}).get("score"), (int, float))
        ]
        entry: dict[str, Any] = {
            "n": len(rows),
            "chosen": dict(chosen.most_common()),
            "bands": dict(bands.most_common()),
            "reasons": dict(reasons.most_common(8)),
        }
        if scores:
            scores.sort()
            entry["score"] = {
                "p05": round(scores[max(0, int(len(scores) * 0.05) - 1)], 3),
                "median": round(statistics.median(scores), 3),
                "p95": round(scores[min(len(scores) - 1, int(len(scores) * 0.95))], 3),
            }
        by_point[point] = entry
    return by_point


def shadow_disagreements(decisions: list[dict]) -> dict[str, dict[str, Any]]:
    """``policy=shadow`` の判定点で規則と事例が食い違った件を集める。"""
    out: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in decisions:
        ctx = rec.get("context") or {}
        if ctx.get("policy") != "shadow":
            continue
        grouped[rec.get("decision_point") or "?"].append(rec)
    for point, rows in sorted(grouped.items()):
        pairs = Counter()
        agreed = disagreed = abstained = 0
        for r in rows:
            ctx = r.get("context") or {}
            shadow_band = ctx.get("shadow_band")
            if shadow_band == "abstain" or shadow_band is None:
                abstained += 1
                continue
            rule = str(r.get("chosen"))
            shadow = str(ctx.get("shadow_value"))
            if rule == shadow:
                agreed += 1
            else:
                disagreed += 1
                pairs[(rule, shadow)] += 1
        decided = agreed + disagreed
        out[point] = {
            "n": len(rows),
            "agreed": agreed,
            "disagreed": disagreed,
            "abstained": abstained,
            "agreement_rate": round(agreed / decided, 3) if decided else None,
            "pairs": {f"{a} -> {b}": n for (a, b), n in pairs.most_common()},
        }
    return out


def outcome_correlation(
    decisions: list[dict], outcomes: list[dict],
) -> dict[str, dict[str, Any]]:
    """``(decision_point, chosen)`` ごとの失敗率 (``trace_id`` で突き合わせ)。

    ログは **旧方針の下で** 生成されているので、この相関は「どの判定点の
    どの選択が失敗ターンと共起したか」であって因果ではない。脆い判定点の
    当たりを付けるための材料として読む (docs/c_17 §5.3)。
    """
    failed: set[str] = set()
    total: set[str] = set()
    for rec in outcomes:
        trace = rec.get("trace_id")
        if not trace or rec.get("kind") != "chat_response":
            continue
        total.add(trace)
        signals = rec.get("quality_signals") or {}
        if rec.get("success") is False or signals.get("turn_outcome") == "failure":
            failed.add(trace)
    if not total:
        return {}

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rec in decisions:
        trace = rec.get("trace_id")
        if not trace or trace not in total:
            continue
        grouped[(rec.get("decision_point") or "?", str(rec.get("chosen")))].append(trace)

    base_rate = len(failed) / len(total)
    out: dict[str, dict[str, Any]] = {}
    for (point, chosen), traces in grouped.items():
        uniq = set(traces)
        n_fail = len(uniq & failed)
        rate = n_fail / len(uniq)
        out[f"{point}:{chosen}"] = {
            "turns": len(uniq),
            "failed": n_fail,
            "failure_rate": round(rate, 3),
            "lift": round(rate / base_rate, 2) if base_rate else None,
        }
    return dict(
        sorted(out.items(), key=lambda kv: (-(kv[1]["lift"] or 0), -kv[1]["turns"])),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dir", type=Path, default=DEFAULT_LOG_DIR, help="JSONL の置き場",
    )
    ap.add_argument("--point", default="", help="判定点名で絞る (部分一致)")
    ap.add_argument("--top", type=int, default=15, help="相関の表示件数")
    ap.add_argument("--json", type=Path, default=None, help="結果を JSON で保存")
    args = ap.parse_args(argv)

    if not args.dir.exists():
        print(f"ログディレクトリがありません: {args.dir}", file=sys.stderr)
        return 2
    decisions = load_category(args.dir, "decision")
    outcomes = load_category(args.dir, "outcome")
    if args.point:
        decisions = [
            d for d in decisions if args.point in str(d.get("decision_point"))
        ]
    if not decisions:
        print(
            "decision レコードがありません。"
            "--develop=investigate 以上で起動したログを指定してください。",
            file=sys.stderr,
        )
        return 2

    points = summarize_points(decisions)
    shadows = shadow_disagreements(decisions)
    correlation = outcome_correlation(decisions, outcomes)

    print(f"decision {len(decisions)} 件 / outcome {len(outcomes)} 件\n")

    print("■ 判定点ごとの分布")
    for point, e in points.items():
        line = f"   {point}: n={e['n']} chosen={e['chosen']}"
        if e.get("bands") and set(e["bands"]) != {"-"}:
            line += f" bands={e['bands']}"
        if e.get("score"):
            line += f" score={e['score']}"
        print(line)
        top_reasons = list(e["reasons"].items())[:3]
        if top_reasons and top_reasons[0][0]:
            print(f"      理由: {dict(top_reasons)}")
    print()

    if shadows:
        print("■ shadow の不一致 (規則 -> 事例)")
        for point, e in shadows.items():
            print(
                f"   {point}: 一致 {e['agreed']} / 不一致 {e['disagreed']} / "
                f"棄権 {e['abstained']} (一致率 {e['agreement_rate']})",
            )
            for pair, n in e["pairs"].items():
                print(f"      {pair}: {n}")
        print()

    if correlation and all(e["failed"] == 0 for e in correlation.values()):
        print(
            "■ 結末との相関: このログには失敗ターンが 1 件も無いので相関は出ない "
            "(全ターン成功の監査では当然。失敗が出た監査のログで見ること)\n",
        )
    elif correlation:
        print("■ 結末との相関 (lift = その選択の失敗率 / 全体の失敗率)")
        for key, e in list(correlation.items())[: args.top]:
            print(
                f"   {key}: turns={e['turns']} failed={e['failed']} "
                f"rate={e['failure_rate']} lift={e['lift']}",
            )
        print()
    elif outcomes:
        print("■ 結末との相関: 突き合わせられる chat_response がありません\n")
    else:
        print("■ 結末との相関: outcome ログなし (--develop=evolve が要る)\n")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "n_decisions": len(decisions),
                    "n_outcomes": len(outcomes),
                    "points": points,
                    "shadow": shadows,
                    "correlation": correlation,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"JSON を書き出しました: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
