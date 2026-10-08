"""自己学習の信号の量と、学習側の追加開発を続けてよいかの判定を出す (読むだけ)。

自然な個人利用で、学習に使える信号が実際にどれだけ貯まっているかを数える。経験 (``experience.jsonl``)
だけを読み、データ根へは何も書かない。判定の基準 (2026-10-07 決定):

- 経験の始まりから今まで 30 日以上、かつ直近 30 日の窓に自然な利用日が 15 日以上あって初めて判定する
  (自然な利用日 = UTC の日で 3 ターン以上あり、バーストでない日。バーストの日 = その日の任意の
  60 分に 30 ターン以上ある日 = 監査・試験のトラフィック)。満たさない間は「判定保留」。
- 判定できるとき、自然な利用日のターンで検出された失敗が 30 件未満、または検証済み訂正が 5 件未満なら、
  学習側の追加開発を止める (監査は欠陥を狙って失敗を膨らませるので、全体の件数は参考に留める)。
- 全体が 3 日以内に収まる経験は、自然な利用として数えない旨を警告する。

使い方::

    python scripts/learning_health_report.py                # 既定のデータ根
    python scripts/learning_health_report.py --data-root userdata --json
    python scripts/learning_health_report.py --since 2026-10-07
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_root import resolve_data_root  # noqa: E402
from backend.free.learning.health_report import (  # noqa: E402,F401 — 判定の本体は backend 側 (再公開)
    BURST_DAYS,
    BURST_TURNS_PER_HOUR,
    IDLE_GAP_MINUTES,
    MIN_DETECTED_FAILURES,
    MIN_NATURAL_DAYS,
    MIN_TURNS_PER_ACTIVE_DAY,
    MIN_VERIFIED_CORRECTIONS,
    NATURAL_USE_DAYS,
    evaluate,
    load_records,
    summarize,
)
from backend.utils import parse_utc  # noqa: E402

def render(summary: dict[str, Any], verdict: dict[str, Any]) -> str:
    """人が読む形に整形する。"""
    lines = [
        "== 自己学習の信号の量 ==",
        f"期間: {summary['first_turn']} → {summary['last_turn']} ({summary['span_days']} 日、"
        f"利用のあった日 {summary['active_days']} 日)",
        f"自然な利用: 経験の始まりから {summary['history_days']} 日、直近 {NATURAL_USE_DAYS} 日のターン "
        f"{summary['window_turns']} 件・自然な利用日 {summary['natural_days']} 日 (バーストの日 "
        f"{summary['burst_days']} 日を除外)、その日の失敗 {summary['natural_failures']} 件 / "
        f"検証済み訂正 {summary['natural_corrections']} 件",
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
    # cp932 のコンソールは 👎 を符号化できず UnicodeEncodeError で落ちる。
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    if args.json:
        print(json.dumps({"summary": summary, "verdict": verdict}, ensure_ascii=False, indent=2))
    else:
        print(render(summary, verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
