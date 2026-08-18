"""COST_WEIGHT の較正 — 品質項とコスト項の tick span を実測して重みを導く。

``PolicyParamEvolver`` の fitness は

    fitness = (1 - w) * 品質 + w * (1 - コスト)

で、選択圧になるのは各項の **tick 間 span** (経験集合の平均が tick ごとに動く幅)。
両項の寄与を釣り合わせる重みは ``w* = span_Q / (span_Q + span_C)``。

コスト標本の入手経路は 2 つ:

1. ``local/learning/<model>/experience.json`` の ``prompt_tokens`` /
   ``cached_prompt_tokens`` (2026-08-18 に配線)。本命だが、配線以降に蓄積した
   ぶんしか無い。
2. ``local/logs/llama-base.stderr.log`` の ``slot print_timing``。配線以前の
   履歴からも再プリフィル率を復元できる。1 が足りないときの代替。

使い方::

    python scripts/calibrate_cost_weight.py
    python scripts/calibrate_cost_weight.py --experience <path> --llama-log <path>

``COST_WEIGHT`` を変更したら ``FITNESS_SCHEMA_VERSION`` も上げること
(旧尺度の best_fitness が更新不能な基準として残るため)。
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.free.learning.policy_evolver import (  # noqa: E402
    COST_WEIGHT,
    EVOLVABLE_DOMAINS,
    MIN_FITNESS_SAMPLES,
    MONOTONE_PARAMS,
    _FITNESS_FUNCTIONS,
    _PROMPT_COST_DOMAINS,
    _calc_fitness_default,
    cost_samples,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PROMPT = re.compile(r"task (\d+) \| prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens")
_EVAL = re.compile(r"task (\d+) \|\s+eval time =\s*[\d.]+ ms /\s*(\d+) tokens")
_RELEASE = re.compile(r"task (\d+) \| stop processing: n_tokens = (\d+)")
_SLOT = re.compile(r"slot launch_slot_: id\s+(\d+) \| task (\d+)")

#: tick シミュレーションの最小プレフィックス長と分割数。
#:
#: 最小長は Level 1 の起動下限 (``learning.level1_min_experiences`` 既定 20) に
#: 合わせる。実際に進化が回り始めるのがこの規模なので、それより大きい窓から
#: 測ると本番より安定側に偏る。逆に小さくしすぎると平均が暴れて span が
#: 過大になる。
_MIN_TICK = 20
_TICKS = 12


def _tick_span(values: list[float]) -> tuple[float, int] | None:
    """プレフィックス平均の span (最大 - 最小) と tick 数を返す。

    fitness は毎 tick「バッファ全体の平均」を採るので、実際に選択圧となるのは
    個々の標本のばらつきではなく **平均が tick 間で動く幅**。
    """
    if len(values) < _MIN_TICK:
        return None
    step = max(1, len(values) // _TICKS)
    means = [
        statistics.mean(values[:t])
        for t in range(_MIN_TICK, len(values) + 1, step)
    ]
    if len(means) < 2:
        return None
    return max(means) - min(means), len(means)


def quality_spans(experiences: list[dict]) -> dict[str, tuple[float, int]]:
    """ドメイン別の品質項 tick span。コスト項は含めない (素の品質関数を直接叩く)。"""
    out: dict[str, tuple[float, int]] = {}
    if len(experiences) < _MIN_TICK:
        return out
    step = max(1, len(experiences) // _TICKS)
    prefixes = [
        experiences[:t] for t in range(_MIN_TICK, len(experiences) + 1, step)
    ]
    for domain in EVOLVABLE_DOMAINS:
        fn = _FITNESS_FUNCTIONS.get(domain, _calc_fitness_default)
        vals = [v for p in prefixes if (v := fn(p)) is not None]
        if len(vals) < 2:
            continue
        out[domain] = (max(vals) - min(vals), len(vals))
    return out


def cost_spans_from_experience(
    experiences: list[dict],
) -> dict[str, tuple[float, int]]:
    """経験バッファのコスト項 tick span (本命経路)。"""
    out: dict[str, tuple[float, int]] = {}
    for domain in EVOLVABLE_DOMAINS:
        samples = cost_samples(domain, experiences)
        if (res := _tick_span(samples)) is not None:
            out[domain] = res
    return out


def reprefill_from_llama_log(path: Path) -> list[dict]:
    """llama-server のログから ``(slot, 再プリフィル率)`` を復元する。

    1 タスクにつき ``prompt eval time = ... / P tokens`` (再評価分)、
    ``eval time = ... / G tokens`` (生成分)、``stop processing: n_tokens = T``
    (スロット総量) が出る。プロンプト全体は ``T - G`` なので
    再プリフィル率 = ``P / (T - G)``。
    """
    prompt_eval: dict[int, int] = {}
    gen: dict[int, int] = {}
    release: dict[int, int] = {}
    slot: dict[int, int] = {}

    for raw in io.open(path, encoding="utf-8", errors="replace"):
        line = _ANSI.sub("", raw)
        if (m := _SLOT.search(line)):
            slot[int(m.group(2))] = int(m.group(1))
        if (m := _PROMPT.search(line)):
            prompt_eval[int(m.group(1))] = int(m.group(2))
        elif (m := _EVAL.search(line)):
            gen[int(m.group(1))] = int(m.group(2))
        if (m := _RELEASE.search(line)):
            release[int(m.group(1))] = int(m.group(2))

    rows = []
    for task, reprefill in sorted(prompt_eval.items()):
        if task not in gen or task not in release:
            continue
        prompt_total = release[task] - gen[task]
        if prompt_total <= 0 or reprefill > prompt_total:
            continue
        rows.append({
            "slot": slot.get(task),
            "ratio": reprefill / prompt_total,
        })
    return rows


def _fmt(res: tuple[float, int] | None) -> str:
    return "(不足)" if res is None else f"{res[0]:.4f} (tick={res[1]})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experience", type=Path, help="experience.json のパス")
    ap.add_argument("--llama-log", type=Path, help="llama-base.stderr.log のパス")
    ap.add_argument("--mode", default="chat", help="対象モード (既定 chat)")
    ap.add_argument(
        "--chat-slot", type=int, default=0,
        help="チャット側の llama スロット ID (既定 0)。背景タスクと分離する",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    exp_path = args.experience
    if exp_path is None:
        found = sorted((root / "local" / "learning").glob("*/experience.json"))
        exp_path = found[0] if found else root / "local" / "experience.json"
    log_path = args.llama_log or root / "local" / "logs" / "llama-base.stderr.log"

    print(f"experience : {exp_path}")
    print(f"llama log  : {log_path}")
    print(f"現在の COST_WEIGHT = {COST_WEIGHT}")
    print()

    experiences: list[dict] = []
    if exp_path.exists():
        experiences = [
            e for e in json.loads(exp_path.read_text(encoding="utf-8"))
            if e.get("mode") == args.mode
        ]
    print(f"経験 ({args.mode}): {len(experiences)} 件")

    q = quality_spans(experiences)
    c_exp = cost_spans_from_experience(experiences)

    print()
    print("== 品質項の tick span ==")
    for domain in EVOLVABLE_DOMAINS:
        print(f"  {domain:<10} {_fmt(q.get(domain))}")

    print()
    print("== コスト項の tick span (経験バッファ = 本命経路) ==")
    for domain in EVOLVABLE_DOMAINS:
        n = len(cost_samples(domain, experiences))
        note = "" if n >= MIN_FITNESS_SAMPLES else f"  ← 標本 {n} 件で fitness に未反映"
        print(f"  {domain:<10} {_fmt(c_exp.get(domain))}{note}")

    fallback_span: float | None = None
    if log_path.exists():
        rows = reprefill_from_llama_log(log_path)
        chat = [r["ratio"] for r in rows if r["slot"] == args.chat_slot]
        print()
        print(f"== 代替経路: llama ログの再プリフィル率 (n={len(rows)}) ==")
        for label, vals in (
            ("全体", [r["ratio"] for r in rows]),
            (f"slot {args.chat_slot}", chat),
        ):
            res = _tick_span(vals)
            if vals:
                print(f"  {label:<10} n={len(vals):>3} mean={statistics.mean(vals):.3f} "
                      f"tick span={_fmt(res)}")
            if label != "全体" and res is not None:
                fallback_span = res[0]

    print()
    print("== 推奨重み  w* = span_Q / (span_Q + span_C) ==")
    any_row = False
    for domain in EVOLVABLE_DOMAINS:
        if domain not in MONOTONE_PARAMS:
            continue  # 解除対象キーを持たないドメインは較正の対象外
        span_q = q.get(domain, (None, 0))[0]
        span_c = c_exp.get(domain, (None, 0))[0]
        source = "experience"
        # llama ログから復元できるのは **プロンプト側** の再プリフィル率だけ。
        # 生成側コスト (long_form の予算消費率) の代替にはならないので流用しない。
        if span_c is None and fallback_span is not None and domain in _PROMPT_COST_DOMAINS:
            span_c, source = fallback_span, "llama-log"
        if span_q is None or span_c is None or (span_q + span_c) == 0:
            print(f"  {domain:<10} 測定不能 (span_Q={span_q} span_C={span_c})")
            continue
        any_row = True
        print(f"  {domain:<10} span_Q={span_q:.4f} span_C={span_c:.4f} "
              f"({source}) → w*={span_q / (span_q + span_c):.3f}")
    if not any_row:
        print("  (データ不足。稼働してコスト付き経験を貯めてから再実行する)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
