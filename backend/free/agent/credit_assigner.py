"""ステップ単位のクレジット割当

MDP トレースの各ステップに対し、最終結果からの貢献度を
逆伝播的に推定する。

参考: Agent Lightning (arXiv:2508.03680)
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.free.agent.agent_tracer import MDPStep
from backend.log_config import get_logger

logger = get_logger("agent.credit")

# 時間減衰率: 後のステップほど結果に近い → 重み大
DEFAULT_DECAY = 0.8


@dataclass
class StepCredit:
    """1ステップの貢献度"""

    step_index: int
    action: str
    credit: float


def assign_credit(
    steps: list[MDPStep],
    final_outcome: float,
    *,
    decay: float = DEFAULT_DECAY,
) -> list[StepCredit]:
    """各ステップの貢献度を推定

    アルゴリズム:
    1. 各ステップに時間減衰重み（後方ほど大）を付与
    2. ステップ報酬 × 重みの加重和を正規化
    3. final_outcome で全体をスケーリング

    Args:
        steps: エピソード内の全ステップ
        final_outcome: 最終結果の評価値 (0.0〜1.0)
        decay: 時間減衰率（0 < decay < 1）

    Returns:
        各ステップの貢献度リスト
    """
    if not steps:
        return []

    n = len(steps)

    # 単一ステップの場合は退化ケース
    if n == 1:
        return [StepCredit(
            step_index=steps[0].step_index,
            action=steps[0].action,
            credit=round(steps[0].reward * final_outcome, 3),
        )]

    # 時間減衰重み: 後のステップほど大きい
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_weight = sum(weights)

    credits: list[StepCredit] = []
    for step, w in zip(steps, weights):
        # 局所 reward × 正規化重み × final_outcome
        raw = step.reward * (w / total_weight) * final_outcome
        credits.append(StepCredit(
            step_index=step.step_index,
            action=step.action,
            credit=round(raw, 3),
        ))

    logger.debug(
        "Credit assigned: %d steps, outcome=%.2f, credits=%s",
        n, final_outcome,
        [(c.action, c.credit) for c in credits],
    )

    return credits


def compute_final_outcome(
    tasks_total: int,
    tasks_done: int,
    tasks_failed: int,
) -> float:
    """タスク完了状況から最終結果の評価値を算出

    Args:
        tasks_total: 全タスク数
        tasks_done: 成功タスク数
        tasks_failed: 失敗タスク数

    Returns:
        0.0〜1.0 の評価値
    """
    if tasks_total == 0:
        return 0.0

    success_rate = tasks_done / tasks_total
    # 失敗ペナルティ: 失敗が多いほどスコアを下げる
    penalty = tasks_failed / tasks_total * 0.5

    return round(max(0.0, min(1.0, success_rate - penalty)), 3)
