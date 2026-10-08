"""一対比較の採用ゲートの判定 (f_04 §4.5 / §4.7)。

プロンプトの採用ゲート (``LearningScheduler._measure_pairwise``) と生成パラメータの
実測ゲート (``GenerationParamEvolver.evolve_measured``) が同じ不等式で採否を
決めるための純関数。勝敗は ``{case_id: +1 | -1 | 0}`` (候補の勝ち / 負け / 引き分け)。
"""

from __future__ import annotations

from collections.abc import Mapping

#: 純勝ちが下限と雑音フロアの両方を超えた (標本の無退行検査へ進む / 採用)
PAIRWISE_NET_WINS = "pairwise_net_wins"
#: 純勝ちは下限に届いたが雑音フロアを超えない
WITHIN_NOISE_FLOOR = "within_noise_floor"
#: 純勝ちが下限に届かない
NO_PAIRWISE_GAIN = "no_pairwise_gain"


def tally(outcomes: Mapping[str, int]) -> tuple[int, int, int]:
    """``(勝ち, 負け, 引き分け)`` の件数。"""
    wins = sum(1 for v in outcomes.values() if v > 0)
    losses = sum(1 for v in outcomes.values() if v < 0)
    return wins, losses, len(outcomes) - wins - losses


def net_wins(outcomes: Mapping[str, int]) -> int:
    """純勝ち (勝ち − 負け)。"""
    wins, losses, _ = tally(outcomes)
    return wins - losses


def noise_floor(canary: Mapping[str, int]) -> int:
    """同じ構成同士 (カナリア) の |純勝ち|。"""
    return abs(net_wins(canary))


def pairwise_verdict(net: int, noise: int, min_net_wins: int) -> str:
    """純勝ちが ``min_net_wins`` 以上 **かつ** 雑音フロアを超えれば ``PAIRWISE_NET_WINS``。"""
    if net >= min_net_wins and net > noise:
        return PAIRWISE_NET_WINS
    return WITHIN_NOISE_FLOOR if net >= min_net_wins else NO_PAIRWISE_GAIN
