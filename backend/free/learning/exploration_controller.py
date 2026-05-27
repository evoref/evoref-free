"""ExplorationController: 探索/活用バランスの適応制御

fitness 安定度に基づいて変異スケール（σ）を自動調整し、
初期は多様な変異（探索）、成熟後は微調整（活用）に自動移行する。

LLM 不要。numpy のみ。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backend.free.learning.json_state_store import JsonPayload, JsonStateStore
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("learning.exploration_controller")

# σ の上下限
SIGMA_MAX: float = 0.15
SIGMA_MIN: float = 0.01

# fitness 分散の閾値（これ以上ならまだ収束していない → 探索モード）
VARIANCE_THRESHOLD: float = 0.01

# 連続改善回数がこの閾値以上で活用モードへ移行
CONSECUTIVE_IMPROVEMENT_N: int = 3

# fitness 急低下の閾値（前回から DROP_THRESHOLD 以上下がったらリセット）
DROP_THRESHOLD: float = 0.1

# 直近の fitness 履歴の窓サイズ
WINDOW_SIZE: int = 5


class ExplorationController(JsonStateStore):
    """探索/活用バランスの適応制御

    PolicyEvolver および Darwinian Evolver の変異スケール（σ）を
    fitness 履歴に基づいて自動調整する。

    - 探索モード（σ 大）: 初期 or fitness 不安定 or 急低下
    - 活用モード（σ 小）: fitness 安定・連続改善中
    - 遷移モード: 中間（分散に基づく線形補間）
    """

    _state_logger = logger

    def __init__(self, debug_logger: DebugLogger | None = None) -> None:
        self._debug_logger = debug_logger
        # (domain, mode) → {"sigma": float, "phase": str}
        self._state: dict[tuple[str, str], dict] = {}

    def get_mutation_scale(self, domain: str, mode: str) -> float:
        """現在の変異スケール（σ）を返す

        初期状態（履歴なし）は SIGMA_MAX（探索モード）を返す。
        """
        key = (domain, mode)
        state = self._state.get(key)
        if state is None:
            return SIGMA_MAX
        return state.get("sigma", SIGMA_MAX)

    def get_phase(self, domain: str, mode: str) -> str:
        """現在のフェーズを返す（"explore" | "exploit" | "transition" | "explore_reset"）"""
        key = (domain, mode)
        state = self._state.get(key)
        if state is None:
            return "explore"
        return state.get("phase", "explore")

    def update(
        self,
        domain: str,
        mode: str,
        fitness_history: list[float],
    ) -> None:
        """fitness 履歴から σ を更新する

        Args:
            domain: ポリシードメイン
            mode: "chat" | "coding"
            fitness_history: そのドメイン・モードの fitness 値の時系列
        """
        key = (domain, mode)

        if len(fitness_history) < 2:
            self._state[key] = {"sigma": SIGMA_MAX, "phase": "explore"}
            return

        recent = fitness_history[-min(WINDOW_SIZE, len(fitness_history)):]
        variance = float(np.var(recent))

        # 1. fitness 急低下 → 探索モードにリセット
        if recent[-1] < recent[-2] - DROP_THRESHOLD:
            sigma = SIGMA_MAX
            phase = "explore_reset"
            logger.info(
                "Exploration reset: domain=%s, mode=%s, "
                "fitness dropped %.4f → %.4f",
                domain, mode, recent[-2], recent[-1],
            )

        # 2. 分散大 → まだ収束していない → 探索モード
        elif variance > VARIANCE_THRESHOLD:
            sigma = SIGMA_MAX
            phase = "explore"

        # 3. 連続改善 → 活用モード
        elif _consecutive_improvements(recent) >= CONSECUTIVE_IMPROVEMENT_N:
            sigma = SIGMA_MIN
            phase = "exploit"

        # 4. それ以外 → 遷移モード（分散に基づく線形補間）
        else:
            ratio = min(1.0, variance / VARIANCE_THRESHOLD)
            sigma = SIGMA_MIN + ratio * (SIGMA_MAX - SIGMA_MIN)
            phase = "transition"

        self._state[key] = {"sigma": sigma, "phase": phase}
        logger.debug(
            "Exploration updated: domain=%s, mode=%s, "
            "sigma=%.4f, phase=%s, variance=%.6f",
            domain, mode, sigma, phase, variance,
        )

        # DebugLogger に構造化ログを出力
        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=1, data={
                "component": "exploration_controller",
                "action": "update",
                "domain": domain,
                "mode": mode,
                "sigma": round(sigma, 4),
                "phase": phase,
                "variance": round(variance, 6),
                "history_len": len(fitness_history),
            })

    def get_status(self) -> dict[str, dict]:
        """全ドメイン・モードの状態を返す"""
        return {
            f"{d}:{m}": s
            for (d, m), s in self._state.items()
        }

    # ── 永続化 (JsonStateStore) ──

    def _to_payload(self) -> JsonPayload:
        return {
            f"{d}:{m}": s
            for (d, m), s in self._state.items()
        }

    def _from_payload(self, payload: JsonPayload) -> None:
        if not isinstance(payload, dict):
            raise TypeError(
                f"exploration_state.json must be a dict, "
                f"got {type(payload).__name__}"
            )
        self._state.clear()
        for key_str, state in payload.items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                self._state[(parts[0], parts[1])] = state

    def _on_save_success(self, path: Path) -> None:
        logger.debug("Exploration state saved: %s", path)

    def _on_load_success(self, path: Path) -> None:
        logger.info(
            "Exploration state loaded: %d entries from %s",
            len(self._state), path,
        )


def _consecutive_improvements(values: list[float]) -> int:
    """末尾からの連続改善回数を数える"""
    count = 0
    for i in range(len(values) - 1, 0, -1):
        if values[i] > values[i - 1]:
            count += 1
        else:
            break
    return count
