"""ExplorationController: 探索/活用バランスの適応制御

fitness 安定度に基づいて変異スケール（σ）を自動調整し、
初期は多様な変異（探索）、成熟後は微調整（活用）に自動移行する。

LLM 不要。numpy のみ。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.io.codec import CodecError, codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import JsonPayload, VersionedJsonFile
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


@persisted()
@dataclass
class ExplorationState:
    """1 つの ``(domain, mode)`` の探索状態 (ペイロードの ``"<domain>:<mode>"`` の値)。"""

    sigma: float = SIGMA_MAX
    phase: str = "explore"
    _extra: dict[str, Any] | None = None


#: ペイロードは ``{"<domain>:<mode>": ExplorationState}``。``:`` を含まないキーは
#: この版が知らないトップのキーとして原形のまま書き戻す。
EXPLORATION_FORMAT = register_format(FormatSpec(
    format_id="learning.exploration",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/prompts/exploration_state.json",
    retention="one per partition",
    export=True,
    records=(ExplorationState,),
))


class ExplorationController(VersionedJsonFile):
    """探索/活用バランスの適応制御

    PolicyEvolver および Darwinian Evolver の変異スケール（σ）を
    fitness 履歴に基づいて自動調整する。

    - 探索モード（σ 大）: 初期 or fitness 不安定 or 急低下
    - 活用モード（σ 小）: fitness 安定・連続改善中
    - 遷移モード: 中間（分散に基づく線形補間）
    """

    FORMAT = EXPLORATION_FORMAT
    _state_logger = logger

    def __init__(self, debug_logger: DebugLogger | None = None) -> None:
        self._debug_logger = debug_logger
        # (domain, mode) → 探索状態
        self._state: dict[tuple[str, str], ExplorationState] = {}
        # 読んだファイルの ``domain:mode`` でないトップのキー (書き戻しで戻す)。
        self._payload_extra: dict[str, Any] = {}

    def reset(self) -> None:
        """全 (domain, mode) の探索状態を初期化する (パーティション切替用)。"""
        self._state.clear()
        self._payload_extra = {}

    def get_mutation_scale(self, domain: str, mode: str) -> float:
        """現在の変異スケール（σ）を返す

        初期状態（履歴なし）は SIGMA_MAX（探索モード）を返す。
        """
        state = self._state.get((domain, mode))
        return SIGMA_MAX if state is None else state.sigma

    def get_phase(self, domain: str, mode: str) -> str:
        """現在のフェーズを返す（"explore" | "exploit" | "transition" | "explore_reset"）"""
        state = self._state.get((domain, mode))
        return "explore" if state is None else state.phase

    def update(
        self,
        domain: str,
        mode: str,
        fitness_history: list[float],
    ) -> None:
        """fitness 履歴から σ を更新する

        Args:
            domain: ポリシードメイン
            mode: "chat" | "create"
            fitness_history: そのドメイン・モードの fitness 値の時系列
        """
        key = (domain, mode)

        if len(fitness_history) < 2:
            self._set(key, SIGMA_MAX, "explore")
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

        self._set(key, sigma, phase)
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

    def _set(self, key: tuple[str, str], sigma: float, phase: str) -> None:
        """状態を更新する (読んだ状態の未知キーは残す)。"""
        state = self._state.get(key)
        if state is None:
            self._state[key] = ExplorationState(sigma=sigma, phase=phase)
        else:
            state.sigma, state.phase = sigma, phase

    def get_status(self) -> dict[str, dict]:
        """全ドメイン・モードの状態を返す"""
        return {
            f"{d}:{m}": {"sigma": s.sigma, "phase": s.phase}
            for (d, m), s in self._state.items()
        }

    # ── 永続化 (VersionedJsonFile) ──

    def _to_payload(self) -> JsonPayload:
        codec = codec_for(ExplorationState)
        return {
            **{f"{d}:{m}": codec.encode(s) for (d, m), s in self._state.items()},
            **self._payload_extra,
        }

    def _from_payload(self, payload: JsonPayload) -> None:
        if not isinstance(payload, dict):
            raise TypeError(
                f"exploration_state.json must be a dict, "
                f"got {type(payload).__name__}"
            )
        codec = codec_for(ExplorationState)
        state: dict[tuple[str, str], ExplorationState] = {}
        extra: dict[str, Any] = {}
        skipped = 0
        for key_str, value in payload.items():
            domain, sep, mode = key_str.partition(":")
            if not sep:
                extra[key_str] = value
                continue
            try:
                state[(domain, mode)] = codec.decode(value)
            except CodecError:
                skipped += 1
        if skipped:
            logger.warning("Skipped %d unreadable exploration state(s)", skipped)
        self._state = state
        self._payload_extra = extra

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
