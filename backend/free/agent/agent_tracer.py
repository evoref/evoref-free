"""MDP トレース構造化ログ

エージェントのマルチステップ実行を MDP（マルコフ決定過程）の
エピソード/ステップ形式で構造化ログに記録する。

参考: Agent Lightning (arXiv:2508.03680)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("agent.tracer")


@dataclass
class MDPStep:
    """MDP の1ステップ"""

    step_index: int
    state: dict
    action: str
    observation: str
    reward: float
    timestamp: float = field(default_factory=time.time)


class AgentTracer:
    """エージェント実行を MDP エピソード形式で記録

    デバッグモード時に agent_trace.jsonl へ JSONL 出力し、
    同時にインメモリでステップを保持してクレジット割当に利用する。
    """

    def __init__(self, debug_logger: DebugLogger | None = None) -> None:
        self._debug_logger = debug_logger
        self._episodes: dict[str, list[MDPStep]] = {}

    def begin_episode(self, conversation_id: str, mode: str) -> str:
        """エピソードを開始し episode_id を返す"""
        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        self._episodes[episode_id] = []

        self._log({
            "event": "begin",
            "episode_id": episode_id,
            "conversation_id": conversation_id,
            "mode": mode,
            "timestamp": time.time(),
        })

        logger.debug(
            "Episode started: %s (conversation=%s, mode=%s)",
            episode_id, conversation_id, mode,
        )
        return episode_id

    def record_step(self, episode_id: str, step: MDPStep) -> None:
        """エピソードにステップを記録"""
        if episode_id not in self._episodes:
            logger.warning("Unknown episode_id: %s", episode_id)
            return

        self._episodes[episode_id].append(step)

        self._log({
            "event": "step",
            "episode_id": episode_id,
            **asdict(step),
        })

        logger.debug(
            "Step recorded: ep=%s step=%d action=%s reward=%.2f",
            episode_id, step.step_index, step.action, step.reward,
        )

    def end_episode(self, episode_id: str, outcome: str) -> None:
        """エピソードを終了"""
        self._log({
            "event": "end",
            "episode_id": episode_id,
            "outcome": outcome,
            "total_steps": len(self._episodes.get(episode_id, [])),
            "timestamp": time.time(),
        })

        logger.debug("Episode ended: %s outcome=%s", episode_id, outcome)

    def get_steps(self, episode_id: str) -> list[MDPStep]:
        """指定エピソードの全ステップを取得（クレジット割当用）"""
        return list(self._episodes.get(episode_id, []))

    def cleanup_episode(self, episode_id: str) -> None:
        """エピソードのインメモリデータを破棄"""
        self._episodes.pop(episode_id, None)

    def _log(self, data: dict) -> None:
        """DebugLogger 経由で agent_trace.jsonl に書き込み"""
        dl = self._debug_logger
        if dl is not None:
            dl.log_agent_trace_event(data)
