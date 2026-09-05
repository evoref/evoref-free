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
    from backend.free.agent.agent_trace_store import AgentTraceStore

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

    イベントは常設の :class:`AgentTraceStore` (``local/memory/agent_trace/``、
    エピソード記憶の入力) へ書き、develop モードでは同じものを DebugLogger の
    ``agent_trace`` JSONL にも出す (観測用)。同時にインメモリでステップを
    保持してクレジット割当に利用する。
    """

    def __init__(
        self,
        debug_logger: DebugLogger | None = None,
        trace_store: AgentTraceStore | None = None,
    ) -> None:
        self._debug_logger = debug_logger
        self._trace_store = trace_store
        self._episodes: dict[str, list[MDPStep]] = {}
        #: episode_id → conversation_id。例外 / キャンセルで ``end_episode`` に
        #: 到達しなかったエピソードを ``abort_open_episodes`` で閉じるための索引。
        self._conversation_of: dict[str, str] = {}
        #: private エピソードの ID 集合。private 判定はリクエスト単位の
        #: contextvar で executor 境界を越えると落ちるため、エピソード単位で
        #: 保持し **全イベントに印を打つ** (2026-09-05 監査)。
        self._private_episodes: set[str] = set()

    def begin_episode(
        self, conversation_id: str, mode: str, *, private: bool = False,
    ) -> str:
        """エピソードを開始し episode_id を返す

        ``private`` はリクエストの private フラグ。begin イベントに刻み、
        MDP ingest (``mdp_ingester``) が STM の private ノートの残存に依らず
        当該エピソードをエピソード記憶へ昇格させないための一次情報にする。
        """
        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        self._episodes[episode_id] = []
        self._conversation_of[episode_id] = conversation_id

        event: dict = {
            "event": "begin",
            "episode_id": episode_id,
            "conversation_id": conversation_id,
            "mode": mode,
            "timestamp": time.time(),
        }
        if private:
            self._private_episodes.add(episode_id)
            event["private"] = True
        self._log(event)

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
        self._conversation_of.pop(episode_id, None)
        self._private_episodes.discard(episode_id)

    def abort_open_episodes(
        self, conversation_id: str, outcome: str = "failure: aborted",
    ) -> int:
        """``conversation_id`` の未終了エピソードを ``outcome`` で閉じて破棄する。

        呼出側が例外 / タイムアウト / 切断で ``end_episode`` に届かなかった
        場合の後始末。``end`` が無いエピソードは MDP ingest の保留に永久滞留し、
        インメモリの ``_episodes`` も増え続ける。正常終了後に呼んでも no-op。
        """
        open_ids = [
            ep for ep, conv in self._conversation_of.items() if conv == conversation_id
        ]
        for ep in open_ids:
            self.end_episode(ep, outcome)
            self.cleanup_episode(ep)
        return len(open_ids)

    def close(self) -> None:
        """常設ストアのファイルハンドルを閉じる (lifespan shutdown)。"""
        store = self._trace_store
        if store is not None:
            store.close()

    def _log(self, data: dict) -> None:
        """常設ストアと (develop 時は) DebugLogger の両方へ書き込み"""
        episode_id = data.get("episode_id")
        if episode_id in self._private_episodes:
            data = {**data, "private": True}
        store = self._trace_store
        if store is not None:
            store.append(data)
        dl = self._debug_logger
        if dl is not None:
            dl.log_agent_trace_event(data)
