"""Self-RAG assist_judge のセッション / クエリ単位カウンタ

ルールベース Self-RAG の marginal 判定 (既定 quality="medium") 時に
アシストモデル LLM による品質再判定を発火させる際、
セッション内累計とクエリ単位の発火回数を抑制するためのトラッカー。

``AppState.assist_judge_tracker`` に 1 個保持し、``search_pipeline``
が ``session_id`` と config (``rag.self_rag.assist_judge``) を渡して
許容可否を問い合わせる。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from backend.log_config import get_logger

logger = get_logger("rag.assist_judge_tracker")


@dataclass(frozen=True)
class AssistJudgeDecision:
    """assist_judge 発火許可判定の結果。

    Attributes:
        allowed: True なら発火可、False なら skip すべき。
        reason: skip 理由 (``session_cap`` / ``query_cap`` / ``disabled``
            / ``quality_not_applicable``)。``allowed=True`` なら
            空文字列。DebugLogger の ``assist_judge_skipped_reason``
            に転記する。
        session_count: 判定時点でのセッション累計発火回数 (発火前の値)。
        query_count: 判定時点でのクエリ内発火回数 (発火前の値)。
    """

    allowed: bool
    reason: str
    session_count: int
    query_count: int


class AssistJudgeUsageTracker:
    """セッション単位の assist_judge 発火回数をカウントする。

    スレッドセーフに実装し、複数並列リクエストから同じ session_id に
    対する許可判定が衝突しても確定した数値で判定する。セッション
    切替時は ``reset_session`` で明示クリアする (chat_service が
    WorkingMemory.clear と同じタイミングで呼ぶ)。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_counts: dict[str, int] = {}

    def check(
        self,
        *,
        session_id: str,
        quality: str,
        query_count: int,
        config: dict | None,
    ) -> AssistJudgeDecision:
        """発火許可判定 (カウントは増やさない)。

        Args:
            session_id: チャットセッション識別子。``WorkingMemory.session_id``
                相当。空文字列でもキー化する (``"default"`` セッションなど)。
            quality: ルールベース Self-RAG の判定結果 ("high"/"medium"/"low")。
                ``only_when_quality`` に含まれない場合は ``disabled``
                相当の skip にする。
            query_count: 現在のクエリ内で既に発火した回数。
                ``_maybe_assist_judge_quality`` 側で整備する。
            config: ``rag.self_rag.assist_judge`` セクションの dict。
                ``None`` / 欠落時はデフォルト値 (enabled=True, session=5,
                query=1, only=["medium"]) を適用する。
        """
        cfg = config or {}
        if not cfg.get("enabled", True):
            return AssistJudgeDecision(
                allowed=False,
                reason="disabled",
                session_count=self._peek(session_id),
                query_count=query_count,
            )

        only = cfg.get("only_when_quality", ["medium"]) or []
        if quality not in only:
            return AssistJudgeDecision(
                allowed=False,
                reason="quality_not_applicable",
                session_count=self._peek(session_id),
                query_count=query_count,
            )

        max_per_query = int(cfg.get("max_per_query", 1))
        if 0 < max_per_query <= query_count:
            return AssistJudgeDecision(
                allowed=False,
                reason="query_cap",
                session_count=self._peek(session_id),
                query_count=query_count,
            )

        max_per_session = int(cfg.get("max_per_session", 5))
        session_count = self._peek(session_id)
        if 0 < max_per_session <= session_count:
            return AssistJudgeDecision(
                allowed=False,
                reason="session_cap",
                session_count=session_count,
                query_count=query_count,
            )

        return AssistJudgeDecision(
            allowed=True,
            reason="",
            session_count=session_count,
            query_count=query_count,
        )

    def record(self, session_id: str) -> int:
        """発火を記録する。戻り値は記録後の累計。"""
        with self._lock:
            new_count = self._session_counts.get(session_id, 0) + 1
            self._session_counts[session_id] = new_count
            return new_count

    def reset_session(self, session_id: str) -> None:
        """セッション切替時にカウンタをクリアする。"""
        with self._lock:
            self._session_counts.pop(session_id, None)

    def get_session_count(self, session_id: str) -> int:
        """デバッグ / テスト用: セッション累計を取得する。"""
        return self._peek(session_id)

    def _peek(self, session_id: str) -> int:
        with self._lock:
            return self._session_counts.get(session_id, 0)
