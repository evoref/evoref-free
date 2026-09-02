"""補助タスク判定 (content gate 等) のセッション / クエリ単位カウンタ

チャット応答パスで補助タスク LLM を発火させる箇所が、セッション内累計と
クエリ単位の発火回数を抑制するためのトラッカー。

``AppState.judge_tracker`` に 1 個保持し、呼出元 (現在は
``ChunkContentGate``、namespace ``"content_gate"``) が ``session_id`` と
**自前で組んだ quota dict** (``enabled`` / ``max_per_session`` /
``max_per_query`` / ``only_when_quality``) を渡して許容可否を問い合わせる。
かつて想定していた ``rag.self_rag.quality_judge`` セクションは config
スキーマ (``SelfRagConfig``) が拒否するため存在しない — 補助タスクによる
品質再判定そのものが撤去済みで、Self-RAG の品質判定は純ルールのみ。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from backend.log_config import get_logger

logger = get_logger("rag.judge_tracker")


@dataclass(frozen=True)
class AuxJudgeDecision:
    """補助タスク判定の発火許可判定の結果。

    Attributes:
        allowed: True なら発火可、False なら skip すべき。
        reason: skip 理由 (``session_cap`` / ``query_cap`` / ``disabled``
            / ``quality_not_applicable``)。``allowed=True`` なら
            空文字列。DebugLogger の ``judge_skipped_reason``
            に転記する。
        session_count: 判定時点でのセッション累計発火回数 (発火前の値)。
        query_count: 判定時点でのクエリ内発火回数 (発火前の値)。
    """

    allowed: bool
    reason: str
    session_count: int
    query_count: int


class JudgeUsageTracker:
    """セッション単位の補助タスク判定の発火回数をカウントする。

    スレッドセーフに実装し、複数並列リクエストから同じ session_id に
    対する許可判定が衝突しても確定した数値で判定する。セッション
    切替時は ``reset_session`` で明示クリアする (chat_service が
    WorkingMemory.clear と同じタイミングで呼ぶ)。

    カウンタは ``(session_id, namespace)`` の組でキー化する。namespace は
    呼出元 purpose を表す文字列 (``"necessity"`` / ``"quality"`` /
    ``"content_gate"`` 等) で、purpose 間の発火予算が独立するようにする。
    以前は session_id のみをキーにしており、よく発火する purpose (例:
    necessity) が他の purpose (quality/content_gate) の予算を意図せず
    食い潰すバグがあった。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_counts: dict[tuple[str, str], int] = {}

    def check(
        self,
        *,
        session_id: str,
        namespace: str,
        quality: str,
        query_count: int,
        config: dict | None,
    ) -> AuxJudgeDecision:
        """発火許可判定 (カウントは増やさない)。

        Args:
            session_id: チャットセッション識別子。``WorkingMemory.session_id``
                相当。空文字列でもキー化する (``"default"`` セッションなど)。
            namespace: 呼出元 purpose を識別する文字列 (``"necessity"`` /
                ``"quality"`` / ``"content_gate"`` 等)。予算はこの単位で
                独立に管理される。
            quality: 呼出元が ``only_when_quality`` と突き合わせる値。
                content gate は ``"content_gate"`` 固定で渡す。含まれない
                場合は ``quality_not_applicable`` で skip。
            query_count: 現在のクエリ内で既に発火した回数 (呼出元が管理)。
            config: 呼出元が組む quota dict (``enabled`` / ``max_per_session``
                / ``max_per_query`` / ``only_when_quality``)。``None`` /
                欠落時はデフォルト値 (enabled=True, session=5, query=1,
                only=["medium"]) を適用する。
        """
        cfg = config or {}
        if not cfg.get("enabled", True):
            return AuxJudgeDecision(
                allowed=False,
                reason="disabled",
                session_count=self._peek(session_id, namespace),
                query_count=query_count,
            )

        only = cfg.get("only_when_quality", ["medium"]) or []
        if quality not in only:
            return AuxJudgeDecision(
                allowed=False,
                reason="quality_not_applicable",
                session_count=self._peek(session_id, namespace),
                query_count=query_count,
            )

        max_per_query = int(cfg.get("max_per_query", 1))
        if 0 < max_per_query <= query_count:
            return AuxJudgeDecision(
                allowed=False,
                reason="query_cap",
                session_count=self._peek(session_id, namespace),
                query_count=query_count,
            )

        max_per_session = int(cfg.get("max_per_session", 5))
        session_count = self._peek(session_id, namespace)
        if 0 < max_per_session <= session_count:
            return AuxJudgeDecision(
                allowed=False,
                reason="session_cap",
                session_count=session_count,
                query_count=query_count,
            )

        return AuxJudgeDecision(
            allowed=True,
            reason="",
            session_count=session_count,
            query_count=query_count,
        )

    def record(self, session_id: str, namespace: str) -> int:
        """発火を記録する。戻り値は記録後の累計。"""
        with self._lock:
            key = (session_id, namespace)
            new_count = self._session_counts.get(key, 0) + 1
            self._session_counts[key] = new_count
            return new_count

    def refund(self, session_id: str, namespace: str) -> int:
        """``record`` を取り消す (0 未満にはしない)。戻り値は取消後の累計。

        タイムアウト等のインフラ的失敗で判定が完了しなかった場合、事前に
        ``record`` した予約分を呼出元が払い戻すために使う。
        """
        with self._lock:
            key = (session_id, namespace)
            new_count = max(0, self._session_counts.get(key, 0) - 1)
            self._session_counts[key] = new_count
            return new_count

    def reset_session(self, session_id: str) -> None:
        """セッション切替時に全 namespace のカウンタをクリアする。"""
        with self._lock:
            for key in [k for k in self._session_counts if k[0] == session_id]:
                del self._session_counts[key]

    def get_session_count(self, session_id: str, namespace: str) -> int:
        """デバッグ / テスト用: セッション累計を取得する。"""
        return self._peek(session_id, namespace)

    def _peek(self, session_id: str, namespace: str) -> int:
        with self._lock:
            return self._session_counts.get((session_id, namespace), 0)
