"""NullReranker — 未設定時の no-op リランカー

reranker セクション未定義時に自動使用され、入力をそのまま返す。
"""


class NullReranker:
    """未設定時の no-op リランカー（入力をそのまま返す）"""

    @property
    def is_active(self) -> bool:
        """NullReranker は常に無効"""
        return False

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        *,
        mode: str = "chat",
    ) -> list[tuple[int, float]]:
        """入力順序をそのまま返す（リランキングなし）。

        ``mode`` は Protocol 互換性のために受けるが NullReranker
        では使用しない。
        """
        return [(i, 1.0) for i in range(min(top_n, len(documents)))]
