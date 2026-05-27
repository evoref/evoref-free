"""RerankerBackend Protocol 定義

リランカーバックエンドの抽象インターフェース。
LlamaCppReranker / NullReranker がこの Protocol に準拠する。

Qwen3-Reranker 系は instruction-aware であり、``mode``
(``chat`` / ``coding``) によって異なる task description を渡せる。
"""

from typing import Protocol, runtime_checkable


# モード未指定時のデフォルト
DEFAULT_MODE = "chat"


@runtime_checkable
class RerankerBackend(Protocol):
    """リランカーバックエンドの抽象インターフェース"""

    @property
    def is_active(self) -> bool:
        """リランカーが有効か（NullReranker: False, LlamaCppReranker: True）"""
        ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        *,
        mode: str = DEFAULT_MODE,
    ) -> list[tuple[int, float]]:
        """ドキュメントをクエリとの関連性でリランキング

        Args:
            query: 検索クエリ
            documents: リランキング対象のドキュメントテキストリスト
            top_n: 返却する上位件数
            mode: ``chat`` / ``coding`` のいずれか。Qwen3-Reranker の
                ``<Instruct>: {task}`` で渡す task description を切替える。

        Returns:
            [(元インデックス, スコア), ...] top_n 件、スコア降順
        """
        ...
