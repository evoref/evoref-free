"""

EvorefGen (llm / rag / generation pillar) が他 pillar に公開する補助タスク
モデル呼び出しの契約を型として固定する。全エディションで
`backend.free.llm.aux_client.AuxClient` (ローカル llama-server)
が本 Protocol を満たす
Pro 実装は存在しない。

設計原則 (CLAUDE.md §8 / `docs/f_02_memory_system.md` §6.5):
- 最小 API 原則: ``generate`` と ``health_check`` のみ
- 実装は duck typing で Protocol を満たす
- Protocol ファイルは他 pillar を import しない
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AuxClientProtocol(Protocol):
    """補助タスク呼び出しの抽象。

    メモリ / RAG / 要約 / 判定など「チャット応答パスから切り離された
    補助タスク処理」で使われる LLM クライアントの共通契約。
    ベースモデル (メイン応答生成) 用クライアントとは別インスタンス。

    最小 API:
    - ``generate(messages, ...) -> dict``: OpenAI 互換レスポンス dict
      (``{"choices": [{"message": {"content": "..."}}]}``) を返す
    - ``health_check() -> bool``: バックエンド可用性チェック
    """

    async def generate(
        self,
        messages: list[dict],
        **kwargs: object,
    ) -> dict:
        """補助タスクで推論を実行し、OpenAI 互換レスポンス dict を返す。

        ``kwargs`` には ``temperature`` / ``max_tokens`` / ``timeout`` /
        ``purpose`` など、実装依存のオプションが含まれる。Protocol としては
        ``messages`` 以外を固定せず、呼び出し側はキーワード指定する。
        """
        ...

    async def health_check(self) -> bool:
        """補助タスクバックエンドが疎通可能かを返す。"""
        ...


__all__ = ["AuxClientProtocol"]
