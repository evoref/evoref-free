"""ローカル LLM (llama.cpp) クライアントのファサード

LocalClient を薄くラップし、フォアグラウンド (チャット応答) の進行中数を
``chat_in_flight()`` / ``is_serving_user`` で公開する。バックグラウンド処理
(学習サイクル / sleep-time) との協調はこの進行中数を読む側
(SleepTimeScheduler.is_user_active → LearningScheduler.should_yield) が行う。

ローカル llama-server 専用となった。アシストモデル (別 llama-server
インスタンス) は `AssistModelClient` が独立に管理する。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import math

from backend.free.llm.local_client import LocalClient
from backend.log_config import get_logger

logger = get_logger("llm.llm_client")


class LLMClient:
    """ローカル LLM (llama.cpp) のファサード。

    フォアグラウンドのチャット応答は ``chat_in_flight()`` で進行中数を
    カウントするだけ。バックグラウンド処理 (学習 / sleep-time) との協調は、
    この進行中数 (``is_serving_user``) を読む SleepTimeScheduler.is_user_active
    → LearningScheduler.should_yield の協調 yield 機構が担う
    (f_04_self_learning.md §8.1)。
    """

    def __init__(self, local: LocalClient):
        self.local = local
        # f_04 §8.1 / §8.2: フォアグラウンドのチャット進行中数
        self._in_flight_chat_count: int = 0

    @property
    def in_flight_chat_count(self) -> int:
        """現在進行中のフォアグラウンドチャット数"""
        return self._in_flight_chat_count

    @property
    def is_serving_user(self) -> bool:
        """現在ユーザーへのチャット応答を生成中かどうか (f_04 §8.1)"""
        return self._in_flight_chat_count > 0

    @asynccontextmanager
    async def chat_in_flight(self):
        """フォアグラウンドのチャット応答進行中マーカー (f_04 §8.1)

        ユーザー応答パスはこのコンテキストマネージャでくくる。フォアグラウンド
        応答は決してブロックされない。バックグラウンド処理は ``is_serving_user``
        (= in_flight_chat_count > 0) を SleepTimeScheduler.is_user_active 経由で
        見て協調 yield する。

        Usage:
            async with llm_client.chat_in_flight():
                async for token in llm_client.generate(...):
                    yield token
        """
        self._in_flight_chat_count += 1
        try:
            yield
        finally:
            self._in_flight_chat_count -= 1

    @property
    def chat_slot(self) -> int:
        """チャット用 KV キャッシュスロット"""
        return self.local.chat_slot

    @property
    def background_slot(self) -> int:
        """バックグラウンド用 KV キャッシュスロット"""
        return self.local.background_slot

    @property
    def metadata(self):
        """モデルメタデータ (ローカル LLM)"""
        return self.local.metadata

    async def generate(
        self,
        messages: list[dict],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        id_slot: int | None = None,
        request_timeout: float | None = None,
    ) -> dict | AsyncIterator[str]:
        """推論リクエスト

        Args:
            messages: メッセージリスト (OpenAI 互換形式)
            stream: ストリーミング
            temperature: 温度パラメータ
            max_tokens: 最大トークン数
            top_p: Top-P サンプリング (None で送信しない)
            top_k: Top-K サンプリング (None または 0 で送信しない)
            presence_penalty: 存在ペナルティ (None で送信しない)
            id_slot: KV キャッシュスロット
            request_timeout: 非ストリーミング呼び出し専用の per-request
                タイムアウト上書き (秒)。``LocalClient.generate`` に透過する

        Returns:
            dict (非ストリーミング) or AsyncIterator[str] (ストリーミング)
        """
        return await self.local.generate(
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            id_slot=id_slot,
            request_timeout=request_timeout,
        )

    async def generate_with_logprobs(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 256,
        id_slot: int | None = None,
    ) -> dict:
        """logprobs 付き推論

        Returns:
            {"content": str, "logprobs": list[float]}
        """
        return await self.local.generate_with_logprobs(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            id_slot=id_slot,
        )

    async def evaluate(
        self,
        response: str,
        criteria: str,
        query: str = "",
    ) -> float:
        """応答品質を評価する。

        ローカル LLM の perplexity ベースのスコアで近似する。

        Args:
            response: 評価対象の応答テキスト
            criteria: 評価基準 (本実装では未使用、互換のため残置)
            query: 元のユーザークエリ (本実装では未使用、互換のため残置)

        Returns:
            品質スコア (0.0〜1.0)
        """
        del criteria, query
        return await self._evaluate_perplexity(response)

    async def _evaluate_perplexity(self, text: str) -> float:
        """ローカル LLM で perplexity ベースのスコアを算出

        perplexity が低いほど品質が高い → 0.0〜1.0 にスケーリング。
        """
        try:
            result = await self.local.generate_with_logprobs(
                messages=[{"role": "user", "content": text}],
                temperature=0.0,
                max_tokens=256,
                id_slot=self.local.background_slot,
            )
            logprobs = result.get("logprobs", [])
            if not logprobs:
                return 0.5

            avg_log_prob = sum(logprobs) / len(logprobs)
            perplexity = math.exp(-avg_log_prob)

            # perplexity → 品質スコア変換
            # perplexity 1.0 → score 1.0, perplexity 100+ → score ~0.0
            score = 1.0 / (1.0 + math.log1p(perplexity) / 5.0)
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning("Perplexity evaluation failed: %s", e)
            return 0.5

    async def health_check(self) -> bool:
        """ローカル LLM のヘルスチェック"""
        return await self.local.health_check()

    async def aclose(self) -> None:
        """HTTP クライアントを閉じる"""
        await self.local.aclose()
