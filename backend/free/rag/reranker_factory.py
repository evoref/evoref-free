"""リランカーバックエンドのファクトリ関数

config.yaml の reranker セクションからバックエンドを生成する。
reranker セクション未定義時は NullReranker（自動 SKIP）を返す。
起動時にヘルスチェックを行い、失敗時は LazyReranker で定期再試行する。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from backend.free.rag.reranker_backend import RerankerBackend
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("rag.reranker_factory")

# ヘルスチェック再試行間隔（秒）
_RETRY_INTERVAL_SEC = 300  # 5分


class LazyReranker:
    """起動時にヘルスチェック失敗した場合の遅延再接続リランカー

    定期的にヘルスチェックを再試行し、成功したら実際の LlamaCppReranker に委譲する。
    再試行までの間は NullReranker と同じ動作（no-op）。
    """

    def __init__(self, reranker_cfg: dict, debug_logger: DebugLogger | None = None):
        self._reranker_cfg = reranker_cfg
        self._debug_logger = debug_logger
        self._inner: RerankerBackend | None = None
        self._last_retry: float = 0.0

    @property
    def is_active(self) -> bool:
        """内部リランカーが接続済みの場合のみ有効"""
        return self._inner is not None and self._inner.is_active

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        *,
        mode: str = "chat",
    ) -> list[tuple[int, float]]:
        """リランキング（未接続時は自動再試行後、no-op フォールバック）。

        ``mode`` は内部リランカーへ素通り
        """
        if self._inner is None:
            await self._try_reconnect()

        if self._inner is not None:
            return await self._inner.rerank(query, documents, top_n, mode=mode)

        # no-op: 入力順序をそのまま返す
        return [(i, 1.0) for i in range(min(top_n, len(documents)))]

    async def _try_reconnect(self) -> None:
        """再試行間隔を守りつつヘルスチェックを実行"""
        now = time.monotonic()
        if now - self._last_retry < _RETRY_INTERVAL_SEC:
            return
        self._last_retry = now

        reranker = _create_llamacpp_reranker(self._reranker_cfg, self._debug_logger)
        if hasattr(reranker, "health_check"):
            healthy = await reranker.health_check()
            if healthy:
                self._inner = reranker
                logger.info("Reranker reconnected successfully")
            else:
                await reranker.aclose()
                logger.debug("Reranker retry: health check still failing")
        else:
            self._inner = reranker

    async def health_check(self) -> bool:
        """内部リランカーのヘルスチェック（未接続時は再試行を試みる）"""
        if self._inner is None:
            await self._try_reconnect()
        if self._inner is not None and hasattr(self._inner, "health_check"):
            return await self._inner.health_check()
        return False

    async def aclose(self) -> None:
        """内部リランカーを閉じる"""
        if self._inner is not None and hasattr(self._inner, "aclose"):
            await self._inner.aclose()
            self._inner = None


async def create_reranker_backend_async(cfg: dict, debug_logger: DebugLogger | None = None) -> RerankerBackend:
    """config.yaml の reranker セクションからバックエンドを生成（async 版）

    ヘルスチェック付き: Reranker サーバーが応答しない場合は LazyReranker で
    定期的に再試行する。
    """
    reranker_cfg = cfg.get("reranker")
    if not reranker_cfg or not reranker_cfg.get("enabled", False):
        from backend.free.rag.reranker_null import NullReranker

        logger.info("Reranker not configured, using NullReranker (auto SKIP)")
        return NullReranker()

    reranker = _create_llamacpp_reranker(reranker_cfg, debug_logger)

    # 起動レース対策: reranker llama-server プロセスが listen するまで
    # /health をポーリング (base/assist と同じパターン)。失敗しても
    # 続行 → 既存の LazyReranker フォールバックに倒れる
    from backend.free.llm._base_client import wait_for_server_ready
    host = reranker_cfg.get("host", "localhost")
    port = reranker_cfg.get("port", 8083)
    await wait_for_server_ready(
        f"http://{host}:{port}/health",
        label="llama-server (reranker)",
    )

    # ヘルスチェック: サーバーが起動していなければ LazyReranker で遅延再接続
    if hasattr(reranker, "health_check"):
        healthy = await reranker.health_check()
        if not healthy:
            logger.warning(
                "Reranker health check failed, using LazyReranker "
                "(will retry every %ds)",
                _RETRY_INTERVAL_SEC,
            )
            await reranker.aclose()
            return LazyReranker(reranker_cfg, debug_logger)

    # セルフテスト: /health は通るが /v1/rerank が壊れているケースを検出
    if hasattr(reranker, "selftest"):
        if not await reranker.selftest():
            logger.warning(
                "Reranker selftest failed (health OK but /v1/rerank broken), "
                "using LazyReranker (will retry every %ds)",
                _RETRY_INTERVAL_SEC,
            )
            await reranker.aclose()
            return LazyReranker(reranker_cfg, debug_logger)

    return reranker


def _create_llamacpp_reranker(reranker_cfg: dict, debug_logger: DebugLogger | None = None) -> RerankerBackend:
    """LlamaCppReranker インスタンスを生成"""
    backend = reranker_cfg.get("backend", "llama-cpp")

    match backend:
        case "llama-cpp":
            from backend.free.rag.reranker_llamacpp import LlamaCppReranker

            host = reranker_cfg.get("host", "localhost")
            port = reranker_cfg.get("port", 8083)
            model_name = reranker_cfg.get("model_name", "reranker")
            timeout = reranker_cfg.get("timeout", 30.0)

            max_doc_chars = reranker_cfg.get("max_doc_chars", 4096)
            fast_path_timeout = reranker_cfg.get("fast_path_timeout", 8.0)
            cb_failure_threshold = reranker_cfg.get("cb_failure_threshold", 5)
            cb_cooldown_sec = reranker_cfg.get("cb_cooldown_sec", 30.0)
            # instruction-aware リランカー (Qwen3 等) 用 instructions。schema 既定値が入る
            instructions = reranker_cfg.get("instructions", {})
            # クエリ整形テンプレート。schema 既定値は Qwen3 仕様
            # (``"<Instruct>: {task}\n<Query>: {query}"``)。
            # 空文字列で素通り (BGE-Reranker-v2-m3 等)。
            query_template = reranker_cfg.get(
                "query_template", "<Instruct>: {task}\n<Query>: {query}",
            )

            reranker = LlamaCppReranker(
                host=host,
                port=port,
                model_name=model_name,
                timeout=timeout,
                max_doc_chars=max_doc_chars,
                fast_path_timeout=fast_path_timeout,
                cb_failure_threshold=cb_failure_threshold,
                cb_cooldown_sec=cb_cooldown_sec,
                instructions=instructions,
                query_template=query_template,
                debug_logger=debug_logger,
            )
            logger.info(
                "Created LlamaCppReranker: %s:%d (model=%s, instruction_modes=%s, "
                "query_template=%r)",
                host, port, model_name, sorted(instructions.keys()),
                query_template,
            )
            return reranker

        case _:
            raise ValueError(f"Unknown reranker backend: {backend}")
