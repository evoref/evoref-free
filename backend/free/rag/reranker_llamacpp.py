"""llama-server /v1/rerank 経由のリランキング（RerankerBackend 準拠）

ポート :8083 で独立した llama-server インスタンスとして起動する。

Qwen3-Reranker 系は instruction-aware で、ネイティブフォーマット
``<Instruct>: {task}\\n<Query>: {query}\\n<Document>: {doc}`` を要求する。
``/v1/rerank`` API の ``query`` フィールドに ``<Instruct>:.../<Query>:...`` 部分を
押し込むことで、llama-server 側のチャットテンプレートに依らず instruction を
注入する (方針 A)。``mode`` (``chat`` / ``coding``) で task description を切替可能。
"""

import asyncio
import time

import httpx

from backend.free.llm._base_client import (
    BaseHTTPClient,
    async_retry_http_call,
    make_retry_logger,
)
from backend.free.rag.instruction_resolver import format_with_instruction
from backend.log_config import get_logger

logger = get_logger("rag.reranker_llamacpp")

# ファストパスタイムアウト デフォルト値: GPU ウォームアップ直後や
# 他リクエストとの競合で 3.0s では間に合わず連続タイムアウト→サーキット
# ブレーカー OPEN を招いていたため引き上げ。8.0s でも頻繁にタイムアウトし
# rerank 結果が元順序で流れて prefill 遅延を悪化させていたため、12.0s に
# 緩める (SSE keepalive と組み合わせて待機時間は許容可能)。
# config.yaml の `reranker.fast_path_timeout` でさらに調整可能。
_FAST_PATH_TIMEOUT = 12.0

# サーキットブレーカー設定 デフォルト値
_CB_FAILURE_THRESHOLD = 5     # 連続失敗回数でトリップ
_CB_COOLDOWN_SEC = 30.0       # トリップ後のクールダウン秒数

# ドキュメントトランケーション: 1 ペア (query + doc) が n_ctx を超えるのを防止。
# Qwen3-Reranker-4B は日本語テキストで 約 3 chars/token。query ~100 tokens + 特殊トークン
# を差し引いて、doc 側は ~7500 tokens ≈ 22500 chars が上限。余裕を持って 4096 chars。
_DEFAULT_MAX_DOC_CHARS = 4096

# Qwen3-Reranker 公式の instruction-aware フォーマット (既定値)
_DEFAULT_QUERY_TEMPLATE = "<Instruct>: {task}\n<Query>: {query}"

# モード未指定時のデフォルト
_DEFAULT_MODE = "chat"

# セルフテスト用定数
# セルフテストは明確に識別可能なペアを使う (1 件目が関連・2 件目が無関連)。
# 退化したリランカー (全候補スコア 0 / 無分散) を起動時に検出するため、
# 動作する reranker なら 1 件目が高スコアになる質問とドキュメントにする。
_SELFTEST_QUERY = "How do I read a file in Python?"
_SELFTEST_DOCS = [
    "Use the built-in open() function to read a file in Python.",
    "The weather in Tokyo is sunny with a high of 18 degrees today.",
]
# relevance_score がこの値未満なら実質ゼロとみなす (退化スコア判定の閾値)。
_SELFTEST_SCORE_EPSILON = 1e-6


class LlamaCppReranker(BaseHTTPClient):
    """llama-server /v1/rerank 経由のリランキング"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8083,
        model_name: str = "reranker",
        timeout: float = 30.0,
        max_doc_chars: int = _DEFAULT_MAX_DOC_CHARS,
        fast_path_timeout: float = _FAST_PATH_TIMEOUT,
        cb_failure_threshold: int = _CB_FAILURE_THRESHOLD,
        cb_cooldown_sec: float = _CB_COOLDOWN_SEC,
        instructions: dict[str, str] | None = None,
        query_template: str = _DEFAULT_QUERY_TEMPLATE,
        debug_logger=None,
    ):
        super().__init__(timeout=timeout)
        self._url = f"http://{host}:{port}/v1/rerank"
        self._model_name = model_name
        self._max_doc_chars = max_doc_chars
        self._fast_path_timeout = fast_path_timeout
        self._cb_failure_threshold = cb_failure_threshold
        self._cb_cooldown_sec = cb_cooldown_sec
        # instruction-aware モデル (Qwen3 等) 用の mode 別 task description
        self._instructions: dict[str, str] = dict(instructions or {})
        # クエリ整形テンプレート。空文字列で instruction prefix を一切付与しない
        # (BGE-Reranker-v2-m3 等の非 instruction-aware モデル運用)。
        self._query_template = query_template
        self._debug_logger = debug_logger

        # サーキットブレーカー状態
        self._consecutive_failures: int = 0
        self._cb_tripped_at: float = 0.0

        logger.info(
            "LlamaCppReranker initialized: url=%s, model=%s, max_doc_chars=%d, "
            "fast_path_timeout=%.1fs, cb_failure_threshold=%d, cb_cooldown_sec=%.1fs, "
            "instruction_modes=%s, query_template=%r",
            self._url, model_name, max_doc_chars,
            fast_path_timeout, cb_failure_threshold, cb_cooldown_sec,
            sorted(self._instructions.keys()),
            self._query_template,
        )

    def _format_query(self, query: str, mode: str) -> str:
        """テンプレート駆動でクエリを整形する

        ``query_template`` が空文字列のときは素のクエリを返す
        (BGE-Reranker-v2-m3 等の非 instruction-aware モデル向け fast-path)。
        instruction 解決とフォーマットは :mod:`instruction_resolver` に委譲する。
        """
        return format_with_instruction(
            query, self._query_template, self._instructions, mode,
            backend_label="rerank",
        )

    @property
    def is_active(self) -> bool:
        """LlamaCppReranker は常に有効"""
        return True

    def _cb_is_open(self) -> bool:
        """サーキットブレーカーがオープン（トリップ中）か判定

        クールダウン経過後は自動的にハーフオープン状態に移行する。
        """
        if self._consecutive_failures < self._cb_failure_threshold:
            return False
        elapsed = time.monotonic() - self._cb_tripped_at
        if elapsed >= self._cb_cooldown_sec:
            # ハーフオープン: 次のリクエストで復帰を試みる
            logger.info(
                "Reranker circuit breaker half-open after %.1fs cooldown", elapsed,
            )
            self._consecutive_failures = 0
            return False
        return True

    def _cb_record_success(self) -> None:
        """成功時にサーキットブレーカーをリセット"""
        self._consecutive_failures = 0

    def _cb_record_failure(self) -> None:
        """失敗時にサーキットブレーカーのカウンタを更新"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._cb_failure_threshold:
            self._cb_tripped_at = time.monotonic()
            logger.warning(
                "Reranker circuit breaker OPEN: %d consecutive failures, "
                "cooldown %.0fs",
                self._consecutive_failures, self._cb_cooldown_sec,
            )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        *,
        mode: str = _DEFAULT_MODE,
    ) -> list[tuple[int, float]]:
        """ドキュメントをクエリとの関連性でリランキング

        ファストパスタイムアウト付き: _FAST_PATH_TIMEOUT 秒以内に応答がなければ
        リランキングをスキップし、入力順序をそのまま返す。
        サーキットブレーカー付き: 連続失敗時はリクエスト自体をスキップする。

        ``mode`` (``chat``/``coding``) によって Qwen3-Reranker の
        ``<Instruct>: {task}`` を切替える。

        Returns:
            [(元インデックス, スコア), ...] top_n 件、スコア降順
        """
        if not documents:
            return []

        fallback = [(i, 1.0) for i in range(min(top_n, len(documents)))]

        # サーキットブレーカーチェック
        if self._cb_is_open():
            logger.debug("Reranker circuit breaker open: skipping rerank")
            return fallback

        # テンプレート駆動でクエリを整形する。
        # Qwen3-Reranker 既定では ``<Instruct>:.../<Query>:...`` を前置し、
        # llama-server ``/v1/rerank`` の ``query`` フィールド全体をチャット
        # テンプレート上の ``<Query>`` 位置に流し込む。BGE-Reranker 等の
        # 非 instruction-aware モデルでは ``query_template: ""`` で素通り。
        wrapped_query = self._format_query(query, mode)

        try:
            result = await asyncio.wait_for(
                self._rerank_with_retry(wrapped_query, documents, top_n),
                timeout=self._fast_path_timeout,
            )
            self._cb_record_success()
            return result
        except (asyncio.TimeoutError, httpx.TimeoutException):
            logger.warning(
                "Reranker fast-path timeout (%.1fs): skipping rerank for %d docs, "
                "returning original order",
                self._fast_path_timeout, len(documents),
            )
            self._cb_record_failure()
            return fallback
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            logger.warning(
                "Reranker request failed (%s): skipping rerank, returning original order", e,
            )
            self._cb_record_failure()
            return fallback

    async def _rerank_with_retry(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """tenacity 経由のリトライ

        共通の ``async_retry_http_call`` で 5xx (500/502/503/504) と
        transient I/O 例外をリトライする。``rerank`` 側のサーキット
        ブレーカー / fast-path timeout はリトライ枠の外側で動作する。
        """
        retry_logger = make_retry_logger(
            self._debug_logger, backend="reranker",
        )

        async def _do_rerank() -> list[tuple[int, float]]:
            return await self._rerank_impl(query, documents, top_n)

        return await async_retry_http_call(
            _do_rerank,
            request_label="Reranker request",
            retry_logger=retry_logger,
        )

    def _truncate_docs(self, documents: list[str]) -> list[str]:
        """max_doc_chars を超えるドキュメントをトランケーション"""
        limit = self._max_doc_chars
        if limit <= 0:
            return documents
        truncated = 0
        result: list[str] = []
        for doc in documents:
            if len(doc) > limit:
                result.append(doc[:limit])
                truncated += 1
            else:
                result.append(doc)
        if truncated:
            logger.debug(
                "rerank: truncated %d/%d docs to max_doc_chars=%d",
                truncated, len(documents), limit,
            )
        return result

    async def _rerank_impl(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """リランキングの実処理"""
        client = self._get_http_client()

        # ドキュメント長トランケーション — n_ctx 超過による 500 エラーを予防
        documents = self._truncate_docs(documents)

        payload = {
            "model": self._model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        logger.debug("rerank: query=%r, docs=%d, top_n=%d", query[:50], len(documents), top_n)

        t0 = time.monotonic()
        resp = await client.post(self._url, json=payload)
        # 500 エラー時にレスポンスボディを診断ログに残す
        if resp.status_code >= 500:
            body_preview = resp.text[:500] if resp.text else "(empty)"
            logger.warning(
                "Reranker %d response body: %s", resp.status_code, body_preview,
            )
        resp.raise_for_status()
        elapsed = time.monotonic() - t0
        data = resp.json()

        results = data.get("results", [])
        ranked = [
            (item["index"], item["relevance_score"])
            for item in results
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "rerank: top scores=[%s], elapsed=%.3fs",
            ", ".join(f"{s:.4f}" for _, s in ranked[:5]), elapsed,
        )

        dl = self._debug_logger
        if dl:
            dl.log_rerank_result(
                query_preview=query,
                doc_count=len(documents),
                top_scores=[s for _, s in ranked[:5]],
                elapsed_sec=elapsed,
            )

        return ranked[:top_n]

    async def health_check(self) -> bool:
        """Reranker サーバーのヘルスチェック

        起動時に呼び出し、失敗時は NullReranker にフォールバックできる。
        """
        try:
            client = self._get_http_client()
            # llama-server の /health エンドポイントで確認
            base_url = self._url.rsplit("/v1/rerank", 1)[0]
            resp = await client.get(f"{base_url}/health", timeout=5.0)
            healthy = resp.status_code == 200
            logger.debug("Reranker health check: status=%d, healthy=%s", resp.status_code, healthy)
            return healthy
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            # %r で型名を残す: TimeoutException 等は __str__ が空になり得る。
            logger.warning("Reranker health check failed: %r", e)
            return False

    async def selftest(self) -> bool:
        """起動時セルフテスト: 実際に /v1/rerank を呼び出して動作確認

        health_check (GET /health) は HTTP サーバーの生存確認のみ。
        このメソッドは POST /v1/rerank で推論パイプライン全体を検証する。
        失敗時は詳細なエラーメッセージをログ出力する。

        transient I/O 失敗 (ConnectError / ReadTimeout / 5xx 等) は
        ``async_retry_http_call`` でリトライする。最終的に失敗 (永続エラー /
        empty results / リトライ枯渇) した場合は ``False`` を返し、
        ``reranker_factory`` 側で NullReranker フォールバックに切替える。
        """
        client = self._get_http_client()
        payload = {
            "model": self._model_name,
            "query": _SELFTEST_QUERY,
            "documents": _SELFTEST_DOCS,
            "top_n": len(_SELFTEST_DOCS),
        }
        retry_logger = make_retry_logger(self._debug_logger, backend="reranker")

        async def _do_post() -> httpx.Response:
            resp = await client.post(self._url, json=payload, timeout=10.0)
            resp.raise_for_status()
            return resp

        try:
            resp = await async_retry_http_call(
                _do_post,
                request_label="Reranker selftest",
                retry_logger=retry_logger,
            )
        except httpx.HTTPStatusError as e:
            body_preview = e.response.text[:500] if e.response.text else "(empty)"
            logger.warning(
                "Reranker selftest failed: status=%d, body=%s",
                e.response.status_code, body_preview,
            )
            return False
        except Exception as e:
            logger.warning("Reranker selftest exception: %s: %s", type(e).__name__, e)
            return False
        try:
            data = resp.json()
        except ValueError as e:
            logger.warning("Reranker selftest exception: %s: %s", type(e).__name__, e)
            return False
        results = data.get("results", [])
        if not results:
            logger.warning("Reranker selftest: 200 but empty results: %s", data)
            return False

        scores = [float(r.get("relevance_score", 0.0)) for r in results]
        max_abs = max(abs(s) for s in scores)
        if max_abs < _SELFTEST_SCORE_EPSILON:
            # 全候補が実質ゼロ = ランキング信号ゼロ。Qwen3-Reranker と
            # llama.cpp --reranking (rank pooling) の流儀不整合などで退化した
            # ケースに該当する。active のままだと毎クエリ 2-3s を空費して並びを
            # 一切変えないため、selftest を失敗させ reranker_factory 側で
            # LazyReranker (no-op + 定期再試行) に倒す。
            logger.warning(
                "Reranker selftest: degenerate all-zero scores %s — reranker "
                "produces no ranking signal; failing selftest to fall back to "
                "no-op. Check model / --reranking compatibility.",
                [f"{s:.4f}" for s in scores],
            )
            return False
        if max(scores) - min(scores) < _SELFTEST_SCORE_EPSILON:
            # 非ゼロだが全候補同値 = 識別力が無い。誤判定回避のため致命扱いには
            # せず WARNING のみ (弱いテストペアで同値になる可能性を考慮)。
            logger.warning(
                "Reranker selftest: scores show no variance %s (weak "
                "discrimination); leaving active but ranking value is low.",
                [f"{s:.4f}" for s in scores],
            )
        logger.info(
            "Reranker selftest passed: %d results, top_score=%.4f",
            len(results), results[0].get("relevance_score", 0.0),
        )
        return True
