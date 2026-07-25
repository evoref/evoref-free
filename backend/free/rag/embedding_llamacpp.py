"""llama-server /v1/embeddings 経由の埋め込み生成（EmbeddingBackend 準拠）

Qwen3-Embedding 等の GGUF モデルを llama-server 経由で使用する。
LoRA 適用が可能なため、自己学習パイプラインの三角形フィードバックループに
組み込める。ポート :8082 で独立した llama-server インスタンスとして起動する。

Qwen3-Embedding 公式の instruction-aware フォーマットを実装する
- ``is_query=True``: ``f"Instruct: {instructions[mode]}\\nQuery: {text}"`` に整形
- ``is_query=False``: 素のテキストをそのまま投入 (Qwen3 はドキュメント側に
  prefix を付けない非対称設計)
"""

import time

import httpx
import numpy as np

from backend.free.llm._base_client import (
    BaseHTTPClient,
    HealthLogGate,
    async_retry_http_call,
    make_retry_logger,
)
from backend.free.rag.embedding_backend import (
    DEFAULT_MODE,
    QueryCacheMixin,
)
from backend.free.rag.instruction_resolver import format_with_instruction
from backend.log_config import get_logger

logger = get_logger("rag.embedding_llamacpp")

# 1 HTTP リクエストあたりの最大テキスト数。フル reindex 等の大バッチ
# (例 492 チャンクを 1 回で送る) を分割し、単一リクエストが embedding.timeout を
# 超過する (低速 iGPU で顕著、httpx.ReadTimeout) のを防ぐ。各サブバッチは
# embed() の単一リクエスト経路 (clamp / prefix / 5xx per-text フォールバック) を通る。
_MAX_HTTP_BATCH = 64


class LlamaCppEmbedder(QueryCacheMixin, BaseHTTPClient):
    """llama-server /v1/embeddings 経由の埋め込み生成"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8082,
        model_name_str: str = "qwen3-embedding",
        dim_size: int = 1024,
        timeout: float = 30.0,
        max_length: int = 8192,
        instructions: dict[str, str] | None = None,
        query_template: str = "Instruct: {task}\nQuery: {query}",
        doc_template: str = "",
        debug_logger=None,
    ):
        super().__init__(timeout=timeout)
        # health check の DEBUG を状態変化時のみに絞る (ポーリングでログが埋まるのを防ぐ)
        self._health_log_gate = HealthLogGate()
        self._url = f"http://{host}:{port}/v1/embeddings"
        self._model_name = model_name_str
        self._dim = dim_size
        # 入力長クランプ (Fail Safe)
        # Qwen3-Embedding 系の日本語トークナイズは概ね 1 tok ≈ 2-3 文字なので、
        # `max_length * 2` の文字数に安全側で切り詰める。llama-server の
        # ubatch_size 不足に対する最終防衛ラインとして機能する (per-text
        # フォールバックでも救えない単一入力超過ケースを抑止する)。
        self._max_length = max_length
        self._char_limit = max(1, max_length * 2)
        # instruction-aware モデル (Qwen3 等) 用の mode 別 task description。
        # 空辞書を渡されても起動を止めないよう、フォールバック文字列を内部で保持する。
        self._instructions: dict[str, str] = dict(instructions or {})
        # クエリ / ドキュメント整形テンプレート。
        # 空文字列の場合は prefix を一切付与せず素のテキストを送る (BGE-M3 等)。
        self._query_template = query_template
        self._doc_template = doc_template
        self._debug_logger = debug_logger
        # 初回レスポンスフラグ: 初回は次元 auto-detect を許可
        # 2 回目以降の変化は異常とみなして例外送出する
        self._dim_detected = False
        self._init_query_cache()
        logger.info(
            "LlamaCppEmbedder initialized: url=%s, model=%s, dim=%d, max_length=%d, "
            "instruction_modes=%s, query_template=%r, doc_template=%r",
            self._url, model_name_str, dim_size, max_length,
            sorted(self._instructions.keys()),
            self._query_template, self._doc_template,
        )

    def _format_text(self, text: str, *, is_query: bool, mode: str) -> str:
        """テンプレート駆動でテキストを整形する

        ``is_query=True`` で ``query_template``、``False`` で ``doc_template``
        を適用する。テンプレートが空文字列なら素のテキストを返す
        (BGE-M3 等の非 instruction-aware モデル向け fast-path)。instruction
        解決とフォーマットは :mod:`instruction_resolver` に委譲する。
        """
        template = self._query_template if is_query else self._doc_template
        return format_with_instruction(
            text, template, self._instructions, mode, backend_label="embed_query",
        )

    async def embed(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        mode: str = DEFAULT_MODE,
    ) -> np.ndarray:
        """テキストリストを埋め込みベクトルに変換（async）

        llama-server の /v1/embeddings API を呼び出す。
        ``is_query=True`` のときは Qwen3-Embedding の instruction-aware フォーマット
        (``f"Instruct: {task}\\nQuery: {text}"``) でクエリを整形する
        ``is_query=False`` (ドキュメント側) は素のテキストをそのまま投入する。
        """
        if not texts:
            return np.array([]).reshape(0, self._dim)

        # 大バッチは HTTP リクエストを分割する (timeout / メモリ過大防止)。各サブバッチは
        # 下の単一リクエスト経路を再帰で通る。フル reindex (全チャンクを 1 embed() で送る)
        # が低速 iGPU で 30s timeout を超えて落ちるのを防ぐ。
        if len(texts) > _MAX_HTTP_BATCH:
            parts: list[np.ndarray] = []
            for i in range(0, len(texts), _MAX_HTTP_BATCH):
                sub = await self.embed(
                    texts[i:i + _MAX_HTTP_BATCH], is_query=is_query, mode=mode,
                )
                parts.append(sub)
            return np.vstack(parts)

        client = self._get_http_client()

        logger.debug(
            "embed: batch_size=%d, is_query=%s, mode=%s",
            len(texts), is_query, mode,
        )

        # 入力長クランプ (instruction 付与前に実行)
        # llama-server の ubatch_size を単一入力が超えた場合、per-text
        # フォールバックでも救えないため、事前に文字数で安全側にトリミングする。
        char_limit = self._char_limit
        clamped_texts: list[str] = []
        clamp_count = 0
        for t in texts:
            if len(t) > char_limit:
                clamp_count += 1
                clamped_texts.append(t[:char_limit])
            else:
                clamped_texts.append(t)
        if clamp_count:
            logger.debug(
                "embed: clamped %d/%d texts to %d chars (max_length=%d)",
                clamp_count, len(texts), char_limit, self._max_length,
            )

        # テンプレート駆動でクエリ / ドキュメントを整形する。
        # - クエリ側: ``query_template`` (Qwen3 既定: ``"Instruct: ...\nQuery: ..."``)
        # - ドキュメント側: ``doc_template`` (Qwen3 既定では空 = 素のテキスト)
        # テンプレート空文字列なら fast-path で素のテキストをそのまま渡す。
        prefixed = [
            self._format_text(t, is_query=is_query, mode=mode)
            for t in clamped_texts
        ]

        payload = {
            "input": prefixed,
            "model": self._model_name,
        }

        retry_logger = make_retry_logger(
            self._debug_logger, backend="embedding",
        )

        async def _post_batch() -> httpx.Response:
            r = await client.post(self._url, json=payload)
            # 5xx (>=500) は raise_for_status で HTTPStatusError 化し、
            # tenacity 側で retryable_statuses に該当すればリトライされる。
            # ただし embedding は ubatch_size 超過時のみ per-text フォールバック
            # を発動したいため、5xx かつ batch>1 のケースは tenacity を経由
            # せず後段で個別処理する (下の hand-off ブロック参照)。
            if r.status_code >= 500 and len(prefixed) > 1:
                return r
            r.raise_for_status()
            return r

        t0 = time.monotonic()
        resp = await async_retry_http_call(
            _post_batch,
            request_label="embedding request",
            retry_logger=retry_logger,
        )
        # 500 でかつバッチが 1 件より多い場合は 1 件ずつフォールバック
        # 通常は config.yaml の embedding.ubatch_size を上げてサーバ側で吸収するが、
        # 設定漏れや一時的に長すぎる入力が来ても sleep-time update を完全停止させない。
        if resp.status_code >= 500 and len(prefixed) > 1:
            logger.warning(
                "embed: server %d on batch=%d, falling back to per-text requests "
                "(consider increasing embedding.ubatch_size in config.yaml)",
                resp.status_code, len(prefixed),
            )
            single_results: list[list[float]] = []
            for one in prefixed:
                async def _post_single(text=one) -> httpx.Response:
                    r = await client.post(
                        self._url,
                        json={"input": [text], "model": self._model_name},
                    )
                    r.raise_for_status()
                    return r

                single_resp = await async_retry_http_call(
                    _post_single,
                    request_label="embedding per-text request",
                    retry_logger=retry_logger,
                )
                single_data = single_resp.json().get("data", [])
                if not single_data:
                    raise ValueError("Empty embeddings response from llama-server")
                single_results.append(single_data[0]["embedding"])
            elapsed = time.monotonic() - t0
            data = {"data": [
                {"embedding": v, "index": i} for i, v in enumerate(single_results)
            ]}
        else:
            elapsed = time.monotonic() - t0
            data = resp.json()

        logger.debug("embed: response in %.3fs, status=%d", elapsed, resp.status_code)

        # レスポンスから埋め込みベクトルを抽出
        embeddings_data = data.get("data", [])
        if not embeddings_data:
            raise ValueError("Empty embeddings response from llama-server")

        # インデックス順にソート（API は順序保証しない場合がある）
        embeddings_data.sort(key=lambda x: x.get("index", 0))
        vectors = [item["embedding"] for item in embeddings_data]

        result = np.array(vectors, dtype=np.float32)

        # 次元数を初回レスポンスから検出
        # 初回: 設定値と異なっても auto-detect として受け入れる
        # 2 回目以降: 動的な変化は異常としてエラー
        if result.shape[1] != self._dim:
            if not self._dim_detected:
                logger.info(
                    "Embedding dimension auto-detected: %d (configured: %d)",
                    result.shape[1], self._dim,
                )
                self._dim = int(result.shape[1])
            else:
                raise RuntimeError(
                    f"Embedding dimension changed mid-session: "
                    f"{self._dim} -> {result.shape[1]}. "
                    f"This usually means the llama-server model was switched."
                )
        self._dim_detected = True

        # L2 正規化
        norms = np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-9)
        result = result / norms

        # DebugLogger
        dl = self._debug_logger
        if dl:
            dl.log_embedding(
                batch_size=len(texts), backend="llama-cpp",
                elapsed_sec=elapsed, is_query=is_query,
            )

        return result

    async def _embed_single_query(
        self, query: str, mode: str = DEFAULT_MODE
    ) -> np.ndarray:
        """単一クエリの埋め込み生成（QueryCacheMixin 用）"""
        result = await self.embed([query], is_query=True, mode=mode)
        return result[0]

    def dim(self) -> int:
        """出力ベクトル次元数"""
        return self._dim

    def model_name(self) -> str:
        """モデル識別名"""
        return self._model_name

    def backend_type(self) -> str:
        """バックエンド種別"""
        return "llama-cpp"

    def supports_lora(self) -> bool:
        """llama-server は --lora オプションで LoRA 適用可能"""
        return True

    def supports_instructions(self) -> bool:
        """検索指示プレフィックスの動的変更が可能"""
        return True

    def set_instruction(self, text: str, *, mode: str | None = None) -> None:
        """進化採用された検索 instruction を稼働中に反映する (Learn→Gen 動的更新)。

        ``mode=None`` で全 mode (既存キー + ``DEFAULT_MODE``) を同一 instruction で
        共通 override する (進化器・eval が構造上 mode 非フィルタなため、単一の
        進化結果を全 mode に適用するのが整合的)。``mode`` 指定でそのキーのみ更新。
        instruction を変えると旧 prefix で計算済みの query 埋め込みキャッシュが
        stale になるため必ず :meth:`clear_query_cache` する。
        """
        text = text.strip()
        if not text:
            return
        if mode is None:
            for key in set(self._instructions) | {DEFAULT_MODE}:
                self._instructions[key] = text
        else:
            self._instructions[mode] = text
        self.clear_query_cache()
        logger.info(
            "Embed instruction updated at runtime (mode=%s, modes=%s)",
            mode or "ALL", sorted(self._instructions.keys()),
        )

    async def health_check(self) -> bool:
        """埋め込み用 llama-server のヘルスチェック

        /api/status からサーバプロセスの生存を確認するために呼ぶ。
        ConnectError / Timeout 系は False を返し、例外を握りつぶす。
        """
        try:
            client = self._get_http_client()
            base_url = self._url.rsplit("/v1/embeddings", 1)[0]
            resp = await client.get(f"{base_url}/health", timeout=5.0)
            healthy = resp.status_code == 200
            if self._health_log_gate.should_log(resp.status_code, healthy):
                logger.debug(
                    "Embedding health check: status=%d, healthy=%s"
                    " (suppressed %d identical)",
                    resp.status_code, healthy,
                    self._health_log_gate.take_suppressed(),
                )
            return healthy
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
            logger.debug("Embedding health check failed: %s", e)
            return False
