"""EmbeddingBackend Protocol 定義 + QueryCacheMixin

埋め込みバックエンドの抽象インターフェース。
llama-cpp 実装がこの Protocol に準拠する。
QueryCacheMixin はクエリ埋め込みの LRU キャッシュロジックを共通化する。

Qwen3-Embedding は instruction-aware かつ非対称な埋め込みモデルで
- クエリ側 (``is_query=True``): ``f"Instruct: {task}\\nQuery: {query}"`` 形式で integration
- ドキュメント側 (``is_query=False``): prefix を一切付与しない

を要求する。``task`` は ``mode`` (``chat`` / ``create``) によって切替えられ、
``config.yaml`` の ``embedding.instructions`` で定義される。
"""

import asyncio
import re
from collections import OrderedDict
from typing import Protocol, runtime_checkable

import numpy as np

from backend.log_config import get_logger

logger = get_logger("rag.embedding_backend")

# LRU キャッシュデフォルトサイズ
DEFAULT_QUERY_CACHE_MAXSIZE = 256

# モード未指定時のデフォルト
DEFAULT_MODE = "chat"

# キャッシュキー正規化用パターン
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[?？!！。.、,]+$")


def normalize_cache_key(text: str) -> str:
    """キャッシュキーの正規化（空白・末尾句読点の揺れを吸収）"""
    key = _WHITESPACE_RE.sub(" ", text.strip())
    key = _TRAILING_PUNCT_RE.sub("", key)
    return key


def make_query_cache_key(query: str, mode: str) -> str:
    """LRU クエリキャッシュ用のキー生成

    同一 query でも mode が違えば異なるキャッシュエントリになる。``\\x00``
    (NUL) を区切り文字に使うことで、mode 値とクエリ本文の境界がクエリ内文字
    と衝突しない。
    """
    return f"{mode}\x00{normalize_cache_key(query)}"


@runtime_checkable
class EmbeddingBackend(Protocol):
    """埋め込みバックエンドの抽象インターフェース"""

    async def embed(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        mode: str = DEFAULT_MODE,
    ) -> np.ndarray:
        """テキストリストを埋め込みベクトルに変換（async）

        Args:
            texts: 埋め込み対象テキストのリスト
            is_query: True でクエリ用 instruction を付与、False で素のテキスト
            mode: ``chat`` / ``create`` のいずれか。``is_query=True`` のときに
                ``embedding.instructions[mode]`` を解決して prefix を生成する。
                ``is_query=False`` 時は無視される (Qwen3 はドキュメント側に
                prefix を付けない仕様)。

        Returns:
            (N, dim) の numpy 配列
        """
        ...

    async def embed_query(
        self, query: str, *, mode: str = DEFAULT_MODE
    ) -> np.ndarray:
        """単一クエリの埋め込み（キャッシュ対応）

        Args:
            query: 検索クエリ文字列
            mode: ``chat`` / ``create`` のいずれか。LRU キャッシュキーは mode
                ごとに分離される。

        Returns:
            (dim,) の numpy 配列
        """
        ...

    def dim(self) -> int:
        """出力ベクトル次元数"""
        ...

    def model_name(self) -> str:
        """モデル識別名（metadata.json 記録用）"""
        ...

    def backend_type(self) -> str:
        """バックエンド種別（"llama-cpp"）"""
        ...

    def supports_lora(self) -> bool:
        """LoRA 適用可能か"""
        ...

    def supports_instructions(self) -> bool:
        """検索指示プレフィックスの動的変更が可能か"""
        ...


class QueryCacheMixin:
    """クエリ埋め込みの LRU キャッシュ Mixin

    LlamaCppEmbedder で使用するキャッシュロジックを共通化。
    サブクラスは ``_embed_single_query(query, mode)`` のみ実装する。

    キャッシュキーは ``(mode, normalized_query)`` で名前空間分離する
    chat 用埋め込みが create 検索で誤って再利用されるのを防ぐ。
    """

    _query_cache: OrderedDict[str, np.ndarray]
    _cache_hits: int
    _cache_misses: int
    _cache_maxsize: int
    _debug_logger: object | None
    #: 進行中の埋め込み。同じキーの 2 人目以降はこの Future を待つ。
    _query_inflight: dict[str, "asyncio.Future[np.ndarray]"]
    #: 単一クエリ埋め込みのデッドライン (秒)。0 で無効。
    _query_deadline_sec: float

    def _init_query_cache(
        self,
        maxsize: int = DEFAULT_QUERY_CACHE_MAXSIZE,
        deadline_sec: float = 0.0,
    ) -> None:
        """キャッシュの初期化（__init__ から呼び出す）"""
        self._query_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_maxsize = maxsize
        self._query_inflight = {}
        self._query_deadline_sec = max(0.0, float(deadline_sec))

    async def _embed_single_query(
        self, query: str, mode: str = DEFAULT_MODE
    ) -> np.ndarray:
        """単一クエリの埋め込み生成（サブクラスで実装）"""
        raise NotImplementedError

    def _record_cache_hit(self, mode: str) -> None:
        self._cache_hits += 1
        logger.debug(
            "embed_query: cache hit mode=%s (hits=%d, misses=%d)",
            mode, self._cache_hits, self._cache_misses,
        )
        dl = self._debug_logger
        if dl:
            dl.log_embedding(
                batch_size=1, backend=self.backend_type(),
                elapsed_sec=0.0, is_query=True, cache_hit=True,
            )

    async def embed_query(
        self, query: str, *, mode: str = DEFAULT_MODE
    ) -> np.ndarray:
        """単一クエリの埋め込み（LRU キャッシュ対応 + mode 名前空間分離）

        キャッシュへの書き込みは ``await`` の **後** に起きるので、同じクエリを
        並行で投げる呼び出し元が複数あると全員がミスして同じ HTTP 往復を重複
        実行する。チャット 1 ターンでは検索パイプラインとツール判定が
        ``asyncio.create_task`` で同時に走るため、これが常態になっていた
        (実測 2026-08-16 ライブ監査: is_query の埋め込み 123 件すべて cache_hit=false、
        1 ターンあたり同一クエリを 2〜3 回埋め込み ≒ 0.7〜0.9 秒の無駄)。

        進行中の呼び出しを Future で共有し、2 人目以降はそれを待つ。

        ``_query_deadline_sec`` が正なら、埋め込み往復にデッドラインを掛ける
        (``embedding.query_timeout``)。超過は :class:`asyncio.TimeoutError` を
        送出し、呼出側 (``run_search_pipeline``) がそのターンだけ記憶なしで
        応答を続ける。**バッチ / ドキュメント側には掛からない** — sleep-time の
        64 件バッチは本来長く、同じ棒を当てると埋め込み工程が永久に完了しない。
        """
        cache_key = make_query_cache_key(query, mode)
        if cache_key in self._query_cache:
            self._query_cache.move_to_end(cache_key)
            self._record_cache_hit(mode)
            return self._query_cache[cache_key]

        inflight = self._query_inflight.get(cache_key)
        if inflight is not None:
            self._record_cache_hit(mode)
            return await asyncio.shield(inflight)

        # 生産者は **独立タスク** で走らせ、呼出側はそれを shield して待つ。
        # 以前は呼出側のコルーチンがそのまま埋め込みを実行しており、その
        # 呼出側が cancel されると共有 Future に CancelledError が載って、
        # 同じクエリを待っていた他の呼出側 (同ターンのツール判定 等) まで
        # 巻き添えで落ちていた。生産者を独立させれば、最初の呼出側が消えても
        # 埋め込みは完走し、待ち手は結果を受け取れる。
        loop = asyncio.get_running_loop()
        self._cache_misses += 1
        task = loop.create_task(self._produce_query_embedding(cache_key, query, mode))
        # 待ち手が全員消えても "Task exception was never retrieved" を出さない。
        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        self._query_inflight[cache_key] = task
        return await asyncio.shield(task)

    async def _produce_query_embedding(
        self, cache_key: str, query: str, mode: str,
    ) -> np.ndarray:
        """進行中テーブルに登録された 1 クエリの埋め込みを生成し LRU へ入れる。"""
        deadline = self._query_deadline_sec
        try:
            if deadline > 0:
                vec = await asyncio.wait_for(
                    self._embed_single_query(query, mode), timeout=deadline,
                )
            else:
                vec = await self._embed_single_query(query, mode)
        except asyncio.TimeoutError:
            logger.warning(
                "embed_query exceeded the %.1fs deadline (mode=%s, query=%r); "
                "this turn continues without memory retrieval",
                deadline, mode, query[:60],
            )
            raise
        finally:
            self._query_inflight.pop(cache_key, None)

        if len(self._query_cache) >= self._cache_maxsize:
            self._query_cache.popitem(last=False)
        self._query_cache[cache_key] = vec
        return vec

    @property
    def cache_stats(self) -> dict[str, int]:
        """キャッシュ統計を返す"""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._query_cache),
            "maxsize": self._cache_maxsize,
        }

    def clear_query_cache(self) -> None:
        """クエリ埋め込み LRU を破棄する。

        instruction (検索 prefix) を変更した後に呼ぶ。旧 instruction で計算済みの
        ベクトルがそのまま返ると進化が逆効果 (古い prefix で検索) になるため、
        変更後は必ず clear する。累積統計 (hits/misses) は観測連続性のため残す。
        """
        n = len(self._query_cache)
        self._query_cache.clear()
        if n:
            logger.info("Query embedding cache cleared (%d entries)", n)
