"""EmbeddingBackend Protocol 定義 + QueryCacheMixin

埋め込みバックエンドの抽象インターフェース。
llama-cpp 実装がこの Protocol に準拠する。
QueryCacheMixin はクエリ埋め込みの LRU キャッシュロジックを共通化する。

Qwen3-Embedding は instruction-aware かつ非対称な埋め込みモデルで
- クエリ側 (``is_query=True``): ``f"Instruct: {task}\\nQuery: {query}"`` 形式で integration
- ドキュメント側 (``is_query=False``): prefix を一切付与しない

を要求する。``task`` は ``mode`` (``chat`` / ``coding``) によって切替えられ、
``config.yaml`` の ``embedding.instructions`` で定義される。
"""

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
            mode: ``chat`` / ``coding`` のいずれか。``is_query=True`` のときに
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
            mode: ``chat`` / ``coding`` のいずれか。LRU キャッシュキーは mode
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
    chat 用埋め込みが coding 検索で誤って再利用されるのを防ぐ。
    """

    _query_cache: OrderedDict[str, np.ndarray]
    _cache_hits: int
    _cache_misses: int
    _cache_maxsize: int
    _debug_logger: object | None

    def _init_query_cache(self, maxsize: int = DEFAULT_QUERY_CACHE_MAXSIZE) -> None:
        """キャッシュの初期化（__init__ から呼び出す）"""
        self._query_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_maxsize = maxsize

    async def _embed_single_query(
        self, query: str, mode: str = DEFAULT_MODE
    ) -> np.ndarray:
        """単一クエリの埋め込み生成（サブクラスで実装）"""
        raise NotImplementedError

    async def embed_query(
        self, query: str, *, mode: str = DEFAULT_MODE
    ) -> np.ndarray:
        """単一クエリの埋め込み（LRU キャッシュ対応 + mode 名前空間分離）"""
        cache_key = make_query_cache_key(query, mode)
        if cache_key in self._query_cache:
            self._cache_hits += 1
            self._query_cache.move_to_end(cache_key)
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
            return self._query_cache[cache_key]

        self._cache_misses += 1
        vec = await self._embed_single_query(query, mode)

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
