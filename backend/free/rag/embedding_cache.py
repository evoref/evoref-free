"""チャンクハッシュベースの永続埋め込みキャッシュ

同一チャンクの再登録・再構築時に埋め込み計算をスキップする。
キャッシュキーは ``model_name:text`` の SHA-256 ハッシュ (16 文字)。
永続化層は ``diskcache.Cache`` (SQLite-backed, WAL モード) を用いる。

dim / model_name 整合性メタは ``cache_dir/meta.json`` に保存し、diskcache の
LRU eviction の影響を受けないようにする。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from diskcache import Cache

from backend.io import AtomicWriter
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("rag.embedding_cache")

# デフォルトキャッシュ上限 (MB)
_DEFAULT_CACHE_MAX_MB = 100

# 整合性メタファイル (dim / model_name)
_META_FILENAME = "meta.json"


def _cache_key(text: str, model_name: str) -> str:
    """テキスト + モデル名の SHA-256 ハッシュ（16文字）"""
    content = f"{model_name}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _read_meta(meta_path: Path) -> dict | None:
    """``meta.json`` を読み込む (存在しない・破損は ``None``)"""
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Failed to read embedding cache meta %s: %s (treated as missing)",
            meta_path, e,
        )
        return None
    return data if isinstance(data, dict) else None


def _write_meta_atomic(meta_path: Path, meta: dict) -> None:
    """``meta.json`` を原子的に書き換える (:class:`AtomicWriter` 委譲)。

    内部実装は ``backend.io.AtomicWriter`` に委譲。``tempfile.mkstemp`` で
    呼出ごとにユニークな tmp ファイル名を発番し、``os.replace`` の
    Windows ``PermissionError`` (= ``ERROR_SHARING_VIOLATION``) は retry
    される (詳細は :mod:`backend.io._retry` 参照、4 process × 20 init を
    含むフル 7629 test を Windows 上で 3 連続全緑で実証済)。

    複数プロセスが同時に書き換える場合は後勝ちとなるが、同じ embedder
    設定なら内容が一致するため問題ない。書込頻度の削減自体は
    :func:`_verify_meta` 側の double-check で対応する。
    """
    with AtomicWriter(meta_path) as f:
        f.write(json.dumps(meta, ensure_ascii=False))


def _write_meta_if_changed(meta_path: Path, expected: dict) -> bool:
    """メタが既に ``expected`` と一致していれば書き込みをスキップする。

    複数プロセスが同時に同じ内容のメタを書き戻すレース を
    抑止するための double-check。既存メタの読込とコンパクトな等価判定で
    no-op を保証することで、`_write_meta_atomic` の発火回数自体を減らす。

    戻り値: 実際に書き込みが発生した場合 True、スキップ時 False。
    """
    current = _read_meta(meta_path)
    if isinstance(current, dict) and current == expected:
        return False
    _write_meta_atomic(meta_path, expected)
    return True


class CachedEmbeddingBackend:
    """永続埋め込みキャッシュ付き EmbeddingBackend ラッパー

    内部バックエンド（llama-cpp）をラップし、``embed()`` 呼び出し時に
    チャンクテキストのハッシュでキャッシュを検索する。
    キャッシュミスのテキストのみ実際の埋め込み計算を実行する。

    永続化は ``diskcache.Cache`` (SQLite + WAL) に委譲。LRU eviction は
    ``size_limit`` (= ``max_mb * 1024 * 1024``) を超えたタイミングで diskcache が
    自動実行する。

    複数プロセス並行アクセスは SQLite の WAL モードにより安全。
    """

    def __init__(
        self,
        inner: EmbeddingBackend,
        cache_dir: Path,
        max_mb: float = _DEFAULT_CACHE_MAX_MB,
        debug_logger: DebugLogger | None = None,
    ):
        self._inner = inner
        self._cache_dir = Path(cache_dir)
        self._max_mb = max_mb
        self._debug_logger = debug_logger

        # 統計
        self._hits: int = 0
        self._misses: int = 0

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        size_limit = max(int(max_mb * 1024 * 1024), 1)
        self._cache: Cache = Cache(
            str(self._cache_dir),
            size_limit=size_limit,
            eviction_policy="least-recently-used",
            sqlite_journal_mode="wal",
        )

        self._verify_meta()

        logger.info(
            "Embedding cache opened (diskcache): dir=%s, size_limit=%.1f MB, entries=%d",
            self._cache_dir, max_mb, self._entry_count(),
        )

    # ── メタ整合性検証 ───────────────────────────────

    def _verify_meta(self) -> None:
        """埋め込み次元・モデル名が以前と一致するか検証。

        次元 (`dim`) が異なる場合はモデル変更とみなしキャッシュ全体を破棄する
。`model_name` 不一致は
        キャッシュキー (`sha256(model_name:text)`) によって自然にミスするため、
        メタを更新するのみで全クリアは行わない。
        """
        try:
            expected_dim = self._inner.dim()
        except Exception:
            return
        expected_model = self._inner.model_name()
        expected_meta = {"dim": expected_dim, "model_name": expected_model}

        meta_path = self._cache_dir / _META_FILENAME
        meta = _read_meta(meta_path)

        if not isinstance(meta, dict):
            _write_meta_if_changed(meta_path, expected_meta)
            return

        cached_dim = meta.get("dim")
        if cached_dim is not None and cached_dim != expected_dim:
            logger.warning(
                "Embedding cache dimension mismatch: cached=%s, expected=%d. Clearing cache.",
                cached_dim, expected_dim,
            )
            self._cache.clear()
            _write_meta_if_changed(meta_path, expected_meta)
            return

        _write_meta_if_changed(meta_path, expected_meta)

    # ── ヘルパ ──────────────────────────────────────

    def _entry_count(self) -> int:
        """エントリ数 (メタは diskcache 外なので除外不要)"""
        return len(self._cache)

    # ── EmbeddingBackend Protocol 委譲 ─────────────

    async def embed(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        mode: str = "chat",
    ) -> np.ndarray:
        """テキストリストを埋め込みベクトルに変換（キャッシュ対応）

        ``is_query=True`` の場合はキャッシュを使用しない（クエリは内部 LRU で対応）。
        ``mode`` (``chat``/``coding``) は内部バックエンドへ素通り
        ドキュメント側 (``is_query=False``) は Qwen3 仕様で prefix を付けないため、
        永続キャッシュキーは ``(model_name, text)`` のままで mode 分離は不要。
        """
        if not texts:
            return np.array([]).reshape(0, self._inner.dim())

        # クエリ埋め込みはキャッシュ対象外（内部バックエンドの LRU に委譲）
        if is_query:
            return await self._inner.embed(texts, is_query=True, mode=mode)

        model = self._inner.model_name()
        dim = self._inner.dim()

        keys = [_cache_key(t, model) for t in texts]

        result = np.empty((len(texts), dim), dtype=np.float32)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, key in enumerate(keys):
            data = self._cache.get(key)
            vec = self._decode_vector(data, dim)
            if vec is not None:
                result[i] = vec
                self._hits += 1
                continue
            miss_indices.append(i)
            miss_texts.append(texts[i])
            self._misses += 1

        if miss_texts:
            computed = await self._inner.embed(miss_texts, is_query=False)
            for j, idx in enumerate(miss_indices):
                vec = computed[j].astype(np.float32, copy=False)
                result[idx] = vec
                self._cache.set(keys[idx], vec.tobytes())

        logger.debug(
            "embed: %d texts, %d hits, %d misses",
            len(texts), len(texts) - len(miss_texts), len(miss_texts),
        )

        # DebugLogger
        dl = self._debug_logger
        if dl and (len(texts) - len(miss_texts)) > 0:
            dl.log_embedding(
                batch_size=len(texts) - len(miss_texts),
                backend="cache",
                elapsed_sec=0.0,
                is_query=False,
                cache_hit=True,
            )

        return result

    @staticmethod
    def _decode_vector(data: object, dim: int) -> np.ndarray | None:
        """diskcache から取得した bytes を float32 ベクトルに復元する。

        破損データや次元不一致の場合は ``None`` を返してキャッシュミス扱いにする。
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            return None
        try:
            arr = np.frombuffer(bytes(data), dtype=np.float32)
        except Exception:
            return None
        if arr.shape != (dim,):
            return None
        return arr

    async def embed_query(self, query: str, *, mode: str = "chat") -> np.ndarray:
        """単一クエリの埋め込み（内部バックエンドの LRU キャッシュに委譲）"""
        return await self._inner.embed_query(query, mode=mode)

    def dim(self) -> int:
        return self._inner.dim()

    def model_name(self) -> str:
        return self._inner.model_name()

    def backend_type(self) -> str:
        return self._inner.backend_type()

    def supports_lora(self) -> bool:
        return self._inner.supports_lora()

    def supports_instructions(self) -> bool:
        return self._inner.supports_instructions()

    def set_instruction(self, text: str, *, mode: str | None = None) -> None:
        """進化採用 instruction を内部バックエンドへ委譲する (Learn→Gen 動的更新)。

        本番経路は cache_enabled=True で本ラッパが ``state.embedder`` になるため、
        委譲しないと採用しても runtime に効かないサイレント失敗になる。doc 側の
        ディスクキャッシュは instruction 非依存なので触らない (inner 側で query
        LRU のみ clear される)。
        """
        if hasattr(self._inner, "set_instruction"):
            self._inner.set_instruction(text, mode=mode)

    def clear_query_cache(self) -> None:
        """query 埋め込みキャッシュの clear を内部バックエンドへ委譲する。"""
        if hasattr(self._inner, "clear_query_cache"):
            self._inner.clear_query_cache()

    async def health_check(self) -> bool:
        """内部バックエンドのヘルスチェックを委譲する。

        `_collect_component_statuses` (status.py) は `hasattr(embedder, "health_check")`
        で生存確認の可否を判定するため、本ラッパが委譲しないと
        `state.embedder` が `CachedEmbeddingBackend` でラップされている場合に
        embed の状態が常に「未接続」と判定されてしまう

        内部バックエンドが `health_check` を持たない場合は True を返す。
        """
        if hasattr(self._inner, "health_check"):
            return await self._inner.health_check()
        return True

    async def aclose(self) -> None:
        """diskcache をクローズし、内部バックエンドを閉じる。

        diskcache は ``set()`` ごとに SQLite に commit するため明示 flush は不要。
        """
        try:
            self._cache.close()
        except Exception as e:
            logger.warning("diskcache close failed: %s", e)
        if hasattr(self._inner, "aclose"):
            await self._inner.aclose()

    @property
    def cache_stats(self) -> dict[str, int | float]:
        """キャッシュ統計"""
        try:
            volume_bytes = self._cache.volume()
        except Exception:
            volume_bytes = 0
        size_mb = round(volume_bytes / (1024 * 1024), 2)
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": self._entry_count(),
            "size_mb": size_mb,
            "max_mb": self._max_mb,
        }

    @property
    def inner(self) -> EmbeddingBackend:
        """ラップされた内部バックエンド"""
        return self._inner
