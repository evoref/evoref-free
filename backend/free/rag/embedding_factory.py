"""埋め込みバックエンドのファクトリ関数

config.yaml の embedding セクションからバックエンドを生成する。
cache_enabled が true の場合は CachedEmbeddingBackend でラップする。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.rag.embedding_backend import EmbeddingBackend
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("rag.embedding_factory")


def validate_embedding_config(emb_cfg: dict) -> None:
    """Embedding 設定の整合性を起動時に検証する

    設定ミス (ubatch_size 未設定 / max_length より小さい / batch_size の逆転) は
    llama-server 500 エラーの再発要因になる。問題を起動時に WARNING / INFO として
    可視化し、運用者が気付けるようにする。ログは英語固定 (i18n 対象外)。

    Args:
        emb_cfg: config.yaml の embedding セクション (dict)
    """
    max_length = emb_cfg.get("max_length", 8192)
    batch_size = emb_cfg.get("batch_size")
    ubatch_size = emb_cfg.get("ubatch_size")

    if ubatch_size is None:
        logger.info(
            "embedding.ubatch_size not set: llama-server default (512) will be used. "
            "Long inputs near max_length=%d may trigger 500 errors during "
            "EvorefMem sleep-time update. See docs/f_01_rag_engine.md.",
            max_length,
        )
    elif ubatch_size < max_length:
        logger.warning(
            "embedding.ubatch_size=%d < max_length=%d: single inputs of %d tokens "
            "will cause llama-server to return 500. Raise ubatch_size to >= max_length "
            "in config.yaml. See docs/f_01_rag_engine.md.",
            ubatch_size, max_length, max_length,
        )

    if batch_size is None and ubatch_size is not None:
        logger.info(
            "embedding.batch_size not set while ubatch_size=%d: llama-server default "
            "will be used for -b. Set batch_size >= ubatch_size for safety.",
            ubatch_size,
        )
    elif batch_size is not None and ubatch_size is not None and batch_size < ubatch_size:
        logger.warning(
            "embedding.batch_size=%d < ubatch_size=%d: llama-server requires -b >= -ub. "
            "Raise batch_size in config.yaml to match ubatch_size or higher.",
            batch_size, ubatch_size,
        )

    logger.info(
        "embedding config: max_length=%d, batch_size=%s, ubatch_size=%s",
        max_length,
        batch_size if batch_size is not None else "default",
        ubatch_size if ubatch_size is not None else "default",
    )


def create_embedding_backend(
    cfg: dict,
    project_root: Path | None = None,
    debug_logger: DebugLogger | None = None,
) -> EmbeddingBackend:
    """config.yaml の embedding セクションからバックエンドを生成

    Args:
        cfg: config.yaml の辞書全体
        project_root: プロジェクトルート（相対パス解決用）

    Returns:
        EmbeddingBackend 準拠のインスタンス
    """
    emb_cfg = cfg.get("embedding", {})
    # 起動時バリデーション (Fail Fast)
    validate_embedding_config(emb_cfg)
    backend = emb_cfg.get("backend", "llama-cpp")

    match backend:
        case "llama-cpp":
            from backend.free.rag.embedding_llamacpp import LlamaCppEmbedder

            host = emb_cfg.get("llama_host", "localhost")
            port = emb_cfg.get("llama_port", 8082)
            model_name = emb_cfg.get("model_name", "qwen3-embedding")
            dim = emb_cfg.get("dim", 1024)
            timeout = emb_cfg.get("timeout", 30.0)
            max_length = emb_cfg.get("max_length", 8192)
            # instruction-aware プレフィックス (Qwen3 等)。
            # ``embedding.instructions`` が config.yaml に無い場合は schema 既定値が
            # 入る (chat / coding 両方の英語 instruction)。空辞書は LlamaCppEmbedder
            # 側の _FALLBACK_INSTRUCTION で救済。
            instructions = emb_cfg.get("instructions", {})
            # クエリ / ドキュメント整形テンプレート。schema 既定値は Qwen3 仕様
            # (``"Instruct: {task}\nQuery: {query}"`` / 文書側は空)。
            # 空文字列で素のテキスト送信 (BGE-M3 等の非 instruction-aware モデル)。
            query_template = emb_cfg.get(
                "query_template", "Instruct: {task}\nQuery: {query}",
            )
            doc_template = emb_cfg.get("doc_template", "")

            embedder = LlamaCppEmbedder(
                host=host,
                port=port,
                model_name_str=model_name,
                dim_size=dim,
                timeout=timeout,
                max_length=max_length,
                instructions=instructions,
                query_template=query_template,
                doc_template=doc_template,
                debug_logger=debug_logger,
            )
            logger.info(
                "Created LlamaCppEmbedder: %s:%d (model=%s, dim=%d, "
                "instruction_modes=%s, query_template=%r, doc_template=%r)",
                host, port, model_name, dim, sorted(instructions.keys()),
                query_template, doc_template,
            )

        case _:
            raise ValueError(f"Unknown embedding backend: {backend}")

    # キャッシュラッパーで包む（embedding.cache_enabled: true 時）
    cache_enabled = emb_cfg.get("cache_enabled", True)
    if cache_enabled:
        from backend.free.rag.embedding_cache import CachedEmbeddingBackend

        cache_dir_str = emb_cfg.get("cache_dir", "local/cache/embeddings/")
        cache_dir = Path(cache_dir_str)
        if project_root and not cache_dir.is_absolute():
            cache_dir = project_root / cache_dir
        cache_max_mb = emb_cfg.get("cache_max_mb", 100)

        embedder = CachedEmbeddingBackend(
            inner=embedder,
            cache_dir=cache_dir,
            max_mb=cache_max_mb,
            debug_logger=debug_logger,
        )
        logger.info(
            "Embedding cache enabled: dir=%s, max_mb=%s",
            cache_dir, cache_max_mb,
        )

    return embedder
