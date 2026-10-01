"""埋め込みバックエンドのファクトリ関数

config.yaml の embedding セクションからバックエンドを生成する。
cache_enabled が true の場合は CachedEmbeddingBackend でラップする。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.core.tuning.tuners.embed_params import EmbedParams, resolve_embed_params
from backend.free.rag.embedding_backend import EmbeddingBackend
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("rag.embedding_factory")


def validate_embedding_config(params: EmbedParams, max_length: int) -> None:
    """埋め込みの実効値 (``embed_params`` の解決値と明示値) を起動時にログへ出す

    ``auto`` の batch / ubatch は ``ubatch >= max_length`` が保証済み。明示値がそれ (と
    ``batch >= ubatch``) を破っていれば WARNING を出す (値は変えない、c_16 §7.2.3)。
    ログは英語固定 (i18n 対象外)。
    """
    for warning in params.warnings:
        logger.warning("%s", warning)
    logger.info(
        "embedding config: max_length=%d, batch_size=%d (%s), ubatch_size=%d (%s), "
        "http_batch=%d, timeout=%.1fs (%s), query_timeout=%.1fs (%s)",
        max_length,
        params.batch_size, params.sources["batch_size"],
        params.ubatch_size, params.sources["ubatch_size"],
        params.http_batch,
        params.timeout, params.sources["timeout"],
        params.query_timeout, params.sources["query_timeout"],
    )


def _embed_model_key(cfg: dict, project_root: Path | None) -> str:
    """``model_paths.embed_model`` の model_key (宣言が無ければ空文字 = model_name を使う)。"""
    raw = (cfg.get("model_paths") or {}).get("embed_model") or ""
    if not raw:
        return ""
    from backend.config import PathResolver, get_project_root

    return PathResolver(cfg, project_root or get_project_root()).model_key_for(raw)


def create_embedding_backend(
    cfg: dict,
    project_root: Path | None = None,
    debug_logger: DebugLogger | None = None,
) -> EmbeddingBackend:
    """config.yaml の embedding セクションからバックエンドを生成

    Args:
        cfg: config.yaml の辞書全体
        project_root: インストール根 (キャッシュの置き場 = そのデータ根の解決用)

    Returns:
        EmbeddingBackend 準拠のインスタンス
    """
    emb_cfg = cfg.get("embedding", {})
    # batch / ubatch / HTTP バッチ件数 / timeout の実効値 (環境調整の保存値を読むだけ、c_16 §7.2.3)。
    # 起動スクリプトの -b / -ub と同じ解決値。
    params = resolve_embed_params(cfg, project_root, allow_decide=False)
    validate_embedding_config(params, int(emb_cfg.get("max_length") or 8192))
    backend = emb_cfg.get("backend", "llama-cpp")

    match backend:
        case "llama-cpp":
            from backend.free.rag.embedding_llamacpp import LlamaCppEmbedder

            host = emb_cfg.get("llama_host", "localhost")
            port = emb_cfg.get("llama_port", 8082)
            model_name = emb_cfg.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
            dim = emb_cfg.get("dim", 1024)
            timeout = params.timeout
            # チャット応答パスの単一クエリだけに掛かるデッドライン。
            query_timeout = params.query_timeout
            max_length = emb_cfg.get("max_length", 8192)
            # instruction-aware プレフィックス (Qwen3 等)。
            # ``embedding.instructions`` が config.yaml に無い場合は schema 既定値が
            # 入る (chat / create 両方の英語 instruction)。空辞書は LlamaCppEmbedder
            # 側の _FALLBACK_INSTRUCTION で救済。
            instructions = emb_cfg.get("instructions", {})
            # クエリ / ドキュメント整形テンプレート。schema 既定値は Qwen3 仕様
            # (``"Instruct: {task}\nQuery: {query}"`` / 文書側は空)。
            # 空文字列で素のテキスト送信 (BGE-M3 等の非 instruction-aware モデル)。
            query_template = emb_cfg.get(
                "query_template", "Instruct: {task}\nQuery: {query}",
            )
            doc_template = emb_cfg.get("doc_template", "")

            # ストアのディレクトリ名に使う埋め込みモデルの識別子 (c_05 §0.5.7)
            model_key = _embed_model_key(cfg, project_root)
            embedder = LlamaCppEmbedder(
                host=host,
                port=port,
                model_name_str=model_name,
                model_key=model_key,
                dim_size=dim,
                timeout=timeout,
                query_timeout=query_timeout,
                max_length=max_length,
                instructions=instructions,
                query_template=query_template,
                doc_template=doc_template,
                debug_logger=debug_logger,
                http_batch=params.http_batch,
            )
            logger.info(
                "Created LlamaCppEmbedder: %s:%d (model=%s, dim=%d, "
                "instruction_modes=%s, query_template=%r, doc_template=%r)",
                host, port, model_name, dim, sorted(instructions.keys()),
                query_template, doc_template,
            )

        case _:
            raise ValueError(f"Unknown embedding backend: {backend}")

    # 埋め込みサーバの調停 (c_16 §6.5)。キャッシュの内側に置く — キャッシュに
    # 当たった呼び出しは順番を待たない。
    from backend.free.rag.embed_scheduler import ScheduledEmbeddingBackend

    embedder = ScheduledEmbeddingBackend(embedder)

    # キャッシュラッパーで包む（embedding.cache_enabled: true 時）
    cache_enabled = emb_cfg.get("cache_enabled", True)
    if cache_enabled:
        from backend.free.rag.embedding_cache import CachedEmbeddingBackend

        from backend.config import resolve_data_path

        # 置き場はデータ根の cache/embeddings/ 固定 (c_05 §0.2)。
        cache_dir = resolve_data_path("embedding_cache_dir", project_root)
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
