"""RAG 管理 API (c_16 §4.3 — 手動投入も corpus パッケージ)

`POST /api/rag/ingest` は「アップロードした 1 文書だけを入れた ``.evocart``
パッケージをその場で組み、corpus へインストールする」。旧実装は単独の
``VectorStore`` (``state.vector_store``) へ直接チャンクを書いていたが、c_16 で
注入材料の永続層が `Evidence` 3 ストアへ統一されたため、書き込み先は corpus
だけになった。``state.vector_store`` は閾値較正 / 次元検査の読み手として残る。
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from backend.app_state import AppState, get_app_state
from backend.i18n_helper import msg
from backend.free.api.content._rag_helpers import (
    corpus_store_not_initialized_error,
    embedder_not_initialized_error,
    rag_file_empty_error,
    rag_filename_required_error,
    rag_no_chunks_error,
    rag_unsupported_format_error,
)
from backend.free.api.schemas import RagIngestResponse, RagStatsResponse, RagSourceInfo
from backend.config import get_config
from backend.log_config import get_logger
from backend.free.rag.corpus import (
    PackageError,
    PackageMeta,
    package_filename,
    write_package,
)
from backend.free.rag.text_extractor import (
    SUPPORTED_DOC_EXTENSIONS,
    extract_text_from_bytes,
    parse_csv_bytes_to_chunks,
)

logger = get_logger("api.rag")

router = APIRouter(prefix="/api/rag", tags=["rag"])

SUPPORTED_EXTENSIONS = SUPPORTED_DOC_EXTENSIONS

#: 手動投入で作るパッケージ id の接頭辞 (``manual-<sha8>``)。
MANUAL_PACKAGE_PREFIX = "manual-"

#: 手動投入パッケージの版。**固定値**にする。id が本文の sha256 由来なので
#: 内容が変われば id が変わり、内容が同じなら同じ id + 同じ版に落ちて
#: 入れ直しがその場で上書きになる (``CorpusStore.install`` が同名版を作り
#: 直す)。版に日付や連番を入れると、同じ文書を 2 回入れただけで版ディレクトリ
#: が増えて GC 待ちのゴミになる。
MANUAL_PACKAGE_VERSION = "1.0.0"

#: ``docs/`` に原文のまま置く拡張子。それ以外 (pdf / docx / xlsx / pptx) は
#: 抽出済みテキストを ``.txt`` として置く。``.csv`` を原文で残すのは、corpus
#: の chunker が拡張子を見て「行ごとに 1 チャンク (ヘッダー付与)」の規則へ
#: 分岐するため (§4.9.9 の CSV 特例をそのまま引き継ぐ)。
_KEEP_SUFFIXES = {".txt", ".md", ".csv"}

#: ファイル名から取り除く Windows 予約文字。
_UNSAFE_NAME_RE = re.compile(r'[\\/:*?"<>|]+')


def _content_sha8(text: str) -> str:
    """文書本文の sha256 (UTF-8) の先頭 8 hex。

    ``compute_content_digest`` ではなく :mod:`hashlib` を直接使う。あちらは
    ``docs/`` 配下の**相対パスも**ハッシュに食わせるので、同じ本文でも
    ファイル名が違えば別の値になり、「同じ文書は同じパッケージ」にならない。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _safe_doc_name(filename: str, suffix: str) -> str:
    """アップロード名から ``docs/`` 内の文書名を作る。

    パス成分は捨てる (zip の外へ書かせない / 階層を作らない)。
    """
    stem = Path(os.path.basename(filename)).stem.strip()
    stem = _UNSAFE_NAME_RE.sub("_", stem)[:80]
    return f"{stem or 'document'}{suffix}"


def _prepare_document(filename: str, ext: str, content: bytes) -> tuple[str, str]:
    """アップロード本体を ``(docs 内の文書名, 本文)`` にする。

    Raises:
        HTTPException: 抽出結果が空 (400)。
    """
    if ext == ".csv":
        # CSV は行ごとに 1 チャンク (§4.9.9)。分割自体は corpus の chunker が
        # 同じ規則でやるので、ここでは「空でないこと」だけ確かめて原文を渡す。
        if not parse_csv_bytes_to_chunks(content):
            raise rag_file_empty_error()
        return _safe_doc_name(filename, ".csv"), content.decode(
            "utf-8", errors="replace",
        )

    text = extract_text_from_bytes(content, filename)
    if not text.strip():
        raise rag_file_empty_error()
    suffix = ext if ext in _KEEP_SUFFIXES else ".txt"
    return _safe_doc_name(filename, suffix), text


@router.post("/ingest", response_model=RagIngestResponse, status_code=201)
async def ingest_document(
    state: AppState = Depends(get_app_state),
    file: UploadFile = File(...),
    category: str = Form("document"),
):
    """アップロード文書を corpus パッケージにして取り込む (c_16 §4.3)。

    パッケージ id は ``manual-<sha8>`` (sha8 = 本文 UTF-8 の sha256 先頭 8
    hex)、版は :data:`MANUAL_PACKAGE_VERSION` 固定。同じ内容を入れ直しても
    同じ版へ上書きされ、内容が変われば別パッケージになる。
    """
    if not file.filename:
        raise rag_filename_required_error()

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise rag_unsupported_format_error(ext)

    manager = state.cartridge_manager
    if manager is None:
        raise corpus_store_not_initialized_error()
    if state.embedder is None:
        raise embedder_not_initialized_error()

    start = time.time()
    content = await file.read()
    doc_name, doc_text = _prepare_document(file.filename, ext, content)

    cfg = get_config()
    meta = PackageMeta(
        id=f"{MANUAL_PACKAGE_PREFIX}{_content_sha8(doc_text)}",
        name=file.filename,
        version=MANUAL_PACKAGE_VERSION,
        language=str((cfg.get("i18n") or {}).get("locale") or "ja"),
        description=file.filename,
        # 未知キーは package.json のトップレベルへ書き戻され、読み直しで
        # ``_extra`` に戻る (c_05 §0.5)。統計の category 表示がこれを読む。
        _extra={"category": category, "origin": "manual_ingest"},
    )

    with tempfile.TemporaryDirectory(prefix="evoref-ingest-") as tmp:
        zip_path = Path(tmp) / package_filename(meta)
        try:
            write_package({doc_name: doc_text}, zip_path, meta)
            info = await manager.install(zip_path, embedder=state.embedder)
        except PackageError as exc:
            # 「文書が無い」「チャンクが 0 件」はどちらも投入内容の問題。
            logger.warning("Manual ingest rejected %s: %s", file.filename, exc)
            raise rag_no_chunks_error() from exc

    # 概算トークン数 (旧実装と同じ 2 文字 = 1 トークンの粗い見積り)。
    tokens_total = max(1, len(doc_text) // 2)
    elapsed = time.time() - start

    logger.info(
        "Ingested %s as corpus package %s v%s: %d chunk(s) in %.2fs",
        file.filename, info.id, info.version, info.chunks, elapsed,
    )

    return RagIngestResponse(
        source=file.filename,
        chunks_created=info.chunks,
        tokens_total=tokens_total,
        ingest_time_sec=round(elapsed, 3),
    )


@router.get("/stats", response_model=RagStatsResponse)
async def get_rag_stats(state: AppState = Depends(get_app_state)):
    """corpus (文書由来チャンク) の統計情報。

    フィールド名は frontend (`RAGStats.svelte` / `stores/dashboard.ts`) が
    そのまま読むので据え置き、中身だけ corpus 由来へ移した:

    - ``total_chunks`` / ``total_vectors``: 全パッケージのチャンク数の合計
      (corpus は snapshot の全行に埋め込みを作るので 2 つは常に同値)
    - ``total_sources`` / ``sources``: インストール済みパッケージ (= 投入元
      文書) の数と内訳
    - ``index_size_mb``: 各パッケージの ``embeddings/`` の合計
    - ``created_at`` / ``last_reindex_at``: 版ディレクトリの更新時刻の最古 /
      最新。rebuild でも更新されるので、後者が「最後に索引を組んだ時刻」
    """
    logger.debug("GET /api/rag/stats")
    cfg = get_config()
    rag_cfg = cfg.get("rag", {})
    embedding_cfg = cfg.get("embedding", {})

    manager = state.cartridge_manager
    packages = [] if manager is None else list(manager.corpus.list_packages())

    total_chunks = sum(int(p.chunk_count) for p in packages)
    index_size_mb = round(sum(float(p.size_mb) for p in packages), 3)
    stored_dim = next(
        (int(p.embedding_dim) for p in packages if p.embedding_dim), None,
    )
    stored_model = next(
        (p.embedding_model_id for p in packages if p.embedding_model_id), None,
    )
    stamps = sorted(p.installed_at for p in packages if p.installed_at)

    sources = [
        RagSourceInfo(
            filename=p.meta.name or p.id,
            chunks=int(p.chunk_count),
            added_at=p.installed_at,
            category=str(p.meta._extra.get("category") or "document"),
        )
        for p in packages
    ]

    embedder_dim = (
        state.embedder.dim() if state.embedder is not None
        else int(embedding_cfg.get("dim", 1024))
    )

    return RagStatsResponse(
        total_chunks=total_chunks,
        total_vectors=total_chunks,
        total_sources=len(packages),
        index_size_mb=index_size_mb,
        embedding_dim=embedder_dim,
        embedding_dim_stored=stored_dim,
        embedding_dim_mismatch=bool(state.embedding_dim_mismatch),
        chunking_strategy=rag_cfg.get("chunking_strategy", "semantic"),
        created_at=stamps[0] if stamps else None,
        last_reindex_at=stamps[-1] if stamps else None,
        embedding_model=stored_model,
        embedding_backend=(
            state.embedder.backend_type() if state.embedder is not None else None
        ),
        sources=sources,
    )


@router.post("/reindex")
async def reindex_vectors(
    state: AppState = Depends(get_app_state),
    dry_run: bool = False,
    cartridge: str | None = None,
):
    """3 ストア (episodic / semantic / corpus) の索引を再構築する

    Query params:
        dry_run: True なら対象件数だけ返して実行しない
        cartridge: corpus パッケージ ID を指定すると当該パッケージのみ
    """
    from backend.free.rag.dimension_check import embedder_config_mismatch
    from backend.free.rag.reindex import plan_reindex, run_reindex

    if state.embedder is None:
        raise embedder_not_initialized_error()

    plan = plan_reindex(state, cartridge_id=cartridge)
    if dry_run:
        return {
            "dry_run": True,
            "rag_chunks": plan.rag_chunks,
            "cartridge_chunks": plan.cartridge_chunks,
            "cartridges": plan.cartridges,
            "memory_notes": plan.memory_notes,
        }

    # migrate 直後で embedder が旧モデルのまま実行すると、旧モデルのベクトルで
    # 再構築して stale マーカーまでクリアしてしまう (順序依存の罠)。embed
    # サーバ再起動 + embedder reload が済むまで実行を拒否する。
    stale = embedder_config_mismatch(state)
    if stale is not None:
        raise HTTPException(
            status_code=409,
            detail=msg(
                "error.rag.stale_embedder", current=stale[0], expected=stale[1],
            ),
        )

    result = await run_reindex(state, cartridge_id=cartridge)
    return {
        "dry_run": False,
        "rag_chunks": result.rag_chunks,
        "cartridge_chunks": result.cartridge_chunks,
        "cartridges_rebuilt": result.cartridges_rebuilt,
        "cartridges_failed": result.cartridges_failed,
        "memory_notes_reset": result.memory_notes_reset,
        "elapsed_sec": result.elapsed_sec,
        "embedding_dim_mismatch": state.embedding_dim_mismatch,
    }


@router.post("/calibrate-thresholds")
async def calibrate_thresholds_endpoint(
    state: AppState = Depends(get_app_state),
):
    """再構築済み RAG ベクトルのスコア分布から rag.* 閾値の推奨値を返す。

    埋め込みモデル切替後にスコアスケールへ閾値を合わせるための分布ヒューリスティック。
    正解ラベル不在のため **自動適用はせず提案のみ** を返す (UI でレビュー → 適用)。
    """
    from backend.free.rag.threshold_calibration import calibrate_thresholds

    if state.embedder is None:
        raise embedder_not_initialized_error()
    return calibrate_thresholds(state)
