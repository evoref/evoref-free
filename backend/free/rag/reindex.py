"""3 ストアの埋め込み索引を作り直す (c_16 §6.1)

埋め込みモデルを切り替えたあとに呼ばれ、``Evidence`` を持つ 3 つのストアの
ベクトル索引を現在の Embedder で組み直す。

| ストア | 実体 | 作り直し方 |
|---|---|---|
| episodic | :class:`~backend.free.memory.episodic.store.EpisodicStore` | ``EvidenceStore.embed_and_index_snapshot()`` |
| semantic | :class:`~backend.free.memory.semantic.store.SemanticStore` | 同上 |
| corpus | :class:`~backend.free.rag.corpus.store.CorpusStore` | パッケージごとに ``rebuild`` (``docs/`` から作り直し) |

旧実装が持っていた「メイン RAG ``VectorStore`` の再埋め込み」
(``_reindex_rag_store``) は無くなった。手動投入の書き込み先が corpus
パッケージへ移り (``POST /api/rag/ingest``)、``state.vector_store`` は
閾値較正 / 次元検査の読み手だけが残る単独ストアになったため。

``embed_and_index_snapshot`` は増分で、前の版に同じ本文 (``text_hash``) が
残っていれば int8 の行を流用する。埋め込みモデルが変われば manifest の宣言と
食い違うので全件やり直しになる — つまり本 module は「モデルを替えたときだけ
全件、それ以外は差分」で正しく振る舞う。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from backend.exceptions import (
    LLMConnectionError,
    LLMRequestRejectedError,
    LLMTimeoutError,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("rag.reindex")


@dataclass
class ReindexPlan:
    """再構築対象のドライランサマリ。

    **フィールド名は変えていない** — API 応答 (``POST /api/rag/reindex``) と
    frontend (``EmbeddingRebuildButton.svelte`` / ``api/rag.ts``) と CLI
    (``evoref reindex``) が同じ名前で読むため。c_16 で中身の意味だけ移した:

    Attributes:
        rag_chunks: **semantic ストア**の構造化事実の件数 (旧: メイン RAG
            VectorStore のチャンク数)。
        cartridge_chunks: corpus パッケージのチャンク数の合計。
        cartridges: corpus パッケージ id の一覧。
        memory_notes: **episodic ストア**の会話由来ノートの件数。
    """

    rag_chunks: int = 0
    cartridge_chunks: int = 0
    cartridges: list[str] = field(default_factory=list)
    memory_notes: int = 0


@dataclass
class ReindexResult:
    """再構築結果 (フィールド名の据え置き理由は :class:`ReindexPlan` と同じ)。

    Attributes:
        rag_chunks: semantic ストアで索引に入れた件数。
        cartridge_chunks: 作り直した corpus パッケージのチャンク数の合計。
        cartridges_rebuilt: 成功した corpus パッケージ id。
        cartridges_failed: 失敗した corpus パッケージ id。
        memory_notes_reset: episodic ストアで索引に入れた件数 (旧: 埋め込みを
            None に落としたノート数)。
        elapsed_sec: 所要秒。
    """

    rag_chunks: int = 0
    cartridge_chunks: int = 0
    cartridges_rebuilt: list[str] = field(default_factory=list)
    cartridges_failed: list[str] = field(default_factory=list)
    memory_notes_reset: int = 0
    elapsed_sec: float = 0.0


def _store_len(store: Any) -> int:
    """ストアの件数を安全に数える (読めなければ 0)。"""
    if store is None:
        return 0
    try:
        return len(store)
    except (TypeError, OSError) as exc:
        logger.warning("Reindex plan: failed to count records: %s", exc)
        return 0


def plan_reindex(
    state: "AppState",
    cartridge_id: str | None = None,
) -> ReindexPlan:
    """再構築対象を集計する (実際の再構築は行わない)。

    ``cartridge_id`` を指定した場合は corpus パッケージ 1 本だけを数える
    (episodic / semantic は対象外なので 0 のまま)。
    """
    plan = ReindexPlan()
    if cartridge_id is None:
        plan.rag_chunks = _store_len(getattr(state, "semantic_memory", None))
        plan.memory_notes = _store_len(getattr(state, "episodic_memory", None))

    manager = getattr(state, "cartridge_manager", None)
    if manager is not None:
        for info in manager.list_cartridges():
            if cartridge_id is not None and info.id != cartridge_id:
                continue
            plan.cartridges.append(info.id)
            plan.cartridge_chunks += info.chunks
    return plan


def _translate_embed_error(exc: httpx.HTTPError, label: str) -> Exception:
    """埋め込み呼び出しの素の httpx 例外を構造化エラーへ変換する。

    そのまま投げると ``backend.error_handlers`` の汎用 "Unhandled server
    error" になって生 traceback が API 呼出元へ漏れる (2026-06-27 実機)。
    """
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError(
            f"Embedding request timed out during reindex ({label})",
        )
    if isinstance(exc, httpx.HTTPStatusError):
        # サーバーは応答したが個別リクエストを拒否した (例: context 長超過)。
        # 「接続できません」に丸めると実態と違い、案内した再起動も効かない。
        try:
            detail = (
                exc.response.json().get("error", {}).get("message")
                or exc.response.text[:300]
            )
        except Exception:
            detail = exc.response.text[:300] if exc.response.text else str(exc)
        return LLMRequestRejectedError(
            f"Embedding server rejected request during reindex: {detail}",
            detail=detail,
        )
    # httpx.ReadError 等の transient I/O 失敗もここ (リトライ対象外で素通し)。
    return LLMConnectionError(
        f"Embedding server connection failed during reindex ({label})",
    )


async def reindex_evidence_store(
    store: Any,
    embedder: "EmbeddingBackend",
    label: str,
) -> int:
    """``EvidenceStore`` を持つ 1 ストアの埋め込みと索引を作り直す。

    Args:
        store: :class:`~backend.free.memory.episodic.store.EpisodicStore` /
            :class:`~backend.free.memory.semantic.store.SemanticStore`
            (``.evidence`` を持つラッパ)、または ``EvidenceStore`` そのもの。
        embedder: 現在の埋め込みバックエンド。
        label: ログ用のストア名。

    版がまだ 1 つも無い場合は先にラッパ側の ``create_snapshot()`` を呼ぶ。
    ``EvidenceStore.embed_and_index_snapshot`` は active snapshot が無いと
    黙って 0 を返すので、ここを飛ばすと「呼んでいるのに一生効かない」
    経路になる。ラッパ経由にするのは在メモリの写しも作り直させるため。

    Returns:
        索引に入れた件数 (ストアが無い / 空なら 0)。
    """
    if store is None:
        return 0
    evidence = getattr(store, "evidence", store)
    # 起動時に差した Embedder とモデル切替後の Embedder は別インスタンスに
    # なりうる。ここで必ず現在のものへ差し替える。
    evidence.embedding_backend = embedder
    try:
        if not evidence.manifest.active_snapshot:
            if int(getattr(evidence.manifest, "events_since_snapshot", 0)) <= 0:
                logger.info("Reindex: %s store is empty, nothing to index", label)
                return 0
            await store.create_snapshot()
        indexed = int(await evidence.embed_and_index_snapshot())
    except httpx.HTTPError as exc:
        raise _translate_embed_error(exc, label) from exc
    logger.info("Reindex: %s store indexed %d record(s)", label, indexed)
    return indexed


async def run_reindex(
    state: "AppState",
    cartridge_id: str | None = None,
) -> ReindexResult:
    """3 ストアのベクトル索引を現在の Embedder で作り直す。

    Args:
        state: AppState。
        cartridge_id: 指定があればその corpus パッケージだけ作り直す。
            ``None`` なら episodic / semantic / 全 corpus パッケージ。
    """
    embedder = state.embedder
    if embedder is None:
        raise RuntimeError("Embedder is not initialized")

    t0 = time.monotonic()
    result = ReindexResult()

    if cartridge_id is None:
        result.memory_notes_reset = await reindex_evidence_store(
            getattr(state, "episodic_memory", None), embedder, "episodic",
        )
        result.rag_chunks = await reindex_evidence_store(
            getattr(state, "semantic_memory", None), embedder, "semantic",
        )

    manager = getattr(state, "cartridge_manager", None)
    if manager is not None:
        target_ids = (
            [cartridge_id] if cartridge_id is not None
            else [c.id for c in manager.list_cartridges()]
        )
        for cart_id in target_ids:
            try:
                info = await manager.rebuild(cart_id, embedder)
                result.cartridges_rebuilt.append(cart_id)
                result.cartridge_chunks += info.chunks
            except Exception as exc:
                result.cartridges_failed.append(cart_id)
                logger.error("Reindex failed for corpus package %s: %s", cart_id, exc)

    # 状態更新: embed 切替 reindex マーカーを消してから次元不一致フラグを再評価。
    # (フル reindex かつパッケージ全成功時のみ。特定パッケージのみの reindex や、
    # 失敗が残る場合は stale 状態が解消していないためマーカーを残す。)
    try:
        from backend.free.rag.dimension_check import (
            check_embedding_dim_consistency,
            clear_embed_reindex_required,
        )
        if cartridge_id is None and not result.cartridges_failed:
            clear_embed_reindex_required()
        check_embedding_dim_consistency(state)
    except Exception as exc:
        logger.warning("Post-reindex dimension check failed: %s", exc)

    result.elapsed_sec = round(time.monotonic() - t0, 3)
    logger.info(
        "Reindex complete: semantic=%d, corpus=%d (%d packages, %d failed), "
        "episodic=%d, %.2fs",
        result.rag_chunks, result.cartridge_chunks,
        len(result.cartridges_rebuilt), len(result.cartridges_failed),
        result.memory_notes_reset, result.elapsed_sec,
    )
    return result


__all__ = [
    "ReindexPlan",
    "ReindexResult",
    "plan_reindex",
    "reindex_evidence_store",
    "run_reindex",
]
