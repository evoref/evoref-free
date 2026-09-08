"""`CartridgeManager` — `CorpusStore` への薄いファサード

カートリッジは c_16 §4.3 で **corpus パッケージ** に置き換わった。実体は
:mod:`backend.free.rag.corpus` (パッケージ形式 / インストーラ / 実行時ストア)
にあり、本モジュールは既存の呼出面 (`/api/cartridges` / 検索パイプライン /
factory 配線 / Pro のイベントハンドラ) が使っている **名前だけ** を残す。

## 旧実装から落としたもの (c_16 §8)

| 落としたもの | 理由 |
|---|---|
| ``CartridgeInfo.priority`` | 順位式の ``store_prior`` へ。掛けた値を閾値に流すと閾値を偽装する |
| ``needs_rebuild`` / ``docs_digest`` / ``docs_changed`` | ``content_digest`` + ``chunker_version`` で決定論再現できる |
| ``CartridgeRegistryStore`` (``registry.json``) | ``corpus/manifest.json`` の ``active`` / ``loaded`` |
| カートリッジごとの ``VectorStore`` レイアウト | 版ディレクトリ = 1 ``EvidenceStore`` |

``search_detailed`` の 4 タプルは ``(f"{package_id}:{evidence_id}", cosine,
score, text)``。``score`` は ``CorpusStore.search`` が計算した c_16 §7.2 の
順位式 (``cos × freshness × confidence × store_prior``) で、呼出側
(``search_pipeline._search_corpus_layer``) は **順位付けには score、ゲートには
素の cosine** を使う (§7.1: 合成スコアを閾値に流さない)。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.rag.corpus.package import PackageError, PackageMeta
from backend.free.rag.corpus.store import (
    DEFAULT_CARTRIDGE_GATE_THRESHOLD,
    CorpusInstallCancelled,
    CorpusPackage,
    CorpusStore,
    merge_rag_evidence_config,
    resolve_cartridge_gate_threshold,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.vector_store import VectorStore

logger = get_logger("rag.cartridge_manager")

#: 進捗通知コールバック型 / キャンセル判定型 (旧名のまま維持)。
ProgressCallback = Callable[[dict], Awaitable[None]]
CancelCheck = Callable[[], bool]


class CartridgeInstallCancelled(CorpusInstallCancelled):
    """インストール処理がユーザー要求でキャンセルされたことを示す例外。

    API 層 (`/api/cartridges/install/stream`) がこの名前で捕まえている。
    """


@dataclass
class CartridgeInfo:
    """API / CLI に返すパッケージの見え方 (`PackageMeta` + 実行時の状態)。

    ``priority`` / ``needs_rebuild`` / ``docs_digest`` / ``kind`` は無い
    (c_16 §8)。
    """

    id: str
    name: str
    version: str = "1.0.0"
    author: str = ""
    license: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    language: str = "ja"
    status: str = "installed"  # "installed" | "loaded"
    doc_count: int = 0
    chunks: int = 0
    size_mb: float = 0.0
    installed_at: str = ""
    compatibility: str = ">=0.1.0"
    tool_hints: list[dict] = field(default_factory=list)
    #: 現在 active なパッケージ版 (``version`` と同じ値。版切替の明示用)。
    active_version: str = ""
    #: ``docs/`` の内容ダイジェスト (再現性の鍵)。
    content_digest: str = ""
    embedding_model_id: str = ""
    embedding_dim: int = 0
    schema_version: int = 1


def cartridge_info_of(package: CorpusPackage) -> CartridgeInfo:
    """:class:`CorpusPackage` を :class:`CartridgeInfo` にする。"""
    meta: PackageMeta = package.meta
    return CartridgeInfo(
        id=meta.id,
        name=meta.name,
        version=meta.version,
        author=meta.author,
        license=meta.license,
        description=meta.description,
        tags=list(meta.tags),
        language=meta.language,
        status="loaded" if package.loaded else "installed",
        doc_count=package.doc_count,
        chunks=package.chunk_count,
        size_mb=package.size_mb,
        installed_at=package.installed_at,
        compatibility=meta.compatibility,
        tool_hints=list(meta.tool_hints),
        active_version=meta.version,
        content_digest=meta.content_digest,
        embedding_model_id=package.embedding_model_id,
        embedding_dim=package.embedding_dim,
        schema_version=meta.schema_version,
    )


class CartridgeManager:
    """corpus パッケージの install / load / unload / uninstall / search。

    Args:
        corpus_dir: ``local/memory/corpus``。
        rag_config: ``rag`` セクション (c_16 §9 の ``memory.evidence.*`` を
            重ねたもの。:func:`merge_rag_evidence_config` を使う)。
        debug_logger: ``rag.jsonl`` へゲートの採否を残す。
        embedder: 埋め込みバックエンド。起動順の都合で後から
            :meth:`set_embedder` で差せる。
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        rag_config: dict | None = None,
        debug_logger: "DebugLogger | None" = None,
        embedder: "EmbeddingBackend | None" = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.rag_config = rag_config or {}
        self._corpus = CorpusStore(
            self.corpus_dir,
            embedding_backend=embedder,
            rag_config=self.rag_config,
            debug_logger=debug_logger,
        )
        self._learned_patterns: "LearnedPatternStore | None" = None

    # ── 配線互換 ──

    @property
    def corpus(self) -> CorpusStore:
        """実体の :class:`CorpusStore`。"""
        return self._corpus

    def set_embedder(self, embedder: "EmbeddingBackend | None") -> None:
        """埋め込みバックエンドを差し替える。"""
        self._corpus.set_embedding_backend(embedder)

    def set_learned_patterns(self, learned_patterns: "LearnedPatternStore") -> None:
        """学習済みパターンストアを設定 (配線互換のため保持。参照しない)。

        以前はロードのたびに ``tool_hints`` を learned_patterns へ +0.15 で
        登録していたが、減衰と 200 件上限の押し出しで会話から学習した語を
        追い出していたため撤去した (2026-09-02 監査 R-C4)。``tool_hints`` は
        :meth:`get_tool_hints` 経由で ``tool_call_judge`` が直接読む。
        """
        self._learned_patterns = learned_patterns

    def on_change(self, callback: Callable[[str, str], None]) -> None:
        """``(event, cartridge_id)`` を受けるコールバックを登録する。"""
        self._corpus.on_change(callback)

    # ── install / rebuild / uninstall ──

    async def install(
        self,
        zip_path: str | Path,
        embedder: "EmbeddingBackend | None" = None,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> CartridgeInfo:
        """`.evocart` パッケージをインストールする。

        Raises:
            CartridgeInstallCancelled: ``cancel_check`` が ``True`` を返した。
            ValueError: パッケージ形式が壊れている (:class:`PackageError`)。
        """
        if embedder is not None:
            self._corpus.set_embedding_backend(embedder)
        try:
            result = await self._corpus.install(
                zip_path, progress_cb=progress_callback, cancel_check=cancel_check,
            )
        except CorpusInstallCancelled as e:
            raise CartridgeInstallCancelled(str(e)) from e
        return cartridge_info_of(result.package)

    async def rebuild(
        self,
        cartridge_id: str,
        embedder: "EmbeddingBackend | None" = None,
    ) -> CartridgeInfo:
        """``docs/`` からチャンクと埋め込みを作り直す。"""
        if embedder is not None:
            self._corpus.set_embedding_backend(embedder)
        result = await self._corpus.rebuild(cartridge_id)
        return cartridge_info_of(result.package)

    def uninstall(self, cartridge_id: str) -> None:
        """全版をディスクから消す。"""
        self._corpus.uninstall(cartridge_id)

    # ── load / unload ──

    def load(self, cartridge_id: str) -> CartridgeInfo:
        return cartridge_info_of(self._corpus.load(cartridge_id))

    def unload(self, cartridge_id: str) -> CartridgeInfo:
        return cartridge_info_of(self._corpus.unload(cartridge_id))

    # ── 目録 ──

    def list_cartridges(self) -> list[CartridgeInfo]:
        return [cartridge_info_of(p) for p in self._corpus.list_packages()]

    def get_cartridge(self, cartridge_id: str) -> CartridgeInfo | None:
        package = self._corpus.get(cartridge_id)
        return None if package is None else cartridge_info_of(package)

    @property
    def loaded(self) -> dict[str, CorpusPackage]:
        """検索対象のパッケージ (id → パッケージ)。

        呼出側 (``chat_recorder`` / ``LearningScheduler`` / Level 2 trainer) が
        使うのは **キー (= id)** だけ。
        """
        return self._corpus.loaded

    @property
    def loaded_count(self) -> int:
        return len(self._corpus.loaded_ids)

    def get_loaded_ids(self) -> list[str]:
        return self._corpus.loaded_ids

    def get_tool_hints(self) -> list[dict]:
        return self._corpus.get_tool_hints()

    def get_loaded_stores(self) -> dict[str, "VectorStore"]:
        """空 dict を返す (corpus パッケージは不変なので追記対象が無い)。

        旧実装はここでカートリッジの ``VectorStore`` を返し、sleep-time の
        contextual prefix 生成 (step 5.8) がチャンクへプレフィックスを **書き
        戻して** いた。corpus は「版が履歴そのもので、内容は不変」(c_16 §2.1)
        なので、稼働中のパッケージへ書き戻す経路は持たない。プレフィックスを
        付けたければ作成側 (Pro のパッケージ作成) が docs に入れる。
        """
        return {}

    def check_dimension_consistency(self, embedder_dim: int) -> list[str]:
        """埋め込み次元が現在のモデルと違うパッケージを検索対象から外す。"""
        return self._corpus.check_dimension_consistency(embedder_dim)

    # ── 検索 ──

    def search_detailed(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        rescore_candidates: int = 0,  # noqa: ARG002 — 呼出面の互換のため受ける
    ) -> list[tuple[str, float, float, str]]:
        """ロード済みパッケージ横断検索 (素の cosine と順位式スコアを分けて返す)。

        戻り値は ``(f"{package_id}:{evidence_id}", cosine, score, text)``。
        ``score`` は :meth:`CorpusStore.search` が **ストアの中で** 計算した
        c_16 §7.2 の順位式 ``cos × freshness × confidence × store_prior``。
        呼出側は cosine を品質判定 / floor に、``score`` を順位付けに使う —
        合成した値を閾値に流すと閾値を偽装する (2026-09-02 監査 S-A4、
        c_16 §7.1)。

        件数の上限は旧実装と同じ ``top_k × ロード数`` に保つ。上流
        (salience ranker / ``_ensure_cartridge_fairness``) が候補プールの広さを
        前提にしているため。

        Args:
            rescore_candidates: 受け取るが使わない。``EvidenceStore`` は
                snapshot 生成時に作ったクラスタ索引と int8 → float32 の 2 段
                検索を内部で持ち、候補数は c_16 §6.3 の構造上限で決まる。
        """
        loaded = self._corpus.loaded_ids
        if not loaded:
            return []
        hits = self._corpus.search(
            "", query_vec, top_k=top_k * len(loaded), per_package_k=top_k,
        )
        return [
            (
                f"{hit.package_id}:{hit.evidence_id}",
                hit.cosine,
                hit.score,
                hit.text,
            )
            for hit in hits
        ]

    def search(
        self, query_vec: np.ndarray, top_k: int = 5, rescore_candidates: int = 0,
    ) -> list[tuple[str, float, str]]:
        """順位式スコアの 3 タプル (:meth:`search_detailed` の薄い包み)。"""
        return [
            (chunk_id, score, text)
            for chunk_id, _cosine, score, text in self.search_detailed(
                query_vec, top_k=top_k, rescore_candidates=rescore_candidates,
            )
        ]


__all__ = [
    "DEFAULT_CARTRIDGE_GATE_THRESHOLD",
    "CancelCheck",
    "CartridgeInfo",
    "CartridgeInstallCancelled",
    "CartridgeManager",
    "PackageError",
    "ProgressCallback",
    "cartridge_info_of",
    "merge_rag_evidence_config",
    "resolve_cartridge_gate_threshold",
]
