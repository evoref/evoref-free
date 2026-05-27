"""ナレッジカートリッジ管理"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backend.io import CheckpointEntry, CheckpointStore
from backend.log_config import get_logger
from backend.trace_context import get_trace_id
from backend.free.rag.chunker import SemanticChunker
from backend.free.rag.vector_store import DEFAULT_MEMMAP_THRESHOLD, VectorStore
from backend.free.rag.text_extractor import (
    SUPPORTED_DOC_EXTENSIONS,
    extract_text,
    parse_csv_to_chunks,
)

if TYPE_CHECKING:
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("rag.cartridge_manager")


# 進捗通知コールバック型: フレーム dict を受け取って await 可能に処理する
ProgressCallback = Callable[[dict], Awaitable[None]]
# キャンセル判定型: True ならキャンセル要求あり
CancelCheck = Callable[[], bool]


class CartridgeInstallCancelled(Exception):
    """インストール処理がユーザー要求でキャンセルされたことを示す例外"""


async def _emit(callback: ProgressCallback | None, frame: dict) -> None:
    """progress_callback が None でない場合のみ呼び出すユーティリティ"""
    if callback is not None:
        await callback(frame)


def _check_cancel(cancel_check: CancelCheck | None) -> None:
    """cancel_check が True を返したら CartridgeInstallCancelled を raise"""
    if cancel_check is not None and cancel_check():
        raise CartridgeInstallCancelled("Cartridge install cancelled")


@dataclass
class CartridgeInfo:
    """カートリッジメタ情報"""
    id: str
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    language: str = "ja"
    status: str = "installed"  # "installed" | "loaded"
    doc_count: int = 0
    chunks: int = 0
    size_mb: float = 0.0
    priority: float = 1.0
    installed_at: str = ""
    compatibility: str = ">=0.1.0"
    needs_rebuild: bool = False
    tool_hints: list[dict] = field(default_factory=list)
    # 埋め込みモデル情報
    embedding_model: str = ""
    embedding_backend: str = ""
    embedding_dim: int = 0


@dataclass
class CartridgeData:
    """ロード済みカートリッジのデータ"""
    info: CartridgeInfo
    store: VectorStore
    # 未構築カートリッジでは None (従来動作にフォールバック)。
    centroid: np.ndarray | None = None


CENTROID_FILENAME = "centroid.npy"


def _load_centroid(cart_dir: Path) -> np.ndarray | None:
    """cart_dir/centroid.npy があれば読み込み、なければ None"""
    path = cart_dir / CENTROID_FILENAME
    if not path.exists():
        return None
    try:
        arr = np.load(str(path))
        if arr.ndim != 1 or arr.size == 0:
            return None
        return arr.astype(np.float32, copy=False)
    except (OSError, ValueError) as e:
        logger.warning("Failed to load centroid for %s: %s", cart_dir.name, e)
        return None


def _save_centroid(cart_dir: Path, store: VectorStore) -> np.ndarray | None:
    """VectorStore から centroid を計算し cart_dir/centroid.npy に保存する。

    保存できた場合はその centroid を返す。ベクトル未登録時は None。
    """
    centroid = store.compute_centroid()
    path = cart_dir / CENTROID_FILENAME
    if centroid is None:
        # 既存の centroid.npy が残っていたら削除 (再構築で空になった場合)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        return None
    np.save(str(path), centroid)
    return centroid


class CartridgeManager:
    """ナレッジカートリッジの install / load / unload / uninstall / search"""

    def __init__(self, cartridges_dir: str | Path, rag_config: dict | None = None):
        self.cartridges_dir = Path(cartridges_dir)
        self.cartridges_dir.mkdir(parents=True, exist_ok=True)
        self.rag_config = rag_config or {}
        self._memmap_threshold = self.rag_config.get(
            "memmap_threshold", DEFAULT_MEMMAP_THRESHOLD,
        )
        self._max_loaded = int(self.rag_config.get("max_loaded_cartridges", 20))
        self._large_warn_chunks = int(
            self.rag_config.get("large_cartridge_warn_chunks", 50000),
        )
        gate_cfg = self.rag_config.get("cartridge_gate", {}) or {}
        self._gate_enabled = bool(gate_cfg.get("enabled", True))
        self._gate_threshold = float(gate_cfg.get("threshold", 0.3))
        self._gate_max_cartridges = int(gate_cfg.get("max_cartridges", 10))
        self._gate_fallback_when_empty = bool(
            gate_cfg.get("fallback_when_empty", False),
        )
        cluster_cfg = self.rag_config.get("cluster_index", {}) or {}
        self._cluster_enabled = bool(cluster_cfg.get("enabled", True))
        self._cluster_threshold = int(cluster_cfg.get("threshold", 5000))
        self._cluster_n_probe_ratio = float(
            cluster_cfg.get("n_probe_ratio", 0.125),
        )
        # OrderedDict: 末尾 = 最近使用、先頭 = LRU 候補
        self._loaded: OrderedDict[str, CartridgeData] = OrderedDict()
        self._registry: dict[str, CartridgeInfo] = {}
        self._on_change_callbacks: list[Callable[[str, str], None]] = []
        self._learned_patterns: LearnedPatternStore | None = None
        self._load_registry()

    def set_learned_patterns(self, learned_patterns: LearnedPatternStore) -> None:
        """学習済みパターンストアを設定"""
        self._learned_patterns = learned_patterns

    @staticmethod
    def _find_file_in_zip(names: list[str], target: str) -> str | None:
        """ZIP 内のファイル名リストからターゲットファイルを検索する

        ディレクトリ階層を無視し、ファイル名（basename）が一致する
        最初のエントリを返す。

        Args:
            names: ZIP 内のファイル名リスト
            target: 検索するファイル名（例: "cartridge.json"）

        Returns:
            見つかった場合は ZIP 内のフルパス、見つからなければ None
        """
        for name in names:
            basename = name.split("/")[-1]
            if basename == target:
                return name
        return None

    async def install(
        self,
        zip_path: str | Path,
        embedder: EmbeddingBackend,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> CartridgeInfo:
        """ZIP パッケージからカートリッジをインストール

        Args:
            zip_path: カートリッジ ZIP パス
            embedder: EmbeddingBackend インスタンス（埋め込み生成用）
            progress_callback: 進捗フレーム ({"phase","status","current"?,"total"?,"detail"?}) を受け取る async コールバック
            cancel_check: True を返したら CartridgeInstallCancelled を raise する判定関数

        Raises:
            CartridgeInstallCancelled: cancel_check が True を返したとき
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        if not zipfile.is_zipfile(str(zip_path)):
            raise ValueError("Not a valid ZIP file")

        # ZIP 展開・メタデータ解析
        await _emit(progress_callback, {"phase": "extract", "status": "running"})
        _check_cancel(cancel_check)
        meta, cart_dir, doc_count = self._extract_cartridge_zip(zip_path)
        cart_id = meta["id"]
        await _emit(progress_callback, {"phase": "extract", "status": "done", "detail": cart_id})

        # ドキュメントをチャンク分割 + 埋め込み
        docs_dir = cart_dir / "docs"
        store = VectorStore(cart_dir, memmap_threshold=self._memmap_threshold)
        store.load()

        total_chunks, _ = await self._chunk_and_embed_docs(
            docs_dir, store, embedder,
            progress_callback=progress_callback, cancel_check=cancel_check,
        )
        store.save()

        await _emit(progress_callback, {"phase": "index", "status": "running"})
        _check_cancel(cancel_check)
        self._maybe_build_cluster_index(store)
        await _emit(progress_callback, {"phase": "index", "status": "done"})

        # レジストリ登録
        info = self._register_cartridge(meta, cart_dir, store, doc_count, total_chunks)

        logger.info("Installed cartridge %s: %d docs, %d chunks", cart_id, doc_count, total_chunks)
        return info

    def _extract_cartridge_zip(self, zip_path: Path) -> tuple[dict, Path, int]:
        """ZIP パッケージを展開し、メタデータ・ドキュメント・eval.json を配置する

        Args:
            zip_path: カートリッジ ZIP ファイルパス

        Returns:
            (meta, cart_dir, doc_count) のタプル

        Raises:
            ValueError: cartridge.json がない、id が未設定、既にインストール済み
        """
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            names = zf.namelist()

            meta_file = self._find_file_in_zip(names, "cartridge.json")
            if meta_file is None:
                raise ValueError("cartridge.json not found in ZIP")

            meta = json.loads(zf.read(meta_file))
            cart_id = meta.get("id")
            if not cart_id:
                raise ValueError("cartridge.json missing 'id' field")
            if cart_id in self._registry:
                raise ValueError(f"Cartridge '{cart_id}' already installed")

            cart_dir = self.cartridges_dir / cart_id
            cart_dir.mkdir(parents=True, exist_ok=True)

            # docs/ ファイル展開（early continue で平坦化）
            docs_dir = cart_dir / "docs"
            docs_dir.mkdir(exist_ok=True)
            doc_count = 0
            for name in names:
                if name.endswith("/"):
                    continue
                parts = name.replace("\\", "/").split("/")
                if "docs" not in parts:
                    continue
                filename = parts[-1]
                target = docs_dir / filename
                target.write_bytes(zf.read(name))
                doc_count += 1

            # cartridge.json 保存
            (cart_dir / "cartridge.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # eval.json 展開（存在する場合）
            eval_file = self._find_file_in_zip(names, "eval.json")
            if eval_file is not None:
                (cart_dir / "eval.json").write_bytes(zf.read(eval_file))

        return meta, cart_dir, doc_count

    def _register_cartridge(
        self,
        meta: dict,
        cart_dir: Path,
        store: VectorStore,
        doc_count: int,
        total_chunks: int,
    ) -> CartridgeInfo:
        """カートリッジをレジストリに登録しロード状態にする"""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        index_size = store.index_path.stat().st_size / (1024 * 1024) if store.index_path.exists() else 0
        cart_id = meta["id"]

        info = CartridgeInfo(
            id=cart_id,
            name=meta.get("name", cart_id),
            version=meta.get("version", "1.0.0"),
            author=meta.get("author", ""),
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            language=meta.get("language", "ja"),
            status="loaded",
            doc_count=doc_count,
            chunks=total_chunks,
            size_mb=round(index_size, 3),
            priority=meta.get("priority", 1.0),
            installed_at=now,
            compatibility=meta.get("compatibility", ">=0.1.0"),
            tool_hints=meta.get("tool_hints", []),
            embedding_model=str(store.store_info.get("embedding_model", "")),
            embedding_backend=str(store.store_info.get("embedding_backend", "")),
            embedding_dim=int(store.store_info.get("embedding_dim", 0) or 0),
        )

        self._registry[cart_id] = info
        self._enforce_max_loaded_before_insert(cart_id)
        centroid = _save_centroid(cart_dir, store)
        self._loaded[cart_id] = CartridgeData(
            info=info, store=store, centroid=centroid,
        )
        self._warn_if_large_cartridge(info)
        self._save_registry()

        return info

    def load(self, cartridge_id: str) -> CartridgeInfo:
        """カートリッジを検索対象に追加"""
        if cartridge_id not in self._registry:
            raise KeyError(f"Cartridge '{cartridge_id}' not found")

        if cartridge_id in self._loaded:
            # LRU: 既にロード済みなら末尾へ移動
            self._loaded.move_to_end(cartridge_id)
            return self._loaded[cartridge_id].info

        cart_dir = self.cartridges_dir / cartridge_id
        store = VectorStore(cart_dir, memmap_threshold=self._memmap_threshold)
        store.load()

        info = self._registry[cartridge_id]
        info.status = "loaded"
        self._enforce_max_loaded_before_insert(cartridge_id)
        centroid = _load_centroid(cart_dir)
        self._loaded[cartridge_id] = CartridgeData(
            info=info, store=store, centroid=centroid,
        )
        self._warn_if_large_cartridge(info)
        self._save_registry()

        logger.info(
            "Loaded cartridge %s (centroid=%s)",
            cartridge_id, "yes" if centroid is not None else "no",
        )

        # tool_hints のキーワードを tool_routing カテゴリで学習パターンに追加
        self._register_tool_hint_patterns(info)

        self._notify_change("load", cartridge_id)
        return info

    def unload(self, cartridge_id: str) -> CartridgeInfo:
        """カートリッジを検索対象から除外"""
        if cartridge_id not in self._registry:
            raise KeyError(f"Cartridge '{cartridge_id}' not found")

        if cartridge_id in self._loaded:
            del self._loaded[cartridge_id]

        info = self._registry[cartridge_id]
        info.status = "installed"
        self._save_registry()

        logger.info("Unloaded cartridge %s", cartridge_id)

        # tool_hints のキーワードの source_count を減算
        self._unregister_tool_hint_patterns(info)

        self._notify_change("unload", cartridge_id)
        return info

    def uninstall(self, cartridge_id: str) -> None:
        """カートリッジを完全削除"""
        if cartridge_id not in self._registry:
            raise KeyError(f"Cartridge '{cartridge_id}' not found")

        # ロード中なら解除
        self._loaded.pop(cartridge_id, None)

        # ファイル削除
        cart_dir = self.cartridges_dir / cartridge_id
        if cart_dir.exists():
            shutil.rmtree(str(cart_dir))

        del self._registry[cartridge_id]
        self._save_registry()

        logger.info("Uninstalled cartridge %s", cartridge_id)
        self._notify_change("uninstall", cartridge_id)

    async def rebuild(
        self,
        cartridge_id: str,
        embedder: EmbeddingBackend,
    ) -> CartridgeInfo:
        """カートリッジのベクトルインデックスを再構築

        既存ドキュメントを現在の rag_config で再チャンキングし、
        embedder で実際の埋め込みを生成する。

        Args:
            cartridge_id: カートリッジ ID
            embedder: EmbeddingBackend インスタンス（埋め込み生成用）
        """
        if cartridge_id not in self._registry:
            raise KeyError(f"Cartridge '{cartridge_id}' not found")

        was_loaded = cartridge_id in self._loaded

        # ロード中なら解除
        if was_loaded:
            del self._loaded[cartridge_id]

        cart_dir = self.cartridges_dir / cartridge_id
        docs_dir = cart_dir / "docs"
        chunks_dir = cart_dir / "chunks"

        if not docs_dir.exists() or not any(docs_dir.iterdir()):
            raise ValueError(f"No documents found for cartridge '{cartridge_id}'")

        self._clean_old_index(cart_dir, chunks_dir)

        # 新しい VectorStore を作成
        store = VectorStore(cart_dir, memmap_threshold=self._memmap_threshold)
        store.load()

        total_chunks, doc_count = await self._chunk_and_embed_docs(docs_dir, store, embedder)

        store.save()

        self._maybe_build_cluster_index(store)

        # レジストリ更新
        info = self._registry[cartridge_id]
        info.chunks = total_chunks
        info.doc_count = doc_count
        index_path = store.index_path
        info.size_mb = round(
            index_path.stat().st_size / (1024 * 1024) if index_path.exists() else 0, 3
        )
        # 埋め込みモデル情報を更新
        info.embedding_model = str(store.store_info.get("embedding_model", ""))
        info.embedding_backend = str(store.store_info.get("embedding_backend", ""))
        info.embedding_dim = int(store.store_info.get("embedding_dim", 0) or 0)
        info.needs_rebuild = False

        # centroid を再計算・保存
        centroid = _save_centroid(cart_dir, store)

        # リビルド前にロードされていたなら再ロード
        if was_loaded:
            info.status = "loaded"
            self._loaded[cartridge_id] = CartridgeData(
                info=info, store=store, centroid=centroid,
            )
        else:
            info.status = "installed"

        self._save_registry()

        logger.info(
            "Rebuilt cartridge %s: %d docs, %d chunks",
            cartridge_id, doc_count, total_chunks,
        )
        return info

    @staticmethod
    def _clean_old_index(cart_dir: Path, chunks_dir: Path) -> None:
        """古いベクトルインデックスとチャンクディレクトリを削除する（I/O）"""
        for fname in (
            "index_q8.npy", "scales.npy", "index.npy",
            "metadata.json", CENTROID_FILENAME, "cluster_index.npz",
        ):
            fpath = cart_dir / fname
            if fpath.exists():
                fpath.unlink()
        if chunks_dir.exists():
            shutil.rmtree(str(chunks_dir))

    async def _chunk_and_embed_docs(
        self,
        docs_dir: Path,
        store: VectorStore,
        embedder: EmbeddingBackend,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
        enable_checkpoint: bool = False,
        cart_dir: Path | None = None,
    ) -> tuple[int, int]:
        """ドキュメントのチャンク分割・埋め込み・VectorStore 登録の共通処理

        Args:
            docs_dir: ドキュメントディレクトリ
            store: 登録先 VectorStore
            embedder: EmbeddingBackend インスタンス
            progress_callback: 各ドキュメント処理開始/終了時に進捗フレームを emit する async コールバック
            cancel_check: 各ドキュメント先頭で True を返したら CartridgeInstallCancelled を raise
            enable_checkpoint: ``True`` の場合、``cart_dir / "install_progress.jsonl"`` に
                doc 単位の処理結果を append-only で記録し、再 install 時に
                既処理 doc を自動 skip する。``cart_dir`` の指定が必須。
            cart_dir: checkpoint ファイルの配置先 (``enable_checkpoint=True`` 時必須)。

        Returns:
            (total_chunks, doc_count) のタプル

        Raises:
            CartridgeInstallCancelled: cancel_check が True を返したとき
            ValueError: ``enable_checkpoint=True`` で ``cart_dir`` が None のとき
        """
        chunker = SemanticChunker(
            chunk_size=self.rag_config.get("chunk_size", 512),
            chunk_overlap=self.rag_config.get("chunk_overlap", 128),
            max_chunk=self.rag_config.get("max_chunk", 512),
        )

        # 事前にループ対象を確定 (進捗 total を確定するため)
        target_docs = [
            p for p in sorted(docs_dir.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
        ]
        total_docs = len(target_docs)

        # checkpoint 初期化 (opt-in)
        checkpoint: CheckpointStore | None = None
        skip_keys: set[str] = set()
        if enable_checkpoint:
            if cart_dir is None:
                raise ValueError("cart_dir is required when enable_checkpoint=True")
            checkpoint = CheckpointStore(cart_dir / "install_progress.jsonl")
            skip_keys = checkpoint.load_done_keys()

        total_chunks = 0
        doc_count = 0
        for idx, doc_path in enumerate(target_docs, start=1):
            _check_cancel(cancel_check)
            await _emit(progress_callback, {
                "phase": "chunk_embed", "status": "running",
                "current": idx, "total": total_docs, "detail": doc_path.name,
            })

            # 前回までに done として記録済の doc はスキップ
            if doc_path.name in skip_keys:
                doc_count += 1
                continue

            try:
                chunks, source_text = self._extract_and_chunk(doc_path, chunker)
                if chunks:
                    # ソーステキスト保存（Contextual Retrieval 用）
                    if source_text:
                        store.save_source_text(doc_path.name, source_text)

                    vecs = await embedder.embed(chunks, is_query=False)
                    store.add_vectors(
                        vecs, chunks, source=doc_path.name, category="cartridge",
                        embedding_model=embedder.model_name(),
                        embedding_backend=embedder.backend_type(),
                    )
                    total_chunks += len(chunks)
                doc_count += 1
                if checkpoint is not None:
                    checkpoint.record(CheckpointEntry(
                        key=doc_path.name,
                        status="done",
                        ts=time.time(),
                        trace_id=get_trace_id(),
                    ))
            except BaseException as e:
                # 例外時は checkpoint に failed を記録してから伝播
                if checkpoint is not None:
                    checkpoint.record(CheckpointEntry(
                        key=doc_path.name,
                        status="failed",
                        ts=time.time(),
                        trace_id=get_trace_id(),
                        detail={"error": repr(e)},
                    ))
                raise

        await _emit(progress_callback, {
            "phase": "chunk_embed", "status": "done",
            "current": total_docs, "total": total_docs,
        })

        # 完了 → checkpoint ファイルを削除
        if checkpoint is not None:
            checkpoint.unlink()

        # store_info を Embedder の現在値で更新
        if total_chunks > 0:
            store.mark_reindexed(
                embedding_model=embedder.model_name(),
                embedding_backend=embedder.backend_type(),
                embedding_dim=embedder.dim(),
            )

        return total_chunks, doc_count

    def _apply_cartridge_gate(
        self, query_vec: np.ndarray, cart_ids: list[str],
    ) -> list[str]:
        """

        query_vec (L2 正規化) と各カートリッジの centroid の cosine 類似度を
        計算し、threshold 以上かつ max_cartridges 件までに絞る。

        - centroid 未構築のカートリッジは常に通過 (従来動作 / 段階移行のため)
        - 全件不通過時の挙動は ``fallback_when_empty`` で切替:
          False (既定) なら空リストを返してカートリッジ検索を skip し、
          True なら旧挙動の全件フォールバック (recall 保護) を維持する。
        """
        if not self._gate_enabled or not cart_ids:
            return cart_ids

        # クエリ正規化
        qnorm = float(np.linalg.norm(query_vec))
        if qnorm < 1e-9:
            return cart_ids
        q = (query_vec / qnorm).astype(np.float32)

        scored: list[tuple[str, float]] = []
        uncomputed: list[str] = []
        for cid in cart_ids:
            data = self._loaded[cid]
            if data.centroid is None:
                # 未構築カートリッジは無条件通過
                uncomputed.append(cid)
                continue
            sim = float(np.dot(q, data.centroid))
            if sim >= self._gate_threshold:
                scored.append((cid, sim))

        # スコア上位 max_cartridges 件 (<=0 で無制限) + 未構築を常に含める
        if self._gate_max_cartridges > 0:
            scored.sort(key=lambda x: -x[1])
            scored = scored[: self._gate_max_cartridges]

        passed = [cid for cid, _ in scored] + uncomputed
        # 全件不通過時: fallback_when_empty で挙動を切替
        if not passed:
            if self._gate_fallback_when_empty:
                logger.debug(
                    "Cartridge gate: 0 passed of %d (threshold=%.2f), "
                    "falling back to all",
                    len(cart_ids), self._gate_threshold,
                )
                return cart_ids
            logger.debug(
                "Cartridge gate: 0 passed of %d (threshold=%.2f), "
                "skipping cartridge search (fallback_when_empty=False)",
                len(cart_ids), self._gate_threshold,
            )
            return []
        if len(passed) < len(cart_ids):
            logger.debug(
                "Cartridge gate: %d/%d passed (threshold=%.2f)",
                len(passed), len(cart_ids), self._gate_threshold,
            )
        # 元の LRU 順を保持して返す (ラウンドロビンの決定性維持)
        passed_set = set(passed)
        return [cid for cid in cart_ids if cid in passed_set]

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[str, float, str]]:
        """ロード済みカートリッジ横断検索

        各カートリッジから top_k 件を取得し、ラウンドロビンでインターリーブして
        返す。スコアでグローバルソートしてから top_k で切り詰める旧実装は、
        埋め込みスコアが近い (例: 同ジャンルの複数カートリッジ) ケースで
        特定のカートリッジが結果を独占し、他のカートリッジが完全に欠落する
        問題があった

        ラウンドロビンにより各カートリッジが少なくとも 1 件は結果に含まれる
        ことが保証される。最終件数は ``top_k * n_loaded`` を上限とし、上流の
        reranker / salience ranker / `_ensure_cartridge_fairness` で最終的な
        絞り込みを行う。
        """
        if not self._loaded:
            return []

        # 各カートリッジから top_k 件を取得 (priority 補正済み、スコア降順)
        # items() のスナップショットを取ってから move_to_end する (イテレーション中の変更を避ける)
        cart_ids = list(self._loaded.keys())
        cart_ids = self._apply_cartridge_gate(query_vec, cart_ids)
        n_loaded = len(cart_ids)
        if n_loaded == 0:
            return []

        per_cart: list[list[tuple[str, float, str]]] = []
        for cart_id in cart_ids:
            data = self._loaded[cart_id]
            results = data.store.search(query_vec, top_k=top_k)
            adjusted = [
                (f"{cart_id}:{chunk_id}", score * data.info.priority, text)
                for chunk_id, score, text in results
            ]
            per_cart.append(adjusted)
            self._loaded.move_to_end(cart_id)

        # ラウンドロビン: i 周回目に全カートリッジから i 番目に良いチャンクを
        # 取り出す。これにより各カートリッジの 1 位 → 2 位 → ... が交互に
        # 並び、必ず全カートリッジが代表される。
        interleaved: list[tuple[str, float, str]] = []
        max_len = max((len(c) for c in per_cart), default=0)
        for i in range(max_len):
            for cart_results in per_cart:
                if i < len(cart_results):
                    interleaved.append(cart_results[i])

        # 上流レイヤが余裕を持って候補を扱えるよう、上限は top_k * n_loaded。
        return interleaved[: top_k * n_loaded]

    def list_cartridges(self) -> list[CartridgeInfo]:
        """インストール済みカートリッジ一覧"""
        return list(self._registry.values())

    def get_cartridge(self, cartridge_id: str) -> CartridgeInfo | None:
        """カートリッジ情報取得"""
        return self._registry.get(cartridge_id)

    @property
    def loaded(self) -> dict[str, CartridgeData]:
        """ロード済みカートリッジマップを返す"""
        return self._loaded

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    def on_change(self, callback: Callable[[str, str], None]) -> None:
        """カートリッジ変更コールバックを登録

        Args:
            callback: (event, cartridge_id) を受け取る関数。
                      event は "load" | "unload" | "uninstall" のいずれか。
        """
        self._on_change_callbacks.append(callback)

    def _notify_change(self, event: str, cartridge_id: str) -> None:
        """登録済みコールバックにカートリッジ変更を通知する"""
        for cb in self._on_change_callbacks:
            try:
                cb(event, cartridge_id)
            except Exception as e:
                logger.warning("Cartridge change callback failed: %s", e)

    def get_loaded_stores(self) -> dict[str, VectorStore]:
        """ロード済みカートリッジの VectorStore マップを返す"""
        return {cid: data.store for cid, data in self._loaded.items()}

    def check_dimension_consistency(self, embedder_dim: int) -> list[str]:
        """全カートリッジの次元と現在の Embedder 次元を照合

        不一致のカートリッジは `needs_rebuild = True` を立て、ID を返す。
        起動時 / リロード時に呼び出す。

        Returns:
            次元不一致で needs_rebuild になったカートリッジ ID のリスト
        """
        mismatched: list[str] = []
        registry_changed = False
        for cart_id, info in self._registry.items():
            cart_dim = info.embedding_dim
            if cart_dim and cart_dim != embedder_dim:
                if not info.needs_rebuild:
                    info.needs_rebuild = True
                    registry_changed = True
                mismatched.append(cart_id)
                logger.warning(
                    "Cartridge '%s' embedding_dim=%d != current embedder dim=%d, "
                    "marked needs_rebuild. Run 'evoref reindex --cartridge %s'.",
                    cart_id, cart_dim, embedder_dim, cart_id,
                )
        if registry_changed:
            self._save_registry()
        return mismatched

    def _register_tool_hint_patterns(self, info: CartridgeInfo) -> None:
        """カートリッジの tool_hints キーワードを tool_routing パターンとして登録"""
        if self._learned_patterns is None or not info.tool_hints:
            return
        for hint in info.tool_hints:
            for pattern in hint.get("patterns", []):
                if len(pattern) >= 2:
                    self._learned_patterns.add_pattern(
                        pattern, category="tool_routing",
                    )
        logger.debug(
            "Registered tool_hint patterns for cartridge %s", info.id,
        )

    def _unregister_tool_hint_patterns(self, info: CartridgeInfo) -> None:
        """カートリッジ unload 時: tool_hints キーワードの source_count を減算"""
        if self._learned_patterns is None or not info.tool_hints:
            return
        for hint in info.tool_hints:
            for pattern in hint.get("patterns", []):
                if len(pattern) >= 2:
                    self._learned_patterns.decrement_source_count(pattern)
        logger.debug(
            "Unregistered tool_hint patterns for cartridge %s", info.id,
        )

    def get_tool_hints(self) -> list[dict]:
        """ロード済みカートリッジの tool_hints を集約して返す"""
        hints: list[dict] = []
        for data in self._loaded.values():
            hints.extend(data.info.tool_hints)
        return hints

    @staticmethod
    def _extract_and_chunk(
        doc_path: Path, chunker: SemanticChunker,
    ) -> tuple[list[str], str]:
        """ドキュメントからテキスト抽出 + チャンク分割

        CSV は行ごとに1チャンク（ヘッダー付与）で分割する（§4.9.9）。
        PDF は pypdf でテキスト抽出後、SemanticChunker でチャンキング。
        その他はテキスト読込み後、SemanticChunker でチャンキング。

        Returns:
            (chunks, source_text) のタプル。chunks が空の場合も source_text は返す。
        """
        suffix = doc_path.suffix.lower()

        if suffix == ".csv":
            chunks = parse_csv_to_chunks(doc_path)
            return chunks, "\n".join(chunks)

        text = extract_text(doc_path)
        if not text.strip():
            return [], ""
        return chunker.chunk(text), text

    def _load_registry(self) -> None:
        """レジストリをロード

        status が "loaded" のカートリッジは VectorStore をディスクから
        自動的にメモリへ復元する。インデックスファイルが欠損している場合は
        status を "installed" にリセットする。

        永続化 I/O は `CartridgeRegistryStore` に委譲し、本メソッドは
        ドメイン側 (auto_load 起動 + status リセット) に専念する。
        """
        from backend.free.rag.cartridge_registry_store import CartridgeRegistryStore

        loaded = CartridgeRegistryStore.load(self.cartridges_dir)
        if loaded is None:
            return
        self._registry = loaded

        # status: loaded のカートリッジを自動ロード
        needs_save = False
        for cart_id, info in self._registry.items():
            if info.status != "loaded":
                continue
            changed = self._auto_load_cartridge(cart_id, info)
            if changed:
                needs_save = True

        if needs_save:
            self._save_registry()

    def _auto_load_cartridge(self, cart_id: str, info: CartridgeInfo) -> bool:
        """起動時にカートリッジを自動ロードする

        Args:
            cart_id: カートリッジ ID
            info: カートリッジ情報

        Returns:
            レジストリの保存が必要な変更があった場合 True
        """
        cart_dir = self.cartridges_dir / cart_id
        index_path = cart_dir / "index_q8.npy"

        if not index_path.exists():
            logger.warning(
                "Auto-load skipped %s: index_q8.npy missing, resetting to installed",
                cart_id,
            )
            info.status = "installed"
            return True

        try:
            store = VectorStore(cart_dir, memmap_threshold=self._memmap_threshold)
            store.load()
            self._enforce_max_loaded_before_insert(cart_id)
            centroid = _load_centroid(cart_dir)
            self._loaded[cart_id] = CartridgeData(
                info=info, store=store, centroid=centroid,
            )
            self._warn_if_large_cartridge(info)
            logger.info(
                "Auto-loaded cartridge %s on startup (centroid=%s)",
                cart_id, "yes" if centroid is not None else "no",
            )
            return False
        except Exception as exc:
            logger.warning(
                "Auto-load failed for %s: %s, resetting to installed "
                "and cleaning corrupt index",
                cart_id, exc,
            )
            for fname in (
                "index_q8.npy", "scales.npy", "index.npy",
                "metadata.json", CENTROID_FILENAME, "cluster_index.npz",
            ):
                fpath = cart_dir / fname
                if fpath.exists():
                    try:
                        fpath.unlink()
                    except OSError:
                        pass
            info.status = "installed"
            info.needs_rebuild = True
            return True

    def _enforce_max_loaded_before_insert(self, about_to_insert: str) -> None:
        """新規 load 前に LRU eviction を行い、上限を維持する

        ``about_to_insert`` は eviction の対象外 (再ロード時の自己 evict 防止)。
        ``max_loaded_cartridges`` が 0 以下の場合は無制限。
        """
        if self._max_loaded <= 0:
            return
        # 挿入後のサイズが上限を超えないよう、現在サイズ >= max なら evict
        while len(self._loaded) >= self._max_loaded:
            victim_id: str | None = None
            for cid in self._loaded:  # 先頭 = 最古参
                if cid != about_to_insert:
                    victim_id = cid
                    break
            if victim_id is None:
                return
            victim_info = self._loaded.pop(victim_id).info
            if victim_id in self._registry:
                self._registry[victim_id].status = "installed"
            logger.info(
                "Cartridge LRU eviction: %s (max_loaded=%d, L3)",
                victim_id, self._max_loaded,
            )
            self._unregister_tool_hint_patterns(victim_info)
            self._notify_change("unload", victim_id)

    def _maybe_build_cluster_index(self, store: VectorStore) -> None:
        """設定と閾値に応じて cluster index を構築する"""
        if not self._cluster_enabled:
            return
        store.build_cluster_index(
            threshold=self._cluster_threshold,
            n_probe_ratio=self._cluster_n_probe_ratio,
        )

    def _warn_if_large_cartridge(self, info: CartridgeInfo) -> None:
        """巨大カートリッジ load 時に WARNING を出す"""
        if self._large_warn_chunks <= 0:
            return
        if info.chunks >= self._large_warn_chunks:
            logger.warning(
                "Large cartridge loaded: %s has %d chunks (>= %d). "
                "Search latency may degrade until cluster index L2 is built.",
                info.id, info.chunks, self._large_warn_chunks,
            )

    def _save_registry(self) -> None:
        """レジストリを保存 (infra 層 `CartridgeRegistryStore` に委譲)"""
        from backend.free.rag.cartridge_registry_store import CartridgeRegistryStore

        CartridgeRegistryStore.save(self._registry, self.cartridges_dir)
