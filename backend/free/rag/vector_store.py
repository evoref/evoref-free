"""numpy ベースのベクトルインデックス（int8 量子化 + memmap 対応）"""

import json
from pathlib import Path

import numpy as np

from backend.exceptions import VectorDimensionMismatchError
from backend.io import atomic_write_text
from backend.log_config import get_logger
from backend.utils import utc_now_dt

logger = get_logger("rag.vector_store")

# rescore 候補数のデフォルト下限
_DEFAULT_RESCORE_MIN = 50

# memmap 切替閾値のデフォルト（ベクトル件数）
DEFAULT_MEMMAP_THRESHOLD = 10000


def quantize_int8(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """float32 ベクトルを int8 に量子化

    Returns:
        (quantized, scales) — quantized: int8 (N, dim), scales: float32 (N, 1)
    """
    scales = np.abs(vectors).max(axis=1, keepdims=True).clip(min=1e-9)
    quantized = np.round(vectors / scales * 127).astype(np.int8)
    return quantized, scales.astype(np.float32)


def dequantize_int8(
    quantized: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """int8 ベクトルを float32 に復元"""
    return quantized.astype(np.float32) * scales / 127


def _kmeans_numpy(
    vectors: np.ndarray,
    k: int,
    n_iter: int = 8,
    seed: int = 42,
    batch: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """numpy 依存のみの単純 K-means 実装

    L2 距離で割り当てを行う。L2 正規化済み embedding では euclidean と
    cosine は単調関係のため spherical K-means と同等の挙動になる。
    外部ライブラリ (scikit-learn / faiss 等) への依存を避けるため numpy
    のみで実装する。

    Returns:
        (centroids (K, dim) float32, assignments (N,) int32)
    """
    rng = np.random.default_rng(seed)
    n = len(vectors)
    vectors = vectors.astype(np.float32, copy=False)
    k = int(min(max(1, k), n))
    # 初期化: ランダムに K 点を選択 (重複なし)
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = vectors[init_idx].astype(np.float32).copy()
    assignments = np.zeros(n, dtype=np.int32)

    for _ in range(n_iter):
        # ── assignment step (batched) ──
        c_norm_sq = (centroids ** 2).sum(axis=1)  # (K,)
        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = vectors[start:end]
            # dist^2 = ||x||^2 + ||c||^2 - 2 x·c; ||x||^2 は argmin で不要
            cross = chunk @ centroids.T  # (b, K)
            dists = -2.0 * cross + c_norm_sq[None, :]
            assignments[start:end] = dists.argmin(axis=1).astype(np.int32)

        # ── update step ──
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = assignments == c
            cnt = int(mask.sum())
            if cnt > 0:
                new_centroids[c] = vectors[mask].mean(axis=0)
            else:
                # 空クラスタ: ランダム再初期化 (収束助ける)
                new_centroids[c] = vectors[rng.integers(0, n)]
        centroids = new_centroids

    return centroids, assignments


class VectorStore:
    """numpy ベースのベクトル検索（int8 量子化 + memmap 対応）"""

    def __init__(
        self,
        vectors_dir: str | Path,
        memmap_threshold: int = DEFAULT_MEMMAP_THRESHOLD,
        quantization: str = "int8",
    ):
        """
        Args:
            vectors_dir: インデックスの配置ディレクトリ
            memmap_threshold: memmap に切り替えるベクトル件数の閾値
            quantization: 検索時の量子化利用方針 (config ``rag.quantization``)。

                - ``int8`` (既定): int8 粗検索で候補を絞り、候補のみ float32 に
                  復元して rescore する 2 段階検索
                - ``none``: 粗検索で候補を絞らず全ベクトルを float32 に復元して
                  正確なスコアを計算する (低速だが粗検索の取りこぼしが無い)

                ディスク上の保持形式は常に int8 で、本設定は検索時の挙動のみを
                変える。
        """
        self.quantization = quantization
        self.vectors_dir = Path(vectors_dir)
        self.chunks_dir = self.vectors_dir / "chunks"
        self.source_texts_dir = self.vectors_dir / "source_texts"

        # int8 量子化ファイルパス
        self.index_q8_path = self.vectors_dir / "index_q8.npy"
        self.scales_path = self.vectors_dir / "scales.npy"

        self.metadata_path = self.vectors_dir / "metadata.json"

        # int8 量子化ベクトル + スケール
        self.vectors_q8: np.ndarray | None = None  # int8 (N, dim)
        self.scales: np.ndarray | None = None       # float32 (N, 1)
        self.metadata: list[dict] = []
        # ストア情報ヘッダー（埋め込みモデル/バックエンド/次元等を記録）
        # 埋め込みモデル変更時のガードレール
        self.store_info: dict = {}

        # memmap 設定
        self._memmap_threshold = memmap_threshold
        self._is_memmap = False

        self.cluster_index_path = self.vectors_dir / "cluster_index.npz"
        self.cluster_centroids: np.ndarray | None = None  # float32 (K, dim)
        self.cluster_assignments: np.ndarray | None = None  # int32 (N,)
        self.cluster_n_probe: int = 0

    def _ensure_dirs(self) -> None:
        """ベクトル / chunk / source_text ディレクトリを作成する (冪等)。"""
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.source_texts_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """ディスクからインデックスをロード（int8 量子化形式）

        メタデータを先に読込み、件数が memmap_threshold 以上なら
        memmap モードでベクトルをロードする（OS ページキャッシュに委譲）。
        """
        self._ensure_dirs()

        # メタデータを先に読込み（件数で memmap 判定するため）
        if self.metadata_path.exists():
            with open(self.metadata_path, encoding="utf-8") as f:
                raw = json.load(f)
            self._unpack_metadata(raw)
        else:
            self.metadata = []
            self.store_info = {}

        use_memmap = len(self.metadata) >= self._memmap_threshold

        if self.index_q8_path.exists() and self.scales_path.exists():
            if use_memmap:
                # memmap モード: OS ページキャッシュに委譲
                self.vectors_q8 = np.load(str(self.index_q8_path), mmap_mode="r")
                self.scales = np.load(str(self.scales_path), mmap_mode="r")
                self._is_memmap = True
                logger.info(
                    "Loaded quantized vector index (memmap): %d vectors (int8)",
                    len(self.vectors_q8),
                )
            else:
                # 通常モード: 全件 RAM 常駐
                self.vectors_q8 = np.load(str(self.index_q8_path))
                self.scales = np.load(str(self.scales_path))
                self._is_memmap = False
                logger.info(
                    "Loaded quantized vector index: %d vectors (int8)",
                    len(self.vectors_q8),
                )
        else:
            self.vectors_q8 = None
            self.scales = None
            self._is_memmap = False

        self._load_cluster_index()

    def _load_cluster_index(self) -> None:
        """cluster_index.npz があればロード

        ベクトル数と assignments 長が不一致なら無効化する (整合性ガード)。
        """
        self.cluster_centroids = None
        self.cluster_assignments = None
        self.cluster_n_probe = 0
        if not self.cluster_index_path.exists():
            return
        if self.vectors_q8 is None:
            return
        try:
            data = np.load(str(self.cluster_index_path))
            centroids = data["centroids"]
            assignments = data["assignments"]
            n_probe = int(data["n_probe"])
        except (OSError, KeyError, ValueError) as e:
            logger.warning(
                "cluster_index.npz invalid for %s: %s (ignored)",
                self.vectors_dir.name, e,
            )
            return
        if len(assignments) != len(self.vectors_q8):
            logger.warning(
                "cluster_index.npz N mismatch (%d vs %d) for %s; ignored",
                len(assignments), len(self.vectors_q8), self.vectors_dir.name,
            )
            return
        self.cluster_centroids = centroids.astype(np.float32, copy=False)
        self.cluster_assignments = assignments.astype(np.int32, copy=False)
        self.cluster_n_probe = max(1, n_probe)
        logger.info(
            "Loaded cluster index: K=%d, n_probe=%d, N=%d for %s",
            len(self.cluster_centroids), self.cluster_n_probe,
            len(self.cluster_assignments), self.vectors_dir.name,
        )

    def build_cluster_index(
        self,
        threshold: int = 5000,
        n_probe_ratio: float = 0.125,
        n_iter: int = 8,
        seed: int = 42,
    ) -> bool:
        """IVF-KMeans クラスタインデックスを構築・保存する

        ベクトル数が ``threshold`` 未満なら構築せず False を返す (ブルート
        フォースの方が速い規模)。numpy のみで K-means を実行する。

        Returns:
            構築した場合 True、閾値未満などでスキップした場合 False
        """
        if self.vectors_q8 is None or self.scales is None:
            return False
        n_vectors = len(self.vectors_q8)
        if n_vectors < threshold:
            # 既存の index が残っている可能性があるので削除する
            if self.cluster_index_path.exists():
                try:
                    self.cluster_index_path.unlink()
                except OSError:
                    pass
            self.cluster_centroids = None
            self.cluster_assignments = None
            self.cluster_n_probe = 0
            return False

        k = max(16, int(np.sqrt(n_vectors)))
        k = min(k, n_vectors)
        n_probe = max(1, int(k * max(n_probe_ratio, 1e-6)))

        # 全件 float32 復元 (memmap 経由でも一度 RAM に展開)
        floats = dequantize_int8(
            np.asarray(self.vectors_q8), np.asarray(self.scales),
        )
        centroids, assignments = _kmeans_numpy(
            floats, k=k, n_iter=n_iter, seed=seed,
        )
        np.savez(
            str(self.cluster_index_path),
            centroids=centroids.astype(np.float32),
            assignments=assignments.astype(np.int32),
            n_probe=np.int32(n_probe),
        )
        self.cluster_centroids = centroids.astype(np.float32)
        self.cluster_assignments = assignments.astype(np.int32)
        self.cluster_n_probe = n_probe
        logger.info(
            "Built cluster index for %s: N=%d, K=%d, n_probe=%d",
            self.vectors_dir.name, n_vectors, k, n_probe,
        )
        return True

    def _ensure_writable(self) -> None:
        """memmap モードの場合、書き込み可能な通常配列に変換

        memmap は読み取り専用。書き込み操作（add/delete/update/save）の前に呼ぶ。
        Windows では memmap 中のファイルがロックされるため、save 前にも必須。
        """
        if self._is_memmap and self.vectors_q8 is not None:
            self.vectors_q8 = np.array(self.vectors_q8)
            self.scales = np.array(self.scales)
            self._is_memmap = False

    def save(self) -> None:
        """インデックスをディスクに永続化（int8 量子化形式）"""
        self._ensure_dirs()

        # memmap 中のファイルへの書き込みを防ぐため通常配列に変換
        self._ensure_writable()
        self._save_vectors()

        # _store_info をヘッダーとして metadata.json の先頭に書き出す
        payload = self._pack_metadata()
        atomic_write_text(
            self.metadata_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    # ── _store_info 永続化ヘルパー ──────────────────────

    def _pack_metadata(self) -> list[dict]:
        """metadata.json への書き出し用ペイロードを構築する

        先頭に `_store_info` マーカー付きエントリを置き、続いてチャンク
        メタデータを並べる。store_info が空のときはマーカーを付けない。
        """
        if not self.store_info:
            return list(self.metadata)
        header = {"_store_info": True, **self.store_info}
        return [header, *self.metadata]

    def _unpack_metadata(self, raw: list[dict]) -> None:
        """metadata.json から読み込んだリストを `store_info` と
        チャンクメタデータに分離する。

        - 先頭エントリが `_store_info` マーカー付きならヘッダーとして取り込む
        - マーカーがない旧フォーマットの場合は、ベクトル shape から
          次元を推定して store_info を自動生成する（ロード時に補完）
        """
        if not isinstance(raw, list):
            self.metadata = []
            self.store_info = {}
            return

        if raw and isinstance(raw[0], dict) and raw[0].get("_store_info") is True:
            header = dict(raw[0])
            header.pop("_store_info", None)
            self.store_info = header
            self.metadata = list(raw[1:])
        else:
            self.metadata = list(raw)
            self.store_info = {}

    def stored_dim(self) -> int | None:
        """保存済みベクトルの次元数を返す（インデックスが空なら None）

        - `store_info.embedding_dim` がある場合はそれを返す
        - なければベクトル shape[1] から推定し、可能なら store_info に補完
        """
        dim = self.store_info.get("embedding_dim") if self.store_info else None
        if dim:
            return int(dim)
        if self.vectors_q8 is not None and self.vectors_q8.ndim == 2 and len(self.vectors_q8) > 0:
            return int(self.vectors_q8.shape[1])
        return None

    def ensure_store_info(
        self,
        embedding_model: str,
        embedding_backend: str,
        embedding_dim: int,
    ) -> bool:
        """store_info が未設定の場合、現在の Embedder 情報で初期化する

        既存ストアに `_store_info` がない（旧フォーマット）かつ
        ベクトル件数が 0 件のときに、起動時に Embedder 情報で
        ヘッダーを充足する用途。次元が既存ベクトルと食い違う場合は
        書き換えず False を返す（呼び出し側で reindex を案内する）。

        Returns:
            store_info を更新したら True、変更なしなら False
        """
        existing_dim = self.stored_dim()
        if existing_dim is not None and existing_dim != embedding_dim:
            return False
        if not self.store_info:
            self.store_info = {
                "embedding_model": embedding_model,
                "embedding_backend": embedding_backend,
                "embedding_dim": int(embedding_dim),
                "created_at": utc_now_dt().isoformat(),
                "last_reindex_at": None,
            }
            return True
        # 既存 store_info に dim が抜けているケースの補完
        changed = False
        for key, value in (
            ("embedding_model", embedding_model),
            ("embedding_backend", embedding_backend),
            ("embedding_dim", int(embedding_dim)),
        ):
            if not self.store_info.get(key):
                self.store_info[key] = value
                changed = True
        return changed

    def mark_reindexed(
        self,
        embedding_model: str,
        embedding_backend: str,
        embedding_dim: int,
    ) -> None:
        """reindex 完了時に store_info を更新する"""
        now = utc_now_dt().isoformat()
        if not self.store_info:
            self.store_info = {"created_at": now}
        self.store_info["embedding_model"] = embedding_model
        self.store_info["embedding_backend"] = embedding_backend
        self.store_info["embedding_dim"] = int(embedding_dim)
        self.store_info["last_reindex_at"] = now

    def _save_vectors(self) -> None:
        """ベクトルファイルのみ保存"""
        if (
            self.vectors_q8 is not None
            and len(self.vectors_q8) > 0
            and self.scales is not None
        ):
            np.save(str(self.index_q8_path), self.vectors_q8)
            np.save(str(self.scales_path), self.scales)
        else:
            # 空の場合はファイル削除
            if self.index_q8_path.exists():
                self.index_q8_path.unlink()
            if self.scales_path.exists():
                self.scales_path.unlink()

    def add_vectors(
        self,
        vectors: np.ndarray,
        chunks: list[str],
        source: str,
        category: str = "document",
        embedding_model: str = "",
        embedding_backend: str = "",
        has_context: bool = False,
        speaker: str | None = None,
    ) -> list[str]:
        """ベクトル（float32）を受け取り、int8 量子化して追加

        has_context=True は「コンテキストプレフィックス生成の対象外」を表す。
        自己完結したチャンク（memory ノート等）は prefix を付けても価値が薄く、
        Step 5.8 (contextual) の source text 探索を空振りさせるだけのため、
        最初から has_context=True で登録し未処理スキャンから除外する。

        ``speaker`` は memory 由来チャンクの発話者 (``user`` / ``assistant`` /
        ``rag`` / ``system``)。検索側が「ユーザーが述べた事実」と「アシスタント
        自身が過去に答えた内容」を区別するために使う。``LongTermMemory.note_meta``
        にも同じ値が入るが、あちらは **プロセス内メモリのみで永続化されない**
        ため、再起動後は読めない。ドキュメント由来のチャンクでは ``None``。
        """
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")

        # 次元検証: 既存ストアと不整合な次元を弾く
        if vectors.ndim == 2 and len(vectors) > 0:
            new_dim = int(vectors.shape[1])
            stored = self.stored_dim()
            if stored is not None and stored != new_dim:
                raise VectorDimensionMismatchError(
                    f"New vector dim ({new_dim}) != stored dim ({stored}). "
                    f"Run 'evoref reindex' to rebuild vectors.",
                    stored_dim=stored,
                    new_dim=new_dim,
                )

        self._ensure_writable()

        logger.debug(
            "add_vectors: %d vectors, source=%s, category=%s, dim=%d",
            len(vectors), source, category,
            vectors.shape[1] if vectors.ndim > 1 else 0,
        )

        # float32 → int8 量子化
        new_q8, new_scales = quantize_int8(vectors)

        start_id = len(self.metadata)
        chunk_ids = []
        now = utc_now_dt().isoformat()

        for i, chunk in enumerate(chunks):
            chunk_id = f"{start_id + i:04d}"
            chunk_ids.append(chunk_id)

            meta_entry: dict = {
                "id": chunk_id,
                "source": source,
                "chunk_index": i,
                "created_at": now,
                "category": category,
                "tokens": max(1, len(chunk) // 2),
            }
            if has_context:
                meta_entry["has_context"] = True
            if speaker:
                meta_entry["speaker"] = speaker
            if embedding_model:
                meta_entry["embedding_model"] = embedding_model
            if embedding_backend:
                meta_entry["embedding_backend"] = embedding_backend
            self.metadata.append(meta_entry)

            chunk_path = self.chunks_dir / f"{chunk_id}.txt"
            chunk_path.write_text(chunk, encoding="utf-8")

        # int8 ベクトルを追加
        if self.vectors_q8 is None or len(self.vectors_q8) == 0:
            self.vectors_q8 = new_q8
            self.scales = new_scales
        else:
            self.vectors_q8 = np.vstack([self.vectors_q8, new_q8])
            self.scales = np.vstack([self.scales, new_scales])

        return chunk_ids

    def _restrict_to_clusters(
        self, query_f32: np.ndarray, top_k: int,
    ) -> np.ndarray | None:
        """クラスタインデックスで候補インデックスを絞り込む

        未構築 / 整合性なし / 候補が少なすぎる (< top_k) 場合は None を返し、
        呼び出し側に全件走査へフォールバックさせる。

        Returns:
            (M,) int64 の global index 配列、または None (フォールバック)
        """
        if (
            self.cluster_centroids is None
            or self.cluster_assignments is None
            or self.cluster_n_probe <= 0
        ):
            return None
        if len(self.cluster_centroids) == 0:
            return None
        q_n = float(np.linalg.norm(query_f32)) + 1e-9
        c_norms = np.linalg.norm(self.cluster_centroids, axis=1) + 1e-9
        sims = self.cluster_centroids @ query_f32 / (c_norms * q_n)
        n_probe = min(self.cluster_n_probe, len(self.cluster_centroids))
        if n_probe >= len(self.cluster_centroids):
            probe_clusters = np.arange(len(self.cluster_centroids))
        else:
            probe_clusters = np.argpartition(sims, -n_probe)[-n_probe:]
        mask = np.isin(self.cluster_assignments, probe_clusters)
        indices = np.nonzero(mask)[0]
        if len(indices) < top_k:
            # recall 保護: 候補が少ない場合は全件走査へフォールバック
            return None
        return indices

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        rescore_candidates: int = 0,
    ) -> list[tuple[str, float, str]]:
        """2段階検索: int8 粗検索 → float32 rescore

        Args:
            query_vec: クエリベクトル (float32)
            top_k: 返却する上位件数
            rescore_candidates: rescore 候補数（0 でデフォルト: max(50, top_k*3)）

        Returns:
            (chunk_id, score, chunk_text) のリスト
        """
        if self.vectors_q8 is None or len(self.vectors_q8) == 0:
            logger.debug("vector search: empty index, returning []")
            return []

        # 次元検証: クエリベクトルとストアの次元不一致を弾く
        if query_vec.ndim == 1:
            query_dim = int(query_vec.shape[0])
        else:
            query_dim = int(query_vec.shape[-1])
        stored = self.stored_dim()
        if stored is not None and stored != query_dim:
            raise VectorDimensionMismatchError(
                f"Query vector dim ({query_dim}) != stored dim ({stored}). "
                f"Run 'evoref reindex' to rebuild vectors.",
                stored_dim=stored,
                query_dim=query_dim,
            )

        n_vectors = len(self.vectors_q8)
        if self.quantization == "none":
            # 粗検索で絞らず全件を float32 rescore に回す
            rescore_candidates = n_vectors
        elif rescore_candidates <= 0:
            rescore_candidates = max(_DEFAULT_RESCORE_MIN, top_k * 3)

        # ── 第1段階: int8 粗検索 ──
        query_f32 = query_vec.astype(np.float32)

        restricted_indices = self._restrict_to_clusters(query_f32, top_k)

        if restricted_indices is None:
            # 通常経路: 全件 int8 粗検索
            dots = self.vectors_q8.astype(np.float32) @ query_f32
            n_candidates = min(rescore_candidates, n_vectors)
            if n_candidates >= n_vectors:
                candidate_indices = np.arange(n_vectors)
            else:
                candidate_indices = np.argpartition(dots, -n_candidates)[
                    -n_candidates:
                ]
        else:
            # IVF 経路: 対象クラスタ内のみ dot を計算
            sub_q8 = np.asarray(self.vectors_q8)[restricted_indices].astype(np.float32)
            sub_dots = sub_q8 @ query_f32
            n_candidates = min(rescore_candidates, len(restricted_indices))
            if n_candidates >= len(restricted_indices):
                candidate_indices = restricted_indices
            else:
                local_top = np.argpartition(sub_dots, -n_candidates)[
                    -n_candidates:
                ]
                candidate_indices = restricted_indices[local_top]

        # ── 第2段階: float32 rescore ──
        # 候補のみ float32 に復元して正確なコサイン類似度を計算
        restored = dequantize_int8(
            self.vectors_q8[candidate_indices],
            self.scales[candidate_indices],
        )
        norms = np.linalg.norm(restored, axis=1).clip(min=1e-9)
        query_norm = np.linalg.norm(query_f32).clip(min=1e-9)
        similarities = restored @ query_f32 / (norms * query_norm)

        # top_k 選択
        k = min(top_k, len(similarities))
        local_top = np.argsort(similarities)[-k:][::-1]

        results = []
        for local_idx in local_top:
            global_idx = candidate_indices[local_idx]
            chunk_id = self.metadata[global_idx]["id"]
            score = float(similarities[local_idx])
            chunk_text = self.load_chunk(chunk_id)
            results.append((chunk_id, score, chunk_text))

        logger.debug(
            "vector search (int8+rescore): %d/%d vectors, "
            "candidates=%d, top_k=%d, scores=[%s]",
            len(results), n_vectors, n_candidates, top_k,
            ", ".join(f"{s:.3f}" for _, s, _ in results),
        )
        return results

    def delete(self, chunk_ids: list[str]) -> int:
        """指定されたチャンクを削除"""
        self._ensure_writable()
        logger.debug("delete: removing %d chunk IDs", len(chunk_ids))
        ids_to_remove = set(chunk_ids)
        keep_indices = []

        for i, meta in enumerate(self.metadata):
            if meta["id"] not in ids_to_remove:
                keep_indices.append(i)
            else:
                chunk_path = self.chunks_dir / f"{meta['id']}.txt"
                if chunk_path.exists():
                    chunk_path.unlink()

        removed = len(self.metadata) - len(keep_indices)

        if keep_indices and self.vectors_q8 is not None:
            self.vectors_q8 = self.vectors_q8[keep_indices]
            self.scales = self.scales[keep_indices]
            self.metadata = [self.metadata[i] for i in keep_indices]
        else:
            if (
                self.vectors_q8 is not None
                and self.vectors_q8.ndim == 2
            ):
                dim = self.vectors_q8.shape[1]
                self.vectors_q8 = np.empty((0, dim), dtype=np.int8)
                self.scales = np.empty((0, 1), dtype=np.float32)
            else:
                self.vectors_q8 = None
                self.scales = None
            self.metadata = []

        return removed

    def compute_centroid(self) -> np.ndarray | None:
        """全ベクトルの L2 正規化済み平均 (centroid) を float32 で返す。

        int8 量子化を float32 に復元してから平均・正規化する。
        """
        if self.vectors_q8 is None or self.scales is None:
            return None
        if len(self.vectors_q8) == 0:
            return None
        floats = dequantize_int8(
            np.asarray(self.vectors_q8), np.asarray(self.scales),
        )
        mean = floats.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-9:
            return None
        return (mean / norm).astype(np.float32)

    def load_chunk(self, chunk_id: str) -> str:
        """チャンク本文を読込み"""
        chunk_path = self.chunks_dir / f"{chunk_id}.txt"
        if chunk_path.exists():
            return chunk_path.read_text(encoding="utf-8")
        return ""

    # ── ソーステキスト保存 / 読込み（Contextual Retrieval 用）──

    def save_source_text(self, source: str, text: str) -> None:
        """ソースドキュメントの全文を保存（プレフィックス生成時に参照）"""
        self.source_texts_dir.mkdir(parents=True, exist_ok=True)
        path = self.source_texts_dir / f"{source}.txt"
        path.write_text(text, encoding="utf-8")

    def load_source_text(self, source: str) -> str:
        """ソースドキュメントの全文を読込み"""
        path = self.source_texts_dir / f"{source}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def get_contextual_text(self, chunk_id: str) -> str:
        """プレフィックス付きチャンクテキストを返す（埋め込み・BM25 用）

        context_prefix がある場合は「プレフィックス + 改行 + 元テキスト」、
        ない場合は元テキストのみ。
        """
        meta = self._find_meta(chunk_id)
        chunk_text = self.load_chunk(chunk_id)
        if meta and meta.get("context_prefix"):
            return meta["context_prefix"] + "\n" + chunk_text
        return chunk_text

    def get_chunks_without_context(self) -> list[dict]:
        """コンテキストプレフィックス未生成のチャンク metadata リストを返す"""
        return [m for m in self.metadata if not m.get("has_context")]

    def mark_has_context(self, chunk_id: str) -> None:
        """source text が存在せずプレフィックス生成不能なチャンクを
        ``get_chunks_without_context`` の対象から恒久的に除外する

        ``update_context_prefix`` と異なり prefix もベクトルも更新しない
        （生成すべきプレフィックスが無いチャンク向け）。
        """
        self._ensure_writable()
        for meta in self.metadata:
            if meta["id"] == chunk_id:
                meta["has_context"] = True
                return

    def increment_access_count(self, chunk_id: str) -> int:
        """retrieval ヒット回数をメモリ上でインクリメントする

        Lazy Contextual Retrieval 用。hit 回数が閾値に達した chunk のみ
        プレフィックスを生成・永続化する判定に使う。永続化はプレフィックス
        生成時の save() までメモリ上に留まる。

        Returns:
            更新後の access_count。chunk_id が見つからない場合は 0。
        """
        for meta in self.metadata:
            if meta.get("id") == chunk_id:
                meta["access_count"] = int(meta.get("access_count", 0)) + 1
                return meta["access_count"]
        return 0

    def update_context_prefix(
        self, chunk_id: str, prefix: str, new_vector: np.ndarray | None = None,
    ) -> None:
        """チャンクのコンテキストプレフィックスとベクトルを更新"""
        self._ensure_writable()
        for i, meta in enumerate(self.metadata):
            if meta["id"] == chunk_id:
                meta["context_prefix"] = prefix
                meta["has_context"] = True
                if new_vector is not None and self.vectors_q8 is not None:
                    # float32 → int8 量子化して更新
                    vec = new_vector.reshape(1, -1)
                    q8, sc = quantize_int8(vec)
                    self.vectors_q8[i] = q8[0]
                    self.scales[i] = sc[0]
                return
        logger.warning("update_context_prefix: chunk_id=%s not found", chunk_id)

    def _find_meta(self, chunk_id: str) -> dict | None:
        """chunk_id に対応する metadata を返す"""
        for meta in self.metadata:
            if meta["id"] == chunk_id:
                return meta
        return None

    @property
    def index_path(self) -> Path:
        """ベクトルインデックスのパス（ストレージサイズ計測用）"""
        return self.index_q8_path

    @property
    def count(self) -> int:
        return len(self.metadata)

    @property
    def is_memmap(self) -> bool:
        """現在 memmap モードでロードされているか"""
        return self._is_memmap
