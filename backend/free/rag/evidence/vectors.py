"""Evidence Store の埋め込み版ディレクトリ (c_16 §2.1 / §5.6 / §6.1)

``embeddings/<model_id>/v<N>/`` の中身:

- ``index_q8.npy`` / ``scales.npy`` / ``cluster_index.npz`` — :class:`VectorStore` と同じ
- ``ids.npy`` (``S24`` の ASCII バイト列、c_05 §0.5.5) / ``text_hash.npy`` / ``embed_side.npy`` — 行の列
- ``stamp.json`` — この版がどの snapshot の版に対応するか (版名・行数・id ハッシュ) と
  モデル・次元。**最後に書く** ので、無い版は書きかけとして扱う (読まず、流用もしない)

G0 の ``metadata.json`` (行ごとの dict の JSON 配列) は廃止した — 50k 行で 14MB、
``json.dumps`` でループが 75ms 止まる (G1 設計 §17.3)。どれも derived なので fsync
しない (壊れていれば作り直す、c_16 §5.6)。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.rag.evidence.columns import encode_id_column
from backend.free.rag.vector_store import DEFAULT_MEMMAP_THRESHOLD, VectorStore
from backend.io import atomic_write_text
from backend.log_config import get_logger

logger = get_logger("rag.evidence.vectors")

IDS_FILE = "ids.npy"
TEXT_HASH_FILE = "text_hash.npy"
EMBED_SIDE_FILE = "embed_side.npy"
STAMP_FILE = "stamp.json"


def ids_digest(ids: Iterable[Any]) -> str:
    """行 id の並びのハッシュ (sha256 先頭 16 桁)。埋め込み版と snapshot の照合に使う。"""
    joined = "\n".join(str(record_id) for record_id in ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _bytes_column(values: list[str]) -> np.ndarray:
    """文字列の列を固定幅の ASCII バイト列にする (幅は最長の値に合わせる)。"""
    if not values:
        return np.zeros(0, dtype="S1")
    return np.array([value.encode("ascii") for value in values])


def _str_column(array: np.ndarray) -> list[str]:
    return [value.decode("ascii") for value in array.tolist()]


class EvidenceVectorStore(VectorStore):
    """``metadata.json`` を持たない埋め込み版 (行 id は ``ids.npy``)。

    検索・クラスタ索引は :class:`VectorStore` のまま使う。本文は ``records.jsonl``
    から lazy に読むので ``chunks/`` を持たない。
    """

    def __init__(
        self,
        vectors_dir: str | Path,
        memmap_threshold: int = DEFAULT_MEMMAP_THRESHOLD,
        quantization: str = "int8",
    ) -> None:
        super().__init__(vectors_dir, memmap_threshold=memmap_threshold,
                         quantization=quantization)
        self.row_ids: list[str] = []
        #: 増分埋め込みの再利用鍵 (``embed_reuse_key``、埋め込み側 + 本文)。
        self.text_hashes: list[str] = []
        #: 行の埋め込み側 (``d`` / ``q:<mode>``、観測用)。
        self.embed_sides: list[str] = []
        #: ``stamp.json`` の中身 (無ければ空 = 書きかけ / 未生成)。
        self.stamp: dict[str, Any] = {}

    def _reset(self) -> None:
        self.row_ids = []
        self.text_hashes = []
        self.embed_sides = []
        self.stamp = {}
        self.store_info = {}
        self.vectors_q8 = None
        self.scales = None
        self._is_memmap = False

    def load(self) -> None:
        """版ディレクトリを読む。``stamp.json`` が無い / 列が揃わない版は空として扱う。"""
        self._reset()
        stamp = read_stamp(self.vectors_dir)
        if stamp is None:
            self._load_cluster_index()
            return
        try:
            ids = np.load(str(self.vectors_dir / IDS_FILE), allow_pickle=False)
            hashes = np.load(str(self.vectors_dir / TEXT_HASH_FILE), allow_pickle=False)
            sides = np.load(str(self.vectors_dir / EMBED_SIDE_FILE), allow_pickle=False)
            rows = int(ids.shape[0])
            mmap_mode = "r" if rows >= self._memmap_threshold else None
            q8 = np.load(str(self.index_q8_path), mmap_mode=mmap_mode)
            scales = np.load(str(self.scales_path), mmap_mode=mmap_mode)
        except (OSError, ValueError) as e:
            logger.warning("Ignoring an unreadable embedding version %s: %s", self.vectors_dir, e)
            self._load_cluster_index()
            return
        if not (len(q8) == len(scales) == len(hashes) == len(sides) == rows):
            logger.error(
                "Embedding version %s is misaligned (%d ids, %d vectors); ignoring it",
                self.vectors_dir, rows, len(q8),
            )
            self._load_cluster_index()
            return
        self.vectors_q8 = q8
        self.scales = scales
        self._is_memmap = mmap_mode is not None
        self.row_ids = _str_column(ids)
        self.text_hashes = _str_column(hashes)
        self.embed_sides = _str_column(sides)
        self.stamp = stamp
        self.store_info = {
            "embedding_model": str(stamp.get("model_id") or ""),
            "embedding_backend": str(stamp.get("backend_type") or ""),
            "embedding_dim": int(stamp.get("dim") or 0),
        }
        self._load_cluster_index()

    def save(self) -> None:
        """行列と行の列を書き、**最後に** ``stamp.json`` を書く。"""
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_writable()
        assert self.vectors_q8 is not None and self.scales is not None
        np.save(str(self.index_q8_path), self.vectors_q8)
        np.save(str(self.scales_path), self.scales)
        np.save(str(self.vectors_dir / IDS_FILE), encode_id_column(self.row_ids))
        np.save(str(self.vectors_dir / TEXT_HASH_FILE), _bytes_column(self.text_hashes))
        np.save(str(self.vectors_dir / EMBED_SIDE_FILE), _bytes_column(self.embed_sides))
        atomic_write_text(
            self.vectors_dir / STAMP_FILE,
            json.dumps(self.stamp, ensure_ascii=False, separators=(",", ":")),
        )

    def _row_id(self, row: int) -> str:
        return self.row_ids[row]

    def load_chunk(self, chunk_id: str) -> str:  # noqa: ARG002 — 本文は records.jsonl 側
        return ""

    @property
    def count(self) -> int:
        return len(self.row_ids)


def read_stamp(directory: Path | str) -> dict[str, Any] | None:
    """埋め込み版の ``stamp.json`` (無い / 読めなければ ``None``)。"""
    path = Path(directory) / STAMP_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.warning("unreadable %s: %s", path, e)
        return None
    return raw if isinstance(raw, dict) else None


__all__ = [
    "EMBED_SIDE_FILE",
    "IDS_FILE",
    "STAMP_FILE",
    "TEXT_HASH_FILE",
    "EvidenceVectorStore",
    "ids_digest",
    "read_stamp",
]
