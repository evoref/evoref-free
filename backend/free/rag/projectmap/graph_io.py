"""``graph/nodes.json`` + ``graph/edges.npz`` の書き出し・読み込み (c_16 §4.4)

辺は snapshot の **行番号** (int32) で持つ — 文字列 id を並べるより軽く、
``numpy`` の argsort だけで両方向の近傍探索が O(E log E) で済む。
``nodes.json`` はその行番号への変換表 (``{ev_id: row}``)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.free.rag.evidence.store import EvidenceStore
from backend.free.rag.projectmap.graph import Edge, Node
from backend.io import AtomicWriter, atomic_write_text

GRAPH_DIR = "graph"
NODES_FILE = "nodes.json"
EDGES_FILE = "edges.npz"

#: ``etype`` の文字列 ↔ id (c_16 §4.4)。**値は永続化されるので変えない**。
ETYPE_IDS: dict[str, int] = {"contains": 0, "imports": 1, "calls": 2, "inherits": 3}
ETYPE_NAMES: dict[int, str] = {v: k for k, v in ETYPE_IDS.items()}


@dataclass(slots=True)
class GraphIndex:
    """読み込んだグラフ (行番号ベース)。"""

    id_to_row: dict[str, int]
    row_to_id: list[str]
    src: np.ndarray
    dst: np.ndarray
    etype: np.ndarray
    weight: np.ndarray

    def __len__(self) -> int:
        return int(self.src.shape[0])


def write_graph(
    directory: Path | str, store: EvidenceStore, nodes: list[Node], edges: list[Edge],
) -> None:
    """version ディレクトリへ ``graph/nodes.json`` + ``graph/edges.npz`` を書く。

    ``store`` は ``create_snapshot()`` 済みであること (行番号は snapshot 基準)。
    """
    graph_dir = Path(directory) / GRAPH_DIR
    graph_dir.mkdir(parents=True, exist_ok=True)
    snapshot = store.snapshot

    id_to_row: dict[str, int] = {}
    if snapshot is not None:
        for node in nodes:
            row = snapshot.row_of(node.id)
            if row is not None:
                id_to_row[node.id] = row

    atomic_write_text(
        graph_dir / NODES_FILE,
        json.dumps(id_to_row, ensure_ascii=False, indent=2),
    )

    src_rows: list[int] = []
    dst_rows: list[int] = []
    etypes: list[int] = []
    weights: list[float] = []
    for edge in edges:
        s = id_to_row.get(edge.src)
        d = id_to_row.get(edge.dst)
        if s is None or d is None:
            continue
        src_rows.append(s)
        dst_rows.append(d)
        etypes.append(ETYPE_IDS.get(edge.etype, 255))
        weights.append(float(edge.weight))

    with AtomicWriter(graph_dir / EDGES_FILE, mode="wb") as f:
        np.savez(
            f,
            src=np.array(src_rows, dtype=np.int32),
            dst=np.array(dst_rows, dtype=np.int32),
            etype=np.array(etypes, dtype=np.uint8),
            weight=np.array(weights, dtype=np.float32),
        )


def read_graph(directory: Path | str) -> GraphIndex:
    """version ディレクトリの ``graph/`` を読む (無ければ空)。"""
    graph_dir = Path(directory) / GRAPH_DIR
    nodes_path = graph_dir / NODES_FILE
    id_to_row: dict[str, int] = {}
    if nodes_path.exists():
        raw = json.loads(nodes_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            id_to_row = {str(k): int(v) for k, v in raw.items()}

    row_to_id: list[str] = [""] * ((max(id_to_row.values()) + 1) if id_to_row else 0)
    for node_id, row in id_to_row.items():
        if 0 <= row < len(row_to_id):
            row_to_id[row] = node_id

    edges_path = graph_dir / EDGES_FILE
    if edges_path.exists():
        with np.load(str(edges_path), allow_pickle=False) as data:
            src = np.array(data["src"], dtype=np.int32)
            dst = np.array(data["dst"], dtype=np.int32)
            etype = np.array(data["etype"], dtype=np.uint8)
            weight = np.array(data["weight"], dtype=np.float32)
    else:
        src = np.zeros(0, dtype=np.int32)
        dst = np.zeros(0, dtype=np.int32)
        etype = np.zeros(0, dtype=np.uint8)
        weight = np.zeros(0, dtype=np.float32)

    return GraphIndex(
        id_to_row=id_to_row, row_to_id=row_to_id,
        src=src, dst=dst, etype=etype, weight=weight,
    )


__all__ = [
    "EDGES_FILE",
    "ETYPE_IDS",
    "ETYPE_NAMES",
    "GRAPH_DIR",
    "NODES_FILE",
    "GraphIndex",
    "read_graph",
    "write_graph",
]
