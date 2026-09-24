"""fingerprint (c_16 §4.4)

``{path: sha256(bytes)}`` を版ディレクトリの ``fingerprints.json`` に持つ。
差分更新の分類 (``classify.py``) が前版とこの版を突き合わせる基準になる。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from backend.free.rag.projectmap.ids import _PACKAGE_ID_PREFIX
from backend.free.rag.projectmap.scanner import ScannedFile
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedJsonFile

FINGERPRINTS_FILE = "fingerprints.json"

#: ProjectMap のパッケージ (``pm-<hash>``) は丸ごと走査したソースから作り直せる。
_PROJECTMAP_PACKAGE_KEY = f"store/corpus/packages/{_PACKAGE_ID_PREFIX}<hash>"

PROJECTMAP_FORMAT = register_format(FormatSpec(
    format_id="projectmap",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key=f"{_PROJECTMAP_PACKAGE_KEY}/<version>/{FINGERPRINTS_FILE}",
    retention="rebuilt from the scanned sources",
))

#: 版ディレクトリの残り (package.json / snapshot / graph / 埋め込み)。
PROJECTMAP_PACKAGE_FORMAT = register_format(FormatSpec(
    format_id="projectmap.package",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key=f"{_PROJECTMAP_PACKAGE_KEY}/**",
    retention="rebuilt from the scanned sources",
    encodings=("dir",),
))


#: ``{path: [(qualname, signature), ...]}``。cosmetic 判定 (前版と定義の形が同じか) の材料。
Shapes = dict[str, list[tuple[str, str]]]


class FingerprintStore(VersionedJsonFile):
    """1 パッケージ版の ``fingerprints.json`` (封筒付き、``AtomicWriter``)。

    fingerprint に加えて、次版の cosmetic 判定に要る **定義の形** (path ごとの
    qualname + signature) とノード / 辺の件数も持つ。これが無いと無変更の
    確認のたびに snapshot の全レコード (2,000 ファイルで 38,000 件・15 秒級) を
    読み直すことになる。
    """

    FORMAT = PROJECTMAP_FORMAT
    RAISE_ON_SAVE_ERROR = True

    def __init__(self, directory: Path | str) -> None:
        super().__init__(Path(directory) / FINGERPRINTS_FILE)
        self.fingerprints: dict[str, str] = {}
        self.shapes: Shapes = {}
        self.node_count: int = 0
        self.edge_count: int = 0

    def _to_payload(self) -> dict[str, Any]:
        return {
            "fingerprints": dict(self.fingerprints),
            "shapes": {p: [list(s) for s in shapes] for p, shapes in self.shapes.items()},
            "counts": {"nodes": int(self.node_count), "edges": int(self.edge_count)},
        }

    def _from_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("fingerprints payload must be an object")
        raw = payload.get("fingerprints")
        self.fingerprints = {
            str(k): str(v) for k, v in raw.items()
        } if isinstance(raw, dict) else {}
        raw_shapes = payload.get("shapes")
        self.shapes = {}
        if isinstance(raw_shapes, dict):
            for path, shapes in raw_shapes.items():
                if isinstance(shapes, list):
                    self.shapes[str(path)] = [
                        (str(s[0]), str(s[1])) for s in shapes
                        if isinstance(s, (list, tuple)) and len(s) == 2
                    ]
        counts = payload.get("counts")
        if isinstance(counts, dict):
            self.node_count = int(counts.get("nodes", 0) or 0)
            self.edge_count = int(counts.get("edges", 0) or 0)


def compute_fingerprint(path: Path) -> str:
    """1 ファイルの sha256 hex。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_fingerprints(
    root: Path | str, files: Sequence[ScannedFile],
) -> dict[str, str]:
    """走査結果の全ファイルから ``{path: sha256}`` を作る。

    読めないファイル (競合で削除された等) はそのファイルだけ飛ばす。
    """
    base = Path(root)
    out: dict[str, str] = {}
    for file in files:
        try:
            out[file.path] = compute_fingerprint(base / file.path)
        except OSError:
            continue
    return out


def load_fingerprints(directory: Path | str) -> dict[str, str]:
    """版ディレクトリの ``fingerprints.json`` を読む (無ければ空 dict)。"""
    return load_fingerprint_store(directory).fingerprints


def load_fingerprint_store(directory: Path | str) -> FingerprintStore:
    """版ディレクトリの ``fingerprints.json`` を丸ごと読む (無ければ空)。"""
    store = FingerprintStore(directory)
    store.load()
    return store


def save_fingerprints(
    directory: Path | str,
    fingerprints: dict[str, str],
    *,
    shapes: Shapes | None = None,
    node_count: int = 0,
    edge_count: int = 0,
) -> None:
    """版ディレクトリへ ``fingerprints.json`` を書く。"""
    store = FingerprintStore(directory)
    store.fingerprints = dict(fingerprints)
    store.shapes = dict(shapes or {})
    store.node_count = node_count
    store.edge_count = edge_count
    store.save()


__all__ = [
    "FINGERPRINTS_FILE",
    "FingerprintStore",
    "Shapes",
    "compute_fingerprint",
    "compute_fingerprints",
    "load_fingerprint_store",
    "load_fingerprints",
    "save_fingerprints",
]
