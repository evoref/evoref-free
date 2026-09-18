"""前版との差分分類 (c_16 §4.4)

fingerprint の突き合わせで new / deleted / changed に分け、``changed`` の
うち **ノード集合 (qualname + signature) が前版と同じ** (cosmetic、空白や
コメントだけの変更) を除いた「構造変更ファイル数」から
``skip`` / ``partial`` / ``architecture`` / ``full`` を決める。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: ``update_kind`` の値 (c_16 §4.4)。
UPDATE_SKIP = "skip"
UPDATE_PARTIAL = "partial"
UPDATE_ARCHITECTURE = "architecture"
UPDATE_FULL = "full"


@dataclass(frozen=True, slots=True)
class UpdateThresholds:
    """``rag.project_map.update`` の閾値 (c_16 §9)。"""

    architecture_threshold: int = 10
    full_threshold: int = 30
    full_ratio: float = 0.5


def top_level_dirs(paths: Iterable[str]) -> frozenset[str]:
    """相対パス集合からトップレベルディレクトリ名の集合を作る。"""
    return frozenset(p.split("/", 1)[0] for p in paths if "/" in p)


def classify_update(
    *,
    old_fingerprints: dict[str, str],
    new_fingerprints: dict[str, str],
    cosmetic_paths: frozenset[str] = frozenset(),
    thresholds: UpdateThresholds = UpdateThresholds(),
) -> str:
    """前版と今回の走査を突き合わせて ``update_kind`` を決める。

    Args:
        old_fingerprints: 前版の ``{path: sha256}`` (初回構築なら空 dict)。
        new_fingerprints: 今回走査した ``{path: sha256}``。
        cosmetic_paths: ``changed`` のうち再抽出したノード集合
            (qualname + signature) が前版と同じだったファイル。
        thresholds: 分類の閾値。
    """
    old_paths = set(old_fingerprints)
    new_paths = set(new_fingerprints)
    added = new_paths - old_paths
    deleted = old_paths - new_paths
    changed = {
        path for path in (old_paths & new_paths)
        if old_fingerprints[path] != new_fingerprints[path]
    }
    structural_changed = changed - cosmetic_paths
    structural = added | deleted | structural_changed
    if not structural:
        return UPDATE_SKIP

    total = len(new_paths) or 1
    count = len(structural)
    topdirs_changed = top_level_dirs(old_paths) != top_level_dirs(new_paths)

    if count > thresholds.full_threshold or count > thresholds.full_ratio * total:
        return UPDATE_FULL
    if count > thresholds.architecture_threshold or topdirs_changed:
        return UPDATE_ARCHITECTURE
    return UPDATE_PARTIAL


__all__ = [
    "UPDATE_ARCHITECTURE",
    "UPDATE_FULL",
    "UPDATE_PARTIAL",
    "UPDATE_SKIP",
    "UpdateThresholds",
    "classify_update",
    "top_level_dirs",
]
