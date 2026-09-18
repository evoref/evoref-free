"""ProjectMap — 既存プロジェクトの code グラフ (c_16 §4.4)

corpus の 1 パッケージ = 1 版 = 1 EvidenceStore の形で、file / class /
function / method ノード (``kind=code_node``) と contains / imports /
calls / inherits 辺を持つ。抽出は tree-sitter (無ければ Python だけ
``ast`` に縮退) による決定論処理で、LLM は使わない。書き手は sleep-time
だけ、チャット応答パスは :class:`ProjectMapReader` 経由で読むだけ。

このモジュールの公開 API はここに集約する — 実装は同パッケージ内の
``builder.py`` / ``query.py`` / ``graph.py`` / ``ids.py`` に分かれている。
"""

from __future__ import annotations

from backend.free.rag.projectmap.builder import EXTRACTOR_VERSION, ProjectMapBuilder, UpdateResult
from backend.free.rag.projectmap.extractors.treesitter import is_available
from backend.free.rag.projectmap.graph import Edge, Node, ProjectGraph
from backend.free.rag.projectmap.ids import PROJECT_MAP_KIND, project_map_package_id
from backend.free.rag.projectmap.query import MultiProjectMapReader, ProjectMapReader

__all__ = [
    "EXTRACTOR_VERSION",
    "PROJECT_MAP_KIND",
    "Edge",
    "Node",
    "ProjectGraph",
    "ProjectMapBuilder",
    "MultiProjectMapReader",
    "ProjectMapReader",
    "UpdateResult",
    "is_available",
    "project_map_package_id",
]
