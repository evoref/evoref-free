"""設計↔実装のドリフト検査 (f_10 §8.1、Phase 2.5)。

Software Reflexion Model (Murphy / Notkin / Sullivan) の実装: 意図したグラフ
(planner の ``module_deps``、f_10 §2) と実装のグラフ (生成物を ProjectMap の
抽出器に掛けて ``build_graph`` した ``imports`` 辺) を突き合わせ、
convergence (両方にある) / divergence (実装にだけある = 設計外の結合) /
absence (設計にだけある = 未実装の依存) の 3 分類で返す。決定論・LLM 不使用。

finalize (``_finalize_staged_checks_events``) が全生成物を ``code_map`` として
俯瞰できる唯一の場所で呼ぶ。観測専用 — ここでは何も直さない。
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.free.rag.projectmap.extractors import python_ast, treesitter
from backend.free.rag.projectmap.graph import ExtractedFile, build_graph
from backend.free.rag.projectmap.scanner import LANGUAGE_EXTENSIONS
from backend.log_config import get_logger

logger = get_logger("loop.staged.design_drift")


@dataclass(frozen=True, slots=True)
class DriftReport:
    """設計↔実装ドリフトの分類結果 (決定論・ソート済み ``(src, dst)`` タプル列)。"""

    convergence: list[tuple[str, str]]
    divergence: list[tuple[str, str]]
    absence: list[tuple[str, str]]


def _extract_file(path: str, source: str) -> ExtractedFile | None:
    """1 ファイルを tree-sitter で抽出し、不在 / 非対応言語 (Python のみ) は
    ``python_ast`` へ縮退する。拡張子から言語が分からなければ ``None``。
    """
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    lang = LANGUAGE_EXTENSIONS.get(ext.lower())
    if lang is None:
        return None
    extracted = treesitter.extract_file(path, lang, source.encode("utf-8"))
    if extracted is None and lang == "python":
        extracted = python_ast.extract_file(path, source)
    return extracted


def _implemented_edges(code_map: dict[str, str]) -> set[tuple[str, str]]:
    """``code_map`` (path -> source) から実装の ``imports`` 辺 (path, path) を抽出する。"""
    extracted: list[ExtractedFile] = []
    for path in sorted(code_map):
        file = _extract_file(path, code_map[path])
        if file is not None:
            extracted.append(file)
    graph = build_graph(extracted, fingerprints={})
    path_by_file_id = {
        n.id: n.path for n in graph.nodes if n.node_type == "file"
    }
    edges: set[tuple[str, str]] = set()
    for edge in graph.edges:
        if edge.etype != "imports":
            continue
        src = path_by_file_id.get(edge.src)
        dst = path_by_file_id.get(edge.dst)
        if src and dst:
            edges.add((src, dst))
    return edges


def check_design_drift(
    code_map: dict[str, str], module_deps: dict[str, list[str]],
) -> DriftReport:
    """意図したグラフ (``module_deps``) と実装のグラフ (``code_map`` の import 辺) を突き合わせる。

    観測のみ (LLM 不使用、決定論)。呼出側 (finalize) が結果をどう扱うか
    (validation_errors に畳み込むか / spec 見直しのトリガにするか) は別判断
    (f_10 §8.1 は「観測から始める」)。
    """
    intended = {
        (src, dst) for src, deps in module_deps.items() for dst in deps
    }
    implemented = _implemented_edges(code_map)
    return DriftReport(
        convergence=sorted(intended & implemented),
        divergence=sorted(implemented - intended),
        absence=sorted(intended - implemented),
    )


__all__ = ["DriftReport", "check_design_drift"]
