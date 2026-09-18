"""ProjectMap の構築 (c_16 §4.4)

``ProjectMapBuilder.update()`` が 1 回の走査 → 分類 → (必要なら) 新版の書き
出しを行う。書き手は sleep-time だけ (c_16 §2.1) で、このクラス自身は
呼出元がいつ・どの頻度で呼ぶかを知らない — 与えられた ``is_cancelled`` /
``should_pause`` を尊重するだけの決定論処理。
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.free.rag.corpus.package import PackageMeta, write_package_meta
from backend.free.rag.corpus.store import PACKAGES_DIR, CorpusManifest
from backend.free.rag.evidence.store import EvidenceStore
from backend.free.rag.evidence.types import Evidence
from backend.free.rag.projectmap import graph_io
from backend.free.rag.projectmap.classify import (
    UPDATE_SKIP,
    UpdateThresholds,
    classify_update,
)
from backend.free.rag.projectmap.extractors import python_ast, treesitter
from backend.free.rag.projectmap.fingerprint import (
    compute_fingerprints,
    load_fingerprint_store,
    load_fingerprints,
    save_fingerprints,
)
from backend.free.rag.projectmap.graph import (
    ExtractedFile,
    ProjectGraph,
    build_graph,
    validate_graph,
)
from backend.free.rag.projectmap.ids import PROJECT_MAP_KIND, project_map_package_id
from backend.free.rag.projectmap.scanner import (
    DEFAULT_MAX_FILE_BYTES,
    ScannedFile,
    scan_project,
)
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("rag.projectmap.builder")

#: 抽出規則の版 (§4.4)。分割の結果が変わる変更で上げる。
EXTRACTOR_VERSION = 1

#: 対応する ``update_kind`` (c_16 §4.4)。
UPDATE_INITIAL = "initial"
UPDATE_UNAVAILABLE = "unavailable"
UPDATE_PAUSED = "paused"
UPDATE_INVALID = "invalid"

_VERSION_RE = re.compile(r"^1\.0\.(\d+)$")


def _cfg(section: Any, key: str, default: Any) -> Any:
    if section is None:
        return default
    value = section.get(key) if isinstance(section, dict) else getattr(section, key, None)
    return default if value is None else value


def _next_version(previous: str | None) -> str:
    """``1.0.<n>`` の版番号を 1 つ進める (c_16 §4.4)。"""
    if not previous:
        return "1.0.0"
    match = _VERSION_RE.match(previous)
    if match:
        return f"1.0.{int(match.group(1)) + 1}"
    return "1.0.0"


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """:meth:`ProjectMapBuilder.update` の結果。"""

    update_kind: str
    package_id: str
    version: str | None
    nodes: int
    edges: int
    changed_files: int


class ProjectMapBuilder:
    """1 プロジェクトルート分の ProjectMap を構築する (c_16 §4.4)。"""

    def __init__(
        self,
        corpus_dir: Path,
        root: Path,
        *,
        rag_config: dict,
        now_provider: Callable[[], str] | None = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.root = Path(root)
        self.rag_config = rag_config or {}
        self._now = now_provider or utc_now
        self.package_id = project_map_package_id(self.root)
        self._packages_dir = self.corpus_dir / PACKAGES_DIR

    # ── 設定 ──

    def _pm_config(self) -> Any:
        return _cfg(self.rag_config, "project_map", None)

    def _exclude_globs(self) -> list[str]:
        pm = self._pm_config()
        globs = _cfg(pm, "exclude_globs", [])
        return [str(g) for g in globs] if globs else []

    def _max_file_bytes(self) -> int:
        pm = self._pm_config()
        return int(_cfg(pm, "max_file_bytes", DEFAULT_MAX_FILE_BYTES))

    def _thresholds(self) -> UpdateThresholds:
        pm = self._pm_config()
        update_cfg = _cfg(pm, "update", None)
        defaults = UpdateThresholds()
        return UpdateThresholds(
            architecture_threshold=int(
                _cfg(update_cfg, "architecture_threshold", defaults.architecture_threshold),
            ),
            full_threshold=int(_cfg(update_cfg, "full_threshold", defaults.full_threshold)),
            full_ratio=float(_cfg(update_cfg, "full_ratio", defaults.full_ratio)),
        )

    # ── 走査 ──

    def _scan(self) -> list[ScannedFile]:
        """``self.root`` を走査する。

        ``rag.project_map.roots`` (複数のプロジェクトルート) は
        ``ProjectMapBuilder`` 自身は読まない — 1 インスタンス = 1 root
        (``package_id`` も root から導出、c_16 §4.4) で、複数 root を回すのは
        呼出元 (sleep-time Step 5.87) の責務。ここで ``roots`` も読むと、
        同じ設定を渡された全 root のビルダーが互いの root まで多重に走査する。
        """
        return scan_project(
            self.root,
            exclude_globs=self._exclude_globs(),
            max_file_bytes=self._max_file_bytes(),
        )

    def _extract_one(self, scanned: ScannedFile) -> ExtractedFile | None:
        try:
            source_bytes = (self.root / scanned.path).read_bytes()
        except OSError as e:
            logger.warning("projectmap: failed to read %s: %s", scanned.path, e)
            return None
        result = treesitter.extract_file(scanned.path, scanned.lang, source_bytes)
        if result is not None:
            return result
        if scanned.lang == "python":
            try:
                source_text = source_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                logger.warning("projectmap: failed to decode %s: %s", scanned.path, e)
                return None
            return python_ast.extract_file(scanned.path, source_text)
        return None

    # ── 旧版の読み出し ──

    def _package_dir(self, version: str) -> Path:
        return self._packages_dir / self.package_id / version

    def _load_old_state(
        self, version: str,
    ) -> tuple[dict[str, list[tuple[str, str]]], int, int]:
        """旧版の ``{path: [(qualname, signature), ...]}`` とノード/辺件数。"""
        # snapshot の全レコードは読まない — 形と件数は書込時に fingerprints.json へ
        # 一緒に残してある (FingerprintStore)。無変更の確認を秒単位に保つため。
        state = load_fingerprint_store(self._package_dir(version))
        return dict(state.shapes), state.node_count, state.edge_count

    # ── 更新 ──

    async def update(
        self,
        *,
        is_cancelled: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> UpdateResult:
        """走査 → 分類 → (必要なら) 新版の書き出し。"""
        if self._should_stop(is_cancelled, should_pause):
            return self._paused()

        scanned = self._scan()
        new_fingerprints = compute_fingerprints(self.root, scanned)

        manifest = CorpusManifest(self.corpus_dir)
        manifest.load()
        old_version = manifest.active.get(self.package_id)

        has_python = any(f.lang == "python" for f in scanned)
        if not treesitter.is_available() and not has_python:
            return UpdateResult(
                update_kind=UPDATE_UNAVAILABLE, package_id=self.package_id,
                version=old_version, nodes=0, edges=0, changed_files=0,
            )

        old_fingerprints: dict[str, str] = {}
        old_nodes_by_path: dict[str, list[tuple[str, str]]] = {}
        old_node_count = 0
        old_edge_count = 0
        if old_version:
            old_fingerprints = load_fingerprints(self._package_dir(old_version))
            old_nodes_by_path, old_node_count, old_edge_count = self._load_old_state(old_version)

        if self._should_stop(is_cancelled, should_pause):
            return self._paused()

        # fingerprint が変わったファイルだけを先に抽出して cosmetic 判定 → skip なら
        # ここで抜ける。全ファイルの再抽出 (2,000 ファイルで 30 秒級) は新版を書くと
        # 決まってからにする — Full サイクルごとの無変更チェックを秒単位に保つため。
        touched = {
            p for p, fp in new_fingerprints.items()
            if p not in old_fingerprints or old_fingerprints[p] != fp
        }
        extracted_by_path: dict[str, ExtractedFile] = {}

        def _extract_into(files: list[ScannedFile]) -> UpdateResult | None:
            for scanned_file in files:
                if self._should_stop(is_cancelled, should_pause):
                    return self._paused()
                result = self._extract_one(scanned_file)
                if result is not None:
                    extracted_by_path[scanned_file.path] = result
                elif not treesitter.is_available() or scanned_file.lang != "python":
                    logger.warning(
                        "projectmap: skipped %s (%s): no usable extractor",
                        scanned_file.path, scanned_file.lang,
                    )
            return None

        paused = _extract_into([f for f in scanned if f.path in touched])
        if paused is not None:
            return paused

        cosmetic_paths: set[str] = set()
        for path in touched & set(old_fingerprints):
            new_file = extracted_by_path.get(path)
            if new_file is None:
                continue
            new_shape = {(n.qualname, n.signature) for n in new_file.nodes}
            old_shape = set(old_nodes_by_path.get(path, ()))
            if new_shape == old_shape:
                cosmetic_paths.add(path)

        if old_version:
            kind = classify_update(
                old_fingerprints=old_fingerprints,
                new_fingerprints=new_fingerprints,
                cosmetic_paths=frozenset(cosmetic_paths),
                thresholds=self._thresholds(),
            )
        else:
            kind = UPDATE_INITIAL

        structural_changed = (
            (set(old_fingerprints) - set(new_fingerprints))
            | {p for p in touched if p not in cosmetic_paths}
        )

        if kind == UPDATE_SKIP:
            return UpdateResult(
                update_kind=UPDATE_SKIP, package_id=self.package_id, version=old_version,
                nodes=old_node_count, edges=old_edge_count, changed_files=0,
            )

        paused = _extract_into([f for f in scanned if f.path not in touched])
        if paused is not None:
            return paused
        extracted = [extracted_by_path[f.path] for f in scanned if f.path in extracted_by_path]

        graph = build_graph(extracted, fingerprints=new_fingerprints)
        violations = validate_graph(graph)
        if violations:
            for line in violations[:20]:
                logger.warning("projectmap: graph validation failed: %s", line)
            return UpdateResult(
                update_kind=UPDATE_INVALID, package_id=self.package_id, version=old_version,
                nodes=0, edges=0, changed_files=len(structural_changed),
            )

        if self._should_stop(is_cancelled, should_pause):
            return self._paused()

        new_version = _next_version(old_version)
        await self._write_version(new_version, graph, kind)

        manifest.active[self.package_id] = new_version
        manifest.save()

        return UpdateResult(
            update_kind=kind, package_id=self.package_id, version=new_version,
            nodes=len(graph.nodes), edges=len(graph.edges),
            changed_files=len(structural_changed),
        )

    def _should_stop(
        self,
        is_cancelled: Callable[[], bool] | None,
        should_pause: Callable[[], bool] | None,
    ) -> bool:
        if is_cancelled is not None and is_cancelled():
            return True
        return should_pause is not None and should_pause()

    def _paused(self) -> UpdateResult:
        manifest = CorpusManifest(self.corpus_dir)
        manifest.load()
        return UpdateResult(
            update_kind=UPDATE_PAUSED, package_id=self.package_id,
            version=manifest.active.get(self.package_id), nodes=0, edges=0, changed_files=0,
        )

    # ── 書き出し ──

    def _fan_counts(self, graph: ProjectGraph) -> tuple[dict[str, int], dict[str, int]]:
        fan_in: dict[str, int] = {}
        fan_out: dict[str, int] = {}
        for edge in graph.edges:
            fan_out[edge.src] = fan_out.get(edge.src, 0) + 1
            fan_in[edge.dst] = fan_in.get(edge.dst, 0) + 1
        return fan_in, fan_out

    def _node_to_evidence(
        self, node, graph: ProjectGraph, fan_in: dict[str, int], fan_out: dict[str, int],
        version: str, now: str,
    ) -> Evidence:
        attrs: dict[str, Any] = {
            "package_id": self.package_id,
            "package_version": version,
            "node_type": node.node_type,
            "path": node.path,
            "name": node.name,
            "qualname": node.qualname,
            "lang": node.lang,
            "line_start": node.line_start,
            "line_end": node.line_end,
            "signature": node.signature,
            "fan_in": fan_in.get(node.id, 0),
            "fan_out": fan_out.get(node.id, 0),
        }
        if node.parent_id:
            attrs["parent_id"] = node.parent_id
        if node.node_type == "file":
            externals = graph.external_imports.get(node.path)
            if externals:
                attrs["external_imports"] = list(externals)

        return Evidence(
            id=node.id,
            kind="code_node",
            store="corpus",
            text=node.text,
            origin="tool",
            provenance=[{
                "extractor": "projectmap",
                "extractor_version": EXTRACTOR_VERSION,
                "captured_at": now,
            }],
            observed_at=now,
            created_at=now,
            attrs=attrs,
        )

    async def _write_version(
        self, version: str, graph: ProjectGraph, update_kind: str,
    ) -> None:
        """新しい版ディレクトリへ snapshot / graph / fingerprints / meta を書く。

        corpus パッケージ (c_16 §4.3) と同じ規約 — 版は常に完全な新版、
        最後に呼出元が ``manifest.active`` を指し替える。
        """
        directory = self._package_dir(version)
        if directory.exists():
            # 前回の書きかけ (manifest.active を指し替える前に落ちた版)。番号は
            # active + 1 なので生きた版ではない。c_16 §5.4 と同じく改名してから消す
            # (途中で落ちても「生きた版」か「trash」のどちらかに落ち着く)。
            trash = directory.with_name(f".trash-{directory.name}")
            shutil.rmtree(trash, ignore_errors=True)
            directory.rename(trash)
            shutil.rmtree(trash, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        now = self._now()

        store = EvidenceStore(
            directory, store_name="corpus", embedding_backend=None,
            rag_config=self.rag_config, by="projectmap_builder",
            shard_key_for=lambda _record, key=self.package_id: key,
        )
        store.load()
        fan_in, fan_out = self._fan_counts(graph)
        try:
            for node in graph.nodes:
                store.put(
                    self._node_to_evidence(node, graph, fan_in, fan_out, version, now),
                    by="projectmap_builder",
                )
            await store.create_snapshot()
            graph_io.write_graph(directory, store, graph.nodes, graph.edges)
        finally:
            store.close()
        # corpus は events を持たない — 版そのものが履歴で内容は不変 (c_16 §4.3)。
        # 残すと 1 版あたり records と同量 (2,000 ノードで 2.6 MB) が二重になる。
        shutil.rmtree(directory / "events", ignore_errors=True)

        # file ノードは extractors の ExtractedFile.nodes に含まれない (build_graph が
        # 合成する) ので、cosmetic 判定の形からは外す — 混ぜると必ず「file 分だけ
        # 違う」になり、cosmetic な変更まで partial 扱いになる。
        shapes: dict[str, list[tuple[str, str]]] = {}
        for node in graph.nodes:
            if node.node_type != "file":
                shapes.setdefault(node.path, []).append((node.qualname, node.signature))
        save_fingerprints(
            directory, graph.fingerprints, shapes=shapes,
            node_count=len(graph.nodes), edge_count=len(graph.edges),
        )

        meta = PackageMeta(
            id=self.package_id,
            name=self.root.name or self.package_id,
            version=version,
            language="",
            _extra={
                "kind": PROJECT_MAP_KIND,
                "root": self.root.as_posix(),
                "languages": dict(graph.languages),
                "extractor_version": EXTRACTOR_VERSION,
                "update_kind": update_kind,
            },
        )
        write_package_meta(directory, meta)

    def close(self) -> None:
        """このビルダー自身は永続ハンドルを持たない (no-op、対称性のため)。"""


__all__ = [
    "EXTRACTOR_VERSION",
    "UPDATE_INITIAL",
    "UPDATE_INVALID",
    "UPDATE_PAUSED",
    "UPDATE_UNAVAILABLE",
    "ProjectMapBuilder",
    "UpdateResult",
]
