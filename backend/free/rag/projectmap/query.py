"""``ProjectMapReader`` — ProjectMap の読み出し専用 API (c_16 §4.4)

統合検索 (§7.4) には載せない (``_extra.kind == "project_map"`` の除外は
``CorpusStore`` 側、c_16 §4.4)。読み手はここの 3 操作だけを使う:

- :meth:`lookup` — 名前 / qualname で code_node を引く
- :meth:`neighborhood` — 対象を起点に fan-in 順で近傍を辿り、予算内で描画
- :meth:`overview` — ディレクトリ / 言語別件数とエントリ候補
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from pathlib import Path

from backend.free.rag.corpus.package import PACKAGE_FILE
from backend.free.rag.corpus.store import PACKAGES_DIR, CorpusManifest
from backend.free.rag.evidence.store import EvidenceStore
from backend.free.rag.projectmap import graph_io
from backend.free.rag.projectmap.graph import Node
from backend.free.rag.projectmap.ids import code_node_id, project_map_package_id
from backend.utils import estimate_tokens

#: エントリ候補と見なすファイル stem (c_16 §4.4)。
_ENTRY_STEMS: frozenset[str] = frozenset({"main", "app", "cli", "index", "__main__"})
#: overview のエントリ候補の上限件数。
_ENTRY_LIMIT = 5


def _fit_budget(lines: list[str], budget_tokens: int) -> str:
    """行を先頭から足し、予算 (推定トークン数) を超える手前で打ち切る。

    1 行目は予算を超えていても必ず入れる (空の描画より情報がある方がよい)。
    """
    out: list[str] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line)
        if out and used + cost > budget_tokens:
            break
        out.append(line)
        used += cost
    return "\n".join(out)


class ProjectMapReader:
    """ProjectMap の active 版を開いて読む (c_16 §4.4)。"""

    def __init__(
        self, root: Path, package_id: str, version: str, directory: Path,
    ) -> None:
        self.root = root
        self.package_id = package_id
        self.version = version
        self.directory = directory
        self._store = EvidenceStore(directory, store_name="corpus", embedding_backend=None)
        self._store.load()
        self._graph = graph_io.read_graph(directory)
        # ノード表は open 時に records.jsonl を 1 回だけ順読みして常駐させる。
        # 呼出しごとに ``iter_records`` (行単位の seek + read、0.3 ms/件) で全件を
        # なめると 38k ノードの map で lookup 1 回が 12 秒になる (チャット応答パス)。
        self._nodes: dict[str, Node] = {}
        self._fan_in: dict[str, int] = {}
        self._by_name: dict[str, list[str]] = {}
        self._by_qualname: dict[str, list[str]] = {}
        self._dir_counts: dict[str, int] = {}
        self._lang_counts: dict[str, int] = {}
        self._entry_candidates: list[tuple[int, str]] = []
        self._load_nodes()
        self._sorted_names = sorted(self._by_name)
        self._sorted_qualnames = sorted(self._by_qualname)

    def _load_nodes(self) -> None:
        snapshot = self._store.snapshot
        if snapshot is None or not snapshot.records_path.exists():
            return
        with snapshot.records_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except ValueError:
                    continue
                if raw.get("kind") != "code_node":
                    continue
                attrs = raw.get("attrs") or {}
                node = Node(
                    id=str(raw.get("id") or ""),
                    node_type=str(attrs.get("node_type") or ""),
                    path=str(attrs.get("path") or ""),
                    name=str(attrs.get("name") or ""),
                    qualname=str(attrs.get("qualname") or ""),
                    lang=str(attrs.get("lang") or ""),
                    line_start=int(attrs.get("line_start") or 0),
                    line_end=int(attrs.get("line_end") or 0),
                    parent_id=attrs.get("parent_id"),
                    signature=str(attrs.get("signature") or ""),
                    text=str(raw.get("text") or ""),
                )
                if not node.id:
                    continue
                self._nodes[node.id] = node
                self._fan_in[node.id] = int(attrs.get("fan_in") or 0)
                if node.name:
                    self._by_name.setdefault(node.name, []).append(node.id)
                if node.qualname:
                    self._by_qualname.setdefault(node.qualname, []).append(node.id)
                if node.node_type == "file":
                    top = node.path.split("/", 1)[0] if "/" in node.path else "(root)"
                    self._dir_counts[top] = self._dir_counts.get(top, 0) + 1
                    if node.lang:
                        self._lang_counts[node.lang] = self._lang_counts.get(node.lang, 0) + 1
                    if Path(node.path).stem in _ENTRY_STEMS:
                        self._entry_candidates.append((self._fan_in[node.id], node.path))
        self._entry_candidates.sort(key=lambda item: (-item[0], item[1]))

    def _prefix_ids(self, keys: list[str], index: dict[str, list[str]], prefix: str) -> list[str]:
        start = bisect.bisect_left(keys, prefix)
        out: list[str] = []
        for key in keys[start:]:
            if not key.startswith(prefix):
                break
            out.extend(index[key])
        return out

    @classmethod
    def open(cls, corpus_dir: Path, root: Path) -> "ProjectMapReader | None":
        """active 版を開く。無ければ ``None``。"""
        package_id = project_map_package_id(root)
        manifest = CorpusManifest(corpus_dir)
        manifest.load()
        version = manifest.active.get(package_id)
        if not version:
            return None
        directory = Path(corpus_dir) / PACKAGES_DIR / package_id / version
        if not (directory / PACKAGE_FILE).exists():
            return None
        return cls(root, package_id, version, directory)

    def close(self) -> None:
        """memmap を握った snapshot / 索引を手放す (Windows で版を GC できるように)。"""
        self._store.close()

    # ── lookup ──

    def lookup(self, name: str, *, limit: int = 10) -> list[Node]:
        """名前 / qualname の完全一致 → 前方一致 → 転置索引の順で引く。"""
        if limit <= 0 or not name:
            return []
        exact_ids = set(self._by_name.get(name, ())) | set(self._by_qualname.get(name, ()))
        prefix_ids = (
            set(self._prefix_ids(self._sorted_names, self._by_name, name))
            | set(self._prefix_ids(self._sorted_qualnames, self._by_qualname, name))
        ) - exact_ids
        exact = sorted((self._nodes[i] for i in exact_ids), key=lambda n: (n.path, n.line_start))
        prefix = sorted((self._nodes[i] for i in prefix_ids), key=lambda n: (n.path, n.line_start))
        combined = exact + prefix
        if len(combined) >= limit:
            return combined[:limit]

        seen_ids = {n.id for n in combined}
        if self._store.snapshot is not None:
            # 転置索引は bi-gram の部分一致でも候補を返す (存在しない名前でも
            # 数件ヒットする) ので、名前 / qualname / signature に問い合わせ文字列
            # が実際に含まれるものだけを採る。
            needle = name.lower()
            rows, _scores = self._store.lexical_candidates(name, limit * 4)
            for row in rows:
                r = int(row)
                if not (0 <= r < len(self._graph.row_to_id)):
                    continue
                node = self._nodes.get(self._graph.row_to_id[r])
                if node is None or node.id in seen_ids:
                    continue
                haystack = f"{node.name}\n{node.qualname}\n{node.signature}".lower()
                if needle not in haystack:
                    continue
                combined.append(node)
                seen_ids.add(node.id)
                if len(combined) >= limit:
                    break
        return combined[:limit]

    # ── neighborhood ──

    def _resolve_target(self, target: str) -> str | None:
        file_id = code_node_id(target, "file", "")
        if file_id in self._nodes:
            return file_id
        hits = self.lookup(target, limit=1)
        return hits[0].id if hits else None

    def _fan_in_of(self, node_id: str) -> int:
        return self._fan_in.get(node_id, 0)

    def _neighbors(self, node_id: str) -> list[tuple[str, str]]:
        """``(neighbor_id, direction)``。direction は ``->`` (出辺) / ``<-`` (入辺)。"""
        row = self._graph.id_to_row.get(node_id)
        if row is None:
            return []
        out_rows = self._graph.dst[self._graph.src == row]
        in_rows = self._graph.src[self._graph.dst == row]
        seen: set[int] = set()
        out: list[tuple[str, str]] = []
        for direction, rows in (("->", out_rows.tolist()), ("<-", in_rows.tolist())):
            for raw_row in rows:
                r = int(raw_row)
                if r == row or r in seen:
                    continue
                seen.add(r)
                if 0 <= r < len(self._graph.row_to_id):
                    node_id_at_row = self._graph.row_to_id[r]
                    if node_id_at_row:
                        out.append((node_id_at_row, direction))
        return out

    def _render_line(self, node_id: str) -> str:
        node = self._nodes.get(node_id)
        if node is None:
            return node_id
        return f"{node.path}:{node.line_start}  {node.signature or node.name}".rstrip()

    def neighborhood(
        self, target: str, *, depth: int = 2, budget_tokens: int = 800,
    ) -> str:
        """``target`` (path または symbol) を起点に fan-in 順で近傍を辿る。

        BFS で ``depth`` 段まで広げ、各段は fan-in の高い順 (同点は id 昇順)
        に並べる — 予算で切れても「より参照されているもの」を優先して残す。
        """
        start_id = self._resolve_target(target)
        if start_id is None:
            return ""
        visited = {start_id}
        frontier = [start_id]
        # (node_id, 起点から見た向き)。起点は無印、1 段目は ``->`` (呼ぶ / import する)
        # と ``<-`` (呼ばれる / import される) を付け、2 段目以降は最初に到達した
        # 向きを引き継ぐ — 「誰が呼ぶか」と「何を呼ぶか」を読み分けられるように。
        order: list[tuple[str, str]] = [(start_id, "")]
        direction_of: dict[str, str] = {}
        for _ in range(max(depth, 0)):
            candidates: dict[str, int] = {}
            for node_id in frontier:
                for neighbor_id, direction in self._neighbors(node_id):
                    if neighbor_id in visited:
                        continue
                    fan_in = self._fan_in_of(neighbor_id)
                    if neighbor_id not in candidates or fan_in > candidates[neighbor_id]:
                        candidates[neighbor_id] = fan_in
                    direction_of.setdefault(
                        neighbor_id, direction_of.get(node_id) or direction,
                    )
            if not candidates:
                break
            ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
            frontier = [node_id for node_id, _ in ranked]
            visited.update(frontier)
            order.extend((node_id, direction_of.get(node_id, "")) for node_id in frontier)

        lines = [
            f"{direction} {self._render_line(node_id)}" if direction else self._render_line(node_id)
            for node_id, direction in order
        ]
        return _fit_budget(lines, budget_tokens)

    # ── overview ──

    def overview(self, *, budget_tokens: int = 800) -> str:
        """トップレベルディレクトリ / 言語別件数とエントリ候補 (open 時の集計を描画)。"""
        dir_counts = self._dir_counts
        lang_counts = self._lang_counts
        entry_candidates = self._entry_candidates

        lines: list[str] = [f"project: {self.root.name or self.package_id}"]
        lines.append("directories:")
        lines.extend(
            f"  {name}: {dir_counts[name]} file(s)" for name in sorted(dir_counts)
        )
        lines.append("languages:")
        lines.extend(
            f"  {name}: {lang_counts[name]} file(s)" for name in sorted(lang_counts)
        )
        if entry_candidates:
            lines.append("entry points:")
            lines.extend(
                f"  {path} (fan_in={fan_in})"
                for fan_in, path in entry_candidates[:_ENTRY_LIMIT]
            )

        return _fit_budget(lines, budget_tokens)


class MultiProjectMapReader:
    """複数 root の :class:`ProjectMapReader` を 1 つの読み手として束ねる。

    root が 1 つなら出力は単独 reader と byte 一致 (見出しを付けない)。2 つ以上
    のときは root 名の見出しで区切る。ツール (`search_code` / `project_map`) は
    どちらでも同じ 3 操作だけを呼ぶ。
    """

    def __init__(self, readers: Sequence[ProjectMapReader]) -> None:
        self.readers: list[ProjectMapReader] = list(readers)

    @classmethod
    def open(
        cls, corpus_dir: Path, roots: Sequence[Path],
    ) -> "MultiProjectMapReader | None":
        """各 root の active 版を開く。1 つも開けなければ ``None``。"""
        readers: list[ProjectMapReader] = []
        for root in roots:
            reader = ProjectMapReader.open(corpus_dir, root)
            if reader is not None:
                readers.append(reader)
        return cls(readers) if readers else None

    def close(self) -> None:
        for reader in self.readers:
            reader.close()

    def _label(self, reader: ProjectMapReader) -> str:
        return reader.root.name or reader.package_id

    def lookup(self, name: str, *, limit: int = 10) -> list[Node]:
        """root 順に集め、上限は全体に掛ける。"""
        out: list[Node] = []
        for reader in self.readers:
            out.extend(reader.lookup(name, limit=limit))
            if len(out) >= limit:
                break
        return out[:limit]

    def neighborhood(
        self, target: str, *, depth: int = 2, budget_tokens: int = 800,
    ) -> str:
        parts: list[str] = []
        for reader in self.readers:
            text = reader.neighborhood(target, depth=depth, budget_tokens=budget_tokens)
            if not text:
                continue
            if len(self.readers) > 1:
                text = f"[{self._label(reader)}]\n{text}"
            parts.append(text)
        return "\n".join(parts)

    def overview(self, *, budget_tokens: int = 800) -> str:
        parts: list[str] = []
        for reader in self.readers:
            text = reader.overview(budget_tokens=budget_tokens)
            if len(self.readers) > 1:
                text = f"[{self._label(reader)}]\n{text}"
            parts.append(text)
        return "\n".join(parts)


__all__ = ["MultiProjectMapReader", "ProjectMapReader"]
