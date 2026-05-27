"""``evorefmem_cli inspect`` 実装

副作用なしで EvorefMem 全 scope の統計情報を集計する.

- fact 数 / 型別分布 / pillar 別分布
- subject 上位 N 件
- ``.idx`` ファイルサイズ
- embedding 行数 / dim
- manifest.json サマリ
- orphan 検出 (詳細は :mod:`.verify_cmd` が担当、本コマンドは件数のみ)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.free.memory.semantic.cli._paths import (
    ScopeInfo,
    enumerate_scopes,
)
from backend.free.memory.semantic.embedding_store import (
    list_stored_models,
    row_to_id_path,
    vectors_path,
)
from backend.free.memory.semantic.manifest import (
    Manifest,
    load_manifest,
    manifest_path,
)
from backend.free.memory.semantic.store import (
    FACTS_FILENAME,
    INDEX_JSONL_FILENAME,
)
from backend.free.memory.semantic.subject_key import SubjectKey


@dataclass
class ScopeInspectReport:
    """1 scope 分の inspect 結果."""

    scope: str
    """scope 名 (``global`` / ``project:<id>``)。"""

    facts_total: int = 0
    """``facts.jsonl`` 内のユニーク id 数。"""

    facts_jsonl_lines: int = 0
    """``facts.jsonl`` の行数 (重複行を含む生の行数)。"""

    facts_jsonl_bytes: int = 0
    """``facts.jsonl`` のバイトサイズ。"""

    duplicate_lines: int = 0
    """``facts_jsonl_lines - facts_total``。compact で削減可能な行数。"""

    by_type: dict[str, int] = field(default_factory=dict)
    """FactType ごとのユニーク fact 数。"""

    by_pillar: dict[str, int] = field(default_factory=dict)
    """pillar (``loop`` / ``learn`` / ``mem``) ごとの fact 数 (subject から判定)。"""

    pillar_unparsed: int = 0
    """pillar prefix を持たない subject の fact 数 (legacy / 自然文 subject)。"""

    pinned: int = 0
    """``pinned == True`` な fact 数。"""

    superseded: int = 0
    """``superseded_by`` がセットされている fact 数。"""

    top_subjects: list[tuple[str, int]] = field(default_factory=list)
    """subject 上位 N 件 (デフォルト 10)。"""

    idx_sizes: dict[str, int] = field(default_factory=dict)
    """``.idx`` ファイル名 -> バイトサイズ。"""

    embedding_models: list[str] = field(default_factory=list)
    """``embeddings/`` 配下に存在する model_id 一覧。"""

    embedding_active_model: str | None = None
    """manifest.embedding.model_id が指す現行 active model。"""

    embedding_count_by_model: dict[str, int] = field(default_factory=dict)
    """各 model_id の vectors.npy 行数。"""

    embedding_dim_by_model: dict[str, int] = field(default_factory=dict)
    """各 model_id の vectors.npy 次元 (空なら 0)。"""

    orphan_idx_ids: int = 0
    """``.idx`` 内に列挙されているが ``facts.jsonl`` に存在しない fact_id 数。"""

    orphan_embedding_ids: int = 0
    """``embeddings/<active>/row_to_id.json`` に居るが facts.jsonl に居ない id 数。"""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InspectReport:
    """全 scope を集約した inspect 結果."""

    memory_dir: str
    schema_version_marker: int | None
    manifest: dict[str, Any] | None
    scopes: list[ScopeInspectReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_dir": self.memory_dir,
            "schema_version_marker": self.schema_version_marker,
            "manifest": self.manifest,
            "scopes": [s.to_dict() for s in self.scopes],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ──────────────────────────────────────────────────────────────────────────
# 内部ヘルパ
# ──────────────────────────────────────────────────────────────────────────


def _read_facts_jsonl(
    facts_path: Path,
) -> tuple[dict[str, Any], int, int, int]:
    """``facts.jsonl`` から (id -> 最新 fact dict) と統計を返す.

    Returns:
        (latest_by_id, lines, bytes, malformed)

        - latest_by_id: id -> (fact dict) (last-write-wins; 不正な行はスキップ)
        - lines: 行数 (空行 / 不正行を含む合計行数)
        - bytes: ファイルバイトサイズ
        - malformed: 不正な行の数
    """
    if not facts_path.exists():
        return {}, 0, 0, 0
    latest: dict[str, dict[str, Any]] = {}
    lines = 0
    malformed = 0
    with facts_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                lines += 1
                continue
            lines += 1
            try:
                data = json.loads(line)
                fid = data.get("id")
                if not isinstance(fid, str) or not fid:
                    raise ValueError("missing id field")
                latest[fid] = data
            except (json.JSONDecodeError, ValueError):
                malformed += 1
    size = facts_path.stat().st_size
    return latest, lines, size, malformed


def _read_index_jsonl_ids(index_path: Path) -> set[str]:
    """新形式 ``index.jsonl`` (M5-d) から fact_id 集合を抽出する。

    tombstone 行は除外し、生存 fact のみを返す。
    """
    if not index_path.exists():
        return set()
    out: set[str] = set()
    try:
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("_tombstone"):
                    key = obj.get("_key")
                    if isinstance(key, str):
                        out.discard(key)
                    continue
                fact_id = obj.get("fact_id")
                if isinstance(fact_id, str):
                    out.add(fact_id)
    except OSError:
        return set()
    return out


def _read_row_to_id(row_to_id_p: Path) -> list[str]:
    if not row_to_id_p.exists():
        return []
    try:
        data = json.loads(row_to_id_p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    ids = data.get("row_to_id", []) if isinstance(data, dict) else []
    return [str(x) for x in ids] if isinstance(ids, list) else []


def _vectors_dim(vectors_p: Path) -> tuple[int, int]:
    """``vectors.npy`` の (rows, dim) を返す。読み込み失敗時は (0, 0)。"""
    if not vectors_p.exists():
        return 0, 0
    try:
        import numpy as np

        arr = np.load(vectors_p, mmap_mode="r")
    except (OSError, ValueError):
        return 0, 0
    if arr.ndim != 2:
        return int(arr.shape[0]) if arr.ndim >= 1 else 0, 0
    return int(arr.shape[0]), int(arr.shape[1])


# ──────────────────────────────────────────────────────────────────────────
# 公開 API
# ──────────────────────────────────────────────────────────────────────────


def inspect_scope(
    scope: ScopeInfo,
    *,
    top_subjects: int = 10,
    active_model_id: str | None = None,
) -> ScopeInspectReport:
    """1 scope 分の inspect を実行する."""
    rep = ScopeInspectReport(scope=scope.name)

    facts_path = scope.root_dir / FACTS_FILENAME
    latest, lines, size, malformed = _read_facts_jsonl(facts_path)
    rep.facts_total = len(latest)
    rep.facts_jsonl_lines = lines
    rep.facts_jsonl_bytes = size
    # last-write-wins 圧縮で削減できる行数 = (有効行 - ユニーク id 数)
    valid_lines = lines - malformed
    rep.duplicate_lines = max(0, valid_lines - rep.facts_total)

    # 集計
    type_ctr: Counter[str] = Counter()
    pillar_ctr: Counter[str] = Counter()
    subject_ctr: Counter[str] = Counter()
    for fact_dict in latest.values():
        ftype = fact_dict.get("type")
        if isinstance(ftype, str):
            type_ctr[ftype] += 1
        subj = fact_dict.get("subject")
        if isinstance(subj, str):
            subject_ctr[subj] += 1
            key = SubjectKey.try_parse(subj)
            if key is not None:
                pillar_ctr[key.pillar] += 1
            else:
                rep.pillar_unparsed += 1
        if bool(fact_dict.get("pinned", False)):
            rep.pinned += 1
        if fact_dict.get("superseded_by"):
            rep.superseded += 1
    rep.by_type = dict(sorted(type_ctr.items()))
    rep.by_pillar = dict(sorted(pillar_ctr.items()))
    rep.top_subjects = subject_ctr.most_common(top_subjects)

    # idx サイズ (M5-d 以降は統合 index.jsonl 1 本のみ)
    idx_path = scope.root_dir / INDEX_JSONL_FILENAME
    if idx_path.exists():
        rep.idx_sizes[INDEX_JSONL_FILENAME] = idx_path.stat().st_size

    # embedding model 一覧
    rep.embedding_models = list_stored_models(scope.root_dir)
    rep.embedding_active_model = active_model_id
    for model_id in rep.embedding_models:
        rows, dim = _vectors_dim(vectors_path(scope.root_dir, model_id))
        rep.embedding_count_by_model[model_id] = rows
        rep.embedding_dim_by_model[model_id] = dim

    # orphan: idx ID で latest (facts.jsonl 由来) に居ない物 / row_to_id 同様
    fact_id_set = set(latest.keys())
    # M5-d 以降は統合 index.jsonl 1 本から fact_id 集合を取得
    idx_ids: set[str] = _read_index_jsonl_ids(scope.root_dir / INDEX_JSONL_FILENAME)
    rep.orphan_idx_ids = len(idx_ids - fact_id_set)

    if active_model_id and active_model_id in rep.embedding_models:
        emb_ids = set(_read_row_to_id(
            row_to_id_path(scope.root_dir, active_model_id),
        ))
        rep.orphan_embedding_ids = len(emb_ids - fact_id_set)
    elif rep.embedding_models:
        # active 未指定でもいずれかの model から最大の orphan 件数を見せる
        max_orphan = 0
        for model_id in rep.embedding_models:
            emb_ids = set(_read_row_to_id(
                row_to_id_path(scope.root_dir, model_id),
            ))
            max_orphan = max(max_orphan, len(emb_ids - fact_id_set))
        rep.orphan_embedding_ids = max_orphan

    return rep


def run_inspect(
    memory_dir: Path,
    *,
    top_subjects: int = 10,
    scope_filter: str | None = None,
) -> InspectReport:
    """全 scope (または ``scope_filter`` 指定時の単一 scope) を inspect する.

    Args:
        memory_dir: ``local/memory/`` ルート。
        top_subjects: subject 上位件数 (1 scope あたり)。
        scope_filter: ``"global"`` または ``"project:<id>"`` で 1 scope 限定。

    Returns:
        :class:`InspectReport` (全 scope 集約)。
    """
    from backend.free.memory.init_evorefmem import read_schema_version

    rep = InspectReport(
        memory_dir=str(memory_dir),
        schema_version_marker=read_schema_version(memory_dir),
        manifest=None,
    )

    manifest = load_manifest(memory_dir)
    active_model_id: str | None = None
    if manifest is not None:
        rep.manifest = _manifest_summary(manifest)
        active_model_id = manifest.embedding.model_id
    else:
        if manifest_path(memory_dir).exists():
            rep.manifest = {"_error": "manifest.json unreadable"}

    for scope in enumerate_scopes(memory_dir):
        if scope_filter is not None and scope.name != scope_filter:
            continue
        rep.scopes.append(
            inspect_scope(
                scope,
                top_subjects=top_subjects,
                active_model_id=active_model_id,
            ),
        )
    return rep


def _manifest_summary(m: Manifest) -> dict[str, Any]:
    return {
        "schema_version": m.schema_version,
        "component_versions": dict(m.component_versions),
        "embedding": {
            "model_id": m.embedding.model_id,
            "dim": m.embedding.dim,
            "normalized": m.embedding.normalized,
        },
        "created_at": m.created_at,
        "last_migrated_at": m.last_migrated_at,
        "last_compacted_at": m.last_compacted_at,
        "index_generations": dict(m.index_generations),
    }


def format_report_text(report: InspectReport) -> str:
    """人間可読な簡易テキスト形式で整形する (CLI default 出力)。"""
    lines: list[str] = []
    lines.append(f"memory_dir       : {report.memory_dir}")
    lines.append(f"schema_version   : marker={report.schema_version_marker}")
    if report.manifest is None:
        lines.append("manifest         : (missing)")
    else:
        m = report.manifest
        if "_error" in m:
            lines.append(f"manifest         : ERROR — {m['_error']}")
        else:
            emb = m.get("embedding", {})
            lines.append(
                f"manifest         : v{m.get('schema_version')} "
                f"embedding={emb.get('model_id')} dim={emb.get('dim')} "
                f"normalized={emb.get('normalized')}",
            )
    lines.append("")
    if not report.scopes:
        lines.append("(no scopes found)")
        return "\n".join(lines)
    for s in report.scopes:
        lines.append(f"## scope: {s.scope}")
        lines.append(
            f"  facts            : total={s.facts_total} "
            f"lines={s.facts_jsonl_lines} dup={s.duplicate_lines} "
            f"size={s.facts_jsonl_bytes}B",
        )
        lines.append(
            f"  pinned/superseded: {s.pinned} / {s.superseded}",
        )
        if s.by_type:
            type_str = ", ".join(f"{k}={v}" for k, v in s.by_type.items())
            lines.append(f"  by_type          : {type_str}")
        if s.by_pillar or s.pillar_unparsed:
            pillar_str = ", ".join(f"{k}={v}" for k, v in s.by_pillar.items())
            if s.pillar_unparsed:
                pillar_str += f", _unparsed={s.pillar_unparsed}"
            lines.append(f"  by_pillar        : {pillar_str}")
        if s.top_subjects:
            top_str = ", ".join(f"{k}({v})" for k, v in s.top_subjects[:5])
            lines.append(f"  top_subjects     : {top_str}")
        if s.embedding_models:
            for mid in s.embedding_models:
                marker = "*" if mid == s.embedding_active_model else " "
                lines.append(
                    f"  embedding{marker}       : "
                    f"model_id={mid} rows={s.embedding_count_by_model.get(mid, 0)} "
                    f"dim={s.embedding_dim_by_model.get(mid, 0)}",
                )
        if s.orphan_idx_ids or s.orphan_embedding_ids:
            lines.append(
                f"  orphans          : idx={s.orphan_idx_ids} "
                f"embedding={s.orphan_embedding_ids}",
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "InspectReport",
    "ScopeInspectReport",
    "format_report_text",
    "inspect_scope",
    "run_inspect",
]
