"""``evorefmem_cli verify`` 実装

SemMem の整合性を副作用なしで検査する (c_16 §4.2 の 1 ストア構成)。

検査項目:

- ``SCHEMA_VERSION`` マーカーと code 側 ``SCHEMA_VERSION`` の一致
- manifest の存在と ``record_version`` の一致、active snapshot の実在
- レコードの読めなさ (``attrs.fact_type`` 欠損 = ownership を引けない)
- supersession の健全性 — 未知 id を指す ``superseded_by`` / 閉路
- ``contradicts`` の相互参照と ``veracity=disputed`` の整合
- 埋め込みの被覆率 (snapshot に載っているのにベクトルが無い行)
- ``know.*`` claim の ``provenance[].source_id`` が items 台帳に実在するか
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from backend.free.memory.init_evorefmem import SCHEMA_VERSION, read_schema_version
from backend.free.memory.semantic.cli._paths import (
    open_semantic_store,
    scope_names,
)
from backend.free.memory.semantic.fact import FactRecordError, evidence_to_fact
from backend.free.rag.evidence import RECORD_VERSION

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticStore

Severity = Literal["info", "warning", "error"]


@dataclass
class Issue:
    """検査で検出された 1 件の不整合."""

    severity: Severity
    scope: str
    """``"global"`` / ``"project:<id>"`` / ``"_marker"`` / ``"_manifest"``。"""

    code: str
    """機械可読なコード (``schema_version_mismatch`` 等)。"""

    message: str
    """人間可読な説明文。"""

    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyReport:
    """verify 実行結果サマリ."""

    memory_dir: str
    schema_version_marker: int | None
    expected_schema_version: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    def exit_code(self) -> int:
        """error が 1 件でもあれば 1、そうでなければ 0."""
        return 1 if self.has_errors else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_dir": self.memory_dir,
            "schema_version_marker": self.schema_version_marker,
            "expected_schema_version": self.expected_schema_version,
            "issues": [asdict(i) for i in self.issues],
            "summary": {
                "errors": sum(1 for i in self.issues if i.severity == "error"),
                "warnings": sum(1 for i in self.issues if i.severity == "warning"),
                "infos": sum(1 for i in self.issues if i.severity == "info"),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ──────────────────────────────────────────────────────────────────────────
# 検査ロジック
# ──────────────────────────────────────────────────────────────────────────


def _check_store_meta(store: "SemanticStore", rep: VerifyReport) -> None:
    """manifest と snapshot の存在・版を検査する。"""
    manifest = store.evidence.manifest
    if not manifest.path.exists():
        rep.issues.append(Issue(
            severity="warning",
            scope="_manifest",
            code="manifest_missing",
            message=(
                "semantic/manifest.json not found; the store will be "
                "initialized on the next snapshot"
            ),
        ))
        return
    if int(manifest.record_version) != RECORD_VERSION:
        rep.issues.append(Issue(
            severity="error",
            scope="_manifest",
            code="record_version_mismatch",
            message=(
                f"manifest record_version={manifest.record_version} but code "
                f"expects {RECORD_VERSION}"
            ),
            details={"manifest": manifest.record_version, "code": RECORD_VERSION},
        ))
    version = manifest.active_snapshot
    if version and not (store.store_dir / "snapshot" / version).exists():
        rep.issues.append(Issue(
            severity="error",
            scope="_manifest",
            code="active_snapshot_missing",
            message=f"active snapshot {version} does not exist on disk",
            details={"version": version},
        ))


def _check_records(store: "SemanticStore", rep: VerifyReport) -> None:
    """レコードが ``SemanticFact`` として読めるかを検査する。"""
    unreadable = 0
    for record in store.evidence.iter_records():
        if record.kind not in ("fact", "claim"):
            continue
        try:
            evidence_to_fact(record)
        except FactRecordError:
            unreadable += 1
    if unreadable:
        rep.issues.append(Issue(
            severity="error",
            scope="_store",
            code="unreadable_records",
            message=(
                f"{unreadable} record(s) lack attrs.fact_type and cannot be "
                "mapped to a SemanticFact (ownership is unresolvable)"
            ),
            details={"count": unreadable},
        ))


def verify_scope(store: "SemanticStore", scope: str, rep: VerifyReport) -> None:
    """1 scope 分の検査を実行し、``rep.issues`` へ追記する。"""
    facts = store.all_facts(include_superseded=True, scope=scope)
    known = {f.id for f in store.all_facts(include_superseded=True)}

    dangling = [
        f.id for f in facts
        if f.superseded_by and f.superseded_by not in known
    ]
    if dangling:
        rep.issues.append(Issue(
            severity="error",
            scope=scope,
            code="dangling_supersede",
            message=(
                f"{len(dangling)} fact(s) point at a superseder that no longer "
                "exists; their slot has no live value"
            ),
            details={"fact_ids": dangling[:20]},
        ))

    cycles = sorted(_supersede_cycles(facts))
    if cycles:
        rep.issues.append(Issue(
            severity="error",
            scope=scope,
            code="supersede_cycle",
            message=(
                f"{len(cycles)} fact(s) form a supersede cycle; the slot reads "
                "as empty even though the value is stored"
            ),
            details={"fact_ids": cycles[:20]},
        ))

    by_id = {f.id: f for f in facts}
    broken_contradicts = [
        f.id for f in facts
        if f.veracity == "disputed"
        and not any(
            other in by_id and f.id in (by_id[other].contradicts or [])
            for other in (f.contradicts or [])
        )
    ]
    if broken_contradicts:
        rep.issues.append(Issue(
            severity="warning",
            scope=scope,
            code="one_sided_dispute",
            message=(
                f"{len(broken_contradicts)} disputed fact(s) have no partner "
                "pointing back; the conflict cannot be reviewed as a group"
            ),
            details={"fact_ids": broken_contradicts[:20]},
        ))

    item_ids = {f"item:{item.id}" for item in store.items.all()}
    orphan_claims = [
        f.id for f in facts
        if f.type == "claim"
        and not any(
            (p.source_id or "") in item_ids for p in (f.provenances or ())
        )
    ]
    if orphan_claims:
        rep.issues.append(Issue(
            severity="warning",
            scope=scope,
            code="claim_without_item",
            message=(
                f"{len(orphan_claims)} claim(s) reference an acquisition item "
                "that is not in items.jsonl"
            ),
            details={"fact_ids": orphan_claims[:20]},
        ))


def _supersede_cycles(facts: list[Any]) -> set[str]:
    """``superseded_by`` の閉路に属する fact id を返す。"""
    nexts = {f.id: f.superseded_by for f in facts if f.superseded_by}
    cyclic: set[str] = set()
    for start in list(nexts):
        seen: list[str] = []
        cur: str | None = start
        while cur is not None and cur in nexts and cur not in seen:
            seen.append(cur)
            cur = nexts[cur]
        if cur is not None and cur in seen:
            cyclic.update(seen[seen.index(cur):])
    return cyclic


def _check_embedding_coverage(store: "SemanticStore", rep: VerifyReport) -> None:
    """snapshot に載っているのにベクトルが無い行を数える。"""
    snapshot = store.evidence.snapshot
    if snapshot is None or len(snapshot) == 0:
        return
    vector_store = store.evidence.vector_store()
    if vector_store is None:
        rep.issues.append(Issue(
            severity="warning",
            scope="_store",
            code="embeddings_missing",
            message=(
                "no embedding index for the active snapshot; dense recall is "
                "unavailable until the next sleep-time snapshot"
            ),
        ))
        return
    embedded = {str(meta.get("id") or "") for meta in vector_store.metadata}
    missing = sum(
        1 for row in range(len(snapshot)) if snapshot.id_at(row) not in embedded
    )
    if missing:
        rep.issues.append(Issue(
            severity="info",
            scope="_store",
            code="embedding_gap",
            message=f"{missing} snapshot row(s) have no vector yet",
            details={"count": missing, "total": len(snapshot)},
        ))


def run_verify(
    memory_dir: Path, *, scope_filter: str | None = None,
) -> VerifyReport:
    """SemMem 全体を検査する (読み取りのみ)。"""
    memory_dir = Path(memory_dir)
    marker = read_schema_version(memory_dir)
    rep = VerifyReport(
        memory_dir=str(memory_dir),
        schema_version_marker=marker,
        expected_schema_version=SCHEMA_VERSION,
    )
    if marker is None:
        rep.issues.append(Issue(
            severity="warning",
            scope="_marker",
            code="schema_version_missing",
            message="SCHEMA_VERSION marker not found (store not initialized)",
        ))
    elif marker != SCHEMA_VERSION:
        rep.issues.append(Issue(
            severity="error",
            scope="_marker",
            code="schema_version_mismatch",
            message=(
                f"SCHEMA_VERSION marker={marker} but code expects "
                f"{SCHEMA_VERSION}"
            ),
            details={"marker": marker, "expected": SCHEMA_VERSION},
        ))

    store = open_semantic_store(memory_dir)
    try:
        _check_store_meta(store, rep)
        _check_records(store, rep)
        _check_embedding_coverage(store, rep)
        for scope in scope_names(store):
            if scope_filter is not None and scope != scope_filter:
                continue
            verify_scope(store, scope, rep)
    finally:
        store.close()
    return rep


def format_report_text(report: VerifyReport) -> str:
    """人間可読なテキスト形式で整形する。"""
    lines: list[str] = []
    lines.append(f"memory_dir       : {report.memory_dir}")
    lines.append(
        f"schema_version   : marker={report.schema_version_marker} "
        f"expected={report.expected_schema_version}",
    )
    severity_ctr: Counter[str] = Counter(i.severity for i in report.issues)
    lines.append(
        f"summary          : errors={severity_ctr['error']} "
        f"warnings={severity_ctr['warning']} infos={severity_ctr['info']}",
    )
    lines.append("")
    if not report.issues:
        lines.append("status           : OK (no issues found)")
        return "\n".join(lines)
    for i in report.issues:
        lines.append(f"[{i.severity:7}] {i.scope:20} {i.code}")
        lines.append(f"    {i.message}")
    return "\n".join(lines)


__all__ = [
    "Issue",
    "Severity",
    "VerifyReport",
    "format_report_text",
    "run_verify",
    "verify_scope",
]
