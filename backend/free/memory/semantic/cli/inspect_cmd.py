"""``evorefmem_cli inspect`` 実装

副作用なしで SemMem の統計情報を集計する (c_16 §4.2 の 1 ストア構成)。

- fact 数 / 型別分布 / namespace 別分布 / scope 別分布
- subject 上位 N 件
- 事象ログ / snapshot / 埋め込みの状態 (manifest 由来)
- 競合中 (``veracity=disputed``) と supersede 済みの件数
- ``know.*`` の取得元 / 取得単位の件数
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.free.memory.semantic.cli._paths import (
    open_semantic_store,
    scope_names,
)
from backend.free.memory.semantic.namespaces import namespace_of

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticStore


@dataclass
class ScopeInspectReport:
    """1 scope 分の inspect 結果."""

    scope: str
    """scope 名 (``global`` / ``project:<id>``)。"""

    facts_total: int = 0
    """その scope の live なファクト数 (retracted は含まない)。"""

    by_type: dict[str, int] = field(default_factory=dict)
    """FactType (``attrs.fact_type``) ごとの件数。"""

    by_namespace: dict[str, int] = field(default_factory=dict)
    """namespace (``mem`` / ``know`` / ``idx`` / ``loop`` / ``learn``) ごとの件数。"""

    by_origin: dict[str, int] = field(default_factory=dict)
    """``origin`` ごとの件数 (``user`` / ``tool`` / ``assistant`` …)。"""

    pinned: int = 0
    """``pinned == True`` な fact 数。"""

    superseded: int = 0
    """``superseded_by`` がセットされている fact 数。"""

    disputed: int = 0
    """``veracity == "disputed"`` (競合が未解決) の fact 数。"""

    top_subjects: list[tuple[str, int]] = field(default_factory=list)
    """subject 上位 N 件 (デフォルト 10)。"""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InspectReport:
    """ストア全体を集約した inspect 結果."""

    memory_dir: str
    schema_version_marker: int | None
    store: dict[str, Any] = field(default_factory=dict)
    """manifest 由来のストア状態 (active snapshot / 事象数 / 埋め込みモデル)。"""

    knowledge: dict[str, int] = field(default_factory=dict)
    """``know.*`` の台帳件数 (``sources`` / ``items``)。"""

    scopes: list[ScopeInspectReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_dir": self.memory_dir,
            "schema_version_marker": self.schema_version_marker,
            "store": self.store,
            "knowledge": self.knowledge,
            "scopes": [s.to_dict() for s in self.scopes],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def inspect_scope(
    store: "SemanticStore", scope: str, *, top_subjects: int = 10,
) -> ScopeInspectReport:
    """1 scope 分の inspect を実行する (副作用なし)。"""
    rep = ScopeInspectReport(scope=scope)
    type_ctr: Counter[str] = Counter()
    ns_ctr: Counter[str] = Counter()
    origin_ctr: Counter[str] = Counter()
    subject_ctr: Counter[str] = Counter()
    for fact in store.all_facts(include_superseded=True, scope=scope):
        rep.facts_total += 1
        type_ctr[str(fact.type)] += 1
        ns_ctr[namespace_of(fact.subject)] += 1
        origin_ctr[str(fact.origin)] += 1
        subject_ctr[fact.subject] += 1
        if fact.pinned:
            rep.pinned += 1
        if fact.superseded_by:
            rep.superseded += 1
        if fact.veracity == "disputed":
            rep.disputed += 1
    rep.by_type = dict(sorted(type_ctr.items()))
    rep.by_namespace = dict(sorted(ns_ctr.items()))
    rep.by_origin = dict(sorted(origin_ctr.items()))
    rep.top_subjects = subject_ctr.most_common(top_subjects)
    return rep


def run_inspect(
    memory_dir: Path,
    *,
    top_subjects: int = 10,
    scope_filter: str | None = None,
) -> InspectReport:
    """SemMem 全体を inspect する (読み取りのみ)。"""
    from backend.free.memory.init_evorefmem import read_schema_version

    memory_dir = Path(memory_dir)
    store = open_semantic_store(memory_dir)
    manifest = store.evidence.manifest
    report = InspectReport(
        memory_dir=str(memory_dir),
        schema_version_marker=read_schema_version(memory_dir),
        store={
            "record_version": manifest.record_version,
            "active_snapshot": manifest.active_snapshot,
            "events_since_snapshot": manifest.events_since_snapshot,
            "embedding_model_id": manifest.embedding_model_id,
            "embedding_dim": manifest.embedding_dim,
            "retention": dict(manifest.retention),
            "records": len(store.evidence),
            "facts": len(store),
        },
        knowledge={
            "sources": len(store.sources.all()),
            "items": len(store.items.all()),
        },
    )
    for scope in scope_names(store):
        if scope_filter is not None and scope != scope_filter:
            continue
        report.scopes.append(
            inspect_scope(store, scope, top_subjects=top_subjects),
        )
    store.close()
    return report


def format_report_text(report: InspectReport) -> str:
    """人間向けのテキスト表現に整形する。"""
    lines = [
        f"memory_dir            : {report.memory_dir}",
        f"schema_version marker : {report.schema_version_marker}",
        f"active snapshot       : {report.store.get('active_snapshot') or '(none)'}",
        f"events since snapshot : {report.store.get('events_since_snapshot')}",
        f"embedding model       : {report.store.get('embedding_model_id') or '(none)'}"
        f" (dim={report.store.get('embedding_dim')})",
        f"records / facts       : {report.store.get('records')} / "
        f"{report.store.get('facts')}",
        f"knowledge sources     : {report.knowledge.get('sources', 0)}",
        f"knowledge items       : {report.knowledge.get('items', 0)}",
        "",
    ]
    for scope in report.scopes:
        lines.append(f"[{scope.scope}]")
        lines.append(
            f"  facts={scope.facts_total} pinned={scope.pinned} "
            f"superseded={scope.superseded} disputed={scope.disputed}",
        )
        lines.append(f"  by_type      : {scope.by_type}")
        lines.append(f"  by_namespace : {scope.by_namespace}")
        lines.append(f"  by_origin    : {scope.by_origin}")
        if scope.top_subjects:
            lines.append("  top subjects :")
            lines.extend(
                f"    {count:5d}  {subject}"
                for subject, count in scope.top_subjects
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
