"""``evorefmem_cli rebuild-indices`` 実装

各 scope の統合索引 ``index.jsonl`` を ``facts.jsonl`` から決定論的に
再生成する (M5-d 以降の新形式)。

実装は :class:`SemanticFactStore` のロード時の自動 rebuild に乗る:
``SemanticFactStore(scope.root_dir)`` を作るだけで in-memory に索引が
構築される。``--apply`` 時は store の ``_index_updater`` を経由して全 fact を
upsert し、最後に ``compact()`` で物理ファイルを ``facts.jsonl`` ベースの最新
状態に書き直す。
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.free.memory.semantic.cli._paths import (
    ScopeInfo,
    cli_backup_root,
    enumerate_scopes,
)
from backend.free.memory.semantic.store import (
    FACTS_FILENAME,
    INDEX_JSONL_FILENAME,
    SemanticFactStore,
    _fact_to_index_attrs,
)
from backend.free.memory.semantic.subject_key import SubjectKey

#: rebuild 対象の永続索引ファイル名 (M5-d 以降は統合 index.jsonl 1 本)。
IDX_FILENAMES: tuple[str, ...] = (INDEX_JSONL_FILENAME,)


@dataclass
class RebuildScopeResult:
    scope: str
    facts_total: int = 0
    by_subject_keys: int = 0
    by_type_keys: int = 0
    by_pillar_keys: int = 0
    pinned_count: int = 0
    backup_path: str | None = None
    applied: bool = False
    """True なら .idx を実際に上書きした。"""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RebuildReport:
    memory_dir: str
    applied: bool
    scopes: list[RebuildScopeResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_dir": self.memory_dir,
            "applied": self.applied,
            "scopes": [s.to_dict() for s in self.scopes],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ──────────────────────────────────────────────────────────────────────────
# 内部
# ──────────────────────────────────────────────────────────────────────────


def _scan_facts(facts_path: Path) -> dict[str, dict[str, Any]]:
    """facts.jsonl を last-write-wins で読んで {id: dict} を返す."""
    if not facts_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with facts_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                fid = d.get("id")
                if isinstance(fid, str) and fid:
                    out[fid] = d
            except json.JSONDecodeError:
                continue
    return out


def _build_index_summary(latest: dict[str, dict[str, Any]]) -> dict[str, int]:
    """facts dict から {idx_filename: key 数 / id 数} を計算する."""
    by_subject: dict[str, set[str]] = defaultdict(set)
    by_type: dict[str, set[str]] = defaultdict(set)
    by_pillar: dict[str, set[str]] = defaultdict(set)
    pinned: set[str] = set()
    for fid, d in latest.items():
        subj = d.get("subject")
        ftype = d.get("type")
        if isinstance(subj, str):
            by_subject[subj].add(fid)
            key = SubjectKey.try_parse(subj)
            if key is not None:
                by_pillar[key.pillar].add(fid)
        if isinstance(ftype, str):
            by_type[ftype].add(fid)
        if bool(d.get("pinned", False)):
            pinned.add(fid)
    return {
        "subject_keys": len(by_subject),
        "type_keys": len(by_type),
        "pillar_keys": len(by_pillar),
        "pinned_count": len(pinned),
    }


def plan_rebuild_scope(scope: ScopeInfo) -> RebuildScopeResult:
    facts_path = scope.root_dir / FACTS_FILENAME
    latest = _scan_facts(facts_path)
    summary = _build_index_summary(latest)
    return RebuildScopeResult(
        scope=scope.name,
        facts_total=len(latest),
        by_subject_keys=summary["subject_keys"],
        by_type_keys=summary["type_keys"],
        by_pillar_keys=summary["pillar_keys"],
        pinned_count=summary["pinned_count"],
        applied=False,
    )


def apply_rebuild_scope(
    scope: ScopeInfo,
    backup_root: Path,
) -> RebuildScopeResult:
    res = plan_rebuild_scope(scope)
    # 退避
    scope_safe = scope.name.replace(":", "_").replace("/", "_")
    backup_dir = backup_root / scope_safe
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up = False
    for fname in IDX_FILENAMES:
        src = scope.root_dir / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
            backed_up = True
    if backed_up:
        res.backup_path = str(backup_dir)
    # SemanticFactStore を再構築。コンストラクタが in-memory 索引を facts.jsonl
    # から再構築する。M5-d 以降は新形式 index.jsonl への書き出しが永続索引なので、
    # 既存 in-memory state を _index_updater 経由で全件 upsert + compact する。
    store = SemanticFactStore(scope.root_dir)
    for fact in store._facts.values():  # noqa: SLF001 (rebuild の本責務)
        store._index_updater.upsert(fact.id, _fact_to_index_attrs(fact))
    store._index_updater.compact()
    res.applied = True
    return res


# ──────────────────────────────────────────────────────────────────────────
# 公開 API
# ──────────────────────────────────────────────────────────────────────────


def run_rebuild_indices(
    memory_dir: Path,
    migration_archive_dir: Path,
    *,
    apply: bool = False,
    scope_filter: str | None = None,
    now: float | None = None,
) -> RebuildReport:
    rep = RebuildReport(memory_dir=str(memory_dir), applied=apply)
    backup_root: Path | None = None
    if apply:
        backup_root = cli_backup_root(
            migration_archive_dir, "rebuild_indices", now=now,
        )
    for scope in enumerate_scopes(memory_dir):
        if scope_filter is not None and scope.name != scope_filter:
            continue
        if apply:
            assert backup_root is not None
            rep.scopes.append(apply_rebuild_scope(scope, backup_root))
        else:
            rep.scopes.append(plan_rebuild_scope(scope))
    return rep


def format_report_text(report: RebuildReport) -> str:
    lines: list[str] = []
    lines.append(f"memory_dir       : {report.memory_dir}")
    mode = "apply" if report.applied else "dry-run"
    lines.append(f"mode             : {mode}")
    if not report.scopes:
        lines.append("(no scopes found)")
        return "\n".join(lines)
    for s in report.scopes:
        action = "APPLIED" if s.applied else "PLAN"
        lines.append(
            f"  {s.scope:30} [{action}] facts={s.facts_total} "
            f"subject_keys={s.by_subject_keys} type_keys={s.by_type_keys} "
            f"pillar_keys={s.by_pillar_keys} pinned={s.pinned_count}",
        )
        if s.backup_path:
            lines.append(f"      backup: {s.backup_path}")
    return "\n".join(lines)


__all__ = [
    "IDX_FILENAMES",
    "RebuildReport",
    "RebuildScopeResult",
    "apply_rebuild_scope",
    "format_report_text",
    "plan_rebuild_scope",
    "run_rebuild_indices",
]
