"""``evorefmem_cli verify`` 実装

EvorefMem ストアの整合性を検査する。`init_evorefmem.py --check` の上位互換。

検査項目:

1. ``SCHEMA_VERSION`` マーカーが ``backend.free.memory.init_evorefmem.SCHEMA_VERSION``
   と一致するか
2. ``manifest.json`` が読み込み可能で ``schema_version`` がマーカーと一致するか
3. 各 scope の ``embeddings/<active>/vectors.npy`` の dim が
   ``manifest.embedding.dim`` と一致するか
4. 各 scope の ``.idx`` 群が ``facts.jsonl`` と整合 (ID リーク無し / ID 漏れ無し)
5. 各 scope の ``embeddings/<active>/row_to_id.json`` の ID が ``facts.jsonl``
   に存在するか (orphan 検出)
6. ``facts.jsonl`` 内に重複 id があれば WARNING レベルの issue として報告
   (compact が削減可能)

検査結果は :class:`VerifyReport` にまとめて返す。issue が 1 件でも error 以上で
あれば :meth:`VerifyReport.exit_code` は 1、warning までなら 0。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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
    MANIFEST_SCHEMA_VERSION,
    load_manifest,
    manifest_path,
)
from backend.free.memory.semantic.store import (
    FACTS_FILENAME,
    INDEX_JSONL_FILENAME,
    PILLAR_SUBJECT_PREFIXES,
)
from backend.free.memory.semantic.subject_key import SubjectKey

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


def _read_facts(
    facts_path: Path,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """facts.jsonl を読み、(id -> 最新 dict, valid lines, malformed) を返す."""
    if not facts_path.exists():
        return {}, 0, 0
    latest: dict[str, dict[str, Any]] = {}
    valid = 0
    malformed = 0
    with facts_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                fid = d.get("id")
                if not isinstance(fid, str) or not fid:
                    raise ValueError("missing id")
                latest[fid] = d
                valid += 1
            except (json.JSONDecodeError, ValueError):
                malformed += 1
    return latest, valid, malformed


def _read_index_jsonl(idx_path: Path) -> dict[str, dict] | None:
    """新形式 ``index.jsonl`` を ``{fact_id: attrs}`` 辞書として読む.

    tombstone 行は該当 key を除外。読み込みエラー / 欠損時は None。
    """
    if not idx_path.exists():
        return None
    out: dict[str, dict] = {}
    try:
        with idx_path.open("r", encoding="utf-8") as f:
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
                        out.pop(key, None)
                    continue
                fact_id = obj.get("fact_id")
                if isinstance(fact_id, str):
                    out[fact_id] = {
                        "subject": obj.get("subject"),
                        "type": obj.get("type"),
                        "pillar": obj.get("pillar"),
                        "pinned": bool(obj.get("pinned", False)),
                    }
    except OSError:
        return None
    return out


def _check_index_jsonl_against_facts(
    rep: VerifyReport,
    scope: ScopeInfo,
    *,
    expected_by_subject: dict[str, set[str]],
    expected_by_type: dict[str, set[str]],
    expected_by_pillar: dict[str, set[str]],
    expected_pinned: set[str],
) -> None:
    """新形式 ``index.jsonl`` と ``facts.jsonl`` 由来の期待値を cross-check する。

    各 fact_id について subject / type / pillar / pinned の 4 属性が一致するか
    確認する。M5-d で 4 索引が統合された結果、cross-check も 1 ファイル走査で済む。
    """
    idx_path = scope.root_dir / INDEX_JSONL_FILENAME

    # 期待値を fact_id → 属性 dict に変換
    expected_attrs: dict[str, dict] = {}
    for subj, ids in expected_by_subject.items():
        for fid in ids:
            expected_attrs.setdefault(fid, {})["subject"] = subj
    for ftype, ids in expected_by_type.items():
        for fid in ids:
            expected_attrs.setdefault(fid, {})["type"] = ftype
    for pillar, ids in expected_by_pillar.items():
        for fid in ids:
            expected_attrs.setdefault(fid, {})["pillar"] = pillar
    # pinned/pillar は属性として明示
    for fid in expected_attrs:
        expected_attrs[fid].setdefault("pillar", None)
        expected_attrs[fid]["pinned"] = fid in expected_pinned

    actual_attrs = _read_index_jsonl(idx_path)
    if actual_attrs is None:
        if expected_attrs:
            rep.issues.append(Issue(
                severity="error",
                scope=scope.name,
                code="index_jsonl_missing_or_malformed",
                message=f"{INDEX_JSONL_FILENAME} is missing or malformed but "
                        f"facts.jsonl requires {len(expected_attrs)} entries",
                details={"idx_path": str(idx_path)},
            ))
        return

    expected_ids = set(expected_attrs.keys())
    actual_ids = set(actual_attrs.keys())
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing:
        rep.issues.append(Issue(
            severity="error",
            scope=scope.name,
            code="index_jsonl_missing_facts",
            message=f"{INDEX_JSONL_FILENAME} is missing {len(missing)} fact(s)",
            details={"ids_sample": sorted(missing)[:5]},
        ))
    if extra:
        rep.issues.append(Issue(
            severity="error",
            scope=scope.name,
            code="index_jsonl_extra_facts",
            message=f"{INDEX_JSONL_FILENAME} has {len(extra)} fact(s) "
                    "not present in facts.jsonl",
            details={"ids_sample": sorted(extra)[:5]},
        ))
    # 属性 cross-check
    for fid in expected_ids & actual_ids:
        exp = expected_attrs[fid]
        act = actual_attrs[fid]
        mismatches: list[tuple[str, object, object]] = []
        for attr in ("subject", "type", "pillar"):
            if exp.get(attr) != act.get(attr):
                mismatches.append((attr, exp.get(attr), act.get(attr)))
        if bool(exp.get("pinned", False)) != bool(act.get("pinned", False)):
            mismatches.append((
                "pinned",
                bool(exp.get("pinned", False)),
                bool(act.get("pinned", False)),
            ))
        if mismatches:
            rep.issues.append(Issue(
                severity="error",
                scope=scope.name,
                code="index_jsonl_attr_mismatch",
                message=(
                    f"{INDEX_JSONL_FILENAME}[{fid}] attribute mismatch: "
                    f"{mismatches}"
                ),
                details={"fact_id": fid, "mismatches": mismatches},
            ))


def verify_scope(
    scope: ScopeInfo,
    rep: VerifyReport,
    *,
    active_model_id: str | None,
    expected_dim: int | None,
) -> None:
    """1 scope の整合性を検査して issue を ``rep`` に追加する."""
    facts_path = scope.root_dir / FACTS_FILENAME
    latest, valid_lines, malformed = _read_facts(facts_path)

    if malformed > 0:
        rep.issues.append(Issue(
            severity="error",
            scope=scope.name,
            code="facts_jsonl_malformed",
            message=f"{malformed} malformed line(s) in facts.jsonl",
            details={"facts_path": str(facts_path)},
        ))
    if valid_lines > len(latest):
        rep.issues.append(Issue(
            severity="warning",
            scope=scope.name,
            code="facts_jsonl_duplicates",
            message=f"facts.jsonl contains {valid_lines - len(latest)} "
                    "duplicate line(s); run `evorefmem_cli compact` to reduce",
            details={"valid_lines": valid_lines, "unique_ids": len(latest)},
        ))

    # 期待される .idx を facts.jsonl から再構築
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

    # M5-d 以降: 4 索引 → 統合 index.jsonl 1 本に統合された。verify は
    # facts.jsonl から構築した期待値 (by_subject / by_type / by_pillar /
    # pinned) と、新形式 index.jsonl 上の各 fact 属性を cross-check する。
    _check_index_jsonl_against_facts(
        rep,
        scope,
        expected_by_subject=dict(by_subject),
        expected_by_type=dict(by_type),
        expected_by_pillar=dict(by_pillar),
        expected_pinned=pinned,
    )

    # embedding 整合性
    fact_ids = set(latest.keys())
    models = list_stored_models(scope.root_dir)

    if active_model_id is not None and active_model_id not in models and models:
        rep.issues.append(Issue(
            severity="warning",
            scope=scope.name,
            code="embedding_active_missing",
            message=f"manifest.embedding.model_id={active_model_id!r} but "
                    f"scope has only {models!r}",
            details={"models": models},
        ))

    for model_id in models:
        vec_p = vectors_path(scope.root_dir, model_id)
        rid_p = row_to_id_path(scope.root_dir, model_id)
        rows, dim = _vectors_dim(vec_p)
        ids = _read_row_to_id(rid_p)
        if rows != len(ids):
            rep.issues.append(Issue(
                severity="error",
                scope=scope.name,
                code="embedding_rows_id_mismatch",
                message=f"{model_id}: vectors.npy has {rows} rows but "
                        f"row_to_id.json has {len(ids)} ids",
                details={"model_id": model_id, "rows": rows, "ids": len(ids)},
            ))
        if active_model_id is not None and model_id == active_model_id:
            if expected_dim is not None and dim and expected_dim != dim:
                rep.issues.append(Issue(
                    severity="error",
                    scope=scope.name,
                    code="embedding_dim_mismatch",
                    message=f"manifest.embedding.dim={expected_dim} but "
                            f"vectors.npy has dim={dim} (model_id={model_id})",
                    details={
                        "model_id": model_id,
                        "expected": expected_dim,
                        "actual": dim,
                    },
                ))
        # orphan
        orphan = set(ids) - fact_ids
        if orphan:
            rep.issues.append(Issue(
                severity="warning",
                scope=scope.name,
                code="embedding_orphan_ids",
                message=f"{model_id}: {len(orphan)} embedding id(s) not in facts.jsonl",
                details={
                    "model_id": model_id,
                    "ids_sample": sorted(orphan)[:5],
                },
            ))

    # subject prefix の妥当性 (情報レベル)
    bad_subj = sum(
        1 for d in latest.values()
        if isinstance(d.get("subject"), str)
        and not any(d["subject"].startswith(p) for p in PILLAR_SUBJECT_PREFIXES)
    )
    if bad_subj > 0:
        rep.issues.append(Issue(
            severity="info",
            scope=scope.name,
            code="subject_without_pillar_prefix",
            message=f"{bad_subj} fact(s) have subject without pillar prefix "
                    "(legacy / natural subjects)",
            details={"count": bad_subj},
        ))


def _vectors_dim(vectors_p: Path) -> tuple[int, int]:
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


def _read_row_to_id(rid_path: Path) -> list[str]:
    if not rid_path.exists():
        return []
    try:
        data = json.loads(rid_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    ids = data.get("row_to_id", []) if isinstance(data, dict) else []
    return [str(x) for x in ids] if isinstance(ids, list) else []


# ──────────────────────────────────────────────────────────────────────────
# 公開 API
# ──────────────────────────────────────────────────────────────────────────


def run_verify(
    memory_dir: Path,
    *,
    scope_filter: str | None = None,
) -> VerifyReport:
    """全 scope の整合性を検査する.

    Args:
        memory_dir: ``local/memory/`` ルート。
        scope_filter: ``"global"`` / ``"project:<id>"`` で 1 scope 限定。

    Returns:
        :class:`VerifyReport`。``exit_code()`` で CLI 終了コードへ変換可能。
    """
    from backend.free.memory.init_evorefmem import (
        SCHEMA_VERSION,
        read_schema_version,
    )

    marker = read_schema_version(memory_dir)
    rep = VerifyReport(
        memory_dir=str(memory_dir),
        schema_version_marker=marker,
        expected_schema_version=SCHEMA_VERSION,
    )

    if marker is None:
        rep.issues.append(Issue(
            severity="error",
            scope="_marker",
            code="schema_marker_missing",
            message="SCHEMA_VERSION marker not found; "
                    "run `python scripts/evorefmem_cli.py init`",
            details={"path": str(memory_dir / "semantic" / "SCHEMA_VERSION")},
        ))
    elif marker != SCHEMA_VERSION:
        rep.issues.append(Issue(
            severity="error",
            scope="_marker",
            code="schema_marker_mismatch",
            message=f"SCHEMA_VERSION marker={marker} but expected={SCHEMA_VERSION}",
            details={"marker": marker, "expected": SCHEMA_VERSION},
        ))

    manifest = load_manifest(memory_dir)
    active_model_id: str | None = None
    expected_dim: int | None = None
    if manifest is None:
        if manifest_path(memory_dir).exists():
            rep.issues.append(Issue(
                severity="error",
                scope="_manifest",
                code="manifest_unreadable",
                message="manifest.json exists but is unreadable / malformed",
                details={"path": str(manifest_path(memory_dir))},
            ))
        else:
            rep.issues.append(Issue(
                severity="warning",
                scope="_manifest",
                code="manifest_missing",
                message="manifest.json not found; embedding dim check skipped",
                details={"path": str(manifest_path(memory_dir))},
            ))
    else:
        active_model_id = manifest.embedding.model_id
        expected_dim = manifest.embedding.dim
        # manifest.schema_version は manifest 自体のスキーマ版 (MANIFEST_SCHEMA_VERSION)
        # で、EvorefMem 全体の SCHEMA_VERSION マーカーとは別軸。両者が乖離する
        # ことは設計上正常 (M5-d 以降は marker=2 / manifest.schema_version=1)。
        # 比較対象は MANIFEST_SCHEMA_VERSION に揃える。
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            rep.issues.append(Issue(
                severity="warning",
                scope="_manifest",
                code="manifest_schema_version_mismatch",
                message=f"manifest.schema_version={manifest.schema_version} "
                        f"differs from MANIFEST_SCHEMA_VERSION="
                        f"{MANIFEST_SCHEMA_VERSION}",
                details={
                    "manifest": manifest.schema_version,
                    "expected": MANIFEST_SCHEMA_VERSION,
                },
            ))

    for scope in enumerate_scopes(memory_dir):
        if scope_filter is not None and scope.name != scope_filter:
            continue
        verify_scope(
            scope, rep,
            active_model_id=active_model_id,
            expected_dim=expected_dim,
        )

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
