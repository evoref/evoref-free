"""``evorefmem_cli compact`` 実装

各 scope の ``facts.jsonl`` を **last-write-wins** で圧縮する.

``facts.jsonl`` は追記式 (`store.add_fact` / `store.update_fact` がそれぞれ
1 行追記する) のため、長期運用では同じ id の行が大量に積み上がる。本コマンドは
最後の行だけを残して書き換える (``SemanticFactStore._rewrite_facts_log`` の
スタンドアロン版)。

破壊的なため:

- デフォルトで dry-run (``--apply`` 必須)
- ``--apply`` 時は ``facts.jsonl`` を ``migration_archive/cli_<utc_ts>/compact/<scope>/``
  へコピー退避してから rewrite
- 完了後に :func:`update_manifest` で ``last_compacted_at`` を更新
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.free.memory.semantic.cli._paths import (
    ScopeInfo,
    cli_backup_root,
    enumerate_scopes,
)
from backend.free.memory.semantic.manifest import (
    load_manifest,
    update_manifest,
)
from backend.free.memory.semantic.store import FACTS_FILENAME
from backend.io import AtomicWriter


@dataclass
class CompactScopeResult:
    """1 scope の compact 計画 / 結果."""

    scope: str
    facts_path: str
    valid_lines: int = 0
    """有効な行数 (空行と malformed を除く)。"""

    malformed_lines: int = 0
    unique_ids: int = 0
    duplicates_removed: int = 0
    """``valid_lines - unique_ids`` で削減される行数。dry-run でも算出。"""

    bytes_before: int = 0
    bytes_after: int = 0
    """dry-run 時は 0 (実書き換えしないため未測定)。"""

    backup_path: str | None = None
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompactReport:
    memory_dir: str
    applied: bool
    scopes: list[CompactScopeResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_dir": self.memory_dir,
            "applied": self.applied,
            "scopes": [s.to_dict() for s in self.scopes],
            "totals": {
                "duplicates_removed": sum(
                    s.duplicates_removed for s in self.scopes
                ),
                "unique_ids": sum(s.unique_ids for s in self.scopes),
                "bytes_freed": sum(
                    max(0, s.bytes_before - s.bytes_after) for s in self.scopes
                ),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ──────────────────────────────────────────────────────────────────────────
# 内部
# ──────────────────────────────────────────────────────────────────────────


def _scan_facts(facts_path: Path) -> tuple[list[str], int, int]:
    """facts.jsonl を走査して (id ごとの最新行 list, valid lines, malformed) を返す.

    出力 list は **挿入順 (=最初に発見した順)** を保つが、各 id について
    最後に見つかった行で値を上書きする。これにより rewrite 後のファイル順序が
    決定論的になる (同 id の最初の出現位置でソート)。
    """
    if not facts_path.exists():
        return [], 0, 0
    order: list[str] = []
    seen: dict[str, str] = {}
    valid = 0
    malformed = 0
    with facts_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            try:
                d = json.loads(stripped)
                fid = d.get("id")
                if not isinstance(fid, str) or not fid:
                    raise ValueError("missing id")
                if fid not in seen:
                    order.append(fid)
                seen[fid] = line
                valid += 1
            except (json.JSONDecodeError, ValueError):
                malformed += 1
    return [seen[fid] for fid in order], valid, malformed


def plan_compact_scope(scope: ScopeInfo) -> CompactScopeResult:
    """1 scope の dry-run plan を作る (副作用なし)."""
    facts_path = scope.root_dir / FACTS_FILENAME
    rewritten, valid, malformed = _scan_facts(facts_path)
    res = CompactScopeResult(
        scope=scope.name,
        facts_path=str(facts_path),
        valid_lines=valid,
        malformed_lines=malformed,
        unique_ids=len(rewritten),
        duplicates_removed=max(0, valid - len(rewritten)),
        bytes_before=facts_path.stat().st_size if facts_path.exists() else 0,
        bytes_after=0,
        backup_path=None,
        applied=False,
    )
    return res


def apply_compact_scope(
    scope: ScopeInfo,
    backup_root: Path,
) -> CompactScopeResult:
    """1 scope の compact を実行する (rewrite). ``backup_root`` 配下へ事前退避."""
    facts_path = scope.root_dir / FACTS_FILENAME
    rewritten, valid, malformed = _scan_facts(facts_path)
    bytes_before = facts_path.stat().st_size if facts_path.exists() else 0
    res = CompactScopeResult(
        scope=scope.name,
        facts_path=str(facts_path),
        valid_lines=valid,
        malformed_lines=malformed,
        unique_ids=len(rewritten),
        duplicates_removed=max(0, valid - len(rewritten)),
        bytes_before=bytes_before,
        bytes_after=0,
        backup_path=None,
        applied=False,
    )
    if not facts_path.exists():
        # 何もない scope は触らない
        return res
    if res.duplicates_removed == 0 and res.malformed_lines == 0:
        # 削減対象なしならスキップ (no-op、応答は applied=False のまま)
        res.bytes_after = bytes_before
        return res

    scope_safe = scope.name.replace(":", "_").replace("/", "_")
    backup_dir = backup_root / scope_safe
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_target = backup_dir / FACTS_FILENAME
    shutil.copy2(facts_path, backup_target)
    res.backup_path = str(backup_target)

    # AtomicWriter が tmp 書込 → os.replace (Windows retry 付き) を担い、
    # 途中例外時は tmp を unlink して facts_path を不変のまま残す。
    with AtomicWriter(facts_path) as f:
        for line in rewritten:
            f.write(line + "\n")
    res.bytes_after = facts_path.stat().st_size
    res.applied = True
    return res


# ──────────────────────────────────────────────────────────────────────────
# 公開 API
# ──────────────────────────────────────────────────────────────────────────


def run_compact(
    memory_dir: Path,
    migration_archive_dir: Path,
    *,
    apply: bool = False,
    scope_filter: str | None = None,
    now: float | None = None,
    update_manifest_timestamp: bool = True,
) -> CompactReport:
    """全 scope (または ``scope_filter`` 指定時の単一 scope) で compact 実行 / 計画."""
    rep = CompactReport(memory_dir=str(memory_dir), applied=apply)
    backup_root: Path | None = None
    if apply:
        backup_root = cli_backup_root(migration_archive_dir, "compact", now=now)

    scopes = enumerate_scopes(memory_dir)
    for scope in scopes:
        if scope_filter is not None and scope.name != scope_filter:
            continue
        if apply:
            assert backup_root is not None
            rep.scopes.append(apply_compact_scope(scope, backup_root))
        else:
            rep.scopes.append(plan_compact_scope(scope))

    if (
        apply
        and update_manifest_timestamp
        and any(s.applied for s in rep.scopes)
        and load_manifest(memory_dir) is not None
    ):
        import time as _time

        t = _time.time() if now is None else now
        update_manifest(
            memory_dir,
            last_compacted_at=_time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", _time.gmtime(t),
            ),
        )
    return rep


def format_report_text(report: CompactReport) -> str:
    lines: list[str] = []
    lines.append(f"memory_dir       : {report.memory_dir}")
    mode = "apply" if report.applied else "dry-run"
    lines.append(f"mode             : {mode}")
    if not report.scopes:
        lines.append("(no scopes found)")
        return "\n".join(lines)
    for s in report.scopes:
        action = "APPLIED" if s.applied else (
            "PLAN" if s.duplicates_removed > 0 else "NOOP"
        )
        lines.append(
            f"  {s.scope:30} [{action}] valid={s.valid_lines} "
            f"unique={s.unique_ids} dup={s.duplicates_removed} "
            f"malformed={s.malformed_lines} "
            f"bytes={s.bytes_before}->{s.bytes_after}",
        )
        if s.backup_path:
            lines.append(f"      backup: {s.backup_path}")
    totals = report.to_dict()["totals"]
    lines.append(
        f"\ntotals: duplicates_removed={totals['duplicates_removed']} "
        f"bytes_freed={totals['bytes_freed']}",
    )
    return "\n".join(lines)


__all__ = [
    "CompactReport",
    "CompactScopeResult",
    "apply_compact_scope",
    "format_report_text",
    "plan_compact_scope",
    "run_compact",
]
