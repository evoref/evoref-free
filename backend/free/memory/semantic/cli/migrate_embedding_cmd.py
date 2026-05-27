"""``evorefmem_cli migrate-embedding`` 実装

:func:`backend.free.memory.semantic.embedding_store.swap_active_model_id`
を CLI から駆動する.

このコマンド自身は **manifest.embedding を atomic 書換える** だけで、
新モデルでの再埋め込み計算は行わない (再埋め込みは
sleep-time バックグラウンド処理または別途運用ジョブの責務)。

破壊的なため:
- デフォルトで dry-run (``--apply`` 必須)
- ``--apply`` 時は既存 manifest を ``migration_archive/cli_<utc_ts>/migrate_embedding/``
  に退避してから ``swap_active_model_id`` を呼ぶ
- 新 ``model_id`` 配下の ``embeddings/<new_model_id>/`` ディレクトリは
  事前に :func:`register_new_model` で作成 (実埋め込み未投入でも可)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.free.memory.semantic.cli._paths import (
    cli_backup_root,
    enumerate_scopes,
)
from backend.free.memory.semantic.embedding_store import (
    list_stored_models,
    register_new_model,
    swap_active_model_id,
)
from backend.free.memory.semantic.manifest import (
    MANIFEST_FILENAME,
    load_manifest,
)


@dataclass
class MigrateEmbeddingReport:
    memory_dir: str
    current_model_id: str | None
    current_dim: int | None
    new_model_id: str
    new_dim: int
    normalized: bool
    applied: bool
    backup_path: str | None = None
    scopes_with_new_dir: list[str] = field(default_factory=list)
    """新 model_id 用ディレクトリが既に存在 / 新規作成された scope 一覧。"""

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def run_migrate_embedding(
    memory_dir: Path,
    migration_archive_dir: Path,
    *,
    new_model_id: str,
    new_dim: int,
    normalized: bool = True,
    apply: bool = False,
    create_dirs: bool = True,
    now: float | None = None,
) -> MigrateEmbeddingReport:
    """埋め込み active model を切り替える"""
    if not new_model_id:
        raise ValueError("new_model_id must be non-empty")
    if new_dim < 1:
        raise ValueError("new_dim must be >= 1")

    manifest = load_manifest(memory_dir)
    rep = MigrateEmbeddingReport(
        memory_dir=str(memory_dir),
        current_model_id=manifest.embedding.model_id if manifest else None,
        current_dim=manifest.embedding.dim if manifest else None,
        new_model_id=new_model_id,
        new_dim=new_dim,
        normalized=normalized,
        applied=apply,
    )
    if manifest is None:
        rep.error = (
            "manifest.json not found; embedding swap requires an initialized "
            "manifest (run `evorefmem_cli init` first)"
        )
        return rep

    if (
        manifest.embedding.model_id == new_model_id
        and manifest.embedding.dim == new_dim
        and manifest.embedding.normalized == normalized
    ):
        rep.error = (
            f"already at model_id={new_model_id} dim={new_dim} "
            f"normalized={normalized}; nothing to do"
        )
        return rep

    if not apply:
        # dry-run: どの scope が新 model dir を持つか / 持っていないかだけ報告
        for scope in enumerate_scopes(memory_dir):
            existing = list_stored_models(scope.root_dir)
            if new_model_id in existing:
                rep.scopes_with_new_dir.append(scope.name)
        return rep

    # apply path
    backup_root = cli_backup_root(
        migration_archive_dir, "migrate_embedding", now=now,
    )
    manifest_src = memory_dir / "semantic" / MANIFEST_FILENAME
    if manifest_src.exists():
        shutil.copy2(manifest_src, backup_root / MANIFEST_FILENAME)
        rep.backup_path = str(backup_root / MANIFEST_FILENAME)

    if create_dirs:
        for scope in enumerate_scopes(memory_dir):
            register_new_model(scope.root_dir, new_model_id)
            rep.scopes_with_new_dir.append(scope.name)

    swap_active_model_id(
        memory_dir, new_model_id, new_dim=new_dim, normalized=normalized,
    )
    return rep


def format_report_text(report: MigrateEmbeddingReport) -> str:
    lines: list[str] = []
    lines.append(f"memory_dir       : {report.memory_dir}")
    mode = "apply" if report.applied else "dry-run"
    lines.append(f"mode             : {mode}")
    lines.append(
        f"current          : model_id={report.current_model_id} "
        f"dim={report.current_dim}",
    )
    lines.append(
        f"target           : model_id={report.new_model_id} "
        f"dim={report.new_dim} normalized={report.normalized}",
    )
    if report.error:
        lines.append("")
        lines.append(f"ERROR: {report.error}")
        return "\n".join(lines)
    if report.scopes_with_new_dir:
        lines.append(
            f"new dir prepared : {len(report.scopes_with_new_dir)} scope(s)",
        )
    if report.backup_path:
        lines.append(f"manifest backup  : {report.backup_path}")
    if not report.applied:
        lines.append("")
        lines.append(
            "(dry-run; rerun with --apply to swap manifest.embedding)",
        )
    return "\n".join(lines)


__all__ = [
    "MigrateEmbeddingReport",
    "format_report_text",
    "run_migrate_embedding",
]
