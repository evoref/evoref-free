"""ローカル状態 (state.json) の永続化層

EvorefMem 統合仕様 で追加される `local/state.json` を扱う
プロジェクト ID のキャッシュ・モード保持・alias 上書き・最終アクセス時刻
追跡を一元化する。

ファイル例 (`local/state.json`)::

    {
      "schema_version": 1,
      "current_project_id": "git_abc123def456",
      "mode": "coding",
      "project_aliases": {
        "https://github.com/owner/repo": "git_abc123def456"
      },
      "projects": {
        "git_abc123def456": {
          "project_id": "git_abc123def456",
          "path": "/path/to/project",
          "remote": "https://github.com/owner/repo",
          "first_seen_at": 1712627200.0,
          "last_accessed_at": 1712713600.0,
          "archived": false
        }
      }
    }

設計原則 (CLAUDE.md / .claude/rules/backend.md):
- 純粋関数 (`serialize` / `deserialize`) と I/O (`load` / `save`) を分離
- 書き込みはアトミック (一時ファイル + replace) でクラッシュ耐性
- 後方互換不要
- 180 日無アクセスのアーカイブ提案は **提案のみ**
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.io import AtomicWriter
from backend.log_config import get_logger

logger = get_logger("memory.local_state_store")


SCHEMA_VERSION = 1
DEFAULT_MODE: "MemoryMode" = "chat"
DEFAULT_INACTIVE_DAYS = 180

MemoryMode = Literal["chat", "coding"]


@dataclass
class ProjectMeta:
    """1 プロジェクトのメタ情報"""

    project_id: str
    path: str | None = None
    remote: str | None = None
    first_seen_at: float = 0.0
    last_accessed_at: float = 0.0
    archived: bool = False


@dataclass
class LocalState:
    """`local/state.json` の in-memory 表現"""

    schema_version: int = SCHEMA_VERSION
    current_project_id: str | None = None
    mode: MemoryMode = DEFAULT_MODE
    project_aliases: dict[str, str] = field(default_factory=dict)
    projects: dict[str, ProjectMeta] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# シリアライズ / デシリアライズ (純粋)
# ──────────────────────────────────────────────────────────────────────────


def serialize(state: LocalState) -> dict[str, Any]:
    """`LocalState` を JSON-serializable な dict にする純粋関数"""
    return {
        "schema_version": state.schema_version,
        "current_project_id": state.current_project_id,
        "mode": state.mode,
        "project_aliases": dict(state.project_aliases),
        "projects": {pid: asdict(meta) for pid, meta in state.projects.items()},
    }


def deserialize(data: dict[str, Any] | None) -> LocalState:
    """JSON dict から `LocalState` を再構築する純粋関数。

    壊れたフィールドは既定値で埋める (起動時に state.json を破壊しない)。
    schema_version が想定外なら警告ログを出すが例外は投げない。
    """
    if not isinstance(data, dict):
        return LocalState()

    raw_version = data.get("schema_version")
    version = raw_version if isinstance(raw_version, int) else SCHEMA_VERSION
    if version != SCHEMA_VERSION:
        logger.warning(
            "local state schema_version mismatch: expected=%s actual=%s. "
            "Loading with default fields.",
            SCHEMA_VERSION, version,
        )

    mode_raw = data.get("mode")
    mode: MemoryMode = mode_raw if mode_raw in ("chat", "coding") else DEFAULT_MODE

    current = data.get("current_project_id")
    current_pid = current if isinstance(current, str) and current else None

    aliases_raw = data.get("project_aliases") or {}
    aliases: dict[str, str] = {}
    if isinstance(aliases_raw, dict):
        for k, v in aliases_raw.items():
            if isinstance(k, str) and isinstance(v, str):
                aliases[k] = v

    projects: dict[str, ProjectMeta] = {}
    projects_raw = data.get("projects") or {}
    if isinstance(projects_raw, dict):
        for pid, meta_raw in projects_raw.items():
            if not isinstance(pid, str) or not isinstance(meta_raw, dict):
                continue
            try:
                meta = ProjectMeta(
                    project_id=str(meta_raw.get("project_id") or pid),
                    path=meta_raw.get("path"),
                    remote=meta_raw.get("remote"),
                    first_seen_at=float(meta_raw.get("first_seen_at") or 0.0),
                    last_accessed_at=float(meta_raw.get("last_accessed_at") or 0.0),
                    archived=bool(meta_raw.get("archived", False)),
                )
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed project meta %s: %s", pid, exc)
                continue
            projects[pid] = meta

    return LocalState(
        schema_version=SCHEMA_VERSION,
        current_project_id=current_pid,
        mode=mode,
        project_aliases=aliases,
        projects=projects,
    )


# ──────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────


class LocalStateStore:
    """`local/state.json` の純粋永続化担当"""

    @staticmethod
    def load(path: Path) -> LocalState:
        """state.json を読み込む。存在しなければ既定値で返す"""
        if not path.exists():
            return LocalState()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read local state %s: %s. Falling back to defaults.",
                path, exc,
            )
            return LocalState()
        return deserialize(raw)

    @staticmethod
    def save(path: Path, state: LocalState) -> None:
        """state.json をアトミックに書き出す (:class:`AtomicWriter` 委譲)。

        親ディレクトリは自動作成。書き込み失敗時は tmp ファイルが除去される
        (詳細は :mod:`backend.io.atomic` 参照)。Windows ``PermissionError``
        は :mod:`backend.io._retry` の retry ポリシーで吸収される。
        """
        payload = json.dumps(
            serialize(state), ensure_ascii=False, indent=2, sort_keys=True,
        )
        with AtomicWriter(path) as f:
            f.write(payload)
        logger.debug("Saved local state: %s", path)


# ──────────────────────────────────────────────────────────────────────────
# 補助操作 (純粋関数)
# ──────────────────────────────────────────────────────────────────────────


def touch_project(
    state: LocalState,
    project_id: str,
    *,
    path: str | None = None,
    remote: str | None = None,
    now: float | None = None,
) -> ProjectMeta:
    """プロジェクトを登録 / 既存なら ``last_accessed_at`` を更新する。

    `state` を破壊的に更新する (戻り値は更新後の `ProjectMeta`)。
    アーカイブ済プロジェクトに再アクセスした場合は ``archived=False`` に戻す
    (ユーザーが再開した扱い)。
    """
    if not project_id:
        raise ValueError("project_id must be non-empty")
    ts = time.time() if now is None else now
    meta = state.projects.get(project_id)
    if meta is None:
        meta = ProjectMeta(
            project_id=project_id,
            path=path,
            remote=remote,
            first_seen_at=ts,
            last_accessed_at=ts,
            archived=False,
        )
        state.projects[project_id] = meta
    else:
        if path is not None:
            meta.path = path
        if remote is not None:
            meta.remote = remote
        meta.last_accessed_at = ts
        if meta.archived:
            meta.archived = False
    return meta


def set_current_project(
    state: LocalState,
    project_id: str,
    *,
    mode: MemoryMode | None = None,
) -> None:
    """`current_project_id` を更新し、必要なら `mode` も切替える。

    project_id に対応する `ProjectMeta` が無ければ何もしない (登録は
    `touch_project` 経由を強制し、二重管理を防ぐ)。
    """
    if project_id not in state.projects:
        raise KeyError(f"unknown project_id: {project_id} (call touch_project first)")
    state.current_project_id = project_id
    if mode is not None:
        state.mode = mode


def add_alias(state: LocalState, alias_key: str, project_id: str) -> None:
    """alias を登録 / 上書きする。存在しない project_id を指す alias は禁止"""
    if not alias_key:
        raise ValueError("alias_key must be non-empty")
    if project_id not in state.projects:
        raise KeyError(f"unknown project_id: {project_id}")
    state.project_aliases[alias_key] = project_id


def remove_alias(state: LocalState, alias_key: str) -> bool:
    """alias を削除する。削除した場合 True"""
    return state.project_aliases.pop(alias_key, None) is not None


def archive_project(
    state: LocalState,
    project_id: str,
    *,
    semantic_root: Path,
    archive_dir: Path,
    now: float | None = None,  # noqa: ARG001
) -> Path | None:
    """指定プロジェクトを物理アーカイブする

    `semantic_root/projects/<project_id>/` 以下を `archive_dir/<project_id>/`
    へ移動し、`state.projects[project_id].archived = True` を設定する。
    実体ディレクトリが存在しない場合は state のフラグだけ更新する
    (履歴のみのプロジェクトを許容)。

    既に archive_dir 側に同名ディレクトリが存在する場合は、移動せず
    state フラグだけ更新する (再アーカイブ抑止 / 上書き防止)。

    Returns:
        実際にファイルを移動した場合は移動先パス、移動しなかった場合は None。
        プロジェクトが state に存在しない場合は KeyError。
    """
    import shutil

    meta = state.projects.get(project_id)
    if meta is None:
        raise KeyError(f"unknown project_id: {project_id}")
    if state.current_project_id == project_id:
        raise ValueError(
            f"cannot archive currently active project: {project_id}",
        )

    src = Path(semantic_root) / "projects" / project_id
    dst = Path(archive_dir) / project_id
    moved_to: Path | None = None
    if src.exists() and src.is_dir():
        if dst.exists():
            logger.warning(
                "Archive destination already exists, leaving source intact: %s",
                dst,
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved_to = dst
            logger.info("Archived project semantic dir: %s -> %s", src, dst)

    meta.archived = True
    return moved_to


def propose_archives(
    state: LocalState,
    *,
    threshold_days: int = DEFAULT_INACTIVE_DAYS,
    now: float | None = None,
) -> list[str]:
    """`threshold_days` 以上アクセスのないプロジェクト ID を返す。

    実アーカイブは 実装する。本関数は提案リストのみ返し
    state は変更しない。すでに ``archived=True`` のものは除外する
    (再提案を防ぐ)。`current_project_id` も除外する。
    """
    if threshold_days <= 0:
        return []
    ts = time.time() if now is None else now
    cutoff = ts - threshold_days * 86400
    proposals: list[str] = []
    for pid, meta in state.projects.items():
        if meta.archived:
            continue
        if pid == state.current_project_id:
            continue
        if meta.last_accessed_at and meta.last_accessed_at < cutoff:
            proposals.append(pid)
    proposals.sort()
    return proposals
