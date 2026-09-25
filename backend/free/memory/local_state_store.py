"""ローカル状態 (state.json) の永続化層

EvorefMem 統合仕様 で追加される `<data_root>/g1/store/state.json` を扱う
プロジェクト ID のキャッシュ・モード保持・alias 上書き・最終アクセス時刻
追跡を一元化する。

ファイル例 (`<data_root>/g1/store/state.json` の封筒の ``payload``、形式 ``state``)::

    {
      "current_project_id": "git_abc123def456",
      "mode": "create",
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
- 書き込みは封筒付きでアトミック (:class:`backend.io.versioned.VersionedJsonFile`)。
  G1 の封筒でない / 新しい版のファイルは読まず書き戻さない (readonly)、
  壊れたファイルは ``state.json.corrupt-<stamp>`` へ退避して既定値で続ける
- 180 日無アクセスのアーカイブ提案は **提案のみ**
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.free.core.session_mode import normalize_session_mode
from backend.io.codec import CodecError, codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedJsonFile
from backend.log_config import get_logger
from backend.utils import epoch_to_utc, utc_to_epoch

logger = get_logger("memory.local_state_store")


DEFAULT_MODE: "MemoryMode" = "chat"
DEFAULT_INACTIVE_DAYS = 180

MemoryMode = Literal["chat", "create"]


@dataclass
class ProjectMeta:
    """1 プロジェクトのメタ情報"""

    project_id: str
    path: str | None = None
    remote: str | None = None
    first_seen_at: float = 0.0
    last_accessed_at: float = 0.0
    archived: bool = False
    #: 読み戻したときの、この版が知らないキー (書き戻しでそのまま戻す)。
    _extra: dict[str, Any] | None = None


@dataclass
class LocalState:
    """`<data_root>/g1/store/state.json` の in-memory 表現"""

    current_project_id: str | None = None
    mode: MemoryMode = DEFAULT_MODE
    project_aliases: dict[str, str] = field(default_factory=dict)
    projects: dict[str, ProjectMeta] = field(default_factory=dict)
    #: ディスク上のファイルが書き戻すと壊す状態 (新しい版 / G1 の封筒でない)。
    #: :meth:`LocalStateStore.save` はこの印が立っていたら書き込まない。
    readonly: bool = False
    #: payload の未知キー (書き戻しでそのまま戻す)。
    _extra: dict[str, Any] | None = None


# ──────────────────────────────────────────────────────────────────────────
# 永続形 (コーデックの表、c_05 §0.5.2)
# ──────────────────────────────────────────────────────────────────────────


@persisted()
@dataclass
class ProjectRecord:
    """``projects`` の 1 項目の永続形 (時刻は ISO 8601 UTC μs ``Z``、作業型は epoch 秒)。"""

    project_id: str = ""
    path: str | None = None
    remote: str | None = None
    first_seen_at: str | None = None
    last_accessed_at: str | None = None
    archived: bool = False
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class StatePayload:
    """``state.json`` の payload。"""

    current_project_id: str | None = None
    mode: str = DEFAULT_MODE
    project_aliases: dict[str, str] = field(default_factory=dict)
    projects: dict[str, ProjectRecord] = field(default_factory=dict)
    _extra: dict[str, Any] | None = None


_PROJECT_CODEC = codec_for(ProjectRecord)
_PAYLOAD_CODEC = codec_for(StatePayload)

STATE_FORMAT = register_format(FormatSpec(
    format_id="state",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/state.json",
    retention="rewritten in place; projects are archived, not deleted",
    records=(StatePayload,),
))


# ──────────────────────────────────────────────────────────────────────────
# シリアライズ / デシリアライズ (純粋)
# ──────────────────────────────────────────────────────────────────────────


def serialize(state: LocalState) -> dict[str, Any]:
    """`LocalState` を JSON-serializable な dict にする純粋関数"""
    return _PAYLOAD_CODEC.encode(StatePayload(
        current_project_id=state.current_project_id,
        mode=state.mode,
        project_aliases=dict(state.project_aliases),
        projects={
            pid: ProjectRecord(
                project_id=meta.project_id,
                path=meta.path,
                remote=meta.remote,
                first_seen_at=epoch_to_utc(meta.first_seen_at),
                last_accessed_at=epoch_to_utc(meta.last_accessed_at),
                archived=meta.archived,
                _extra=meta._extra,
            )
            for pid, meta in state.projects.items()
        },
        _extra=state._extra,
    ))


def deserialize(data: dict[str, Any] | None) -> LocalState:
    """JSON dict から `LocalState` を再構築する純粋関数。

    形の崩れたプロジェクトの項目だけを飛ばし、読めない時刻は未設定 (0.0) として
    読む。それ以外の型の違う値は :class:`~backend.io.codec.CodecError` (壊れた
    ファイル)。版の判定は封筒 (``format_version``) が持つので、ここでは見ない。
    """
    if not isinstance(data, dict):
        return LocalState()
    projects_raw = data.get("projects") or {}
    if not isinstance(projects_raw, dict):
        raise CodecError("StatePayload.projects: expected an object")
    payload = _PAYLOAD_CODEC.decode({k: v for k, v in data.items() if k != "projects"})

    projects: dict[str, ProjectMeta] = {}
    for pid, meta_raw in projects_raw.items():
        try:
            record = _PROJECT_CODEC.decode(meta_raw)
        except CodecError as exc:
            logger.warning("Skipping malformed project meta %s: %s", pid, exc)
            continue
        projects[pid] = ProjectMeta(
            project_id=record.project_id or pid,
            path=record.path,
            remote=record.remote,
            first_seen_at=utc_to_epoch(record.first_seen_at, 0.0),
            last_accessed_at=utc_to_epoch(record.last_accessed_at, 0.0),
            archived=record.archived,
            _extra=record._extra,
        )

    return LocalState(
        current_project_id=payload.current_project_id or None,
        mode=normalize_session_mode(payload.mode, default=DEFAULT_MODE),
        project_aliases=payload.project_aliases,
        projects=projects,
        _extra=payload._extra,
    )


# ──────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────


class _LocalStateFile(VersionedJsonFile):
    """``state.json`` の封筒 (形式 ``state``)。"""

    FORMAT = STATE_FORMAT
    RAISE_ON_SAVE_ERROR = True
    _state_logger = logger

    def __init__(self, path: Path, state: LocalState | None = None) -> None:
        super().__init__(path)
        self.state = state if state is not None else LocalState()

    def _to_payload(self) -> dict[str, Any]:
        return serialize(self.state)

    def _from_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("local state payload must be an object")
        self.state = deserialize(payload)


class LocalStateStore:
    """`<data_root>/g1/store/state.json` の純粋永続化担当"""

    @staticmethod
    def load(path: Path) -> LocalState:
        """state.json を読み込む。存在しなければ既定値で返す。

        新しい版 / G1 の封筒でないファイルは読まず ``readonly=True`` の既定値を
        返す (ファイルはそのまま残る)。壊れたファイルは退避して既定値を返す。
        """
        f = _LocalStateFile(path)
        if f.load():
            return f.state
        return LocalState(readonly=f.readonly)

    @staticmethod
    def save(path: Path, state: LocalState) -> None:
        """state.json を封筒付きでアトミックに書き出す。

        親ディレクトリは自動作成。``state.readonly`` なら書き込まない。
        書き込み失敗は送出する (呼出側の sleep-time Step 10 が握る)。
        """
        if state.readonly:
            logger.warning(
                "Skipping local state save: the on-disk file must not be "
                "overwritten (%s)", path,
            )
            return
        _LocalStateFile(path, state).save()
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
    now: float | None = None,  # noqa: ARG001
) -> None:
    """指定プロジェクトを ``archived=True`` にする。

    c_16 (2026-09-07) でスコープはディレクトリではなく ``Evidence.scope``
    フィールドになったため (§4.2)、``semantic/projects/<id>/`` という実体は
    存在しない。かつてここにあったディレクトリ移動は新規インストールでは
    src が一度も存在せず **到達しない分岐** だったので落とした。ファクトの
    退役は呼出側 (``sleep/archive.py`` の ``retire_project_facts``) が
    ``retract(reason="project_archived")`` で行う (c_16 §3)。

    Raises:
        KeyError: ``project_id`` が state に無い。
        ValueError: 現在アクティブなプロジェクトを指定した。
    """
    meta = state.projects.get(project_id)
    if meta is None:
        raise KeyError(f"unknown project_id: {project_id}")
    if state.current_project_id == project_id:
        raise ValueError(
            f"cannot archive currently active project: {project_id}",
        )

    meta.archived = True


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
