"""``evorefmem_cli purge-private`` 実装

private セッション由来のまま SemMem に残ったキュレーターファクトを掃除する。

## なぜ必要か

Step 8.4 / 8.5 / 8.6 の 3 キュレーターは ``note.private`` を見ずに
``world_fact`` を書いていた (2026-09-01 監査 F2)。しかも ``make_fact()`` を
素で呼んでいたため ``provenances=[]`` / ``session_ids=set()`` で、生成された
ファクトは ``private=False``。注入除外にもリコール除外にも掛からず、
**後続の通常セッションから引き当てられる**。

書込側は :func:`~backend.free.memory.note_facts.fact_from_note` へ寄せて
privacy を継承するようにしたが、**それは今後書かれるファクトの話**。既に
書かれた行はこのコマンドで消す。

## 何を根拠に選ぶか

3 系統で追跡情報の残り方が違う:

======================================  ==========================================
subject 接頭辞                           使える手掛かり
======================================  ==========================================
``mem.world.assertion.*``               ``_extra.source_note_id`` → STM ノート →
                                        ``private`` (**厳密**)
``mem.world.executable_command.*``      無し (``last_query`` / ``mode`` のみ)
``mem.world.url.*``                     無し (``url`` / ``last_query`` のみ)
======================================  ==========================================

したがって:

- ``--strict`` (既定): ノートに解決できて ``private=True`` のものだけ。
  取りこぼしは残るが誤削除しない。
- ``--all-curated``: 3 系統を **まるごと** 候補にする。取りこぼしゼロだが
  正当な索引も一度消える。

## まるごと消してよい理由

この 3 系統は **再生成可能な索引** であって、ユーザーの言明ではない。正当な
ものはノート側の冪等マーカー (``assertion_curated_at`` / ``command_curated_at``
/ ``url_curated_at``) を戻せば次の Full で作り直される。したがって
``--all-curated`` の実コストは ``exec_count`` / ``success_history`` /
``score_history`` の統計を失うことだけ。

**マーカーのリセットは非 private ノートに限る** — private ノートのマーカーを
戻すと、次の Full で同じものが再生成されてしまう (今は ``public_notes`` が
入口で落とすので実際には作られないが、二重防御として明示的に除外する)。

破壊的なので:

- 既定で dry-run (``--apply`` 必須)
- ``--apply`` 時は ``facts.jsonl`` と ``short_term_notes.json`` を
  ``migration_archive/cli_<utc_ts>/purge_private/`` へ退避してから書き換える
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
from backend.free.memory.semantic.store import FACTS_FILENAME, SemanticFactStore
from backend.log_config import get_logger

logger = get_logger("memory.semantic.cli.purge_private")

#: キュレーター由来の world_fact の subject 接頭辞 (再生成可能な索引)。
CURATED_SUBJECT_PREFIXES: tuple[str, ...] = (
    "mem.world.assertion.",
    "mem.world.executable_command.",
    "mem.world.url.",
)

#: 接頭辞 → その系統を作るキュレーターの冪等マーカー (MemoryNote の属性名)。
_MARKER_BY_PREFIX: dict[str, str] = {
    "mem.world.assertion.": "assertion_curated_at",
    "mem.world.executable_command.": "command_curated_at",
    "mem.world.url.": "url_curated_at",
}

#: STM ノートの永続化ファイル名 (``local/memory/`` 直下)。
NOTES_FILENAME = "short_term_notes.json"


@dataclass
class PurgeCandidate:
    """削除候補 1 件."""

    fact_id: str
    scope: str
    subject: str
    reason: str
    """``private_note`` (厳密照合) / ``curated_index`` (--all-curated) /
    ``time_window`` / ``session``。"""

    object_preview: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PurgeReport:
    memory_dir: str
    applied: bool
    mode: str
    """``strict`` または ``all_curated``。"""

    candidates: list[PurgeCandidate] = field(default_factory=list)
    deleted: int = 0
    notes_unmarked: int = 0
    """再生成のためにマーカーを戻した (非 private の) ノート数。"""

    notes_available: bool = True
    """STM ノートを読めたか。``False`` なら厳密照合は成立しない。"""

    backup_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_dir": self.memory_dir,
            "applied": self.applied,
            "mode": self.mode,
            "notes_available": self.notes_available,
            "candidates": [c.to_dict() for c in self.candidates],
            "totals": {
                "candidates": len(self.candidates),
                "deleted": self.deleted,
                "notes_unmarked": self.notes_unmarked,
            },
            "backup_path": self.backup_path,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ──────────────────────────────────────────────────────────────────────────
# 内部
# ──────────────────────────────────────────────────────────────────────────


def _notes_path(memory_dir: Path) -> Path:
    return Path(memory_dir) / NOTES_FILENAME


def _load_notes(memory_dir: Path) -> tuple[dict[str, dict], bool]:
    """``{note_id: note_dict}`` と読み込み成否を返す。"""
    path = _notes_path(memory_dir)
    if not path.exists():
        return {}, False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("purge-private: failed to read %s: %s", path, exc)
        return {}, False
    notes = raw.get("notes", raw) if isinstance(raw, dict) else raw
    if isinstance(notes, dict):
        entries = list(notes.values())
    elif isinstance(notes, list):
        entries = notes
    else:
        return {}, False
    return {n.get("id"): n for n in entries if isinstance(n, dict) and n.get("id")}, True


def _is_curated(subject: str) -> bool:
    return subject.startswith(CURATED_SUBJECT_PREFIXES)


def _marker_for(subject: str) -> str | None:
    for prefix, marker in _MARKER_BY_PREFIX.items():
        if subject.startswith(prefix):
            return marker
    return None


def _select(
    store: SemanticFactStore,
    scope: str,
    notes: dict[str, dict],
    *,
    all_curated: bool,
    since: float | None,
    until: float | None,
    session_ids: set[str],
) -> list[PurgeCandidate]:
    """1 scope 分の削除候補を選ぶ (副作用なし)。"""
    out: list[PurgeCandidate] = []
    for fact in store.all_facts(include_superseded=True):
        subject = fact.subject or ""
        if not _is_curated(subject):
            continue

        reason: str | None = None

        # (1) 厳密照合 — assertion のみ source_note_id を持つ。
        note_id = (fact._extra or {}).get("source_note_id")
        if not note_id:
            for prov in fact.provenances or ():
                if prov.note_id:
                    note_id = prov.note_id
                    break
        if note_id and notes.get(note_id, {}).get("private"):
            reason = "private_note"

        # (2) セッション指定。
        if reason is None and session_ids:
            fact_sessions = set(fact.session_ids or ())
            for prov in fact.provenances or ():
                if prov.session_id:
                    fact_sessions.add(prov.session_id)
            if fact_sessions & session_ids:
                reason = "session"

        # (3) 時間窓。
        if reason is None and (since is not None or until is not None):
            created = float(fact.created_at or 0.0)
            lo_ok = since is None or created >= since
            hi_ok = until is None or created <= until
            if lo_ok and hi_ok:
                reason = "time_window"

        # (4) まるごと。
        if reason is None and all_curated:
            reason = "curated_index"

        if reason is None:
            continue
        out.append(PurgeCandidate(
            fact_id=fact.id,
            scope=scope,
            subject=subject,
            reason=reason,
            object_preview=(fact.object or "")[:80],
            created_at=float(fact.created_at or 0.0),
        ))
    out.sort(key=lambda c: (c.scope, c.subject, c.created_at))
    return out


def _unmark_notes(
    memory_dir: Path,
    purged_subjects: set[str],
    *,
    apply: bool,
) -> int:
    """再生成のため、非 private ノートのキュレーションマーカーを戻す。

    private ノートは対象外 — マーカーを戻すと次の Full で同じものが再生成
    されうる (現在は ``public_notes`` が入口で落とすが、二重防御として
    ここでも明示的に除外する)。
    """
    markers = {
        _marker_for(subject) for subject in purged_subjects
    } - {None}
    if not markers:
        return 0
    path = _notes_path(memory_dir)
    if not path.exists():
        return 0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0

    container = raw.get("notes", raw) if isinstance(raw, dict) else raw
    entries = (
        list(container.values()) if isinstance(container, dict)
        else container if isinstance(container, list) else []
    )
    changed = 0
    for note in entries:
        if not isinstance(note, dict) or note.get("private"):
            continue
        for marker in markers:
            if note.get(marker) is not None:
                note[marker] = None
                changed += 1
                break
    if changed and apply:
        from backend.io import atomic_write_text

        atomic_write_text(path, json.dumps(raw, ensure_ascii=False, indent=2))
    return changed


def _backup(
    scopes: list[ScopeInfo], memory_dir: Path, archive_root: Path,
) -> Path:
    dest = cli_backup_root(archive_root, "purge_private")
    for scope in scopes:
        facts = scope.root_dir / FACTS_FILENAME
        if facts.exists():
            target = dest / scope.name.replace(":", "_") / FACTS_FILENAME
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(facts, target)
    notes = _notes_path(memory_dir)
    if notes.exists():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(notes, dest / NOTES_FILENAME)
    return dest


# ──────────────────────────────────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────────────────────────────────


def run_purge_private(
    memory_dir: Path,
    archive_root: Path,
    *,
    apply: bool = False,
    all_curated: bool = False,
    scope_filter: str | None = None,
    since: float | None = None,
    until: float | None = None,
    sessions: list[str] | None = None,
) -> PurgeReport:
    """private 由来のキュレーターファクトを掃除する。

    Args:
        memory_dir: ``local/memory/`` ルート。
        archive_root: ``local/migration_archive/`` ルート (退避先)。
        apply: ``True`` で実際に削除する。既定は dry-run。
        all_curated: 3 系統をまるごと候補にする (取りこぼしゼロ / 統計は失う)。
        scope_filter: ``"global"`` 等、特定 scope に限定。
        since / until: ``created_at`` の窓 (epoch 秒)。
        sessions: この session_id 由来のものを候補にする。

    Returns:
        :class:`PurgeReport`。``apply=False`` なら ``deleted == 0``。
    """
    memory_dir = Path(memory_dir)
    report = PurgeReport(
        memory_dir=str(memory_dir),
        applied=apply,
        mode="all_curated" if all_curated else "strict",
    )
    notes, ok = _load_notes(memory_dir)
    report.notes_available = ok

    scopes = [
        s for s in enumerate_scopes(memory_dir)
        if scope_filter is None or s.name == scope_filter
    ]
    session_ids = set(sessions or ())

    stores: dict[str, SemanticFactStore] = {}
    for scope in scopes:
        try:
            store = SemanticFactStore(scope.root_dir)
        except Exception as exc:
            logger.warning(
                "purge-private: failed to open %s: %s", scope.name, exc,
            )
            continue
        stores[scope.name] = store
        report.candidates.extend(
            _select(
                store, scope.name, notes,
                all_curated=all_curated,
                since=since, until=until, session_ids=session_ids,
            ),
        )

    purged_subjects = {c.subject for c in report.candidates}
    if apply and report.candidates:
        report.backup_path = str(_backup(scopes, memory_dir, archive_root))
        by_scope: dict[str, list[str]] = {}
        for c in report.candidates:
            by_scope.setdefault(c.scope, []).append(c.fact_id)
        for scope_name, ids in by_scope.items():
            store = stores.get(scope_name)
            if store is None:
                continue
            try:
                report.deleted += store.delete_facts(ids)
            except Exception as exc:
                logger.warning(
                    "purge-private: delete failed in %s: %s", scope_name, exc,
                )
    report.notes_unmarked = _unmark_notes(
        memory_dir, purged_subjects, apply=apply,
    )
    if apply:
        logger.info(
            "purge-private: deleted %d fact(s), unmarked %d note(s)",
            report.deleted, report.notes_unmarked,
        )
    return report


__all__ = [
    "CURATED_SUBJECT_PREFIXES",
    "PurgeCandidate",
    "PurgeReport",
    "run_purge_private",
]
