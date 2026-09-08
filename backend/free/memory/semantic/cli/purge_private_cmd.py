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
``mem.world.assertion.*``               ``_extra.source_note_id`` → ノート →
                                        ``private`` (**厳密**)
``idx.command.*``                       無し (``last_query`` / ``mode`` のみ)
``idx.url.*``                           無し (``url`` / ``last_query`` のみ)
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
- ``--apply`` 時は SemMem の事象ログを
  ``migration_archive/cli_<utc_ts>/purge_private/`` へ退避してから取り下げる
  (レコードは物理削除ではなく ``veracity=retracted``、c_16 §3)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.free.memory.episodic.note import MemoryNote
from backend.free.memory.episodic.store import EpisodicStore
from backend.free.memory.semantic.cli._paths import (
    cli_backup_root,
    open_semantic_store,
    scope_names,
)
from backend.free.memory.sleep._curator_common import (
    EXECUTABLE_COMMAND_SUBJECT_PREFIX,
    URL_SUBJECT_PREFIX,
)
from backend.free.memory.semantic.store import SemanticStore
from backend.log_config import get_logger

logger = get_logger("memory.semantic.cli.purge_private")

#: キュレーター由来のファクトの subject 接頭辞 (再生成可能な索引)。
#: URL / コマンドは ``idx.*`` namespace (c_16 §4.2)、assertion は言明なので
#: ``mem.world.*`` のまま。接頭辞の SSOT は ``sleep._curator_common``。
ASSERTION_SUBJECT_PREFIX = "mem.world.assertion."

CURATED_SUBJECT_PREFIXES: tuple[str, ...] = (
    ASSERTION_SUBJECT_PREFIX,
    EXECUTABLE_COMMAND_SUBJECT_PREFIX,
    URL_SUBJECT_PREFIX,
)

#: 接頭辞 → その系統を作るキュレーターの冪等マーカー (MemoryNote の属性名)。
_MARKER_BY_PREFIX: dict[str, str] = {
    ASSERTION_SUBJECT_PREFIX: "assertion_curated_at",
    EXECUTABLE_COMMAND_SUBJECT_PREFIX: "command_curated_at",
    URL_SUBJECT_PREFIX: "url_curated_at",
}


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
    """エピソード記憶を読めたか。``False`` なら厳密照合は成立しない。"""

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


def _open_episodic(memory_dir: Path) -> "EpisodicStore | None":
    """エピソード記憶を読み取り専用で開く (失敗は ``None``)。"""
    try:
        store = EpisodicStore(Path(memory_dir))
        store.load()
    except Exception as exc:  # noqa: BLE001 — 読めなくても strict 判定を諦めるだけ
        logger.warning("purge-private: failed to open the episodic store: %s", exc)
        return None
    return store


def _load_notes(
    episodic: "EpisodicStore | None",
) -> tuple[dict[str, MemoryNote], bool]:
    """``{note_id: MemoryNote}`` と読み込み成否を返す。

    ノートの永続形は ``Evidence`` 1 本になったので、旧
    ``short_term_notes.json`` は読まない (c_16 §4.1)。private ノートも
    含めて引く — private かどうかがまさに判定材料。
    """
    if episodic is None:
        return {}, False
    notes = {
        note.id: note
        for note in episodic.iter_notes(include_private=True)
    }
    return notes, True


def _is_curated(subject: str) -> bool:
    return subject.startswith(CURATED_SUBJECT_PREFIXES)


def _marker_for(subject: str) -> str | None:
    for prefix, marker in _MARKER_BY_PREFIX.items():
        if subject.startswith(prefix):
            return marker
    return None


def _select(
    store: SemanticStore,
    scope: str,
    notes: dict[str, MemoryNote],
    *,
    all_curated: bool,
    since: float | None,
    until: float | None,
    session_ids: set[str],
) -> list[PurgeCandidate]:
    """1 scope 分の削除候補を選ぶ (副作用なし)。"""
    out: list[PurgeCandidate] = []
    for fact in store.all_facts(include_superseded=True, scope=scope):
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
        note = notes.get(note_id) if note_id else None
        if note is not None and note.private:
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
    episodic: "EpisodicStore | None",
    notes: dict[str, MemoryNote],
    purged_subjects: set[str],
    *,
    apply: bool,
) -> int:
    """再生成のため、非 private ノートのキュレーションマーカーを戻す。

    private ノートは対象外 — マーカーを戻すと次の Full で同じものが再生成
    されうる (現在は ``public_notes`` が入口で落とすが、二重防御として
    ここでも明示的に除外する)。

    書き込みは ``patch`` 事象 1 件ずつ (c_16 §5.2)。snapshot は sleep-time が
    作るので、ここでは版を積まない。
    """
    markers = {
        marker for marker in (_marker_for(s) for s in purged_subjects)
        if marker is not None
    }
    if not markers or episodic is None:
        return 0
    changed = 0
    for note in notes.values():
        if note.private:
            continue
        hit = next(
            (m for m in markers if getattr(note, m, None) is not None), None,
        )
        if hit is None:
            continue
        changed += 1
        if apply:
            episodic.patch_note(note.id, attrs={**_note_attrs(note, hit)})
    return changed


def _note_attrs(note: MemoryNote, marker: str) -> dict[str, Any]:
    """``marker`` を ``None`` へ戻した ``attrs`` を組む (他のキーはそのまま)。

    ``patch`` の ``attrs`` は **マージ** される (``snapshot.apply_patch``) ので、
    キーを ``pop`` しても消えない。既定値 (``None``) を明示的に書くこと —
    ``evidence_to_note`` はそれをそのまま属性へ写すので、次の Full で
    キュレーターが再生成できる状態に戻る。
    """
    from backend.free.memory.episodic.note import note_to_evidence

    attrs = dict(note_to_evidence(note).attrs)
    attrs[marker] = None
    return attrs


def _backup(memory_dir: Path, archive_root: Path) -> Path:
    """SemMem の事象ログを退避先へコピーする。"""
    dest = cli_backup_root(archive_root, "purge_private")
    events = Path(memory_dir) / "semantic" / "events"
    if events.exists():
        shutil.copytree(events, dest / "events", dirs_exist_ok=True)
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
    episodic = _open_episodic(memory_dir)
    notes, ok = _load_notes(episodic)
    report.notes_available = ok

    store = open_semantic_store(memory_dir)
    session_ids = set(sessions or ())
    for scope in scope_names(store):
        if scope_filter is not None and scope != scope_filter:
            continue
        report.candidates.extend(
            _select(
                store, scope, notes,
                all_curated=all_curated,
                since=since, until=until, session_ids=session_ids,
            ),
        )

    purged_subjects = {c.subject for c in report.candidates}
    if apply and report.candidates:
        report.backup_path = str(_backup(memory_dir, archive_root))
        try:
            report.deleted += store.delete_facts(
                [c.fact_id for c in report.candidates],
            )
        except Exception as exc:  # noqa: BLE001 — 1 件の失敗で全体を止めない
            logger.warning("purge-private: retract failed: %s", exc)
    report.notes_unmarked = _unmark_notes(
        episodic, notes, purged_subjects, apply=apply,
    )
    store.close()
    if apply:
        logger.info(
            "purge-private: retracted %d fact(s), unmarked %d note(s)",
            report.deleted, report.notes_unmarked,
        )
    return report


__all__ = [
    "ASSERTION_SUBJECT_PREFIX",
    "CURATED_SUBJECT_PREFIXES",
    "PurgeCandidate",
    "PurgeReport",
    "run_purge_private",
]
