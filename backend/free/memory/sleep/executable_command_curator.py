"""Step 8.6: executable command リコール用のキュレーター

ユーザの直近セッションで ``run_command`` が実行されたターンについて、
コマンド文字列と成否を ``world_fact`` (subject =
``mem.world.executable_command.*``) として SemMem に永続化する。次回類似
クエリで ``ToolCallJudge`` がアシスト呼出より先に引き当てる (読み取りは A3)。

CLAUDE.md §6 不変則 #2 より、SemMem への書込は sleep-time に限定される。
本モジュールは ``SleepTimeWorker.run_full`` の Step 8.6 として呼び出され、
チャット応答パスでは利用しない。

設計ポリシー (url_curator と対称):

- 新 FactType を追加せず ``world_fact`` を流用する (CLAUDE.md §3 / §6 #2)。
- subject = ``mem.world.executable_command.<mode>.<sha1_12(command_normalized)>``。
  同一コマンドの決定論的キーで既存 fact を引き当て可能。
- ``_extra`` にコマンド専用メタ (command / command_normalized / mode /
  success_history / success_avg / exec_count / last_query /
  last_executed_at) を載せる。SemanticFact の round-trip でそのまま JSONL に
  保持される。
- url_curator と違い **アシスト採点はしない**。``MemoryNote.tool_command_success``
  (run_command 戻り値が "Error:" prefix でないか) を真偽として記録する。
  ``success=False`` は既存 fact を penalize するのみで新規作成しない。

データ源は PR-A1 で追加した ``MemoryNote.tool_command`` /
``tool_command_name`` / ``tool_command_success`` (assistant note に載る)。
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.memory.types import SemanticFact, make_fact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.stores.short_term import MemoryNote
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.sleep.executable_command_curator")

_SUBJECT_PREFIX = "mem.world.executable_command."
_WS_RE = re.compile(r"\s+")


def _normalize_command(command: str) -> str:
    """コマンド文字列を正規化する (空白圧縮 + lower)。

    パスや引数の大小は意味を持ち得るが、subject キーの安定化を優先して
    lower + 連続空白の単一化のみ行う (過度な正規化は別コマンドの衝突を招く)。
    """
    return _WS_RE.sub(" ", command).strip().lower()


def _make_subject(mode: str, command_normalized: str) -> str:
    """executable command リコール用 subject を構築する。"""
    digest = hashlib.sha1(command_normalized.encode("utf-8")).hexdigest()[:12]
    safe_mode = mode if mode in ("chat", "coding") else "chat"
    return f"{_SUBJECT_PREFIX}{safe_mode}.{digest}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _iter_command_pairs(
    notes: list["MemoryNote"],
) -> list[tuple["MemoryNote", "MemoryNote"]]:
    """``run_command`` を実行した assistant note と直前 user note の組を返す。

    同一 session 内で ``created_at`` 昇順にソートし、``tool_command`` を持つ
    assistant note について、その直前で最も近い ``source=="user"`` note を
    query として対にする。直前 user note が無い assistant note は除外する。
    """
    by_session: dict[str, list["MemoryNote"]] = {}
    for note in notes:
        by_session.setdefault(note.session_id or "_", []).append(note)
    pairs: list[tuple["MemoryNote", "MemoryNote"]] = []
    for sess_notes in by_session.values():
        ordered = sorted(sess_notes, key=lambda n: n.created_at)
        for idx, note in enumerate(ordered):
            if not note.tool_command:
                continue
            if note.tool_command_name != "run_command":
                continue
            # 直前で最も近い user note を探す
            user_note: "MemoryNote | None" = None
            for before in reversed(ordered[:idx]):
                if before.source == "user":
                    user_note = before
                    break
            if user_note is None:
                continue
            pairs.append((user_note, note))
    return pairs


def _existing_command_fact(
    store: "SemanticFactStore", subject: str,
) -> SemanticFact | None:
    """同一 subject の既存 executable command fact を探す。"""
    for fact in store.search_by_subject(subject):
        if fact.type == "world_fact":
            return fact
    return None


def _record_success(
    fact: SemanticFact,
    success: bool,
    query: str,
    *,
    history_size: int,
    now: float,
) -> dict[str, Any]:
    """既存 fact の ``_extra`` を成否 1 件分だけ更新した dict を返す。"""
    extra = dict(fact._extra) if fact._extra else {}
    history = list(extra.get("success_history") or [])
    history.append(1.0 if success else 0.0)
    if len(history) > history_size:
        history = history[-history_size:]
    exec_count = int(extra.get("exec_count") or 0) + 1
    success_avg = sum(history) / len(history) if history else 0.0
    extra.update(
        {
            "exec_count": exec_count,
            "success_history": history,
            "success_avg": round(success_avg, 4),
            "last_query": _truncate(query or "", 200),
            "last_executed_at": now,
        },
    )
    return extra


async def curate_executable_command_facts(
    notes: list["MemoryNote"],
    *,
    config: dict | None,
    store_provider: Callable[[str], "SemanticFactStore | None"] | None,
    embedder: "EmbeddingBackend | None",
    profile_id: str = "default",
    debug_logger=None,
    now_provider: Callable[[], float] | None = None,
) -> int:
    """executable command リコール用の world_fact を sleep-time で書き込む。

    Args:
        notes: 直近の MemoryNote 群 (通常 ``ShortTermMemory.notes.values()``)。
        config: 全体 config。``tools.executable_command_recall_*`` を参照する。
        store_provider: ``scope -> SemanticFactStore | None`` のコールバック。
        embedder: query topic embedding 生成用。``None`` で no-op。
        profile_id: 書込先 fact の profile_id。
        debug_logger: 任意の DebugLogger。
        now_provider: 時刻供給。テスト用。

    Returns:
        新規に書き込まれた / 更新された command fact 件数。
    """
    if embedder is None:
        logger.debug("executable_command_curator: embedder is None, skipping")
        return 0
    if store_provider is None:
        logger.debug(
            "executable_command_curator: store_provider is None, skipping",
        )
        return 0
    cfg = (config or {}).get("tools") or {}
    if not bool(cfg.get("executable_command_recall_enabled", True)):
        logger.debug(
            "executable_command_curator: recall disabled, skipping",
        )
        return 0
    history_size = int(cfg.get("executable_command_recall_record_history_size", 10))

    pairs = _iter_command_pairs(notes)
    if not pairs:
        return 0
    store = store_provider("global")
    if store is None:
        logger.debug(
            "executable_command_curator: global store not available, skipping",
        )
        return 0

    now_fn = now_provider or time.time
    written = 0
    seen_subjects: set[str] = set()
    for user_note, assistant_note in pairs:
        command = assistant_note.tool_command or ""
        if not command:
            continue
        mode = assistant_note.mode or "chat"
        normalized = _normalize_command(command)
        if not normalized:
            continue
        subject = _make_subject(mode, normalized)
        if subject in seen_subjects:
            continue
        seen_subjects.add(subject)
        success = bool(assistant_note.tool_command_success)
        topic = _truncate((user_note.content or "").strip(), 200)
        now = now_fn()

        existing = _existing_command_fact(store, subject)
        try:
            if existing is not None:
                new_extra = _record_success(
                    existing, success, user_note.content,
                    history_size=history_size, now=now,
                )
                new_avg = float(new_extra.get("success_avg") or 0.0)
                store.update_fact(
                    existing.id, _extra=new_extra, confidence=new_avg,
                )
                written += 1
            elif success:
                # 新規作成は成功時のみ (失敗コマンドを学習対象にしない)
                embedding = None
                try:
                    emb = await embedder.embed([topic], is_query=False)
                    if emb is not None and len(emb) > 0:
                        embedding = np.asarray(emb[0], dtype=np.float32)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "executable_command_curator: embed failed: %s", exc,
                    )
                extra = {
                    "command": command,
                    "command_normalized": normalized,
                    "mode": mode,
                    "exec_count": 1,
                    "success_history": [1.0],
                    "success_avg": 1.0,
                    "last_query": topic,
                    "last_executed_at": now,
                }
                fact = make_fact(
                    subject=subject,
                    predicate="answers_query",
                    object_=topic,
                    type="world_fact",
                    scope="global",
                    confidence=1.0,
                    now=now,
                    profile_id=profile_id,
                    embedding=embedding,
                    _extra=extra,
                )
                store.add_fact(fact)
                written += 1
            else:
                # 既存 fact が無く success=False → 学習しない
                logger.debug(
                    "executable_command_curator: skip failed command with no "
                    "existing fact (mode=%s)", mode,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "executable_command_curator: persist failed: %s", exc,
            )

    if written and debug_logger is not None:
        try:
            debug_logger.log_memory_op(
                op="executable_command_curator",
                stats={"facts_curated": written},
            )
        except Exception:  # noqa: BLE001
            pass
    if written:
        logger.info(
            "executable_command_curator: curated %d command fact(s)", written,
        )
    return written


__all__ = ["curate_executable_command_facts"]
