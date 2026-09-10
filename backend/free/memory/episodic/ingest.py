"""会話ターン → ``short`` ノート (sleep-time のノート生成)。

旧 ``ShortTermMemory.absorb`` の判定 (ツール出力の除外 / 揮発する計測値の
除外 / 自動 pin 検出 / 値の言い直しの拾い直し) をそのまま持ってきた。違いは
**いつ走るか** だけ — 応答パスではなく sleep-time で、入力は WM の押し出し
バッファではなく会話履歴のターン列 (c_16 §4.1)。

同じターンを二度ノートにしないための進捗は
:class:`~backend.free.memory.episodic.progress.EpisodicProgress` が持つ。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.free.core.text_quality import (
    detect_lang,
    is_query_echo,
    strip_echoed_query,
)
from backend.free.memory.episodic.note import MemoryNote
from backend.free.memory.episodic.store import EpisodicStore
from backend.free.memory.episodic.turn_source import (
    DEFAULT_SESSION_LIMIT,
    SessionTurns,
    TurnSource,
)
from backend.free.memory.notes.note_builder import (
    get_note_builder,
    restates_attribute_value,
)
from backend.free.memory.notes.pin_detector import detect_pin, get_pin_triggers_for
from backend.free.memory.volatile_values import is_volatile_measurement_report
from backend.free.rag.evidence import new_evidence_id
from backend.log_config import get_logger
from backend.utils import parse_utc

logger = get_logger("memory.episodic.ingest")


def _turn_timestamp(turn: dict) -> float:
    """ターンの時刻を epoch 秒で取る (float / ISO 文字列のどちらでも)。"""
    raw = turn.get("timestamp")
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    parsed = parse_utc(str(raw or ""))
    return parsed.timestamp() if parsed is not None else 0.0


def build_note_from_turn(
    turn: dict,
    session_id: str,
    *,
    mode: str = "chat",
    project_id: str | None = None,
    turn_index: int = 0,
    triggers_dir: str | Path | None = None,
    auto_pin: bool = True,
) -> MemoryNote | None:
    """1 ターンから :class:`MemoryNote` を作る。ノートにしないターンは ``None``。

    ノートにしない条件 (旧 ``absorb`` と同じ):

    - ``is_tool_output`` のターン — ツール出力はエピソード記憶に残さない
    - ツール出力を **言い直しただけ** のアシスタント発話 — 1 ホップで上の
      除外を迂回し、揮発する計測値が焼き付く
    - 本文が空
    """
    content = str(turn.get("content") or "")
    if not content.strip():
        return None
    meta = turn.get("meta") if isinstance(turn.get("meta"), dict) else {}
    role = str(turn.get("role") or "user")
    source = meta.get("source") or turn.get("source")
    turn_mode = str(meta.get("mode") or turn.get("mode") or mode or "chat")
    is_tool_output = bool(meta.get("is_tool_output") or turn.get("is_tool_output"))
    if is_tool_output:
        return None
    if role == "assistant" and is_volatile_measurement_report(content):
        logger.debug(
            "Episodic ingest: skipped a volatile measurement report (session=%s)",
            session_id,
        )
        return None

    builder = get_note_builder(turn_mode if turn_mode in ("chat", "create") else "chat")
    data = builder.build(
        content,
        session_id,
        role=role,
        source=source,
        mode=turn_mode if turn_mode in ("chat", "create") else "chat",
        project_id=project_id,
        is_tool_output=False,
    )

    # private ターンは会話履歴に残らないので普通は届かないが、届いたら
    # 「揮発する」契約どおり印を立てて抽出から外す (ストアの private フラグは
    # 注入側のアクティブマスクも落とす)。
    private = bool(turn.get("private"))

    is_correction = bool(meta.get("is_correction") or turn.get("is_correction"))
    if not is_correction and data["source"] == "user":
        # 応答パス側の判定が取りこぼした値の言い直しを、属性の裏取り付きで
        # 拾い直す (旧 ``absorb`` と同じ。記憶層は「その属性の現在値は何か」を
        # 持つので、学習層より広い範囲を訂正とみなす)。
        is_correction = restates_attribute_value(content, mode=data["mode"])

    pin_flag = False
    pin_reason: str | None = None
    if (
        auto_pin
        and not data["is_code_block"]
        and data["source"] == "user"
    ):
        triggers = get_pin_triggers_for(triggers_dir)
        if not triggers.empty:
            detection = detect_pin(content, data["mode"], triggers)
            if detection.should_pin:
                pin_flag = True
                pin_reason = detection.reason
            elif detection.negated:
                pin_reason = detection.reason

    created_at = _turn_timestamp(turn)
    note = MemoryNote(
        id=new_evidence_id(),
        content=data["content"],
        keywords=data["keywords"],
        tags=data["tags"],
        created_at=created_at,
        accessed_at=created_at,
        session_id=session_id,
        source=data["source"],
        confidence=data["confidence"],
        mode=data["mode"],
        project_id=data["project_id"],
        is_code_block=data["is_code_block"],
        extraction_skipped=data["extraction_skipped"] or private,
        extraction_skip_reason=(
            data["extraction_skip_reason"]
            or ("private" if private else None)
        ),
        private=private,
        pin_flag=pin_flag,
        pin_reason=pin_reason,
        is_correction=is_correction,
        trace_id=str(turn.get("trace_id") or "") or None,
        turn_id=str(turn.get("turn_id") or ""),
        turn_index=turn_index,
        lang=detect_lang(content),
        tool_command=meta.get("tool_command") or turn.get("tool_command"),
        tool_command_name=meta.get("tool_command_name") or turn.get("tool_command_name"),
        tool_command_success=(
            meta.get("tool_command_success")
            if meta.get("tool_command_success") is not None
            else turn.get("tool_command_success")
        ),
        tool_command_source=(
            meta.get("tool_command_source") or turn.get("tool_command_source")
        ),
        tool_command_query=(
            meta.get("tool_command_query") or turn.get("tool_command_query")
        ),
        tier="short",
    )
    return note


def ingest_session(
    store: EpisodicStore,
    session: SessionTurns,
    *,
    triggers_dir: str | Path | None = None,
    auto_pin: bool = True,
) -> int:
    """1 セッションの未ノート化ターンをノートにして ``put`` する。"""
    start = store.progress.start_index(session.session_id, session.turns)
    if start >= len(session.turns):
        return 0
    created = 0
    dropped_echo = 0
    # 直前の user 発話。エコー落とし (下記) の比較対象で、ノート化していない
    # ターンから始めても正しく引けるよう、開始位置の 1 つ前から拾う。
    last_user = ""
    for turn in session.turns[max(0, start - 1):start]:
        if str(turn.get("role") or "") == "user":
            last_user = str(turn.get("content") or "")
    # 直前にノート化した user 発話。直後の assistant ノートと **問い ↔ 答え** で
    # 結ぶ (``answered_by`` / ``answers``)。問いだけのノート (「発表の日付と
    # テーマを確認させてください」) は注入時に捨てられるが、その答えこそが
    # 別セッションから引きたい証拠 (2026-09-10 ライブ監査 (g) G-05)。
    last_user_note: MemoryNote | None = None
    for index in range(start, len(session.turns)):
        turn = session.turns[index]
        content = str(turn.get("content") or "")
        role = str(turn.get("role") or "user")
        if role == "user":
            last_user = content
        elif last_user and content:
            # 直前のユーザー発言を逐語コピーしただけの応答は記憶しない。
            # 保存すると同じ問いで想起されて再生産され、繰り返し回数が増える
            # 自己増幅ループになる (2026-08-04 ライブ監査)。エコーを落として
            # 中身が残ればその中身だけを残す。
            if is_query_echo(content, last_user):
                dropped_echo += 1
                continue
            cleaned = strip_echoed_query(content, last_user)
            if cleaned != content:
                turn = {**turn, "content": cleaned}
        note = build_note_from_turn(
            turn,
            session.session_id,
            mode=session.mode,
            project_id=session.project_id,
            turn_index=index,
            triggers_dir=triggers_dir,
            auto_pin=auto_pin,
        )
        if note is None:
            continue
        if note.source == "user":
            last_user_note = note
        elif last_user_note is not None and note.source == "assistant":
            note.answers = last_user_note.id
        store.put_note(note, tier="short")
        created += 1
        if note.answers and last_user_note is not None:
            store.patch_note(last_user_note.id, attrs={"answered_by": note.id})
            last_user_note = None
    if dropped_echo:
        logger.info(
            "Episodic ingest: dropped %d echo-only assistant turn(s) (session=%s)",
            dropped_echo, session.session_id,
        )
    store.progress.mark(session.session_id, session.turns, len(session.turns))
    if created:
        logger.info(
            "Episodic ingest: %d note(s) from session %s (turns %d..%d)",
            created, session.session_id, start, len(session.turns) - 1,
        )
    return created


def ingest_new_turns(
    store: EpisodicStore,
    source: TurnSource,
    *,
    session_limit: int = DEFAULT_SESSION_LIMIT,
    triggers_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> int:
    """未ノート化のターンを全セッション分ノートにする。作った件数を返す。"""
    pin_cfg = ((config or {}).get("memory") or {}).get("pin") or {}
    auto_pin = bool(pin_cfg.get("auto_detect", True))
    total = 0
    for session in source.recent_sessions(limit=session_limit):
        total += ingest_session(
            store, session, triggers_dir=triggers_dir, auto_pin=auto_pin,
        )
    return total


__all__ = ["build_note_from_turn", "ingest_new_turns", "ingest_session"]
