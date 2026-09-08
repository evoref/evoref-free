"""``MemoryNote`` ↔ ``Evidence`` (kind=``note``) の対応 (c_16 §3.5 / §4.1)。

``MemoryNote`` は **sleep-time の作業用インメモリ表現** で、永続形は
``Evidence`` 1 本。旧 STM (``stores/short_term.py``) の dataclass から、
書き戻し後も消費者が残っているフィールドだけを引き継いだ。

## 落としたもの

===================== ===============================================
落としたフィールド     理由
===================== ===============================================
``embedding``         ベクトルは ``embeddings/<model_id>/`` (c_16 §6.1)。
                      レコードには持たない。作業用の一時値としてのみ
                      :attr:`MemoryNote.embedding` に載る (永続化しない)。
``embed_failures``    埋め込みは snapshot 生成時の増分処理になり、
                      ノート単位の失敗回数を持ち越す意味が無くなった。
``access_count``      LightMem スコアの frequency 項の入力だった。順位式は
                      ``cos × freshness × confidence × store_prior`` に一本化
                      され (c_16 §7.2)、保持順は ``last_used_at`` (§5.4)。
``lightmem_score``    同上。0.6cos+0.4lightmem の合成は廃止 (c_16 §7.1)。
===================== ===============================================

## ``attrs`` に載せたものと消費者

``Evidence.attrs`` のキー名は :data:`NOTE_ATTR_FIELDS` が SSOT。c_16 §3.5 に
無いキー (下表) は phase 3 時点で **まだ生きている sleep-time 工程** が読む。

================== ==================================================
attrs キー          消費者
================== ==================================================
``turn_index``      c_16 §3.5。会話内の位置 (表示・順序の再現用)
``summary_of``      c_16 §3.5。要約ノート → 元ノート id
``mode``            c_16 §3.5。``chat`` / ``coding`` (本リポジトリの
                    ``create`` モードは ``coding`` に対応させる)
``keywords``        ``corrections.corrections_by_target`` (訂正の宛先)、
                    ``sleep.url_curator`` / ``assertion_curator``
``tags``            ``sleep.extraction`` (抽出器の適格判定)、
                    ``notes.note_evolver``
``context_description`` ``notes.note_evolver`` (Step 7 のノート進化)
``is_correction``   ``sleep.extraction`` / ``assertion_curator`` /
                    ``corrections``
``is_tool_output``  ``sleep.extraction`` (``BaseExtractor.is_eligible``)
``is_code_block``   同上
``extraction_skipped`` / ``extraction_skip_reason`` 同上
``extraction_deferred`` ``sleep.extraction`` のセッション別上限
``extracted_fact_ids`` ``sleep.extraction`` (二重抽出の抑止)
``tool_command`` /
``tool_command_name`` /
``tool_command_success`` /
``tool_command_source`` /
``tool_command_query``  ``sleep.executable_command_curator`` (Step 8.6)
``links`` / ``cluster_id``  ``notes.note_evolver.rebuild_links_and_clusters``
``evolution_pending``   ``notes.note_evolver`` (Step 7 の対象選別)
``conflict_candidate`` /
``conflict_partner_id`` /
``conflict_fail_count`` /
``conflict_cooldown_until`` ``pipeline.conflict_resolver`` (Step 6)
``url_curated_at``      ``sleep.url_curator`` (Step 8.5) の冪等マーカー
``command_curated_at``  ``sleep.executable_command_curator`` (Step 8.6)
``assertion_curated_at`` / ``assertion_slug`` ``sleep.assertion_curator``
``pin_reason``          pin の理由 (可視化 / デバッグ)
``episode_id``          ``notes.mdp_ingester`` 由来のノート
================== ==================================================

phase 4 が SemMem を置き換えるときに、これらの消費者ごと畳む予定。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from backend.free.memory.types import MemoryMode, NoteSource
from backend.free.rag.evidence import (
    Evidence,
    compute_claim_key,
    derive_confidence,
    new_evidence_id,
)
from backend.log_config import get_logger
from backend.utils import format_utc, parse_utc, utc_now, utc_now_dt

logger = get_logger("memory.episodic.note")

#: ``MemoryNote.source`` (NoteSource) → ``Evidence.origin``。
#: ``rag`` は「取り込んだ文書由来」なので ``document`` に対応する。
_SOURCE_TO_ORIGIN: dict[str, str] = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "rag": "document",
}
_ORIGIN_TO_SOURCE: dict[str, str] = {v: k for k, v in _SOURCE_TO_ORIGIN.items()}

#: 本リポジトリの ``MemoryMode`` (``chat`` / ``create``) と c_16 §3.5 の
#: ``attrs.mode`` (``chat`` / ``coding``) の対応。同じ概念に別名が付いている
#: だけなので、レコードには c_16 の語彙で書き、読み戻しで元へ戻す。
_MODE_TO_ATTR: dict[str, str] = {"chat": "chat", "create": "coding"}
_ATTR_TO_MODE: dict[str, str] = {"chat": "chat", "coding": "create"}

#: ``provenance[].extractor_version``。ノートの作り方 (本文の切り出し / タグ付け)
#: を変えたら上げる — 「どの版の抽出器が作ったノートか」を後から数えられるように。
NOTE_BUILDER_VERSION = 1


#: ``attrs`` へ載せるフィールド名 (``MemoryNote`` の属性名と同じ)。
#: 名前を変えないのは、既存の消費者 (getattr ベース) がそのまま動くようにするため。
NOTE_ATTR_FIELDS: tuple[str, ...] = (
    "keywords",
    "tags",
    "context_description",
    "is_correction",
    "is_tool_output",
    "is_code_block",
    "extraction_skipped",
    "extraction_skip_reason",
    "extraction_deferred",
    "extracted_fact_ids",
    "tool_command",
    "tool_command_name",
    "tool_command_success",
    "tool_command_source",
    "tool_command_query",
    "links",
    "cluster_id",
    "evolution_pending",
    "conflict_candidate",
    "conflict_partner_id",
    "conflict_fail_count",
    "conflict_cooldown_until",
    "url_curated_at",
    "command_curated_at",
    "assertion_curated_at",
    "assertion_slug",
    "pin_reason",
    "episode_id",
    "turn_index",
    "summary_of",
)

#: ``attrs`` に既定値と同じ値しか入っていないときは書かない (レコードを太らせない)。
_ATTR_DEFAULTS: dict[str, Any] = {
    "keywords": [],
    "tags": [],
    "context_description": "",
    "is_correction": False,
    "is_tool_output": False,
    "is_code_block": False,
    "extraction_skipped": False,
    "extraction_skip_reason": None,
    "extraction_deferred": False,
    "extracted_fact_ids": [],
    "tool_command": None,
    "tool_command_name": None,
    "tool_command_success": None,
    "tool_command_source": None,
    "tool_command_query": None,
    "links": [],
    "cluster_id": None,
    "evolution_pending": True,
    "conflict_candidate": False,
    "conflict_partner_id": None,
    "conflict_fail_count": 0,
    "conflict_cooldown_until": None,
    "url_curated_at": None,
    "command_curated_at": None,
    "assertion_curated_at": None,
    "assertion_slug": None,
    "pin_reason": None,
    "episode_id": None,
    "turn_index": 0,
    "summary_of": [],
}


@dataclass
class MemoryNote:
    """会話由来ノートの作業用表現 (永続形は :class:`Evidence`)。

    ``id`` は ``Evidence.id`` と同一 (``ev_`` + hex12)。ファクトの
    ``Provenance.note_id`` / 経験の ``source_memory_ids`` はこの id で
    エピソードと結ばれる (c_05 §0.6 の ID 連鎖)。
    """

    id: str
    content: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    accessed_at: float = 0.0
    session_id: str = ""
    context_description: str = ""
    evolution_pending: bool = True
    conflict_candidate: bool = False
    conflict_partner_id: str | None = None

    source: NoteSource = "user"
    """発生源 (user / assistant / system / rag)。``Evidence.origin`` と対応。"""

    confidence: float = 1.0
    """決定論導出の確度 (c_16 §3.3)。``origin`` から導く。"""

    pin_flag: bool = False
    pin_reason: str | None = None

    is_correction: bool = False
    """ユーザーが **自分の値を言い直した** ターンか。

    ``sleep.extraction`` が直前の名前付き属性を継承する条件であり、
    ``corrections.corrections_by_target`` が訂正の宛先を解くための印。
    """

    extracted_fact_ids: list[str] = field(default_factory=list)
    private: bool = False
    mode: MemoryMode = "chat"
    project_id: str | None = None
    lang: str = ""
    turn_id: str = ""
    turn_index: int = 0
    trace_id: str | None = None
    episode_id: str | None = None

    is_tool_output: bool = False
    is_code_block: bool = False
    extraction_skipped: bool = False
    extraction_skip_reason: str | None = None
    extraction_deferred: bool = False

    tool_command: str | None = None
    tool_command_name: str | None = None
    tool_command_success: bool | None = None
    tool_command_source: str | None = None
    tool_command_query: str | None = None

    links: list[str] = field(default_factory=list)
    cluster_id: str | None = None

    url_curated_at: float | None = None
    command_curated_at: float | None = None
    assertion_curated_at: float | None = None
    assertion_slug: str | None = None

    conflict_fail_count: int = 0
    conflict_cooldown_until: float | None = None

    tier: str = "short"
    """``working`` / ``short`` / ``long`` (c_16 §4.1)。遷移は ``patch`` のみ。"""

    summary_of: list[str] = field(default_factory=list)
    superseded_by: str | None = None

    embedding: np.ndarray | None = None
    """**永続化しない** 作業用ベクトル。

    Step 6 (競合検出) / Step 7 (ノート進化) は互いの類似度を必要とする。
    snapshot の ``embeddings/`` に載っている分は
    :meth:`EpisodicStore.vectors_for` が復元し、無いものだけ埋め込む。
    """

    _extra: dict = field(default_factory=dict)
    """未知キーの退避先 (``Evidence._extra`` と往復する)。"""


def _epoch(value: str | None, default: float = 0.0) -> float:
    """ISO 8601 (UTC) → epoch 秒。読めなければ ``default``。

    文字列のまま比較しない (c_05 §0.5) ため、読み戻しは必ずここを通す。
    """
    parsed = parse_utc(value) if value else None
    return parsed.timestamp() if parsed is not None else default


def _iso(epoch: float | None) -> str | None:
    """epoch 秒 → ISO 8601 UTC μs ``Z``。``None`` / 0 は ``None``。"""
    if not epoch:
        return None
    from datetime import UTC, datetime

    return format_utc(datetime.fromtimestamp(float(epoch), tz=UTC))


def note_to_evidence(note: MemoryNote, *, tier: str | None = None) -> Evidence:
    """``MemoryNote`` を ``Evidence`` (kind=``note``) にする。

    ``confidence`` は :func:`derive_confidence` で origin から決める
    (LLM にも呼出側にも答えさせない、c_16 §3.3)。``claim_key`` は本文の
    正規化ハッシュで、同一発話の重複注入を注入時に 1 件へ畳むための鍵。
    """
    origin = _SOURCE_TO_ORIGIN.get(str(note.source), "user")
    observed_at = _iso(note.created_at) or utc_now()
    provenance = [
        {
            "session_id": note.session_id,
            "turn_id": note.turn_id,
            "trace_id": note.trace_id or "",
            "extractor": "note_builder",
            "extractor_version": NOTE_BUILDER_VERSION,
            "captured_at": observed_at,
        },
    ]
    attrs: dict[str, Any] = {"tier": tier or note.tier or "short"}
    mode_attr = _MODE_TO_ATTR.get(str(note.mode), "chat")
    attrs["mode"] = mode_attr
    for name in NOTE_ATTR_FIELDS:
        value = getattr(note, name, None)
        if value == _ATTR_DEFAULTS.get(name) and name not in ("turn_index",):
            continue
        attrs[name] = value
    if note.turn_index:
        attrs["turn_index"] = int(note.turn_index)
    else:
        attrs.pop("turn_index", None)

    scope = f"project:{note.project_id}" if note.project_id else "global"
    return Evidence(
        id=note.id or new_evidence_id(),
        kind="note",
        store="episodic",
        scope=scope,
        text=note.content,
        lang=note.lang or None,
        origin=origin,  # type: ignore[arg-type]
        provenance=provenance,
        observed_at=observed_at,
        confidence=derive_confidence(origin, 1),
        claim_key=compute_claim_key(note.content),
        superseded_by=note.superseded_by,
        private=bool(note.private),
        pinned=bool(note.pin_flag),
        created_at=observed_at,
        updated_at=utc_now(),
        last_used_at=_iso(note.accessed_at),
        attrs=attrs,
        _extra=dict(note._extra or {}),
    )


def evidence_to_note(record: Evidence) -> MemoryNote:
    """``Evidence`` (kind=``note``) から作業用 :class:`MemoryNote` を復元する。

    ``attrs`` の未知キーは ``_extra`` へ退避する (捨てない)。
    """
    attrs = dict(record.attrs or {})
    known = {f.name for f in fields(MemoryNote)}
    extra = dict(record._extra or {})
    note = MemoryNote(
        id=record.id,
        content=record.text,
        created_at=_epoch(record.observed_at),
        accessed_at=_epoch(record.last_used_at, _epoch(record.observed_at)),
        source=_ORIGIN_TO_SOURCE.get(str(record.origin), "user"),  # type: ignore[arg-type]
        confidence=float(record.confidence),
        pin_flag=bool(record.pinned),
        private=bool(record.private),
        lang=record.lang or "",
        tier=str(attrs.get("tier") or "short"),
        superseded_by=record.superseded_by,
    )
    note.mode = _ATTR_TO_MODE.get(str(attrs.get("mode") or "chat"), "chat")  # type: ignore[assignment]
    if record.scope.startswith("project:"):
        note.project_id = record.scope.split(":", 1)[1] or None
    provenance = record.provenance[0] if record.provenance else {}
    note.session_id = str(provenance.get("session_id") or "")
    note.turn_id = str(provenance.get("turn_id") or "")
    note.trace_id = str(provenance.get("trace_id") or "") or None
    for key, value in attrs.items():
        if key in ("tier", "mode"):
            continue
        if key in known:
            setattr(note, key, value)
        else:
            extra[key] = value
    note._extra = extra
    return note


def note_age_days(note: MemoryNote, now: float | None = None) -> float:
    """ノートの経過日数 (tier 昇格の判定用)。"""
    reference = utc_now_dt().timestamp() if now is None else float(now)
    created = float(note.created_at or 0.0)
    if created <= 0:
        return 0.0
    return max(0.0, (reference - created) / 86400.0)


__all__ = [
    "NOTE_ATTR_FIELDS",
    "NOTE_BUILDER_VERSION",
    "MemoryNote",
    "evidence_to_note",
    "note_age_days",
    "note_to_evidence",
]
