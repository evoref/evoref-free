"""

コーディングモード由来の ``MemoryNote`` から SemanticFact 候補を抽出する。

抽出される type は統合仕様 (``coding_task`` 含む) に従い:

- ``project`` — プロジェクトルール (subject = ``mem.project.<project_id>``)
- ``decision`` — 採用/不採用の判断 (subject = ``mem.decision.<project_id>``)
- ``commitment`` — 締切・予定 (subject = ``mem.commitment.user``)
- ``coding_task`` — タスク依頼 (subject = ``mem.coding_task.<project_id>``)
  Loop driver の ``task`` FactType と構造差があるため別 FactType に分離
- ``coding`` — コード関連知識 (subject = ``mem.coding.<keyword>``)

スコープは可能な限り ``project:<project_id>``
``ctx.project_id`` が ``None`` の場合は extractor が no-op となる
(``project`` タグが ``project`` スコープ必須のため)。

ロジックは ``CodingNoteBuilder.candidate_fact_tags`` のトリガ判定をそのまま再利用
してモード一貫性を保つ。LLM 呼び出しは行わない。

subject の pillar namespace (``mem.*``) を全面適用した
"""

from __future__ import annotations

from collections.abc import Iterable

from backend.free.memory.extractors.base import (
    BaseExtractor,
    ExtractionContext,
    ExtractionResult,
)
from backend.free.memory.notes.note_builder import CodingNoteBuilder
from backend.free.memory.stores.short_term import MemoryNote
from backend.free.memory.notes.subject_ns import make_mem_subject
from backend.free.memory.types import FactType, SemanticFact
from backend.log_config import get_logger

logger = get_logger("memory.extractors.coding")


#: ``CodingNoteBuilder.candidate_fact_tags`` が返すタグ → 実際に書き込む FactType。
#:
#: ``task`` タグは ``coding_task`` FactType に変換する
#: (CodingExtractor の task は Loop driver の ``task`` と構造差が
#: あるため別 FactType に分離)。
_FACT_TYPE_BY_TAG: dict[str, FactType] = {
    "project": "project",
    "decision": "decision",
    "commitment": "commitment",
    "task": "coding_task",  # D4: task → coding_task に変換
    "coding": "coding",
}

_PREDICATE_BY_TAG: dict[str, str] = {
    "project": "rule",
    "decision": "decided",
    "commitment": "promised",
    "task": "requested",
    "coding": "notes",
}


_SAFE_KEYWORD_FALLBACK = "unknown"


def _sanitize_keyword(raw: str) -> str:
    """``mem.<kind>.<parts>`` で許容される文字列にサニタイズする。

    `subject_ns._SAFE_PART_RE` は ASCII の英数字 / ``_`` / ``-`` のみを許容する
    ため、Unicode 文字 (日本語等) は ``-`` に置換する。
    """
    sanitized: list[str] = []
    for ch in raw:
        if (ch.isascii() and ch.isalnum()) or ch in ("_", "-"):
            sanitized.append(ch)
        else:
            sanitized.append("-")
    out = "".join(sanitized).strip("-_")
    while out and not (out[0].isascii() and out[0].isalnum()):
        out = out[1:]
    return out or _SAFE_KEYWORD_FALLBACK


def _coding_keyword(note: MemoryNote) -> str:
    """``coding`` subject 用の kind キーワードをノートから導く (サニタイズ済)。"""
    if note.keywords:
        return _sanitize_keyword(note.keywords[0])
    text = " ".join((note.content or "").split())
    return _sanitize_keyword(text[:24]) if text else "coding"


def _build_subject(tag: str, *, project_id: str, note: MemoryNote) -> str:
    """Coding extractor の subject を mem.* namespace で構築する"""
    if tag == "commitment":
        return make_mem_subject("commitment", "user")
    if tag == "project":
        return make_mem_subject("project", _sanitize_keyword(project_id))
    if tag == "decision":
        return make_mem_subject("decision", _sanitize_keyword(project_id))
    if tag == "task":
        return make_mem_subject("coding_task", _sanitize_keyword(project_id))
    if tag == "coding":
        return make_mem_subject("coding", _coding_keyword(note))
    # フォールバック (理論上到達しない)
    return make_mem_subject("coding", _SAFE_KEYWORD_FALLBACK)


class CodingExtractor(BaseExtractor):
    """コーディングモード用 SemanticFact 抽出器。"""

    mode = "coding"

    #: ``CodingNoteBuilder.candidate_fact_tags`` が返しうるタグ集合。
    #: 実際に書き込む FactType は :data:`_FACT_TYPE_BY_TAG` を経由する
    SUPPORTED_TAGS: tuple[str, ...] = (
        "project",
        "decision",
        "commitment",
        "task",
        "coding",
    )

    def __init__(self, builder: CodingNoteBuilder | None = None) -> None:
        self._builder = builder or CodingNoteBuilder()

    def extract(
        self,
        notes: Iterable[MemoryNote],
        ctx: ExtractionContext,
    ) -> ExtractionResult:
        result = ExtractionResult()
        if not ctx.project_id:
            logger.debug(
                "CodingExtractor: no project_id in context, skipping (project scope required)"
            )
            return result

        scope = SemanticFact.make_project_scope(ctx.project_id)
        candidates: list[tuple[MemoryNote, SemanticFact]] = []
        for note in notes:
            if not self.is_eligible(note, self.mode):
                result.notes_skipped += 1
                continue
            if note.extracted_fact_ids:
                result.already_extracted += 1
                continue
            result.notes_processed += 1
            tags = self._builder.candidate_fact_tags(note.content or "")
            for tag in tags:
                if tag not in self.SUPPORTED_TAGS:
                    continue
                fact_type = _FACT_TYPE_BY_TAG.get(tag)
                if fact_type is None:
                    continue
                subject = _build_subject(
                    tag, project_id=ctx.project_id, note=note,
                )
                overrides: dict = {}
                if tag == "task":
                    # coding_task ファクトの初期 confidence
                    # (task → coding_task FactType に分離済)
                    overrides["confidence"] = 0.6
                fact = self.make_fact(
                    subject=subject,
                    predicate=_PREDICATE_BY_TAG.get(tag, "notes"),
                    object_text=note.content or "",
                    fact_type=fact_type,
                    scope=scope,
                    note=note,
                    ctx=ctx,
                    **overrides,
                )
                candidates.append((note, fact))

        kept, dropped = self.apply_session_caps(candidates, ctx)
        result.cap_dropped = dropped
        result.facts = [fact for _, fact in kept]
        for note, fact in kept:
            if fact.id not in note.extracted_fact_ids:
                note.extracted_fact_ids.append(fact.id)
        logger.debug(
            "CodingExtractor: processed=%d skipped=%d already=%d facts=%d dropped=%d project=%s",
            result.notes_processed,
            result.notes_skipped,
            result.already_extracted,
            len(result.facts),
            result.cap_dropped,
            ctx.project_id,
        )
        return result
