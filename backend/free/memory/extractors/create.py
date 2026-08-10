"""

クリエイトモード由来の ``MemoryNote`` から SemanticFact 候補を抽出する。

抽出される type は統合仕様 (``create_task`` 含む) に従い:

- ``project`` — プロジェクトルール (subject = ``mem.project.<project_id>``)
- ``decision`` — 採用/不採用の判断 (subject = ``mem.decision.<project_id>``)
- ``commitment`` — 締切・予定 (subject = ``mem.commitment.user``)
- ``create_task`` — タスク依頼 (subject = ``mem.create_task.<project_id>``)
  Loop driver の ``task`` FactType と構造差があるため別 FactType に分離
- ``create`` — コード関連知識 (subject = ``mem.create.<keyword>``)

スコープは可能な限り ``project:<project_id>``
``ctx.project_id`` が ``None`` の場合は extractor が no-op となる
(``project`` タグが ``project`` スコープ必須のため)。

ロジックは ``CreateNoteBuilder.candidate_fact_tags`` のトリガ判定をそのまま再利用
してモード一貫性を保つ。LLM 呼び出しは行わない。

subject の pillar namespace (``mem.*``) を全面適用した
"""

from __future__ import annotations

import hashlib

from collections.abc import Iterable

from backend.free.memory.extractors.base import (
    BaseExtractor,
    ExtractionContext,
    ExtractionResult,
)
from backend.free.memory.notes.note_builder import CreateNoteBuilder
from backend.free.memory.stores.short_term import MemoryNote
from backend.free.memory.notes.subject_ns import make_mem_subject
from backend.free.memory.types import FactType, SemanticFact
from backend.log_config import get_logger

logger = get_logger("memory.extractors.create")


#: ``CreateNoteBuilder.candidate_fact_tags`` が返すタグ → 実際に書き込む FactType。
#:
#: ``task`` タグは ``create_task`` FactType に変換する
#: (CreateExtractor の task は Loop driver の ``task`` と構造差が
#: あるため別 FactType に分離)。
_FACT_TYPE_BY_TAG: dict[str, FactType] = {
    "project": "project",
    "decision": "decision",
    "commitment": "commitment",
    "task": "create_task",  # D4: task → create_task に変換
    "create": "create",
}

_PREDICATE_BY_TAG: dict[str, str] = {
    "project": "rule",
    "decision": "decided",
    "commitment": "promised",
    "task": "requested",
    "create": "notes",
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


def _create_keyword(note: MemoryNote) -> str:
    """``create`` subject 用の kind キーワードをノートから導く (サニタイズ済)。"""
    if note.keywords:
        return _sanitize_keyword(note.keywords[0])
    text = " ".join((note.content or "").split())
    return _sanitize_keyword(text[:24]) if text else "create"


def _task_signature(note: MemoryNote) -> str:
    """``create_task`` subject 用のタスク識別子 (内容ハッシュ、12 hex)。

    従来 ``create_task`` の subject は project_id だけだったため、1 プロジェクト
    内の全タスクが 1 subject に集まり、競合検出 (``(subject, predicate)`` キー)
    が別々のタスク依頼を「同一タスクの競合版」と誤判定して恒久 pending 化して
    いた (2026-07-25 実測: 9 タスクが 1 subject、project scope の pending 16 件)。

    ``subject_ns.make_mem_subject`` の docstring が示す
    ``mem.create_task.<project>.<id>`` 形式に合わせる。同一依頼の再アサートでは
    同じ signature になり、正しく更新/競合判定される。
    """
    text = " ".join((note.content or "").split()).lower()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _build_subject(tag: str, *, project_id: str, note: MemoryNote) -> str:
    """Create extractor の subject を mem.* namespace で構築する"""
    if tag == "commitment":
        return make_mem_subject("commitment", "user")
    if tag == "project":
        return make_mem_subject("project", _sanitize_keyword(project_id))
    if tag == "decision":
        return make_mem_subject("decision", _sanitize_keyword(project_id))
    if tag == "task":
        return make_mem_subject(
            "create_task", _sanitize_keyword(project_id), _task_signature(note),
        )
    if tag == "create":
        return make_mem_subject("create", _create_keyword(note))
    # フォールバック (理論上到達しない)
    return make_mem_subject("create", _SAFE_KEYWORD_FALLBACK)


class CreateExtractor(BaseExtractor):
    """クリエイトモード用 SemanticFact 抽出器。"""

    mode = "create"

    #: ``CreateNoteBuilder.candidate_fact_tags`` が返しうるタグ集合。
    #: 実際に書き込む FactType は :data:`_FACT_TYPE_BY_TAG` を経由する
    SUPPORTED_TAGS: tuple[str, ...] = (
        "project",
        "decision",
        "commitment",
        "task",
        "create",
    )

    def __init__(self, builder: CreateNoteBuilder | None = None) -> None:
        self._builder = builder or CreateNoteBuilder()

    def extract(
        self,
        notes: Iterable[MemoryNote],
        ctx: ExtractionContext,
    ) -> ExtractionResult:
        result = ExtractionResult()
        if not ctx.project_id:
            logger.debug(
                "CreateExtractor: no project_id in context, skipping (project scope required)"
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
                    # create_task ファクトの初期 confidence
                    # (task → create_task FactType に分離済)
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
            "CreateExtractor: processed=%d skipped=%d already=%d facts=%d dropped=%d project=%s",
            result.notes_processed,
            result.notes_skipped,
            result.already_extracted,
            len(result.facts),
            result.cap_dropped,
            ctx.project_id,
        )
        return result
