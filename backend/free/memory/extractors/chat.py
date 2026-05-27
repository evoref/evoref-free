"""

チャットモード由来の ``MemoryNote`` から SemanticFact 候補を抽出する。

抽出される type は統合仕様に従い:

- ``personal_fact`` — ユーザー固有事実 (subject = ``mem.personal.user``)
- ``world_fact`` — 一般知識 (subject = ``mem.world.<keyword>``)
- ``preference`` — 嗜好 (subject = ``mem.preference.user``)
- ``emotion`` — 感情 (subject = ``mem.emotion.user``)
- ``opinion`` — 意見 (subject = ``mem.opinion.user``)

ロジックは ``ChatNoteBuilder.candidate_fact_tags`` のトリガ判定をそのまま再利用
してモード一貫性を保つ (Builder 側のトリガ更新に追従できる)。LLM 呼び出しは
行わない。

スコープは常に ``global``

subject の pillar namespace (``mem.*``) を全面適用した
"""

from __future__ import annotations

from collections.abc import Iterable

from backend.free.memory.extractors.base import (
    BaseExtractor,
    ExtractionContext,
    ExtractionResult,
)
from backend.free.memory.notes.note_builder import ChatNoteBuilder
from backend.free.memory.stores.short_term import MemoryNote
from backend.free.memory.notes.subject_ns import make_mem_subject
from backend.free.memory.types import FactType, SemanticFact
from backend.log_config import get_logger

logger = get_logger("memory.extractors.chat")


#: FactType → ``mem.*`` subject の ``<kind>`` 対応表
_KIND_BY_TAG: dict[str, str] = {
    "personal_fact": "personal",
    "world_fact": "world",
    "preference": "preference",
    "emotion": "emotion",
    "opinion": "opinion",
}

# chat extractor で生成する subject の固定「parts」(= user を主語とする)
_USER_SUBJECT_TAGS: frozenset[str] = frozenset(
    {"personal_fact", "preference", "emotion", "opinion"},
)

_PREDICATE_BY_TAG: dict[str, str] = {
    "personal_fact": "states",
    "world_fact": "is",
    "preference": "prefers",
    "emotion": "feels",
    "opinion": "thinks",
}


_SAFE_KEYWORD_FALLBACK = "unknown"


def _sanitize_keyword(raw: str) -> str:
    """``mem.<kind>.<parts>`` に使える安全な文字列に変換する。

    ``make_mem_subject`` (``subject_ns._SAFE_PART_RE``) は ASCII の英数字 /
    ``_`` / ``-`` のみ許容するため、Unicode 文字 (日本語等) は ``-`` に置換する。
    先頭非英数字を除去し、全て置換された場合は :data:`_SAFE_KEYWORD_FALLBACK`
    を返す。
    """
    sanitized: list[str] = []
    for ch in raw:
        if (ch.isascii() and ch.isalnum()) or ch in ("_", "-"):
            sanitized.append(ch)
        else:
            sanitized.append("-")
    out = "".join(sanitized).strip("-_")
    # 先頭が英数字でないと validate 側で弾かれる
    while out and not (out[0].isascii() and out[0].isalnum()):
        out = out[1:]
    return out or _SAFE_KEYWORD_FALLBACK


def _world_fact_keyword(note: MemoryNote) -> str:
    """world_fact の subject kind キーワードをノートから導く。

    最初の `keywords` を採用し、無ければ内容先頭 24 文字を使う。
    ``make_mem_subject`` 互換のサニタイズを掛けて返す。
    """
    if note.keywords:
        return _sanitize_keyword(note.keywords[0])
    text = " ".join((note.content or "").split())
    return _sanitize_keyword(text[:24]) if text else _SAFE_KEYWORD_FALLBACK


class ChatExtractor(BaseExtractor):
    """チャットモード用 SemanticFact 抽出器。"""

    mode = "chat"

    #: ``candidate_fact_tags`` から実際に SemanticFact 化する type 集合
    SUPPORTED_TAGS: tuple[FactType, ...] = (
        "personal_fact",
        "world_fact",
        "preference",
        "emotion",
        "opinion",
    )

    def __init__(self, builder: ChatNoteBuilder | None = None) -> None:
        self._builder = builder or ChatNoteBuilder()

    def extract(
        self,
        notes: Iterable[MemoryNote],
        ctx: ExtractionContext,
    ) -> ExtractionResult:
        result = ExtractionResult()
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
                fact_type: FactType = tag  # type: ignore[assignment]
                kind = _KIND_BY_TAG[tag]
                if tag in _USER_SUBJECT_TAGS:
                    subject = make_mem_subject(kind, "user")
                else:
                    # world_fact のみ: keyword (sanitized) を parts に使用
                    subject = make_mem_subject(kind, _world_fact_keyword(note))
                fact = self.make_fact(
                    subject=subject,
                    predicate=_PREDICATE_BY_TAG.get(tag, "states"),
                    object_text=note.content or "",
                    fact_type=fact_type,
                    scope=SemanticFact.make_global_scope(),
                    note=note,
                    ctx=ctx,
                )
                candidates.append((note, fact))

        kept, dropped = self.apply_session_caps(candidates, ctx)
        result.cap_dropped = dropped
        result.facts = [fact for _, fact in kept]
        # ノート → fact ID の双方向リンクを書き戻し
        for note, fact in kept:
            if fact.id not in note.extracted_fact_ids:
                note.extracted_fact_ids.append(fact.id)
        logger.debug(
            "ChatExtractor: processed=%d skipped=%d already=%d facts=%d dropped=%d",
            result.notes_processed,
            result.notes_skipped,
            result.already_extracted,
            len(result.facts),
            result.cap_dropped,
        )
        return result
