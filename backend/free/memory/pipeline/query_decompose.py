"""決定論の問い分解 (D2) の影の記録 (f_01 §8.1 の 7.9 / c_17 §3.22)。

2 チャンクを跨ぐ複合の問い (「家族構成と住所を教えて」「X は？ Y は？」) は、問い全体の
埋め込みが片側に寄り、もう片側のチャンクが top-k に残らない。問いを **構文だけ** で
2 つに割り、それぞれで統合検索を引けば片側を拾える — ただし疑似クエリ索引 (f_01 §6) の
先頭 interleave がある本番構成では、分解の結果を 3 本目の列として interleave すると
元の問いの席を奪って下がった (2026-09-11 の実測、f_01 §8.1 の 7.9)。

そこで本番化する場合の形を **元の問いの採用 (位置の集合) を一切変えず、空いた枠だけを
分解後の各項の採用で埋める** に固定し、まず影だけを記録する。応答には何も足さない。

- 発火は構文だけで決める (語彙の網目は足さない): 問い符で終わる節が 2 つ以上ある /
  名詞の連なり同士を並列の「と」が結び、直後が助詞 (ひらがな) で続く。
- LLM は呼ばない (不変則 #1)。分解の項ごとに埋め込み 1 回 + 統合検索 1 回。
- 記録は ``decision.jsonl`` の 2 行: 判定点 ``query_decompose`` (would_decompose) と
  ``query_decompose_fill`` (空き枠と、分解後に追加で入る id)。問いの文は書かない。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.free.core.script_ranges import HIRAGANA, KANJI, KANJI_MARKS, KATAKANA_WORD
from backend.log_config import get_logger

logger = get_logger("memory")

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "query_decompose"

#: 影の結果 (空いた枠に入る id) を書く決定点。
FILL_DECISION_POINT = "query_decompose_fill"

#: 分解できる (発火のラベル)。
DECOMPOSE_LABEL = "decompose"

#: 分解の項の上限 (埋め込みの追加回数の上限)。
MAX_SUBQUERIES = 2

#: 問い符で終わる節として数える最小の文字数 (空白を除く)。「え？」「本当？」を数えない。
MIN_CLAUSE_CHARS = 4

#: 並列の「と」で結ばれる名詞の連なりの最小の長さ。
MIN_CONJUNCT_CHARS = 2

_QUESTION_CLAUSE_RE = re.compile(r"[^？?]+[？?]")

_NOUN_CHARS = f"{KANJI}{KANJI_MARKS}{KATAKANA_WORD}A-Za-z0-9"
#: 名詞の連なり (漢字・カタカナ・英数字) + 「と」 + 名詞の連なりで、後ろが助詞 (ひらがな)・
#: 読点・問い符・文末。「とは」「と言う」は「と」の直後がひらがななので当たらない。
_COORDINATION_RE = re.compile(
    f"(?P<a>[{_NOUN_CHARS}]{{{MIN_CONJUNCT_CHARS},}})と"
    f"(?P<b>[{_NOUN_CHARS}]{{{MIN_CONJUNCT_CHARS},}})"
    f"(?=[{HIRAGANA}、。，．？?]|$)",
)


def decompose_query(query: str) -> tuple[list[str], str]:
    """問いを構文だけで分解する (純粋関数)。

    Returns:
        ``(分解の項, 根拠)``。分解しないときは ``([], "no_structure")``。根拠は
        ``question_clauses`` (問い符で終わる節が 2 つ以上) か ``coordination``
        (並列の「と」。最初のものを割る)。
    """
    text = (query or "").strip()
    if not text:
        return [], "no_structure"
    clauses = [
        c.strip() for c in _QUESTION_CLAUSE_RE.findall(text)
        if len(re.sub(r"\s", "", c)) - 1 >= MIN_CLAUSE_CHARS
    ]
    if len(clauses) >= 2:
        return clauses[:MAX_SUBQUERIES], "question_clauses"
    m = _COORDINATION_RE.search(text)
    if m:
        head, tail = text[: m.start()], text[m.end():]
        return [head + m.group("a") + tail, head + m.group("b") + tail], "coordination"
    return [], "no_structure"


class _DecomposeRule:
    """字句段 (``Predicate`` プロトコル)。根拠を構造ごとに書き分ける。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:  # noqa: ARG002 - Predicate プロトコルの引数
        parts, evidence = decompose_query(text or "")
        if not parts:
            return Verdict(
                value=NEGATIVE_LABEL, score=0.0, band="skip",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=DECOMPOSE_LABEL, score=1.0, band="fire",
            evidence=evidence, predicate=self.name, stage="lexical",
            detail={"n_sub": len(parts)},
        )


#: プロセス共通の判定点。影の記録なので方針は ``shadow`` (返すのは常に規則の結果)。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_DecomposeRule(),
        policy="shadow",
        candidates=[DECOMPOSE_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def decompose_mode(cfg: Mapping[str, Any] | None) -> str:
    """``rag.query_decompose.mode`` を返す (``off`` / ``shadow``)。"""
    rag_cfg = (cfg or {}).get("rag") or {}
    return str((rag_cfg.get("query_decompose") or {}).get("mode") or "off")


def fill_empty_seats(
    original_ids: Sequence[str], sub_ids: Sequence[Sequence[str]], top_k: int,
) -> list[str]:
    """元の採用を変えず、空いた枠を分解後の各項の採用で埋める id を返す (純粋関数)。

    各項の採用を順位ごとに交互に取り (項 1 の 1 位、項 2 の 1 位、項 1 の 2 位…)、
    元の採用と重なる id は飛ばす。元の問いの席は奪わない (f_01 §8.1 の 7.9)。
    """
    empty = top_k - len(original_ids)
    if empty <= 0:
        return []
    seen = set(original_ids)
    added: list[str] = []
    depth = max((len(ids) for ids in sub_ids), default=0)
    for rank in range(depth):
        for ids in sub_ids:
            if rank >= len(ids) or ids[rank] in seen:
                continue
            seen.add(ids[rank])
            added.append(ids[rank])
            if len(added) >= empty:
                return added
    return added


#: 分解の項 1 つを検索して、採用された id 列を返す呼出 (埋め込み + 統合検索)。
SubSearch = Callable[[str], Awaitable[list[str]]]


async def record_decompose_shadow(
    query: str,
    *,
    original_ids: Sequence[str],
    top_k: int,
    sub_search: SubSearch,
    debug_logger: Any = None,
) -> list[str]:
    """判定点を記録し、発火したら空いた枠に入る id を影として記録する。

    応答には何も足さない。空き枠が無ければ分解の項を検索しない (埋め込みを払わない)。
    戻り値は記録した id (テスト用)。
    """
    verdict = predicate.evaluate(query)
    if not verdict.fired:
        return []
    parts, evidence = decompose_query(query)
    empty = max(0, top_k - len(original_ids))
    sub_ids: list[list[str]] = []
    if empty > 0:
        results = await asyncio.gather(
            *(sub_search(part) for part in parts), return_exceptions=True,
        )
        for res in results:
            if isinstance(res, BaseException):
                logger.info("Query decompose shadow: sub-search failed: %s", type(res).__name__)
                sub_ids.append([])
            else:
                sub_ids.append(list(res))
    added = fill_empty_seats(original_ids, sub_ids, top_k)
    if added:
        reason = "filled"
    elif empty == 0:
        reason = "no_empty_seat"
    else:
        reason = "no_new_ids"
    logger.info(
        "Query decompose shadow (%s): %d sub-queries, %d empty seat(s), %d id(s) would be added",
        evidence, len(parts), empty, len(added),
    )
    if debug_logger is not None:
        try:
            debug_logger.log_decision(
                decision_point=FILL_DECISION_POINT,
                chosen="fill" if added else "none",
                candidates=["fill", "none"],
                reason=reason,
                context={
                    "would_decompose": True,
                    "structure": evidence,
                    "n_sub": len(parts),
                    "top_k": top_k,
                    "empty_seats": empty,
                    "original_ids": list(original_ids),
                    "sub_adopted": [len(ids) for ids in sub_ids],
                    "added_ids": added,
                },
                scope="request",
            )
        except Exception as e:  # pragma: no cover - ログで検索を落とさない
            logger.debug("Query decompose shadow log failed: %s", e)
    return added


__all__ = [
    "DECOMPOSE_LABEL",
    "FILL_DECISION_POINT",
    "PREDICATE_NAME",
    "bind_debug_logger",
    "decompose_mode",
    "decompose_query",
    "fill_empty_seats",
    "predicate",
    "record_decompose_shadow",
]
