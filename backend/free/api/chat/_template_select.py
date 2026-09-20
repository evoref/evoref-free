"""判定点 `template_select` — 文書テンプレートの選択 (c_16 §4.5.2 / c_17 §3.8)。

語彙はコードではなく **パッケージが持つ** (``doc_type`` / ``aliases``)。判定点が
固定で持つのは様式語 (テンプレート / ひな形 / 様式 / フォーマット / template)
だけ。事例段は初版では持たない (c_17 §3.8: fire の条件が 2 語の連言で狭く、
語彙の片方はパッケージ由来で事前に事例を用意できない)。

インストール済みテンプレートが 0 件なら、呼出側 (``chat.py``) はこの判定点を
一切呼ばない (記録も出さない、c_16 §4.5.2)。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from backend.free.core.intent_vocab import ascii_boundary_alternation
from backend.free.core.predicate import (
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.free.rag.corpus.templates import TemplateCandidate

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "template_select"

#: 様式語 (c_16 §4.5.2)。
_STYLE_WORD_RE = re.compile(
    r"テンプレート|ひな形|雛形|様式|フォーマット|"
    + ascii_boundary_alternation("template"),
    re.IGNORECASE,
)

#: abstain の evidence (c_17 §3.8 の表)。
CANDIDATE_EVIDENCE = "candidate"
AMBIGUOUS_EVIDENCE = "ambiguous"


class _TemplateSelectLexical:
    """字句段 (唯一の段)。``ctx["entries"]`` にこのターンの候補を渡す。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        entries: Sequence[TemplateCandidate] = (ctx or {}).get("entries") or ()
        if not text or not entries:
            return Verdict(
                value=None, score=0.0, band="skip", evidence="no_entries",
                predicate=self.name, stage="lexical",
            )
        named = [e for e in entries if e.matches_naming(text)]
        if not named:
            return Verdict(
                value=None, score=0.0, band="skip", evidence="no_naming",
                predicate=self.name, stage="lexical",
            )
        if not _STYLE_WORD_RE.search(text):
            return Verdict(
                value=None, score=0.0, band="abstain", evidence=CANDIDATE_EVIDENCE,
                predicate=self.name, stage="lexical",
            )
        if len(named) > 1:
            return Verdict(
                value=None, score=0.0, band="abstain", evidence=AMBIGUOUS_EVIDENCE,
                predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=named[0].label, score=1.0, band="fire", evidence="template_named",
            predicate=self.name, stage="lexical",
        )


#: プロセス共通の判定点。候補ラベルは install 済みパッケージ次第で動的なので、
#: ``candidates=`` は空のまま (``ABSTAIN_LABEL`` だけが自動で足される)。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_TemplateSelectLexical(),
        policy="complement",
        candidates=[],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def select_template(text: str, entries: Sequence[TemplateCandidate]) -> Verdict:
    """``entries`` (このターンで有効な全パッケージの命名候補) から 1 件選ぶ。

    呼出側の契約: ``entries`` が空ならこの関数を呼ばない (0 件は判定点を
    経由せず素通りする、c_16 §4.5.2)。
    """
    return predicate.evaluate(text or "", {"entries": entries})


__all__ = [
    "AMBIGUOUS_EVIDENCE",
    "CANDIDATE_EVIDENCE",
    "PREDICATE_NAME",
    "bind_debug_logger",
    "predicate",
    "select_template",
]
