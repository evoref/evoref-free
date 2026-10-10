"""発話が **その発話のあった会話そのもの** を対象にしているか (純粋関数 + 判定点)。

「ここまでの内容を3行でまとめて」「今日の話を5つの原則にまとめて」「上の内容を
英語にして」は、答えの材料がその会話の中にしか無い。同じ規則を 3 つの読み手が
読む (不変則 #14 (a)):

- エピソード検索の範囲 (``search_pipeline.episodic_session_scope``): 問いがこの形
  なら自セッションに閉じる。
- 記憶の注入 (``chat_service._query_session_scoped`` → ``MemoryInjector.inject``):
  問いがこの形なら別の会話のノートを ``[関連する記憶]`` に載せない
  (判定点 ``query_conversation_scope``)。
- 記憶の注入のノート側 (``MemoryInjector.inject``): **別の会話の** ノートの本文が
  この形なら、どの問いにも載せない — 本文の「ここまで」「今日の話」はその会話を
  指していて、今の会話へ持ち出すと情報量が無いまま今の指示を別の会話へ誘導する。

実インシデント (2026-10-09 実機 301 ターン): 会話全体を対象にした問いのまとめ漏れ
9 件のうち 6 件で、``[関連する記憶]`` に別の会話の「ここまでの内容を3行で
まとめて。」「今日の話を5つの原則にまとめて。」が載っていた。注入されたメタ発話が
次の会話のノートの候補になり、会話を重ねるごとに行が増えた (最後は 1 ターンに 6 行)。

手掛かりは 3 つで、語を足さず既存の判定と構造で取る:

- 直前の出力を指す後方参照 (:func:`~backend.free.core.intent_vocab.refers_to_previous_output`)
- 会話そのものへのアンカー (:func:`~backend.free.core.intent_vocab.refers_to_ongoing_session`)
- 依頼の目的語が **発話日 + の + 内容語を持たない名詞** だけでできている
  (「今日の話を」「本日の内容を」)。発話日の語は ``temporal_deixis.present_day_terms``、
  内容語の有無は ``query_anchors`` (どちらも既存の SSOT)。「今日の天気を」「今日の
  予定を」は目的語に内容語があるので当たらない。「今日の会話」(日付で区切った
  全セッション、:func:`~backend.free.core.intent_vocab.is_today_scope_query`) と
  「この会話とは別に」は除く。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from backend.free.core.intent_vocab import (
    excludes_current_conversation,
    is_today_scope_query,
    refers_to_ongoing_session,
    refers_to_previous_output,
)
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.free.core.query_anchors import query_anchors
from backend.free.core.script_ranges import KANJI
from backend.free.core.temporal_deixis import present_day_terms
from backend.free.core.text_quality import mentions_self

#: 文の区切り (句点・感嘆符・疑問符・改行)。
_SENTENCE_RE = re.compile(r"[^。．.！!？?\n]+")
#: 目的語の名詞句の切れ目 (読点)。
_PHRASE_BREAK_RE = re.compile(r"[、,，]")
_KANJI_RE = re.compile(f"[{KANJI}]")
#: 目的語として見る名詞句の最大長。「今日の話」「本日の内容」程度の短い句だけを見る。
_MAX_OBJECT_CHARS = 12
_PRESENT_DAY_TERMS = frozenset(present_day_terms())


def day_scoped_contentless_object(text: str) -> bool:
    """依頼の目的語が「発話日 + の + 内容語を持たない名詞」か (純粋関数)。

    文ごとに最初の ``を`` の手前 (読点より後) を目的語の名詞句と読み、最後の
    ``の`` で修飾部と名詞に分ける。修飾部が発話日の語 (今日 / 本日) そのもので、
    名詞が漢字を含みかつ内容語 (``query_anchors``) を持たない (「話」「内容」) とき真。
    """
    for sentence in _SENTENCE_RE.findall(text or ""):
        head, sep, _ = sentence.partition("を")
        if not sep:
            continue
        phrase = _PHRASE_BREAK_RE.split(head)[-1].strip()
        if not phrase or len(phrase) > _MAX_OBJECT_CHARS:
            continue
        modifier, sep_no, noun = phrase.rpartition("の")
        if not sep_no or modifier not in _PRESENT_DAY_TERMS:
            continue
        if _KANJI_RE.search(noun) and not query_anchors(noun):
            return True
    return False


# ── 判定点 ``query_conversation_scope`` ───────────────────────────────

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "query_conversation_scope"

#: 発話が、その発話のあった会話そのものを対象にしている。
CONVERSATION_SCOPE_LABEL = "conversation_scope"


def conversation_scope_reason(text: str) -> tuple[str | None, str]:
    """``(label, evidence)`` を返す (純粋関数)。

    - 直前の出力への後方参照 → ``(CONVERSATION_SCOPE_LABEL, "previous_output")``
    - 会話そのものへのアンカー → ``(CONVERSATION_SCOPE_LABEL, "session_anchor")``
    - 日付で区切った会話 / 今の会話の除外 → ``(NEGATIVE_LABEL, "other_scope")``
    - 目的語が「今日の話」の形で、一人称が無い →
      ``(CONVERSATION_SCOPE_LABEL, "day_contentless_object")``
    - 同じ形で一人称を伴う (「今日の話を私の日記に」) → ``(None, "self_reference")``
      — 本人の記憶を要しうるので棄権
    - どれも無い → ``(NEGATIVE_LABEL, "no_match")``

    前 2 つの順序と判定は ``episodic_session_scope`` の従来の第 1 分岐と同じ
    (日付の除外より前に立てる — 既存の挙動を変えない)。
    """
    t = (text or "").strip()
    if not t:
        return NEGATIVE_LABEL, "empty"
    if refers_to_previous_output(t):
        return CONVERSATION_SCOPE_LABEL, "previous_output"
    if refers_to_ongoing_session(t):
        return CONVERSATION_SCOPE_LABEL, "session_anchor"
    if is_today_scope_query(t) or excludes_current_conversation(t):
        return NEGATIVE_LABEL, "other_scope"
    if not day_scoped_contentless_object(t):
        return NEGATIVE_LABEL, "no_match"
    if mentions_self(t):
        return None, "self_reference"
    return CONVERSATION_SCOPE_LABEL, "day_contentless_object"


def addresses_ongoing_conversation(text: str) -> bool:
    """発話がその発話のあった会話そのものを対象にしているか (純粋関数、発火だけ真)。

    判定点を通さない読み手 (エピソード検索の範囲、注入のノート側) が読む。規則は
    :func:`conversation_scope_reason` の 1 本 — 棄権は偽。
    """
    label, _ = conversation_scope_reason(text)
    return label == CONVERSATION_SCOPE_LABEL


class _ConversationScopeRule:
    """字句段 (``Predicate`` プロトコル)。根拠を形ごとに書き分けるため
    ``LexicalPredicate`` ではなく ``Verdict`` を直接返す。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:  # noqa: ARG002 - Predicate プロトコルの引数
        label, evidence = conversation_scope_reason(text or "")
        if label is None:
            return Verdict(
                value=None, score=0.5, band="abstain",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        if label == CONVERSATION_SCOPE_LABEL:
            return Verdict(
                value=label, score=1.0, band="fire",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=NEGATIVE_LABEL, score=0.0, band="skip",
            evidence=evidence, predicate=self.name, stage="lexical",
        )


_RULE = _ConversationScopeRule()

#: プロセス共通の判定点。チャット応答パスの記憶注入がターンに 1 回引く。
#: 誤発火のほうが重い (別の会話の記憶を落とす) ので ``confirm``。事例段は未装着。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="confirm",
        candidates=[CONVERSATION_SCOPE_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def query_conversation_scope_verdict(query: str) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する。ターンに 1 回)。"""
    return predicate.evaluate(query or "")


__all__ = [
    "CONVERSATION_SCOPE_LABEL",
    "PREDICATE_NAME",
    "addresses_ongoing_conversation",
    "bind_debug_logger",
    "conversation_scope_reason",
    "day_scoped_contentless_object",
    "predicate",
    "query_conversation_scope_verdict",
]
