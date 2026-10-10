"""値の span が **話者本人の言明** の中にあるかを見る判定点 (sleep-time)。

Step 8.3 (``sleep/personal_fact_curator``) は補助タスクが返した値を逐語 span と
スロット表で検証するが、**誰の / どんな枠の言明か** は検証していなかった。
問いの混ざった発話 (「犬を飼い始めました。最初に用意するものは？」) を Step 8.3 へ
回すと (2026-10-10)、同じ形の例文・虚構・他者の話も届く:

- 「例文: 私は京都に住んでいます。これを英訳して。」 — 依頼の素材
- 「小説の主人公が犬を飼い始めました。この後の展開は？」 — 虚構の主体
- 「兄が来月、京都に2泊で旅行します。おすすめは？」 — 他者の予定
- 「去年まで犬を飼っていました。また飼うなら？」 — 終わった状態
- 「I wish I were taller. How do I say it politely?」 — 仮定

補助タスクのプロンプトは他者・依頼・質問を除外させているが、モデルの出力は
候補でしかない。同じ除外を **決定論の門** として二重化する。判定は語形の列挙では
なく、値の span の位置から見た構造で決める:

1. ``request_material`` — 値の文が依頼の素材 (ラベルのコロンの後ろで、発話に
   依頼がある / 後ろの依頼文の目的語が指示語「これを英訳して」)
2. ``quoted`` — 値が引用 (鉤括弧) の中で、引用の中身が文になっている
   (『ノルウェイの森』のような題名は文ではないので通す)
3. ``in_question`` — 値が問い・依頼の文の、依頼の節そのものにある
   (「勉強の**進め方**を教えて」。根拠の文を依頼文にしない)
4. ``hypothetical`` — 値の節に仮定の標識 (``correction_verdict.HYPOTHETICAL_MARKER``
   と同じ語彙) / 英語の仮定 (I wish / if I were)
5. ``ended_state`` — 値の節が「〜まで / 以前は … ていました」で終わった状態
6. ``hearsay`` — 値の文に伝聞の標識 (``correction_verdict.HEARSAY_MARKER``)
7. ``other_subject`` — 値の手前の直近の主語が本人以外の人
   (``text_quality.another_person_is_subject_before`` と同じ判定)
8. ``agent_subject`` — 値の手前に「X の Y が … を」の動作主 (X が本人でない)

字句段だけのカスケード (不変則 #14 の契約、``log_decision`` を出す)。発火は
``not_own`` (本人の言明ではない)、通過は陰性ラベル、値が発話に見つからなければ
棄権。呼出側は **通過 (skip) 以外は書かない**。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from backend.free.core.correction_verdict import (
    HEARSAY_MARKER,
    HYPOTHETICAL_MARKER,
    QUOTE_PAIRS,
)
from backend.free.core.intent_vocab import (
    PAYLOAD_COLON_RE,
    is_asking_sentence,
    is_plain_statement,
    memorize_request,
    split_sentences,
)
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.free.core.text_quality import (
    _SELF_REFERENCE_RE,
    another_person_is_subject_before,
)

PREDICATE_NAME = "speaker_own_statement"

#: 発火ラベル (値は話者本人の言明の中に無い)。
NOT_OWN_LABEL = "not_own"

#: 門が返す理由 (``Verdict.evidence``)。Step 8 は依頼の素材だけを読む。
REQUEST_MATERIAL = "request_material"

_HYPOTHETICAL_RE = re.compile(
    HYPOTHETICAL_MARKER
    + r"|\bI\s+wish\b|\bif\s+only\b|\bif\s+I\s+(?:were|was|had)\b",
    re.IGNORECASE,
)
_HEARSAY_RE = re.compile(HEARSAY_MARKER)
#: 終わった状態: 終点 (まで) / 以前の対比のあとに過去の継続・断定で閉じる節。
_ENDED_STATE_RE = re.compile(
    r"(?:まで|以前は)[^。！？!?\n]*?"
    r"(?:ていました|でいました|ていた|でいた|でした|だった)\s*[。．.!！]?\s*$",
)
#: 節の末尾に残る逆接・理由の接続 (「住んでいましたが、」の が)。
_CLAUSE_CONJUNCTION_TAIL_RE = re.compile(r"(?:けれども|けれど|けど|ので|から|が)$")
_CLAUSE_BREAK_CHARS = "、，,"
#: 「X の Y が … を」の動作主。X は本人以外 (一人称なら他の判定に任せる)。
_GENITIVE_AGENT_RE = re.compile(
    r"(?:^|[、，,])\s*(?P<owner>[^、，,。はがをにのへと\s]{1,15})の"
    r"[^、，,。はがをにのへと\s]{1,10}が[^。！？!?\n]*?を",
)
#: 後ろの依頼文の目的語が指示語 (「これを英訳して」「この文を直してください」)。
#: 動詞句は短く、途中に別の目的語 (を) や て形 (「踏まえて考えて」) を挟まない。
_DEMONSTRATIVE_OBJECT_REQUEST_RE = re.compile(
    r"^\s*(?:これ|それ|この文章?|その文章?|上の文章?|今の文章?)を\s*"
    r"[^\s、，,。をて]{1,12}?(?:して|て|で)"
    r"(?:ください|下さい|くれ\w*|もらえ\w*|ほしい|欲しい)?\s*[。．！!？?]*\s*$",
)
#: 英語の言い換え・翻訳の依頼の目的語が裸の代名詞で、文の終わりにある
#: (「How do I say it politely?」)。「What do you think of it?」は素材の依頼ではない。
_EN_PRONOUN_OBJECT_TAIL_RE = re.compile(
    r"\b(?:say|write|translate|rewrite|phrase|put|express|correct)\s+(?:this|it|that)\b"
    r"(?:\s+\w+){0,2}\s*[?.!]?\s*$",
    re.IGNORECASE,
)


def _locate(value: str, content: str) -> tuple[int, int] | None:
    """``value`` の発話中の位置 (空白の違いは無視する / 純粋関数)。"""
    if not value or not content:
        return None
    at = content.find(value)
    if at >= 0:
        return at, at + len(value)
    index = [i for i, ch in enumerate(content) if not ch.isspace()]
    flat = "".join(content[i] for i in index)
    needle = "".join(value.split())
    if not needle:
        return None
    j = flat.find(needle)
    if j < 0:
        return None
    return index[j], index[j + len(needle) - 1] + 1


def _sentence_spans(content: str) -> list[tuple[int, int]]:
    """文の位置 (``intent_vocab.split_sentences`` の区切り)。"""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in split_sentences(content):
        at = content.find(piece, cursor)
        if at < 0:
            continue
        spans.append((at, at + len(piece)))
        cursor = at + len(piece)
    return spans


def _clause_end(content: str, start: int, end: int) -> int:
    """``start`` 以降の最初の節の区切り (無ければ ``end``)。"""
    positions = [content.find(ch, start, end) for ch in _CLAUSE_BREAK_CHARS]
    hits = [p for p in positions if p >= 0]
    return min(hits) if hits else end


def _quoted_statement(content: str, start: int, end: int) -> bool:
    """値が引用の中にあり、引用の中身が文 (平叙の文末) になっているか。"""
    for open_q, close_q in QUOTE_PAIRS:
        search = 0
        while (o := content.find(open_q, search)) >= 0:
            c = content.find(close_q, o + len(open_q))
            if c < 0:
                break
            search = c + len(close_q)
            if o < start and end <= c:
                inner = content[o + len(open_q):c].strip()
                if is_plain_statement(inner) or inner.endswith(("。", ".")):
                    return True
    return False


def _request_material(
    content: str, sentences: list[tuple[int, int]], index: int, value_end: int,
) -> bool:
    """値の文が依頼の素材か (コロンの後ろの例文 / 後ろの依頼の目的語が指示語)。"""
    s_start, s_end = sentences[index]
    sentence = content[s_start:s_end]
    others = [content[a:b] for i, (a, b) in enumerate(sentences) if i != index]
    colon = PAYLOAD_COLON_RE.search(sentence)
    if (
        colon is not None
        and s_start + colon.end() <= value_end
        and (
            is_asking_sentence(sentence[:colon.start()])
            or any(is_asking_sentence(o) for o in others)
        )
    ):
        return True
    for a, b in sentences[index + 1:]:
        later = content[a:b].strip()
        if not is_asking_sentence(later) or memorize_request(later):
            continue
        if _DEMONSTRATIVE_OBJECT_REQUEST_RE.match(later) or _EN_PRONOUN_OBJECT_TAIL_RE.search(later):
            return True
    return False


def frame_reason(
    content: str, start: int, end: int, own_words: tuple[str, ...] = (),
) -> str | None:
    """``content[start:end]`` の値が本人の言明でない理由 (本人の言明なら ``None`` / 純粋関数)。"""
    sentences = _sentence_spans(content)
    index = next(
        (i for i, (a, b) in enumerate(sentences) if a <= start < b),
        None,
    )
    if index is None:
        return None
    s_start, s_end = sentences[index]
    sentence = content[s_start:s_end]
    if _request_material(content, sentences, index, end):
        return REQUEST_MATERIAL
    if _quoted_statement(content, start, end):
        return "quoted"
    clause_end = _clause_end(content, end, s_end)
    if is_asking_sentence(sentence) and clause_end == s_end:
        return "in_question"
    clause = content[s_start:clause_end].strip()
    if _HYPOTHETICAL_RE.search(clause):
        return "hypothetical"
    if _ENDED_STATE_RE.search(_CLAUSE_CONJUNCTION_TAIL_RE.sub("", clause)):
        return "ended_state"
    if _HEARSAY_RE.search(sentence):
        return "hearsay"
    head = content[s_start:end]
    if another_person_is_subject_before(head, own_words):
        return "other_subject"
    agent = _GENITIVE_AGENT_RE.search(head)
    if agent is not None and not _SELF_REFERENCE_RE.search(agent.group("owner")):
        return "agent_subject"
    return None


class _FrameRule:
    """字句段 (:func:`frame_reason`)。``ctx`` に ``value`` と ``own_words`` を受ける。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        ctx = ctx or {}
        located = _locate(str(ctx.get("value") or ""), text or "")
        if located is None:
            return Verdict(
                value=None, score=0.0, band="abstain",
                evidence="span_not_found", predicate=self.name, stage="lexical",
            )
        reason = frame_reason(
            text, located[0], located[1], tuple(ctx.get("own_words") or ()),
        )
        if reason is None:
            return Verdict(
                value=NEGATIVE_LABEL, score=0.0, band="skip",
                evidence="own_statement", predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=NOT_OWN_LABEL, score=1.0, band="fire",
            evidence=reason, predicate=self.name, stage="lexical",
        )


#: プロセス共通の判定点。sleep-time の同期経路から ``evaluate`` で引く。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_FrameRule(),
        policy="complement",
        candidates=[NOT_OWN_LABEL, NEGATIVE_LABEL],
        scope="sleep",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def judge(content: str, value: str, own_words: tuple[str, ...] = ()) -> Verdict:
    """``value`` (発話の逐語 span) が話者本人の言明の中にあるかを判定して記録する。"""
    return predicate.evaluate(
        content or "", {"value": value, "own_words": tuple(own_words)},
    )


__all__ = [
    "NOT_OWN_LABEL",
    "PREDICATE_NAME",
    "REQUEST_MATERIAL",
    "bind_debug_logger",
    "frame_reason",
    "judge",
    "predicate",
]
