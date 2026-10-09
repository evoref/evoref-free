"""Step 8.35: 集合値の属性の **編集** キュレーター

利用者は属性の現在値を丸ごと言い直さず、要素を足し引きして伝えることがある:

    「趣味は釣りとキャンプです。」
    「趣味に写真を加えて、キャンプは外してください。」
    (あるいは 2 ターンに分けて「趣味にもう一つ、写真を加えてください。」
     「キャンプはもうやめたので趣味から外してください。今の趣味は？」)

編集の発話は依頼形なので Step 8 (``ChatExtractor``) の自己開示にも、Step 8.3
(属性分割) の平叙文にも当たらず、SemMem には編集前の「釣りとキャンプ」だけが
残った。答えの経路も編集を最新の状態と見なさず、別セッションの「私の趣味は？」に
「釣りとキャンプです」と答えた (2026-10-05 実機、2 回とも同じ)。

この段は、属性語でスロットに解決した利用者の発話のうち、**そのスロットに値を
書いていない** (= 言明として抽出されなかった) ものを、スロットの現在値と一緒に
補助タスク ``slot_edit`` へ出す。補助タスクは現在値の要素分け・外す要素・加える
要素を逐語 span で返し、コード側の門 (:func:`check_slot_edit`) が

- 現在値の要素は現在値の逐語 span で、**現在値を網羅する** (要素の取りこぼしで
  値が黙って消えない — 要素の間に残ってよいのは区切りと 2 文字以下のひらがな)
- 外す要素は現在値の要素で、発話に逐語で在る
- 加える要素は発話に逐語で在り、現在値の要素を含まず、属性語を含まない
  (「趣味に写真」のように属性語ごと抜いた span を値にしない)。要素は名詞句の
  形 (「加えて」「写真を」は落とす)
- 外す / 加える要素を持つ文 (引用の内側は落とす) が本人の編集である — 問い・
  是非の相談・否定の依頼・仮定 / 伝聞でなく、属性語の持ち主も要素の持ち主も
  本人以外の人でない (「息子の趣味に」「妻は趣味に」「外さないで」「加えたら
  どう？」「外すべきか迷っている」)。補助タスクのプロンプトだけに任せない

を確かめたときだけ、編集後の要素集合を新しいファクトとして書き、編集した行を
畳む。

属性語の無い発話 (「キャンプはもうやめました。」) は、**要素からスロットを引く**
(:func:`slots_named_by_member`): 発話の主題・目的語 (「<要素>は / を / も」) に置いた
名詞が、編集の対象になる live のスロットの **ちょうど 1 つ** で現在値の要素として
(区切りで) 現れるときだけ、そのスロットを宛先にする。語彙は足さない — 宛先を決めるのは
SemMem の現在値で、属性語の辞書ではない。属性を名指さない発話から値を足すことは
できないので、この経路の編集は **外すだけ** (加える要素があれば却下)。持ち主の検査は
要素の語を属性語の代わりに使う (「娘はキャンプをやめた」「息子のキャンプ」)。

**訂正ではない** (不変則 #12): 旧値は誤りではなく、状態が変わっただけ
(「やめた」「もう〜していない」は ``correction_verify`` の ``premise_change``)。
``correction_verdict`` の門 (「X ではなく Y」の向き・同値・既述) は当てはまらない
ので、別の検証済みの判定 (この段の門) を持つ。書くファクトは ``from_correction``
を立てず訂正の力 (競合の即時解決・値アンカー) を持たない。

**畳むのは編集した値の行だけ** (不変則 #13): 編集の対象は「その発話より前の、
そのスロットの最新の live 行」1 件で、同じ値を持つ (同じ発話を別のセッションで
抽出した) 行だけを一緒に畳む。並列多値 (``multi_valued``) と ``span_only_fold``
のスロットは要素ごとに別の行を持つ / 他者の値が落ちうるので対象にしない。

規約:

- 走るのは sleep-time だけ (CLAUDE.md §6 不変則 #2)。補助タスクは
  ``background_slot`` の文法制約 JSON (不変則 #1)。
- 書き込みは :func:`~backend.free.memory.sleep.extraction.write_sleep_facts`
  を通す (private / 持ち主 / #13 の畳み)。
- 冪等マーカーは ``MemoryNote.slot_edit_curated_at`` (補助タスクが答えたら立つ。
  例外では立てず ``curation_backoff`` へ積む)。適用したら ``slot_edit_slot`` に
  slug を刻み、注入とエピソードがそのノートをスロットの最新の言明に数える
  (:func:`~backend.free.memory.notes.note_builder.note_state_slot`)。
- ``aux_client`` / ``embedder`` / store が無い (degraded) 場合は no-op。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.free.core.correction_verdict import (
    marks_not_own_restatement,
    mask_quoted_speech,
    norm_span,
    strip_copula,
)
from backend.free.core.script_ranges import HIRAGANA, KANJI, KANJI_MARKS, KATAKANA_WORD
from backend.free.llm.json_schemas import SlotEdit
from backend.free.memory.sleep._curator_common import public_notes
from backend.free.memory.sleep.curation_backoff import (
    clear_failure,
    in_cooldown,
    record_transient_failure,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.episodic.note import MemoryNote
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.sleep.slot_edit_curator")

#: ``Provenance.extractor`` に刻む名前と版 (c_05 §0.5)。
EXTRACTOR_NAME = "SlotEditCurator"
EXTRACTOR_VERSION = 1

#: 1 サイクルで補助タスクへ出す上限 (1 回 45 秒、``PURPOSE_TIMEOUT_DEFAULTS``)。
#: 超過分はマーカーが立たないので次サイクルが拾う。
_MAX_PER_CYCLE = 4

#: ``slot_edit`` の出力トークン上限と生成温度。
_SLOT_EDIT_MAX_TOKENS = 384
_SLOT_EDIT_TEMPERATURE = 0.1

#: 判定に出す発話の最大文字数 (長文は 1 つの編集ではない)。
_MAX_CHARS = 400

#: 加える要素の最小 / 要素の最大文字数。
_MIN_ADDED_CHARS = 2
_MAX_MEMBER_CHARS = 30

#: 編集後の要素を並べる区切り。
_JOINER = "、"

#: ``curation_backoff`` のキー (aux purpose 名と同じ)。
_FAILURE_KEY = "slot_edit"

#: 属性を持たない受け皿スロット (編集の宛先にしない)。
_FALLBACK_SLUG = "user"

#: ``fact_attributes.yaml`` の fact_type → ``mem.<kind>``。本人の属性だけ。
_KIND_BY_FACT_TYPE: dict[str, str] = {
    "preference": "preference",
    "personal_fact": "personal",
}

#: 要素の間に残ってよい区切り記号 (語彙ではない)。
_SEPARATOR_RE = re.compile(r"[、，,・/／。．.!！?？&＆]")
#: 区切り記号を落とした後に要素の **間** に残ってよい並べの助詞 (「と」「や」
#: 「とか」)。2 文字以下のひらがなだけ — 漢字・カタカナ・英数字が残れば要素の
#: 取りこぼし (網羅していない)。
_JOINER_RUN_RE = re.compile(f"^[{HIRAGANA}]{{0,2}}$")
#: 先頭の残りが終わってよい主題・主格の助詞 (「休日は」「私の趣味は」)。
_LEADING_TOPIC_TAIL = "はがも"
#: 末尾の残りが始まってはいけない並べの助詞 (「と読書」は要素の取りこぼし)。
_TRAILING_JOINER_HEAD = "とや"

#: 内容語の文字 (漢字 / カタカナ / 英数字)。要素は内容語の連なりを核に持つ。
_CONTENT = f"{KANJI}{KANJI_MARKS}{KATAKANA_WORD}A-Za-z0-9"
#: 要素の形 — 内容語の連なりと、その間・後ろの 2 文字以下の送り仮名
#: (「釣り」「山登り」「お菓子作り」)。「加えて」「写真を」は下の末尾の検査で落ちる。
_MEMBER_SHAPE_RE = re.compile(f"^[{HIRAGANA}]?(?:[{_CONTENT}]+[{HIRAGANA}]{{0,2}})+$")
#: 要素の末尾に来てはいけない助詞・テ形 (「写真を」「加えて」)。
_PARTICLE_END = "をがにはでともへやてのかよねだ"
#: ひらがなで終わる要素の **直後** に続けば、要素は述語の語幹 (「加え|て」「外し|た」)。
_INFLECTION_NEXT = "てたるなまれ"

#: 文末の否定の依頼 (「外さないで」「加えないでください」) — 編集しないでほしい。
_NEGATED_REQUEST_END_RE = re.compile(
    r"ないで(?:ください|下さい|ほしい|欲しい|くれ|ね)?[。．.！!\s]*$",
)

#: 属性語の無い発話で、編集の対象の要素の候補 — 内容語の連なり (と 2 文字以下の
#: 送り仮名) を主題・目的・並列の助詞 (は / を / も) の直前に置いたもの (「キャンプは」
#: 「釣りを」)。連体 (「キャンプの道具」) は要素を修飾に使っているだけで対象にしない。
_MEMBER_TOPIC_RE = re.compile(
    f"(?<![{_CONTENT}])(?P<member>[{_CONTENT}]+[{HIRAGANA}]{{0,2}}?)(?=[はをも])",
)
_CONTENT_CHAR_RE = re.compile(f"[{_CONTENT}]")
#: 要素から引いたスロットの最小の要素の文字数 (1 文字の語は偶然に当たりやすい)。
_MIN_MEMBER_LOOKUP_CHARS = 2

#: 却下のうち、補助タスクの答え方の揺れで起きうるもの — 結論にせず
#: ``curation_backoff`` へ積んで次サイクルでもう一度聞く (閾値で cooldown)。
_RETRYABLE_REASONS = frozenset({"members_do_not_cover_current_value"})


@dataclass(frozen=True)
class EditTarget:
    """編集の宛先 — スロットと、そのスロットの編集前の値の行。"""

    subject: str
    fact_type: str
    slug: str
    fact: Any
    #: 発話でスロットに解決した属性語 (持ち主の検査に使う)。要素から引いたスロット
    #: (``by_member``) では発話の要素の語。
    slot_words: tuple[str, ...] = ()
    #: 属性語ではなく現在値の要素からスロットを引いた (:func:`slots_named_by_member`)。
    by_member: bool = False


@dataclass(frozen=True)
class SlotEditCheck:
    """門の結果。``ok`` が偽なら ``reason`` に却下理由。"""

    ok: bool
    reason: str
    members: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    added: tuple[str, ...] = ()


def _reject(reason: str) -> SlotEditCheck:
    return SlotEditCheck(ok=False, reason=reason)


def members_cover(current_text: str, members: list[str]) -> bool:
    """``members`` が現在値の逐語 span で、現在値を **網羅する** か (純粋関数)。

    各要素を現在値の (他の要素と重ならない) 位置に当て、当たらなかった文字の
    連なりを見る。補助タスクが要素を 1 つ言い落とすと、その要素は「外す」と
    言われていないのに編集後の値から黙って消える — それを防ぐ門。

    - 要素の **間** は区切りと 2 文字以下のひらがなだけ (「と」「や」「、」)
    - **先頭** は区切りを含まず主題・主格の助詞で終わる前置き (「休日は」) なら可
    - **末尾** は文末の語尾を落とした後 (:func:`strip_copula`)、区切りでも並べの
      助詞でもなく始まる述語 (「をしています」) なら可。「と読書」は取りこぼし
    """
    text = norm_span(strip_copula(current_text))
    if not text or not members:
        return False
    covered = [False] * len(text)
    for member in members:
        needle = norm_span(member)
        if not needle:
            return False
        start = text.find(needle)
        while start >= 0 and any(covered[start:start + len(needle)]):
            start = text.find(needle, start + 1)
        if start < 0:
            return False
        for i in range(start, start + len(needle)):
            covered[i] = True
    first = covered.index(True)
    last = len(covered) - 1 - covered[::-1].index(True)
    leading, trailing = text[:first], text[last + 1:]
    if leading and (
        _SEPARATOR_RE.search(leading) or leading[-1] not in _LEADING_TOPIC_TAIL
    ):
        return False
    if trailing and (
        _SEPARATOR_RE.match(trailing) or trailing[0] in _TRAILING_JOINER_HEAD
    ):
        return False
    gaps: list[str] = []
    run = ""
    for ch, done in zip(text[first:last + 1], covered[first:last + 1], strict=True):
        if done:
            if run:
                gaps.append(run)
            run = ""
        else:
            run += ch
    return all(_JOINER_RUN_RE.match(_SEPARATOR_RE.sub("", g)) for g in gaps)


def _member_shape_ok(key: str) -> bool:
    """正規化済みの要素が 1 つの名詞句の形か (純粋関数)。"""
    return bool(_MEMBER_SHAPE_RE.match(key)) and key[-1] not in _PARTICLE_END


def _is_predicate_stem(key: str, said: str) -> bool:
    """ひらがなで終わる ``key`` が、発話のどの出現でも活用語尾に続くか (純粋関数)。"""
    if not key or not re.match(f"[{HIRAGANA}]", key[-1]):
        return False
    nexts = []
    start = said.find(key)
    while start >= 0:
        end = start + len(key)
        nexts.append(said[end] if end < len(said) else "")
        start = said.find(key, start + 1)
    return bool(nexts) and all(ch and ch in _INFLECTION_NEXT for ch in nexts)


def _sentences_with(masked: str, keys: list[str]) -> list[str]:
    """``keys`` のどれかを含む文 (引用を落とした発話から、純粋関数)。"""
    from backend.free.core.intent_vocab import split_sentences

    return [
        s for s in split_sentences(masked)
        if any(k in norm_span(s) for k in keys)
    ]


def _mood_or_owner_reason(
    utterance: str, spans: list[str], slot_words: tuple[str, ...],
) -> str | None:
    """編集の span を持つ文が本人の編集の言明 / 依頼でなければ却下理由 (純粋関数)。

    補助タスクのプロンプトだけに任せず、コードで決められる範囲をここに持つ
    (訂正の ``restated_own_value`` と同じ考え方)。見るのは引用を落とした発話の、
    外す / 加える要素を含む文:

    - 問い (「加えたらどう？」「外そうか。」)・是非の相談 (「外すべきか迷っている」)
    - 否定の依頼 (「外さないで」)
    - 仮定・時間の対比・伝聞の標識 (:func:`marks_not_own_restatement`)
    - 属性語の持ち主が本人でない (「息子の趣味に」「妻は趣味に」) / 要素が本人以外の
      人を修飾する (「娘のサッカー」)
    """
    from backend.free.core.intent_vocab import (
        KA_QUESTION_END_RE,
        QUESTION_END_RE,
        is_practice_advice_query,
    )
    from backend.free.core.text_quality import (
        attribute_belongs_to_another_person,
        value_followed_by_another_persons_subject,
        value_modifies_another_person,
    )
    from backend.free.memory.notes.note_builder import _possessor_is_self

    if marks_not_own_restatement(utterance):
        return "hypothetical_or_hearsay"
    masked = mask_quoted_speech(utterance or "")
    sentences = _sentences_with(masked, [norm_span(s) for s in spans])
    if not sentences:
        return "span_only_in_quote"
    for sentence in sentences:
        tail = sentence.strip()
        if QUESTION_END_RE.search(tail) or KA_QUESTION_END_RE.search(tail):
            return "question"
        if is_practice_advice_query(sentence):
            return "deliberation"
        if _NEGATED_REQUEST_END_RE.search(tail):
            return "negated_request"
        if slot_words and attribute_belongs_to_another_person(sentence, slot_words):
            return "slot_of_another_person"
        for word in slot_words:
            pos = sentence.find(word)
            while pos >= 0:
                if not _possessor_is_self(sentence, pos, explicit_only=True):
                    return "slot_of_another_person"
                pos = sentence.find(word, pos + 1)
        for member in spans:
            if value_modifies_another_person(sentence, member) or (
                value_followed_by_another_persons_subject(sentence, member)
            ):
                return "member_of_another_person"
    return None


def _bare_predicate_after(utterance: str, member: str) -> bool:
    """``member`` を主題・目的語に置いた文の述語が、内容語を含まない形か (純粋関数)。

    属性語の無い発話で、要素について何かを述べただけの文 (「キャンプは雨で中止に
    なりました」「写真は後で送ります」「釣りは最近行けてない」) を、要素をやめた・外した
    という言明と区別する語彙は持たない (状態変化の語の SSOT が無い)。代わりに形で
    絞る — 「<要素>は / を / も」の後ろから文末までに内容語 (漢字・カタカナ・英数字) が
    無いこと (「キャンプはもうやめました。」「釣りはもうしていません」)。漢字で書いた
    述語 (「辞めました」) も落ちるが、落ちた側は値を残すだけ (#13)。
    """
    from backend.free.core.intent_vocab import split_sentences

    found = False
    for sentence in split_sentences(mask_quoted_speech(utterance or "")):
        for m in _MEMBER_TOPIC_RE.finditer(sentence):
            if m.group("member") != member:
                continue
            found = True
            if _CONTENT_CHAR_RE.search(sentence, m.end() + 1):
                return False
    return found


def check_slot_edit(
    current_text: str,
    utterance: str,
    *,
    is_edit: bool,
    current_members: list[str],
    removed: list[str],
    added: list[str],
    fact_type: str,
    slot_words: tuple[str, ...] = (),
    slot_named: bool = True,
    slug: str | None = None,
    triggers_dir: str | Path | None = None,
) -> SlotEditCheck:
    """補助タスクの答えを検証し、編集後の要素集合を返す (純粋関数に近い)。

    どれか 1 つでも門を通らなければ **編集全体を却下** する — 一部だけ適用すると
    「外す」が落ちて「加える」だけ通る、のような半端な値が現在値になる。発話の
    照合は引用の内側を落とした形で行う (伝聞は本人の編集ではない)。
    ``slot_named`` が偽 (発話が属性を名指さず、要素からスロットを引いた) なら、
    加える要素を持つ編集は却下する — 何の属性に足すのかを発話が言っていない。

    ``slug`` (編集するスロット) を渡すと、要素が属性語を含むかの検査で **その
    スロット自身の trigger は使わず**、発話がスロットを名指した語 (``slot_words``)
    で見る。food の trigger には値そのもの (ラーメン / パン / 肉) が並ぶので、
    trigger で見ると「ラーメン」という要素が属性語を含むことになる。名指した語の
    うち、要素に助詞を続けて現れるもの (「趣味に写真」「食べ物は」) と、外す /
    加える要素の span の外で発話に現れるもの (「好きな食べ物に」の 食べ物) を
    属性名とみなす — 外す要素そのもの (「ラーメンを外して」の ラーメン) は値。
    他のスロットの trigger は従来どおり見る。
    """
    from backend.free.memory.notes.note_builder import resolve_fact_attribute_matches

    said = norm_span(mask_quoted_speech(utterance or ""))
    outside = said
    for raw in (*removed, *added):
        key = norm_span(str(raw or ""))
        if key:
            outside = outside.replace(key, "\n")
    own_words = tuple(
        w for w in (norm_span(word) for word in slot_words) if w
    ) if slot_named else ()
    name_words = tuple(w for w in own_words if w in outside)

    def names_slot(text: str) -> bool:
        matches = resolve_fact_attribute_matches(
            text, fact_type, mode="chat", triggers_dir=triggers_dir,
        )
        if slug is None:
            return bool(matches)
        if any(s != slug for s, _ in matches):
            return True
        key = norm_span(text)
        return any(w in key for w in name_words) or any(
            re.search(f"{re.escape(w)}[{HIRAGANA}]", key) for w in own_words
        )

    if not is_edit:
        return _reject("not_an_edit")
    members: list[str] = []
    seen: set[str] = set()
    for raw in current_members:
        member = str(raw or "").strip()
        key = norm_span(member)
        if not key or key in seen:
            continue
        if len(key) > _MAX_MEMBER_CHARS:
            return _reject("member_too_long")
        if not _member_shape_ok(key) or names_slot(member):
            return _reject("member_is_not_a_noun_phrase")
        seen.add(key)
        members.append(member)
    if not members_cover(current_text, members):
        return _reject("members_do_not_cover_current_value")
    dropped: list[str] = []
    spans: list[str] = []
    for raw in removed:
        key = norm_span(str(raw or ""))
        if key not in seen:
            return _reject("removed_is_not_a_current_member")
        if key not in said:
            return _reject("removed_is_not_named_in_utterance")
        if key not in dropped:
            dropped.append(key)
            spans.append(str(raw).strip())
    new: list[str] = []
    for raw in added:
        member = str(raw or "").strip()
        key = norm_span(member)
        if key in seen:
            continue  # 既に要素 (言い直し) — 変化ではない
        if not (_MIN_ADDED_CHARS <= len(key) <= _MAX_MEMBER_CHARS):
            return _reject("added_length")
        if key not in said:
            return _reject("added_is_not_verbatim")
        if _SEPARATOR_RE.search(key):
            return _reject("added_spans_several_members")
        if any(other in key for other in seen):
            return _reject("added_contains_a_current_member")
        if names_slot(member):
            return _reject("added_carries_an_attribute_word")
        if not _member_shape_ok(key) or _is_predicate_stem(key, said):
            return _reject("added_is_not_a_noun_phrase")
        if member not in new:
            new.append(member)
            spans.append(member)
    if not dropped and not new:
        return _reject("no_change")
    if new and not slot_named:
        return _reject("added_without_the_slot_word")
    if not slot_named and not all(
        _bare_predicate_after(utterance, span) for span in spans
    ):
        return _reject("member_statement_is_not_a_bare_removal")
    reason = _mood_or_owner_reason(utterance, spans, slot_words)
    if reason is not None:
        return _reject(reason)
    result = [m for m in members if norm_span(m) not in dropped] + new
    if not result:
        # 全要素を外す編集は「値が無い」状態で、空の値は書けない。旧値を残す側に
        # 倒す (畳むと復旧できない、#13)。
        return _reject("empty_result")
    return SlotEditCheck(
        ok=True, reason="applied",
        members=tuple(result), removed=tuple(dropped), added=tuple(new),
    )


def _note_ids(fact: object) -> set[str]:
    return {
        str(prov.note_id)
        for prov in getattr(fact, "provenances", None) or ()
        if getattr(prov, "note_id", None)
    }


def _at(obj: object) -> float:
    return float(getattr(obj, "created_at", 0.0) or 0.0)


def find_edit_target(
    note: "MemoryNote",
    store: "SemanticFactStore",
    *,
    triggers_dir: str | Path | None = None,
    slots: list[tuple[str, str, str, tuple[str, ...]]] | None = None,
    by_member: bool = False,
) -> EditTarget | None:
    """ノートが編集しうるスロットと、その編集前の値の行 (規則だけで決める)。

    宛先は ``slots`` (省略時は :func:`editable_slots`。要素から引いたスロットなら
    ``by_member`` を立てて渡す) のうち、次をすべて満たす最初の 1 つ:

    - このノートがまだそのスロットに値を書いていない — 書いていれば言明として
      抽出済み (Step 8 / 8.3)、または編集を適用済み
    - そのスロットに、発話より **前** の live 行がある (編集する値がある)
    - 発話より **後** の live 行が無い (後の言い直しが既に現在値を決めている)

    発話より前の live 行が複数あれば (同じスロットの別の値。注入は 1 値へ畳む)、
    発話時刻が最も新しい行を宛先にし、同着は id の大きい方に決める (決定論)。
    """
    uttered_at = _at(note)
    if slots is None:
        slots = editable_slots(
            (getattr(note, "content", "") or "").strip(), triggers_dir=triggers_dir,
        )
    for fact_type, slug, subject, slot_words in slots:
        try:
            rows = store.search_by_subject(subject, include_superseded=True)
        except Exception as exc:  # noqa: BLE001 - 読めなければ宛先なし
            logger.warning("slot_edit_curator: failed to list %s: %s", subject, exc)
            continue
        if any(note.id in _note_ids(row) for row in rows):
            continue
        live = [row for row in rows if not getattr(row, "superseded_by", None)]
        if any(_at(row) > uttered_at for row in live):
            continue
        older = [row for row in live if _at(row) < uttered_at]
        if not older:
            continue
        return EditTarget(
            subject=subject, fact_type=fact_type, slug=slug,
            fact=max(older, key=lambda row: (_at(row), str(row.id))),
            slot_words=slot_words,
            by_member=by_member,
        )
    return None


def _member_in_value(value: str, member: str) -> bool:
    """``member`` が ``value`` に要素として (前後を内容語の文字に接さずに) 現れるか。"""
    start = value.find(member)
    while start >= 0:
        end = start + len(member)
        before = value[start - 1] if start else ""
        after = value[end] if end < len(value) else ""
        if not _CONTENT_CHAR_RE.match(before) and not _CONTENT_CHAR_RE.match(after):
            return True
        start = value.find(member, start + 1)
    return False


def slots_named_by_member(
    note: "MemoryNote",
    store: "SemanticFactStore",
    *,
    triggers_dir: str | Path | None = None,
) -> list[tuple[str, str, str, tuple[str, ...]]]:
    """属性語の無い発話の主題・目的語の名詞を現在値の要素に持つスロット (規則だけ)。

    「キャンプはもうやめました。」のキャンプが、編集の対象になりうる (:func:`editable_slots`
    と同じ除外の) スロットの、発話より前の最新の live 行の値に要素として現れるとき、
    そのスロットを返す。当たるスロットが **ちょうど 1 つ** のときだけ返し、複数なら
    宛先を決められないので空にする。返す組の属性語の欄は発話の要素の語 (持ち主の検査に
    使う)。
    """
    from backend.free.memory.notes.note_builder import (
        get_fact_attributes,
        is_multi_valued_subject,
        is_single_valued_subject,
        is_span_only_fold_subject,
        resolve_fact_attributes_path,
    )
    from backend.free.memory.notes.subject_ns import make_mem_subject

    said = mask_quoted_speech((getattr(note, "content", "") or "").strip())
    candidates = []
    for m in _MEMBER_TOPIC_RE.finditer(said):
        member = m.group("member")
        if len(norm_span(member)) >= _MIN_MEMBER_LOOKUP_CHARS and member not in candidates:
            candidates.append(member)
    if not candidates:
        return []
    uttered_at = _at(note)
    attrs = get_fact_attributes(resolve_fact_attributes_path(triggers_dir)).get("chat") or {}
    hits: list[tuple[str, str, str, tuple[str, ...]]] = []
    for fact_type, kind in _KIND_BY_FACT_TYPE.items():
        for spec in attrs.get(fact_type) or ():
            slug = spec.slug
            if slug == _FALLBACK_SLUG:
                continue
            subject = make_mem_subject(kind, slug)
            if (
                is_multi_valued_subject(subject, triggers_dir=triggers_dir)
                or is_span_only_fold_subject(subject, triggers_dir=triggers_dir)
                or is_single_valued_subject(subject, triggers_dir=triggers_dir)
            ):
                continue
            try:
                rows = store.search_by_subject(subject, include_superseded=True)
            except Exception as exc:  # noqa: BLE001 - 読めなければ当たりなし
                logger.warning("slot_edit_curator: failed to list %s: %s", subject, exc)
                continue
            # 現在値の行は find_edit_target と同じ選び方 (live = superseded_by 無し)
            live = [row for row in rows if not getattr(row, "superseded_by", None)]
            older = [row for row in live if _at(row) < uttered_at]
            if not older:
                continue
            latest = max(older, key=lambda row: (_at(row), str(row.id)))
            value = str(getattr(latest, "text", "") or "")
            members = tuple(c for c in candidates if _member_in_value(value, c))
            if members:
                hits.append((fact_type, slug, subject, members))
    if len(hits) > 1:
        logger.info(
            "slot_edit_curator: note %s names members of %d slots (%s); no target",
            getattr(note, "id", ""), len(hits), [h[2] for h in hits],
        )
        return []
    return hits


def editable_slots(
    content: str, *, triggers_dir: str | Path | None = None,
) -> list[tuple[str, str, str, tuple[str, ...]]]:
    """発話が属性語で解決する、編集の対象になりうるスロット (純粋関数)。

    ``(fact_type, slug, subject, 当たった属性語)`` の列。辞書は Step 8 と同じ
    ``fact_attributes.yaml``。除くもの:

    - 並列多値 (``multi_valued``) — 要素ごとに別の行を持つ。編集は訂正の経路
    - ``span_only_fold`` (name / birthday) — 他者の値が落ちうる
    - 単値 (``single_valued``、age / color) — 要素の集合ではない。言い直しは
      Step 8 が単値の畳みで扱う
    - 受け皿 (``user``)
    """
    from backend.free.memory.notes.note_builder import (
        is_multi_valued_subject,
        is_single_valued_subject,
        is_span_only_fold_subject,
        resolve_fact_attribute_matches,
    )
    from backend.free.memory.notes.subject_ns import make_mem_subject

    out: list[tuple[str, str, str, tuple[str, ...]]] = []
    for fact_type, kind in _KIND_BY_FACT_TYPE.items():
        for slug, words in resolve_fact_attribute_matches(
            content, fact_type, mode="chat", triggers_dir=triggers_dir,
        ):
            if slug == _FALLBACK_SLUG:
                continue
            subject = make_mem_subject(kind, slug)
            if (
                is_multi_valued_subject(subject, triggers_dir=triggers_dir)
                or is_span_only_fold_subject(subject, triggers_dir=triggers_dir)
                or is_single_valued_subject(subject, triggers_dir=triggers_dir)
            ):
                continue
            out.append((fact_type, slug, subject, tuple(words)))
    return out


def _eligible(note: "MemoryNote", now: float) -> bool:
    if getattr(note, "slot_edit_curated_at", None) is not None:
        return False
    if getattr(note, "source", "user") != "user" or getattr(note, "private", False):
        return False
    content = (getattr(note, "content", "") or "").strip()
    if not content or len(content) > _MAX_CHARS:
        return False
    return not in_cooldown(note, _FAILURE_KEY, now)


def build_prompt(
    meaning: str, current_value: str, utterance: str, *, slot_named: bool = True,
) -> str:
    """編集判定の user プロンプト (純粋関数)。

    ``slot_named`` が偽 (発話が属性を名指さず、要素からスロットを引いた) なら、要素に
    ついての出来事・予定 (中止になった / 後で送る / 最近行けていない) は編集ではない
    ことを足す — 属性を名指さない発話は、要素の語を別の意味で使っていることが多い。
    """
    unnamed = (
        "- 発話は属性「" + meaning + "」を名指していない。要素の語についての出来事・"
        "予定・頻度 (「〜は雨で中止になった」「〜は後で送る」「〜は最近行けていない」) は"
        "編集ではない。本人がその要素を属性として **やめた / もう持たない** と述べた"
        "ときだけ is_edit を true にし、added は空にすること。\n"
    ) if not slot_named else ""
    return (
        "ユーザー本人の属性「" + meaning + "」の現在の値と、ユーザーの発話が"
        "あります。発話が **この属性の値の要素を加える / 外す** (「〜を加えて」"
        "「〜は外して」「〜はやめた」「もう〜していない」) ことを述べているかを"
        "判定してください。\n"
        "- current_members: 現在の値を要素ごとに分けたもの。現在の値に現れる"
        "文字列をそのまま使い、要素を 1 つも省かないこと (助詞・区切りは含めない)。"
        "is_edit が false でも埋めること。\n"
        "- removed: 発話で外す・やめたと述べた要素。current_members の語をそのまま使う。\n"
        "- added: 発話で新しく加えると述べた要素。発話に現れる文字列をそのまま、"
        "要素の名詞だけ (属性名・助詞・動詞を含めない)。\n"
        "- 発話がこの属性の編集でなければ (物を外した話、他の人の属性、質問だけ、"
        "別の属性の話) is_edit を false にし、removed と added を空にすること。\n"
        + unnamed
        + f"\n現在の値: {current_value}\n発話: {utterance}\n"
    )


async def _ask(aux_client, prompt: str) -> dict:
    """補助タスクに編集を判定させる。例外は呼出側へ伝播する (一過性失敗の再試行)。"""
    parsed = await aux_client.generate_json(
        prompt,
        purpose="slot_edit",
        max_tokens=_SLOT_EDIT_MAX_TOKENS,
        temperature=_SLOT_EDIT_TEMPERATURE,
        response_schema=SlotEdit,
    )
    return parsed if isinstance(parsed, dict) else {}


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, str)]


async def curate_slot_edits(
    notes: list["MemoryNote"],
    *,
    store_provider: Callable[[str], "SemanticFactStore | None"] | None,
    aux_client,
    embedder: "EmbeddingBackend | None",
    profile_id: str = "default",
    triggers_dir: str | Path | None = None,
    now_provider: Callable[[], float] | None = None,
    should_pause: Callable[[], bool] | None = None,
    max_per_cycle: int = _MAX_PER_CYCLE,
) -> dict[str, int]:
    """集合値の属性を編集する発話を、編集後の値のファクトとして SemMem へ書く。

    ノートは発話順に処理する — 「写真を加えて」の次の「キャンプは外して」は、
    前の編集で書いた行を編集前の値として読む。

    Returns:
        ``{"applied", "judged", "rejected"}`` の件数 (死活監視用、c_07 §7.1)。
        ``judged`` は補助タスクが答えた件数、``rejected`` はそのうち門で却下した件数。
    """
    stats = {"applied": 0, "judged": 0, "rejected": 0}
    if aux_client is None or embedder is None or store_provider is None:
        logger.debug("slot_edit_curator: aux/embedder/store not wired, skipping")
        return stats
    store = store_provider("global")
    if store is None:
        logger.debug("slot_edit_curator: global store not available, skipping")
        return stats
    from backend.free.memory.sleep.personal_fact_curator import slot_meanings

    now_fn = now_provider or time.time
    now = now_fn()
    pending = sorted(
        (n for n in public_notes(notes) if _eligible(n, now)), key=_at,
    )
    meanings = slot_meanings(triggers_dir)
    asked = 0
    for note in pending:
        if should_pause is not None and should_pause():
            logger.info("slot_edit_curator paused for the user turn")
            break
        slots = editable_slots((note.content or "").strip(), triggers_dir=triggers_dir)
        by_member = not slots
        if by_member:
            # 属性語の無い発話は、主題・目的語の名詞を現在値の要素に持つスロットへ
            slots = slots_named_by_member(note, store, triggers_dir=triggers_dir)
        if not slots:
            # 本文は変わらないので、編集の対象になるスロットが無いノートは二度と
            # 見ない (毎サイクルの走査を省く)。スロットはあるが宛先の値がまだ
            # 無いノートは、値が後から書かれうるので印を付けない。
            note.slot_edit_curated_at = now_fn()
            continue
        target = find_edit_target(
            note, store, triggers_dir=triggers_dir, slots=slots, by_member=by_member,
        )
        if target is None:
            if by_member:
                # 要素から引いたスロットは発話より前の値があって当たったので、宛先が
                # 無いのは後の言い直し / 書込み済み — 後から変わらないので二度と見ない
                note.slot_edit_curated_at = now_fn()
            continue
        if asked >= max_per_cycle:
            logger.info(
                "slot_edit_curator: reached %d judgement(s) this cycle; the rest "
                "carry over", max_per_cycle,
            )
            break
        asked += 1
        content = (note.content or "").strip()
        current = str(getattr(target.fact, "text", "") or "")
        meaning = meanings.get((target.fact_type, target.slug)) or target.slug
        try:
            answer = await _ask(aux_client, build_prompt(
                meaning, current, content, slot_named=not target.by_member,
            ))
        except Exception as exc:  # noqa: BLE001 - 一過性失敗は次サイクルで再試行
            logger.warning("slot_edit_curator: judgement failed for note %s: %s", note.id, exc)
            record_transient_failure(
                note, _FAILURE_KEY, now_fn(),
                counts=not getattr(exc, "contended", False),
            )
            continue
        check = check_slot_edit(
            current, content,
            is_edit=bool(answer.get("is_edit")),
            current_members=_strings(answer.get("current_members")),
            removed=_strings(answer.get("removed")),
            added=_strings(answer.get("added")),
            fact_type=target.fact_type,
            slot_words=target.slot_words,
            slot_named=not target.by_member,
            slug=target.slug,
            triggers_dir=triggers_dir,
        )
        stats["judged"] += 1
        if not check.ok:
            stats["rejected"] += 1
            logger.info(
                "slot_edit_curator: note %s left %s as is (%s)",
                note.id, target.subject, check.reason,
            )
            if check.reason in _RETRYABLE_REASONS:
                # 要素分けの揺れ — 結論にせず次サイクルで聞き直す (閾値で cooldown)
                record_transient_failure(note, _FAILURE_KEY, now_fn())
                continue
        elif not await _apply_edit(
            store, note, target, check,
            embedder=embedder, profile_id=profile_id, now=now_fn(),
        ):
            # 書けなかった (埋め込み失敗等) — マーカーを立てず次サイクルで再試行
            record_transient_failure(note, _FAILURE_KEY, now_fn())
            continue
        else:
            note.slot_edit_slot = target.slug
            stats["applied"] += 1
        # 補助タスクが答えて結論が出たときだけ立てる (却下も結論)
        note.slot_edit_curated_at = now_fn()
        clear_failure(note, _FAILURE_KEY)
    return stats


async def _apply_edit(
    store: "SemanticFactStore",
    note: "MemoryNote",
    target: EditTarget,
    check: SlotEditCheck,
    *,
    embedder: "EmbeddingBackend",
    profile_id: str,
    now: float,
) -> bool:
    """編集後の値を書き、編集した値の行 (と同じ値の行) を畳む。"""
    from backend.free.memory.note_facts import fact_from_note
    from backend.free.memory.sleep.extraction import write_sleep_facts

    value = _JOINER.join(check.members)
    try:
        embedding = await embedder.embed([value], is_query=False)
    except Exception as exc:  # noqa: BLE001 - 埋め込めなければ書かない (次サイクルで再試行)
        logger.warning("slot_edit_curator: embedding failed for %s: %s", target.subject, exc)
        return False
    old = target.fact
    fact = fact_from_note(
        note,
        subject=target.subject,
        predicate=old.predicate,
        object_=value,
        type=old.type,
        scope="global",
        now=now,
        profile_id=profile_id,
        embedding=embedding[0] if len(embedding) else None,
        _extra={"source_note_id": note.id, "raw_utterance": note.content or ""},
    )
    prov = fact.provenances[0] if fact.provenances else None
    if prov is not None:
        prov.turn_id = getattr(note, "turn_id", "") or None
        prov.extractor = EXTRACTOR_NAME
        prov.extractor_version = EXTRACTOR_VERSION
    if not write_sleep_facts(store, [fact], label="slot_edit"):
        return False
    if fact.id not in note.extracted_fact_ids:
        note.extracted_fact_ids.append(fact.id)
    edited = norm_span(str(getattr(old, "text", "") or ""))
    folded = 0
    for row in store.search_by_subject(target.subject, include_superseded=False):
        if row.id == fact.id or row.predicate != fact.predicate or _at(row) > _at(fact):
            continue
        if row.id != old.id and norm_span(str(getattr(row, "text", "") or "")) != edited:
            continue
        try:
            store.supersede(row.id, fact.id)
            folded += 1
        except (KeyError, ValueError) as exc:
            logger.warning(
                "slot_edit_curator: failed to supersede %s -> %s: %s", row.id, fact.id, exc,
            )
    logger.info(
        "slot_edit_curator: note %s edited %s (removed=%s added=%s) -> %r; "
        "superseded %d row(s)",
        note.id, target.subject, list(check.removed), list(check.added), value, folded,
    )
    return True


__all__ = [
    "EditTarget",
    "SlotEditCheck",
    "build_prompt",
    "check_slot_edit",
    "curate_slot_edits",
    "editable_slots",
    "find_edit_target",
    "members_cover",
    "slots_named_by_member",
]
