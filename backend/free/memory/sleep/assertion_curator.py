"""Step 8.4: 型付けできなかった言明のキュレーター

``ChatExtractor`` (Step 8) は ``candidate_fact_tags`` のトリガ辞書で
FactType を決める。ところが **日本語の断定の大半は「です」で終わり**、
``world_fact`` のトリガ (``である`` / ``とは``) には掛からない。「です」を
トリガに足すと疑問文・依頼文まで全て候補になるため足せない。

さらに ``world_fact`` の subject は ``_world_fact_keyword`` が ASCII 英字を
1 文字以上要求するので、キーワードが日本語だけのノートは必ずスキップされる
(``_is_usable_world_keyword``)。``fact_attributes.yaml`` の JA→ASCII 辞書は
登録済みの話題しか拾えない。

実インシデント (2026-08-19 ライブ監査、4 テーマ 40 ターン): ユーザーが
「忘れないでください」と明示して伝えた「あさひプロジェクトの締切は9月30日
です。」「チームは私を含めて4人です。」が **どのファクトにもならず**、
新セッションでの想起に失敗した。ノートは pin されており Step 8 まで届いて
いた (``apply_session_caps`` は pinned を per-session cap から除外する =
設計上は抽出する意図) が、候補タグが 1 段手前で空だったため届いていなかった。

そこで **命名だけを補助タスクへ出す**。Step 8.5 (url) / 8.6
(executable_command) と同じ curator 型で、SemMem 書込は
sleep-time に閉じる (CLAUDE.md §6 不変則 #2)。

設計ポリシー:

- 新 FactType を追加せず ``world_fact`` を流用する (CLAUDE.md §3 / §6 #2)。
- subject = ``mem.world.assertion.<slug>``。**内容ハッシュを付けない**。
  ``extractors.chat._world_fact_subject_parts`` がハッシュを足すのは keyword が
  本文から拾った任意の語で衝突が怖いからだが、ここでの slug は補助タスクが
  「何についての言明か」を答えたものなので、**同じ話題の言い直しが同じ
  subject に並ぶ方が正しい**。並べば競合検出 (``(subject, predicate)``) が
  対にでき、訂正が supersede できる。
- 規則側で落とせるもの (assistant 発話 / コード断片 / 疑問形 / 依頼形 /
  既に型付けできたノート) は補助タスクへ**出す前に**落とす。モデルに
  拒否権を与えるのではなく、決定論で絞ってから命名だけ任せる。
- ``aux_client is None`` (degraded) では何もせず ``0`` を返す。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING

from backend.free.core.intent_vocab import NUMBER_LITERAL_RE
from backend.free.core.response_dates import complete_years, literal_dates, nearest_date
from backend.free.core.text_quality import (
    contradicts_asserted_value,
    detect_lang,
    states_own_intention,
    value_was_adopted,
)
from backend.free.llm.json_schemas import AssertionNaming
from backend.free.memory.corrections import correction_target, has_correction_shape
from backend.free.core.correction_verdict import strip_copula
from backend.free.memory.extractors.base import (
    OWN_VALUE_UPDATE_VERDICTS,
    note_is_verified_correction,
    note_verification_rejected,
    value_update_spans,
)
from backend.free.memory.notes.pin_detector import note_is_pinned
from backend.free.memory.note_facts import fact_from_note
from backend.free.memory.sleep._curator_common import public_notes
from backend.free.memory.sleep.extraction import write_sleep_facts
from backend.free.memory.sleep.curation_backoff import (
    clear_failure,
    in_cooldown,
    record_transient_failure,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.episodic.note import MemoryNote
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.sleep.assertion_curator")

_SUBJECT_PREFIX = "assertion"

#: 命名を依頼する言明の最小 / 最大文字数。短すぎる相槌はファクトにならず、
#: 長文はそもそも 1 つの言明ではない。
_MIN_CHARS = 8
_MAX_CHARS = 200

#: 1 サイクルで命名に出す上限。補助タスク 1 回 45 秒 (PURPOSE_TIMEOUT_DEFAULTS)
#: なので、アイドル窓を食い潰さないよう明示的に絞る。超過分は次サイクルへ回る
#: (``assertion_curated_at`` を立てないため)。
_MAX_PER_CYCLE = 8

#: slug に許す文字 (``subject_ns._SAFE_PART_RE`` と同じ規約)。
_SLUG_MAX_LEN = 40

#: ``curation_backoff`` の失敗カウンタキー (aux purpose 名と同じ)。
_FAILURE_KEY = "assertion_naming"


def _sanitize_slug(raw: str) -> str | None:
    """補助タスクが返した slug を subject part に使える形へ正規化する。

    ``subject_ns._SAFE_PART_RE`` は ASCII 英数字 / ``_`` / ``-`` のみ許し、
    先頭は英数字でなければならない。命名を任せている以上ここは信用せず、
    通らないものは ``None`` を返して呼出側でスキップさせる。
    """
    if not isinstance(raw, str):
        return None
    out = "".join(
        ch if ((ch.isascii() and ch.isalnum()) or ch in ("_", "-")) else "_"
        for ch in raw.strip().lower()
    ).strip("_-")
    while out and not (out[0].isascii() and out[0].isalnum()):
        out = out[1:]
    if not out:
        return None
    return out[:_SLUG_MAX_LEN]


def assertive_body(content: str) -> str | None:
    """発話から **言明の文だけ** を取り出して返す (純粋関数)。1 文も無ければ None。

    ノート全体の文末で判定してはいけない。「あさひプロジェクトの締切は9月30日
    です。**忘れないでください。**」は文末が依頼形だが、本体は言明そのもの
    である。しかも「忘れないでください」は pin トリガなので、**記憶しろという
    依頼ほど依頼形ゲートに掛かる**という逆転が起きる (2026-08-19 実測: 観測
    された取りこぼし 2 件のうち 1 件がこれで落ちた)。

    ``_tag_evidence_is_question_only`` が文単位で見ているのと同じ理由・同じ
    分割規則を使い、疑問形でも依頼形でもない文だけを命名対象として残す。
    """
    from backend.free.memory.extractors.chat import (
        _QUESTION_ENDING_RE,
        _REQUEST_ENDING_RE,
        _SENTENCE_SPLIT_RE,
    )

    kept = [
        sentence
        for raw in _SENTENCE_SPLIT_RE.split(content or "")
        if (sentence := raw.strip())
        and not _QUESTION_ENDING_RE.search(sentence)
        and not _REQUEST_ENDING_RE.search(sentence)
    ]
    return "".join(kept) or None


def world_assertion_body(content: str) -> str | None:
    """世界の事実として命名に出す文だけを返す (純粋関数)。1 文も無ければ None。

    :func:`assertive_body` (疑問形・依頼形を除いた文) から、さらに **話者自身の
    願望・意志** で終わる文 (「〜にしたいです」「〜するつもりです」「〜する予定
    です」「〜しようと思います」、:func:`~backend.free.core.text_quality.states_own_intention`)
    を除く。本人の予定・希望は世界の事実ではなく、``mem.world.assertion`` に
    すると subject が本人を指さないまま別の会話へ注入される (本人の属性の
    取りこぼしは Step 8.3 ``personal_fact_curator`` の担当)。

    実インシデント (2026-10-05 trace 8232204694b3): 「1日目は嵐山、2日目は東山に
    したいです。それぞれの見どころを3つずつ。」が
    ``mem.world.assertion.travel_itinerary`` になり、東京の観光の会話へ
    「(world_fact) … is: 1日目は嵐山、2日目は東山にしたい。」として注入された。
    判定は文法の閉じた類で、補助タスクへ出す **前に** 決める。
    """
    from backend.free.memory.extractors.chat import _SENTENCE_SPLIT_RE

    body = assertive_body(content)
    if body is None:
        return None
    kept = [
        sentence
        for raw in _SENTENCE_SPLIT_RE.split(body)
        if (sentence := raw.strip()) and not states_own_intention(sentence)
    ]
    return "".join(kept) or None


def _next_assistant_note(note: "MemoryNote", notes: list) -> "MemoryNote | None":
    """``note`` の直後にある同一セッションの assistant ノート (無ければ None)。"""
    at = float(getattr(note, "created_at", 0.0) or 0.0)
    session = getattr(note, "session_id", None)
    best = None
    best_at = None
    for other in notes:
        if getattr(other, "source", "user") != "assistant":
            continue
        if getattr(other, "session_id", None) != session:
            continue
        other_at = float(getattr(other, "created_at", 0.0) or 0.0)
        if other_at < at:
            continue
        if best_at is None or other_at < best_at:
            best, best_at = other, other_at
    return best


def _assistant_rejected_the_claim(note: "MemoryNote", notes: list) -> bool:
    """ユーザーが述べた値を、直後のアシスタント応答が **採らなかった** か。

    採らなかった主張を world_fact として残すと、アシスタントが会話では
    正しく反論しているのに記録側だけが誤りを永続化する。

    実インシデント (2026-08-27 ライブ監査): 「いや、それは間違いです。
    答えは 63800 ですよ。」(誤) が ``mem.world.assertion.correct_answer`` として
    live になった。アシスタントは同じ会話で 3 回とも 63802 を維持しており、
    subject 名が ``correct_answer`` なので次の算術質問で注入されうる状態だった。

    判定は学習層 (``FeedbackCollector._settle_pending_correction``) と **同じ
    条件・同じ述語** を使う:

    - ユーザー側に数値があり、直後のアシスタント応答にも数値がある
    - アシスタントが **自分の値を出し** (両者の数値が重ならない)
    - かつユーザーの値を **採用していない**
      (:func:`~backend.free.core.text_quality.value_was_adopted`。
      「約100kmという値は事実と異なります」のように打ち消しながら言及する
      ケースを出現だけで採用と誤判定しないため)

    判定材料が無いケース (アシスタントが「承知しました。」とだけ返した等) は
    ``False`` = 従来どおり curate する。**安全側は「残す」**。
    """
    reply = _next_assistant_note(note, notes)
    if reply is None:
        return False
    reply_text = reply.content or ""
    # 疑問形しか無い応答 (「10月1日からでよいですか？」) は反論ではなく確認。
    # 値が違っても却下と見なさない。
    if assertive_body(reply_text) is None:
        return False
    # 数値を含まない誤主張は数値集合では見えない。同じ話題に別の値が対置された
    # かを先に見る (2026-08-28 ライブ監査 T11-3:「日本の首都は大阪ですよね。」に
    # 「日本の首都は東京です。大阪は……首都ではありません。」と返しているのに
    # ``mem.world.assertion.capital_of_japan`` = 「日本の首都は大阪です。」が
    # live になった。2026-08-27 に入れた数値ゲートと同じ欠陥の非数値版)。
    if contradicts_asserted_value(note.content or "", reply_text):
        return True
    claimed = set(NUMBER_LITERAL_RE.findall(note.content or ""))
    if not claimed:
        return False
    prior = set(NUMBER_LITERAL_RE.findall(reply_text))
    if not (prior - claimed):
        # アシスタントが数値を出していない / ユーザーの値しか含まない
        # = 自分の値を対置していない。
        return False
    # 出現の有無で見てはいけない。「約100kmという値は事実と異なります」のように
    # **打ち消しながら言及する** ため、含まれていても採用とは限らない。
    return not value_was_adopted(reply_text, claimed)


def _is_curatable(note: "MemoryNote", builder) -> bool:
    """補助タスクへ命名を依頼してよいノートかを規則だけで判定する。

    Step 8 が型付けできたノートは対象外 — 本 curator は **取りこぼしだけ**を
    拾う純粋な追加であり、既存経路の判断を上書きしない。

    **ユーザー自身の属性の取りこぼしはここでは拾わない。** 「私は…」で
    ``personal_fact`` タグが付いた発話は ``candidate_fact_tags`` が非空になる
    ので下の分岐で落ちるが、それは正しい — 属性の取りこぼし (スロットが
    足りない / object が粗い) は Step 8.3
    (:mod:`~backend.free.memory.sleep.personal_fact_curator`) の担当で、
    そちらは ``mem.personal.<slot>`` へ属性単位で書くため訂正で supersede
    できる。ここで拾うと ``mem.world.assertion.<slug>`` の別 subject に
    落ちて、同じ属性の言い直しと対にならない。
    """
    from backend.free.memory.extractors.chat import _looks_like_code_fragment

    if getattr(note, "assertion_curated_at", None) is not None:
        return False
    if getattr(note, "source", "user") != "user":
        return False
    if getattr(note, "extracted_fact_ids", None):
        return False
    content = (note.content or "").strip()
    if not (_MIN_CHARS <= len(content) <= _MAX_CHARS):
        return False
    if _looks_like_code_fragment(content):
        return False
    # 言明の文が 1 つも無い (全文が疑問形 / 依頼形 / 本人の願望・意志) なら対象外。
    body = world_assertion_body(content)
    if body is None or len(body) < _MIN_CHARS:
        return False
    # Step 8 が型付けできたものは Step 8 に任せる。
    if builder.candidate_fact_tags(content):
        return False
    # ノート分類器が「事実を述べている」と見たもの、またはユーザーが明示的に
    # 覚えておけと言ったもの (pin) だけを対象にする。検証済みの自己訂正で旧値の
    # span が取れるものも対象 — 「試験日は4月18日ではなく4月25日でした。…計算し
    # 直してください」は分類器が task しか付けず、訂正がファクトにならないまま
    # 旧値の assertion が残った (2026-09-27 監査 F8)。字句の候補だけでは通さない (#12)。
    return bool(
        "fact" in (note.tags or [])
        or note_is_pinned(note)
        or _is_verified_self_restatement(note, content)
    )


def _replaces_a_value(note: "MemoryNote") -> bool:
    """訂正候補のノートが旧値を置き換える力 (slug の継承・旧値の span での畳み) を持つか。

    検証済みの訂正 (assistant / self) と、本人の値更新 (:data:`~backend.free.memory.extractors.base.OWN_VALUE_UPDATE_VERDICTS`)
    だけ。検証で他人の値と判った候補 (「取引先の創立記念日は6月1日ではなく6月2日だ
    そうです」= ``third_party``) が本人の言明の slug を継いで「当社の創立記念日は
    2001年6月1日です。」を畳んでいた (2026-09-28 独立レビュー、不変則 #12)。
    """
    return note_is_verified_correction(note) or (
        str(getattr(note, "correction_verdict", "") or "") in OWN_VALUE_UPDATE_VERDICTS
    )


def _is_verified_self_restatement(note: "MemoryNote", content: str) -> bool:
    """検証済み ``self`` の訂正で、「X ではなく Y」の旧値 span が取れるノートか。"""
    return (
        str(getattr(note, "correction_verdict", "") or "") == "self"
        and value_update_spans(content) is not None
    )


def _old_value_target(
    note: "MemoryNote", notes: list, span: str,
) -> "MemoryNote | None":
    """旧値の span を逐語で述べ、**話題語も共有する** 同じセッションの前の命名済み
    ユーザーノート (純粋関数)。

    span の文字列一致だけでは話題を特定できない — 「18日」は試験日にも給料日にも
    在り、給料日の slug を継いで給料日を畳んだ (2026-09-27 独立レビュー H2)。
    訂正と keyword を 1 語以上共有する候補がちょうど 1 つのときだけ返し、0 件や
    複数なら ``None`` (呼出側は語の重なり ``correction_target`` へ戻る)。
    """
    from backend.free.core.correction_verdict import norm_span

    needle = norm_span(span)
    if not needle:
        return None
    at = float(getattr(note, "created_at", 0.0) or 0.0)
    session = getattr(note, "session_id", None)
    topic = set(getattr(note, "keywords", None) or ())
    found = []
    for other in notes:
        if other is note or getattr(other, "source", "user") != "user":
            continue
        if getattr(other, "session_id", None) != session:
            continue
        if not getattr(other, "assertion_slug", None):
            continue
        other_at = float(getattr(other, "created_at", 0.0) or 0.0)
        if other_at >= at or needle not in norm_span(other.content or ""):
            continue
        if topic & set(getattr(other, "keywords", None) or ()):
            found.append(other)
    return found[0] if len(found) == 1 else None


def _old_value_span(note: "MemoryNote", content: str) -> str:
    """訂正 / 値更新のノートが置き換える旧値の span (無ければ空文字)。

    検証済みの訂正なら検証器の ``wrong_claim`` (門を通った逐語 span)、無ければ
    発話の「X ではなく Y」の X (本人の値更新、``value_update_spans`` と同じ分解)。
    """
    if note_is_verified_correction(note):
        wrong = strip_copula(str(getattr(note, "correction_wrong_claim", "") or ""))
        if wrong:
            return wrong
    pair = value_update_spans(content)
    return pair[0] if pair else ""


def _build_prompt(content: str) -> str:
    """命名用 user プロンプトを組み立てる (純粋関数)。"""
    return (
        "次の発話は、ユーザーが述べた事実の言明かどうかを判定し、"
        "言明なら「何についての言明か」を表す短い英語の slug を付けてください。\n"
        "slug は ASCII 英小文字・数字・アンダースコアのみ (例: project_deadline, "
        "team_size, office_location)。\n"
        "object には言明の内容を 1 文で簡潔に書き直してください。object は "
        "**UTTERANCE と同じ言語** で書くこと (日本語の発話なら日本語。英語にしない。"
        "英語なのは slug だけ)。日付・数値は発話の表記のまま写し、年を省いたり"
        "発話に無い年を足したりしないこと。言い直し・訂正なら新しい値を述べること。\n"
        "質問・依頼・相槌など、事実の言明でないものは is_assertion=false にしてください。\n"
        f"\nUTTERANCE: {content}\n"
    )


def _object_departs(obj: str, body: str, anchor: date | None) -> str | None:
    """補助タスクの object が発話から外れていればその理由を返す (純粋関数)。

    object は補助タスクの書き直しで、発話の言語も日付も保証されない
    (2026-09-28 再監査 R5: 日本語の訂正「試験日は4月18日ではなく4月25日でした」が
    「The exam date is April 25, not April 18.」になり、年も落ちた)。コードで
    確かめられる 2 点を確かめる:

    - 言語: :func:`detect_lang` が発話と違う
    - 日付: object の日付が発話に無い / 発話の年を落とした・変えた / 発話に無い年を
      足した (``anchor`` から :func:`complete_years` が足す年と同じなら可) /
      発話の日付を 1 つも残さない
    """
    body_lang = detect_lang(body)
    if body_lang and detect_lang(obj) != body_lang:
        return "language"
    body_dates = literal_dates(body)
    obj_dates = literal_dates(obj)
    if body_dates and not obj_dates:
        return "date_lost"
    for year, month, day in obj_dates:
        years = {y for y, m, d in body_dates if (m, d) == (month, day)}
        if not years:
            return "date_changed"
        if year in years:
            continue
        if year is not None and None in years and anchor is not None:
            resolved = nearest_date(month, day, anchor)
            if resolved is not None and resolved.year == year:
                continue
        return "year_changed"
    return None


def _year_anchor(
    note: "MemoryNote", target: "MemoryNote | None", content: str,
) -> date | None:
    """訂正が言い直さない年を解く基準 = **置き換える旧値の日付** (純粋関数)。

    - 検証済みの訂正 (assistant / self) だけ。検証で退けられたノート (第三者の
      日付・前提の変更) に宛先の年を足すと、他人の日付に自分の年が付く
      (2026-09-28 独立レビュー P2-1)。
    - 旧値の span (``_old_value_span``) の月日と一致する、宛先ノートの年つき日付が
      **ちょうど 1 つ** のときだけ。宛先の年つき日付をすべて基準にすると、別の
      日付 (2026-04-01) の年が混ざる (P3-2)。span 自体が年を持てばそれを使う。
    """
    if not note_is_verified_correction(note):
        return None
    old = literal_dates(_old_value_span(note, content))
    if len(old) != 1:
        return None
    year, month, day = old[0]
    if year is None:
        years = {
            y for y, m, d in literal_dates(getattr(target, "content", "") or "")
            if y is not None and (m, d) == (month, day)
        }
        if len(years) != 1:
            return None
        year = years.pop()
    try:
        return date(year, month, day)
    except ValueError:
        return None


async def _name_assertion(aux_client, content: str) -> tuple[str, str] | None:
    """補助タスクに ``(slug, object)`` を付けさせる。

    ``None`` は「答えたが言明でない / slug が使えない」場合。**例外は呼出側へ
    伝播する** — ここで握り潰すと一過性の aux timeout も「LLM が答えた」と
    同じ扱いになり、``curate_assertion_facts`` が冪等マーカーを立てて二度と
    再試行しなくなる (2026-09-08 監査 G-03)。
    """
    parsed = await aux_client.generate_json(
        _build_prompt(content),
        purpose="assertion_naming",
        max_tokens=256,
        temperature=0.1,
        response_schema=AssertionNaming,
    )
    if not isinstance(parsed, dict):
        logger.debug("assertion_curator: unexpected payload type: %r", type(parsed))
        return None
    if not parsed.get("is_assertion"):
        return None
    slug = _sanitize_slug(parsed.get("slug", ""))
    if slug is None:
        logger.debug("assertion_curator: unusable slug %r", parsed.get("slug"))
        return None
    obj = str(parsed.get("object") or "").strip() or content
    return slug, obj


async def curate_assertion_facts(
    notes: list["MemoryNote"],
    *,
    store_provider: Callable[[str], "SemanticFactStore | None"] | None,
    aux_client,
    embedder: "EmbeddingBackend | None",
    builder=None,
    profile_id: str = "default",
    now_provider: Callable[[], float] | None = None,
    should_pause: Callable[[], bool] | None = None,
    max_per_cycle: int = _MAX_PER_CYCLE,
) -> int:
    """型付けできなかった言明を ``world_fact`` として sleep-time で書き込む。

    Args:
        notes: 直近の MemoryNote 群 (通常 ``EpisodicWorkspace.notes.values()``)。
        store_provider: ``scope -> SemanticFactStore | None`` のコールバック。
        aux_client: 命名に使う補助タスククライアント。``None`` なら no-op。
        embedder: fact embedding 生成用。``None`` なら no-op。
        builder: ``ChatNoteBuilder`` (候補タグ判定用)。省略時は既定を作る。
        profile_id: 書込先 fact の profile_id。
        now_provider: 時刻供給。テスト用。
        should_pause: ``True`` を返したらノート境界でループを打ち切る協調
            yield。残りのノートは ``assertion_curated_at`` が立たないままなので
            次サイクルが拾う。
        max_per_cycle: 1 サイクルの上限 (sleep-time が実測 tps で伸縮する、c_16 §7.2.3)。

    Returns:
        新規に書き込まれた fact 件数。
    """
    if aux_client is None:
        logger.debug("assertion_curator: aux_client is None, skipping")
        return 0
    if embedder is None:
        logger.debug("assertion_curator: embedder is None, skipping")
        return 0
    if store_provider is None:
        logger.debug("assertion_curator: store_provider is None, skipping")
        return 0
    store = store_provider("global")
    if store is None:
        logger.debug("assertion_curator: global store not available, skipping")
        return 0

    if builder is None:
        from backend.free.memory.notes.note_builder import ChatNoteBuilder

        builder = ChatNoteBuilder()

    now_fn = now_provider or time.time
    now = now_fn()

    # private セッション由来のノートは SemMem へ昇格させない。
    # (``_curator_common.public_notes`` の docstring に実害と経緯)
    notes = public_notes(notes)

    candidates = [
        n for n in notes
        if _is_curatable(n, builder)
        # アシスタントが採らなかった主張は永続化しない
        # (_assistant_rejected_the_claim の docstring 参照)。
        and not _assistant_rejected_the_claim(n, notes)
        and not in_cooldown(n, _FAILURE_KEY, now)
    ]
    if not candidates:
        return 0
    candidates.sort(key=lambda n: float(getattr(n, "created_at", 0.0) or 0.0))
    if len(candidates) > max_per_cycle:
        logger.info(
            "assertion_curator: %d candidate(s), naming the oldest %d this cycle "
            "(the rest carry over)", len(candidates), max_per_cycle,
        )
        candidates = candidates[:max_per_cycle]

    written = 0
    all_notes = list(notes)
    for idx, note in enumerate(candidates):
        # 協調 yield: チャット生成が走っている間はノート境界で手を止める
        # (note_evolver と同じ実測。CLAUDE.md 不変則 #1)。
        if should_pause is not None and should_pause():
            remaining = len(candidates) - idx
            logger.info(
                "assertion_curator paused for the user turn: %d note(s) "
                "left pending for the next cycle", remaining,
            )
            break
        content = (note.content or "").strip()
        # 訂正候補は **検証済みイベントとしてのみ消費する** (CLAUDE.md §6 #12)。
        # 未検証のまま命名すると「正しい年は2018年ではなく2020年です」が
        # ``mem.world.assertion.year_correction`` になり、宛先 (employer の
        # 現在値) を畳めず誤りの値まで世界の事実として残る (2026-09-10 (h)
        # H-10、Step 8.0 の aux がチャットに横取りされた直後の Full で実測)。
        # 未検証はマークせず次サイクルへ持ち越し、検証で訂正でないと判った
        # ものは通常の言明として進む。検証済みでも宛先が言明でない (属性
        # スロットの訂正) ものは Step 8 の値アンカーが受け持つのでここでは
        # 書かない。
        if has_correction_shape(note):
            if not note_is_verified_correction(note):
                if not note_verification_rejected(note):
                    logger.debug(
                        "assertion_curator: correction candidate %s awaits "
                        "verification (Step 8.0); carried over", note.id,
                    )
                    continue
                if str(getattr(note, "correction_verdict", "") or "") == "disputed":
                    # アシスタントがその場で退けた値は世界の事実として書かない
                    # (「302 は恒久的な移転」が assertion に残った、2026-09-11 (j) J-04)。
                    note.assertion_curated_at = now_fn()
                    logger.info(
                        "assertion_curator: %s was disputed by the reply; not written",
                        note.id,
                    )
                    continue
            elif _correction_targets_attribute(store, note):
                # 誤りの span がユーザー属性の live 値に当たる訂正は Step 8 の
                # 値アンカーが employer / location 等のスロットへ書く。ここで
                # 世界の事実として二重に持たない。
                note.assertion_curated_at = now_fn()
                logger.debug(
                    "assertion_curator: verified correction %s targets a user "
                    "attribute; left to the attribute path", note.id,
                )
                continue
        # 命名には言明の文だけを渡す。「忘れないでください」等の依頼節が
        # 混ざると補助タスクが is_assertion=false に倒れる。
        body = world_assertion_body(content) or content
        try:
            named = await _name_assertion(aux_client, body)
        except Exception as exc:
            # 一過性失敗 (aux timeout 等) はマーカーを立てず、次サイクルで
            # 再試行させる (2026-09-08 監査 G-03)。
            logger.warning(
                "assertion_curator: naming failed for note %s: %s", note.id, exc,
            )
            record_transient_failure(
                note, _FAILURE_KEY, now_fn(),
                counts=not getattr(exc, "contended", False),
            )
            continue
        # 命名できなかった / 言明でないと判定された場合もマークする。同じ
        # ノートを毎サイクル補助タスクへ出し続けないため (url_curator が
        # 全分岐で url_curated_at を立てるのと同じ理由)。これは「補助タスクが
        # 答えた」場合であって、失敗ではない。
        note.assertion_curated_at = now_fn()
        clear_failure(note, _FAILURE_KEY)
        if named is None:
            continue
        slug, obj = named
        target = None
        # 訂正は **対象と同じ slug** を継ぐ。訂正は話題語を落として言うため
        # 単独で命名させると別 slug になり (実測 2026-08-20:
        # ``asahi_project_deadline`` に対し訂正が ``deadline_change``)、
        # subject が分かれて競合検出が対にできず supersede できない。
        if getattr(note, "is_correction", False) and _replaces_a_value(note):
            # 旧値を逐語で述べたノートが宛先 (語の重なりより確か)。無ければ従来の
            # 語の重なり。語の重なりだけだと、同じ語を持つ後の問い (「試験日までの
            # 総勉強時間は？」) が選ばれて slug を継げない (2026-09-27 監査 F8)。
            target = _old_value_target(
                note, all_notes, _old_value_span(note, content),
            ) or correction_target(note, all_notes)
            inherited = getattr(target, "assertion_slug", None) if target else None
            if inherited:
                logger.info(
                    "assertion_curator: correction %s inherits slug %s from %s",
                    note.id, inherited, getattr(target, "id", "?"),
                )
                slug = inherited
        note.assertion_slug = slug
        # 本文は発話の言語と日付を保つ。外れた書き直しは採らず発話の言明文へ戻し、
        # 訂正が言い直さない年は宛先の日付から足す (2026-09-28 再監査 R5)。
        anchor = _year_anchor(note, target, content)
        departure = _object_departs(obj, body, anchor)
        if departure is not None:
            logger.info(
                "assertion_curator: named object for note %s departs from the "
                "utterance (%s); keeping the utterance", note.id, departure,
            )
            obj = body
        obj = complete_years(obj, anchor)
        subject = f"mem.world.{_SUBJECT_PREFIX}.{slug}"
        # 同じ命題が既に live なら積まない。命名は補助タスクが「何についての
        # 言明か」を答えるので、言い直し・別セッションの再言明が同じ
        # (subject, object) に収束する。読出しは claim_key で畳むが、書き手が
        # 積み続けると 1 文が 5 件並ぶ (2026-09-09 ライブ監査 P-3:
        # holiday_exclusion ×5、同一セッション内でも copyright_clause ×2)。
        # 訂正は畳む対象 (旧値) を持つので除外しない。
        if not getattr(note, "is_correction", False) and _same_claim_is_live(
            store, subject, obj,
        ):
            note.extracted_fact_ids = list(
                getattr(note, "extracted_fact_ids", None) or [],
            )
            logger.debug(
                "assertion_curator: %s already holds the same claim; note %s not re-added",
                subject, note.id,
            )
            continue
        try:
            embedding = await embedder.embed([obj], is_query=False)
            vec = embedding[0] if len(embedding) else None
            # アシスタントの **知識の回答** に対するユーザーの訂正は、どちらが
            # 正しいかをシステムが確かめられない主張。世界の事実として断定せず
            # ``unverified`` (注入では「未確認」のラベル) で残す — アシスタントが
            # 一度は折れて「おっしゃる通り」と答えた誤った訂正 (「302 は恒久的な
            # 移転」) が (過去の記録) として素通りしていた (2026-09-11 (j) J-08)。
            veracity = (
                "unverified"
                if str(getattr(note, "correction_verdict", "") or "") == "assistant"
                else "stated"
            )
            fact = fact_from_note(
                note,
                subject=subject,
                predicate="is",
                object_=obj,
                type="world_fact",
                scope="global",
                now=now_fn(),
                profile_id=profile_id,
                embedding=vec,
                veracity=veracity,
                # 訂正の力 (時刻を跨いだ supersede) は検証済みの訂正だけ (#12)。
                from_correction=note_is_verified_correction(note),
                _extra={"source_note_id": note.id, "raw_utterance": content},
            )
            # 畳む範囲は旧値の span で絞る (#13)。assertion は多値が既定なので、
            # span の無い訂正は兄弟を畳まない (共通入口 write_sleep_facts が畳む)。
            if has_correction_shape(note) and _replaces_a_value(note):
                fact.value_update = _old_value_span(note, content)  # type: ignore[attr-defined]
            if not write_sleep_facts(store, [fact], label="assertion"):
                continue
            written += 1
            logger.info(
                "assertion_curator: wrote mem.world.%s.%s from note %s",
                _SUBJECT_PREFIX, slug, note.id,
            )
        except Exception as exc:
            logger.warning(
                "assertion_curator: persist failed for slug=%s: %s", slug, exc,
            )
    return written


def _correction_targets_attribute(store: "SemanticFactStore", note: object) -> bool:
    """検証済み訂正の ``wrong_claim`` がユーザー属性 (``mem.personal.*`` /
    ``mem.preference.*``) の live 値に逐語で含まれるか。"""
    from backend.free.core.correction_verdict import norm_span

    wrong = norm_span(str(getattr(note, "correction_wrong_claim", "") or ""))
    if not wrong:
        return False
    search = getattr(store, "search_by_pillar_prefix", None)
    if search is None:
        return False
    for prefix in ("mem.personal.", "mem.preference."):
        try:
            facts = search(prefix, include_superseded=False)
        except Exception:  # ストアの状態に依存しない (取れなければ無い扱い)
            continue
        if any(wrong in norm_span(getattr(f, "object", "") or "") for f in facts):
            return True
    return False


def _same_claim_is_live(store: "SemanticFactStore", subject: str, obj: str) -> bool:
    """同じ (subject, object) の live なファクトが既にあるか (純粋関数に近い)。"""
    try:
        siblings = store.search_by_subject(subject, include_superseded=False)
    except Exception:  # ストアの状態に依存しない (取れなければ無い扱い)
        return False
    wanted = (obj or "").strip()
    return any(
        (getattr(f, "object", "") or "").strip() == wanted
        and getattr(f, "predicate", "is") == "is"
        for f in siblings
    )
