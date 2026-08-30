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
from typing import TYPE_CHECKING

from backend.free.core.intent_vocab import NUMBER_LITERAL_RE
from backend.free.core.text_quality import (
    contradicts_asserted_value,
    value_was_adopted,
)
from backend.free.llm.json_schemas import AssertionNaming
from backend.free.memory.corrections import correction_target
from backend.free.memory.types import make_fact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.stores.short_term import MemoryNote
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
    # 言明の文が 1 つも無い (全文が疑問形 / 依頼形) なら対象外。
    body = assertive_body(content)
    if body is None or len(body) < _MIN_CHARS:
        return False
    # Step 8 が型付けできたものは Step 8 に任せる。
    if builder.candidate_fact_tags(content):
        return False
    # ノート分類器が「事実を述べている」と見たもの、またはユーザーが明示的に
    # 覚えておけと言ったもの (pin) だけを対象にする。
    return bool("fact" in (note.tags or []) or note.pin_flag)


def _build_prompt(content: str) -> str:
    """命名用 user プロンプトを組み立てる (純粋関数)。"""
    return (
        "次の発話は、ユーザーが述べた事実の言明かどうかを判定し、"
        "言明なら「何についての言明か」を表す短い英語の slug を付けてください。\n"
        "slug は ASCII 英小文字・数字・アンダースコアのみ (例: project_deadline, "
        "team_size, office_location)。\n"
        "object には言明の内容を 1 文で簡潔に書き直してください (原文の言語のまま)。\n"
        "質問・依頼・相槌など、事実の言明でないものは is_assertion=false にしてください。\n"
        f"\nUTTERANCE: {content}\n"
    )


async def _name_assertion(aux_client, content: str) -> tuple[str, str] | None:
    """補助タスクに ``(slug, object)`` を付けさせる。失敗時は ``None``。"""
    try:
        parsed = await aux_client.generate_json(
            _build_prompt(content),
            purpose="assertion_naming",
            max_tokens=256,
            temperature=0.1,
            response_schema=AssertionNaming,
        )
    except Exception as exc:
        logger.warning("assertion_curator: naming failed: %s", exc)
        return None
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
) -> int:
    """型付けできなかった言明を ``world_fact`` として sleep-time で書き込む。

    Args:
        notes: 直近の MemoryNote 群 (通常 ``ShortTermMemory.notes.values()``)。
        store_provider: ``scope -> SemanticFactStore | None`` のコールバック。
        aux_client: 命名に使う補助タスククライアント。``None`` なら no-op。
        embedder: fact embedding 生成用。``None`` なら no-op。
        builder: ``ChatNoteBuilder`` (候補タグ判定用)。省略時は既定を作る。
        profile_id: 書込先 fact の profile_id。
        now_provider: 時刻供給。テスト用。

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

    candidates = [
        n for n in notes
        if _is_curatable(n, builder)
        # アシスタントが採らなかった主張は永続化しない
        # (_assistant_rejected_the_claim の docstring 参照)。
        and not _assistant_rejected_the_claim(n, notes)
    ]
    if not candidates:
        return 0
    candidates.sort(key=lambda n: float(getattr(n, "created_at", 0.0) or 0.0))
    if len(candidates) > _MAX_PER_CYCLE:
        logger.info(
            "assertion_curator: %d candidate(s), naming the oldest %d this cycle "
            "(the rest carry over)", len(candidates), _MAX_PER_CYCLE,
        )
        candidates = candidates[:_MAX_PER_CYCLE]

    now_fn = now_provider or time.time
    written = 0
    all_notes = list(notes)
    for note in candidates:
        content = (note.content or "").strip()
        # 命名には言明の文だけを渡す。「忘れないでください」等の依頼節が
        # 混ざると補助タスクが is_assertion=false に倒れる。
        body = assertive_body(content) or content
        named = await _name_assertion(aux_client, body)
        # 命名できなかった / 言明でないと判定された場合もマークする。同じ
        # ノートを毎サイクル補助タスクへ出し続けないため (url_curator が
        # 全分岐で url_curated_at を立てるのと同じ理由)。
        note.assertion_curated_at = now_fn()
        if named is None:
            continue
        slug, obj = named
        # 訂正は **対象と同じ slug** を継ぐ。訂正は話題語を落として言うため
        # 単独で命名させると別 slug になり (実測 2026-08-20:
        # ``asahi_project_deadline`` に対し訂正が ``deadline_change``)、
        # subject が分かれて競合検出が対にできず supersede できない。
        if getattr(note, "is_correction", False):
            target = correction_target(note, all_notes)
            inherited = getattr(target, "assertion_slug", None) if target else None
            if inherited:
                logger.info(
                    "assertion_curator: correction %s inherits slug %s from %s",
                    note.id, inherited, getattr(target, "id", "?"),
                )
                slug = inherited
        note.assertion_slug = slug
        try:
            embedding = await embedder.embed([obj], is_query=False)
            vec = embedding[0] if len(embedding) else None
            fact = make_fact(
                subject=f"mem.world.{_SUBJECT_PREFIX}.{slug}",
                predicate="is",
                object_=obj,
                type="world_fact",
                scope="global",
                now=now_fn(),
                profile_id=profile_id,
                embedding=vec,
                _extra={"source_note_id": note.id, "raw_utterance": content},
            )
            store.add_fact(fact)
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
