"""Step 8.3: 自己開示発話の **属性分割** キュレーター

Step 8 (``ChatExtractor``) は ``fact_attributes.yaml`` のトリガ語で属性を
型付けする。この設計は「語形が 1 つ外れると、その属性のファクトが 0 件になる」
という失敗様式を構造として持っており、過去 2 か月で **4 回** 同じ形の事故を
出している (2026-08-31 / 2026-09-04 ×2 / 2026-09-08)。しかも取りこぼしは
黙って起きるうえ、残った属性が文末まで飲み込むので object まで汚れる。

実インシデント (2026-09-08 ライブ監査 T01#1)::

    「はじめまして。私は久我山蒼真（くがやま そうま）といいます。金沢市の
      野々市寄りに住んでいて、印刷会社で色管理と製版ワークフローのソフト
      開発をしています。よろしくお願いします。」

    → SemMem に入ったのは 1 件だけ:
      mem.personal.location states:
        「金沢市の野々市寄りに住んでいて、印刷会社で…開発をしています」
      name も occupation も無く、前の人物の氏名が live のまま復唱された。

そこで **regex が型付けし、LLM が分割・補完する** 二段構えにする。語彙を
足す対処は必ずまた漏れるが、この段は語形に依存しない。

規約:

- 走るのは sleep-time だけ (CLAUDE.md §6 不変則 #2 / c_16 §2.1)。
- **slot は ``fact_attributes.yaml`` のキー集合で検証** する (enum 相当)。
  新しいスロットは生えない。
- **value は発話の逐語 span であることをコード側で検証** する (空白を無視した
  部分一致)。含まれない値は捨てる — 幻覚した値が「ユーザーの属性」として
  永続化されるのが最悪の失敗なので、モデルの出力は素材ではなく **候補** として
  扱い、決定論で門を作る。
- 書き込みは Step 8 と同じ :func:`~backend.free.memory.sleep.extraction.
  persist_facts` を通す。supersede / ``extracted_fact_ids`` / ID 連鎖の扱いが
  regex 経路と 1 ミリも変わらないようにするため。
- ``aux_client`` / ``embedder`` / store が無い (degraded) 場合は no-op。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.free.llm.json_schemas import PersonalFactSplit
from backend.free.memory.note_facts import fact_from_note
from backend.free.memory.notes.note_builder import (
    MAX_ATTRIBUTES_PER_TEXT,
    get_fact_attributes,
    resolve_fact_attributes_path,
)
from backend.free.memory.notes.subject_ns import make_mem_subject
from backend.free.memory.sleep._curator_common import public_notes
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

logger = get_logger("memory.sleep.personal_fact_curator")

#: ``Provenance.extractor`` に刻む名前と版。抽出規則を変えたら版を上げる
#: (「どの版が作ったファクトか」で選り分けて再導出するため / c_05 §0.5)。
EXTRACTOR_NAME = "PersonalFactCurator"
EXTRACTOR_VERSION = 1

#: 分割を依頼する発話の最小 / 最大文字数。短い相槌は分ける余地が無く、
#: 長文は 1 つの自己紹介ではない。
_MIN_CHARS = 12
_MAX_CHARS = 400

#: 1 サイクルで分割に出す上限。補助タスク 1 回 60 秒
#: (``PURPOSE_TIMEOUT_DEFAULTS["personal_fact_split"]``) なのでアイドル窓を
#: 食い潰さないよう明示的に絞る。超過分は次サイクルへ回る
#: (``personal_fact_curated_at`` を立てないため)。
_MAX_PER_CYCLE = 4

#: 1 発話から採る属性の上限 (regex 側 ``MAX_ATTRIBUTES_PER_TEXT`` と同じ)。
_MAX_FACTS_PER_NOTE = MAX_ATTRIBUTES_PER_TEXT

#: 値として受ける最小文字数。1 文字の値は属性の裏取りにならない。
_MIN_VALUE_CHARS = 2

#: ``fact_attributes.yaml`` の fact_type → ``mem.<kind>`` / 述語。
#: 分割の対象は **ユーザー自身の属性** に限る (world_fact は Step 8.4 の担当)。
_KIND_BY_FACT_TYPE: dict[str, str] = {
    "personal_fact": "personal",
    "preference": "preference",
}
_PREDICATE_BY_FACT_TYPE: dict[str, str] = {
    "personal_fact": "states",
    "preference": "prefers",
}

#: 節の区切り (``extractors.chat._CLAUSE_SPLIT_RE`` と同じ規則)。
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[。！？!?、，,])\s*")

#: 値の中に現れてはいけない文の区切り。1 スロットの値が文をまたぐなら
#: それは「分割できていない」ので採らない (末尾の 1 個だけは許す)。
_SENTENCE_BREAK_RE = re.compile(r"[。．.！!？?]")

#: 属性を持たない受け皿スロット。ここへ分割しても粒度が上がらない。
_FALLBACK_SLUG = "user"

#: ``curation_backoff`` の失敗カウンタキー (aux purpose 名と同じにして
#: どの補助タスクの失敗かを一目で辿れるようにする)。
_FAILURE_KEY = "personal_fact_split"


def _allowed_slots(
    triggers_dir: str | Path | None = None,
) -> dict[str, tuple[str, str]]:
    """``{"<kind>.<slug>": (fact_type, slug)}`` の許可表を組む。

    ``fact_attributes.yaml`` (chat 節) の ``personal_fact`` / ``preference``
    のキーがそのまま値域になる。**表に無い slot は捨てる** ので、補助タスクが
    新しいスロット名を発明しても SemMem には入らない。
    """
    attrs = get_fact_attributes(resolve_fact_attributes_path(triggers_dir))
    per_type = attrs.get("chat") or {}
    allowed: dict[str, tuple[str, str]] = {}
    for fact_type, kind in _KIND_BY_FACT_TYPE.items():
        for spec in per_type.get(fact_type) or ():
            if spec.slug == _FALLBACK_SLUG:
                continue
            allowed[f"{kind}.{spec.slug}"] = (fact_type, spec.slug)
    return allowed


def resolve_slot(
    raw: str, allowed: dict[str, tuple[str, str]],
) -> tuple[str, str, str] | None:
    """補助タスクが返した slot を ``(fact_type, kind, slug)`` に解決する。

    受けるのは ``"<kind>.<slug>"`` (プロンプトで指示する形)。裸の ``<slug>``
    も、**どちらか一方の fact_type にしか無いとき** だけ受ける — 曖昧なもの
    (``beverage`` は personal_fact と preference の両方にある) は捨てる。
    表に無いものは ``None``。
    """
    key = (raw or "").strip().lower().replace(" ", "")
    if key in allowed:
        fact_type, slug = allowed[key]
        return fact_type, _KIND_BY_FACT_TYPE[fact_type], slug
    hits = [
        (fact_type, kind_slug.split(".", 1)[0], slug)
        for kind_slug, (fact_type, slug) in allowed.items()
        if slug == key
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _normalize_ws(text: str) -> str:
    """空白を全て落とした比較用の形 (純粋関数)。"""
    return "".join((text or "").split())


def is_verbatim_span(value: str, content: str) -> bool:
    """``value`` が発話の **逐語 span** か (空白の違いは無視する / 純粋関数)。

    補助タスクの出力を素材にしてよい唯一の条件。言い換え・要約・幻覚は
    ここで落ちる。空白を無視するのは、モデルが読点前後の空白を落とす /
    足すことがあるだけで、それは値の同一性を壊さないため。
    """
    v = _normalize_ws(value)
    if len(v) < _MIN_VALUE_CHARS:
        return False
    return v in _normalize_ws(content)


def _is_tight_value(value: str, content: str) -> bool:
    """値が「1 属性の span」として妥当か (純粋関数)。

    文をまたぐ値は分割できていない (元の粗い object と同じ問題を作る)。
    発話全体と同じ値も分割ではない。
    """
    body = (value or "").strip()
    if not body:
        return False
    if _SENTENCE_BREAK_RE.search(body[:-1] if len(body) > 1 else body):
        return False
    return _normalize_ws(body) != _normalize_ws(content)


def _clause_count(text: str) -> int:
    """節 (読点・句点で区切った単位) の数。"""
    return len([u for u in _CLAUSE_SPLIT_RE.split(text or "") if u.strip()])


def _existing_attribute_facts(
    store: "SemanticFactStore", note: "MemoryNote",
) -> dict[str, Any]:
    """このノートから既に作られた ``mem.personal.*`` / ``mem.preference.*``。

    ``{subject: fact}``。regex 経路 (Step 8) が同じサイクルで書いたものを
    引き当てるために ``extracted_fact_ids`` を辿る (ID 連鎖 / c_05 §0.6)。

    **supersede 済みのファクトも返す。** 粗い object (「札幌市の中央区に住んで
    いて、出版社で雑誌の編集をしています」) が、split の前に後続の訂正
    (「中央区ではなく北区です」) で畳まれると、live だけ見る判定では
    「2 節に跨る object」が見えなくなり split が走らず、隣の属性
    (occupation) が **永久に落ちる** (2026-09-09 ライブ監査 C-03)。分割の要否は
    そのノートの regex が何を掴んだかで決まり、その後に畳まれたかは無関係。
    再分割で書く古い location は ``persist_facts`` が発話時刻順で新しい live
    (北区) に負けさせるので、陳腐値が復活することはない。
    """
    out: dict[str, Any] = {}
    for fact_id in getattr(note, "extracted_fact_ids", None) or []:
        try:
            fact = store.get_fact(fact_id)
        except Exception:  # ストアの状態に依存しない (取れなければ無い扱い)
            fact = None
        if fact is None:
            continue
        parts = (fact.subject or "").split(".")
        if len(parts) == 3 and parts[0] == "mem" and parts[1] in (
            "personal", "preference",
        ):
            out[fact.subject] = fact
    return out


def needs_split(
    note: "MemoryNote", *, builder, store: "SemanticFactStore | None" = None,
) -> bool:
    """このノートを補助タスクへ出すか (規則だけで判定 / 純粋関数に近い)。

    出す条件は 2 つのどちらか:

    1. **節が 2 つ以上あるのに、解決したスロットが 1 個以下** — 語形の
       取りこぼしで属性が丸ごと落ちている形。
    2. **抽出済みの object が 2 節以上に跨る** — 1 スロットが隣の属性まで
       飲み込んでいる形。

    どちらも「regex が仕事をしきれなかった」ことの観測可能な兆候で、
    発話の語彙には依存しない。型付けが十分なノート (自己紹介 1 属性など) は
    出さない — 補助タスクは 1 回 60 秒なので、疑わしいものだけに絞る。
    """
    if getattr(note, "personal_fact_curated_at", None) is not None:
        return False
    if getattr(note, "source", "user") != "user":
        return False
    if getattr(note, "private", False):
        return False
    content = (note.content or "").strip()
    if not (_MIN_CHARS <= len(content) <= _MAX_CHARS):
        return False

    from backend.free.memory.extractors.chat import _looks_like_code_fragment

    if _looks_like_code_fragment(content):
        return False
    # 自己開示として型付けされた発話だけを対象にする。質問文 / 依頼文は
    # builder 側の ``is_plain_statement`` / ``states_no_user_value`` で落ちる。
    tags = builder.candidate_fact_tags(content)
    if not any(tag in _KIND_BY_FACT_TYPE for tag in tags):
        return False

    from backend.free.memory.notes.note_builder import (
        resolve_fact_attribute_matches,
    )

    slots = {
        (fact_type, slug)
        for fact_type in _KIND_BY_FACT_TYPE
        for slug, _ in resolve_fact_attribute_matches(
            content, fact_type, mode="chat",
            triggers_dir=getattr(builder, "triggers_dir", None),
        )
        if slug != _FALLBACK_SLUG
    }
    if _clause_count(content) >= 2 and len(slots) <= 1:
        return True
    if store is None:
        return False
    return any(
        _clause_count(fact.object or "") >= 2
        for fact in _existing_attribute_facts(store, note).values()
    )


def build_prompt(content: str, allowed: dict[str, tuple[str, str]]) -> str:
    """分割用 user プロンプトを組み立てる (純粋関数)。

    許可スロットを列挙するのは、``json_schema`` が型しか守らない
    (値域は守らない) ため。コード側でも検証するが、選択肢を見せた方が
    有効な出力の率が上がる。
    """
    slots = "\n".join(f"- {key}" for key in sorted(allowed))
    return (
        "次の発話から、ユーザー自身の属性を **属性ごとに 1 件ずつ** 取り出して"
        "ください。\n"
        "slot は下のリストの語をそのまま使うこと (リストに無い slot は禁止)。\n"
        "value は **発話に現れる文字列をそのまま** 抜き出すこと "
        "(要約・言い換え・補完は禁止)。属性 1 つ分の最小の範囲にすること。\n"
        "ユーザー本人の属性でないもの (ペット・家族・同僚の値、依頼、質問、"
        "挨拶) は返さないでください。該当が無ければ facts を空にしてください。\n"
        "slot の意味: personal.location は **本人の居住地** (通勤先・旅行先・"
        "趣味で行く場所は含めない)、personal.occupation は本人の職業・勤務先、"
        "personal.name は本人の氏名。趣味の行き先は location にしないこと。\n"
        f"\n許可された slot:\n{slots}\n"
        f"\nUTTERANCE: {content}\n"
    )


async def _split_facts(
    aux_client, content: str, allowed: dict[str, tuple[str, str]],
) -> list[dict[str, str]]:
    """補助タスクに属性分割させる。

    パース不能な応答は「答えたが使えない」ので空リスト。**例外は呼出側へ
    伝播する** — ここで握り潰すと一過性の aux timeout も「LLM が答えた」と
    同じ扱いになり、``curate_personal_facts`` が冪等マーカーを立てて二度と
    再試行しなくなる (2026-09-08 監査 G-03)。
    """
    parsed = await aux_client.generate_json(
        build_prompt(content, allowed),
        purpose="personal_fact_split",
        max_tokens=512,
        temperature=0.1,
        response_schema=PersonalFactSplit,
        list_key="facts",
    )
    if isinstance(parsed, dict):
        parsed = parsed.get("facts")
    if not isinstance(parsed, list):
        logger.debug(
            "personal_fact_curator: unexpected payload type: %r", type(parsed),
        )
        return []
    return [item for item in parsed if isinstance(item, dict)]


def accept_items(
    items: list[dict[str, str]],
    content: str,
    allowed: dict[str, tuple[str, str]],
) -> list[tuple[str, str, str, str]]:
    """補助タスクの出力を検証して ``(fact_type, kind, slug, value)`` に直す。

    落とすもの (すべて **黙って落とさず** DEBUG に残す):

    - 表に無い slot (新スロットを生やさない)
    - 発話の逐語 span でない value (幻覚)
    - 文をまたぐ / 発話全体と同じ value (分割になっていない)
    - 同じスロットの 2 件目 (先勝ち)
    """
    accepted: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        resolved = resolve_slot(str(item.get("slot") or ""), allowed)
        if resolved is None:
            logger.debug(
                "personal_fact_curator: unknown slot %r (dropped)",
                item.get("slot"),
            )
            continue
        fact_type, kind, slug = resolved
        value = str(item.get("value") or "").strip()
        if not is_verbatim_span(value, content):
            logger.debug(
                "personal_fact_curator: value not verbatim (dropped): %r", value,
            )
            continue
        if not _is_tight_value(value, content):
            logger.debug(
                "personal_fact_curator: value is not a single span (dropped): %r",
                value,
            )
            continue
        if (kind, slug) in seen:
            continue
        seen.add((kind, slug))
        accepted.append((fact_type, kind, slug, value))
        if len(accepted) >= _MAX_FACTS_PER_NOTE:
            break
    return accepted


def _is_refinement(value: str, existing_object: str) -> bool:
    """``value`` が既存 object の **より狭い span** か (純粋関数)。"""
    v, o = _normalize_ws(value), _normalize_ws(existing_object)
    return bool(v) and v in o and len(v) < len(o)


async def curate_personal_facts(
    notes: list["MemoryNote"],
    *,
    store_provider: Callable[[str], "SemanticFactStore | None"] | None,
    aux_client,
    embedder: "EmbeddingBackend | None",
    builder=None,
    profile_id: str = "default",
    now_provider: Callable[[], float] | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> int:
    """自己開示発話を属性ごとの逐語 span に分けて SemMem へ書く。

    Args:
        notes: 直近の MemoryNote 群 (通常 ``EpisodicWorkspace.notes.values()``)。
        store_provider: ``scope -> SemanticFactStore | None`` のコールバック。
        aux_client: 分割に使う補助タスククライアント。``None`` なら no-op。
        embedder: fact embedding 生成用。``None`` なら no-op。
        builder: ``ChatNoteBuilder`` (候補タグ判定用)。省略時は既定を作る。
        profile_id: 書込先 fact の profile_id。
        now_provider: 時刻供給。テスト用。
        should_pause: ``True`` を返したらノート境界でループを打ち切る協調
            yield。残りのノートは ``personal_fact_curated_at`` が立たない
            ままなので次サイクルが拾う。

    Returns:
        新規に書き込まれた fact 件数。
    """
    if aux_client is None:
        logger.debug("personal_fact_curator: aux_client is None, skipping")
        return 0
    if embedder is None:
        logger.debug("personal_fact_curator: embedder is None, skipping")
        return 0
    if store_provider is None:
        logger.debug("personal_fact_curator: store_provider is None, skipping")
        return 0
    store = store_provider("global")
    if store is None:
        logger.debug("personal_fact_curator: global store not available, skipping")
        return 0

    if builder is None:
        from backend.free.memory.notes.note_builder import ChatNoteBuilder

        builder = ChatNoteBuilder()

    now_fn = now_provider or time.time
    now = now_fn()

    # private セッション由来のノートは SemMem へ昇格させない
    # (``_curator_common.public_notes`` の docstring に実害と経緯)。
    notes = public_notes(notes)
    candidates = [
        n for n in notes
        if needs_split(n, builder=builder, store=store)
        and not in_cooldown(n, _FAILURE_KEY, now)
    ]
    if not candidates:
        return 0
    candidates.sort(key=lambda n: float(getattr(n, "created_at", 0.0) or 0.0))
    if len(candidates) > _MAX_PER_CYCLE:
        logger.info(
            "personal_fact_curator: %d candidate(s), splitting the oldest %d "
            "this cycle (the rest carry over)", len(candidates), _MAX_PER_CYCLE,
        )
        candidates = candidates[:_MAX_PER_CYCLE]

    allowed = _allowed_slots(getattr(builder, "triggers_dir", None))
    written = 0
    for idx, note in enumerate(candidates):
        # 協調 yield: チャット生成が走っている間はノート境界で手を止める
        # (note_evolver と同じ実測。CLAUDE.md 不変則 #1)。
        if should_pause is not None and should_pause():
            remaining = len(candidates) - idx
            logger.info(
                "personal_fact_curator paused for the user turn: %d note(s) "
                "left pending for the next cycle", remaining,
            )
            break
        content = (note.content or "").strip()
        try:
            items = await _split_facts(aux_client, content, allowed)
        except Exception as exc:
            # 一過性失敗 (aux timeout 等) はマーカーを立てず、次サイクルで
            # 再試行させる (2026-09-08 監査 G-03)。チャット併走由来の
            # contended timeout はこの purpose の予算不足ではないので
            # quarantine のカウントに含めない。
            logger.warning(
                "personal_fact_curator: split failed for note %s: %s",
                note.id, exc,
            )
            record_transient_failure(
                note, _FAILURE_KEY, now_fn(),
                counts=not getattr(exc, "contended", False),
            )
            continue
        # 分割できなかった場合もマークする。同じノートを毎サイクル補助タスクへ
        # 出し続けないため (url_curator が全分岐で url_curated_at を立てるのと
        # 同じ理由)。これは「補助タスクが答えた」場合であって、失敗ではない。
        note.personal_fact_curated_at = now_fn()
        clear_failure(note, _FAILURE_KEY)
        accepted = accept_items(items, content, allowed)
        if not accepted:
            continue
        written += await _persist_split(
            store, note, accepted,
            embedder=embedder, profile_id=profile_id, now=now_fn(),
        )
    return written


async def _persist_split(
    store: "SemanticFactStore",
    note: "MemoryNote",
    accepted: list[tuple[str, str, str, str]],
    *,
    embedder: "EmbeddingBackend",
    profile_id: str,
    now: float,
) -> int:
    """検証済みの分割結果を Step 8 と同じ経路で書き込む。

    ``persist_facts`` を通すので、単値スロット / 訂正の supersede は regex 経路
    と同じ規則で走る (:func:`~backend.free.memory.sleep.extraction.
    _supersede_corrected_slots`)。

    regex が同じスロットに **より粗い object** を書いていた場合は、その 1 件を
    明示的に supersede する。同じノート・同じスロットの **狭め直し** であって
    競合ではないので、値が失われる方向の失敗が無い (単値スロットなら
    ``persist_facts`` 側でも畳まれるが、``name`` のような多値スロットは
    ここで畳まないと粗い object が live のまま残る)。

    **既存スロットを狭められるのは「実際に分割できた」ときだけ。** 補助タスクが
    1 スロットしか返さなかったら、残りの節が *別の属性* なのか *同じ属性の敷衍*
    なのかの証拠が無い。実インシデント (2026-09-05 監査 F-10):
    「趣味は登山で、去年は北アルプスの槍ヶ岳に登りました。」を「登山」へ
    狭めると、次ターンの訂正「…去年は白馬岳でした。」が既存スロットの現在値を
    名指せなくなり (値アンカーの手掛かりが消える)、訂正が hobby に届かない。
    よって単独スロットの返答は **既に regex が持っているスロットには触らない**
    (取りこぼしていたスロットの追加だけ行う)。
    """
    from backend.free.memory.extractors import ExtractionResult
    from backend.free.memory.sleep.extraction import persist_facts

    existing = _existing_attribute_facts(store, note)
    #: 分割の証拠 = 2 スロット以上に割れたこと。1 件しか返らなかった場合は
    #: 既存スロットを狭めない (docstring の 2026-09-05 F-10 参照)。
    split_confirmed = len(accepted) >= 2
    facts = []
    refined: list[tuple[Any, Any]] = []
    for fact_type, kind, slug, value in accepted:
        subject = make_mem_subject(kind, slug)
        prior = existing.get(subject)
        if (
            prior is not None
            and not split_confirmed
            and _normalize_ws(value) in _normalize_ws(prior.object or "")
        ):
            # F-10 の保護は「regex の object が既にその値を含む」ときだけ。
            # 含まないなら regex は **値を持たない文** (「今日は新しい趣味の話を
            # します」= トリガ語だけの前置き) を掴んでおり、狭めるのではなく
            # 値を与えることになる (2026-09-09 検証 W02: 陶芸教室が捨てられ、
            # 次ターンの自己訂正が hobby の現在値を名指せなかった)。
            logger.debug(
                "personal_fact_curator: single-slot answer, keeping the regex "
                "object for %s", subject,
            )
            continue
        if prior is not None and _normalize_ws(prior.object or "") == _normalize_ws(
            value,
        ):
            continue  # regex が既に同じ span を書いている
        try:
            embedding = await embedder.embed([value], is_query=False)
        except Exception as exc:
            logger.warning(
                "personal_fact_curator: embedding failed for %s: %s", subject, exc,
            )
            continue
        fact = fact_from_note(
            note,
            subject=subject,
            predicate=_PREDICATE_BY_FACT_TYPE[fact_type],
            object_=value,
            type=fact_type,
            scope="global",
            now=now,
            profile_id=profile_id,
            embedding=embedding[0] if len(embedding) else None,
            _extra={"source_note_id": note.id, "raw_utterance": note.content or ""},
        )
        # ノート由来の追跡情報を Step 8 と同じ粒度で刻む (c_05 §0.6)。
        prov = fact.provenances[0] if fact.provenances else None
        if prov is not None:
            prov.turn_id = getattr(note, "turn_id", "") or None
            prov.extractor = EXTRACTOR_NAME
            prov.extractor_version = EXTRACTOR_VERSION
        facts.append(fact)
        # 同じノート由来の粗い object は、狭めた span (refinement) でも、値を
        # 含まない前置き文だった場合でも、逐語 span の方が現在値になる。
        if prior is not None and (
            _is_refinement(value, prior.object or "")
            or _normalize_ws(value) not in _normalize_ws(prior.object or "")
        ):
            refined.append((prior, fact))

    if not facts:
        return 0
    written = persist_facts(
        store, ExtractionResult(facts=facts), "personal_fact_split",
    )
    for fact in facts:
        if fact.id not in note.extracted_fact_ids:
            note.extracted_fact_ids.append(fact.id)
    _supersede_coarser_siblings(store, refined)
    if written:
        logger.info(
            "personal_fact_curator: wrote %d fact(s) from note %s (%s)",
            written, note.id, ", ".join(f.subject for f in facts),
        )
    return written


def _supersede_coarser_siblings(
    store: "SemanticFactStore", refined: list[tuple[Any, Any]],
) -> int:
    """同じノート・同じスロットの **粗い object** を新しい方で畳む。

    粗い object が **既に別の値 (訂正など) に畳まれている** なら、狭め直した
    値も同じ後継に畳む。狭め直しは同じ発話の言い直しであって、無効化された
    値を live に戻す根拠ではない。実データ (2026-09-10 ライブ監査 (f) F-10):
    「休日は登山に行くことが多いです」の粗い hobby が訂正「登山ではなく
    トレイルランニング」に畳まれた後、Step 8.3 の分割が「登山」を書き、多値
    スロットなので ``persist_facts`` の畳み込みにも掛からず live に戻った。
    """
    folded = 0
    for old, new in refined:
        if old.id == new.id:
            continue
        successor = getattr(old, "superseded_by", None)
        if successor and successor != new.id:
            try:
                store.supersede(new.id, successor)
                folded += 1
                logger.info(
                    "personal_fact_curator: refinement %s inherits the supersession "
                    "of %s -> %s", new.id, old.id, successor,
                )
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "personal_fact_curator: failed to supersede %s -> %s: %s",
                    new.id, successor, exc,
                )
            continue
        if successor:
            continue
        try:
            store.supersede(old.id, new.id)
        except (KeyError, ValueError) as exc:
            # 閉路ガード等で弾かれた場合は残す (競合解決 / TTL に委ねる)。
            logger.warning(
                "personal_fact_curator: failed to supersede %s -> %s: %s",
                old.id, new.id, exc,
            )
            continue
        folded += 1
    if folded:
        logger.info(
            "personal_fact_curator: superseded %d coarse object(s)", folded,
        )
    return folded


__all__ = [
    "accept_items",
    "build_prompt",
    "curate_personal_facts",
    "is_verbatim_span",
    "needs_split",
    "resolve_slot",
]
