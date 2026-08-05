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

import hashlib
import re
from collections.abc import Iterable

from backend.free.memory.extractors.base import (
    BaseExtractor,
    ExtractionContext,
    ExtractionResult,
)
from backend.free.memory.notes.note_builder import (
    ChatNoteBuilder,
    resolve_fact_attribute,
)
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

#: world_fact の subject kind として無意味な英語機能語 / フォーマット例トークン。
#: 生成物断片の抽出で ``mem.world.from`` / ``mem.world.YYYYMMDD`` 等のゴミ
#: subject が量産された (2026-07-15) ため、これらは keyword 候補から除外する。
#: 2026-07-26 追加: 依頼文・質問文の先頭語が subject になった実例
#: (``mem.world.Please`` ← "Please answer in English. What is a race condition")。
_WORLD_KEYWORD_STOPWORDS: frozenset[str] = frozenset({
    "from", "import", "the", "and", "for", "with", "this", "that",
    "class", "def", "return", "none", "true", "false",
    "yyyymmdd", "yyyy-mm-dd", "hhmmss", _SAFE_KEYWORD_FALLBACK,
    # 依頼・質問・接続の機能語 (話題を指さないので subject に使えない)
    "please", "what", "when", "where", "which", "who", "why", "how",
    "is", "are", "was", "were", "be", "been", "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "might", "must",
    "tell", "give", "show", "explain", "answer", "about", "into",
    "you", "your", "yours", "me", "my", "mine", "we", "our", "it", "its",
    "a", "an", "of", "in", "on", "at", "to", "by", "as", "or", "but",
    "not", "no", "yes", "if", "then", "than", "so", "there", "here",
})

#: 英字を 1 文字も含まない sanitized keyword を弾く判定。``_sanitize_keyword`` は
#: 非 ASCII を ``-`` に潰すため、数字と記号だけの語は無意味な subject になる
#: (2026-07-26 実測: 「0.8% と 2.3%」→ ``mem.world.0-8----2-3``)。
_HAS_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

#: subject に付ける内容ハッシュの長さ。curator 系 (``mem.world.url.<host>.<sha1_12>``
#: / ``mem.world.executable_command.<mode>.<sha1_12>``) と同じ規約に合わせる。
_WORLD_SUBJECT_HASH_LEN = 8

#: 明示的なローカルパス (ファイル出力依頼の指示文検出用)
_LOCAL_PATH_HINT_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"']+")

#: 文末が疑問形 (質問文) かどうかの判定。personal_fact/preference/emotion/opinion
#: は「ユーザーが表明した事実」であるべきで、質問文 (例: 「私の好きなプログラミング
#: 言語は？」) はユーザー自身の嗜好の表明ではない。トリガ語の単純部分一致だけでは
#: 質問文と平叙文を区別できず、ノート全文がそのまま object_text になる (2026-07-18:
#: pending コンフリクトに質問文がそのまま preference ファクトとして混入していた実
#: インシデント)。
#:
#: 日本語の疑問文は疑問符を伴わないことが多い。``[?？]`` だけを見ていたため、
#: 「〜ますか。」で終わる質問が平叙文として通り、ユーザーの質問文がそのまま
#: ファクトとして保存されていた (2026-08-05 ライブ監査で注入ブロックに
#: ``mem.personal.birthday states: 私の猫の名前と誕生日を覚えていますか。``
#: ``mem.preference.user prefers: 私の名前と好きな季節を覚えていますか。``
#: が出現)。
#:
#: 疑問符なしの判定は**丁寧形の助動詞 + か**という閉じた集合に限る。裸の「か」
#: まで拾うと「イカ。」「〜とか。」のような名詞・引用の文末を巻き込む。
#: 活用形を列挙するのではなく助動詞そのものを列挙するのは、語尾のバリエーションで
#: 取りこぼす失敗を繰り返さないため。
_QUESTION_ENDING_RE = re.compile(
    r"(?:[?？]"
    r"|(?:ます|です|ました|でした|ません|ませんでした|でしょう|ましょう"
    r"|だろう|であろう)か)"
    r"[。．.、,！!\s\"'」』）)]*\s*$",
)

#: 文区切り (。！？!?) の直後で分割する。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")


def _tag_evidence_is_question_only(content: str, trigger_words: tuple[str, ...]) -> bool:
    """トリガ語を含む文が、すべて疑問形かどうかを判定する。

    ノート全体の文末だけで判定すると、平叙文の嗜好表明に無関係な質問が
    続く複合発話 (例:「Pythonが好きです。あなたは何が好きですか?」) まで
    丸ごと除外してしまう (レビューで判明)。トリガ語を含む文だけを見て、
    それが全て疑問形の場合のみ「質問文のみの根拠」とみなす。該当文が
    無ければ (呼出側の判定に委ねるため) False を返す。
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(content) if s.strip()]
    text_lower_sentences = [(s, s.lower()) for s in sentences]
    relevant = [
        s for s, s_lower in text_lower_sentences
        if any(t in s_lower for t in trigger_words)
    ]
    if not relevant:
        return False
    return all(_QUESTION_ENDING_RE.search(s.strip()) for s in relevant)

#: コード断片らしさの指標 (EvorefMem は EvorefLoop の utils を import できない
#: ため、meta_cognitive_utils.contains_code_indicator と同旨の最小実装を持つ)
_CODE_FRAGMENT_MARKERS: tuple[str, ...] = (
    "import ", "from __future__", "def ", "class ", "return ",
    "#!/", "```python", "@dataclass",
)


def _looks_like_code_fragment(content: str) -> bool:
    """ノート内容がプログラムコードの断片らしいかを判定する。"""
    if not content:
        return False
    hits = sum(1 for m in _CODE_FRAGMENT_MARKERS if m in content)
    return hits >= 2


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


def _is_usable_world_keyword(sanitized: str) -> bool:
    """sanitized keyword が world_fact の subject に使えるかを判定する。

    ストップワードでないこと、かつ英字を 1 文字以上含むことを要求する。
    数字と記号だけの語 (「0.8% と 2.3%」→ ``0-8----2-3``) は話題を指さない。
    """
    if sanitized.lower() in _WORLD_KEYWORD_STOPWORDS:
        return False
    return bool(_HAS_ASCII_LETTER_RE.search(sanitized))


def _world_fact_keyword(note: MemoryNote) -> str:
    """world_fact の subject kind キーワードをノートから導く。

    `keywords` からストップワード (英語機能語 / 依頼・質問の機能語 / フォーマット
    例トークン) と英字を含まない語を除いた最初の候補を採用し、無ければ内容先頭
    24 文字を使う。全候補が無効な場合は :data:`_SAFE_KEYWORD_FALLBACK` を返す
    (呼出側で抽出をスキップする)。
    """
    for kw in note.keywords or []:
        sanitized = _sanitize_keyword(kw)
        if _is_usable_world_keyword(sanitized):
            return sanitized
    text = " ".join((note.content or "").split())
    if text:
        sanitized = _sanitize_keyword(text[:24])
        if _is_usable_world_keyword(sanitized):
            return sanitized
    return _SAFE_KEYWORD_FALLBACK


def _world_fact_subject_parts(keyword: str, content: str) -> tuple[str, str]:
    """world_fact の subject を ``<keyword>`` + ``<内容ハッシュ>`` に分ける。

    keyword だけでは subject が衝突する。keyword は本文から拾った任意の語なので、
    無関係な事実が同一 subject に相乗りし、競合検出が ``(subject, predicate)``
    でグルーピングするため「同一事実の別版」と誤判定される
    (2026-07-26 実測: 自転車通勤の走行距離がスペイン語の ``mem.world.ser`` に
    同居していた)。curator 系が既に採っている
    ``mem.world.url.<host>.<sha1_12>`` / ``mem.world.executable_command.<mode>.
    <sha1_12>`` と同じ規約で、内容ハッシュを 1 セグメント足して一意にする。
    """
    normalized = " ".join((content or "").split())
    digest = hashlib.sha1(
        normalized.encode("utf-8"), usedforsecurity=False,
    ).hexdigest()[:_WORLD_SUBJECT_HASH_LEN]
    return keyword, digest


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
            content = note.content or ""
            is_assistant_note = getattr(note, "source", "user") != "user"
            # 生成物 (コード等) の断片は user/world ファクトの素材にならない
            # (2026-07-15: 誤ルート生成の Python コードが mem.world.from として
            # 保存された)。コードらしいノートは抽出対象から外す。
            if _looks_like_code_fragment(content):
                result.notes_skipped += 1
                continue
            # ロールガード: assistant 発話は SemanticFact の素材にしない。
            # personal/preference/emotion/opinion は 2026-07-15 に対策済み
            # (「私は AI ですので…」が mem.personal.user に保存された) だが、
            # world_fact だけガードの外にあり、アシスタント自身の生成文が
            # 「世界の事実」として蓄積されていた。未検証の生成物を権威ある
            # 事実として後日想起する構造になる。
            # 2026-07-26 の実データ (global scope) では world_fact 11 件が
            # すべてこの経路の産物で、内容も subject も事実として機能して
            # いなかった: アシスタントの長文回答がそのまま object になり、
            # subject は本文から拾った任意の ASCII 語 (mem.world.ser /
            # mem.world.Cpk / mem.world.LDK)、数字混じりのゴミ
            # (mem.world.0-8----2-3)、ユーザーの質問文の先頭語
            # (mem.world.Please) だった。自転車通勤の走行距離がスペイン語の
            # 「ser」に相乗りする subject 衝突も起きていた。
            # world_fact はユーザーが断定した知識 (「水は H2O である」) に限る。
            if is_assistant_note:
                result.notes_skipped += 1
                continue
            tags = self._builder.candidate_fact_tags(content)
            for tag in tags:
                if tag not in self.SUPPORTED_TAGS:
                    continue
                fact_type: FactType = tag  # type: ignore[assignment]
                kind = _KIND_BY_TAG[tag]
                if tag in _USER_SUBJECT_TAGS:
                    # ファイル出力の指示文 (明示パス付き) は嗜好/感情ではなく
                    # 作業依頼なので preference/emotion/opinion にしない。
                    if tag != "personal_fact" and _LOCAL_PATH_HINT_RE.search(content):
                        continue
                    # 質問文はユーザー自身の事実の表明ではないので候補にしない
                    # (例: 「私の好きなプログラミング言語は？」は preference ではない)。
                    # トリガ語を含む文だけを見て判定し、無関係な質問が後続する
                    # 平叙文の嗜好表明 (例:「Pythonが好きです。何が好きですか?」)
                    # までは除外しない。
                    trigger_words = self._builder.fact_triggers.get(tag, ())
                    if _tag_evidence_is_question_only(content, trigger_words):
                        continue
                    # subject を属性単位に分割する (mem.personal.machine_spec 等)。
                    # 一律 "user" だと、競合検出が (subject, predicate) キーで
                    # 類似度を見ずにグルーピングするため、無関係な事実
                    # (コメント方針 vs GPU 仕様) まで同一事実の競合版と誤判定され
                    # 恒久 pending 化する (2026-07-25)。辞書に無ければ従来どおり
                    # "user" へフォールバックするので退行しない。
                    attr = resolve_fact_attribute(
                        content, tag, mode="chat",
                        triggers_dir=self._builder.triggers_dir,
                    )
                    subject = make_mem_subject(kind, attr or "user")
                else:
                    # world_fact のみ: keyword (sanitized) + 内容ハッシュを
                    # parts に使用 (keyword 単独では subject が衝突する)
                    keyword = _world_fact_keyword(note)
                    if keyword == _SAFE_KEYWORD_FALLBACK:
                        # 有効な keyword を導けないノートは world_fact 化しない
                        # (mem.world.unknown の量産防止)
                        continue
                    subject = make_mem_subject(
                        kind, *_world_fact_subject_parts(keyword, content),
                    )
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
