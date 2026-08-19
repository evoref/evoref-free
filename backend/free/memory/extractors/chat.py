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

from backend.free.core.text_quality import (
    strip_discourse_prefix,
    strip_first_person_topic,
    strip_interrogative_sentences,
)
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


#: アシスタントへの **依頼** の文末。疑問形ではないが、ユーザー自身の事実の
#: 表明でもない。
#:
#: 実インシデント (2026-08-18 ライブ監査): 「データ分析で**よく使う**可視化
#: ライブラリを 3 つ挙げてください。」が preference トリガ ``よく使う`` に
#: 一致し、依頼文がまるごと ``mem.preference.user`` の object として保存された。
#: 疑問符も「〜ですか」も無いため ``_QUESTION_ENDING_RE`` では拾えない。
_REQUEST_ENDING_RE = re.compile(
    r"(?:(?:て|で)(?:ください|下さい)"
    r"|(?:して|で)(?:ほしい|欲しい)"
    r"|お願いします|願います"
    r"|(?:教え|挙げ|見せ|出し|作っ|書い|説明し|列挙し|示し)て)"
    r"[。．.、,！!\s\"'」』）)]*\s*$",
)

#: 一人称マーカー。依頼形でもこれを伴う文は本人の事実表明を含みうるため
#: (例:「私はダークテーマが好きなので、そう設定してください。」)、依頼を
#: 理由に捨てない。ただし **一人称があるだけでは免除しない** —
#: :func:`_asserts_before_request` を参照。
_SELF_REFERENCE_RE = re.compile(r"(?:私|僕|俺|自分|わたし|ぼく|うち)")

#: 従属節の切れ目 (接続助詞 + 読点)。依頼文の中で「言明の節」と「依頼の節」を
#: 分ける境界として使う。読点を必須にするのは、体言の並列 (「AとB、Cを…」) を
#: 節の切れ目と誤認しないため。
#:
#: 実インシデント (2026-08-19 ライブ監査): 「私の好きな飲み物をもう一度教えて
#: ください。」が ``mem.personal.beverage states`` / ``mem.preference.beverage
#: prefers`` の 2 件として保存され、さらに本人の実際の言明
#: (「私の好きな飲み物は緑茶です」) と同じ (subject, predicate) に並んだため
#: 競合の当事者になり pending に滞留した。依頼形ゲート自体は存在したが、
#: 一人称を含むだけで無条件に免除していたため機能していなかった。
#:
#: ``で`` / ``て`` は入れない。「私の好きな飲み物を調べて、教えてください。」の
#: ような**依頼の中の依頼**まで免除してしまい、直そうとしている誤りが戻る。
#: 取りこぼす側 (「私は東京在住で、近くの店を教えてください。」) の損失は
#: 候補 1 件であり、ゴミを入れる損失より小さい。
_CLAUSE_BREAK_RE = re.compile(
    r"(?:ので|のに|から|ため|けれども|けれど|けど|ですが|だが|ますが)[、,]",
)


def _asserts_before_request(sentence: str) -> bool:
    """依頼文が、依頼節より **前の節** に本人の言明を含むかを判定する。

    一人称の有無だけで判定すると、一人称が依頼の**目的語**でしかない文
    (「私の好きな飲み物をもう一度教えてください。」) まで本人の表明として
    通ってしまう。言明は依頼とは別の節に立つはずなので、従属節の切れ目
    (:data:`_CLAUSE_BREAK_RE`) より前に一人称があることを要求する。

    - ``私はダークテーマが好きなので、そう設定してください。`` → ``ので、``
      より前に「私」がある → True (本人の表明を含む)
    - ``私の好きな飲み物をもう一度教えてください。`` → 節の切れ目が無い
      → False (依頼でしかない)
    - ``明日の予定を、私の代わりに調べてください。`` → 読点はあるが接続助詞
      ではなく、そもそも「私」は読点より後 → False
    """
    last_break = -1
    for m in _CLAUSE_BREAK_RE.finditer(sentence):
        last_break = m.end()
    if last_break < 0:
        return False
    return bool(_SELF_REFERENCE_RE.search(sentence[:last_break]))


def _tag_evidence_is_question_only(content: str, trigger_words: tuple[str, ...]) -> bool:
    """トリガ語を含む文が、すべて **ユーザー自身の表明でない** かを判定する。

    ノート全体の文末だけで判定すると、平叙文の嗜好表明に無関係な質問が
    続く複合発話 (例:「Pythonが好きです。あなたは何が好きですか?」) まで
    丸ごと除外してしまう (レビューで判明)。トリガ語を含む文だけを見て、
    それが全てそうである場合のみ「根拠が表明でない」とみなす。該当文が
    無ければ (呼出側の判定に委ねるため) False を返す。

    「そうでない文」は 2 種類:

    - **疑問形** (``_QUESTION_ENDING_RE``) — 「あなたは何が好きですか?」
    - **依頼形** (``_REQUEST_ENDING_RE``) — 「よく使うライブラリを挙げて
      ください。」。ただし依頼節より前の節に本人の言明がある複合文
      (「私は〜なので、〜してください。」) は本人の事実表明を兼ねるので
      除外しない (``_asserts_before_request``)。
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(content) if s.strip()]
    text_lower_sentences = [(s, s.lower()) for s in sentences]
    relevant = [
        s for s, s_lower in text_lower_sentences
        if any(t in s_lower for t in trigger_words)
    ]
    if not relevant:
        return False

    def _is_non_assertive(sentence: str) -> bool:
        s = sentence.strip()
        if _QUESTION_ENDING_RE.search(s):
            return True
        if not _REQUEST_ENDING_RE.search(s):
            return False
        return not _asserts_before_request(s)

    return all(_is_non_assertive(s) for s in relevant)

#: 規則ベース正規化で残す文の下限 (これ未満なら正規化失敗として原文を使う)。
_STATEMENT_MIN_CHARS = 2

#: 接頭辞剥がしの最大反復。「ところで、私は…」のような入れ子を解くため繰り返すが、
#: 無限ループにはしない。実際は 2 回で収束する。
_STATEMENT_STRIP_PASSES = 3

#: ``compress_turn(style="summary")`` の圧縮マークと末尾の元文字数。
_SUMMARY_MARK = "[要約] "
_SUMMARY_TAIL_RE = re.compile(r"…（\d+文字）\s*$")


def _collapse_equivalent_candidates(
    candidates: list[tuple[MemoryNote, SemanticFact]],
) -> tuple[list[tuple[MemoryNote, SemanticFact]], int]:
    """同じ命題を指す候補を 1 本に畳む (純粋関数)。

    同一発話が STM に 2 本のノートとして残ることがあり (原文と
    ``compress_turn(style="summary")`` の ``[要約]`` 版)、それぞれからファクトが
    起きて **statement が完全一致する重複** ができる。両方 active のまま残ると
    競合検出が「内容が矛盾している」と見なして恒久 pending 化し、
    ``[記憶の競合 — 未解決]`` として無関係な質問にまで注入される
    (2026-08-16 実データ: ``sf_127618cb29fb`` 原文 / ``sf_852c5461ce09`` ``[要約]``、
    同一 subject・同一秒・statement 一致)。

    :func:`normalize_statement` が既に ``[要約]`` を剥がしているので、突き合わせは
    ``(subject, predicate, statement)`` で行う。``statement`` が立たなかった候補は
    ``object`` で代用する (従来と同じ挙動)。先に現れた候補 = 原文側を残す。

    Returns:
        ``(畳んだ後の候補, 落とした件数)``。
    """
    seen: set[tuple[str, str, str]] = set()
    kept: list[tuple[MemoryNote, SemanticFact]] = []
    collapsed = 0
    for note, fact in candidates:
        key = (
            fact.subject or "",
            fact.predicate or "",
            (fact.statement or fact.object or "").strip(),
        )
        if key[2] and key in seen:
            collapsed += 1
            continue
        if key[2]:
            seen.add(key)
        kept.append((note, fact))
    if collapsed:
        logger.debug(
            "ChatExtractor: collapsed %d duplicate fact candidate(s) "
            "(same subject/predicate/statement)", collapsed,
        )
    return kept, collapsed


def normalize_statement(
    content: str, trigger_words: tuple[str, ...],
) -> str | None:
    """発話原文から「ユーザーについての命題」を規則だけで切り出す (純粋関数)。

    ``object`` には発話原文が入るため、``[関連する記憶]`` には会話の足場や
    一人称のついた行が並ぶ (2026-08-16 監査時点の実データ:
    ``mem.personal.user states: コーヒー派？紅茶派？私はコーヒーを1日3杯は飲んじゃう。``)。

    ここで落とすのは **語彙に依存しない構造的なノイズ** だけ:

    1. 疑問文だけの文 (問いは主張ではない)
    2. トリガ語を含まない文 (その属性と無関係な部分)
    3. 文頭の談話標識 (「ところで」「実は」)
    4. 文頭の一人称主題 (「私は」— 主語は subject が持つ)

    「紅茶派です」→「紅茶」のような **値の抽出** は語順・助詞の解析が要るので
    ここでは行わない (LLM 併用の Phase 2 で扱う想定)。

    Returns:
        正規化後の命題。原文と変わらない / 短くなりすぎる場合は ``None``
        (呼出側は ``statement`` を立てず ``object`` をそのまま使う)。
    """
    text = (content or "").strip()
    if not text:
        return None
    kept = strip_interrogative_sentences(text)
    if trigger_words:
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(kept) if s.strip()]
        relevant = [
            s for s in sentences
            if any(t in s.lower() for t in trigger_words)
        ]
        if relevant:
            kept = "".join(relevant)
    kept = kept.strip()
    # 圧縮マークはシステムが足したもので、ユーザーの発話ではない。
    # これを落とすと原文と [要約] が同じ命題に畳まれ、競合・重複判定でも揃う。
    if kept.startswith(_SUMMARY_MARK):
        kept = kept[len(_SUMMARY_MARK):].strip()
    kept = _SUMMARY_TAIL_RE.sub("", kept).strip()
    # 談話標識と一人称主題は入れ子になる (「ところで、私は…」「私、実は…」)。
    # 変化が無くなるまで繰り返し剥がす。
    for _ in range(_STATEMENT_STRIP_PASSES):
        peeled = strip_first_person_topic(strip_discourse_prefix(kept)).strip()
        if peeled == kept:
            break
        kept = peeled
    if len(kept) < _STATEMENT_MIN_CHARS or kept == text:
        return None
    return kept


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


def resolve_inherited_attributes(
    notes: list[MemoryNote], builder: "ChatNoteBuilder",
) -> dict[tuple[str, str], str]:
    """訂正ノートが継ぐべき属性を ``(note_id, tag) -> attr`` で返す (純粋関数)。

    ``resolve_fact_attribute`` は **その発話自身の文** から属性を引く。ところが
    訂正は属性名詞を落として言うのが普通で、「違います、私はほうじ茶が一番
    好きです。」には ``飲み物`` も ``コーヒー`` も出てこない。結果 ``None`` →
    ``mem.preference.user`` へフォールバックし、訂正対象の
    ``mem.preference.beverage`` と **別スロット**になる。スロットが違うと
    競合検出が対にできず、``from_correction`` の即時解決も
    ``_collapse_to_current_values`` の畳み込みも効かない。

    実測 (2026-08-19 ライブ検証): 「私の好きな飲み物は何ですか？」への注入候補で
    訂正済みの「緑茶」が sim 0.762 で最上位、訂正後の「ほうじ茶」が 0.487 で
    下位に並んでいた。どちらも同日なので ``(N日前の記録)`` ラベルも付かない。

    **埋め込みでスロットを束ねる案は使えない**。訂正と対象の類似度は実測で
    真 0.575〜0.714 / 偽 0.429〜0.529 と重なり、閾値を置ける分離が無い
    (``split_by_attribute_similarity`` が使う「同じ属性の言い直しか」の分離は
    真 0.796〜0.963 / 偽 0.316〜0.418 で、そちらとは別物)。

    そこで **直前の言明から継ぐ**。発火条件を絞って副作用を閉じる:

    - ノートに ``is_correction`` が立っている (値の言い直し)
    - **かつ** そのノート自身の属性解決が ``None`` (既存の解決を上書きしない)
    - **かつ** 同一セッションの直前に、同じ tag で属性を解決できたノートがある

    実測 (STM 42 ノートでのシミュレーション): 発火は狙った 1 件のみで、
    残り 41 件は無変化だった。
    """
    ordered = sorted(
        notes, key=lambda n: (n.session_id or "", float(n.created_at or 0.0)),
    )
    last_named: dict[tuple[str, str], str] = {}
    inherited: dict[tuple[str, str], str] = {}
    for note in ordered:
        content = note.content or ""
        for tag in _USER_SUBJECT_TAGS:
            key = (note.session_id or "", tag)
            attr = resolve_fact_attribute(
                content, tag, mode="chat", triggers_dir=builder.triggers_dir,
            )
            if attr:
                last_named[key] = attr
            elif getattr(note, "is_correction", False) and key in last_named:
                inherited[(note.id, tag)] = last_named[key]
    return inherited


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
        note_list = list(notes)
        # 訂正ノートが継ぐ属性を先に決める (本ループの順序は変えない —
        # apply_session_caps の採否順に影響するため)。
        inherited = resolve_inherited_attributes(note_list, self._builder)
        for note in note_list:
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
                    # 属性語を落とした訂正は直前の言明からスロットを継ぐ
                    # (resolve_inherited_attributes の説明を参照)。
                    if attr is None:
                        attr = inherited.get((note.id, tag))
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
                    # 末尾の問いは記憶として意味を持たないうえ、[関連する記憶] に
                    # 「答えではなく問い」が根拠として並ぶ原因になる。
                    # (2026-08-16 監査時点の実データ:
                    #  mem.emotion.user feels: 夜更かしすると次の日つらいですよね。
                    #  何かいい対策ありますか？)
                    object_text=strip_interrogative_sentences(note.content or ""),
                    # 規則で切り出せる分だけ命題化する。object (原文) は証拠として
                    # 残し、提示・比較・埋め込みは fact.text 経由で statement を
                    # 優先する。切り出せなければ None のままで従来と同じ挙動。
                    statement=normalize_statement(
                        note.content or "",
                        tuple(self._builder.fact_triggers.get(tag, ())),
                    ),
                    fact_type=fact_type,
                    scope=SemanticFact.make_global_scope(),
                    note=note,
                    ctx=ctx,
                )
                candidates.append((note, fact))

        candidates, collapsed = _collapse_equivalent_candidates(candidates)
        kept, dropped = self.apply_session_caps(candidates, ctx)
        result.cap_dropped = dropped
        result.facts = [fact for _, fact in kept]
        # ノート → fact ID の双方向リンクを書き戻し
        for note, fact in kept:
            if fact.id not in note.extracted_fact_ids:
                note.extracted_fact_ids.append(fact.id)
        logger.debug(
            "ChatExtractor: processed=%d skipped=%d already=%d facts=%d "
            "dropped=%d collapsed=%d",
            result.notes_processed,
            result.notes_skipped,
            result.already_extracted,
            len(result.facts),
            result.cap_dropped,
            collapsed,
        )
        return result
