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

from backend.free.core.intent_vocab import is_plain_statement
from backend.free.core.text_quality import (
    _asserts_before_request,
    _REQUEST_ENDING_RE,
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
    _CONFIRMATION_SEEKING_RE,
    _normalize_trigger,
    resolve_fact_attribute_match,
    resolve_fact_attribute_matches,
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

#: 走査順を固定した版 (frozenset の反復順は実行ごとに変わりうる)。
_USER_SUBJECT_TAGS_ORDERED: tuple[str, ...] = (
    "personal_fact",
    "preference",
    "emotion",
    "opinion",
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
#: **体言止めの問い** (「私の好きな飲み物は。」) も値を持たない。助動詞も疑問符も
#: 伴わないため上の 2 つの規則をすり抜けていた。実インシデント
#: (2026-08-29 ライブ監査): F11 修正後の Full で新規抽出された 9 件のうち 4 件が
#: 想起の問いで、``mem.preference.media = 私が好きな音楽のジャンルは。``
#: ``mem.personal.beverage = 私の好きな飲み物は。`` が **その属性の現在値** として
#: 保存された。次の想起でこの質問文が「過去の記録」として注入されるため、
#: **想起するたびに記憶が汚染される**。
#:
#: 主題の ``は`` が文の最後に来る形は日本語の平叙文には無い — 値の言明は必ず
#: ``は`` の後ろに語が続く (「私の猫の名前は**ミケです**。」)。よって末尾の ``は``
#: だけで問いと言明を分離できる (router の ``_PERSONAL_RECALL_RE`` と同じ判別)。
_QUESTION_ENDING_RE = re.compile(
    r"(?:[?？]"
    r"|(?:ます|です|ました|でした|ません|ませんでした|でしょう|ましょう"
    r"|だろう|であろう)か"
    r"|(?<=.)は)"
    r"[。．.、,！!\s\"'」』）)]*\s*$",
)

#: 文区切り (。！？!?) の直後で分割する。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")

#: 節区切り (読点を含む) の直後で分割する。文単位で属性を分けられないときの
#: 第 2 段。日本語は 1 文へ複数属性を読点で並べるのが常態で、文単位だけでは
#: 「職業はデータベース管理者で、名古屋に住んでいます。」を分離できない。
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[。！？!?、，,])\s*")

#: 節を根拠に採ったときに末尾へ残る接続。読点と、その直前の連用形の
#: 断定辞 ``で`` (「職業はデータベース管理者で、」) を落として命題にする。
#: ``て`` (「住んでいて、」) は動詞の一部なので対象外。
_CLAUSE_TAIL_RE = re.compile(r"(?<![ていしっ])で?[、，,]\s*$")


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
        # 疑問の判定だけ抽出側の式を使う。読み出し側 (text_quality) の
        # ``_INTERROGATIVE_TAIL_RE`` は語尾の閉じた集合で、こちらは
        # 「丁寧形の助動詞 + か」を拾う (体言止めの取りこぼしが違う)。
        # 依頼形の判定は共通実装を使う。
        s = sentence.strip()
        if _QUESTION_ENDING_RE.search(s):
            return True
        # 「〜でしたよね」「〜でしたっけ」は確認を求める問い。``です/ます + か``
        # を見る _QUESTION_ENDING_RE では拾えないが、本人の言明ではないので
        # ファクトの根拠にしない (2026-08-27 ライブ監査 T14: 「私の名前は御堂
        # ではなく田中でしたよね。」が mem.personal.name のファクトになり、
        # 訂正前の値と並んで live のまま残った)。
        if _CONFIRMATION_SEEKING_RE.search(s):
            return True
        if not _REQUEST_ENDING_RE.search(s):
            return False
        return not _asserts_before_request(s)

    return all(_is_non_assertive(s) for s in relevant)


def _mentions_any(sentence: str, words: tuple[str, ...]) -> bool:
    """``sentence`` が ``words`` のいずれかを含むか (正規化して比較)。"""
    if not words:
        return False
    normalized = _normalize_trigger(sentence)
    return any(w in normalized for w in words)


#: 「値」とみなす内容語の並び (漢字 / カタカナ / ラテン文字 / 数字が 2 文字以上)。
#: ひらがなを入れないのは、述語 (「変わりました」「しています」) が全部
#: ひらがなで、値の有無を弁別できなくなるため。
_CONTENT_RUN_RE = re.compile(
    r"[一-鿿゠-ヿA-Za-z0-9][一-鿿゠-ヿA-Za-z0-9]+"
)


def _states_attribute_value(sentence: str, attr_words: tuple[str, ...]) -> bool:
    """``sentence`` がその属性の **値** を述べているか (純粋関数)。

    属性語を取り除いてなお内容語が残れば値がある、と判定する:

        「職業は建築設計士です。」 − 職業 → 「建築設計士」が残る → 値あり
        「職業も変わりました。」   − 職業 → 「変」しか残らない   → 値なし
        「私の名前は小川です。」   − 名前 → 「小川」が残る       → 値あり

    値なしと判定した文だけが後続文を引き継ぐので、**誤って「値あり」と
    出るのは安全側** (従来どおり絞り込むだけ)。
    """
    stripped = sentence
    for word in attr_words:
        stripped = stripped.replace(word, " ")
    return bool(_CONTENT_RUN_RE.search(stripped))


#: **常体 (だ・である体) の言明** の文末。用言で終わる形を採る。
#:
#: ``is_plain_statement`` の ``_STATEMENT_TAIL_RE`` は敬体 (です / ます) を
#: 前提にしているため、「このやり方が良いと思う」「リモートの方が集中できる」
#: のような常体の自己申告を落としてしまう。
#:
#: 過剰に通るのは **安全側** — このゲートは既存の否定形ガード
#: (``_tag_evidence_is_question_only`` / ``_assertive_evidence``) に AND で
#: 重ねる追加条件なので、通しすぎても新しい漏れは作らない。逆に落としすぎると
#: 正当な言明が記憶されなくなる。
#:
#: 想起の問いは主題の ``は`` か体言で終わる (「私の職業は。」「ペットをもう
#: 一度。」) ので、用言の語尾では拾われない。
_PLAIN_FORM_TAIL_RE = re.compile(
    r"(?:う|る|た|い|だ|ぬ|く|す|む|ぶ|つ|ぐ|ず|ん)[。．.!！\s]*$",
)


def _states_a_value(content: str, attr_words: tuple[str, ...] = ()) -> bool:
    """発話が **値を言明している** か (純粋関数)。

    ユーザー由来のファクト (personal_fact / preference / emotion / opinion) を
    作ってよいかの **肯定側のゲート**。既存の判定はすべて「疑問形か / 依頼形か」
    を語彙で列挙する否定形で、外れた形は素通りする。

    ``is_plain_statement`` を **文単位** で見て 1 文でも平叙なら通す。
    発話全体に掛けると「言明 + 依頼」の複合文
    (「私はコーヒーが好きですが、紅茶のおすすめも教えてください。」) が
    依頼マーカーで落ちてしまい、2026-07 以来の
    ``test_request_guard_does_not_swallow_real_preferences`` が退行する。
    1 文に収まった複合形は ``_asserts_before_request`` が拾う (同じ判定を
    ``_carries_no_value`` も使っている)。

    ``attr_words`` を渡すと **体言止めの言明** も通す。``is_plain_statement``
    は平叙の文末 (``です`` / ``ます`` 等) を要求するため、「私は東京在住」の
    ような名詞で終わる自己申告を落としてしまう
    (``test_memory_note_trace_id_propagates_to_semantic_fact``)。属性語を
    取り除いてなお内容語が残るなら値を述べている
    (:func:`_states_attribute_value`) ので、そちらでも通す。
    """
    if not content:
        return False
    if _asserts_before_request(content):
        return True
    for raw in _SENTENCE_SPLIT_RE.split(content):
        sentence = raw.strip()
        if not sentence:
            continue
        if is_plain_statement(sentence) or _PLAIN_FORM_TAIL_RE.search(sentence):
            return True
    return bool(attr_words) and _states_attribute_value(content, attr_words)


def _assertive_evidence(evidence: str) -> str:
    """証拠テキストから **言明の文だけ** を残す (純粋関数)。

    ``strip_interrogative_sentences`` は問いだけを落としていたので、依頼の文
    (「〜してください。」) が object に残っていた。属性が解決できない発話は
    フォールバック subject (``mem.personal.user``) へ落ちる設計なので、
    **依頼だけの発話ほどそのスロットに溜まる**。

    実インシデント (2026-08-28 ライブ監査、実ストアで確認):
    ``mem.personal.user states: 私の好みを踏まえて、通勤についてアドバイスを
    ください。`` が live で残り、以後 ``[関連する記憶]`` に依頼文が並んだ。
    2026-08-19 の監査でも「依頼形の発話がファクト化して競合の当事者になる」と
    記録されている。

    判定は ``_carries_no_value`` (= ``_tag_evidence_is_question_only`` の内側)
    と同じ規則。依頼節より前に本人の言明がある複合文
    (「私は〜なので、〜してください。」) は残る。
    """
    kept = [
        sentence
        for raw in _SENTENCE_SPLIT_RE.split(evidence or "")
        if (sentence := raw.strip()) and not _carries_no_value(sentence)
    ]
    return "".join(kept)


def _carries_no_value(sentence: str) -> bool:
    """1 文が「値の表明ではない」か (疑問形 または 純粋な依頼形)。

    ``_tag_evidence_is_question_only`` の内側と同じ規則。属性文に続く文を
    引き継ぐかどうかの判定に使う (「よろしくお願いします。」を値として
    連れて行かないため)。
    """
    s = sentence.strip()
    if _QUESTION_ENDING_RE.search(s):
        return True
    if not _REQUEST_ENDING_RE.search(s):
        return False
    return not _asserts_before_request(s)


def _attribute_evidence_text(
    content: str,
    attr_words: tuple[str, ...],
    all_attr_words: tuple[str, ...] = (),
) -> str:
    """``attr_words`` を含む文だけを連結して返す (絞れなければ空文字列)。

    1 発話で複数の属性を述べても、fact の subject は 1 つしか付かない。
    object に発話全文を入れると、その subject と無関係な値まで同居する。
    実インシデント 2026-08-22 ライブ監査 (100 ターン): 初回の自己紹介
    「私の名前は小川です。いま取り組んでいるプロジェクトは EvorefAudit と
    いう名前で、締め切りは 2026年9月30日です。」が丸ごと
    ``mem.personal.name`` の object になった。その後ユーザーは
    プロジェクト名と締め切りを訂正したが、訂正は
    ``mem.world.assertion.deadline_correction`` 等の **別 subject** に入るため
    ``mem.personal.name`` は supersede されない。結果、両方が
    ``[関連する記憶]`` に並び、より直截な自己申告に見える古い方が採用されて
    「EvorefAudit / 2026年9月30日」と回答した。

    絞り込みは **文が 2 つ以上あり、実際に減る** ときだけ行う。該当文が
    無い / 全文が該当する場合は空文字列を返し、呼出側は従来どおり全文を使う。

    **属性語を含まない文は直前の属性文に属する** (``all_attr_words`` を渡した
    場合)。日本語は「属性名を出す文」と「値を述べる文」が分かれるのが普通で、
    属性語だけで絞ると **値そのものが落ちる**:

        「職業も変わりました。今は構造設計士です。」
          → 旧: 「職業も変わりました。」   (値が無い)
          → 新: 「職業も変わりました。今は構造設計士です。」

    実インシデント (2026-08-27 ライブ監査): 上の発話から
    ``mem.personal.occupation`` の object が「職業も変わりました。」になり、
    新セッションで「あなたの現在の職業は**「職業も変わりました」という記録しか
    確認できず**、具体的な職種は不明です。」とユーザーへ露出した。

    引き継ぐのは **どの属性語も含まない** 文だけなので、別属性の値を巻き込む
    ことはない (含んでいればその属性の文として扱われる)。ただし疑問形 /
    純粋な依頼形は値を運ばないので引き継がない — 日本語の自己紹介は
    「…といいます。よろしくお願いします。」で終わるのが常態で、
    そのまま連れて行くと object に依頼文が混ざる。

    Args:
        attr_words: この属性のトリガ語。
        all_attr_words: この発話で解決した **全属性** のトリガ語。省略時は
            引き継ぎを行わず従来どおりの絞り込みになる (後方互換)。
    """
    if not attr_words or not content:
        return ""
    narrowed = _narrow_by_units(
        content, _SENTENCE_SPLIT_RE, attr_words, all_attr_words,
    )
    if narrowed:
        return narrowed
    # 文単位で分けられなかった。1 文へ複数属性を読点で並べた形
    # (「職業はデータベース管理者で、名古屋に住んでいます。」) は節単位なら
    # 分離できる。**他属性のトリガ語が渡っているときだけ** 試す — 単一属性の
    # 発話を節へ刻むと、値を運ぶ節が落ちて命題が壊れる。
    if not _has_other_attribute_words(attr_words, all_attr_words):
        return ""
    narrowed = _narrow_by_units(
        content, _CLAUSE_SPLIT_RE, attr_words, all_attr_words,
    )
    return _CLAUSE_TAIL_RE.sub("", narrowed) if narrowed else ""


def _has_other_attribute_words(
    attr_words: tuple[str, ...], all_attr_words: tuple[str, ...],
) -> bool:
    """``all_attr_words`` に ``attr_words`` 以外の属性語が含まれるか。"""
    own = set(attr_words)
    return any(word not in own for word in all_attr_words)


def _narrow_by_units(
    content: str,
    splitter: re.Pattern[str],
    attr_words: tuple[str, ...],
    all_attr_words: tuple[str, ...],
) -> str:
    """``splitter`` の単位で ``attr_words`` の根拠だけを残す (絞れなければ "")。

    :func:`_attribute_evidence_text` の本体。文単位・節単位で 2 回使うため、
    分割規則だけを差し替えられるように切り出してある。
    """
    units = [u for u in splitter.split(content) if u.strip()]
    if len(units) < 2:
        return ""

    kept: list[str] = []
    awaiting_value = False
    for unit in units:
        if _mentions_any(unit, attr_words):
            kept.append(unit)
            # 値を述べていない属性文 (「職業も変わりました。」) だけが
            # 後続文を引き継ぐ。値がある文は従来どおり単独で完結させる。
            awaiting_value = not _states_attribute_value(unit, attr_words)
            continue
        if not all_attr_words:
            # 引き継ぎ無効 (呼出側が全属性語を渡していない) = 従来どおり。
            continue
        if _mentions_any(unit, all_attr_words):
            # 別属性の文。以降の「属性語なし」文はそちらに属する。
            awaiting_value = False
            continue
        if awaiting_value and not _carries_no_value(unit):
            kept.append(unit)
            awaiting_value = False  # 引き継ぐのは 1 文だけ

    if not kept or len(kept) == len(units):
        return ""
    return "".join(kept).strip()


#: 規則ベース正規化で残す文の下限 (これ未満なら正規化失敗として原文を使う)。
_STATEMENT_MIN_CHARS = 2

#: 接頭辞剥がしの最大反復。「ところで、私は…」のような入れ子を解くため繰り返すが、
#: 無限ループにはしない。実際は 2 回で収束する。
_STATEMENT_STRIP_PASSES = 3

#: ``compress_turn(style="summary")`` の圧縮マークと末尾の元文字数。
_SUMMARY_MARK = "[要約] "
_SUMMARY_TAIL_RE = re.compile(r"…（\d+文字）\s*$")

#: 文末の断定辞と句点。命題には要らないので落とす。
#: ``になりました`` 等の複合述語は **落とさない** — 値の一部を運んでいる
#: (「11月10日に変更になりました」)。
_STATEMENT_TAIL_RE = re.compile(r"(?:です|でした|だ|である|でございます)?\s*[。．.]?\s*$")

#: 「正しくは B」— 訂正の後半だけが現在値。前半 (「さっき神戸に住んでいると
#: 言いましたが、」) は **古い値を運ぶだけ** なので落とす。
_CORRECTION_RIGHT_RE = re.compile(r"^.*?正しくは、?\s*(?P<new>.+)$")

#: 「<枠>は<旧>ではなく<新>」— 枠を残して値だけ差し替える。
#: 枠 (``猫の名前は``) を残すのは、``[関連する記憶]`` の 1 行が
#: ``mem.personal.pet states: ルナ`` だけになると何の名前か読めなくなるため。
#: 枠が無い形 (「緑茶ではなく紅茶です」) は新しい値だけを残す。
#: ``ない`` (単なる否定) は含めない。「急ぎではないので、ゆっくりで大丈夫です。」を
#: 差し替えると「ので、ゆっくりで大丈夫」という壊れた命題になる。訂正の意味を
#: 持つのは対比の ``なく`` / ``なくて`` だけで、これは
#: ``note_builder.CORRECTION_FORM_TRIGGERS`` (``ではなく`` / ``じゃなくて``) とも揃う。
_CORRECTION_NOT_RE = re.compile(
    r"^(?P<head>.*?)(?:では|じゃ)なく(?:て)?、?\s*(?P<new>.+)$",
)

#: ``ではなく`` が訂正でない形。これらは「A に加えて B」「A というわけではない」で、
#: 差し替えると **A が消える**。実例: 「Python だけではなく TypeScript も書きます」を
#: 差し替えると Python が失われる。
_NOT_A_CORRECTION_RE = re.compile(r"(?:だけ|のみ|ばかり|わけ|訳|はず|そう)(?:では|じゃ)なく")

#: 枠の切れ目 (最後の話題マーカー)。``猫の名前はソラではなく`` の ``は`` を指す。
_FRAME_SPLIT_RE = re.compile(r"^(?P<frame>.*[はが])(?P<old>[^はが]*)$")


def _reduce_correction_statement(text: str) -> str | None:
    """訂正形の発話から **現在値だけ** を残した命題を返す (純粋関数)。

    訂正は必ず「古い値」と「新しい値」を 1 文に同居させる。``object`` は原文を
    残す設計なので、正規化しないと ``[関連する記憶]`` に古い値がそのまま並ぶ。
    実インシデント (2026-08-25 ライブ監査の追調査): 訂正後の
    ``mem.personal.location`` の本文が「さっき**神戸**に住んでいると言いましたが、
    正しくは横浜です。」で、**訂正ファクト自身が古い値を運んで**いた。

    Returns:
        正規化後の命題。訂正形でなければ ``None``。
    """
    t = (text or "").strip()
    if not t:
        return None
    m = _CORRECTION_RIGHT_RE.match(t)
    if m:
        return _STATEMENT_TAIL_RE.sub("", m.group("new").strip()).strip() or None
    if _NOT_A_CORRECTION_RE.search(t):
        return None
    m = _CORRECTION_NOT_RE.match(t)
    if m is None:
        return None
    new = _STATEMENT_TAIL_RE.sub("", m.group("new").strip()).strip()
    if not new:
        return None
    frame_m = _FRAME_SPLIT_RE.match(m.group("head").strip())
    frame = frame_m.group("frame") if frame_m else ""
    return (frame + new) if frame else new


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


def _relevant_sentences(
    text: str, trigger_words: tuple[str, ...],
) -> list[str]:
    """トリガ語を含む文と、**その値を運ぶ後続文** を返す (純粋関数)。

    トリガ語を含む文だけを残すと、日本語でごく普通の
    「属性名を出す文」+「値を述べる文」の組から **値の側が落ちる**:

        「職業も変わりました。今は構造設計士です。」
          → 旧: 「職業も変わりました」   (値が無い)
          → 新: 「職業も変わりました。今は構造設計士です」

    :func:`_attribute_evidence_text` は同じ規則を既に持っていたが、
    ``statement`` 側にだけ無かった。``fact.text`` は statement を優先する
    ため、根拠本文を正しく絞っても **注入される 1 行は値を失っていた**
    (2026-08-27 ライブ監査の再検証で、新規セッションへ「職業も変わりました」
    という記録だけが露出した)。
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    kept: list[str] = []
    awaiting_value = False
    for sentence in sentences:
        if any(t in sentence.lower() for t in trigger_words):
            kept.append(sentence)
            awaiting_value = not _states_attribute_value(sentence, trigger_words)
            continue
        if awaiting_value and not _carries_no_value(sentence):
            kept.append(sentence)
            awaiting_value = False  # 引き継ぐのは 1 文だけ
    return kept


def normalize_statement(
    content: str,
    trigger_words: tuple[str, ...],
    attr_words: tuple[str, ...] = (),
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

    5. 訂正形の古い値 (``_reduce_correction_statement``)
    6. 文末の断定辞と句点 (``_STATEMENT_TAIL_RE``)

    5 と 6 を足したのは、``statement`` が **一度も埋まっていなかった** ため
    (2026-08-25 実測: live 143 件中 0 件)。3/4 だけでは素直な平叙文
    (「私の名前は小川浩之です。」) が ``kept == text`` で ``None`` に落ち、
    ``fact.text`` が実運用上ずっと ``object`` と同一だった。訂正形はとくに
    **古い値を本文に同居させる** ので、正規化しないと訂正後も陳腐値が
    ``[関連する記憶]`` に並ぶ。

    「紅茶派です」→「紅茶」のような助詞解析を要する **値の抽出** は依然として
    ここでは行わない (取りこぼしは LLM 併用で扱う想定)。

    Returns:
        正規化後の命題。原文と変わらない / 短くなりすぎる場合は ``None``
        (呼出側は ``statement`` を立てず ``object`` をそのまま使う)。
    """
    text = (content or "").strip()
    if not text:
        return None
    kept = strip_interrogative_sentences(text)
    # 文の取捨は **その属性のトリガ語** で行う。fact_type のトリガ語
    # (``私は`` / ``ではなく`` / ``変わりました`` …) は属性を指さないので、
    # 「値なしの属性文」を判定できず後続文の引き継ぎが働かない
    # (``職業も変わりました。今は構造設計士です。`` の値が落ちていた)。
    relevance_words = attr_words or trigger_words
    if relevance_words:
        relevant = _relevant_sentences(kept, relevance_words)
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
    corrected = _reduce_correction_statement(kept)
    if corrected:
        kept = corrected
    else:
        kept = _STATEMENT_TAIL_RE.sub("", kept).strip()
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


#: 値アンカーで採用する最短の値。1 文字の値 (「猫」「A」) は別の文へ偶然
#: 含まれるため、スロットの決め手にしない。
_VALUE_ANCHOR_MIN_CHARS = 2


def _value_anchors(value: str) -> tuple[str, ...]:
    """スロットの現在値から、照合に使う文字列を取り出す (純粋関数)。

    ``fact.text`` は命題 (「名古屋に住んでいます」) であって値そのもの
    (「名古屋」) ではない。訂正はふつう値だけを名指す — 「さっき**名古屋**と
    言いましたが、正しくは横浜です。」— ので、命題まるごとで部分一致を取ると
    永久に当たらない。

    形態素解析は入れず、``_CONTENT_RUN_RE`` の内容語 (漢字 / カタカナ /
    ラテン文字 / 数字が 2 文字以上) を候補にする。助詞と述語はひらがな
    なので自然に落ちる:

        「名古屋に住んでいます」        → ("名古屋に住んでいます", "名古屋")
        「職業はデータベース管理者」    → (..., "職業", "データベース管理者")

    命題そのものも候補に残す (値がひらがなだけの場合の保険)。
    """
    text = (value or "").strip()
    if len(text) < _VALUE_ANCHOR_MIN_CHARS:
        return ()
    anchors = [text]
    anchors.extend(
        run for run in _CONTENT_RUN_RE.findall(text)
        if len(run) >= _VALUE_ANCHOR_MIN_CHARS
    )
    return tuple(dict.fromkeys(anchors))


def resolve_value_anchored_attributes(
    notes: Iterable[MemoryNote],
    live_values: dict[tuple[str, str], tuple[str, ...]],
) -> dict[tuple[str, str], str]:
    """訂正が **どのスロットの現在値** を名指しているかで属性を決める (純粋関数)。

    ``(note_id, tag) -> attr`` を返す。

    従来、訂正のスロットは属性語 (``住んで`` / ``在住`` / ``職業`` …) の
    列挙で解決していた。日本語の訂正は属性名を落として値だけを言い直すのが
    普通なので、この列挙は繰り返し破れてきた — 2026-07-26 / 2026-08-22 /
    2026-08-25 と 3 回、実インシデントのたびに語を足している。

    2026-08-27 ライブ監査で 4 回目が出た::

        「さっき名古屋と言いましたが、正しくは横浜です。訂正します。」

    ``location`` のトリガ語 (``在住`` / ``住んで`` / ``住まい`` / ``出身`` /
    ``引っ越`` / ``地元`` / ``勤務地``) を **1 つも含まない**。結果
    ``candidate_fact_tags`` が 0 件を返し、``resolve_inherited_attributes``
    による継承にすら到達せず、``横浜`` を含むファクトが 1 件も生まれなかった。
    新規セッションでの想起は 2 回とも訂正前の「名古屋」を返した。

    語を足す代わりに **ストアが既に持っている値** を手掛かりにする。訂正は
    必ず古い値と新しい値を 1 文へ同居させるので、古い値の側が既存スロットの
    現在値と一致すれば、それがこの訂正の宛先である。

    - 「さっき **名古屋** と言いましたが、正しくは横浜です。」
      → ``名古屋`` は ``mem.personal.location`` の現在値 → location への訂正
    - 「さっきの 1234 × 5678 の答えは間違っています。正しくは 7006653 です。」
      → どのスロットの現在値とも一致しない → **不採用** (2026-08-23 の
        実インシデントで birthday と food の両方へ書かれた発話)

    語彙表の保守が要らず、宛先が実在するスロットに限られるので誤爆も閉じる。
    自属性が解決できる発話・継承で解決できる発話には手を出さない
    (呼出側が最後の手段として使う)。

    Args:
        notes: 走査対象のノート。``is_correction`` が立っているものだけ見る。
        live_values: ``{(tag, attr): (現在値, ...)}``。ストアの live ファクトの
            ``text`` から呼出側が組む。

    Returns:
        ``(note_id, tag) -> attr``。解決できなかったノートは含まない。
    """
    if not live_values:
        return {}
    resolved: dict[tuple[str, str], str] = {}
    for note in notes:
        if not getattr(note, "is_correction", False):
            continue
        content = note.content or ""
        if not content:
            continue
        haystack = _normalize_trigger(content)
        #: 同じ発話が複数スロットの値を含むことがある (「職業はデータベース
        #: 管理者ではなく…」は occupation の値と pet の「猫」を同時に含みうる)。
        #: **より長く一致した方** を採る — 短い値ほど偶然の混入になりやすい。
        best: dict[str, tuple[int, str]] = {}
        for (tag, attr), values in live_values.items():
            for value in values:
                for anchor in _value_anchors(value):
                    if _normalize_trigger(anchor) not in haystack:
                        continue
                    current = best.get(tag)
                    if current is None or len(anchor) > current[0]:
                        best[tag] = (len(anchor), attr)
        for tag, (_length, attr) in best.items():
            resolved[(note.id, tag)] = attr
    return resolved


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
    #: セッションごとの「直近で属性が確定した 1 発話」の tag -> attribute。
    #:
    #: 継承は **その 1 発話が解決したスロットだけ** を丸ごと写す。タグごとに
    #: 独立した「最後に見た属性」を継ぐと、別々の発話に由来する無関係な
    #: スロットが 1 件の訂正へ同時に流れ込む。実インシデント (2026-08-23
    #: ライブ監査): 算術の訂正 1 件が mem.personal.birthday と
    #: mem.preference.food の両方へ書かれ、どちらも supersede されずに残った。
    #:
    #: 同一発話が複数タグで同じ属性を解決した場合 (「私の好きな飲み物は緑茶です」
    #: は personal_fact / preference の双方が beverage) は両方を継ぐ — 先行発話の
    #: スロット構成をそのまま写すだけなので、無関係なスロットは混ざらない。
    last_resolved: dict[str, dict[str, str]] = {}
    inherited: dict[tuple[str, str], str] = {}
    for note in ordered:
        content = note.content or ""
        session = note.session_id or ""
        resolved: dict[str, str] = {}
        for tag in _USER_SUBJECT_TAGS_ORDERED:
            attr, _words = resolve_fact_attribute_match(
                content, tag, mode="chat", triggers_dir=builder.triggers_dir,
            )
            if attr:
                resolved[tag] = attr
        if resolved:
            last_resolved[session] = resolved
            continue
        if not getattr(note, "is_correction", False):
            continue
        for tag, attr in (last_resolved.get(session) or {}).items():
            inherited[(note.id, tag)] = attr
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
        # 属性語も継承も効かない訂正を、既存スロットの現在値で拾い直す。
        # 「さっき名古屋と言いましたが、正しくは横浜です。」は location の
        # トリガ語を 1 つも含まず、候補タグ 0 件で入口に到達しなかった
        # (resolve_value_anchored_attributes の説明を参照)。
        value_anchored = resolve_value_anchored_attributes(
            note_list, ctx.live_attribute_values,
        )
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
            # 値アンカーで宛先が決まったタグは、トリガ語が無くても候補にする。
            tags = tags + [
                tag for (note_id, tag) in value_anchored
                if note_id == note.id and tag not in tags
            ]
            for tag in tags:
                if tag not in self.SUPPORTED_TAGS:
                    continue
                fact_type: FactType = tag  # type: ignore[assignment]
                kind = _KIND_BY_TAG[tag]
                #: (subject, その属性の根拠に絞った本文) の列。
                #: 1 発話が複数属性を述べていれば複数要素になる。
                #: (subject, 根拠本文, その属性のトリガ語) の列。
                attr_specs: list[tuple[str, str, tuple[str, ...]]] = []
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
                    # **否定形のガードだけでは想起の問いがファクトになる。**
                    # 上の判定も ``_assertive_evidence`` も「疑問形か / 依頼形か」
                    # を語彙で列挙する閉じた規則で、体言止めの想起は素通りする。
                    # そして属性スロットの多くは ``single_valued`` なので、
                    # **問いから作られたゴミが正しい値を supersede して消す**。
                    #
                    # 実インシデント (2026-08-30 ライブ監査の検証、実ストア)::
                    #
                    #   LIVE  mem.personal.pet         | ペットをもう一度。
                    #   SUPER mem.personal.pet         | 柴犬を1匹飼っています。
                    #   LIVE  mem.personal.occupation  | 職業
                    #   SUPER mem.personal.occupation  | …SREになりました。
                    #   LIVE  mem.personal.origin      | 私の出身地と居住地をもう一度。
                    #
                    # 「私の名前、住所、職業、ペットをもう一度。」という **問い
                    # そのもの** が、その問いの答えを永久に破壊していた。実機の
                    # 回答は「あなたの職業は会社員です」「ペットは、猫です」。
                    #
                    # そこで注入側 (``MemoryInjector._restated_slots``) と同じく
                    # **肯定の証拠を要求する**。代償は非対称 — 取りこぼしても
                    # ファクトが 1 件増えないだけだが、誤って作ると正しい値が消える。
                    # 実データ (experience 313 件中、候補になった 102 件) で
                    # 言明 18 / 問い・依頼 84 を完全分離した。
                    if not _states_a_value(content, trigger_words):
                        result.notes_skipped += 1
                        continue
                    # subject を属性単位に分割する (mem.personal.machine_spec 等)。
                    # 一律 "user" だと、競合検出が (subject, predicate) キーで
                    # 類似度を見ずにグルーピングするため、無関係な事実
                    # (コメント方針 vs GPU 仕様) まで同一事実の競合版と誤判定され
                    # 恒久 pending 化する (2026-07-25)。辞書に無ければ従来どおり
                    # "user" へフォールバックするので退行しない。
                    # **1 発話に複数の属性があれば、そのぶんファクトを作る。**
                    # 単数版は YAML 記載順で最初の 1 件を返して打ち切るため、
                    # 「私は小川宏之といいます。埼玉県川口市に住んでいます。」
                    # から location しか作られず名前が 1 件も残らなかった
                    # (2026-08-25 ライブ監査:「私の名前を覚えていますか。」に
                    # 答えられなかった)。日本語の自己紹介は 1 発話へ複数属性を
                    # 詰めるのが普通なので、打ち切る設計そのものが噛み合わない。
                    # 根拠文は属性ごとの trigger 語で絞るので object は混ざらない。
                    matches = resolve_fact_attribute_matches(
                        content, tag, mode="chat",
                        triggers_dir=self._builder.triggers_dir,
                    )
                    if not matches:
                        # 属性語を落とした訂正は直前の言明からスロットを継ぐ
                        # (resolve_inherited_attributes の説明を参照)。
                        # 継承も効かないときは、訂正文が名指した既存スロットの
                        # 現在値から決める (value_anchored)。継承より後に置くのは
                        # 「直前の言明」の方が近い文脈だから。
                        matches = [(
                            inherited.get((note.id, tag))
                            or value_anchored.get((note.id, tag))
                            or "user",
                            (),
                        )]
                    # 「属性語を含まない文は直前の属性文に属する」判定に、
                    # この発話で解決した全属性のトリガ語を渡す
                    # (詳細は _attribute_evidence_text の docstring)。
                    all_attr_words = tuple({
                        word for _, words in matches for word in words
                    })
                    attr_specs = [
                        (
                            make_mem_subject(kind, attr or "user"),
                            _attribute_evidence_text(
                                content, attr_words, all_attr_words,
                            ) or content,
                            attr_words,
                        )
                        for attr, attr_words in matches
                    ]
                else:
                    # world_fact もユーザーが断定した知識に限る。トリガ語
                    # (「とは」「である」) を含む文が疑問形/依頼形しかない
                    # ノートは、_USER_SUBJECT_TAGS と同じ理由で候補にしない。
                    # このガードが world_fact 側だけ抜けていたため、質問文が
                    # そのまま「世界の事実」になっていた (2026-08-22 ライブ監査:
                    # mem.world.RRF.3fa274d4 is:「RRF (Reciprocal Rank Fusion)
                    # とは何ですか？」)。
                    if _tag_evidence_is_question_only(
                        content, self._builder.fact_triggers.get(tag, ()),
                    ):
                        continue
                    # keyword (sanitized) + 内容ハッシュを parts に使用
                    # (keyword 単独では subject が衝突する)
                    keyword = _world_fact_keyword(note)
                    if keyword == _SAFE_KEYWORD_FALLBACK:
                        # 有効な keyword を導けないノートは world_fact 化しない
                        # (mem.world.unknown の量産防止)
                        continue
                    attr_specs = [(
                        make_mem_subject(
                            kind, *_world_fact_subject_parts(keyword, content),
                        ),
                        content,
                        (),
                    )]
                for subject, evidence, attr_words in attr_specs:
                    # 依頼の文は記憶として意味を持たない。問いを落とすのと
                    # 同じ理由で落とし、**1 文も残らなければファクトにしない**
                    # (2026-08-28 ライブ監査: 「私の好みを踏まえて、通勤について
                    # アドバイスをください。」が mem.personal.user の live な
                    # ファクトになり、以後 [関連する記憶] に依頼文が並んだ。
                    # 属性が解決できない発話はフォールバック subject へ落ちる
                    # ので、**依頼だけの発話ほどこのスロットに溜まる**)。
                    object_text = _assertive_evidence(evidence)
                    if not object_text:
                        continue
                    fact = self.make_fact(
                        subject=subject,
                        predicate=_PREDICATE_BY_TAG.get(tag, "states"),
                        # 末尾の問いは記憶として意味を持たないうえ、
                        # [関連する記憶] に「答えではなく問い」が根拠として
                        # 並ぶ原因になる (2026-08-16 監査時点の実データ:
                        #  mem.emotion.user feels: 夜更かしすると次の日つらい
                        #  ですよね。何かいい対策ありますか？)。
                        object_text=object_text,
                        # 規則で切り出せる分だけ命題化する。object (原文) は
                        # 証拠として残し、提示・比較・埋め込みは fact.text 経由で
                        # statement を優先する。切り出せなければ None のままで
                        # 従来と同じ挙動。
                        statement=normalize_statement(
                            evidence,
                            tuple(self._builder.fact_triggers.get(tag, ())),
                            attr_words,
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
