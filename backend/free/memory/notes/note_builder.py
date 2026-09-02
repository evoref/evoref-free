"""NoteBuilder: A-MEM ノート構築（LLM 不要）

設計書 §5.2 / EvorefMem 統合仕様に基づく、キーワード抽出・
タグ付け・候補ファクト分類を一体的に行うクラス群。すべての処理は 0.1ms 以内で
完了し、LLM 呼び出しは行わない。

クラス階層:
- :class:`NoteBuilder` — 共通基底。``extract_keywords`` / ``auto_tag`` /
  ``initial_score`` の汎用ヘルパを提供する。``build()`` は EvorefMem の
  ``MemoryNote`` 拡張フィールド (mode / project_id / source / is_tool_output /
  is_code_block / extraction_skipped) を含む dict を返す。
- :class:`ChatNoteBuilder` — チャットモード用レンズ。``personal_fact`` /
  ``world_fact`` / ``preference`` / ``emotion`` / ``opinion`` の候補抽出を行う。
- :class:`CreateNoteBuilder` — クリエイトモード用レンズ。``project`` /
  ``decision`` / ``commitment`` / ``task`` / ``create`` の候補抽出を行う。
  コードブロックは完全スキップ、ツール出力は extraction_skipped を立てる。

候補ファクトタグは sleep-time Step 8 の Extractor がこのノートを
拾い上げる際のヒントとして使用される。本フェーズでは抽出までは行わない。
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache
import time
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import yaml

from backend.free.core.intent_vocab import is_plain_statement
from backend.free.core.session_mode import is_create_mode
from backend.free.core.text_quality import states_no_user_value
from backend.free.memory.types import MemoryMode, NoteSource
from backend.log_config import get_logger

logger = get_logger("memory.note_builder")


# reasoning モデル (LFM2 / Qwen3 等) が応答に残す <think>...</think> を除去する。
# メモリ抽出が思考を STM ノートに焼き込むと、後続チャットで意味検索により再注入され
# 話題汚染 (例: ニュース質問に過去の天気ノートで返答) を招く。
# cf. backend.free.optimizer.prompt_evolver の同名処理 (pillar 境界のため非共有)。
_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """``<think>...</think>`` (未閉鎖含む) を除去する。

    閉じたブロックは削除し、未閉鎖 ``<think>`` (暴走/打ち切り) 以降はすべて思考と
    みなして破棄する。本文に思考が無ければ原文をそのまま返す。
    """
    if "<think" not in text.lower():
        return text
    text = _THINK_TAG_RE.sub("", text)
    m = _THINK_OPEN_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────
# 候補ファクト判定用トリガ辞書のロード
# ──────────────────────────────────────────────────────────────────────────


#: ``<mode>`` → ``<fact_type>`` → 部分一致トリガ語 tuple
FactTriggerMap = dict[str, dict[str, tuple[str, ...]]]

#: 訂正の **構文形**。``fact_triggers.yaml`` の各 fact_type にも載っているが、
#: これらは「これは訂正だ」としか言わず **何についての訂正か** を示さない。
#: 単独では候補化の根拠にせず、属性が解決できたときだけ採る
#: (:meth:`_ModeAwareNoteBuilder.candidate_fact_tags` を参照)。
#:
#: YAML 側との同期は ``test_correction_form_triggers_are_declared_weak`` が検証する。
CORRECTION_FORM_TRIGGERS: frozenset[str] = frozenset({
    "ではなく",
    "じゃなくて",
    "間違いでした",
    "間違えました",
    "正しくは",
    # 値の **更新告知** 形。訂正は「誤りの指摘」だけでなく「変わった」という
    # 申告でも起きる。記憶層が必要とするのは「その属性の現在値」なので、
    # どちらも同じ扱いをする (学習層は「アシスタントが誤ったか」を数えるので
    # 必要な範囲が違う — feedback.restates_a_value は据え置き)。
    #
    # 実測 (2026-08-27 ライブ監査の修正検証): 「職業も変わりました。今は
    # 構造設計士です。」が from_correction=False で書かれ、自動 supersede が
    # 効かないまま occupation スロットに世代が積み上がった。値の欠落自体は
    # 直った (a8e1607) が、旧世代を畳む側が残っていた。
    "変わりました",
    "変わった",
    "変更になりました",
    "変更しました",
    "actually it's",
    "i meant",
    "has changed",
})

#: 属性スロットを持つ fact_type (``fact_attributes.yaml`` の chat 節)。
#: :func:`restates_attribute_value` が走査する対象。
_ATTRIBUTE_FACT_TYPES: tuple[str, ...] = (
    "personal_fact", "preference", "emotion", "opinion",
)

#: **確認を求める終助詞**。「〜でしたよね」「〜ですよね」「〜でしたっけ」は
#: 値の言明ではなく、記憶の**問い合わせ**である。訂正形の語 (``ではなく``) と
#: 同居しても、ユーザーは値を確定していない。
#:
#: 実インシデント (2026-08-27 ライブ監査 T14): 「私の名前は御堂ではなく田中
#: でしたよね。」の 1 発話で ``mem.personal.name`` が「田中」に置き換わり、
#: 以後そのセッションでずっと「田中です」と答え、成果物 ``summary.md`` にも
#: ``ユーザー名: 田中`` として書き出された。同じ会話でユーザーが「1マイルは
#: 1.2kmですよね」と数値の誤りを主張した際は正しく否定できており、
#: **記憶属性にだけ抵抗機構が無い**という非対称になっていた。
#:
#: 裸の ``ね`` は採らない — 「正しくは横浜ですね」のような相槌混じりの
#: 言い直しまで落ちる。確認を求める形に限る ``よね`` / ``っけ`` だけを見る。
_CONFIRMATION_SEEKING_RE = re.compile(
    r"(?:よ\s*ね|っけ)[。．.、,！!？?\s\"'」』）)]*$",
)


def restates_attribute_value(
    text: str,
    *,
    mode: str = "chat",
    triggers_dir: str | Path | None = None,
) -> bool:
    """この発話が **既知の属性スロットの値を言い直している** か (純粋関数)。

    「訂正の構文形が出ている」+「どの属性の話かが解決できる」の AND。
    ``agent.feedback.restates_a_value`` が取りこぼす ``XではなくY`` 形を、
    記憶層の要件 (「その属性の現在値は何か」) で拾い直すための述語。

    属性の裏取りを必須にするのは、訂正形が **何についての訂正か** を示さない
    ため。「さっきの 1234 × 5678 の答えは間違っています。正しくは 7006653
    です。」のような値の言い直しでない訂正は、属性が解決できないので False。
    これは :meth:`_ModeAwareNoteBuilder.candidate_fact_tags` が訂正形単独の
    証拠に課しているゲートと同じ。
    """
    if not text:
        return False
    if _CONFIRMATION_SEEKING_RE.search(text.strip()):
        # 「〜でしたよね」は記憶の問い合わせ。値は確定していないので
        # スロットを書き換えない (:data:`_CONFIRMATION_SEEKING_RE` 参照)。
        return False
    haystack = _normalize_trigger(text)
    if not any(t in haystack for t in CORRECTION_FORM_TRIGGERS):
        return False
    return any(
        resolve_fact_attribute(
            text, fact_type, mode=mode, triggers_dir=triggers_dir,
        ) is not None
        for fact_type in _ATTRIBUTE_FACT_TYPES
    )


#: **ユーザー自身の属性** を表す fact_type。
#: (``extractors.chat._USER_SUBJECT_TAGS`` と同じ集合。tag 語彙はこちらが持つ)
_USER_SUBJECT_FACT_TAGS: frozenset[str] = frozenset(
    {"personal_fact", "preference", "emotion", "opinion"},
)


def _drop_user_subject_tags_from_assistant(
    tags: list[str], source: str | None,
) -> list[str]:
    """アシスタント発話から **ユーザー自身の属性** タグを外す (純粋関数)。

    ユーザーの属性を主張できるのはユーザーだけ。アシスタントの発話に
    ``personal_fact`` 等が付くと、モデルが口にしただけの値が「過去の記録」
    として STM から再注入され、さらに sleep-time が SemMem の現在値へ
    昇格させる。

    実インシデント (2026-08-31 ライブ監査 T06#3): 「私の誕生日を当ててみて。」
    に **「誕生日は1995年4月12日です。」** と答え (ユーザーは一度も誕生日を
    述べていない)、その発話が ``source=assistant`` / ``tags=['personal_fact']``
    の STM ノートとして保存された。以後この捏造値がユーザーの誕生日として
    想起される。

    アシスタントがユーザーの申告を復唱した場合もタグは落ちるが、**同じ内容は
    ユーザー自身のターンから既に抽出されている** ので失うものは無い。
    """
    if source != "assistant" or not tags:
        return tags
    return [t for t in tags if t not in _USER_SUBJECT_FACT_TAGS]


#: 同梱 default で期待される fact_type 集合 (mode 別)
_EXPECTED_TAGS: dict[MemoryMode, tuple[str, ...]] = {
    "chat": ("personal_fact", "world_fact", "preference", "emotion", "opinion"),
    "create": ("project", "decision", "commitment", "task", "create"),
}

_TRIGGERS_LOCK = threading.Lock()
_TRIGGERS_CACHE: dict[str, FactTriggerMap] = {}

#: プロセス全体で共有される user override 配置先 (通常 ``local/triggers/``)。
#: app_factory が :func:`set_default_triggers_dir` で起動時にセットする。
#: これにより、明示的な ``triggers_dir`` を渡さずに生成された Builder /
#: Extractor (``sleep/extraction.py`` 内の ``ChatExtractor()`` 等) も
#: user override を参照できるようになる。
_DEFAULT_TRIGGERS_DIR: str | Path | None = None


def set_default_triggers_dir(triggers_dir: str | Path | None) -> None:
    """モジュールレベルの default triggers_dir を設定し、singleton builder を
    刷新する。``None`` を渡すと無効化 (= 常に package 同梱 default 使用)。

    app_factory 起動時に ``PathResolver.resolve_local("triggers_dir")`` の
    値を渡す想定。テスト / 再設定のため何度呼んでも安全。
    """
    global _DEFAULT_TRIGGERS_DIR, _CHAT_BUILDER, _CREATE_BUILDER
    _DEFAULT_TRIGGERS_DIR = triggers_dir
    # Singleton を再構築してキャッシュを無効化 (新しい default を読ませる)。
    _CHAT_BUILDER = ChatNoteBuilder()
    _CREATE_BUILDER = CreateNoteBuilder()


def get_default_triggers_dir() -> str | Path | None:
    """現在の default triggers_dir を返す (テスト / introspection 用)。"""
    return _DEFAULT_TRIGGERS_DIR


def _normalize_trigger(text: str) -> str:
    """trigger 語を保存用に正規化 (NFKC + lowercase、空白は保持)。

    ``candidate_fact_tags`` 側は入力テキストを ``str.lower()`` でのみ比較
    するため、全角英数字をそのまま保存すると一致しない。YAML 取込時に
    NFKC + lowercase を掛けることで pin_triggers.yaml と同じ ja/en フラット
    混在運用を許す。
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).lower()


def _coerce_triggers(items: Any) -> tuple[str, ...]:
    if not isinstance(items, list):
        return ()
    out: list[str] = []
    for it in items:
        if not isinstance(it, str):
            continue
        n = _normalize_trigger(it)
        if n:
            out.append(n)
    return tuple(out)


def _coerce_attribute(slug: str, raw: Any) -> "AttributeSpec | None":
    """YAML の 1 スロットを :class:`AttributeSpec` へ変換する。

    2 つの記法を受ける (既存の YAML を書き換えずに済ませるため):

    - ``slug: [trigger, ...]`` — ガード無し (従来形)
    - ``slug: {triggers: [...], requires_self_possessor: true, single_valued: true}``

    trigger が 1 つも無ければ ``None`` (呼出側がスキップする)。
    """
    if isinstance(raw, dict):
        words = _coerce_triggers(raw.get("triggers"))
        requires_self = bool(raw.get("requires_self_possessor", False))
        single_valued = bool(raw.get("single_valued", False))
    else:
        words = _coerce_triggers(raw)
        requires_self = False
        single_valued = False
    if not words:
        return None
    return AttributeSpec(
        slug=slug,
        triggers=words,
        requires_self_possessor=requires_self,
        single_valued=single_valued,
    )


def load_fact_triggers(path: str | Path) -> FactTriggerMap:
    """``fact_triggers.yaml`` をロードして ``{mode: {fact_type: triggers}}``
    を返す。ファイル欠落 / パース失敗時は空辞書を返し、呼び出し側は
    「候補判定無効」相当として扱える (例外を投げない)。
    """
    p = Path(path)
    result: FactTriggerMap = {"chat": {}, "create": {}}
    if not p.exists():
        logger.warning("fact_triggers file not found: %s — candidates disabled", p)
        return result
    try:
        raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("fact_triggers load failed (%s): %s", p, exc)
        return result
    if not isinstance(raw, dict):
        logger.warning("fact_triggers root is not mapping: %s", p)
        return result

    for mode_key in ("chat", "create"):
        section = raw.get(mode_key) or {}
        if not isinstance(section, dict):
            continue
        mapping: dict[str, tuple[str, ...]] = {}
        for tag, triggers in section.items():
            if not isinstance(tag, str):
                continue
            words = _coerce_triggers(triggers)
            if words:
                mapping[tag] = words
        result[mode_key] = mapping

    logger.info(
        "fact_triggers loaded: chat=%d tags / create=%d tags (from %s)",
        len(result["chat"]),
        len(result["create"]),
        p,
    )
    return result


def get_fact_triggers(path: str | Path) -> FactTriggerMap:
    """パスをキーとした ``FactTriggerMap`` のキャッシュ取得 (プロセス内シングルトン)。"""
    key = str(Path(path).resolve())
    with _TRIGGERS_LOCK:
        cached = _TRIGGERS_CACHE.get(key)
        if cached is not None:
            return cached
        triggers = load_fact_triggers(path)
        _TRIGGERS_CACHE[key] = triggers
        return triggers


def reset_fact_triggers_cache() -> None:
    """テスト用: キャッシュ全消去。"""
    with _TRIGGERS_LOCK:
        _TRIGGERS_CACHE.clear()


#: 一人称の所有者。``requires_self_possessor`` の判定に使う。
#: ``_normalize_trigger`` 後 (NFKC + lowercase) の文字列に対して照合する。
_SELF_POSSESSOR_RE = re.compile(
    r"(?:私|僕|俺|自分|わたし|ぼく|おれ|あたし|わたくし|うち)\s*の\s*$",
)

#: 直前が「〜の」で修飾されているか (所有者が明示されている形)。
#: 読点・句点・空白を挟む場合は修飾が切れているとみなす。
_ANY_POSSESSOR_RE = re.compile(r"[^\s、。，．,.]\s*の\s*$")

#: 指示語 (この / その / あの / どの / もの)。「この名前は」「その職業は」の
#: ``の`` は連体詞の一部で所有者ではない。所有格として扱うと
#: :data:`_SELF_POSSESSOR_RE` に落ちて「一人称でない所有者」と判定され、
#: 「私の名前は太郎です。この名前は仮名です。」の後半が本人の name スロットに
#: 結び付かず ``mem.personal.user`` へ流れていた。
#:
#: 文頭・区切り・助詞の直後にあるときだけ指示語と読む — 「ね**この**名前」
#: 「いと**この**名前」の語尾を指示語と誤認すると、ペット / 親族の名前が
#: 話題判定 (:func:`_last_topic_noun`) 経由で本人に付いてしまう。
_DEMONSTRATIVE_NO_RE = re.compile(
    r"(?:^|[\s、。，．,.はがをにでもへ])(?:こ|そ|あ|ど|も)\s*の\s*$",
)


def _has_explicit_possessor(before: str) -> bool:
    """``before`` の末尾が「<所有者>の」で終わるか (指示語は所有者に数えない)。"""
    if not _ANY_POSSESSOR_RE.search(before):
        return False
    return not _DEMONSTRATIVE_NO_RE.search(before)

#: 文の区切り。話題 (``〜は`` / ``〜が`` / ``〜を``) のスコープはここで切れる。
#: 読点は **切らない** — 「猫を飼っていて、名前は…」のような te 形の連結節では
#: 話題が持ち越されるため。
_SENTENCE_BREAK_RE = re.compile(r"[。．.!！?？\n]")

#: 一人称の話題語 (``私は`` / ``僕が`` …) の名詞部分。
_SELF_NOUNS: frozenset[str] = frozenset({
    "私", "僕", "俺", "自分", "わたし", "ぼく", "おれ", "あたし", "わたくし",
    "うち", "i", "me", "my",
})

#: 話題マーカー。直前の名詞がその文の話題を握る。
_TOPIC_MARKERS = "はがを"


def _last_topic_noun(before: str) -> str | None:
    """``before`` の **同じ文の中** で最後に話題マーカーを取った名詞を返す。

    見つからなければ ``None`` (話題が明示されていない)。
    """
    break_match = None
    for break_match in _SENTENCE_BREAK_RE.finditer(before):
        pass
    clause = before[break_match.end():] if break_match else before
    marker_at = max((clause.rfind(m) for m in _TOPIC_MARKERS), default=-1)
    if marker_at <= 0:
        return None
    noun = clause[:marker_at]
    # 名詞は直前の連続した非区切り文字。助詞・記号で切る。
    noun = re.split(r"[\s、，,のにでとへやもか]", noun)[-1]
    return noun or None


def _possessor_is_self(haystack: str, start: int) -> bool:
    """``haystack[start:]`` の trigger がユーザー自身のものか (純粋関数)。

    2 段で見る:

    1. 所有格が明示されている (「X の名前は」) なら X が一人称かどうか。
    2. 所有格が無ければ、**同じ文で最後に話題マーカーを取った名詞** が
       一人称か / 話題が明示されていないか。

    2 が要るのは te 形の連結節のため。「猫を 1 匹飼っていて、名前はソラです。」は
    ``名前は`` の直前に所有格が無いので 1 だけでは通ってしまい、ペットの名前が
    ユーザー本人の名前スロットへ入る。読点で話題は切れないが句点では切れるので、
    「趣味は登山です。名前は小川です。」は従来どおり本人の名前として通る。

    漏れたときの着地は ``mem.personal.user`` で、**本人のスロットは汚れない**。
    """
    before = haystack[:start]
    if _has_explicit_possessor(before):
        return bool(_SELF_POSSESSOR_RE.search(before))
    topic = _last_topic_noun(before)
    return topic is None or topic in _SELF_NOUNS


@lru_cache(maxsize=4096)
def _trigger_variants(word: str) -> tuple[str, ...]:
    """trigger 語とその並列形 (``AttributeSpec._variants`` の説明を参照)。

    trigger 語は YAML 由来で種類が少なく、属性解決はノート × fact_type の
    二重ループで何度も回るため結果をキャッシュする (毎回組み直すと
    ``backend/free/memory`` のテストが 249s → 600s 超に落ちた)。
    """
    if word.endswith(_COPULA_SUFFIX) and len(word) > len(_COPULA_SUFFIX):
        # 「<役割>です」は「<役割>をしています」等とも言う。実インシデント
        # (2026-08-31 ライブ監査): 「医療機器メーカーで組込みエンジニアを
        # しています。」から occupation が 1 件も作られず、新セッションの
        # 「私の職業は？」に答えられなかった (trigger は ``エンジニアです`` のみ)。
        stem = word[: -len(_COPULA_SUFFIX)]
        return (word, *(stem + v for v in _COPULA_ACTIVITY_FORMS))
    if len(word) < 2 or word[-1] not in _TOPIC_PARTICLES:
        return (word,)
    stem = word[:-1]
    return (word, *(stem + c for c in _ENUMERATION_CONNECTORS))


#: 属性 trigger の末尾に付く話題・格助詞。語幹を切り出す判定に使う。
_TOPIC_PARTICLES: frozenset[str] = frozenset({"は", "が", "を"})
#: 属性を並べるときに助詞の代わりに入る連結詞
#: (``AttributeSpec._variants`` の説明を参照)。
_ENUMERATION_CONNECTORS: tuple[str, ...] = ("と", "や", "、", ",", "・")
#: 「<役割>です」形の trigger の語尾。
_COPULA_SUFFIX = "です"
#: ``<役割>です`` と同じ属性を指す「〜している」系の言い回し。
_COPULA_ACTIVITY_FORMS: tuple[str, ...] = (
    "をしています", "をしている", "をやっています", "をやっている",
    "として働", "を務めて",
)


@dataclass(frozen=True, slots=True)
class AttributeSpec:
    """``fact_attributes.yaml`` の 1 スロット。

    ``requires_self_possessor`` は「このスロットはユーザー自身の属性である」
    という宣言。真のとき、trigger の直前が ``<一人称以外>の`` である出現は
    一致とみなさない。

    実インシデント (2026-08-25 ライブ監査の追調査): ``name`` の trigger
    ``名前は`` が「**猫の**名前はソラではなくルナでした。」にも一致し、
    ペットの名前が ``mem.personal.name`` (ユーザー本人の名前スロット) へ入った。
    実ストアで同一 subject に「私の名前は小川浩之です。」と並んで live になり、
    injector の slot collapse は ``(subject, predicate)`` 単位なので
    **構造的に片方が必ず落ちる**。セッション要約からの補完 (別経路) を塞いだ
    直後に「名前については、確認できる情報を持ち合わせていません。」が出た。

    語彙を足すだけの対策 (``pet_name`` スロットの追加) は漏れが必ず残るが、
    このガードがあれば漏れたときの着地点が ``mem.personal.user`` になる —
    **本人のスロットを汚さない安全な失敗**に変わる。「妻の名前は」
    「会社の名前は」「プロジェクトの名前は」にも同じ理屈で効く。
    """

    slug: str
    triggers: tuple[str, ...]
    requires_self_possessor: bool = False
    #: 1 人につき値が 1 つしか成立しないスロットか。
    #:
    #: 真のとき、Step 8 は同じスロットの旧ファクトを **訂正でなくても**
    #: supersede する (:func:`sleep.extraction.supersede_stale_slot_values`)。
    #: 既定は偽 — 宣言していないスロットの挙動は一切変わらない。
    #:
    #: 実インシデント (2026-08-29 ライブ監査): supersede は
    #: ``from_correction`` が立ったファクトにしか走らないため、**訂正ではない
    #: 更新** (「先月、横浜から札幌に引っ越しました」「転職してデータエンジニア
    #: になりました」) で旧値が live のまま残った。``mem.personal.location`` に
    #: 横浜 / 札幌 / 名古屋 が 3 つとも live になり、次セッションの想起で
    #: 旧値が返った (T28: 「39歳, 横浜市, ソフトウェアエンジニア」)。
    #:
    #: 一括で畳まないのは、``pet`` / ``family`` / ``hobby`` のように **1 人が
    #: 複数値を持つのが正当な** スロットで正しい値を落とすため (既存の
    #: supersede 制限の理由そのもの)。宣言は「値が 1 つしか成立し得ない」
    #: スロットだけに限る。
    #:
    #: **``name`` / ``birthday`` には付けない。** これらは他者の値が同じ
    #: スロットへ落ちうる。実測 (2026-08-29): 「猫も1匹飼い始めました。
    #: 名前はミルクです。」が ``name`` へ解決され、単値化していると
    #: ペットの名前が **本人の名前を supersede** する。所有者ガード
    #: (``requires_self_possessor``) は trigger 直前の「<非一人称>の」しか
    #: 見ないため、文をまたいだ暗黙の所有者は落とせない。重複 live のままなら
    #: injector の collapse が選ぶだけで、正しい値が消えることはない。
    #:
    #: 現在の宣言は ``location`` / ``origin`` / ``occupation`` /
    #: ``preference.editor`` の 4 つだけ
    #: (範囲は ``TestSingleValuedSlotDeclaration`` が固定している)。
    single_valued: bool = False

    def _variants(self, word: str) -> tuple[str, ...]:
        """``word`` と、その **並列形** を返す。

        属性の trigger は「名前**は**」「名前**が**」のように助詞で終わるものが
        多い。ところが日本語で属性を並べると助詞は連結詞に置き換わる::

            「私の名前は？」            → 名前は  … 一致
            「私の名前と職業は？」      → 名前と  … **どの trigger にも一致しない**

        その結果 **並べられた属性は、最後の 1 つ以外すべて失われる**。読み出し側
        では尋ねた属性が解決できず、その属性のファクトはコサインの棒を免除
        されないまま落ちる (``pipeline.injector._asked_attributes``)。

        実インシデント (2026-08-31 ライブ監査 t20#1): 「私の名前と職業を教えて
        ください。」に **「記載がありません」**。``mem.personal.name`` は live で
        在ったのに ``asked_attrs=['occupation']`` で name が免除されなかった。
        同 2026-08-30 の検証 V4「私の名前、住所、職業、ペットをもう一度。」も
        同じ形で occupation だけが解決していた。

        語彙を足す対処 (``名前と`` を trigger に追加) は並びの数だけ漏れるので、
        **trigger の構造** から並列形を導く: 助詞で終わる trigger は、その語幹 +
        連結詞も同じ属性を指す。
        """
        return _trigger_variants(word)

    def match(self, haystack: str) -> tuple[str, ...]:
        """``haystack`` に **実際に現れた** trigger 語形を返す (無ければ空タプル)。

        返すのは YAML の原語ではなく :meth:`_variants` が導いた **一致した
        語形そのもの**。抽出側はこの語を ``in`` で本文に当てて根拠文を絞る
        (:func:`~backend.free.memory.extractors.chat._attribute_evidence_text`)
        ため、本文に無い原語を返すと **絞り込みが必ず空振りし、発話全文が
        object になる**。

        実インシデント (2026-08-31 ライブ監査 T01#1):
        「はじめまして。私は佐藤健一といいます。名古屋市中区に住んでいて、
        精密機械メーカーで生産技術のエンジニアをしています。」から
        ``mem.personal.occupation`` の object が **発話全文** になった。
        occupation が一致したのは派生形 ``エンジニアをしています`` なのに、
        返していたのは原語 ``エンジニアです`` で、本文のどの文にも含まれない。
        """
        hit: list[str] = []
        for word in self.triggers:
            if not word:
                continue
            for variant in self._variants(word):
                if not self.requires_self_possessor:
                    if variant in haystack:
                        hit.append(variant)
                        break
                    continue
                # 所有者ガード付きは **出現ごと** に判定する。1 つでも自己所有の
                # 出現があれば一致 (「猫の名前はソラで、私の名前は小川です」)。
                start = haystack.find(variant)
                matched = False
                while start != -1:
                    if _possessor_is_self(haystack, start):
                        hit.append(variant)
                        matched = True
                        break
                    start = haystack.find(variant, start + 1)
                if matched:
                    break
        return tuple(hit)


#: ``{mode: {fact_type: (AttributeSpec, ...)}}``。
#: attribute は YAML 記載順を保持する (先勝ちの優先順位になるため)。
FactAttributeMap = dict[str, dict[str, tuple[AttributeSpec, ...]]]

_ATTRS_LOCK = threading.Lock()
_ATTRS_CACHE: dict[str, FactAttributeMap] = {}


def load_fact_attributes(path: str | Path) -> FactAttributeMap:
    """``fact_attributes.yaml`` をロードする。

    欠落 / パース失敗時は空辞書を返す (属性分割が無効になるだけで、呼出側は
    従来どおり ``"user"`` へフォールバックする)。例外は投げない。
    """
    p = Path(path)
    result: FactAttributeMap = {"chat": {}, "create": {}}
    if not p.exists():
        logger.warning(
            "fact_attributes file not found: %s — attribute subjects disabled", p,
        )
        return result
    try:
        raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("fact_attributes load failed (%s): %s", p, exc)
        return result
    if not isinstance(raw, dict):
        logger.warning("fact_attributes root is not mapping: %s", p)
        return result

    for mode_key in ("chat", "create"):
        section = raw.get(mode_key) or {}
        if not isinstance(section, dict):
            continue
        per_type: dict[str, tuple[AttributeSpec, ...]] = {}
        for tag, attrs in section.items():
            if not isinstance(tag, str) or not isinstance(attrs, dict):
                continue
            ordered: list[AttributeSpec] = []
            for slug, triggers in attrs.items():
                if not isinstance(slug, str):
                    continue
                spec = _coerce_attribute(slug, triggers)
                if spec is not None:
                    ordered.append(spec)
            if ordered:
                per_type[tag] = tuple(ordered)
        result[mode_key] = per_type

    logger.info(
        "fact_attributes loaded: chat=%d types / create=%d types (from %s)",
        len(result["chat"]), len(result["create"]), p,
    )
    return result


def get_fact_attributes(path: str | Path) -> FactAttributeMap:
    """パスをキーとした ``FactAttributeMap`` のキャッシュ取得。"""
    key = str(Path(path).resolve())
    with _ATTRS_LOCK:
        cached = _ATTRS_CACHE.get(key)
        if cached is not None:
            return cached
        attrs = load_fact_attributes(path)
        _ATTRS_CACHE[key] = attrs
        return attrs


def reset_fact_attributes_cache() -> None:
    """テスト用: キャッシュ全消去。"""
    with _ATTRS_LOCK:
        _ATTRS_CACHE.clear()


def resolve_fact_attributes_path(triggers_dir: str | Path | None = None) -> Path:
    """``fact_attributes.yaml`` の解決パスを返す (user override → default)。"""
    from backend.free.memory._defaults import resolve_trigger_file

    return resolve_trigger_file("fact_attributes.yaml", triggers_dir=triggers_dir)


def resolve_fact_attribute_match(
    text: str,
    fact_type: str,
    *,
    mode: str = "chat",
    triggers_dir: str | Path | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """``resolve_fact_attribute`` の内部実装 — スラグと採用した trigger 語を返す。

    trigger 語まで返すのは、抽出側が **その属性の根拠になった文だけ** を
    fact の object に採るため。複数の属性を 1 発話で述べた
    (「私の名前は小川です。プロジェクトは X で、締め切りは 9/30 です。」)
    ときに発話全文を object にすると、``mem.personal.name`` の中へ後から
    訂正される値まで同居し、訂正ファクトは別 subject なので supersede されず
    陳腐化した値が恒久的に注入され続ける (2026-08-22 ライブ監査 ターン100:
    訂正後にも「EvorefAudit / 2026年9月30日」を回答)。
    """
    if not text:
        return None, ()
    if triggers_dir is None:
        triggers_dir = _DEFAULT_TRIGGERS_DIR
    attrs = get_fact_attributes(resolve_fact_attributes_path(triggers_dir))
    per_type = attrs.get(mode) or {}
    ordered = per_type.get(fact_type) or ()
    if not ordered:
        return None, ()
    haystack = _normalize_trigger(text)
    for spec in ordered:
        matched = spec.match(haystack)
        if matched:
            return spec.slug, matched
    return None, ()


#: 1 発話から取り出す属性の上限。セッション上限
#: (``memory.facts.extraction_max_per_session``) を 1 発話で食い潰さないための
#: 保険で、超えた分は YAML 記載順で後ろの属性から **黙って** 落ちていた。
#:
#: 3 では足りない: 日本語の自己紹介は「名前・居住地・職業・ペット」の 4 属性を
#: 1 発話に詰めるのが普通で、記載順で最後に来る ``name`` が押し出されて
#: 本人の名前が 1 件も作られなかった (2026-09-02 監査 B-2)。上限に当たった
#: ときは WARNING で落とした slug を残す (:func:`resolve_fact_attribute_matches`)。
MAX_ATTRIBUTES_PER_TEXT = 6

#: **読み出し側** の上限。書き込み側の保険 (ファクトを作りすぎない) は
#: 読み出しには要らない — 尋ねられた属性を余分に免除しても、その属性の
#: ファクトが無ければ何も注入されないだけで害が無い。逆に切ると
#: 「私の名前、住所、職業、ペットをもう一度。」のような列挙で **後ろの属性が
#: 免除されずコサインのゲートに落ちる** (2026-08-31: 住所を語彙に足した途端、
#: 上限に当たって name が押し出された)。
MAX_ASKED_ATTRIBUTES = 8


def resolve_fact_attribute_matches(
    text: str,
    fact_type: str,
    *,
    mode: str = "chat",
    triggers_dir: str | Path | None = None,
    limit: int = MAX_ATTRIBUTES_PER_TEXT,
) -> list[tuple[str, tuple[str, ...]]]:
    """``text`` に含まれる **すべての** 属性を YAML 記載順で返す。

    ``resolve_fact_attribute_match`` は最初の 1 件で打ち切るため、1 発話で
    複数の属性を述べると **記載順で先に来た属性以外は丸ごと失われる**。

    実インシデント (2026-08-25 ライブ監査): 「私は小川宏之といいます。埼玉県
    川口市に住んでいます。」から ``mem.personal.location`` だけが作られ、
    名前のファクトは 1 件も作られなかった (YAML の記載順で location が name
    より前にある)。その結果「私の名前を覚えていますか。」に答えられなかった。
    日本語の自己紹介は 1 発話に複数属性を詰めるのが普通なので、最初の 1 件で
    打ち切る設計そのものが噛み合っていない。

    根拠文の絞り込み (``_attribute_evidence_text``) は属性ごとの trigger 語で
    行うので、属性を複数返しても object が混ざることはない。
    """
    if not text:
        return []
    if triggers_dir is None:
        triggers_dir = _DEFAULT_TRIGGERS_DIR
    attrs = get_fact_attributes(resolve_fact_attributes_path(triggers_dir))
    per_type = attrs.get(mode) or {}
    ordered = per_type.get(fact_type) or ()
    if not ordered:
        return []
    haystack = _normalize_trigger(text)
    out: list[tuple[str, tuple[str, ...]]] = []
    for spec in ordered:
        matched = spec.match(haystack)
        if matched:
            out.append((spec.slug, matched))
    cap = max(1, limit)
    if len(out) > cap:
        logger.warning(
            "resolve_fact_attribute_matches: %d attributes in one text exceed "
            "cap=%d (fact_type=%s); dropping %s",
            len(out), cap, fact_type, [slug for slug, _ in out[cap:]],
        )
        out = out[:cap]
    return _reassign_name_to_owning_entity(out, haystack)


#: ``mem.<kind>.<attr>`` の ``kind`` → ``fact_attributes.yaml`` の fact_type。
_ATTR_FACT_TYPE_BY_KIND: dict[str, str] = {
    "personal": "personal_fact",
    "preference": "preference",
    "emotion": "emotion",
    "opinion": "opinion",
}


def is_single_valued_subject(
    subject: str,
    *,
    mode: str = "chat",
    triggers_dir: str | Path | None = None,
) -> bool:
    """``subject`` が **単値スロット** として宣言されているか (純粋関数)。

    ``mem.personal.location`` のように ``mem.<kind>.<attr>`` の形をした subject
    だけを見る。辞書に無い / 形が違う / 宣言が無い場合は ``False`` — つまり
    **既定は従来どおり多値扱い**で、宣言したスロット以外の挙動は変わらない。
    """
    if not subject:
        return False
    parts = subject.split(".")
    if len(parts) != 3 or parts[0] != "mem":
        return False
    fact_type = _ATTR_FACT_TYPE_BY_KIND.get(parts[1])
    if fact_type is None:
        return False
    if triggers_dir is None:
        triggers_dir = _DEFAULT_TRIGGERS_DIR
    attrs = get_fact_attributes(resolve_fact_attributes_path(triggers_dir))
    for spec in (attrs.get(mode) or {}).get(fact_type) or ():
        if spec.slug == parts[2]:
            return spec.single_valued
    return False


def states_single_valued_attribute(
    text: str,
    *,
    mode: str = "chat",
    triggers_dir: str | Path | None = None,
) -> bool:
    """**単値スロットの値を言明している**発話か (純粋関数)。

    「先週、仙台から京都に引っ越しました。」「転職してデータアナリストに
    なりました。」のような **属性の更新** を拾う。問い (「私の好きな飲み物は。」)
    と純粋な依頼は :func:`states_no_user_value` で落とす。

    **なぜ要るか** — 単値スロットの旧値を畳むのは Step 8 (Full sleep-time) だが、
    Full は既定でアイドル 10 分 / 繰り延べ上限 30 分。その間、注入は

    - 旧値: SemMem ファクト (Tier 1)
    - 新値: STM ノートだけ (Tier 2)

    となり、**Tier の序列上どうやっても旧値が勝つ**。実測 (2026-08-29 ライブ監査
    F38): 「転職してデータサイエンティストになりました」の直後の新セッションで
    「インフラエンジニアです」と旧値を返し、自己検査も「古い情報は含まれて
    いません」と保証した。

    そこで更新を観測したターンだけ Full を前倒しする
    (:meth:`SleepTimeScheduler.request_full_soon`)。誤爆しても代償は
    「Full が少し早く走る」だけで、**正しい値を消す方向の失敗が無い**。

    訂正 (``restates_a_value``) とは別の経路。あちらは「〜ではなく〜」型の
    言い直しを拾うが、引っ越し・転職のような **訂正ではない更新** は拾えない。
    """
    if not text:
        return False
    if states_no_user_value(text):
        return False
    for kind, fact_type in _ATTR_FACT_TYPE_BY_KIND.items():
        for slug, _ in resolve_fact_attribute_matches(
            text, fact_type, mode=mode, triggers_dir=triggers_dir,
        ):
            if is_single_valued_subject(
                f"mem.{kind}.{slug}", mode=mode, triggers_dir=triggers_dir,
            ):
                return True
    return False


#: 「名前」を持ちうる **ユーザー以外のエンティティ** のスロット。
#: 同じ発話でこれらが解決していれば、所有格の無い「名前は…」は
#: そのエンティティのものと読む。
_NAME_OWNING_ENTITY_SLUGS: frozenset[str] = frozenset({"pet", "family"})


def _reassign_name_to_owning_entity(
    matches: list[tuple[str, tuple[str, ...]]], haystack: str,
) -> list[tuple[str, tuple[str, ...]]]:
    """別エンティティの名前を ``name`` から **その所有者のスロットへ移す** (純粋関数)。

    ``AttributeSpec.requires_self_possessor`` は所有格 (「猫**の**名前は」) と
    **同じ文の** 話題マーカーを見る。話題のスコープは句点で切れる設計なので
    (「趣味は登山です。名前は小川です。」を本人の名前として通すため)、
    **句点で区切られた形は素通りする**::

        「猫を飼っています。名前はミケです。」
            → pet と name の両方が解決し、``mem.personal.name`` に
              「名前はミケ」が live で入る

    実インシデント (2026-08-27 ライブ監査の修正検証、クリーン状態):
    上の 1 発話だけで、新規セッションの「私の名前を覚えていますか。」が
    **「あなたの名前はミケですね。」** を返した。さらにこのファクトは
    「猫の名前はミケではなくトラでした。」の訂正が ``pet`` スロットへ行くため
    **永久に supersede されず**、猫の名前を聞いても訂正前の「ミケ」が勝った。

    語彙 (猫 / 犬 / 妻 …) を数えるのではなく、**同じ発話で解決した他の
    スロット** を見る。``pet`` / ``family`` は「名前を持ちうる別の存在」を
    指すスロットそのものなので、これが立っていて ``name`` の側に一人称の
    裏付けが無ければ、その名前は本人のものではない。

    ``requires_self_possessor`` が既に自己所有を確認できた場合
    (「猫の名前はソラで、私の名前は小川です」) は移さない —
    その出現には一人称の所有格が付いている。

    **落とすだけでは名前が消える。** 根拠文の絞り込み
    (:func:`~backend.free.memory.extractors.chat._attribute_evidence_text`) は
    **その属性のトリガ語を含む文** だけを残すので、``name`` を捨てると
    「名前は…」の文はどのスロットにも属さなくなり、object から消える::

        「柴犬を1匹飼っています。名前はソラです。」
          → mem.personal.pet の object = 「柴犬を1匹飼っています。」

    実インシデント (2026-08-30 ライブ監査 T02#9): 上の 1 発話のあと、
    新セッションの「飼っている犬の名前は。」に **「確認できていません。」**
    を返した。SemMem には ``pet`` ファクトが live で在ったのに、
    **名前がその object に入っていなかった**。

    そこで捨てずに、``name`` のトリガ語を **所有者スロットのトリガ語へ併合**
    する。所有者側の根拠文が「名前は…」の文まで伸びるので、値が残る。
    ``name`` スロット自体は生成されないままなので、本人の名前が別エンティティ
    の名前で汚れる元の防御はそのまま効く。

    所有者が複数解決した場合 (``pet`` と ``family`` が同時) は、**「名前」の
    出現位置より前にある最後のトリガ語** を持つスロットへ寄せる。日本語では
    直前に述べたエンティティの名前を続けるのが自然なため。
    """
    slugs = {slug for slug, _ in matches}
    if "name" not in slugs or not (slugs & _NAME_OWNING_ENTITY_SLUGS):
        return matches
    if _SELF_POSSESSED_NAME_RE.search(haystack):
        return matches
    name_words = next(
        (words for slug, words in matches if slug == "name"), (),
    )
    owner = _nearest_owning_slug(matches, haystack, name_words)
    return [
        (slug, tuple(dict.fromkeys(words + name_words)) if slug == owner else words)
        for slug, words in matches
        if slug != "name"
    ]


def _nearest_owning_slug(
    matches: list[tuple[str, tuple[str, ...]]],
    haystack: str,
    name_words: tuple[str, ...],
) -> str | None:
    """「名前」の直前に現れた所有者スロットを返す (純粋関数)。

    位置が取れない場合は記載順で最初の所有者スロットへ寄せる。
    """
    name_at = min(
        (haystack.find(w) for w in name_words if haystack.find(w) >= 0),
        default=-1,
    )
    best: str | None = None
    best_at = -1
    for slug, words in matches:
        if slug not in _NAME_OWNING_ENTITY_SLUGS:
            continue
        at = max((haystack.find(w) for w in words), default=-1)
        if at < 0:
            continue
        if name_at >= 0 and at > name_at:
            continue
        if at > best_at:
            best, best_at = slug, at
    if best is not None:
        return best
    return next(
        (slug for slug, _ in matches if slug in _NAME_OWNING_ENTITY_SLUGS), None,
    )


#: 一人称の所有格が直接付いた「名前」。これがあれば本人の名前と読む。
_SELF_POSSESSED_NAME_RE = re.compile(
    r"(?:私|僕|俺|自分|わたし|ぼく|おれ|あたし|わたくし|うち)\s*の\s*名前",
)


def resolve_fact_attribute(
    text: str,
    fact_type: str,
    *,
    mode: str = "chat",
    triggers_dir: str | Path | None = None,
) -> str | None:
    """``text`` から ``fact_type`` の属性スラグを決定する。

    YAML 記載順に走査し、最初に部分一致した attribute を返す。どれにも
    一致しなければ ``None`` (呼出側は ``"user"`` 等へフォールバックする)。

    subject の粒度を属性単位にすることで、競合検出 (``(subject, predicate)``
    キー) が無関係な事実を同一事実の競合版と誤判定するのを防ぐ。
    """
    return resolve_fact_attribute_match(
        text, fact_type, mode=mode, triggers_dir=triggers_dir,
    )[0]


def resolve_fact_triggers_path(triggers_dir: str | Path | None = None) -> Path:
    """``fact_triggers.yaml`` の解決パスを返す (user override → default)。"""
    from backend.free.memory._defaults import resolve_trigger_file

    return resolve_trigger_file("fact_triggers.yaml", triggers_dir=triggers_dir)


def _default_fact_triggers() -> FactTriggerMap:
    """package 同梱 default を取得 (初期化時 / テスト既定値として使用)。"""
    return get_fact_triggers(resolve_fact_triggers_path(None))


# ──────────────────────────────────────────────────────────────────────────
# 共通基底
# ──────────────────────────────────────────────────────────────────────────


class NoteBuilder:
    """A-MEM ノート構築の共通基底 — LLM ゼロ。"""

    #: このビルダのデフォルトモード。サブクラスで上書きする。
    mode: MemoryMode = "chat"

    # 日本語キーワード抽出: 形態素解析不要の軽量版
    KEYWORD_PATTERNS = [
        re.compile(r"[A-Za-z][A-Za-z0-9_.-]+"),   # 英数字トークン
        re.compile(r"[\u4e00-\u9fff]{2,8}"),        # 漢字2-8文字
        re.compile(r"[\u30a0-\u30ff]{2,}"),          # カタカナ2文字以上
    ]

    # 自動タグ: キーワードからルールベースで付与（汎用）
    TAG_RULES: dict[str, list[str]] = {
        "create": ["python", "code", "バグ", "実装", "関数", "class", "def", "import"],
        "model": ["gguf", "llama", "qwen", "lora", "モデル", "推論"],
        "preference": ["好き", "嫌い", "いつも", "よく使う", "お気に入り"],
        "fact": ["です", "である", "とは", "定義"],
        "task": ["やって", "して", "作って", "教えて", "確認"],
    }

    #: assistant 発話には付けないタグ。
    #:
    #: ``fact`` のトリガは文末表現 (``です`` 等) なので、**assistant の回答は
    #: ほぼ全てが該当する**。この付与自体が問題なのは、``fact`` ノートが
    #: ``MemoryInjector`` 経由でプロンプトへ「(過去の記録)」として再注入され、
    #: システムプロンプトが「[関連する記憶] は自分の記憶より優先して根拠にする」
    #: と規定しているため — **未検証の生成物が、以後のターンでモデル自身の知識を
    #: 上書きする**。誤答が一度出ると記録として恒久化し自己増幅する。
    #:
    #: 実インシデント (2026-08-15 ライブ監査 ターン3): 「日本の三名園は、
    #: 大徳寺(京都)、西芳寺(京都)、兼六園(石川)です。」(正しくは兼六園・後楽園・
    #: 偕楽園) が ``tags:["fact"]`` で STM に残り、別セッションのプロンプトへ
    #: 「(過去の記録)」として注入されているのを確認した。
    #:
    #: user 発話の ``fact`` は「ユーザーがそう述べた」という記録であり残す。
    ASSISTANT_EXCLUDED_TAGS: frozenset[str] = frozenset({"fact"})

    # ─── markdown フェンス検知 ──
    _CODE_FENCE_RE = re.compile(r"```")

    # ノート本文 (``content``) は ``build()`` では切らない。STM / LTM の読み手は
    # 全文を前提にしているため、上限は **使う側** が持つ:
    #   - 埋め込み: ``sleep_update._EMBED_MAX_CHARS`` (1500 字、embed の n_ctx 対策)
    #   - LLM 進化: ``note_evolver.MAX_NOTE_CONTENT_LEN`` (800 字、プロンプト対策)
    # 長いアシスタント応答が丸ごと保存されるのは仕様で、ここで truncate すると
    # 両者の前提 (原文が残っている) が崩れる。

    # ─── 公開 API ───────────────────────────────────────────────────────

    def build(
        self,
        content: str,
        session_id: str,
        *,
        role: str = "user",
        source: NoteSource | None = None,
        mode: MemoryMode | None = None,
        project_id: str | None = None,
        is_tool_output: bool = False,
    ) -> dict[str, Any]:
        """ノート構築 dict を返す — 0.1ms 以内・LLM 呼び出し無し。

        Args:
            content: ノート内容テキスト
            session_id: セッション ID
            role: 発言者ロール (``user`` / ``assistant``)
            source: 発生源 (``NoteSource``)。``None`` の場合は ``role`` から推測
            mode: モード。``None`` の場合はビルダの ``self.mode`` を使う
            project_id: クリエイトモード時のプロジェクト ID
            is_tool_output: ツール出力か。``True`` の場合 STM 以降は除外される

        Returns:
            ``MemoryNote`` 構築用の dict
        """
        # reasoning モデルの思考漏れ (<think>...</think>) をノート化しない。
        # 残すと STM に焼き込まれ後続チャットへ再注入され話題汚染を招く。
        content = _strip_think_tags(content)

        effective_source: NoteSource = source if source is not None else (
            "assistant" if role == "assistant" else "user"
        )
        effective_mode: MemoryMode = mode if mode is not None else self.mode

        is_code_block = self._detect_code_block(content)
        extraction_skipped = False
        skip_reason: str | None = None

        if is_code_block:
            extraction_skipped = True
            skip_reason = "code_block"
        elif is_tool_output:
            extraction_skipped = True
            skip_reason = "tool_output"

        # コードブロックなら keywords/tags も付与しない (extraction を完全に止める)
        if is_code_block:
            keywords: list[str] = []
            generic_tags: list[str] = []
            candidate_tags: list[str] = []
        else:
            keywords = self.extract_keywords(content)
            generic_tags = self.auto_tag(content, effective_source)
            # ツール出力は候補抽出も行わない (sleep-time Extractor に流さない)
            candidate_tags = (
                [] if is_tool_output else self.candidate_fact_tags(content)
            )
            candidate_tags = _drop_user_subject_tags_from_assistant(
                candidate_tags, effective_source,
            )

        merged_tags = sorted(set(generic_tags + candidate_tags))

        now = time.time()
        return {
            "id": uuid4().hex[:12],
            "content": content,
            "keywords": keywords,
            "tags": merged_tags,
            "embedding": None,
            "lightmem_score": self.initial_score(content, role),
            "confidence": self.source_confidence(effective_source),
            "created_at": now,
            "accessed_at": now,
            "access_count": 0,
            "session_id": session_id,
            "context_description": "",
            "evolution_pending": True,
            # ── 拡張フィールド ──
            "source": effective_source,
            "mode": effective_mode,
            "project_id": project_id,
            "is_tool_output": is_tool_output,
            "is_code_block": is_code_block,
            "extraction_skipped": extraction_skipped,
            "extraction_skip_reason": skip_reason,
            "candidate_fact_tags": candidate_tags,
        }

    # ─── ヘルパ ─────────────────────────────────────────────────────────

    @classmethod
    def extract_keywords(cls, content: str) -> list[str]:
        """正規表現ベースのキーワード抽出（LLM 不要）"""
        keywords: list[str] = []
        for pattern in cls.KEYWORD_PATTERNS:
            keywords.extend(pattern.findall(content))
        seen: set[str] = set()
        result: list[str] = []
        for kw in keywords:
            lower = kw.lower()
            if lower not in seen and len(kw) >= 2:
                seen.add(lower)
                result.append(kw)
        return result[:10]

    @classmethod
    def auto_tag(cls, content: str, source: str = "user") -> list[str]:
        """ルールベースの自動タグ付け (汎用)

        Args:
            content: ノート内容。
            source: 発生源 (``NoteSource``)。``"assistant"`` のときは
                :data:`ASSISTANT_EXCLUDED_TAGS` を付けない (理由は同定数の
                コメント参照)。既定 ``"user"`` は従来挙動。
        """
        content_lower = content.lower()
        excluded = cls.ASSISTANT_EXCLUDED_TAGS if source == "assistant" else frozenset()
        tags: list[str] = []
        for tag, trigger_words in cls.TAG_RULES.items():
            if tag in excluded:
                continue
            if any(w in content_lower for w in trigger_words):
                tags.append(tag)
        return tags

    @staticmethod
    def initial_score(content: str, role: str = "user") -> float:
        """初期 LightMem スコア（ルールベース）"""
        score = 0.5
        if len(content) > 200:
            score += 0.1
        if role == "user":
            score += 0.1
        return min(1.0, score)

    # 発生源別の初期 confidence。NoteEvolver はこの値が
    # ``memory.note_evolver.confidence_threshold`` (既定 0.7) 未満のノートのみ
    # LLM 進化 (context_description 生成) の対象にする。ユーザー発話は権威性が
    # 高く LLM 進化不要なので閾値以上、assistant / rag / system 由来のノートは
    # context 補強の価値があるので閾値未満に置く。
    _SOURCE_CONFIDENCE: dict[str, float] = {
        "user": 1.0,
        "assistant": 0.5,
        "rag": 0.6,
        "system": 0.6,
    }

    @classmethod
    def source_confidence(cls, source: str) -> float:
        """発生源 (``NoteSource``) 別の初期 confidence を返す。"""
        return cls._SOURCE_CONFIDENCE.get(source, 1.0)

    @classmethod
    def _detect_code_block(cls, content: str) -> bool:
        """markdown フェンス (``` ... ```) を含むかを判定。

        クリエイトモードで生成された markdown 応答にフェンスが含まれると、
        コード片はファクト抽出対象から外したいので、ここで早期検出する。
        部分一致でも True を返す (フェンス開閉が片方のみでも保守的に skip)。
        """
        if not content:
            return False
        return bool(cls._CODE_FENCE_RE.search(content))

    # サブクラスでオーバーライドする
    def candidate_fact_tags(self, content: str) -> list[str]:  # noqa: ARG002
        """モード別の候補ファクトタイプを返す。

        基底クラスは何も返さない。``ChatNoteBuilder`` /
        ``CreateNoteBuilder`` でオーバーライドする。
        """
        return []


# ──────────────────────────────────────────────────────────────────────────
# ChatNoteBuilder
# ──────────────────────────────────────────────────────────────────────────


class _ModeAwareNoteBuilder(NoteBuilder):
    """mode 固有のトリガ辞書を YAML からロードする共通基底。

    shipped default は :mod:`backend.free.memory._defaults` 配下の
    ``fact_triggers.yaml`` に、user override は ``<triggers_dir>/
    fact_triggers.yaml`` (通常 ``local/triggers/``) に置く。
    ``triggers_dir=None`` の場合は default のみ参照する。
    """

    mode: MemoryMode

    def __init__(self, triggers_dir: str | Path | None = None) -> None:
        # 明示的に ``triggers_dir`` を渡された場合はそれを優先、
        # 省略時はモジュールレベル default (:data:`_DEFAULT_TRIGGERS_DIR`) を
        # 動的に参照する。後者は ``set_default_triggers_dir`` による後追い
        # 変更を反映するため property 内で遅延解決する。
        self._explicit_triggers_dir = triggers_dir
        self._fact_triggers_cached: dict[str, tuple[str, ...]] | None = None

    @property
    def _effective_triggers_dir(self) -> str | Path | None:
        if self._explicit_triggers_dir is not None:
            return self._explicit_triggers_dir
        return _DEFAULT_TRIGGERS_DIR

    @property
    def triggers_dir(self) -> str | Path | None:
        """解決に使う triggers_dir (明示指定 → モジュール default)。

        Extractor 側が ``resolve_fact_attribute`` へ同じ override を渡すための
        公開アクセサ。
        """
        return self._effective_triggers_dir

    @property
    def fact_triggers(self) -> dict[str, tuple[str, ...]]:
        """mode 対応セクションの ``{fact_type: triggers}`` マップ。

        初回呼出しで YAML からロードしてキャッシュする。プロセス内で
        同じ解決パスを複数インスタンスが共有する (get_fact_triggers 側で
        再利用される)。
        """
        if self._fact_triggers_cached is None:
            path = resolve_fact_triggers_path(self._effective_triggers_dir)
            table = get_fact_triggers(path)
            self._fact_triggers_cached = table.get(self.mode, {})
        return self._fact_triggers_cached

    def candidate_fact_tags(self, content: str) -> list[str]:
        """候補ファクトタイプを返す。

        **訂正マーカーだけが根拠のときは属性の裏取りを要求する。**
        ``ではなく`` / ``正しくは`` 等は「これは訂正だ」としか言っておらず、
        **何についての訂正か** を一切示さない。訂正は算術・ファイルパス・
        アシスタントの主張など何にでも掛かるので、これを単独の根拠にすると
        無関係な発話が personal_fact / preference の候補になる。

        実インシデント (2026-08-23 ライブ監査): 「さっきの 1234 × 5678 の答えは
        間違っています。正しくは 7006653 です。」が ``正しくは`` だけを根拠に
        ``personal_fact`` と ``preference`` の候補になり、自属性が解決できない
        ため ``is_correction`` の継承が直前の名前付きスロットを埋めて、
        ``mem.personal.birthday states: …7006653…`` と
        ``mem.preference.food prefers: …7006653…`` の 2 件が SemMem に残った。

        ``fact_triggers.yaml`` のコメントは既に「属性を伴う訂正形だけを採り、
        話題語を落とした訂正は curator に委ねる」と宣言している。ここはその
        宣言をコードで満たす (従来は宣言だけで、実際は全訂正形を採っていた)。
        """
        text = content.lower()
        results: list[str] = []
        for tag, triggers in self.fact_triggers.items():
            hits = [t for t in triggers if t in text]
            if not hits:
                continue
            if all(h in CORRECTION_FORM_TRIGGERS for h in hits):
                own_attribute = resolve_fact_attribute(
                    content, tag,
                    mode=self.mode,
                    triggers_dir=self._effective_triggers_dir,
                )
                if own_attribute is None:
                    continue
            results.append(tag)
        results.extend(
            tag for tag in self._attribute_backed_tags(content)
            if tag not in results
        )
        return results

    #: 属性の解決だけで候補にしてよい fact_type。``fact_attributes.yaml`` の
    #: スロットは「ユーザーが持ちうる属性」の定義そのものなので、平叙文が
    #: スロットに解決できた時点で自己開示とみなせる。
    ATTRIBUTE_BACKED_TAGS: tuple[str, ...] = ()

    def _attribute_backed_tags(self, content: str) -> list[str]:
        """属性が解決できる平叙文を候補タグにする (動詞リストに依存しない経路)。

        ``fact_triggers.yaml`` の ``personal_fact`` は一人称代名詞か、断定の
        述語形 (``住んでいます`` / ``勤めています`` …) の **列挙** を要求する。
        日本語は一人称を落とすのが常態なので、この列挙は繰り返し漏れてきた
        (2026-07-26 / 2026-08-22 の各ライブ監査で 1 度ずつ語を追加している)。

        2026-08-25 ライブ監査では 4 発話が候補タグ **0 件** で、sleep-time
        Step 8 の抽出対象にすらならなかった:

        - 「猫を1匹飼っていて、名前はソラです。」
        - 「来月の9月14日に大阪で技術カンファレンスの登壇があります。」
        - 「登壇日は9月14日ではなく9月20日に変更になりました。」
        - 「職業はソフトウェアエンジニアで、主にPythonとTypeScriptを書いています。」

        語を足し続ける代わりに、判定の向きを変える: ``fact_attributes.yaml`` の
        スロットは **ユーザーが持ちうる属性の定義そのもの** なので、
        「平叙文である」+「既知の属性スロットに解決できる」が揃えば自己開示と
        みなしてよい。依頼文・質問文は :func:`is_plain_statement` と
        :func:`states_no_user_value` の 2 つで落ちる (「このファイルの名前を
        変更してください。」「予定を教えてください。」「あなたの職業は何ですか？」は
        いずれも両方に掛かる)。

        誤って拾っても着地は候補タグまでで、Step 8 側の質問文・依頼文フィルタと
        assistant 発話の除外が後段に残る。
        """
        if not self.ATTRIBUTE_BACKED_TAGS or not content:
            return []
        if not is_plain_statement(content) or states_no_user_value(content):
            return []
        return [
            tag for tag in self.ATTRIBUTE_BACKED_TAGS
            if resolve_fact_attribute(
                content, tag,
                mode=self.mode,
                triggers_dir=self._effective_triggers_dir,
            ) is not None
        ]


class ChatNoteBuilder(_ModeAwareNoteBuilder):
    """チャットモード用 NoteBuilder。

    会話重視のレンズ — ユーザーの自己開示・好み・感情・意見を捕捉する。
    実際のファクト抽出は sleep-time Step 8 で行われ、本クラスは
    候補タグを ``MemoryNote.tags`` に積んでヒントを残すだけ。

    候補ファクトタイプ: ``personal_fact`` / ``world_fact`` / ``preference`` /
    ``emotion`` / ``opinion``。実際のトリガ語は
    ``backend/free/memory/_defaults/triggers/fact_triggers.yaml`` を参照。
    """

    mode: MemoryMode = "chat"

    #: personal_fact / preference のスロットは「ユーザーが持ちうる属性」の
    #: 定義そのものなので、平叙文が解決できれば自己開示とみなす
    #: (:meth:`_ModeAwareNoteBuilder._attribute_backed_tags`)。
    #: emotion / opinion は入れない — 属性語 (``workload`` / ``tech_stack``) が
    #: 一般名詞で、平叙文であれば何にでも当たってしまう。
    ATTRIBUTE_BACKED_TAGS: tuple[str, ...] = ("personal_fact", "preference")


# ──────────────────────────────────────────────────────────────────────────
# CreateNoteBuilder
# ──────────────────────────────────────────────────────────────────────────


class CreateNoteBuilder(_ModeAwareNoteBuilder):
    """クリエイトモード用 NoteBuilder。

    目的達成重視のレンズ — プロジェクトルール・判断・約束・タスク・コード関連
    知識を捕捉する。コードブロック (``` フェンス) は完全スキップ、ツール出力は
    extraction_skipped を立てる (基底クラスの ``build()`` 内で処理)。

    候補ファクトタイプ: ``project`` / ``decision`` / ``commitment`` /
    ``task`` / ``create``。実際のトリガ語は
    ``backend/free/memory/_defaults/triggers/fact_triggers.yaml`` を参照。
    """

    mode: MemoryMode = "create"


# ──────────────────────────────────────────────────────────────────────────
# Builder ファクトリ
# ──────────────────────────────────────────────────────────────────────────


_CHAT_BUILDER = ChatNoteBuilder()
_CREATE_BUILDER = CreateNoteBuilder()


def get_note_builder(mode: MemoryMode) -> NoteBuilder:
    """モード別 NoteBuilder のシングルトンを返す。

    builder インスタンスはステートレスなので使い回して問題ない。
    ``fact_triggers.yaml`` の user override を使いたい場合は、
    ``ChatNoteBuilder(triggers_dir=...)`` / ``CreateNoteBuilder(triggers_dir=...)``
    を直接構築すること (通常は extractor / STM 側から注入される)。
    """
    if is_create_mode(mode):
        return _CREATE_BUILDER
    return _CHAT_BUILDER
