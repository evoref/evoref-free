"""アシスタント自身の好み・感情を尋ねる問いか — 判定点 ``persona_question`` (c_17 §3.20)。

読み手は人格注記 (``core.inference._persona_question_note``、c_02 §6.3)。帯ごとに
注記を変える:

- ``fire`` (**1 つの文** に二人称の主語と嗜好・感情・意見の語があり、方法・是非の
  助言の形ではない): 「あなた自身の好みを尋ねている」と言い切る注記。
- ``abstain``: **言い切らない** 注記。自分のことを尋ねているならそう答え、ユーザーの
  ための選択肢・提案・助言を求めているなら自分の体験として語らずに答える、と両方の
  答え方を書く。次の 3 つの形:
  - 主語の無い疑問文に嗜好・感情の語 (``subjectless_preference_question``)
  - 二人称の主語と話題が別の文にある (``second_person_cross_sentence`` —
    「週末に猫と楽しめることはありますか？あなたのおすすめを教えて」)
  - 二人称の主語と話題が同じ文にあるが、方法・是非の助言の形
    (``second_person_advice_frame`` — 「猫と楽しく過ごすコツについて、あなたの意見を
    聞かせて」)
- ``skip``: 注記なし。

発火の帯も字句の判定なので誤りうる (宛先が二人称でも、尋ねているのがユーザーの
ための選択肢である文はありうる)。言い切るのは、宛先と話題が 1 つの文で結び付いて
いて、助言の形でもない場合に絞る。

主語の無い文は宛先を字句で決められない。「猫派？犬派？」「好きな季節は？」は自分への
問いだが、「週末に猫と一緒に楽しめることはありますか？」(猫を飼っていると述べた
ユーザーの問い) は提案を求めている。後者に言い切りの注記が付き「猫と一緒にのんびりと
読書をするのが好きです。」と自分の体験を語った (2026-10-05 ライブ監査 trace
99abfeaf22d6)。同型の取りこぼしを語形 (「どうすれば」「〜べきか」) で塞いできたが、
「〜ことはありますか」のように語形は尽きない (#12 / #14 — 語形を足す方向では直さない)。
言い切りの注記の誤発火は体験の捏造を起こすが、両方の答え方を書いた注記はどちらの
読みでも害が残らないので、宛先が字句で読めない帯は「判定してから言い切る」のでは
なく「判断の材料を渡す」形にする (#15)。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from backend.free.core.intent_vocab import is_practice_advice_query
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)

#: アシスタント自身の好み・感情・体験を尋ねる質問のシグナル。
#: 主語が相手 (あなた / 君 / you) であることと、感情・嗜好語の共起を要求する。
_PERSONA_SUBJECT_RE = re.compile(
    r"あなた|君は|きみは|(?<![A-Za-z])(?:you|your)(?![A-Za-z])",
    re.IGNORECASE,
)
_PERSONA_PREFERENCE_PATTERN = (
    r"好き|嫌い|好み|嬉し|うれし|悲し|楽し|寂し|感情|気持ち|心|感じ(?:ます|る|て)"
    r"|性格|人格|内面"
    # 雑談での嗜好の訊き方。「猫派？犬派？」「コーヒー派？紅茶派？」
    # 「最近ハマってるものある？」は日常的だが、どれも従来の語彙に無かった。
    r"|[^\s、。]派\b|[^\s、。]派[？?]|ハマ(?:って|る|った)"
    r"|(?<![A-Za-z])(?:feel|feelings|emotion|emotions|favou?rite|enjoy|prefer"
    r"|like\s+best|personality)(?![A-Za-z])"
)
#: 意見を求める語。何について の意見かは文が決めるので、二人称の主語がある
#: ときだけ人格質問とみなす (主語の無い「〜んだけど、どう思う?」はユーザーが
#: 述べた事柄への意見を求めている。2026-10-03 run10)。
_PERSONA_OPINION_PATTERN = r"どう思(?:い|う)|意見|(?<![A-Za-z])opinion(?![A-Za-z])"
_PERSONA_TOPIC_RE = re.compile(
    _PERSONA_PREFERENCE_PATTERN + "|" + _PERSONA_OPINION_PATTERN, re.IGNORECASE,
)
_PERSONA_PREFERENCE_RE = re.compile(_PERSONA_PREFERENCE_PATTERN, re.IGNORECASE)
#: 一人称の主語。これがある文は「ユーザー自身について」述べている。
_FIRST_PERSON_RE = re.compile(
    r"私|僕|俺|自分|うち|(?<![A-Za-z])(?:i|me|my|mine)(?![A-Za-z])",
    re.IGNORECASE,
)
#: 文の切れ目 (人格質問の判定を文単位で行うため)。
_PERSONA_SENTENCE_RE = re.compile(r"[^。．.!！?？\n]+[。．.!！?？]?")
#: 主語省略の文を「アシスタントへの問い」と見なすための疑問形。これが無いと
#: 「今日は嬉しいことがありました」のような **ユーザー自身の報告** まで拾う。
_PERSONA_QUESTION_TAIL_RE = re.compile(
    r"[?？]\s*$"
    r"|(?:です|ます|ました|でしょう|ません)か[。．.]?\s*$",
)
#: 主語省略の文で「ユーザー自身がどうすればよいか」を尋ねる形 (方法・手順の問い)。
#: 嗜好語 (楽し / 気持ち / 心 / 寂し) を含んでも、相手の内面ではなくユーザーへの
#: 助言を求めている (「心が疲れたときはどうすればいいですか？」)。是非の問い
#: (「〜するべきですか」) は :func:`is_practice_advice_query` が受ける。
_PERSONA_ADVICE_FRAME_RE = re.compile(
    r"どう(?:したら|すれば|すると|やったら|過ごせば|過ごしたら|過ごすと)|方法|コツ",
)


#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "persona_question"

#: アシスタント自身への問い (宛先が二人称で書かれている)。
PERSONA_LABEL = "persona"


def _is_advice_framed(sentence: str) -> bool:
    """方法・手順 (どうすれば / コツ) か是非 (〜するべきか) の助言の形か。"""
    return bool(_PERSONA_ADVICE_FRAME_RE.search(sentence)) or is_practice_advice_query(sentence)


def persona_question_reason(text: str) -> tuple[str | None, str]:
    """``(label, evidence)`` を返す (純粋関数)。``label`` が ``None`` なら棄権。

    判定は文単位。嗜好の話題を含む文が

    - 二人称の主語を持つ (「あなたは何が好き？」) → 発火。ただしその文が方法・
      是非の助言の形 (「〜のコツについて、あなたの意見を」) なら棄権、
    - 主語を持たず、かつ疑問形である (「猫派？犬派？」「ハマってるものある？」)
      → 棄権 (宛先が字句で読めない)

    一人称の主語を持つ文 (「私は猫派なんだけど」) はユーザー自身についての記述
    なので数えない。主語省略の側で疑問形を要求しないと「今日は嬉しいことが
    ありました」のようなユーザー自身の報告まで拾う。主語省略の側は意見を求める
    語 (「どう思う」) を話題に数えない — 「昨日友達と話したんだけど、どう思う?」に
    人格ノートが付き「私には何の感想もありません…」と返した (2026-10-03 run10)。
    主語省略の側は助言の問い (「どうすれば」「〜するべきか」) も数えない (2026-10-05)。

    主語と話題が別の文に分かれる形 (「あなたはただのプログラムでしょう。気持ち
    なんて無いはずです」) は棄権にする。以前は発火にしており、「週末に猫と楽しめる
    ことはありますか？あなたのおすすめを教えて」(楽し と あなた が別の文) にも
    言い切りの注記が付いた (2026-10-05 独立レビュー)。
    """
    if not text:
        return NEGATIVE_LABEL, "empty"
    sentences = [raw.strip() for raw in _PERSONA_SENTENCE_RE.findall(text) if raw.strip()]
    advice_framed = False
    for sentence in sentences:
        if not (_PERSONA_SUBJECT_RE.search(sentence) and _PERSONA_TOPIC_RE.search(sentence)):
            continue
        if _is_advice_framed(sentence):
            advice_framed = True
            continue
        return PERSONA_LABEL, "second_person_topic"
    if advice_framed:
        return None, "second_person_advice_frame"
    if _PERSONA_SUBJECT_RE.search(text) and _PERSONA_TOPIC_RE.search(text):
        return None, "second_person_cross_sentence"
    for sentence in sentences:
        if not _PERSONA_PREFERENCE_RE.search(sentence):
            continue
        if (
            not _FIRST_PERSON_RE.search(sentence)
            and _PERSONA_QUESTION_TAIL_RE.search(sentence)
            and not _is_advice_framed(sentence)
        ):
            return None, "subjectless_preference_question"
    return NEGATIVE_LABEL, "no_match"


def is_persona_question(text: str) -> bool:
    """自分自身への問いでありうるか (発火または棄権。純粋関数、記録しない)。"""
    label, _evidence = persona_question_reason(text)
    return label != NEGATIVE_LABEL


class _PersonaQuestionRule:
    """字句段 (``Predicate`` プロトコル)。根拠を形ごとに書き分けるため
    ``LexicalPredicate`` ではなく ``Verdict`` を直接返す。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:  # noqa: ARG002 - Predicate プロトコルの引数
        label, evidence = persona_question_reason(text or "")
        if label is None:
            return Verdict(
                value=None, score=0.5, band="abstain",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        if label == PERSONA_LABEL:
            return Verdict(
                value=label, score=1.0, band="fire",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=NEGATIVE_LABEL, score=0.0, band="skip",
            evidence=evidence, predicate=self.name, stage="lexical",
        )


_RULE = _PersonaQuestionRule()

#: プロセス共通の判定点。``build_messages`` がターンに 1 回引く。
#: 言い切りの注記の誤発火 (体験の捏造) のほうが重いので ``confirm``。
#: 事例段は未装着。``exemplar=`` を渡すだけでは効かない — 同期の ``evaluate`` は
#: 字句段しか評価せず、``confirm`` は字句の棄権をそのまま返す。事例段を使うには
#: 読み手を ``aevaluate`` へ移し、棄権の帯を埋める方針を選び直す必要がある。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="confirm",
        candidates=[PERSONA_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def persona_question_verdict(text: str) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する。ターンに 1 回)。"""
    return predicate.evaluate(text or "")


def persona_question_rule(text: str) -> Verdict:
    """字句段だけを **記録せずに** 評価する (純粋関数。同じターンの 2 度目以降の読み手)。"""
    return _RULE.evaluate(text or "")


__all__ = [
    "PERSONA_LABEL",
    "PREDICATE_NAME",
    "bind_debug_logger",
    "is_persona_question",
    "persona_question_reason",
    "persona_question_rule",
    "persona_question_verdict",
    "predicate",
]
