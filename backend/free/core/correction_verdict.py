"""訂正の **検証** に関わる純粋関数 (pillar 横断の正準置き場)。

「訂正」は *過去のターンについての主張* であって、発話の形 (「違います」
「ではなく」) だけでは成立しない。字句で拾った候補を消費するのは、

- EvorefLearn: ``learning.correction_verifier`` (few-shot / 採用ゲート / eval_core)
- EvorefMem: ``memory.sleep.correction_curator`` (``from_correction`` ファクト /
  スロットの supersede / 値アンカーの宛先)

の 2 pillar で、どちらも同じ判定を使わなければ「学習側は偽陽性を弾いたのに
記憶側は書いてしまう」という食い違いが残る (2026-09-08 夜の監査: 物理の訂正
「違います。最初の答えを…100 m のはずです」が SemMem の ``mem.personal.family``
を supersede し、検証器は wrong_claim="100" / correct_value="100 m" の **同値**
を訂正と判定した)。

本モジュールは LLM を呼ばない。プロンプトの組み立てと、LLM の出力に対する
**コード側の門** (逐語 span / 同値 / 既述 / 引用の除外) だけを持つ。

- :func:`mask_quoted_speech` — 鉤括弧の内側は本人の主張ではない (伝聞・引用)。
  「営業から『色が違う』というクレーム」の「違う」で訂正候補を立てない。
- :func:`claims_equivalent` — ``wrong_claim`` と ``correct_value`` が同じ値を
  指すなら、それは訂正ではなく **確認 / 言い直し**。
- :func:`response_already_states` — 相手の応答が既にその値を述べているなら、
  ユーザーは誤りを指摘していない。
- :func:`reversed_restatement` — 検証器の span の組が、発話の「X ではなく Y」と
  逆向き (ユーザーが退けた X を正しい値としている) なら判定ごと捨てる
  (却下理由は ``invalid_span``。永続の語彙は増やさない)。
- :func:`widen_to_restated_value` — ``correct_value`` が発話の「X ではなく Y」の
  Y の途中で切れていて ``wrong_claim`` と同値に見えるなら、Y へ広げる。
- :func:`check_verdict` — 上の門を LLM 出力へ順に当て、通ったものだけ
  :class:`VerdictCheck` として返す。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, get_args
from backend.free.core.correction_target import restatement_pairs, split_sentences
from backend.free.core.intent_vocab import is_plain_statement
from backend.free.core.script_ranges import (
    KANJI,
    KATAKANA_WORD,
)
from backend.log_config import get_logger

logger = get_logger("core.correction_verdict")

#: 引用として扱う開き / 閉じの対。内側は発話者本人の主張ではない。
QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ("「", "」"), ("『", "』"), ("“", "”"), ('"', '"'),
)

#: 検証器が返す帰属。``assistant`` / ``self`` だけが「誤りの指摘」。
CorrectionTarget = Literal[
    "assistant", "self", "third_party", "premise_change", "none",
]
CORRECTION_TARGETS: frozenset[str] = frozenset(
    ("assistant", "self", "third_party", "premise_change", "none"),
)
#: 誤りの指摘として扱う帰属 (消費側はこの部分集合をさらに絞ってよい)。
POINTING_TARGETS: frozenset[str] = frozenset(("assistant", "self"))

#: 桁区切り・小数付きの数値 (全角は :func:`_ascii` で半角へ寄せてから当てる)。
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

#: 検証プロンプトへ入れる本文の上限。
RESPONSE_CAP = 1200
QUERY_CAP = 600

#: 検証結果の却下理由。消費側はログ / 監査用にそのまま記録する。
RejectReason = Literal[
    "no_verdict", "invalid_target", "not_correction", "invalid_span",
    "same_value", "already_stated",
]

#: 経験に刻む検証結果 (``FeedbackSignals.correction_verdict``) の語彙。帰属・
#: コード側の却下理由と、検証器が付ける ``no_context`` (直前応答を解決できない) /
#: ``disputed`` (訂正への回答が値を退けた)。台帳では ``open`` の列挙 (c_05 §0.5.3)。
VerdictCode = Literal[CorrectionTarget, RejectReason, Literal["no_context", "disputed"]]
VERDICT_CODES: frozenset[str] = frozenset(get_args(VerdictCode))


def mask_quoted_speech(text: str) -> str:
    """引用 (鉤括弧 / 二重引用符) の内側を落としたコピーを返す (純粋関数)。

    括弧そのものは残す — 「『』というクレーム」のように文の骨格は保ち、
    内側の語だけを訂正語彙の照合対象から外す。閉じ括弧が無い開き括弧は
    そのまま (引用と断定できない)。
    """
    if not text:
        return text
    out = text
    for open_q, close_q in QUOTE_PAIRS:
        if open_q == close_q:
            pattern = re.escape(open_q) + r"[^" + re.escape(open_q) + r"]*" + re.escape(close_q)
        else:
            pattern = re.escape(open_q) + r"[^" + re.escape(close_q) + r"]*" + re.escape(close_q)
        out = re.sub(pattern, f"{open_q}{close_q}", out)
    return out


def _ascii(text: str) -> str:
    """全角英数字・記号を半角へ寄せ、空白を全て落とす。"""
    return "".join(unicodedata.normalize("NFKC", text or "").split())


#: 文末の断定・丁寧の語尾と句読点。値の span に付いて来やすい (「2015年です。」)。
#: 動詞語尾 (``ます`` / ``だ``) は落とさない — 「住んでいます」「パンダ」を
#: 切ってしまう。丁寧・断定の助動詞に限る。
_COPULA_TAIL_RE = re.compile(
    r"(?:(?:です|でした|でしょう|である|であった)?(?:ね|よ|か)?[。．.、,!！?？\s]*)+$",
)


def strip_copula(span: str) -> str:
    """値の span から文末の語尾・句読点を落とす (純粋関数)。

    LLM は「2015年です。」のように **文** を span として返すことがある。
    逐語の門 (:func:`check_verdict`) は前応答が「2015年です。」なので通るが、
    消費側はこの span を live 値 (「2015年から勤めています」) に当てるため、
    語尾が付いたままだと宛先に当たらず訂正が迷子になる (2026-09-10 ライブ
    監査 (h) H-11: 検証は通ったのに employer の値が畳まれず、訂正文が
    ``mem.world.assertion.year_correction`` になった)。語尾だけの span は
    空にしない (元のまま返す)。
    """
    text = (span or "").strip()
    stripped = _COPULA_TAIL_RE.sub("", text).strip()
    return stripped or text


def norm_span(text: str) -> str:
    """逐語 span 照合用の正規化 (空白無視 / 全角半角同一視)。"""
    return _ascii(text).lower()


#: 直後に続けば「その値を否定している」と読める標識 (訂正への回答側)。
_DISPUTE_MARKER_RE = re.compile(
    r"\s*[」』）)]?\s*(?:ではなく|ではありません|ではない|じゃなく|とは限らず"
    r"|は誤り|は間違|というのは誤|は正しくあり|is\s+not|isn't|incorrect)",
)
_CONTENT_RUN_RE = re.compile(f"[{KANJI}{KATAKANA_WORD}A-Za-z0-9]{{2,}}")
#: 値の内容語の直後 (この文字数以内) に否定標識があれば否定とみなす。
_DISPUTE_WINDOW = 14


def answer_disputes_value(answer: str, correct_value: str) -> bool:
    """訂正への回答が、訂正で示された値そのものを否定しているか (純粋関数)。

    検証器は (直前の応答, 訂正発話) だけを見て「アシスタントの誤りを指して
    いる」と判定するが、**その訂正が正しいか** は判定しない。ユーザーが
    誤った訂正 (「302 は恒久的な移転を示すコードです」) をし、アシスタントが
    その場で退けた (「302 は恒久的な移転を示すコードではなく、一時的な…」)
    場合、値は受け入れられていない。受け入れられなかった値を検証済み訂正
    として記憶 (world assertion / 属性の supersede) や学習 (訂正ペア) に
    流すと、**退けた誤りをシステムが採用する** (2026-09-11 ライブ監査 (j)
    J-04)。値の内容語の直後に否定標識が続けば否定と読む — 「〜ではなく、
    正しくは…」の形は回答の常態で、語彙ではなく構造で取れる。
    """
    body = answer or ""
    if not body or not (correct_value or "").strip():
        return False
    for run in _CONTENT_RUN_RE.findall(correct_value):
        start = 0
        while (pos := body.find(run, start)) >= 0:
            start = pos + len(run)
            tail = body[start:start + _DISPUTE_WINDOW]
            if _DISPUTE_MARKER_RE.match(tail):
                return True
    return False


def claim_numbers(text: str) -> tuple[str, ...]:
    """主張に含まれる数値を正規化して返す (桁区切り除去 / 末尾ゼロ整理)。"""
    out: list[str] = []
    for raw in _NUMBER_RE.findall(_ascii(text)):
        body = raw.replace(",", "")
        if "." in body:
            body = body.rstrip("0").rstrip(".")
        out.append(body or "0")
    return tuple(out)


def claims_equivalent(wrong_claim: str, correct_value: str) -> bool:
    """``wrong_claim`` と ``correct_value`` が同じ値を指すか (純粋関数)。

    数値を含むなら **数値の集合** で比べる (「100」と「100 m」は同じ、「10」と
    「100」は違う)。数値が無ければ正規化文字列の包含で比べる (「横浜」と
    「横浜市」は同じ扱い — 訂正なら別の値が出るはず)。どちらかが空なら False
    (空 span は「抜き出せなかった」であって同値ではない)。
    """
    w = norm_span(wrong_claim)
    c = norm_span(correct_value)
    if not w or not c:
        return False
    wn, cn = claim_numbers(wrong_claim), claim_numbers(correct_value)
    if wn and cn:
        return set(wn) == set(cn)
    if wn or cn:
        return False
    return w in c or c in w


def response_already_states(correct_value: str, prev_response: str) -> bool:
    """訂正が示す正しい値を、相手の応答が既に述べているか (純粋関数)。

    数値だけの値 (「100 m」) は応答内の **数値集合** に含まれるかで見る —
    「100 m」を「100」と書いた応答は同じ値を述べている。文字列の値は正規化
    包含。空なら False。
    """
    c = norm_span(correct_value)
    r = norm_span(prev_response)
    if not c or not r:
        return False
    cn = claim_numbers(correct_value)
    if cn:
        rn = set(claim_numbers(prev_response))
        residue = re.sub(r"[\d.,]+", "", c)
        # 数値以外の残りが単位程度 (3 文字以下) なら数値だけで判定する。
        if len(residue) <= 3:
            return all(n in rn for n in cn)
    return c in r


def build_correction_verify_prompt(
    prev_response: str,
    correction: str,
    *,
    prev_user: str = "",
) -> str:
    """検証用プロンプト (本文は原文のまま渡す)。

    ``prev_user`` は訂正より前のユーザー発話 (複数なら改行連結)。``target=self``
    の ``wrong_claim`` はこちら、または訂正発話自身 (「X ではなく Y」の X) の
    逐語 span になる。
    """
    prev_user_block = (
        f"PREVIOUS_USER_UTTERANCES:\n{prev_user[:QUERY_CAP * 3]}\n\n"
        if prev_user else ""
    )
    return (
        "直前のアシスタント応答と、その次のユーザー発話を読んでください。\n"
        "ユーザー発話が「過去の発言の誤りを指摘して正しい値を述べている」"
        "ものかどうかを判定します。\n\n"
        "target の意味:\n"
        "- assistant: アシスタントの応答が誤っていると指摘している\n"
        "- self: ユーザー自身の過去の発言を訂正している "
        "(例: 「すみません、住んでいるのは盛岡市ではなく花巻市でした」— "
        "アシスタントが言及していなくても self)。"
        "自分の前の値を「X ではなく Y」と言い直しているなら、"
        "計算などのやり直しの依頼を伴っても self "
        "(例: 「試験日は4月18日ではなく4月25日でした。計算し直してください」)\n"
        "- third_party: 第三者・世間・引用元の誤りに言及している\n"
        "- premise_change: 前の値は誤っていないが、前提や条件 (計画) を変えて"
        "やり直しを頼んでいる、または仮定の問いをしている (誤りの指摘ではない。"
        "例: 「金利が1.5%ではなく2%だったら月々いくら？」)\n"
        "- none: 訂正ではない (質問・比較・依頼・伝聞・確認など)\n\n"
        "重要: アシスタントの応答が **既にユーザーの示す値と同じ** なら、それは"
        "誤りの指摘ではありません (is_correction=false, target=none)。\n"
        "鉤括弧の内側 (引用・伝聞) は本人の主張ではありません。\n\n"
        "wrong_claim には誤っていた箇所を **そのまま逐語で** 抜き出してください "
        "(target=assistant なら直前のアシスタント応答から、target=self なら"
        "前のユーザー発話か、ユーザー発話中の「X ではなく」の X から。無ければ空文字)。\n"
        "correct_value にはユーザー発話が示した正しい値を"
        "**そのまま逐語で** 抜き出してください (無ければ空文字)。\n"
        "言い換え・要約・補完をしてはいけません。項目名を付け足さず値だけを抜き出します "
        "(例: 「住んでいるのは盛岡市ではなく花巻市でした」→ wrong_claim=盛岡市, "
        "correct_value=花巻市)。\n\n"
        f"{prev_user_block}"
        f"ASSISTANT_RESPONSE:\n{prev_response[:RESPONSE_CAP]}\n\n"
        f"USER_UTTERANCE:\n{correction[:QUERY_CAP]}\n"
    )


def _is_number_char(ch: str) -> bool:
    return ch.isdigit() or ch in ",."


def narrow_to_new_value(
    wrong_claim: str, correct_value: str, candidate: str,
) -> tuple[str, str] | None:
    """合成された span の組を、訂正発話に逐語で在る差分へ縮める (純粋関数)。

    訂正発話は「A は X ではなく Y」の形なので、項目名と新しい値がつながった
    ``correct_value`` (「年利4%」「経費200万円」) は発話に無い。LLM は
    ``wrong_claim`` の形に揃えて ``correct_value`` を合成して返すことが多く、
    逐語の門で全件落ちていた (2026-09-26 ライブ監査: 4 件とも invalid_span)。

    2 つの共通の前置・後置を剥がした差分 (数字の途中で切れたら数全体へ広げる)
    が発話に逐語で在るときだけ、差分を含み発話に逐語で現れる最長の部分へ
    両方を同じ位置で縮めて返す。新しい情報が逐語であるという門の保証は保つ。
    差分が発話に無ければ None (却下のまま)。
    """
    if not wrong_claim or not correct_value:
        return None
    limit = min(len(wrong_claim), len(correct_value))
    pre = 0
    while pre < limit and wrong_claim[pre] == correct_value[pre]:
        pre += 1
    suf = 0
    while (suf < limit - pre
           and wrong_claim[-1 - suf] == correct_value[-1 - suf]):
        suf += 1
    # 共通部分が数の途中で切れていたら数全体を差分に含める (150/200 → 15/20 にしない)
    while pre > 0 and _is_number_char(correct_value[pre - 1]) and (
        _is_number_char(correct_value[pre]) if pre < len(correct_value) - suf else True
    ):
        pre -= 1
    while suf > 0 and _is_number_char(correct_value[len(correct_value) - suf]) and (
        _is_number_char(correct_value[len(correct_value) - suf - 1])
        if len(correct_value) - suf - 1 >= pre else True
    ):
        suf -= 1
    c_end = len(correct_value) - suf
    diff = correct_value[pre:c_end]
    if not diff.strip():
        return None
    target = norm_span(candidate)
    if norm_span(diff) not in target:
        return None
    best: tuple[int, int] = (pre, c_end)
    for left in range(pre, -1, -1):
        for right in range(len(correct_value), c_end - 1, -1):
            if right - left <= best[1] - best[0]:
                break
            if norm_span(correct_value[left:right]) in target:
                best = (left, right)
                break
    left, right = best
    w_end = len(wrong_claim) - (len(correct_value) - right)
    return strip_copula(wrong_claim[left:w_end]), strip_copula(correct_value[left:right])


#: 前の値の言い直しではないと読める発話の標識 — 仮定 (仮に / もし / としたら /
#: なら)・時間の対比 (以前は / 先月 / 今は / 今月は)・伝聞 (によると / と言って /
#: と言われ / らしい)。「以前は1.5%ではなく1.2%でしたが、今は2%です」「部下に
#: よると、試験日は18日ではなく25日でした。」を self に倒さない (独立レビュー M-c)。
#: 門は LLM の判定を上げる方向なので、標識があれば倒さず LLM の答えに任せる。
_NOT_OWN_RESTATEMENT_RE = re.compile(
    r"仮に|もし|としたら|とすると|なら"
    r"|以前は|先月|今は|今月は"
    r"|によると|と言って|と言われ|らしい",
)


def marks_not_own_restatement(text: str) -> bool:
    """発話 (引用の内側は除く) に仮定・時間の対比・伝聞の標識があるか (純粋関数)。

    :func:`restated_own_value` の門と、記憶側の本人の値更新 (J-03、
    ``extractors.base.note_may_update_own_value``) が同じこの判定を読む (不変則 #14a)。
    """
    return bool(_NOT_OWN_RESTATEMENT_RE.search(mask_quoted_speech(text or "")))


def restated_own_value(candidate: str, prev_user: str) -> tuple[str, str] | None:
    """ユーザーが **自分の前の値** を言い直しているなら ``(旧値, 新値)`` (純粋関数)。

    検証器は「自分の申告を言い直したうえでやり直しを頼む」発話 (「試験日は4月18日
    ではなく4月25日でした。…計算し直してください」) を ``premise_change`` と答える
    ことがあり、訂正が記憶に届かなかった (2026-09-27 監査 F8)。LLM の判定は単体では
    閉じないので、コードで決められる範囲をここに持つ:

    - 引用の内側は落とす (伝聞は本人の主張ではない)
    - **平叙の文** だけを見る (問い・依頼の文の対比は仮定・条件)
    - 対比は断定 (です / でした) で閉じるものだけ (:func:`restatement_pairs`)。
      「だったら」の仮定・「に変更」の計画変更は対比にならない
    - 旧値が **前のユーザー発話** に逐語で在る (アシスタントの値の訂正ではない)
    - 述語の直後が文末 (「でしたが」「でしたっけ」「でした、と言われたら」は除く)
    - 同じ発話に仮定・時間の対比・伝聞の標識が無い (:data:`_NOT_OWN_RESTATEMENT_RE`)
    """
    source = norm_span(prev_user)
    if not source:
        return None
    if marks_not_own_restatement(candidate):
        return None
    masked = mask_quoted_speech(candidate or "")
    for sentence in split_sentences(masked):
        if not is_plain_statement(sentence):
            continue
        for old, new in restatement_pairs(sentence, sentence_final=True):
            if norm_span(old) in source:
                return strip_copula(old), strip_copula(new)
    return None


def reversed_restatement(wrong_claim: str, correct_value: str, candidate: str) -> bool:
    """span の組が訂正発話の「X ではなく Y」と **逆向き** か (純粋関数)。

    逐語 span / 同値 / 既述の門は向きを見ない。「金利は1.5%ではなく1.2%でした」に
    ``wrong_claim=1.2%`` / ``correct_value=1.5%`` が返ると、どちらも逐語で在るので
    全部通っていた (2026-09-28 実機再監査 R4)。発話 (引用の内側は落とす) の文末で
    断定に閉じた対比 (:func:`restatement_pairs`、:func:`restated_own_value` と同じ
    分解) について、``correct_value`` が X と同値で Y とは同値でなく、``wrong_claim``
    が Y と同値 (空なら不問) の組があれば真。

    偽に倒すもの (向きが構造で決まらない):

    - X と Y 自体が同値で両方の向きに当たる組
    - 同じ発話に **順向き** の対比も在る (「1.5%ではなく1.2%でした。いえ、やはり
      1.2%ではなく1.5%でした。」) — 言い直しの最終形を構造だけでは決めない
    - 文末で閉じない対比 (「でしたが」「です、というのは誤りです」)
    """
    if not correct_value:
        return False

    def _wrong_is(value: str) -> bool:
        return not wrong_claim or claims_equivalent(wrong_claim, value)

    backward = forward = False
    for sentence in split_sentences(mask_quoted_speech(candidate or "")):
        for old, new in restatement_pairs(sentence, sentence_final=True):
            if claims_equivalent(correct_value, new) and _wrong_is(old):
                forward = True
            elif claims_equivalent(correct_value, old) and _wrong_is(new):
                backward = True
    return backward and not forward


def widen_to_restated_value(
    wrong_claim: str, correct_value: str, candidate: str,
) -> str | None:
    """途中で切れた ``correct_value`` を発話の「X ではなく Y」の Y へ広げる (純粋関数)。

    検証器は「長女はひなたではなくひなのです。」に ``wrong_claim=ひなた`` /
    ``correct_value=ひな`` を返した (2026-10-01 実機、4/4 回)。ひらがなの値に
    ひらがなの述語 (「のです」) が続くと、LLM は値を述語の手前で切る。「ひな」は
    発話に逐語で在るので逐語の門は通り、包含で比べる同値の門 (:func:`claims_equivalent`
    — 「横浜」と「横浜市」を同じとみなす) が「ひな」⊂「ひなた」で ``same_value`` に
    落としていた。文字種の境界では切れ目が決まらない (値も述語もひらがな) ので、
    発話の構造 (:func:`restatement_pairs`、:func:`reversed_restatement` /
    :func:`restated_own_value` と同じ分解。語彙は足さない) で決める。

    広げるのは次をすべて満たすときだけ:

    - ``wrong_claim`` と ``correct_value`` が同値に見える (広げなければ ``same_value``
      で却下される組だけを救う。通っていた判定の span は変えない)
    - 発話 (引用の内側は除く) の文末で断定に閉じた対比 (X, Y) で、X が
      ``wrong_claim`` と同値、Y が ``correct_value`` を真に含み、X と Y は同値でない
    - そういう Y が 1 通りに決まる
    - 発話に仮定・時間の対比・伝聞の標識が無い (:func:`marks_not_own_restatement`)
    - Y が否定 (「ない」) で終わらない (値ではない)

    帰属は上げない span の修復 (:func:`narrow_to_new_value` と同類) で、
    不変則 #12 の例外 (:func:`restated_own_value`) ではない。

    該当しなければ ``None`` (呼出側は従来どおり ``same_value`` で却下する)。
    """
    c = norm_span(correct_value)
    if not c or not norm_span(wrong_claim) or not claims_equivalent(wrong_claim, correct_value):
        return None
    if marks_not_own_restatement(candidate):
        return None
    widened: set[str] = set()
    for sentence in split_sentences(mask_quoted_speech(candidate or "")):
        for old, new in restatement_pairs(sentence, sentence_final=True):
            value = strip_copula(new)
            n = norm_span(value)
            if (
                n != c and c in n and not n.endswith("ない")
                and claims_equivalent(wrong_claim, old)
                and not claims_equivalent(old, value)
            ):
                widened.add(value)
    return widened.pop() if len(widened) == 1 else None


@dataclass(frozen=True)
class VerdictCheck:
    """LLM 出力にコード側の門を当てた結果。

    ``ok`` が False のとき ``reason`` に却下理由が入る。``target`` は却下時も
    (返ってきていれば) 保持する — 消費側が「premise_change だった」と記録できる。
    """

    ok: bool
    reason: RejectReason | None
    target: str
    wrong_claim: str
    correct_value: str
    #: 門が LLM の答えを上書きしたときの理由 (``"restated"`` = premise_change を
    #: :func:`restated_own_value` で self に上げた、``"widened"`` = 切れた
    #: ``correct_value`` を :func:`widen_to_restated_value` で広げた)。LLM の答えのままなら ``None``。
    overridden: str | None = None
    #: 却下理由の内訳 (非永続。ログ / テスト用)。``"reversed"`` = span の組が発話の
    #: 「X ではなく Y」と逆向き (:func:`reversed_restatement`) で ``invalid_span``。
    detail: str | None = None


def check_verdict(
    payload: Any,
    *,
    candidate: str,
    prev_response: str,
    prev_user: str = "",
) -> VerdictCheck:
    """LLM の判定に **コード側の門** を順に当てる (純粋関数)。

    1. 形 (dict / target が既知の値)
    2. ``is_correction`` と帰属 — ``assistant`` / ``self`` 以外は訂正でない。
       ただし ``premise_change`` でも、自分の前の値の言い直し
       (:func:`restated_own_value`) なら ``self`` に倒し、その span で以降の門を当てる
    3. 逐語 span — ``wrong_claim`` は帰属先の本文に、``correct_value`` は
       訂正発話に、空白を無視して含まれること
    4. 向き — 発話の「X ではなく Y」と逆向きの組 (:func:`reversed_restatement`)
       は入れ替えずに ``invalid_span`` で却下する (帰属も誤っている見込みが高い。
       ``detail="reversed"`` とログで区別する)
    5. 同値 — ``wrong_claim`` と ``correct_value`` が同じ値なら訂正でない。ただし
       ``correct_value`` が発話の「X ではなく Y」の Y の途中で切れていたなら Y へ
       広げて通す (:func:`widen_to_restated_value`)
    6. 既述 — 帰属先の本文が ``correct_value`` を既に述べているなら訂正でない

    消費側は ``ok`` のものをさらに ``target`` で絞る (学習側は ``assistant`` のみ)。
    """
    if not isinstance(payload, dict):
        return VerdictCheck(False, "no_verdict", "", "", "")
    target = str(payload.get("target") or "")
    wrong_claim = strip_copula(str(payload.get("wrong_claim") or ""))
    correct_value = strip_copula(str(payload.get("correct_value") or ""))
    if target not in CORRECTION_TARGETS:
        return VerdictCheck(False, "invalid_target", target, wrong_claim, correct_value)
    restated = (
        restated_own_value(candidate, prev_user) if target == "premise_change" else None
    )
    overridden: str | None = None
    if restated is not None:
        # LLM の判定を字句の門で上げる唯一の経路なので、必ず記録する (#14(c))。
        target, overridden = "self", "restated"
        wrong_claim, correct_value = restated
        logger.info(
            "Correction verdict premise_change overridden to self (restated): "
            "wrong_claim=%r correct_value=%r", wrong_claim[:40], correct_value[:40],
        )
    elif not payload.get("is_correction") or target not in POINTING_TARGETS:
        return VerdictCheck(False, "not_correction", target, wrong_claim, correct_value)

    def _check(
        ok: bool, reason: RejectReason | None, detail: str | None = None,
    ) -> VerdictCheck:
        return VerdictCheck(
            ok, reason, target, wrong_claim, correct_value, overridden, detail,
        )

    # self の古い値は前のユーザー発話にあるか、訂正文自身に同居している
    # (「盛岡市ではなく花巻市でした」)。既述の判定 (下) は前の発話だけで見る。
    source = prev_response if target == "assistant" else (prev_user or prev_response)
    span_source = source if target == "assistant" else f"{source}\n{candidate}"
    if wrong_claim and norm_span(wrong_claim) not in norm_span(span_source):
        return _check(False, "invalid_span")
    if correct_value and norm_span(correct_value) not in norm_span(candidate):
        narrowed = narrow_to_new_value(wrong_claim, correct_value, candidate)
        if narrowed is None:
            return _check(False, "invalid_span")
        wrong_claim, correct_value = narrowed
    if reversed_restatement(wrong_claim, correct_value, candidate):
        logger.info(
            "Correction verdict rejected: invalid_span (reversed against the "
            "utterance's contrast; target=%s, wrong_claim=%r, correct_value=%r)",
            target, wrong_claim[:40], correct_value[:40],
        )
        return _check(False, "invalid_span", "reversed")
    if claims_equivalent(wrong_claim, correct_value):
        widened = widen_to_restated_value(wrong_claim, correct_value, candidate)
        if widened is None:
            return _check(False, "same_value")
        logger.info(
            "Correction verdict correct_value widened to the utterance's restated "
            "value: %r -> %r (wrong_claim=%r)",
            correct_value[:40], widened[:40], wrong_claim[:40],
        )
        correct_value, overridden = widened, "widened"
    if response_already_states(correct_value, source):
        return _check(False, "already_stated")
    return _check(True, None)
