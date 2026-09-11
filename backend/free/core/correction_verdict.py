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
- :func:`check_verdict` — 上の門を LLM 出力へ順に当て、通ったものだけ
  :class:`VerdictCheck` として返す。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

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
_CONTENT_RUN_RE = re.compile(r"[一-龥ァ-ヶーA-Za-z0-9]{2,}")
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
        "アシスタントが言及していなくても self)\n"
        "- third_party: 第三者・世間・引用元の誤りに言及している\n"
        "- premise_change: 前提や条件を変えてやり直しを頼んでいる (誤りの指摘ではない)\n"
        "- none: 訂正ではない (質問・比較・依頼・伝聞・確認など)\n\n"
        "重要: アシスタントの応答が **既にユーザーの示す値と同じ** なら、それは"
        "誤りの指摘ではありません (is_correction=false, target=none)。\n"
        "鉤括弧の内側 (引用・伝聞) は本人の主張ではありません。\n\n"
        "wrong_claim には誤っていた箇所を **そのまま逐語で** 抜き出してください "
        "(target=assistant なら直前のアシスタント応答から、target=self なら"
        "前のユーザー発話か、ユーザー発話中の「X ではなく」の X から。無ければ空文字)。\n"
        "correct_value にはユーザー発話が示した正しい値を"
        "**そのまま逐語で** 抜き出してください (無ければ空文字)。\n"
        "言い換え・要約・補完をしてはいけません。\n\n"
        f"{prev_user_block}"
        f"ASSISTANT_RESPONSE:\n{prev_response[:RESPONSE_CAP]}\n\n"
        f"USER_UTTERANCE:\n{correction[:QUERY_CAP]}\n"
    )


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


def check_verdict(
    payload: Any,
    *,
    candidate: str,
    prev_response: str,
    prev_user: str = "",
) -> VerdictCheck:
    """LLM の判定に **コード側の門** を順に当てる (純粋関数)。

    1. 形 (dict / target が既知の値)
    2. ``is_correction`` と帰属 — ``assistant`` / ``self`` 以外は訂正でない
    3. 逐語 span — ``wrong_claim`` は帰属先の本文に、``correct_value`` は
       訂正発話に、空白を無視して含まれること
    4. 同値 — ``wrong_claim`` と ``correct_value`` が同じ値なら訂正でない
    5. 既述 — 帰属先の本文が ``correct_value`` を既に述べているなら訂正でない

    消費側は ``ok`` のものをさらに ``target`` で絞る (学習側は ``assistant`` のみ)。
    """
    if not isinstance(payload, dict):
        return VerdictCheck(False, "no_verdict", "", "", "")
    target = str(payload.get("target") or "")
    wrong_claim = strip_copula(str(payload.get("wrong_claim") or ""))
    correct_value = strip_copula(str(payload.get("correct_value") or ""))
    if target not in CORRECTION_TARGETS:
        return VerdictCheck(False, "invalid_target", target, wrong_claim, correct_value)
    if not payload.get("is_correction") or target not in POINTING_TARGETS:
        return VerdictCheck(False, "not_correction", target, wrong_claim, correct_value)

    # self の古い値は前のユーザー発話にあるか、訂正文自身に同居している
    # (「盛岡市ではなく花巻市でした」)。既述の判定 (下) は前の発話だけで見る。
    source = prev_response if target == "assistant" else (prev_user or prev_response)
    span_source = source if target == "assistant" else f"{source}\n{candidate}"
    if wrong_claim and norm_span(wrong_claim) not in norm_span(span_source):
        return VerdictCheck(False, "invalid_span", target, wrong_claim, correct_value)
    if correct_value and norm_span(correct_value) not in norm_span(candidate):
        return VerdictCheck(False, "invalid_span", target, wrong_claim, correct_value)
    if claims_equivalent(wrong_claim, correct_value):
        return VerdictCheck(False, "same_value", target, wrong_claim, correct_value)
    if response_already_states(correct_value, source):
        return VerdictCheck(False, "already_stated", target, wrong_claim, correct_value)
    return VerdictCheck(True, None, target, wrong_claim, correct_value)
