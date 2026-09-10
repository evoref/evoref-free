"""訂正ペア — ユーザーの訂正で確定した「問い → 正しい回答」の組 (純粋関数)。

ユーザーがアシスタントの誤りを訂正すると、経験バッファには **訂正発話** の
ターン (query = 訂正文、response = 訂正後の回答、``signals.user_correction`` 付き)
と、その直前の **訂正された** ターン (query = 元の問い、response = 誤った回答)
が並ぶ。学習側はこれまで前者を「失敗」として数えるだけで、**訂正後の正しい回答は
一度も手本にならなかった** (few-shot は ``user_correction`` 付きのターンを丸ごと
除外する)。2026-09-05 の失敗 32 件を全数確認したところ、on-device の重み学習
(Level 2) はこのデータを目的関数に載せても動かず、失敗の受け皿は Level 1 側
(few-shot / 採用ゲート / eval_core) しか無い。本モジュールはその受け皿に流す
「元の問い → 訂正後の回答」ペアを経験から組み立てる。

判定の要点:

- ``user_correction`` は **検証済み** の訂正だけが入る
  (``learning.correction_verifier`` が直前応答と突き合わせて帰属を判定し、
  ``assistant`` のものだけを昇格させる)。ここでは帰属を再判定しない。
  期待語は検証器が抜いた ``signals.correction_correct_value`` を優先する。
- 訂正後の回答から謝罪・受諾の前置き (「おっしゃる通りです。訂正いたします。」)
  を剥がす。前置きが手本に残ると、few-shot が「まず謝る」型を再生産する。
- 剥がした後が受諾だけ (「承知しました。住まいは福岡ですね。」) なら回答では
  ないので捨てる。
- ``depends_on_context`` は、元の問いが直前ターン・記憶・ツール結果を前提に
  しているか (照応語 / 一人称 / ファイル操作語) を見る。文脈無しで再生成・
  再評価できるペアだけが eval_core の候補になる。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from backend.free.core.correction_target import (
    IDENTIFIER_RE,
    NUMBER_LITERAL_RE,
    QUOTED_RE,
    STOP_IDENTIFIERS,
    wrong_side_tokens,
)

#: 訂正後の回答の先頭に付く謝罪・受諾の前置き。1 文単位で繰り返し剥がす。
_PREAMBLE_SENTENCE_RE = re.compile(
    r"^\s*(?:"
    r"(?:大変|誠に)?(?:申し訳(?:ありません|ございません)|すみません|失礼(?:しました|いたしました))"
    r"|おっしゃる通りです|ご指摘(?:の通り|ありがとうございます)[^。]*"
    r"|(?:訂正|修正)(?:します|いたします)|承知(?:しました|いたしました)|かしこまりました"
    r"|はい[、,]?"
    r")[。、,.!！\s]*",
)

#: 前置きを剥がした後に残るのが受諾・復唱だけの形 (回答ではない)。
_ACK_ONLY_RE = re.compile(
    r"^[^。\n]{0,60}(?:ですね|として(?:修正|記録|認識)します|と(?:認識|記録)しました"
    r"|に修正します|で承知しました)[。.!！]?\s*$",
)

#: 直前ターンへの照応・継続 (「修正版に」「2 案を採用」「その」「続けて」)。
#:
#: ``その他`` / ``それぞれ`` は **照応ではない** — 前者は「その他大勢」の
#: 一語、後者は複数対象への配分を表す副詞で、どちらも直前ターンを指さない。
#: 素の ``その`` / ``それ`` で拾うと、文脈非依存の問いまで文脈依存に倒れる。
#: 実測 (2026-09-06): 正しく組めた訂正ペア 4 件のうち 2 件がこの 2 語だけで
#: 棄却され、eval_core / few-shot への追加が 0 件のままだった (F-01 の宛先を
#: 直しても受け皿に届かない)。
#:
#: やり直しの動詞 (「計算し直して」「書き直して」) と、述べた値の訂正
#: (「18 kg ではなく 22 kg でした」) も直前ターンを指す — やり直す対象、
#: 訂正で入れ替わらなかった残りの前提 (積載量 4.5 t / 6 便) は前のターンに
#: しか無い。実インシデント (2026-09-09 ライブ監査 (d) D-08): 「すみません、
#: 荷物は 18 kg ではなく 22 kg でした。1 台あたりの個数と 1 日の総個数を
#: 計算し直してください。」が few-shot に採用され、手本の応答が問いに無い
#: 4.5 トン・6 便を前提に答える形 (= 無い前提を補う型) を教えていた。
#: 「見直す」(検討する) は対象を含む新規の問いに現れるので含めない。
_ANAPHORA_RE = re.compile(
    r"それ(?!ぞれ)|その(?!他)|これ|この|あれ|あの|さっき|先(?:ほど|程)|直前"
    r"|(?<![名以事手])前の"
    r"|(?<!の)上の|上記|同じ|同様|続き|続けて|もう一度|再度|最初の|ここまで|今の"
    r"|修正版|最終版|改訂版|案を採用|を採用"
    r"|[しりきぎみびいえけせてねめれ]直(?:し|す|せ|さ)"
    r"|(?:ではなく|じゃなく)[^。！？!?\n]{0,24}?でした",
)
#: 記憶・ツール結果を前提にしている手掛かり (照応に加えて)。
_MEMORY_OR_TOOL_RE = re.compile(
    r"私|僕|自分|俺|覚え|記憶|言いました|言ったか|でしたか|でしたっけ"
    r"|ファイル|保存|読んで|読み|実行|検索|コマンド|ディレクトリ|フォルダ",
)

#: 訂正文から期待キーワードを拾う正規表現。**``core.correction_target`` が
#: SSOT**。同じトークン抽出を 2 箇所に書くと、片方だけ直したときに
#: 「訂正の宛先を決めた根拠」と「期待キーワード」が別基準になる。
_NUMBER_RE = NUMBER_LITERAL_RE
_IDENTIFIER_RE = IDENTIFIER_RE
_QUOTED_RE = QUOTED_RE
#: 「X ではなく Y」の境界。X 側 (誤りだった値) は期待値ではない。「それは
#: 間違いです」型の全体否定は境界にしない (回答全体を指すだけで値を挟まない)。
_NEGATED_VALUE_RE = re.compile(r"(?:ではなく|じゃなく)")
#: 「… = 51,148.8 円」の右辺。訂正が式で示されたら、答えの値は必須の期待語。
_EQUATION_RHS_RE = re.compile(r"[=＝]\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_STOP_IDENTIFIERS = STOP_IDENTIFIERS
#: 短い継続指示 (「表にしてください。」「箇条書きで。」「続けて。」): 対象を
#: 言わない依頼形の文末で、この長さ以下。「べき等性とは何ですか。」のような
#: 短い問いは対象を含むので拾わない。
_MAX_CONTINUATION_CHARS = 14
_CONTINUATION_TAIL_RE = re.compile(
    r"(?:にして|で|に|を|も)(?:ください|下さい|お願いします|くれ|ね)?[。.!！]?\s*$"
    r"|(?:続けて|続きを|もう一度|再度)[。.!！]?\s*$",
)
#: ユーザー自身の属性の話 (記憶が要る)。
_PERSONAL_ATTR_RE = re.compile(
    r"趣味|職業|名前|住ま|住んで|出身|誕生日|好きな|勤め|会社|年齢|家族|ペット"
    r"|はじめまして|申します|と言います|といいます",
)


@dataclass(frozen=True)
class CorrectedPair:
    """訂正で確定した 1 組。"""

    query: str
    response: str
    correction: str
    mode: str
    timestamp: str = ""
    #: 訂正前の (誤った) 回答。判定・ログ用で、手本には使わない。
    wrong_response: str = ""
    #: 検証器 (``learning.correction_verifier``) が確定した「正しい値」の逐語
    #: span。空でなければ期待語はここから採る (字句の否定境界に頼らない)。
    correct_value: str = ""

    @property
    def pair_id(self) -> str:
        return hashlib.blake2b(
            f"{self.query.strip()}\x00{self.response.strip()}".encode("utf-8"),
            digest_size=6,
        ).hexdigest()


def strip_correction_preamble(response: str) -> str:
    """訂正後の回答から謝罪・受諾の前置きを剥がす (純粋関数)。

    「おっしゃる通りです。訂正いたします。\\n\\n`TaskGroup`は…」→「`TaskGroup`は…」。
    前置きしか無い / 受諾・復唱だけなら空文字列を返す。
    """
    text = (response or "").strip()
    for _ in range(4):
        m = _PREAMBLE_SENTENCE_RE.match(text)
        if not m or not m.group(0).strip():
            break
        text = text[m.end():].lstrip()
    if not text or _ACK_ONLY_RE.match(text):
        return ""
    return text


def refers_to_previous_turn(query: str) -> bool:
    """問いが直前ターンへの照応・継続か (純粋関数)。

    照応語 (「その」「修正版に」「2 案を採用」) と、短すぎる継続指示
    (「表にしてください。」) を拾う。few-shot の入口 (``find_content_rejection``)
    がこれで単独では意味を成さない問いを手本から外す。
    """
    q = (query or "").strip()
    if len(q) <= _MAX_CONTINUATION_CHARS and _CONTINUATION_TAIL_RE.search(q):
        return True
    return bool(_ANAPHORA_RE.search(q))


def depends_on_context(query: str) -> bool:
    """元の問いが直前ターン・記憶・ツール結果を前提にしているか (純粋関数)。

    :func:`refers_to_previous_turn` に加えて、一人称 / ユーザー属性の話 (記憶が
    要る) とファイル操作語 (ツール結果が要る) も文脈依存とみなす。eval_core の
    候補 (文脈無しで再生成・再評価するケース) の足切りに使う。
    """
    q = (query or "").strip()
    if refers_to_previous_turn(q):
        return True
    return bool(_MEMORY_OR_TOOL_RE.search(q) or _PERSONAL_ATTR_RE.search(q))


def expected_keywords_from_correction(
    correction: str, *, correct_value: str = "",
) -> list[str]:
    """訂正文から「正しい回答に含まれるべき語」を拾う (純粋関数、順序保持・重複なし)。

    「正しくは 1,280 × 37 × 1.08 = 51,148.8 円です」→ ``["1280", "37", "1.08", "51148.8"]``。
    「BaseExceptionGroup は「単一の例外の場合」ではなく、KeyboardInterrupt など…」
    → 「ではなく」の **前** にある値は誤りだった側なので落とし、
    ``["KeyboardInterrupt", ...]`` のように訂正後の側だけを残す。

    ``correct_value`` (検証器が訂正発話から逐語で抜いた正しい値) があれば
    **そちらだけ** を語源にする。字句の否定境界は ``ではなく`` /
    ``じゃなく`` の 2 語しか無く、それ以外の言い回しでは誤り側が期待語に
    載っていた (2026-09-07 監査 F-01: eval_core 追加 5 件中 4 件)。
    """
    text = (correct_value or "").strip() or (correction or "")
    # 「X ではなく Y」の X 側 (誤り) を落とす: 否定語より前の鉤括弧語は捨てる。
    neg = None
    for neg in _NEGATED_VALUE_RE.finditer(text):
        pass
    quoted_scope = text[neg.end():] if neg else text
    # 誤りの側の語 (「56,417 件という数字は間違いです」の 56417、「「会話の前半は
    # 参照できない」という注記は間違いです」の引用) は期待語にしない。
    # ``ではなく`` だけを境界にしていた頃はこれらが素通りし、eval_core に
    # 「訂正された誤りを含むこと」が正解条件として載った (2026-09-07 ライブ
    # 監査: 追加 5 件中 4 件)。判定は ``core.correction_target`` が SSOT。
    wrong = {_normalize_for_match(t) for t in wrong_side_tokens(text)}
    out: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        tok = tok.strip()
        if not tok or tok.lower() in _STOP_IDENTIFIERS or tok in seen:
            return
        if _normalize_for_match(tok) in wrong:
            return
        seen.add(tok)
        out.append(tok)

    for m in _QUOTED_RE.finditer(quoted_scope):
        _add(m.group(1))
    for m in _IDENTIFIER_RE.finditer(quoted_scope):
        _add(m.group(0))
    for m in _NUMBER_RE.finditer(text):
        num = m.group(0).replace(",", "")
        # 1 桁の数はどんな回答にも現れる (「8%」「37 個」の 8)。期待語にしない。
        if len(num) >= 2:
            _add(num)
    return out


def _normalize_for_match(text: str) -> str:
    return (text or "").replace(",", "").lower()


def response_honors_correction(
    response: str, correction: str, *, correct_value: str = "",
) -> bool:
    """訂正後の回答が、訂正文から拾える期待語を実際に含んでいるか (純粋関数)。

    訂正されても同じ誤りを繰り返す回答 (2026-09-05 実機: 「正しくは 51,148.8
    円です」への「1,280円 × 37個 × 1.08 = 50,688円です」) を手本にしない。
    期待語が 1 つも拾えない訂正 (「違います」だけ等) は判定できないので通す。
    """
    keywords = expected_keywords_from_correction(
        correction, correct_value=correct_value,
    )
    if not keywords:
        return True
    body = _normalize_for_match(response)
    # 式の右辺 (答え) は必須。被演算子が揃っていても答えが違えば訂正を
    # 受け入れていない (「1,280 × 37 × 1.08 = 50,688円」は 3/4 語が一致する)。
    rhs = [_normalize_for_match(m.group(1)) for m in _EQUATION_RHS_RE.finditer(correction or "")]
    if rhs:
        # 答えが載っていれば被演算子の再掲は要らない (「税込合計は 51,148.8 円です」)。
        return all(v in body for v in rhs)
    hits = sum(1 for k in keywords if _normalize_for_match(k) in body)
    return hits * 2 >= len(keywords)


def resolve_corrected_turn(
    experiences: list[dict], index: int,
) -> dict | None:
    """``experiences[index]`` の訂正が指す **誤っていたターン** を返す (純粋関数)。

    優先順:

    1. ``signals.corrected_entry_id`` — 記録時に
       ``core.correction_target.resolve_correction_target`` が本文の重なりで
       確定した宛先。**これが唯一の正しい情報源**。
    2. 同一 ``session_id`` の直前ターン — 旧データ (宛先未記録) 向け。
       会話をまたがないので、少なくとも別の会話のターンとは組まれない。
    3. 双方に ``session_id`` が無い場合のみ、バッファ上の直前ターン。
       セッションの概念が無かった時期のデータを取りこぼさないための最終手段。

    2 と 3 を分けているのは、``session_id`` を持つデータで 3 に落ちると
    **本モジュールが直そうとしている欠陥そのもの** (別会話のターンと組む) が
    再現するため。
    """
    exp = experiences[index]
    signals = exp.get("signals") or {}
    target_id = signals.get("corrected_entry_id")
    if target_id:
        for cand in reversed(experiences[:index]):
            if cand.get("id") == target_id:
                return cand
        # ID はあるが対象がバッファから溢れている = 学習に使える素材が無い。
        # 位置で代用すると誤ったペアになるので諦める。
        return None
    if "corrected_entry_id" in signals:
        # 記録時に宛先を解決した世代のデータで、それでも None = 同一セッションに
        # 候補が 1 つも無かった。位置で代用しない — 訂正が会話の最後に来ると
        # 「そのセッションの最終ターン」が拾われ、「Q: サビで転調させたい… /
        # A: E7」のような別の問いへの手本ができる (2026-09-07 ライブ監査:
        # 組まれた 10 ペアのうち 7 件が Q/A 不一致で、内容ゲートは止められない)。
        return None

    session = str(exp.get("session_id") or "")
    if session:
        for cand in reversed(experiences[:index]):
            if str(cand.get("session_id") or "") == session:
                return cand
        return None

    prev = experiences[index - 1] if index > 0 else None
    if prev is not None and str(prev.get("session_id") or ""):
        # 訂正側にセッションが無く直前ターンにはある = 別会話の可能性が高い。
        return None
    return prev


def build_corrected_pairs(
    experiences: list[dict], mode: str | None = None,
) -> list[CorrectedPair]:
    """経験バッファ (時系列順) から訂正ペアを組み立てる (純粋関数)。

    ``mode`` を渡すとそのモードのターンだけを見る。同一 (query, response) は
    最新 1 件に畳む。

    対応付けは :func:`resolve_corrected_turn` に委ねる。以前はここが
    **バッファ上の直前エントリ** を無条件に元の問いとみなしており、会話を
    またいで訂正が並ぶと問いと訂正が食い違った (2026-09-06 監査 F-01)。

    訂正後の回答が **自身の検証に失敗している** ターン
    (``turn_outcome == "failed"``: 算術矛盾 / 出力破損 / 制約違反) は手本にも
    評価ケースにもしない。訂正で 1 つ直しても別の欠陥を持ち込んだ回答を
    「正しい回答」として再生産すると、few-shot が壊れた形を増幅する
    (2026-09-06 監査 F-05)。
    """
    pairs: dict[str, CorrectedPair] = {}
    scoped = [
        e for e in experiences
        if mode is None or e.get("mode") == mode
    ]
    for i, exp in enumerate(scoped):
        signals = exp.get("signals") or {}
        correction = signals.get("user_correction")
        if not correction:
            continue
        if signals.get("truncated", False):
            continue
        if signals.get("turn_outcome") == "failed":
            continue
        prev = resolve_corrected_turn(scoped, i)
        if prev is None:
            continue
        query = str(prev.get("query") or "").strip()
        fixed = strip_correction_preamble(
            str(exp.get("response_full") or exp.get("response_summary") or ""),
        )
        if not (query and fixed):
            continue
        correct_value = str(signals.get("correction_correct_value") or "").strip()
        if not response_honors_correction(
            fixed, str(correction), correct_value=correct_value,
        ):
            continue
        pair = CorrectedPair(
            query=query,
            response=fixed,
            correction=str(correction).strip(),
            mode=str(exp.get("mode") or "chat"),
            timestamp=str(exp.get("timestamp") or ""),
            wrong_response=str(
                prev.get("response_full") or prev.get("response_summary") or ""
            ),
            correct_value=correct_value,
        )
        pairs[pair.pair_id] = pair
    return list(pairs.values())


def build_demoted_pairs(experiences: list[dict], mode: str | None = None) -> list[CorrectedPair]:
    """検証で **却下 / 格下げ** された訂正候補が組んでいたはずのペアを返す (純粋関数)。

    :func:`build_corrected_pairs` の裏返し。``correction_candidate`` があり、
    検証済み (``correction_verdict`` が立っている) なのに ``user_correction`` が
    無いエントリ — 検証器が最初から却下したもの、または
    :func:`~backend.free.learning.correction_verifier.recheck_promoted` が後から
    候補へ戻したもの — について、同じ (元の問い, 訂正後の回答) を組む。

    用途は **取り消し**。few-shot プールと eval_core は昇格時にペアを受け取る
    だけで、格下げを伝える経路が無かった。門を足しても既存の偽陽性
    (100 → 100 m) が fitness 1.0 の手本と評価ケースに残り続けた (2026-09-09
    ライブ監査 P-1: recheck は 11:31 に格下げしたが、どちらにも残ったまま)。

    受け皿側の内容ゲート (``response_honors_correction`` 等) は掛けない —
    取り消すべきものを取りこぼす方が害が大きい。
    """
    pairs: dict[str, CorrectedPair] = {}
    scoped = [
        e for e in experiences
        if mode is None or e.get("mode") == mode
    ]
    for i, exp in enumerate(scoped):
        signals = exp.get("signals") or {}
        if signals.get("user_correction"):
            continue
        candidate = signals.get("correction_candidate")
        if not candidate or not signals.get("correction_verdict"):
            continue
        prev = resolve_corrected_turn(scoped, i)
        if prev is None:
            continue
        query = str(prev.get("query") or "").strip()
        fixed = strip_correction_preamble(
            str(exp.get("response_full") or exp.get("response_summary") or ""),
        )
        if not (query and fixed):
            continue
        pair = CorrectedPair(
            query=query,
            response=fixed,
            correction=str(candidate).strip(),
            mode=str(exp.get("mode") or "chat"),
            timestamp=str(exp.get("timestamp") or ""),
            correct_value=str(signals.get("correction_correct_value") or "").strip(),
        )
        pairs[pair.pair_id] = pair
    return list(pairs.values())


__all__ = [
    "CorrectedPair",
    "build_corrected_pairs",
    "build_demoted_pairs",
    "depends_on_context",
    "expected_keywords_from_correction",
    "refers_to_previous_turn",
    "resolve_corrected_turn",
    "response_honors_correction",
    "strip_correction_preamble",
]
