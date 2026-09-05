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

- ``user_correction`` は記録時点で **アシスタントの誤りに対する訂正** に絞られて
  いる (``classify_correction_target``)。ここでは帰属を再判定しない。
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
_ANAPHORA_RE = re.compile(
    r"それ|その|これ|この|あれ|あの|さっき|先(?:ほど|程)|直前|(?<![名以事手])前の"
    r"|(?<!の)上の|上記|同じ|同様|続き|続けて|もう一度|再度|最初の|ここまで|今の"
    r"|修正版|最終版|改訂版|案を採用|を採用",
)
#: 記憶・ツール結果を前提にしている手掛かり (照応に加えて)。
_MEMORY_OR_TOOL_RE = re.compile(
    r"私|僕|自分|俺|覚え|記憶|言いました|言ったか|でしたか|でしたっけ"
    r"|ファイル|保存|読んで|読み|実行|検索|コマンド|ディレクトリ|フォルダ",
)

#: 訂正文から期待キーワードを拾う: 数値 (桁区切り・小数付き)、識別子/英単語
#: (3 文字以上)、鉤括弧で囲まれた語。
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.*]{2,}")
_QUOTED_RE = re.compile(r"[「『]([^」』]{1,40})[」』]")
#: 「X ではなく Y」の境界。X 側 (誤りだった値) は期待値ではない。「それは
#: 間違いです」型の全体否定は境界にしない (回答全体を指すだけで値を挟まない)。
_NEGATED_VALUE_RE = re.compile(r"(?:ではなく|じゃなく)")
#: 「… = 51,148.8 円」の右辺。訂正が式で示されたら、答えの値は必須の期待語。
_EQUATION_RHS_RE = re.compile(r"[=＝]\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_STOP_IDENTIFIERS = frozenset({
    "the", "and", "for", "not", "but", "with", "from", "import", "def", "class",
    "return", "print",
})
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


def expected_keywords_from_correction(correction: str) -> list[str]:
    """訂正文から「正しい回答に含まれるべき語」を拾う (純粋関数、順序保持・重複なし)。

    「正しくは 1,280 × 37 × 1.08 = 51,148.8 円です」→ ``["1280", "37", "1.08", "51148.8"]``。
    「BaseExceptionGroup は「単一の例外の場合」ではなく、KeyboardInterrupt など…」
    → 「ではなく」の **前** にある値は誤りだった側なので落とし、
    ``["KeyboardInterrupt", ...]`` のように訂正後の側だけを残す。
    """
    text = correction or ""
    # 「X ではなく Y」の X 側 (誤り) を落とす: 否定語より前の鉤括弧語は捨てる。
    neg = None
    for neg in _NEGATED_VALUE_RE.finditer(text):
        pass
    quoted_scope = text[neg.end():] if neg else text
    out: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        tok = tok.strip()
        if tok and tok.lower() not in _STOP_IDENTIFIERS and tok not in seen:
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


def response_honors_correction(response: str, correction: str) -> bool:
    """訂正後の回答が、訂正文から拾える期待語を実際に含んでいるか (純粋関数)。

    訂正されても同じ誤りを繰り返す回答 (2026-09-05 実機: 「正しくは 51,148.8
    円です」への「1,280円 × 37個 × 1.08 = 50,688円です」) を手本にしない。
    期待語が 1 つも拾えない訂正 (「違います」だけ等) は判定できないので通す。
    """
    keywords = expected_keywords_from_correction(correction)
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


def build_corrected_pairs(
    experiences: list[dict], mode: str | None = None,
) -> list[CorrectedPair]:
    """経験バッファ (時系列順) から訂正ペアを組み立てる (純粋関数)。

    ``mode`` を渡すとそのモードのターンだけを見る。同一 (query, response) は
    最新 1 件に畳む。
    """
    pairs: dict[str, CorrectedPair] = {}
    prev: dict | None = None
    for exp in experiences:
        if mode is not None and exp.get("mode") != mode:
            continue
        signals = exp.get("signals") or {}
        correction = signals.get("user_correction")
        if correction and prev is not None:
            query = str(prev.get("query") or "").strip()
            fixed = strip_correction_preamble(
                str(exp.get("response_full") or exp.get("response_summary") or ""),
            )
            if (
                query and fixed
                and not signals.get("truncated", False)
                and response_honors_correction(fixed, str(correction))
            ):
                pair = CorrectedPair(
                    query=query,
                    response=fixed,
                    correction=str(correction).strip(),
                    mode=str(exp.get("mode") or "chat"),
                    timestamp=str(exp.get("timestamp") or ""),
                    wrong_response=str(
                        prev.get("response_full") or prev.get("response_summary") or ""
                    ),
                )
                pairs[pair.pair_id] = pair
        prev = exp
    return list(pairs.values())


__all__ = [
    "CorrectedPair",
    "build_corrected_pairs",
    "depends_on_context",
    "expected_keywords_from_correction",
    "refers_to_previous_turn",
    "response_honors_correction",
    "strip_correction_preamble",
]
