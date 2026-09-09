"""訂正が指す「誤っていたターン」を証拠で同定する (純粋関数、pillar 横断)。

ユーザーの訂正発話は、**どのターンを訂正しているか** をどこにも記録して
いなかった。学習側 (``learning.corrected_pairs``) は後段で経験バッファの
**直前エントリ** を元の問いとみなして再導出しており、これが 2 つの前提を
置いている:

1. 訂正は対象ターンの **すぐ次** に来る
2. バッファに並ぶ隣接エントリは **同じ会話** のもの

どちらも実運用では成り立たない。2026-09-06 のライブ監査では、複数セッション
へ交互に訂正を送った結果、組まれたペアの ``query`` が **1 つ前の別セッション
の訂正文** になり (期待キーワードと問いが 1 つずれる)、訂正 7 件が few-shot
にも eval_core にも 1 件も入らなかった。設計上「Level 2 の重み学習は
on-device で動かないので、失敗の受け皿は Level 1 側しかない」としている、
その唯一の受け皿が沈黙したまま機能していなかった。

本モジュールは **記録時** (``agent.feedback``) に対象ターンの ID を確定する
ための判定を提供する。記録時点ではセッションが確定しており、直近ターンの
応答本文も手元にあるので、位置ではなく **本文の重なり** という証拠で選べる。

判定の骨子: 訂正文はほぼ必ず、誤っていた応答から **値や識別子を引用する**
(「冒頭の 1,096万4,771円 は…」「あなたが示した push の実装は…」)。よって
候補ターンの応答と訂正文の distinctive トークンの重なりを数え、最も重なる
ターンを対象とする。重なりが 1 つも無ければ直近ターン (従来の前提) へ倒す。

ここに置く理由: 記録側は EvorefLoop (``backend/free/agent``)、消費側は
EvorefLearn (``backend/free/learning``) で、Loop → Learn の import は
禁止されている。``core`` は両 pillar から参照できる純粋関数の正準置き場
(``core.text_quality`` と同じ立場) で、**判定を 1 本にしておかないと
「記録した対象」と「後段が想定する対象」がまた食い違う**。

到達範囲 (2026-09-06 の実データ 8 件で測定、宛先が既知のもので照合):

- 本モジュール: 6/8 正解。外した 2 件はどちらも **同一セッション内の話題が
  近いターン** で、別会話のターンと組む従来の壊れ方とは質が違う。
- 従来 (バッファ上の直前エントリ): 0/8。

外れる形は「同じ話題を複数ターンが扱っていて、訂正が語彙で区別できない」
場合 (「最初の回答は違います」のような **順序による指示**)。順序表現の
解釈は本モジュールでは扱わない — 語彙ごとの分岐を足すと、語形が 1 つ外れた
だけで壊れる判定をまた増やすことになる。外れても下流の ``depends_on_context``
/ ``response_honors_correction`` が誤ったペアを弾く多段防御に委ねる。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

#: 桁区切り・小数付きの数値。
NUMBER_LITERAL_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
#: 識別子 / 英単語 (3 文字以上)。``self.head`` のようなドット付きも 1 語で拾う。
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.*]{2,}")
#: 鉤括弧で囲まれた語 (引用された誤り / 正しい値)。
QUOTED_RE = re.compile(r"[「『]([^」』]{1,40})[」』]")
#: 和文の内容語。**ASCII 識別子と数値だけでは日本語の訂正から証拠が 1 つも
#: 取れない**。2026-09-06 の実データでは、「もう 1 つは 月曜:佐々木、火曜:川島…」
#: という訂正が (人名も曜日も拾えず) 全候補スコア 0 になり、宛先を直近ターンへ
#: 落としていた。漢字列 2 文字以上とカタカナ列 3 文字以上を内容語とみなす。
KANJI_RUN_RE = re.compile(r"[一-龥々]{2,}")
KATAKANA_RUN_RE = re.compile(r"[ァ-ヴ][ァ-ヴー]{2,}")
#: 英字 1〜2 文字 + 数字の短い記号 (``D7`` / ``E7`` / ``v2`` / ``T01``)。
#: ``IDENTIFIER_RE`` は 3 文字以上を要求するので、コードネームや版番号のような
#: 2 文字の答えが証拠から漏れていた (2026-09-07 ライブ監査: 「先ほどの D7 という
#: 回答は間違いです」の D7 が拾えず、応答が「D7」だけのターンを同定できなかった)。
CODE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]{1,2}\d{1,3}(?![A-Za-z0-9_])")

#: 「誤りだった側」を示す語。この語より **前** に置かれた値・引用は、訂正が
#: 「アシスタントが述べた誤り」として引いているものとみなす。
#:
#: 「X ではなく Y」の ``ではなく`` だけを境界にしていた頃は、日本語で誤りを指す
#: 最も普通の形 — 「…は間違いです」「…に掛けるものではありません」「「…」という
#: 注記は間違いです」 — がどれも境界にならず、誤りの側の値が **期待キーワード**
#: として eval_core に載った (2026-09-07 ライブ監査: 追加 5 件中 4 件が訂正された
#: 誤りを期待値にしていた。ユーザーが「消せ」と言った注記の文言が、その問いの
#: 正解条件になった)。
WRONG_MARKER_RE = re.compile(
    r"間違[いっえ]|誤[りっ]|違います|違う|ではなく|じゃなく"
    r"|ではありません|ではない|正しくありません|あり得ません|ありえません|おかしい"
    # 「「5 件のタスクをすべて完了しました」の一行 **だけで**、…返ってきていません」
    # — 引用が「それしか返っていない」誤りの側。
    r"|(?:だけ|のみ)(?:で|です|だ)",
)
#: 「正しい側」を示す語。この語より後ろは訂正後の値。
RIGHT_MARKER_RE = re.compile(r"正しくは|正解は|本当は|実際は|が正しい")
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?\n]")

#: どんな文にも現れる語。証拠にならない。
STOP_IDENTIFIERS = frozenset({
    "the", "and", "for", "not", "but", "with", "from", "import", "def", "class",
    "return", "print", "you", "your", "that", "this", "are", "was", "were",
})

#: 1 桁の数はどんな応答にも現れる (「8%」「37 個」の 8)。証拠にしない。
_MIN_NUMBER_CHARS = 2

#: 誤りの側の語 (:func:`wrong_side_tokens`) に掛ける重み。応答本文にその語が
#: あれば、それだけで通常の証拠 3 つ分に相当する。文書頻度で割らない —
#: 誤った値は後続ターンへ引き継がれて複数の応答に現れるのが普通で
#: (「56,417 件」が 4 ターンに残っていた)、割ると引き継いだ側と同じ重みまで
#: 薄まって、24,814 を述べただけの隣のターンに負ける。
WRONG_SIDE_WEIGHT = 3.0
#: 応答本文が誤りの値 **そのもの** (「41168%」「D7」) のときの加点。短い答えは
#: 他に証拠を持てないので、値の一致がほぼ確定の証拠になる。
WRONG_VALUE_ONLY_BONUS = 3.0
_WRONG_VALUE_ONLY_SLACK = 8
#: 「先ほどの **回答** は間違い」の回答 / 計算 / 数字 — 応答を指す語であって
#: 応答の内容ではない。誤りの側として重くするとどの候補にも付いて回る。
WRONG_SIDE_META_WORDS = frozenset({
    "回答", "答え", "計算", "数字", "数値", "説明", "結果", "出力", "注記", "末尾",
    "冒頭", "最後", "最初", "実装", "コード", "表現", "記述", "内容", "部分",
})

#: 遡って探す上限。これより前のターンを訂正するのは実運用でほぼ無く、
#: 広げるほど無関係なターンとの偶然の重なりを拾いやすくなる。
DEFAULT_LOOKBACK = 12


def _normalize(token: str) -> str:
    return token.replace(",", "").strip().lower()


def correction_evidence_tokens(correction: str) -> list[str]:
    """訂正文が引用している distinctive トークンを返す (順序保持・重複なし)。

    ``expected_keywords_from_correction`` (訂正 **後** の正しい値だけを残す) とは
    目的が違う。こちらは **誤っていた応答を同定する** ためのものなので、
    「X ではなく Y」の X 側 (誤りだった値) も落とさず残す — 誤りの側こそ、
    訂正対象の応答にだけ現れる最強の手掛かりになる。
    """
    text = correction or ""
    out: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        norm = _normalize(token)
        if not norm or norm in STOP_IDENTIFIERS or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    for m in QUOTED_RE.finditer(text):
        _add(m.group(1))
    for m in IDENTIFIER_RE.finditer(text):
        _add(m.group(0))
    for m in CODE_TOKEN_RE.finditer(text):
        _add(m.group(0))
    for m in KATAKANA_RUN_RE.finditer(text):
        _add(m.group(0))
    for m in KANJI_RUN_RE.finditer(text):
        _add(m.group(0))
    for m in NUMBER_LITERAL_RE.finditer(text):
        num = m.group(0).replace(",", "")
        if len(num) >= _MIN_NUMBER_CHARS:
            _add(num)
    return out


def wrong_side_spans(correction: str) -> list[str]:
    """訂正文のうち「誤りだった側」を述べている区間を返す (純粋関数)。

    文ごとに見て、:data:`WRONG_MARKER_RE` より前の部分を誤りの側とみなす。
    同じ文に :data:`RIGHT_MARKER_RE` があれば、そこから後ろは正しい側なので
    誤りの側から外す (「D7 は間違いで、正しくは E7」)。
    """
    spans: list[str] = []
    for sent in _SENTENCE_SPLIT_RE.split(correction or ""):
        if not sent.strip():
            continue
        right = RIGHT_MARKER_RE.search(sent)
        head = sent[:right.start()] if right else sent
        wrong = WRONG_MARKER_RE.search(head)
        if wrong is None:
            continue
        spans.append(head[:wrong.start()])
    return spans


def wrong_side_tokens(correction: str) -> list[str]:
    """訂正文が「誤り」として引いている値・識別子・引用 (順序保持・重複なし)。

    誤りの側こそ **訂正対象の応答にだけ現れる最強の手掛かり** なので、
    宛先の同定ではこれを重く見る。逆に期待キーワード (訂正後の正しい回答に
    含まれるべき語) からは必ず外す。

    **値だけ** を採る (数値 / 識別子 / 短い記号 / 鉤括弧の引用)。漢字・カタカナの
    内容語は採らない — 否定語より前の区間は「10 年後の売却価格が購入価格の何 %
    を下回ると購入が不利になるかを聞いており」のように **問いの言い直し** で
    埋まっていて、そこから拾った語 (売却価格 / 購入価格 / テーブル) を重くすると
    同じ話題を長く述べた隣のターンが勝つ。内容語は通常の重みで数える。
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        norm = _normalize(token)
        if not norm or norm in STOP_IDENTIFIERS or norm in seen:
            return
        if norm in WRONG_SIDE_META_WORDS:
            return
        seen.add(norm)
        out.append(norm)

    for span in wrong_side_spans(correction):
        for m in QUOTED_RE.finditer(span):
            _add(m.group(1))
        for m in IDENTIFIER_RE.finditer(span):
            _add(m.group(0))
        for m in CODE_TOKEN_RE.finditer(span):
            _add(m.group(0))
        for m in NUMBER_LITERAL_RE.finditer(span):
            num = m.group(0).replace(",", "")
            if len(num) >= _MIN_NUMBER_CHARS:
                _add(num)
    return out


def score_correction_match(correction: str, response: str) -> float:
    """訂正文と応答本文で重なる distinctive トークン数を返す (純粋関数)。

    単一の応答しか見られないので重み付けはしない。順位付けには
    :func:`resolve_correction_target` (候補集合で IDF を効かせる) を使う。
    """
    tokens = correction_evidence_tokens(correction)
    if not tokens:
        return 0.0
    body = _normalize(response or "")
    if not body:
        return 0.0
    return float(sum(1 for t in tokens if t in body))


def resolve_correction_target(
    correction: str,
    candidates: Sequence[tuple[str, str]],
    *,
    lookback: int = DEFAULT_LOOKBACK,
) -> str:
    """訂正が指すターンの ID を返す (該当なしは空文字列)。

    重なりの数え上げは **候補集合内の出現頻度で重み付け** する。どの応答にも
    出る語 (「RTT」「回答」「実装」) は宛先を分けないのに、素の件数では
    トークン数の多い長い応答が常に勝ってしまう。実データでは 0-RTT の
    リプレイ耐性への訂正が、「RTT」を共有するだけの前ターン (TLS 1.3 の
    ハンドシェイク解説) に吸われていた。1 つの候補にしか出ない語を重く、
    全候補に出る語を軽く扱う。

    証拠は 3 層で見る (2026-09-07 ライブ監査で 13 件中 2 件しか当たらなかった
    ことへの対処):

    1. **誤りの側の語** (:func:`wrong_side_tokens`) — 「先ほどの D7 という回答は
       間違いです」の D7。訂正対象の応答にしか無いはずの最強の手掛かりなので
       :data:`WRONG_SIDE_WEIGHT` 倍で数える。
    2. 応答本文との重なり (従来)。
    3. **元の問いとの重なり** — 訂正は「コードネームだけで答え直して」「総学習
       時間の見積もりを数値だけで」のように **依頼の形を言い直す** ことが多い。
       応答が「D7」「確認できていません」のように短いと本文からは証拠が取れず、
       同じ話題の長い応答に必ず負けていた (実測の失敗はまさにその短い最終
       ターンに集中する)。問い側の重なりは応答側と同じ重みで足す。

    Args:
        correction: 訂正発話の本文。
        candidates: **同一セッション** の ``(entry_id, response_text)`` または
            ``(entry_id, response_text, query_text)`` を古い順に並べたもの。
            呼出側がセッションで絞り、**ユーザーの訂正ターン自身は候補に
            入れない** 責務を持つ (入れると訂正が 1 つ前の訂正を宛先に選ぶ)。
            ここでセッションを跨いだ候補を渡すと、本モジュールが直そうと
            している欠陥をそのまま再現する。
        lookback: 遡る上限ターン数。

    Returns:
        最も証拠が重なるターンの ID。重なりが無ければ直近ターンの ID
        (訂正は直前ターンに向くという従来の前提へ倒す)。候補が空なら
        空文字列。
    """
    window = [c for c in candidates[-lookback:] if c and c[0]]
    if not window:
        return ""

    tokens = correction_evidence_tokens(correction)
    if not tokens:
        return window[-1][0]
    wrong = set(wrong_side_tokens(correction))

    rows = [
        (
            c[0],
            _normalize(c[1] or ""),
            _normalize(c[2] or "") if len(c) > 2 else "",
        )
        for c in window
    ]
    # 文書頻度 (この語を含む候補数)。0 件の語は誰の得点にもならないので無視。
    body_freq = {
        token: sum(1 for _, body, _q in rows if token in body)
        for token in tokens
    }
    query_freq = {
        token: sum(1 for _, _b, query in rows if query and token in query)
        for token in tokens
    }

    # 誤りの側の語 (最強の手掛かり) を **直近ターンも持っている** とき、古い
    # ターンは応答本文の固有の証拠でしか勝てない (問い側の重なりは数えない)。
    # 同じ誤値を複数ターンが述べている場面で、訂正が依頼を言い直す語
    # (「印刷費」「計算」) は古い問いにも現れ、それだけで 2 つ前の印刷費だけの
    # 回答が宛先になり「直前の回答の話題には触れないこと」の注記が製本費込みの
    # 合計を抑えかねなかった (2026-09-09 ライブ監査 C-01)。一方、誤値が後続へ
    # 引き継がれたときに **計算したターン** を指す訂正 (「1日1,900件…実施日数を
    # 計算し直して」) は本文の固有の値 (1,900) で古いターンが勝てるので壊れない。
    latest_body, latest_query = rows[-1][1], rows[-1][2]
    present_wrong = {
        t for t in wrong
        if any(t in body or t in query for _, body, query in rows)
    }
    latest_holds_wrong = bool(present_wrong) and all(
        t in latest_body or t in latest_query for t in present_wrong
    )

    best_id = ""
    best_score = 0.0
    # 新しい順に見る。同点なら新しい方を採る (同じ値を複数ターンが述べている
    # ときは直近を訂正しているとみなすのが自然)。
    for entry_id, body, query in reversed(rows):
        score = 0.0
        for token in tokens:
            if body_freq[token] and token in body:
                if token in wrong:
                    score += WRONG_SIDE_WEIGHT
                    if len(body) <= len(token) + _WRONG_VALUE_ONLY_SLACK:
                        score += WRONG_VALUE_ONLY_BONUS
                else:
                    score += 1.0 / body_freq[token]
            if query_freq[token] and token in query and not (
                latest_holds_wrong and entry_id != rows[-1][0]
            ):
                score += 1.0 / query_freq[token]
        if score > best_score:
            best_id, best_score = entry_id, score

    if best_score > 0:
        return best_id
    return window[-1][0]


__all__ = [
    "DEFAULT_LOOKBACK",
    "IDENTIFIER_RE",
    "KANJI_RUN_RE",
    "KATAKANA_RUN_RE",
    "NUMBER_LITERAL_RE",
    "QUOTED_RE",
    "STOP_IDENTIFIERS",
    "CODE_TOKEN_RE",
    "RIGHT_MARKER_RE",
    "WRONG_MARKER_RE",
    "WRONG_SIDE_META_WORDS",
    "WRONG_SIDE_WEIGHT",
    "WRONG_VALUE_ONLY_BONUS",
    "correction_evidence_tokens",
    "resolve_correction_target",
    "score_correction_match",
    "wrong_side_spans",
    "wrong_side_tokens",
]
