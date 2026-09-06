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

#: どんな文にも現れる語。証拠にならない。
STOP_IDENTIFIERS = frozenset({
    "the", "and", "for", "not", "but", "with", "from", "import", "def", "class",
    "return", "print", "you", "your", "that", "this", "are", "was", "were",
})

#: 1 桁の数はどんな応答にも現れる (「8%」「37 個」の 8)。証拠にしない。
_MIN_NUMBER_CHARS = 2

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
    for m in KATAKANA_RUN_RE.finditer(text):
        _add(m.group(0))
    for m in KANJI_RUN_RE.finditer(text):
        _add(m.group(0))
    for m in NUMBER_LITERAL_RE.finditer(text):
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

    Args:
        correction: 訂正発話の本文。
        candidates: **同一セッション** の ``(entry_id, response_text)`` を
            古い順に並べたもの。呼出側がセッションで絞る責務を持つ —
            ここでセッションを跨いだ候補を渡すと、本モジュールが直そうと
            している欠陥をそのまま再現する。
        lookback: 遡る上限ターン数。

    Returns:
        最も証拠が重なるターンの ID。重なりが無ければ直近ターンの ID
        (訂正は直前ターンに向くという従来の前提へ倒す)。候補が空なら
        空文字列。
    """
    window = [c for c in candidates[-lookback:] if c[0]]
    if not window:
        return ""

    tokens = correction_evidence_tokens(correction)
    if not tokens:
        return window[-1][0]

    bodies = [(entry_id, _normalize(text or "")) for entry_id, text in window]
    # 文書頻度 (この語を含む候補数)。0 件の語は誰の得点にもならないので無視。
    doc_freq = {
        token: sum(1 for _, body in bodies if token in body)
        for token in tokens
    }

    best_id = ""
    best_score = 0.0
    # 新しい順に見る。同点なら新しい方を採る (同じ値を複数ターンが述べている
    # ときは直近を訂正しているとみなすのが自然)。
    for entry_id, body in reversed(bodies):
        score = sum(
            1.0 / doc_freq[token]
            for token in tokens
            if doc_freq[token] and token in body
        )
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
    "correction_evidence_tokens",
    "resolve_correction_target",
    "score_correction_match",
]
