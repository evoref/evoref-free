"""クエリの内容語 (語彙アンカー) — 埋め込みのスケールに依存しない決定論の根拠。

「その話題を話しているか」を判定する読み手は 2 つある:

- EvorefMem の注入ゲート (:mod:`backend.free.memory.pipeline.injector`):
  属性辞書に無い話題 (「蕎麦」「苦手」「約束」) はコサインでしか拾えず、実測では
  背景と重なって届かない。クエリの内容語がファクト本文にそのまま出ていれば
  コサインの棒を免除する。
- EvorefLoop のツール判定ガード (:mod:`backend.free.agent.tool_judge_guards`):
  「答えは進行中の会話の窓の中にある」を根拠に除外検索 (現在セッションを除いた
  ``search_history``) を抑止するが、窓にクエリの内容語が 1 つも無いならその
  根拠は成り立たない (2026-09-09 ライブ監査 (d) D-09: 新規セッション 2 ターン目の
  「私が相談したプログラミング言語と、読み込んだファイルの形式」が抑止され
  「確認できません」に落ちた。答えは別セッションにしか無い)。

同じ語の切り出しを 2 箇所に書くと片方だけ直る (2026-09-09 D-02 と同じ形) ので、
どの pillar にも属さない ``core/`` に置く。
"""

from __future__ import annotations

import re
from backend.free.core.script_ranges import (
    KANJI,
    KATAKANA_WORD,
)

#: クエリから取り出す **内容語**。2 文字以上の漢字 / カタカナ / 英数字の連なり。
#: 1 文字を採らないのは助詞・接辞の断片が全文にマッチしてしまうため。
QUERY_ANCHOR_RE = re.compile(
    f"[{KANJI}]{{2,}}|[{KATAKANA_WORD}]{{2,}}|[A-Za-z0-9]{{2,}}",
)

#: 想起の **足場語**。どの想起クエリにも現れるので、これが一致しても
#: 「その話題だ」とは言えない。アンカーから除く。
ANCHOR_SCAFFOLD: frozenset[str] = frozenset({
    "自分", "今回", "会話", "記録", "情報", "内容", "以前", "過去", "確認",
    "教えて", "何度", "全部", "一度", "本当", "具体", "詳細", "最初", "最後",
    "さっき", "先ほど", "いま", "現在",
})


def query_anchors(query_text: str) -> tuple[str, ...]:
    """クエリの内容語 (語彙アンカー) を返す (純粋関数)。"""
    if not query_text:
        return ()
    return tuple({
        w for w in QUERY_ANCHOR_RE.findall(query_text)
        if w not in ANCHOR_SCAFFOLD
    })


def has_anchor(text: str, anchors: tuple[str, ...]) -> bool:
    """``text`` に語彙アンカーのいずれかがそのまま出現するか (純粋関数)。"""
    return bool(anchors) and any(a in text for a in anchors)


#: :func:`mentions_anchor` が数える ASCII の語の最短長。2 文字の語 (``it`` / ``km`` /
#: ``10``) は無関係な本文の語の一部や数値に当たりすぎる。
MIN_ASCII_ANCHOR_LEN = 3

_ASCII_ANCHOR_RE = re.compile(r"[A-Za-z0-9]+")


def word_anchors(anchors: tuple[str, ...]) -> tuple[str, ...]:
    """:func:`mentions_anchor` で照合する語だけを残す (純粋関数)。

    ASCII の語は :data:`MIN_ASCII_ANCHOR_LEN` 文字以上だけ (1〜2 桁の数値も落ちる)。
    """
    return tuple(
        a for a in anchors
        if not _ASCII_ANCHOR_RE.fullmatch(a) or len(a) >= MIN_ASCII_ANCHOR_LEN
    )


def mentions_anchor(text: str, anchors: tuple[str, ...]) -> bool:
    """``text`` が語彙アンカーのいずれかを **語として** 含むか (純粋関数)。

    :func:`has_anchor` (素の部分一致) より厳しい照合。ASCII の語は大小を問わず
    英数字の境界で照合し (``it`` が "with" に当たらない)、短い ASCII の語は
    数えない (:func:`word_anchors`)。日本語の語は部分一致のまま (語境界が無い)。
    """
    for anchor in word_anchors(anchors):
        if _ASCII_ANCHOR_RE.fullmatch(anchor):
            pattern = rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])"
            if re.search(pattern, text or "", re.IGNORECASE):
                return True
        elif anchor in (text or ""):
            return True
    return False
