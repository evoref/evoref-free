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

#: クエリから取り出す **内容語**。2 文字以上の漢字 / カタカナ / 英数字の連なり。
#: 1 文字を採らないのは助詞・接辞の断片が全文にマッチしてしまうため。
QUERY_ANCHOR_RE = re.compile(r"[一-鿿]{2,}|[ァ-ヴー]{2,}|[A-Za-z0-9]{2,}")

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
