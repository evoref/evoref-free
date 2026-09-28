"""本文をコードフェンスで囲むときの囲みの長さと、切った本文の閉じ (純粋関数)。

囲みの長さの決定を 1 か所に置く (2026-09-28 レビュー L1)。書き込んだ内容の
提示 (meta_cognitive)、書込みを断ったときの本文 (meta_cognitive)、逐語の
エコー (deliberative) が、それぞれ「``` を含むなら 4 連」と素朴に決めていた —
本文に 4 連があると囲みが内側で閉じる。本文の構造を読むだけで語彙は持たない。
"""

from __future__ import annotations

import re

#: 行頭のコードフェンス (字下げ可)。
FENCE_LINE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")


def outer_fence(body: str) -> str:
    """本文を囲むバッククォート列。本文の行頭のバッククォート列より 1 つ長くする (最短 3)。"""
    longest = max(
        (
            len(m.group("fence"))
            for line in (body or "").splitlines()
            if (m := FENCE_LINE_RE.match(line)) and m.group("fence")[0] == "`"
        ),
        default=0,
    )
    return "`" * max(3, longest + 1)


def close_open_fence(body: str) -> str:
    """切り詰めた本文の末尾で開いたままのフェンスを、開きと同じ字下げ・長さで閉じる。"""
    open_fence: tuple[str, str] | None = None
    for line in (body or "").splitlines():
        m = FENCE_LINE_RE.match(line)
        if m is None:
            continue
        fence = m.group("fence")
        if open_fence is None:
            open_fence = (m.group("indent"), fence)
        elif (
            fence[0] == open_fence[1][0]
            and len(fence) >= len(open_fence[1])
            and not m.group("info").strip()
        ):
            open_fence = None
    if open_fence is None:
        return body
    return f"{body}\n{open_fence[0]}{open_fence[1]}"


__all__ = ["FENCE_LINE_RE", "close_open_fence", "outer_fence"]
