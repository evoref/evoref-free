"""文書抽出器が共有する表記 — 画像・図形の印と GFM 表

設計書: [docs/f_11_file_export.md](../../../../docs/f_11_file_export.md) §3.3 / §5.1。
"""

from __future__ import annotations

import re

#: 本文に画像が在ることを示す印。抽出はテキストなので画像そのものは読み戻せない
#: (f_11 §5.1) が、画像だけの文書を「中身が無い」と読むと追記が止まる (§3.3)。
IMAGE_MARK = "[image]"
#: 文字を持たない図形 (線・塗りだけの四角形など) が在ることを示す印。
SHAPE_MARK = "[shape]"

#: GFM 表の区切り行 (``|---|:--:|``)。前後のパイプを要する。抽出・取得・分割が
#: 同じ規則で表を見分けるための SSOT (不変則 #14(a))。
GFM_TABLE_SEP_LINE_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$", re.MULTILINE)

_GFM_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")


def gfm_table_lines(rows: list[list[str]]) -> list[str]:
    """セルの文字列の行列を GFM テーブルにする (先頭行をヘッダとみなす)。

    空行は捨てる。全行が空なら空リスト。セル内の改行・連続空白は 1 つの空白に畳む
    (改行が残ると GFM の行が割れ、続きがヘッダ無しの本文になる)。
    """
    rows = [[" ".join(cell.split()).replace("|", r"\|") for cell in row] for row in rows]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("|" + "---|" * width)
    lines.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return lines


def split_gfm_row(line: str) -> list[str]:
    r"""GFM 表の 1 行をセルに分ける (前後のパイプを除く。``\|`` はセル内の文字)。"""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith(r"\|"):
        body = body[:-1]
    return [cell.strip() for cell in _GFM_CELL_SPLIT_RE.split(body)]
