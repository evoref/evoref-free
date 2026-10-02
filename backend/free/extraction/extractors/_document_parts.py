"""文書抽出器が共有する表記 — 画像・図形の印と GFM 表

設計書: [docs/f_11_file_export.md](../../../../docs/f_11_file_export.md) §3.3 / §5.1。
"""

from __future__ import annotations

#: 本文に画像が在ることを示す印。抽出はテキストなので画像そのものは読み戻せない
#: (f_11 §5.1) が、画像だけの文書を「中身が無い」と読むと追記が止まる (§3.3)。
IMAGE_MARK = "[image]"
#: 文字を持たない図形 (線・塗りだけの四角形など) が在ることを示す印。
SHAPE_MARK = "[shape]"


def gfm_table_lines(rows: list[list[str]]) -> list[str]:
    """セルの文字列の行列を GFM テーブルにする (先頭行をヘッダとみなす)。

    空行は捨てる。全行が空なら空リスト。
    """
    rows = [[cell.strip().replace("|", r"\|") for cell in row] for row in rows]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("|" + "---|" * width)
    lines.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return lines
