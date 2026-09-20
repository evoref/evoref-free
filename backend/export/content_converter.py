"""Markdown → 構造化コンテンツ変換

外部ライブラリ不使用（正規表現ベース）。
対応構造: heading, paragraph, code block, table, list (ordered/unordered),
          blockquote, horizontal rule, inline formatting (bold/italic/code)
"""

from __future__ import annotations

import re
from typing import Any

from backend.export.base import ContentBlock, ExportContent
from backend.log_config import get_logger

logger = get_logger("export.content_converter")

# パターン定義
_RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_RE_CODE_FENCE = re.compile(r"^```(\w*)$")
_RE_HR = re.compile(r"^(?:---|\*\*\*|___)\s*$")
_RE_TABLE_ROW = re.compile(r"^\|(.+)\|$")
_RE_TABLE_SEP = re.compile(r"^\|[\s\-:|]+\|$")
_RE_UL = re.compile(r"^(\s*)[-*+]\s+(.+)$")
_RE_OL = re.compile(r"^(\s*)\d+[.)]\s+(.+)$")
_RE_QUOTE = re.compile(r"^>\s?(.*)")

#: 入れ子リストの打ち切り段数 (0 始まりで 0-2 の 3 段まで)。
_MAX_LIST_DEPTH = 2
#: 行全体が 1 個の画像記法だけのとき (f_11 §2.1)。段落の途中に混ざった
#: インライン画像はブロック化せず paragraph の文字列として残す。
_RE_IMAGE_ONLY = re.compile(r"^!\[([^\]]*)\]\(\s*<?([^)>]+?)>?\s*\)$")


def _recover_shapes(text: str) -> list[dict] | None:
    """言語指定を落とした図形 DSL を拾い直す。図形でなければ ``None``。"""
    from backend.export.shapes import looks_like_shapes_payload

    return looks_like_shapes_payload(text)


class ContentConverter:
    """Markdown テキストを ContentBlock リストに変換する"""

    @staticmethod
    def _parse_code_fence(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """``` フェンスコードブロックを 1 件パース。マッチしなければ ``(None, i)``。"""
        m = _RE_CODE_FENCE.match(lines[i])
        if not m:
            return None, i
        lang = m.group(1)
        code_lines: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            code_lines.append(lines[i])
            i += 1
        i += 1  # 閉じ ``` をスキップ
        source = "\n".join(code_lines)
        if lang.strip().lower() == "shapes":
            # 図形 DSL (f_11 §4.2)。Markdown に図形の記法が無いため、
            # 言語識別子 shapes のフェンスを図形ブロックとして扱う。
            from backend.export.shapes import parse_shapes_source

            return ContentBlock(
                type="shapes", content="", shapes=parse_shapes_source(source),
            ), i
        # 言語指定を落として JSON だけ吐いた場合の拾い直し (f_11 §4.2)。
        recovered = _recover_shapes(source)
        if recovered is not None:
            return ContentBlock(type="shapes", content="", shapes=recovered), i
        return ContentBlock(
            type="code",
            content=source,
            language=lang,
        ), i

    @staticmethod
    def _parse_heading(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """`# 見出し` をパース。マッチしなければ ``(None, i)``。"""
        m = _RE_HEADING.match(lines[i])
        if not m:
            return None, i
        return ContentBlock(
            type="heading",
            content=m.group(2).strip(),
            level=len(m.group(1)),
        ), i + 1

    @staticmethod
    def _parse_hr(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """水平線をパース。"""
        if not _RE_HR.match(lines[i]):
            return None, i
        return ContentBlock(type="hr", content=""), i + 1

    @staticmethod
    def _parse_table(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """`|...|...|` テーブルをパース。"""
        if not _RE_TABLE_ROW.match(lines[i]):
            return None, i
        table_rows: list[list[str]] = []
        while i < len(lines) and _RE_TABLE_ROW.match(lines[i]):
            if _RE_TABLE_SEP.match(lines[i]):
                i += 1
                continue
            cells = [
                c.strip() for c in lines[i].strip().strip("|").split("|")
            ]
            table_rows.append(cells)
            i += 1
        return ContentBlock(type="table", content="", rows=table_rows), i

    @staticmethod
    def _parse_quote(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """`>` 引用をパース。"""
        if not _RE_QUOTE.match(lines[i]):
            return None, i
        quote_lines: list[str] = []
        while i < len(lines):
            qm = _RE_QUOTE.match(lines[i])
            if qm:
                quote_lines.append(qm.group(1))
                i += 1
            else:
                break
        return ContentBlock(
            type="quote", content="\n".join(quote_lines),
        ), i

    @staticmethod
    def _match_list_item(line: str) -> tuple[int, bool, str] | None:
        """行がリスト項目なら ``(インデント幅, ordered, text)`` を返す。

        タブは 4 幅として展開する。2 / 3 / 4 スペースとタブのどれでも、
        後続の比較 (:meth:`_parse_list` のスタック) だけで段が決まる。
        """
        m = _RE_OL.match(line)
        if m:
            return len(m.group(1).expandtabs(4)), True, m.group(2)
        m = _RE_UL.match(line)
        if m:
            return len(m.group(1).expandtabs(4)), False, m.group(2)
        return None

    @staticmethod
    def _continues_after_blank(
        stack: list[int], candidate: tuple[int, bool, str], block_ordered: bool,
    ) -> bool:
        """空行の次の行が、いま組んでいるリストの続きとみなせるか。

        ``stack`` は変更しない (判定用のシミュレーション)。深い段の子として
        続くなら常に継続、0 段目に戻るなら種別が揃っているときだけ継続する。
        """
        indent, ordered, _text = candidate
        sim = list(stack)
        while sim and indent < sim[-1]:
            sim.pop()
        pushes = not sim or indent > sim[-1]
        depth = len(sim) - 1 + (1 if pushes else 0)
        if depth > 0:
            return True
        return ordered == block_ordered

    @classmethod
    def _is_item_continuation(cls, line: str, stack: list[int]) -> bool:
        """リスト記号の無い行が、直前の項目の続き (説明の段落) か。

        リストの 0 段目の記号より深く字下げされた、他の構造要素の開始でない行。
        LLM は「1. 題名」の直下に字下げした説明文を書くことがあり、ここで
        リストを切ると番号付きが 1 項目ずつの別リストに割れる (実機: .txt /
        .html で番号が毎回 1 から、.pptx で説明が項目より上の段に出た)。
        字下げの無い行は従来どおりリストの終わり。
        """
        if not stack or not line.strip():
            return False
        indent = len(line[: len(line) - len(line.lstrip())].expandtabs(4))
        return indent > stack[0] and not cls._is_block_start(line.strip())

    @classmethod
    def _parse_list(
        cls, lines: list[str], i: int, warn_state: list[bool],
    ) -> tuple[ContentBlock | None, int]:
        """``-`` / ``*`` / ``+`` / ``1.`` 混在の入れ子リストを 1 ブロックへパース。

        階層はインデント幅のスタックで決める (CommonMark のゆるい規則と同じ):
        直前の項目より深ければ 1 段下がり、同じ幅なら同じ段、浅くなればその
        幅以下の段まで戻る。以前は箇条書きと番号付きが別パーサだったため、
        「番号付きの下に箇条書き」が 3 ブロックに分断され、番号が振り直されて
        いた。項目間の空行 1 つは、次の非空行がリスト項目ならリストの続きと
        みなす (LLM は入れ子の前後に空行を入れることが多い)。
        """
        first = cls._match_list_item(lines[i])
        if first is None:
            return None, i
        items: list[str] = []
        item_levels: list[int] = []
        item_ordered: list[bool] = []
        block_ordered = first[1]
        stack: list[int] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                nxt_match = cls._match_list_item(nxt)
                # 空行を挟んでも良いのは、次の項目が入れ子の子 (深い段) か、
                # 0 段目どうしで種別 (ordered) が同じ「ゆるいリスト」の続きの
                # ときだけ。マーカーが変わる 0 段目は別リストとして終える
                # (CommonMark と同じ扱い。番号付き 3 個 → 空行 → 箇条書き 3 個
                # のような**独立した平らなリスト**を 1 ブロックへ誤って
                # 溶接しないため)。
                if nxt_match is not None and cls._continues_after_blank(
                    stack, nxt_match, block_ordered,
                ):
                    i += 1
                    continue
                if nxt_match is None and cls._is_item_continuation(nxt, stack):
                    i += 1
                    continue
                break
            matched = cls._match_list_item(line)
            if matched is None:
                if cls._is_item_continuation(line, stack):
                    # 項目テキストの中の改行として持つ (paragraph の複数行と同じ規約)。
                    items[-1] += "\n" + line.strip()
                    i += 1
                    continue
                break
            indent, ordered, text = matched
            while stack and indent < stack[-1]:
                stack.pop()
            if not stack or indent > stack[-1]:
                stack.append(indent)
            depth = len(stack) - 1
            # 0 段目でマーカーの種別が変われば別のリスト (空行の有無に依らない。
            # 以前の「箇条書きの直後の番号付きは別ブロック」を保つ)。
            if depth == 0 and items and ordered != block_ordered:
                break
            if depth > _MAX_LIST_DEPTH:
                depth = _MAX_LIST_DEPTH
                if not warn_state[0]:
                    logger.warning(
                        "list nesting exceeds %d levels; deeper items are "
                        "folded to level %d",
                        _MAX_LIST_DEPTH + 1, _MAX_LIST_DEPTH,
                    )
                    warn_state[0] = True
            items.append(text)
            item_levels.append(depth)
            item_ordered.append(ordered)
            i += 1
        # 平らなリスト (入れ子を含まない) は並行配列を空にし、従来の
        # ContentBlock と完全に一致させる (f_11 入れ子対応、後方互換)。
        if all(lv == 0 for lv in item_levels) and all(
            o == block_ordered for o in item_ordered
        ):
            item_levels = []
            item_ordered = []
        return ContentBlock(
            type="list", content="", ordered=block_ordered, items=items,
            item_levels=item_levels, item_ordered=item_ordered,
        ), i

    @staticmethod
    def _parse_image(
        lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """行単独の ``![alt](src)`` をパース。マッチしなければ ``(None, i)``。"""
        m = _RE_IMAGE_ONLY.match(lines[i].strip())
        if not m:
            return None, i
        return ContentBlock(
            type="image", content=m.group(1).strip(), src=m.group(2).strip(),
        ), i + 1

    @staticmethod
    def _is_block_start(line: str) -> bool:
        """段落終了判定: 次の構造要素の開始行か？"""
        return bool(
            _RE_HEADING.match(line)
            or _RE_CODE_FENCE.match(line)
            or _RE_HR.match(line)
            or _RE_TABLE_ROW.match(line)
            or _RE_QUOTE.match(line)
            or _RE_UL.match(line)
            or _RE_OL.match(line)
            or _RE_IMAGE_ONLY.match(line.strip())
        )

    @classmethod
    def _parse_paragraph(
        cls, lines: list[str], i: int,
    ) -> tuple[ContentBlock | None, int]:
        """段落 (連続行) をパース。次の構造要素開始行で停止。"""
        para_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            if cls._is_block_start(lines[i]) and para_lines:
                break
            para_lines.append(lines[i])
            i += 1
        if not para_lines:
            return None, i
        text = "\n".join(para_lines)
        # フェンスごと落ちた図形 DSL を拾い直す (f_11 §4.2)。
        recovered = _recover_shapes(text)
        if recovered is not None:
            return ContentBlock(type="shapes", content="", shapes=recovered), i
        return ContentBlock(type="paragraph", content=text), i

    def convert(self, markdown: str) -> list[ContentBlock]:
        """Markdown → ContentBlock リスト"""
        lines = markdown.split("\n")
        blocks: list[ContentBlock] = []
        # 深さ打ち切りの WARNING は文書につき 1 回だけ (可変な 1 要素リストで
        # convert() 呼び出しごとに独立させる)。
        list_depth_warned = [False]
        parsers = (
            self._parse_code_fence,
            self._parse_heading,
            self._parse_hr,
            self._parse_table,
            self._parse_quote,
            self._parse_image,
            lambda lns, idx: self._parse_list(lns, idx, list_depth_warned),
        )
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue

            matched = False
            for parser in parsers:
                block, new_i = parser(lines, i)
                if block is not None:
                    blocks.append(block)
                    i = new_i
                    matched = True
                    break
            if matched:
                continue

            block, i = self._parse_paragraph(lines, i)
            if block is not None:
                blocks.append(block)
        return blocks

    @staticmethod
    def from_markdown(markdown: str, title: str = "", **metadata: Any) -> ExportContent:
        """Markdown テキストから ExportContent を一括生成"""
        converter = ContentConverter()
        blocks = converter.convert(markdown)

        # タイトル自動検出: 最初の heading があればそれを使用
        auto_title = title
        if not auto_title:
            for block in blocks:
                if block.type == "heading":
                    auto_title = block.content
                    break

        return ExportContent(
            title=auto_title,
            blocks=blocks,
            metadata=dict(metadata),
            raw_markdown=markdown,
        )

    @staticmethod
    def from_data(
        data: list[dict] | list[list],
        title: str = "",
        **metadata: Any,
    ) -> ExportContent:
        """構造化データから ExportContent を生成（CSV/XLSX/JSON 用）"""
        return ExportContent(
            title=title,
            blocks=[],
            metadata=dict(metadata),
            raw_data=data,
        )
