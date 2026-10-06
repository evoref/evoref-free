"""テキストチャンク分割"""

import re
from dataclasses import dataclass
from itertools import zip_longest

import tiktoken

from backend.free.extraction.extractors._document_parts import (
    GFM_TABLE_SEP_LINE_RE,
    split_gfm_row,
)
from backend.log_config import get_logger

logger = get_logger("rag.chunker")

# cl100k_base エンコーディング（日英混在テキストに対応）
_encoding = tiktoken.get_encoding("cl100k_base")

# 文境界パターン（優先順位順）
SENTENCE_BOUNDARIES = [
    re.compile(r"\n\n+"),                    # 段落区切り
    re.compile(r"(?<=[。！？…])\s*"),         # 日本語文末
    re.compile(r"(?<=[.!?])\s+"),            # 英語文末
    re.compile(r"\n"),                        # 改行
]

#: 文の切れ目。全角の文末 (。！？) は無条件、ASCII の文末 (.!?) は **後ろが
#: 空白か行末のときだけ**。空白の無いピリオドで切ると「Step 5.9」「§6.1」
#: 「rag.pseudo_query.enabled」が「5. 9」「§6. 1」「rag. pseudo_query. enabled」に
#: 壊れ、技術文書の固有語が転置索引でも埋め込みでも当たらなくなる
#: (2026-09-12 (b) ライブ監査、f_01 §3.1.2)。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])(?:\s+|$)")

#: markdown の見出し行。チャンクの **硬い境界** (見出しの前で閉じる、f_01 §3.1.3)。
_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S", re.MULTILINE)


#: 見出し行の段と題 (表のチャンクに見出しパスを添えるため)。
_HEADING_PARTS_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")

#: 表の直前の行を題 (caption) とみなす最大長。長い行は前置きの本文として残す。
_MAX_CAPTION_CHARS = 120

#: 見出しパスの区切り。
_HEADING_PATH_SEP = " > "

#: 表の連続する行を 1 チャンクに束ねる目安 (前置き込みのトークン数、``max_chunk`` を
#: 超えない)。1 行 1 チャンクだと 60 行の表が 60 チャンクになり、疑似クエリの生成
#: (1 チャンク約 3.5 秒) と索引が行数に比例して膨らむ (f_01 §3.1.5)。
_TABLE_PACK_TOKENS = 300


@dataclass(frozen=True)
class ChunkPiece:
    """チャンク 1 つ。表のチャンクは文書内の表の番号とデータ行の範囲を持つ (f_01 §3.1.5)。"""

    text: str
    #: 文書の中で何番目の表か (0 始まり)。表のチャンクでなければ ``None``。
    table_index: int | None = None
    #: 含むデータ行の範囲 ``(最初, 最後)`` (0 始まり・両端を含む)。ヘッダだけの表は ``None``。
    table_rows: tuple[int, int] | None = None
    #: 本文の先頭にある前置き (見出しパス・caption・ヘッダ行・区切り行のうち付けたもの、
    #: 末尾の改行込み) の文字数。近似重複は前置きが同じチャンクの間でだけ、残りの行で比べる。
    table_prefix_chars: int = 0
    #: 表のヘッダ行。「列名: 値」に開いたチャンクでも列名を比較から除くために持つ。
    table_header: str = ""


@dataclass(frozen=True)
class _TableBlock:
    """本文から切り出した GFM 表 1 つ (f_01 §3.1)。"""

    index: int
    heading_path: str
    caption: str
    header: str
    separator: str
    rows: tuple[str, ...]
    #: caption と表の原文 (表として扱えないときに本文として分割し直す)。
    raw: str


def _is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def _split_tables(text: str) -> list[str | _TableBlock]:
    """本文を「表以外の区間」と GFM 表に分ける (純粋関数)。

    表はヘッダ行 + 区切り行 (``GFM_TABLE_SEP_LINE_RE``) + ``|`` で始まる行の続き。
    表の直前に空行を挟まず置かれた短い行は表の題 (caption) として表に持たせる。
    コードフェンスの中は表とみなさない。見出しパスは表より前の見出し行から作る。
    """
    lines = text.split("\n")
    out: list[str | _TableBlock] = []
    buf: list[str] = []
    path: list[tuple[int, str]] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            heading = _HEADING_PARTS_RE.match(line)
            if heading:
                level = len(heading.group(1))
                path = [p for p in path if p[0] < level] + [(level, heading.group(2).strip())]
            elif (
                _is_table_row(line)
                and i + 1 < len(lines)
                and GFM_TABLE_SEP_LINE_RE.match(lines[i + 1])
            ):
                end = i + 2
                while end < len(lines) and _is_table_row(lines[end]):
                    end += 1
                caption = ""
                if (
                    buf and buf[-1].strip()
                    and len(buf[-1].strip()) <= _MAX_CAPTION_CHARS
                    and not _HEADING_PARTS_RE.match(buf[-1])
                    and not buf[-1].lstrip().startswith("```")  # フェンスを壊さない
                ):
                    caption = buf.pop().strip()
                # 表の直前の見出しは表のチャンクの見出しパスに載るので、前の区間の末尾に残さない
                while buf and (not buf[-1].strip() or _HEADING_PARTS_RE.match(buf[-1])):
                    buf.pop()
                if "\n".join(buf).strip():
                    out.append("\n".join(buf))
                buf = []
                out.append(_TableBlock(
                    index=sum(isinstance(block, _TableBlock) for block in out),
                    heading_path=_HEADING_PATH_SEP.join(title for _, title in path),
                    caption=caption,
                    header=line.strip(),
                    separator=lines[i + 1].strip(),
                    rows=tuple(
                        row.strip() for row in lines[i + 2:end] if any(split_gfm_row(row))
                    ),
                    raw="\n".join(([caption] if caption else []) + lines[i:end]),
                ))
                i = end
                continue
        buf.append(line)
        i += 1
    if "\n".join(buf).strip():
        out.append("\n".join(buf))
    return out


def _split_on_headings(text: str) -> list[str]:
    """見出し行の直前で本文を区切る (純粋関数)。見出しの無い本文はそのまま 1 区間。"""
    positions = [m.start() for m in _HEADING_LINE_RE.finditer(text)]
    if not positions:
        return [text]
    bounds = ([0] if positions[0] > 0 else []) + positions + [len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:]) if text[a:b].strip()]


class SemanticChunker:
    """セマンティック境界ベースのチャンク分割"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk: int = 64,
        max_chunk: int = 512,
        strategy: str = "semantic",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.strategy = strategy

    def chunk(self, text: str) -> list[str]:
        """テキストをチャンクに分割"""
        return [piece.text for piece in self.chunk_pieces(text)]

    def chunk_pieces(self, text: str) -> list[ChunkPiece]:
        """テキストをチャンクに分割し、表のチャンクには表の番号と行範囲を付ける。"""
        if not text.strip():
            logger.debug("chunk: empty text, returning []")
            return []

        logger.debug(
            "chunk: strategy=%s, text_length=%d, chunk_size=%d, overlap=%d",
            self.strategy, len(text), self.chunk_size, self.chunk_overlap,
        )

        result: list[ChunkPiece] = []
        if self.strategy == "fixed":
            result = [ChunkPiece(t) for t in self._fixed_chunk(text)]
        else:
            # 表は硬い境界で、連続する行を列名付きのチャンクに束ねる (f_01 §3.1.5)。
            # 文分割器に通すと表が途中で切れ、続きの片がヘッダを失う。
            for block in _split_tables(text):
                if isinstance(block, _TableBlock):
                    result.extend(self._table_chunks(block))
                else:
                    result.extend(ChunkPiece(t) for t in self._prose_chunks(block))

        logger.debug(
            "chunk: produced %d chunks, sizes=[%s]",
            len(result),
            ", ".join(str(len(c.text)) for c in result[:10])
            + ("..." if len(result) > 10 else ""),
        )
        return result

    def _prose_chunks(self, text: str) -> list[str]:
        """表以外の区間を見出しの前で閉じながら分割する。"""
        # 見出しの前でチャンクを閉じる (chunker v3、f_01 §3.1.3)。節の途中から
        # 始まるチャンクが答えを主語抜きで持つのを避ける。
        result: list[str] = []
        carry = ""
        for section in _split_on_headings(text):
            # 短い節 (min_chunk 未満) は次の節と併せる — 見出しごとに閉じると
            # 細切れになり内容が薄くなる (実測: 1124 → 1666 チャンクで recall 低下)。
            merged = carry + section
            if self._estimate_tokens(merged) < self.min_chunk:
                carry = merged
                continue
            carry = ""
            result.extend(self._semantic_chunk(merged))
        if carry.strip():
            result.extend(self._semantic_chunk(carry))
        return result

    def _table_prefix(self, table: _TableBlock) -> list[str]:
        """表のチャンクに付ける前置き。``max_chunk`` の半分を超えるなら縮める。

        見出しパス + caption + ヘッダ行 + 区切り行 → (過大なら) ヘッダ行 + 区切り行
        → (それでも過大なら) 見出しパス + caption だけ → 無し。ヘッダ行が無いときは
        各行を「列名: 値」で開くので、列名は値の行に残る。前置きが予算を食い潰して
        1 行が数十片に割れるのを防ぐ。
        """
        limit = self.max_chunk // 2
        context = [part for part in (table.heading_path, table.caption) if part]
        for candidate in (
            [*context, table.header, table.separator],
            [table.header, table.separator],
            context,
        ):
            if candidate and self._estimate_tokens("\n".join(candidate)) <= limit:
                return candidate
        return []

    def _table_chunks(self, table: _TableBlock) -> list[ChunkPiece]:
        """表 1 つを「前置き + 連続する行」のチャンク列にする (f_01 §3.1.5)。

        CSV の「ヘッダ + 行」と同じ形。連続する行を前置き込みで
        ``_TABLE_PACK_TOKENS`` (``max_chunk`` 以下) まで束ねる。単独で ``max_chunk``
        を超える行だけ「列名: 値」の行に開いて分け、続きの片にも前置きを付ける。
        ヘッダだけの表が ``max_chunk`` を超える (HTML のレイアウト用の表で本文が
        1 セルに潰れた等) ときは本文として分割し直す。
        """
        prefix = self._table_prefix(table)
        has_header = bool(prefix) and prefix[-1] == table.separator
        prefix_chars = len("\n".join(prefix)) + 1 if prefix else 0

        def piece(text: str, rows: tuple[int, int] | None) -> ChunkPiece:
            return ChunkPiece(
                text, table.index, rows,
                table_prefix_chars=prefix_chars, table_header=table.header,
            )

        if not table.rows:
            text = "\n".join(prefix if has_header else [table.header, table.separator])
            if self._estimate_tokens(text) <= self.max_chunk:
                return [ChunkPiece(text, table.index, table_header=table.header)]
            return [ChunkPiece(t) for t in self._prose_chunks(table.raw)]

        target = min(_TABLE_PACK_TOKENS, self.max_chunk)
        prefix_tokens = self._estimate_tokens("\n".join(prefix)) if prefix else 0
        pieces: list[ChunkPiece] = []
        group: list[str] = []
        first = 0
        used = prefix_tokens

        def flush() -> None:
            if group:
                pieces.append(piece("\n".join([*prefix, *group]), (first, first + len(group) - 1)))
                group.clear()

        for position, row in enumerate(table.rows):
            cost = self._estimate_tokens(row) + 1
            if not has_header or prefix_tokens + cost > self.max_chunk:
                flush()
                used = prefix_tokens
                pieces.extend(
                    piece(text, (position, position))
                    for text in self._split_long_row(prefix, table.header, row)
                )
                continue
            if group and used + cost > target:
                flush()
                used = prefix_tokens
            if not group:
                first = position
            group.append(row)
            used += cost
        flush()
        return pieces

    def _split_long_row(self, prefix: list[str], header: str, row: str) -> list[str]:
        """長い行を「列名: 値」の行に開き、予算ごとに区切る (各片に前置きを付ける)。

        各片は ``max_chunk`` を超えない (前置きは ``_table_prefix`` が半分以下に抑える)。
        """
        prefix_text = "\n".join(prefix)
        budget = self.max_chunk - (self._estimate_tokens(prefix_text) + 1 if prefix else 0)
        lines: list[str] = []
        for name, value in zip_longest(split_gfm_row(header), split_gfm_row(row), fillvalue=""):
            if not value:
                continue
            label = f"{name}: " if name else ""
            if self._estimate_tokens(label) > budget // 2:
                label = ""  # 列名だけで予算の半分を食う列は列名を落とす
            line = f"{label}{value}"
            if self._estimate_tokens(line) + 1 <= budget:
                lines.append(line)
                continue
            # 1 セルだけで予算を超える: 文字で切り (トークン境界で切ると多バイト文字が
            # 割れる)、続きの片にも列名を付ける
            step = max(budget - self._estimate_tokens(label) - 2, 1)
            start = 0
            while start < len(value):
                end = min(len(value), start + step * 4)
                while end - start > 1:
                    used_tokens = self._estimate_tokens(value[start:end])
                    if used_tokens <= step:
                        break
                    end = start + max(1, (end - start) * step // used_tokens)
                piece = value[start:end].strip()
                if piece:
                    lines.append(f"{label}{piece}")
                start = end
        pieces: list[str] = []
        current: list[str] = []
        used = 0
        for line in lines:
            cost = self._estimate_tokens(line) + 1
            if current and used + cost > budget:
                pieces.append("\n".join([*prefix, *current]))
                current, used = [], 0
            current.append(line)
            used += cost
        if current:
            pieces.append("\n".join([*prefix, *current]))
        return pieces

    def _semantic_chunk(self, text: str) -> list[str]:
        """セマンティック分割: 文境界で分割し、トークン予算に収める"""
        # まず文に分割
        paragraphs = re.split(r"\n\n+", text)
        para_count = len([p for p in paragraphs if p.strip()])
        sentences = self._split_sentences(text)
        if not sentences:
            logger.debug("_semantic_chunk: no sentences extracted")
            return []
        avg_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        logger.debug(
            "_semantic_chunk: paragraphs=%d, sentences=%d, avg_sentence_len=%.1f",
            para_count, len(sentences), avg_len,
        )

        chunks = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            # 1文がmax_chunkを超える場合は固定長で分割
            if sent_tokens > self.max_chunk:
                logger.debug(
                    "_semantic_chunk: sentence exceeds max_chunk (%d > %d), "
                    "falling back to fixed chunking",
                    sent_tokens, self.max_chunk,
                )
                # 現在のバッファをフラッシュ
                if current_sentences:
                    chunks.append("".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                # 長い文を固定長で分割
                sub_chunks = self._fixed_chunk(sentence)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + sent_tokens > self.max_chunk and current_sentences:
                # バッファをフラッシュ
                chunks.append("".join(current_sentences))
                # オーバーラップ: 最後の文を引き継ぐ
                last = current_sentences[-1] if current_sentences else ""
                current_sentences = [last] if self._estimate_tokens(last) < self.chunk_overlap else []
                current_tokens = self._estimate_tokens("".join(current_sentences))

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # 残りをフラッシュ
        if current_sentences:
            chunk_text = "".join(current_sentences)
            if self._estimate_tokens(chunk_text) >= self.min_chunk or not chunks:
                chunks.append(chunk_text)
            elif chunks:
                # min_chunk未満なら最後のチャンクに結合
                chunks[-1] += chunk_text

        return [c.strip() for c in chunks if c.strip()]

    def _fixed_chunk(self, text: str) -> list[str]:
        """固定長分割（トークン単位）"""
        tokens = _encoding.encode(text)
        if not tokens:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = _encoding.decode(tokens[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)
            start += step

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """文境界で分割"""
        # 段落 → 文の2段階で分割
        paragraphs = re.split(r"\n\n+", text)
        sentences = []

        for para in paragraphs:
            if not para.strip():
                continue
            # 日本語・英語の文末で分割 (ASCII の文末は空白 / 行末が続くときだけ)
            parts = _SENTENCE_SPLIT_RE.split(para)
            for part in parts:
                if part.strip():
                    sentences.append(part.strip() + " ")

        return sentences

    def _estimate_tokens(self, text: str) -> int:
        """tiktoken による正確なトークン数計算"""
        if not text:
            return 0
        return max(1, len(_encoding.encode(text)))
