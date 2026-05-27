"""ファイル分割読み込み (ストリーミング)

大ファイルを一気に :func:`pathlib.Path.read_text` で読まず、strategy 関数を
通じて :class:`Chunk` 単位で yield する。RAG ドキュメントインジェスト時の
プロセスメモリ削減を目的とする。

公開要素:

- :class:`Chunk` — 1 単位の分割結果 (text または data)
- :class:`ReadOptions` — encoding / メモリ cap
- :class:`ChunkedReader` — ファイルから iterator として Chunk を yield
- :data:`ChunkStrategy` — strategy 関数の型エイリアス
- :func:`csv_row_strategy` — CSV をヘッダー付き行単位に分割

将来追加予定の strategy: ``markdown_heading`` / ``paragraph`` /
``fixed_chars`` / ``fixed_bytes``。
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

__all__ = [
    "Chunk",
    "ChunkStrategy",
    "ChunkedReader",
    "ReadOptions",
    "csv_row_strategy",
]


@dataclass(frozen=True)
class Chunk:
    """ChunkedReader が yield する 1 単位。

    ``text`` / ``data`` のうち strategy 種別に応じて片方のみが値を持つ
    (テキスト系 strategy は ``text``、バイナリ系は ``data``)。

    ``byte_offset`` は将来の resumability (``iter_from(N)``) のための再開キー
    枠。CSV など行可変長フォーマットでは 0 で埋めて差し支えない。
    ``fixed_bytes_strategy`` 等の固定長 strategy で利用する。
    """

    index: int
    text: str | None = None
    data: bytes | None = None
    byte_offset: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadOptions:
    """ChunkedReader の動作オプション。"""

    encoding: str = "utf-8"
    errors: str = "replace"
    max_chunks: int | None = None        # 何チャンク yield したら停止するか
    max_total_bytes: int | None = None   # text/data の累積バイト数で停止


#: strategy 関数の型エイリアス。
#: バイナリストリームと ReadOptions を受け取り Chunk を yield する generator。
ChunkStrategy = Callable[[BinaryIO, ReadOptions], Iterator[Chunk]]


class ChunkedReader:
    """ファイルをストリームで読み、strategy 関数で Chunk 単位に分割する。

    Usage:
        >>> reader = ChunkedReader(file_path, strategy=csv_row_strategy)
        >>> for chunk in reader:
        ...     process(chunk.text)

    ``options.max_chunks`` / ``options.max_total_bytes`` で上限を設ければ、
    巨大ファイルでもメモリ消費を一定範囲に抑えられる。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        strategy: ChunkStrategy,
        options: ReadOptions = ReadOptions(),
    ) -> None:
        self._path = Path(path)
        self._strategy = strategy
        self._options = options

    def __iter__(self) -> Iterator[Chunk]:
        opts = self._options
        max_chunks = opts.max_chunks
        max_total = opts.max_total_bytes
        emitted = 0
        total_bytes = 0
        with self._path.open("rb") as stream:
            for chunk in self._strategy(stream, opts):
                yield chunk
                emitted += 1
                if chunk.text is not None:
                    total_bytes += len(chunk.text.encode(opts.encoding, errors=opts.errors))
                elif chunk.data is not None:
                    total_bytes += len(chunk.data)
                if max_chunks is not None and emitted >= max_chunks:
                    return
                if max_total is not None and total_bytes >= max_total:
                    return


# ─────────────────────────────────────────────────────────────────────
# 同梱 strategy: CSV
# ─────────────────────────────────────────────────────────────────────


def csv_row_strategy(stream: BinaryIO, options: ReadOptions) -> Iterator[Chunk]:
    """CSV をヘッダー付き行単位の :class:`Chunk` に分割する strategy。

    挙動 (``backend.free.rag.text_extractor._parse_csv_text_to_chunks`` 互換):

    - 完全空ファイル → 何も yield しない
    - ヘッダー行のみ (データ行 0 件) で、ヘッダーに非空白セルが 1 つでも
      あれば ``Chunk(index=0, text=header_line)`` を 1 件 yield する
    - 通常 (2 行以上) → 各データ行を ``"header_line\\ndata_line"`` 形式で
      1 件ずつ yield する
    - 全セルが空白だけのデータ行は **スキップ**
    - フィールドのカンマ join は素朴な ``,`` 結合 (``csv.writer`` の
      引用符付けは行わない。元実装と同等)
    """
    text_stream = io.TextIOWrapper(
        stream,
        encoding=options.encoding,
        errors=options.errors,
        newline="",  # csv モジュール推奨: \r\n 自動変換を抑止
    )
    reader = csv.reader(text_stream)
    try:
        header_row = next(reader)
    except StopIteration:
        return  # 完全空ファイル
    header_line = ",".join(header_row)
    has_data_rows = False
    chunk_index = 0
    for row in reader:
        has_data_rows = True
        if not any(cell.strip() for cell in row):
            continue
        data_line = ",".join(row)
        yield Chunk(
            index=chunk_index,
            text=f"{header_line}\n{data_line}",
        )
        chunk_index += 1
    if not has_data_rows and any(cell.strip() for cell in header_row):
        yield Chunk(index=0, text=header_line)
