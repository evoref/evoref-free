"""ファイル書出しの基本型定義

ExportContent, WriteResult, FileWriter Protocol, ExportError を提供する。
各形式の Writer はこのモジュールの Protocol を実装する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol


@dataclass
class ContentBlock:
    """コンテンツの構成要素

    種別の語彙は docs/f_11_file_export.md §2 が SSOT。増やすときは §2.2 の
    表にある場所を全部直す (未知 type はどの Writer でも例外にならず黙って
    落ちるため、抜けをテストが検知できない)。
    """
    # "heading", "paragraph", "code", "table", "list", "quote", "hr",
    # "image", "shapes"
    type: str
    content: str    # テキスト内容 (image では alt テキスト)
    level: int = 0  # heading レベル (1-6)
    language: str = ""  # code block の言語
    rows: list[list[str]] = field(default_factory=list)  # table の行データ
    ordered: bool = False  # list の順序付き/なし (= 先頭の 0 段目の項目の種別)
    items: list[str] = field(default_factory=list)  # list の項目 (階層フラット)
    #: ``items`` と並行する各項目の段 (0 始まり、0-2)。空なら全項目が 0 段目
    #: (入れ子を含まない従来どおりの ``ContentBlock``)。``items`` と長さが
    #: 合わなければ壊れた入力として空扱いする。
    item_levels: list[int] = field(default_factory=list)
    #: ``items`` と並行する各項目の順序付き/なし。空なら全項目が ``ordered``
    #: に従う。``items`` と長さが合わなければ壊れた入力として空扱いする。
    item_ordered: list[bool] = field(default_factory=list)
    src: str = ""  # image の所在 (絶対パス、または出力先からの相対パス)
    shapes: list[dict[str, Any]] = field(default_factory=list)  # shapes の図形定義

    def iter_items(self) -> Iterator[tuple[str, int, bool]]:
        """``(text, level, ordered)`` を項目順に列挙する。

        ``item_levels`` / ``item_ordered`` が ``items`` と同じ長さのときだけ
        使い、そうでなければ全項目を 0 段目・``ordered`` 従属として扱う
        (壊れた入力で例外にしない)。
        """
        levels = self.item_levels if len(self.item_levels) == len(self.items) else None
        ordered_flags = (
            self.item_ordered if len(self.item_ordered) == len(self.items) else None
        )
        for i, text in enumerate(self.items):
            level = levels[i] if levels is not None else 0
            ordered = ordered_flags[i] if ordered_flags is not None else self.ordered
            yield text, level, ordered

    def has_nested_items(self) -> bool:
        """1 つでも 0 段目より深い項目を持つか。"""
        return any(level > 0 for _, level, _ in self.iter_items())


@dataclass
class ListItemNode:
    """入れ子リストの 1 項目 (Writer 共通のツリー表現)。

    ``ContentBlock`` の並行配列 (``items`` / ``item_levels`` / ``item_ordered``)
    は文字列の列として扱う読み手を壊さないためにフラットなまま持つが、
    HTML / LaTeX / plaintext / ODF のように入れ子構造を直接描く Writer は
    ツリーの方が書きやすいので :func:`build_item_tree` で変換する。
    """
    text: str
    ordered: bool
    children: list["ListItemNode"] = field(default_factory=list)


def build_item_tree(block: ContentBlock) -> list[ListItemNode]:
    """``ContentBlock`` の並行配列から入れ子ツリーを組む。

    段が親より 2 段以上飛ぶ壊れた入力でも、直前に置ける親の 1 段下へ畳んで
    例外にしない。
    """
    roots: list[ListItemNode] = []
    stack: list[ListItemNode] = []
    for text, level, ordered in block.iter_items():
        level = min(level, len(stack))
        node = ListItemNode(text=text, ordered=ordered)
        if level == 0:
            roots.append(node)
        else:
            stack[level - 1].children.append(node)
        stack = stack[:level] + [node]
    return roots


def sibling_runs(nodes: list[ListItemNode]) -> list[list[ListItemNode]]:
    """兄弟を、同じ種別 (``ordered``) が続く区間ごとに区切る。

    同じ段に箇条書きと番号付きが混ざったとき、先頭の種別で全体を包むと
    後ろの項目の番号 (または記号) が失われる。種別が変わる所で別のリスト
    として描けるように区切って返す (CommonMark もマーカーの種別が変われば
    別リスト)。
    """
    runs: list[list[ListItemNode]] = []
    for node in nodes:
        if runs and runs[-1][0].ordered == node.ordered:
            runs[-1].append(node)
        else:
            runs.append([node])
    return runs


@dataclass
class ExportContent:
    """書出し対象コンテンツ"""
    title: str = ""
    blocks: list[ContentBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata 例: {"author": "evoref", "date": "2026-03-23", "language": "ja"}
    raw_markdown: str = ""  # 元の Markdown テキスト（フォールバック用）
    raw_data: Any = None    # 構造化データ（csv/xlsx/json 用: list[dict] 等）


#: 数値にしてよい表記 (f_11 §3.1)。任意の ``-``、先頭 0 の無い整数部、任意の小数部だけ。
#: ``int()`` / ``float()`` は ``0012`` / ``+81`` / ``1_000`` / ``NaN`` / ``1e400`` も受けてしまう。
_PLAIN_DECIMAL_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
#: 表計算 (IEEE 754 倍精度) が保てる 10 進の有効桁。超えると末尾が化ける (カード番号等)。
_MAX_NUMERIC_DIGITS = 15


def is_plain_decimal(text: str) -> bool:
    """``text`` が数値として書いてよい素の 10 進表記か (f_11 §3.1)。"""
    return (
        _PLAIN_DECIMAL_RE.fullmatch(text) is not None
        and sum(c.isdigit() for c in text) <= _MAX_NUMERIC_DIGITS
    )


def coerce_cell_value(value: object) -> object:
    """表セルの文字列を数値へ型推定する (xlsx / ods 共通)。

    "1200" を文字列のままセルへ入れると表計算側で集計できない。素の 10 進
    表記 (:func:`is_plain_decimal`) だけを数値にし、それ以外はそのまま返す。
    """
    if not isinstance(value, str) or not is_plain_decimal(value):
        return value
    return float(value) if "." in value else int(value)


#: 同一シート内の集計だけに使える関数 (外部参照・文字列・ハイパーリンクを作れない)。
_SAFE_FORMULA_FUNCS = frozenset({
    "SUM", "AVERAGE", "MIN", "MAX", "COUNT", "ROUND", "ABS", "PRODUCT",
})
_CELL = r"\$?[A-Za-z]{1,3}\$?[1-9][0-9]{0,6}"
_COL = r"\$?[A-Za-z]{1,3}"
_OPERAND_RE = re.compile(
    rf"(?P<func>[A-Za-z]+)\("  # 関数名 + 開き括弧
    rf"|(?P<ref>{_CELL}(?::{_CELL})?|{_COL}:{_COL})"
    r"|(?P<num>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)%?)"
    r"|(?P<lp>\()"
    r"|(?P<sign>[-+])",
    re.ASCII,
)
_OPERATOR_RE = re.compile(r"(?P<bin>[-+*/^])|(?P<comma>,)|(?P<rp>\))|(?P<pct>%)", re.ASCII)


def is_safe_formula(text: str) -> bool:
    """``=`` 始まりの文字列が、同一シートの算術・集計だけの式かを返す (f_11 §3.1)。

    許すのは数値・セル参照 (範囲・列範囲可)・四則と ``^``・括弧・``_SAFE_FORMULA_FUNCS`` の
    関数だけ。オペランドと演算子が交互に並ぶ文法も検査する (Excel が修復を求める式を作らない)。
    ``!`` ``[`` ``"`` ``&`` ``@`` や未知の関数名は 1 つでもあれば偽 (HYPERLINK / 外部参照を弾く)。
    """
    if not text.startswith("=") or len(text) > 255:
        return False
    pos, depth, n = 1, 0, len(text)
    expect_operand = True
    while True:
        while pos < n and text[pos] == " ":
            pos += 1
        if pos >= n:
            return depth == 0 and not expect_operand
        if expect_operand:
            m = _OPERAND_RE.match(text, pos)
            if m is None:
                return False
            if m.group("func") is not None:
                if m.group("func").upper() not in _SAFE_FORMULA_FUNCS or depth >= 20:
                    return False
                depth += 1
            elif m.group("lp") is not None:
                if depth >= 20:
                    return False
                depth += 1
            elif m.group("ref") is not None or m.group("num") is not None:
                expect_operand = False
            # sign / func / lp は次もオペランド
        else:
            m = _OPERATOR_RE.match(text, pos)
            if m is None:
                return False
            if m.group("bin") is not None or (m.group("comma") is not None and depth > 0):
                expect_operand = True
            elif m.group("comma") is not None:
                return False
            elif m.group("rp") is not None:
                depth -= 1
                if depth < 0:
                    return False
        pos = m.end()


def assign_xlsx_cell(cell: Any, value: object, *, allow_safe_formula: bool = False) -> None:
    """openpyxl のセルへ値を入れる。``=`` 始まりの文字列を数式にしない (f_11 §3.1)。

    openpyxl は ``=`` で始まる str を数式セルにするので、LLM の出力や差し込み値が
    ``=HYPERLINK(...)`` なら開いた人の Excel で評価される。文字列セルへ戻し、Excel が
    ``'`` 付きで入力されたセルに付けるのと同じ ``quotePrefix`` を立てる。
    ``allow_safe_formula`` のときだけ :func:`is_safe_formula` を通った式を数式のまま残す
    (表の「合計を数式で」。帳票の穴埋めなど差し込み値は常に文字列)。
    """
    cell.value = value
    if isinstance(value, str) and value.startswith("="):
        if allow_safe_formula and is_safe_formula(value):
            return
        cell.data_type = "s"
        cell.quotePrefix = True


@dataclass
class WriteResult:
    """書出し結果"""
    path: Path | None = None
    data: bytes | None = None  # バイトデータ出力（API レスポンス用）
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class FileWriter(Protocol):
    """ファイル書出しプロトコル — 各形式はこれを実装"""

    @property
    def extensions(self) -> frozenset[str]:
        """対応する拡張子の集合（例: frozenset({".txt", ".md"})）"""
        ...

    @property
    def requires(self) -> list[str]:
        """必要な pip パッケージ名"""
        ...

    def write(self, content: ExportContent, path: Path) -> WriteResult:
        """ファイルに書出し"""
        ...

    def write_to_bytes(self, content: ExportContent, ext: str) -> bytes:
        """バイトデータとして書出し（API ダウンロード用）"""
        ...

    def is_available(self) -> bool:
        """必要ライブラリがインストール済みか"""
        ...


class ExportError(Exception):
    """書出しエラー"""
    def __init__(self, code: str, message: str = ""):
        self.code = code  # unsupported_format, missing_library, conversion_error, etc.
        super().__init__(message or code)
