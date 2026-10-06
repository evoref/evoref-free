"""取得した表の数値集計を決定論で行う (docs/f_03 §4.2.2)。

2026-10-05 ライブ監査: 「sales.csv を読んで、月ごとの売上金額 (units×unit_price) の
合計を出して」で、9 行の CSV の月別合計 3 つがすべて誤っていた (正 310,700 /
363,880 / 320,400、答え 307,100 / 371,880 / 312,540)。読みは成功し、行の数値は
プロンプトに全部載っていた — 誤りはモデルの暗算そのもの。``calculate`` は式が
400 字までで、行の数値をモデルに写させる経路は表が大きくなると写し間違いと上限で
崩れる。

日付演算 (``date_intent``) と同じ立て付けにする: **モデルには集計のパラメータだけを
文法制約 JSON で取らせ** (列名の式・グループの列・集計の種類・絞り込み)、計算は
ここで全行に対して行う。モデルにコードは書かせない。式は列名・数・四則演算と括弧
だけを AST の許可リストで受ける。

このモジュールは純粋関数だけを持つ (表の解析・式の評価・集計・整形・答えの照合)。
パラメータを取る往復は ``agent/table_aggregate_intent.py`` が持つ。
"""

from __future__ import annotations

import ast
import csv
import io
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from backend.free.constants import READ_FILE_META_PREFIX
from backend.free.core.script_ranges import KANJI, KANJI_EXT_A, KANJI_MARKS, KATAKANA_WORD

__all__ = [
    "AGGREGATIONS",
    "DATE_PARTS",
    "DERIVES",
    "AggregateError",
    "AggregateResult",
    "AggregateSpec",
    "DerivedValue",
    "Table",
    "describe_spec",
    "find_table",
    "format_derived",
    "format_number",
    "mismatch_note",
    "parse_table",
    "result_block",
    "run_aggregate",
    "contradicted_derived",
    "contradicted_values",
]

#: 集計の種類 (``json_schemas.TableAggregation.agg`` と同じ語彙)。
AGGREGATIONS: tuple[str, ...] = ("sum", "mean", "min", "max", "count")
#: グループの列が日付のときに取る部分 (``json_schemas.TableAggregation.group_date_part``)。
DATE_PARTS: tuple[str, ...] = ("none", "year", "month")
#: 集計値の上で求める派生値 (``json_schemas.TableAggregation.derive``)。``pct_change`` /
#: ``diff`` はキーの自然な順で 1 つ前のグループからの増減率 (%) / 差、``share_of_total`` は
#: 合計に占める割合 (%)。
DERIVES: tuple[str, ...] = ("none", "pct_change", "diff", "share_of_total")
#: 派生値のうち百分率のもの (応答の「%」の数と照合する)。
_PERCENT_DERIVES = ("pct_change", "share_of_total")
#: 割合を出してよい集計 (平均・最小・最大の合計に占める割合は意味を持たない)。
_SHARE_AGGREGATIONS = ("sum", "count")

#: 表とみなす最小のデータ行数。1 行の「表」は集計するまでもない。
_MIN_DATA_ROWS = 2
#: 集計の結果をプロンプトへ並べる上限のグループ数。超えたら集計を諦める (並べきれない
#: 結果を一部だけ渡すと、残りをモデルが暗算で補う)。
MAX_GROUPS = 50
#: 平均など整数にならない値の表示の桁 (小数 4 桁で四捨五入)。
_FRACTION_QUANT = Decimal("0.0001")
#: 応答の数を集計値と比べる桁の帯 (集計値の絶対値の何倍から何倍まで)。暗算の誤りは
#: 集計値の近くに出るが、範囲には別の量 (人数・列の数・前月比の %) も現れる。
_CONTRADICTION_BAND = (Decimal("0.5"), Decimal(2))
#: 百分率の派生値の表示の桁 (小数 1 桁で四捨五入)。
_PERCENT_QUANT = Decimal("0.1")
#: 小数で述べた百分率を派生値と同じとみなす幅 (ポイント)。小数 1 桁の四捨五入 (±0.05) を
#: 許し、暗算の誤り (−11.95 を −11.7) は拾う。整数で述べた率は四捨五入した値と比べる
#: (``_matches_rate``)。
_PERCENT_TOLERANCE = Decimal("0.15")
#: 応答の百分率 (「11.7%」「+17.1 %」。NFKC で ``％`` は ``%`` になる)。
_PERCENT_RE = re.compile(r"(?<![\d.,])[-−+]?(?P<num>\d+(?:\.\d+)?)\s*%")
#: 自然な順に並べるためのキーの数字の連なり。
_DIGITS_RE = re.compile(r"(\d+)")

#: ``read_file`` の結果が全体でないことを示す印 (行の範囲指定 / 大きすぎて切った)。
#: 一部の行だけで集計すると、全体の値として渡すことになる。
_PARTIAL_READ_RE = re.compile(r"\| showing lines \d+-\d+|\(truncated, file too large\)")
#: ``read_file`` のメタ行の対象のパス。
_META_PATH_RE = re.compile(r"^\[file: (?P<path>.+?) \| lines: ")
#: 日付のセル (``2026-07`` / ``2026-07-15`` / ``2026/7/15`` / ``2026年7月``)。
_DATE_CELL_RE = re.compile(r"^\s*(?P<y>\d{4})\s*[-/.年]\s*(?P<m>\d{1,2})(?!\d)")
#: セルの数値から除く通貨・単位・桁区切り (NFKC の後)。
_NUMBER_NOISE_RE = re.compile(r"[,\s¥$€£円]")
#: 応答の中の数 (桁区切りあり / なし、小数、負号、「31万700」「2万」「1.5億」の万・億・千)。
_NUMBER_PART = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_RESPONSE_NUMBER_RE = re.compile(
    rf"(?<![\d.])(?P<sign>[-−])?(?P<body>(?:{_NUMBER_PART}\s*[億万千])+(?:{_NUMBER_PART})?"
    rf"|{_NUMBER_PART})",
)
_NUMBER_UNIT_PART_RE = re.compile(rf"(?P<num>{_NUMBER_PART})\s*(?P<unit>[億万千]?)")
_NUMBER_UNITS = {"億": Decimal(10) ** 8, "万": Decimal(10) ** 4, "千": Decimal(1000), "": Decimal(1)}
#: グループの値の境界に来てはいけない語の文字 (英数字・カタカナ・漢字。ひらがなの助詞は可)。
#: 「東」が「東京」の中で当たらないように (2026-10-05 レビュー)。
_KEY_WORD_CHARS = f"A-Za-z0-9{KATAKANA_WORD}{KANJI}{KANJI_EXT_A}{KANJI_MARKS}"
#: 集計値の桁の上限 (10^30)。超える結果は表の集計として扱わない。
_MAX_ADJUSTED = 30
#: ``;`` 区切りの表で小数点がカンマのセル (「12,5」)。数として読むと桁を誤る。
_DECIMAL_COMMA_CELL_RE = re.compile(r"^\s*-?\d+,\d{1,2}\s*$")
#: 式の中で列名の代わりに置く名前 (``_c0`` / ``_c1`` …)。
_PLACEHOLDER = "_c{}"
_ASCII_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


class AggregateError(ValueError):
    """パラメータが表と合わない・式が許可されない・値が数でない (集計を諦める理由)。"""


@dataclass(frozen=True)
class Table:
    """取得結果から読み取った表 (ヘッダと全データ行)。"""

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    #: ``read_file`` のメタ行のパス (無ければ空文字)。
    source: str = ""


@dataclass(frozen=True)
class AggregateSpec:
    """モデルが取った集計のパラメータ (``json_schemas.TableAggregation`` の検証済みの形)。"""

    value: str
    agg: str = "sum"
    group_by: tuple[str, ...] = ()
    group_date_part: str = "none"
    where_column: str = ""
    where_equals: str = ""
    derive: str = "none"


@dataclass(frozen=True)
class DerivedValue:
    """集計値の上で求めた派生値 1 つ (``value`` が ``None`` は基準が 0 以下で算出できない)。"""

    key: str
    value: Decimal | None
    #: 比べた 1 つ前のグループ (``pct_change`` / ``diff``)。割合は空文字。
    base_key: str = ""


@dataclass(frozen=True)
class AggregateResult:
    """全行に対する集計の結果。``groups`` は (グループの値, 集計値) の並び (初出順)。"""

    spec: AggregateSpec
    groups: tuple[tuple[str, Decimal], ...]
    #: 絞り込みの後に集計へ入った行数。
    rows_used: int
    #: 表の全データ行数。
    rows_total: int
    source: str = ""
    #: ``spec.derive`` の派生値 (求めない・求められない集計では空)。
    derived: tuple[DerivedValue, ...] = ()


# ---------------------------------------------------------------------------
# 表の解析
# ---------------------------------------------------------------------------


def _body_and_source(text: str) -> tuple[str, str] | None:
    """``read_file`` のメタ行を除いた本文と対象のパス。一部しか読めていなければ ``None``。"""
    if not text.startswith(READ_FILE_META_PREFIX):
        return text, ""
    head, _sep, rest = text.partition("\n")
    if _PARTIAL_READ_RE.search(head) or _PARTIAL_READ_RE.search(rest[-80:]):
        return None
    match = _META_PATH_RE.match(head)
    return rest, (match.group("path") if match else "")


def _parse_number(cell: str) -> Decimal | None:
    """セルを数として読む (通貨記号・桁区切り・単位「円」を除く)。読めなければ ``None``。"""
    text = _NUMBER_NOISE_RE.sub("", unicodedata.normalize("NFKC", cell or ""))
    if not text or text in ("-", "+", "."):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite() or (value != 0 and value.adjusted() > _MAX_ADJUSTED):
        return None
    return value


def _markdown_table(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    """GFM の表 (``| a | b |`` + 区切り行) を読む。無ければ ``None``。"""
    rows = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(rows) < 2 + _MIN_DATA_ROWS:
        return None

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    if not re.fullmatch(r"\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?", rows[1]):
        return None
    header = cells(rows[0])
    data = [cells(r) for r in rows[2:]]
    if any(len(r) != len(header) for r in data):
        return None
    return header, data


def _delimited_table(body: str) -> tuple[list[str], list[list[str]]] | None:
    """区切り文字の表 (CSV / TSV / セミコロン) を読む。全行の列数が揃うものだけ。"""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if len(lines) < 1 + _MIN_DATA_ROWS:
        return None
    for delimiter in (",", "\t", ";"):
        if delimiter not in lines[0]:
            continue
        parsed = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
        width = len(parsed[0])
        if width < 2 or any(len(r) != width for r in parsed):
            continue
        if delimiter == ";" and any(
            _DECIMAL_COMMA_CELL_RE.match(c) for r in parsed[1:] for c in r
        ):
            # 小数点がカンマの地域の CSV (「12,5」)。桁区切りとして読むと 125 になる
            return None
        return [c.strip() for c in parsed[0]], [[c.strip() for c in r] for r in parsed[1:]]
    return None


def parse_table(text: str) -> Table | None:
    """取得結果 (``read_file`` の出力等) を表として読む (純粋関数)。

    表と認めるのは、ヘッダ行 + 2 行以上のデータ行で、全行の列数が揃い、ヘッダの
    セルが数でなく、数の列が 1 つ以上あるものだけ。``read_file`` が一部しか返して
    いない (行の範囲指定 / 大きすぎて切った) 結果は表として扱わない — 一部の行の
    集計を全体の値として渡すことになる。
    """
    if not text or not text.strip():
        return None
    split = _body_and_source(text)
    if split is None:
        return None
    body, source = split
    parsed = _markdown_table(body.splitlines()) or _delimited_table(body)
    if parsed is None:
        return None
    header, data = parsed
    if len(data) < _MIN_DATA_ROWS:
        return None
    if any(not h or _parse_number(h) is not None for h in header):
        return None
    numeric_columns = [
        i for i in range(len(header))
        if all(_parse_number(r[i]) is not None for r in data)
    ]
    if not numeric_columns:
        return None
    return Table(
        header=tuple(header), rows=tuple(tuple(r) for r in data), source=source,
    )


def find_table(outputs: list[str]) -> Table | None:
    """取得結果の並びの ``read_file`` の表がただ 1 つに決まればそれを返す (無ければ ``None``)。

    対象は ``read_file`` の結果 (メタ行で始まる) だけ。取得したページ (``fetch_url``) は
    幅の同じ別の表が並びうり、1 つの表として読むと行を取り違える。**別のファイルの表が
    2 つ以上あれば集計しない** — どちらを問われたかを決められず、最後に読んだ方を選ぶと
    別のファイルの合計を確定値として渡す (2026-10-05 レビュー)。同じファイルの読み直しは
    最後の読みを使う。
    """
    tables: dict[object, Table] = {}
    for text in outputs or []:
        if not (text or "").startswith(READ_FILE_META_PREFIX):
            continue
        table = parse_table(text)
        if table is None:
            continue
        key = table.source or (table.header, table.rows)
        tables.pop(key, None)
        tables[key] = table
    if len(tables) != 1:
        return None
    return next(iter(tables.values()))


# ---------------------------------------------------------------------------
# 式の評価
# ---------------------------------------------------------------------------


def _column_index(table: Table, name: str) -> int:
    """列名を列の位置へ解決する (完全一致、無ければ大小文字・全角を無視して一致)。"""
    wanted = (name or "").strip()
    if wanted in table.header:
        return table.header.index(wanted)
    folded = unicodedata.normalize("NFKC", wanted).casefold()
    for i, h in enumerate(table.header):
        if unicodedata.normalize("NFKC", h).casefold() == folded:
            return i
    raise AggregateError(f"unknown column: {name!r}")


def _substitute_columns(expression: str, table: Table) -> str:
    """式の列名を ``_c<i>`` に置き換える (長い名前から。英数字の名前は語の境界で)。"""
    expr = unicodedata.normalize("NFKC", expression or "")
    expr = expr.replace("×", "*").replace("÷", "/")
    order = sorted(range(len(table.header)), key=lambda i: -len(table.header[i]))
    for i in order:
        name = unicodedata.normalize("NFKC", table.header[i])
        if not name:
            continue
        if _ASCII_NAME_RE.match(name):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
            expr = re.sub(pattern, _PLACEHOLDER.format(i), expr, flags=re.IGNORECASE)
        else:
            expr = expr.replace(name, _PLACEHOLDER.format(i))
    return expr


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)


def _compile_value(expression: str, table: Table) -> ast.expr:
    """値の式を検証済みの AST にする。列名・数・四則演算・括弧以外は拒む。"""
    source = _substitute_columns(expression, table)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise AggregateError(f"not an expression: {expression!r}") from e
    placeholders = {_PLACEHOLDER.format(i) for i in range(len(table.header))}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
            continue
        if isinstance(node, _ALLOWED_BINOPS + _ALLOWED_UNARY):
            continue
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) >= 10 ** _MAX_ADJUSTED:
                # ``1E+5000`` は inf に、``b * 1e999`` は Infinity になる
                raise AggregateError(f"constant out of range: {node.value!r}")
            continue
        if isinstance(node, ast.Name) and node.id in placeholders:
            continue
        raise AggregateError(
            f"unsupported element in the value expression: {type(node).__name__}",
        )
    return tree.body


def _evaluate(node: ast.expr, row: tuple[str, ...]) -> Decimal:
    """検証済みの式を 1 行に対して Decimal で評価する。"""
    if isinstance(node, ast.Constant):
        return Decimal(str(node.value))
    if isinstance(node, ast.Name):
        index = int(node.id[2:])
        value = _parse_number(row[index])
        if value is None:
            raise AggregateError(f"non-numeric cell: {row[index]!r}")
        return value
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, row)
        return -operand if isinstance(node.op, ast.USub) else operand
    assert isinstance(node, ast.BinOp)
    left, right = _evaluate(node.left, row), _evaluate(node.right, row)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if right == 0:
        raise AggregateError("division by zero")
    return left / right


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------


def _group_value(cell: str, date_part: str) -> str:
    """グループの値。日付の部分を取るなら ``YYYY`` / ``YYYY-MM`` (日付でなければセルのまま)。"""
    if date_part != "none":
        match = _DATE_CELL_RE.match(unicodedata.normalize("NFKC", cell))
        if match:
            year, month = match.group("y"), int(match.group("m"))
            if date_part == "year":
                return year
            if 1 <= month <= 12:
                return f"{year}-{month:02d}"
    return cell.strip()


def run_aggregate(table: Table, spec: AggregateSpec) -> AggregateResult:
    """表の全行に対して集計する (純粋関数)。合わないパラメータは ``AggregateError``。"""
    if spec.agg not in AGGREGATIONS:
        raise AggregateError(f"unknown aggregation: {spec.agg!r}")
    if spec.group_date_part not in DATE_PARTS:
        raise AggregateError(f"unknown date part: {spec.group_date_part!r}")
    if spec.derive not in DERIVES:
        raise AggregateError(f"unknown derived value: {spec.derive!r}")
    value_node = None
    if spec.agg != "count" or spec.value.strip():
        if not spec.value.strip():
            raise AggregateError("empty value expression")
        value_node = _compile_value(spec.value, table)
    group_columns = [_column_index(table, name) for name in spec.group_by]
    where_index = (
        _column_index(table, spec.where_column) if spec.where_column.strip() else None
    )

    # 絞り込みの値も日付の部分で比べる (「2026-07」で 2026-07-15 の行を取る)
    wanted = _group_value(spec.where_equals, spec.group_date_part)
    buckets: dict[str, list[Decimal]] = {}
    used = 0
    for row in table.rows:
        if where_index is not None and row[where_index].strip() != spec.where_equals.strip() and (
            _group_value(row[where_index], spec.group_date_part) != wanted
        ):
            continue
        key = " / ".join(
            _group_value(row[i], spec.group_date_part) for i in group_columns
        )
        value = Decimal(1) if value_node is None else _evaluate(value_node, row)
        buckets.setdefault(key, []).append(value)
        used += 1
    if not used:
        raise AggregateError("no rows matched")
    if len(buckets) > MAX_GROUPS:
        raise AggregateError(f"too many groups: {len(buckets)}")

    def reduce(values: list[Decimal]) -> Decimal:
        if spec.agg == "sum":
            return sum(values, Decimal(0))
        if spec.agg == "mean":
            return sum(values, Decimal(0)) / Decimal(len(values))
        if spec.agg == "min":
            return min(values)
        if spec.agg == "max":
            return max(values)
        return Decimal(len(values))

    groups = tuple((key, reduce(values)) for key, values in buckets.items())
    for _key, value in groups:
        if not value.is_finite() or (value != 0 and value.adjusted() > _MAX_ADJUSTED):
            raise AggregateError(f"result out of range: {value}")
    derived = _derive(spec, groups)
    return AggregateResult(
        # 求められない派生値 (グループが 1 つ・列の組・平均の割合) は spec からも落とす —
        # 枠と注記の「集計:」に、値を渡していない派生値の名前を出さない
        spec=spec if derived or spec.derive == "none" else replace(spec, derive="none"),
        groups=groups,
        rows_used=used,
        rows_total=len(table.rows),
        source=table.source,
        derived=derived,
    )


def _key_order(groups: tuple[tuple[str, Decimal], ...]) -> list[tuple[str, Decimal]]:
    """増減を求めるグループの順。

    キーがみな同じ型 (数字以外の部分が同じ) で、先頭の数字が 4 桁 (年) のとき
    (``2026-07`` / ``2026-10``、``2026``、``2026/7/15``) だけ、数字を数として比べた順
    (``2026-9`` < ``2026-10``)。それ以外は表に現れた順 — 地域名、``Jan 2026`` / ``Feb 2026``
    のように数字以外が違うキー、年の無い月・四半期・月日 (``11月`` / ``12月`` / ``1月``、
    会計年度の ``4月``…``3月``、``Q4`` / ``Q1``) は、数で並べると年の境目で前後を誤る。
    """
    split = [
        _DIGITS_RE.split(unicodedata.normalize("NFKC", key)) for key, _v in groups
    ]
    templates = {tuple(parts[0::2]) for parts in split}
    if len(templates) != 1 or not all(len(parts) > 1 and len(parts[1]) == 4 for parts in split):
        return list(groups)
    order = sorted(
        range(len(groups)), key=lambda i: tuple(int(n) for n in split[i][1::2]),
    )
    return [groups[i] for i in order]


def _derive(
    spec: AggregateSpec, groups: tuple[tuple[str, Decimal], ...],
) -> tuple[DerivedValue, ...]:
    """集計値の上の派生値を Decimal で求める (純粋関数)。求められなければ空。

    ``pct_change`` / ``diff`` はグループの列が 1 つで 2 グループ以上のときだけ (複数の列の
    組の「1 つ前」は決まらない)。``pct_change`` の基準が 0 以下なら ``None`` (0 除算・符号の
    反転した率を出さない)。``share_of_total`` は ``sum`` / ``count`` で 2 グループ以上のとき
    だけで、合計が 0 以下かどれかの値が負なら全グループ ``None``。
    """
    if spec.derive == "none" or len(groups) < 2:
        return ()
    if spec.derive == "share_of_total":
        if spec.agg not in _SHARE_AGGREGATIONS:
            return ()
        total = sum((v for _k, v in groups), Decimal(0))
        # 負の値があると割合が 100% を超え・負になり、構成比として読めない
        defined = total > 0 and all(v >= 0 for _k, v in groups)
        return tuple(
            DerivedValue(key, value * 100 / total if defined else None)
            for key, value in groups
        )
    if len(spec.group_by) != 1:
        return ()
    ordered = _key_order(groups)
    out: list[DerivedValue] = []
    for (prev_key, prev), (key, value) in zip(ordered, ordered[1:], strict=False):
        if spec.derive == "diff":
            out.append(DerivedValue(key, value - prev, prev_key))
        else:
            change = (value - prev) * 100 / prev if prev > 0 else None
            out.append(DerivedValue(key, change, prev_key))
    return tuple(out)


# ---------------------------------------------------------------------------
# 整形と照合
# ---------------------------------------------------------------------------


def format_number(value: Decimal) -> str:
    """集計値の表示 (整数は桁区切り、端数は小数 4 桁で四捨五入して末尾の 0 を落とす)。"""
    if value == value.to_integral_value():
        return f"{int(value):,}"
    rounded = value.quantize(_FRACTION_QUANT, rounding=ROUND_HALF_UP)
    integer, _dot, fraction = f"{rounded:f}".partition(".")
    sign = "-" if integer.startswith("-") else ""
    head = f"{abs(int(integer)):,}" if integer.lstrip("-") else "0"
    fraction = fraction.rstrip("0")
    return f"{sign}{head}.{fraction}" if fraction else f"{sign}{head}"


def _response_numbers(text: str) -> list[Decimal]:
    """応答の数を読む (桁区切り・小数・負号・「31万700」「2万」「1.5億」の万・億・千)。"""
    out: list[Decimal] = []
    for match in _RESPONSE_NUMBER_RE.finditer(text or ""):
        total = Decimal(0)
        try:
            for part in _NUMBER_UNIT_PART_RE.finditer(match.group("body")):
                total += Decimal(part.group("num").replace(",", "")) * _NUMBER_UNITS[
                    part.group("unit")
                ]
        except InvalidOperation:
            continue
        out.append(-total if match.group("sign") else total)
    return out


def _stated(value: Decimal, numbers: list[Decimal]) -> bool:
    """``value`` が数の並びに在るか (端数は小数 1〜4 桁の四捨五入の表示も同じに数える)。

    整数への丸め (1.6667 → 2) は数えない — 平均を整数で述べると別の値と区別できない。
    """
    if value == value.to_integral_value():
        return value in numbers
    for places in range(1, 5):
        quant = Decimal(1).scaleb(-places)
        if value.quantize(quant, rounding=ROUND_HALF_UP) in numbers:
            return True
    return False


def _key_spans(text: str, keys: list[str]) -> list[tuple[int, int, str]]:
    """応答の中のグループの値の出現 (開始, 終了, 値)。語の途中の一致 (「東京」の「東」) は除く。"""
    spans: list[tuple[int, int, str]] = []
    for key in sorted({k for k in keys if k}, key=len, reverse=True):
        pattern = re.compile(
            rf"(?<![{_KEY_WORD_CHARS}]){re.escape(key)}(?![{_KEY_WORD_CHARS}])",
        )
        for m in pattern.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e, _k in spans):
                spans.append((m.start(), m.end(), key))
    return sorted(spans)


def _near(value: Decimal, numbers: list[Decimal]) -> list[Decimal]:
    """``numbers`` のうち絶対値が ``value`` の桁の帯に入るもの。"""
    low, high = _CONTRADICTION_BAND
    size = abs(value)
    return [n for n in numbers if size * low <= abs(n) <= size * high]


def contradicted_values(response: str, result: AggregateResult) -> list[tuple[str, Decimal]]:
    """応答が集計と **別の数** を述べたグループ (純粋関数)。述べなかったグループは数えない。

    グループの値 (``2026-07``) が現れた位置から次のグループの値の手前までをそのグループの
    範囲とし (同じ行の「2026-07: 310,700」も、見出しの次の行の値も入る)、範囲に数が
    あるのに集計値が無ければ食い違いとする。グループの値が応答に無いグループ (省いた・
    「7月」と書き換えた) は見ない — 省略は誤りではない。グループの無い集計は応答全体が範囲。
    比べるのは絶対値が集計値の ``_CONTRADICTION_BAND`` に入る数だけ (別の量を食い違いと
    読まない、docs/f_03 §4.2.2 の 5)。
    """
    text = unicodedata.normalize("NFKC", response or "")
    contradicted: list[tuple[str, Decimal]] = []
    if len(result.groups) == 1 and not result.groups[0][0]:
        value = result.groups[0][1]
        numbers = _near(value, _response_numbers(text))
        if numbers and not _stated(value, numbers):
            contradicted.append(result.groups[0])
        return contradicted
    spans = _key_spans(text, [k for k, _v in result.groups])
    scoped: dict[str, list[Decimal]] = {}
    for i, (_start, end, key) in enumerate(spans):
        stop = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        scoped.setdefault(key, []).extend(_base_numbers(text[end:stop], result))
    for key, value in result.groups:
        numbers = _near(value, scoped.get(key, []))
        if numbers and not _stated(value, numbers):
            contradicted.append((key, value))
    return contradicted


def _matches_rate(stated: Decimal, rates: list[Decimal]) -> bool:
    """述べた率 (絶対値) が派生値の率 (絶対値) のどれかと合うか。

    小数のある数は ``_PERCENT_TOLERANCE`` 以内、整数で述べた数は率を整数へ四捨五入した
    値と等しいこと (「33% / 17%」は 33.3 / 16.7 と合い、「11%」は 11.95 と合わない)。
    """
    if stated.as_tuple().exponent >= 0:
        return any(r.quantize(Decimal(1), rounding=ROUND_HALF_UP) == stated for r in rates)
    return any(abs(stated - r) <= _PERCENT_TOLERANCE for r in rates)


def _known_rates(result: AggregateResult) -> list[Decimal]:
    return [abs(d.value) for d in result.derived if d.value is not None]


def _base_numbers(segment: str, result: AggregateResult) -> list[Decimal]:
    """範囲の数のうち、集計値と比べるもの。派生値を述べた数 (「17.1%」・差の値) は除く。

    派生値を渡したターンでは答えに増減率・差が並ぶ。それを集計値の食い違いに数えると、
    基準の小さい集計 (平均) や伸びの大きい月の差 (集計値の 0.5 倍以上) で誤った注記になる。
    除くのは派生値と合う率だけ — 集計する量そのものが率 (率の列の平均) のとき、合わない
    「%」の数まで除くと誤った値を照合から隠す。
    """
    if not result.derived:
        return _response_numbers(segment)
    if result.spec.derive in _PERCENT_DERIVES:
        rates = _known_rates(result)
        return _response_numbers(_PERCENT_RE.sub(
            lambda m: " " if _matches_rate(Decimal(m.group("num")), rates) else m.group(0),
            segment,
        ))
    diffs = [abs(d.value) for d in result.derived if d.value is not None]
    return [
        n for n in _response_numbers(segment)
        if not any(_stated(d, [abs(n)]) for d in diffs)
    ]


def contradicted_derived(response: str, result: AggregateResult) -> list[DerivedValue]:
    """応答がグループに帰して述べた百分率が、どの派生値とも合わないもの (純粋関数)。

    照合するのは百分率の派生値 (``pct_change`` / ``share_of_total``) だけ。グループの範囲は
    :func:`contradicted_values` と同じ (グループの値の位置から次のグループの値の手前まで)。
    範囲の「%」の数のうち絶対値が派生値の絶対値の ``_CONTRADICTION_BAND`` に入るものを
    候補とし、候補がありどれも **いずれかの** 派生値と合わなければ食い違い
    (:func:`_matches_rate`: 小数は ±``_PERCENT_TOLERANCE``、整数は四捨五入した値)。

    - 符号は比べない: 「11.9%減」は数としては 11.9。増・減の語を読むと語彙が要る (#14)。
    - いずれかの派生値と合えば数えない: 「2026-09 は 2026-08 比 11.9% 減」は 11.9% が
      2026-08 の範囲に落ちるので、そのグループの値とだけ比べると正しい答えに注記を付ける。
      代わりにグループ間の取り違えは捕まえない。
    - 差 (``diff``) は照合しない: 符号・「増」「減」の書き方が多様で、桁の帯で別の量と
      分けられない。
    """
    if result.spec.derive not in _PERCENT_DERIVES or not result.derived:
        return []
    known = _known_rates(result)
    if not known:
        return []
    text = unicodedata.normalize("NFKC", response or "")
    spans = _key_spans(text, [k for k, _v in result.groups])
    scoped: dict[str, list[Decimal]] = {}
    for i, (_start, end, key) in enumerate(spans):
        stop = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        scoped.setdefault(key, []).extend(
            Decimal(m.group("num")) for m in _PERCENT_RE.finditer(text[end:stop])
        )
    contradicted: list[DerivedValue] = []
    for item in result.derived:
        if item.value is None:
            continue
        candidates = _near(item.value, scoped.get(item.key, []))
        if candidates and not any(_matches_rate(c, known) for c in candidates):
            contradicted.append(item)
    return contradicted


def describe_spec(spec: AggregateSpec) -> str:
    """集計の中身の短い表記 (``sum(units * unit_price) by month (month)``、言語に依らない)。"""
    head = f"{spec.agg}({spec.value.strip()})" if spec.value.strip() else f"{spec.agg}(rows)"
    if spec.group_by:
        part = f" ({spec.group_date_part})" if spec.group_date_part != "none" else ""
        head += f" by {', '.join(spec.group_by)}{part}"
    if spec.where_column.strip():
        head += f" where {spec.where_column.strip()} = {spec.where_equals.strip()}"
    if spec.derive != "none":
        head += f", {spec.derive}"
    return head


def format_derived(item: DerivedValue, derive: str) -> str | None:
    """派生値の表示 (百分率は小数 1 桁で四捨五入して ``%``、差は符号付き)。算出不可は ``None``。"""
    if item.value is None:
        return None
    if derive in _PERCENT_DERIVES:
        rounded = item.value.quantize(_PERCENT_QUANT, rounding=ROUND_HALF_UP)
        if rounded == 0:
            rounded = abs(rounded)  # 「-0.0%」と書かない
        sign = "+" if derive == "pct_change" and rounded > 0 else ""
        return f"{sign}{rounded:f}%"
    sign = "+" if item.value > 0 else ""
    return f"{sign}{format_number(item.value)}"


#: 派生値の見出しと算出不可の表記 (``i18n.prompt_locale`` 別)。
_DERIVED_LABELS: dict[str, dict[str, str]] = {
    "ja": {
        "pct_change": "1 つ前のグループからの増減率 (小数 1 桁で四捨五入):",
        "diff": "1 つ前のグループとの差:",
        "share_of_total": "合計に占める割合 (小数 1 桁で四捨五入):",
        "versus": "{key} ({base} 比)",
        "undefined": "算出できない (基準の値が 0 以下)",
        "note_pct_change": "1 つ前のグループからの増減率",
        "note_diff": "1 つ前のグループとの差",
        "note_share_of_total": "合計に占める割合",
        "instruction": "これらの値も自分で計算し直さず、ここからそのまま取ること。",
    },
    "en": {
        "pct_change": "Percent change from the previous group (rounded to 1 decimal):",
        "diff": "Difference from the previous group:",
        "share_of_total": "Share of the total (rounded to 1 decimal):",
        "versus": "{key} (vs {base})",
        "undefined": "undefined (the base value is zero or negative)",
        "note_pct_change": "percent change from the previous group",
        "note_diff": "difference from the previous group",
        "note_share_of_total": "share of the total",
        "instruction": "Take these values as they are too; do not recompute them.",
    },
}


def _derived_lines(result: AggregateResult, locale: str) -> list[str]:
    if not result.derived:
        return []
    labels = _DERIVED_LABELS.get(locale, _DERIVED_LABELS["ja"])
    lines = [labels[result.spec.derive]]
    for item in result.derived:
        label = (
            labels["versus"].format(key=item.key, base=item.base_key)
            if item.base_key else item.key
        )
        shown = format_derived(item, result.spec.derive) or labels["undefined"]
        lines.append(f"- {label}: {shown}")
    lines.append(labels["instruction"])
    return lines


def _value_lines(result: AggregateResult) -> list[str]:
    return [
        f"- {key}: {format_number(value)}" if key else f"- {format_number(value)}"
        for key, value in result.groups
    ]


#: プロンプトへ渡す集計結果の枠 (``i18n.prompt_locale`` 別)。
_RESULT_BLOCKS: dict[str, str] = {
    "ja": (
        "## 表の集計結果 (コードで全 {rows} 行から計算した確定値)\n"
        "集計: {spec} (対象 {used} 行)\n{values}\n"
        "答えの数値はこの集計結果からそのまま取ること。表の数値を自分で計算し直したり、"
        "別の値を述べたりしないこと。"
    ),
    "en": (
        "## Table aggregation (exact values computed by code over all {rows} rows)\n"
        "Aggregation: {spec} ({used} rows)\n{values}\n"
        "Take the numbers of the answer from this result as they are. Do not recompute "
        "the table yourself or state different values."
    ),
}
#: 持ち越した資料の表への続きの問い用 (docs/f_03 §4.2.2 の 6)。往復は語のゲート無しで
#: 撃つので、問いと別の集計を取りうる — その値で答えさせない条件付きの文面。
_CONDITIONAL_RESULT_BLOCKS: dict[str, str] = {
    "ja": (
        "## 表の集計結果 (コードで全 {rows} 行から計算した確定値)\n"
        "集計: {spec} (対象 {used} 行)\n{values}\n"
        "今回の問いがこの集計を求めていれば、答えの数値はこの集計結果からそのまま取ること"
        " (表の数値を自分で計算し直したり、別の値を述べたりしない)。求めていなければ、"
        "この集計に触れないこと。"
    ),
    "en": (
        "## Table aggregation (exact values computed by code over all {rows} rows)\n"
        "Aggregation: {spec} ({used} rows)\n{values}\n"
        "If the current question asks for this aggregation, take the numbers of the "
        "answer from this result as they are (do not recompute the table or state "
        "different values). If it does not, do not mention this aggregation."
    ),
}


def result_block(result: AggregateResult, *, conditional: bool = False) -> str:
    """集計結果をモデルへ渡す枠 (``i18n.prompt_locale`` の言語)。

    ``conditional`` は問いがこの集計を求めているときだけ使わせる文面 (続きのターン用)。
    """
    from backend.i18n_helper import prompt_locale

    blocks = _CONDITIONAL_RESULT_BLOCKS if conditional else _RESULT_BLOCKS
    locale = prompt_locale()
    template = blocks.get(locale, blocks["ja"])
    return template.format(
        rows=result.rows_total, used=result.rows_used,
        spec=describe_spec(result.spec),
        values="\n".join(_value_lines(result) + _derived_lines(result, locale)),
    )


def mismatch_note(response: str, result: AggregateResult | None) -> str:
    """応答が集計と別の値を述べたら、正しい値を示す末尾の注記 (無ければ空文字)。

    述べなかったグループでは足さない (:func:`contradicted_values`)。

    本文は書き換えない (ストリーミングで送り済み)。ja は ``(注: …)``、en は
    ``(System note: …)`` の形で、``strip_system_notes`` が記憶・経験へ積む前に落とす。
    """
    if result is None or not (response or "").strip():
        return ""
    if not contradicted_values(response, result) and not contradicted_derived(
        response, result,
    ):
        return ""
    from backend.i18n_helper import msg_for_locale, prompt_locale

    locale = prompt_locale()
    source = result.source or msg_for_locale(locale, "agent.output_note.table_fallback")
    values = ", ".join(
        f"{key}: {format_number(value)}" if key else format_number(value)
        for key, value in result.groups
    )
    derived = [
        (item.key, format_derived(item, result.spec.derive)) for item in result.derived
    ]
    if any(shown for _k, shown in derived):
        labels = _DERIVED_LABELS.get(locale, _DERIVED_LABELS["ja"])
        values += f"; {labels['note_' + result.spec.derive]}: " + ", ".join(
            f"{key}: {shown}" for key, shown in derived if shown
        )
    # 注記の括弧は 1 段の入れ子までしか ``strip_system_notes`` が落とせないので、
    # セルやパス由来の丸括弧は角括弧へ替える
    return "\n\n" + msg_for_locale(
        locale, "agent.output_note.table_aggregate",
        source=source.translate(_NOTE_PARENS), rows=str(result.rows_total),
        # 派生値の種類は値の並びに言語別の名前で書くので、「集計:」の表記には付けない
        spec=describe_spec(replace(result.spec, derive="none")).translate(_NOTE_PARENS),
        values=values.translate(_NOTE_PARENS),
    )


_NOTE_PARENS = str.maketrans({"(": "[", ")": "]", "（": "[", "）": "]"})
