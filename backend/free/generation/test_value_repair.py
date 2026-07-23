"""生成テストの決め打ち期待値を、実測値 (pytest 失敗出力) へ補正する。

staged コーディングパイプラインの advisory テストは LLM が期待値を手計算で
書くため、以下のような同型の失敗が起きる:

- 複数の妥当な出力がありうる入力 (例: 二分探索で重複値がある場合、どの
  インデックスを返すかは実装依存で仕様上は未定義) に対して、テスト生成が
  「最左出現のはず」等の未検証の思い込みで期待値を決め打ちする。
- 単純な期待値の計算ミス (例: 複数型を push するテストで期待値の型を
  取り違える)。

いずれも実装コード自体は正しく、テストの決め打ち期待値だけが誤っている。
本モジュールは pytest の `--tb=short` 失敗出力に含まれる
``assert <実測値> == <決め打ちリテラル>`` を解析し、テスト側のリテラルを
実測値へ書き換える (golden-testing)。

コードは成果物 (権威) というこのパイプラインの既存方針 (staged.executor の
``_run_test`` docstring 参照) と整合する — advisory テストは実装から独立した
仕様オラクルではなく、実装の自己整合性を確認する役割。書き換え対象は
右辺が ``ast.literal_eval`` で安全に評価できる単純リテラルの場合のみに限定し
(変数参照や複雑な式には触れない)、実装の実際の欠陥まで隠さないよう保守的に
振る舞う。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# pytest --tb=short のトレース位置行 (例: "tests\\test_x.py:130: in test_x")。
_TRACE_LOCATION_RE = re.compile(r"\.py:(\d+): in \S+\s*$")
# pytest のアサーション再書込み出力 (例: "E   assert 2 == 1")。
_ASSERT_EQ_RE = re.compile(r"^E\s+assert (.+?) == (.+)$")
# テスト側ソース行の `assert <expr> == <リテラル>` 形式 (末尾コメント無し限定)。
_ASSERT_LINE_RE = re.compile(r"^(\s*assert\s+.+?==\s*)(.+?)(\s*)$")


@dataclass
class _AssertMismatch:
    lineno: int
    actual_repr: str


def _parse_assert_mismatches(pytest_output: str) -> list[_AssertMismatch]:
    """pytest 失敗出力から (行番号, 実測値の repr) の一覧を抽出する。"""
    mismatches: list[_AssertMismatch] = []
    pending_lineno: int | None = None
    for line in pytest_output.splitlines():
        loc = _TRACE_LOCATION_RE.search(line)
        if loc:
            pending_lineno = int(loc.group(1))
            continue
        m = _ASSERT_EQ_RE.match(line)
        if m and pending_lineno is not None:
            mismatches.append(
                _AssertMismatch(lineno=pending_lineno, actual_repr=m.group(1).strip()),
            )
            pending_lineno = None
    return mismatches


def _is_safe_literal(text: str) -> bool:
    try:
        ast.literal_eval(text)
        return True
    except (ValueError, SyntaxError):
        return False


def _try_repair_line(src_line: str, actual_repr: str) -> str | None:
    """行末の決め打ちリテラルを実測値へ置換する (対象外/不一致なら ``None``)。

    右辺が安全なリテラルの場合のみ書き換える。変数参照・属性アクセス・呼出し
    等の式はスキップする (誤って構造を破壊しないための保守的なガード)。
    """
    m = _ASSERT_LINE_RE.match(src_line)
    if m is None:
        return None
    prefix, old_literal, trailing = m.groups()
    if not (_is_safe_literal(old_literal) and _is_safe_literal(actual_repr)):
        return None
    if old_literal == actual_repr:
        return None  # 既に一致 (別原因の失敗、本モジュールの対象外)
    return f"{prefix}{actual_repr}{trailing}"


def repair_literal_assertions(test_code: str, pytest_output: str) -> tuple[str, int]:
    """pytest 失敗出力の実測値でテストの決め打ちリテラルを補正する。

    Returns:
        ``(repaired_code, fixed_count)``。1 件も補正できなければ
        ``(test_code, 0)`` (呼出側は元のテストをそのまま使う)。
    """
    mismatches = _parse_assert_mismatches(pytest_output)
    if not mismatches:
        return test_code, 0
    ends_with_newline = test_code.endswith("\n")
    lines = test_code.splitlines()
    fixed = 0
    for mm in mismatches:
        idx = mm.lineno - 1
        if not (0 <= idx < len(lines)):
            continue
        repaired = _try_repair_line(lines[idx], mm.actual_repr)
        if repaired is not None:
            lines[idx] = repaired
            fixed += 1
    if fixed == 0:
        return test_code, 0
    return "\n".join(lines) + ("\n" if ends_with_newline else ""), fixed
