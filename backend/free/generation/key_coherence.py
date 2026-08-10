"""生成物のモジュール間で辞書キーが食い違っていないかを静的に見る (advisory)

import スモークは「起動できるか」しか見ないため、**片方が作ったキーをもう片方が
別名で読む**類の欠陥は素通りする。実行して初めて ``KeyError`` になる。

実インシデント (2026-08-07 ライブ監査): ``csv_processor`` が
``{'mean':…, 'max':…, 'min':…}`` を返すのに ``main.py`` が ``stats['average']``
を読んでおり、「起動可能性チェック合格」として配信された (仕様書には
``ColumnStats`` の例として mean/max/min が明記されていたので情報不足ではない)。

**判定は advisory 専用**。プロジェクト外から来る辞書 (環境変数・JSON 入力等) の
キーは当然どこにも「作られて」いないため、ゲートにすると正常な生成を止めうる。
実測 (同監査で生成した 5 プロジェクト) では誤検出 0 / 実欠陥 1 件を検出したが、
それは母数が小さいだけなので警告に留める。
"""

from __future__ import annotations

import ast
from collections.abc import Mapping

#: 添字元がこれらなら外部由来の辞書とみなして対象外にする。
_EXTERNAL_SUBSCRIPT_ROOTS = frozenset({"environ", "os", "sys", "globals", "locals"})


def _is_external_subscript(node: ast.Subscript) -> bool:
    """``os.environ['PATH']`` 等、プロジェクト外の辞書への添字か。"""
    value = node.value
    if isinstance(value, ast.Attribute):
        return value.attr in _EXTERNAL_SUBSCRIPT_ROOTS or (
            isinstance(value.value, ast.Name)
            and value.value.id in _EXTERNAL_SUBSCRIPT_ROOTS
        )
    if isinstance(value, ast.Name):
        return value.id in _EXTERNAL_SUBSCRIPT_ROOTS
    return False


def _collect(code: str) -> tuple[set[str], set[str]]:
    """(辞書リテラルで作られたキー, 添字で読まれたキー) を返す。"""
    produced: set[str] = set()
    read: set[str] = set()
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return produced, read
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    produced.add(key.value)
        elif isinstance(node, ast.Subscript):
            sl = node.slice
            if (
                isinstance(sl, ast.Constant)
                and isinstance(sl.value, str)
                and not _is_external_subscript(node)
            ):
                read.add(sl.value)
    # ``d.get("k")`` は数えない。``app.get("/todos/")`` (FastAPI ルータ) や
    # ``requests.get(url)`` と構文上区別できず、実測 (2026-08-07 監査の 5
    # プロジェクト) で TODO API の ``/todos/`` を誤検出した。添字だけなら
    # 同じ母数で誤検出 0 / 実欠陥 1 件を検出できている。
    return produced, read


def find_unmatched_dict_keys(py_map: Mapping[str, str]) -> list[str]:
    """プロジェクト内のどこでも作られていない添字キーを返す (advisory)。

    Args:
        py_map: ``{logical_path: source}``。生成物の ``.py`` 全体を渡す。

    Returns:
        ソート済みのキー名。辞書リテラルが 1 つも無いプロジェクトでは比較対象が
        無いので常に空 (誤検出を避ける)。
    """
    produced: set[str] = set()
    read: set[str] = set()
    for code in py_map.values():
        p, r = _collect(code)
        produced |= p
        read |= r
    if not produced:
        return []
    return sorted(read - produced)


__all__ = ["find_unmatched_dict_keys"]
