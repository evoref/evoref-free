"""create のパッケージ形 — 平置きのコードから配信形を決定論で作る (f_10 §11.1-1 / §11.1-3)。

staged v2 は Python を平置き (``src/<name>.py``、bare import) で生成・検査する。骨組みの使い方が
``python -m <共通の親>`` のときだけ、配信の直前にパッケージの形 (相対 import・``__main__.py``) に
変換する (2026-09-27 ライブ監査 K04: ``python -m units 10 km mi`` が ``ModuleNotFoundError``)。

引き金は使い方と共通の親の一致だけ — ``__init__.py`` の有無では決めない (平置きのアプリに付いた
余計な ``__init__.py`` で ``python main.py`` が壊れる)、``python -m http.server`` のような無関係な
モジュール名もパッケージにしない (反証レビュー S6)。純粋関数だけを置く (実行は
``smoke_validator.run_usage``)。
"""

from __future__ import annotations

import ast
import builtins
import re
import shlex
import textwrap
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from backend.free.core.check_outcome import CheckKind, CheckOutcome, UncheckedReason
from backend.free.core.intent_vocab import FILE_NAME_IN_TEXT_RE

if TYPE_CHECKING:
    from backend.free.generation.smoke_validator import UsageRun

#: Python の起動コマンド (``python`` / ``python3`` / ``python3.12`` / ``py`` / ``pythonw``、``.exe`` 付きも)。
_PYTHON_EXE_RE = re.compile(r"^(?:py|pythonw?(?:\d+(?:\.\d+)*)?)(?:\.exe)?$", re.IGNORECASE)
#: 値を 1 つ取るインタプリタのオプション (``-X utf8`` / ``-W ignore``)。
_OPTIONS_WITH_VALUE = frozenset({"-X", "-W"})
#: 使い方の引数のうち、実行者が埋める置き場所・シェルの記号 (``<値>`` / ``[file]`` / ``{x}`` / ``...`` / ``>`` / ``|``)。
_PLACEHOLDER_RE = re.compile(r"^[<\[{].*[>\]}]$|[<>|]|\.\.\.|…")
#: 画面を開く (端末を占有する) モジュール。import していたら使い方を実行しない。
_SCREEN_MODULES = frozenset({
    "tkinter", "turtle", "curses", "pygame", "pyglet", "arcade", "kivy", "wx",
    "PyQt5", "PyQt6", "PySide2", "PySide6",
})
#: traceback の中のパッケージ内のファイル (``File ".../units/length.py"``)。
_TRACE_FILE_RE = re.compile(r'File "([^"]+)"')
#: traceback の最後の例外の行 (``ValueError: …`` / ``json.decoder.JSONDecodeError: …``)。
_EXCEPTION_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(?::\s|$)")
_TRACEBACK_HEAD = "Traceback (most recent call last):"


# ── 引き金 ─────────────────────────────────────────────────────────────


def _split(line: str) -> list[str]:
    """コマンド行を語に分ける。``\\`` をエスケープとして読まない (Windows のパスを壊さない)、囲みの引用符は外す。"""
    try:
        tokens = shlex.split(line, posix=False)
    except ValueError:
        tokens = line.split()
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]


def usage_command(usage: str) -> tuple[str, list[str]] | None:
    """使い方 ``python -m <モジュール> <引数…>`` を ``(モジュール, 引数)`` にする (その形でなければ ``None``)。

    起動コマンド (``python`` / ``py`` / ``C:\\Python312\\python.exe`` …) の語から後ろだけを読む — 前置き
    (``$ `` / ``PS> `` / バッククォート / 「コマンド例:」 / ``cd x &&``) は捨てる。起動コマンドのある最初の行を見る。
    引数は ``#`` のコメントと ``;`` の後ろを捨てる。
    """
    for raw in str(usage or "").splitlines():
        tokens = _split(raw.replace("`", " "))
        start = next((i for i, t in enumerate(tokens) if _PYTHON_EXE_RE.match(PureWindowsPath(t).name)), None)
        if start is None:
            continue
        tokens = tokens[start:]
        i = 1
        while i < len(tokens) and tokens[i].startswith("-") and tokens[i] != "-m":
            i += 2 if tokens[i] in _OPTIONS_WITH_VALUE else 1
        if i + 1 >= len(tokens) or tokens[i] != "-m":
            return None
        args: list[str] = []
        for token in tokens[i + 2:]:
            if token.startswith("#"):
                break
            if token.endswith(";"):
                if token[:-1]:
                    args.append(token[:-1])
                break
            args.append(token)
        return tokens[i + 1], args
    return None


def package_from_usage(usage: str, folder: str) -> str:
    """パッケージ形の引き金。使い方が ``python -m <folder>`` (``python -m <folder>.cli`` も) なら
    パッケージのフォルダ (= ``folder``)、それ以外は空。

    ``folder`` は骨組みのモジュールの共通の親 (``normalize_skeleton`` の出力フォルダ)。モジュール名の ``.`` は
    ``/`` と読む。
    """
    command = usage_command(usage)
    folder = str(folder or "").strip("/")
    if command is None or not folder:
        return ""
    path = command[0].replace(".", "/")
    return folder if path == folder or path.startswith(folder + "/") else ""


# ── 配信形への変換 ─────────────────────────────────────────────────────


def _alias(name: str, asname: str | None) -> str:
    return f"{name} as {asname}" if asname else name


def _package_target(anchor: str, module: str) -> str:
    """``from <ここ> import`` の部分 (anchor ``.`` なら ``.length``、``units`` なら ``units.length``)。"""
    if not module:
        return anchor
    return f".{module}" if anchor == "." else f"{anchor}.{module}"


def _import_replacement(node: ast.stmt, siblings: set[str], anchor: str) -> list[str] | None:
    """兄弟モジュールの bare import を置き換える文の列 (置き換えないなら ``None``)。"""
    if isinstance(node, ast.Import):
        if not any(a.name in siblings for a in node.names):
            return None
        return [
            f"from {_package_target(anchor, '')} import {_alias(a.name, a.asname)}" if a.name in siblings
            else f"import {_alias(a.name, a.asname)}"
            for a in node.names
        ]
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        names = ", ".join(_alias(a.name, a.asname) for a in node.names)
        if node.module == "__init__":
            # 平置きの ``from __init__ import main`` はパッケージそのものから
            return [f"from {_package_target(anchor, '')} import {names}"]
        if node.module in siblings:
            return [f"from {_package_target(anchor, node.module)} import {names}"]
    return None


def to_package_imports(source: str, siblings: set[str], anchor: str = ".") -> str:
    """兄弟モジュールの bare import をパッケージ経由の import に書き換える。

    ``anchor`` が ``.`` ならパッケージの中 (``from . import length`` / ``from .length import convert``)、
    パッケージ名ならパッケージの外 (``from units import length``、配信するテスト)。関数の中の import も
    書き換える。``__init__`` / ``__main__`` は兄弟に数えない (``from __init__ import main`` はパッケージから)。
    構文エラーのファイルは触らない。
    """
    names = {s for s in siblings if s not in ("__init__", "__main__")}
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return source
    lines = source.splitlines(keepends=True)
    encoded = [line.encode("utf-8") for line in lines]
    starts = [0]
    for line in encoded:
        starts.append(starts[-1] + len(line))
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        statements = _import_replacement(node, names, anchor)
        if statements is None:
            continue
        # col_offset は UTF-8 のバイト位置
        prefix = encoded[node.lineno - 1][: node.col_offset].decode("utf-8", "replace")
        separator = "\n" + prefix if not prefix.strip() else "; "
        begin = starts[node.lineno - 1] + node.col_offset
        end = starts[(node.end_lineno or node.lineno) - 1] + (node.end_col_offset or 0)
        edits.append((begin, end, separator.join(statements)))
    if not edits:
        return source
    data = b"".join(encoded)
    for begin, end, text in sorted(edits, reverse=True):
        data = data[:begin] + text.encode("utf-8") + data[end:]
    return data.decode("utf-8")


def is_main_guard(node: ast.stmt) -> bool:
    """``if __name__ == "__main__":`` (``"__main__" == __name__`` の向きも) か。"""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
        return False
    sides = [test.left, test.comparators[0]]
    names = [s for s in sides if isinstance(s, ast.Name) and s.id == "__name__"]
    consts = [s for s in sides if isinstance(s, ast.Constant) and s.value == "__main__"]
    return len(names) == 1 and len(consts) == 1


def _bound_names(nodes: list[ast.stmt]) -> set[str]:
    """文の列が束縛する名前 (代入・for / with の的・import・except の名前・def / class)。関数とクラスの中には入らない。"""
    out: set[str] = set()
    stack: list[ast.AST] = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _loaded_names(nodes: list[ast.stmt]) -> list[str]:
    """文の列が読む名前 (出現順、文の中で束縛した名前と組込みは除く)。"""
    bound = _bound_names(nodes)
    out: list[str] = []
    for stmt in nodes:
        for n in ast.walk(stmt):
            if (
                isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                and n.id not in bound and not hasattr(builtins, n.id) and n.id not in out
            ):
                out.append(n.id)
    return out


def synthesize_main(entry_path: str, entry_source: str, siblings: set[str] = frozenset()) -> str:
    """入口のモジュールから ``__main__.py`` を合成する (作れなければ空文字列)。

    入口の ``if __name__ == "__main__":`` の本体を写し、本体が読む名前を入口から import する
    (入口が ``__init__.py`` なら ``from . import main``、``cli.py`` なら ``from .cli import main``)。
    入口の直下の素の ``import sys`` はそのまま写し、それ以外 (try / for / with の中で束縛した名前、兄弟
    モジュール ``siblings``) は入口から取る。本体が無ければ ``main()`` を呼ぶ。本体の中の兄弟 import の
    相対化は :func:`package_code_map` が掛ける。
    """
    try:
        tree = ast.parse(entry_source)
    except (SyntaxError, ValueError):
        return ""
    guard = next((n for n in tree.body if is_main_guard(n)), None)
    rest = [n for n in tree.body if n is not guard]
    bound = _bound_names(rest)
    if guard is not None:
        body = list(guard.body)  # type: ignore[attr-defined]
    elif "main" in bound:
        body = [ast.Expr(ast.Call(ast.Name("main", ast.Load()), [], []))]
    else:
        return ""
    plain_imports: dict[str, str] = {}
    for node in rest:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in siblings:
                    plain_imports.setdefault(a.asname or a.name.split(".")[0], f"import {_alias(a.name, a.asname)}")
    # 素の import でも、後で別の値を束縛し直した名前は入口から取る
    rebound = _bound_names([n for n in rest if not isinstance(n, ast.Import)])
    stem = PurePosixPath(entry_path).stem
    source_module = "." if stem == "__init__" else f".{stem}"
    imports: list[str] = []
    from_entry: list[str] = []
    for name in _loaded_names(body):
        if name in plain_imports and name not in rebound:
            imports.append(plain_imports[name])
        elif name in bound:
            from_entry.append(name)
    code = textwrap.indent("\n".join(ast.unparse(stmt) for stmt in body), "    ")
    blocks = []
    if imports:
        blocks.append("\n".join(imports))
    if from_entry:
        blocks.append(f"from {source_module} import {', '.join(sorted(from_entry))}")
    blocks.append(f'if __name__ == "__main__":\n{code}')
    return "\n\n".join(blocks) + "\n"


def choose_entry(code_map: dict[str, str], declared: str) -> str:
    """入口のモジュール。骨組みの ``entry_module``、無ければ ``__main__`` ガードを持つモジュール、無ければ ``__init__.py``。"""
    if declared in code_map:
        return declared
    for path, source in code_map.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        if any(is_main_guard(n) for n in tree.body):
            return path
    return "__init__.py" if "__init__.py" in code_map else ""


def package_code_map(code_map: dict[str, str], entry: str, *, synthesize_main: bool = True) -> dict[str, str]:
    """平置きのコード (``length.py`` …) からパッケージの中身 (同じ名前 + ``__init__.py`` / ``__main__.py``) を作る。

    ``synthesize_main=False`` は ``__main__.py`` を合成しない (使い方が ``python -m units.cli`` の形)。
    """
    siblings = {PurePosixPath(p).stem for p in code_map if p.endswith(".py")}
    out = {p: (to_package_imports(c, siblings) if p.endswith(".py") else c) for p, c in code_map.items()}
    out.setdefault("__init__.py", "")
    if synthesize_main and "__main__.py" not in out and entry in code_map:
        # ガードの本体の兄弟 import (``from length import convert``) も相対にする
        main = to_package_imports(_synthesize_main(entry, code_map[entry], siblings), siblings)
        if main:
            out["__main__.py"] = main
    return out


_synthesize_main = synthesize_main


# ── 使い方の実行 ─────────────────────────────────────────────────────


def usage_blocker(args: list[str], code_map: dict[str, str], available: set[str]) -> str:
    """使い方を実行できない理由 (入力の置き場所・成果物に無いファイル・画面を開くモジュール)。無ければ空。

    ``available`` は実行するフォルダに置くファイル名 (依頼されたデータファイル)。ファイル名らしい引数と、区切り・
    ドライブ・``~`` を含む引数 (``--input=data.csv`` の値も) は ``available`` に無ければ入力とみなす。
    """
    for arg in args:
        if _PLACEHOLDER_RE.search(arg):
            return arg
        if arg.startswith("-") and "=" not in arg:
            continue
        value = arg.split("=", 1)[1] if arg.startswith("-") else arg
        # 区切り・ホーム・ドライブ (``E:x.txt``) を含む引数はパス
        pathish = "/" in value or "\\" in value or value.startswith("~") or (value[1:2] == ":" and value[:1].isalpha())
        if pathish or FILE_NAME_IN_TEXT_RE.fullmatch(value):
            if value not in available and PurePosixPath(value.replace("\\", "/")).name not in available:
                return arg
    for source in code_map.values():
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            heads = (
                [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import)
                else [(node.module or "").split(".")[0]] if isinstance(node, ast.ImportFrom) and not node.level
                else []
            )
            screen = next((h for h in heads if h in _SCREEN_MODULES), "")
            if screen:
                return screen
    return ""


def _last_exception(stderr: str) -> str:
    for line in reversed([ln.strip() for ln in stderr.splitlines() if ln.strip()]):
        match = _EXCEPTION_LINE_RE.match(line)
        if match:
            return line
    return ""


def _blamed_stem(stderr: str, package: str) -> str:
    """traceback の最後のパッケージ内のファイルの stem (無ければ空)。"""
    parts = [p for p in package.replace("\\", "/").split("/") if p]
    for path in reversed(_TRACE_FILE_RE.findall(stderr)):
        pure = PurePosixPath(path.replace("\\", "/"))
        if len(pure.parts) > len(parts) and list(pure.parts[-len(parts) - 1:-1]) == parts:
            return pure.stem
    return ""


def usage_outcome(
    run: "UsageRun", package: str, *, args: list[str] | None = None,
) -> tuple[CheckOutcome, str]:
    """使い方の実行結果を 3 値にする。戻り値は (結果, 作り直す stem — 無ければ空)。

    終了コード 0 かつ traceback が無ければ合格。隔離が書込みを止めた・時間切れ・``EOFError`` (対話) は未検査。
    ``args`` が空 (引数を渡していない) で traceback 無しの非 0 終了 (argparse の必須引数のエラー) も入力待ちの
    未検査。不合格なら traceback の最後の ``<package>/`` の中のファイルを作り直しの宛先に返す。
    """
    from backend.i18n_helper import msg

    if run.unchecked:
        return CheckOutcome.unchecked(
            CheckKind.USAGE, UncheckedReason.SANDBOX_VIOLATION, detail=", ".join(run.unchecked[:3]),
        ), ""
    if run.timed_out:
        return CheckOutcome.unchecked(
            CheckKind.USAGE, UncheckedReason.NOT_RUN,
            detail=msg("create.check.detail.timed_out", seconds=f"{run.timeout_sec:g}"),
        ), ""
    if run.returncode is None:
        return CheckOutcome.unchecked(CheckKind.USAGE, UncheckedReason.NOT_RUN, detail=run.error[:200]), ""
    stderr = run.stderr or ""
    has_traceback = _TRACEBACK_HEAD in stderr
    exception = _last_exception(stderr) if has_traceback else ""
    if exception.split(":", 1)[0].rsplit(".", 1)[-1] in ("EOFError", "KeyboardInterrupt"):
        return CheckOutcome.unchecked(CheckKind.USAGE, UncheckedReason.NEEDS_INPUT, detail=exception[:200]), ""
    if run.returncode == 0 and not has_traceback:
        return CheckOutcome.passed(CheckKind.USAGE), ""
    if args is not None and not args and not has_traceback:
        last = next((ln.strip() for ln in reversed(stderr.splitlines()) if ln.strip()), f"exit code {run.returncode}")
        return CheckOutcome.unchecked(CheckKind.USAGE, UncheckedReason.NEEDS_INPUT, detail=last[:200]), ""
    summary = exception or f"exit code {run.returncode}: {stderr.strip()[-200:]}"
    return CheckOutcome.failed(CheckKind.USAGE, errors=1, failures=[summary[:300]]), _blamed_stem(stderr, package)


__all__ = [
    "choose_entry",
    "is_main_guard",
    "package_code_map",
    "package_from_usage",
    "synthesize_main",
    "to_package_imports",
    "usage_blocker",
    "usage_command",
    "usage_outcome",
]
