"""create のパッケージ形 — 平置きのコードから配信形を決定論で作る (f_10 §11.1-1 / §11.1-3)。

staged v2 は Python を平置き (``src/<name>.py``、bare import) で生成・検査する。骨組みの使い方が
``python -m <共通の親>`` のときだけ、配信の直前にパッケージの形 (相対 import・``__main__.py``) に
変換する (2026-09-27 ライブ監査 K04: ``python -m units 10 km mi`` が ``ModuleNotFoundError``)。

引き金は使い方と共通の親の一致だけ — ``__init__.py`` の有無では決めない (平置きのアプリに付いた
余計な ``__init__.py`` で ``python main.py`` が壊れる)、``python -m http.server`` のような無関係な
モジュール名もパッケージにしない (反証レビュー S6)。使い方が ``python -m <パッケージ> <サブコマンド> …`` なら
``__main__.py`` は各モジュールの入口への振り分けを作る (2026-10-03 K04)。使い方の読み取りと実行結果の判定は
平置きの使い方 (``python main.py …``) にも使う。純粋関数だけを置く (実行は ``smoke_validator.run_usage``)。
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import keyword
import re
import sys
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
#: 別のプロセスが使用中で消せない (Windows の共有違反。実行フォルダ自身の削除で出る)。
_IN_USE_RE = re.compile(r"\[WinError 32\]")


# ── 引き金 ─────────────────────────────────────────────────────────────


def _split(line: str) -> list[str]:
    """コマンド行を語に分ける。``\\`` をエスケープとして読まない (Windows のパスを壊さない)、囲みの引用符は外す。"""
    try:
        tokens = shlex.split(line, posix=False)
    except ValueError:
        tokens = line.split()
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]


def _command_tokens(usage: str) -> list[str] | None:
    """使い方の起動コマンドの後ろ (インタプリタのオプションを除いた ``-m x …`` / ``main.py …``)。無ければ ``None``。

    起動コマンド (``python`` / ``py`` / ``C:\\Python312\\python.exe`` …) の語から後ろだけを読む — 前置き
    (``$ `` / ``PS> `` / バッククォート / 「コマンド例:」 / ``cd x &&``) は捨てる。起動コマンドのある最初の行を見る。
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
        return tokens[i:]
    return None


def _command_args(tokens: list[str]) -> list[str]:
    """引数の列から ``#`` のコメントと ``;`` の後ろを捨てる。"""
    args: list[str] = []
    for token in tokens:
        if token.startswith("#"):
            break
        if token.endswith(";"):
            if token[:-1]:
                args.append(token[:-1])
            break
        args.append(token)
    return args


def usage_command(usage: str) -> tuple[str, list[str]] | None:
    """使い方 ``python -m <モジュール> <引数…>`` を ``(モジュール, 引数)`` にする (その形でなければ ``None``)。

    前置きの扱いは :func:`_command_tokens`。引数は ``#`` のコメントと ``;`` の後ろを捨てる。
    """
    rest = _command_tokens(usage)
    if not rest or len(rest) < 2 or rest[0] != "-m":
        return None
    return rest[1], _command_args(rest[2:])


def script_command(usage: str) -> tuple[str, list[str]] | None:
    """使い方 ``python <ファイル>.py <引数…>`` を ``(ファイルのパス, 引数)`` にする (その形でなければ ``None``)。

    パスの区切りは ``/`` に揃える。前置きと引数の扱いは :func:`usage_command` と同じ。
    """
    rest = _command_tokens(usage)
    if not rest or not rest[0].lower().endswith(".py"):
        return None
    return rest[0].replace("\\", "/"), _command_args(rest[1:])


#: 依頼文の ASCII の連なりの前後から外す囲みと句読点 (``(python -m x)`` / ``python -m x.``)。
_RUN_EDGE_CHARS = " \t\"'([{)]}.,;:!?"
#: 依頼文のバッククォートで囲んだ 1 行の中身。
_QUOTED_COMMAND_RE = re.compile(r"`([^`\n]+)`")
#: コマンドの後ろの説明の始まりとみなす語 (結果を示す矢印)。
_PROSE_ARROWS = frozenset({"->", "=>", "→", "⇒"})


def _until_prose(args: list[str]) -> list[str]:
    """引数の列を、2 番目以降で説明の文 (ASCII 以外の文字を含む語・矢印) が始まる手前で打ち切る。

    最初の引数 (サブコマンド ``追加`` のような和語もありうる) は残す。
    """
    out: list[str] = []
    for i, arg in enumerate(args):
        if i >= 1 and (arg in _PROSE_ARROWS or not arg.isascii()):
            break
        out.append(arg)
    return out


def usage_from_request(text: str, folder: str = "") -> str:
    """依頼文が書いた ``python -m <モジュール>`` (無ければ空)。骨組みの ``usage`` が空のときの代わり。

    字句の鍵 (語彙ではなく文字の種類で切る): 行をバッククォートと ASCII 以外の文字で切った連なりから
    ``python -m <モジュール>`` を拾う。引数は採らない — 依頼文では使い方の後ろの文 (``-> 6.2 mi と出る`` /
    英文の続き) と区別できず、誤った引数で使い方の実行が落ちると正しいモジュールを作り直してしまう。
    モジュール名の前後の引用符・句読点は剥がす。複数あれば ``folder`` (骨組みの共通の親) で
    パッケージ形が成り立つものを優先し、無ければ最初のもの。

    例外はバッククォートで囲んだコマンド (`` `python -m textkit count file.txt` ``) — 囲みの中がコマンドだけ
    (起動コマンドから始まる) なら後ろの文と混ざらないので引数も採る (2026-10-03 K04: サブコマンド ``count`` が
    分からず ``__main__.py`` を補完できなかった)。囲みの中でもコマンドの後ろに説明が続くこと
    (`` `python -m textkit count file.txt で行数を数える` `` / `` `python -m units 10 km mi -> 6.21` ``) があるので、
    2 番目以降の引数に ASCII 以外の文字を含む語・矢印 (``->`` / ``=>``) が出たらそこで打ち切る (独立レビュー 中 2)。
    """
    quoted: dict[str, str] = {}
    for span in _QUOTED_COMMAND_RE.findall(str(text or "")):
        tokens = _split(span.strip())
        command = usage_command(span) if tokens and _PYTHON_EXE_RE.match(PureWindowsPath(tokens[0]).name) else None
        if command and all(p.isidentifier() for p in command[0].split(".")) and command[0] not in quoted:
            # 空白を含む引数だけ二重引用符で囲む (:func:`_split` が外す。和語・Windows のパスは囲まない)
            quoted[command[0]] = " ".join(
                f'"{a}"' if any(c.isspace() for c in a) else a
                for a in ["python", "-m", command[0], *_until_prose(command[1])]
            )
    modules: list[str] = []
    for line in str(text or "").splitlines():
        run = ""
        for ch in line + "`":
            if ch.isascii() and ch != "`":
                run += ch
                continue
            tokens = [t.strip(_RUN_EDGE_CHARS) for t in run.split()]
            run = ""
            # 1 つの連なりに起動コマンドが複数 (英文の「… python -m pytest … python -m textkit」) でも全部拾う
            for i, token in enumerate(tokens):
                if not _PYTHON_EXE_RE.match(PureWindowsPath(token).name):
                    continue
                command = usage_command(" ".join(tokens[i:]))
                module = command[0].strip(_RUN_EDGE_CHARS) if command else ""
                if module and all(p.isidentifier() for p in module.split(".")) and module not in modules:
                    modules.append(module)
    usages = [quoted.get(m) or f"python -m {m}" for m in modules]
    return next((u for u in usages if package_from_usage(u, folder)), usages[0] if usages else "")


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


# ── サブコマンドの振り分け (``python -m textkit count file.txt``) ─────────────

#: サブコマンドのモジュールの入口として呼ぶ関数の名前 (この順に探す)。
SUBCOMMAND_ENTRY_NAMES = ("main", "run", "cli")
#: 骨組みがサブコマンドのモジュールに足す入口 (argv はサブコマンドより後ろの引数、戻り値は終了コード)。
SUBCOMMAND_ENTRY_SIGNATURE = "def main(argv: list[str] | None = None) -> int"


def subcommand_modules(usage: str, package: str, modules: list[dict], text: str = "") -> list[str]:
    """使い方 ``python -m <package> <サブコマンド> …`` のサブコマンドに当たるモジュールの stem (先頭が使い方のもの)。

    使い方がパッケージそのものを起動し、最初の引数がパッケージのモジュールの stem のときだけ (それ以外は空)。
    残りは骨組みで他のモジュールから使われない (``imports_from`` に現れない) モジュール — 共通の部品
    (``utils.py``) はサブコマンドにしない。``__init__`` / ``__main__`` は数えない。モジュール名で決める (語彙は見ない)。
    骨組みがどのモジュールにも ``imports_from`` を書いていない (使われる・使われないが分からない) ときは、
    使い方か ``text`` (依頼文) に名前が語として出るものだけを残す (独立レビュー 低 a: ``utils`` に入口を強いない)。
    """
    command = usage_command(usage)
    if command is None or not package or command[0] != package.replace("/", ".") or not command[1]:
        return []
    stems = [
        PurePosixPath(m["path"]).stem for m in modules
        if str(m.get("path") or "").endswith(".py") and PurePosixPath(m["path"]).stem not in ("__init__", "__main__")
        and PurePosixPath(m["path"]).stem.isidentifier() and not keyword.iskeyword(PurePosixPath(m["path"]).stem)
    ]
    first = command[1][0]
    if first not in stems:
        return []
    used = {PurePosixPath(str(p)).stem for m in modules for p in m.get("imports_from") or []}
    others = [s for s in stems if s != first and s not in used]
    if not used:
        named = f"{usage}\n{text}"
        others = [s for s in others if re.search(rf"(?<![A-Za-z0-9_]){re.escape(s)}(?![A-Za-z0-9_])", named)]
    return [first, *others]


def subcommand_entry(source: str) -> tuple[str, bool] | None:
    """サブコマンドのモジュールの入口 ``(関数名, argv を渡すか)``。呼べる入口が無ければ ``None``。

    モジュール直下の ``main`` / ``run`` / ``cli`` (この順) のうち、引数 1 つ (argv) で呼べるもの、または引数を
    取らないもの (``sys.argv`` を自分で読む) を採る。必須の引数が 2 つ以上・必須のキーワード専用引数があれば呼べない。
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in SUBCOMMAND_ENTRY_NAMES:
        fn = defs.get(name)
        if fn is None:
            continue
        a = fn.args
        positional = [*a.posonlyargs, *a.args]
        required = len(positional) - len(a.defaults)
        if required > 1 or any(d is None for d in a.kw_defaults):
            continue
        return name, bool(positional) or a.vararg is not None
    return None


def synthesize_dispatcher(code_map: dict[str, str], subcommands: list[str], prog: str) -> str:
    """サブコマンドを各モジュールの入口へ振り分ける ``__main__.py`` (作れなければ空文字列)。

    使い方のサブコマンド (``subcommands[0]``) のモジュールに入口 (:func:`subcommand_entry`) が無ければ作らない —
    入口を推し量った呼出しで使い方の実行を通したように見せない。ほかのサブコマンドは入口のあるものだけを
    振り分けに入れる (入口の無いものは :func:`unwired_subcommands` が挙げる)。入口が argv を取るならサブコマンドより
    後ろの引数を渡し、取らなければ ``sys.argv`` をその形にして呼ぶ。戻り値が int ならそれを終了コードにする。
    """
    entries = {s: subcommand_entry(code_map.get(f"{s}.py", "")) for s in subcommands}
    if not subcommands or entries[subcommands[0]] is None:
        return ""
    wired = [(s, e) for s, e in entries.items() if e is not None]
    table = "\n".join(f"    {s!r}: ({name!r}, {takes_argv})," for s, (name, takes_argv) in wired)
    choices = ",".join(s for s, _ in wired)
    # サブコマンドのモジュールは名前で束縛せず使うときに import する (``sys`` という名前のサブコマンドが標準の
    # sys を上書きしない、独立レビュー 低 c)
    return (
        "import importlib as _importlib\n"
        "import sys as _sys\n\n"
        f"_COMMANDS = {{\n{table}\n}}\n\n\n"
        "def _dispatch(argv):\n"
        "    if not argv or argv[0] not in _COMMANDS:\n"
        f"        print({f'usage: python -m {prog} {{{choices}}} ...'!r}, file=_sys.stderr)\n"
        "        return 2\n"
        "    name, rest = argv[0], argv[1:]\n"
        "    func, takes_argv = _COMMANDS[name]\n"
        "    entry = getattr(_importlib.import_module(\".\" + name, __package__), func)\n"
        "    if takes_argv:\n"
        "        result = entry(rest)\n"
        "    else:\n"
        "        _sys.argv = [f\"{_sys.argv[0]} {name}\", *rest]\n"
        "        result = entry()\n"
        "    return result if type(result) is int else 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    _sys.exit(_dispatch(_sys.argv[1:]))\n"
    )


def unwired_subcommands(code_map: dict[str, str], subcommands: list[str]) -> list[str]:
    """入口 (:func:`subcommand_entry`) が無く振り分けられないサブコマンドのモジュールの stem。"""
    return [s for s in subcommands if subcommand_entry(code_map.get(f"{s}.py", "")) is None]


def package_code_map(
    code_map: dict[str, str], entry: str, *, synthesize_main: bool = True,
    subcommands: list[str] | None = None, prog: str = "",
) -> dict[str, str]:
    """平置きのコード (``length.py`` …) からパッケージの中身 (同じ名前 + ``__init__.py`` / ``__main__.py``) を作る。

    ``synthesize_main=False`` は ``__main__.py`` を合成しない (使い方が ``python -m units.cli`` の形)。
    ``subcommands`` (:func:`subcommand_modules`) があれば、まず各モジュールの入口への振り分け
    (:func:`synthesize_dispatcher`) を作り、作れなければ入口のモジュールのガードから合成する。
    """
    siblings = {PurePosixPath(p).stem for p in code_map if p.endswith(".py")}
    out = {p: (to_package_imports(c, siblings) if p.endswith(".py") else c) for p, c in code_map.items()}
    out.setdefault("__init__.py", "")
    if synthesize_main and "__main__.py" not in out:
        main = synthesize_dispatcher(code_map, list(subcommands or []), prog) if subcommands else ""
        if not main and entry in code_map:
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


#: import したら使い方を実行しないモジュール — 隔離 (f_10 §11.1-4) が止めない外部への作用 (別プロセス・シェル・
#: ネットワークの接続や待受・ブラウザ・ネイティブ呼出し・サーバのフレームワーク)。``a.b`` は ``a.b`` とその下。
_SIDE_EFFECT_MODULES = frozenset({
    "subprocess", "socket", "socketserver", "http.server", "http.client", "urllib.request", "xmlrpc",
    "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "ssl", "webbrowser", "ctypes", "winreg", "_winapi",
    "multiprocessing", "requests", "httpx", "aiohttp", "urllib3", "paramiko", "websocket", "websockets",
    "flask", "fastapi", "uvicorn", "django", "bottle", "tornado", "pyautogui", "pynput",
})
#: ``os.<名前>(…)`` で別プロセス・シェルを起こす / プロセスを止める呼出し (前方一致は exec* / spawn* / posix_spawn*)。
_OS_SIDE_EFFECT_CALLS = frozenset({"system", "popen", "startfile", "kill", "killpg", "fork", "forkpty"})
_OS_SIDE_EFFECT_PREFIXES = ("exec", "spawn", "posix_spawn")
#: どのオブジェクトの属性でも待受・画面を開く呼出し (``asyncio.start_server`` / ``serve_forever`` / ``plt.show``)。
_SERVE_CALLS = frozenset({"start_server", "create_server", "serve_forever", "run_forever"})


def _module_matches(name: str, roots: frozenset[str]) -> str:
    return next((r for r in roots if name == r or name.startswith(r + ".")), "")


def usage_side_effect(code_map: dict[str, str]) -> str:
    """使い方を実行すると隔離の外へ作用する恐れのある書き方 (最初の 1 つ、無ければ空)。

    隔離 (f_10 §11.1-4) は Python の中のファイル書込みしか止めない — シェル (``os.system('echo > 外')``)・
    別プロセス・ソケットの接続や待受・ブラウザ・ネイティブ呼出しは通る (独立レビュー 高)。それらを import する・
    呼ぶ成果物は実行しない (未検査)。``matplotlib.pyplot`` は ``show()`` を呼ぶときだけ (画像の保存は実行する)。
    字句の鍵 (AST の import と呼出し) だけを見る — 語彙の判定ではない。
    """
    for path, source in code_map.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        os_names: set[str] = set()
        pyplot_names: set[str] = set()
        for node in ast.walk(tree):
            names: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                names = [(a.name, a.asname or a.name.split(".")[0]) for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [(f"{node.module}.{a.name}", a.asname or a.name) for a in node.names]
                names.append((node.module, ""))
            for full, bound in names:
                hit = _module_matches(full, _SIDE_EFFECT_MODULES)
                if hit:
                    return hit
                if isinstance(node, ast.Import) and full.split(".")[0] == "os":
                    # ``import os`` / ``import os as o`` / ``import os.path`` (``os`` を束縛する)
                    os_names.add(bound)
                elif isinstance(node, ast.ImportFrom) and full.startswith("os.") and full.count(".") == 1:
                    attr = full.split(".", 1)[1]
                    if attr in _OS_SIDE_EFFECT_CALLS or attr.startswith(_OS_SIDE_EFFECT_PREFIXES):
                        return full
                elif full == "matplotlib.pyplot" and bound:
                    pyplot_names.add(bound)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attr = node.func.attr
            base = node.func.value
            if isinstance(base, ast.Name) and base.id in os_names and (
                attr in _OS_SIDE_EFFECT_CALLS or attr.startswith(_OS_SIDE_EFFECT_PREFIXES)
            ):
                return f"os.{attr}"
            if attr in _SERVE_CALLS:
                return attr
            if attr == "show" and isinstance(base, ast.Name) and base.id in pyplot_names:
                return "matplotlib.pyplot.show"
    return ""


def usage_missing_dependency(code_map: dict[str, str]) -> str:
    """成果物が import する外部依存のうち、この環境に入っていないもの (最初の 1 つ、無ければ空)。

    兄弟モジュール・標準ライブラリ・相対 import は見ない。入っていない依存で使い方を実行すると
    ``ModuleNotFoundError`` の不合格になり、正しいコードを作り直してしまう (環境の欠けでコードの欠陥ではない)。
    """
    siblings = {PurePosixPath(p).stem for p in code_map if p.endswith(".py")}
    for path, source in code_map.items():
        if not path.endswith(".py"):
            continue
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
            for head in heads:
                if not head or head in siblings or head in sys.stdlib_module_names or head == "__future__":
                    continue
                try:
                    found = importlib.util.find_spec(head) is not None
                except (ImportError, ValueError):
                    found = False
                if not found:
                    return head
    return ""


def _last_exception(stderr: str) -> str:
    for line in reversed([ln.strip() for ln in stderr.splitlines() if ln.strip()]):
        match = _EXCEPTION_LINE_RE.match(line)
        if match:
            return line
    return ""


def _blamed_stem(stderr: str, package: str, stems: set[str] | None = None) -> str:
    """traceback の最後のパッケージ内のファイルの stem (無ければ空)。

    平置き (``package`` が空) は ``stems`` (成果物のモジュール) に入る stem のファイルだけを見る (標準ライブラリの
    ``json/decoder.py`` を宛先にしない)。
    """
    parts = [p for p in package.replace("\\", "/").split("/") if p]
    for path in reversed(_TRACE_FILE_RE.findall(stderr)):
        pure = PurePosixPath(path.replace("\\", "/"))
        if not parts:
            if pure.suffix == ".py" and pure.stem in (stems or set()):
                return pure.stem
            continue
        if len(pure.parts) > len(parts) and list(pure.parts[-len(parts) - 1:-1]) == parts:
            return pure.stem
    return ""


def usage_outcome(
    run: "UsageRun", package: str, *, args: list[str] | None = None, stems: set[str] | None = None,
) -> tuple[CheckOutcome, str]:
    """使い方の実行結果を 3 値にする。戻り値は (結果, 作り直す stem — 無ければ空)。

    終了コード 0 かつ traceback が無ければ合格。隔離が書込みを止めた・時間切れ・``EOFError`` (対話) は未検査。
    ``args`` が空 (引数を渡していない) で traceback 無しの非 0 終了 (argparse の必須引数のエラー・「入力ファイルが
    ありません」で終わる) も入力待ちの未検査で、理由に最後の出力の行 (stderr、無ければ stdout) を添える。
    不合格なら traceback の最後の ``<package>/`` の中のファイル (平置きは ``stems`` のファイル) を作り直しの宛先に返す。
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
    if _IN_USE_RE.search(exception):
        # 実行フォルダ自身 (CWD・実行中のファイル) を消そうとして掴まれていた (``shutil.rmtree('.')``) — 実行の場の
        # 制約でコードの欠陥ではない。不合格にして正しいコードを作り直さない (独立レビュー 中 1)
        return CheckOutcome.unchecked(CheckKind.USAGE, UncheckedReason.NOT_RUN, detail=exception[:200]), ""
    if args is not None and not args and not has_traceback:
        lines = [ln.strip() for ln in (stderr or run.stdout or "").splitlines() if ln.strip()]
        last = lines[-1] if lines else f"exit code {run.returncode}"
        return CheckOutcome.unchecked(CheckKind.USAGE, UncheckedReason.NEEDS_INPUT, detail=last[:200]), ""
    summary = exception or f"exit code {run.returncode}: {stderr.strip()[-200:]}"
    return CheckOutcome.failed(CheckKind.USAGE, errors=1, failures=[summary[:300]]), _blamed_stem(
        stderr, package, stems,
    )


__all__ = [
    "SUBCOMMAND_ENTRY_NAMES",
    "SUBCOMMAND_ENTRY_SIGNATURE",
    "choose_entry",
    "is_main_guard",
    "package_code_map",
    "package_from_usage",
    "script_command",
    "subcommand_entry",
    "subcommand_modules",
    "synthesize_dispatcher",
    "synthesize_main",
    "to_package_imports",
    "unwired_subcommands",
    "usage_blocker",
    "usage_command",
    "usage_from_request",
    "usage_missing_dependency",
    "usage_outcome",
    "usage_side_effect",
]
