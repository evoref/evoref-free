"""as-built 文書 — 完成したコードと骨組みから SPEC.md / flowchart.md を決定論で描く (f_10 §11)。

staged v2 は仕様書を LLM に書かせない。仕様書は **出来上がったコード** (AST のシグネチャと
docstring) と骨組み (契約) から組み立てるので、実物とずれない。LLM のトークンも使わない。
純粋関数だけを置く (I/O なし)。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

_HEADINGS = {
    "ja": {
        "note": "この仕様書は完成したコードから自動で作成しました (as-built)。",
        "request": "依頼", "usage": "使い方", "modules": "モジュール",
        "examples": "入出力の例", "verification": "検証", "entry": "エントリポイント",
        "flow_title": "設計フローチャート", "start": "開始", "unverified": "(未検証)",
    },
    "en": {
        "note": "This specification was generated from the finished code (as-built).",
        "request": "Request", "usage": "Usage", "modules": "Modules",
        "examples": "Examples", "verification": "Verification", "entry": "Entry point",
        "flow_title": "Design flowchart", "start": "Start", "unverified": "(unverified)",
    },
}


@dataclass(frozen=True)
class PublicSymbol:
    """モジュールの公開要素 (関数 / クラス / クラスの公開メソッド)。"""

    signature: str
    doc: str = ""
    methods: tuple["PublicSymbol", ...] = field(default_factory=tuple)


def _first_doc_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node) or ""
    for line in doc.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({args}){ret}"


def public_symbols(source: str) -> list[PublicSymbol]:
    """ソースの公開要素を宣言順で返す (構文エラーなら空)。"""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    out: list[PublicSymbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            out.append(PublicSymbol(_signature(node), _first_doc_line(node)))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = tuple(
                PublicSymbol(_signature(m), _first_doc_line(m))
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (not m.name.startswith("_") or m.name == "__init__")
            )
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
            out.append(PublicSymbol(sig, _first_doc_line(node), methods))
    return out


def internal_imports(code_map: dict[str, str]) -> dict[str, list[str]]:
    """成果物内のモジュール間 import (モジュールのパス → import 先のパス)。"""
    stems = {PurePosixPath(p).stem: p for p in code_map if p.endswith(".py")}
    edges: dict[str, list[str]] = {}
    for path, source in code_map.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        targets: list[str] = []
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module.split(".")[0]]
            for name in names:
                dest = stems.get(name)
                if dest and dest != path and dest not in targets:
                    targets.append(dest)
        edges[path] = targets
    return edges


#: 値を取らない ``add_argument`` の action
_FLAG_ACTIONS = frozenset({"store_true", "store_false", "store_const", "count", "help", "version", "append_const"})
#: 引数を持ち主のパーサへそのまま足すグループ
_GROUP_METHODS = ("add_argument_group", "add_mutually_exclusive_group")
_SCOPE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _call_on(node: ast.AST, method: str) -> str:
    """``<名前>.<method>(…)`` の呼出しならその名前 (それ以外は空)。"""
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
    ):
        return node.func.value.id
    return ""


def _kw(call: ast.Call, name: str) -> object:
    for k in call.keywords:
        if k.arg == name and isinstance(k.value, ast.Constant):
            return k.value.value
    return None


def _argument_text(call: ast.Call) -> str:
    """``add_argument(…)`` 1 個の使い方の表記 (``<amount>`` / ``[--limit <limit>]`` / ``[--verbose]``)。"""
    names = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not names:
        return ""
    metavar = _kw(call, "metavar")
    nargs = _kw(call, "nargs")
    if not names[0].startswith("-"):
        value = f"<{metavar or names[0]}>"
        if nargs == "?":
            return f"[{value}]"
        if nargs == "*":
            return f"[{value} ...]"
        return f"{value} ..." if nargs == "+" else value
    flag = max(names, key=len)
    action = next((k.value for k in call.keywords if k.arg == "action"), None)
    if action is not None and ast.unparse(action).rsplit(".", 1)[-1] == "BooleanOptionalAction":
        text = f"{flag} | --no-{flag.lstrip('-')}" if flag.startswith("--") else flag
    elif _kw(call, "action") in _FLAG_ACTIONS:
        text = flag
    else:
        dest = _kw(call, "dest") or flag.lstrip("-").replace("-", "_")
        text = f"{flag} <{metavar or dest}>"
    return text if _kw(call, "required") is True else f"[{text}]"


def _scope_nodes(scope: ast.AST) -> list[ast.AST]:
    """スコープ (モジュール / 関数) の中の節を、入れ子の関数・クラスの中を除いてソース順で返す。"""
    out: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPE_TYPES):
            continue
        out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return sorted(
        (n for n in out if hasattr(n, "lineno")), key=lambda n: (n.lineno, n.col_offset),
    )


def _is_argument_parser(value: ast.AST) -> bool:
    return isinstance(value, ast.Call) and ast.unparse(value.func).rsplit(".", 1)[-1] == "ArgumentParser"


def _top_parser(tree: ast.Module) -> tuple[ast.AST, str] | None:
    """使い方を読むパーサ (スコープ, 変数名)。

    ``parse_args`` / ``parse_known_args`` を同じスコープで呼ぶ ``ArgumentParser`` を採る。無ければ、``ArgumentParser`` を
    組んで返す関数 (``build_parser``) が 1 つだけならそれ、``ArgumentParser`` を組むスコープが 1 つだけならその最初の
    パーサ。決められなければ ``None`` (ファイル内の別のパーサ — 設定の読み込み・テスト用ヘルパ — を混ぜない)。
    """
    scopes = [tree, *(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))]
    built: list[tuple[ast.AST, list[str], list[ast.AST]]] = []
    for scope in scopes:
        nodes = _scope_nodes(scope)
        names = [
            n.targets[0].id for n in nodes
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and _is_argument_parser(n.value)
        ]
        if names:
            built.append((scope, names, nodes))
    for scope, names, nodes in built:
        for n in nodes:
            owner = _call_on(n, "parse_args") or _call_on(n, "parse_known_args")
            if owner in names:
                return scope, owner
    returning = [
        (scope, next(n.value.id for n in nodes if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)
                     and n.value.id in names))
        for scope, names, nodes in built
        if any(isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id in names for n in nodes)
    ]
    if len(returning) == 1:
        return returning[0]
    if len(built) == 1:
        return built[0][0], built[0][1][0]
    return None


def argparse_commands(source: str) -> list[tuple[str, str]]:
    """コードの argparse から ``(サブコマンドの経路, 引数の表記)`` の列を宣言順で返す (無ければ空)。

    骨組みの ``usage`` は 1 行でサブコマンドの 1 つしか書かないことがある (2026-10-03 ライブ再実行 K01:
    ``add_expense`` だけで ``category_summary`` / ``monthly_list`` が SPEC に載らなかった)。使い方は実コードの
    ``ArgumentParser`` / ``add_subparsers`` / ``add_parser`` / ``add_argument`` から決定論で読む。

    - 読むのは 1 個のトップのパーサ (:func:`_top_parser`) とそのスコープの中だけ。別の関数の
      ``ArgumentParser`` (設定用・テスト用ヘルパ) の引数を混ぜない (独立レビュー 中1)。
    - 入れ子のサブコマンド (``db`` → ``migrate``) は経路を連ねて ``"db migrate"`` とし、葉だけを返す。
      親のパーサの引数は子の名前の前に置く。
    - ``add_argument_group`` / ``add_mutually_exclusive_group`` の引数は持ち主のパーサに足す。
    - 名前に束縛したパーサだけを追う (ループ・辞書で組むパーサ、click 等は読まない)。サブコマンドが無ければ
      ``("", 引数)`` を 1 件。``-h`` は argparse が足すので書かない。
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found = _top_parser(tree)
    if found is None:
        return []
    scope, top = found
    parsers: dict[str, tuple[str, ...]] = {top: ()}  # 変数名 → サブコマンドの経路
    subparsers: dict[str, tuple[str, ...]] = {}  # add_subparsers の戻り値 → 親の経路
    args: dict[tuple[str, ...], list[str]] = {(): []}
    for node in _scope_nodes(scope):
        if not isinstance(node, (ast.Assign, ast.Expr)):
            continue
        value = node.value
        target = node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else ""
        if target and (owner := _call_on(value, "add_subparsers")) in parsers:
            subparsers[target] = parsers[owner]
        elif target and (owner := next((o for m in _GROUP_METHODS if (o := _call_on(value, m))), "")) in parsers:
            parsers[target] = parsers[owner]
        elif (
            (owner := _call_on(value, "add_parser")) in subparsers
            and value.args and isinstance(value.args[0], ast.Constant)
        ):
            path = (*subparsers[owner], str(value.args[0].value))
            args.setdefault(path, [])
            if target:
                parsers[target] = path
        elif (owner := _call_on(value, "add_argument")) in parsers:
            text = _argument_text(value)
            if text and text not in ("[-h]", "[--help]"):
                args[parsers[owner]].append(text)
    paths = [p for p in args if p]
    if not paths:
        return [("", " ".join(args[()]))] if args[()] else []
    leaves = [p for p in paths if not any(q[: len(p)] == p and len(q) > len(p) for q in paths)]
    out: list[tuple[str, str]] = []
    for path in leaves:
        words = list(args[()])
        for i, name in enumerate(path):
            words += [name, *args.get(path[: i + 1], [])]
        out.append((" ".join(path), " ".join(words)))
    return out


def _command_prefix(usage: str, entry: str) -> str:
    """骨組みの使い方から起動の部分 (``python cli.py`` / ``python -m units``) を取る。

    無ければ ``python <entry>``。入口が ``__main__.py`` (パッケージの入口) なら、起動の部分は骨組みの使い方からしか
    決められない (``python __main__.py`` とは書かない) — 使い方が無ければ空。
    """
    for raw in usage.splitlines():
        tokens = raw.replace("`", " ").split()
        for i, token in enumerate(tokens):
            if token == "-m" and i + 1 < len(tokens):
                return " ".join(tokens[: i + 2])
            if token.endswith(".py"):
                return " ".join(tokens[: i + 1])
    if not entry or PurePosixPath(entry).name == "__main__.py":
        return ""
    return f"python {entry}"


def _entry_source(entry: str, code_map: dict[str, str]) -> str:
    """入口のソース。完全一致のパス → 末尾がそのパスに一致する 1 件 → 同名のファイルが 1 件だけ、の順。

    同名のファイルが複数あって決められなければ空 (別のモジュールの argparse を入口として読まない)。
    """
    if entry in code_map:
        return code_map[entry]
    suffix = [c for p, c in code_map.items() if p.endswith("/" + entry)]
    if len(suffix) == 1:
        return suffix[0]
    same = [c for p, c in code_map.items() if PurePosixPath(p).name == PurePosixPath(entry).name]
    return same[0] if len(same) == 1 else ""


def supplemental_usages(usage: str, entry: str, code_map: dict[str, str]) -> list[str]:
    """骨組みの使い方に無いサブコマンドの使い方を、入口のコードの argparse から足す行の列。

    骨組みの使い方の語に経路の名前がすべて出るサブコマンドは足さない (骨組みの具体的な例を残す)。骨組みに使い方が
    無く入口に argparse があれば、引数の表記を 1 行足す。入口は配信形のパス (``units/cli.py``) でも引く。
    入口に argparse が無い (``__main__.py`` が ``main(argv)`` へ振り分ける等)・入口が決められないときは空。
    """
    if not entry:
        return []
    commands = argparse_commands(_entry_source(entry, code_map))
    prefix = _command_prefix(usage, entry)
    if not commands or not prefix:
        return []
    words = set(usage.replace("`", " ").split())
    if commands[0][0] == "":
        return [] if usage.strip() else [f"{prefix} {commands[0][1]}".rstrip()]
    return [f"{prefix} {text}" for path, text in commands if not set(path.split()) <= words]


def subcommand_usages(usage: str, prog: str, sources: dict[str, str]) -> list[str]:
    """``python -m <prog> <サブコマンド>`` の振り分け (f_10 §11.1-1 (b)) が呼ぶサブコマンドの使い方の行。

    ``sources`` はサブコマンドの名前 → そのモジュールのソース (振り分けに入ったものだけ、宣言順)。骨組みの使い方の
    語に名前が出るサブコマンドは足さない。モジュールの argparse にトップのパーサがあれば引数の表記を添える
    (振り分けの ``__main__.py`` は argparse を持たないので :func:`supplemental_usages` からは引けない)。
    """
    if not prog:
        return []
    words = set(usage.replace("`", " ").split())
    out = []
    for name, source in sources.items():
        if name in words:
            continue
        commands = argparse_commands(source)
        text = commands[0][1] if commands and commands[0][0] == "" else ""
        out.append(f"python -m {prog} {name} {text}".rstrip())
    return out


def _t(locale: str) -> dict[str, str]:
    return _HEADINGS["ja" if str(locale).startswith("ja") else "en"]


def render_spec(
    *,
    skeleton: dict,
    code_map: dict[str, str],
    request: str,
    locale: str = "ja",
    verification: list[str] | None = None,
    other_facts: dict[str, list[str]] | None = None,
    examples_verified: bool = True,
    extra_usages: list[str] | None = None,
) -> str:
    """as-built の SPEC.md 本文を返す。

    ``extra_usages`` は使い方の節に足す行 (サブコマンドの振り分けの使い方、:func:`subcommand_usages`)。

    Python 以外のファイル (f_10 §12.5) は呼出し側が抽出した事実 (``other_facts``: id・参照先・
    export・テーブル等) をそのまま並べる。``examples_verified`` が偽なら、入出力の例は契約テストで例ごとに
    確かめていない骨組みの例なので、各行に「(未検証)」を添える (f_10 §11.1-4)。
    """
    t = _t(locale)
    roles = {m.get("path", ""): m.get("role", "") for m in skeleton.get("modules") or []}
    summary = (skeleton.get("summary") or "").strip()
    lines = [f"# {summary or t['modules']}", "", f"> {t['note']}", "", f"## {t['request']}", "", request.strip(), ""]
    usage = (skeleton.get("usage") or "").strip()
    entry = (skeleton.get("entry_module") or "").strip()
    if usage or entry:
        lines += [f"## {t['usage']}", ""]
        if entry:
            lines.append(f"- {t['entry']}: `{entry}`")
        if usage:
            lines.append(f"- `{usage}`" if "\n" not in usage else usage)
        lines += [f"- `{line}`" for line in supplemental_usages(usage, entry, code_map)]
        lines += [f"- `{line}`" for line in extra_usages or []]
        lines.append("")
    lines += [f"## {t['modules']}", ""]
    for path in sorted(code_map):
        if not path.endswith(".py"):
            if other_facts is None or path not in other_facts:
                continue
            role = roles.get(path, "")
            lines.append(f"### `{path}`" + (f" — {role}" if role else ""))
            lines.append("")
            lines += [f"- {fact}" for fact in other_facts[path]] or ["- -"]
            lines.append("")
            continue
        role = roles.get(path) or next((r for p, r in roles.items() if PurePosixPath(p).name == PurePosixPath(path).name), "")
        lines.append(f"### `{path}`" + (f" — {role}" if role else ""))
        lines.append("")
        symbols = public_symbols(code_map[path])
        if not symbols:
            lines.append("- (公開要素なし)" if t is _HEADINGS["ja"] else "- (no public symbols)")
        for sym in symbols:
            lines.append(f"- `{sym.signature}`" + (f" — {sym.doc}" if sym.doc else ""))
            for m in sym.methods:
                lines.append(f"  - `{m.signature}`" + (f" — {m.doc}" if m.doc else ""))
        lines.append("")
    examples = [e for e in skeleton.get("examples") or [] if e.get("call")]
    if examples:
        lines += [f"## {t['examples']}", ""]
        mark = "" if examples_verified else f" {t['unverified']}"
        for e in examples:
            lines.append(f"- `{e.get('call')}` → `{e.get('expected')}`{mark}")
        lines.append("")
    if verification:
        lines += [f"## {t['verification']}", ""]
        lines += [f"- {v}" for v in verification]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_NODE_ID_RE = re.compile(r"[^0-9A-Za-z_]")


def _node_id(path: str) -> str:
    return "m_" + _NODE_ID_RE.sub("_", path)


def _label(text: str) -> str:
    return text.replace('"', "'")


def render_flowchart(
    *, skeleton: dict, code_map: dict[str, str], locale: str = "ja",
    references: dict[str, list[str]] | None = None,
) -> str:
    """as-built の mermaid 本文を返す (``flowchart TD`` から。フェンスは付けない)。

    ノードはモジュール (公開要素の名前を添える)、辺は成果物内の import。Python 以外は
    呼出し側が抽出した参照 (``references``: script / link / import / require、f_10 §12.5) を辺にする。
    エントリポイントがあれば開始ノードからつなぐ。
    """
    t = _t(locale)
    py = sorted(p for p in code_map if p.endswith(".py"))
    others = sorted(p for p in (references or {}) if p in code_map)
    lines = ["flowchart TD"]
    for path in others:
        lines.append(f'    {_node_id(path)}["{_label(path)}"]')
    for path in py:
        names = [s.signature.split("(")[0].replace("def ", "").replace("class ", "") for s in public_symbols(code_map[path])]
        shown = ", ".join(n.strip() for n in names[:4]) + (" …" if len(names) > 4 else "")
        # パスのまま (パッケージ形は ``units/length.py``、平置きはファイル名と同じ)
        label = path + (f"<br/>{shown}" if shown else "")
        lines.append(f'    {_node_id(path)}["{_label(label)}"]')
    entry = (skeleton.get("entry_module") or "").strip()
    nodes = py + others
    entry_path = next((p for p in nodes if p == entry or PurePosixPath(p).name == PurePosixPath(entry).name), "") if entry else ""
    if not entry_path and len(nodes) == 1:
        entry_path = nodes[0]
    if entry_path:
        lines.append(f'    start(["{_label(t["start"])}"]) --> {_node_id(entry_path)}')
    for src, dests in internal_imports({p: code_map[p] for p in py}).items():
        for dest in dests:
            lines.append(f"    {_node_id(src)} --> {_node_id(dest)}")
    for src in others:
        for dest in (references or {}).get(src, []):
            if dest in code_map:
                lines.append(f"    {_node_id(src)} --> {_node_id(dest)}")
    return "\n".join(lines) + "\n"


__all__ = [
    "PublicSymbol",
    "internal_imports",
    "public_symbols",
    "render_flowchart",
    "render_spec",
    "subcommand_usages",
]
