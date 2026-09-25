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
        "flow_title": "設計フローチャート", "start": "開始",
    },
    "en": {
        "note": "This specification was generated from the finished code (as-built).",
        "request": "Request", "usage": "Usage", "modules": "Modules",
        "examples": "Examples", "verification": "Verification", "entry": "Entry point",
        "flow_title": "Design flowchart", "start": "Start",
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
) -> str:
    """as-built の SPEC.md 本文を返す。

    Python 以外のファイル (f_10 §12.5) は呼出し側が抽出した事実 (``other_facts``: id・参照先・
    export・テーブル等) をそのまま並べる。
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
        for e in examples:
            lines.append(f"- `{e.get('call')}` → `{e.get('expected')}`")
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
        label = f"{PurePosixPath(path).name}" + (f"<br/>{shown}" if shown else "")
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
]
