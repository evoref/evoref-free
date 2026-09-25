"""staged v2 の言語ごとの扱い (f_10 §12)。

第 1 段の系統は ``python`` / ``web`` (HTML・CSS・JS・TS) / ``php`` / ``sql``。系統の判定・生成指示・
決定論の検査 (構文・参照の整合・実行環境による構文検査)・as-built 文書の事実をここにまとめる。
生成物を丸ごと実行はしない (構文の検査だけ、§12.4)。
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path, PurePosixPath

from backend.log_config import get_logger

logger = get_logger("api.chat.staged_v2_languages")

#: v2 が扱う系統 (``create.staged.v2_families`` の値)。
FAMILIES: tuple[str, ...] = ("python", "web", "php", "sql")

_PY = frozenset({".py"})
_WEB = frozenset({".html", ".htm", ".css", ".js", ".mjs", ".ts"})
_PHP = frozenset({".php"})
_SQL = frozenset({".sql"})
#: コードではないファイル (系統の判定に数えない)。
DATA_SUFFIXES = frozenset({
    "", ".json", ".csv", ".tsv", ".md", ".txt", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml",
})
SUPPORTED_SUFFIXES = _PY | _WEB | _PHP | _SQL

#: 実装言語の言い方 (``content_detector.implementation_languages``) のうち v2 が扱うもの。
_SUPPORTED_LANGUAGES = frozenset({"python", "javascript", "typescript", "php"})

#: 依頼文から拾う「対象外の言語」の拡張子 (URL の ``.com`` 等を拾わないよう既知のものだけ)。
_UNSUPPORTED_CODE_SUFFIXES = frozenset({
    ".go", ".rs", ".java", ".kt", ".kts", ".swift", ".rb", ".cs", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".jsx", ".tsx", ".svelte", ".vue", ".scss", ".lua", ".r", ".pl", ".dart", ".scala",
    ".sh", ".ps1", ".bat",
})
FILE_NAME_IN_TEXT_RE = re.compile(r"[\w.\-]+\.[A-Za-z][A-Za-z0-9]{0,5}(?![A-Za-z0-9])")
#: SQLite で実行できない方言を名指す言い方 (名指されたら構文検査だけにする)。
_OTHER_SQL_DIALECT_RE = re.compile(r"(?i)\b(?:mysql|mariadb|postgres(?:ql)?|sql\s*server|oracle|t-sql)\b")


def family_of(paths: list[str]) -> tuple[str | None, list[str]]:
    """モジュールの拡張子から系統を決める。戻り値は (系統 or None, 対象外の根拠)。

    Python と Web / PHP の混在 (Flask 等) は第 1 段では対象外 (§12.1)。
    """
    suffixes = {PurePosixPath(p).suffix.lower() for p in paths} - DATA_SUFFIXES
    unsupported = sorted(p for p in paths if PurePosixPath(p).suffix.lower() not in SUPPORTED_SUFFIXES
                         and PurePosixPath(p).suffix.lower() not in DATA_SUFFIXES)
    if unsupported:
        return None, unsupported
    if suffixes & _PY and suffixes & (_WEB | _PHP):
        return None, sorted(p for p in paths if PurePosixPath(p).suffix.lower() in _WEB | _PHP)
    if suffixes & _PY:
        return "python", []
    if suffixes & _PHP:
        return "php", []
    if suffixes & _WEB:
        return "web", []
    if suffixes & _SQL:
        return "sql", []
    return None, []


def unsupported_request(query: str) -> list[str]:
    """依頼が第 1 段の対象外を名指す根拠 (骨組みの前、LLM を呼ばない)。

    対象外の拡張子のファイル名、対象外の実装言語 (「Go で」)、Python と Web / PHP の
    ファイル名の混在。空なら v2 で骨組みを作る (言語名の無い依頼は骨組みの後で見る)。
    """
    from backend.free.generation.content_detector import implementation_languages

    names = sorted({m.group() for m in FILE_NAME_IN_TEXT_RE.finditer(query or "")})
    evidence = [n for n in names if PurePosixPath(n).suffix.lower() in _UNSUPPORTED_CODE_SUFFIXES]
    evidence += sorted(implementation_languages(query) - _SUPPORTED_LANGUAGES)
    code_names = [n for n in names if PurePosixPath(n).suffix.lower() in SUPPORTED_SUFFIXES]
    family, mixed = family_of(code_names)
    if family is None and mixed:
        evidence += mixed
    return evidence


def names_other_sql_dialect(query: str) -> bool:
    """依頼が SQLite 以外の方言 (MySQL / PostgreSQL …) を名指すか。"""
    return bool(_OTHER_SQL_DIALECT_RE.search(query or ""))


# ── 生成指示 ────────────────────────────────────────────────────────────


def fence_language(path: str) -> str:
    from backend.free.core.code_syntax import language_label

    return language_label(path)


def module_rules(path: str, *, siblings: list[str]) -> str:
    """ファイルの言語ごとの生成規則 (``_MODULE_TASK`` の Rules に足す行)。"""
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".py":
        stems = [PurePosixPath(s).stem for s in siblings if s.endswith(".py")]
        example = stems[0] if stems else "module"
        return (
            "- Import sibling modules of this deliverable by their bare module name\n"
            f"  (e.g. `from {example} import ...`) — they are placed in the same directory.\n"
            "- Standard library only unless the request says otherwise.\n"
        )
    lang = fence_language(path)
    rules = [f"- This file is {lang}, not Python. Write idiomatic {lang}."]
    if suffix in (".html", ".htm", ".php"):
        rules.append(
            "- Reference the other files of this deliverable (scripts, stylesheets) by exactly the "
            "relative paths in the design, and give every element other files use the exact id / class "
            "the design declares.",
        )
    if suffix in (".html", ".htm"):
        # file:// で開くと ES モジュールは CORS で読めない — 静的ページは素の script で書く
        rules.append(
            "- The page must work when opened directly from disk (file://): load scripts with plain "
            "`<script src=\"...\" defer></script>` (not type=\"module\").",
        )
    if suffix in (".js", ".mjs"):
        # Coder 系は .js にも TypeScript の型注釈 / ``as`` を書き、作り直しでも直らなかった (ベンチ w2)
        rules.append(
            "- Plain JavaScript: no type annotations, no `as` casts, no interfaces or other TypeScript syntax.",
        )
    if suffix in (".js", ".mjs", ".ts"):
        rules.append(
            "- Use only element ids / classes that the HTML of this deliverable defines. Attach an event "
            "listener to every control (button, input, form) the request describes — defining a function "
            "is not enough. No external libraries or CDNs unless the request asks for them.",
        )
        rules.append(
            "- Import sibling modules with relative paths (`./x.js`) and only names they export."
            if suffix != ".js" else
            "- A script loaded by an HTML page is a plain script (no import / export); share values "
            "between scripts through globals declared in the design.",
        )
    if suffix == ".php":
        rules.append(
            "- Include sibling PHP files with `require_once __DIR__ . '/<path>';` using the design's paths. "
            "No frameworks or Composer packages unless the request asks for them. Functions do not see "
            "script variables: read `$argv` / `$argc` at the top level and pass the values in.",
        )
    if suffix == ".sql":
        rules.append(
            "- Plain SQL that runs on SQLite unless the request names another database. End every "
            "statement with `;`. No comments that are not SQL comments.",
        )
    return "\n".join(rules) + "\n"


# ── 決定論の検査 (§12.3 / §12.4) ─────────────────────────────────────────

_SCRIPT_OR_LINK_RE = re.compile(
    r"<(?:script|link)\b[^>]*?\b(?:src|href)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE,
)
_HTML_ID_RE = re.compile(r"\bid\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_JS_ID_USE_RE = re.compile(
    r"getElementById\(\s*[\"'`]([\w\-:.]+)[\"'`]\s*\)"
    r"|querySelector(?:All)?\(\s*[\"'`]#([\w\-]+)[\"'`]\s*\)",
)
_PHP_INCLUDE_RE = re.compile(
    r"\b(?:require|include)(?:_once)?\s*\(?\s*(?:__DIR__\s*\.\s*)?[\"']/?([^\"']+\.php)[\"']",
)


def _is_external(spec: str) -> bool:
    return bool(re.match(r"(?i)^(?:[a-z][a-z0-9+.\-]*:|//|#)", spec))


def _resolve(from_path: str, spec: str) -> str:
    """``from_path`` から見た相対参照を成果物内のパスにする (クエリ・断片は落とす)。"""
    spec = spec.split("?", 1)[0].split("#", 1)[0]
    base = PurePosixPath(from_path).parent
    parts: list[str] = []
    for part in (PurePosixPath(spec.lstrip("/")) if spec.startswith("/") else base / spec).parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _html_refs(path: str, code: str) -> list[str]:
    """HTML / PHP の ``<script src>`` / ``<link href>`` (外部 URL を除く)。"""
    specs: list[str] = []
    if PurePosixPath(path).suffix.lower() in (".html", ".htm"):
        try:
            from backend.free.rag.projectmap.extractors.html_tree import get_parser
            from backend.free.rag.projectmap.extractors.markup import html_import_specs

            parser = get_parser("html")
            if parser is not None:
                source = code.encode("utf-8")
                specs = html_import_specs(parser.parse(source).root_node, source)
        except Exception as exc:  # noqa: BLE001 - 抽出器が使えなければ正規表現へ
            logger.debug("html_import_specs unavailable: %s", exc)
    if not specs:
        specs = _SCRIPT_OR_LINK_RE.findall(code)
    return [s for s in specs if s and not _is_external(s)]


def references(code_map: dict[str, str]) -> dict[str, list[str]]:
    """ファイル → 参照する成果物内のファイル (HTML の script/link、JS の import、PHP の require)。"""
    from backend.free.loop.staged.js_modules import relative_js_imports

    out: dict[str, list[str]] = {}
    for path, code in code_map.items():
        suffix = PurePosixPath(path).suffix.lower()
        refs: list[str] = []
        if suffix in (".html", ".htm", ".php"):
            refs += [_resolve(path, s) for s in _html_refs(path, code)]
        if suffix in (".js", ".mjs", ".ts"):
            refs += [_resolve(path, src) for src in relative_js_imports(code)]
        if suffix == ".php":
            refs += [_resolve(path, s) for s in _PHP_INCLUDE_RE.findall(code)]
        out[path] = [r for r in dict.fromkeys(refs) if r != path]
    return out


_CONTROL_TAG_RE = re.compile(r"<(button|input|select|textarea)\b([^>]*)>", re.IGNORECASE)
_INLINE_HANDLER_RE = re.compile(r"\bon[a-z]+\s*=", re.IGNORECASE)
_INPUT_TYPE_RE = re.compile(r"\btype\s*=\s*[\"']?(\w+)", re.IGNORECASE)


def _unwired_control_errors(code_map: dict[str, str]) -> list[str]:
    """静的な HTML の操作要素 (ボタン等) が、どのスクリプトからも参照されていない。

    関数を定義しただけでボタンに結び付けないページは押しても何も起きない (ベンチ w1)。
    id を持つ操作要素が、スクリプトの ``getElementById`` / ``querySelector('#id')`` にも
    inline のハンドラにも現れなければ、そのページが読み込むスクリプトの誤りとして返す。
    PHP はフォームをサーバで受けるので対象外。
    """
    errors: list[str] = []
    scripts = {p: c for p, c in code_map.items() if PurePosixPath(p).suffix.lower() in (".js", ".mjs", ".ts")}
    for path, code in code_map.items():
        if PurePosixPath(path).suffix.lower() not in (".html", ".htm"):
            continue
        loaded = [r for r in references({path: code, **scripts}).get(path, []) if r in scripts]
        if not loaded:
            continue
        inline = "\n".join(re.findall(r"<script\b[^>]*>(.*?)</script>", code, re.IGNORECASE | re.DOTALL))
        used = {a or b for s in [inline, *(scripts[r] for r in loaded)] for a, b in _JS_ID_USE_RE.findall(s)}
        unwired = []
        for tag, attrs in _CONTROL_TAG_RE.findall(code):
            m = _HTML_ID_RE.search(attrs)
            kind = (_INPUT_TYPE_RE.search(attrs) or [None, ""])[1].lower()
            if m is None or _INLINE_HANDLER_RE.search(attrs) or kind in ("hidden", "submit"):
                continue
            if m.group(1) not in used:
                unwired.append(f"#{m.group(1)} ({tag.lower()})")
        if unwired:
            errors.append(
                f"{loaded[0]}: the controls {', '.join(unwired)} in {path} are never used — no script looks "
                "them up, so pressing / typing does nothing. Look each one up and attach its event listener "
                "(or read its value) as the request describes.",
            )
    return errors


def reference_errors(code_map: dict[str, str]) -> list[str]:
    """参照の整合 (§12.3)。エラーは ``<path>: …`` (作り直す宛先はそのパス)。"""
    from backend.free.loop.staged.js_modules import missing_js_imports

    errors: list[str] = []
    names = set(code_map)
    for path, refs in references(code_map).items():
        for ref in refs:
            if ref not in names:
                errors.append(f"{path}: references '{ref}', which is not a file of this deliverable "
                              f"(files: {', '.join(sorted(names))})")
    markup = [c for p, c in code_map.items() if PurePosixPath(p).suffix.lower() in (".html", ".htm", ".php")]
    ids = {i for c in markup for i in _HTML_ID_RE.findall(c)}
    errors += _unwired_control_errors(code_map)
    for path, code in code_map.items():
        if PurePosixPath(path).suffix.lower() not in (".js", ".mjs", ".ts"):
            continue
        siblings = {p: c for p, c in code_map.items() if p != path}
        errors += [f"{path}: {problem}" for problem in missing_js_imports(code, siblings)]
        if markup:
            used = {a or b for a, b in _JS_ID_USE_RE.findall(code)}
            missing = sorted(used - ids)
            if missing:
                errors.append(f"{path}: uses element id(s) {', '.join(missing)} that no HTML of this "
                              f"deliverable defines (defined ids: {', '.join(sorted(ids)) or 'none'})")
    return errors


_PHP_SUPERGLOBALS = frozenset({
    "this", "GLOBALS", "_SERVER", "_GET", "_POST", "_FILES", "_COOKIE", "_SESSION", "_REQUEST", "_ENV",
    "http_response_header",
})
_PHP_BINDING_PARENTS = frozenset({
    "simple_parameter", "variadic_parameter", "property_promotion_parameter", "global_declaration",
    "static_variable_declaration", "catch_clause", "anonymous_function_use_clause", "list_literal",
})
_PHP_DYNAMIC_SCOPE_RE = re.compile(r"\b(?:extract|compact|eval|get_defined_vars)\s*\(|\$\$|\b(?:include|require)(?:_once)?\b")


def _php_undefined_variable_errors(path: str, code: str) -> list[str]:
    """関数の中で、引数でも代入でも global でもない変数を使っている (``php -l`` は構文しか見ない)。

    PHP の関数からはスクリプトの変数 (``$argv`` / ``$argc`` を含む) が見えない。
    ``function main() { if ($argc !== 3) … }`` は構文としては正しく、実行して初めて
    未定義の警告で止まる (create ベンチ p1)。束縛は関数の中のどこで起きても数え
    (順序は見ない)、``extract()`` / ``$$`` / 関数内の include がある関数は判定しない。
    """
    try:
        from backend.free.rag.projectmap.extractors.html_tree import get_parser, iter_nodes, node_text
    except Exception:  # noqa: BLE001 - tree-sitter が無ければ検査しない
        return []
    parser = get_parser("php")
    if parser is None:
        return []
    source = code.encode("utf-8")
    errors: list[str] = []
    for fn in iter_nodes(parser.parse(source).root_node):
        if fn.type not in ("function_definition", "method_declaration"):
            continue
        body_text = node_text(fn, source)
        if _PHP_DYNAMIC_SCOPE_RE.search(body_text):
            continue
        bound: set[str] = set()
        used: list[str] = []

        def _walk(node, parent_type: str, lhs: bool) -> None:
            if node.type == "variable_name":
                name = node_text(node, source).lstrip("$")
                if lhs or parent_type in _PHP_BINDING_PARENTS:
                    bound.add(name)
                else:
                    used.append(name)
                return
            children = list(node.children)
            for i, child in enumerate(children):
                child_lhs = False
                if node.type in ("assignment_expression", "augmented_assignment_expression",
                                 "reference_assignment_expression"):
                    child_lhs = i == 0
                elif node.type == "foreach_statement":
                    # ``foreach ($items as $k => $v)`` の as 以降は束縛
                    seen_as = any(c.type == "as" for c in children[:i])
                    child_lhs = seen_as and child.type != "compound_statement"
                elif node.type == "pair" and lhs:
                    child_lhs = True
                _walk(child, node.type, lhs or child_lhs)

        _walk(fn, "", False)
        name_node = fn.child_by_field_name("name")
        fn_name = node_text(name_node, source) if name_node is not None else "?"
        missing = sorted({u for u in used if u not in bound and u not in _PHP_SUPERGLOBALS})
        if missing:
            errors.append(
                f"{path}: function {fn_name}() uses {', '.join('$' + m for m in missing)}, which is never "
                "assigned, passed in or declared global there — PHP functions do not see script variables "
                "(including $argv / $argc). Pass them as arguments or declare them `global`.",
            )
    return errors


def syntax_errors(code_map: dict[str, str]) -> list[str]:
    """tree-sitter の構文検査 (Python 以外)。"""
    from backend.free.core.code_syntax import syntax_error_detail

    errors = []
    for path, code in code_map.items():
        if path.endswith(".py"):
            continue
        detail = syntax_error_detail(code, path)
        if detail:
            errors.append(f"{path}: syntax error: {detail}")
        elif path.endswith(".php"):
            errors += _php_undefined_variable_errors(path, code)
    return errors


def _sqlite_errors(code_map: dict[str, str], order: list[str]) -> list[str]:
    """``.sql`` を依存順に 1 つのメモリ上の SQLite で流す (最初に失敗したファイルだけ返す)。"""
    conn = sqlite3.connect(":memory:")
    try:
        for path in order:
            try:
                conn.executescript(code_map[path])
            except sqlite3.Error as exc:
                return [f"{path}: SQLite error: {exc}"]
    finally:
        conn.close()
    return []


def _sql_order(code_map: dict[str, str], skeleton: dict) -> list[str]:
    deps = {m["path"]: list(m.get("imports_from") or []) for m in skeleton.get("modules") or []}
    order: list[str] = []

    def _visit(p: str, seen: set[str]) -> None:
        if p in order or p in seen or p not in code_map:
            return
        seen.add(p)
        for d in deps.get(p, []):
            if d.endswith(".sql"):
                _visit(d, seen)
        order.append(p)

    # 骨組みが依存を書かないことがある (report.sql → schema.sql の順で流れ、正しい SQL が
    # 「no such table」で落ちた。ベンチ q1) — 定義 → データ → 問い合わせの順を先に決める
    def _phase(p: str) -> int:
        text = code_map[p]
        if _SQL_DDL_RE.search(text):
            return 0
        return 1 if _SQL_DML_RE.search(text) else 2

    for p in sorted((p for p in code_map if p.endswith(".sql")), key=lambda p: (_phase(p), p)):
        _visit(p, set())
    return order


_SQL_DDL_RE = re.compile(r"(?i)\bcreate\s+(?:temp(?:orary)?\s+)?(?:table|view|index|trigger)\b")
_SQL_DML_RE = re.compile(r"(?i)\b(?:insert\s+into|update\s+\w+\s+set|delete\s+from)\b")


def runtime_errors(
    code_map: dict[str, str], *, cfg: dict, skeleton: dict, query: str, timeout_sec: float = 30.0,
) -> list[str]:
    """実行環境による構文検査 (§12.4): ``node --check`` / ``php -l`` / SQLite。見つからない検査は飛ばす。"""
    from backend.free.core.runtimes import resolve_runtime, run_runtime

    errors: list[str] = []
    checks = {
        # .ts は node --check が型注釈を外さないので tree-sitter の構文検査だけ
        "node": ((".js", ".mjs"), ("--check",)),
        "php": ((".php",), ("-l",)),
    }
    targets = {name: [p for p in code_map if PurePosixPath(p).suffix.lower() in sfx]
               for name, (sfx, _args) in checks.items()}
    if any(targets.values()):
        with tempfile.TemporaryDirectory(prefix="evoref_v2_check_") as tmp:
            root = Path(tmp)
            for path, code in code_map.items():
                dest = root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(code, encoding="utf-8")
            for name, paths in targets.items():
                if not paths:
                    continue
                exe, why = resolve_runtime(name, cfg)
                if exe is None:
                    logger.info("staged v2: %s not available (%s); skipping its syntax check", name, why)
                    continue
                for path in paths:
                    code, out = run_runtime(exe, (*checks[name][1], str(root / path)), cwd=root,
                                            timeout_sec=timeout_sec)
                    if code not in (0, None):
                        detail = out.replace(str(root), "").strip()[-1200:]
                        errors.append(f"{path}: {name} {' '.join(checks[name][1])} failed: {detail}")
    if any(p.endswith(".sql") for p in code_map) and not names_other_sql_dialect(query):
        errors += _sqlite_errors(code_map, _sql_order(code_map, skeleton))
    return errors


def check(code_map: dict[str, str], *, cfg: dict, skeleton: dict, query: str) -> list[str]:
    """Python 以外の決定論の検査を全部 (構文 → 参照 → 実行環境)。"""
    others = {p: c for p, c in code_map.items() if not p.endswith(".py")}
    if not others:
        return []
    return (
        syntax_errors(others)
        + reference_errors(others)
        + runtime_errors(others, cfg=cfg, skeleton=skeleton, query=query)
    )


# ── as-built 文書の事実 (§12.5) ─────────────────────────────────────────

_CSS_RULE_RE = re.compile(r"[^{}]+\{")
_JS_FUNCTION_RE = re.compile(r"\bfunction\s+(\w+)")
_SQL_TABLE_RE = re.compile(r"(?i)\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?[\"`\[]?(\w+)")
_PHP_DECL_RE = re.compile(r"(?m)^\s*(?:(?:final|abstract)\s+)?(function\s+\w+\s*\([^)]*\)|class\s+\w+)")


def facts(path: str, code: str) -> list[str]:
    """SPEC.md に載せる事実 (Python 以外)。"""
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in (".js", ".mjs", ".ts"):
        from backend.free.loop.staged.js_modules import js_export_summary

        lines = [ln.strip() for ln in js_export_summary(code).splitlines() if ln.strip()]
        functions = _JS_FUNCTION_RE.findall(code)[:12]
        return lines or [f"functions: {', '.join(functions) or 'none'}"]
    if suffix in (".html", ".htm"):
        ids = sorted(set(_HTML_ID_RE.findall(code)))
        refs = _html_refs(path, code)
        out = [f"ids: {', '.join(ids) or 'none'}"]
        if refs:
            out.append(f"loads: {', '.join(refs)}")
        return out
    if suffix == ".css":
        return [f"rules: {len(_CSS_RULE_RE.findall(code))}"]
    if suffix == ".sql":
        return [f"tables: {', '.join(dict.fromkeys(_SQL_TABLE_RE.findall(code))) or 'none'}"]
    if suffix == ".php":
        return [m.strip() for m in _PHP_DECL_RE.findall(code)][:20] or ["(no functions or classes)"]
    return []


__all__ = [
    "DATA_SUFFIXES",
    "FAMILIES",
    "FILE_NAME_IN_TEXT_RE",
    "SUPPORTED_SUFFIXES",
    "check",
    "facts",
    "family_of",
    "fence_language",
    "module_rules",
    "names_other_sql_dialect",
    "reference_errors",
    "references",
    "runtime_errors",
    "syntax_errors",
    "unsupported_request",
]
