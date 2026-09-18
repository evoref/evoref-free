"""tree-sitter によるコード抽出 (c_16 §4.4)

言語ごとのクエリ (``queries.py``) で class / function / method / import /
call のアンカーノードを捕まえ、フィールド名 (``child_by_field_name``) で
名前 / 基底 / 呼出し名 / import 指定子を読む。tree-sitter / 該当言語の
grammar が読めない場合は :func:`extract_file` が ``None`` を返す — 呼出側
(``builder.py``) が Python なら :mod:`python_ast` へ縮退させ、他言語は
WARNING 1 行で読み飛ばす。

同じ入力から同じ出力にするため、LLM は一切使わない決定論抽出。
"""

from __future__ import annotations

from typing import Any

from backend.free.rag.projectmap.extractors.queries import LANGUAGE_QUERIES, LanguageQuery
from backend.free.rag.projectmap.graph import ExtractedFile, Node, RawCall
from backend.free.rag.projectmap.ids import code_node_id
from backend.log_config import get_logger

logger = get_logger("rag.projectmap.treesitter")

#: 言語 → (parser, 構築済み Query)。構築に失敗した言語は ``None`` を持つ
#: (次回以降の再試行 / 再警告をしない)。
_PARSER_CACHE: dict[str, tuple[Any, Any] | None] = {}
_WARNED_LANGS: set[str] = set()
_TREE_SITTER_AVAILABLE: bool | None = None

_COMMENT_TYPES: frozenset[str] = frozenset({"comment", "line_comment", "block_comment"})
_STRING_PREFIXES: tuple[str, ...] = ("r", "R", "b", "B", "f", "F", "u", "U")


def is_available() -> bool:
    """``tree_sitter`` / ``tree_sitter_language_pack`` が import できるか。"""
    global _TREE_SITTER_AVAILABLE
    if _TREE_SITTER_AVAILABLE is None:
        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            _TREE_SITTER_AVAILABLE = False
        else:
            _TREE_SITTER_AVAILABLE = True
    return _TREE_SITTER_AVAILABLE


def _get_parser_and_query(lang: str) -> tuple[Any, Any] | None:
    """言語の parser + 構築済み Query (キャッシュ、失敗は 1 度だけ警告)。"""
    if lang in _PARSER_CACHE:
        return _PARSER_CACHE[lang]
    result: tuple[Any, Any] | None = None
    lang_query = LANGUAGE_QUERIES.get(lang)
    if is_available() and lang_query is not None:
        try:
            import tree_sitter
            from tree_sitter_language_pack import get_language, get_parser

            parser = get_parser(lang)  # type: ignore[arg-type]
            language = get_language(lang)  # type: ignore[arg-type]
            query = tree_sitter.Query(language, lang_query.query)
            result = (parser, query)
        except Exception as e:  # noqa: BLE001 — 1 言語の失敗で全体の走査を止めない
            if lang not in _WARNED_LANGS:
                _WARNED_LANGS.add(lang)
                logger.warning("tree-sitter unavailable for language %s: %s", lang, e)
    _PARSER_CACHE[lang] = result
    return result


def _text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _signature(node: Any, source: bytes) -> str:
    try:
        line = source.splitlines()[node.start_point.row]
    except IndexError:
        return ""
    return line.decode("utf-8", "replace").strip()


_COMMENT_LINE_PREFIXES = ("#", "//", "/*", "*", "*/")


def _leading_comment(node: Any, source: bytes) -> str:
    """定義の直前に連続するコメント行の先頭 1 行を返す。

    tree-sitter 0.26 の runtime は language-pack 1.20 の grammar と組むと
    ``prev_sibling`` / ``parent`` / ``children`` で得たノードの byte 位置を
    読んだ瞬間にアクセス違反で落ちる (実測 2026-09-18、``cartridge_manager.py``
    で再現。0.25.2 では再現しない)。requirements は ``<0.26`` に pin したうえで、
    木を遡る API に依存しない実装にしておく。
    """
    lines = source.splitlines()
    row = node.start_point.row - 1
    block: list[str] = []
    while row >= 0:
        stripped = lines[row].decode("utf-8", "replace").strip()
        if not stripped or not stripped.startswith(_COMMENT_LINE_PREFIXES):
            break
        block.append(stripped)
        row -= 1
    for line in reversed(block):
        cleaned = line.lstrip("/*#").strip()
        if cleaned:
            return cleaned
    return ""


def _python_docstring(node: Any, source: bytes) -> str:
    body = node.child_by_field_name("body")
    if body is None or body.child_count == 0:
        return ""
    first = body.children[0]
    # grammar 版によって、先頭の docstring が ``expression_statement`` に
    # 包まれる場合と ``string`` が直接子になる場合がある。両方受ける。
    string_node = first
    if first.type == "expression_statement":
        if first.child_count == 0:
            return ""
        string_node = first.children[0]
    if string_node.type != "string":
        return ""
    raw = _text(string_node, source).strip()
    for prefix in _STRING_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    raw = raw.strip("\"'")
    for line in raw.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _doc_line(node: Any, lang: str, source: bytes) -> str:
    if lang == "python":
        doc = _python_docstring(node, source)
        if doc:
            return doc
    return _leading_comment(node, source)


# ── 言語別: 定義ノードの名前 ────────────────────────────────────────────


def _definition_name(node: Any, lang: str) -> Any | None:
    if lang in ("c", "cpp") and node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")
        if declarator is not None and declarator.type == "function_declarator":
            return declarator.child_by_field_name("declarator")
        return None
    if lang == "kotlin":
        for child in node.children:
            if child.type in ("type_identifier", "simple_identifier"):
                return child
        return None
    return node.child_by_field_name("name")


# ── 言語別: 基底クラス ──────────────────────────────────────────────────


def _definition_bases(node: Any, lang: str) -> list[Any]:
    if lang == "python":
        superclasses = node.child_by_field_name("superclasses")
        if superclasses is None:
            return []
        return [c for c in superclasses.children if c.type == "identifier"]
    if lang in ("javascript", "typescript", "tsx"):
        for child in node.children:
            if child.type != "class_heritage":
                continue
            out = []
            for gc in child.children:
                if gc.type == "extends_clause":
                    value = gc.child_by_field_name("value")
                    if value is not None:
                        out.append(value)
            return out
        return []
    if lang == "java":
        out = []
        superclass = node.child_by_field_name("superclass")
        if superclass is not None:
            out.extend(c for c in superclass.children if c.type == "type_identifier")
        interfaces = node.child_by_field_name("interfaces")
        if interfaces is not None:
            for c in interfaces.children:
                if c.type == "type_list":
                    out.extend(gc for gc in c.children if gc.type == "type_identifier")
        return out
    if lang == "cpp":
        out = []
        for child in node.children:
            if child.type == "base_class_clause":
                out.extend(c for c in child.children if c.type == "type_identifier")
        return out
    if lang == "php":
        out = []
        for child in node.children:
            if child.type == "base_clause":
                out.extend(c for c in child.children if c.type == "name")
        return out
    if lang == "ruby":
        superclass = node.child_by_field_name("superclass")
        if superclass is None:
            return []
        return [c for c in superclass.children if c.type == "constant"]
    if lang == "swift":
        out = []
        for child in node.children:
            if child.type == "inheritance_specifier":
                out.extend(c for c in child.children if c.type == "user_type")
        return out
    return []


# ── 言語別: 呼出し名 ────────────────────────────────────────────────────


def _call_name(node: Any, lang: str) -> Any | None:
    if lang == "python":
        fn = node.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type == "identifier":
            return fn
        if fn.type == "attribute":
            return fn.child_by_field_name("attribute")
        return None
    if lang in ("javascript", "typescript", "tsx"):
        fn = node.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type == "identifier":
            return fn
        if fn.type == "member_expression":
            return fn.child_by_field_name("property")
        return None
    if lang == "go":
        fn = node.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type == "identifier":
            return fn
        if fn.type == "selector_expression":
            return fn.child_by_field_name("field")
        return None
    if lang == "rust":
        fn = node.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type == "identifier":
            return fn
        if fn.type == "field_expression":
            return fn.child_by_field_name("field")
        if fn.type == "scoped_identifier":
            return fn.child_by_field_name("name")
        return None
    if lang == "java":
        return node.child_by_field_name("name")
    if lang in ("c", "cpp"):
        fn = node.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type == "identifier":
            return fn
        if fn.type == "field_expression":
            return fn.child_by_field_name("field")
        return None
    if lang == "ruby":
        return node.child_by_field_name("method")
    if lang == "php":
        if node.type == "function_call_expression":
            return node.child_by_field_name("function")
        return node.child_by_field_name("name")
    if lang in ("kotlin", "swift"):
        for child in node.children:
            if child.type == "simple_identifier":
                return child
        return None
    return None


# ── 言語別: import 指定子 ───────────────────────────────────────────────


#: PHP の ``require``/``include`` 系ノード型 (動的パスは引数が文字列リテラル
#: でないので、その場合は spec を出さず素通しする)。
_PHP_INCLUDE_LIKE_TYPES: frozenset[str] = frozenset({
    "require_expression", "require_once_expression",
    "include_expression", "include_once_expression",
})


def _import_specs(node: Any, lang: str, source: bytes) -> list[str]:
    if lang == "python":
        field = "name" if node.type == "import_statement" else "module_name"
        out = []
        for child in node.children_by_field_name(field):
            target = child
            if child.type == "aliased_import" and child.child_count:
                target = child.children[0]
            out.append(_text(target, source))
        return out
    if lang in ("javascript", "typescript", "tsx"):
        source_node = node.child_by_field_name("source")
        return [_text(source_node, source).strip("'\"`")] if source_node is not None else []
    if lang == "go":
        path = node.child_by_field_name("path")
        return [_text(path, source).strip('"')] if path is not None else []
    if lang == "rust":
        if node.type == "mod_item":
            # body 付き (``mod a { ... }``) はインラインモジュールでファイル
            # 参照ではないので import 扱いしない (誤った辺の元になる)。
            if node.child_by_field_name("body") is not None:
                return []
            name = node.child_by_field_name("name")
            return [f"mod {_text(name, source)}"] if name is not None else []
        argument = node.child_by_field_name("argument")
        return [_text(argument, source)] if argument is not None else []
    if lang == "java":
        for child in node.children:
            if child.type in ("scoped_identifier", "identifier"):
                return [_text(child, source)]
        return []
    if lang in ("c", "cpp"):
        path = node.child_by_field_name("path")
        if path is None:
            return []
        if path.type == "system_lib_string":
            # ``<...>`` システムヘッダは山括弧付きのまま残す — 実在パスは
            # 山括弧を含まないので graph.py 側は自然に unresolved になる。
            return [_text(path, source)]
        return [_text(path, source).strip('"')]
    if lang == "php":
        if node.type in _PHP_INCLUDE_LIKE_TYPES:
            if node.child_count == 0:
                return []
            arg = node.children[-1]
            if arg.type not in ("string", "encapsed_string"):
                return []  # ``__DIR__ . '...'`` 等の動的パスは決定論解決できない
            raw = _text(arg, source).strip("'\"")
            # ``<keyword> '<path>'`` の実ソースに近い形にしておく — graph.py
            # 側の解決キーであると同時に、unresolved 時に external_imports
            # へそのまま出す表示形にもなる。
            keyword = node.type.removesuffix("_expression")
            return [f"{keyword} '{raw}'"] if raw else []
        for child in node.children:
            if child.type in ("qualified_name", "name"):
                return [_text(child, source)]
        return []
    if lang == "ruby":
        method = node.child_by_field_name("method")
        args = node.child_by_field_name("arguments")
        if method is None or args is None:
            return []
        method_name = _text(method, source).strip()
        for child in args.children:
            if child.type == "string":
                raw = _text(child, source).strip("'\"")
                # ``<require|require_relative> "<path>"`` の実ソースに近い形
                # (php と同じ理由)。
                return [f'{method_name} "{raw}"'] if raw else []
        return []
    if lang == "kotlin":
        for child in node.children:
            if child.type == "identifier":
                return [_text(child, source)]
        return []
    if lang == "swift":
        for child in node.children:
            if child.type == "identifier":
                return [_text(child, source)]
        return []
    return []


def _nearest_enclosing(
    key: tuple[int, int], def_keys: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """``key`` を真に含む定義範囲のうち最も内側 (start が最大) のものを返す。

    構文木は入れ子なので byte 範囲の包含だけで親子が決まる。``node.parent``
    を辿らないのは :func:`_leading_comment` と同じ理由 (native 側の
    アクセス違反)。
    """
    start, end = key
    best: tuple[int, int] | None = None
    for cand in def_keys:
        if cand == key:
            continue
        if cand[0] <= start and end <= cand[1] and (best is None or cand[0] > best[0]):
            best = cand
    return best


def _containing_depth(key: tuple[int, int], def_keys: set[tuple[int, int]]) -> int:
    start, end = key
    return sum(
        1 for cand in def_keys
        if cand != key and cand[0] <= start and end <= cand[1]
    )


def extract_file(path: str, lang: str, source: bytes) -> ExtractedFile | None:
    """1 ファイルを tree-sitter で抽出する。

    言語のクエリが構築できない (tree-sitter 不在 / grammar 未対応 / クエリ
    不正) 場合は ``None``。呼出側 (``builder.py``) が Python ならこの後で
    :mod:`python_ast` へ縮退し、他言語は読み飛ばして件数を警告する。
    """
    resolved = _get_parser_and_query(lang)
    if resolved is None:
        return None
    parser, query = resolved
    lang_query: LanguageQuery = LANGUAGE_QUERIES[lang]

    import tree_sitter

    tree = parser.parse(source)
    cursor = tree_sitter.QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    # ``QueryCursor.captures`` の返す順序は内部実装に依存し、同じ入力でも
    # 呼出しごとに揺れることがある。決定論出力のため byte 位置で固定する。
    def _by_start(nodes: list[Any]) -> list[Any]:
        return sorted(nodes, key=lambda n: (n.start_byte, n.end_byte))

    class_ts = _by_start(captures.get("class.def", []))
    function_ts = _by_start(captures.get("function.def", []))
    method_ts = _by_start(captures.get("method.def", []))
    import_ts = _by_start(captures.get("import.stmt", []))
    call_ts = _by_start(captures.get("call.expr", []))

    hints: dict[tuple[int, int], tuple[Any, str]] = {}
    for n in class_ts:
        hints[(n.start_byte, n.end_byte)] = (n, "class")
    for n in function_ts:
        hints.setdefault((n.start_byte, n.end_byte), (n, "function"))
    for n in method_ts:
        hints[(n.start_byte, n.end_byte)] = (n, "method")
    def_keys = set(hints)

    entries: dict[tuple[int, int], dict[str, Any]] = {}
    inherits: list[tuple[str, str]] = []
    # 同じ qualname の 2 つ目以降 (再定義 / TYPE_CHECKING の二重定義 / TS の
    # オブジェクトリテラルのメソッド) は ``#<n>`` を付けて id を分ける。最初の
    # 定義が素の qualname を保つので、版を跨いだ同一性は主定義で安定する。
    seen_qualnames: dict[str, int] = {}
    # 外側の定義から順に処理する (親の entry が先に出来ている必要がある)。
    # 同じ深さの順序は byte 位置で固定 (決定論)。
    for key in sorted(hints, key=lambda k: (_containing_depth(k, def_keys), k)):
        ts_node, kind_hint = hints[key]
        name_node = _definition_name(ts_node, lang)
        name = _text(name_node, source).strip() if name_node is not None else ""
        if not name:
            continue

        parent_key = _nearest_enclosing(key, def_keys)
        if kind_hint == "function":
            parent_entry = entries.get(parent_key) if parent_key else None
            # 最も内側の定義が class なら method (JS/TS の class_body 直下も
            # class の byte 範囲に含まれるので、木を遡らずに判定できる)
            if parent_entry is not None and parent_entry["node_type"] == "class":
                node_type = "method"
            else:
                node_type = "function"
        else:
            node_type = kind_hint

        parent_entry = entries.get(parent_key) if parent_key else None
        qualname = f"{parent_entry['qualname']}.{name}" if parent_entry else name
        occurrence = seen_qualnames.get(qualname, 0) + 1
        seen_qualnames[qualname] = occurrence
        if occurrence > 1:
            qualname = f"{qualname}#{occurrence}"
        parent_id = parent_entry["node"].id if parent_entry else None
        signature = _signature(ts_node, source)
        doc = _doc_line(ts_node, lang, source)
        text = f"{signature}\n{doc}" if doc else signature
        node_id = code_node_id(path, node_type, qualname)
        node = Node(
            id=node_id,
            node_type=node_type,
            path=path,
            name=name,
            qualname=qualname,
            lang=lang,
            line_start=ts_node.start_point.row + 1,
            line_end=ts_node.end_point.row + 1,
            parent_id=parent_id,
            signature=signature,
            text=text,
        )
        entries[key] = {"node": node, "qualname": qualname, "node_type": node_type}

        if node_type == "class":
            for base_node in _definition_bases(ts_node, lang):
                base_name = _text(base_node, source).strip()
                if base_name:
                    inherits.append((node_id, base_name))

    nodes = [entry["node"] for entry in entries.values()]
    nodes.sort(key=lambda n: n.line_start)

    imports: list[str] = []
    for imp_node in import_ts:
        for spec in _import_specs(imp_node, lang, source):
            spec = spec.strip()
            if spec:
                imports.append(spec)

    calls: list[RawCall] = []
    for call_node in call_ts:
        name_node = _call_name(call_node, lang)
        if name_node is None:
            continue
        callee = _text(name_node, source).strip()
        if not callee:
            continue
        parent_key = _nearest_enclosing(
            (call_node.start_byte, call_node.end_byte), def_keys,
        )
        parent_entry = entries.get(parent_key) if parent_key else None
        if parent_entry is None:
            continue  # モジュール直下の呼出しは呼出し元が定まらないので捨てる
        calls.append(RawCall(caller_id=parent_entry["node"].id, callee_name=callee))

    line_count = source.count(b"\n") + (0 if source.endswith(b"\n") else 1)
    return ExtractedFile(
        path=path, lang=lang, line_count=max(line_count, 1),
        nodes=nodes, imports=imports, calls=calls, inherits=inherits,
    )


__all__ = ["extract_file", "is_available"]
