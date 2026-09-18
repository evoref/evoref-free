"""言語別の tree-sitter S 式クエリ表 (c_16 §4.4)

クエリは **アンカーノードを捕まえるだけ**にする — クラス名 / 基底名 / 呼出し名
/ import 指定子は、文法ごとにフィールド名が違ったり (Python は
``superclasses``、Java は ``superclass`` + ``interfaces``)、フィールドを
持たない文法もある (Kotlin) ため、クエリの捕獲だけで安全に対応させるより
:mod:`backend.free.rag.projectmap.extractors.treesitter` 側で
``child_by_field_name`` を使って直接読む方が壊れにくい。

捕獲名の規約:

- ``class.def`` — クラス (相当) 定義
- ``function.def`` — 関数**とメソッドの両方を同じ構文ノードで表す言語**
  (Python / Ruby / Rust / C / C++ / Kotlin / Swift)。メソッドかどうかは
  :attr:`LanguageQuery.class_ancestor_types` に挙がる祖先ノード型を持つかで
  決める (``treesitter.py`` 側の仕事)
- ``method.def`` — 文法上 **別のノード型** でメソッドを区別できる言語
  (JS/TS/TSX の ``method_definition``、Java / PHP / Go の
  ``method_declaration``)。祖先判定は不要
- ``import.stmt`` — import 文。文法上 import 専用のノード型を持たない言語は、
  述語 (``#match?``) で絞った ``call`` / 式ノードを流用する
  (Ruby の ``require``/``require_relative``、PHP の
  ``require``/``require_once``/``include``/``include_once``、Rust の
  bodyless ``mod_item``)。流用ノードは同時に ``call.expr`` 等にも二重捕獲
  されうるが、``graph.py`` 側の解決が失敗するだけで害は無い
- ``call.expr`` — 呼出し式

未対応の種別が無い言語は、その捕獲を持たない (クエリ実行自体は他の捕獲に
ついて行う)。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LanguageQuery:
    """1 言語分のクエリ定義。"""

    #: tree-sitter クエリ文字列 (複数パターンを連結してよい)。
    query: str
    #: ``function.*`` で捕獲した定義ノードの祖先にこの型があれば method 扱い
    #: にする (メソッドが関数と同じノード型の言語だけ使う)。
    class_ancestor_types: frozenset[str] = field(default_factory=frozenset)


_PYTHON = LanguageQuery(
    query="""
(class_definition) @class.def
(function_definition) @function.def
(import_statement) @import.stmt
(import_from_statement) @import.stmt
(call) @call.expr
""",
    class_ancestor_types=frozenset({"class_definition"}),
)

_JAVASCRIPT = LanguageQuery(
    query="""
(class_declaration) @class.def
(function_declaration) @function.def
(method_definition) @method.def
(import_statement) @import.stmt
(call_expression) @call.expr
""",
)

# tsx は typescript 文法のスーパーセット (grammar 上のノード名は同じ)。
_TYPESCRIPT = _JAVASCRIPT
_TSX = _JAVASCRIPT

_GO = LanguageQuery(
    query="""
(function_declaration) @function.def
(method_declaration) @method.def
(import_spec) @import.stmt
(call_expression) @call.expr
""",
)

_RUST = LanguageQuery(
    query="""
(function_item) @function.def
(use_declaration) @import.stmt
(mod_item) @import.stmt
(call_expression) @call.expr
""",
    class_ancestor_types=frozenset({"impl_item"}),
)

_JAVA = LanguageQuery(
    query="""
(class_declaration) @class.def
(method_declaration) @method.def
(import_declaration) @import.stmt
(method_invocation) @call.expr
""",
)

_C = LanguageQuery(
    query="""
(function_definition) @function.def
(preproc_include) @import.stmt
(call_expression) @call.expr
""",
)

_CPP = LanguageQuery(
    query="""
(class_specifier) @class.def
(function_definition) @function.def
(preproc_include) @import.stmt
(call_expression) @call.expr
""",
    class_ancestor_types=frozenset({"field_declaration_list"}),
)

_RUBY = LanguageQuery(
    query="""
(class) @class.def
(method) @function.def
(call
  method: (identifier) @_require_method
  (#match? @_require_method "^require(_relative)?$")
  arguments: (argument_list . (string))
) @import.stmt
(call) @call.expr
""",
    class_ancestor_types=frozenset({"class"}),
)

_PHP = LanguageQuery(
    query="""
(class_declaration) @class.def
(function_definition) @function.def
(method_declaration) @method.def
(namespace_use_clause) @import.stmt
(require_expression) @import.stmt
(require_once_expression) @import.stmt
(include_expression) @import.stmt
(include_once_expression) @import.stmt
(function_call_expression) @call.expr
(member_call_expression) @call.expr
""",
)

_KOTLIN = LanguageQuery(
    query="""
(class_declaration) @class.def
(function_declaration) @function.def
(import_header) @import.stmt
(call_expression) @call.expr
""",
    class_ancestor_types=frozenset({"class_body"}),
)

_SWIFT = LanguageQuery(
    query="""
(class_declaration) @class.def
(function_declaration) @function.def
(import_declaration) @import.stmt
(call_expression) @call.expr
""",
    class_ancestor_types=frozenset({"class_body"}),
)

#: 言語名 (``scanner.LANGUAGE_EXTENSIONS`` の値) → クエリ定義。
LANGUAGE_QUERIES: dict[str, LanguageQuery] = {
    "python": _PYTHON,
    "javascript": _JAVASCRIPT,
    "typescript": _TYPESCRIPT,
    "tsx": _TSX,
    "go": _GO,
    "rust": _RUST,
    "java": _JAVA,
    "c": _C,
    "cpp": _CPP,
    "ruby": _RUBY,
    "php": _PHP,
    "kotlin": _KOTLIN,
    "swift": _SWIFT,
}

__all__ = ["LANGUAGE_QUERIES", "LanguageQuery"]
