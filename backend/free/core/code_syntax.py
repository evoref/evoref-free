"""生成コードの言語別構文検査 (staged create、f_10 §4.2)。

staged の code 工程は全ファイルを Python の ``compile`` に通していたため、
HTML / CSS / JavaScript の成果物が「invalid character '、'」で全件失敗した
(2026-09-19 ライブ監査 K05)。Python は従来どおり ``compile`` (コンパイル時にしか
出ない誤りも拾う)、それ以外は tree-sitter の構文木に ERROR / MISSING が
あるかで判定する。tree-sitter が無い環境、または対応表に無い拡張子は
**検査しない** (``None``) — 検査できないことを構文エラーと扱わない。

同梱の対応表は ProjectMap (c_16 §4.4) と同じ拡張子を持つ (``.scss`` /
``.svelte`` / ``.vue`` を 2026-09-20 に追加)。その上に言語パック
(c_16 §4.5.3) を重ねて引く — ``core`` は corpus を import できないので、
:func:`set_language_overlay` を押し込み口にする。corpus 側
(``CorpusStore`` の install / uninstall / 版切替 / 起動時の open) が
呼ぶ。同梱の拡張子は上書きできない (押し込み時に弾く)。

単一ファイルコンポーネント (``.svelte`` / ``.vue``) は **``<script>`` ブロック
だけ**を JavaScript / TypeScript の grammar で検査し、テンプレート部は検査しない。
language-pack 1.20 の svelte grammar は Svelte 5 の構文 (``{@render x?.()}`` /
型注釈付きの ``{#snippet}``) を ERROR にし、このリポジトリの正しい ``.svelte``
75 本のうち 3 本を構文エラーと判定した (2026-09-20 実測)。ファイル全体を通すと
正しい成果物を失敗にして repair が書き換える。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from backend.log_config import get_logger

logger = get_logger("core.code_syntax")

#: 拡張子 → tree-sitter 言語名 (Python 以外で構文検査する言語)。
_TREE_SITTER_LANGUAGES: dict[str, str] = {
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
}

#: 単一ファイルコンポーネント。``<script>`` ブロックだけを検査する (モジュール docstring)。
_SFC_LANGUAGES: dict[str, str] = {
    ".svelte": "svelte",
    ".vue": "vue",
}

_SFC_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.DOTALL | re.IGNORECASE)
_SFC_TS_LANG_RE = re.compile(r"\blang\s*=\s*[\"']?(?:ts|typescript)\b", re.IGNORECASE)

_PARSERS: dict[str, Any] = {}

#: 言語パック (c_16 §4.5.3) が登録した拡張子表。``{拡張子: (grammar, label)}``。
#: ``set_language_overlay`` 経由でしか変更しない。
_LANGUAGE_OVERLAY: dict[str, tuple[str, str]] = {}


def set_language_overlay(overlay: Mapping[str, tuple[str, str]]) -> None:
    """言語パックの拡張子表を丸ごと差し替える (空 dict で解除、c_16 §4.5.3)。

    ``overlay`` は ``{拡張子: (tree-sitter grammar 名, 表示ラベル)}``。同梱の
    拡張子 (Python / :data:`_TREE_SITTER_LANGUAGES` / :data:`_SFC_LANGUAGES`)
    は上書きできない — 呼出側 (``CorpusStore``) が既にその衝突を弾いているが、
    防御的にここでも同梱側を優先する。
    """
    global _LANGUAGE_OVERLAY
    _LANGUAGE_OVERLAY = {
        ext: (str(grammar), str(label))
        for ext, (grammar, label) in overlay.items()
        if ext not in _TREE_SITTER_LANGUAGES
        and ext not in _SFC_LANGUAGES
        and ext not in ("", ".py")
    }


def is_python_path(path: str) -> bool:
    """Python のソースか (拡張子無しは従来どおり Python 扱い)。"""
    suffix = Path(path).suffix.lower()
    return suffix in ("", ".py")


def language_label(path: str) -> str:
    """指示文に書く言語名 (``python`` / ``javascript`` / ``html`` …、不明は拡張子)。

    同梱表 → 言語パック (c_16 §4.5.3) の順で引く。
    """
    if is_python_path(path):
        return "python"
    suffix = Path(path).suffix.lower()
    if suffix in _SFC_LANGUAGES:
        return _SFC_LANGUAGES[suffix]
    if suffix in _TREE_SITTER_LANGUAGES:
        return _TREE_SITTER_LANGUAGES[suffix]
    if suffix in _LANGUAGE_OVERLAY:
        return _LANGUAGE_OVERLAY[suffix][1]
    return suffix.lstrip(".")


def _parser(lang: str) -> Any | None:
    if lang in _PARSERS:
        return _PARSERS[lang]
    parser = None
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(lang)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001 — 無い環境は検査を省く
        logger.warning("tree-sitter parser unavailable for %s; skipping syntax check: %s", lang, e)
    _PARSERS[lang] = parser
    return parser


def _first_error_line(node: Any) -> int | None:
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing:
            return int(n.start_point[0]) + 1
        if n.has_error:
            stack.extend(reversed(n.children))
    return None


def _sfc_script_error(code: str, label: str) -> str | None:
    """SFC の ``<script>`` ブロックを順に検査する (行番号は元ファイルの位置で返す)。"""
    for match in _SFC_SCRIPT_RE.finditer(code):
        lang = "typescript" if _SFC_TS_LANG_RE.search(match.group(1)) else "javascript"
        parser = _parser(lang)
        if parser is None:
            continue
        tree = parser.parse(match.group(2).encode("utf-8"))
        if not tree.root_node.has_error:
            continue
        line = _first_error_line(tree.root_node)
        if line is None:
            return f"unknown line: {label} <script> syntax error"
        offset = code.count("\n", 0, match.start(2))
        return f"line {offset + line}: {label} <script> syntax error"
    return None


#: 誤りをほとんど出さない文法。「構文が通った」を本文かどうかの判定に使えない。
_ERROR_TOLERANT_LANGUAGES = frozenset({"html"})


def checks_syntax(path: str) -> bool:
    """:func:`syntax_error_detail` がこのパスの構文の誤りを実際に検出できるか。

    ``None`` (誤りなし) と「検査できない」を区別する呼出側 (生成応答からの本文の抜き出し) 用。
    """
    if is_python_path(path):
        return True
    suffix = Path(path).suffix.lower()
    if suffix in _SFC_LANGUAGES:
        return False
    lang = _TREE_SITTER_LANGUAGES.get(suffix)
    if lang is None and suffix in _LANGUAGE_OVERLAY:
        lang = _LANGUAGE_OVERLAY[suffix][0]
    return lang is not None and lang not in _ERROR_TOLERANT_LANGUAGES and _parser(lang) is not None


#: 構文検査器の任意依存 (pip の配布名)。無い環境では Python 以外の構文検査を飛ばす。
SYNTAX_CHECKER_PACKAGE = "tree-sitter-language-pack"


def syntax_checker_installed() -> bool:
    """構文検査器 (``tree_sitter_language_pack``) が import できるか (読み込まずに見る)。"""
    import importlib.util

    try:
        return importlib.util.find_spec("tree_sitter_language_pack") is not None
    except (ImportError, ValueError):
        return False


def syntax_checker_missing(path: str) -> str | None:
    """このパスの構文を tree-sitter で検査できないとき、欠けているもの (検査できる・対象外なら ``None``)。

    パッケージが無ければ :data:`SYNTAX_CHECKER_PACKAGE`、パッケージはあるのにその言語の文法が取れなければ
    ``tree-sitter grammar <言語>`` (f_10 §12.4: 欠けているものを正しく名指す)。
    Python (``compile``) と、表に無い拡張子 (PHP / SQL は実行環境で検査する) は対象外。
    """
    if is_python_path(path):
        return None
    suffix = Path(path).suffix.lower()
    if suffix in _SFC_LANGUAGES:
        lang: str | None = "javascript"
    else:
        lang = _TREE_SITTER_LANGUAGES.get(suffix)
        if lang is None and suffix in _LANGUAGE_OVERLAY:
            lang = _LANGUAGE_OVERLAY[suffix][0]
    if lang is None or _parser(lang) is not None:
        return None
    return SYNTAX_CHECKER_PACKAGE if not syntax_checker_installed() else f"tree-sitter grammar {lang}"


def syntax_check_unavailable(path: str) -> bool:
    """このパスは tree-sitter で構文を検査する言語なのに、構文検査器が無くて検査できないか。

    :func:`syntax_error_detail` の ``None`` (誤りなし) と「検査しなかった」を分けるため
    (f_10 §12.4: 飛ばした検査を「合格」と書かない、2026-09-27 ライブ監査 S9)。
    """
    return syntax_checker_missing(path) is not None


def runtime_environment() -> dict[str, Any]:
    """起動した Python の実行ファイルと任意依存の有無 (起動ログと ``evoref doctor`` が出す)。"""
    import sys

    return {
        "python": sys.executable,
        "optional_dependencies": {SYNTAX_CHECKER_PACKAGE: syntax_checker_installed()},
    }


def runtime_environment_line() -> str:
    """:func:`runtime_environment` の 1 行 (ログ用、英語固定)。"""
    env = runtime_environment()
    installed = env["optional_dependencies"][SYNTAX_CHECKER_PACKAGE]
    state = "installed" if installed else "missing; non-Python syntax checks are reported as not checked"
    return f"Python runtime: {env['python']} ({SYNTAX_CHECKER_PACKAGE}: {state})"


#: 終了タグを持たない要素 (HTML Living Standard の void elements と旧要素)
_HTML_VOID = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr",
    "basefont", "bgsound", "frame", "keygen", "command", "isindex", "image",
})
#: 終了タグを省略してよい要素 (閉じていなくても誤りにしない。親の終了タグが暗黙に閉じる)
_HTML_OPTIONAL_END = frozenset({
    "html", "head", "body", "p", "li", "dt", "dd", "rt", "rp", "rb", "rtc", "optgroup", "option",
    "colgroup", "caption", "thead", "tbody", "tfoot", "tr", "td", "th",
})
#: 中身を文字として読む要素 (中のタグを数えない)
_HTML_RCDATA = frozenset({"title", "textarea"})
#: テンプレートの構文 (条件で開閉が分かれるので平衡を数えない)
_HTML_TEMPLATE_RE = re.compile(r"\{%|\{\{|<\?|<%")


def html_balance_error(code: str) -> str | None:
    """HTML のタグの開閉の対応を検査し、明確な不平衡があれば ``line N: 理由`` を返す (無ければ ``None``)。

    tree-sitter の HTML 文法は閉じていない ``<div>`` を誤りにしない (2026-10-03 ライブ再実行 K02: 閉じていない
    ``<div class="timer">`` のまま静的検査が「合格」と書いた)。標準ライブラリの ``html.parser`` で開始・終了タグを
    数える。不合格にするのは、終了タグを省略できない要素 (``div`` / ``span`` / ``section`` …) が閉じていない・対応の
    無い終了タグがある、の 2 つだけ。void 要素 (``br`` / ``img`` …)、省略が合法な要素 (``p`` / ``li`` / ``td`` …)、
    自己終了の書き方 (``<path/>``) は数えない。テンプレートの構文 (``{% %}`` / ``{{ }}`` / ``<? ?>``) を含む本文は
    開閉が条件で分かれるので検査しない。
    """
    from html.parser import HTMLParser

    if _HTML_TEMPLATE_RE.search(code):
        return None
    problems: list[tuple[int, str]] = []
    stack: list[tuple[str, int]] = []

    def _in_rcdata() -> str:
        # title / textarea の中身は文字だけ (RCDATA)。html.parser が RCDATA として読むかは Python の版で違う
        # (3.12 はタグとして渡す) ので、自前で中身のタグを無視して版によらず同じ結果にする
        return stack[-1][0] if stack and stack[-1][0] in _HTML_RCDATA else ""

    class _Balance(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list) -> None:  # noqa: ARG002 - HTMLParser の署名
            if _in_rcdata():
                return
            if tag not in _HTML_VOID:
                stack.append((tag, self.getpos()[0]))

        def handle_endtag(self, tag: str) -> None:
            if tag in _HTML_VOID or _in_rcdata() not in ("", tag):
                return
            if not any(name == tag for name, _ in stack):
                if tag not in _HTML_OPTIONAL_END:
                    problems.append((self.getpos()[0], f"</{tag}> has no matching <{tag}>"))
                return
            while stack:
                name, line = stack.pop()
                if name == tag:
                    break
                if name not in _HTML_OPTIONAL_END:
                    problems.append((line, f"<{name}> is not closed (closed by </{tag}>)"))

    parser = _Balance(convert_charrefs=True)
    try:
        parser.feed(code)
        parser.close()
    except Exception:  # noqa: BLE001 — 読めない本文は平衡を判定しない (誤検出を避ける)
        return None
    problems += [(line, f"<{name}> is not closed") for name, line in stack if name not in _HTML_OPTIONAL_END]
    if not problems:
        return None
    line, reason = min(problems)
    return f"line {line}: html {reason}"


def syntax_error_detail(code: str, path: str) -> str | None:
    """``path`` の言語で構文を検査し、誤りがあれば ``line N: 理由`` を返す (無ければ ``None``)。"""
    if is_python_path(path):
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as exc:
            where = f"line {exc.lineno}" if exc.lineno else "unknown line"
            return f"{where}: {exc.msg}"
        except ValueError as exc:
            return str(exc)
        return None
    suffix = Path(path).suffix.lower()
    if suffix in _SFC_LANGUAGES:
        return _sfc_script_error(code, _SFC_LANGUAGES[suffix])
    lang = _TREE_SITTER_LANGUAGES.get(suffix)
    if lang is None and suffix in _LANGUAGE_OVERLAY:
        lang = _LANGUAGE_OVERLAY[suffix][0]
    if lang is None:
        return None
    parser = _parser(lang)
    if parser is None:
        return None
    tree = parser.parse(code.encode("utf-8"))
    if not tree.root_node.has_error:
        return None
    line = _first_error_line(tree.root_node)
    where = f"line {line}" if line else "unknown line"
    return f"{where}: {lang} syntax error"
