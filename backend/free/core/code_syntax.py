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


def is_known_non_python_code_path(path: str) -> bool:
    """Python 以外の **コード** として扱える拡張子か (同梱表 → 言語パック)。``.json`` はデータなので除く。

    URL の ``.com`` のような未知の拡張子は偽 (依頼文から拾うときの誤爆を避ける)。
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return False
    return suffix in _TREE_SITTER_LANGUAGES or suffix in _SFC_LANGUAGES or suffix in _LANGUAGE_OVERLAY


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
