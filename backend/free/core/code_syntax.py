"""生成コードの言語別構文検査 (staged create、f_10 §4.2)。

staged の code 工程は全ファイルを Python の ``compile`` に通していたため、
HTML / CSS / JavaScript の成果物が「invalid character '、'」で全件失敗した
(2026-09-19 ライブ監査 K05)。Python は従来どおり ``compile`` (コンパイル時にしか
出ない誤りも拾う)、それ以外は tree-sitter の構文木に ERROR / MISSING が
あるかで判定する。tree-sitter が無い環境、または対応表に無い拡張子は
**検査しない** (``None``) — 検査できないことを構文エラーと扱わない。
"""

from __future__ import annotations

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
    ".json": "json",
}

_PARSERS: dict[str, Any] = {}


def is_python_path(path: str) -> bool:
    """Python のソースか (拡張子無しは従来どおり Python 扱い)。"""
    suffix = Path(path).suffix.lower()
    return suffix in ("", ".py")


def language_label(path: str) -> str:
    """指示文に書く言語名 (``python`` / ``javascript`` / ``html`` …、不明は拡張子)。"""
    if is_python_path(path):
        return "python"
    suffix = Path(path).suffix.lower()
    return _TREE_SITTER_LANGUAGES.get(suffix, suffix.lstrip("."))


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
    lang = _TREE_SITTER_LANGUAGES.get(Path(path).suffix.lower())
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
