"""プロジェクトの走査 (c_16 §4.4)

除外ディレクトリ・``exclude_globs``・``max_file_bytes``・バイナリ判定を適用し、
言語が分かるファイルだけを **path 昇順の固定順**で返す。同じ入力から同じ
出力を作るための最初の段。
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: 既定の除外ディレクトリ (c_16 §4.4)。
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "models",
    "local", "dist", "build", "htmlcov", "coverage",
})

#: 拡張子 → tree-sitter 言語名 (c_16 §4.4 の対応言語)。
LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
}

#: 既定のバイト上限 (c_16 §9)。
DEFAULT_MAX_FILE_BYTES = 1_000_000

#: バイナリ判定に読む先頭バイト数。
_BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True, slots=True)
class ScannedFile:
    """走査で見つかった 1 ファイル。"""

    #: プロジェクトルートからの相対 posix パス。
    path: str
    #: tree-sitter 言語名。
    lang: str


def _is_excluded_dir(name: str, exclude_globs: Sequence[str]) -> bool:
    if name in DEFAULT_EXCLUDED_DIRS:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in exclude_globs)


def _matches_exclude_glob(rel_posix: str, exclude_globs: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(rel_posix, pattern) for pattern in exclude_globs)


def _looks_binary(path: Path) -> bool:
    """先頭バイトに NUL があればバイナリとみなす。"""
    try:
        with path.open("rb") as f:
            chunk = f.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True
    return b"\x00" in chunk


def scan_project(
    root: Path | str,
    *,
    exclude_globs: Sequence[str] = (),
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[ScannedFile]:
    """``root`` 配下を走査し、対応言語のファイルを path 昇順で返す。

    除外は :data:`DEFAULT_EXCLUDED_DIRS` + ``exclude_globs`` (ディレクトリ名 /
    相対パスの両方に掛ける) + ``max_file_bytes`` + バイナリ判定 (NUL バイト)。
    """
    base = Path(root)
    if not base.is_dir():
        return []
    out: list[ScannedFile] = []
    # ``rglob`` は除外ディレクトリ (node_modules / .venv / local / models …) の中まで
    # 列挙してから捨てるので、2,000 ファイルのリポジトリで 17 秒掛かった。
    # ``os.walk`` で降りる前に刈る (``search_code`` ツールと同じ形)。
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(
            d for d in dirnames if not _is_excluded_dir(d, exclude_globs)
        )
        rel_dir = Path(dirpath).relative_to(base)
        for filename in filenames:
            lang = LANGUAGE_EXTENSIONS.get(Path(filename).suffix.lower())
            if lang is None:
                continue
            rel_posix = (rel_dir / filename).as_posix() if rel_dir.parts else filename
            if _matches_exclude_glob(rel_posix, exclude_globs):
                continue
            path = Path(dirpath) / filename
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_file_bytes:
                continue
            if _looks_binary(path):
                continue
            out.append(ScannedFile(path=rel_posix, lang=lang))
    out.sort(key=lambda f: f.path)
    return out


__all__ = [
    "DEFAULT_EXCLUDED_DIRS",
    "DEFAULT_MAX_FILE_BYTES",
    "LANGUAGE_EXTENSIONS",
    "ScannedFile",
    "scan_project",
]
