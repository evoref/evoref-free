"""Diff パース・適用サービス

LLM 応答からの unified diff 検出・解析・ファイル適用ロジック。
CLI の diff_applier および GUI の diff 適用 API から利用される。

hunk 単位のパース層は ``whatthepatch`` に委譲し
LLM 応答固有のコードフェンス抽出と前後 50 行ファジー適用は
本モジュールで自前実装を維持する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import whatthepatch

from backend.log_config import get_logger

logger = get_logger("services.diff_service")

# ```diff ... ``` ブロックを抽出する正規表現
_DIFF_BLOCK_RE = re.compile(
    r"```diff\s*\n(.*?)```",
    re.DOTALL,
)

# unified diff のファイルパスヘッダー
_DIFF_HEADER_RE = re.compile(
    r"^(?:---|\+\+\+)\s+(?:[ab]/)?(.*?)(?:\s|$)",
    re.MULTILINE,
)


@dataclass
class DiffBlock:
    """抽出された diff ブロック"""
    raw: str
    file_path: str | None

    @property
    def has_file_path(self) -> bool:
        return self.file_path is not None

    @property
    def has_hunks(self) -> bool:
        return bool(_parse_hunks(self.raw))


@dataclass
class Hunk:
    """パース済み hunk"""
    old_start: int  # 1-based
    old_count: int
    new_start: int
    new_count: int
    context_lines: list[str]   # ' ' で始まる行（改行なし）
    remove_lines: list[str]    # '-' で始まる行（改行なし）
    add_lines: list[str]       # '+' で始まる行（改行なし）
    lines: list[tuple[str, str]]  # (type, content) — type: ' ', '-', '+'


class DiffServiceError(Exception):
    """Diff サービスエラー"""
    pass


def extract_diffs(response: str) -> list[DiffBlock]:
    """LLM 応答テキストから ```diff``` ブロックを抽出

    Returns:
        DiffBlock のリスト（検出順）
    """
    blocks: list[DiffBlock] = []
    for m in _DIFF_BLOCK_RE.finditer(response):
        raw = m.group(1).strip()
        if not raw:
            continue
        file_path = _extract_file_path(raw)
        blocks.append(DiffBlock(raw=raw, file_path=file_path))
    return blocks


def _extract_file_path(diff_text: str) -> str | None:
    """diff テキストからファイルパスを抽出

    優先順位:
    1. +++ ヘッダーのパス（変更後ファイル）
    2. --- ヘッダーのパス（変更前ファイル）
    /dev/null は除外する
    """
    plus_path: str | None = None
    minus_path: str | None = None

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = _parse_header_path(line)
            if path and path != "/dev/null":
                plus_path = path
        elif line.startswith("--- "):
            path = _parse_header_path(line)
            if path and path != "/dev/null":
                minus_path = path

    return plus_path or minus_path


def _parse_header_path(header_line: str) -> str | None:
    """--- / +++ ヘッダー行からパスを抽出"""
    m = _DIFF_HEADER_RE.match(header_line)
    if m:
        return m.group(1).strip()
    return None


def apply_unified_diff(file_path: str, diff_text: str) -> tuple[bool, str]:
    """unified diff をファイルに適用（Python 純正実装）

    patch コマンドに依存せず、hunk ヘッダーを解析して適用する。
    適用失敗時は原文を保持する。

    Returns:
        (success, message)
    """
    p = Path(file_path)
    if not p.exists():
        return False, f"File not found: {file_path}"

    try:
        original = p.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"Failed to read file: {e}"

    original_lines = original.splitlines(keepends=True)

    # hunk を解析
    hunks = _parse_hunks(diff_text)
    if not hunks:
        return False, "No valid hunks found in diff"

    # hunk を逆順に適用（後方から適用することで行番号のずれを防ぐ）
    result_lines = list(original_lines)
    for hunk in reversed(hunks):
        success, result_lines = _apply_hunk(result_lines, hunk)
        if not success:
            return False, f"Failed to apply hunk at line {hunk.old_start}"

    try:
        p.write_text("".join(result_lines), encoding="utf-8")
    except OSError as e:
        # 書き込み失敗時に原文を復元
        try:
            p.write_text(original, encoding="utf-8")
        except OSError:
            pass
        return False, f"Failed to write file: {e}"

    return True, f"Diff applied to {file_path}"


def _parse_hunks(diff_text: str) -> list[Hunk]:
    """diff テキストから hunk のリストをパース

    ``whatthepatch.parse_patch`` でコア解析を行い
    Change イテレータを ``Change.hunk`` 単位でグループ化して
    本モジュールの ``Hunk`` dataclass に変換する。
    no-newline-at-end-of-file マーカーや単一行 hunk
    (``@@ -1 +1 @@``) など標準的な unified diff 形式の細部は
    whatthepatch 側で吸収される。
    """
    hunks: list[Hunk] = []

    try:
        patches = whatthepatch.parse_patch(diff_text)
    except Exception as e:  # noqa: BLE001 — whatthepatch の例外型は private
        logger.debug("whatthepatch.parse_patch failed: %s", e)
        return []

    for patch in patches:
        if not patch or not patch.changes:
            continue

        # Change.hunk (1-based) 単位でグループ化
        # whatthepatch は changes を unified diff の出現順に保持するため
        # groupby で連続 hunk を素直に分割できる
        for _, group_iter in groupby(patch.changes, key=lambda c: c.hunk):
            group = list(group_iter)

            hunk_lines: list[tuple[str, str]] = []
            context: list[str] = []
            removes: list[str] = []
            adds: list[str] = []

            for change in group:
                content = change.line if change.line is not None else ""
                if change.old is not None and change.new is not None:
                    hunk_lines.append((" ", content))
                    context.append(content)
                elif change.old is not None:
                    hunk_lines.append(("-", content))
                    removes.append(content)
                elif change.new is not None:
                    hunk_lines.append(("+", content))
                    adds.append(content)

            olds = [c.old for c in group if c.old is not None]
            news = [c.new for c in group if c.new is not None]
            # 純追加 hunk (old_count=0) では old_start を new 側基準で代用、
            # 純削除 hunk (new_count=0) は逆。ファジー適用は context/remove
            # の存在に依存するため、空 hunk のフォールバックは安全側
            old_start = min(olds) if olds else (min(news) if news else 1)
            new_start = min(news) if news else (min(olds) if olds else 1)
            old_count = len(olds)
            new_count = len(news)

            hunks.append(Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                context_lines=context,
                remove_lines=removes,
                add_lines=adds,
                lines=hunk_lines,
            ))

    return hunks


def _apply_hunk(
    file_lines: list[str], hunk: Hunk,
) -> tuple[bool, list[str]]:
    """1つの hunk をファイル行リストに適用

    Returns:
        (success, result_lines)
    """
    # 0-based index
    start = hunk.old_start - 1

    # hunk の old 行（コンテキスト + 削除行）を構築
    old_lines: list[str] = []
    new_lines: list[str] = []
    for line_type, content in hunk.lines:
        if line_type in (" ", "-"):
            old_lines.append(content)
        if line_type in (" ", "+"):
            new_lines.append(content)

    # まず指定位置でマッチを試行
    if _lines_match(file_lines, start, old_lines):
        return True, _replace_lines(file_lines, start, len(old_lines), new_lines)

    # マッチしない場合、前後 50 行の範囲でファジー検索
    for offset in range(1, 51):
        for candidate in (start - offset, start + offset):
            if 0 <= candidate <= len(file_lines):
                if _lines_match(file_lines, candidate, old_lines):
                    logger.debug(
                        "Hunk at line %d matched at offset %+d (line %d)",
                        hunk.old_start, candidate - start, candidate + 1,
                    )
                    return True, _replace_lines(
                        file_lines, candidate, len(old_lines), new_lines,
                    )

    return False, file_lines


def _lines_match(
    file_lines: list[str], start: int, expected: list[str],
) -> bool:
    """ファイル行と期待行がマッチするかチェック（末尾空白を無視）"""
    if start < 0 or start + len(expected) > len(file_lines):
        return False
    for i, exp in enumerate(expected):
        actual = file_lines[start + i].rstrip("\n").rstrip("\r")
        if actual.rstrip() != exp.rstrip():
            return False
    return True


def _replace_lines(
    file_lines: list[str],
    start: int,
    remove_count: int,
    new_lines: list[str],
) -> list[str]:
    """ファイル行リスト内の指定範囲を新しい行で置換"""
    # 改行を付与
    newline = "\n"
    if file_lines and file_lines[0].endswith("\r\n"):
        newline = "\r\n"

    formatted_new = [line + newline for line in new_lines]

    result = file_lines[:start] + formatted_new + file_lines[start + remove_count:]
    return result
