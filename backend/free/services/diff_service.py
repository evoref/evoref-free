"""Diff パース・適用サービス (docs/f_11 §5.7)

LLM 応答からの unified diff 検出・解析・ファイル適用ロジック。
CLI の diff_applier とエージェントの ``apply_diff`` ツールが
:func:`apply_diff_to_file` 1 本を通る。

解析は自前で行う。``whatthepatch`` は hunk 見出しの行数で本文を打ち切るため
行数の書き損じで余った行を黙って捨て、見出しの無い 2 ファイル目の hunk を
1 ファイル目へ混ぜていた。ここでは行の記号で本文を拾い、見出しの行数は
検証にだけ使う。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from backend.io.text_file import TextFile, read_text_for_edit
from backend.io.user_file import write_user_text
from backend.log_config import get_logger

logger = get_logger("services.diff_service")

#: 適用先として読むファイルの上限 (``read_file`` ツールの上限と同じ)。
DEFAULT_MAX_FILE_BYTES = 2_000_000

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

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_HUNK_HEADER_SEARCH_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)

#: 1 ファイルの本文編集として当てられない diff (バイナリ / 改名・複製 / モード変更)。
_UNSUPPORTED_RE = re.compile(
    r"^(?:Binary files .* differ$|GIT binary patch"
    r"|rename (?:from|to) |copy (?:from|to) |(?:dis)?similarity index "
    r"|old mode |new mode |new file mode |deleted file mode )",
)

_DEV_NULL = "/dev/null"


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
        return _HUNK_HEADER_SEARCH_RE.search(self.raw) is not None


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

    @property
    def old_lines(self) -> list[str]:
        """適用前の並び (文脈行 + 削除行)。"""
        return [c for t, c in self.lines if t != "+"]

    @property
    def new_lines(self) -> list[str]:
        """適用後の並び (文脈行 + 追加行)。"""
        return [c for t, c in self.lines if t != "-"]


@dataclass(frozen=True)
class ParsedDiff:
    """1 ファイル分の diff。見出しが無ければパスは ``None``。"""
    old_path: str | None
    new_path: str | None
    hunks: list[Hunk]


@dataclass(frozen=True)
class DiffApplyResult:
    """適用の結果。``already_applied`` なら何も書いていない。

    ``moved_hunks`` は見出しの行番号と違う位置で一致した hunk の数。
    """
    path: Path
    already_applied: bool
    moved_hunks: int
    bytes_written: int
    backup: Path | None


class DiffServiceError(Exception):
    """Diff サービスエラー。``code`` は断った理由 (docs/f_11 §5.7)。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


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


# ── 解析 ──────────────────────────────────────────────────────────────────


def parse_unified_diff(diff_text: str) -> ParsedDiff:
    """1 ファイル分の unified diff を解析する。

    Raises:
        DiffServiceError: 複数ファイル・バイナリ等・hunk の外の行・行数の食い違い・hunk 無し。
    """
    text = diff_text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    old_path: str | None = None
    new_path: str | None = None
    file_headers = 0
    diff_lines = 0
    hunks: list[Hunk] = []
    started = False
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HUNK_HEADER_RE.match(line)
        if m:
            hunk, i = _read_hunk(lines, i, m, len(hunks) + 1)
            hunks.append(hunk)
            started = True
            continue
        if _is_file_header(lines, i):
            file_headers += 1
            if file_headers > 1 or hunks:
                raise _multi_file()
            old_path = _header_name(line)
            new_path = _header_name(lines[i + 1])
            if _DEV_NULL in (old_path, new_path):
                raise DiffServiceError(
                    "unsupported",
                    "unsupported diff: creating or deleting a file (/dev/null) "
                    "is not an edit of the existing file",
                )
            started = True
            i += 2
            continue
        if line.startswith(("diff --git ", "diff -")):
            diff_lines += 1
            if diff_lines > 1 or file_headers or hunks:
                raise _multi_file()
            i += 1
            continue
        if _UNSUPPORTED_RE.match(line):
            raise DiffServiceError("unsupported", f"unsupported diff (binary / rename / mode change): {line[:80]}")
        if line.startswith("@@"):
            raise DiffServiceError("malformed", f"malformed hunk header at line {i + 1}: {line[:80]}")
        if started and line.strip():
            # 最後の hunk の後ろの説明文は読み飛ばす。後ろにまだ hunk や見出しが
            # あれば、見出しの無い別ファイルの hunk を混ぜかねないので断る。
            if any(_starts_patch_part(lines, k) for k in range(i + 1, len(lines))):
                raise DiffServiceError(
                    "unexpected_line",
                    f"unexpected line {i + 1} outside a hunk: {line[:80]!r}",
                )
            break
        # 最初の見出しより前の前置き (git の index 行・説明文) と空行は読み飛ばす
        i += 1
    if not hunks:
        raise DiffServiceError("no_hunks", "No valid hunks found in diff")
    return ParsedDiff(old_path=old_path, new_path=new_path, hunks=hunks)


def _multi_file() -> DiffServiceError:
    return DiffServiceError("multi_file", "diff must change exactly one file (found another file header)")


def _starts_patch_part(lines: list[str], i: int) -> bool:
    line = lines[i]
    return (
        line.startswith(("@@", "diff --git ", "diff -"))
        or _is_file_header(lines, i)
        or _UNSUPPORTED_RE.match(line) is not None
    )


def _is_file_header(lines: list[str], i: int) -> bool:
    return (
        lines[i].startswith("--- ")
        and i + 1 < len(lines)
        and lines[i + 1].startswith("+++ ")
    )


def _header_name(line: str) -> str:
    """``--- path\\tdate`` からパスを取り出す (タブより後は日付)。"""
    name = line[4:].split("\t", 1)[0].strip()
    if len(name) >= 2 and name[0] == name[-1] == '"':
        name = name[1:-1]
    return name


def _read_hunk(lines: list[str], i: int, m: re.Match[str], index: int) -> tuple[Hunk, int]:
    """``lines[i]`` の見出しから本文を記号で拾い、次に読む位置を返す。"""
    old_start, new_start = int(m[1]), int(m[3])
    old_count = int(m[2]) if m[2] is not None else 1
    new_count = int(m[4]) if m[4] is not None else 1
    body: list[tuple[str, str]] = []
    old_seen = new_seen = 0
    j = i + 1
    while j < len(lines):
        line = lines[j]
        short = old_seen < old_count and new_seen < new_count
        if line == "":
            # エディタが文脈行の先頭の空白を削った空行。後に同じ hunk の行が
            # 続くか、見出しの行数がまだ足りないときだけ空の文脈行とみなす。
            k = j
            while k < len(lines) and lines[k] == "":
                k += 1
            follows = (
                k < len(lines)
                and lines[k][:1] in (" ", "+", "-", "\\")
                and not _is_file_header(lines, k)
            )
            if not (follows or short):
                break
            body.append((" ", ""))
            old_seen += 1
            new_seen += 1
            j += 1
            continue
        kind = line[0]
        if kind == "\\":  # "\ No newline at end of file" — 末尾改行は元のファイルに従う
            j += 1
            continue
        if kind == "-" and _is_file_header(lines, j) and not short:
            break
        if kind not in (" ", "+", "-"):
            break
        body.append((kind, line[1:]))
        if kind != "+":
            old_seen += 1
        if kind != "-":
            new_seen += 1
        j += 1
    if not body:
        raise DiffServiceError("malformed", f"hunk {index} has no lines")
    if (old_seen, new_seen) != (old_count, new_count):
        raise DiffServiceError(
            "count_mismatch",
            f"hunk {index} header says -{old_start},{old_count} +{new_start},{new_count} "
            f"but its body has {old_seen} old / {new_seen} new line(s); "
            "fix the line counts in the @@ header",
        )
    return Hunk(
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
        context_lines=[c for t, c in body if t == " "],
        remove_lines=[c for t, c in body if t == "-"],
        add_lines=[c for t, c in body if t == "+"],
        lines=body,
    ), j


# ── 宛先の照合 ─────────────────────────────────────────────────────────────


def check_header_paths(parsed: ParsedDiff, target: Path) -> None:
    """見出しのパスが ``target`` そのものを指すか検査する (見出しが無ければ通す)。

    相対パスは ``a/`` / ``b/`` を外した形か外さない形が ``target`` の絶対パスの
    末尾の成分と一致すること。絶対パスは同じパスであること。

    Raises:
        DiffServiceError: ``..`` を含む / 別のファイルを指す。
    """
    resolved = target.resolve()
    for name, prefix in ((parsed.old_path, "a/"), (parsed.new_path, "b/")):
        if not name:
            continue
        slashed = name.replace("\\", "/")
        if ".." in slashed.split("/"):
            raise DiffServiceError("traversal", f"path traversal not allowed in diff header: {name}")
        candidates = [slashed]
        if slashed.startswith(prefix):
            candidates.append(slashed[len(prefix):])
        if not any(_names_target(c, resolved) for c in candidates):
            raise DiffServiceError(
                "header_mismatch",
                f"diff header {name} does not name the target file {target}",
            )


def _names_target(name: str, resolved: Path) -> bool:
    """``name`` (``/`` 区切り) が ``resolved`` を指すか。相対なら末尾の成分と比べる。"""
    if os.path.isabs(name) or re.match(r"^[A-Za-z]:", name):
        try:
            return os.path.normcase(str(Path(name).resolve())) == os.path.normcase(str(resolved))
        except (OSError, ValueError):
            return False
    parts = [os.path.normcase(p) for p in name.split("/") if p not in ("", ".")]
    target_parts = [os.path.normcase(p) for p in resolved.parts]
    if not parts or len(parts) >= len(target_parts):
        return False
    return parts == target_parts[-len(parts):]


# ── 位置合わせと適用 ──────────────────────────────────────────────────────


def _find(lines: list[str], block: list[str], limit: int) -> list[int]:
    """``block`` が ``lines`` に連続して現れる位置 (先頭から ``limit`` 個まで)。"""
    found: list[int] = []
    first, size = block[0], len(block)
    for pos in range(len(lines) - size + 1):
        if lines[pos] == first and lines[pos:pos + size] == block:
            found.append(pos)
            if len(found) >= limit:
                break
    return found


def patch_lines(lines: list[str], hunks: list[Hunk]) -> tuple[list[str] | None, int]:
    """hunk を当てた行のリストと、見出しと違う位置で当たった hunk の数を返す。

    各 hunk の適用前の並びがファイル全体でちょうど 1 か所に完全一致することを求める
    (fuzz なし)。当て済みなら ``(None, 0)``。

    Raises:
        DiffServiceError: 一致しない / 2 か所以上に一致する / 順序が逆・重なる。
    """
    positions: list[int] = []
    missing: int | None = None
    for index, hunk in enumerate(hunks, 1):
        old = hunk.old_lines
        if not old:
            # 文脈の無い純粋な追加は見出しの行番号 (-a,0 の a 行目の直後) に入れる
            if hunk.old_start > len(lines):
                raise DiffServiceError(
                    "no_match",
                    f"hunk {index} inserts after line {hunk.old_start} but the file has {len(lines)} line(s)",
                )
            positions.append(hunk.old_start)
            continue
        found = _find(lines, old, limit=2)
        if len(found) > 1:
            raise DiffServiceError(
                "ambiguous",
                f"hunk {index} is ambiguous: its context and removed lines appear more than once "
                f"(lines {found[0] + 1} and {found[1] + 1}); add more context lines",
            )
        if not found:
            missing = missing or index
            positions.append(-1)
            continue
        positions.append(found[0])
    if missing is not None:
        if _already_applied(lines, hunks):
            return None, 0
        raise DiffServiceError(
            "no_match",
            f"hunk {missing} does not match the file: its context and removed lines were not found "
            "exactly (the file may have changed, or only part of the diff is already applied)",
        )
    out: list[str] = []
    end = 0
    moved = 0
    for index, (hunk, pos) in enumerate(zip(hunks, positions, strict=True), 1):
        if pos < end:
            raise DiffServiceError(
                "overlap",
                f"hunk {index} is out of order or overlaps the previous hunk",
            )
        if hunk.old_lines and pos != hunk.old_start - 1:
            moved += 1
        out.extend(lines[end:pos])
        out.extend(hunk.new_lines)
        end = pos + len(hunk.old_lines)
    out.extend(lines[end:])
    return out, moved


def _already_applied(lines: list[str], hunks: list[Hunk]) -> bool:
    """全 hunk の適用後の並びが順に一意に見つかり、適用前の並びがどこにも無いか。"""
    end = 0
    for hunk in hunks:
        old, new = hunk.old_lines, hunk.new_lines
        if old and _find(lines, old, limit=1):
            return False
        if not new:
            continue
        if not old:
            pos = hunk.new_start - 1
            if lines[pos:pos + len(new)] != new:
                return False
        else:
            found = _find(lines, new, limit=2)
            if len(found) != 1:
                return False
            pos = found[0]
        if pos < end:
            return False
        end = pos + len(new)
    return True


def apply_diff_to_file(
    path: Path | str, diff_text: str, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> DiffApplyResult:
    """1 ファイル分の unified diff を既存ファイルへ当てる (docs/f_11 §5.7)。

    読みは ``read_text_for_edit``、書きは ``write_user_text`` (符号化・改行・触って
    いない行のバイト列を保ち、上書きの前に退避する)。同期 I/O。

    Raises:
        DiffServiceError: diff を当てられない (``code`` が理由。``existing_unreadable``
            はファイルを編集の素材として読めない)。
        UnencodableTextError / UserFileWriteError / OSError: 書込みの失敗 (元のファイルは残る)。
    """
    target = Path(path)
    parsed = parse_unified_diff(diff_text)
    check_header_paths(parsed, target)
    existing = read_text_for_edit(target, max_bytes=max_bytes)
    if existing is None:
        raise DiffServiceError("not_found", f"File not found: {target}")
    if not isinstance(existing, TextFile):
        detail = f": {existing.detail}" if existing.detail else ""
        raise DiffServiceError(
            "existing_unreadable",
            f"cannot read {target} as text for editing (existing_unreadable: {existing.reason}{detail})",
        )
    lines = existing.text.split("\n") if existing.text else []
    if existing.trailing_newline:
        lines.pop()
    new_lines, moved = patch_lines(lines, parsed.hunks)
    if new_lines is None:
        logger.info("Diff already applied to %s; nothing written", target)
        return DiffApplyResult(target, True, 0, 0, None)
    new_text = "\n".join(new_lines)
    if new_lines and (existing.trailing_newline or not lines):
        new_text += "\n"
    if moved:
        logger.info("%d hunk(s) matched away from their header line in %s", moved, target)
    result = write_user_text(target, new_text, like=existing)
    return DiffApplyResult(target, False, moved, result.bytes_written, result.backup)


def apply_unified_diff(file_path: str, diff_text: str) -> tuple[bool, str]:
    """unified diff をファイルに適用する (CLI 用。利用者の確認の後に呼ぶ)。

    見出しの行番号と違う位置でも一意に一致すれば当てる (確認済みなので)。
    当てられなければ原文を保持する。

    Returns:
        (success, message)
    """
    p = Path(file_path)
    if not p.is_file():
        return False, f"File not found: {file_path}"
    try:
        result = apply_diff_to_file(p, diff_text)
    except DiffServiceError as e:
        return False, str(e)
    except OSError as e:  # UserFileWriteError / UnencodableTextError を含む
        return False, f"Failed to write file: {e}"
    if result.already_applied:
        return True, f"Diff already applied to {file_path} (no changes written)"
    if result.moved_hunks:
        return True, (
            f"Diff applied to {file_path} "
            f"({result.moved_hunks} hunk(s) matched at a different line than the header)"
        )
    return True, f"Diff applied to {file_path}"
