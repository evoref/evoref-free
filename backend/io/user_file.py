"""利用者のファイルへ書く (docs/f_11 §5.5 / §5.6)。

出力先・既存ファイルの編集・export の成果物など、利用者が見るファイルの書き手は
ここの 2 本 (:func:`write_user_text` / :func:`write_user_bytes`) だけにする。

- 既存ファイルの符号化 (BOM 含む)・改行・末尾改行を保ち、**触っていない行は元の
  バイト列のまま**書く。cp932 には符号化し直すと元に戻らない字 (NEC 選定 IBM 拡張
  などの重複符号) があるので、全体を符号化し直すと編集していない行まで変わる。
- 中身の違う既存ファイルは置き換える前に ``<data_root>/bk/overwrite/`` へ退避する。
  退避できなければ書かない。
- 同じフォルダの一時ファイルへバイト列で書いてから置き換える (途中で失敗しても元の
  ファイルは残る)。利用者の成果物なので ``fsync`` はしない。
"""

from __future__ import annotations

import contextlib
import contextvars
import difflib
import os
import re
import secrets
import shutil
import stat
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from backend.io._retry import _replace_with_retry
from backend.io.text_file import TextFile
from backend.log_config import get_logger

logger = get_logger("io.user_file")

#: 新規に作るとき CRLF で書く拡張子 (cmd.exe は LF だけのバッチを読み違える)。
_CRLF_SUFFIXES = frozenset({".bat", ".cmd"})
_UTF8_BOM = b"\xef\xbb\xbf"
#: 行末の分割。cp932 / UTF-8 の多バイト文字は 0x0A / 0x0D を含まないのでバイト列で切れる。
_LINE_BREAK_RE = re.compile(rb"(\r\n|\n|\r)")
_ZONE_IDENTIFIER = ":Zone.Identifier"
_MAX_REPORTED_CHARS = 10

#: 退避の保持 (f_11 §5.6)。退避のたびに刈る。
BACKUP_KEEP_DAYS = 30
BACKUP_MAX_BYTES = 1 << 30
#: 退避のファイル名に使う元の名前の上限 (長い名前は切り詰める)。
_MAX_BACKUP_NAME = 80

_backup_sink: contextvars.ContextVar[list[tuple[Path, Path]] | None] = contextvars.ContextVar(
    "evoref_user_file_backups", default=None,
)


class UserFileWriteError(OSError):
    """利用者のファイルを書けなかった。``code`` は理由 (``hardlink`` / ``readonly`` /
    ``not_a_file`` / ``backup_failed`` / ``verify_failed`` / ``unencodable``)。"""

    def __init__(self, code: str, path: Path, detail: str = "") -> None:
        self.code = code
        self.path = path
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"cannot write {path} ({code}){suffix}")


class UnencodableTextError(UserFileWriteError):
    """既存ファイルの符号化で表せない文字がある (``chars`` はその文字、先頭 10 種)。"""

    def __init__(self, path: Path, encoding: str, chars: str) -> None:
        self.encoding = encoding
        self.chars = chars
        super().__init__("unencodable", path, f"{encoding} cannot encode {chars!r}")


@dataclass(frozen=True, slots=True)
class UserWriteResult:
    """書いた結果。``path`` は実際に書いたパス (シンボリックリンクならリンク先)。

    ``encoding`` / ``newline`` は :func:`write_user_bytes` では ``None``。``backup`` は
    上書きの前に退避した先 (新規 / 中身が同じなら ``None``)。
    """

    path: Path
    bytes_written: int
    encoding: str | None
    newline: str | None
    backup: Path | None


@contextlib.contextmanager
def record_backups() -> Iterator[list[tuple[Path, Path]]]:
    """この文脈で行った退避を ``(書いたパス, 退避先)`` のリストに集める。

    ツールの戻り値の形式を変えずに退避先を知りたい呼び出し側 (create の配信記録) が使う。
    ``asyncio.to_thread`` などは文脈を写すので、スレッドで書いても同じリストに届く。
    """
    recorded: list[tuple[Path, Path]] = []
    token = _backup_sink.set(recorded)
    try:
        yield recorded
    finally:
        _backup_sink.reset(token)


def write_user_text(path: Path | str, text: str, *, like: TextFile | None) -> UserWriteResult:
    """テキストを書く。``like`` は既存ファイルを :func:`read_text_for_edit` で読んだ結果。

    ``like`` があれば符号化・BOM・改行・末尾改行を保ち、変わらない行は元のバイト列の
    まま書く。無ければ UTF-8 (BOM なし)・LF (``.bat`` / ``.cmd`` は CRLF)。

    Raises:
        UnencodableTextError: 既存ファイルの符号化で表せない文字がある (何も書かない)。
        UserFileWriteError: ハードリンク・読み取り専用・退避の失敗・読み戻しの不一致。
        OSError: そのほかの書込みの失敗 (元のファイルは残る)。
    """
    target = _resolve_target(Path(path))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if like is None:
        newline = "\r\n" if target.suffix.lower() in _CRLF_SUFFIXES else "\n"
        data = _encode(text.replace("\n", newline), "utf-8", target)
        encoding = "utf-8"
    else:
        data = _splice(like, _match_trailing_newline(text, like), target)
        encoding, newline = like.encoding, like.newline
    size, backup = _write_bytes(target, data)
    return UserWriteResult(target, size, encoding, newline, backup)


def write_user_bytes(path: Path | str, data: bytes) -> UserWriteResult:
    """バイト列 (export の成果物) を書く。退避・置き換え・検査は :func:`write_user_text` と同じ。"""
    target = _resolve_target(Path(path))
    size, backup = _write_bytes(target, data)
    return UserWriteResult(target, size, None, None, backup)


def roll_back_user_write(path: Path | str, previous: bytes | None) -> None:
    """直前の :func:`write_user_bytes` を取り消し、書く前の状態へ戻す。

    ``previous`` が ``None`` (書く前は無かった) なら書いたファイルを消す。シンボリック
    リンクならリンクではなく書いたリンク先を消す (リンクは書く前から在った)。
    戻すのは書き直しではないので **退避しない**。直前の書込みが :func:`record_backups`
    へ積んだ退避の記録も外す (配信記録に「置き換え」を残さない、f_11 §3.3)。

    Raises:
        OSError: 戻せなかった (書いた中身が残っている)。
    """
    target = _resolve_target(Path(path))
    sink = _backup_sink.get()
    if sink is not None:
        for index in range(len(sink) - 1, -1, -1):
            if sink[index][0] == target:
                del sink[index]
                break
    if previous is None:
        target.unlink(missing_ok=True)
        return
    tmp = _write_tmp(target, previous, target.stat())
    try:
        _replace_with_retry(tmp, target)
    except BaseException:
        _discard(tmp)
        raise


# ── 符号化 ────────────────────────────────────────────────────────────────


def _match_trailing_newline(text: str, like: TextFile) -> str:
    """末尾改行の有無を既存ファイルに合わせる (改行 1 つだけを足す / 外す)。"""
    if not text or not like.text:
        return text
    if like.trailing_newline and not text.endswith("\n"):
        return text + "\n"
    if not like.trailing_newline and text.endswith("\n"):
        return text[:-1]
    return text


def _encode(text: str, codec: str, path: Path) -> bytes:
    try:
        return text.encode(codec)
    except UnicodeEncodeError:
        raise UnencodableTextError(path, codec, _unencodable_chars(text, codec)) from None


def _unencodable_chars(text: str, codec: str) -> str:
    bad: list[str] = []
    for ch in dict.fromkeys(text):
        try:
            ch.encode(codec)
        except UnicodeEncodeError:
            bad.append(ch)
            if len(bad) >= _MAX_REPORTED_CHARS:
                break
    return "".join(bad)


def _splice(like: TextFile, text: str, path: Path) -> bytes:
    """``text`` を ``like`` の形で符号化する。``like.text`` と同じ行は元のバイト列と行末を使う。"""
    codec = "utf-8" if like.encoding == "utf-8-sig" else like.encoding
    raw, bom = like.raw, b""
    if like.encoding == "utf-8-sig" and raw.startswith(_UTF8_BOM):
        raw, bom = raw[len(_UTF8_BOM):], _UTF8_BOM
    parts = _LINE_BREAK_RE.split(raw)
    old_raw, old_ends = parts[0::2], parts[1::2]
    old_lines = like.text.split("\n")
    newline = like.newline.encode("ascii")
    if len(old_lines) != len(old_raw):
        # read_text_for_edit の正規化と食い違うことは無いはずだが、行の対応が取れないなら
        # 既存の行を使わずに全体を符号化する (内容は正しいまま)。
        logger.warning("Line split mismatch for %s; re-encoding the whole text", path)
        return bom + _encode(text.replace("\n", like.newline), codec, path)
    new_lines = text.split("\n")
    source = _map_unchanged_lines(old_lines, new_lines)
    out: list[bytes] = [bom]
    changed: list[str] = []
    last = len(new_lines) - 1
    for i, line in enumerate(new_lines):
        j = source[i]
        if j is None:
            try:
                out.append(line.encode(codec))
            except UnicodeEncodeError:
                changed.append(line)
                continue
        else:
            out.append(old_raw[j])
        if i < last:
            out.append(old_ends[j] if j is not None and j < len(old_ends) else newline)
    if changed:
        raise UnencodableTextError(path, codec, _unencodable_chars("".join(changed), codec))
    return b"".join(out)


def _map_unchanged_lines(old: list[str], new: list[str]) -> list[int | None]:
    """新しい各行について、同じ内容の既存行の番号 (無ければ ``None``) を返す。

    先頭と末尾の一致を先に外してから ``difflib`` に掛ける (追記は中身が空になる)。
    """
    mapping: list[int | None] = [None] * len(new)
    lo = 0
    while lo < len(old) and lo < len(new) and old[lo] == new[lo]:
        mapping[lo] = lo
        lo += 1
    hi_old, hi_new = len(old), len(new)
    while hi_old > lo and hi_new > lo and old[hi_old - 1] == new[hi_new - 1]:
        hi_old -= 1
        hi_new -= 1
        mapping[hi_new] = hi_old
    matcher = difflib.SequenceMatcher(None, old[lo:hi_old], new[lo:hi_new])
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            mapping[lo + b + k] = lo + a + k
    return mapping


# ── 書き込み ──────────────────────────────────────────────────────────────


def _resolve_target(path: Path) -> Path:
    """シンボリックリンクならリンク先 (リンクを普通のファイルで置き換えない)。"""
    if path.is_symlink():
        return Path(os.path.realpath(path))
    return path


def _write_bytes(target: Path, data: bytes) -> tuple[int, Path | None]:
    from backend.io.readonly import guard_write

    guard_write(target)
    try:
        st = target.stat()
    except FileNotFoundError:
        st = None
    if st is not None:
        if not stat.S_ISREG(st.st_mode):
            raise UserFileWriteError("not_a_file", target)
        # 置き換えるとリンクが切れ、他の名前からは古い中身が見え続ける。
        if st.st_nlink > 1:
            raise UserFileWriteError("hardlink", target, f"{st.st_nlink} links")
        # 置き換えは読み取り専用を黙って外す (Windows では 5.6 秒の再試行の末に失敗する)。
        if not st.st_mode & stat.S_IWRITE:
            raise UserFileWriteError("readonly", target)
        if st.st_size == len(data) and target.read_bytes() == data:
            return len(data), None
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = _back_up(target) if st is not None else None
    tmp = _write_tmp(target, data, st)
    try:
        if st is not None:
            _copy_zone_identifier(target, tmp)
        _replace_with_retry(tmp, target)
    except BaseException:
        _discard(tmp)
        raise
    if target.read_bytes() != data:
        raise UserFileWriteError("verify_failed", target, "on-disk bytes differ from the written bytes")
    if backup is not None:
        sink = _backup_sink.get()
        if sink is not None:
            sink.append((target, backup))
    return len(data), backup


def _write_tmp(target: Path, data: bytes, st: os.stat_result | None) -> Path:
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    # mkstemp は 0600 で作るので、新規ファイルが umask どおりのパーミッションにならない。
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if st is not None and os.name != "nt":
            os.chmod(tmp, stat.S_IMODE(st.st_mode))
    except BaseException:
        _discard(tmp)
        raise
    return tmp


def _discard(tmp: Path) -> None:
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Failed to remove tmp file %s: %s", tmp, e)


def _copy_zone_identifier(src: Path, dst: Path) -> None:
    """インターネットから来た印 (Windows の ADS ``Zone.Identifier``) を写す (best-effort)。"""
    if os.name != "nt":
        return
    try:
        with open(f"{src}{_ZONE_IDENTIFIER}", "rb") as f:
            zone = f.read()
    except OSError:
        return
    try:
        with open(f"{dst}{_ZONE_IDENTIFIER}", "wb") as f:
            f.write(zone)
    except OSError as e:
        logger.warning("Could not carry Zone.Identifier over to %s: %s", dst, e)


# ── 退避 (f_11 §5.6) ──────────────────────────────────────────────────────


def _backup_dir() -> Path:
    from backend.config import resolve_data_path

    return resolve_data_path("backup_overwrite_dir")


def _back_up(target: Path) -> Path:
    """``target`` を ``<bk/overwrite>/<UTC 日付>/<stamp>_<hex6>_<名前>`` へ複製する。"""
    from backend.utils import utc_compact_stamp

    root = _backup_dir()
    stamp = utc_compact_stamp()
    day = root / f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"
    name = target.name
    if len(name) > _MAX_BACKUP_NAME:  # MAX_PATH (c_03 §10.1)。拡張子は残す
        suffix = target.suffix[:20]
        name = target.name[: _MAX_BACKUP_NAME - len(suffix)] + suffix
    dest = day / f"{stamp}_{secrets.token_hex(3)}_{name}"
    try:
        day.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, dest)
    except OSError as e:
        raise UserFileWriteError("backup_failed", target, str(e)) from e
    logger.info("Backed up %s to %s before overwriting", target, dest)
    try:
        _prune_backups(root, keep=dest)
    except OSError as e:  # 刈り込みの失敗で書込みは止めない
        logger.warning("Pruning overwrite backups under %s failed: %s", root, e)
    return dest


def _prune_backups(root: Path, *, keep: Path) -> None:
    """30 日を過ぎたものと、合計が上限を超えた分を古い順に消す (``keep`` は残す)。"""
    entries: list[tuple[float, int, Path]] = []
    for p in root.rglob("*"):
        if p.name == ".gitkeep" or p == keep or not p.is_file():
            continue
        st = p.stat()
        entries.append((st.st_mtime, st.st_size, p))
    entries.sort()
    total = keep.stat().st_size + sum(size for _, size, _ in entries)
    cutoff = time.time() - BACKUP_KEEP_DAYS * 86400
    for mtime, size, p in entries:
        if mtime >= cutoff and total <= BACKUP_MAX_BYTES:
            break
        try:
            p.unlink()
        except OSError as e:
            logger.warning("Could not remove old backup %s: %s", p, e)
            continue
        total -= size
    for d in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
        with contextlib.suppress(OSError):
            d.rmdir()  # 空のときだけ消える
