"""利用者のテキストファイルを編集の素材として読む (docs/f_11 §5.4)。

既存ファイルへの追記・書き直しは、読んだ内容をそのまま土台にする。ここで置換
文字に落として「読めたことにする」と、化けた文字列でファイル全体が置き換わる。
UTF-16 の本文は cp932 として「読めて」しまい、NUL を含むバイナリも UTF-8 として
通る。そうしたものは :class:`Unreadable` で返し、呼び出し側が編集を断る。

``read_file`` ツールの表示 (:func:`decode_text_for_display`) も同じ候補順を使い、
厳格に読めないときだけ従来どおり寛容に読む — 表示は壊さず、編集には使わせない。
"""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

_UTF8_BOM = b"\xef\xbb\xbf"
#: UTF-16 LE / BE の BOM (UTF-32 LE の BOM も先頭 2 バイトは FF FE)。
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


@dataclass(frozen=True, slots=True)
class TextFile:
    """厳格に読めたテキストファイル。

    Attributes:
        text: 改行を ``\\n`` へ正規化した本文 (単独の ``\\r`` も ``\\n`` になる)。
        encoding: ``"utf-8"`` / ``"utf-8-sig"`` / ``"cp932"``。
        newline: 元の改行 (``"\\n"`` / ``"\\r\\n"``、混在なら多い方)。
        trailing_newline: 本文が改行で終わっていたか。
        raw: 読んだバイト列そのもの。
    """

    text: str
    encoding: str
    newline: str
    trailing_newline: bool
    raw: bytes


@dataclass(frozen=True, slots=True)
class Unreadable:
    """編集の素材として読めなかった。

    ``reason`` は ``too_large`` / ``utf16_bom`` / ``nul_bytes`` / ``undecodable`` /
    ``not_a_file`` / ``os_error`` (リッチ文書の抽出失敗は
    呼び出し側が ``extraction_failed`` を使う)。
    """

    reason: str
    detail: str = ""


def _strict_decode(raw: bytes) -> tuple[str, str] | Unreadable:
    """``(本文, エンコーディング)`` を厳格に求める。改行は触らない。"""
    if raw.startswith(_UTF16_BOMS):
        return Unreadable("utf16_bom")
    if b"\x00" in raw:
        return Unreadable("nul_bytes")
    if raw.startswith(_UTF8_BOM):
        try:
            return raw.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError:
            return Unreadable("undecodable")
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        text = raw.decode("cp932")
    except UnicodeDecodeError:
        return Unreadable("undecodable")
    # cp932 には同じ文字に 2 つの符号を持つもの (NEC 選定 IBM 拡張など) があり、
    # 符号化し直すと元のバイト列に戻らない。読むこと自体は正しいので断らない —
    # 触っていない行のバイト列は書き手 (backend/io/user_file.py) が元のまま保つ。
    return text, "cp932"


def decode_text_for_edit(raw: bytes) -> TextFile | Unreadable:
    """バイト列を編集の素材として厳格にデコードする (純粋関数)。"""
    decoded = _strict_decode(raw)
    if isinstance(decoded, Unreadable):
        return decoded
    text, encoding = decoded
    crlf = text.count("\r\n")
    lf_only = text.count("\n") - crlf
    text_lf = text.replace("\r\n", "\n").replace("\r", "\n")
    return TextFile(
        text=text_lf,
        encoding=encoding,
        newline="\r\n" if crlf > lf_only else "\n",
        trailing_newline=text_lf.endswith("\n"),
        raw=raw,
    )


def read_text_for_edit(path: Path | str, *, max_bytes: int) -> TextFile | Unreadable | None:
    """ファイルを編集の素材として読む。無ければ ``None``。

    同期 I/O なので、イベントループからは ``run_in_executor_with_context`` 経由で呼ぶ。
    """
    p = Path(path)
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    except OSError as e:
        return Unreadable("os_error", str(e))
    if not stat.S_ISREG(st.st_mode):
        return Unreadable("not_a_file")
    if st.st_size > max_bytes:
        return Unreadable("too_large", f"{st.st_size} bytes > {max_bytes}")
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as e:
        return Unreadable("os_error", str(e))
    if len(raw) > max_bytes:
        return Unreadable("too_large", f"{len(raw)} bytes > {max_bytes}")
    return decode_text_for_edit(raw)


def decode_text_prefix_for_display(raw: bytes, *, truncated: bool) -> str:
    """先頭だけ読んだバイト列を表示用に読む (改行は ``\\n`` へ正規化)。

    ``truncated`` のときは末尾で切れた文字 (最大 3 バイト) を落として試す。
    UTF-8 を全ての切り方で先に試すのは、途中で切れた UTF-8 がたまたま cp932 として
    読めて化けるのを避けるため。どれも読めなければ ``decode_text_for_display``。
    """
    cuts = range(4) if truncated else (0,)
    encodings = ("utf-8-sig",) if raw.startswith(_UTF8_BOM) else ("utf-8", "cp932")
    for enc in encodings:
        for cut in cuts:
            try:
                text = raw[: len(raw) - cut].decode(enc)
            except UnicodeDecodeError:
                continue
            return text.replace("\r\n", "\n").replace("\r", "\n")
    return decode_text_for_display(raw)[0]


def decode_text_for_display(raw: bytes) -> tuple[str, str]:
    """表示用に ``(本文, 使ったエンコーディング)`` を返す (改行は ``\\n`` へ正規化)。

    厳格に読めればその結果。読めなければ BOM 付きは ``utf-8-sig``、無ければ
    ``utf-8`` → ``cp932`` を往復検査なしで試し、全て失敗したら ``utf-8`` の置換
    デコード (``"utf-8 (replace)"``) に落ちる。編集の素材には使わない。
    """
    strict = decode_text_for_edit(raw)
    if isinstance(strict, TextFile):
        return strict.text, strict.encoding
    candidates = ("utf-8-sig",) if raw.startswith(_UTF8_BOM) else ("utf-8", "cp932")
    text: str | None = None
    used = "utf-8"
    for enc in candidates:
        try:
            text = raw.decode(enc)
            used = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
        used = "utf-8 (replace)"
    return text.replace("\r\n", "\n").replace("\r", "\n"), used
