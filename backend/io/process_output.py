"""子プロセス出力のデコード。

``subprocess`` を ``text=True`` (``encoding`` なし) で呼ぶと、出力はサーバ側の
ロケール既定で読まれる。日本語 Windows の既定は cp932 だが、``scripts/evoref*.bat``
は ``PYTHONUTF8=1`` で起動するので UTF-8 になり、子 (cmd / git / python 等) が
別のコードページで書くと文字化けするか、読み取りスレッドが ``UnicodeDecodeError``
で落ちて stdout / stderr が ``None`` になる (失敗理由ごと消える)。子の出力は
bytes で受け、ここで読む。
"""

from __future__ import annotations

import locale
import sys


def decode_process_output(raw: bytes | str | None) -> str:
    """子プロセスの出力を例外なしで str にする。

    厳格 UTF-8 → (Windows) OEM / mbcs → ロケール推奨 → UTF-8 (置換) の順に試す。
    cp932 を先に試すと UTF-8 の日本語が「成功裏に」化けるので UTF-8 が先。
    差し替えられた runner が既に str を返した場合はそのまま返す。
    """
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    candidates: list[str] = []
    if sys.platform == "win32":
        candidates.extend(["oem", "mbcs"])
    pref = locale.getpreferredencoding(False)
    if pref and pref.lower() not in {c.lower() for c in candidates}:
        candidates.append(pref)
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")
