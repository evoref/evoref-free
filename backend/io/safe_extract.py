"""アーカイブ展開の共通関数 (docs/c_06 §1.5 / c_05 §0.2)

export の取り込み・テーマ・``.evocart``・semantic の tar は、どれも利用者が外から
持ち込むアーカイブを展開する。エントリ名をそのまま展開先に連結すると、Windows では
``Path(base) / "C:/Users/..."`` が ``C:/Users/...`` に置き換わり、任意の場所へ書ける
(2026-09-23 監査で export の取り込みに実在した)。全経路をここに寄せ、同じ規則で弾く。

規則:
- エントリ名は ``/`` と ``\\`` の両方を区切りとして分解し、NFC に正規化する。
- 拒否: 絶対パス・ドライブ・UNC・``..``・任意位置の ``:`` (ADS / ドライブ)・制御文字・
  Windows で使えない文字・末尾のドット / 空白・予約デバイス名 (拡張子付きも)・8.3 短縮名。
- NFC + casefold で同じになる名前の重複、シンボリックリンク / ハードリンクを拒否。
- 展開先は ``resolve()`` した上で ``is_relative_to`` でも確かめる (二重確認)。
- サイズの上限は**実際に展開したバイト数**で数える (自己申告の ``file_size`` を信用しない)。
"""

from __future__ import annotations

import re
import stat
import unicodedata
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO

_CHUNK = 1 << 20
_FORBIDDEN_CHARS = frozenset('<>:"|?*')
_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{c}" for c in "0123456789¹²³"}
    | {f"lpt{c}" for c in "0123456789¹²³"},
)
_SHORT_NAME_RE = re.compile(r"~\d")


class UnsafeArchiveError(ValueError):
    """展開してはいけないエントリ / 上限超過。"""


@dataclass(frozen=True)
class ExtractLimits:
    """展開の上限。``None`` は無制限。"""

    max_entries: int | None = 100_000
    max_member_bytes: int | None = None
    max_total_bytes: int | None = None
    #: 1 エントリの展開後 / 圧縮後の比 (zip bomb 対策)。
    max_ratio: float | None = 1000.0


def normalize_member_name(name: str) -> PurePosixPath | None:
    """エントリ名を展開先相対の安全なパスへ正規化する。

    ディレクトリエントリ (末尾が区切り) と空の名前は ``None``。危険な名前は
    :class:`UnsafeArchiveError`。
    """
    if not name:
        return None
    unified = unicodedata.normalize("NFC", name.replace("\\", "/"))
    if unified.endswith("/"):
        return None
    if unified.startswith("/") or unified.startswith("//"):
        raise UnsafeArchiveError(f"absolute path in archive: {name!r}")
    parts: list[str] = []
    for part in unified.split("/"):
        if part in ("", "."):
            continue
        _check_component(part, name)
        parts.append(part)
    if not parts:
        return None
    return PurePosixPath(*parts)


def _check_component(part: str, original: str) -> None:
    if part == "..":
        raise UnsafeArchiveError(f"parent reference in archive: {original!r}")
    if any(ord(ch) < 0x20 or ch in _FORBIDDEN_CHARS for ch in part):
        # ``:`` はドライブ指定と NTFS の代替データストリームの両方を塞ぐ
        raise UnsafeArchiveError(f"forbidden character in archive entry: {original!r}")
    if part.endswith((".", " ")):
        raise UnsafeArchiveError(f"trailing dot or space in archive entry: {original!r}")
    stem = part.split(".", 1)[0].strip().casefold()
    if stem in _RESERVED_STEMS:
        raise UnsafeArchiveError(f"reserved device name in archive entry: {original!r}")
    if _SHORT_NAME_RE.search(part):
        raise UnsafeArchiveError(f"8.3 short name in archive entry: {original!r}")


def resolve_under(base: Path, member: PurePosixPath) -> Path:
    """``base`` 配下に解決する。外を指したら :class:`UnsafeArchiveError`。"""
    base_resolved = base.resolve()
    target = (base / Path(*member.parts)).resolve()
    if target == base_resolved or not target.is_relative_to(base_resolved):
        raise UnsafeArchiveError(f"archive entry escapes destination: {member}")
    return target


def check_zip_members(zf: zipfile.ZipFile, limits: ExtractLimits) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """zip の全エントリを検査し、(ZipInfo, 正規化名) を返す (ディレクトリは除く)。"""
    infos = zf.infolist()
    if limits.max_entries is not None and len(infos) > limits.max_entries:
        raise UnsafeArchiveError(f"archive has too many entries: {len(infos)}")
    seen: set[str] = set()
    out: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for info in infos:
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise UnsafeArchiveError(f"symlink in archive: {info.filename!r}")
        member = normalize_member_name(info.filename)
        if member is None:
            continue
        key = unicodedata.normalize("NFC", str(member)).casefold()
        if key in seen:
            raise UnsafeArchiveError(f"duplicate entry in archive: {info.filename!r}")
        seen.add(key)
        if limits.max_member_bytes is not None and info.file_size > limits.max_member_bytes:
            raise UnsafeArchiveError(f"archive entry too large: {info.filename!r}")
        out.append((info, member))
    return out


class ExtractBudget:
    """展開したバイト数の累計を数える (上限は実際に書いたバイトで判定する)。"""

    def __init__(self, limits: ExtractLimits) -> None:
        self.limits = limits
        self.total = 0

    def copy(self, src: IO[bytes], target: Path, *, compressed_size: int | None = None, overwrite: bool = False) -> int:
        """``src`` を ``target`` へ書く。上限を超えたら書きかけを消して送出する。"""
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        limits = self.limits
        ratio_cap = (
            None
            if limits.max_ratio is None or not compressed_size
            else int(compressed_size * limits.max_ratio) + _CHUNK
        )
        try:
            with open(target, "wb" if overwrite else "xb") as dst:
                while True:
                    chunk = src.read(_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    self.total += len(chunk)
                    if limits.max_member_bytes is not None and written > limits.max_member_bytes:
                        raise UnsafeArchiveError(f"archive entry exceeds size limit: {target.name}")
                    if limits.max_total_bytes is not None and self.total > limits.max_total_bytes:
                        raise UnsafeArchiveError("archive exceeds total size limit")
                    if ratio_cap is not None and written > ratio_cap:
                        raise UnsafeArchiveError(f"archive entry compression ratio too high: {target.name}")
                    dst.write(chunk)
        except FileExistsError:
            raise
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return written


def extract_zip(
    zf: zipfile.ZipFile,
    destination: Path,
    *,
    limits: ExtractLimits,
    members: Iterable[tuple[zipfile.ZipInfo, PurePosixPath]] | None = None,
) -> list[Path]:
    """検査済みのエントリを ``destination`` 配下へ展開する (既存ファイルは上書きしない)。"""
    selected = list(members) if members is not None else check_zip_members(zf, limits)
    budget = ExtractBudget(limits)
    written: list[Path] = []
    for info, member in selected:
        target = resolve_under(destination, member)
        with zf.open(info) as src:
            budget.copy(src, target, compressed_size=info.compress_size)
        written.append(target)
    return written
