"""世代印 ``<data_root>/g1/store/.generation`` (c_05 §0.4.6)。

G1 の封筒 (``store.generation`` v1) に包む。ペイロードは恒久に凍結する形::

    {"formats": {"<format_id>": {"version": 1, "writers": ["free"]}},
     "pending": null,
     "written_by": {"app_version": "1.0.0", "edition": "free"},
     "last_edition": "free"}

- 封筒でない / 新しい版 / 壊れた世代印は読めない (:class:`SealError`、起動は readonly)。
- 起動ゲートの分類は世代印だけで決める (ファイルごとの版・破損は読み手が開いた
  ときに判定する)。
- 旧版は未知キーを無視する。書き戻すときは未知キー・自分の知らない形式の項目を
  原形のまま保つ (Free は ``writers`` が pro だけの項目を触らない)。
- ``pending`` = 移行の途中。自分が読む形式が含まれ、その移行器を持たない版は
  起動を拒否する (G1 は移行器を持たないので、読む形式が含まれていれば拒否)。
- 書き込みは fsync (壊れると起動の判定ができない)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.io.format_registry import FormatRegistry, FormatSpec, register_format
from backend.io.versioned import read_versioned, write_versioned

SEAL_FILENAME = ".generation"

GENERATION_SEAL_FORMAT = register_format(FormatSpec(
    format_id="store.generation",
    version=1,
    klass="system",
    writers=frozenset({"free", "pro"}),
    path_key=f"store/{SEAL_FILENAME}",
    retention="one per data root",
))
_KNOWN_KEYS = ("formats", "pending", "written_by", "last_edition")

FormatState = Literal["current", "newer", "older", "absent"]


class SealError(ValueError):
    """世代印が読めない (形が壊れている)。"""


@dataclass(slots=True)
class Seal:
    """世代印の中身。``extra`` は知らないトップレベルキー (原形で書き戻す)。"""

    formats: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending: dict[str, Any] | None = None
    written_by: dict[str, Any] = field(default_factory=dict)
    last_edition: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def version_of(self, format_id: str) -> int | None:
        entry = self.formats.get(format_id)
        version = entry.get("version") if isinstance(entry, dict) else None
        return version if isinstance(version, int) and not isinstance(version, bool) else None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "formats": {k: self.formats[k] for k in sorted(self.formats)},
            "pending": self.pending,
            "written_by": self.written_by,
            "last_edition": self.last_edition,
        }
        for key, value in self.extra.items():
            out.setdefault(key, value)
        return out


def read_seal(path: Path) -> Seal | None:
    """世代印を読む。無ければ ``None``、読めなければ (封筒でない・新しい版・壊れた形) :class:`SealError`。"""
    result = read_versioned(
        path,
        format_id=GENERATION_SEAL_FORMAT.format_id,
        format_version=GENERATION_SEAL_FORMAT.version,
    )
    if result.status == "absent":
        return None
    if not result.ok:
        raise SealError(f"{path}: {result.status} ({result.detail})")
    raw = result.payload
    if not isinstance(raw, dict) or not isinstance(raw.get("formats"), dict):
        raise SealError(f"{path}: missing 'formats'")
    pending = raw.get("pending")
    if pending is not None and not isinstance(pending, dict):
        raise SealError(f"{path}: 'pending' must be an object or null")
    written_by = raw.get("written_by")
    last_edition = raw.get("last_edition")
    return Seal(
        formats={str(k): v for k, v in raw["formats"].items()},
        pending=pending,
        written_by=written_by if isinstance(written_by, dict) else {},
        last_edition=last_edition if isinstance(last_edition, str) else None,
        extra={k: v for k, v in raw.items() if k not in _KNOWN_KEYS},
    )


def write_seal(path: Path, seal: Seal) -> None:
    """世代印を封筒に包んで fsync 付きで原子的に書く。"""
    write_versioned(
        path,
        format_id=GENERATION_SEAL_FORMAT.format_id,
        format_version=GENERATION_SEAL_FORMAT.version,
        payload=seal.to_json(),
        component="generation_seal",
        fsync=True,
        indent=2,
    )


def classify(seal: Seal | None, registry: FormatRegistry, edition: str) -> dict[str, FormatState]:
    """実行中エディションが読む形式ごとに、世代印とコードの版を比べる (c_05 §0.4.3)。"""
    states: dict[str, FormatState] = {}
    for spec in registry.read_by(edition):  # type: ignore[arg-type]
        on_disk = seal.version_of(spec.format_id) if seal is not None else None
        if on_disk is None:
            states[spec.format_id] = "absent"
        elif on_disk > spec.version:
            states[spec.format_id] = "newer"
        elif on_disk < spec.version:
            states[spec.format_id] = "older"
        else:
            states[spec.format_id] = "current"
    return states


def pending_blocks(seal: Seal | None, registry: FormatRegistry, edition: str) -> list[str]:
    """移行途中の形式のうち、自分が読むもの (1 つでもあれば起動拒否)。"""
    if seal is None or not seal.pending:
        return []
    listed = seal.pending.get("formats") or []
    ours = {spec.format_id for spec in registry.read_by(edition)}  # type: ignore[arg-type]
    return sorted(str(f) for f in listed if str(f) in ours)


def updated_seal(
    seal: Seal | None, registry: FormatRegistry, *, edition: str, app_version: str,
) -> Seal:
    """書き戻す世代印を作る (readonly でないときだけ呼ぶ)。

    自分が知っている形式の版を現行へ揃え、知らない形式・他エディション専用の
    項目は原形のまま残す。
    """
    base = seal or Seal()
    formats = dict(base.formats)
    for spec in registry.all():
        if edition == "free" and "free" not in spec.writers:
            continue  # Free は pro 専用の項目を触らない (c_05 §0.4.6)
        formats[spec.format_id] = {"version": spec.version, "writers": sorted(spec.writers)}
    return Seal(
        formats=formats,
        pending=base.pending,
        written_by={"app_version": app_version, "edition": edition},
        last_edition=edition,
        extra=dict(base.extra),
    )


#: ディレクトリ構成だけを残す印 (リポジトリの ``userdata/**/.gitkeep``、c_03 §10.1)。
GITKEEP_FILENAME = ".gitkeep"


def store_has_data(store_dir: Path) -> bool:
    """``store/`` に世代印・ロック・``.gitkeep`` 以外のファイルがあるか (世代印の欠落を readonly にする条件)。

    空のディレクトリ構成 (setup / ``ensure_local_dirs`` / ``.gitkeep`` の骨組み) はデータではない。
    """
    import os

    from backend.io.writer_lock import LOCK_FILENAME

    if not store_dir.is_dir():
        return False
    ignored = {LOCK_FILENAME, SEAL_FILENAME, GITKEEP_FILENAME}
    for _current, _dirs, files in os.walk(store_dir):
        if any(name not in ignored for name in files):
            return True
    return False


@dataclass
class SealCheck:
    """``store/`` の世代印を現行のコードと突き合わせた結果 (起動ゲート・export / import・doctor 共通)。"""

    seal: Seal | None = None
    #: 読めなかった理由 (読めたら ``None``)。
    unreadable: str | None = None
    #: 自分が読む形式のうち移行途中のもの (起動拒否)。
    pending: list[str] = field(default_factory=list)
    #: 世代印が無いのに ``store/`` にデータがある。
    missing_with_data: bool = False
    states: dict[str, FormatState] = field(default_factory=dict)

    @property
    def newer(self) -> list[str]:
        return sorted(k for k, v in self.states.items() if v == "newer")

    @property
    def older(self) -> list[str]:
        return sorted(k for k, v in self.states.items() if v == "older")

    def readonly_reasons(self, store_dir: Path) -> list[str]:
        """readonly で起動する理由 (移行途中は起動拒否なので含めない)。空なら書いてよい。"""
        if self.unreadable is not None:
            return [f"generation seal unreadable: {self.unreadable}"]
        if self.missing_with_data:
            return [f"{SEAL_FILENAME} is missing but {store_dir} holds data"]
        reasons: list[str] = []
        if self.newer:
            reasons.append(f"formats written by a newer version: {', '.join(self.newer)}")
        if self.older:
            reasons.append(
                f"formats need a migration this version does not have: {', '.join(self.older)}",
            )
        return reasons


def check_seal(store_dir: Path, registry: FormatRegistry, edition: str) -> SealCheck:
    """世代印を読み、現行の台帳と突き合わせる (書き込みはしない)。"""
    try:
        seal = read_seal(store_dir / SEAL_FILENAME)
    except SealError as e:
        return SealCheck(unreadable=str(e))
    check = SealCheck(seal=seal, pending=pending_blocks(seal, registry, edition))
    if seal is None and store_has_data(store_dir):
        check.missing_with_data = True
        return check
    check.states = classify(seal, registry, edition)
    return check


__all__ = [
    "SEAL_FILENAME",
    "SealCheck",
    "check_seal",
    "store_has_data",
    "FormatState",
    "Seal",
    "SealError",
    "classify",
    "pending_blocks",
    "read_seal",
    "updated_seal",
    "write_seal",
]
