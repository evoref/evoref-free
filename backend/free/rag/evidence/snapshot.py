"""snapshot の畳み込み・書き出し・読み出し (c_16 §5.3)

snapshot 生成 = 前 snapshot + 事象を畳んで ``records.jsonl`` を **新しい版
ディレクトリ** に書き、同時に ``offsets.npy`` / ``columns.npz`` を作る。
稼働中の版は 1 バイトも書き換えない (in-place 更新はクラッシュ窓と memmap
中の上書きを生む — 2026-09-05 監査)。

読み出しは id とカラムだけ常駐させ、**本文は top-k だけ** ``offsets.npy``
経由で lazy に読む。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.rag.evidence.columns import (
    EvidenceColumns,
    build_columns,
    load_columns,
    save_columns,
)
from backend.free.rag.evidence.types import (
    PATCHABLE_FIELDS,
    Evidence,
    EvidenceRecordError,
    from_record,
)
from backend.io import AtomicWriter
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("rag.evidence.snapshot")

RECORDS_FILE = "records.jsonl"
OFFSETS_FILE = "offsets.npy"
SNAPSHOT_DIR = "snapshot"

#: 版ディレクトリ名 (``v0007``)。
_VERSION_FORMAT = "v{:04d}"


def version_name(seq: int) -> str:
    """版番号 → ディレクトリ名 (``v0007``)。"""
    return _VERSION_FORMAT.format(int(seq))


def list_versions(store_dir: Path | str) -> list[str]:
    """``snapshot/`` 配下の版ディレクトリ名を昇順で返す。"""
    root = Path(store_dir) / SNAPSHOT_DIR
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / RECORDS_FILE).exists()
    )


def snapshot_dir(store_dir: Path | str, version: str) -> Path:
    return Path(store_dir) / SNAPSHOT_DIR / version


class SnapshotWriter:
    """事象の畳み込みと版の書き出し。"""

    @staticmethod
    def fold(
        prev_records: Iterable[Evidence],
        events: Iterable[dict[str, Any]],
    ) -> list[Evidence]:
        """前 snapshot + 事象 → 現在状態 (c_16 §5.2)。

        - ``put``: 完全なレコードで置き換え / 追加
        - ``patch``: :data:`PATCHABLE_FIELDS` のみ差し替え
        - ``retract``: ``veracity=retracted`` (物理削除はしない)
        - ``touch``: ``last_used_at`` のみ更新 (複数 id を 1 事象で)

        未知 id への patch / retract / touch は件数を WARNING に出して飛ばす
        (全体を落とさない)。順序は put された順を保つ。
        """
        folded: dict[str, Evidence] = {r.id: r for r in prev_records}
        unknown = 0
        broken = 0

        for event in events:
            op = event.get("op")
            payload = event.get("payload") or {}
            record_id = str(event.get("id") or "")
            at = str(event.get("at") or utc_now())

            if op == "put":
                raw = payload.get("record")
                if not isinstance(raw, dict):
                    broken += 1
                    continue
                try:
                    record = from_record(raw)
                except EvidenceRecordError as e:
                    logger.warning("skipping malformed put event %s: %s", record_id, e)
                    broken += 1
                    continue
                folded[record.id] = record
                continue

            if op == "touch":
                ids = payload.get("ids") or ([record_id] if record_id else [])
                stamp = str(payload.get("last_used_at") or at)
                for rid in ids:
                    current = folded.get(str(rid))
                    if current is None:
                        unknown += 1
                        continue
                    folded[current.id] = replace(current, last_used_at=stamp)
                continue

            current = folded.get(record_id)
            if current is None:
                unknown += 1
                continue

            if op == "patch":
                fields = payload.get("fields")
                if not isinstance(fields, dict):
                    broken += 1
                    continue
                folded[record_id] = apply_patch(current, fields, at=at)
            elif op == "retract":
                reason = str(payload.get("reason") or "")
                extra = dict(current._extra)
                extra["retract_reason"] = reason
                extra["retracted_at"] = at
                folded[record_id] = replace(
                    current, veracity="retracted", updated_at=at, _extra=extra,
                )
            else:
                broken += 1

        if unknown:
            logger.warning("fold: %d event(s) referenced unknown record ids", unknown)
        if broken:
            logger.warning("fold: %d malformed event(s) skipped", broken)
        return list(folded.values())

    @staticmethod
    def write_snapshot(
        store_dir: Path | str,
        version: str,
        records: Sequence[Evidence],
    ) -> Path:
        """``records.jsonl`` / ``offsets.npy`` / ``columns.npz`` を書き出す。

        offsets は各行の **先頭 byte offset** (int64)。本文を lazy に読むため
        の索引なので、行の書き出しと同じ 1 パスで作る (別々に数えるとずれる)。
        """
        directory = snapshot_dir(store_dir, version)
        directory.mkdir(parents=True, exist_ok=True)

        offsets: list[int] = []
        cursor = 0
        with AtomicWriter(directory / RECORDS_FILE, mode="wb") as f:
            for record in records:
                line = (record.to_json_line() + "\n").encode("utf-8")
                offsets.append(cursor)
                f.write(line)
                cursor += len(line)

        with AtomicWriter(directory / OFFSETS_FILE, mode="wb") as f:
            np.save(f, np.array(offsets, dtype=np.int64))

        save_columns(directory, build_columns(records))
        logger.info(
            "Wrote snapshot %s: %d record(s), %d bytes",
            directory, len(records), cursor,
        )
        return directory


class SnapshotReader:
    """1 版分の snapshot への読み取り専用アクセス。

    常駐するのは ``ids`` / カラム / offsets だけ。本文は :meth:`text_at` /
    :meth:`record_at` で要求された行だけ読む。
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.records_path = self.directory / RECORDS_FILE
        self.columns: EvidenceColumns = load_columns(self.directory)
        offsets_path = self.directory / OFFSETS_FILE
        if offsets_path.exists():
            self.offsets: np.ndarray = np.load(str(offsets_path), allow_pickle=False)
        else:
            self.offsets = np.zeros(0, dtype=np.int64)
        self._id_index: dict[str, int] = self.columns.id_index()
        if len(self.offsets) != len(self.columns):
            logger.error(
                "Snapshot misalignment at %s: %d offsets vs %d columns. "
                "Rebuild the snapshot.",
                self.directory, len(self.offsets), len(self.columns),
            )

    # ── 位置引き ──

    def __len__(self) -> int:
        return len(self.columns)

    @property
    def ids(self) -> np.ndarray:
        return self.columns.ids

    def row_of(self, record_id: str) -> int | None:
        return self._id_index.get(record_id)

    def id_at(self, row: int) -> str:
        return str(self.columns.ids[row])

    # ── 本文の lazy 読み ──

    def raw_at(self, row: int) -> dict[str, Any] | None:
        """指定行の生 dict を読む (offsets で seek)。"""
        if row < 0 or row >= len(self.offsets) or not self.records_path.exists():
            return None
        with self.records_path.open("rb") as f:
            f.seek(int(self.offsets[row]))
            line = f.readline()
        if not line:
            return None
        try:
            raw = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning("unreadable snapshot line %s:%d: %s", self.records_path, row, e)
            return None
        return raw if isinstance(raw, dict) else None

    def text_at(self, row: int) -> str:
        """指定行の本文だけを返す (見つからなければ空文字)。"""
        raw = self.raw_at(row)
        if raw is None:
            return ""
        return str(raw.get("text") or "")

    def record_at(self, row: int) -> Evidence | None:
        """指定行を :class:`Evidence` として読む。"""
        raw = self.raw_at(row)
        if raw is None:
            return None
        try:
            return from_record(raw)
        except EvidenceRecordError as e:
            logger.warning("skipping unreadable record at row %d: %s", row, e)
            return None

    def get(self, record_id: str) -> Evidence | None:
        row = self.row_of(record_id)
        return None if row is None else self.record_at(row)

    def iter_records(self) -> Iterable[Evidence]:
        """全レコードを順に読む (snapshot 再生成・移行用の全走査)。"""
        if not self.records_path.exists():
            return
        skipped = 0
        with self.records_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    yield from_record(json.loads(text))
                except (json.JSONDecodeError, EvidenceRecordError):
                    skipped += 1
        if skipped:
            logger.warning(
                "%s: skipped %d unreadable record(s)", self.records_path, skipped,
            )


def read_snapshot(store_dir: Path | str, version: str) -> SnapshotReader:
    """版名を指定して :class:`SnapshotReader` を開く。"""
    return SnapshotReader(snapshot_dir(store_dir, version))


def apply_patch(
    record: Evidence, fields: dict[str, Any], *, at: str | None = None,
) -> Evidence:
    """``patch`` 事象のフィールドを 1 レコードへ適用する。

    :data:`PATCHABLE_FIELDS` 以外は無視する (put で置き換えるべきコア部分を
    事象で少しずつ壊さない)。``tier`` は ``attrs.tier`` のショートカット。
    """
    stamp = at or utc_now()
    updates: dict[str, Any] = {}
    attrs: dict[str, Any] | None = None
    ignored: list[str] = []

    for key, value in fields.items():
        if key not in PATCHABLE_FIELDS:
            ignored.append(key)
            continue
        if key == "tier":
            attrs = dict(record.attrs) if attrs is None else attrs
            attrs["tier"] = value
            continue
        if key == "attrs":
            if isinstance(value, dict):
                attrs = {**(attrs if attrs is not None else record.attrs), **value}
            continue
        updates[key] = value

    if attrs is not None:
        updates["attrs"] = attrs
    if ignored:
        logger.warning(
            "patch on %s ignored non-patchable field(s): %s",
            record.id, ", ".join(sorted(ignored)),
        )
    updates["updated_at"] = stamp
    return replace(record, **updates)


def prune_snapshots(
    store_dir: Path | str, keep: int, *, protect: Iterable[str] = (),
) -> list[str]:
    """古い版ディレクトリを削除する (c_16 §5.4: 直近 3 版)。

    ``protect`` の版は件数に関わらず残す (active 版を消して起動不能にしない)。

    Returns:
        削除した版名。
    """
    if keep < 1:
        return []
    versions = list_versions(store_dir)
    protected = {v for v in protect if v}
    removable = [v for v in versions if v not in protected]
    surplus = len(versions) - keep
    if surplus <= 0:
        return []
    removed: list[str] = []
    for version in removable[:surplus]:
        directory = snapshot_dir(store_dir, version)
        try:
            for path in sorted(directory.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                else:
                    path.rmdir()
            directory.rmdir()
        except OSError as e:
            logger.warning("failed to prune snapshot %s: %s", version, e)
            continue
        removed.append(version)
    if removed:
        logger.info("Pruned %d old snapshot(s): %s", len(removed), ", ".join(removed))
    return removed


__all__ = [
    "OFFSETS_FILE",
    "RECORDS_FILE",
    "SNAPSHOT_DIR",
    "SnapshotReader",
    "SnapshotWriter",
    "apply_patch",
    "list_versions",
    "prune_snapshots",
    "read_snapshot",
    "snapshot_dir",
    "version_name",
]
