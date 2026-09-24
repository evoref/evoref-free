"""snapshot の畳み込み・書き出し・読み出し (c_16 §5.3)

snapshot 生成 = 前 snapshot + 事象を畳んで ``records.jsonl`` を **新しい版
ディレクトリ** に書き、同時に ``offsets.npy`` / ``columns.npz`` を作る。
稼働中の版は 1 バイトも書き換えない (in-place 更新はクラッシュ窓と memmap
中の上書きを生む — 2026-09-05 監査)。

読み出しは id とカラムだけ常駐させ、**本文は top-k だけ** ``offsets.npy``
経由で lazy に読む。

版の確定の単位は ``COMPLETE`` (c_16 §5.6)。版データと埋め込み版を書き終えた後に
``{folded_through, rows, embedding_version, records_segments}`` を fsync して書き、
これが無い版は読まない (:func:`list_versions` から外れ、回復 / GC が trash へ送る)。
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.rag.evidence.columns import (
    EvidenceColumns,
    build_columns,
    encode_id,
    load_columns,
    save_columns,
)
from backend.free.rag.evidence.events import EventPosition
from backend.free.rag.evidence.types import (
    PATCHABLE_FIELDS,
    UNSETTABLE_PREFIXES,
    Evidence,
    EvidenceRecordError,
    EvidenceVersionError,
    from_record,
)
from backend.io import AtomicWriter
from backend.io.codec import CodecError, codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import ReadResult, quarantine, read_versioned, write_versioned
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("rag.evidence.snapshot")

RECORDS_FILE = "records.jsonl"
OFFSETS_FILE = "offsets.npy"
SNAPSHOT_DIR = "snapshot"
#: 版の確定の印 (c_16 §5.6)。これが無い版は無効。
COMPLETE_FILE = "COMPLETE"

@persisted()
@dataclass
class CompletePosition:
    """``COMPLETE`` の ``folded_through`` (事象ログ上の畳み込み位置、:class:`EventPosition` の永続形)。"""

    month: str = ""
    line: int = 0
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class CompleteEmbedding:
    """``COMPLETE`` の ``embedding_version`` (版と一緒に作った埋め込み版の刻印)。"""

    model_id: str = ""
    version: str = ""
    rows: int = 0
    id_hash: str = ""
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class RecordsSegment:
    """``COMPLETE`` の ``records_segments`` の 1 本 (予約。当面 ``records.jsonl`` 1 本)。"""

    file: str = RECORDS_FILE
    rows: int = 0
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass(kw_only=True)
class SnapshotComplete:
    """版を確定する ``COMPLETE`` の payload (c_16 §5.6)。sha256 は持たない。"""

    folded_through: CompletePosition = dataclasses.field(default_factory=CompletePosition)
    rows: int
    embedding_version: CompleteEmbedding | None = None
    records_segments: list[RecordsSegment] = dataclasses.field(default_factory=list)
    #: この版が知らないキー (各階層が自分の ``_extra`` を持つ)。
    _extra: dict[str, Any] | None = None

    @property
    def position(self) -> EventPosition:
        """畳み込み位置 (事象ログの型)。"""
        return EventPosition(self.folded_through.month, self.folded_through.line)


_COMPLETE_CODEC = codec_for(SnapshotComplete)

#: snapshot 版の確定印 (c_16 §5.6)。無い版は読まれないので records と一緒に運ぶ。
SNAPSHOT_COMPLETE_FORMAT = register_format(FormatSpec(
    format_id="evidence.snapshot_complete",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key=f"store/memory/<store>/{SNAPSHOT_DIR}/<ver>/{COMPLETE_FILE}",
    retention="one per snapshot version (snapshots_keep)",
    export=True,
    records=(SnapshotComplete,),
))

#: 版ディレクトリ名 (``v0007``)。
_VERSION_FORMAT = "v{:04d}"

#: :func:`version_seq` が読む版ディレクトリ名。
_VERSION_RE = re.compile(r"v(\d+)")

#: 退避の種別 (``v0007.trash-<utcstamp>``、c_05 §0.5.8 の退避名の文法)。
TRASH_KIND = "trash"
#: 退避名に含まれる印。``list_versions`` から外れ、次の prune が掃く。G0 の
#: 接頭辞形 (``.trash-v0007``) も同じ印で拾う。
TRASH_MARK = f".{TRASH_KIND}-"


def version_name(seq: int) -> str:
    """版番号 → ディレクトリ名 (``v0007``)。"""
    return _VERSION_FORMAT.format(int(seq))


def version_seq(name: str) -> int | None:
    """ディレクトリ名 → 版番号 (``v0007`` → 7)。読めなければ ``None``。"""
    match = _VERSION_RE.fullmatch(str(name).strip())
    return int(match.group(1)) if match else None


def _version_dirs(store_dir: Path | str) -> list[Path]:
    """版名 (``v0007``) を持つディレクトリを版番号の昇順で返す (trash は含まない)。"""
    root = Path(store_dir) / SNAPSHOT_DIR
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and version_seq(p.name) is not None]
    return sorted(dirs, key=lambda p: (version_seq(p.name) or 0, p.name))


def list_versions(store_dir: Path | str) -> list[str]:
    """``snapshot/`` 配下の **有効な** (``COMPLETE`` のある) 版名を昇順で返す。"""
    return [p.name for p in _version_dirs(store_dir) if (p / COMPLETE_FILE).exists()]


def list_incomplete_versions(store_dir: Path | str) -> list[str]:
    """``COMPLETE`` の無い版名 (書きかけで落ちた版) を昇順で返す。"""
    return [p.name for p in _version_dirs(store_dir) if not (p / COMPLETE_FILE).exists()]


def snapshot_dir(store_dir: Path | str, version: str) -> Path:
    return Path(store_dir) / SNAPSHOT_DIR / version


def read_complete_result(directory: Path | str) -> ReadResult:
    """版ディレクトリの ``COMPLETE`` を封筒の分類つきで読む。

    読めたときの ``payload`` は :class:`SnapshotComplete`。表で読めない payload
    (必須の ``rows`` の欠損・型の違う値) は ``corrupt``。
    """
    result = read_versioned(
        Path(directory) / COMPLETE_FILE,
        format_id=SNAPSHOT_COMPLETE_FORMAT.format_id,
        format_version=SNAPSHOT_COMPLETE_FORMAT.version,
    )
    if not result.ok:
        return result
    try:
        complete = _COMPLETE_CODEC.decode(result.payload)
    except CodecError as e:
        return ReadResult("corrupt", version=result.version, detail=f"payload: {e}")
    return ReadResult(result.status, complete, result.version)


def read_complete(directory: Path | str) -> SnapshotComplete | None:
    """版ディレクトリの ``COMPLETE`` を読む (無い / 読めなければ ``None``)。"""
    result = read_complete_result(directory)
    if result.ok:
        return result.payload
    if result.status != "absent":
        logger.warning(
            "unreadable %s (%s: %s)", Path(directory) / COMPLETE_FILE, result.status, result.detail,
        )
    return None


def new_complete(
    *, folded_through: EventPosition, rows: int, embedding_version: CompleteEmbedding | None,
) -> SnapshotComplete:
    """版を作ったときの ``COMPLETE`` (``records_segments`` は予約で当面 1 本)。"""
    return SnapshotComplete(
        folded_through=CompletePosition(folded_through.month, int(folded_through.line)),
        rows=int(rows),
        embedding_version=embedding_version,
        records_segments=[RecordsSegment(RECORDS_FILE, int(rows))],
    )


def write_complete(directory: Path | str, complete: SnapshotComplete) -> Path:
    """版を確定する ``COMPLETE`` を fsync して書く (c_16 §5.6 の手順 4)。

    版データと埋め込み版を書き終えた後にだけ呼ぶ (これが無い版は読まない)。
    """
    path = Path(directory) / COMPLETE_FILE
    write_versioned(
        path,
        format_id=SNAPSHOT_COMPLETE_FORMAT.format_id,
        format_version=SNAPSHOT_COMPLETE_FORMAT.version,
        payload=_COMPLETE_CODEC.encode(complete),
        component="evidence.snapshot",
        fsync=True,
    )
    return path


def trash_version(store_dir: Path | str, version: str) -> Path | None:
    """版ディレクトリを ``<版>.trash-<utcstamp>`` へ改名する (消さない)。

    readonly 中は改名しない (``None``)。中身は次の :func:`sweep_trashed_versions` が消す。
    """
    directory = snapshot_dir(store_dir, version)
    if not directory.exists():
        return None
    target = quarantine(directory, TRASH_KIND)
    if target is None:
        logger.warning("could not move snapshot %s to trash", directory)
    return target


class SnapshotWriter:
    """事象の畳み込みと版の書き出し。"""

    @staticmethod
    def fold(
        prev_records: Iterable[Evidence],
        events: Iterable[dict[str, Any]],
    ) -> list[Evidence]:
        """前 snapshot + 事象 → 現在状態 (c_16 §5.2)。

        - ``create``: 新しい id の追加。**既に在る id への create は飛ばす** (書き手
          ``EvidenceStore.create`` が拒否するので、ログに現れるのは壊れた / 重複した
          ログだけ。既存のレコードを勝たせ、件数を WARNING に出す)
        - ``put``: 完全なレコードで置き換え / 追加 (明示の全置換)
        - ``patch``: :data:`PATCHABLE_FIELDS` のみ差し替え
        - ``retract``: ``veracity=retracted`` (物理削除はしない)
        - ``touch``: ``last_used_at`` のみ更新 (複数 id を 1 事象で)

        未知 id への patch / retract / touch は件数を WARNING に出して飛ばす
        (全体を落とさない)。順序は put された順を保つ。
        """
        folded: dict[str, Evidence] = {r.id: r for r in prev_records}
        unknown = 0
        broken = 0
        duplicate_creates = 0

        for event in events:
            op = event.get("op")
            payload = event.get("payload") or {}
            record_id = str(event.get("id") or "")
            at = str(event.get("at") or utc_now())

            if op in ("create", "put"):
                raw = payload.get("record")
                if not isinstance(raw, dict):
                    broken += 1
                    continue
                try:
                    record = from_record(raw)
                except EvidenceVersionError:
                    # 版が新しいレコードは飛ばさない — 飛ばして畳むと、次の
                    # 版で「知らないフィールドを落としたレコード」が正になる。
                    raise
                except EvidenceRecordError as e:
                    logger.warning("skipping malformed %s event %s: %s", op, record_id, e)
                    broken += 1
                    continue
                if op == "create" and record.id in folded:
                    duplicate_creates += 1
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
                fields = payload.get("fields", {})
                unset = payload.get("unset", [])
                if not isinstance(fields, dict) or not isinstance(unset, list):
                    broken += 1
                    continue
                try:
                    folded[record_id] = apply_patch(current, fields, at=at, unset=unset)
                except EvidenceRecordError as e:  # 型付きの入れ子に読めない値
                    logger.warning("skipping malformed patch event %s: %s", record_id, e)
                    broken += 1
            elif op == "retract":
                reason = str(payload.get("reason") or "")
                extra = dict(current._extra or {})
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
        if duplicate_creates:
            logger.warning(
                "fold: %d create event(s) for existing record ids skipped", duplicate_creates,
            )
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

        fsync するのは SoT の ``records.jsonl`` だけ (c_16 §5.6)。offsets /
        columns は壊れていれば作り直す derived。
        """
        directory = snapshot_dir(store_dir, version)
        directory.mkdir(parents=True, exist_ok=True)

        offsets: list[int] = []
        cursor = 0
        with AtomicWriter(directory / RECORDS_FILE, mode="wb", fsync=True) as f:
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
        #: id (ASCII バイト列) → 行。``ids`` は ``S24`` なのでキーもバイト列のまま持つ。
        self._id_index: dict[bytes, int] = self.columns.id_index()
        #: 読んだ行に :data:`RECORD_VERSION` より新しいものがあった (§5.1)。
        #: 立った版は畳んで書き戻してはいけないので、``EvidenceStore`` は
        #: これを見て readonly に落ちる。
        self.unsupported_version: bool = False
        #: 版の確定の印 (無ければ ``None``、c_16 §5.6)。
        complete = read_complete_result(self.directory)
        #: ``COMPLETE`` の読み取りの分類。新しい版 / G1 でない印の版は無効扱いに
        #: せず readonly にする (``EvidenceStore`` が見る。trash へ送ると壊す)。
        self.complete_status = complete.status
        self.complete: SnapshotComplete | None = complete.payload if complete.ok else None
        #: ``COMPLETE`` があり、その ``rows`` が offsets / columns の行数と合う。
        #: 偽の版は使わない (``EvidenceStore.load`` が回復と同じく前の版へ落ちる)。
        self.valid: bool = self._check_complete()

    def _check_complete(self) -> bool:
        """``COMPLETE`` の ``rows`` と ``len(offsets)`` を照合する (追加 I/O なし)。"""
        if len(self.offsets) != len(self.columns):
            logger.error(
                "Snapshot misalignment at %s: %d offsets vs %d columns. "
                "Rebuild the snapshot.",
                self.directory, len(self.offsets), len(self.columns),
            )
            return False
        if self.complete is None:
            logger.error("Snapshot %s has no readable %s (%s); it is not a valid version",
                         self.directory, COMPLETE_FILE, self.complete_status)
            return False
        rows = self.complete.rows
        if rows != len(self.offsets):
            logger.error(
                "Snapshot %s: %s declares %r row(s) but offsets hold %d",
                self.directory, COMPLETE_FILE, rows, len(self.offsets),
            )
            return False
        return True

    @property
    def folded_through(self) -> EventPosition:
        """``COMPLETE`` が記録した畳み込み位置 (無ければ先頭)。"""
        return self.complete.position if self.complete is not None else EventPosition()

    # ── 位置引き ──

    def __len__(self) -> int:
        return len(self.columns)

    @property
    def ids(self) -> np.ndarray:
        return self.columns.ids

    def row_of(self, record_id: str) -> int | None:
        key = encode_id(record_id)
        return None if key is None else self._id_index.get(key)

    def id_at(self, row: int) -> str:
        return self.columns.ids[row].decode("ascii")

    # ── 本文の lazy 読み ──

    def raw_at(self, row: int) -> dict[str, Any] | None:
        """指定行の生 dict を読む (offsets で seek)。"""
        if row < 0 or row >= len(self.offsets) or not self.records_path.exists():
            return None
        with self.records_path.open("rb") as f:
            f.seek(int(self.offsets[row]))
            line = f.readline()
        return self._parse_line(line, row)

    def iter_raw(self) -> Iterator[tuple[int, dict[str, Any] | None]]:
        """全行の ``(行, 生 dict)`` を行順に返す (読めない行は ``None``)。

        全件走査はこちらを使う。:meth:`raw_at` を行ごとに呼ぶと 1 行ずつ open
        し直し (Windows で約 74µs)、50k 行で 7〜11 秒かかる (G1 設計 §11-1)。
        ファイルは 1 回だけ開いて順に読み、``offsets`` と位置がずれた行だけ
        seek する (:meth:`raw_at` と同じ行を返す)。版ディレクトリを掴み続け
        ないよう、走査が終わればすぐ閉じる。
        """
        count = len(self.offsets)
        if count == 0:
            return
        try:
            f = self.records_path.open("rb")
        except OSError:
            return
        with f:
            position = 0
            for row in range(count):
                start = int(self.offsets[row])
                if start != position:
                    f.seek(start)
                line = f.readline()
                position = start + len(line)
                yield row, self._parse_line(line, row)

    def _parse_line(self, line: bytes, row: int) -> dict[str, Any] | None:
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
        return self.record_from_raw(self.raw_at(row), row)

    def record_from_raw(self, raw: dict[str, Any] | None, row: int) -> Evidence | None:
        """:meth:`raw_at` / :meth:`iter_raw` の生 dict を :class:`Evidence` にする。"""
        if raw is None:
            return None
        try:
            return from_record(raw)
        except EvidenceVersionError as e:
            self.unsupported_version = True
            logger.error(
                "Snapshot %s row %d is a newer record version: %s",
                self.directory, row, e,
            )
            return None
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
        # 多バイト文字の途中で切れた行が 1 本あるだけで全体の読み出しを落とさない
        # (壊れた行として飛ばす)。NUL を含む行も読み手が飛ばす (c_05 §0.5.8)。
        with self.records_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                if "\x00" in text:
                    skipped += 1
                    continue
                try:
                    yield from_record(json.loads(text))
                except EvidenceVersionError as e:
                    self.unsupported_version = True
                    logger.error(
                        "Snapshot %s holds a newer record version: %s",
                        self.records_path, e,
                    )
                    raise
                except (json.JSONDecodeError, EvidenceRecordError):
                    skipped += 1
        if skipped:
            logger.warning(
                "%s: skipped %d unreadable record(s)", self.records_path, skipped,
            )


def read_snapshot(store_dir: Path | str, version: str) -> SnapshotReader:
    """版名を指定して :class:`SnapshotReader` を開く。"""
    return SnapshotReader(snapshot_dir(store_dir, version))


def _field_default(name: str) -> tuple[bool, Any]:
    """``Evidence`` のフィールド既定値 (``(持つか, 値)``)。必須フィールドは持たない。"""
    for f in dataclasses.fields(Evidence):
        if f.name != name:
            continue
        if f.default is not dataclasses.MISSING:
            return True, f.default
        if f.default_factory is not dataclasses.MISSING:
            return True, f.default_factory()
        return False, None
    return False, None


def apply_patch(
    record: Evidence,
    fields: dict[str, Any],
    *,
    at: str | None = None,
    unset: Iterable[str] = (),
) -> Evidence:
    """``patch`` 事象のフィールドを 1 レコードへ適用する (G1 の patch op)。

    :data:`PATCHABLE_FIELDS` 以外は無視する (put で置き換えるべきコア部分を
    事象で少しずつ壊さない)。``tier`` は ``attrs.tier`` のショートカット。
    ``attrs`` / ``_extra`` はキー単位で重ね、それ以外は置き換える。

    ``unset`` は ``fields`` の後に適用する: ``"attrs.<key>"`` / ``"_extra.<key>"`` は
    そのキーを消し、``"tier"`` は ``attrs.tier`` を消し、それ以外の patch 可能な
    フィールド名は既定値へ戻す (必須フィールドは戻せないので無視)。
    """
    stamp = at or utc_now()
    updates: dict[str, Any] = {}
    attrs: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None
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
        if key == "_extra":
            if isinstance(value, dict):
                extra = {**(extra if extra is not None else record._extra or {}), **value}
            continue
        updates[key] = value

    for name in unset:
        name = str(name)
        head, dot, key = name.partition(".")
        if dot and head in UNSETTABLE_PREFIXES:
            if head == "attrs":
                attrs = dict(record.attrs) if attrs is None else attrs
                attrs.pop(key, None)
            else:
                extra = dict(record._extra or {}) if extra is None else extra
                extra.pop(key, None)
            continue
        if name == "tier":
            attrs = dict(record.attrs) if attrs is None else attrs
            attrs.pop("tier", None)
            continue
        has_default, default = _field_default(name)
        if dot or name not in PATCHABLE_FIELDS or not has_default:
            ignored.append(f"unset:{name}")
            continue
        updates[name] = default

    if attrs is not None:
        updates["attrs"] = attrs
    if extra is not None:
        updates["_extra"] = extra
    if ignored:
        logger.warning(
            "patch on %s ignored non-patchable field(s): %s",
            record.id, ", ".join(sorted(ignored)),
        )
    updates["updated_at"] = stamp
    return replace(record, **updates)


def sweep_trashed_versions(store_dir: Path | str) -> int:
    """前回の prune / 回復が trash へ送った版を掃く (c_16 §5.4)。

    削除は「まず改名 → 中身を消す」の 2 段で行う。改名は原子的なので、
    途中で落ちても **``records.jsonl`` を失った版ディレクトリが残らない**
    (残ると :func:`list_versions` から永久に見えず、誰も再試行しない)。
    掃き残しはここで次回に回収する。

    Returns:
        消せたディレクトリ数。
    """
    root = Path(store_dir) / SNAPSHOT_DIR
    if not root.exists():
        return 0
    swept = 0
    for path in sorted(root.iterdir()):
        if not path.is_dir() or TRASH_MARK not in path.name:
            continue
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            logger.warning("failed to sweep trashed snapshot %s", path)
            continue
        swept += 1
    return swept


def _discard_version(store_dir: Path | str, version: str) -> bool:
    """版を trash へ改名してから中身を消す (消し残しは次の sweep が拾う)。"""
    trash = trash_version(store_dir, version)
    if trash is None:
        return False
    shutil.rmtree(trash, ignore_errors=True)
    if trash.exists():
        logger.warning(
            "snapshot %s was moved to %s but could not be deleted; "
            "the next prune will sweep it", version, trash.name,
        )
    return True


def prune_snapshots(
    store_dir: Path | str, keep: int, *, protect: Iterable[str] = (),
) -> list[str]:
    """古い版ディレクトリを削除する (c_16 §5.4: 直近 3 版)。

    ``protect`` の版は件数に関わらず残す (active 版を消して起動不能にしない)。
    ``COMPLETE`` の無い版 (書きかけで落ちた版、c_16 §5.6) は件数に数えず、
    保護されていなければ一緒に trash へ送る。

    削除は **``<版>.trash-<utcstamp>`` へ改名してから中身を消す**。ファイル単位で
    消していくと、``records.jsonl`` だけ消えた版ディレクトリが残り、
    :func:`list_versions` の対象から外れて二度と再試行されない (Windows では
    memmap を掴んだままの索引が消せず、実際にこの形で残る)。改名は原子的
    なので、どこで落ちても「生きた版」か「trash」かのどちらかに落ち着く。

    Returns:
        削除した (= trash へ移した) 有効な版名。
    """
    if keep < 1:
        return []
    sweep_trashed_versions(store_dir)
    protected = {v for v in protect if v}
    incomplete = [
        v for v in list_incomplete_versions(store_dir)
        if v not in protected and _discard_version(store_dir, v)
    ]
    if incomplete:
        logger.warning(
            "Discarded %d snapshot version(s) without %s: %s",
            len(incomplete), COMPLETE_FILE, ", ".join(incomplete),
        )
    versions = list_versions(store_dir)
    removable = [v for v in versions if v not in protected]
    surplus = len(versions) - keep
    if surplus <= 0:
        return []
    removed = [v for v in removable[:surplus] if _discard_version(store_dir, v)]
    if removed:
        logger.info("Pruned %d old snapshot(s): %s", len(removed), ", ".join(removed))
    return removed


__all__ = [
    "COMPLETE_FILE",
    "SNAPSHOT_COMPLETE_FORMAT",
    "OFFSETS_FILE",
    "RECORDS_FILE",
    "SNAPSHOT_DIR",
    "TRASH_KIND",
    "TRASH_MARK",
    "CompleteEmbedding",
    "CompletePosition",
    "RecordsSegment",
    "SnapshotComplete",
    "SnapshotReader",
    "SnapshotWriter",
    "apply_patch",
    "list_incomplete_versions",
    "list_versions",
    "new_complete",
    "prune_snapshots",
    "read_complete",
    "read_complete_result",
    "read_snapshot",
    "snapshot_dir",
    "sweep_trashed_versions",
    "trash_version",
    "version_name",
    "version_seq",
    "write_complete",
]
