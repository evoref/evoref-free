"""データ根の export / import v1 (G1 設計 §8.6 / §8.7 / §15.5 / §17.1)。

停止中に単一書き手ロックの下で CLI (``evoref export`` / ``evoref import``) から
呼ぶ。稼働中の backend からは呼ばない (API は持たない)。

export:

- 対象は形式台帳 (:mod:`backend.io.ledger_files`) で ``export=True`` かつ ``sot``
  に分類されたファイルだけ。台帳外・derived・volatile・system・退避名は出ない。
- カテゴリ (:data:`CATEGORIES`) は置き場で決める (:data:`CATEGORY_PREFIXES`)。
  CLI の手動保存セッション (``store/cli_sessions/``) は ``history`` に含める。
  ``learning`` は全 model_key のパーティション (Pro なら ``store/pro/learning/`` も)。
- Evidence Store (episodic / semantic) は事象ログや snapshot を写さず、畳み込み済みで
  取り消されていないレコードを ``memory/<store>/records.jsonl`` へ論理ダンプする。
  ``private`` / ``secret`` のレコードは ``include_private`` が無い限り除く。
- テキスト (json / jsonl / md / yaml) は ``redact_string`` で走査し、既定で伏せる。
- 書き出し先はデータ根の外か ``<data_root>/outputs/`` の下。

import:

- パッケージの manifest を先に検査する (G0 / 新しい版 / 新しい形式は全体を拒否)。
- 全エントリを :mod:`backend.io.safe_extract` で検査し、パスは export が作る形
  (台帳で ``export=True`` の sot に分類される ``store/...`` とダンプ) だけを許す。パッケージの
  ``store/...`` は世代フォルダ ``g<N>/`` からの相対で、取り込み先もこのリリースの世代フォルダ。
- カテゴリごとに取り込み先が **空** のときだけ置換する (統合・上書きはしない)。
  ``learning`` はパーティション (``store/learning/<key>/`` 等) ごとに判定する。
- Evidence のダンプは新しいストアへ ``create`` し直す (snapshot・埋め込み・索引は
  起動後の sleep-time が作る)。他のファイルは ``AtomicWriter`` で書く。
- 取り込み後に世代印を現行へ書く (次の起動ゲートが readonly にしないため)。
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from backend.config import PathResolver
from backend.data_root import generation_root, store_root
from backend.free.memory.episodic.store import STORE_DIRNAME as EPISODIC_DIRNAME
from backend.free.memory.semantic.store import STORE_DIRNAME as SEMANTIC_DIRNAME
from backend.io import jsoncodec, ledger_files
from backend.io.atomic import AtomicWriter
from backend.io.format_registry import FORMATS, PRO_STORE_PREFIX, FormatRegistry, FormatSpec
from backend.io.generation_seal import (
    SEAL_FILENAME,
    Seal,
    check_seal,
    updated_seal,
    write_seal,
)
from backend.io.safe_extract import (
    ExtractLimits,
    UnsafeArchiveError,
    check_zip_members,
    extract_zip,
    normalize_member_name,
    resolve_under,
)
from backend.log_config import get_logger
from backend.structlog_config import redact_string
from backend.utils import utc_compact_stamp, utc_now

logger = get_logger("export_import")

#: export パッケージの manifest (zip の中にだけ在り、データ根の台帳には載らない)。
EXPORT_FORMAT_ID = "evoref.export"
EXPORT_FORMAT_VERSION = 1
MANIFEST_NAME = "export-manifest.json"
PACKAGE_SUFFIX = ".evoref-export.zip"
EXPORT_COMPONENT = "cli.export"
IMPORT_COMPONENT = "cli.import"
#: manifest は c_05 §0.5.1 の封筒 (p_02 §0.5.1)。トップレベルのキー。
MANIFEST_KEYS: tuple[str, ...] = ("format_id", "format_version", "written_at", "producer", "payload")
#: ``payload`` のキー (Pro だけ ``pro_data_generation`` が ``data_generation`` の後に加わる)。
PAYLOAD_KEYS: tuple[str, ...] = ("data_generation", "categories", "entries", "stats")
#: ``payload.entries`` の各要素のキー。
ENTRY_KEYS: tuple[str, ...] = ("path", "format_id", "format_version", "edition", "bytes")

CATEGORIES: tuple[str, ...] = ("memory", "history", "learning", "corpus", "overrides")

_LAYOUT = PathResolver.LAYOUT
#: 置き場の接頭辞 → カテゴリ (最初に当たったもの)。
CATEGORY_PREFIXES: tuple[tuple[str, str], ...] = (
    (_LAYOUT["memory_dir"], "memory"),
    (_LAYOUT["history_dir"], "history"),
    (_LAYOUT["cli_sessions_dir"], "history"),
    (_LAYOUT["learning_dir"], "learning"),
    (f"{PRO_STORE_PREFIX}learning/", "learning"),
    (_LAYOUT["corpus_dir"], "corpus"),
    (f"{PRO_STORE_PREFIX}created/", "corpus"),
    (str(PurePosixPath(_LAYOUT["triggers_dir"]).parent) + "/", "overrides"),
)
_LEARNING_ROOTS: tuple[str, ...] = (_LAYOUT["learning_dir"], f"{PRO_STORE_PREFIX}learning/")

#: 論理ダンプする Evidence Store (``store/memory/<name>/``)。
EVIDENCE_STORES: tuple[str, ...] = (EPISODIC_DIRNAME, SEMANTIC_DIRNAME)
#: Evidence Store の中身の形式 (ファイルとしては写さずダンプに置き換える)。
_EVIDENCE_PREFIX = "evidence."
EVIDENCE_RECORD_FORMAT_ID = "evidence.record"
_DUMP_FILENAME = "records.jsonl"

_TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".yaml", ".yml"})

#: 取り込みの上限 (記憶・履歴・学習データ。モデル本体は含まない)。
IMPORT_LIMITS = ExtractLimits(
    max_entries=500_000, max_member_bytes=4 << 30, max_total_bytes=16 << 30, max_ratio=1000.0,
)


class DataTransferError(Exception):
    """export / import を止める理由。``key`` は CLI の案内 (i18n) の鍵。"""

    def __init__(self, key: str, detail: str = "", **params: Any) -> None:
        self.key = key
        self.params = {"detail": detail, **params}
        super().__init__(detail or key)


# ── 分類 ──


def category_of(rel: str) -> str | None:
    """データ根からの相対 posix パスのカテゴリ (export の対象外なら ``None``)。"""
    for prefix, category in CATEGORY_PREFIXES:
        if rel.startswith(prefix):
            return category
    return None


def learning_partition(rel: str) -> str:
    """学習データのパーティション (``store/learning/<key>`` / ``store/pro/learning/<key>``)。"""
    for root in _LEARNING_ROOTS:
        if rel.startswith(root):
            key = rel[len(root):].split("/", 1)[0]
            return f"{root}{key}"
    raise ValueError(f"not a learning path: {rel}")


def dump_path(store_name: str) -> str:
    """Evidence Store の論理ダンプのパッケージ内パス。"""
    return f"memory/{store_name}/{_DUMP_FILENAME}"


def _is_evidence(spec: FormatSpec) -> bool:
    return spec.format_id.startswith(_EVIDENCE_PREFIX)


def _edition_of(spec: FormatSpec) -> str:
    return "pro" if spec.writers == frozenset({"pro"}) else "free"


def _visible(spec: FormatSpec, edition: str) -> bool:
    """実行中のエディションが扱う形式か (Free は Pro だけが書く形式を見ない)。"""
    return edition == "pro" or "free" in spec.writers


def exportable_files(
    data_root: Path, registry: FormatRegistry, *, edition: str, categories: Iterable[str],
) -> list[tuple[Path, str, FormatSpec, str]]:
    """バイト列のまま写すファイル: (パス, 相対パス, 形式, カテゴリ)。

    全形式で分類してから ``export`` / ``klass`` で絞る (台帳外・退避名は落ちる)。
    Evidence Store の中身はダンプに置き換えるので含めない。
    """
    wanted = set(categories)
    out: list[tuple[Path, str, FormatSpec, str]] = []
    for path, spec in ledger_files.iter_files(generation_root(data_root), registry):
        if not (spec.export and spec.klass == "sot") or _is_evidence(spec) or not _visible(spec, edition):
            continue
        rel = path.relative_to(generation_root(data_root)).as_posix()
        category = category_of(rel)
        if category in wanted:
            out.append((path, rel, spec, category))
    return out


def evidence_stores(data_root: Path, registry: FormatRegistry) -> list[str]:
    """``store/memory/`` に在る Evidence Store の名前 (中身の形式のファイルがあるもの)。"""
    memory = _LAYOUT["memory_dir"]
    names: set[str] = set()
    for path, spec in ledger_files.iter_files(generation_root(data_root), registry):
        if not _is_evidence(spec):
            continue
        rel = path.relative_to(generation_root(data_root)).as_posix()
        if rel.startswith(memory):
            names.add(rel[len(memory):].split("/", 1)[0])
    return sorted(n for n in names if n in EVIDENCE_STORES)


# ── 世代印 (起動ゲートと同じ判定) ──


def seal_problems(store_dir: Path, registry: FormatRegistry, *, edition: str) -> tuple[Seal | None, list[str]]:
    """起動ゲートが readonly / 拒否にする理由 (判定は ``generation_seal.check_seal`` 1 つ)。

    空のリストなら書いてよい。
    """
    check = check_seal(store_dir, registry, edition)
    if check.unreadable is None and check.pending:
        return check.seal, [f"a data migration is in progress for {', '.join(check.pending)}"]
    return check.seal, check.readonly_reasons(store_dir)


def _require_writable_seal(store_dir: Path, registry: FormatRegistry, edition: str) -> Seal | None:
    seal, reasons = seal_problems(store_dir, registry, edition=edition)
    if reasons:
        raise DataTransferError("cli.data_transfer_readonly", "; ".join(reasons))
    return seal


# ── 伏せ字 ──


def _mask_value(value: Any) -> tuple[Any, int]:
    """文字列の値に ``redact_string`` を掛ける。(結果, 伏せた文字列の数)。"""
    if isinstance(value, str):
        masked = redact_string(value)
        return (value, 0) if masked == value else (masked, 1)
    if isinstance(value, dict):
        hits = 0
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[key], n = _mask_value(item)
            hits += n
        return (out, hits) if hits else (value, 0)
    if isinstance(value, list):
        hits = 0
        items: list[Any] = []
        for item in value:
            masked_item, n = _mask_value(item)
            items.append(masked_item)
            hits += n
        return (items, hits) if hits else (value, 0)
    return value, 0


def _mask_json_text(text: str) -> tuple[str, int]:
    try:
        parsed = jsoncodec.loads(text)
    except ValueError:
        masked = redact_string(text)  # 読めない JSON は字面で伏せる
        return masked, int(masked != text)
    masked_value, hits = _mask_value(parsed)
    return (jsoncodec.dumps(masked_value), hits) if hits else (text, 0)


def mask_text(data: bytes, suffix: str) -> tuple[bytes, int]:
    """テキストファイルの中身を伏せる。JSON は値だけを伏せて形を保つ。"""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, 0
    if suffix == ".json":
        masked, hits = _mask_json_text(text)
    elif suffix == ".jsonl":
        lines: list[str] = []
        hits = 0
        for line in text.split("\n"):
            if line.strip():
                line, n = _mask_json_text(line)
                hits += n
            lines.append(line)
        masked = "\n".join(lines)
    else:
        masked = redact_string(text)
        hits = int(masked != text)
    return (masked.encode("utf-8"), hits) if hits else (data, 0)


# ── export ──


@dataclass(slots=True)
class CategoryStats:
    """export したカテゴリの件数。"""

    files: int = 0
    bytes: int = 0
    records: int = 0
    private_excluded: int = 0
    ignored_records: int = 0
    redacted: int = 0

    def to_json(self) -> dict[str, int]:
        return {
            "files": self.files, "bytes": self.bytes, "records": self.records,
            "private_excluded": self.private_excluded, "ignored_records": self.ignored_records,
            "redacted": self.redacted,
        }


@dataclass(slots=True)
class ExportReport:
    path: Path
    manifest: dict[str, Any]
    stats: dict[str, CategoryStats]
    masked: bool


def default_export_name() -> str:
    return f"evoref-export-{utc_compact_stamp()}{PACKAGE_SUFFIX}"


def resolve_destination(data_root: Path, output: Path | None) -> Path:
    """書き出し先を決める (データ根の中なら ``outputs/`` の下だけを許す)。"""
    outputs = PathResolver.layout_path(data_root, "outputs_dir")
    if output is None:
        path = outputs / default_export_name()
    elif output.is_dir():
        path = output / default_export_name()
    elif output.name.endswith(PACKAGE_SUFFIX):
        path = output
    else:
        path = output.with_name(output.name + PACKAGE_SUFFIX)
    resolved = path.resolve()
    if resolved.is_relative_to(data_root.resolve()) and not resolved.is_relative_to(outputs.resolve()):
        raise DataTransferError(
            "cli.export_destination_inside", str(resolved), path=str(resolved), outputs=str(outputs),
        )
    if path.exists():
        raise DataTransferError("cli.export_destination_exists", str(path), path=str(path))
    return path


def _iter_dump_lines(
    data_root: Path, store_name: str, *, include_private: bool, mask: bool, stats: CategoryStats,
) -> Iterator[str]:
    from backend.free.rag.evidence.store import EvidenceStore

    store = EvidenceStore(PathResolver.layout_path(data_root, "memory_dir") / store_name, store_name=store_name,
                          by=EXPORT_COMPONENT)
    store.load()
    try:
        if store.readonly:
            raise DataTransferError(
                "cli.data_transfer_readonly", f"{store_name} store: {store.readonly_reason}",
            )
        for record in store.iter_records():
            if record.veracity == "retracted":
                continue
            if record.ignored:
                stats.ignored_records += 1
                continue
            if (record.private or record.confidentiality == "secret") and not include_private:
                stats.private_excluded += 1
                continue
            data = record.to_record()
            masked, hits = _mask_value(data)
            stats.redacted += hits
            stats.records += 1
            yield jsoncodec.dumps(masked if mask else data)
    finally:
        store.close()


def _generations(edition: str) -> dict[str, int]:
    from backend.io.format_lock import current_generations

    generations = current_generations()
    out = {"data_generation": generations["generation"]}
    if edition == "pro" and "pro_generation" in generations:
        out["pro_data_generation"] = generations["pro_generation"]
    return out


def _app_version(edition: str) -> str:
    from backend.version import get_version_info

    info = get_version_info()
    return info.pro if edition == "pro" and info.pro else info.free


def export_data(
    data_root: Path,
    destination: Path,
    *,
    edition: str,
    categories: Iterable[str] = CATEGORIES,
    include_private: bool = False,
    mask: bool = True,
    registry: FormatRegistry = FORMATS,
) -> ExportReport:
    """``data_root`` を ``destination`` (:func:`resolve_destination` で決めた) へ書き出す。

    書き手ロックは呼出側が持つ。
    """
    selected = [c for c in CATEGORIES if c in set(categories)]
    _require_writable_seal(store_root(data_root), registry, edition)
    stats = {c: CategoryStats() for c in selected}
    entries: list[dict[str, Any]] = []
    record_spec = registry.get(EVIDENCE_RECORD_FORMAT_ID)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with AtomicWriter(destination, mode="wb") as fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as zf:
        if "memory" in stats:
            for store_name in evidence_stores(data_root, registry):
                lines = list(_iter_dump_lines(
                    data_root, store_name, include_private=include_private, mask=mask,
                    stats=stats["memory"],
                ))
                data = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
                rel = dump_path(store_name)
                zf.writestr(rel, data)
                entries.append({"path": rel, "format_id": record_spec.format_id,
                                "format_version": record_spec.version, "edition": "free",
                                "bytes": len(data)})
                stats["memory"].files += 1
                stats["memory"].bytes += len(data)
        for path, rel, spec, category in exportable_files(
            data_root, registry, edition=edition, categories=selected,
        ):
            data = path.read_bytes()
            if path.suffix in _TEXT_SUFFIXES:
                masked, hits = mask_text(data, path.suffix)
                stats[category].redacted += hits
                if mask:
                    data = masked
            zf.writestr(rel, data)
            entries.append({"path": rel, "format_id": spec.format_id, "format_version": spec.version,
                            "edition": _edition_of(spec), "bytes": len(data)})
            stats[category].files += 1
            stats[category].bytes += len(data)
        manifest: dict[str, Any] = {
            "format_id": EXPORT_FORMAT_ID,
            "format_version": EXPORT_FORMAT_VERSION,
            "written_at": utc_now(),
            "producer": {"component": EXPORT_COMPONENT, "app_version": _app_version(edition),
                         "edition": edition},
            "payload": {
                **_generations(edition),
                "categories": selected,
                "entries": entries,
                "stats": {c: s.to_json() for c, s in stats.items()},
            },
        }
        zf.writestr(MANIFEST_NAME, jsoncodec.dumps_bytes(manifest, indent=2))
    logger.info(
        "Exported %d file(s) in %s to %s (masked=%s)", len(entries), ",".join(selected), destination, mask,
    )
    return ExportReport(path=destination, manifest=manifest, stats=stats, masked=mask)


# ── import ──


@dataclass(slots=True)
class Entry:
    path: str
    format_id: str
    format_version: int
    edition: str
    bytes: int


@dataclass(slots=True)
class CategoryResult:
    """取り込みのカテゴリごとの結果。"""

    status: str = "imported"  # imported | not_empty
    files: int = 0
    records: int = 0
    invalid_records: int = 0
    #: learning で取り込み先が空でなかったパーティション。
    skipped_partitions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ImportReport:
    categories: dict[str, CategoryResult] = field(default_factory=dict)
    skipped_pro: int = 0
    skipped_unknown: int = 0


def _bad_manifest(detail: str) -> DataTransferError:
    return DataTransferError("cli.import_not_a_package", detail)


def read_manifest(zf: zipfile.ZipFile) -> tuple[dict[str, Any], list[Entry]]:
    """manifest を読んで検査する (取り込みの前に必ず通す)。"""
    try:
        raw = zf.read(MANIFEST_NAME)
    except KeyError:
        raise _bad_manifest(f"{MANIFEST_NAME} not found") from None
    try:
        manifest = jsoncodec.loads(raw)
    except ValueError as e:
        raise _bad_manifest(f"{MANIFEST_NAME} is not JSON: {e}") from None
    if not isinstance(manifest, dict):
        raise _bad_manifest(f"{MANIFEST_NAME} is not an object")
    if "format_id" not in manifest:
        raise DataTransferError("cli.import_g0_package", "the package has no format_id (G0 export)")
    if manifest["format_id"] != EXPORT_FORMAT_ID:
        raise _bad_manifest(f"format_id {manifest['format_id']!r}")
    version = manifest.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise _bad_manifest(f"format_version {version!r}")
    if version > EXPORT_FORMAT_VERSION:
        raise DataTransferError("cli.import_newer_package", f"format_version {version}", version=version)
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise _bad_manifest("payload must be an object")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise _bad_manifest("entries must be a list")
    entries: list[Entry] = []
    seen: set[str] = set()
    for item in raw_entries:
        if not isinstance(item, dict):
            raise _bad_manifest("an entry is not an object")
        path, format_id, fversion, edition, size = (item.get(k) for k in ENTRY_KEYS)
        if not (isinstance(path, str) and isinstance(format_id, str) and edition in ("free", "pro")
                and isinstance(size, int) and isinstance(fversion, int)
                and not isinstance(fversion, bool) and fversion >= 1):
            raise _bad_manifest(f"bad entry {item!r}")
        if path in seen or path == MANIFEST_NAME:
            raise _bad_manifest(f"duplicate entry {path}")
        seen.add(path)
        entries.append(Entry(path=path, format_id=format_id, format_version=fversion,
                             edition=edition, bytes=size))
    return manifest, entries


def _check_versions(entries: list[Entry], registry: FormatRegistry) -> None:
    newer = sorted({
        f"{e.format_id} v{e.format_version}" for e in entries
        if (spec := registry.find(e.format_id)) is not None and e.format_version > spec.version
    })
    if newer:
        raise DataTransferError("cli.import_newer_format", ", ".join(newer), formats=", ".join(newer))


def _entry_category(entry: Entry, classify: ledger_files.Classifier) -> str:
    """entry の置き場を検査してカテゴリを返す (export が作る形でなければ拒否)。"""
    normalized = normalize_member_name(entry.path)
    if normalized is None or str(normalized) != entry.path:
        raise DataTransferError("cli.import_unsafe_member", entry.path)
    if entry.format_id == EVIDENCE_RECORD_FORMAT_ID and entry.path in {dump_path(s) for s in EVIDENCE_STORES}:
        return "memory"
    spec = classify(entry.path)
    category = category_of(entry.path)
    if (
        spec is None or spec.format_id != entry.format_id or not (spec.export and spec.klass == "sot")
        or _is_evidence(spec) or category is None or _edition_of(spec) != entry.edition
    ):
        raise DataTransferError("cli.import_unsafe_member", entry.path)
    return category


def _occupied(data_root: Path, registry: FormatRegistry, edition: str) -> tuple[set[str], set[str]]:
    """取り込み先で sot のファイルを持つ (カテゴリ, learning のパーティション)。"""
    categories: set[str] = set()
    partitions: set[str] = set()
    for path, spec in ledger_files.iter_files(generation_root(data_root), registry):
        if spec.klass != "sot" or not _visible(spec, edition):
            continue
        rel = path.relative_to(generation_root(data_root)).as_posix()
        category = category_of(rel)
        if category is None:
            continue
        categories.add(category)
        if category == "learning":
            partitions.add(learning_partition(rel))
    return categories, partitions


def _copy_into_store(source: Path, target: Path) -> None:
    with open(source, "rb") as src, AtomicWriter(target, mode="wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)


def _restore_evidence(data_root: Path, store_name: str, dump: Path, result: CategoryResult) -> None:
    from backend.free.rag.evidence.store import EvidenceIdExistsError, EvidenceStore
    from backend.free.rag.evidence.types import EvidenceRecordError, EvidenceVersionError, from_record

    store_dir = PathResolver.layout_path(data_root, "memory_dir") / store_name
    store_dir.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(store_dir, store_name=store_name, by=IMPORT_COMPONENT)
    store.load()
    try:
        with open(dump, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = from_record(jsoncodec.loads(line))
                    if record.store != store_name:
                        raise EvidenceRecordError(f"record {record.id} belongs to {record.store}")
                    store.create(record, by=IMPORT_COMPONENT)
                except (ValueError, EvidenceRecordError, EvidenceVersionError, EvidenceIdExistsError) as e:
                    result.invalid_records += 1
                    logger.warning("Skipped an evidence record in %s: %s", dump.name, e)
                    continue
                result.records += 1
        store.save_manifest()
    finally:
        store.close()


def import_data(
    data_root: Path,
    package: Path,
    *,
    edition: str,
    categories: Iterable[str] | None = None,
    registry: FormatRegistry = FORMATS,
) -> ImportReport:
    """export パッケージを空の取り込み先へ置換で取り込む (書き手ロックは呼出側が持つ)。"""
    wanted = set(CATEGORIES if categories is None else categories)
    report = ImportReport()
    store_dir = store_root(data_root)
    try:
        zf = zipfile.ZipFile(package)
    except (OSError, zipfile.BadZipFile) as e:
        raise _bad_manifest(str(e)) from None
    with zf:
        manifest, entries = read_manifest(zf)
        _check_versions(entries, registry)
        try:
            members = {str(m): (info, m) for info, m in check_zip_members(zf, IMPORT_LIMITS)}
        except UnsafeArchiveError as e:
            raise DataTransferError("cli.import_unsafe_member", str(e)) from None
        expected = {e.path for e in entries} | {MANIFEST_NAME}
        if set(members) != expected:
            extra = sorted(set(members) - expected) or sorted(expected - set(members))
            raise DataTransferError("cli.import_unsafe_member", ", ".join(extra[:5]))

        classify = ledger_files.classifier(registry)
        planned: list[tuple[Entry, str]] = []
        for entry in entries:
            if entry.edition == "pro" and edition != "pro":
                report.skipped_pro += 1
                continue
            if registry.find(entry.format_id) is None:
                report.skipped_unknown += 1
                continue
            category = _entry_category(entry, classify)
            if category in wanted:
                planned.append((entry, category))

        seal = _require_writable_seal(store_dir, registry, edition)
        occupied, occupied_partitions = _occupied(data_root, registry, edition)
        selected: list[tuple[Entry, str]] = []
        for entry, category in planned:
            result = report.categories.setdefault(category, CategoryResult())
            if category in occupied and category != "learning":
                result.status = "not_empty"
                continue
            if category == "learning":
                partition = learning_partition(entry.path)
                if partition in occupied_partitions:
                    if partition not in result.skipped_partitions:
                        result.skipped_partitions.append(partition)
                    continue
            selected.append((entry, category))
        if not selected:
            return report

        tmp_root = PathResolver.layout_path(data_root, "tmp_dir")
        tmp_root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="import-", dir=tmp_root))
        try:
            extract_zip(zf, workdir, limits=IMPORT_LIMITS,
                        members=[members[entry.path] for entry, _ in selected])
            for entry, category in selected:
                result = report.categories[category]
                source = resolve_under(workdir, PurePosixPath(entry.path))
                if category == "memory" and entry.format_id == EVIDENCE_RECORD_FORMAT_ID:
                    _restore_evidence(data_root, PurePosixPath(entry.path).parent.name, source, result)
                else:
                    _copy_into_store(source, resolve_under(generation_root(data_root), PurePosixPath(entry.path)))
                result.files += 1
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    write_seal(store_dir / SEAL_FILENAME,
               updated_seal(seal, registry, edition=edition, app_version=_app_version(edition)))
    logger.info(
        "Imported %s from %s", ", ".join(f"{c}={r.files}" for c, r in report.categories.items()), package,
    )
    return report


__all__ = [
    "CATEGORIES",
    "CATEGORY_PREFIXES",
    "EVIDENCE_STORES",
    "EXPORT_FORMAT_ID",
    "EXPORT_FORMAT_VERSION",
    "ENTRY_KEYS",
    "MANIFEST_KEYS",
    "PAYLOAD_KEYS",
    "MANIFEST_NAME",
    "PACKAGE_SUFFIX",
    "CategoryResult",
    "CategoryStats",
    "DataTransferError",
    "ExportReport",
    "ImportReport",
    "category_of",
    "dump_path",
    "export_data",
    "exportable_files",
    "import_data",
    "learning_partition",
    "mask_text",
    "read_manifest",
    "resolve_destination",
    "seal_problems",
]
