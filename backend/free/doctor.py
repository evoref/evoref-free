"""evoref doctor — データ根の全件の照合 (読むだけ、G1 設計 §10.4 / §17.4)。

起動ゲートと読み手はファイルを開いたときにしか版・破損を見ない (§17.4)。全件の照合は
ここでまとめて行う。``store/`` には一切書かない (Evidence ストアの読み込みが途中で
行う修復 — 事象ログの穴埋め・無効な版の退避・manifest の退避 — も、プロセスを
readonly にして止める)。serve の稼働中でも動く (書き手ロックは取らない)。

検査 (形式ごとに ``ledger_files`` で束ねる):

- ``store/`` の台帳外のファイル (error) と退避・取り残しのファイル (info、件数だけ)
- 版付き JSON: 読み手と同じ分類 (:func:`read_versioned`) の件数。current 以外は error
- JSONL: 行数・読めない行 (warning)・新しい版の行 (error)
- Evidence ストア (``store/memory/<store>``): 実際の読み手で畳んだ件数 (全体 / 有効 /
  取り消し / 置き換え済み / 未知の列挙値)、id の一意性、active 版の COMPLETE と
  offsets の行数、ぶら下がった参照 (gc_log の墓標は許容)
- 履歴のセッション id と経験の id の一意性、未知の列挙値の経験 (計数だけ)
- スコープを跨ぐ参照 (Pro の LoRA meta → 共有の経験) は弱い参照で、欠けは計数だけ (§15.3)
- 世代印・G0 の検出・データ根の置き場

``--bundle`` の束 (:func:`write_bundle`) は報告・形式表・版・redact 済みのログの末尾だけで、
ストアの中身 (レコード・本文) は入れない。
"""

from __future__ import annotations

import contextlib
import io
import json
import platform
import zipfile
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.io import jsoncodec, ledger_files
from backend.io.format_registry import FORMATS, FormatSpec
from backend.io.generation_seal import check_seal
from backend.io.readonly import disable_readonly, enable_readonly, is_readonly
from backend.io.versioned import producer_edition, read_versioned
from backend.io.writer_lock import lock_held

#: 報告 (``--json``) の形の版。キーを足すだけなら上げない。
REPORT_VERSION = 1

Severity = Literal["error", "warning", "info"]
SEVERITIES: tuple[Severity, ...] = ("error", "warning", "info")

#: 1 つの指摘に並べるパスの数。
_MAX_PATHS = 5
#: 束に入れるログ (データ根からの相対) と末尾の行数。
BUNDLE_LOGS: tuple[str, ...] = ("logs/backend.log", "logs/learning.log")
BUNDLE_LOG_LINES = 2000

_EVIDENCE_MANIFEST = "evidence.manifest"
_HISTORY_SESSION = "history.session"
_HISTORY_TURNS = "history.turns"
_EXPERIENCE = "learning.experience"
_ADAPTER_META = "pro.adapter_versions"
_TURNS_SUFFIX = ".turns.jsonl"


class DoctorError(RuntimeError):
    """検査を走らせられない (データ根が無い / 束の置き場が不正)。"""


@dataclass(slots=True)
class Finding:
    """1 種類の指摘 (同じ ``code`` × ``format_id`` × ``detail`` は件数に畳む)。"""

    severity: Severity
    code: str
    format_id: str | None = None
    count: int = 0
    detail: str = ""
    #: 例のパス (データ根からの相対、最大 :data:`_MAX_PATHS`)。
    paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _FormatStats:
    files: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    records: int = 0
    torn: int = 0
    newer_rows: int = 0
    #: 行の版 ``_v`` の無い行 (読み手は壊れた行として飛ばす)。
    unversioned_rows: int = 0


class _Report:
    """検査の途中の状態 (最後に :meth:`to_dict`)。"""

    def __init__(self) -> None:
        self.findings: dict[tuple[str, str | None, str], Finding] = {}
        self.formats: dict[str, _FormatStats] = {}
        self.sections: dict[str, Any] = {}

    def add(
        self, severity: Severity, code: str, *, format_id: str | None = None,
        path: str | None = None, detail: str = "", count: int = 1,
    ) -> None:
        key = (code, format_id, detail)
        finding = self.findings.get(key)
        if finding is None:
            finding = self.findings[key] = Finding(severity, code, format_id, detail=detail)
        finding.count += count
        if path is not None and len(finding.paths) < _MAX_PATHS:
            finding.paths.append(path)

    def stats(self, spec: FormatSpec) -> _FormatStats:
        stats = self.formats.get(spec.format_id)
        if stats is None:
            stats = self.formats[spec.format_id] = _FormatStats()
        return stats

    def to_dict(self) -> dict[str, Any]:
        findings = sorted(
            self.findings.values(),
            key=lambda f: (SEVERITIES.index(f.severity), f.code, f.format_id or "", f.detail),
        )
        count = {s: sum(f.count for f in findings if f.severity == s) for s in SEVERITIES}
        summary = {
            "errors": count["error"], "warnings": count["warning"], "info": count["info"],
            "files": sum(s.files for s in self.formats.values()),
        }
        formats: dict[str, Any] = {}
        for format_id in sorted(self.formats):
            stats = self.formats[format_id]
            spec = FORMATS.find(format_id)
            entry: dict[str, Any] = {
                "version": spec.version if spec is not None else None,
                "class": spec.klass if spec is not None else None,
                "files": stats.files,
            }
            if stats.outcomes:
                entry["outcomes"] = dict(sorted(stats.outcomes.items()))
            if stats.records or stats.torn or stats.newer_rows or stats.unversioned_rows:
                entry.update(
                    records=stats.records, torn=stats.torn, newer_rows=stats.newer_rows,
                    unversioned_rows=stats.unversioned_rows,
                )
            formats[format_id] = entry
        return {
            "report_version": REPORT_VERSION,
            **self.sections,
            "summary": summary,
            "formats": formats,
            "findings": [asdict(f) for f in findings],
        }


@contextlib.contextmanager
def _no_store_writes(store_dir: Path) -> Iterator[None]:
    """検査の間だけプロセスを readonly にする (読み手の修復が ``store/`` へ書かないように)。"""
    if is_readonly():
        yield
        return
    enable_readonly(store_dir, "evoref doctor is read-only")
    try:
        yield
    finally:
        disable_readonly()


def _rel(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root).as_posix()


# ── 版付き JSON / JSONL ──


def _check_json(
    report: _Report, spec: FormatSpec, path: Path, rel: str, collected: dict[str, Any],
) -> None:
    stats = report.stats(spec)
    result = read_versioned(path, format_id=spec.format_id, format_version=spec.version)
    stats.outcomes[result.status] += 1
    payload = result.payload if result.ok and isinstance(result.payload, dict) else None
    if spec.format_id == _EVIDENCE_MANIFEST:
        # 読めない manifest のストアも読み手で開いて readonly の理由を出す
        collected["evidence_manifests"][path.parent] = payload
    if not result.ok:
        detail = result.detail or (f"format_version {result.version}" if result.version else "")
        report.add("error", f"json_{result.status}", format_id=spec.format_id, path=rel, detail=detail)
        return
    if spec.format_id == _HISTORY_SESSION and payload is not None:
        collected["session_ids"].append((str(payload.get("session_id") or ""), rel))
    elif spec.format_id == _ADAPTER_META and payload is not None:
        window = payload.get("experience_window")
        if isinstance(window, list):
            collected["weak_experience_refs"].extend(str(i) for i in window if i)


def _row_version(obj: dict[str, Any]) -> int | None:
    """JSONL の行の版 (``_v``、c_05 §0.5.1)。"""
    value = obj.get("_v")
    return value if type(value) is int else None


def _check_jsonl(
    report: _Report, spec: FormatSpec, path: Path, rel: str, collected: dict[str, Any],
) -> None:
    """1 行ずつ読んで数える (ファイル全体を持たない)。"""
    stats = report.stats(spec)
    torn = newer = unversioned = 0
    experience = spec.format_id == _EXPERIENCE
    ids: Counter[str] = Counter()
    try:
        f = path.open("rb")
    except OSError as e:
        report.add("error", "unreadable", format_id=spec.format_id, path=rel, detail=type(e).__name__)
        return
    with f:
        for line in f:
            if not line.strip():
                continue
            if b"\x00" in line:
                torn += 1
                continue
            try:
                obj = jsoncodec.loads(line)
            except ValueError:
                torn += 1
                continue
            if not isinstance(obj, dict):
                torn += 1
                continue
            stats.records += 1
            version = _row_version(obj)
            if version is None:
                unversioned += 1
            elif version > spec.version:
                newer += 1
            if experience and obj.get("op") != "patch" and obj.get("id"):
                ids[str(obj["id"])] += 1
    stats.torn += torn
    stats.newer_rows += newer
    stats.unversioned_rows += unversioned
    if torn:
        report.add("warning", "jsonl_torn", format_id=spec.format_id, path=rel, count=torn)
    if newer:
        report.add("error", "jsonl_newer_rows", format_id=spec.format_id, path=rel, count=newer)
    if unversioned:
        # 読み手は `_v` の無い行を壊れた行として飛ばす (c_05 §0.5.1) — 黙って使われない行を見せる
        report.add("warning", "jsonl_unversioned_rows", format_id=spec.format_id, path=rel, count=unversioned)
    if experience:
        _check_experience(report, spec, path, rel, ids, collected)
    elif spec.format_id == _HISTORY_TURNS and path.name.endswith(_TURNS_SUFFIX):
        collected["active_sessions"].append((path.name[: -len(_TURNS_SUFFIX)], rel))


def _check_experience(
    report: _Report, spec: FormatSpec, path: Path, rel: str, ids: Counter[str],
    collected: dict[str, Any],
) -> None:
    from backend.free.learning.level0_instant import fold_experience_file, unknown_enums

    duplicates = sum(n - 1 for n in ids.values() if n > 1)
    if duplicates:
        report.add("error", "experience_duplicate_id", format_id=spec.format_id, path=rel, count=duplicates)
    collected["experience_ids"].update(ids)
    records, _ = fold_experience_file(path)
    ignored = sum(1 for record in records if unknown_enums(record))
    section = collected["experience"]
    section["files"] += 1
    section["records"] += len(records)
    section["ignored"] += ignored
    if ignored:
        report.add("info", "experience_ignored", format_id=spec.format_id, path=rel, count=ignored)


# ── Evidence ストア ──


def _gc_log_ids(store_dir: Path) -> set[str]:
    """gc_log の墓標 (物理 GC で落とした id)。"""
    from backend.free.rag.evidence.store import GC_LOG_FILE, GC_LOG_ROW_VERSION

    out: set[str] = set()
    try:
        f = (store_dir / GC_LOG_FILE).open("rb")
    except OSError:
        return out
    with f:
        for line in f:
            try:
                obj = jsoncodec.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict) or _row_version(obj) != GC_LOG_ROW_VERSION:
                continue  # 版が新しい / 無い行の形は読まない (jsonl の検査が数える)
            payload = obj.get("payload")
            dropped = payload.get("dropped_ids") if isinstance(payload, dict) else None
            if isinstance(dropped, list):
                out.update(str(i) for i in dropped)
    return out


def _check_evidence_store(
    report: _Report, data_root: Path, store_dir: Path, manifest: dict[str, Any] | None,
) -> tuple[set[str], list[tuple[str, str]], dict[str, Any]]:
    """1 ストアを実際の読み手で畳んで数える。返り値は (id, 参照 (欄, 宛先), 集計)。"""
    from backend.free.rag.evidence.snapshot import list_incomplete_versions, read_snapshot
    from backend.free.rag.evidence.store import EvidenceStore

    name = store_dir.name
    rel = _rel(store_dir, data_root)
    section: dict[str, Any] = {}
    active = str((manifest or {}).get("active_snapshot") or "")
    section["active_snapshot"] = active or None
    if active:
        reader = read_snapshot(store_dir, active)
        complete_rows = reader.complete.rows if reader.complete is not None else None
        section["complete_rows"] = complete_rows
        section["offsets"] = len(reader.offsets)
        if not reader.valid:
            detail = (
                "COMPLETE missing" if reader.complete is None
                else f"COMPLETE rows {complete_rows!r}, offsets {len(reader.offsets)}, columns {len(reader.columns)}"
            )
            report.add(
                "error", "evidence_snapshot_invalid", format_id="evidence.snapshot_complete",
                path=f"{rel}/snapshot/{active}", detail=detail,
            )
    incomplete = list_incomplete_versions(store_dir)
    if incomplete:
        report.add(
            "warning", "evidence_incomplete_version", format_id="evidence.snapshot_index",
            path=f"{rel}/snapshot", count=len(incomplete),
        )

    store = EvidenceStore(store_dir, name)
    ids: set[str] = set()
    refs: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    duplicates = 0
    try:
        store.load()
        if store.readonly:
            section["readonly_reason"] = store.readonly_reason
            report.add("error", "evidence_readonly", format_id="evidence.record", path=rel, detail=store.readonly_reason)
        section["pending"] = len(store.pending_ids())
        for record in store.iter_records():
            counts["total"] += 1
            if record.id in ids:
                duplicates += 1
            ids.add(record.id)
            if record.ignored:
                counts["ignored"] += 1
            elif record.veracity == "retracted":
                counts["retracted"] += 1
            elif record.superseded_by:
                counts["superseded"] += 1
            else:
                counts["live"] += 1
            if record.superseded_by:
                refs.append(("superseded_by", record.superseded_by))
            refs.extend(("contradicts", target) for target in record.contradicts if target)
            summary_of = record.attrs.get("summary_of") if isinstance(record.attrs, dict) else None
            if isinstance(summary_of, list):
                refs.extend(("summary_of", str(t)) for t in summary_of if t)
            refs.extend(
                ("provenance.note_id", entry.note_id)
                for entry in record.provenance if getattr(entry, "note_id", None)
            )
    except Exception as e:  # noqa: BLE001 — 1 ストアが読めなくても残りを検査する
        report.add("error", "evidence_unreadable", format_id="evidence.record", path=rel, detail=type(e).__name__)
    finally:
        store.close()
    for key in ("total", "live", "retracted", "superseded", "ignored"):
        section[key] = counts[key]
    section["duplicate_ids"] = duplicates
    if duplicates:
        report.add("error", "evidence_duplicate_id", format_id="evidence.record", path=rel, count=duplicates)
    if counts["ignored"]:
        report.add("info", "evidence_ignored", format_id="evidence.record", path=rel, count=counts["ignored"])
    return ids, refs, section


def _check_evidence(report: _Report, data_root: Path, manifests: dict[Path, dict[str, Any]]) -> None:
    """記憶の Evidence ストア (``store/memory/<store>``) を全部見て、参照を解く。"""
    stores: dict[str, Any] = {}
    all_ids: set[str] = set()
    tombstones: set[str] = set()
    pending: list[tuple[str, str, list[tuple[str, str]]]] = []
    for store_dir in sorted(manifests):
        ids, refs, section = _check_evidence_store(report, data_root, store_dir, manifests[store_dir])
        stores[store_dir.name] = section
        all_ids |= ids
        tombstones |= _gc_log_ids(store_dir)
        pending.append((store_dir.name, _rel(store_dir, data_root), refs))
    for name, rel, refs in pending:
        dangling: Counter[str] = Counter()
        tolerated = 0
        for field_name, target in refs:
            if target in all_ids:
                continue
            if target in tombstones:
                tolerated += 1
                continue
            dangling[field_name] += 1
        stores[name]["dangling"] = dict(sorted(dangling.items()))
        stores[name]["tolerated_by_gc_log"] = tolerated
        for field_name, count in sorted(dangling.items()):
            # provenance は別ストア (episodic のノート) を指すので、書き手の順序次第で
            # 一時的に欠けうる。同じストアの中の参照だけを error にする。
            severity: Severity = "warning" if field_name.startswith("provenance.") else "error"
            report.add(severity, "dangling_reference", format_id="evidence.record", path=rel, detail=field_name, count=count)
    report.sections["evidence"] = stores


# ── 履歴・経験・スコープを跨ぐ参照 ──


def _check_history(report: _Report, collected: dict[str, Any]) -> None:
    closed: Counter[str] = Counter(sid for sid, _ in collected["session_ids"] if sid)
    paths = {sid: rel for sid, rel in collected["session_ids"]}
    for sid, n in closed.items():
        if n > 1:
            report.add("error", "history_duplicate_session", format_id=_HISTORY_SESSION, path=paths[sid], count=n - 1)
    # 閉じたセッションに後からターンが届くと追記ログが作り直され、次に畳むときに
    # セッション JSON と重ねる (``merge_turn_log``)。両方あるのは正常なので数えるだけ。
    report.sections["history"] = {
        "sessions": sum(closed.values()),
        "active": len(collected["active_sessions"]),
        "resumed": sum(1 for sid, _ in collected["active_sessions"] if sid in closed),
    }


def _check_weak_refs(report: _Report, collected: dict[str, Any]) -> None:
    refs = collected["weak_experience_refs"]
    missing = sum(1 for ref in refs if ref not in collected["experience_ids"])
    report.sections["weak_references"] = {"checked": len(refs), "missing": missing}
    if missing:
        report.add("info", "weak_reference_missing", format_id=_ADAPTER_META, count=missing)


# ── 世代印・置き場・G0 ──


def _check_seal(report: _Report, store_dir: Path, edition: str) -> None:
    section: dict[str, Any] = {"state": "ok"}
    check = check_seal(store_dir, FORMATS, edition)
    seal = check.seal
    if check.unreadable is not None:
        section["state"] = "unreadable"
        report.add("error", "seal_unreadable", format_id="store.generation", detail=check.unreadable.split(": ", 1)[-1])
        report.sections["seal"] = section
        return
    if seal is None:
        section["state"] = "missing"
        if check.missing_with_data:
            report.add("error", "seal_missing", format_id="store.generation")
        report.sections["seal"] = section
        return
    section.update(
        last_edition=seal.last_edition,
        written_by=seal.written_by,
        pending=seal.pending,
        newer=check.newer,
        older=check.older,
    )
    for format_id in section["newer"]:
        report.add("error", "seal_newer", format_id=format_id)
    for format_id in section["older"]:
        report.add("error", "seal_older", format_id=format_id)
    if seal.pending:
        report.add("error", "seal_pending", detail=", ".join(str(f) for f in seal.pending.get("formats") or []))
    report.sections["seal"] = section


def _check_location(report: _Report, data_root: Path) -> None:
    from backend.data_location import inspect_location

    location = inspect_location(data_root)
    report.sections["location"] = {
        "fs_type": location.fs_type, "remote": location.remote, "onedrive": location.onedrive,
    }
    if location.unsafe:
        report.add("warning", "location_unsafe", detail="network" if location.remote else "OneDrive")
    elif location.weak_fs:
        report.add("info", "location_weak_fs", detail=location.fs_type or "")


def _check_g0(report: _Report, install_root: Path | None) -> None:
    if install_root is None:
        return
    from backend.factory._data_gate import detect_g0

    g0 = detect_g0(install_root)
    report.sections["g0"] = {"detected": g0.found, "signatures": len(g0.signatures)}
    if g0.found:
        report.add("info", "g0_detected", count=len(g0.signatures))


def _app_section(edition: str) -> dict[str, Any]:
    from backend.utils import utc_now
    from backend.version import get_version_info

    info = get_version_info()
    return {
        "generated_at": utc_now(),
        "app": {
            "free_version": info.free,
            "pro_version": info.pro,
            "edition": edition,
            "generation": info.generation,
            "pro_generation": info.pro_generation,
        },
    }


# ── 入口 ──


def run_doctor(
    data_root: Path, *, install_root: Path | None = None, edition: str | None = None,
) -> dict[str, Any]:
    """データ根を検査して報告 (JSON にできる dict) を返す。``store/`` には書かない。

    Raises:
        DoctorError: データ根が無い。
    """
    from backend.formats import load_all_formats

    data_root = Path(data_root)
    if not data_root.is_dir():
        raise DoctorError(f"data root does not exist: {data_root}")
    load_all_formats()
    edition = edition or producer_edition()
    store_dir = data_root / ledger_files.STORE_DIR
    report = _Report()
    report.sections.update(_app_section(edition))
    report.sections["serve_running"] = lock_held(store_dir)
    collected: dict[str, Any] = {
        "evidence_manifests": {},
        "session_ids": [],
        "active_sessions": [],
        "experience_ids": Counter(),
        "experience": {"files": 0, "records": 0, "ignored": 0},
        "weak_experience_refs": [],
    }
    with _no_store_writes(store_dir):
        _check_location(report, data_root)
        _check_g0(report, install_root)
        _check_seal(report, store_dir, edition)
        for path, spec in ledger_files.walk(data_root):
            rel = _rel(path, data_root)
            if spec is None:
                report.add("error", "unledgered_file", path=rel)
                continue
            report.stats(spec).files += 1
            if spec is ledger_files.RESIDUE_FORMAT:
                report.add("info", "residue_file", format_id=spec.format_id)
                continue
            if "jsonl" in spec.encodings and path.suffix == ".jsonl":
                _check_jsonl(report, spec, path, rel, collected)
            elif "json" in spec.encodings and (path.suffix == ".json" or spec.encodings == ("json",)):
                _check_json(report, spec, path, rel, collected)
        memory_manifests = {
            d: p for d, p in collected["evidence_manifests"].items()
            if _rel(d, data_root).startswith("store/memory/")
        }
        _check_evidence(report, data_root, memory_manifests)
    _check_history(report, collected)
    report.sections["experience"] = collected["experience"]
    _check_weak_refs(report, collected)
    return report.to_dict()


def exit_code(report: dict[str, Any]) -> int:
    """0 = error なし、1 = error あり。"""
    return 1 if report["summary"]["errors"] else 0


# ── 不具合報告の束 ──


def _tail(path: Path, lines: int) -> list[str]:
    try:
        f = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return []
    with f:
        return list(deque(f, maxlen=lines))


def bundle_path(dest: Path) -> Path:
    """束の置き場 (ディレクトリならその中に ``evoref-doctor-<stamp>.zip``)。"""
    from backend.utils import utc_compact_stamp

    dest = Path(dest)
    if dest.is_dir():
        return dest / f"evoref-doctor-{utc_compact_stamp()}.zip"
    return dest


def write_bundle(
    report: dict[str, Any], data_root: Path, dest: Path, *, log_lines: int = BUNDLE_LOG_LINES,
) -> Path:
    """不具合報告の束 (zip) を書く。ストアの中身は入れず、ログは redact して末尾だけ。

    Raises:
        DoctorError: 置き場が ``store/`` の中。
    """
    from backend.structlog_config import redact_string

    target = bundle_path(dest).resolve()
    store_dir = (Path(data_root) / ledger_files.STORE_DIR).resolve()
    if target == store_dir or store_dir in target.parents:
        raise DoctorError(f"the bundle must be written outside {store_dir}")
    formats = {spec.format_id: spec.version for spec in FORMATS.all()}
    environment = {
        **{k: report.get(k) for k in ("generated_at", "app", "serve_running", "location", "seal", "g0")},
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.json", redact_string(json.dumps(report, ensure_ascii=False, indent=2)))
        zf.writestr("formats.json", json.dumps(formats, indent=2))
        zf.writestr("environment.json", redact_string(json.dumps(environment, ensure_ascii=False, indent=2)))
        for rel in BUNDLE_LOGS:
            buf = io.StringIO()
            for line in _tail(Path(data_root) / rel, log_lines):
                buf.write(redact_string(line))
            zf.writestr(f"{rel}.tail.txt", buf.getvalue())
    return target


__all__ = [
    "BUNDLE_LOGS",
    "REPORT_VERSION",
    "DoctorError",
    "Finding",
    "bundle_path",
    "exit_code",
    "run_doctor",
    "write_bundle",
]
