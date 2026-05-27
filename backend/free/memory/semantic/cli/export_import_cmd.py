"""``evorefmem_cli export`` / ``import`` 実装

``<memory_dir>/semantic/`` 全体を tar アーカイブにバックアップ / リストアする.

## 圧縮形式

- デフォルト: ``.tar.gz`` (stdlib ``tarfile`` のみ、追加依存なし)
- ``.tar.zst``: ``zstandard`` パッケージがインストール済みの場合のみ
  サポート (互換性最優先のため、現行のリポジトリ依存には追加しない)
- 拡張子から自動判定。``.tar`` も無圧縮で受け付ける。

## アーカイブ構造

アーカイブ内には ``semantic/`` ディレクトリと、トップレベルに
``_export_meta.json`` を入れる:

```text
my_export.tar.gz
├── _export_meta.json    # schema_version / created_at / source_memory_dir
└── semantic/
    ├── SCHEMA_VERSION
    ├── manifest.json
    ├── global/...
    ├── projects/<id>/...
    └── archive/...
```

## import の安全性

破壊的なため:
- デフォルトで dry-run (``--apply`` 必須)
- ``--apply`` 時は **既存 ``semantic/`` 全体を**
  ``migration_archive/cli_<utc_ts>/import/semantic/`` へ退避してから
  アーカイブ展開する (旧データは退避して残る)
- アーカイブ内 ``_export_meta.json`` の ``schema_version`` が現行
  ``SCHEMA_VERSION`` と一致するか検査。不一致時は ``--allow-version-mismatch``
  フラグなしで拒否する
"""

from __future__ import annotations

import json
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.free.memory.init_evorefmem import (
    SCHEMA_VERSION,
    read_schema_version,
)

EXPORT_META_FILENAME = "_export_meta.json"
"""tar アーカイブ内のメタファイル名 (トップレベル)。"""

SUPPORTED_GZIP_SUFFIXES = (".tar.gz", ".tgz")
SUPPORTED_ZSTD_SUFFIXES = (".tar.zst", ".tzst")
SUPPORTED_PLAIN_SUFFIXES = (".tar",)


@dataclass
class ExportReport:
    memory_dir: str
    archive_path: str
    archive_format: str
    """``"gzip"`` / ``"zstd"`` / ``"plain"``。"""

    file_count: int
    bytes_uncompressed: int
    bytes_archive: int
    schema_version: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class ImportReport:
    memory_dir: str
    archive_path: str
    archive_format: str
    file_count: int
    schema_version_in_archive: int | None
    schema_version_current: int | None
    backup_path: str | None
    applied: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ──────────────────────────────────────────────────────────────────────────
# 圧縮形式判定
# ──────────────────────────────────────────────────────────────────────────


def _detect_format(path: Path) -> str:
    """``path`` の拡張子から ``"gzip"`` / ``"zstd"`` / ``"plain"`` を返す.

    対応していない拡張子は :class:`ValueError`.
    """
    s = path.name.lower()
    for suf in SUPPORTED_GZIP_SUFFIXES:
        if s.endswith(suf):
            return "gzip"
    for suf in SUPPORTED_ZSTD_SUFFIXES:
        if s.endswith(suf):
            return "zstd"
    for suf in SUPPORTED_PLAIN_SUFFIXES:
        if s.endswith(suf):
            return "plain"
    raise ValueError(
        f"unsupported archive suffix: {path.name!r} "
        f"(supported: {SUPPORTED_GZIP_SUFFIXES + SUPPORTED_ZSTD_SUFFIXES + SUPPORTED_PLAIN_SUFFIXES})",
    )


def _open_zstd_for_write(path: Path):
    """zstandard が利用可能なら write 用 stream を返す. なければ ImportError."""
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            "zstandard package is not installed. Use .tar.gz suffix instead, "
            "or install zstandard via `pip install zstandard`.",
        ) from exc
    cctx = zstd.ZstdCompressor(level=10)
    return cctx.stream_writer(path.open("wb"))


def _open_zstd_for_read(path: Path):
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            "zstandard package is not installed; cannot open .tar.zst. "
            "Re-export as .tar.gz or install zstandard.",
        ) from exc
    dctx = zstd.ZstdDecompressor()
    return dctx.stream_reader(path.open("rb"))


# ──────────────────────────────────────────────────────────────────────────
# export
# ──────────────────────────────────────────────────────────────────────────


def run_export(
    memory_dir: Path,
    archive_path: Path,
    *,
    now: float | None = None,
) -> ExportReport:
    """``<memory_dir>/semantic/`` を tar アーカイブにエクスポートする.

    Args:
        memory_dir: ``local/memory/`` ルート。
        archive_path: 出力ファイルパス。拡張子 (``.tar.gz`` /
            ``.tar.zst`` / ``.tar``) で圧縮形式を選択。
        now: メタの ``created_at`` 上書き (テスト用)。

    Raises:
        FileNotFoundError: ``semantic/`` が存在しない。
        ValueError: 未対応の拡張子。
        ImportError: ``.tar.zst`` 指定だが zstandard が未インストール。
    """
    semantic_dir = Path(memory_dir) / "semantic"
    if not semantic_dir.exists():
        raise FileNotFoundError(
            f"semantic/ not found under {memory_dir}; nothing to export",
        )
    fmt = _detect_format(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    t = time.time() if now is None else now
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    schema_version = read_schema_version(memory_dir)
    if schema_version is None:
        schema_version = SCHEMA_VERSION  # marker missing → 現行値を記録
    meta = {
        "schema_version": schema_version,
        "created_at": created_at,
        "source_memory_dir": str(memory_dir),
        "format": fmt,
        "exporter": "evorefmem_cli",
        "exporter_version": "1",
    }
    meta_bytes = (
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    file_count = 0
    bytes_uncompressed = 0

    def _add_meta(tar: tarfile.TarFile) -> None:
        nonlocal file_count, bytes_uncompressed
        info = tarfile.TarInfo(name=EXPORT_META_FILENAME)
        info.size = len(meta_bytes)
        info.mtime = int(t)
        info.mode = 0o644
        import io

        tar.addfile(info, io.BytesIO(meta_bytes))
        file_count += 1
        bytes_uncompressed += len(meta_bytes)

    def _add_semantic(tar: tarfile.TarFile) -> None:
        nonlocal file_count, bytes_uncompressed
        for path in sorted(semantic_dir.rglob("*")):
            arcname = "semantic/" + str(
                path.relative_to(semantic_dir),
            ).replace("\\", "/")
            tar.add(path, arcname=arcname, recursive=False)
            if path.is_file():
                file_count += 1
                bytes_uncompressed += path.stat().st_size

    if fmt == "gzip":
        with tarfile.open(archive_path, "w:gz") as tar:
            _add_meta(tar)
            _add_semantic(tar)
    elif fmt == "plain":
        with tarfile.open(archive_path, "w") as tar:
            _add_meta(tar)
            _add_semantic(tar)
    elif fmt == "zstd":
        with _open_zstd_for_write(archive_path) as stream:
            with tarfile.open(fileobj=stream, mode="w|") as tar:
                _add_meta(tar)
                _add_semantic(tar)
    else:
        raise AssertionError(f"unhandled fmt: {fmt}")

    return ExportReport(
        memory_dir=str(memory_dir),
        archive_path=str(archive_path),
        archive_format=fmt,
        file_count=file_count,
        bytes_uncompressed=bytes_uncompressed,
        bytes_archive=archive_path.stat().st_size,
        schema_version=schema_version,
        created_at=created_at,
    )


# ──────────────────────────────────────────────────────────────────────────
# import
# ──────────────────────────────────────────────────────────────────────────


def _safe_extract_member(
    member: tarfile.TarInfo, dest_root: Path,
) -> Path | None:
    """tar メンバーを ``dest_root`` 配下に解決し、parent traversal を防ぐ.

    異常時は ``None`` を返す (extract 対象外)。
    """
    name = member.name.replace("\\", "/")
    if name.startswith("/") or ".." in Path(name).parts:
        return None
    out = (dest_root / name).resolve()
    try:
        out.relative_to(dest_root.resolve())
    except ValueError:
        return None
    return out


def _open_for_read(archive_path: Path, fmt: str) -> tarfile.TarFile:
    if fmt == "gzip":
        return tarfile.open(archive_path, "r:gz")
    if fmt == "plain":
        return tarfile.open(archive_path, "r")
    if fmt == "zstd":
        stream = _open_zstd_for_read(archive_path)
        return tarfile.open(fileobj=stream, mode="r|")
    raise AssertionError(f"unhandled fmt: {fmt}")


def _read_meta(archive_path: Path, fmt: str) -> dict[str, Any] | None:
    """アーカイブから ``_export_meta.json`` のみを読み出す (副作用なし)."""
    with _open_for_read(archive_path, fmt) as tar:
        for member in tar:
            if member.name in (EXPORT_META_FILENAME, f"./{EXPORT_META_FILENAME}"):
                f = tar.extractfile(member)
                if f is None:
                    return None
                try:
                    return json.loads(f.read().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return None
    return None


def _count_archive(archive_path: Path, fmt: str) -> int:
    n = 0
    with _open_for_read(archive_path, fmt) as tar:
        for _ in tar:
            n += 1
    return n


def run_import(
    memory_dir: Path,
    archive_path: Path,
    migration_archive_dir: Path,
    *,
    apply: bool = False,
    allow_version_mismatch: bool = False,
    now: float | None = None,
) -> ImportReport:
    """tar アーカイブから ``<memory_dir>/semantic/`` を復元する.

    Args:
        memory_dir: ``local/memory/`` ルート。
        archive_path: 入力アーカイブ (``.tar.gz`` / ``.tar.zst`` / ``.tar``)。
        migration_archive_dir: 既存 ``semantic/`` の退避先。
        apply: True で実書換、False で dry-run。
        allow_version_mismatch: archive 内 schema_version != 現行 SCHEMA_VERSION
            を許容するか。デフォルト不可。
    """
    fmt = _detect_format(archive_path)
    file_count = _count_archive(archive_path, fmt)
    meta = _read_meta(archive_path, fmt)
    archive_schema = (
        int(meta["schema_version"])
        if meta and isinstance(meta.get("schema_version"), int)
        else None
    )
    rep = ImportReport(
        memory_dir=str(memory_dir),
        archive_path=str(archive_path),
        archive_format=fmt,
        file_count=file_count,
        schema_version_in_archive=archive_schema,
        schema_version_current=read_schema_version(memory_dir),
        backup_path=None,
        applied=apply,
    )
    if archive_schema is None:
        rep.error = (
            f"{EXPORT_META_FILENAME} missing or malformed in archive; "
            "this archive may not have been produced by evorefmem_cli export"
        )
        return rep
    if archive_schema != SCHEMA_VERSION and not allow_version_mismatch:
        rep.error = (
            f"archive schema_version={archive_schema} differs from current "
            f"SCHEMA_VERSION={SCHEMA_VERSION}; pass --allow-version-mismatch "
            "to override"
        )
        return rep
    if not apply:
        return rep

    # apply path
    semantic_dir = Path(memory_dir) / "semantic"
    if semantic_dir.exists():
        from backend.free.memory.semantic.cli._paths import cli_backup_root

        backup_root = cli_backup_root(migration_archive_dir, "import", now=now)
        backup_dest = backup_root / "semantic"
        shutil.copytree(semantic_dir, backup_dest)
        rep.backup_path = str(backup_dest)
        shutil.rmtree(semantic_dir)
    semantic_dir.mkdir(parents=True, exist_ok=True)

    # 抽出
    with _open_for_read(archive_path, fmt) as tar:
        for member in tar:
            target = _safe_extract_member(member, memory_dir)
            if target is None:
                continue
            if member.name in (EXPORT_META_FILENAME, f"./{EXPORT_META_FILENAME}"):
                # メタは memory_dir/_export_meta.json として残さず捨てる
                continue
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            f = tar.extractfile(member)
            if f is None:
                continue
            with target.open("wb") as out:
                shutil.copyfileobj(f, out)
    return rep


def format_export_report_text(report: ExportReport) -> str:
    return (
        f"archive_path     : {report.archive_path}\n"
        f"format           : {report.archive_format}\n"
        f"schema_version   : {report.schema_version}\n"
        f"created_at       : {report.created_at}\n"
        f"files            : {report.file_count}\n"
        f"bytes (raw/arc)  : {report.bytes_uncompressed} / {report.bytes_archive}"
    )


def format_import_report_text(report: ImportReport) -> str:
    lines: list[str] = []
    lines.append(f"archive_path     : {report.archive_path}")
    lines.append(f"format           : {report.archive_format}")
    lines.append(f"files            : {report.file_count}")
    lines.append(
        f"schema_version   : archive={report.schema_version_in_archive} "
        f"current={report.schema_version_current}",
    )
    mode = "apply" if report.applied else "dry-run"
    lines.append(f"mode             : {mode}")
    if report.backup_path:
        lines.append(f"backup           : {report.backup_path}")
    if report.error:
        lines.append("")
        lines.append(f"ERROR: {report.error}")
    elif not report.applied:
        lines.append("")
        lines.append("(dry-run; rerun with --apply to restore)")
    return "\n".join(lines)


__all__ = [
    "EXPORT_META_FILENAME",
    "ExportReport",
    "ImportReport",
    "format_export_report_text",
    "format_import_report_text",
    "run_export",
    "run_import",
]
