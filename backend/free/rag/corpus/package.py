"""`.evocart` パッケージ形式 (c_16 §4.3)

**パッケージは配布単位、corpus は実行時ストア** の二層のうち、配布単位側。
zip の中身は次の形に固定する:

```
<id>-<version>.evocart (zip)
├── package.json      # PackageMeta
├── docs/             # 原本 (md / txt / 変換済み text)。これが真実
└── prebuilt/         # 任意 (作成側が現在の埋め込みモデルで作っておく)
    ├── build.json    # {chunker_version, embedding_model_id, embedding_dim}
    ├── chunks.jsonl  # doc_chunk の Evidence
    └── embeddings/<model_id>/   # VectorStore 形式
```

`docs/` が真実で、`prebuilt/` は再現可能な派生物にすぎない。``content_digest``
(docs の sha256) と ``chunker_version`` があれば同じチャンクを決定論で再生成
できるため、旧カートリッジの ``needs_rebuild`` / ``docs_digest`` は持たない
(c_16 §8)。

## zip を扱うときの規約

- **展開先の外へ書かせない** (zip slip)。エントリ名に ``..`` / 絶対パス /
  ドライブレターが混ざっていれば :class:`PackageError` で弾く。
- ``package.json`` は :class:`~backend.io.AtomicWriter` 経由で書く。docs の
  本体はレコードではない (壊れても形式で気付ける) のでそのまま書く。
- 未知キーは捨てずに :attr:`PackageMeta._extra` へ退避し、書き戻しで復元する
  (c_05 §0.5)。
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any

from backend.io import AtomicWriter, atomic_write_text
from backend.log_config import get_logger

logger = get_logger("rag.corpus.package")

#: パッケージレコードの版。意味を変える変更で上げ、migrator を用意する。
PACKAGE_SCHEMA_VERSION = 1

#: 配布ファイルの拡張子。
PACKAGE_SUFFIX = ".evocart"

PACKAGE_FILE = "package.json"
DOCS_DIR = "docs"
PREBUILT_DIR = "prebuilt"
PREBUILT_BUILD_FILE = "build.json"
PREBUILT_CHUNKS_FILE = "chunks.jsonl"
PREBUILT_EMBEDDINGS_DIR = "embeddings"

#: パッケージ id。ディレクトリ名・シャード名・検索結果の接頭辞に素で入るので
#: 小文字英数 + ``_`` / ``-`` だけに絞る (2〜64 文字)。
PACKAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")

#: semver 2.0.0 (prerelease / build metadata まで許す)。
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$",
)


class PackageError(ValueError):
    """`.evocart` として読めない / 書けない (形式違反・不正な id・zip slip)。"""


# ── メタデータ ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class PackageMeta:
    """``package.json`` の中身 (c_16 §4.3)。

    ``priority`` は無い — 順位式の ``store_prior`` を PC 固有に上書きしたい
    場合は corpus 側の ``manifest.store_prior_overrides`` を使う (c_16 §4.3)。
    """

    id: str
    name: str
    version: str = "1.0.0"
    author: str = ""
    license: str = ""
    language: str = "ja"
    tags: list[str] = field(default_factory=list)
    description: str = ""
    tool_hints: list[dict[str, Any]] = field(default_factory=list)
    compatibility: str = ">=0.1.0"
    #: ``docs/`` の内容ダイジェスト (:func:`compute_content_digest`)。
    content_digest: str = ""
    schema_version: int = PACKAGE_SCHEMA_VERSION
    #: 未知キーの退避先 (前方互換)。
    _extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """JSON レコードにする。

        キーは :func:`dataclasses.fields` から機械生成する。**手書きで列挙
        しないこと** — フィールドを足したときに書き漏れて値が消える
        (c_05 §0.5)。``_extra`` は展開して戻す。
        """
        record = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "_extra"
        }
        for key, value in self._extra.items():
            record.setdefault(key, value)
        return record

    def to_json(self) -> str:
        return json.dumps(self.to_record(), ensure_ascii=False, indent=2)


def validate_package_id(package_id: str) -> str:
    """パッケージ id を検査して返す (違反は :class:`PackageError`)。"""
    if not isinstance(package_id, str) or not PACKAGE_ID_RE.match(package_id):
        raise PackageError(
            f"invalid package id: {package_id!r} "
            f"(expected {PACKAGE_ID_RE.pattern})",
        )
    return package_id


def validate_version(version: str) -> str:
    """semver を検査して返す (違反は :class:`PackageError`)。"""
    if not isinstance(version, str) or not SEMVER_RE.match(version):
        raise PackageError(f"invalid package version: {version!r} (expected semver)")
    return version


def meta_from_record(data: Any) -> PackageMeta:
    """``package.json`` の dict から :class:`PackageMeta` を復元する。

    未対応の新しい ``schema_version`` は読まない (c_05 §0.5.1) — 旧版で読んで
    書き戻すと新版のフィールドを落として壊すため。
    """
    if not isinstance(data, dict):
        raise PackageError("package.json must be a JSON object")

    schema_version = int(data.get("schema_version") or PACKAGE_SCHEMA_VERSION)
    if schema_version > PACKAGE_SCHEMA_VERSION:
        raise PackageError(
            f"package.json schema_version {schema_version} is newer than "
            f"supported {PACKAGE_SCHEMA_VERSION}",
        )

    package_id = validate_package_id(str(data.get("id") or ""))
    version = validate_version(str(data.get("version") or "1.0.0"))

    known = {f.name for f in fields(PackageMeta)}
    extra = dict(data.get("_extra") or {})
    for key, value in data.items():
        if key not in known:
            extra[key] = value

    tags = data.get("tags") or []
    if not isinstance(tags, list):
        raise PackageError("package.json tags must be a list")
    tool_hints = data.get("tool_hints") or []
    if not isinstance(tool_hints, list):
        raise PackageError("package.json tool_hints must be a list")

    return PackageMeta(
        id=package_id,
        name=str(data.get("name") or package_id),
        version=version,
        author=str(data.get("author") or ""),
        license=str(data.get("license") or ""),
        language=str(data.get("language") or "ja"),
        tags=[str(t) for t in tags],
        description=str(data.get("description") or ""),
        tool_hints=[dict(h) for h in tool_hints if isinstance(h, dict)],
        compatibility=str(data.get("compatibility") or ">=0.1.0"),
        content_digest=str(data.get("content_digest") or ""),
        schema_version=schema_version,
        _extra=extra,
    )


@dataclass(slots=True, frozen=True)
class PrebuiltInfo:
    """``prebuilt/build.json`` (c_16 §4.3)。"""

    chunker_version: int
    embedding_model_id: str
    embedding_dim: int

    def to_record(self) -> dict[str, Any]:
        return {
            "chunker_version": int(self.chunker_version),
            "embedding_model_id": self.embedding_model_id,
            "embedding_dim": int(self.embedding_dim),
        }


def prebuilt_from_record(data: Any) -> PrebuiltInfo | None:
    """``build.json`` の dict から :class:`PrebuiltInfo` を作る。

    読めない / 欠けている場合は ``None`` を返す — prebuilt は任意の派生物
    なので、壊れていても ``docs/`` から作り直せばよい。
    """
    if not isinstance(data, dict):
        return None
    try:
        return PrebuiltInfo(
            chunker_version=int(data.get("chunker_version") or 0),
            embedding_model_id=str(data.get("embedding_model_id") or ""),
            embedding_dim=int(data.get("embedding_dim") or 0),
        )
    except (TypeError, ValueError) as e:
        logger.warning("unreadable prebuilt build.json: %s", e)
        return None


# ── content_digest ──────────────────────────────────────────────────────


def compute_content_digest(docs_dir: Path | str) -> str:
    """``docs/`` の内容ダイジェスト (相対パス + 本文の sha256)。

    旧 ``compute_docs_digest`` は「相対パス + サイズ + mtime」だった。mtime は
    zip 展開・コピー・チェックアウトのたびに変わるので、**中身が同じでも別物
    と判定される**。ここでは本文そのものを食わせ、同じ ``docs/`` なら別 PC で
    展開しても必ず同じ値になるようにする。チャンクの evidence id はこの値から
    導出するので (``chunking.py``)、再インストールで id が変わらない。

    ディレクトリが無い / 空なら空文字 (「未計算」と「空」は区別しない)。
    """
    root = Path(docs_dir)
    if not root.is_dir():
        return ""
    digest = hashlib.sha256()
    counted = 0
    for path in sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    ):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        try:
            digest.update(path.read_bytes())
        except OSError as e:
            logger.warning("unreadable document %s: %s", path, e)
            continue
        digest.update(b"\x00")
        counted += 1
    return digest.hexdigest() if counted else ""


# ── 読み込み ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PackageContents:
    """展開済みパッケージ (:func:`read_package` の戻り)。"""

    meta: PackageMeta
    #: 展開先ルート (``extract_to`` を渡さなかった場合は ``None``)。
    root: Path | None
    #: ``docs/`` 配下の相対パス (posix、昇順)。
    doc_names: list[str]
    #: ``prebuilt/build.json`` (無ければ ``None``)。
    prebuilt: PrebuiltInfo | None = None

    @property
    def docs_dir(self) -> Path | None:
        return None if self.root is None else self.root / DOCS_DIR

    @property
    def prebuilt_dir(self) -> Path | None:
        if self.root is None:
            return None
        directory = self.root / PREBUILT_DIR
        return directory if directory.is_dir() else None


def _safe_member_path(name: str) -> PurePosixPath | None:
    """zip エントリ名を展開先相対の安全なパスへ正規化する。

    ``..`` / 絶対パス / ドライブレター混じりは ``None`` (呼出側が弾く)。
    ディレクトリエントリも ``None``。
    """
    normalized = name.replace("\\", "/")
    if not normalized or normalized.endswith("/"):
        return None
    if normalized.startswith("/") or ":" in normalized.split("/")[0]:
        return None
    parts = [p for p in PurePosixPath(normalized).parts if p not in (".", "")]
    if any(p == ".." for p in parts):
        return None
    return PurePosixPath(*parts) if parts else None


def _strip_single_root(members: list[PurePosixPath]) -> list[PurePosixPath]:
    """zip 全体が 1 つのディレクトリで包まれていれば剥がす。

    作成側が ``<id>/package.json`` の形で固めることがある (旧カートリッジの
    ``CartridgeCreator`` がそう)。1 段だけ剥がす。2 段以上は剥がさない —
    意図した階層を潰すと docs のパスが壊れる。

    戻り値は **入力と同じ長さ・同じ並び** (呼出側が zip 内の元の名前と位置で
    突き合わせるため)。剥がすと消える要素が出る形 (トップレベルにファイルが
    ある) では、そもそも剥がさない。
    """
    if any(len(m.parts) == 1 for m in members):
        return members
    roots = {m.parts[0] for m in members}
    if len(roots) != 1:
        return members
    return [PurePosixPath(*m.parts[1:]) for m in members]


def read_package(
    zip_path: Path | str, extract_to: Path | str | None = None,
) -> PackageContents:
    """`.evocart` を読む (``extract_to`` 指定時は展開もする)。

    Args:
        zip_path: パッケージ zip。
        extract_to: 展開先。``None`` ならメタデータと目録だけ返す。

    Raises:
        PackageError: zip でない / ``package.json`` が無い / id・版が不正 /
            zip slip を検出した。
    """
    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(f"package not found: {path}")
    if not zipfile.is_zipfile(str(path)):
        raise PackageError(f"not a zip archive: {path}")

    destination = Path(extract_to) if extract_to is not None else None

    with zipfile.ZipFile(str(path), "r") as zf:
        members: list[PurePosixPath] = []
        source_names: list[str] = []
        for raw_name in zf.namelist():
            member = _safe_member_path(raw_name)
            if member is None:
                if raw_name.endswith("/"):
                    continue
                raise PackageError(f"unsafe entry in package: {raw_name!r}")
            members.append(member)
            source_names.append(raw_name)
        #: 展開先相対のパス → zip 内の元の名前。
        remap: dict[PurePosixPath, str] = dict(
            zip(_strip_single_root(members), source_names),
        )

        meta_key = PurePosixPath(PACKAGE_FILE)
        if meta_key not in remap:
            raise PackageError(f"{PACKAGE_FILE} not found in package")
        meta = meta_from_record(json.loads(zf.read(remap[meta_key])))

        doc_names = sorted(
            str(PurePosixPath(*m.parts[1:]))
            for m in remap
            if len(m.parts) > 1 and m.parts[0] == DOCS_DIR
        )
        prebuilt_key = PurePosixPath(PREBUILT_DIR, PREBUILT_BUILD_FILE)
        prebuilt = None
        if prebuilt_key in remap:
            try:
                prebuilt = prebuilt_from_record(json.loads(zf.read(remap[prebuilt_key])))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("unreadable %s: %s", prebuilt_key, e)

        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            for member, source_name in sorted(remap.items()):
                target = destination / Path(*member.parts)
                # 念のための二重確認: 正規化後に展開先の外を指していないか。
                resolved = target.resolve()
                if not str(resolved).startswith(str(destination.resolve())):
                    raise PackageError(f"unsafe entry in package: {source_name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(source_name))

    logger.info(
        "Read package %s v%s: %d doc(s), prebuilt=%s",
        meta.id, meta.version, len(doc_names),
        "yes" if prebuilt is not None else "no",
    )
    return PackageContents(
        meta=meta, root=destination, doc_names=doc_names, prebuilt=prebuilt,
    )


def iter_prebuilt_chunks(prebuilt_dir: Path | str) -> Iterator[dict[str, Any]]:
    """``prebuilt/chunks.jsonl`` を 1 行ずつ生 dict で返す。

    壊れた行はそのレコードだけ飛ばして件数を WARNING に出す (c_05 §0.5.2)。
    """
    path = Path(prebuilt_dir) / PREBUILT_CHUNKS_FILE
    if not path.exists():
        return
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(record, dict):
                yield record
            else:
                skipped += 1
    if skipped:
        logger.warning("%s: skipped %d unreadable chunk line(s)", path, skipped)


# ── 書き出し ────────────────────────────────────────────────────────────


def write_package_meta(directory: Path | str, meta: PackageMeta) -> Path:
    """``package.json`` を原子的に書く (壊れると版が読めないので ``fsync``)。"""
    path = Path(directory) / PACKAGE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, meta.to_json(), fsync=True)
    return path


def write_prebuilt_build(directory: Path | str, info: PrebuiltInfo) -> Path:
    """``prebuilt/build.json`` を原子的に書く。"""
    path = Path(directory) / PREBUILT_BUILD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path, json.dumps(info.to_record(), ensure_ascii=False, indent=2),
    )
    return path


def write_prebuilt_chunks(directory: Path | str, records: list[dict[str, Any]]) -> Path:
    """``prebuilt/chunks.jsonl`` を原子的に書く。"""
    path = Path(directory) / PREBUILT_CHUNKS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    atomic_write_text(path, body)
    return path


def _staging_from_mapping(
    docs: Mapping[str, str | bytes], staging: Path,
) -> None:
    """``{相対名: 本文}`` を ``docs/`` として書き出す。"""
    docs_dir = staging / DOCS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, content in docs.items():
        member = _safe_member_path(name)
        if member is None:
            raise PackageError(f"unsafe document name: {name!r}")
        target = docs_dir / Path(*member.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")


def write_package(
    source: Path | str | Mapping[str, str | bytes],
    zip_path: Path | str,
    meta: PackageMeta | None = None,
) -> PackageMeta:
    """`.evocart` を書き出し、確定した :class:`PackageMeta` を返す。

    Args:
        source: パッケージのディレクトリ (``package.json`` + ``docs/`` +
            任意の ``prebuilt/``)、または ``{docs 内の相対名: 本文}`` の写像。
            写像を渡す場合は ``meta`` が必須。
        zip_path: 出力先。
        meta: ``source`` がディレクトリなら上書き用 (``None`` で
            ``package.json`` をそのまま使う)。写像なら必須。

    ``content_digest`` は書き出す直前に ``docs/`` から**必ず計算し直す**。
    呼出側が持ち回った値を信じると、docs を差し替えたのに digest が古いまま
    という状態が配布物に載る (再インストールで別内容が同じ id になる)。
    """
    output = Path(zip_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(source, Mapping):
        if meta is None:
            raise PackageError("meta is required when source is a mapping")
        import tempfile

        with tempfile.TemporaryDirectory(prefix="evocart-") as tmp:
            staging = Path(tmp)
            _staging_from_mapping(source, staging)
            return write_package(staging, output, meta)

    root = Path(source)
    if not root.is_dir():
        raise PackageError(f"package source directory not found: {root}")

    if meta is None:
        meta_path = root / PACKAGE_FILE
        if not meta_path.exists():
            raise PackageError(f"{PACKAGE_FILE} not found under {root}")
        meta = meta_from_record(json.loads(meta_path.read_text(encoding="utf-8")))
    validate_package_id(meta.id)
    validate_version(meta.version)

    docs_dir = root / DOCS_DIR
    if not docs_dir.is_dir() or not any(p.is_file() for p in docs_dir.rglob("*")):
        raise PackageError(f"package has no documents under {docs_dir}")

    from dataclasses import replace

    meta = replace(meta, content_digest=compute_content_digest(docs_dir))
    write_package_meta(root, meta)

    with AtomicWriter(output, mode="wb") as raw:
        with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(PACKAGE_FILE, meta.to_json())
            for path in sorted(
                (p for p in root.rglob("*") if p.is_file()),
                key=lambda p: p.relative_to(root).as_posix(),
            ):
                rel = path.relative_to(root).as_posix()
                if rel == PACKAGE_FILE:
                    continue
                zf.write(str(path), rel)

    logger.info(
        "Wrote package %s to %s (%d bytes)",
        meta.id, output, output.stat().st_size if output.exists() else 0,
    )
    return meta


def package_filename(meta: PackageMeta) -> str:
    """配布ファイル名 (``<id>-<version>.evocart``)。"""
    return f"{meta.id}-{meta.version}{PACKAGE_SUFFIX}"


__all__ = [
    "DOCS_DIR",
    "PACKAGE_FILE",
    "PACKAGE_ID_RE",
    "PACKAGE_SCHEMA_VERSION",
    "PACKAGE_SUFFIX",
    "PREBUILT_BUILD_FILE",
    "PREBUILT_CHUNKS_FILE",
    "PREBUILT_DIR",
    "PREBUILT_EMBEDDINGS_DIR",
    "SEMVER_RE",
    "PackageContents",
    "PackageError",
    "PackageMeta",
    "PrebuiltInfo",
    "compute_content_digest",
    "iter_prebuilt_chunks",
    "meta_from_record",
    "package_filename",
    "prebuilt_from_record",
    "read_package",
    "validate_package_id",
    "validate_version",
    "write_package",
    "write_package_meta",
    "write_prebuilt_build",
    "write_prebuilt_chunks",
]
