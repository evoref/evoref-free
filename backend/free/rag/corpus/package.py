"""`.evocart` パッケージ形式 (c_16 §4.3)

**パッケージは配布単位、corpus は実行時ストア** の二層のうち、配布単位側。
zip の中身は次の形に固定する:

```
<id>-<version>.evocart (zip)
├── package.json      # PackageMeta {id, name, version, kind, provides, requires,
│                     #  content_digest, section_digests, ...}
├── docs/             # 任意。原本 (md / txt / 変換済み text)。これが真実。検索に載るのはここだけ
├── prebuilt/         # 任意 (docs/ の派生物。作成側が現在の埋め込みモデルで作っておく)
│   ├── build.json    # {chunker_version, embedding_model_id, embedding_dim}
│   ├── chunks.jsonl  # doc_chunk の Evidence
│   └── embeddings/<model_id>/   # VectorStore 形式
├── templates/        # 任意。文書テンプレート (c_16 §4.5.2)。索引に載せない
└── language/         # 任意。言語パック (c_16 §4.5.3)。索引に載せない
```

`docs/` が真実で、`prebuilt/` は再現可能な派生物にすぎない。``content_digest``
(docs の sha256) と ``chunker_version`` があれば同じチャンクを決定論で再生成
できるため、旧カートリッジの ``needs_rebuild`` / ``docs_digest`` は持たない
(c_16 §8)。``docs/`` は任意 — 1 枚のパッケージは ``docs`` / ``templates`` /
``language`` のセクションを 1 つ以上持てばよく (``provides`` が宣言する)、
``docs/`` を持たないパッケージは索引を作らない (c_16 §4.3 / §4.5)。

## zip を扱うときの規約

- **展開先の外へ書かせない** (zip slip)。エントリ名に ``..`` / 絶対パス /
  ドライブレターが混ざっていれば :class:`PackageError` で弾く。
- ``package.json`` は G1 の封筒 (``evoref.package`` v1、c_16 §4.3 / c_05 §0.4.10) に
  包んで原子的に書く。封筒の無い ``package.json`` は G0 として案内付きで、新しい版は
  それと分かる案内付きで拒否する (:class:`PackageFormatError`)。docs の本体は
  レコードではない (壊れても形式で気付ける) のでそのまま書く。
- 未知キーは捨てずに :attr:`PackageMeta._extra` へ退避し、書き戻しで復元する
  (c_05 §0.5)。
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any

from backend.io.safe_extract import (
    ExtractBudget,
    ExtractLimits,
    UnsafeArchiveError,
    check_zip_members,
    normalize_member_name,
    resolve_under,
)
from backend.io import AtomicWriter, atomic_write_text, jsoncodec
from backend.io.versioned import (
    ReadResult,
    build_envelope,
    read_versioned,
    read_versioned_bytes,
    write_versioned,
)
from backend.log_config import get_logger

logger = get_logger("rag.corpus.package")

#: ``package.json`` の封筒 (c_16 §4.3)。配布物 (``.evocart``) の中でもインストール先の
#: 版ディレクトリでも同じ形。台帳の宣言は ``corpus.store`` (インストール先の置き場)。
PACKAGE_FORMAT_ID = "evoref.package"
PACKAGE_FORMAT_VERSION = 1

#: 配布ファイルの拡張子。
PACKAGE_SUFFIX = ".evocart"

PACKAGE_FILE = "package.json"
DOCS_DIR = "docs"
PREBUILT_DIR = "prebuilt"
PREBUILT_BUILD_FILE = "build.json"
PREBUILT_CHUNKS_FILE = "chunks.jsonl"
PREBUILT_EMBEDDINGS_DIR = "embeddings"
#: 検索に載らないセクション (c_16 §4.5)。索引 (snapshot / 埋め込み /
#: centroid / 疑似クエリ) は作らず、指名して使う資材として読む。
TEMPLATES_DIR = "templates"
LANGUAGE_DIR = "language"
#: ``provides`` に載る既知のセクション名 (この 3 つ + 将来のセクション)。
SECTION_DIRS = (DOCS_DIR, TEMPLATES_DIR, LANGUAGE_DIR)

#: ``kind`` の既知値 (c_16 §4.3)。``package`` = 配布物 (既定)、
#: ``project_map`` = 機械生成 (c_16 §4.4)。
PACKAGE_KIND_DEFAULT = "package"
PACKAGE_KINDS = frozenset({PACKAGE_KIND_DEFAULT, "project_map"})

#: ``requires`` の既知の機能フラグ (c_16 §4.3)。``templates/1`` は段階 B-0 で
#: 追加 (c_16 §4.5.6)。``language/1`` は段階 C-1 で追加。``language.verify/1``
#: は段階 C-3 で追加 — install を通すだけで、実行を有効にするのは
#: ``create.staged.verify.enabled`` (既定 OFF、c_16 §4.5.4)。このフラグを
#: requires していないパッケージの ``verify`` 宣言は構造検証は受けるが
#: 実行時は全部無視される (フラグを宣言せずに verify を運ばせない)。
KNOWN_FEATURES: frozenset[str] = frozenset({
    "templates/1", "language/1", "language.verify/1",
})

#: ``language.verify/1`` を ``requires`` に持つパッケージだけが、宣言した
#: verify コマンドを実行時に有効化できる (c_16 §4.5.4)。
LANGUAGE_VERIFY_FEATURE = "language.verify/1"
#: ``edition/pro`` だけは固定集合に置かず動的に評価する (``backend.edition.is_pro()``)。
EDITION_PRO_FEATURE = "edition/pro"

#: パッケージ id。ディレクトリ名・シャード名・検索結果の接頭辞に素で入るので
#: 小文字英数 + ``_`` / ``-`` だけに絞る (2〜40 文字)。長さは Windows の MAX_PATH
#: の予算 (c_05 §0.7.2 R11: インストール根 60 文字 + 組み立てパス ≤ 240) で決まる。
PACKAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$")
#: 版文字列の最大長 (版ディレクトリ名になる。R11 と同じ予算)。
MAX_VERSION_LENGTH = 24

#: semver 2.0.0 (prerelease / build metadata まで許す)。
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$",
)


class PackageError(ValueError):
    """`.evocart` として読めない / 書けない (形式違反・不正な id・zip slip)。"""


class PackageFormatError(PackageError):
    """``package.json`` / セクションの ``manifest.json`` が G1 の封筒でない (G0) / 新しい版。

    メッセージ (ログ用、英語) とは別に、利用者への案内の i18n キーを持つ。
    """

    def __init__(self, message: str, i18n_key: str, **context: Any) -> None:
        super().__init__(message)
        self.i18n_key = i18n_key
        self.context = context


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
    #: 所有と生まれ方だけを表す (c_16 §4.3)。``package`` (配布物) /
    #: ``project_map`` (機械生成、c_16 §4.4)。中身の種類は :attr:`provides` が表す。
    kind: str = PACKAGE_KIND_DEFAULT
    #: 運ぶセクション名の列 (``docs`` / ``templates`` / ``language``)。宣言が
    #: 無ければ install 時にディレクトリの実在から導出する。
    provides: list[str] = field(default_factory=list)
    #: 機能フラグの列 (``"templates/1"`` / ``"edition/pro"`` 等)。install は
    #: 知らないフラグを拒否する (:data:`KNOWN_FEATURES`)。
    requires: list[str] = field(default_factory=list)
    author: str = ""
    license: str = ""
    language: str = "ja"
    tags: list[str] = field(default_factory=list)
    description: str = ""
    tool_hints: list[dict[str, Any]] = field(default_factory=list)
    compatibility: str = ">=0.1.0"
    #: ``docs/`` の内容ダイジェスト (:func:`compute_content_digest`)。
    content_digest: str = ""
    #: ``docs/`` / ``prebuilt/`` 以外のセクションディレクトリのダイジェスト
    #: (:func:`compute_section_digests`)。``{section: sha256}``。
    section_digests: dict[str, str] = field(default_factory=dict)
    #: 未知キーの退避先 (前方互換)。
    _extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """JSON レコードにする。

        キーは :func:`dataclasses.fields` から機械生成する。**手書きで列挙
        しないこと** — フィールドを足したときに書き漏れて値が消える
        (c_05 §0.5)。``_extra`` は展開して戻す。

        第一級フィールドが常に勝つ。``_extra`` に同名のキーがあっても書き出さない
        — 検証済みの値 (``kind`` / ``requires`` / ``content_digest`` …) を、検証を
        通っていない ``_extra`` の値で上書きさせないため。読み取り側
        (:func:`meta_from_record`) も既知名を ``_extra`` に残さない。
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
        """``package.json`` の本文 (G1 の封筒、手で読めるよう字下げ)。"""
        envelope = build_envelope(
            format_id=PACKAGE_FORMAT_ID, format_version=PACKAGE_FORMAT_VERSION,
            payload=self.to_record(), component="corpus.package",
        )
        return jsoncodec.dumps(envelope, indent=2)


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
    if len(version) > MAX_VERSION_LENGTH:
        raise PackageError(
            f"package version {version!r} is longer than {MAX_VERSION_LENGTH} characters",
        )
    return version


def _meta_from_read(result: ReadResult, source: str) -> PackageMeta:
    """封筒の分類から :class:`PackageMeta` を作る (読めなければ :class:`PackageError`)。"""
    if result.ok:
        return meta_from_record(result.payload)
    if result.status == "foreign" and result.detail == "not a G1 envelope":
        raise PackageFormatError(
            f"{source} has no format_id (a G0 package)", "api.cartridge_g0_package",
        )
    if result.status == "newer":
        raise PackageFormatError(
            f"{source} format_version {result.version} is newer than {PACKAGE_FORMAT_VERSION}",
            "api.cartridge_newer_package", version=result.version,
        )
    raise PackageError(f"{source} is unreadable ({result.status}: {result.detail})")


def meta_from_json(data: bytes | str, source: str = PACKAGE_FILE) -> PackageMeta:
    """``package.json`` の本文 (G1 の封筒) から :class:`PackageMeta` を復元する。

    Raises:
        PackageFormatError: 封筒が無い (G0) / 新しい版。
        PackageError: 読めない / 別の形式 / 中身が不正。
    """
    raw = data.encode("utf-8") if isinstance(data, str) else data
    result = read_versioned_bytes(
        raw, format_id=PACKAGE_FORMAT_ID, format_version=PACKAGE_FORMAT_VERSION,
    )
    return _meta_from_read(result, source)


def read_package_meta(directory: Path | str) -> PackageMeta:
    """版ディレクトリ / 展開先の ``package.json`` を読む (:func:`meta_from_json` と同じ規則)。"""
    path = Path(directory) / PACKAGE_FILE
    result = read_versioned(
        path, format_id=PACKAGE_FORMAT_ID, format_version=PACKAGE_FORMAT_VERSION,
    )
    if result.status == "absent":
        raise PackageError(f"{PACKAGE_FILE} not found under {directory}")
    return _meta_from_read(result, str(path))


def meta_from_record(data: Any) -> PackageMeta:
    """``package.json`` のペイロード (dict) から :class:`PackageMeta` を復元する。"""
    if not isinstance(data, dict):
        raise PackageError("package.json must be a JSON object")

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

    # 既知のフィールド名は `_extra` に残さない (to_record は第一級フィールドを
    # 優先するので、残しても書き出されず黙って消える)。
    for name in known:
        extra.pop(name, None)
    kind = data.get("kind")
    kind = str(kind) if kind else PACKAGE_KIND_DEFAULT
    if kind not in PACKAGE_KINDS:
        logger.warning(
            "package.json %s declares unknown kind %r; treating as %r",
            package_id, kind, PACKAGE_KIND_DEFAULT,
        )
        kind = PACKAGE_KIND_DEFAULT

    provides = data.get("provides")
    if provides is None:
        provides = []
    elif not isinstance(provides, list):
        raise PackageError("package.json provides must be a list")

    requires = data.get("requires") or []
    if not isinstance(requires, list):
        raise PackageError("package.json requires must be a list")

    section_digests = data.get("section_digests") or {}
    if not isinstance(section_digests, dict):
        raise PackageError("package.json section_digests must be an object")

    return PackageMeta(
        id=package_id,
        name=str(data.get("name") or package_id),
        version=version,
        kind=kind,
        provides=[str(p) for p in provides],
        requires=[str(r) for r in requires],
        author=str(data.get("author") or ""),
        license=str(data.get("license") or ""),
        language=str(data.get("language") or "ja"),
        tags=[str(t) for t in tags],
        description=str(data.get("description") or ""),
        tool_hints=[dict(h) for h in tool_hints if isinstance(h, dict)],
        compatibility=str(data.get("compatibility") or ">=0.1.0"),
        content_digest=str(data.get("content_digest") or ""),
        section_digests={str(k): str(v) for k, v in section_digests.items()},
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


def _dir_content_digest(root: Path) -> str:
    """ディレクトリの内容ダイジェスト (相対パス + 本文の sha256)。

    :func:`compute_content_digest` (``docs/``) と :func:`compute_section_digests`
    (その他のセクション) が共有する方式。mtime は zip 展開・コピー・
    チェックアウトのたびに変わるので、**中身が同じでも別物と判定される**
    旧方式 (相対パス + サイズ + mtime) は使わない。本文そのものを食わせ、
    同じディレクトリなら別 PC で展開しても必ず同じ値になるようにする。

    ディレクトリが無い / 空なら空文字 (「未計算」と「空」は区別しない)。
    """
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


def compute_content_digest(docs_dir: Path | str) -> str:
    """``docs/`` の内容ダイジェスト (:func:`_dir_content_digest`)。

    チャンクの evidence id はこの値から導出するので (``chunking.py``)、
    再インストールで id が変わらない。``docs/`` 以外を混ぜると意味が変わる
    ため、対象は常に ``docs/`` だけ (c_16 §4.3)。
    """
    return _dir_content_digest(Path(docs_dir))


def discover_sections(root: Path | str) -> list[str]:
    """展開済みパッケージのうち、ファイルを持つ既知セクション名を返す (昇順)。

    ``provides`` の宣言が無い旧 package.json (c_16 §4.3) の補完、および
    install 時の宣言との突き合わせに使う。
    """
    root = Path(root)
    found = []
    for name in SECTION_DIRS:
        section = root / name
        if section.is_dir() and any(p.is_file() for p in section.rglob("*")):
            found.append(name)
    return sorted(found)


def has_any_section_content(root: Path | str) -> bool:
    """``prebuilt/`` を除く、ファイルを持つトップレベルディレクトリが 1 つでもあるか。

    ``docs/`` も他のセクション (未知のものを含む) も無いパッケージは
    install / write を拒否する根拠 (c_16 §4.3)。
    """
    root = Path(root)
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if not child.is_dir() or child.name == PREBUILT_DIR:
            continue
        if any(p.is_file() for p in child.rglob("*")):
            return True
    return False


def compute_section_digests(root: Path | str) -> dict[str, str]:
    """``docs/`` / ``prebuilt/`` 以外のトップレベルディレクトリのダイジェスト。

    セクションごとに :func:`_dir_content_digest` と同じ方式 (相対パス + バイト列)
    の sha256 を計算する。ファイルの無いディレクトリは含めない。
    """
    root = Path(root)
    if not root.is_dir():
        return {}
    digests: dict[str, str] = {}
    for child in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        if child.name in (DOCS_DIR, PREBUILT_DIR):
            continue
        digest = _dir_content_digest(child)
        if digest:
            digests[child.name] = digest
    return digests


def validate_requires(requires: Sequence[str]) -> None:
    """``requires`` の機能フラグを検査する (c_16 §4.3)。

    ``edition/pro`` だけは動的に評価する (:func:`backend.edition.is_pro`)。
    それ以外は :data:`KNOWN_FEATURES` に無ければ拒否する — 段階 A では
    空集合なので、``edition/pro`` 以外の全フラグが未知として拒否される
    (後続段階が自分のフラグをここに足す)。
    """
    from backend.edition import is_pro

    unknown: list[str] = []
    for flag in requires:
        if flag == EDITION_PRO_FEATURE:
            if not is_pro():
                raise PackageError(
                    f"package requires {EDITION_PRO_FEATURE!r} but this is not "
                    "the Pro edition",
                )
            continue
        if flag not in KNOWN_FEATURES:
            unknown.append(flag)
    if unknown:
        raise PackageError(
            f"package requires unknown feature flag(s): {', '.join(unknown)}",
        )


def section_manifest_payload(
    result: ReadResult,
    *,
    section: str,
    source: str,
    format_version: int,
    error: type[PackageError] = PackageError,
) -> Any:
    """セクションの ``manifest.json`` の封筒の分類から ``payload`` を返す (c_16 §4.5.1)。

    封筒の無い (G0 / 独自の ``schema_version``) manifest と新しい版は、
    ``package.json`` と同じく案内付きの :class:`PackageFormatError` で拒否する。
    それ以外の読めない manifest は ``error`` (セクションの例外型) で送出する。
    """
    if result.ok:
        return result.payload
    if result.status == "foreign" and result.detail == "not a G1 envelope":
        raise PackageFormatError(
            f"{source} has no format_id (a G0 section manifest)",
            "api.cartridge_g0_section_manifest", section=section,
        )
    if result.status == "newer":
        raise PackageFormatError(
            f"{source} format_version {result.version} is newer than {format_version}",
            "api.cartridge_newer_section_manifest", section=section, version=result.version,
        )
    raise error(f"{source} is unreadable ({result.status}: {result.detail})")


def validate_section_feature(
    requires: Sequence[str], section: str, manifest_version: int, package_id: str,
) -> None:
    """セクション manifest の ``format_version`` と ``requires`` の ``<section>/<N>`` を突き合わせる。

    ``requires`` はそのセクションのフラグをちょうど 1 つ、manifest と同じ版で
    宣言しなければならない (c_16 §4.5.1)。``language.verify/1`` のような
    セクション内の機能フラグは別の名前なので数えない。
    """
    expected = f"{section}/{manifest_version}"
    declared = [flag for flag in requires if flag.partition("/")[0] == section]
    if declared != [expected]:
        raise PackageError(
            f"package '{package_id}' provides {section}/manifest.json with format_version "
            f"{manifest_version} but requires declares {declared or 'no ' + section + ' flag'} "
            f"(expected exactly {expected!r})",
        )


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

    規則は全アーカイブ共通の ``backend.io.safe_extract`` (docs/c_06 §1.5)。危険な名前は
    :class:`PackageError`、ディレクトリエントリは ``None``。
    """
    try:
        return normalize_member_name(name)
    except UnsafeArchiveError as e:
        raise PackageError(f"unsafe entry in package: {name!r}") from e


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
    zip_path: Path | str,
    extract_to: Path | str | None = None,
    *,
    max_package_bytes: int | None = None,
    max_unpacked_bytes: int | None = None,
) -> PackageContents:
    """`.evocart` を読む (``extract_to`` 指定時は展開もする)。

    Args:
        zip_path: パッケージ zip。
        extract_to: 展開先。``None`` ならメタデータと目録だけ返す。
        max_package_bytes: zip 本体のサイズ上限 (``rag.packages.max_package_bytes``)。
            ``None`` で無検査。
        max_unpacked_bytes: 展開後の合計サイズ上限 (zip bomb 対策)。``None`` で無検査。

    Raises:
        PackageFormatError: ``package.json`` が G1 の封筒でない (G0) / 新しい版。
        PackageError: zip でない / ``package.json`` が無い / id・版が不正 /
            zip slip を検出した / サイズ上限を超えた。
    """
    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(f"package not found: {path}")
    if max_package_bytes is not None:
        actual_bytes = path.stat().st_size
        if actual_bytes > max_package_bytes:
            raise PackageError(
                f"package exceeds max_package_bytes: {actual_bytes} > "
                f"{max_package_bytes}",
            )
    if not zipfile.is_zipfile(str(path)):
        raise PackageError(f"not a zip archive: {path}")

    destination = Path(extract_to) if extract_to is not None else None

    with zipfile.ZipFile(str(path), "r") as zf:
        limits = ExtractLimits(max_total_bytes=max_unpacked_bytes)
        if max_unpacked_bytes is not None:
            # 自己申告の file_size で早めに断る (実際の展開量は下の budget でも数える)
            unpacked = sum(info.file_size for info in zf.infolist())
            if unpacked > max_unpacked_bytes:
                raise PackageError(
                    f"package unpacks to more than max_unpacked_bytes: "
                    f"{unpacked} > {max_unpacked_bytes}",
                )
        try:
            checked = check_zip_members(zf, limits)
        except UnsafeArchiveError as e:
            raise PackageError(f"unsafe entry in package: {e}") from e
        members: list[PurePosixPath] = [member for _, member in checked]
        source_names: list[str] = [info.filename for info, _ in checked]
        infos_by_name = {info.filename: info for info, _ in checked}
        #: 展開先相対のパス → zip 内の元の名前。
        remap: dict[PurePosixPath, str] = dict(
            zip(_strip_single_root(members), source_names),
        )

        meta_key = PurePosixPath(PACKAGE_FILE)
        if meta_key not in remap:
            raise PackageError(f"{PACKAGE_FILE} not found in package")
        meta = meta_from_json(zf.read(remap[meta_key]))

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
            budget = ExtractBudget(limits)
            for member, source_name in sorted(remap.items()):
                try:
                    target = resolve_under(destination, member)
                    info = infos_by_name[source_name]
                    with zf.open(info) as src:
                        budget.copy(src, target, compressed_size=info.compress_size, overwrite=True)
                except UnsafeArchiveError as e:
                    raise PackageError(f"unsafe entry in package: {source_name!r}: {e}") from e

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
    """``package.json`` を G1 の封筒で原子的に書く (壊れると版が読めないので ``fsync``)。"""
    path = Path(directory) / PACKAGE_FILE
    write_versioned(
        path, format_id=PACKAGE_FORMAT_ID, format_version=PACKAGE_FORMAT_VERSION,
        payload=meta.to_record(), component="corpus.package", fsync=True, indent=2,
    )
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
        meta = read_package_meta(root)
    validate_package_id(meta.id)
    validate_version(meta.version)

    docs_dir = root / DOCS_DIR
    if not has_any_section_content(root):
        raise PackageError(
            f"package has no documents and no other sections under {root}",
        )

    from dataclasses import replace

    meta = replace(
        meta,
        content_digest=compute_content_digest(docs_dir),
        section_digests=compute_section_digests(root),
        provides=meta.provides or discover_sections(root),
    )
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
    "EDITION_PRO_FEATURE",
    "KNOWN_FEATURES",
    "LANGUAGE_DIR",
    "LANGUAGE_VERIFY_FEATURE",
    "MAX_VERSION_LENGTH",
    "PACKAGE_FILE",
    "PACKAGE_ID_RE",
    "PACKAGE_KIND_DEFAULT",
    "PACKAGE_FORMAT_ID",
    "PACKAGE_FORMAT_VERSION",
    "PACKAGE_KINDS",
    "PACKAGE_SUFFIX",
    "PREBUILT_BUILD_FILE",
    "PREBUILT_CHUNKS_FILE",
    "PREBUILT_DIR",
    "PREBUILT_EMBEDDINGS_DIR",
    "SECTION_DIRS",
    "SEMVER_RE",
    "TEMPLATES_DIR",
    "PackageContents",
    "PackageError",
    "PackageFormatError",
    "PackageMeta",
    "PrebuiltInfo",
    "compute_content_digest",
    "compute_section_digests",
    "discover_sections",
    "has_any_section_content",
    "iter_prebuilt_chunks",
    "meta_from_json",
    "meta_from_record",
    "package_filename",
    "prebuilt_from_record",
    "read_package",
    "read_package_meta",
    "validate_package_id",
    "section_manifest_payload",
    "validate_requires",
    "validate_section_feature",
    "validate_version",
    "write_package",
    "write_package_meta",
    "write_prebuilt_build",
    "write_prebuilt_chunks",
]
