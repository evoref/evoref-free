"""`templates/` セクション — 文書テンプレートの manifest 読み手 (c_16 §4.5.1 / §4.5.2)

``templates/manifest.json`` は「文書種別 1 つ = 1 エントリ」の列で、モード別の
ディレクトリには分けない (提案資料は「体裁」と「構成」を同時に使うため)。本
モジュールは manifest の読み込み・検証・install 時の拒否判定だけを持つ。
選択判定 (``template_select``) は :mod:`backend.free.api.chat._template_select`、
体裁の継承 (Writer 側) は :mod:`backend.free.export.writers.docx` /
:mod:`backend.free.export.writers.pptx` が別々に持つ (c_16 §4.5.6 の段階分け)。

``fields`` の語彙検証・プレースホルダの走査は :mod:`backend.export.template_writer`
(段階 B-1b、f_11 §9.2) が持つ (一方向依存: corpus → export)。install 時は
``fields`` の語彙が壊れている / 様式ファイルのプレースホルダと一致しない
エントリをまるごと無効にする (c_16 §4.5.2)。構成テンプレートの seed (B-2)
はまだ実装しない。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from backend.free.rag.corpus.package import (
    PACKAGE_ID_RE,
    TEMPLATES_DIR,
    PackageError,
    PackageFormatError,
    section_manifest_payload,
    validate_section_feature,
)
from backend.io.versioned import read_versioned, read_versioned_bytes, write_versioned
from backend.log_config import get_logger

logger = get_logger("rag.corpus.templates")

#: ``templates/manifest.json`` の封筒 (c_16 §4.5.1)。版は ``requires`` の
#: ``"templates/<N>"`` と一致させる。台帳の宣言は ``corpus.store``。
TEMPLATES_FORMAT_ID = "evoref.package.templates"
TEMPLATES_FORMAT_VERSION = 1

TEMPLATES_MANIFEST_FILE = "manifest.json"

#: ``base`` に許される拡張子 (c_16 §4.5.2)。``.dotx`` 等は
#: ``store._REJECTED_TEMPLATE_SUFFIXES`` が別途拒否する。
BASE_SUFFIXES = frozenset({".docx", ".pptx", ".xlsx"})


class TemplateManifestError(PackageError):
    """``templates/manifest.json`` が install を拒否するレベルで壊れている。"""


@dataclass(frozen=True, slots=True)
class TemplateEntry:
    """``templates/manifest.json`` の 1 エントリ。"""

    id: str
    doc_type: str
    aliases: tuple[str, ...] = ()
    lang: str = "ja"
    description: str = ""
    #: セクションディレクトリ相対 (posix)。``None`` = 体裁の継承元を持たない。
    base: str | None = None
    #: セクションディレクトリ相対 (posix)。``fields`` があれば無視 (WARNING)。
    outline: str | None = None
    #: 帳票の列定義 (f_11 §9.2)。生 dict のまま保持する — 語彙の検証は
    #: install 時に :func:`backend.export.template_writer.parse_field_specs`
    #: で行い (無効ならエントリごと無効化)、実際の解釈 (穴埋め) は
    #: 消費側 (chat.py) が都度 ``parse_field_specs`` を呼んで行う。
    fields: tuple[dict[str, Any], ...] | None = None
    #: 未知キーの退避先。
    _extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TemplateManifest:
    """``templates/manifest.json`` の ``payload`` (有効なエントリだけを持つ)。"""

    entries: tuple[TemplateEntry, ...]


def _safe_relative_file(section_dir: Path, name: Any) -> str | None:
    """セクションディレクトリ相対のファイル参照を検査する。

    ``..`` / 絶対パス / ドライブレターは ``None`` (呼出側がエントリごと飛ばす)。
    実在しないファイルも ``None``。返り値は posix 形の相対パス文字列。
    """
    if not isinstance(name, str) or not name.strip():
        return None
    normalized = name.replace("\\", "/")
    if not normalized or normalized.startswith("/") or ":" in normalized.split("/")[0]:
        return None
    parts = [p for p in PurePosixPath(normalized).parts if p not in (".", "")]
    if not parts or any(p == ".." for p in parts):
        return None
    rel = PurePosixPath(*parts)
    if not (section_dir / Path(*parts)).is_file():
        return None
    return str(rel)


def _validate_fields_vocabulary(
    fields: tuple[dict[str, Any], ...],
    base_rel: str,
    section_dir: Path,
    package_id: str,
    entry_id: str,
) -> bool:
    """``fields`` の語彙・様式との突き合わせを検証する (f_11 §9.2 が SSOT)。

    無効なら ``False`` (呼出側がエントリごと無効化して件数を WARNING に出す)。
    """
    from backend.export.template_writer import (
        TemplateFillError,
        parse_field_specs,
        required_placeholder_names,
        scan_placeholders,
        validate_xlsx_repeat_rows,
    )

    field_specs = parse_field_specs(fields)
    if field_specs is None:
        logger.warning(
            "corpus package %s: templates entry %s has an invalid fields "
            "vocabulary (unknown type/repeat/compute, duplicate name, or "
            "empty list); skipped", package_id, entry_id,
        )
        return False

    base_suffix = PurePosixPath(base_rel).suffix.lower()
    if base_suffix == ".pptx" and any(f.repeat for f in field_specs):
        logger.warning(
            "corpus package %s: templates entry %s uses 'repeat' with a "
            ".pptx base, which is not supported; skipped",
            package_id, entry_id,
        )
        return False

    base_path = section_dir / base_rel
    try:
        present = scan_placeholders(base_path)
    except TemplateFillError as e:
        logger.warning(
            "corpus package %s: templates entry %s base could not be "
            "scanned for placeholders: %s; skipped", package_id, entry_id, e,
        )
        return False

    required = required_placeholder_names(field_specs)
    extra = sorted(present - required)
    missing = sorted(required - present)
    if extra or missing:
        logger.warning(
            "corpus package %s: templates entry %s fields do not match the "
            "base's placeholders (extra in base=%s, missing from base=%s); "
            "skipped", package_id, entry_id, extra, missing,
        )
        return False

    if base_suffix == ".xlsx":
        try:
            validate_xlsx_repeat_rows(base_path, field_specs)
        except TemplateFillError as e:
            logger.warning(
                "corpus package %s: templates entry %s violates the xlsx "
                "repeat-row requirement: %s; skipped", package_id, entry_id, e,
            )
            return False

    return True


def _parse_entry(raw: Any, section_dir: Path, package_id: str) -> TemplateEntry | None:
    """1 エントリを検証する。無効なら ``None`` (呼出側が件数を WARNING に出す)。"""
    if not isinstance(raw, dict):
        return None

    entry_id = raw.get("id")
    if not isinstance(entry_id, str) or not PACKAGE_ID_RE.match(entry_id):
        logger.warning(
            "corpus package %s: templates entry has invalid id %r", package_id, entry_id,
        )
        return None

    doc_type = raw.get("doc_type")
    if not isinstance(doc_type, str) or not doc_type.strip():
        logger.warning(
            "corpus package %s: templates entry %s missing doc_type",
            package_id, entry_id,
        )
        return None

    aliases_raw = raw.get("aliases") or []
    if not isinstance(aliases_raw, list):
        logger.warning(
            "corpus package %s: templates entry %s aliases must be a list",
            package_id, entry_id,
        )
        return None
    aliases = tuple(
        a.strip() for a in aliases_raw if isinstance(a, str) and a.strip()
    )

    lang = str(raw.get("lang") or "ja")
    description = str(raw.get("description") or "")

    base_rel: str | None = None
    base_name = raw.get("base")
    if base_name is not None:
        base_rel = _safe_relative_file(section_dir, base_name)
        if base_rel is None:
            logger.warning(
                "corpus package %s: templates entry %s has an unreadable base %r",
                package_id, entry_id, base_name,
            )
            return None
        if PurePosixPath(base_rel).suffix.lower() not in BASE_SUFFIXES:
            logger.warning(
                "corpus package %s: templates entry %s base %r is not "
                ".docx/.pptx/.xlsx", package_id, entry_id, base_rel,
            )
            return None

    outline_rel: str | None = None
    outline_name = raw.get("outline")
    if outline_name is not None:
        outline_rel = _safe_relative_file(section_dir, outline_name)
        if outline_rel is None:
            logger.warning(
                "corpus package %s: templates entry %s has an unreadable outline %r",
                package_id, entry_id, outline_name,
            )
            return None

    fields_raw = raw.get("fields")
    fields: tuple[dict[str, Any], ...] | None = None
    if fields_raw is not None:
        if not isinstance(fields_raw, list) or not fields_raw:
            logger.warning(
                "corpus package %s: templates entry %s fields must be a "
                "non-empty list", package_id, entry_id,
            )
            return None
        if base_rel is None:
            # fields があるのに base が無い = 差し込む先が無い (c_16 §4.5.2 の表)。
            logger.warning(
                "corpus package %s: templates entry %s has fields but no base "
                "(nothing to fill in); skipped", package_id, entry_id,
            )
            return None
        fields = tuple(f for f in fields_raw if isinstance(f, dict))
        if outline_rel is not None:
            logger.warning(
                "corpus package %s: templates entry %s has both fields and "
                "outline; outline is ignored (fields wins, c_16 §4.5.2)",
                package_id, entry_id,
            )
            outline_rel = None
        if not _validate_fields_vocabulary(fields, base_rel, section_dir, package_id, entry_id):
            return None

    known = {
        "id", "doc_type", "aliases", "lang", "description", "base", "outline",
        "fields",
    }
    extra = {k: v for k, v in raw.items() if k not in known}

    return TemplateEntry(
        id=entry_id,
        doc_type=doc_type.strip(),
        aliases=aliases,
        lang=lang,
        description=description,
        base=base_rel,
        outline=outline_rel,
        fields=fields,
        _extra=extra,
    )


def parse_template_manifest(
    data: Any, section_dir: Path, package_id: str,
) -> TemplateManifest:
    """``templates/manifest.json`` の ``payload`` を検証済みの :class:`TemplateManifest` にする。

    無効なエントリはそのエントリだけ飛ばして件数を WARNING に出す (c_05 §0.5.2)。
    封筒 (版) の検査は呼出側 (:func:`load_template_manifest` /
    :func:`validate_templates_install`) が済ませている。
    """
    if not isinstance(data, dict):
        raise TemplateManifestError(
            f"templates manifest of package '{package_id}' must be a JSON object",
        )

    raw_entries = data.get("templates") or []
    if not isinstance(raw_entries, list):
        raise TemplateManifestError(
            f"templates manifest of package '{package_id}': 'templates' must "
            "be a list",
        )

    entries: list[TemplateEntry] = []
    skipped = 0
    for raw in raw_entries:
        entry = _parse_entry(raw, section_dir, package_id)
        if entry is None:
            skipped += 1
            continue
        entries.append(entry)
    if skipped:
        logger.warning(
            "corpus package %s: skipped %d invalid templates entry(ies)",
            package_id, skipped,
        )
    return TemplateManifest(entries=tuple(entries))


def write_template_manifest(section_dir: Path, templates: Sequence[dict[str, Any]]) -> Path:
    """``<section_dir>/manifest.json`` を G1 の封筒で書く (パッケージを組む側、c_16 §4.5.1)。"""
    path = Path(section_dir) / TEMPLATES_MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    write_versioned(
        path, format_id=TEMPLATES_FORMAT_ID, format_version=TEMPLATES_FORMAT_VERSION,
        payload={"templates": list(templates)}, component="corpus.templates",
        fsync=False, indent=2,
    )
    return path


def load_template_manifest(
    section_dir: Path, package_id: str,
) -> TemplateManifest | None:
    """``<section_dir>/manifest.json`` を読む (無い・壊れていれば ``None``)。

    版ディレクトリは install 後不変なので、install 時に検証済みの manifest を
    ここで再度壊れているとみなす必要は薄いが、手動でディレクトリを触られた
    場合にも本体の起動を落とさない。ただし封筒の無い (G0) / 新しい版の
    manifest は黙って空にせず :class:`PackageFormatError` を送出する —
    呼出側はパッケージの読み込みを拒否する (c_16 §4.5.1)。
    """
    path = section_dir / TEMPLATES_MANIFEST_FILE
    result = read_versioned(
        path, format_id=TEMPLATES_FORMAT_ID, format_version=TEMPLATES_FORMAT_VERSION,
    )
    if result.status == "absent":
        return None
    try:
        payload = section_manifest_payload(
            result, section=TEMPLATES_DIR, source=str(path),
            format_version=TEMPLATES_FORMAT_VERSION, error=TemplateManifestError,
        )
        return parse_template_manifest(payload, section_dir, package_id)
    except PackageFormatError:
        raise
    except TemplateManifestError as e:
        logger.warning(
            "corpus package %s: invalid templates manifest: %s", package_id, e,
        )
        return None


def validate_templates_install(
    directory: Path, package_id: str, requires: Sequence[str],
) -> None:
    """install 時の拒否判定 (c_16 §4.5.1)。

    ``templates/`` セクションを ``provides`` しているのに ``manifest.json`` が
    無い / 壊れている / G1 の封筒でない / 新しい版 / 版が ``requires`` の
    ``templates/<N>`` と一致しない場合は install を拒否する。呼出側
    (``CorpusStore.install``) は ``provides`` の宣言と実ディレクトリが既に
    突き合わせ済みの後でこれを呼ぶ。
    """
    section_dir = directory / TEMPLATES_DIR
    if not section_dir.is_dir():
        return
    source = f"{TEMPLATES_DIR}/{TEMPLATES_MANIFEST_FILE}"
    manifest_path = section_dir / TEMPLATES_MANIFEST_FILE
    if not manifest_path.is_file():
        raise TemplateManifestError(
            f"package '{package_id}' provides templates but {source} is missing",
        )
    try:
        data = manifest_path.read_bytes()
    except OSError as e:
        raise TemplateManifestError(
            f"package '{package_id}' has an unreadable templates manifest: {e}",
        ) from e
    # 取り込み中の版は data_health に載せない (拒否すれば版ごと消える)
    result = read_versioned_bytes(
        data, format_id=TEMPLATES_FORMAT_ID, format_version=TEMPLATES_FORMAT_VERSION,
    )
    payload = section_manifest_payload(
        result, section=TEMPLATES_DIR, source=source,
        format_version=TEMPLATES_FORMAT_VERSION, error=TemplateManifestError,
    )
    validate_section_feature(requires, TEMPLATES_DIR, TEMPLATES_FORMAT_VERSION, package_id)
    # ここは検証だけ (結果は _open_package / _build_version が改めて読む)。
    parse_template_manifest(payload, section_dir, package_id)


@dataclass(frozen=True, slots=True)
class TemplateCandidate:
    """``template_select`` 判定点へ渡す 1 エントリの命名材料 (c_17 §3.8)。"""

    package_id: str
    version: str
    entry: TemplateEntry

    @property
    def label(self) -> str:
        """判定点が ``fire`` で返すラベル (= 参照鍵)。"""
        return f"{self.package_id}:{self.entry.id}"

    def matches_naming(self, text: str) -> bool:
        """発話が ``doc_type`` / ``aliases`` のいずれかを指名しているか。"""
        return any(
            _term_in_text(text, term)
            for term in (self.entry.doc_type, *self.entry.aliases)
        )


def _term_in_text(text: str, term: str) -> bool:
    """指名語の照合 (locale で排他しない、英語は大文字小文字を無視、c_16 §4.5.2)。

    ASCII だけの語は語境界付きで当たる (``go`` が ``algorithm`` の一部に
    誤爆するのと同じ理由で、短い英語別名の部分一致を避ける)。CJK を含む語は
    素の部分一致 (日本語に語境界という概念が無いため)。
    """
    term = term.strip()
    if not term:
        return False
    if term.isascii():
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])"
        return re.search(pattern, text, re.IGNORECASE) is not None
    return term in text


def load_templates_for_package(directory: Path, package_id: str) -> tuple[TemplateEntry, ...]:
    """版ディレクトリの ``templates/`` を開く (open 時に常駐させる、c_16 §4.5.1)。

    Raises:
        PackageFormatError: manifest が G1 の封筒でない (G0) / 新しい版。
    """
    section_dir = directory / TEMPLATES_DIR
    if not section_dir.is_dir():
        return ()
    manifest = load_template_manifest(section_dir, package_id)
    if manifest is None:
        return ()
    return manifest.entries


__all__ = [
    "BASE_SUFFIXES",
    "TEMPLATES_FORMAT_ID",
    "TEMPLATES_FORMAT_VERSION",
    "TEMPLATES_MANIFEST_FILE",
    "TemplateCandidate",
    "TemplateEntry",
    "TemplateManifest",
    "TemplateManifestError",
    "load_template_manifest",
    "load_templates_for_package",
    "parse_template_manifest",
    "validate_templates_install",
    "write_template_manifest",
]
