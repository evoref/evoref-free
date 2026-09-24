"""`language/` セクション — 言語パックの manifest 読み手 (c_16 §4.5.1 / §4.5.3)

``language/manifest.json`` は「対応言語 1 つ = 1 エントリ」の列。第三者が
ProjectMap (c_16 §4.4) / staged create の構文検査 (``core/code_syntax.py``)
の対応言語を足す拡張点で、**同梱言語 (§4.4 の一覧) は上書きできない**。

本モジュールは manifest の読み込み・構造検証・install 時の拒否判定と、
tree-sitter を使った 1 エントリぶんの実行可能性検証
(:func:`resolve_pack_language_entry`) を持つ。パック同士の拡張子の取り合い
(package id 昇順の先勝ち) と、同梱の拡張子・言語名との衝突判定は
:mod:`backend.free.rag.corpus.store` の ``CorpusStore.language_overlay``
(全 active パッケージを横断する必要があるため) が担う。

段階 C-2 (c_16 §4.5.6) から ``imports`` の指定子・解決規則を検証する
(:class:`ImportRule` / :func:`parse_import_rule`)。抽出は
:mod:`backend.free.rag.projectmap.extractors.treesitter`、解決は
:mod:`backend.free.rag.projectmap.graph` が本モジュールの検証済みの形を読む。
「定義名の照合 (spec 準拠)」と ``verify`` (段階 C-3、コマンド実行) は
引き続き読み込んで保持するだけ。パッケージは Python コードを運ばない —
ここで扱うのは manifest (JSON) と ``.scm`` クエリ (データ) だけ。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from backend.free.rag.corpus.package import (
    LANGUAGE_DIR,
    PACKAGE_ID_RE,
    PackageError,
    PackageFormatError,
    section_manifest_payload,
    validate_section_feature,
)
from backend.io.versioned import read_versioned, read_versioned_bytes, write_versioned
from backend.log_config import get_logger

logger = get_logger("rag.corpus.language")

#: ``language/manifest.json`` の封筒 (c_16 §4.5.1)。版は ``requires`` の
#: ``"language/<N>"`` と一致させる。``language.verify/1`` はこの manifest の中の
#: 機能フラグで、manifest の版とは別に数える (c_16 §4.5.4)。台帳の宣言は ``corpus.store``。
LANGUAGE_FORMAT_ID = "evoref.package.language"
LANGUAGE_FORMAT_VERSION = 1

LANGUAGE_MANIFEST_FILE = "manifest.json"

#: ``.scm`` クエリファイルのサイズ上限 (c_16 §4.5.3)。
MAX_QUERY_BYTES = 65_536

#: §4.4 と同じ捕獲名規約。これ以外の捕獲名を含むクエリは無効 (c_16 §4.5.3)。
ALLOWED_CAPTURE_NAMES: frozenset[str] = frozenset({
    "class.def", "function.def", "method.def", "import.stmt", "call.expr",
})

#: 同梱言語 (c_16 §4.4) が使う拡張子。SSOT は
#: ``backend.free.rag.projectmap.scanner.LANGUAGE_EXTENSIONS`` の値集合と
#: ``backend.free.core.code_syntax`` の対応表の和 — ここでは import しない
#: (corpus は projectmap の依存先なので、逆方向の import は循環になる。
#: store.py の ``PROJECT_MAP_PACKAGE_KIND`` と同じ理由で複製する)。値を
#: 変えるときは両方合わせて直す。
BUNDLED_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".mts", ".cts", ".tsx",
    ".go", ".rs", ".java", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
    ".hxx", ".rb", ".php", ".kt", ".kts", ".swift", ".html", ".htm", ".css",
    ".scss", ".svelte", ".vue", ".json",
})

#: 同梱言語のグラマ名 (tree-sitter-language-pack の名前)。上と同じ理由で複製。
BUNDLED_GRAMMAR_NAMES: frozenset[str] = frozenset({
    "python", "javascript", "typescript", "tsx", "go", "rust", "java", "c",
    "cpp", "ruby", "php", "kotlin", "swift", "html", "css", "scss", "svelte",
    "vue", "json",
})

#: ``tree-sitter-language-pack`` の配布名 (``importlib.metadata.version``)。
_LANGUAGE_PACK_DISTRIBUTION = "tree-sitter-language-pack"

#: ``imports.specifier`` の閉じた語彙 (c_16 §4.5.3、段階 C-2)。パッケージから
#: 任意の式・正規表現を受けない — 値はコード側の固定実装を選ぶだけ。
IMPORT_SPECIFIER_VALUES: frozenset[str] = frozenset({"string_literal", "dotted_name"})

#: ``imports.resolve`` の閉じた語彙 (c_16 §4.5.3、段階 C-2)。上と同じ理由で
#: 閉じている。
IMPORT_RESOLVE_VALUES: frozenset[str] = frozenset({
    "relative", "root_relative", "extension_completion", "index_file", "dotted_to_path",
})

#: ``verify.args`` の要素が丸ごと一致してよいプレースホルダ (c_16 §4.5.4、段階 C-3)。
#: 要素単位の置換だけを許し、文字列連結 (``--out={file}``) は受けない。
VERIFY_ARG_PLACEHOLDERS: frozenset[str] = frozenset({"{file}", "{workspace}"})


class LanguageManifestError(PackageError):
    """``language/manifest.json`` が install を拒否するレベルで壊れている。"""


@dataclass(frozen=True, slots=True)
class ImportRule:
    """``imports`` の検証済みの形 (c_16 §4.5.3、段階 C-2)。

    ``specifier`` / ``resolve`` は閉じた語彙 (:data:`IMPORT_SPECIFIER_VALUES` /
    :data:`IMPORT_RESOLVE_VALUES`) の値だけを持つ。指定子の取り出しは
    :mod:`backend.free.rag.projectmap.extractors.treesitter`、path への解決は
    :mod:`backend.free.rag.projectmap.graph` が読む — 本クラスは検証済みの
    値を運ぶだけで、解決ロジックは持たない。
    """

    specifier: str
    resolve: tuple[str, ...]
    extension_completion: tuple[str, ...] = ()
    index_file: tuple[str, ...] = ()
    external_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifyCommand:
    """検証済みの ``verify`` コマンド 1 件 (c_16 §4.5.4、段階 C-3)。

    構造検証だけを行う — 実行時の allow-list 照合・``which`` 解決・
    ワークスペース境界検査は :mod:`backend.free.loop.staged.language_verify`
    (Loop pillar 側、corpus を import しない複製の作法) が担う。
    """

    id: str
    executable: str
    args: tuple[str, ...]
    timeout_sec: float
    success_exit_codes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VerifyEntry:
    """``verify`` 列の 1 項目 (c_16 §4.5.4)。無効なら ``command`` が ``None``。

    無効な項目 1 件は他の verify・言語自体を止めない (理由だけ保持し、
    API 詳細 (``language_report``) で見える形にする)。
    """

    id: str
    command: VerifyCommand | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class LanguageEntry:
    """``language/manifest.json`` の 1 エントリ (c_16 §4.5.3)。"""

    id: str
    grammar: str
    grammar_range: str | None = None
    extensions: tuple[str, ...] = ()
    #: セクションディレクトリ相対 (posix)。``None`` = クエリを持たない。
    query: str | None = None
    class_ancestor_types: tuple[str, ...] = ()
    #: 検証済み (段階 C-2)。無効な宣言は ``None`` になり、imports 辺を張らない
    #: だけで言語自体は有効のまま (理由は WARNING に残す)。
    imports: ImportRule | None = None
    #: 構造検証済み (段階 C-3)。``language.verify/1`` を requires していない
    #: パッケージでは、実行時にすべて無視される (:mod:`backend.free.rag.corpus.store`
    #: の overlay 計算が package.meta.requires を見て絞り込む)。
    verify: tuple[VerifyEntry, ...] = ()
    #: 未知キーの退避先。
    _extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LanguageManifest:
    """``language/manifest.json`` の ``payload`` (有効なエントリだけを持つ)。"""

    entries: tuple[LanguageEntry, ...]


def _safe_relative_file(
    section_dir: Path, name: Any, *, max_bytes: int | None = None,
) -> str | None:
    """セクションディレクトリ相対のファイル参照を検査する (templates.py と同じ作法)。

    ``..`` / 絶対パス / ドライブレターは ``None``。実在しない・サイズ超過の
    ファイルも ``None``。返り値は posix 形の相対パス文字列。
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
    full = section_dir / Path(*parts)
    if not full.is_file():
        return None
    if max_bytes is not None:
        try:
            if full.stat().st_size > max_bytes:
                return None
        except OSError:
            return None
    return str(rel)


def _parse_extensions(raw: Any) -> tuple[str, ...] | None:
    if not isinstance(raw, list) or not raw:
        return None
    extensions: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            return None
        ext = item.strip()
        if not ext.startswith(".") or ext != ext.lower() or ext == ".":
            return None
        extensions.append(ext)
    return tuple(extensions)


def _parse_string_tuple(raw: Any) -> tuple[str, ...] | None:
    """``None`` → 空タプル、文字列のリスト → タプル、それ以外 → ``None`` (無効)。"""
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
        return None
    return tuple(raw)


def parse_import_rule(raw: Any) -> tuple["ImportRule | None", str | None]:
    """``imports`` オブジェクト 1 つを検証する (c_16 §4.5.3、段階 C-2)。

    ``specifier`` / ``resolve`` は閉じた語彙 (:data:`IMPORT_SPECIFIER_VALUES` /
    :data:`IMPORT_RESOLVE_VALUES`) の値だけを受ける。未知の語・型違いは
    ``(None, 理由)`` — 呼出側 (:func:`_parse_entry`) はこの結果をそのまま
    entry の ``imports`` に入れる (エントリ自体は無効にしない)。
    """
    if not isinstance(raw, dict):
        return None, "imports must be an object"

    specifier = raw.get("specifier")
    if specifier not in IMPORT_SPECIFIER_VALUES:
        return None, (
            f"imports.specifier must be one of {sorted(IMPORT_SPECIFIER_VALUES)}, "
            f"got {specifier!r}"
        )

    resolve_raw = raw.get("resolve")
    if not isinstance(resolve_raw, list) or not resolve_raw:
        return None, "imports.resolve must be a non-empty list"
    resolve: list[str] = []
    for step in resolve_raw:
        if step not in IMPORT_RESOLVE_VALUES:
            return None, (
                f"imports.resolve has an unknown step {step!r} "
                f"(allowed: {sorted(IMPORT_RESOLVE_VALUES)})"
            )
        resolve.append(step)

    extension_completion = _parse_string_tuple(raw.get("extension_completion"))
    if extension_completion is None:
        return None, "imports.extension_completion must be a list of strings"
    index_file = _parse_string_tuple(raw.get("index_file"))
    if index_file is None:
        return None, "imports.index_file must be a list of strings"
    external_prefixes = _parse_string_tuple(raw.get("external_prefixes"))
    if external_prefixes is None:
        return None, "imports.external_prefixes must be a list of strings"

    if "extension_completion" in resolve and not extension_completion:
        return None, "imports.resolve uses extension_completion but none is declared"
    if "index_file" in resolve and not index_file:
        return None, "imports.resolve uses index_file but none is declared"

    return ImportRule(
        specifier=specifier,
        resolve=tuple(resolve),
        extension_completion=extension_completion,
        index_file=index_file,
        external_prefixes=external_prefixes,
    ), None


def parse_verify_command(raw: Any, index: int) -> VerifyEntry:
    """``verify`` 列の 1 項目を検証する (c_16 §4.5.4、段階 C-3)。

    違反した項目はその 1 件だけ無効にする (:attr:`VerifyEntry.command` が
    ``None``、:attr:`VerifyEntry.reason` に理由)。他の verify・言語自体は
    有効のまま。``label`` (表示用 id) は宣言された ``id`` が読める限りそれを
    使う — 無効な id でも API 詳細でどの宣言かを見分けられるようにする。
    """
    label = f"#{index}"
    if isinstance(raw, dict):
        raw_id = raw.get("id")
        if isinstance(raw_id, str) and raw_id:
            label = raw_id

    if not isinstance(raw, dict):
        return VerifyEntry(id=label, command=None, reason="verify entry must be an object")

    entry_id = raw.get("id")
    if not isinstance(entry_id, str) or not PACKAGE_ID_RE.match(entry_id):
        return VerifyEntry(
            id=label, command=None,
            reason=f"verify.id is invalid: {entry_id!r} (expected {PACKAGE_ID_RE.pattern})",
        )

    executable = raw.get("executable")
    if not isinstance(executable, str) or not executable:
        return VerifyEntry(
            id=label, command=None, reason="verify.executable must be a non-empty string",
        )
    if any(c in executable for c in "/\\:. ") or executable != executable.strip():
        return VerifyEntry(
            id=label, command=None,
            reason=(
                f"verify.executable must be a bare name without path separators, "
                f"':', '.', or whitespace: {executable!r}"
            ),
        )

    args_raw = raw.get("args")
    if not isinstance(args_raw, list) or not all(isinstance(a, str) for a in args_raw):
        return VerifyEntry(id=label, command=None, reason="verify.args must be a list of strings")
    args: list[str] = []
    for a in args_raw:
        if a in VERIFY_ARG_PLACEHOLDERS:
            args.append(a)
            continue
        if "{" in a or "}" in a:
            return VerifyEntry(
                id=label, command=None,
                reason=(
                    f"verify.args element {a!r} must be exactly {{file}}/{{workspace}} "
                    "or contain no braces (no string concatenation)"
                ),
            )
        args.append(a)

    timeout_raw = raw.get("timeout_sec")
    if (
        isinstance(timeout_raw, bool)
        or not isinstance(timeout_raw, (int, float))
        or timeout_raw <= 0
    ):
        return VerifyEntry(
            id=label, command=None,
            reason=f"verify.timeout_sec must be a positive number: {timeout_raw!r}",
        )

    codes_raw = raw.get("success_exit_codes")
    if (
        not isinstance(codes_raw, list) or not codes_raw
        or not all(isinstance(c, int) and not isinstance(c, bool) for c in codes_raw)
    ):
        return VerifyEntry(
            id=label, command=None,
            reason="verify.success_exit_codes must be a non-empty list of integers",
        )

    return VerifyEntry(
        id=entry_id,
        command=VerifyCommand(
            id=entry_id,
            executable=executable,
            args=tuple(args),
            timeout_sec=float(timeout_raw),
            success_exit_codes=tuple(codes_raw),
        ),
    )


def _parse_entry(raw: Any, section_dir: Path, package_id: str) -> LanguageEntry | None:
    """1 エントリを検証する。無効なら ``None`` (呼出側が件数を WARNING に出す)。"""
    if not isinstance(raw, dict):
        return None

    entry_id = raw.get("id")
    if not isinstance(entry_id, str) or not PACKAGE_ID_RE.match(entry_id):
        logger.warning(
            "corpus package %s: language entry has invalid id %r", package_id, entry_id,
        )
        return None

    grammar = raw.get("grammar")
    if not isinstance(grammar, str) or not grammar.strip():
        logger.warning(
            "corpus package %s: language entry %s missing grammar", package_id, entry_id,
        )
        return None

    grammar_range = raw.get("grammar_range")
    if grammar_range is not None and not isinstance(grammar_range, str):
        logger.warning(
            "corpus package %s: language entry %s grammar_range must be a string",
            package_id, entry_id,
        )
        return None

    extensions = _parse_extensions(raw.get("extensions"))
    if extensions is None:
        logger.warning(
            "corpus package %s: language entry %s has invalid extensions %r",
            package_id, entry_id, raw.get("extensions"),
        )
        return None

    query_rel: str | None = None
    query_name = raw.get("query")
    if query_name is not None:
        query_rel = _safe_relative_file(section_dir, query_name, max_bytes=MAX_QUERY_BYTES)
        if query_rel is None or PurePosixPath(query_rel).suffix.lower() != ".scm":
            logger.warning(
                "corpus package %s: language entry %s has an unreadable/oversized "
                "query %r (must be a .scm file <= %d bytes)",
                package_id, entry_id, query_name, MAX_QUERY_BYTES,
            )
            return None

    ancestor_raw = raw.get("class_ancestor_types") or []
    if not isinstance(ancestor_raw, list) or not all(isinstance(a, str) for a in ancestor_raw):
        logger.warning(
            "corpus package %s: language entry %s class_ancestor_types must be "
            "a list of strings", package_id, entry_id,
        )
        return None
    class_ancestor_types = tuple(ancestor_raw)

    imports_raw = raw.get("imports")
    imports: ImportRule | None = None
    if imports_raw is not None:
        imports, imports_invalid_reason = parse_import_rule(imports_raw)
        if imports is None:
            logger.warning(
                "corpus package %s: language entry %s has invalid imports (%s); "
                "the language stays enabled without imports edges",
                package_id, entry_id, imports_invalid_reason,
            )

    verify_raw = raw.get("verify")
    verify: tuple[VerifyEntry, ...] = ()
    if verify_raw is not None:
        if not isinstance(verify_raw, list):
            logger.warning(
                "corpus package %s: language entry %s verify must be a list; "
                "the language stays enabled without verify commands",
                package_id, entry_id,
            )
        else:
            parsed_verify: list[VerifyEntry] = []
            invalid_verify = 0
            for i, v in enumerate(verify_raw):
                item = parse_verify_command(v, i)
                if item.command is None:
                    invalid_verify += 1
                    logger.warning(
                        "corpus package %s: language entry %s verify %s disabled: %s",
                        package_id, entry_id, item.id, item.reason,
                    )
                parsed_verify.append(item)
            if invalid_verify:
                logger.warning(
                    "corpus package %s: language entry %s skipped %d invalid "
                    "verify entry(ies)", package_id, entry_id, invalid_verify,
                )
            verify = tuple(parsed_verify)

    known = {
        "id", "grammar", "grammar_range", "extensions", "query",
        "class_ancestor_types", "imports", "verify",
    }
    extra = {k: v for k, v in raw.items() if k not in known}

    return LanguageEntry(
        id=entry_id,
        grammar=grammar.strip(),
        grammar_range=grammar_range,
        extensions=extensions,
        query=query_rel,
        class_ancestor_types=class_ancestor_types,
        imports=imports,
        verify=verify,
        _extra=extra,
    )


def parse_language_manifest(
    data: Any, section_dir: Path, package_id: str,
) -> LanguageManifest:
    """``language/manifest.json`` の ``payload`` を検証済みの :class:`LanguageManifest` にする。

    無効なエントリはそのエントリだけ飛ばして件数を WARNING に出す (c_05 §0.5.2)。
    封筒 (版) の検査は呼出側 (:func:`load_language_manifest` /
    :func:`validate_language_install`) が済ませている。
    """
    if not isinstance(data, dict):
        raise LanguageManifestError(
            f"language manifest of package '{package_id}' must be a JSON object",
        )

    raw_entries = data.get("languages") or []
    if not isinstance(raw_entries, list):
        raise LanguageManifestError(
            f"language manifest of package '{package_id}': 'languages' must "
            "be a list",
        )

    entries: list[LanguageEntry] = []
    skipped = 0
    for raw in raw_entries:
        entry = _parse_entry(raw, section_dir, package_id)
        if entry is None:
            skipped += 1
            continue
        entries.append(entry)
    if skipped:
        logger.warning(
            "corpus package %s: skipped %d invalid language entry(ies)",
            package_id, skipped,
        )
    return LanguageManifest(entries=tuple(entries))


def write_language_manifest(section_dir: Path, languages: Sequence[dict[str, Any]]) -> Path:
    """``<section_dir>/manifest.json`` を G1 の封筒で書く (パッケージを組む側、c_16 §4.5.1)。"""
    path = Path(section_dir) / LANGUAGE_MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    write_versioned(
        path, format_id=LANGUAGE_FORMAT_ID, format_version=LANGUAGE_FORMAT_VERSION,
        payload={"languages": list(languages)}, component="corpus.language",
        fsync=False, indent=2,
    )
    return path


def load_language_manifest(
    section_dir: Path, package_id: str,
) -> LanguageManifest | None:
    """``<section_dir>/manifest.json`` を読む (無い・壊れていれば ``None``)。

    封筒の無い (G0) / 新しい版の manifest は黙って空にせず
    :class:`PackageFormatError` を送出する — 呼出側はパッケージの読み込みを
    拒否する (c_16 §4.5.1)。
    """
    path = section_dir / LANGUAGE_MANIFEST_FILE
    result = read_versioned(
        path, format_id=LANGUAGE_FORMAT_ID, format_version=LANGUAGE_FORMAT_VERSION,
    )
    if result.status == "absent":
        return None
    try:
        payload = section_manifest_payload(
            result, section=LANGUAGE_DIR, source=str(path),
            format_version=LANGUAGE_FORMAT_VERSION, error=LanguageManifestError,
        )
        return parse_language_manifest(payload, section_dir, package_id)
    except PackageFormatError:
        raise
    except LanguageManifestError as e:
        logger.warning(
            "corpus package %s: invalid language manifest: %s", package_id, e,
        )
        return None


def validate_language_install(
    directory: Path, package_id: str, requires: Sequence[str],
) -> None:
    """install 時の拒否判定 (c_16 §4.5.1)。

    ``language/`` セクションを ``provides`` しているのに ``manifest.json`` が
    無い / 壊れている / G1 の封筒でない / 新しい版 / 版が ``requires`` の
    ``language/<N>`` と一致しない場合は install を拒否する。
    """
    section_dir = directory / LANGUAGE_DIR
    if not section_dir.is_dir():
        return
    source = f"{LANGUAGE_DIR}/{LANGUAGE_MANIFEST_FILE}"
    manifest_path = section_dir / LANGUAGE_MANIFEST_FILE
    if not manifest_path.is_file():
        raise LanguageManifestError(
            f"package '{package_id}' provides language but {source} is missing",
        )
    try:
        data = manifest_path.read_bytes()
    except OSError as e:
        raise LanguageManifestError(
            f"package '{package_id}' has an unreadable language manifest: {e}",
        ) from e
    # 取り込み中の版は data_health に載せない (拒否すれば版ごと消える)
    result = read_versioned_bytes(
        data, format_id=LANGUAGE_FORMAT_ID, format_version=LANGUAGE_FORMAT_VERSION,
    )
    payload = section_manifest_payload(
        result, section=LANGUAGE_DIR, source=source,
        format_version=LANGUAGE_FORMAT_VERSION, error=LanguageManifestError,
    )
    validate_section_feature(requires, LANGUAGE_DIR, LANGUAGE_FORMAT_VERSION, package_id)
    # ここは検証だけ (結果は _open_package / _build_version が改めて読む)。
    parse_language_manifest(payload, section_dir, package_id)


def load_language_for_package(
    directory: Path, package_id: str,
) -> tuple[LanguageEntry, ...]:
    """版ディレクトリの ``language/`` を開く (open 時に常駐させる、c_16 §4.5.1)。

    Raises:
        PackageFormatError: manifest が G1 の封筒でない (G0) / 新しい版。
    """
    section_dir = directory / LANGUAGE_DIR
    if not section_dir.is_dir():
        return ()
    manifest = load_language_manifest(section_dir, package_id)
    if manifest is None:
        return ()
    return manifest.entries


# ── tree-sitter を使った実行可能性検証 (c_16 §4.5.3) ──────────────────────


@dataclass(frozen=True, slots=True)
class ResolvedLanguage:
    """1 言語エントリぶんの、抽出に使える形 (c_16 §4.5.3)。"""

    grammar: str
    #: ``.scm`` の中身 (無ければ ``None`` — ProjectMap の抽出には使えないが、
    #: 構文検査には grammar だけあれば足りる)。
    query_text: str | None
    class_ancestor_types: tuple[str, ...]
    #: 検証済み (段階 C-2)。``None`` = imports 辺を張らない。
    imports: ImportRule | None = None


def _installed_language_pack_version() -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(_LANGUAGE_PACK_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def _grammar_range_compatible(grammar_range: str | None) -> bool:
    """``grammar_range`` が導入済みの tree-sitter-language-pack と合うか。

    ``packaging`` が使えない / 版が取れない場合は検査をスキップする (寛容側、
    c_16 §4.5.3)。
    """
    if not grammar_range:
        return True
    installed = _installed_language_pack_version()
    if installed is None:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:
        return True
    try:
        return Version(installed) in SpecifierSet(grammar_range)
    except Exception:  # noqa: BLE001 — 版式が壊れていても検査をブロックしない
        return True


def resolve_pack_language_entry(
    entry: LanguageEntry, section_dir: Path,
) -> tuple[ResolvedLanguage | None, str | None]:
    """1 エントリを tree-sitter 込みで検証する。

    Returns:
        ``(resolved, None)`` — 有効。``(None, reason)`` — 無効 (理由付き)。

    grammar の入手性・``grammar_range``・クエリの構築可能性・捕獲名規約
    (c_16 §4.4 と同じ ``class.def``/``function.def``/``method.def``/
    ``import.stmt``/``call.expr``) を確認する。1 エントリの失敗は他の
    エントリ・他の言語に波及しない (呼出側が個別に扱う)。
    """
    try:
        import tree_sitter
        from tree_sitter_language_pack import get_language
    except ImportError:
        return None, "tree-sitter is not available in this environment"

    if not _grammar_range_compatible(entry.grammar_range):
        installed = _installed_language_pack_version() or "unknown"
        return None, (
            f"grammar_range {entry.grammar_range!r} does not match installed "
            f"tree-sitter-language-pack {installed}"
        )

    try:
        language = get_language(entry.grammar)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001 — 1 言語の失敗で他を止めない
        return None, f"grammar {entry.grammar!r} is not available: {e}"

    query_text: str | None = None
    if entry.query is not None:
        try:
            query_text = (section_dir / entry.query).read_text(encoding="utf-8")
        except OSError as e:
            return None, f"query file {entry.query!r} is unreadable: {e}"
        try:
            query = tree_sitter.Query(language, query_text)
        except Exception as e:  # noqa: BLE001 — Query 構築エラーは多様
            return None, f"query is invalid: {e}"
        for i in range(query.capture_count):
            name = query.capture_name(i)
            if name not in ALLOWED_CAPTURE_NAMES:
                return None, (
                    f"query has an unsupported capture name {name!r} "
                    f"(allowed: {sorted(ALLOWED_CAPTURE_NAMES)})"
                )

    return ResolvedLanguage(
        grammar=entry.grammar,
        query_text=query_text,
        class_ancestor_types=entry.class_ancestor_types,
        imports=entry.imports,
    ), None


__all__ = [
    "ALLOWED_CAPTURE_NAMES",
    "BUNDLED_EXTENSIONS",
    "BUNDLED_GRAMMAR_NAMES",
    "IMPORT_RESOLVE_VALUES",
    "IMPORT_SPECIFIER_VALUES",
    "LANGUAGE_FORMAT_ID",
    "LANGUAGE_FORMAT_VERSION",
    "LANGUAGE_MANIFEST_FILE",
    "MAX_QUERY_BYTES",
    "VERIFY_ARG_PLACEHOLDERS",
    "ImportRule",
    "LanguageEntry",
    "LanguageManifest",
    "LanguageManifestError",
    "ResolvedLanguage",
    "VerifyCommand",
    "VerifyEntry",
    "load_language_for_package",
    "load_language_manifest",
    "parse_import_rule",
    "parse_language_manifest",
    "parse_verify_command",
    "resolve_pack_language_entry",
    "validate_language_install",
    "write_language_manifest",
]
