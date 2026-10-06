"""コードグラフの組み立て (c_16 §4.4)

抽出器 (``extractors/``) が返す 1 ファイル分の生ノード / import / 呼出しを
受け取り、辺を解決してプロジェクト全体の :class:`ProjectGraph` にする。

- ``contains``: 構文木から確定 (file → 直下の定義、定義 → ネストした定義)
- ``imports``: モジュール指定子をプロジェクト内 path へ解決できたものだけ
  (Python は絶対 import のドット区切り、JS/TS/TSX は相対パス + 拡張子補完、
  Rust/Go/Ruby/Java/Kotlin/PHP/C/C++ は言語ごとの決定論規則 — 詳細は各
  ``_resolve_<lang>_import`` の docstring。Swift はモジュール名が path に
  落ちないため常に external。HTML は ``<script src>``/``<link href>``、
  CSS/SCSS は ``@import``/``@use``/``@forward``/``url()``、Svelte/Vue は
  JS/TS 風と CSS 風の両方を試す — c_16 §4.4 マークアップ / SFC。言語パック
  (c_16 §4.5.3、段階 C-2) は ``ImportRule.resolve`` の閉じた語彙
  (``relative``/``root_relative``/``extension_completion``/``index_file``/
  ``dotted_to_path``) を順に試す汎用解決 — :func:`_resolve_pack_import`)。
  解決できないものは file ノードの ``external_imports`` に残す。Go だけは
  1 つの import 指定子が 1 パッケージ (= ディレクトリ) を指すため、複数
  ファイルへ辺を張りうる。JS/TS/Svelte/Vue は先に ``$lib`` (SvelteKit) /
  tsconfig・jsconfig ``paths`` のエイリアス解決 (:mod:`.aliases`、
  :func:`_resolve_via_alias`) を試し、当てはまらなければ通常の相対解決へ
  フォールバックする (``alias_config`` が ``None``、または一致するエイリアス
  が無い場合)

- ``calls`` / ``inherits``: 呼出名 / 基底名を
  **同一ファイル → 同一ディレクトリ → プロジェクト全体で一意** の順で解決し、
  一意でなければ張らない (誤った辺は無い辺より害が大きい)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

from backend.free.rag.corpus.language import ImportRule
from backend.free.rag.projectmap.aliases import AliasConfig
from backend.free.rag.projectmap.ids import code_node_id

#: JS/TS 系の相対 import に補完する拡張子 (順に試す)。``.svelte``/``.vue`` を
#: 足すことで、既存の JS/TS ファイルが SFC を import する場合も解決できる
#: (c_16 §4.4 マークアップ / SFC)。
_JS_LIKE_EXTENSIONS: tuple[str, ...] = (
    "", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".svelte", ".vue",
)
#: ``import "./dir"`` が index ファイルを指す場合の候補。
_JS_INDEX_NAMES: tuple[str, ...] = (
    "index.ts", "index.tsx", "index.js", "index.jsx",
)

#: Python の絶対 import 解決で使う候補サフィックス。
_PY_MODULE_SUFFIXES: tuple[str, ...] = ("", "/__init__")


@dataclass(frozen=True, slots=True)
class Node:
    """コードグラフの 1 ノード (c_16 §3.5 ``code_node`` に対応)。"""

    id: str
    node_type: str
    path: str
    name: str
    qualname: str
    lang: str
    line_start: int
    line_end: int
    parent_id: str | None
    signature: str
    text: str


@dataclass(frozen=True, slots=True)
class Edge:
    """コードグラフの 1 辺。``etype`` は ``contains|imports|calls|inherits``。"""

    src: str
    dst: str
    etype: str
    weight: float


@dataclass(slots=True)
class ProjectGraph:
    """1 回の走査で組み上がったプロジェクト全体のグラフ。"""

    nodes: list[Node]
    edges: list[Edge]
    external_imports: dict[str, list[str]]
    fingerprints: dict[str, str]
    languages: dict[str, int]


@dataclass(frozen=True, slots=True)
class RawCall:
    """抽出器が返す未解決の呼出し (同一ファイル内の呼出し元/呼出し名)。"""

    #: 呼出し元ノードの id (function / method)。呼出し元が特定できない
    #: (定義の外側) 場合は抽出器が捨てる。
    caller_id: str
    callee_name: str
    #: 受け手。``None`` = 素の名前の呼出 (``f()``)、``"self"`` / ``"mod.sub"`` = 名前を
    #: ``.`` で繋いだ受け手 (``self.f()`` / ``mod.sub.f()``)、``"super"`` = ``super().f()`` /
    #: ``super.f()``、``""`` = それ以外の受け手 (呼出の結果・添字など)。受け手を記録しない
    #: 言語は常に ``None`` (c_16 §4.4)。
    receiver: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedFile:
    """1 ファイル分の抽出結果 (file ノードは含まない)。"""

    path: str
    lang: str
    #: ファイルの行数 (file ノードの ``line_end``)。
    line_count: int
    #: class / function / method ノード (``parent_id`` は同一ファイル内の
    #: 親ノード id、無ければ ``None`` = ファイル直下)。
    nodes: list[Node]
    #: 生の import 指定子 (未解決、言語ごとの表記のまま)。
    imports: list[str]
    calls: list[RawCall]
    #: ``(クラスノード id, 基底名)``。基底名は未解決 (build_graph 側で解決)。
    inherits: list[tuple[str, str]] = field(default_factory=list)
    #: Python の ``from M import name`` の ``(M, name)`` (絶対 import のみ)。
    #: ``M.name`` がプロジェクト内のモジュールなら、そのファイルへも ``imports`` 辺を張る。
    import_names: list[tuple[str, str]] = field(default_factory=list)
    #: Python の import が束縛した名前 → ``(モジュール, メンバ or None)``。
    #: ``import a.b as c`` → ``c: ("a.b", None)``、``import a.b`` → ``a: ("a", None)``、
    #: ``from m import x as y`` → ``y: ("m", "x")`` (絶対 import のみ)。
    import_bindings: dict[str, tuple[str, str | None]] = field(default_factory=dict)


def _file_node(file: ExtractedFile) -> Node:
    name = PurePosixPath(file.path).name
    return Node(
        id=code_node_id(file.path, "file", ""),
        node_type="file",
        path=file.path,
        name=name,
        qualname=file.path,
        lang=file.lang,
        line_start=1,
        line_end=max(1, file.line_count),
        parent_id=None,
        signature="",
        text=name,
    )


def _resolve_python_import(module: str, all_paths: frozenset[str]) -> str | None:
    base = module.strip().replace(".", "/")
    if not base:
        return None
    for suffix in _PY_MODULE_SUFFIXES:
        candidate = f"{base}{suffix}.py"
        if candidate in all_paths:
            return candidate
    return None


def _resolve_js_import(
    source_dir: PurePosixPath, spec: str, all_paths: frozenset[str],
) -> str | None:
    raw = spec.strip().strip("'\"`")
    if not raw.startswith((".", "/")):
        return None  # bare specifier = 外部パッケージ
    target = (source_dir / raw) if raw.startswith(".") else PurePosixPath(raw.lstrip("/"))
    normalized = PurePosixPath(_normalize_posix(str(target)))
    for ext in _JS_LIKE_EXTENSIONS:
        candidate = f"{normalized}{ext}"
        if candidate in all_paths:
            return candidate
    for index_name in _JS_INDEX_NAMES:
        candidate = str(normalized / index_name)
        if candidate in all_paths:
            return candidate
    return None


def _normalize_posix(path: str) -> str:
    """``a/b/../c`` のような相対参照を畳む (``..`` / ``.`` を解決する)。"""
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _rust_crate_src_root(file_path: str) -> str:
    """``crate::`` パスの基準ディレクトリ (最も近い祖先の ``src/``)。

    ``Cargo.toml`` はソース言語ファイルではないため走査対象に入らず、実体は
    読めない。ファイル自身の path から最も近い ``src`` ディレクトリを crate
    ルートとみなす (単一 crate は ``src/`` 直下、workspace は
    ``crates/<name>/src/`` でも成立する)。見つからなければ走査ルート直下の
    ``src`` とみなす (c_16 §4.4 の規則どおり)。
    """
    parts = PurePosixPath(file_path).parent.parts
    if "src" in parts:
        idx = len(parts) - 1 - parts[::-1].index("src")
        return "/".join(parts[: idx + 1])
    return "src"


def _rust_own_module_dir(file_path: str) -> str:
    """このファイルが属するモジュールのサブモジュール探索ディレクトリ。

    ``a/b.rs`` (フラット形式) と ``a/mod.rs`` / ``main.rs`` / ``lib.rs``
    (mod.rs 形式) のどちらでも、そのモジュールの子モジュールは同じ規則
    (``a/b/...``) で並ぶようにする。
    """
    path = PurePosixPath(file_path)
    if path.stem in ("mod", "main", "lib"):
        return str(path.parent)
    return str(path.parent / path.stem)


def _rust_module_candidates(base_dir: str, segments: Sequence[str]) -> list[str]:
    """``base_dir`` 配下でモジュール ``segments`` を定義しうるファイル候補。"""
    joined = "/".join([base_dir, *segments]) if segments else base_dir
    candidates = [f"{joined}.rs", f"{joined}/mod.rs"]
    if not segments:
        # crate ルート / 親モジュール自身の定義ファイル (item が直接その
        # ファイルに書かれている場合)。
        candidates += [f"{base_dir}/main.rs", f"{base_dir}/lib.rs"]
    return candidates


def _resolve_rust_import(
    file_path: str, spec: str, all_paths: frozenset[str],
) -> list[str]:
    """Rust の ``use`` / bodyless ``mod`` を解決する。

    ``mod a;`` は現ファイルのサブモジュールディレクトリ基準、``use crate::…``
    は crate ルート基準、``use self::…`` は現モジュール基準、``use super::…``
    は親モジュール基準。曖昧な場合 (どの候補も見つからない) は解決しない。
    """
    text = spec.strip()
    if text.startswith("mod "):
        name = text[len("mod "):].strip()
        if not name:
            return []
        base_dir = _rust_own_module_dir(file_path)
        for candidate in _rust_module_candidates(base_dir, [name]):
            if candidate in all_paths:
                return [candidate]
        return []

    exact = False  # ``{...}`` / ``::*`` は前段までが丸ごとモジュール path
    if "{" in text:
        text = text[: text.index("{")].rstrip(":")
        exact = True
    if text.endswith("::*"):
        text = text[:-3].rstrip(":")
        exact = True
    if " as " in text:
        text = text.split(" as ", 1)[0].strip()

    segments = [s for s in text.split("::") if s]
    if not segments:
        return []
    head, *rest = segments
    if head == "crate":
        base_dir = _rust_crate_src_root(file_path)
    elif head == "self":
        base_dir = _rust_own_module_dir(file_path)
    elif head == "super":
        base_dir = str(PurePosixPath(_rust_own_module_dir(file_path)).parent)
    else:
        return []  # 外部 crate (std / core / サードパーティ) は external
    if not rest:
        return []

    # 末尾セグメントが item (関数/型) か submodule かは構文だけでは決まらない
    # ため、具体的な候補から緩めていく (誤った辺は無い辺より害が大きい —
    # どれも実在しなければ解決しない)。
    tiers: list[list[str]] = [rest]
    if not exact:
        if len(rest) > 1:
            tiers.append(rest[:-1])
        if len(rest) > 2:
            tiers.append(rest[:1])
        tiers.append([])
    for segs in tiers:
        for candidate in _rust_module_candidates(base_dir, segs):
            if candidate in all_paths:
                return [candidate]
    return []


def _resolve_go_import(spec: str, all_paths: frozenset[str]) -> list[str]:
    """Go の import path を解決する。

    ``go.mod`` はソース言語ファイルではなく走査対象外なので ``module`` 行は
    読めない。先頭要素にドットが無ければ標準ライブラリ (external)。それ以外
    は、先頭から 1 つ以上のセグメントを prefix として落としたときに
    プロジェクト内の実在ディレクトリ (``.go`` ファイルを含む、``_test.go``
    除く) に一意に一致する場合だけ解決する — 1 つの import は 1 パッケージ
    (ディレクトリ) なので、そのディレクトリの全 ``.go`` ファイルへ辺を張る。
    """
    raw = spec.strip().strip('"')
    segments = raw.split("/")
    if not segments or "." not in segments[0]:
        return []  # 標準ライブラリ (先頭要素にドット無し) は external
    by_dir: dict[str, list[str]] = defaultdict(list)
    for path in all_paths:
        if path.endswith(".go") and not path.endswith("_test.go"):
            by_dir[str(PurePosixPath(path).parent)].append(path)
    matched_dirs: set[str] = set()
    for drop in range(1, len(segments)):
        suffix = "/".join(segments[drop:])
        if suffix in by_dir:
            matched_dirs.add(suffix)
    if len(matched_dirs) != 1:
        return []  # go.mod が無く prefix を確定できない / 複数一致で曖昧
    return sorted(by_dir[matched_dirs.pop()])


def _resolve_ruby_import(
    file_path: str, spec: str, all_paths: frozenset[str],
) -> list[str]:
    """Ruby の ``require`` / ``require_relative`` を解決する。

    spec は ``treesitter._import_specs`` が ``'<method> "<引数>"'`` 形式で
    渡す (例: ``'require_relative "./foo"'``、実ソースに近い形)。
    """
    kind, _, rest = spec.strip().partition(" ")
    raw = rest.strip().strip('"')
    if not raw:
        return []
    if kind == "require_relative":
        target = PurePosixPath(file_path).parent / raw
        candidate = f"{_normalize_posix(str(target))}.rb"
        return [candidate] if candidate in all_paths else []
    if kind == "require":
        candidate = f"lib/{raw}.rb"
        return [candidate] if candidate in all_paths else []
    return []


def _java_kotlin_ends_with(path: str, suffix: str) -> bool:
    return path == suffix or path.endswith(f"/{suffix}")


def _resolve_java_kotlin_import(spec: str, all_paths: frozenset[str]) -> list[str]:
    """Java ``import`` / Kotlin ``import`` を解決する。

    ``src/main/java`` 等のソースルート prefix は問わず、ルート配下のどこかに
    ``**/a/b/C.java`` または ``**/a/b/C.kt`` が一意に見つかれば張る
    (Java/Kotlin 相互参照もあり得るため両拡張子を探す)。``java.*`` /
    ``kotlin.*`` / ``android.*`` / ``javax.*`` は external。
    """
    raw = spec.strip()
    if raw.endswith(".*"):
        raw = raw[:-2]
    segments = [s for s in raw.split(".") if s]
    if not segments:
        return []
    if segments[0] in ("java", "javax", "kotlin", "android"):
        return []
    # static import (末尾がメソッド/フィールド名で小文字始まり) はクラス名
    # まで遡った候補も試す。
    tier_options = [segments]
    if len(segments) > 1 and not segments[-1][:1].isupper():
        tier_options.append(segments[:-1])
    for segs in tier_options:
        base = "/".join(segs)
        matches = {
            path for path in all_paths
            if _java_kotlin_ends_with(path, f"{base}.java")
            or _java_kotlin_ends_with(path, f"{base}.kt")
        }
        if len(matches) == 1:
            return [matches.pop()]
        if matches:
            return []  # 複数一致 = 曖昧、張らない
    return []


_PHP_INCLUDE_KEYWORDS: frozenset[str] = frozenset({
    "require", "require_once", "include", "include_once",
})


def _resolve_php_import(
    file_path: str, spec: str, all_paths: frozenset[str],
) -> list[str]:
    """PHP の ``require``/``require_once``/``include``/``include_once`` を解決する。

    spec は ``treesitter._import_specs`` が ``"<keyword> '<path>'"`` 形式で
    渡す (実ソースに近い形)。``use A\\B\\C;`` (namespace) は PSR-4 の実配置が
    分からないので常に external — その spec はこの語彙に一致しないのでここで
    弾かれる。
    """
    keyword, _, rest = spec.strip().partition(" ")
    if keyword not in _PHP_INCLUDE_KEYWORDS:
        return []
    raw = rest.strip().strip("'")
    if not raw:
        return []
    target = PurePosixPath(file_path).parent / raw
    candidate = _normalize_posix(str(target))
    if not candidate.endswith(".php"):
        candidate = f"{candidate}.php"
    return [candidate] if candidate in all_paths else []


def _resolve_c_import(
    file_path: str, spec: str, all_paths: frozenset[str],
) -> list[str]:
    """C/C++ の ``#include`` を解決する。

    ``<...>`` (システムヘッダ) は ``treesitter._import_specs`` が山括弧付き
    のまま渡すので常に unresolved になる。``"..."`` (相対) は現ファイル基準
    → ルート基準の順で試す。
    """
    if spec.startswith("<"):
        return []
    source_dir = PurePosixPath(file_path).parent
    candidates = [
        _normalize_posix(str(source_dir / spec)),
        _normalize_posix(spec),
    ]
    for candidate in candidates:
        if candidate in all_paths:
            return [candidate]
    return []


#: マークアップ / SFC の指定子から落とすもの (c_16 §4.4)。
_EXTERNAL_SPEC_PREFIXES: tuple[str, ...] = ("http://", "https://", "//", "data:")


def _strip_query_and_hash(spec: str) -> str:
    """クエリ文字列 (``?v=1``) / フラグメント (``#hash``) を落とす。"""
    for ch in ("?", "#"):
        idx = spec.find(ch)
        if idx != -1:
            spec = spec[:idx]
    return spec


def _is_external_spec(spec: str) -> bool:
    return spec.lower().startswith(_EXTERNAL_SPEC_PREFIXES)


def _resolve_relative_or_root(
    source_dir: PurePosixPath, spec: str,
) -> str:
    """相対 (現ファイル基準) / ``/`` 始まり (走査ルート基準) を正規化する。"""
    target = PurePosixPath(spec.lstrip("/")) if spec.startswith("/") else (source_dir / spec)
    return _normalize_posix(str(target))


def _resolve_html_import(
    file_path: str, spec: str, all_paths: frozenset[str],
) -> str | None:
    """HTML の ``<script src>``/``<link href>`` を解決する (c_16 §4.4)。

    拡張子補完はしない (src/href は実ファイル名をそのまま指す)。
    """
    raw = _strip_query_and_hash(spec.strip())
    if not raw or raw.startswith("#") or _is_external_spec(raw):
        return None
    source_dir = PurePosixPath(file_path).parent
    normalized = _resolve_relative_or_root(source_dir, raw)
    return normalized if normalized in all_paths else None


def _scss_candidates(normalized: str) -> list[str]:
    """SCSS partial 補完 (``_x.scss``) + 拡張子省略の候補。"""
    path = PurePosixPath(normalized)
    stem = path.name
    parent = path.parent
    candidates = [f"{normalized}.scss"]
    if not stem.startswith("_"):
        prefixed = f"_{stem}.scss"
        candidates.append(prefixed if str(parent) == "." else str(parent / prefixed))
    return candidates


def _resolve_css_import(
    file_path: str, lang: str, spec: str, all_paths: frozenset[str],
) -> str | None:
    """CSS/SCSS の ``@import``/``@use``/``@forward``/``url()`` を解決する。

    相対 = 現ファイル基準、``/`` 始まり = 走査ルート基準。SCSS だけ
    ``_x.scss`` の partial 補完 + 拡張子省略を試す (c_16 §4.4)。
    """
    raw = _strip_query_and_hash(spec.strip())
    if not raw or raw.startswith("#") or _is_external_spec(raw):
        return None
    source_dir = PurePosixPath(file_path).parent
    normalized = _resolve_relative_or_root(source_dir, raw)
    if normalized in all_paths:
        return normalized
    if lang == "scss":
        for candidate in _scss_candidates(normalized):
            if candidate in all_paths:
                return candidate
    return None


#: 言語パック (c_16 §4.5.3、段階 C-2) の解決で使う基準ディレクトリの種類。
_PACK_RELATIVE = "relative"
_PACK_ROOT_RELATIVE = "root_relative"


def _pack_normalize(base_dir: str, raw_path: str) -> str | None:
    """``base_dir`` (posix、``.``/空文字列 = ルート) に ``raw_path`` を足して畳む。

    ``..`` が積み上げた部品より多ければ走査ルートの外へ出るので ``None``
    (c_16 §4.4 の ``imports`` と同じ「ルートの外は external」規則)。
    :func:`_normalize_posix` は積み上げが空のときの ``..`` を黙って捨てるため
    (既存の言語別解決と共有)、パック言語の解決はこちらの専用実装を使う。
    """
    parts = [p for p in base_dir.split("/") if p and p != "."]
    for part in raw_path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _pack_base_dir(mode: str, source_dir: PurePosixPath) -> str:
    return "" if mode == _PACK_ROOT_RELATIVE else str(source_dir)


def _pack_base_candidate(mode: str, source_dir: PurePosixPath, spec: str) -> str | None:
    raw = spec.lstrip("/") if mode == _PACK_ROOT_RELATIVE else spec
    return _pack_normalize(_pack_base_dir(mode, source_dir), raw)


def _pack_resolve_sequence(
    steps: Sequence[str], spec: str, source_dir: PurePosixPath,
    all_paths: frozenset[str], rule: ImportRule,
) -> list[str] | None:
    """``resolve`` の語の列を順に試す (c_16 §4.5.3、段階 C-2)。

    ``dotted_to_path`` に出会ったら ``.`` を ``/`` に置き換え、**それより前に
    並んでいる語** (``steps`` の自分より手前のスライス) だけを新しい指定子で
    再度試す (仕様の「上の規則へ」)。他の語は「一意に 1 件だけ実在すれば
    採用」— 0 件・複数件は次の語へ進む (誤った辺は無い辺より害が大きい、
    c_16 §4.4 と同じ判断)。
    """
    mode = _PACK_RELATIVE
    for i, step in enumerate(steps):
        if step == "dotted_to_path":
            transformed = spec.replace(".", "/")
            result = _pack_resolve_sequence(steps[:i], transformed, source_dir, all_paths, rule)
            if result is not None:
                return result
            continue
        if step in (_PACK_RELATIVE, _PACK_ROOT_RELATIVE):
            mode = step
            base = _pack_base_candidate(mode, source_dir, spec)
            if base is not None and base in all_paths:
                return [base]
            continue
        if step == "extension_completion":
            base = _pack_base_candidate(mode, source_dir, spec)
            if base is None:
                continue
            matches = sorted({
                f"{base}{ext}" for ext in rule.extension_completion
                if f"{base}{ext}" in all_paths
            })
            if len(matches) == 1:
                return matches
            continue
        if step == "index_file":
            base = _pack_base_candidate(mode, source_dir, spec)
            if base is None:
                continue
            matches = sorted({
                (f"{base}/{name}" if base else name)
                for name in rule.index_file
                if (f"{base}/{name}" if base else name) in all_paths
            })
            if len(matches) == 1:
                return matches
            continue
    return None


def _resolve_pack_import(
    file_path: str, spec: str, all_paths: frozenset[str], rule: ImportRule,
) -> list[str]:
    """言語パック (c_16 §4.5.3、段階 C-2) の import 指定子を解決する。

    ``rule.external_prefixes`` に前方一致すれば external。それ以外は
    ``rule.resolve`` の語を順に試し、一意に決まったものだけを採る。
    """
    raw = spec.strip()
    if not raw or any(raw.startswith(prefix) for prefix in rule.external_prefixes):
        return []
    source_dir = PurePosixPath(file_path).parent
    result = _pack_resolve_sequence(rule.resolve, raw, source_dir, all_paths, rule)
    # 自己参照の除外は呼出側 (``build_graph``) が全言語共通で行う。
    return result if result else []


def _resolve_via_alias(
    file_path: str, spec: str, all_paths: frozenset[str], alias_config: AliasConfig,
) -> str | None:
    """``kit.alias`` / tsconfig・jsconfig ``paths`` / SvelteKit ``$lib`` を
    解決する (c_16 §4.4)。

    優先順位は ``kit.alias`` → ``paths`` → ``$lib`` — ``kit.alias`` が
    ``$lib`` を上書きしていれば ``$lib`` より先に試されるのでそちらが勝つ。
    どの候補も、既存の相対 import と同じ拡張子補完 / index file 解決
    (:func:`_resolve_js_import`) にそのまま流す。一致するエイリアスが
    無ければ ``None`` (呼出側は従来どおりの解決へフォールバックする —
    エイリアスを使っていない指定子の挙動は変わらない)。
    """
    source_dir = PurePosixPath(file_path).parent
    candidates: list[str] = []
    candidates.extend(alias_config.kit_alias_candidates(file_path, spec))
    candidates.extend(alias_config.ts_paths_candidates(file_path, spec))
    svelte_candidate = alias_config.svelte_lib_candidate(file_path, spec)
    if svelte_candidate is not None:
        candidates.append(svelte_candidate)
    for candidate_spec in candidates:
        target = _resolve_js_import(source_dir, candidate_spec, all_paths)
        if target is not None:
            return target
    return None


def _resolve_import(
    file_path: str, lang: str, spec: str, all_paths: frozenset[str],
    *,
    pack_import_rules: Mapping[str, ImportRule] | None = None,
    alias_config: AliasConfig | None = None,
    py_modules: "_PythonModuleIndex | None" = None,
) -> list[str]:
    if lang == "python":
        if py_modules is not None:
            target = py_modules.resolve(spec, file_path)
        else:
            target = _resolve_python_import(spec, all_paths)
        return [target] if target is not None else []
    if lang in ("javascript", "typescript", "tsx", "svelte", "vue") and alias_config is not None:
        aliased = _resolve_via_alias(file_path, spec, all_paths, alias_config)
        if aliased is not None:
            return [aliased]
    if lang in ("javascript", "typescript", "tsx"):
        source_dir = PurePosixPath(file_path).parent
        target = _resolve_js_import(source_dir, spec, all_paths)
        return [target] if target is not None else []
    if lang in ("svelte", "vue"):
        # script 側 (JS/TS 風 import) を先に試し、駄目なら style 側
        # (CSS 風 @import/url()) を試す — SFC の imports には両方の書式が
        # 混ざりうる (c_16 §4.4)。
        source_dir = PurePosixPath(file_path).parent
        js_target = _resolve_js_import(source_dir, spec, all_paths)
        if js_target is not None:
            return [js_target]
        css_target = _resolve_css_import(file_path, "css", spec, all_paths)
        return [css_target] if css_target is not None else []
    if lang == "html":
        target = _resolve_html_import(file_path, spec, all_paths)
        return [target] if target is not None else []
    if lang in ("css", "scss"):
        target = _resolve_css_import(file_path, lang, spec, all_paths)
        return [target] if target is not None else []
    if lang == "rust":
        return _resolve_rust_import(file_path, spec, all_paths)
    if lang == "go":
        return _resolve_go_import(spec, all_paths)
    if lang == "ruby":
        return _resolve_ruby_import(file_path, spec, all_paths)
    if lang in ("java", "kotlin"):
        return _resolve_java_kotlin_import(spec, all_paths)
    if lang == "php":
        return _resolve_php_import(file_path, spec, all_paths)
    if lang in ("c", "cpp"):
        return _resolve_c_import(file_path, spec, all_paths)
    if lang == "swift":
        return []  # モジュール名が path に落ちないため常に external
    # 言語パック (c_16 §4.5.3、段階 C-2)。``lang`` は同梱表に無いのでここまで
    # 落ちる — manifest の ``imports`` を宣言していない言語は従来どおり空。
    rule = (pack_import_rules or {}).get(lang)
    if rule is None:
        return []
    return _resolve_pack_import(file_path, spec, all_paths, rule)


def _unique_by_name(
    candidates: Iterable[Node], name: str,
) -> Node | None:
    matches = [n for n in candidates if n.name == name]
    return matches[0] if len(matches) == 1 else None


#: 名前一致だけで張る辺 (``calls`` / ``inherits``) の weight (c_16 §4.4)。
#: ``contains`` / ``imports`` (構文木 / パスから確定) の 1.0 より低く置く —
#: 型情報を見ない名前一致は、一意に決まっても「たぶんこれ」でしかない。
NAME_MATCH_WEIGHT = 0.5

#: 名前一致の相手を同じ集合に限る言語ファミリ (c_16 §4.4)。載っていない言語はそれ自身が 1 ファミリ。
_LANGUAGE_FAMILY: dict[str, str] = {
    "javascript": "web", "typescript": "web", "tsx": "web", "svelte": "web", "vue": "web",
    "c": "c", "cpp": "c",
}
#: import の解決が揃っていて、import していないファイルの名前を呼べない言語ファミリ。
#: 同一ファイルの次は import 先だけを見る (同一ディレクトリ・全体で一意の段を使わない)。
_IMPORT_SCOPED_FAMILIES = frozenset({"python", "web"})
#: パッケージの入口。ここを import したら、入口が import しているファイルも 1 段だけ見る (再エクスポート)。
_PACKAGE_ENTRY_NAMES = frozenset({
    "__init__.py", "index.ts", "index.tsx", "index.js", "index.jsx", "index.mjs",
})


def language_family(lang: str) -> str:
    """名前一致の解決で同じ集合として扱う言語の組 (c_16 §4.4)。"""
    return _LANGUAGE_FAMILY.get(lang, lang)


#: 受け手が「自分のインスタンス / クラス」を指す名前 (``self.m()`` / ``this.m()``)。
_SELF_RECEIVERS = frozenset({"self", "cls", "this"})
#: ``super().m()`` / ``super.m()`` の受け手 (抽出器がこの文字列で残す)。
_SUPER_RECEIVER = "super"
#: 継承の連鎖を辿る深さの上限 (循環・病的な階層の保険)。
_MAX_BASE_DEPTH = 8


#: SvelteKit が実行時に提供するモジュール (aliases.py と同じく外部扱い)。
_SVELTEKIT_EXTERNAL_PREFIXES = ("$app/", "$env/", "$service-worker")


def _looks_project_local(spec: str) -> bool:
    """js の import 指定子がプロジェクト内を指す形か (相対・ルート相対・エイリアス)。

    解決できなかったときに「外部パッケージ」と「エイリアスを解けなかったプロジェクト内」
    を分けるのに使う (後者は全体で一意の名前一致へ縮退する、c_16 §4.4)。
    """
    raw = spec.strip().strip("'\"`")
    if raw.startswith(_SVELTEKIT_EXTERNAL_PREFIXES):
        return False  # SvelteKit が提供するモジュール (外部)
    return raw.startswith((".", "/", "$", "~", "@/"))


class _PythonModuleIndex:
    """ドット区切りのモジュール名 → プロジェクト内の path (c_16 §4.4)。

    ルート相対だけでなく、path の末尾が一致するモジュールも候補にする
    (``src/`` レイアウトや、別の木に写しを持つリポジトリ)。候補のうち接頭辞が
    呼出元の path の祖先であるものを優先し (最も長いもの)、無ければ候補が 1 つの
    ドット付きのモジュール名だけ採る。決まらなければ ``None`` (外部とみなす)。

    接頭辞のディレクトリがパッケージ (``__init__.py`` を持つ) なら sys.path の根では
    ないので候補にしない (``backend/`` の中から ``import io`` を ``backend/io/`` へ
    解かない)。標準ライブラリの名前はルート相対で一致するときだけ採る。
    """

    def __init__(self, all_paths: Iterable[str]) -> None:
        self._by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)
        paths = sorted(all_paths)
        packages = {p[: -len("/__init__.py")] for p in paths if p.endswith("/__init__.py")}
        for path in paths:
            if not path.endswith(".py"):
                continue
            parts = path[:-3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            for i in range(len(parts)):
                prefix = "/".join(parts[:i])
                if prefix in packages:
                    continue  # パッケージの内側は sys.path の根ではない
                self._by_key[".".join(parts[i:])].append((f"{prefix}/" if prefix else "", path))
        self._cache: dict[tuple[str, str], str | None] = {}

    def resolve(self, module: str, from_path: str) -> str | None:
        module = module.strip()
        if not module:
            return None
        key = (module, from_path)
        if key in self._cache:
            return self._cache[key]
        candidates = self._by_key.get(module, [])
        if module.split(".", 1)[0] in sys.stdlib_module_names:
            candidates = [(p, path) for p, path in candidates if not p]
        enclosing = [(p, path) for p, path in candidates if from_path.startswith(p)]
        if enclosing:
            longest = max(len(p) for p, _ in enclosing)
            best = [path for p, path in enclosing if len(p) == longest]
            # 同じ接頭辞で ``m.py`` と ``m/__init__.py`` が両方あればモジュール本体を採る
            result = sorted(best, key=lambda p: (p.endswith("__init__.py"), p))[0]
        elif len(candidates) == 1 and "." in module:
            result = candidates[0][1]
        else:
            result = None
        self._cache[key] = result
        return result


class _SymbolResolver:
    """呼出し名 / 基底名を名前一致で解決する (c_16 §4.4)。

    候補は呼出元と同じ言語ファミリに限り、入れ子の定義
    (関数の中の関数・クラス) は呼出元を囲むスコープにあるときだけ相手にする。
    一意に決まらなければ ``None`` (辺を張らない — 誤った辺は無い辺より害が大きい)。

    Python / js ファミリ (import を解決できる言語) は呼出の形で相手を分ける:

    - 素の名前 ``f()``: メソッドは相手にしない。同一ファイル → import が束縛した
      名前ならその import 先 (外部なら張らない) → import 先の全ファイル。
    - ``self.f()`` / ``this.f()``: 囲みのクラスと、解決済みの基底クラスの連鎖。
      無ければ import 先のメソッド (別ファイルの mixin)。
    - ``mod.f()`` / ``C.f()`` (受け手が import したモジュール / クラス): そのファイルの
      関数 / そのクラスのメソッド。外部なら張らない。
    - それ以外の受け手 (変数・呼出の結果): 同一ファイル → import 先のメソッドだけ。

    他の言語は同一ファイル → 同一ディレクトリ → 全体の順で一意かを見る。
    """

    def __init__(
        self,
        *,
        by_file: dict[str, list[Node]],
        by_dir: dict[str, list[Node]],
        by_name_global: dict[str, list[Node]],
        imports_of: Mapping[str, set[str]],
        spec_targets: Mapping[str, Mapping[str, str | None]],
        nested_ids: frozenset[str],
        node_by_id: Mapping[str, Node],
        bindings: Mapping[str, Mapping[str, tuple[str, str | None]]],
        py_modules: _PythonModuleIndex,
    ) -> None:
        self.by_dir = by_dir
        self.by_name_global = by_name_global
        self.imports_of = imports_of
        self.spec_targets = spec_targets
        self.nested_ids = nested_ids
        self.node_by_id = node_by_id
        self.bindings = bindings
        self.py_modules = py_modules
        #: 継承の連鎖 (class id → 解決済みの基底ノード)。inherits を解いた後に入れる。
        self.bases: dict[str, list[Node]] = defaultdict(list)
        self._by_name: dict[str, dict[str, list[Node]]] = {}
        self._by_qualname: dict[str, dict[str, list[Node]]] = {}
        for path, nodes in by_file.items():
            names: dict[str, list[Node]] = defaultdict(list)
            qualnames: dict[str, list[Node]] = defaultdict(list)
            for n in nodes:
                names[n.name].append(n)
                qualnames[n.qualname].append(n)
            self._by_name[path] = names
            self._by_qualname[path] = qualnames
        self._hop_cache: dict[tuple[str, ...], tuple[str, ...]] = {}
        self._ancestor_cache: dict[str, frozenset[str]] = {}
        self._owner_cache: dict[str, str] = {}

    # ── 候補の集合 ──

    def _with_entry_hop(self, paths: Iterable[str]) -> tuple[str, ...]:
        """パッケージの入口なら、入口が import しているファイルも足す (再エクスポート、1 段)。"""
        key = tuple(sorted(paths))
        cached = self._hop_cache.get(key)
        if cached is None:
            out = set(key)
            for path in key:
                if PurePosixPath(path).name in _PACKAGE_ENTRY_NAMES:
                    out.update(self.imports_of.get(path, ()))
            cached = self._hop_cache[key] = tuple(sorted(out))
        return cached

    def _ancestors(self, caller: Node) -> frozenset[str]:
        """呼出元と、それを囲むノードの id (入れ子の定義が見えるスコープ)。"""
        cached = self._ancestor_cache.get(caller.id)
        if cached is None:
            out: set[str] = set()
            current: Node | None = caller
            while current is not None:
                out.add(current.id)
                current = self.node_by_id.get(current.parent_id) if current.parent_id else None
            cached = self._ancestor_cache[caller.id] = frozenset(out)
        return cached

    def _scope_owner(self, node: Node) -> str | None:
        """入れ子の定義を囲む最も内側の関数 / メソッド (そこから内側でだけ見える)。"""
        cached = self._owner_cache.get(node.id)
        if cached is None:
            current = self.node_by_id.get(node.parent_id) if node.parent_id else None
            while current is not None and current.node_type not in ("function", "method"):
                current = self.node_by_id.get(current.parent_id) if current.parent_id else None
            cached = self._owner_cache[node.id] = current.id if current is not None else ""
        return cached or None

    def _eligible(
        self, caller: Node, nodes: Iterable[Node], *, exclude_caller: bool = False,
    ) -> list[Node]:
        """``exclude_caller``: 呼出元自身を除く (``super().f()`` のように受け手が式のとき)。

        自分の名前の呼出 (再帰) は呼出元自身が一意な相手になり、辺を張らずに止まる
        (自己辺は build_graph が捨てる) — 除くと import 先の同名へ誤って結ぶ。
        """
        family = language_family(caller.lang)
        out = []
        for n in nodes:
            if (exclude_caller and n.id == caller.id) or language_family(n.lang) != family:
                continue
            if n.id in self.nested_ids and self._scope_owner(n) not in self._ancestors(caller):
                continue
            out.append(n)
        return out

    def _named(self, paths: Iterable[str], name: str) -> Iterator[Node]:
        for path in paths:
            yield from self._by_name.get(path, {}).get(name, ())

    def _unique(
        self, caller: Node, paths: Iterable[str], name: str, *, methods: bool | None,
        exclude_caller: bool = False,
    ) -> Node | None:
        """``methods``: True ならメソッドだけ、False ならメソッド以外、None なら全部。"""
        pool = [
            n for n in self._eligible(
                caller, self._named(paths, name), exclude_caller=exclude_caller,
            )
            if methods is None or (n.node_type == "method") is methods
        ]
        return pool[0] if len(pool) == 1 else None

    def _unique_qualname(self, caller: Node, paths: Iterable[str], qualname: str) -> Node | None:
        hits = self._eligible(
            caller, (n for path in paths for n in self._by_qualname.get(path, {}).get(qualname, ())),
        )
        return hits[0] if len(hits) == 1 else None

    def _imported(self, caller: Node) -> tuple[str, ...]:
        return tuple(
            p for p in self._with_entry_hop(self.imports_of.get(caller.path, ())) if p != caller.path
        )

    def _scoped(
        self, caller: Node, name: str, *, methods: bool | None, exclude_caller: bool = False,
    ) -> Node | None:
        """同一ファイル → import 先 (入口の再エクスポートを 1 段) の順で一意か。"""
        same_file = self._unique(
            caller, (caller.path,), name, methods=methods, exclude_caller=exclude_caller,
        )
        if same_file is not None:
            return same_file
        return self._unique(
            caller, self._imported(caller), name, methods=methods, exclude_caller=exclude_caller,
        )

    def _binding_path(self, caller: Node, module: str) -> tuple[str | None, bool]:
        """束縛のモジュールが指すファイルと、解けなかったのがプロジェクト内らしいか。"""
        if language_family(caller.lang) == "python":
            return self.py_modules.resolve(module, caller.path), False
        target = self.spec_targets.get(caller.path, {}).get(module)
        return target, target is None and _looks_project_local(module)

    # ── 解決 ──

    def resolve(self, caller: Node, name: str, receiver: str | None = None) -> Node | None:
        if language_family(caller.lang) not in _IMPORT_SCOPED_FAMILIES:
            return self._resolve_unscoped(caller, name)
        if receiver is None:
            return self._resolve_bare(caller, name)
        if receiver in _SELF_RECEIVERS:
            return self._resolve_self(caller, name)
        if receiver == _SUPER_RECEIVER:
            return self._resolve_self(caller, name, skip_own_class=True)
        head = receiver.split(".", 1)[0]
        binding = self.bindings.get(caller.path, {}).get(head) if head else None
        if binding is not None:
            return self._resolve_via_import(caller, name, receiver, head, binding)
        # 受け手のある呼出 (``self`` 以外) は再帰ではない — 呼出元自身を相手にしない
        return self._scoped(caller, name, methods=True, exclude_caller=True)

    def _resolve_unscoped(self, caller: Node, name: str) -> Node | None:
        same_file = self._unique(caller, (caller.path,), name, methods=None)
        if same_file is not None:
            return same_file
        directory = str(PurePosixPath(caller.path).parent)
        same_dir = _unique_by_name(self._eligible(caller, self.by_dir.get(directory, ())), name)
        if same_dir is not None:
            return same_dir
        return _unique_by_name(self._eligible(caller, self.by_name_global.get(name, ())), name)

    def _global_function(self, caller: Node, name: str) -> Node | None:
        """エイリアスを解けなかった import の縮退: 全体で一意な (メソッド以外の) 名前。"""
        pool = [
            n for n in self._eligible(caller, self.by_name_global.get(name, ()))
            if n.node_type != "method" and n.id not in self.nested_ids
        ]
        return pool[0] if len(pool) == 1 else None

    def _resolve_bare(self, caller: Node, name: str) -> Node | None:
        same_file = self._unique(caller, (caller.path,), name, methods=False)
        if same_file is not None:
            return same_file
        binding = self.bindings.get(caller.path, {}).get(name)
        if binding is None:
            return self._scoped(caller, name, methods=False)
        module, member = binding
        if member is None:
            return None  # モジュールそのもの (名前空間) は呼べない
        if language_family(caller.lang) == "python" and self.py_modules.resolve(
            f"{module}.{member}", caller.path,
        ):
            return None  # ``from pkg import sub`` の sub (モジュール) は呼べない
        module_path, local_unresolved = self._binding_path(caller, module)
        if module_path is None:
            return self._global_function(caller, member) if local_unresolved else None
        return self._unique_qualname(caller, self._with_entry_hop([module_path]), member)

    def _class_chain(self, cls: Node) -> list[Node]:
        chain, frontier, seen = [], [cls], {cls.id}
        for _ in range(_MAX_BASE_DEPTH):
            if not frontier:
                break
            chain.extend(frontier)
            nxt = []
            for c in frontier:
                for base in self.bases.get(c.id, ()):
                    if base.id not in seen:
                        seen.add(base.id)
                        nxt.append(base)
            frontier = nxt
        return chain

    def _resolve_self(
        self, caller: Node, name: str, *, skip_own_class: bool = False,
    ) -> Node | None:
        """``self.f()`` は囲みのクラスから、``super().f()`` は基底から連鎖を辿る。"""
        current: Node | None = caller
        while current is not None and current.node_type != "class":
            current = self.node_by_id.get(current.parent_id) if current.parent_id else None
        if current is not None:
            chain = self._class_chain(current)
            for cls in chain[1:] if skip_own_class else chain:
                own = self._unique_qualname(caller, (cls.path,), f"{cls.qualname}.{name}")
                if own is not None:
                    return own
        # 基底が外部 / 未解決、または別ファイルの mixin が定義している: import 先のメソッド
        return self._unique(
            caller, self._imported(caller), name, methods=True, exclude_caller=skip_own_class,
        )

    def _resolve_via_import(
        self, caller: Node, name: str, receiver: str, head: str,
        binding: tuple[str, str | None],
    ) -> Node | None:
        module, member = binding
        rest = receiver[len(head):]  # 受け手が ``head.b.c`` のときの ``.b.c``
        is_python = language_family(caller.lang) == "python"
        if member is not None:
            submodule = f"{module}.{member}"
            if is_python and self.py_modules.resolve(submodule, caller.path) is not None:
                module = submodule
            else:
                # ``from m import C`` / ``import { C } from 'm'`` の ``C.f()`` (クラスのメソッド)。
                # ``C.x.f()`` は決めない
                module_path, _ = self._binding_path(caller, module)
                if module_path is None or rest:
                    return None
                return self._unique_qualname(
                    caller, self._with_entry_hop([module_path]), f"{member}.{name}",
                )
        if is_python:
            module_path = self.py_modules.resolve(f"{module}{rest}", caller.path)
        else:
            module_path = None if rest else self._binding_path(caller, module)[0]
        if module_path is None:
            return None  # 外部モジュールの関数 (``json.dumps()`` / ``asyncio.run()``)
        return self._unique_qualname(caller, self._with_entry_hop([module_path]), name)


def build_graph(
    extracted: Sequence[ExtractedFile],
    *,
    fingerprints: dict[str, str],
    pack_import_rules: Mapping[str, ImportRule] | None = None,
    alias_config: AliasConfig | None = None,
) -> ProjectGraph:
    """1 プロジェクト分の抽出結果からグラフを組む。

    Args:
        extracted: ファイルごとの抽出結果 (path 昇順で渡すこと)。
        fingerprints: ``{path: sha256}`` (差分検出用にそのまま持ち回る)。
        pack_import_rules: 言語パック (c_16 §4.5.3、段階 C-2) の
            ``{lang_id: ImportRule}``。``lang_id`` は ``ExtractedFile.lang``
            と一致するもの (呼出側の manifest エントリ id)。
        alias_config: ``$lib`` / tsconfig・jsconfig ``paths`` のエイリアス
            解決テーブル (c_16 §4.4)。``None`` なら従来どおりエイリアスは
            解決しない (呼出元が設定ファイルを 1 つも見つけなかった場合)。
    """
    languages: dict[str, int] = defaultdict(int)
    all_paths = frozenset(f.path for f in extracted)

    nodes: list[Node] = []
    edges: list[Edge] = []
    external_imports: dict[str, list[str]] = {}

    by_file: dict[str, list[Node]] = defaultdict(list)
    by_dir: dict[str, list[Node]] = defaultdict(list)
    by_name_global: dict[str, list[Node]] = defaultdict(list)
    call_sites: list[tuple[Node, str, str | None]] = []
    inherit_sites: list[tuple[Node, str]] = []
    node_by_id: dict[str, Node] = {}
    bindings: dict[str, dict[str, tuple[str, str | None]]] = {}
    #: file path → import 指定子 → 解決した path (一意に決まらなければ None)
    spec_targets: dict[str, dict[str, str | None]] = defaultdict(dict)
    py_modules = _PythonModuleIndex(all_paths)
    #: file path → import 辺で結んだ先の path (解決済みのものだけ)
    imports_of: dict[str, set[str]] = defaultdict(set)
    #: 親が function / method のノード (入れ子の定義)
    nested_ids: set[str] = set()

    for file in extracted:
        languages[file.lang] += 1
        file_node = _file_node(file)
        nodes.append(file_node)
        id_to_node: dict[str, Node] = {}
        for raw_node in file.nodes:
            resolved = (
                raw_node if raw_node.parent_id is not None
                else replace(raw_node, parent_id=file_node.id)
            )
            nodes.append(resolved)
            id_to_node[resolved.id] = resolved
            node_by_id[resolved.id] = resolved
            parent = id_to_node.get(resolved.parent_id) if resolved.parent_id else None
            if parent is not None and (
                parent.node_type in ("function", "method") or parent.id in nested_ids
            ):
                nested_ids.add(resolved.id)
            edges.append(
                Edge(src=resolved.parent_id, dst=resolved.id, etype="contains", weight=1.0),
            )
            by_file[resolved.path].append(resolved)
            by_dir[str(PurePosixPath(resolved.path).parent)].append(resolved)
            by_name_global[resolved.name].append(resolved)

        unresolved: list[str] = []
        for spec in file.imports:
            targets = [
                t for t in _resolve_import(
                    file.path, file.lang, spec, all_paths,
                    pack_import_rules=pack_import_rules, alias_config=alias_config,
                    py_modules=py_modules,
                )
                if t != file.path
            ]
            spec_targets[file.path][spec] = targets[0] if len(targets) == 1 else None
            if not targets:
                unresolved.append(spec)
                continue
            for target_path in targets:
                if target_path in imports_of[file.path]:
                    continue
                imports_of[file.path].add(target_path)
                target_id = code_node_id(target_path, "file", "")
                edges.append(
                    Edge(src=file_node.id, dst=target_id, etype="imports", weight=1.0),
                )
        # ``from pkg import sub`` の ``sub`` がモジュールなら、そのファイルへも辺を張る
        # (``pkg/__init__.py`` への辺だけだと、呼出の相手が入口に限られる)。``pkg`` が
        # 解けなくても (``__init__.py`` の無い名前空間パッケージ) sub が解ければ外部ではない。
        for module, member in file.import_names:
            target_path = py_modules.resolve(f"{module}.{member}", file.path)
            if target_path is None or target_path == file.path:
                continue
            if module in unresolved:
                unresolved.remove(module)
            if target_path in imports_of[file.path]:
                continue
            imports_of[file.path].add(target_path)
            edges.append(Edge(
                src=file_node.id, dst=code_node_id(target_path, "file", ""),
                etype="imports", weight=1.0,
            ))
        if unresolved:
            external_imports[file.path] = unresolved

        if file.import_bindings:
            bindings[file.path] = dict(file.import_bindings)
        for call in file.calls:
            caller = id_to_node.get(call.caller_id)
            if caller is not None:
                call_sites.append((caller, call.callee_name, call.receiver))

        for class_id, base_name in file.inherits:
            cls_node = id_to_node.get(class_id)
            if cls_node is not None:
                inherit_sites.append((cls_node, base_name))

    resolver = _SymbolResolver(
        by_file=by_file, by_dir=by_dir, by_name_global=by_name_global,
        imports_of=imports_of, spec_targets=spec_targets, nested_ids=frozenset(nested_ids),
        node_by_id=node_by_id, bindings=bindings, py_modules=py_modules,
    )

    # 継承を先に解く — ``self.f()`` が基底クラスの連鎖を辿れるように
    for cls_node, base_name in inherit_sites:
        target = resolver.resolve(cls_node, base_name)
        if target is not None and target.id != cls_node.id:
            resolver.bases[cls_node.id].append(target)
            edges.append(
                Edge(src=cls_node.id, dst=target.id, etype="inherits", weight=NAME_MATCH_WEIGHT),
            )

    for caller, name, receiver in call_sites:
        target = resolver.resolve(caller, name, receiver)
        if target is not None and target.id != caller.id:
            edges.append(
                Edge(src=caller.id, dst=target.id, etype="calls", weight=NAME_MATCH_WEIGHT),
            )

    nodes.sort(key=lambda n: (n.path, n.line_start, n.node_type, n.qualname))
    return ProjectGraph(
        nodes=nodes,
        edges=edges,
        external_imports=external_imports,
        fingerprints=dict(fingerprints),
        languages=dict(languages),
    )


def validate_graph(graph: ProjectGraph) -> list[str]:
    """決定論で検査できる範囲の整合性チェック (c_16 §4.4)。

    違反があれば版を書かない (呼出側の責務)。空リストなら妥当。

    - 辺の両端が実在する
    - id が一意 (衝突は「別々のシンボルが同じ同一性鍵に落ちた」バグの印)
    - file ノードは 1 つの path に 1 つ
    - ``parent_id`` の連鎖が file に到達する (循環 / 迷子が無い)
    - 自己辺が無い
    """
    violations: list[str] = []
    by_id: dict[str, Node] = {}
    for node in graph.nodes:
        if node.id in by_id:
            violations.append(f"duplicate node id: {node.id} ({node.path}:{node.qualname})")
        else:
            by_id[node.id] = node

    file_paths: dict[str, int] = defaultdict(int)
    for node in graph.nodes:
        if node.node_type == "file":
            file_paths[node.path] += 1
    for path, count in file_paths.items():
        if count != 1:
            violations.append(f"path {path!r} has {count} file node(s), expected 1")

    for edge in graph.edges:
        if edge.src not in by_id:
            violations.append(f"edge references unknown src: {edge.src}")
        if edge.dst not in by_id:
            violations.append(f"edge references unknown dst: {edge.dst}")
        if edge.src == edge.dst:
            violations.append(f"self edge on {edge.src} ({edge.etype})")

    for node in graph.nodes:
        if node.node_type == "file":
            continue
        seen: set[str] = set()
        current: Node | None = node
        reached_file = False
        cyclic = False
        while current is not None:
            if current.id in seen:
                cyclic = True
                break
            seen.add(current.id)
            if current.node_type == "file":
                reached_file = True
                break
            current = by_id.get(current.parent_id) if current.parent_id else None
        if cyclic:
            violations.append(f"parent_id cycle involving {node.id}")
        elif not reached_file:
            violations.append(f"parent_id chain does not reach a file node: {node.id}")

    return violations


__all__ = [
    "NAME_MATCH_WEIGHT",
    "Edge",
    "ExtractedFile",
    "Node",
    "ProjectGraph",
    "RawCall",
    "build_graph",
    "validate_graph",
]
