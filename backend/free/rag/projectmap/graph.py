"""コードグラフの組み立て (c_16 §4.4)

抽出器 (``extractors/``) が返す 1 ファイル分の生ノード / import / 呼出しを
受け取り、辺を解決してプロジェクト全体の :class:`ProjectGraph` にする。

- ``contains``: 構文木から確定 (file → 直下の定義、定義 → ネストした定義)
- ``imports``: モジュール指定子をプロジェクト内 path へ解決できたものだけ
  (Python は絶対 import のドット区切り、JS/TS/TSX は相対パス + 拡張子補完、
  Rust/Go/Ruby/Java/Kotlin/PHP/C/C++ は言語ごとの決定論規則 — 詳細は各
  ``_resolve_<lang>_import`` の docstring。Swift はモジュール名が path に
  落ちないため常に external)。解決できないものは file ノードの
  ``external_imports`` に残す。Go だけは 1 つの import 指定子が 1 パッケージ
  (= ディレクトリ) を指すため、複数ファイルへ辺を張りうる
- ``calls`` / ``inherits``: 呼出名 / 基底名を
  **同一ファイル → 同一ディレクトリ → プロジェクト全体で一意** の順で解決し、
  一意でなければ張らない (誤った辺は無い辺より害が大きい)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

from backend.free.rag.projectmap.ids import code_node_id

#: JS/TS 系の相対 import に補完する拡張子 (順に試す)。
_JS_LIKE_EXTENSIONS: tuple[str, ...] = (
    "", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
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


def _resolve_import(
    file_path: str, lang: str, spec: str, all_paths: frozenset[str],
) -> list[str]:
    if lang == "python":
        target = _resolve_python_import(spec, all_paths)
        return [target] if target is not None else []
    if lang in ("javascript", "typescript", "tsx"):
        source_dir = PurePosixPath(file_path).parent
        target = _resolve_js_import(source_dir, spec, all_paths)
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
    return []  # swift はモジュール名が path に落ちないため常に external


def _unique_by_name(
    candidates: Sequence[Node], name: str,
) -> Node | None:
    matches = [n for n in candidates if n.name == name]
    return matches[0] if len(matches) == 1 else None


#: 名前一致だけで張る辺 (``calls`` / ``inherits``) の weight (c_16 §4.4)。
#: ``contains`` / ``imports`` (構文木 / パスから確定) の 1.0 より低く置く —
#: 型情報を見ない名前一致は、一意に決まっても「たぶんこれ」でしかない。
NAME_MATCH_WEIGHT = 0.5


def _resolve_symbol(
    caller: Node,
    name: str,
    *,
    by_file: dict[str, list[Node]],
    by_dir: dict[str, list[Node]],
    by_name_global: dict[str, list[Node]],
) -> Node | None:
    """呼出し名 / 基底名を「同一ファイル → 同一ディレクトリ → 全体で一意」で解決する。

    一意に決まらなければ ``None`` (辺を張らない — 誤った辺は無い辺より害が
    大きい、c_16 §4.4)。
    """
    same_file = _unique_by_name(by_file.get(caller.path, ()), name)
    if same_file is not None:
        return same_file
    directory = str(PurePosixPath(caller.path).parent)
    same_dir = _unique_by_name(by_dir.get(directory, ()), name)
    if same_dir is not None:
        return same_dir
    global_matches = by_name_global.get(name, ())
    return global_matches[0] if len(global_matches) == 1 else None


def build_graph(
    extracted: Sequence[ExtractedFile],
    *,
    fingerprints: dict[str, str],
) -> ProjectGraph:
    """1 プロジェクト分の抽出結果からグラフを組む。

    Args:
        extracted: ファイルごとの抽出結果 (path 昇順で渡すこと)。
        fingerprints: ``{path: sha256}`` (差分検出用にそのまま持ち回る)。
    """
    languages: dict[str, int] = defaultdict(int)
    all_paths = frozenset(f.path for f in extracted)

    nodes: list[Node] = []
    edges: list[Edge] = []
    external_imports: dict[str, list[str]] = {}

    by_file: dict[str, list[Node]] = defaultdict(list)
    by_dir: dict[str, list[Node]] = defaultdict(list)
    by_name_global: dict[str, list[Node]] = defaultdict(list)
    call_sites: list[tuple[Node, str]] = []
    inherit_sites: list[tuple[Node, str]] = []

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
            edges.append(
                Edge(src=resolved.parent_id, dst=resolved.id, etype="contains", weight=1.0),
            )
            by_file[resolved.path].append(resolved)
            by_dir[str(PurePosixPath(resolved.path).parent)].append(resolved)
            by_name_global[resolved.name].append(resolved)

        unresolved: list[str] = []
        for spec in file.imports:
            targets = [
                t for t in _resolve_import(file.path, file.lang, spec, all_paths)
                if t != file.path
            ]
            if not targets:
                unresolved.append(spec)
                continue
            for target_path in targets:
                target_id = code_node_id(target_path, "file", "")
                edges.append(
                    Edge(src=file_node.id, dst=target_id, etype="imports", weight=1.0),
                )
        if unresolved:
            external_imports[file.path] = unresolved

        for call in file.calls:
            caller = id_to_node.get(call.caller_id)
            if caller is not None:
                call_sites.append((caller, call.callee_name))

        for class_id, base_name in file.inherits:
            cls_node = id_to_node.get(class_id)
            if cls_node is not None:
                inherit_sites.append((cls_node, base_name))

    for caller, name in call_sites:
        target = _resolve_symbol(
            caller, name,
            by_file=by_file, by_dir=by_dir, by_name_global=by_name_global,
        )
        if target is not None and target.id != caller.id:
            edges.append(
                Edge(src=caller.id, dst=target.id, etype="calls", weight=NAME_MATCH_WEIGHT),
            )

    for cls_node, base_name in inherit_sites:
        target = _resolve_symbol(
            cls_node, base_name,
            by_file=by_file, by_dir=by_dir, by_name_global=by_name_global,
        )
        if target is not None and target.id != cls_node.id:
            edges.append(
                Edge(src=cls_node.id, dst=target.id, etype="inherits", weight=NAME_MATCH_WEIGHT),
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
