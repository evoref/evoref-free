"""JS/TS/Svelte/Vue の import エイリアス解決 (c_16 §4.4)

SvelteKit 組み込みの ``$lib`` と、tsconfig/jsconfig の
``compilerOptions.paths`` を、既存の相対 import と同じ最終解決 (拡張子補完 /
index file、``graph._resolve_js_import``) に乗せるための前処理。ここでは
「解決すべき root-relative な指定子の候補列」を作るところまでを担い、実際の
ファイル一致判定は :mod:`backend.free.rag.projectmap.graph` 側の既存ロジック
を再利用する (候補が 1 つも実在しなければ、呼出側は従来どおり
``external_imports`` に残す)。

``svelte.config.js``/``.ts`` の ``kit.alias`` は、キー・値とも文字列リテラル
だけで書かれた範囲を tree-sitter で **評価せず構文木だけを歩いて** 読む
(``export default {...}`` / ``const x = {...}; export default x;`` / 1 引数の
呼出しで包んだ ``export default defineConfig({...})``、TS の ``satisfies``/
``as`` 付き可)。``path.resolve(...)`` などの式・変数・スプレッド・計算キー・
補間ありのテンプレートリテラルはそのプロパティだけ読み飛ばす。tree-sitter が
無い環境では ``kit.alias`` を読まないだけで ``$lib``/tsconfig の解決は変わらない。
``tsconfig``/``jsconfig`` の ``extends`` は辿らない — 読めない/壊れた設定は
その 1 ファイルだけ無視して WARNING を出し、走査全体は止めない。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from backend.log_config import get_logger

logger = get_logger("rag.projectmap.aliases")

#: エイリアス解決に使う設定ファイルの basename (scanner が同じ 1 回の走査で拾う、c_16 §4.4)。
ALIAS_CONFIG_BASENAMES: frozenset[str] = frozenset({
    "svelte.config.js", "svelte.config.ts", "tsconfig.json", "jsconfig.json",
})

_SVELTE_CONFIG_NAMES: frozenset[str] = frozenset({"svelte.config.js", "svelte.config.ts"})
_TS_CONFIG_NAMES: frozenset[str] = frozenset({"tsconfig.json", "jsconfig.json"})


def _config_dir(path: str) -> str:
    """設定ファイルの相対パスから、そのディレクトリ (posix、ルート直下 = ``""``) を返す。"""
    parent = PurePosixPath(path).parent
    return "" if str(parent) == "." else str(parent)


def _nearest_ancestor_dir(file_path: str, candidate_dirs: frozenset[str]) -> str | None:
    """``file_path`` の祖先ディレクトリのうち ``candidate_dirs`` に含まれる最寄りのもの。

    自分自身のディレクトリから走査ルート (``""``) まで遡る (c_16 §4.4)。
    """
    if not candidate_dirs:
        return None
    parts = PurePosixPath(file_path).parent.parts
    for i in range(len(parts), -1, -1):
        candidate = "/".join(parts[:i])
        if candidate in candidate_dirs:
            return candidate
    return None


def _normalize_join(base_dir: str, raw: str) -> str:
    """``base_dir`` (posix、``""`` = ルート) に ``raw`` (``./``/``../`` 可) を足して畳む。"""
    parts = [p for p in base_dir.split("/") if p and p != "."]
    for part in raw.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class _PathsPattern:
    """``compilerOptions.paths`` の 1 パターン (c_16 §4.4)。宣言順を保って持ち回る。

    ``*`` は末尾に 1 つだけのものに限る (:func:`_parse_paths_pattern` が検証)。
    """

    prefix: str
    wildcard: bool
    targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TSConfigRules:
    """1 tsconfig/jsconfig ぶんの解決規則。"""

    #: posix、ルート相対。設定ファイルのディレクトリ + ``baseUrl`` を合成済み。
    base_dir: str
    #: 宣言順 (``paths`` オブジェクトのキー順)。
    patterns: tuple[_PathsPattern, ...]


def _strip_jsonc(text: str) -> str:
    """tsconfig の JSONC (行/ブロックコメント・末尾カンマ) を JSON へ落とす。

    文字列リテラルの中の ``//`` やカンマは壊さない — 決定論の 1 文字ずつの
    走査で文字列の内外を追跡し、エスケープ (``\\"``) も飛ばす。
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        out.append(ch)
        i += 1
    stripped = "".join(out)
    # 末尾カンマ (「,」の後に空白/改行を挟んで「}」か「]」) を落とす。
    return re.sub(r",(\s*[}\]])", r"\1", stripped)


def _parse_paths_pattern(key: Any, raw_targets: Any) -> _PathsPattern | None:
    if not isinstance(key, str) or not key:
        return None
    if not isinstance(raw_targets, list) or not raw_targets:
        return None
    targets = tuple(t for t in raw_targets if isinstance(t, str) and t)
    if not targets:
        return None
    star_count = key.count("*")
    if star_count == 0:
        return _PathsPattern(prefix=key, wildcard=False, targets=targets)
    if star_count == 1 and key.endswith("*"):
        return _PathsPattern(prefix=key[:-1], wildcard=True, targets=targets)
    return None  # ``*`` が末尾以外/複数 — 仕様外 (c_16 §4.4)


def _load_tsconfig_rules(root: Path, path: str) -> _TSConfigRules | None:
    """1 つの tsconfig/jsconfig を読む。壊れていれば ``None`` (WARNING 1 行)。"""
    try:
        text = (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("projectmap: unreadable tsconfig %s: %s", path, e)
        return None
    try:
        data = json.loads(_strip_jsonc(text))
    except json.JSONDecodeError as e:
        logger.warning("projectmap: invalid JSON in %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        return None
    compiler_options = data.get("compilerOptions")
    if not isinstance(compiler_options, dict):
        return None
    paths_raw = compiler_options.get("paths")
    if not isinstance(paths_raw, dict) or not paths_raw:
        return None

    base_url = compiler_options.get("baseUrl")
    base_url = base_url if isinstance(base_url, str) else "."
    base_dir = _normalize_join(_config_dir(path), base_url)

    patterns: list[_PathsPattern] = []
    skipped = 0
    for key, targets in paths_raw.items():
        pattern = _parse_paths_pattern(key, targets)
        if pattern is None:
            skipped += 1
            continue
        patterns.append(pattern)
    if skipped:
        logger.warning(
            "projectmap: skipped %d unsupported paths pattern(s) in %s", skipped, path,
        )
    if not patterns:
        return None
    return _TSConfigRules(base_dir=base_dir, patterns=tuple(patterns))


# ── kit.alias (svelte.config.js/.ts、静的な tree-sitter 走査、c_16 §4.4) ────

#: 構築に失敗した grammar は ``None`` を持つ (再試行 / 再警告をしない)。
_KIT_ALIAS_PARSER_CACHE: dict[str, Any] = {}
_KIT_ALIAS_TREE_SITTER_AVAILABLE: bool | None = None

#: ``export default <expr> satisfies Config`` / ``as Config`` の型注釈を
#: 剥がす対象。どちらも構文木上は最初の子が中身の式 (フィールド名は無い)。
_TYPE_ANNOTATION_WRAPPER_TYPES: frozenset[str] = frozenset({
    "satisfies_expression", "as_expression",
})


def _kit_alias_tree_sitter_available() -> bool:
    """``tree_sitter``/``tree_sitter_language_pack`` が import できるか。

    :mod:`backend.free.rag.projectmap.extractors.treesitter` の同名の判定を
    再利用しない — その物は :mod:`graph` を import しており、``graph`` は
    本モジュールの :class:`AliasConfig` を import するため、モジュール
    トップレベルで結ぶと循環 import になる (2026-09-20)。
    """
    global _KIT_ALIAS_TREE_SITTER_AVAILABLE
    if _KIT_ALIAS_TREE_SITTER_AVAILABLE is None:
        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            _KIT_ALIAS_TREE_SITTER_AVAILABLE = False
        else:
            _KIT_ALIAS_TREE_SITTER_AVAILABLE = True
    return _KIT_ALIAS_TREE_SITTER_AVAILABLE


def _get_kit_alias_parser(grammar: str) -> Any | None:
    if grammar in _KIT_ALIAS_PARSER_CACHE:
        return _KIT_ALIAS_PARSER_CACHE[grammar]
    parser: Any | None = None
    if _kit_alias_tree_sitter_available():
        try:
            from tree_sitter_language_pack import get_parser

            parser = get_parser(grammar)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 — 1 grammar の失敗で走査全体を止めない
            logger.warning(
                "projectmap: tree-sitter unavailable for kit.alias parsing (%s): %s",
                grammar, e,
            )
    _KIT_ALIAS_PARSER_CACHE[grammar] = parser
    return parser


def _ts_text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _unwrap_type_annotation(node: Any | None) -> Any | None:
    """``satisfies``/``as`` の型注釈を 1 段以上剥がす。"""
    while node is not None and node.type in _TYPE_ANNOTATION_WRAPPER_TYPES:
        if node.child_count == 0:
            return None
        node = node.children[0]
    return node


def _unwrap_single_arg_call(node: Any | None) -> Any | None:
    """1 引数の呼出し (``defineConfig({...})``) なら、その引数を返す。

    引数が 1 つ (名前付きノードが 1 個) でなければ関数は評価できないので
    そのまま返す (呼出側でオブジェクトリテラル判定に失敗して諦める)。
    """
    if node is None or node.type != "call_expression":
        return node
    args = node.child_by_field_name("arguments")
    if args is None:
        return node
    named = [c for c in args.children if c.is_named]
    if len(named) != 1:
        return node
    return named[0]


def _resolve_top_level_identifier(
    node: Any | None, program: Any, source: bytes,
) -> Any | None:
    """``const``/``let`` で束ねた値を、トップレベルの宣言まで 1 段だけ辿る。

    ``identifier`` でなければそのまま返す。見つからなければ (未定義 / 別種の
    宣言) 元の identifier ノードのまま返し、呼出側でオブジェクトリテラル
    判定に失敗して諦める。
    """
    if node is None or node.type != "identifier":
        return node
    name = _ts_text(node, source)
    for stmt in program.children:
        if stmt.type != "lexical_declaration":
            continue
        for declarator in stmt.children:
            if declarator.type != "variable_declarator":
                continue
            name_node = declarator.child_by_field_name("name")
            if (
                name_node is not None
                and name_node.type == "identifier"
                and _ts_text(name_node, source) == name
            ):
                value = declarator.child_by_field_name("value")
                if value is not None:
                    return value
    return node


def _find_export_default_value(program: Any) -> Any | None:
    for stmt in program.children:
        if stmt.type != "export_statement":
            continue
        value = stmt.child_by_field_name("value")
        if value is not None:
            return value
    return None


def _object_property_key_text(key_node: Any, source: bytes) -> str | None:
    if key_node.type == "property_identifier":
        return _ts_text(key_node, source)
    if key_node.type == "string":
        return _plain_string_content(key_node, source)
    return None


def _object_get_property(obj_node: Any, prop_name: str, source: bytes) -> Any | None:
    """オブジェクトリテラルから ``prop_name: <value>`` の ``<value>`` を取る。

    キーが計算キー / スプレッド等でこの名前と一致しないものは無視する。
    """
    for child in obj_node.children:
        if not child.is_named or child.type != "pair":
            continue
        key_node = child.child_by_field_name("key")
        if key_node is None:
            continue
        key_text = _object_property_key_text(key_node, source)
        if key_text == prop_name:
            return child.child_by_field_name("value")
    return None


def _plain_string_content(node: Any, source: bytes) -> str | None:
    """``string``/``template_string`` の中身を返す。

    補間 (``template_substitution``) やエスケープ (``escape_sequence``) が
    混じっていれば決定論に読めないので ``None`` (呼出側はそのプロパティ/
    キーだけを読み飛ばす)。中身が無い名前付き子は空文字列として扱う。
    """
    named = [c for c in node.children if c.is_named]
    if not named:
        return ""
    if len(named) == 1 and named[0].type == "string_fragment":
        return _ts_text(named[0], source)
    return None


def _kit_alias_object_node(root: Any, source: bytes) -> Any | None:
    """構文木から ``kit.alias`` のオブジェクトリテラルノードを探す。

    受ける形 (c_16 §4.4): ``export default {...}`` / ``const x = {...};
    export default x;`` (同一ファイル内の 1 段だけ) / ``export default
    fn({...})`` (1 引数の呼出しで包んだ形)。TS の ``satisfies``/``as`` 型注釈
    はどの段でも剥がす。どれにも当てはまらない、または ``kit``/``alias`` が
    無ければ ``None``。
    """
    value = _find_export_default_value(root)
    value = _unwrap_type_annotation(value)
    value = _resolve_top_level_identifier(value, root, source)
    value = _unwrap_type_annotation(value)
    value = _unwrap_single_arg_call(value)
    value = _unwrap_type_annotation(value)
    if value is None or value.type != "object":
        return None
    kit_value = _object_get_property(value, "kit", source)
    if kit_value is None or kit_value.type != "object":
        return None
    alias_value = _object_get_property(kit_value, "alias", source)
    if alias_value is None or alias_value.type != "object":
        return None
    return alias_value


def _extract_kit_alias_pairs(alias_node: Any, source: bytes) -> tuple[dict[str, str], int]:
    """``kit.alias`` オブジェクトの ``{キー: 値}`` を宣言順に読む。

    キー・値のどちらかが文字列リテラル (補間なしテンプレート含む) でない
    プロパティは、そのプロパティだけ読み飛ばす (2 個目の戻り値は読み飛ばした
    件数、呼出元で WARNING 1 行にまとめる)。
    """
    pairs: dict[str, str] = {}
    skipped = 0
    for child in alias_node.children:
        if not child.is_named:
            continue
        if child.type != "pair":
            skipped += 1
            continue
        key_node = child.child_by_field_name("key")
        value_node = child.child_by_field_name("value")
        if key_node is None or value_node is None:
            skipped += 1
            continue
        key = _object_property_key_text(key_node, source)
        if key is None:
            skipped += 1
            continue
        if value_node.type not in ("string", "template_string"):
            skipped += 1
            continue
        value = _plain_string_content(value_node, source)
        if value is None:
            skipped += 1
            continue
        pairs[key] = value
    return pairs, skipped


def _load_kit_alias_rules(root: Path, path: str) -> _TSConfigRules | None:
    """1 つの ``svelte.config.js``/``.ts`` から ``kit.alias`` 規則を読む。

    tree-sitter 不在 / grammar 未対応 / 構文エラー / ``kit.alias`` が無い
    場合は ``None`` (呼出側は従来どおり ``$lib`` だけで解決する)。
    """
    grammar = "typescript" if path.endswith(".ts") else "javascript"
    parser = _get_kit_alias_parser(grammar)
    if parser is None:
        return None
    try:
        source = (root / path).read_bytes()
    except OSError as e:
        logger.warning("projectmap: unreadable svelte config %s: %s", path, e)
        return None
    try:
        tree = parser.parse(source)
        alias_node = _kit_alias_object_node(tree.root_node, source)
        if alias_node is None:
            return None
        raw_pairs, skipped = _extract_kit_alias_pairs(alias_node, source)
    except Exception as e:  # noqa: BLE001 — 1 ファイルの走査失敗で全体を止めない
        logger.warning("projectmap: failed to parse kit.alias in %s: %s", path, e)
        return None
    if skipped:
        logger.warning(
            "projectmap: skipped %d unsupported kit.alias entr%s in %s",
            skipped, "y" if skipped == 1 else "ies", path,
        )
    if not raw_pairs:
        return None
    patterns: list[_PathsPattern] = []
    for key, target in raw_pairs.items():
        pattern = _parse_paths_pattern(key, [target])
        if pattern is not None:
            patterns.append(pattern)
    if not patterns:
        return None
    return _TSConfigRules(base_dir=_config_dir(path), patterns=tuple(patterns))


@dataclass(slots=True)
class AliasConfig:
    """走査 1 回分のエイリアス解決テーブル (c_16 §4.4)。

    ``svelte_lib_dirs``: ``$lib`` が有効なディレクトリ (``svelte.config.js``/
    ``.ts`` があるディレクトリ) の集合。``ts_rules_by_dir``: tsconfig/jsconfig
    があるディレクトリ → 解決規則。``kit_alias_by_dir``: ``svelte.config.js``/
    ``.ts`` の ``kit.alias`` があるディレクトリ → 解決規則 (文字列リテラル
    だけで書かれた範囲のみ)。どれも「最寄り」検索は :func:`_nearest_ancestor_dir`
    で行う。
    """

    svelte_lib_dirs: frozenset[str] = field(default_factory=frozenset)
    ts_rules_by_dir: dict[str, _TSConfigRules] = field(default_factory=dict)
    kit_alias_by_dir: dict[str, _TSConfigRules] = field(default_factory=dict)

    def svelte_lib_candidate(self, file_path: str, spec: str) -> str | None:
        """``$lib``/``$lib/x`` を root-relative 指定子 (``/…``) へ変換する。

        最寄りに ``svelte.config.js``/``.ts`` が無ければ ``None`` (呼出側は
        従来どおり external 扱いにする)。``kit.alias`` は見ない (対象外)。
        """
        if spec != "$lib" and not spec.startswith("$lib/"):
            return None
        nearest = _nearest_ancestor_dir(file_path, self.svelte_lib_dirs)
        if nearest is None:
            return None
        lib_dir = f"{nearest}/src/lib" if nearest else "src/lib"
        rest = spec[len("$lib"):]
        return f"/{lib_dir}{rest}"

    def ts_paths_candidates(self, file_path: str, spec: str) -> list[str]:
        """tsconfig/jsconfig の ``paths`` に一致する root-relative 候補の列。

        パターンは宣言順、各パターンの ``targets`` も宣言順で列挙する。
        呼出側 (``graph._resolve_via_alias``) が先頭から試し、最初に実在する
        ファイルへ解決できたものを採る。一致するパターンが無ければ空リスト。
        """
        nearest = _nearest_ancestor_dir(file_path, frozenset(self.ts_rules_by_dir))
        if nearest is None:
            return []
        rules = self.ts_rules_by_dir[nearest]
        candidates: list[str] = []
        for pattern in rules.patterns:
            if pattern.wildcard:
                if not spec.startswith(pattern.prefix):
                    continue
                fragment = spec[len(pattern.prefix):]
            else:
                if spec != pattern.prefix:
                    continue
                fragment = ""
            for target in pattern.targets:
                rendered = target.replace("*", fragment, 1) if "*" in target else target
                candidates.append(f"/{_normalize_join(rules.base_dir, rendered)}")
        return candidates

    def kit_alias_candidates(self, file_path: str, spec: str) -> list[str]:
        """``kit.alias`` に一致する root-relative 候補の列 (c_16 §4.4)。

        SvelteKit の解釈は tsconfig ``paths`` と違い、ワイルドカードでない
        エントリも接頭辞として働く (``$components`` と ``$components/x`` の
        両方に効く) ので :meth:`ts_paths_candidates` とは判定を分けている。
        ワイルドカードエントリ (``キー`` が ``*`` 終端) は tsconfig と同じ
        判定。宣言順・``targets`` の宣言順で列挙する (呼出側が先頭から試す)。
        """
        nearest = _nearest_ancestor_dir(file_path, frozenset(self.kit_alias_by_dir))
        if nearest is None:
            return []
        rules = self.kit_alias_by_dir[nearest]
        candidates: list[str] = []
        for pattern in rules.patterns:
            if pattern.wildcard:
                if not spec.startswith(pattern.prefix):
                    continue
                fragment = spec[len(pattern.prefix):]
            else:
                if spec == pattern.prefix:
                    fragment = ""
                elif spec.startswith(f"{pattern.prefix}/"):
                    fragment = spec[len(pattern.prefix):]  # 先頭 "/" を含めて残す
                else:
                    continue
            for target in pattern.targets:
                rendered = (
                    target.replace("*", fragment, 1) if "*" in target else f"{target}{fragment}"
                )
                candidates.append(f"/{_normalize_join(rules.base_dir, rendered)}")
        return candidates


def build_alias_config(root: Path, config_paths: Sequence[str]) -> AliasConfig:
    """走査で拾った設定ファイル一覧 (c_16 §4.4) からエイリアステーブルを作る。

    ``svelte.config.js``/``.ts`` は ``$lib`` を有効にするだけでなく、
    ``kit.alias`` (文字列リテラルだけの範囲) も tree-sitter で読む。
    ``tsconfig.json``/``jsconfig.json`` は JSONC を剥がして
    ``compilerOptions.paths`` を読む。1 ファイルの読み込み失敗は他の設定
    ファイルを止めない。同じディレクトリに両方あれば ``tsconfig.json`` を
    優先する (先勝ち、走査順に依存させない)。
    """
    svelte_lib_dirs: set[str] = set()
    ts_rules_by_dir: dict[str, _TSConfigRules] = {}
    kit_alias_by_dir: dict[str, _TSConfigRules] = {}
    for path in config_paths:
        name = PurePosixPath(path).name
        if name in _SVELTE_CONFIG_NAMES:
            config_dir = _config_dir(path)
            svelte_lib_dirs.add(config_dir)
            if config_dir not in kit_alias_by_dir:
                kit_rules = _load_kit_alias_rules(root, path)
                if kit_rules is not None:
                    kit_alias_by_dir[config_dir] = kit_rules
        elif name in _TS_CONFIG_NAMES:
            rules = _load_tsconfig_rules(root, path)
            if rules is None:
                continue
            config_dir = _config_dir(path)
            if config_dir not in ts_rules_by_dir or name == "tsconfig.json":
                ts_rules_by_dir[config_dir] = rules
    return AliasConfig(
        svelte_lib_dirs=frozenset(svelte_lib_dirs),
        ts_rules_by_dir=ts_rules_by_dir,
        kit_alias_by_dir=kit_alias_by_dir,
    )


__all__ = [
    "ALIAS_CONFIG_BASENAMES",
    "AliasConfig",
    "build_alias_config",
]
