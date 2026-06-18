"""生成物の API 契約を AST から抽出し、test↔src の不整合を決定論的に検出する。

``smoke_validator`` の ``check_coherence`` (重複定義 / 自由名の未定義) と
``run_import_smoke`` (import が通るか) では拾えない以下を、サブプロセス不要・
純粋・非循環の AST 解析で検出する:

- **関数 arity 不一致**: ``f(a, b, c)`` を ``def f(a, b): ...`` に対して呼ぶ。
- **属性 / メソッド不在**: ``obj.method()`` / ``obj.attr`` を、そのクラスが
  定義しないメンバに対して参照する (例: ``main()`` が ``game.run`` を呼ぶが
  ``run`` メソッドが無い / テストが ``game.board`` を参照するが実体に無い)。

誤検知は破壊的リペア (src の不要な書き換え) を招くため、``check_coherence`` と
同じく **保守的に「確証のある問題のみ」報告**する。型推論を要する判定 (戻り値の
型・比較値の型) は対象外。動的束縛 (``__getattr__`` / ``setattr`` / 生成物外の
基底クラス継承 / ``*args`` / ``**kwargs`` / 型不明変数) があれば該当チェックを
丸ごとスキップする。

この層は EvorefGen 内に閉じる (pillar 越境なし)。``smoke_validator.check_entrypoint``
がクラスシンボル抽出と束縛追跡を再利用する。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

# 任意属性が実行時合成され得る = 属性存在チェックを無効化する dunder。
_DYNAMIC_DUNDERS: frozenset[str] = frozenset({"__getattr__", "__getattribute__"})

# 同名 sibling 定義が正当になるアクセサ系デコレータ (check_coherence と整合)。
_ACCESSOR_DECORATORS: frozenset[str] = frozenset({
    "property", "cached_property", "setter", "deleter", "getter",
})


@dataclass
class FuncSig:
    """関数 / メソッドのシグネチャ契約 (``self`` はメソッドでは除外済み)。"""

    min_pos: int = 0          # デフォルト無し位置引数の数 (最低限の束縛数)
    max_pos: int = 0          # posonly + 通常位置引数の総数
    has_varargs: bool = False  # ``*args`` の有無
    has_kwargs: bool = False   # ``**kwargs`` の有無
    kwonly_all: set[str] = field(default_factory=set)
    kwonly_required: set[str] = field(default_factory=set)
    pos_names: set[str] = field(default_factory=set)  # 位置引数名 (kw 充足判定用)


@dataclass
class ClassApi:
    """クラスの公開シンボル契約。"""

    name: str
    bases: list[str] = field(default_factory=list)
    methods: dict[str, FuncSig] = field(default_factory=dict)
    attrs: set[str] = field(default_factory=set)
    dynamic: bool = False  # True なら属性存在チェックを丸ごとスキップ


@dataclass
class SrcApi:
    """生成 src 群の公開 API (関数 + クラス)。"""

    functions: dict[str, FuncSig] = field(default_factory=dict)
    classes: dict[str, ClassApi] = field(default_factory=dict)


# ── シグネチャ / クラス抽出 ───────────────────────────────────────────────

def _func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_method: bool) -> FuncSig:
    """def ノードから :class:`FuncSig` を構築する (メソッドは ``self`` を除外)。"""
    a = node.args
    posonly = list(a.posonlyargs)
    pos = list(a.args)
    all_pos = posonly + pos
    if is_method and all_pos:
        all_pos = all_pos[1:]  # self / cls を除外
    # デフォルト値は posonly+args の末尾から対応する。
    num_defaults = len(a.defaults)
    total = len(all_pos)
    min_pos = max(0, total - num_defaults)
    kwonly_all = {k.arg for k in a.kwonlyargs}
    kwonly_required = {
        k.arg for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is None
    }
    return FuncSig(
        min_pos=min_pos,
        max_pos=total,
        has_varargs=a.vararg is not None,
        has_kwargs=a.kwarg is not None,
        kwonly_all=kwonly_all,
        kwonly_required=kwonly_required,
        pos_names={p.arg for p in all_pos},
    )


def _base_name(node: ast.expr) -> str | None:
    """基底クラス式から名前を取り出す (``Name``/``Attribute`` のみ)。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _has_dynamic_self(classdef: ast.ClassDef) -> bool:
    """``setattr(self, ...)`` / ``self.__dict__`` 操作の有無 (属性が動的合成され得る)。"""
    for n in ast.walk(classdef):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr":
            if n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == "self":
                return True
        if isinstance(n, ast.Attribute) and n.attr == "__dict__":
            if isinstance(n.value, ast.Name) and n.value.id == "self":
                return True
    return False


def _raw_class_api(classdef: ast.ClassDef) -> ClassApi:
    """1 つの ClassDef から (基底マージ前の) :class:`ClassApi` を作る。"""
    api = ClassApi(name=classdef.name)
    api.bases = [b for b in (_base_name(x) for x in classdef.bases) if b]
    # metaclass 指定 / 動的 self 操作 → 動的扱い。
    if any(kw.arg == "metaclass" for kw in classdef.keywords) or _has_dynamic_self(classdef):
        api.dynamic = True
    for m in classdef.body:
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decos = _decorator_names(m)
            if decos & _DYNAMIC_DUNDERS or m.name in _DYNAMIC_DUNDERS:
                api.dynamic = True
            if decos & _ACCESSOR_DECORATORS:
                api.attrs.add(m.name)  # property 等は属性としてアクセスされる
            else:
                api.methods[m.name] = _func_sig(m, is_method=True)
        elif isinstance(m, ast.Assign):
            for t in m.targets:
                api.attrs |= _assign_target_names(t)
        elif isinstance(m, ast.AnnAssign):  # dataclass フィールド等
            api.attrs |= _assign_target_names(m.target)
    # メソッド本体の ``self.X = ...`` を属性に収集 (条件分岐下も過剰収集)。
    for n in ast.walk(classdef):
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
            if isinstance(n.value, ast.Name) and n.value.id == "self":
                api.attrs.add(n.attr)
    return api


def _decorator_names(node) -> set[str]:
    names: set[str] = set()
    for dec in getattr(node, "decorator_list", None) or []:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _assign_target_names(target) -> set[str]:
    out: set[str] = set()
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for el in target.elts:
            out |= _assign_target_names(el)
    elif isinstance(target, ast.Starred):
        out |= _assign_target_names(target.value)
    return out


def build_src_api(src_files: dict[str, str]) -> SrcApi:
    """生成 src 群から公開 API (top-level 関数 + クラス) を抽出する。

    クラスは生成物内の基底クラスを再帰マージ。生成物外の基底を継承するクラスは
    継承メンバを静的に見られないため ``dynamic=True`` (属性チェック対象外) とする。
    構文エラーのファイルは無視する (validate_python / smoke 側が扱う)。
    """
    api = SrcApi()
    raw_classes: dict[str, ClassApi] = {}
    for code in src_files.values():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not (_decorator_names(n) & _ACCESSOR_DECORATORS):
                    api.functions[n.name] = _func_sig(n, is_method=False)
            elif isinstance(n, ast.ClassDef):
                raw_classes[n.name] = _raw_class_api(n)

    # 基底クラスを再帰マージ (生成物外の基底は dynamic 化)。
    def _merged(name: str, seen: frozenset[str]) -> ClassApi:
        base = raw_classes[name]
        if name in seen:  # 循環継承の保険
            return base
        merged = ClassApi(
            name=name, bases=list(base.bases),
            methods=dict(base.methods), attrs=set(base.attrs), dynamic=base.dynamic,
        )
        for b in base.bases:
            if b == "object":
                continue
            if b in raw_classes:
                parent = _merged(b, seen | {name})
                for mn, ms in parent.methods.items():
                    merged.methods.setdefault(mn, ms)
                merged.attrs |= parent.attrs
                if parent.dynamic:
                    merged.dynamic = True
            else:
                merged.dynamic = True  # 生成物外の基底 → 継承メンバ不明
        return merged

    for cname in raw_classes:
        api.classes[cname] = _merged(cname, frozenset())
    return api


# ── ローカル変数 → クラス束縛追跡 (check_entrypoint と共有) ────────────────

def bind_local_instances(
    body: list[ast.stmt], known_classes: set[str],
) -> dict[str, str]:
    """文列内の ``x = ClassName(...)`` から変数→クラス名束縛を返す (曖昧は除外)。

    同名変数が複数箇所で異なる代入を受ける / 構築以外で再代入される場合は曖昧と
    みなし束縛しない (型不明として以降のチェックをスキップさせる)。
    """
    assigned_class: dict[str, str] = {}
    ambiguous: set[str] = set()
    for stmt in body:
        for n in ast.walk(stmt):
            if not isinstance(n, ast.Assign):
                continue
            cls = None
            if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name):
                if n.value.func.id in known_classes:
                    cls = n.value.func.id
            for t in n.targets:
                if not isinstance(t, ast.Name):
                    continue
                name = t.id
                if cls is None:
                    ambiguous.add(name)
                elif name in assigned_class and assigned_class[name] != cls:
                    ambiguous.add(name)
                else:
                    assigned_class[name] = cls
    return {k: v for k, v in assigned_class.items() if k not in ambiguous}


@dataclass
class AttrUse:
    """インスタンス変数に対する属性 / メソッド参照。"""

    cls: str
    attr: str
    call: ast.Call | None  # メソッド呼び出しなら Call ノード、属性参照なら None


def collect_attr_uses(
    body: list[ast.stmt], bindings: dict[str, str],
) -> list[AttrUse]:
    """束縛済みインスタンス変数への ``var.attr`` / ``var.method(...)`` を収集する。

    ``var.method(...)`` は Call として、素の ``var.attr`` (コールバック渡し含む)
    は属性参照として返す。
    """
    # メソッド呼び出しの Attribute ノードを記録して二重計上を避ける。
    call_attr_nodes: dict[int, ast.Call] = {}
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if isinstance(n.func.value, ast.Name) and n.func.value.id in bindings:
                    call_attr_nodes[id(n.func)] = n
    uses: list[AttrUse] = []
    for stmt in body:
        for n in ast.walk(stmt):
            if not isinstance(n, ast.Attribute):
                continue
            if not (isinstance(n.value, ast.Name) and n.value.id in bindings):
                continue
            if isinstance(n.ctx, ast.Store):
                continue  # 代入先は新規属性定義 = 不在ではない
            cls = bindings[n.value.id]
            uses.append(AttrUse(cls=cls, attr=n.attr, call=call_attr_nodes.get(id(n))))
    return uses


# ── arity 照合 ────────────────────────────────────────────────────────────

def check_call_arity(sig: FuncSig, call: ast.Call, label: str) -> str | None:
    """呼び出しが ``sig`` の arity に反するか判定する (確証のある違反のみ)。

    ``*args``/``**kwargs`` 展開を含む call、可変長を受ける sig は判定を諦める
    (None)。報告するのは「位置引数過多」「未知キーワード」「明確な引数不足」のみ。
    """
    pos_args = [a for a in call.args if not isinstance(a, ast.Starred)]
    has_star_arg = any(isinstance(a, ast.Starred) for a in call.args)
    kw_names = [k.arg for k in call.keywords if k.arg is not None]
    has_kw_unpack = any(k.arg is None for k in call.keywords)
    if has_star_arg or has_kw_unpack:
        return None  # 展開があると正確に数えられない → スキップ

    n_pos = len(pos_args)
    # 位置引数過多 (可変長なし)。
    if not sig.has_varargs and n_pos > sig.max_pos:
        return f"{label}: 引数が多すぎる ({n_pos} 個, 最大 {sig.max_pos})"
    # 未知キーワード (**kwargs なし)。
    if not sig.has_kwargs:
        known = sig.pos_names | sig.kwonly_all
        for kn in kw_names:
            if kn not in known:
                return f"{label}: 未知のキーワード引数 '{kn}'"
    # 引数不足 (kw が位置を埋め得るため過大に見積もって保守判定)。
    if (n_pos + len(kw_names)) < sig.min_pos:
        return f"{label}: 引数が不足 ({n_pos + len(kw_names)} 個, 最低 {sig.min_pos})"
    return None


# ── test ↔ src 契約チェック ───────────────────────────────────────────────

def _resolve_test_imports(
    tree: ast.Module, src_stems: set[str],
) -> tuple[dict[str, str], set[str]]:
    """test の import から (ローカル名→src シンボル名) と (src モジュール別名) を返す。

    - ``from tetris import f, C as K`` → {"f": "f", "K": "C"} (src 由来確証)。
    - ``import tetris`` / ``import tetris as t`` → モジュール別名集合 {"tetris","t"}。
    モジュール/関数本体内の import も拾う (テストは関数内 import を多用するため)。
    """
    name_to_sym: dict[str, str] = {}
    module_aliases: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = (n.module or "").split(".")[0]
            if n.level == 0 and mod in src_stems:
                for a in n.names:
                    if a.name != "*":
                        name_to_sym[a.asname or a.name] = a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in src_stems:
                    module_aliases.add((a.asname or a.name).split(".")[0])
    return name_to_sym, module_aliases


def check_api_contract(
    src_files: dict[str, str], test_files: dict[str, str],
) -> list[str]:
    """test が src の実 API (arity / 属性 / メソッド) に整合するか検証する。

    src を権威とし、不整合は test 側の欠陥として報告する (src は一切見ない方向の
    片方向契約)。誤検知抑制のため、src 由来と確証できる名前のみ照合し、動的束縛・
    可変長・型不明変数は全てスキップする。出力は決定論順 (sorted)。
    """
    api = build_src_api(src_files)
    src_stems = {p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in src_files}
    errors: list[str] = []

    for path, code in test_files.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        name_to_sym, module_aliases = _resolve_test_imports(tree, src_stems)
        if not name_to_sym and not module_aliases:
            continue

        # 関数 arity: 直接名 ``f(...)`` と モジュール経由 ``mod.f(...)``。
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fname: str | None = None
            if isinstance(n.func, ast.Name) and n.func.id in name_to_sym:
                fname = name_to_sym[n.func.id]
            elif isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
                if n.func.value.id in module_aliases:
                    fname = n.func.attr
            if fname and fname in api.functions:
                msg = check_call_arity(api.functions[fname], n, f"{fname}()")
                if msg:
                    errors.append(f"{path}: {msg}")

        # クラス属性 / メソッド: ``g = TetrisGame(); g.board / g.update(...)``。
        known_class_names = set(api.classes)
        # test 内で src クラスに束縛され得る名前 (直接 import / モジュール経由は不可)。
        local_class_names = {
            local for local, sym in name_to_sym.items() if sym in known_class_names
        }
        for fn in _iter_function_bodies(tree):
            bindings_local = bind_local_instances(fn, local_class_names)
            # ローカル名 → 実 src クラス名へ写像し、生成物クラスのみ残す。
            bindings = {
                var: _src_class_for(var, bindings_local, name_to_sym)
                for var in bindings_local
            }
            bindings = {v: c for v, c in bindings.items() if c in api.classes}
            for use in collect_attr_uses(fn, bindings):
                capi = api.classes[use.cls]
                if capi.dynamic:
                    continue
                present_method = use.attr in capi.methods
                present_attr = use.attr in capi.attrs
                if not present_method and not present_attr:
                    errors.append(
                        f"{path}: {use.cls}.{use.attr} は定義が無い "
                        "(メソッド/属性として存在しない)"
                    )
                elif use.call is not None and present_method:
                    msg = check_call_arity(
                        capi.methods[use.attr], use.call, f"{use.cls}.{use.attr}()",
                    )
                    if msg:
                        errors.append(f"{path}: {msg}")
    return sorted(set(errors))


def _src_class_for(
    var: str, bindings_local: dict[str, str], name_to_sym: dict[str, str],
) -> str:
    """ローカル変数の束縛先 (test のローカルクラス名) を src クラス名へ写像する。"""
    local_cls = bindings_local[var]
    return name_to_sym.get(local_cls, local_cls)


def _iter_function_bodies(tree: ast.Module) -> list[list[ast.stmt]]:
    """モジュール内の各関数本体 (文列) を返す。インスタンス束縛は関数スコープ単位。"""
    bodies: list[list[ast.stmt]] = [tree.body]
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies.append(n.body)
    return bodies
