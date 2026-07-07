"""spec 宣言契約と生成コードの決定論照合 (spec 準拠ゲート)。

loop 側 (``spec_contract.parse_declared_contract``) が抽出した宣言契約
(plain dict 列 — 型 import なしで pillar 境界を越える) を、生成 src の AST
(``api_contract.build_src_api`` を再利用) と突き合わせ、**確証のある違反のみ**
を報告する。誤検知は破壊的リペア (spec 通りのコードの不要な書き換え) を招く
ため、判定不能はすべて沈黙スキップに倒す (``check_coherence`` /
``check_api_contract`` と同じ保守則)。

検査セット (f_10 §5):

- C1 存在: 宣言された def/class 名が全生成 src の **任意深度** に 1 つも
  存在しない時のみ違反 (兄弟モジュールへの配置・クラス内への畳み込みは正当)。
- C2 メソッド存在: クラス契約の宣言メソッドが code 側クラスの
  ``methods ∪ attrs`` に不在の時のみ違反 (property 実装は attrs 側で正当化。
  ``ClassApi.dynamic`` = 生成物外基底/``__getattr__``/``setattr``/metaclass は
  丸ごとスキップ、dataclass 系デコレータは ``__init__`` 免除)。
- C3 確定 arity 非互換: 「spec 宣言シグネチャ通りの位置呼出が code 側で必ず
  失敗する」場合のみ (self/cls は含む・含まないの両解釈で試し双方非互換の時
  だけ)。デフォルト引数の追加等、code が spec より寛容になるのは無害。

呼出は executor へ DI 注入する (``contract_checker`` と同型。loop→gen の
top-level import 越境を api 層 (chat_streaming) で回避する確立パターン)。
"""

from __future__ import annotations

import ast

from backend.free.generation.api_contract import (
    FuncSig,
    _decorator_names,
    build_src_api,
)

# __init__ を自動合成するクラスデコレータ (宣言 __init__ の不在を免除する)。
_INIT_SYNTHESIZING_DECORATORS = frozenset({"dataclass", "define", "attrs"})


def _collect_defs_anywhere(
    src_files: dict[str, str],
) -> tuple[set[str], set[str]]:
    """全生成 src の任意深度の定義名と、__init__ 合成クラス名を集める。

    存在 (C1) 判定用。def/class に加え、module 直下の代入束縛
    (``clamp = lambda ...`` / エイリアス ``run = main``) も正当な実装として
    数える (誤検知防止)。
    """
    all_names: set[str] = set()
    init_exempt: set[str] = set()
    for code in src_files.values():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for stmt in tree.body:
            targets: list[ast.expr] = []
            if isinstance(stmt, ast.Assign):
                targets = list(stmt.targets)
            elif isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    all_names.add(t.id)
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                all_names.add(n.name)
                if isinstance(n, ast.ClassDef) and (
                    _decorator_names(n) & _INIT_SYNTHESIZING_DECORATORS
                ):
                    init_exempt.add(n.name)
    return all_names, init_exempt


def _zero_arg_init_ok(msig: object) -> bool:
    """宣言 ``__init__`` が引数 0 で成立するか (object 既定 __init__ で充足)。"""
    if not isinstance(msig, dict):
        return True  # sig 不明 (presence のみの宣言) は既定 __init__ で充足扱い
    return int(msig.get("min_pos", 0)) == 0 or int(msig.get("min_pos_alt", 0)) == 0


def _definitely_incompatible(sig: dict, code_sig: FuncSig) -> bool:
    """spec 宣言通りの位置呼出が code 側で必ず失敗するか (self 両解釈の AND)。"""

    def _incompat(smin: int, smax: int) -> bool:
        # 不足: spec が渡しうる最大の位置引数でも code の必須数に届かない。
        if smax < code_sig.min_pos:
            return True
        # 過多: spec の最小の位置引数でも code の受容上限を超える (varargs 無し)。
        if not code_sig.has_varargs and smin > code_sig.max_pos:
            return True
        return False

    primary = _incompat(int(sig.get("min_pos", 0)), int(sig.get("max_pos", 0)))
    alt = _incompat(
        int(sig.get("min_pos_alt", 0)), int(sig.get("max_pos_alt", 0)),
    )
    return primary and alt


def _sig_range(sig: dict) -> str:
    lo, hi = int(sig.get("min_pos", 0)), int(sig.get("max_pos", 0))
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _code_range(code_sig: FuncSig) -> str:
    hi = "+" if code_sig.has_varargs else str(code_sig.max_pos)
    return (
        str(code_sig.min_pos)
        if str(code_sig.min_pos) == hi else f"{code_sig.min_pos}-{hi}"
    )


def check_spec_conformance(
    declared: list[dict], src_files: dict[str, str], *, primary_path: str = "",
) -> list[str]:
    """宣言契約と生成 src を照合し、確証のある違反メッセージ列を返す。

    ``declared`` は ``spec_contract.parse_declared_contract`` の出力。
    メッセージは英語固定 (リペア指示・spec 見直し judge の evidence として
    自立する文面にする)。違反ゼロ・判定不能はすべて空リスト側へ倒れる。
    """
    violations: list[str] = []
    py_files = {p: c for p, c in src_files.items() if p.endswith(".py")}
    if not py_files or not declared:
        return violations
    # C2/C3 の照合先はファイル毎に解決する。build_src_api の統合 namespace は
    # 同名シンボルを後勝ち上書きするため、兄弟モジュールの同名定義と誤照合して
    # 準拠コードを違反にする (レビュー確定指摘)。primary → 一意所有の順で解決
    # し、多重定義は帰属曖昧 = 判定不能としてスキップする (保守則)。
    # ファイル毎 API では生成物間の基底継承が dynamic 化されるが、これも
    # 「継承メンバ不明 → スキップ」の安全側に倒れる。
    per_file = {p: build_src_api({p: c}) for p, c in py_files.items()}
    all_names, init_exempt = _collect_defs_anywhere(py_files)
    label = primary_path or "the generated sources"

    def _resolve(kind: str, name: str) -> object | None:
        def _get(file_api: object) -> object | None:
            table = (
                file_api.functions if kind == "function" else file_api.classes
            )
            return table.get(name)

        primary_api = per_file.get(primary_path)
        if primary_api is not None:
            found = _get(primary_api)
            if found is not None:
                return found
        owners = [p for p, a in per_file.items() if _get(a) is not None]
        if len(owners) == 1:
            return _get(per_file[owners[0]])
        return None  # 不在 (C1 が扱う) または多重定義 (曖昧 → スキップ)

    for d in declared:
        name = str(d.get("name", "") or "")
        kind = d.get("kind")
        if not name:
            continue

        if kind == "name_only":
            if name not in all_names:
                violations.append(
                    f"spec for `{label}` declares `{name}` but no def/class "
                    f"with that name exists in the generated sources — "
                    f"implement it as specified",
                )
            continue

        if kind == "function":
            if name not in all_names:
                violations.append(
                    f"spec for `{label}` declares function `{name}(...)` but "
                    f"it is not implemented anywhere in the generated "
                    f"sources — implement it with the declared signature",
                )
                continue
            code_fn = _resolve("function", name)
            sig = d.get("sig")
            if (
                code_fn is not None
                and isinstance(sig, dict)
                and _definitely_incompatible(sig, code_fn)
            ):
                violations.append(
                    f"spec for `{label}` declares `{name}` taking "
                    f"{_sig_range(sig)} positional argument(s) but the "
                    f"implementation accepts {_code_range(code_fn)} — a "
                    f"spec-conformant call would always fail; match the "
                    f"declared signature",
                )
            continue

        if kind == "class":
            cls = _resolve("class", name)
            if cls is None:
                if name not in all_names:
                    violations.append(
                        f"spec for `{label}` declares class `{name}` but it "
                        f"is not implemented anywhere in the generated "
                        f"sources — implement it as specified",
                    )
                # 任意深度の別形実装 (ネスト/関数化) / 多重定義は保守的にスキップ
                continue
            if cls.dynamic:
                continue
            members = set(cls.methods) | set(cls.attrs)
            for mname, msig in (d.get("methods") or {}).items():
                if not mname:
                    continue
                if mname == "__init__" and (
                    name in init_exempt
                    # 引数 0 で成立する宣言は object 既定 __init__ で充足
                    # (明示 __init__ を持たない状態レスなクラスを違反にしない)
                    or (mname not in members and _zero_arg_init_ok(msig))
                ):
                    continue
                if mname not in members:
                    violations.append(
                        f"spec for `{label}` declares method "
                        f"`{name}.{mname}(...)` but class `{name}` implements "
                        f"no such method or attribute — implement it as "
                        f"specified",
                    )
                    continue
                code_m = cls.methods.get(mname)
                if (
                    code_m is not None
                    and isinstance(msig, dict)
                    and _definitely_incompatible(msig, code_m)
                ):
                    violations.append(
                        f"spec for `{label}` declares "
                        f"`{name}.{mname}` taking {_sig_range(msig)} "
                        f"positional argument(s) (excluding self) but the "
                        f"implementation accepts {_code_range(code_m)} — a "
                        f"spec-conformant call would always fail; match the "
                        f"declared signature",
                    )
    return violations
