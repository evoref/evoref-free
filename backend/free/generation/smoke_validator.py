"""コード生成物の整合検証 (静的整合ゲート + import スモークテスト)

設計書 f_08 §3.5 の「``validate_python`` は AST 解析のみで実行時 import エラー
(``ModuleNotFoundError``) や dataclass 引数不整合 (``TypeError``) を検出できない」
限界に対し、以下を提供する:

1. :func:`normalize_relative_imports` — flat 出力で解決不能な相対 import の決定的除去
   (除去後に ``import_wirer.wire_imports`` が兄弟定義から正しい絶対 import を再配線)。
2. :func:`check_integrity` — エントリポイント存在 / 残存相対 import の静的検査 (純粋)。
3. :func:`run_import_smoke` — temp dir + サブプロセス import で実行時エラーを捕捉。

(1)(2) は外部リソースを伴わない純粋処理で常時利用可能。(3) はサブプロセスを
起動するため呼出側が config (``long_form.code_smoke_test_enabled``) で gate する。
"""

from __future__ import annotations

import ast
import builtins
import json
import logging
import os
import posixpath
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.core import fs_sandbox
from backend.free.generation.api_contract import (
    bind_local_instances,
    build_src_api,
    collect_attr_uses,
)
from backend.io.process_output import decode_process_output

if TYPE_CHECKING:
    from backend.free.llm.json_schemas import CodeSpec, CodeSpecModule

logger = logging.getLogger("backend.free.generation.smoke_validator")

# Python 標準ライブラリのモジュール名集合 (py3.10+)。``_curses`` 等の私的名を
# ``lstrip("_")`` で正規化して照合するため、欠落モジュールが「ターゲット OS で
# 提供されない標準ライブラリ (起動不能)」か「未インストールの 3rd-party 依存
# (環境要因)」かを区別する (import_wirer._STDLIB_MODULES と同型)。
_STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ()))


def _stems(paths) -> set[str]:
    """ファイルパス集合から「自分の生成物」判定に使うトップレベル名の集合を返す。

    ネストしたパス (``core/game.py``) はトップレベルのパッケージ名 (``core``) を
    返す (``from core.game import ...`` が temp dir でパッケージとして解決される
    場合、失敗時の ``ModuleNotFoundError`` は先頭セグメント ``core`` を指すため)。
    """
    stems: set[str] = set()
    for p in paths:
        head = p.replace("\\", "/").split("/", 1)[0]
        stems.add(head if "/" in p.replace("\\", "/") else os.path.splitext(head)[0])
    return stems


def _module_paths(paths) -> set[str]:
    """各ファイルパスを完全修飾ドット区切りモジュール名に変換する (import smoke 対象)。"""
    return {os.path.splitext(p.replace("\\", "/"))[0].replace("/", ".") for p in paths}


@contextmanager
def _eval_dir(prefix: str) -> Iterator[str]:
    """評価の一時フォルダ。片付けの失敗では例外を上げず、残ったフォルダを WARNING に 1 行出す。

    ウイルス対策等が書いたばかりのファイルを掴むと削除が WinError 32 で落ち、その例外が「起動できなかった」と
    して拾われて、終わった実行の結果が未検査に化けた (f_10 §11.1-3)。
    """
    with tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True) as tmp:
        yield tmp
    if os.path.exists(tmp):
        logger.warning("could not remove the evaluation temp folder %s (a file is still in use)", tmp)


def _run_sandboxed(
    tmp: Path, py_files: dict[str, str], runner: str, args: list[str], exe: str, timeout_sec: float,
    *, stdin=None, base_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """生成物を ``tmp/src`` に書き、``runner`` をスクリプトとして隔離下で実行する (f_10 §11.1-4)。

    ``-c`` で起動すると audit hook を入れられない (入れると import が壊れる) ので、入口を
    ``.sandbox/`` のファイルに書いてから起動する。書込みを許すのは ``src/`` と ``.sandbox`` の
    home / tmp / cwd だけ (違反の記録は許す根の外)。モジュールは CWD (``src/``) から import する。
    """
    src = tmp / "src"
    _write_py_files(str(src), py_files)
    script = fs_sandbox.write_script(
        tmp, "_evoref_runner.py", "import os, sys\nsys.path.insert(0, os.getcwd())\n" + runner,
    )
    # 出力は一時ファイルで受け、終わったら子孫ごと止める (孫が出力を掴むと時間上限が効かなかった)
    return fs_sandbox.run_bounded(
        [exe, str(script), *args],
        cwd=str(src), timeout=timeout_sec, stdin=stdin,
        env=fs_sandbox.sandbox_env(tmp, base_env, write_dirs=[src]),
    )


def _write_py_files(tmp: str, py_files: dict[str, str]) -> None:
    """生成ファイルをネスト構造を保ったまま temp dir に書き出す (namespace package化)。"""
    for path, code in py_files.items():
        dest = os.path.join(tmp, *path.replace("\\", "/").split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(code)


def _resolve_entry_path(files: dict[str, str], entry_module: str) -> str | None:
    """エントリポイント表記に対応する生成ファイルのパスを返す (無ければ None)。

    ``entry_module`` は 3 形態を取り得る: ファイル名 (``main.py``)、素のモジュール名
    (``tetris``)、dotted パッケージ修飾 (``game_of_life.main``)。dotted 修飾は Python
    モジュール表記なので末尾セグメントがファイル stem に対応する
    (``game_of_life.main`` → ``main``)。``os.path.splitext`` で dotted 名を分解すると
    ``.main`` を拡張子として落として ``game_of_life`` を stem 扱いしてしまうため、
    モジュール表記 (``/`` → ``.``) に正規化して full / leaf の両方で一致を見る。
    """
    if entry_module in files:
        return entry_module
    ep = entry_module[:-3] if entry_module.endswith(".py") else entry_module
    ep_full = ep.replace("\\", "/").replace("/", ".")
    ep_leaf = ep_full.rsplit(".", 1)[-1]
    for path in files:
        mod_full = os.path.splitext(path)[0].replace("\\", "/").replace("/", ".")
        if mod_full == ep_full or mod_full.rsplit(".", 1)[-1] == ep_leaf:
            return path
    return None


def _resolve_entry_code(files: dict[str, str], entry_module: str) -> str | None:
    """エントリポイントに対応する生成ファイルの内容を返す (無ければ None)。"""
    path = _resolve_entry_path(files, entry_module)
    return files.get(path) if path is not None else None


def normalize_relative_imports(files: dict[str, str]) -> dict[str, str]:
    """flat 構成で解決不能な相対 import (``from .x import`` / ``from . import``) を除去する。

    生成物は同一ディレクトリの flat な ``.py`` 群として配信されるため、パッケージ
    前提の相対 import は実行時に ``ImportError: attempted relative import with no
    known parent package`` を引き起こす (今回の LINE アプリ失敗の主因)。除去後に
    ``import_wirer.wire_imports`` が兄弟ファイルのトップレベル定義から正しい絶対
    import を再配線する。除去は AST の行範囲ベースで決定的に行い、構文エラーの
    ファイルは触らない (AST 検証側が扱う)。
    """
    out: dict[str, str] = {}
    for path, code in files.items():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            out[path] = code
            continue
        drop_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.level or 0) > 0:
                end = getattr(node, "end_lineno", node.lineno) or node.lineno
                drop_lines.update(range(node.lineno, end + 1))
        if not drop_lines:
            out[path] = code
            continue
        kept = [
            line
            for i, line in enumerate(code.splitlines(keepends=True), start=1)
            if i not in drop_lines
        ]
        out[path] = "".join(kept)
    return out


def check_integrity(
    files: dict[str, str],
    spec: CodeSpec | None = None,
) -> list[str]:
    """生成物の残存整合問題を静的に検出する (サブプロセス不要)。

    - flat 構成に残存する相対 import (``normalize_relative_imports`` 後の取りこぼし)。
    - ``spec.entry_point`` が指定されているのに対応ファイルが無い / ``__main__``
      ガードが無い (起動不能なプログラム = 今回の LINE アプリ失敗の一因)。
    """
    errors: list[str] = []

    for path, code in files.items():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue  # AST 検証 (validate_python) 側が扱う
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.level or 0) > 0:
                errors.append(
                    f"{path}: 相対 import が残存 (flat 構成では解決不可)"
                )
                break

    if spec is not None and spec.entry_point and spec.entry_point.module:
        ep = spec.entry_point.module
        entry_code = _resolve_entry_code(files, ep)
        if entry_code is None:
            errors.append(f"エントリポイント {ep} が生成物に存在しない")
        elif "__main__" not in entry_code:
            errors.append(
                f"エントリポイント {ep} に if __name__ == '__main__' ガードが無い"
            )

    if spec is not None and spec.modules:
        missing = _missing_spec_modules(files, spec.modules)
        if missing:
            # spec.modules[].path (契約合成 LLM) と CodeUnit.file_path (計画合成
            # LLM) は独立した別呼び出しの産物で、両者の対応を強制するコードは
            # 無い。統合/リネーム等の正当な理由もありうるため実失敗にはせず、
            # ドリフトの可視化のみ行う (非ブロッキング)。
            logger.warning(
                "spec で宣言されたモジュールが生成物に見当たらない "
                "(LLM がリネーム/統合した可能性、非致命): %s", ", ".join(missing),
            )

    return errors


def _missing_spec_modules(
    files: dict[str, str], modules: list[CodeSpecModule],
) -> list[str]:
    """``spec.modules[].path`` のうち生成物に対応ファイルが無いものを返す。

    完全一致 → basename 一致 (大小文字無視) の順でフォールバックし、パス表記
    ゆれ (先頭 ``./`` の有無等) による誤検出を避ける。
    """
    declared = [m.path for m in modules if getattr(m, "path", "")]
    generated_base = {
        os.path.splitext(os.path.basename(p))[0].lower() for p in files
    }
    missing = []
    for path in declared:
        if path in files:
            continue
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        if stem in generated_base:
            continue
        missing.append(path)
    return missing


def _is_main_guard(node: ast.stmt) -> bool:
    """``if __name__ == "__main__":`` ガードか判定する。"""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    cmp = node.test
    left = cmp.left
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and len(cmp.ops) == 1
        and isinstance(cmp.ops[0], ast.Eq)
        and len(cmp.comparators) == 1
        and isinstance(cmp.comparators[0], ast.Constant)
        and cmp.comparators[0].value == "__main__"
    )


def check_entrypoint(
    files: dict[str, str],
    spec: CodeSpec | None = None,
) -> list[str]:
    """エントリ起動経路が「存在しないメソッド/属性」を参照していないか静的検出する。

    ``run_import_smoke`` は import が通るかまでしか見ないため、``main()`` が
    ``curses.wrapper(game.run)`` のように **未定義メソッド ``game.run``** を呼ぶ
    起動不能コードを見逃す (今回のテトリス失敗の主因)。これを実行せず AST で捕捉
    する。誤検知 (破壊的リペア誘発) を避け、保守的に「生成物内クラスのインスタンス
    と確証できる変数」への参照のみ照合する。``__getattr__`` / 生成物外基底継承 /
    型不明変数は :mod:`api_contract` 側でスキップされる。
    """
    api = build_src_api(files)
    if not api.classes:
        return []
    known = set(api.classes)

    # エントリ候補コード。spec があればその module、無ければ ``main`` 定義 /
    # ``__main__`` ガードを持つ生成ファイルを候補にする (staged は CodeSpec を持たない)。
    candidates: list[tuple[str, str]] = []  # (label, code)
    ep = getattr(getattr(spec, "entry_point", None), "module", "") if spec else ""
    if ep:
        entry_code = _resolve_entry_code(files, ep)
        if entry_code is None:
            return []  # エントリ欠落は check_integrity が報告する
        candidates.append((ep, entry_code))
    else:
        candidates = [
            (path, code)
            for path, code in files.items()
            if path.endswith(".py") and _looks_like_entry(code)
        ]
    if not candidates:
        return []

    errors: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for label, entry_code in candidates:
        try:
            tree = ast.parse(entry_code)
        except SyntaxError:
            continue  # validate_python 側が扱う
        funcs = {
            n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        # 起動経路ルート = __main__ ガード body + そこから呼ばれる top-level 関数
        # + ``main`` (ガード無しでも main が起点になる今回ケースを拾う)。1 段のみ。
        roots: list[list[ast.stmt]] = []
        called: set[str] = {"main"}
        for n in tree.body:
            if _is_main_guard(n):
                roots.append(n.body)
                for stmt in n.body:
                    for c in ast.walk(stmt):
                        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                            called.add(c.func.id)
        for fname in called:
            if fname in funcs:
                roots.append(funcs[fname].body)

        for body in roots:
            bindings = bind_local_instances(body, known)
            bindings = {v: c for v, c in bindings.items() if c in api.classes}
            for use in collect_attr_uses(body, bindings):
                capi = api.classes[use.cls]
                if capi.dynamic:
                    continue
                if use.attr in capi.methods or use.attr in capi.attrs:
                    continue
                key = (label, use.cls, use.attr)
                if key in seen:
                    continue
                seen.add(key)
                errors.append(
                    f"{label}: エントリ経路が {use.cls}.{use.attr} を参照するが"
                    "定義が無い (起動不能)"
                )
    return sorted(errors)


def check_main_invoked(files: dict[str, str]) -> list[str]:
    """``main.py`` が ``main()`` を定義するのに誰も呼ばない (起動しても何もしない) を検出する。

    import スモークも ``check_entrypoint`` (クラスが無ければ早期 return) も通るため、
    ``python main.py`` が無言で終了する生成物が「起動可能性チェック合格」になった
    (2026-09-19 ライブ監査 K02)。対象は ``main.py`` と、モジュールが 1 本だけの
    ときのそれ — 他モジュールの ``main`` は後続のエントリから呼ばれうる
    (未生成の段階で誤検知しない)。
    """
    errors: list[str] = []
    modules = [
        p for p in files
        if p.endswith(".py")
        and not os.path.basename(p).startswith("test_")
        and os.path.basename(p) != "conftest.py"
    ]
    # main.py に加え、**モジュールが 1 本だけ** ならそれが起点 (単一ファイルの
    # quiz_game.py が main() を呼ばず、無言で終了した。2026-09-21 ライブ監査
    # K06)。複数あるうちの main.py 以外は、後続のエントリから呼ばれうる
    # (staged は全モジュールが揃う前にも検査する)。
    sole = modules[0] if len(modules) == 1 else None
    for path, code in files.items():
        if os.path.basename(path) != "main.py" and path != sole:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue  # validate_python 側が扱う
        defines_main = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main"
            for n in tree.body
        )
        invoked = any(
            _is_main_guard(n)
            or (
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and n.value.func.id == "main"
            )
            for n in tree.body
        )
        if defines_main and not invoked:
            errors.append(
                f"{path}: main() が定義されているが呼ばれない "
                "(if __name__ == '__main__': main() が無く、起動しても何もしない)"
            )
    return errors


def _looks_like_entry(code: str) -> bool:
    """``main`` の top-level 定義か ``__main__`` ガードを持つか (エントリ推論用)。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main":
            return True
        if _is_main_guard(n):
            return True
    return False


# ── 静的整合性チェック (coherence: 重複定義 / 未定義名) ──────────────────
#
# import スモーク (:func:`run_import_smoke`) は「import が通るか」までしか見ない。
# 関数本体内でのみ顕在化する ``NameError`` (どのモジュールにも無いシンボルの使用) や
# 同一モジュール内の重複定義 (生成パスの二重連結 = 後勝ちで前定義が死ぬ) は import
# only では検出できない。これらを AST で決定論的に検出する (サブプロセス不要・非循環)。
# 誤検知は破壊的リペアを招くため、保守的に「確実に未束縛な自由名」「同一ブロック直下の
# 同名 sibling 定義」だけを報告する。

_PSEUDO_NAMES: frozenset[str] = frozenset({
    "__name__", "__file__", "__doc__", "__package__", "__builtins__",
    "__spec__", "__loader__", "__path__", "__dict__", "__class__", "__qualname__",
})

# @overload / property アクセサ等、同名 sibling 定義が正当になるデコレータ。
_REDEF_EXEMPT_DECORATORS: frozenset[str] = frozenset({
    "overload", "property", "setter", "deleter", "getter",
    "cached_property", "singledispatch", "register", "dispatch",
})


def _decorator_names(node) -> set[str]:
    """def/class のデコレータ名集合 (``Name.id`` / ``Attribute.attr``) を返す。"""
    names: set[str] = set()
    for dec in getattr(node, "decorator_list", None) or []:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _is_exempt_redef(node) -> bool:
    """``@overload`` / property アクセサ等で同名 sibling 定義が正当か判定する。"""
    return bool(_decorator_names(node) & _REDEF_EXEMPT_DECORATORS)


def _target_names(target) -> set[str]:
    """代入ターゲット (Name / Tuple / List / Starred) から束縛名を抽出する。"""
    out: set[str] = set()
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for el in target.elts:
            out |= _target_names(el)
    elif isinstance(target, ast.Starred):
        out |= _target_names(target.value)
    return out


def _module_defined_names(tree: ast.Module) -> set[str]:
    """モジュールがトップレベルで公開する名前 (兄弟が import し得る) を返す。"""
    names: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                names |= _target_names(t)
        elif isinstance(n, ast.AnnAssign):
            names |= _target_names(n.target)
        elif isinstance(n, ast.Import):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name != "*":
                    names.add(a.asname or a.name)
        elif isinstance(n, ast.TypeAlias):  # PEP 695 (py3.12+)
            if isinstance(n.name, ast.Name):
                names.add(n.name.id)
    return names


def _duplicate_defs(path: str, tree: ast.Module) -> list[str]:
    """同一ブロック直下の同名 def/class の重複を検出する (条件分岐下は対象外)。"""
    errors: list[str] = []

    def _scan(body, label: str) -> None:
        counts: dict[str, int] = {}
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if _is_exempt_redef(n):
                    continue
                counts[n.name] = counts.get(n.name, 0) + 1
        for name, c in sorted(counts.items()):
            if c >= 2:
                errors.append(f"{path}: {label}'{name}' が重複定義 ({c} 回)")

    _scan(tree.body, "")
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            _scan(
                [m for m in n.body
                 if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))],
                f"クラス '{n.name}' のメソッド ",
            )
    return errors


def _all_bound_names(tree: ast.Module) -> set[str]:
    """モジュール内のどこかで束縛される名前を網羅収集する (関数内含む)。

    未定義名検出の「利用可能集合」を作る保守的 (過剰収集) な集合。関数スコープを
    区別せず全束縛名を集めるため false positive を避ける (false negative は許容)。
    Store-ctx 名で代入 / for / with-as / 内包表記ターゲット / walrus を一括捕捉し、
    引数 (posonly/kwonly/vararg/kwarg/lambda)・except-as・match capture・global/
    nonlocal・全 import alias を補う。
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    bound.add(a.asname or a.name)
        elif isinstance(node, ast.MatchAs) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            bound.add(node.name.id)
    return bound


def _has_dynamic_binding(tree: ast.Module) -> bool:
    """束縛名が静的に確定できない構造 (``import *`` / exec / globals 等) の有無。

    検出時は当該モジュールの未定義名チェックを丸ごとスキップする (= より保守的)。
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            return True
        if isinstance(node, ast.Name) and node.id in {
            "exec", "eval", "globals", "locals", "vars",
        }:
            return True
    return False


def _undefined_names(path: str, tree: ast.Module, cross_module: set[str]) -> list[str]:
    """モジュール内で確実に未束縛な自由 Load 名を検出する (保守的)。

    利用可能 = モジュール内束縛 ∪ builtins ∪ 兄弟モジュール公開名 (cross_module)。
    兄弟が定義する名前は import 配線 (``wire_imports``) で解決可能なため対象外とし、
    **どのモジュールにも存在しない名前** (例: 呼ばれているが未定義の関数) のみ報告する。
    属性アクセス (``obj.attr`` の ``attr``) は対象外。
    """
    if _has_dynamic_binding(tree):
        return []
    available = (
        _all_bound_names(tree)
        | set(dir(builtins))
        | cross_module
        | _PSEUDO_NAMES
    )
    seen: set[str] = set()
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            if name not in available and name not in seen:
                seen.add(name)
                errors.append(f"{path}: 名前 '{name}' が未定義 (import / 定義が無い)")
    return errors


def check_coherence(files: dict[str, str]) -> list[str]:
    """生成物の静的整合性 (重複定義 / 未定義名) を決定論的に検出する。

    ``run_import_smoke`` が見られない「import only では通るが実行時 ``NameError`` に
    なる未定義名」「同一モジュール内の重複定義」を AST で検出する。純粋・サブプロセス
    不要・非循環。誤検知抑制のため保守的に判定する (確証のある問題のみ報告)。
    出力は決定論順 (sorted) で返す。
    """
    parsed: dict[str, ast.Module] = {}
    for path, code in files.items():
        if not path.endswith(".py"):
            continue
        try:
            parsed[path] = ast.parse(code)
        except SyntaxError:
            continue  # validate_python / run_import_smoke 側が扱う
    cross_module: set[str] = set()
    for tree in parsed.values():
        cross_module |= _module_defined_names(tree)
    errors: list[str] = []
    for path, tree in parsed.items():
        errors.extend(_duplicate_defs(path, tree))
        errors.extend(_undefined_names(path, tree, cross_module))
    return sorted(errors)


def check_cross_module_imports(files: dict[str, str]) -> list[str]:
    """生成物間 ``from <sibling> import <name>`` の <name> 実在を静的検証する。

    :func:`run_import_smoke` は外部依存 (pygame 等) が未インストールだと当該モジュール
    の import 自体が ``ModuleNotFoundError`` で止まり、内部の名前欠落
    (``cannot import name 'Game' from 'game'``) を見逃す。これを決定論・サブプロセス
    不要に補完する。対象は **生成物 stem を指す from-import のみ** (stdlib / 3rd-party
    への import は対象外)。再エクスポートが静的に確定できないモジュール (``import *`` /
    exec 等) を import 元とする参照は保留する (誤検知回避)。出力は決定論順 (sorted)。
    """
    parsed: dict[str, ast.Module] = {}
    for path, code in files.items():
        if not path.endswith(".py"):
            continue
        try:
            parsed[path] = ast.parse(code)
        except SyntaxError:
            continue  # validate_python / run_import_smoke 側が扱う
    exports: dict[str, set[str]] = {}
    dynamic_stems: set[str] = set()
    for path, tree in parsed.items():
        stem = os.path.splitext(os.path.basename(path))[0]
        exports[stem] = _module_defined_names(tree)
        if _has_dynamic_binding(tree):
            dynamic_stems.add(stem)
    errors: list[str] = []
    for path, tree in parsed.items():
        own_stem = os.path.splitext(os.path.basename(path))[0]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            sibling = node.module.rsplit(".", 1)[-1]  # dotted パッケージは leaf stem
            if sibling == own_stem:
                # 自己 import (`from breakout import Ball` を breakout.py 自身が
                # 書く)。部分ごと生成で他 part の component を import 参照する
                # 誤解から生じ、実行時に partially-initialized ImportError で
                # 起動不能になる。外部依存欠落が先に発生する環境では runtime
                # smoke がマスクされるため、静的に error とする (2026-07-06 live)。
                errors.append(
                    f"{path}: 自分自身 ('{sibling}') からの import は起動不能 "
                    f"(定義は同一ファイル内にあるため import 文を削除する)"
                )
                continue
            if sibling not in exports or sibling in dynamic_stems:
                continue  # 生成物以外 / 再エクスポート不確定は対象外
            for a in node.names:
                if a.name != "*" and a.name not in exports[sibling]:
                    errors.append(
                        f"{path}: '{a.name}' を '{sibling}' から import できない "
                        f"({sibling} に定義が無い)"
                    )
        errors.extend(_missing_module_attrs(path, tree, own_stem, exports, dynamic_stems))
    return sorted(set(errors))


def _missing_module_attrs(
    path: str, tree: ast.Module, own_stem: str,
    exports: dict[str, set[str]], dynamic_stems: set[str],
) -> list[str]:
    """``import <sibling>`` 経由の ``<sibling>.<name>`` 参照で、定義が無いものを返す。

    from-import だけを照合していたため、``import db`` → ``db.delete_item()`` の
    未定義参照が静的検査を素通りし、実行時に AttributeError で落ちた
    (2026-09-21 ライブ監査 K08: db.py に delete_item が無かった)。
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                sibling = a.name.rsplit(".", 1)[-1]
                if a.asname is None and "." in a.name:
                    continue  # ``import pkg.mod`` の参照は ``pkg.mod.x`` になる
                if sibling != own_stem and sibling in exports and sibling not in dynamic_stems:
                    aliases[a.asname or sibling] = sibling
    if not aliases:
        return []
    # 別名へ代入し直すファイルは束縛が追えないので対象外。
    rebound = {
        t.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for t in node.targets if isinstance(t, ast.Name)
    }
    errors: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name) and node.value.id in aliases
            and node.value.id not in rebound
        ):
            continue
        sibling = aliases[node.value.id]
        if node.attr.startswith("__") or node.attr in exports[sibling]:
            continue
        errors.append(
            f"{path}: '{sibling}.{node.attr}' を参照するが {sibling} に定義が無い"
        )
    return errors


def dedup_top_level_defs(code: str) -> str:
    """同一ブロック直下の同名 def/class 重複を決定論的に除去する (最長 body を保持)。

    生成パスの二重連結で同じ関数/クラスが複数回定義される現象 (後勝ちで前定義が死ぬ・
    可読性破壊) を assembly 段階で解消する。除去対象は「同名かつ非アクセサ
    (= ``@overload`` / property setter 等でない) な top-level sibling」と「同一クラス
    直下の同名メソッド」のみ。固有名の定義は決して落とさない。複数候補からは最長
    body を残す (より完全な実装を保持)。構文エラーは無変更で返す (防御的)。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    drop_lines: set[int] = set()

    def _plan(body) -> None:
        groups: dict[str, list] = {}
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if _is_exempt_redef(n):
                    continue
                groups.setdefault(n.name, []).append(n)
        for nodes in groups.values():
            if len(nodes) < 2:
                continue
            keep = max(nodes, key=lambda n: ((n.end_lineno or n.lineno) - n.lineno))
            for n in nodes:
                if n is keep:
                    continue
                start = n.lineno
                if getattr(n, "decorator_list", None):
                    start = min(start, min(d.lineno for d in n.decorator_list))
                drop_lines.update(range(start, (n.end_lineno or n.lineno) + 1))

    _plan(tree.body)
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            _plan(n.body)
    if not drop_lines:
        return code
    kept = [
        line
        for i, line in enumerate(code.splitlines(keepends=True), start=1)
        if i not in drop_lines
    ]
    return "".join(kept)


@dataclass
class SmokeResult:
    """import スモークテストの結果。

    - ``errors``: 生成物自身の実行時エラー (cross-file の ModuleNotFoundError /
      相対 import / dataclass 引数違い等)。修正対象。
    - ``warnings``: 外部依存未インストール / タイムアウト等の環境要因。修正不要。
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: 作業フォルダの外への書込みを止めたため検査できなかったもの (止めた書込み先、
    #: 記録が無ければモジュール名、f_10 §11.1-4)。不合格ではない (``errors`` に入れない)。
    #: 説明の文は ``warnings`` にも入る。
    unchecked: list[str] = field(default_factory=list)


# サブプロセス内で各モジュールを import し、失敗を JSON で報告する runner。
# __main__ は実行しない (import のみ)。
_SMOKE_RUNNER = (
    "import importlib, json, sys\n"
    "mods = json.loads(sys.argv[1])\n"
    "out = []\n"
    "for m in mods:\n"
    "    try:\n"
    "        importlib.import_module(m)\n"
    "    except BaseException as e:\n"
    "        out.append({'module': m, 'type': type(e).__name__, 'msg': str(e)})\n"
    "print(json.dumps(out))\n"
)


def _classify_failure(
    failure: dict, stems: set[str], component_names: frozenset[str] = frozenset(),
    imported_names: frozenset[str] = frozenset(),
    internal_names: frozenset[str] = frozenset(),
) -> tuple[str, bool]:
    """import 失敗を (メッセージ, is_error) に分類する。

    - 生成物 stem の ``ModuleNotFoundError`` → error (cross-file 欠落)。
    - **生成物内のコンポーネント定義名** (top-level def/class) と一致する欠落
      → error (幻覚 import)。部分ごと生成で LLM が同一ファイル内のコンポーネント
      をモジュールと誤認して ``from initialize_grid import initialize_grid`` の
      ような import を書く実例があり、3rd-party warning に倒すと起動不能コードを
      合格させてしまう (2026-07-06 live)。
    - **標準ライブラリ**の欠落 (``curses``/``_curses`` 等。Windows で POSIX 専用
      stdlib が無い 等) → error。ターゲット OS で起動不能な欠陥であり、warning に
      倒すと「import only では通るが実機で動かない」コードを合格させてしまう。
    - **内部契約名との一致** (case-insensitive の ``component_names`` 一致、
      または欠落モジュールから import している名前 ``imported_names`` が
      内部契約名に含まれる) → error (幻覚内部 import)。planner/spec が宣言する
      プログラム内部の要素を「存在しないモジュール」から import するケース
      (2026-07-07 live: `from game import Game` — Game は spec の Component だが
      game.py は正準に無く、外部依存 warning に降格されて偽 success で配信)。
    - それ以外 (numpy 等の未インストール 3rd-party) → warning (環境要因・修正不要)。
    """
    module = failure.get("module", "?")
    etype = failure.get("type", "Error")
    msg = failure.get("msg", "")
    if etype == "ModuleNotFoundError":
        missing = ""
        # "No module named 'X'" から X を抽出
        if "'" in msg:
            missing = msg.split("'")[1].split(".")[0]
        if missing and missing not in stems:
            if missing in component_names:
                return (
                    f"{module}: 幻覚 import '{missing}' — モジュールではなく"
                    "生成物内のコンポーネント定義名 (同名の def/class が存在する。"
                    "この import を削除し、定義を直接使うこと)",
                    True,
                )
            # ``_curses`` → ``curses`` 等、私的名を正規化して stdlib 判定。
            # stdlib 判定は内部契約名照合より先 (component 名と大文字小文字違いの
            # stdlib を幻覚扱いしないため)。
            norm = missing.lstrip("_")
            if missing in _STDLIB_MODULES or norm in _STDLIB_MODULES:
                return (
                    f"{module}: ターゲット OS で標準ライブラリ '{missing}' が"
                    "利用不可 (起動不能)",
                    True,
                )
            contract = component_names | internal_names
            lowered = {n.lower() for n in contract}
            if missing.lower() in lowered or (
                imported_names and imported_names & contract
            ):
                return (
                    f"{module}: 幻覚 import '{missing}' — このプログラム内部の"
                    "契約名 (spec の Component / 正準モジュール) であり外部"
                    "パッケージではない (この import を削除し、当該定義を"
                    "プログラム内で実装・参照すること)",
                    True,
                )
            return (f"{module}: 外部依存 '{missing}' が未インストール", False)
    return (f"{module}: {etype}: {msg}", True)


def _top_level_component_names(py_files: dict[str, str]) -> frozenset[str]:
    """生成物全ファイルの top-level def/class 名を集める (幻覚 import 判定用)。

    構文エラーのファイルは無視する (防御的)。
    """
    names: set[str] = set()
    for code in py_files.values():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
    return frozenset(names)


def _names_imported_from(py_files: dict[str, str], missing: str) -> frozenset[str]:
    """生成物全ファイル中で ``from <missing> import ...`` されている名前を集める。

    欠落モジュールが「内部契約名を輸出する幻覚モジュール」かの判定材料
    (`from game import Game` → {"Game"})。構文エラーのファイルは無視する。
    """
    names: set[str] = set()
    for code in py_files.values():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and (node.module or "").split(".")[0] == missing
            ):
                names.update(a.name for a in node.names if a.name != "*")
    return frozenset(names)


def run_import_smoke(
    files: dict[str, str],
    timeout_sec: float = 10.0,
    python_exe: str | None = None,
    *,
    internal_names: frozenset[str] = frozenset(),
) -> SmokeResult:
    """生成ファイルを temp dir に書き出し、各モジュールを import してエラーを収集する。

    ``__main__`` は実行せず import のみ行う。top-level 副作用 (サーバ起動等) は
    ``timeout_sec`` で有界化する。書込/実行不可・タイムアウト時は warning に倒し、
    静的ゲートのみで継続する (生成失敗にはしない)。

    ``internal_names`` は呼出側が持つ「プログラム内部の契約名」(staged の場合は
    spec の Component 名 + 正準モジュール stem)。生成コードの top-level 定義名と
    合流させ、幻覚内部 import (契約名を存在しないモジュールから import) を
    外部依存 warning に降格させないための追加判定材料。
    """
    result = SmokeResult()
    py_files = {p: c for p, c in files.items() if p.endswith(".py")}
    if not py_files:
        return result
    stems = _stems(py_files)
    module_paths = _module_paths(py_files)
    component_names = _top_level_component_names(py_files)
    exe = python_exe or sys.executable

    violations: list[str] = []
    try:
        with _eval_dir("evoref_smoke_") as tmp:
            # import 時の副作用で一時フォルダの外へ書かせない (f_10 §11.1-4)
            proc = _run_sandboxed(
                Path(tmp), py_files, _SMOKE_RUNNER, [json.dumps(sorted(module_paths))], exe, timeout_sec,
            )
            violations = fs_sandbox.violation_paths(fs_sandbox.read_violations(Path(tmp)))
    except subprocess.TimeoutExpired:
        result.warnings.append(
            f"import スモークテストが {timeout_sec:.0f}s でタイムアウト "
            "(top-level 副作用の可能性)"
        )
        return result
    except Exception as e:
        result.warnings.append(f"import スモークテスト実行不可: {e}")
        return result

    try:
        failures = json.loads(decode_process_output(proc.stdout).strip() or "[]")
    except (ValueError, TypeError):
        result.warnings.append(
            "import スモークテスト出力を解析できません: "
            f"{decode_process_output(proc.stderr)[:200]}"
        )
        return result

    seen: set[str] = set()
    for path in violations:
        result.unchecked.append(path)
        result.warnings.append(f"import スモーク: 未検査 — 作業フォルダの外への書込みを止めた ({path})")
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        msg = str(failure.get("msg", ""))
        if fs_sandbox.is_sandbox_error(msg):
            # 止めた書込みで import が落ちたのはコードの誤りの証拠ではない (未検査)
            if not violations:
                result.unchecked.append(str(failure.get("module", "?")))
                result.warnings.append(f"{failure.get('module', '?')}: import スモーク: 未検査 — {msg}")
            continue
        missing = msg.split("'")[1].split(".")[0] if "'" in msg else ""
        imported = (
            _names_imported_from(py_files, missing) if missing else frozenset()
        )
        message, is_error = _classify_failure(
            failure, stems, component_names, imported, internal_names,
        )
        if message in seen:
            continue
        seen.add(message)
        (result.errors if is_error else result.warnings).append(message)
    return result


# サブプロセス内で、欠落モジュール (curses 等) をダミー充足してエントリモジュールを
# import し、引数不要のクラスを構築して引数不要の公開メソッドを呼ぶ runner。
# __main__ (ブロッキングなゲームループ等) は実行しない。構築失敗のみ報告し、
# メソッド単体の例外は無視する (stdscr 不足等のノイズを避けるため)。timeout が
# 無限ループの防壁。結果は **すべて advisory (warning)**。
_ENTRY_SMOKE_RUNNER = r'''
import importlib, importlib.util, inspect, json, sys, types

class _Any:
    def __getattr__(self, k): return _Any()
    def __call__(self, *a, **k): return _Any()
    def __iter__(self): return iter(())

class _Loader:
    def __init__(self, name): self.name = name
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__getattr__ = lambda k: _Any()
        return m
    def exec_module(self, m): pass

class _Finder:
    # 実ファイル/実 stdlib の解決に失敗した時だけダミーを合成する (meta_path 末尾)。
    def find_spec(self, name, path=None, target=None):
        return importlib.util.spec_from_loader(name, _Loader(name))

sys.meta_path.append(_Finder())

target = sys.argv[1]
P = inspect.Parameter
issues = []

def _required(sig):
    return [p for p in sig.parameters.values()
            if p.default is p.empty
            and p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD, P.KEYWORD_ONLY)]

try:
    mod = importlib.import_module(target)
except BaseException as e:
    print(json.dumps({"import_error": "%s: %s" % (type(e).__name__, e)}))
    sys.exit(0)

# --- 引数付き純関数の軽量呼出し (型注釈からサンプル値を合成) ---------------
# 引数不要のものしか呼べないと、ライブラリの中心的な関数 (例:
# is_balanced(text: str)) が一度も実行されないまま「起動可能性チェック合格」
# として配信される (実インシデント 2026-07-27: 閉じ括弧のたびに KeyError で
# 落ちる is_balanced が静的検証を通過し、全ての正常入力で例外になった)。
# 型注釈からサンプル値を作れる関数だけを、副作用が疑われる名前を除いて呼ぶ。
_SAMPLES = {
    str: ["", "test", "()[]{}", "a (b) [c] {d}"],
    int: [0, 1, 2],
    float: [0.0, 1.5],
    bool: [True, False],
    list: [[], ["a", "b"]],
    dict: [{}, {"a": 1}],
    tuple: [(), ("a",)],
    set: [set(), {"a"}],
}
# 論理エラーを示す例外だけを issue にする。ValueError / OSError 等は
# 「サンプル値がその関数の入力仕様に合わなかった」だけの可能性が高い。
_LOGIC_ERRORS = (
    KeyError, IndexError, TypeError, AttributeError, NameError,
    ZeroDivisionError, UnboundLocalError, RecursionError,
)
_SIDE_EFFECT_NAME = (
    "write", "save", "delete", "remove", "send", "post", "upload", "download",
    "run", "exec", "main", "install", "drop", "clear", "reset", "update",
    "create", "open", "connect", "commit", "push", "kill", "shutdown",
)


def _sample_args(sig):
    """全必須引数に安全なサンプル値を割り当てた候補リストを返す (作れなければ None)。"""
    required = _required(sig)
    if not required or len(required) > 3:
        return None
    per_param = []
    for p in required:
        ann = p.annotation
        if ann is p.empty or ann not in _SAMPLES:
            return None
        per_param.append(_SAMPLES[ann])
    rounds = max(len(v) for v in per_param)
    calls = []
    for i in range(rounds):
        calls.append([vals[i % len(vals)] for vals in per_param])
    return calls


for fname in dir(mod):
    fn = getattr(mod, fname, None)
    if fname.startswith("_") or not inspect.isfunction(fn):
        continue
    if getattr(fn, "__module__", "") != mod.__name__:
        continue
    if any(w in fname.lower() for w in _SIDE_EFFECT_NAME):
        continue
    try:
        calls = _sample_args(inspect.signature(fn))
    except (TypeError, ValueError):
        continue
    if not calls:
        continue
    for args in calls:
        try:
            fn(*args)
        except _LOGIC_ERRORS as e:
            issues.append(
                "%s(%s) が %s: %s" % (
                    fname, ", ".join(repr(a) for a in args), type(e).__name__, e,
                )
            )
            break
        except BaseException:
            break  # 入力仕様の不一致等は判定不能として次の関数へ

for cname in dir(mod):
    obj = getattr(mod, cname, None)
    if not inspect.isclass(obj) or getattr(obj, "__module__", "") != mod.__name__:
        continue
    try:
        if _required(inspect.signature(obj)):
            continue  # 引数必須クラスは構築しない (誤検知回避)
        inst = obj()
    except BaseException as e:
        issues.append("%s() の構築に失敗: %s: %s" % (cname, type(e).__name__, e))
        continue
    for mname in dir(inst):
        if mname.startswith("_"):
            continue
        try:
            meth = getattr(inst, mname)
            if not callable(meth) or _required(inspect.signature(meth)):
                continue
            meth()  # 戻り値は無視。例外も無視 (stdscr 不足等のノイズ回避)。
        except BaseException:
            pass

print(json.dumps({"issues": issues}))
'''


def run_entry_smoke(
    files: dict[str, str],
    spec: "CodeSpec | None" = None,
    timeout_sec: float = 10.0,
    python_exe: str | None = None,
) -> SmokeResult:
    """エントリモジュールを有界実行し、構築失敗 / ハング / クラッシュを **warning** で返す。

    静的な :func:`check_entrypoint` を補完する advisory 層。``__main__`` は実行せず、
    引数不要のクラス構築＋引数不要の公開メソッド呼び出しのみ行う。欠落モジュール
    (curses 等) はダミー合成で import を通す。タイムアウト / サブプロセス異常は
    warning に倒し、合否ゲートには影響させない (呼出側で warnings として扱う)。
    """
    result = SmokeResult()
    py_files = {p: c for p, c in files.items() if p.endswith(".py")}
    if not py_files:
        return result
    # エントリモジュールを決定。spec があればその module、無ければ ``main`` 定義 /
    # ``__main__`` ガードを持つファイルを推論 (staged は CodeSpec を持たない)。
    ep = getattr(getattr(spec, "entry_point", None), "module", "") if spec else ""
    if ep:
        entry_path = _resolve_entry_path(py_files, ep)
        if entry_path is None:
            return result
    else:
        entry_path = next(
            (p for p, c in py_files.items() if _looks_like_entry(c)), None
        )
        if entry_path is None:
            return result
    ep_module = os.path.splitext(entry_path.replace("\\", "/"))[0].replace("/", ".")
    exe = python_exe or sys.executable
    try:
        with _eval_dir("evoref_entry_") as tmp:
            proc = _run_sandboxed(Path(tmp), py_files, _ENTRY_SMOKE_RUNNER, [ep_module], exe, timeout_sec)
    except subprocess.TimeoutExpired:
        result.warnings.append(
            f"エントリ実行スモークが {timeout_sec:.0f}s でタイムアウト (要確認)"
        )
        return result
    except Exception as e:
        result.warnings.append(f"エントリ実行スモーク実行不可: {e}")
        return result

    try:
        data = json.loads(decode_process_output(proc.stdout).strip() or "{}")
    except (ValueError, TypeError):
        return result  # 解析不能は黙って無視 (advisory)
    if isinstance(data, dict):
        if data.get("import_error"):
            result.warnings.append(f"エントリ実行スモーク: {data['import_error']}")
        for issue in data.get("issues", []) or []:
            result.warnings.append(f"エントリ実行スモーク: {issue}")
    return result


@dataclass
class UsageRun:
    """使い方を配信形で 1 回実行した結果 (f_10 §11.1-3)。判定は ``package_layout.usage_outcome``。"""

    #: 終了コード (時間切れ・起動できなかったときは ``None``)。
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    #: 隔離が止めた書込み先 (あれば未検査)。
    unchecked: list[str] = field(default_factory=list)
    #: 起動できなかった理由。
    error: str = ""
    #: 時間上限 (秒、時間切れの文面に出す)。
    timeout_sec: float = 0.0


# ``python -m <module> <args>`` と同じ形で起こす入口。``-m`` で起動すると audit hook が入らない
# (fs_sandbox_runtime) のでスクリプトとして起動し、runpy で ``__main__`` として走らせる
# (``sys.argv[0]`` は ``__main__.py`` のパスになる)。
_USAGE_RUNNER = (
    "import json, runpy\n"
    "module = sys.argv[1]\n"
    "sys.argv = [module] + json.loads(sys.argv[2])\n"
    "runpy.run_module(module, run_name='__main__', alter_sys=True)\n"
)


#: 使い方の実行の子プロセスへ渡す環境変数 (これと ``PYTHON*`` だけ)。API キー・``EVOREF_*`` 等の秘密を
#: 生成コードに見せない (独立レビュー 高)。Windows の Python とソケット・乱数は SYSTEMROOT を要る。
_USAGE_ENV_KEYS = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE", "OS", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "VIRTUAL_ENV",
})


def usage_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """使い方の実行に渡す最小の環境変数 (:data:`_USAGE_ENV_KEYS` と ``PYTHON*``)。"""
    source = os.environ if environ is None else environ
    return {k: v for k, v in source.items() if k.upper() in _USAGE_ENV_KEYS or k.upper().startswith("PYTHON")}


def run_usage(
    files: dict[str, str], module: str, args: list[str], timeout_sec: float = 30.0,
    python_exe: str | None = None,
) -> UsageRun:
    """配信形 (``units/__init__.py`` … と依頼されたデータファイル) を一時フォルダに置き、その親を CWD にして
    ``python -m <module> <args>`` を隔離の下で 1 回実行する (f_10 §11.1-3 / §11.1-4)。

    stdin は空。書込みを許すのは一時フォルダの中だけで、止めた書込みは ``unchecked`` に入る。環境変数は
    :func:`usage_env` の最小の組だけを渡す。シェル・別プロセス・ソケット等は隔離が止めないので、呼出側が
    ``package_layout.usage_side_effect`` で実行前に外す。
    """
    exe = python_exe or sys.executable
    try:
        with _eval_dir("evoref_usage_") as tmp:
            try:
                proc = _run_sandboxed(
                    Path(tmp), files, _USAGE_RUNNER, [module, json.dumps(list(args))], exe, timeout_sec,
                    stdin=subprocess.DEVNULL, base_env=usage_env(),
                )
            except subprocess.TimeoutExpired as expired:
                return UsageRun(
                    returncode=None, stdout=decode_process_output(expired.output),
                    stderr=decode_process_output(expired.stderr), timed_out=True,
                    unchecked=fs_sandbox.violation_paths(fs_sandbox.read_violations(Path(tmp))),
                    timeout_sec=timeout_sec,
                )
            violations = fs_sandbox.violation_paths(fs_sandbox.read_violations(Path(tmp)))
    except Exception as e:  # noqa: BLE001 - 起動できないのはコードの誤りの証拠ではない (未検査)
        return UsageRun(returncode=None, stdout="", stderr="", error=str(e), timeout_sec=timeout_sec)
    return UsageRun(
        returncode=proc.returncode,
        stdout=decode_process_output(proc.stdout),
        stderr=decode_process_output(proc.stderr),
        unchecked=violations,
        timeout_sec=timeout_sec,
    )


def _signature_of(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    """モジュール直下の関数の引数の形 (位置の上限・必須・名前・可変長)。"""
    a = func.args
    positional = [p.arg for p in (*a.posonlyargs, *a.args)]
    n_defaults = len(a.defaults)
    required = positional[: len(positional) - n_defaults] if n_defaults else list(positional)
    kwonly_required = [
        k.arg for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is None
    ]
    return {
        "positional": positional,
        "required": required + kwonly_required,
        "names": set(positional) | {k.arg for k in a.kwonlyargs},
        "posonly": {p.arg for p in a.posonlyargs},
        "varargs": a.vararg is not None,
        "varkw": a.kwarg is not None,
        "shown": f"{func.name}({', '.join(positional + [k.arg for k in a.kwonlyargs])})",
    }



def _declared_signature(text: str) -> dict | None:
    """骨組みの signature 文字列 (``def f(a, b) -> None``) を :func:`_signature_of` の形へ。"""
    src = (text or "").strip()
    if not src.startswith(("def ", "async def ")):
        return None
    try:
        tree = ast.parse(src.rstrip(":") + ":\n    pass\n")
    except SyntaxError:
        return None
    node = tree.body[0] if tree.body else None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _signature_of(node)
    return None


def _call_problem(sig: dict, n_pos: int, keywords: set[str]) -> str | None:
    """呼び出し (位置の数・キーワード名) が signature に合わない理由 (合えば None)。"""
    if not sig["varargs"] and n_pos > len(sig["positional"]):
        return f"は引数 {len(sig['positional'])} 個まで"
    unknown = sorted(k for k in keywords if k not in sig["names"] or k in sig["posonly"])
    if unknown and not sig["varkw"]:
        return f"に無い引数 {', '.join(unknown)} を渡している"
    given = set(sig["positional"][:n_pos]) | keywords
    missing = [p for p in sig["required"] if p not in given]
    if missing:
        return f"に必須の引数 {', '.join(missing)} が無い"
    return None


def _bound_names(nodes: list) -> set[str]:
    """文の並び (ネストした if / for / with / try を含む) で束縛される名前。

    関数・クラスの本体の中は数えない (別スコープ)。import で束縛した名前は数えない
    (照合の対象そのもの)。
    """
    names: set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda)):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
            continue
        # ``except … as f`` / match の capture (``case f:`` / ``*f`` / ``**f``)
        if isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _import_bound_names(nodes: list) -> set[str]:
    """文の並びの中の import が束縛する名前 (ネストした関数・クラスの中は数えない)。"""
    names: set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
            continue
        stack.extend(ast.iter_child_nodes(node))
    return names


def _function_locals(node) -> set[str]:
    """関数 (lambda を含む) の引数とローカルで束縛される名前。"""
    a = node.args
    names = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    if not isinstance(node, ast.Lambda):
        # 関数内の import もローカルの束縛 (モジュール側の import を遮蔽する)
        names |= _bound_names(node.body) | _import_bound_names(node.body)
    return names


def check_call_arity(
    files: dict[str, str], *, declared: dict[str, str] | None = None,
) -> list[str]:
    """生成物のモジュール間呼び出しの引数の数を、定義と静的に照合する (純粋関数)。

    import smoke と :func:`check_cross_module_imports` は名前の実在しか見ないので、
    ``data_handler.save_expense(a, c, d)`` と ``def save_expense(amount, category)`` の
    食い違いは実行するまで分からなかった (2026-09-27 実機、TypeError)。

    対象は **生成物のモジュール直下の、デコレータの無い関数** を ``import m`` /
    ``from m import f`` (相対 import を含む) 経由で呼ぶ箇所だけ。import 先はファイルの
    パスと一致するときだけ対応づける (``os.path`` は ``path.py`` ではない。同じ stem が
    複数あれば dotted で一意に決まるときだけ)。``*args`` / ``**kw`` を渡す呼び出し・
    メソッド・呼び出し側のスコープで束縛し直した名前 (引数・ローカル・再代入・条件内の
    再定義) は判定しない (誤検知回避、2026-09-27 独立レビュー)。

    ``declared`` (``"stem.関数名"`` → 骨組みの signature) があり、定義が骨組みと合わず
    呼び出しは骨組みに合うなら、誤りを **定義側のファイル** に帰属させる — 呼び出し側に
    機能を落として合わせさせない。出力は決定論順。
    """
    parsed: dict[str, ast.Module] = {}
    for path, code in files.items():
        if path.endswith(".py"):
            try:
                parsed[path.replace("\\", "/")] = ast.parse(code)
            except SyntaxError:
                continue
    by_stem: dict[str, list[str]] = {}
    for path in parsed:
        by_stem.setdefault(posixpath.splitext(posixpath.basename(path))[0], []).append(path)

    def resolve(module: str, level: int, importer: str) -> str | None:
        parts = module.split(".") if module else []
        if level:
            base = posixpath.dirname(importer)
            for _ in range(level - 1):
                base = posixpath.dirname(base)
            if not parts:
                return None
            cand = posixpath.join(base, *parts) + ".py" if base else "/".join(parts) + ".py"
            return cand if cand in parsed else None
        if not parts:
            return None
        if len(parts) == 1:
            hits = by_stem.get(parts[0], [])
            return hits[0] if len(hits) == 1 else None
        suffix = "/".join(parts) + ".py"
        hits = [p for p in parsed if p == suffix or p.endswith("/" + suffix)]
        return hits[0] if len(hits) == 1 else None

    sigs: dict[str, dict[str, dict]] = {
        path: {
            node.name: _signature_of(node)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.decorator_list
        }
        for path, tree in parsed.items()
    }
    declared_sigs = {
        key: sig for key, text in (declared or {}).items()
        if (sig := _declared_signature(text)) is not None
    }
    errors: set[str] = set()
    for path, tree in parsed.items():
        modules: dict[str, str] = {}              # 別名 → 定義ファイル
        funcs: dict[str, tuple[str, str]] = {}    # 名前 → (定義ファイル, 関数名)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = resolve(alias.name, 0, path)
                    if target and target != path and (alias.asname or "." not in alias.name):
                        modules[alias.asname or alias.name] = target
            elif isinstance(node, ast.ImportFrom):
                target = resolve(node.module or "", node.level, path)
                for alias in node.names:
                    if target and target != path and alias.name in sigs[target]:
                        funcs[alias.asname or alias.name] = (target, alias.name)
                        continue
                    sub = resolve(
                        ".".join(x for x in (node.module, alias.name) if x), node.level, path,
                    )
                    if sub and sub != path:
                        modules[alias.asname or alias.name] = sub
        rebound = _bound_names(tree.body)
        funcs = {k: v for k, v in funcs.items() if k not in rebound}
        modules = {k: v for k, v in modules.items() if k not in rebound}
        if not funcs and not modules:
            continue

        def check(node: ast.Call, shadowed: frozenset[str]) -> None:
            target: tuple[str, str] | None = None
            func = node.func
            if isinstance(func, ast.Name) and func.id in funcs and func.id not in shadowed:
                target = funcs[func.id]
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in modules
                and func.value.id not in shadowed
                and func.attr in sigs[modules[func.value.id]]
            ):
                target = (modules[func.value.id], func.attr)
            if target is None:
                return
            if any(isinstance(a, ast.Starred) for a in node.args) or any(
                k.arg is None for k in node.keywords
            ):
                return
            def_path, fname = target
            sig = sigs[def_path][fname]
            n_pos = len(node.args)
            kws = {k.arg for k in node.keywords if k.arg}
            problem = _call_problem(sig, n_pos, kws)
            if problem is None:
                return
            stem = posixpath.splitext(posixpath.basename(def_path))[0]
            # 同じ stem のファイルが複数あると骨組みの宣言がどれを指すか決まらない
            decl = declared_sigs.get(f"{stem}.{fname}") if len(by_stem.get(stem, [])) == 1 else None
            if decl is not None and _call_problem(decl, n_pos, kws) is None:
                errors.add(
                    f"{def_path}: {fname}() の定義が骨組み ({decl['shown']}) と合わず、"
                    f"{path} の呼び出し (位置 {n_pos} 個) が失敗する (定義: {sig['shown']})",
                )
                return
            if problem.startswith("は引数"):
                errors.add(
                    f"{path}: {stem}.{fname}() {problem} (定義: {sig['shown']}) "
                    f"なのに {n_pos} 個で呼んでいる",
                )
            else:
                errors.add(f"{path}: {stem}.{fname}() {problem} (定義: {sig['shown']})")

        def visit(node, shadowed: frozenset[str]) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                inner = frozenset(shadowed | _function_locals(node))
                for child in ast.iter_child_nodes(node):
                    visit(child, inner)
                return
            if isinstance(node, ast.ClassDef):
                # クラス本体での再束縛 (メソッドへは漏れないが、保守的に本体全体で除く)
                inner = frozenset(
                    shadowed | _bound_names(node.body) | _import_bound_names(node.body),
                )
                for child in ast.iter_child_nodes(node):
                    visit(child, inner)
                return
            if isinstance(node, ast.Call):
                check(node, shadowed)
            for child in ast.iter_child_nodes(node):
                visit(child, shadowed)

        visit(tree, frozenset())
    return sorted(errors)
