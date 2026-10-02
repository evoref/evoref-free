"""契約テスト — 骨組みの入出力例から pytest を組み、LLM が書いたテストを検査する (f_10 §11)。

staged v2 (Pro) のテスト工程の決定論部分。純粋関数だけを置く。

- :func:`build_example_tests`: 骨組みの ``examples`` (Python 式 → 期待値) から、LLM を使わずに
  pytest を組む。**合否を決めるのはこのテストだけ** (正の順序: 依頼 > 骨組み > コード > テスト)。
- :func:`lint_generated_tests`: LLM が書いた参考テストから、壊れやすい検査をするテスト関数を落とす。
  標準出力の差し替え (``sys.stdout = …``)、``__annotations__`` の比較、骨組みに無い文言との完全一致、
  本文が ``__file__`` 基準の置き場を持つのに chdir だけでする隔離 (:func:`file_based_data_locations`)。
  2026-09-25: 生成テストが ``sys.stdout`` を StringIO に差し替えたまま戻さず、2 回目の確認で 2 行が
  溜まって落ちた。文言も骨組みに無い和文 (「引数は…」) と完全一致させていた。
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from backend.free.core.check_outcome import UncheckedReason
from backend.free.core.fs_sandbox_runtime import ROOTS_ENV
from backend.free.core.intent_vocab import is_absolute_path_text


@dataclass(frozen=True)
class ExampleCase:
    """組めた入出力例 1 件。"""

    module: str  # import 名 (拡張子なしのモジュール名)
    call: str
    expected: str
    #: 骨組みの signature が宣言する戻り型 (``-> list[str]`` の ``list[str]``)。無ければ空
    returns: str = ""


#: 戻り型が具体的な入れ物の宣言 (反復子を返したら具体化せず不合格にする)
CONTAINER_TYPES = ("list", "tuple", "set", "frozenset", "dict", "str", "bytes")


def _called_name(call: str) -> str:
    """式の外側の呼出しの関数名 (``f(…)`` の ``f``)。属性の呼出し・呼出しでない式は空。"""
    node = ast.parse(call, mode="eval").body
    return node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else ""


def _declared_returns(skeleton: dict, module: str, name: str) -> str:
    """骨組みの ``module`` の部品のうち関数 ``name`` の signature が宣言する戻り型 (無ければ空)。"""
    for mod in skeleton.get("modules") or []:
        if not isinstance(mod, dict) or PurePosixPath(str(mod.get("path") or "")).stem != module:
            continue
        for comp in mod.get("components") or []:
            src = str((comp or {}).get("signature") or "").strip() if isinstance(comp, dict) else ""
            if not src.startswith(("def ", "async def ")):
                continue
            try:
                func = ast.parse(src.rstrip(":") + ":\n    pass\n").body[0]
            except (SyntaxError, ValueError, IndexError):
                continue
            if isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef) and func.name == name and func.returns:
                text = ast.unparse(func.returns)
                # 使うのは具体的な入れ物の宣言だけ (反復子を返したら不合格にする判定)
                head = text.split("[", 1)[0].rsplit(".", 1)[-1].strip().lower()
                return text if head in CONTAINER_TYPES else ""
    return ""


def _is_expr(text: str) -> bool:
    try:
        ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        return False
    return True


def _is_literal(text: str) -> bool:
    try:
        ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return False
    return True


_FILE_NAME_RE = re.compile(r"[\\/]|\.[A-Za-z0-9]{1,5}$")


def _touches_files(call: str) -> bool:
    """式がファイルを名指すか (``sum_column('data.csv', …)`` / ``open(…)``)。

    骨組みの例は純粋な呼出しに限る約束だが、モデルは実在しないファイルを引数に取る例を
    書く。そのままでは正しいコードが FileNotFoundError で「契約テスト不合格」になる
    (2026-09-25 create ベンチ s3 の偽の失敗)。
    """
    for node in ast.walk(ast.parse(call, mode="eval")):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _FILE_NAME_RE.search(node.value):
            return True
        if isinstance(node, ast.Name) and node.id == "open":
            return True
    return False


def usable_examples(skeleton: dict, module_paths: list[str]) -> list[ExampleCase]:
    """骨組みの例のうち、テストに組めるもの (式と期待値が構文として正しく、モジュールが実在し、
    ファイルを名指さない)。"""
    stems = {PurePosixPath(p).stem for p in module_paths if p.endswith(".py")}
    out: list[ExampleCase] = []
    for ex in skeleton.get("examples") or []:
        call = str(ex.get("call") or "").strip()
        expected = str(ex.get("expected") or "").strip()
        module = PurePosixPath(str(ex.get("module") or "")).stem
        if not call or not expected or module not in stems:
            continue
        if _is_expr(call) and _is_literal(expected) and not _touches_files(call):
            returns = _declared_returns(skeleton, module, _called_name(call))
            out.append(ExampleCase(module, call, expected, returns))
    return out


def malformed_example_count(skeleton: dict, module_paths: list[str]) -> int:
    """骨組みの例のうち、形が崩れていて組めないものの件数 (式の構文エラー / 期待値がリテラルでない)。

    ファイルを名指す例は意図した除外なので数えない。組める例が 0 件でこれが 1 件以上なら、契約テストは
    「入出力例なし」ではなく「未検査 (入出力例が不正)」(f_10 §12.4、2026-09-27 ライブ監査 M7: K05 の例は
    閉じの引用符が無く、黙って「入出力例なし」と表示された)。
    """
    stems = {PurePosixPath(p).stem for p in module_paths if p.endswith(".py")}
    count = 0
    for ex in skeleton.get("examples") or []:
        call = str(ex.get("call") or "").strip()
        expected = str(ex.get("expected") or "").strip()
        module = PurePosixPath(str(ex.get("module") or "")).stem
        if not (call and expected and module in stems):
            continue
        if _is_expr(call) and _touches_files(call):
            continue  # 2026-10-02 ライブ監査 K05: 期待値が和文でもファイルを名指す例は意図した除外
        if not (_is_expr(call) and _is_literal(expected)):
            count += 1
    return count


def file_dependent_example_count(skeleton: dict, module_paths: list[str]) -> int:
    """骨組みの例のうち、ファイルを名指すので契約から外した件数 (:func:`malformed_example_count` には数えない)。"""
    stems = {PurePosixPath(p).stem for p in module_paths if p.endswith(".py")}
    count = 0
    for ex in skeleton.get("examples") or []:
        call = str(ex.get("call") or "").strip()
        module = PurePosixPath(str(ex.get("module") or "")).stem
        if call and str(ex.get("expected") or "").strip() and module in stems and _is_expr(call) and _touches_files(call):
            count += 1
    return count


#: 例の結果が遅延評価の反復子のとき、比べる前に取り出す件数と時間の上限。骨組みの例の期待値は
#: 手で書くリテラルなので 1 万件を超える期待は無い。上限を越えたら (無限のジェネレータ等) 固定値と比べられない例として外す。
#: 1 回の ``next`` が戻らない反復子は止められない (テストの実行全体の時間上限に任せる)。
LAZY_MAX_ITEMS = 10_000
LAZY_MAX_SECONDS = 5.0

#: 契約テストが例を skip するときの理由の書き出し (:func:`examples_skip_reason` が読み分ける)
SKIP_FILES = "example reads or writes files"
SKIP_ENVIRONMENT = "example depends on the environment"
SKIP_NONDETERMINISTIC = "example is not deterministic"
SKIP_TOO_LONG = "example returns an iterator too long to compare"


def examples_skip_reason(messages: list[str]) -> UncheckedReason:
    """全件 skip した契約テストの未検査の理由 (pytest の skip の理由の列から)。

    ファイル・環境だけなら :attr:`~UncheckedReason.STATEFUL_EXAMPLES`、呼ぶたびに値が変わるだけなら
    :attr:`~UncheckedReason.NONDETERMINISTIC_EXAMPLES`。混在・長すぎる結果・理由が読めないときは
    :attr:`~UncheckedReason.UNCOMPARABLE_EXAMPLES` (どれか 1 つの理由を名指すと実態とずれる)。
    """
    kinds = set()
    for text in messages:
        if text.startswith((SKIP_FILES, SKIP_ENVIRONMENT)):
            kinds.add(UncheckedReason.STATEFUL_EXAMPLES)
        elif text.startswith(SKIP_NONDETERMINISTIC):
            kinds.add(UncheckedReason.NONDETERMINISTIC_EXAMPLES)
        else:
            kinds.add(UncheckedReason.UNCOMPARABLE_EXAMPLES)
    return kinds.pop() if len(kinds) == 1 else UncheckedReason.UNCOMPARABLE_EXAMPLES


def build_example_tests(cases: list[ExampleCase]) -> str:
    """入出力例の pytest ファイル本文 (例が無ければ空文字列)。

    例の呼出し中にファイル・フォルダへ触れた例は skip する (契約から外す)。値が前の例・前の実行の残したデータで
    決まり、固定値の契約にできない (2026-10-02 ライブ監査 K01)。評価フォルダの中への読み書きは止めて状態を
    残さず、外はサンドボックスに任せる (止めた書込みは「未検査」として記録される)。

    結果が遅延評価の反復子 (ジェネレータ・``map``・``filter`` 等) なら、傍受の内側で取り出して list
    (期待値が tuple なら tuple) にしてから比べる。外で回すとファイルの読込みが傍受の後に走って検出を逃れ、
    期待がリストでジェネレータを返す関数は 2 回の評価のジェネレータが等しくならず「値が食い違う」で skip された
    (検査されない)。具体化して比べるのは、食い違いがコードの欠陥ではなく契約の言い方 (「返す値の並び」を
    list のリテラルで書いた) だから。ただし戻り型を具体的な入れ物 (``-> list`` 等。骨組みの signature、無ければ
    本体の関数の注釈) と宣言して反復子を返す関数は具体化せず不合格にする (呼出側の ``len``・添字が落ちる欠陥)。
    件数・時間の上限 (:data:`LAZY_MAX_ITEMS` / :data:`LAZY_MAX_SECONDS`) を越えた反復子は固定値と比べられない
    例として skip する。
    """
    if not cases:
        return ""
    lines = [
        '"""骨組み (契約) の入出力例から自動で組んだテスト (staged v2)。"""',
        "",
        "import importlib",
        "import importlib.util",
        "",
        "import pytest",
        "",
        "",
        "def _ns(name):",
        "    # 評価のたびにモジュールを新しく読み込む (例どうし・2 回の評価で状態を共有しない)",
        "    spec = importlib.util.find_spec(name)",
        "    module = importlib.util.module_from_spec(spec)",
        "    spec.loader.exec_module(module)",
        "    # 本体の関数が読む大域 (= この辞書) で評価する。名前で取り込んだ関数の傍受も届く",
        "    return vars(module)",
        "",
        "",
        "def _same(got, exp):",
        "    if isinstance(exp, float) or isinstance(got, float):",
        "        return got == pytest.approx(exp)",
        "    return got == exp",
        "",
        "",
        "def _printed(out):",
        "    # 戻り値が None の呼出し (main(['c_to_f', '100']) 等) は表示した最後の行を値として見る",
        "    last = (out.strip().splitlines() or [''])[-1].strip()",
        "    try:",
        "        return ast.literal_eval(last)",
        "    except (SyntaxError, ValueError):",
        "        return last",
        "",
        "",
        "_SYSTEM_ROOTS = tuple({",
        "    os.path.normcase(os.path.abspath(p)).rstrip(os.sep) + os.sep",
        "    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)",
        "})",
        "",
        "",
        "class _FileAccess(Exception):",
        "    pass",
        "",
        "",
        "_SANDBOX_ROOTS = tuple(",
        "    os.path.normcase(os.path.abspath(p)).rstrip(os.sep) + os.sep",
        f"    for p in os.environ.get({ROOTS_ENV!r}, '').split(os.pathsep) if p",
        ")",
        "",
        "",
        "_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC",
        "_CODE_SUFFIXES = ('.py', '.pyc', '.pyd', '.so', '.dll')",
        "#: 傍受する関数 (持ち主, 名前, 種類)。種類は open / os.open (引数で読み書きが決まる) / read / write",
        "_TARGETS = [",
        "    (builtins, 'open', 'open'), (io, 'open', 'open'), (os, 'open', 'os.open'),",
        "    (os, 'listdir', 'read'), (os, 'scandir', 'read'), (os, 'stat', 'read'),",
        "    (os.path, 'exists', 'read'), (os.path, 'isfile', 'read'), (os.path, 'isdir', 'read'),",
        "    (os, 'remove', 'write'), (os, 'unlink', 'write'), (os, 'mkdir', 'write'), (os, 'makedirs', 'write'),",
        "    (os, 'rmdir', 'write'), (os, 'rename', 'write'), (os, 'replace', 'write'),",
        "    (shutil, 'rmtree', 'write'), (shutil, 'copy', 'write'), (shutil, 'copy2', 'write'),",
        "    (shutil, 'copyfile', 'write'), (shutil, 'copytree', 'write'), (shutil, 'move', 'write'),",
        "    (sqlite3, 'connect', 'write'),",
        "]",
        "",
        "",
        "def _writes(kind, args, kwargs):",
        "    if kind == 'open':",
        "        mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')",
        "        return isinstance(mode, str) and any(c in mode for c in 'wax+')",
        "    if kind == 'os.open':",
        "        flags = args[1] if len(args) > 1 else kwargs.get('flags', 0)",
        "        return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)",
        "    return kind == 'write'",
        "",
        "",
        "def _guarded(original, touched, kind):",
        "    # 例の呼出し中のファイル・フォルダへのアクセスを止める (標準ライブラリ・コードの読み込みは通す。",
        "    # 書込みは拡張子・置き場によらず止める)。評価フォルダの外はサンドボックスに任せる (止めた書込みを",
        "    # 「未検査」として記録させる)",
        "    def _call(*args, **kwargs):",
        "        writes = _writes(kind, args, kwargs)",
        "        targets = list(args[:2] if kind == 'write' else args[:1]) or [",
        "            kwargs.get('file', kwargs.get('path', kwargs.get('database', '.'))),",
        "        ]",
        "        for target in targets:",
        "            if not isinstance(target, (str, bytes, os.PathLike)):",
        "                continue",
        "            text = os.fsdecode(target)",
        "            if text in ('', ':memory:') or text.startswith('file::memory:'):",
        "                continue",
        "            norm = os.path.normcase(os.path.abspath(text))",
        "            if not writes and (norm.endswith(_CODE_SUFFIXES) or norm.startswith(_SYSTEM_ROOTS)):",
        "                continue",
        "            touched.append(text)",
        "            if not _SANDBOX_ROOTS or (norm + os.sep).startswith(_SANDBOX_ROOTS):",
        "                raise _FileAccess(text)",
        "        return original(*args, **kwargs)",
        "    return _call",
        "",
        "",
        "def _module_dicts(ns):",
        "    # 名前で取り込んだ関数 (``from os.path import exists``) の束縛先: 評価する本体と、同じフォルダの兄弟",
        "    yield ns",
        "    home = os.path.normcase(os.path.dirname(os.path.abspath(ns.get('__file__') or '.'))) + os.sep",
        "    for module in list(sys.modules.values()):",
        "        path = getattr(module, '__file__', None)",
        "        if path and module.__dict__ is not ns and os.path.normcase(os.path.abspath(path)).startswith(home):",
        "            yield module.__dict__",
        "",
        "",
        "class _TooLong(Exception):",
        "    pass",
        "",
        "",
        f"_LAZY_MAX_ITEMS = {LAZY_MAX_ITEMS}",
        f"_LAZY_MAX_SECONDS = {LAZY_MAX_SECONDS}",
        "",
        "",
        f"_CONTAINERS = {CONTAINER_TYPES!r}",
        "",
        "",
        "def _container(ann):",
        "    # 戻り型の注釈が具体的な入れ物 (list / tuple / set …) ならその名前、それ以外 (無し・Iterator 等) は空",
        "    if ann is None:",
        "        return ''",
        "    text = ann if isinstance(ann, str) else getattr(typing.get_origin(ann) or ann, '__name__', '')",
        "    head = str(text).strip().split('[', 1)[0].rsplit('.', 1)[-1].lower()",
        "    return head if head in _CONTAINERS else ''",
        "",
        "",
        "def _declared(call, ns, returns):",
        "    # 骨組みの signature の戻り型、無ければ本体の関数の戻り型の注釈",
        "    if returns:",
        "        return _container(returns)",
        "    node = ast.parse(call, mode='eval').body",
        "    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):",
        "        return ''",
        "    annotations = getattr(ns.get(node.func.id), '__annotations__', None)",
        "    return _container(annotations.get('return')) if isinstance(annotations, dict) else ''",
        "",
        "",
        "def _materialize(got, exp, declared):",
        "    # 遅延評価の結果 (ジェネレータ・map・filter 等) は傍受の内側で値にする。外で回すとファイルの読込みが",
        "    # 傍受の後に走って検出を逃れ、期待がリストのジェネレータは 2 回の評価が等しくならず「値が食い違う」になる。",
        "    # 等価が同一性のままの反復子だけを対象にする (自前の == を持つ型は従来どおりそのまま比べる)",
        "    if not isinstance(got, collections.abc.Iterator) or type(got).__eq__ is not object.__eq__:",
        "        return got",
        "    if declared:",
        "        # 具体的な入れ物を返すと宣言して反復子を返すのは欠陥 (呼出側の len・添字が落ちる) — 具体化しない",
        "        raise AssertionError(f'returns a lazy {type(got).__name__} but declares -> {declared}')",
        "    items = []",
        "    deadline = time.monotonic() + _LAZY_MAX_SECONDS",
        "    for item in got:",
        "        items.append(item)",
        "        if len(items) > _LAZY_MAX_ITEMS or time.monotonic() > deadline:",
        "            raise _TooLong(f'more than {_LAZY_MAX_ITEMS} items or {_LAZY_MAX_SECONDS} s')",
        "    return tuple(items) if isinstance(exp, tuple) else items",
        "",
        "",
        "def _eval_isolated(call, ns, exp, returns):",
        "    # 結果がファイル (前の例・前の実行が残したデータ) に依存する例は固定値の契約にできない",
        "    # (2026-10-02 ライブ監査 K01: 前の例の追加を前提にした集計の例が、正しい本体を不合格にした)",
        "    touched = []",
        "    too_long = ''",
        "    names = [(owner, name, kind) for owner, name, kind in _TARGETS if owner is not None]",
        "    originals = [getattr(owner, name) for owner, name, _kind in names]",
        "    guards = {}",
        "    for (_owner, _name, kind), original in zip(names, originals):",
        "        guards.setdefault(id(original), _guarded(original, touched, kind))",
        "    rebound = []",
        "    for (owner, name, _kind), original in zip(names, originals):",
        "        setattr(owner, name, guards[id(original)])",
        "    for space in _module_dicts(ns):",
        "        for key, value in list(space.items()):",
        "            if id(value) in guards and any(value is o for o in originals):",
        "                rebound.append((space, key, value))",
        "                space[key] = guards[id(value)]",
        "    try:",
        "        got = _materialize(eval(call, ns), exp, _declared(call, ns, returns))",
        "    except _TooLong as exc:",
        "        got, too_long = None, str(exc)",
        "    except Exception:",
        "        if not touched:",
        "            raise",
        "        got = None",
        "    finally:",
        "        for (owner, name, _kind), original in zip(names, originals):",
        "            setattr(owner, name, original)",
        "        for space, key, value in rebound:",
        "            space[key] = value",
        "    if touched:",
        f"        pytest.skip(f'{SKIP_FILES} ({{touched[0]}}); its value depends on state outside the call')",
        "    if too_long:",
        f"        pytest.skip(f'{SKIP_TOO_LONG} ({{too_long}})')",
        "    return got",
        "",
        "",
        "def _value(call, module, exp, returns=''):",
        "    buf = io.StringIO()",
        "    ns = _ns(module)",
        "    try:",
        "        with contextlib.redirect_stdout(buf):",
        "            got = _eval_isolated(call, ns, exp, returns)",
        "    except OSError as exc:  # 例が環境 (ファイル等) に依存していた — 契約の判定から外す",
        f"        pytest.skip(f'{SKIP_ENVIRONMENT}: {{exc}}')",
        "    if got is None and exp is not None:",
        "        got = _printed(buf.getvalue())",
        "    return got",
    ]
    lines[2:2] = [
        "import ast", "import builtins", "import collections.abc", "import contextlib", "import io", "import os",
        "import shutil", "import sys", "import time", "import typing",
        "",
        "try:",
        "    import sqlite3",
        "except ImportError:  # sqlite3 の無い Python",
        "    sqlite3 = None",
    ]
    for i, case in enumerate(cases, start=1):
        returns = f", {case.returns!r}" if case.returns else ""
        lines += [
            "",
            "",
            f"def test_example_{i}():",
            f"    exp = {case.expected}",
            f"    got = _value({case.call!r}, {case.module!r}, exp{returns})",
            "    # 同じ式で値が変わる (乱数・時刻) 例は固定値の契約にできない — 判定から外す",
            f"    if not _same(_value({case.call!r}, {case.module!r}, exp{returns}), got):",
            f"        pytest.skip('{SKIP_NONDETERMINISTIC} (the same call returned different values)')",
            "    assert _same(got, exp), (got, exp)",
        ]
    return "\n".join(lines) + "\n"


def _test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            out.append(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            out += [
                m for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test")
            ]
    return out


def _is_absolute_path(text: str) -> bool:
    """ホストの絶対パスか (ドライブ + 区切り / UNC / 先頭 ``/`` で第 1 成分がホストに実在)。

    ``/api/todos`` のような URL のパスは残す (第 1 成分がホストに無い)。
    """
    value = text.strip()
    if "\n" in value or not is_absolute_path_text(value):
        return False
    if value.startswith("/"):
        first = value.lstrip("/").split("/", 1)[0]
        try:
            return bool(first) and Path("/" + first).exists()
        except OSError:
            return False
    return True


def _absolute_path_in(node: ast.AST) -> str:
    """ノードの下にある絶対パスの文字列定数 (無ければ空)。"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and _is_absolute_path(sub.value):
            return sub.value
    return ""


def _unrewritable_constants(tree: ast.AST) -> set[int]:
    """書き換えると意味か構文が壊れる文字列定数 (f-string の部品・match のパターン・docstring)。"""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.JoinedStr, ast.MatchValue, ast.MatchSingleton)):
            out.update(id(sub) for sub in ast.walk(node))
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and isinstance(body, list) and body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)
        ):
            out.add(id(body[0].value))
    return out


#: Python の行区切り (``str.splitlines`` は ``\x0c`` / ``\x1c``-``\x1e`` / ``\x85`` / `` `` でも切り、
#: ast の行番号とずれる — 独立レビュー: 書き換えが 1 行上へずれて代入がコメントに飲まれた)。
_LINE_END_RE = re.compile(r"\r\n|\r|\n")


def _source_lines(source: str) -> list[str]:
    """ソースを Python の行 (改行を含む) に分ける。ast の ``lineno`` と 1 対 1。"""
    lines: list[str] = []
    start = 0
    for m in _LINE_END_RE.finditer(source):
        lines.append(source[start:m.end()])
        start = m.end()
    if start < len(source):
        lines.append(source[start:])
    return lines


def _offset_of(source: str) -> Callable[[int, int], int]:
    """ast の (行, 列) → ``source`` の文字の位置。"""
    lines = _source_lines(source)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def _offset(lineno: int, col: int) -> int:
        # ast の列は UTF-8 のバイト数
        line = lines[lineno - 1] if lineno - 1 < len(lines) else ""
        return starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8", errors="ignore"))

    return _offset


def _apply_edits(source: str, edits: list[tuple[int, int, str]]) -> str:
    """重ならない (開始, 終了, 新しい文字列) を後ろから当てる。"""
    out = source
    for start, end, new in sorted(edits, reverse=True):
        out = out[:start] + new + out[end:]
    return out


def _replace_constants(
    source: str, tree: ast.AST, replacement_of: Callable[[str], str | None],
) -> tuple[str, list[str]]:
    """絶対パスの文字列定数を ``replacement_of(値)`` (式の文字列) に差し替える (``None`` は残す)。"""
    skip = _unrewritable_constants(tree)
    _offset = _offset_of(source)
    edits: list[tuple[int, int, str, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
            and node.end_lineno is not None and node.end_col_offset is not None
            and _is_absolute_path(node.value)
        ):
            new = replacement_of(node.value)
            if new is not None:
                edits.append((
                    _offset(node.lineno, node.col_offset), _offset(node.end_lineno, node.end_col_offset),
                    new, node.value,
                ))
    out = _apply_edits(source, [(start, end, new) for start, end, new, _old in edits])
    return out, [old for _s, _e, _n, old in sorted(edits)]


def _binds_os(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Import) and any(a.name == "os" and a.asname in (None, "os") for a in node.names)
        for node in tree.body
    )


def _import_insert_line(tree: ast.Module, lines: list[str]) -> int:
    """``import os`` / ``import pytest`` を足す行 (0 始まり): docstring・``from __future__``・先頭のコメント行の後。"""
    after = 0
    for i, node in enumerate(tree.body):
        is_doc = i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if not (is_doc or is_future):
            break
        after = node.end_lineno or node.lineno
    while after < len(lines) and lines[after].lstrip().startswith("#") and after < 2:
        after += 1
    return after


def here_expression(relative: str) -> str:
    """モジュールの置き場からの相対パス (``/`` 区切り) を ``__file__`` 基準の式にする。"""
    parts = [p for p in relative.replace("\\", "/").split("/") if p not in ("", ".")]
    here = "os.path.dirname(os.path.abspath(__file__))"
    if not parts:
        return here
    return f"os.path.join({here}, {', '.join(repr(p) for p in parts)})"


def rebase_absolute_paths_in_source(
    source: str, relative_of: Callable[[str], str | None],
) -> tuple[str, list[str]]:
    """生成した ``.py`` の文字列定数の絶対パスを ``__file__`` 基準の式へ書き換える (f_10 §11.1-1)。

    ``relative_of(絶対パス)`` はモジュールの置き場からの相対 (``/`` 区切り、``..`` 可) を返す。
    ``None`` のパス (配信先の外 — 依頼が明示した入力等) は残す。``import os`` が無ければ足す。
    戻り値は (本文, 書き換えた絶対パスの列)。構文エラー・書き換えで構文が壊れるなら元のまま。

    作り直しでは直さない: 依頼文が文脈にある限り、作り直しても同じパスが出る
    (2026-09-27 ライブ監査 K01: ``TODO_FILE = "E:\\\\tmp\\\\c0927_01_todo\\\\todos.json"`` を契約テストが
    実行し、利用者のフォルダに todos.json を作って壊した)。
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return source, []

    def _expr(value: str) -> str | None:
        rel = relative_of(value)
        return None if rel is None else here_expression(rel)

    out, rebased = _replace_constants(source, tree, _expr)
    if not rebased:
        return source, []
    if not _binds_os(tree):
        lines = _source_lines(out)
        at = _import_insert_line(tree, _source_lines(source))
        out = "".join(lines[:at]) + "import os\n" + "".join(lines[at:])
    try:
        ast.parse(out)
    except (SyntaxError, ValueError):
        return source, []
    return out, rebased


#: 文字列リテラルの本文 (引用符の内側) がドライブ付きの Windows パスを 1 個の ``\`` で区切っている
_UNESCAPED_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:\\(?!\\)")
_STRING_TOKEN_RE = re.compile(r"(?is)^([a-z]*)('''|\"\"\"|'|\")(.*)\2$")


def repair_windows_path_literals(expression: str, accept: Callable[[str], bool]) -> str:
    r"""Python 式の中の、``\`` を重ねずに書いた Windows パスの文字列リテラルを正しいリテラルにする。

    骨組みの例はモデルが手で書く Python 式で、``'E:\\tmp\\x\\sales.csv'`` ではなく ``'E:\tmp\x\sales.csv'``
    と書く。Python としては ``\t`` がタブになり、絶対パスの相対化 (:func:`rebase_absolute_paths_in_expression`)
    も外れて、SPEC.md にタブに化ける式のまま載った (2026-10-03 ライブ再実行 K03)。接頭辞の無い (``r`` / ``b`` / ``f`` で
    ない) リテラルの本文がドライブ付きのパスを 1 個の ``\`` で区切り、``\\`` を 1 つも含まず、さらに raw として読んだ
    パスを ``accept`` が受け入れる (相対化の対象 = 依頼の出力フォルダの配下) ときだけ、本文をそのままの文字 (raw) として
    読み直す。``'C:\n'`` のように意図したエスケープかもしれない文字列や、出力フォルダの外のパスは直さない
    (独立レビュー 低 c)。それ以外 (正しく重ねた・相対パス・接頭辞付き) もそのまま。
    """
    import io
    import tokenize

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(expression).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return expression
    starts = [0]
    for line in expression.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    edits: list[tuple[int, int, str]] = []
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        m = _STRING_TOKEN_RE.match(tok.string)
        body = m.group(3) if m else ""
        if (
            m and not m.group(1) and _UNESCAPED_DRIVE_PATH_RE.match(body) and "\\\\" not in body
            and accept(body)
        ):
            begin = starts[tok.start[0] - 1] + tok.start[1]
            edits.append((begin, begin + len(tok.string), repr(body)))
    if not edits:
        return expression
    repaired = expression
    for begin, end, text in reversed(edits):
        repaired = repaired[:begin] + text + repaired[end:]
    return repaired if _is_expr(repaired) else expression


def rebase_absolute_paths_in_expression(
    expression: str, relative_of: Callable[[str], str | None],
) -> str:
    """Python 式 (骨組みの ``examples.call``) の絶対パスの文字列定数を相対パスの文字列にする。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return expression

    def _literal(value: str) -> str | None:
        rel = relative_of(value)
        return None if rel is None else repr(rel or ".")

    out, rebased = _replace_constants(expression, tree, _literal)
    return out if rebased and _is_expr(out) else expression


def _brittle_reason(func: ast.AST, allowed_text: str) -> str:
    # テストはサンドボックス (作業フォルダ) の外に触れない。依頼文にある文字列でも絶対パスは落とす
    # (2026-09-26 ライブ監査 K08: 依頼のフォルダへ test_files/ 等を作った)
    path = _absolute_path_in(func)
    if path:
        return f"uses an absolute path ({path})"
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute) and target.attr in ("stdout", "stderr")
                    and isinstance(target.value, ast.Name) and target.value.id == "sys"
                ):
                    return "reassigns sys.stdout/sys.stderr"
        if isinstance(node, ast.Attribute) and node.attr == "__annotations__":
            return "compares __annotations__"
        if isinstance(node, ast.Compare) and any(isinstance(op, ast.Eq) for op in node.ops):
            for side in [node.left, *node.comparators]:
                if (
                    isinstance(side, ast.Constant) and isinstance(side.value, str)
                    and len(side.value.strip()) >= 8 and side.value.strip() not in allowed_text
                    and not _is_structured_data(side.value)
                ):
                    return "asserts exact text that the contract does not specify"
    return ""


#: 表の列の区切り (CSV / TSV / セミコロン / パイプ)
_FIELD_DELIMITERS = (",", "\t", ";", "|")


def _is_structured_data(text: str) -> bool:
    """利用者向けの文言ではなくデータの本文か (JSON の配列・オブジェクト / 2 行以上で列の数が揃った表)。

    lint が落とすのは骨組みに無い文言との完全一致 (表示文・メッセージ)。CSV の本文との ``==`` は値の比較で、
    本体の欠陥 (キーの揃わない行で ValueError) を検出するテストまで落としていた (2026-10-02 ライブ監査 K05)。
    """
    value = text.strip()
    try:
        if isinstance(json.loads(value), (dict, list)):
            return True
    except ValueError:
        pass
    rows = [line for line in value.splitlines() if line.strip()]
    if len(rows) < 2:
        return False
    return any(len({row.count(d) for row in rows}) == 1 and rows[0].count(d) >= 1 for d in _FIELD_DELIMITERS)


@dataclass(frozen=True)
class DataLocation:
    """本文が ``__file__`` から組むデータの置き場 1 つ (f_10 §11.1-4)。"""

    module: str  # import 名
    constant: str  # 差し替えられるモジュール直下の定数名。空 = 関数の中・既定引数で組む (差し替えられない)
    replacement: str  # 参考テストの差し替えの式 (``str(tmp_path / "todos.json")``)。constant が空なら空
    copied_by: tuple[str, ...] = ()  # この定数を名前 import する兄弟 (``from todo_manager import DATA_FILE``)
    # 既定引数の値に焼き込んでいるモジュール (``def load(path=DATA_FILE)`` / ``= config.DATA_FILE``。差し替えが届かない、R17)
    baked: tuple[str, ...] = ()


def _terminal_name(func: ast.AST) -> str:
    """呼出しの関数の末尾の名前 (``os.path.join`` → ``join``)。"""
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


#: パスを返す ``os.path`` の関数 (``basename`` / ``splitext`` のような名前・組を返すものは含めない)
_OS_PATH_FUNCS = frozenset({"join", "dirname", "abspath", "realpath", "normpath", "expanduser", "expandvars"})
#: パスを返す pathlib のメソッド
_PATH_METHODS = frozenset({"resolve", "absolute", "with_name", "with_suffix", "with_stem", "joinpath", "expanduser"})
_PATH_TYPES = frozenset({"Path", "PurePath", "PurePosixPath", "PureWindowsPath", "PosixPath", "WindowsPath"})

_STR, _PATH = "str", "path"


def _is_os_path(func: ast.AST) -> bool:
    """``os.path.<関数>`` か。"""
    return (
        isinstance(func, ast.Attribute) and func.attr in _OS_PATH_FUNCS
        and isinstance(func.value, ast.Attribute) and func.value.attr == "path"
    )


def _is_env_lookup(node: ast.AST) -> bool:
    """``os.getenv(k, 既定)`` / ``os.environ.get(k, 既定)`` か。"""
    return isinstance(node, ast.Call) and (
        _terminal_name(node.func) == "getenv"
        or (isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and _terminal_name(node.func.value) == "environ")
    )


def _path_kind(node: ast.AST, known: dict[str, str]) -> str | None:
    """パスを返す式だけでできていれば ``"str"`` / ``"path"`` (pathlib)、そうでなければ ``None``。

    ``known`` は既知のパス定数 → 種類。``TodoStore(DATA_FILE)`` のような他の呼出しの結果はパスにしない (P2-1)。
    """
    if isinstance(node, ast.Name):
        return _STR if node.id == "__file__" else known.get(node.id)
    if isinstance(node, ast.Constant):
        return _STR if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts = [v.value for v in node.values if isinstance(v, ast.FormattedValue)]
        return _STR if all(_path_kind(p, known) for p in parts) else None
    if isinstance(node, ast.BinOp):
        left, right = _path_kind(node.left, known), _path_kind(node.right, known)
        if isinstance(node.op, ast.Div) and left == _PATH and right:
            return _PATH
        return _STR if isinstance(node.op, ast.Add) and left == right == _STR else None
    if isinstance(node, ast.Attribute):
        return _PATH if node.attr == "parent" and _path_kind(node.value, known) == _PATH else None
    if isinstance(node, ast.Subscript):
        value = node.value
        return _PATH if (
            isinstance(value, ast.Attribute) and value.attr == "parents" and _path_kind(value.value, known) == _PATH
        ) else None
    if isinstance(node, ast.IfExp):
        body, orelse = _path_kind(node.body, known), _path_kind(node.orelse, known)
        return body if body and body == orelse else None
    if not isinstance(node, ast.Call):
        return None
    args_ok = all(_path_kind(a, known) for a in node.args) and not node.keywords
    if _is_env_lookup(node):
        return _path_kind(node.args[1], known) if len(node.args) == 2 else None
    if _terminal_name(node.func) in _PATH_TYPES and isinstance(node.func, ast.Name | ast.Attribute):
        return _PATH if args_ok else None
    if _is_os_path(node.func):
        return _STR if args_ok and node.args else None
    if isinstance(node.func, ast.Name) and node.func.id == "str" or _terminal_name(node.func) == "fspath":
        return _STR if len(node.args) == 1 and args_ok else None
    if isinstance(node.func, ast.Attribute) and node.func.attr in _PATH_METHODS:
        return _PATH if _path_kind(node.func.value, known) == _PATH and args_ok else None
    return None


def _file_name(node: ast.AST) -> str:
    """パスの式がファイル名で終わるなら差し替えに使う名前、フォルダなら空。

    末尾は値の最後の文字列定数。拡張子が無ければフォルダ扱い (P2-2)。条件式・環境変数を含む値では名前を
    推定せず ``data`` + 拡張子にする (P3-5)。
    """
    if isinstance(node, ast.IfExp):
        names = [_file_name(node.body), _file_name(node.orelse)]
        suffix = next((PurePosixPath(n).suffix for n in names if n), "")
        return f"data{suffix}" if suffix else ""
    if _is_env_lookup(node):
        suffix = PurePosixPath(_file_name(node.args[1])).suffix if len(node.args) == 2 else ""
        return f"data{suffix}" if suffix else ""
    strings = sorted(
        (n for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    if not strings:
        return ""
    name = PurePosixPath(strings[-1].value.replace("\\", "/")).name
    if not PurePosixPath(name).suffix or '"' in name:
        return ""
    if any(isinstance(n, ast.IfExp) or _is_env_lookup(n) for n in ast.walk(node)):
        return f"data{PurePosixPath(name).suffix}"
    return name


def _assignments(tree: ast.Module) -> Iterator[tuple[str, ast.expr]]:
    """モジュール直下の単純な代入 (``X = …`` / ``X: T = …``)。"""
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            yield node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            yield node.target.id, node.value


def _builds_file_from_file(node: ast.AST, known: dict[str, str]) -> bool:
    """``node`` の下に ``__file__`` (か由来のパス定数) からファイル名で終わるパスを組む式があるか (P2-2 / P3-4)。"""
    anchors = {"__file__", *known}
    return any(
        isinstance(sub, ast.expr) and not isinstance(sub, ast.Name | ast.Constant)
        and _names_in(sub) & anchors and _path_kind(sub, known) and _file_name(sub)
        for sub in ast.walk(node)
    )


def _path_constants(tree: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    """モジュール直下の ``__file__`` 由来のパス定数 (フォルダを含む) → 種類と、そのうちファイル名で終わるもの → 差し替えの式。"""
    known: dict[str, str] = {}
    files: dict[str, str] = {}
    for name, value in _assignments(tree):
        kind = _path_kind(value, known)
        if kind is None or not _names_in(value) & {"__file__", *known}:
            # 後からパスでない値に再代入された名前は置き場から外す (独立レビュー)
            known.pop(name, None)
            files.pop(name, None)
            continue
        known[name] = kind
        file_name = _file_name(value)
        if file_name:
            target = f'tmp_path / "{file_name}"'
            files[name] = target if kind == _PATH else f"str({target})"
    return known, files


def _default_nodes(tree: ast.AST) -> set[int]:
    """関数の既定引数の値の中のノード (``id``)。"""
    return {
        id(n) for f in ast.walk(tree) if isinstance(f, _FuncDef)
        for d in [*f.args.defaults, *f.args.kw_defaults] if d is not None for n in ast.walk(d)
    }


def _uses(tree: ast.AST, match: Callable[[ast.AST], bool]) -> tuple[bool, bool]:
    """``match`` するノードが (既定引数の外で読まれるか, 既定引数に使われるか)。"""
    in_defaults = _default_nodes(tree)
    read = default = False
    for n in ast.walk(tree):
        if isinstance(getattr(n, "ctx", None), ast.Load) and match(n):
            if id(n) in in_defaults:
                default = True
            else:
                read = True
    return read, default


def _module_imports(tree: ast.AST, module: str) -> set[str]:
    """モジュール直下で ``module`` を束縛する名前 (``import config`` → ``config`` / ``import config as c`` → ``c``)。"""
    body = getattr(tree, "body", [])
    return {
        a.asname or a.name for n in body if isinstance(n, ast.Import)
        for a in n.names if a.name == module
    }


def _names_imported(tree: ast.AST, module: str, name: str) -> set[str]:
    """``from <module> import <name> [as y]`` が束縛する名前 (相対 import は末尾の成分で見る)。"""
    return {
        a.asname or a.name for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and (n.module or "").rsplit(".", 1)[-1] == module
        for a in n.names if a.name == name
    }


def _module_data_locations(module: str, tree: ast.Module, others: dict[str, ast.Module]) -> list[DataLocation]:
    """``module`` の置き場。定数は定義側か兄弟 (``others``) のどこかで読まれるか既定引数に使われるときだけ拾う。"""
    known, files = _path_constants(tree)
    out: list[DataLocation] = []
    for name, target in files.items():
        read, default = _uses(tree, lambda n, name=name: isinstance(n, ast.Name) and n.id == name)
        baked = [module] if default else []
        for other, other_tree in others.items():
            aliases = _module_imports(other_tree, module)
            copies = _names_imported(other_tree, module, name)
            r, d = _uses(other_tree, lambda n, aliases=aliases, copies=copies, name=name: (
                isinstance(n, ast.Attribute) and n.attr == name
                and isinstance(n.value, ast.Name) and n.value.id in aliases
            ) or (isinstance(n, ast.Name) and n.id in copies))
            read = read or r
            if d:
                baked.append(other)
        if read or baked:
            out.append(DataLocation(module, name, target, baked=tuple(baked)))
    functions = [n for n in ast.walk(tree) if isinstance(n, _FuncDef)]
    defaults = [d for f in functions for d in [*f.args.defaults, *f.args.kw_defaults] if d is not None]
    if not out and any(
        _builds_file_from_file(stmt, known) for stmt in [*(s for f in functions for s in f.body), *defaults]
    ):
        # 関数の中・既定引数で __file__ から組む — 差し替えられる定数が無い (2026-09-28 実機確認 K01)
        out.append(DataLocation(module, "", ""))
    return out


def _name_importers(trees: dict[str, ast.Module], module: str, name: str) -> tuple[str, ...]:
    """``from <module> import <name>`` で定数を名前 import するモジュール (P2-4)。"""
    return tuple(
        other for other, tree in trees.items()
        if other != module and any(
            isinstance(n, ast.ImportFrom) and not n.level and n.module == module
            and any(a.name == name for a in n.names)
            for n in ast.walk(tree)
        )
    )


def file_based_data_locations(code_map: dict[str, str]) -> list[DataLocation]:
    """本文 (``.py``) が ``__file__`` から組むデータファイルの置き場 (f_10 §11.1-4)。

    モジュール直下の代入のうち、値がパスを返す式だけでできていて (他の呼出しの結果は除く)、``__file__`` を
    (直接、または同じく ``__file__`` から組んだパス定数を経て) 含み、ファイル名で終わるものを差し替えられる
    置き場として返す (フォルダの定数は返さない)。定数は定義側か兄弟 (``config.DATA_FILE`` / 名前 import) のどこかで
    読まれるか既定引数に使われるときだけ返し、既定引数の値に焼き込んだモジュールを ``baked`` に持つ (差し替えが
    届かない、R17)。差し替えられる定数が無いのに関数の中・既定引数で
    ``__file__`` からファイルのパスを組むモジュールは ``constant=""`` を 1 件返す。相対パス (CWD 基準)
    だけの本文・構文エラーの本文は何も返さない。参考テストのプロンプト (差し替えの指示) と lint の両方が読む。
    """
    trees: dict[str, ast.Module] = {}
    for path, source in code_map.items():
        if path.endswith(".py"):
            try:
                trees[PurePosixPath(path).stem] = ast.parse(source)
            except (SyntaxError, ValueError):
                continue
    out: list[DataLocation] = []
    for module, tree in trees.items():
        others = {m: t for m, t in trees.items() if m != module}
        for loc in _module_data_locations(module, tree, others):
            copied = _name_importers(trees, module, loc.constant) if loc.constant else ()
            out.append(replace(loc, copied_by=copied))
    return out


def _binding_names(node: ast.AST) -> set[str]:
    """``node`` の下で束縛される名前 (代入・引数・import・内側の def / class・except・match・global)。"""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store | ast.Del):
            out.add(sub.id)
        elif isinstance(sub, ast.arg):
            out.add(sub.arg)
        elif isinstance(sub, ast.alias):
            out.add(sub.asname or sub.name.split(".")[0])
        elif isinstance(sub, _FuncDef | ast.ClassDef) and sub is not node:
            out.add(sub.name)
        elif isinstance(sub, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and sub.name:
            out.add(sub.name)
        elif isinstance(sub, ast.Global | ast.Nonlocal):
            out.update(sub.names)
    return out


def _functions_in_scope(node: ast.AST, outer: frozenset[str] = frozenset()) -> Iterator[tuple[_FuncDef, frozenset[str]]]:
    """関数と、その外側のスコープ (クラス・関数) が束縛する名前。"""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _FuncDef | ast.ClassDef):
            if isinstance(child, _FuncDef):
                yield child, outer
            yield from _functions_in_scope(child, outer | _binding_names(child))
        else:
            yield from _functions_in_scope(child, outer)


def _defaults_of(func: _FuncDef) -> list[tuple[ast.arg, ast.expr]]:
    """既定値を持つ引数と既定値 (位置引数・キーワード専用の順)。"""
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
    return pairs + [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]


def _is_docstring(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _body_insert_point(func: _FuncDef, lines: list[str]) -> tuple[int, str] | None:
    """本体の先頭 (docstring の後) に文を足す行 (0 始まり) と字下げ。本体が ``def`` の行に続いていれば ``None``。"""
    first = func.body[0]
    if _is_docstring(first):
        after = first.end_lineno or first.lineno
        if len(func.body) > 1 and func.body[1].lineno == after:
            return None
    else:
        after = first.lineno - 1
    indent = lines[first.lineno - 1].encode("utf-8")[:first.col_offset].decode("utf-8", errors="ignore")
    return None if indent.strip() else (after, indent)


def path_constants_by_module(code_map: dict[str, str]) -> dict[str, frozenset[str]]:
    """本文 (``.py``) ごとの ``__file__`` 由来のパス定数の名前 (フォルダを含む)。:func:`unbake_default_data_paths` の兄弟の情報。"""
    out: dict[str, frozenset[str]] = {}
    for path, source in code_map.items():
        if path.endswith(".py"):
            try:
                known, _files = _path_constants(ast.parse(source))
            except (SyntaxError, ValueError):
                continue
            if known:
                out[PurePosixPath(path).stem] = frozenset(known)
    return out


def unbake_default_data_paths(
    source: str, sibling_constants: dict[str, frozenset[str]] | None = None,
) -> tuple[str, list[str]]:
    """``__file__`` 基準の置き場の定数を既定引数に焼き込んだ関数を、呼び出し時に読む形へ書き換える (f_10 §11.1-1、R17)。

    ``def load(path: str = DATA_FILE)`` → ``def load(path: str = None)`` + 本体の先頭 (docstring の後) に
    ``if path is None: path = DATA_FILE``。既定値が置き場の定数 — 自分のモジュール直下の定数 (:func:`file_based_data_locations`
    と同じ (a)(b)(d) を満たすもの。フォルダも含む) の名前、兄弟 (``sibling_constants``: モジュール名 → 定数名。
    :func:`path_constants_by_module`) から名前 import した定数、モジュール直下で import した兄弟の属性
    (``config.DATA_FILE``) — そのものの引数だけを対象にし、式の既定値・lambda は触らない。注釈は変えない
    (``X | None`` は Python 3.9 以下で def の評価時に TypeError になりうる)。既定値の名前 (属性なら先頭のモジュール名) を
    外側のスコープ (クラス・関数) か関数自身が束縛する関数、本体が ``def`` の行に続く関数も触らない。
    モジュールの他の部分は変えない。戻り値は (本文, 書き換えた ``関数(引数)`` の列)。構文エラー・書き換えで構文が
    壊れるなら元のまま。

    既定引数は定義時の値に束縛されるので、参考テストの ``monkeypatch.setattr(todo_manager, "DATA_FILE", …)`` が
    届かず、テストが配信フォルダの実物の ``todos.json`` を書き換えた (2026-09-28 実機確認 K01)。
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return source, []
    known, _files = _path_constants(tree)
    names = set(known)
    modules: dict[str, frozenset[str]] = {}  # モジュール直下で import した兄弟の名前 → その置き場の定数
    assigned = {name for name, _value in _assignments(tree)}
    for module, constants in (sibling_constants or {}).items():
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").rsplit(".", 1)[-1] == module:
                names.update(
                    a.asname or a.name for a in node.names
                    if a.name in constants and (a.asname or a.name) not in assigned
                )
        for alias in _module_imports(tree, module) - assigned:
            modules[alias] = constants
    if not names and not modules:
        return source, []

    def _head(default: ast.expr) -> str:
        """置き場の定数を指す既定値なら、束縛を確かめる名前 (``DATA_FILE`` / ``config``)。違えば空。"""
        if isinstance(default, ast.Name) and default.id in names:
            return default.id
        if (
            isinstance(default, ast.Attribute) and isinstance(default.value, ast.Name)
            and default.attr in modules.get(default.value.id, ())
        ):
            return default.value.id
        return ""

    lines = _source_lines(source)
    offset = _offset_of(source)
    newline = "\r\n" if "\r\n" in source else "\n"
    edits: list[tuple[int, int, str]] = []
    rewritten: list[str] = []
    for func, outer in _functions_in_scope(tree):
        own = _binding_names(func)
        targets = []
        for a, d in _defaults_of(func):
            head = _head(d)
            if head and head not in outer and head not in own:
                span = (offset(d.lineno, d.col_offset), offset(d.end_lineno or d.lineno, d.end_col_offset or 0))
                targets.append((a, span, source[span[0]:span[1]]))
        at = _body_insert_point(func, lines) if targets else None
        if at is None:
            continue
        after, indent = at
        unit = "\t" if indent.endswith("\t") else "    "
        block = "".join(
            f"{indent}if {a.arg} is None:{newline}{indent}{unit}{a.arg} = {expr}{newline}" for a, _span, expr in targets
        )
        if after >= len(lines) and not source.endswith(("\n", "\r")):
            block = newline + block
        insert_at = offset(after + 1, 0) if after < len(lines) else len(source)
        edits.append((insert_at, insert_at, block))
        for a, (start, end), _expr in targets:
            edits.append((start, end, "None"))
            rewritten.append(f"{func.name}({a.arg})")
    if not edits:
        return source, []
    out = _apply_edits(source, edits)
    try:
        ast.parse(out)
    except (SyntaxError, ValueError):
        return source, []
    return out, rewritten


def _chdir_in(node: ast.AST) -> bool:
    """``monkeypatch.chdir`` / ``os.chdir`` の呼出しがあるか。"""
    return any(isinstance(n, ast.Call) and _terminal_name(n.func) == "chdir" for n in ast.walk(node))


def _replaced_names(tree: ast.AST) -> set[str]:
    """テストが差し替える属性名 (``setattr(m, "X", …)`` / ``setattr("m.X", …)`` / ``patch("m.X")`` / ``m.X = …``)。"""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _terminal_name(node.func) in ("setattr", "patch", "object"):
            out.update(
                a.value.rsplit(".", 1)[-1] for a in node.args[:2]
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            )
        elif isinstance(node, ast.Assign):
            out.update(t.attr for t in node.targets if isinstance(t, ast.Attribute))
    return out


def _is_autouse_fixture(func: ast.AST) -> bool:
    return any(
        isinstance(d, ast.Call) and any(
            k.arg == "autouse" and isinstance(k.value, ast.Constant) and k.value.value is True for k in d.keywords
        )
        for d in getattr(func, "decorator_list", [])
    )


def _usefixtures(node: ast.AST) -> set[str]:
    """``@pytest.mark.usefixtures("a", "b")`` が名指す fixture。"""
    return {
        a.value for d in getattr(node, "decorator_list", [])
        if isinstance(d, ast.Call) and _terminal_name(d.func) == "usefixtures"
        for a in d.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
    }


def _params(func: ast.AST) -> set[str]:
    args = func.args  # type: ignore[attr-defined]
    return {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}


#: 各テストの前に走るフック (クラスの ``setup_method`` 等、モジュールの ``setup_function`` 等)
_SETUP_HOOKS = frozenset({"setup_method", "setup", "setUp", "setup_class", "setup_function", "setup_module"})

_FuncDef = ast.FunctionDef | ast.AsyncFunctionDef


def _tests_that(
    tree: ast.Module, tests: list[_FuncDef], uses: Callable[[ast.AST], bool],
) -> set[int]:
    """``uses`` を満たす処理を通るテスト関数の ``id`` (本体・fixture の連鎖・usefixtures・autouse・setup、P3-2)。"""
    scopes: list[tuple[ast.ClassDef | None, list[_FuncDef]]] = [(None, [
        n for n in tree.body if isinstance(n, _FuncDef) and n not in tests
    ])]
    scopes += [
        (c, [m for m in c.body if isinstance(m, _FuncDef) and m not in tests])
        for c in tree.body if isinstance(c, ast.ClassDef)
    ]
    helpers = {h.name: h for _cls, hs in scopes for h in hs}
    hit = {name for name, h in helpers.items() if uses(h)}
    changed = True
    while changed:  # fixture が受け取る fixture を辿る
        changed = False
        for name, h in helpers.items():
            if name not in hit and _params(h) & hit:
                hit.add(name)
                changed = True
    scope_hit = {
        id(cls): any(h.name in hit and (_is_autouse_fixture(h) or h.name in _SETUP_HOOKS) for h in hs)
        for cls, hs in scopes
    }
    out: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            members = [m for m in node.body if m in tests]
            inherited = scope_hit[id(node)] or bool(_usefixtures(node) & hit)
        else:
            members = [node] if node in tests else []
            inherited = False
        for t in members:
            if (
                scope_hit[id(None)] or inherited or uses(t)
                or (_params(t) | _usefixtures(t)) & hit
            ):
                out.add(id(t))
    return out


def _shared_data_reason(locations: list[DataLocation]) -> str:
    constants = [f"{loc.module}.{loc.constant}" for loc in locations if loc.constant]
    inline = [f"{loc.module} builds them inside functions" for loc in locations if not loc.constant]
    return (
        "isolates data with chdir, but the code locates its data files from __file__ "
        f"({', '.join(constants + inline)}); the tests would share the real data file"
    )


def _copied_reason(loc: DataLocation) -> str:
    who = ", ".join(loc.copied_by)
    return (
        f"replaces {loc.module}.{loc.constant}, but {who} imports it by name "
        f"(from {loc.module} import {loc.constant}), so the replacement does not reach {who}"
    )


def _baked_reason(loc: DataLocation) -> str:
    return (
        f"replaces {loc.module}.{loc.constant}, but {', '.join(loc.baked)} uses it as a default argument value, "
        "so the replacement does not reach those functions"
    )


def _data_file_name(loc: DataLocation) -> str:
    """差し替えの式 (``str(tmp_path / "todos.json")``) のファイル名。"""
    parts = loc.replacement.split('"')
    return parts[1] if len(parts) >= 3 else ""


def _sibling_reason(done: list[DataLocation], missing: list[DataLocation], file_name: str) -> str:
    names = ", ".join(f"{loc.module}.{loc.constant}" for loc in missing)
    return (
        f"replaces {', '.join(f'{loc.module}.{loc.constant}' for loc in done)} but not {names}, which locates "
        f"the same data file ({file_name}), so {', '.join(loc.module for loc in missing)} would still use the real file"
    )


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    """``import todo_manager as tm`` の別名 → モジュール名 (末尾の成分)。"""
    return {
        a.asname: a.name.rsplit(".", 1)[-1]
        for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names if a.asname
    }


def _replaced_targets(node: ast.AST, aliases: dict[str, str]) -> set[tuple[str, str]]:
    """テストが差し替える (モジュール, 属性名)。モジュールが式から分からなければ空文字列 (どのモジュールにも当てる)。"""

    def _module(expr: ast.AST) -> str:
        if isinstance(expr, ast.Name):
            return aliases.get(expr.id, expr.id)
        return expr.attr if isinstance(expr, ast.Attribute) else ""

    out: set[tuple[str, str]] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and _terminal_name(sub.func) in ("setattr", "patch", "object") and sub.args:
            first = sub.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and "." in first.value:
                module, _, name = first.value.rpartition(".")
                out.add((module.rsplit(".", 1)[-1], name))
            elif len(sub.args) >= 2 and isinstance(sub.args[1], ast.Constant) and isinstance(sub.args[1].value, str):
                out.add((_module(first), sub.args[1].value))
        elif isinstance(sub, ast.Assign):
            out.update((_module(t.value), t.attr) for t in sub.targets if isinstance(t, ast.Attribute))
    return out


def _relative_file_name(expr: ast.AST) -> str:
    """CWD 相対のパス (文字列定数 / ``Path("…")``) ならファイル名、それ以外は空。"""
    if (
        isinstance(expr, ast.Call) and _terminal_name(expr.func) in _PATH_TYPES
        and len(expr.args) == 1 and not expr.keywords
    ):
        expr = expr.args[0]
    if not (isinstance(expr, ast.Constant) and isinstance(expr.value, str)) or _is_absolute_path(expr.value):
        return ""
    return PurePosixPath(expr.value.replace("\\", "/")).name


def _is_write_mode(mode: ast.AST | None) -> bool:
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and bool(set(mode.value) & set("wax+"))


def _cwd_written_files(node: ast.AST) -> set[str]:
    """CWD 相対のパスへ書き込むファイル名 (``open(…, "w")`` / ``Path(…).open("w")`` / ``write_text`` / ``write_bytes``)。"""
    out: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        name = _terminal_name(sub.func)
        mode = next((k.value for k in sub.keywords if k.arg == "mode"), None)
        if name == "open" and isinstance(sub.func, ast.Attribute):  # Path("…").open("w")
            target, mode = sub.func.value, sub.args[0] if sub.args else mode
        elif name == "open" and sub.args:
            target, mode = sub.args[0], sub.args[1] if len(sub.args) > 1 else mode
        elif name in ("write_text", "write_bytes") and isinstance(sub.func, ast.Attribute):
            target, mode = sub.func.value, ast.Constant("w")
        else:
            continue
        file_name = _relative_file_name(target)
        if file_name and _is_write_mode(mode):
            out.add(file_name)
    return out


def _cwd_write_reason(file_name: str) -> str:
    return (
        f"writes {file_name} relative to the working directory without moving to tmp_path; "
        "run in the delivered folder it would overwrite the real data file"
    )


def _cwd_data_writes(tree: ast.Module, tests: list[_FuncDef], data_files: set[str]) -> dict[int, str]:
    """置き場と同じ名前のファイルへ CWD 相対で書くテスト関数 → そのファイル名 (補助関数の参照も辿る)。"""
    helpers = [n for n in tree.body if isinstance(n, _FuncDef) and n not in tests]
    helpers += [m for c in tree.body if isinstance(c, ast.ClassDef) for m in c.body if isinstance(m, _FuncDef) and m not in tests]
    writes = {h.name: _cwd_written_files(h) & data_files for h in helpers}
    changed = True
    while changed:  # 補助関数が名前で参照する補助関数 (``setattr("data.save_data", mock_save_data)`` の先) を辿る
        changed = False
        for h in helpers:
            extra = set().union(*(writes[n] for n in _names_in(h) if n in writes and n != h.name)) - writes[h.name]
            if extra:
                writes[h.name] |= extra
                changed = True

    def _written(n: ast.AST) -> set[str]:
        direct = _cwd_written_files(n) & data_files
        return direct.union(*(writes[x] for x in _names_in(n) if x in writes))

    out: dict[int, str] = {}
    for file_name in sorted(data_files):
        for t in _tests_that(tree, tests, lambda n, file_name=file_name: file_name in _written(n)):
            out.setdefault(t, file_name)
    return out


def _isolation_reasons(
    tree: ast.Module, tests: list[_FuncDef], locations: list[DataLocation],
) -> dict[int, str]:
    """データの隔離が効かないテスト関数 → 理由 (f_10 §11.1-4)。判定はテスト関数ごと (P3-3)。"""
    if not locations:
        return {}
    constants = {loc.constant for loc in locations if loc.constant}
    replacing = _tests_that(tree, tests, lambda n: bool(_replaced_names(n) & constants))
    out = {
        t: _shared_data_reason(locations) for t in _tests_that(tree, tests, _chdir_in) if t not in replacing
    }
    for loc in locations:
        if loc.copied_by:
            name = loc.constant
            for t in _tests_that(tree, tests, lambda n, name=name: name in _replaced_names(n)):
                out.setdefault(t, _copied_reason(loc))
    aliases = _module_aliases(tree)

    def _replacing(loc: DataLocation) -> set[int]:
        return _tests_that(tree, tests, lambda n: any(
            name == loc.constant and module in (loc.module, "") for module, name in _replaced_targets(n, aliases)
        ))

    # 既定引数に焼き込んだ定数の差し替えは届かない (R17)
    for loc in locations:
        if loc.constant and loc.baked:
            for t in _replacing(loc):
                out.setdefault(t, _baked_reason(loc))
    # chdir せずに置き場と同じ名前のファイルへ CWD 相対で書く — 配信フォルダで走らせると実物を書き換える (独立レビュー P1)
    data_files = {_data_file_name(loc) for loc in locations if loc.constant} - {""}
    chdir = _tests_that(tree, tests, _chdir_in)
    for t, file_name in _cwd_data_writes(tree, tests, data_files).items():
        if t not in chdir:
            out.setdefault(t, _cwd_write_reason(file_name))
    # 兄弟が同じファイルの置き場を別に定義して読む — 片方だけの差し替えでは、もう片方が実物を使う (R17)
    groups: dict[str, list[DataLocation]] = {}
    for loc in locations:
        if loc.constant:
            groups.setdefault(_data_file_name(loc), []).append(loc)
    for file_name, group in groups.items():
        if len({loc.module for loc in group}) < 2:
            continue
        hits = [(loc, _replacing(loc)) for loc in group]
        for t in set().union(*(h for _loc, h in hits)):
            missing = [loc for loc, h in hits if t not in h]
            if missing:
                out.setdefault(t, _sibling_reason([loc for loc, h in hits if t in h], missing, file_name))
    return out


def _binds_pytest(tree: ast.Module) -> bool:
    """``pytest`` という名前を束縛する import があるか (``import pytest as pt`` は ``pt`` を束縛する)。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any((a.asname or a.name.split(".")[0]) == "pytest" for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if any((a.asname or a.name) == "pytest" for a in node.names):
                return True
    return False


def add_missing_pytest_import(source: str) -> str:
    """``pytest`` の名前を使うのに束縛する import が無ければ、先頭の import 群に ``import pytest`` を足す。

    LLM の参考テストが ``pytest.raises`` を使いながら import を書き忘れ、NameError で偽の不合格に
    なった (2026-09-28 create 再監査 K04)。構文エラー・足して構文が壊れるなら元のまま。
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return source
    uses = any(isinstance(n, ast.Name) and n.id == "pytest" for n in ast.walk(tree))
    if not uses or _binds_pytest(tree):
        return source
    lines = _source_lines(source)
    at = _import_insert_line(tree, lines)
    out = "".join(lines[:at]) + "import pytest\n" + "".join(lines[at:])
    try:
        ast.parse(out)
    except (SyntaxError, ValueError):
        return source
    return out


#: pytest の組み込み fixture のうち、参考テストが引数に取らずに使う名前 (R18。一覧はここ 1 か所)
PYTEST_BUILTIN_FIXTURES: frozenset[str] = frozenset(
    {"monkeypatch", "tmp_path", "tmp_path_factory", "capsys", "capfd", "request"},
)


def _is_fixture(func: _FuncDef) -> bool:
    """``@pytest.fixture`` / ``@pytest.fixture(...)`` / ``@fixture`` か。"""
    return any(
        _terminal_name(d.func if isinstance(d, ast.Call) else d) == "fixture" for d in func.decorator_list
    )


def add_missing_fixture_params(source: str) -> str:
    """テスト関数と fixture の本体が使う組み込み fixture の名前を、引数に無ければ足す (f_10 §11.1-4、R18)。

    名前 (:data:`PYTEST_BUILTIN_FIXTURES`) が関数の中で束縛されている (引数・代入・``for`` / ``with``・内側の関数の
    引数等) か、モジュール直下の名前なら足さない。足す位置は引数の先頭 (メソッドは ``self`` / ``cls`` の後)。
    構文エラー・足して構文が壊れるなら元のまま。

    ライブ監査 K06 (2026-09-28): ``def test_add_task(monkeypatch_load_save, tmp_path):`` の本体が
    ``monkeypatch.setattr(…)`` を使い NameError で偽の不合格になった。
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return source
    tests = _test_functions(tree)
    candidates: list[tuple[_FuncDef, bool]] = []
    for node in tree.body:
        if isinstance(node, _FuncDef) and (node in tests or _is_fixture(node)):
            candidates.append((node, False))
        elif isinstance(node, ast.ClassDef):
            candidates += [(m, True) for m in node.body if isinstance(m, _FuncDef) and (m in tests or _is_fixture(m))]
    module_names: set[str] = set()
    for node in tree.body:
        module_names |= {node.name} if isinstance(node, _FuncDef | ast.ClassDef) else _binding_names(node)
    offset = _offset_of(source)
    edits: list[tuple[int, int, str]] = []
    for func, in_class in candidates:
        used = sorted(
            (n for stmt in func.body for n in ast.walk(stmt)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in PYTEST_BUILTIN_FIXTURES),
            key=lambda n: (n.lineno, n.col_offset),
        )
        bound = _binding_names(func) | module_names
        missing = list(dict.fromkeys(n.id for n in used if n.id not in bound))
        if not missing:
            continue
        names = ", ".join(missing)
        args = func.args
        positional = [*args.posonlyargs, *args.args]
        if in_class and positional and positional[0].arg in ("self", "cls"):
            first = positional[0]
            at = offset(first.end_lineno or first.lineno, first.end_col_offset or 0)
            edits.append((at, at, ", " + names))
        elif positional:
            at = offset(positional[0].lineno, positional[0].col_offset)
            edits.append((at, at, names + ", "))
        else:
            # 位置引数が無い: 開き括弧の直後
            paren = re.compile(rf"def\s+{re.escape(func.name)}\s*(?:\[[^\]]*\])?\s*\(").search(
                source, offset(func.lineno, func.col_offset),
            )
            if paren is None:
                continue
            rest = args.vararg or args.kwonlyargs or args.kwarg
            edits.append((paren.end(), paren.end(), names + (", " if rest else "")))
    if not edits:
        return source
    out = _apply_edits(source, edits)
    try:
        ast.parse(out)
    except (SyntaxError, ValueError):
        return source
    return out


def lint_generated_tests(
    source: str, *, allowed_text: str, data_locations: list[DataLocation] | None = None,
) -> tuple[str, list[str]]:
    """壊れやすいテスト関数を落とした本文と、落とした理由の一覧を返す。

    ``allowed_text`` は骨組み (と依頼文) の全文。そこに現れる文言との一致は契約の一部として残す。
    ``data_locations`` は本文が ``__file__`` から組むデータの置き場 (:func:`file_based_data_locations`)。
    あるのに、置き場の定数を差し替えず chdir で隔離するテスト関数は落とす — chdir では置き場が動かず、
    実物のデータファイルをテスト間で共有する (2026-09-28 実機確認 K01、R12、f_10 §11.1-4)。兄弟が名前
    import した定数を差し替えるテスト関数も落とす (写しに届かない)。判定はテスト関数ごと。
    構文エラーなら (空文字列, [理由])。残るテスト関数が 0 件なら本文は空文字列。
    既定引数に焼き込まれた定数を差し替えるテスト関数と、兄弟が別に定義して読む同じファイルの置き場を片方だけ
    差し替えるテスト関数も落とす (R17)。
    残った本文が ``pytest`` を使うのに import が無ければ足し (:func:`add_missing_pytest_import`)、組み込み fixture を
    引数に取らずに使う関数には引数を足す (:func:`add_missing_fixture_params`、R18)。
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return "", [f"syntax error: {exc}"]
    drop: list[tuple[int, int]] = []
    reasons: list[str] = []
    tests = _test_functions(tree)
    # モジュール直下・fixture (テスト関数以外) の絶対パスは全テストに効くので、ファイルごと落とす
    for node in tree.body:
        if node in tests:
            continue
        if isinstance(node, ast.ClassDef):
            outside = [m for m in node.body if m not in tests]
            path = next((p for p in map(_absolute_path_in, outside) if p), "")
        else:
            path = _absolute_path_in(node)
        if path:
            return "", [f"module: uses an absolute path outside a test ({path})"]
    isolation = _isolation_reasons(tree, tests, list(data_locations or []))
    for func in tests:
        reason = _brittle_reason(func, allowed_text) or isolation.get(id(func), "")
        if reason:
            start = min([func.lineno, *[d.lineno for d in func.decorator_list]])
            drop.append((start, func.end_lineno or func.lineno))
            reasons.append(f"{func.name}: {reason}")
    if len(drop) == len(tests):
        return "", reasons
    lines = [line.rstrip("\r\n") for line in _source_lines(source)]
    for start, end in sorted(drop, reverse=True):
        del lines[start - 1:end]
    return add_missing_pytest_import(add_missing_fixture_params("\n".join(lines).rstrip() + "\n")), reasons


__all__ = [
    "PYTEST_BUILTIN_FIXTURES",
    "DataLocation",
    "ExampleCase",
    "add_missing_fixture_params",
    "add_missing_pytest_import",
    "build_example_tests",
    "file_based_data_locations",
    "file_dependent_example_count",
    "here_expression",
    "lint_generated_tests",
    "rebase_absolute_paths_in_expression",
    "repair_windows_path_literals",
    "path_constants_by_module",
    "rebase_absolute_paths_in_source",
    "unbake_default_data_paths",
    "usable_examples",
]
