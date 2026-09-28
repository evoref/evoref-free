"""staged v2 — 骨組み → 同時生成 → smoke → (Pro) 契約テスト → as-built 文書 (f_10 §11)。

旧経路 (``chat_stream_staged.run_staged_pipeline``: タスクグラフ → spec 本文 → 深化 → フロー合成 →
コード → LLM テスト → spec 見直し) を置き換える。iGPU では生成が 1 本あたり 7〜8 トークン/秒で
頭打ちになり、同時生成で合計 2.2 倍まで伸びる (2026-09-25 実測) ので、

1. **骨組み** (モジュール・シグネチャ・入口・入出力例) を文法制約 JSON 1 回で作る
2. 各モジュールの本文を骨組みを共通の文脈に **別スロットで同時生成** する (チャットスロットは使わない)
3. import の配線 → smoke (import / 静的整合)。エラーがあれば当該モジュールだけ 1 回作り直す
4. (Pro) 骨組みの入出力例から LLM を使わずにテストを組み、合否はそれだけで決める。LLM の参考テストは
   lint を掛けて別に走らせ、落ちても警告だけ (正の順序: 依頼 > 骨組み > コード > テスト)
5. SPEC.md / flowchart.md は完成したコードと骨組みから決定論で描く (as-built)

入出力は旧経路と同じ (``kind`` = ``step`` / ``run_started`` / ``result`` の dict を yield)。
消費者は :class:`backend.free.loop.staged.harness.StagedCodeHarness`。
"""

from __future__ import annotations

import ast
import asyncio
import json
import posixpath
import re
import time
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, AsyncIterator, Callable

from backend.free.api.chat import staged_v2_languages as languages
from backend.free.api.chat.chat_stream_common import cancel_requested, logger
from backend.free.api.chat.chat_stream_staged import (
    _REFERENCE_DOC_MAX_CHARS,
    _STAGE_BUDGET_FLOOR_SEC,
    _STAGED_PROJECT_ID,
    _STAGED_TOTAL_TIMEOUT_DEFAULT_SEC,
    _TURN_TIMEOUT_DEFAULT_SEC,
    _emit_check,
    _finish_run_safely,
    _staged_import_smoke,
    _staged_postprocess,
)
from backend.free.core.file_names import allows_empty_content
from backend.free.core.check_outcome import (
    CheckKind,
    CheckOutcome,
    CheckStatus,
    UncheckedReason,
    unchecked_count,
)
from backend.free.core.intent_vocab import is_absolute_path_text
from backend.free.generation.contract_tests import (
    file_based_data_locations,
    rebase_absolute_paths_in_expression,
    path_constants_by_module,
    rebase_absolute_paths_in_source,
    unbake_default_data_paths,
)
from backend.free.generation.package_layout import (
    choose_entry,
    package_code_map,
    package_from_usage,
    to_package_imports,
    usage_blocker,
    usage_command,
    usage_outcome,
)
from backend.free.generation.smoke_validator import check_call_arity, run_usage
from backend.free.core.prompt_blocks import SHARED_CONTEXT_BOUNDARY
from backend.trace_context import get_trace_id, run_in_executor_with_context
from backend.utils import estimate_tokens as _estimate_tokens

if TYPE_CHECKING:
    from backend.app_state import AppState

#: 1 モジュールの本文の最大トークン (規模に比例させず上限だけ置く)。
_MODULE_MAX_TOKENS = 2048
#: LLM の参考テスト (Pro) の最大トークン。
_ADVISORY_TEST_MAX_TOKENS = 1024
#: 骨組み (JSON) の最大トークン。
_SKELETON_MAX_TOKENS = 1536
#: 作り直しの上限 (1 回目が上限で切れたとき用、f_10 §12.2)。
_SKELETON_RETRY_MAX_TOKENS = 2560
#: 同時生成の上限 (チャット以外のスロット数と小さい方)。
_DEFAULT_PARALLEL = 3
#: 1 LLM 呼出の壁時計の床 (残り予算がこれを切ったら呼ばない)。
_CALL_FLOOR_SEC = 60.0
#: 使い方を配信形で実行する時間の上限 (``test_timeout_sec`` と小さい方、f_10 §11.1-3)。
_USAGE_TIMEOUT_SEC = 30.0

_SKELETON_PROMPT = """\
You design a small software deliverable before any code is written. Return ONLY JSON.

Request (the only specification — design exactly this):
{request}
{reference}
Rules:
- Any background in the system message (facts, memory, prior work) describes earlier and possibly
  unrelated work. Use it only where the request explicitly refers to it; never design a previous
  project instead of the request.
- modules: the source files to write. Use exactly the file names (and the sub-folder) the request names;
  otherwise choose short snake_case names. A path is relative to the output folder: never start it with a
  drive letter, "/" or the folder path written in the request (the files are written there). The folder the
  request writes is the output folder itself: do not repeat its name at the start of a path. Write every file in the language the request asks for
  and give it that language's extension (".py" for Python, ".html", ".js", ".ts", ".css", …);
  Python when the request does not say. Do not put test files, README, SPEC.md or data files in
  modules. Prefer the fewest modules that satisfy the request (one file when the request names one file).
- data_files: the data files the request asks you to create (e.g. "sample.csv" for "prepare a sample
  sample.csv"); [] otherwise. Never list the program's own storage file.
- tests: the test files the request asks for (e.g. ["test_md_toc.py"] for "with tests"); [] when the
  request does not ask for tests.
- components: every public function/class of the file with an exact signature in the file's language
  (Python with type hints: "def add(a: int, b: int) -> int", "class Todo" plus its methods as
  "def Todo.mark_done(self) -> None"; JavaScript: "function start()" (no types);
  TypeScript: "export function start(): void";
  PHP: "function price_with_tax(float $price, float $rate): int"). For a class, list every method
  other files call (e.g. "def Todo.to_dict(self) -> dict"). For HTML list the element ids / classes
  other files use (e.g. "#start button", "#display"); for CSS the main selectors; for SQL the tables
  ("table sales(id, product_id, qty)"). summary is one short line in {language_name}.
- summary: a short title of the deliverable (a few words, not the request itself) in {language_name}.
  role: what the module is for, a few words in {language_name}.
- imports_from: paths of other files of this deliverable that the file uses (imports, the scripts and
  stylesheets an HTML page loads, PHP require, the HTML whose element ids a script uses, the SQL
  schema a query file needs). Keep the dependencies one-directional (no file may use a file that
  uses it).
- entry_module: the file run or opened first ("" for a library). usage: the exact command line or
  action (e.g. "python cli.py add <text>", "php main.php 1000 0.1", "open index.html in a browser").
- examples: only for Python modules: 2-4 input/output examples that pin the behaviour the request
  asks for. call is a Python expression using only the module's public names (e.g. "add(1, 2)");
  expected is a Python literal (e.g. "3"). Only pure, deterministic calls (no I/O, no printing, no
  randomness). Use [] when there is no such Python function.
- Do not invent user-facing message texts, extra options or behaviours the request does not ask for.
"""

#: 本文と参考テストの両方のプロンプトが参照する「絶対パスを書かない」(文言を 2 か所に写さない)。
#: 共通文脈の依頼文には絶対パスがあり、書かれたテストは lint がファイルごと落とす (2026-09-28 R8)。
_ABSOLUTE_PATH_BAN = 'Never write an absolute path (a drive letter, a leading "/" or the folder path written in the request)'

_MODULE_TASK = """\
Write the complete file `{path}`.

It must implement exactly these public components (same names and signatures):
{components}

Rules:
{language_rules}- From other files of this deliverable use only the components the design lists (or the code
  shown below) — do not call methods or attributes they do not define.
- Comments and user-facing messages in {language_name}; keep messages short and do not add
  features the request does not ask for.
- """ + _ABSOLUTE_PATH_BAN + """
  into the code: the files are delivered to that folder, so locate the program's own data files
  relative to this file. When a different location is needed, take it from a command-line argument or
  an environment variable.
{entry_rule}- Output only the file content.
"""

_REPAIR_TASK = """\
The previous version of `{path}` failed the checks below. Fix it and write the complete file again.

Errors:
{errors}

Previous version:
```{fence}
{previous}
```
"""

_ADVISORY_TEST_TASK = """\
Write a pytest file `{path}` with at most 5 focused test functions for the public API above.

Rules:
- Import modules by bare name (e.g. `import {example_module}`).
- Test behaviour only: return values, raised exception TYPES (`pytest.raises(ValueError)` without
  `match=`), and files written. For stdout use the `capsys` fixture and check a short substring at most.
- Never reassign sys.stdout, never compare `__annotations__`, never assert exact user-facing
  message texts.
- """ + _ABSOLUTE_PATH_BAN + """
  into the tests, not even in fixtures or module-level constants: a test file that does is discarded.
  Where the code reads or writes data files, keep every file under pytest's `tmp_path`. When the
  code locates a data file from `__file__` (a module-level constant such as `DATA_FILE`), replace
  that constant in every such test with `monkeypatch.setattr(<module>, "<CONSTANT>", <new path>)`
  where <new path> has the constant's type (`str(tmp_path / "<file name>")` for a string,
  `tmp_path / "<file name>"` for a `pathlib.Path`): changing the working directory does not move
  such a file. Use `monkeypatch.chdir(tmp_path)` only when the code opens its data files relative
  to the working directory.
- Output only the file content.
{data_constants}"""

#: 前の段の本文から拾った置き場の定数 (``_ADVISORY_TEST_TASK`` の末尾、task 部分なので KV 接頭辞は変わらない)。
_ADVISORY_DATA_CONSTANTS = """\
- The code keeps its data files at these module-level constants; in every test that reads or
  writes data, replace each of them first:
{lines}
"""


def _advisory_data_constants(written: dict[str, str]) -> str:
    """書き上がった本文の置き場の定数を、参考テストの差し替えの指示にする (無ければ空、f_10 §11.1-4)。"""
    lines = [
        f'  `monkeypatch.setattr({loc.module}, "{loc.constant}", {loc.replacement})`'
        # 既定引数に焼き込まれた定数は差し替えが届かないので挙げない (R17)
        for loc in file_based_data_locations(written) if loc.constant and not loc.baked
    ]
    return _ADVISORY_DATA_CONSTANTS.format(lines="\n".join(lines)) if lines else ""


def _unbake(cmap: dict[str, str], paths: list[str], record: list[str]) -> dict[str, str]:
    """``paths`` の ``.py`` の本文の既定引数に焼き込んだ置き場を、呼び出し時に読む形へ直す (f_10 §11.1-1、R17)。

    兄弟の置き場の定数は ``cmap`` の全本文から引く。書き換えた ``<ファイル>: <関数>(<引数>)`` を ``record`` に足す。
    """
    out = dict(cmap)
    siblings = path_constants_by_module(cmap)
    for path in paths:
        if not path.endswith(".py") or path not in cmap:
            continue
        stem = PurePosixPath(path).stem
        new, rewritten = unbake_default_data_paths(
            cmap[path], sibling_constants={m: c for m, c in siblings.items() if m != stem},
        )
        if rewritten:
            logger.info(
                "staged v2: %s read its data location from default arguments; "
                "rewrote them to read it at call time: %s", path, ", ".join(rewritten),
            )
            out[path] = new
            record.extend(e for e in (f"{path}: {r}" for r in rewritten) if e not in record)
    return out


def _language_name(locale: str) -> str:
    return "Japanese" if str(locale).startswith("ja") else "English"


def _norm_path(raw: str) -> str:
    p = str(raw or "").strip().replace("\\", "/").lstrip("./")
    parts = [x for x in PurePosixPath(p).parts if x not in ("", ".", "..")]
    return "/".join(parts)


#: 第 1 段の対象外 (言語・系統の混在) の結果 (``notes["fallback"]``)。ハーネスが旧経路へ回す (f_10 §12.1)。
UNSUPPORTED_LANGUAGE_FALLBACK = "unsupported_language"


def _absolute_module_paths(data: dict) -> list[str]:
    """モデルが書いた絶対パス (形の判定は ``is_absolute_path_text`` の 1 本)。出力先は依頼のフォルダで決まる (f_10 §11.1-1)。"""
    return [
        str(m.get("path") or "").strip() for m in data.get("modules") or []
        if is_absolute_path_text(str(m.get("path") or ""))
    ]


def _outside_request(data: dict, query: str) -> list[str]:
    """モジュールの絶対パスのうち、配信先の根 (f_03 §4.4) の外にあるもの。"""
    absolute = _absolute_module_paths(data)
    if not absolute:
        return []
    from backend.free.agent.meta_cognitive_task_exec import delivery_roots, relative_to_roots

    roots = delivery_roots(query)
    return [p for p in absolute if relative_to_roots(p, roots) is None]


def _rebase_absolute_paths(data: dict, query: str) -> dict:
    """モデルが書いた絶対パスを出力先からの相対に直す (f_10 §11.1-1)。

    配信先の根の配下ならその根からの相対。外なら (作り直しても外だった) 外のパスに共通の
    親を捨てる — 出力先は依頼のフォルダで決まる。
    """
    absolute = _absolute_module_paths(data)
    if not absolute:
        return data
    from backend.free.agent.meta_cognitive_task_exec import delivery_roots, relative_to_roots

    roots = delivery_roots(query)
    outside = [p.replace("\\", "/") for p in absolute if relative_to_roots(p, roots) is None]
    try:
        base = posixpath.commonpath([posixpath.dirname(p) for p in outside]) if outside else ""
    except ValueError:
        base = ""

    def _rebase(raw) -> str:
        text = str(raw or "").strip()
        if not is_absolute_path_text(text):
            return text
        rel = relative_to_roots(text, roots)
        if rel is not None:
            return rel
        norm = text.replace("\\", "/")
        if base and norm.startswith(base + "/"):
            return norm[len(base) + 1:]
        return PurePosixPath(norm).name

    return _map_skeleton_paths(data, _rebase)


def _map_skeleton_paths(data: dict, fn: Callable[[object], str]) -> dict:
    """骨組みのパスを持つ欄 (``modules[].path`` / ``imports_from`` / ``entry_module`` / ``examples[].module``) に ``fn`` を掛ける。"""
    return {
        **data,
        "modules": [
            {**m, "path": fn(m.get("path")), "imports_from": [fn(x) for x in m.get("imports_from") or []]}
            for m in data.get("modules") or []
        ],
        "entry_module": fn(data.get("entry_module")),
        "examples": [{**e, "module": fn(e.get("module"))} for e in data.get("examples") or []],
    }


#: 引用符で括った絶対パス (``"E:\\my work\\todo"``)。空白を含むので ``EXPLICIT_WINDOWS_PATH_RE`` では切れる。
_QUOTED_ABS_PATH_RE = re.compile(r"[\"'“「『]\s*[A-Za-z]:[\\/][^\"'”」』]*[\"'”」』]")
#: 名前をフォルダとして名指す形の後ろ (``X フォルダ`` / ``X folder`` / ``X/`` / ``X\\``)。
_FOLDER_NAMING_TAIL = r"\s*(?:フォルダ|folder|[\\/])"


def _names_folder_outside_paths(query: str, names: list[str]) -> bool:
    """依頼文が明示パスの外で ``names`` のどれかをフォルダとして名指しているか (「X フォルダの中に X フォルダを」)。

    名前がただの語として出る (``todo アプリ`` / ``todoリスト`` / ``(timer)``) のは名指しではない。パスの途中
    (``\\`` / ``/`` の直後) も数えない。大文字小文字は剥がす側 (casefold) と揃えて区別しない。
    """
    from backend.free.core.intent_vocab import EXPLICIT_WINDOWS_PATH_RE

    text = _QUOTED_ABS_PATH_RE.sub(" ", query or "")
    text = _POSIX_ABS_IN_TEXT_RE.sub(" ", EXPLICIT_WINDOWS_PATH_RE.sub(" ", text))
    return any(
        re.search(r"(?<![\w.\\/-])" + re.escape(n) + _FOLDER_NAMING_TAIL, text, re.ASCII | re.IGNORECASE)
        for n in names
    )


def _strip_output_folder_prefix(data: dict, query: str) -> dict:
    """相対パスの先頭に繰り返された出力フォルダの名前を剥がす (f_10 §11.1-1、2026-09-28 R15)。

    依頼「``E:\\tmp\\X`` フォルダに…」に骨組みが ``X/cli.py`` を返すと、共通の親 ``X`` が出力フォルダの
    接頭辞になり ``E:\\tmp\\X\\X\\cli.py`` へ配信された。**全モジュール** の先頭 k 段が出力フォルダの末尾
    k 段と一致すれば (Windows なので大文字小文字は区別しない、k は最長一致で全モジュール同じ、ファイル名の
    段は残す)、パスの欄と ``usage`` の先頭の前置を剥がす。一部のモジュールだけなら本物のサブフォルダ
    (``static/app.js``) なので剥がさない。剥がす前の共通の親でパッケージ形 (``python -m X``) が成り立つとき、
    依頼文が明示パスの外でその名前をフォルダとして名指すときも剥がさない。
    """
    from backend.free.agent.meta_cognitive_task_exec import _explicit_output_dir

    out_dir = _explicit_output_dir(query)
    if out_dir is None:
        return data
    win = PureWindowsPath(str(out_dir))
    spelled = list(win.parts[1:] if win.anchor else win.parts)
    tail = [p.casefold() for p in spelled]

    def _prefix_len(raw) -> int:
        parts = [p.casefold() for p in PurePosixPath(_norm_path(str(raw or ""))).parts]
        return next((k for k in range(min(len(parts) - 1, len(tail)), 0, -1) if parts[:k] == tail[-k:]), 0)

    paths = skeleton_paths(data)
    ks = {_prefix_len(p) for p in paths}
    if len(ks) != 1 or 0 in ks:
        return data
    k = ks.pop()
    try:
        common = posixpath.commonpath([posixpath.dirname(p) for p in paths])
    except ValueError:
        common = ""
    if package_from_usage(str(data.get("usage") or ""), common):
        return data
    names = spelled[-k:]
    if _names_folder_outside_paths(query, names):
        return data

    def _strip(raw) -> str:
        text = str(raw or "").strip()
        parts = PurePosixPath(_norm_path(text)).parts
        if len(parts) > k and [p.casefold() for p in parts[:k]] == tail[-k:]:
            return "/".join(parts[k:])
        return text

    # ``python c0928x_01_todo/cli.py add x`` → ``python cli.py add x`` (モジュールのパスと食い違わせない)
    prefix = r"(?<![\w.:\\/-])" + r"[\\/]".join(re.escape(n) for n in names) + r"[\\/](?=[^\s\\/])"
    usage = re.sub(prefix, "", str(data.get("usage") or ""), flags=re.IGNORECASE)
    logger.info(
        "staged v2: stripped the output folder name %s repeated at the head of the skeleton paths",
        "/".join(names),
    )
    return {**_map_skeleton_paths(data, _strip), "usage": usage}


#: 文中の POSIX の絶対パス (``/home/u/x.log``)。URL (``http://h/x``) の途中は拾わない。
_POSIX_ABS_IN_TEXT_RE = re.compile(r"(?<![\w:/.~-])/[^\s\"'<>|]+")


def delivery_relative(query: str, folder: str, module: str = "") -> Callable[[str], str | None]:
    """絶対パス → ``module`` を実際に配信するフォルダからの相対 (``/`` 区切り)。書き換えないなら ``None``。

    書き換えるのは **実際に配信する根 1 つ** の配下だけ。配信先は配信の書込みと同じ確定
    (``_resolve_write_path_from_query`` → ``anchor_relative_output_path``) で求め、その場所を含む
    配信先の根 (f_03 §4.4) を基準にする。依頼が明示した他のフォルダ (入力ファイルの置き場
    ``E:\\data\\in``) の配下は書き換えない — 以前は依頼の全フォルダを根にしていて、入力の
    ``E:\\data\\in\\input.csv`` まで ``__file__`` 基準の ``input.csv`` に直した (独立レビュー)。
    ``module`` を省くと出力フォルダ (使い方・入出力の例の基準)。配信先は最初に絶対パスを見たときに
    1 回だけ求める (絶対パスが無ければ求めない)。
    """
    state: dict = {}

    def _setup() -> None:
        from backend.free.agent.meta_cognitive_task_exec import (
            _TaskExecutionMixin,
            delivery_roots,
            relative_to_roots,
        )
        from backend.free.agent.output_format import anchor_relative_output_path

        name = PurePosixPath(module).name if module else "__evoref_probe__.py"
        target = f"{folder}/{name}" if folder else name
        delivered = anchor_relative_output_path(
            _TaskExecutionMixin._resolve_write_path_from_query(target, query),
        )
        module_dir = str(PurePosixPath(delivered.replace("\\", "/")).parent)
        root = next(
            (r for r in delivery_roots(query) if relative_to_roots(module_dir, [r]) is not None),
            module_dir,
        )
        state["root"] = [root]
        state["module_rel"] = relative_to_roots(module_dir, [root]) or "."
        state["relative_to_roots"] = relative_to_roots

    def _relative(path: str) -> str | None:
        if not state:
            _setup()
        rel = state["relative_to_roots"](path, state["root"])
        if rel is None:
            return None
        return posixpath.relpath(rel or ".", state["module_rel"])

    return _relative


#: 使い方の先頭の「出力フォルダへ移る」部分 (相対化で ``cd .`` になったもの)。
_CD_HERE_RE = re.compile(r"^\s*cd\s+(?:/d\s+)?\.[\\/]?\s*(?:&&|;)\s*", re.IGNORECASE)


def relativize_usage(usage: str, relative_of: Callable[[str], str | None]) -> str:
    """使い方 (コマンド行) の中の配信先の配下の絶対パスを出力フォルダからの相対にする (f_10 §11.1-1)。

    ``python E:\\tmp\\c0927_01_todo\\todo.py add a`` → ``python todo.py add a``。フォルダそのものは ``.``。
    """
    from backend.free.core.intent_vocab import EXPLICIT_WINDOWS_PATH_RE

    def _sub(m: re.Match) -> str:
        text = m.group(0)
        token = text.rstrip(".,;:)")
        rel = relative_of(token) if is_absolute_path_text(token) else None
        return text if rel is None else (rel or ".") + text[len(token):]

    for rx in (EXPLICIT_WINDOWS_PATH_RE, _POSIX_ABS_IN_TEXT_RE):
        usage = rx.sub(_sub, usage)
    # ``cd E:\\tmp\\x && python todo.py`` → ``python todo.py`` (``cd . &&`` を残さない)
    return _CD_HERE_RE.sub("", usage)


def skeleton_problem(data: dict, query: str) -> str:
    """骨組みを作り直すべき理由 (f_10 §12.2)。問題が無ければ空文字列。

    依頼が ``schema.sql`` のようにファイルを名指しているのに、骨組みのどのモジュールも
    その名前でなければ、依頼ではなく別の何か (ブリーフの過去の作業) を設計している
    (ベンチ q1: SQL の依頼に以前のお題の単位変換ツールを設計した)。依頼のフォルダの外の
    絶対パス・依頼が名指した実装言語と違う系統も同じ (2026-09-26 ライブ監査 K04:
    キッチンタイマーの依頼に別の依頼のフォルダの ``md_toc.py``)。
    """
    paths = skeleton_paths(data)
    if not paths:
        return "empty"
    if _outside_request(data, query):
        return "outside the requested folder"
    requested = languages.requested_families(query)
    family, _ = languages.family_of(paths)
    if requested and family is not None and family not in requested:
        return f"{family} files for a {'/'.join(sorted(requested))} request"
    named = {
        PurePosixPath(m.group()).name.lower()
        for m in languages.FILE_NAME_IN_TEXT_RE.finditer(query or "")
        if PurePosixPath(m.group()).suffix.lower() in languages.SUPPORTED_SUFFIXES
    }
    if named and not named & {PurePosixPath(p).name.lower() for p in paths}:
        return "unrelated to the files the request names"
    return ""


def _is_test_file(path: str) -> bool:
    """テストファイルの名前か (``test_x.py`` / ``x_test.py`` / ``x.test.js`` / ``x.spec.ts``)。"""
    pure = PurePosixPath(path)
    stem = pure.name[: -len(pure.suffix)] if pure.suffix else pure.name
    return stem.startswith("test_") or stem.endswith(("_test", ".test", ".spec"))


def skeleton_paths(data: dict) -> list[str]:
    """骨組みのモジュールのパス (テスト・データファイルは除く)。

    ``md_toc_test.py`` のような後置の名前もテスト (2026-09-26 ライブ監査 K04 run2 で
    モジュールとして生成された)。依頼されたテストは ``tests`` に持つ (f_10 §11.1-1)。
    """
    out = []
    for m in data.get("modules") or []:
        path = _norm_path(m.get("path", ""))
        suffix = PurePosixPath(path).suffix.lower()
        if not path or _is_test_file(path) or suffix in languages.DATA_SUFFIXES:
            continue
        out.append(path)
    return out


def _file_names(values) -> list[str]:
    """骨組みの ``tests`` / ``data_files`` を拡張子付きのファイル名の列にする (重複は除く)。"""
    out: list[str] = []
    for value in values or []:
        name = PurePosixPath(_norm_path(str(value))).name
        if name and PurePosixPath(name).suffix and name not in out:
            out.append(name)
    return out


def package_of(modules: list[dict], usage: str, folder: str) -> str:
    """パッケージ形のパッケージのフォルダ (f_10 §11.1-1)。Python だけの成果物で、使い方が
    ``python -m <folder>`` (``<folder>.cli`` も) のときだけ。それ以外は空。"""
    if not modules or not all(str(m.get("path") or "").endswith(".py") for m in modules):
        return ""
    return package_from_usage(usage, folder)


def normalize_skeleton(data: dict, query: str = "") -> tuple[dict, str]:
    """骨組みを検証・正規化する。戻り値は (正規化済み骨組み, 出力フォルダ)。

    モジュールのパスは作業フォルダでは平置き (``src/<name>.py``) にし、依頼が名指した共通の
    フォルダ (``todo_app/``) は出力先の接頭辞として返す。使えなければ ``({}, "")``。
    モデルが書いた絶対パスは先に出力先からの相対へ直し (``_rebase_absolute_paths``)、先頭に繰り返された
    出力フォルダの名前を剥がす (``_strip_output_folder_prefix``)。
    """
    data = _strip_output_folder_prefix(_rebase_absolute_paths(data, query), query)
    kept = set(skeleton_paths(data))
    modules = [
        {**m, "path": _norm_path(m.get("path", ""))} for m in data.get("modules") or []
        if _norm_path(m.get("path", "")) in kept
        and PurePosixPath(_norm_path(m.get("path", ""))).suffix.lower() in languages.SUPPORTED_SUFFIXES
    ]
    if not modules:
        return {}, ""
    family, _ = languages.family_of([m["path"] for m in modules])
    parents = [PurePosixPath(m["path"]).parent.parts for m in modules]
    common: list[str] = []
    for level in zip(*parents):
        if len(set(level)) != 1:
            break
        common.append(level[0])
    folder = "/".join(common)

    def _local(path: str) -> str:
        # Python は平置き (import が解決する)、それ以外は共通フォルダの下の相対パス
        pure = PurePosixPath(_norm_path(path))
        if pure.suffix.lower() == ".py":
            return pure.name
        parts = pure.parts[len(common):] if pure.parts[: len(common)] == tuple(common) else pure.parts
        return "/".join(parts)

    flat: list[dict] = []
    seen: set[str] = set()
    for m in modules:
        local = _local(m["path"])
        if local in seen:
            continue
        seen.add(local)
        flat.append({
            **m, "path": local,
            "imports_from": [_local(x) for x in m.get("imports_from") or [] if _norm_path(x)],
        })
    names = {m["path"] for m in flat}
    for m in flat:
        m["imports_from"] = [x for x in m["imports_from"] if x in names and x != m["path"]]
    entry = _local(data.get("entry_module", "") or "")
    # 使い方・入出力の例の絶対パスも出力フォルダからの相対へ (入口の生成が使い方を丸写しする、f_10 §11.1-1)
    relative_of = delivery_relative(query, folder, entry or flat[0]["path"])
    # パッケージ形の使い方はパッケージの親 (実行する CWD) 基準 (``cd ..`` / ``../in.csv`` にしない、独立レビュー 中 2)
    usage_relative_of = (
        delivery_relative(query, "", "") if package_of(flat, str(data.get("usage") or ""), folder) else relative_of
    )
    examples = [
        {**e, "module": _local(e.get("module", "") or ""),
         "call": rebase_absolute_paths_in_expression(str(e.get("call") or ""), relative_of)}
        for e in data.get("examples") or []
    ]
    return {
        "summary": str(data.get("summary") or ""),
        "language": family or "python",
        "modules": flat,
        "entry_module": entry if entry in names else "",
        "usage": relativize_usage(str(data.get("usage") or ""), usage_relative_of),
        "examples": [e for e in examples if e["module"] in names and e["module"].endswith(".py")],
        "data_files": [
            n for n in _file_names(data.get("data_files"))
            if PurePosixPath(n).suffix.lower() in languages.DATA_SUFFIXES
        ],
        "tests": _file_names(data.get("tests")),
    }, folder


def render_skeleton_context(skeleton: dict, request: str) -> str:
    """同時生成の共通文脈 (全モジュール・全テストで byte 不変、KV の接頭辞を共有する)。"""
    return (
        "Request:\n" + request.strip() + "\n\n"
        "Design (the contract every file must follow):\n"
        + json.dumps(skeleton, ensure_ascii=False, indent=1)
    )


def _module_instruction(
    skeleton: dict, module: dict, *, request: str, brief: str, locale: str,
    written: dict[str, str] | None = None,
) -> str:
    components = "\n".join(
        f"- `{c.get('signature', '')}` — {c.get('summary', '')}" for c in module.get("components") or []
    ) or "- (as described by the role)"
    siblings = [m["path"] for m in skeleton["modules"] if m["path"] != module["path"]]
    is_entry = skeleton.get("entry_module") == module["path"]
    usage = skeleton.get("usage", "")
    if is_entry and module["path"].endswith(".py"):
        entry_rule = (
            f"- This is the entry point. Parse the command line exactly as `{usage}` and "
            "end with `if __name__ == \"__main__\":` calling main(). Print results as plain values "
            "(`print(result)`) — do not round or reformat numbers unless the request specifies a format "
            "for every case.\n"
        )
    elif is_entry:
        entry_rule = (
            f"- This is the entry point (`{usage}`). Print or show results as plain values — do not round "
            "or reformat numbers unless the request specifies a format for every case.\n"
        )
    else:
        entry_rule = ""
    shared = (f"{brief}\n\n" if brief else "") + render_skeleton_context(skeleton, request)
    task = _MODULE_TASK.format(
        path=module["path"], components=components,
        language_rules=languages.module_rules(module["path"], siblings=siblings),
        language_name=_language_name(locale), entry_rule=entry_rule,
    )
    deps = [d for d in module.get("imports_from") or [] if d in (written or {})]
    if deps:
        task += "\nAlready written files this file uses (use exactly their API / ids / paths):\n" + "\n".join(
            f"```{languages.fence_language(d)}\n# {d}\n{written[d][:6000]}\n```" for d in deps
        ) + "\n"
    return f"{shared}{SHARED_CONTEXT_BOUNDARY}{task}"


#: as-built 文書の配信名 (``loop/staged/harness.py`` が出力フォルダへ書く名前)。依頼が名指しても未生成とは言わない。
AS_BUILT_DOC_NAMES = ("SPEC.md", "flowchart.md")


_DATA_FILE_TASK = """\
Write the complete file `{path}`.

It is a data file the request asks for (for example sample input). Keep it small, make it realistic, and
make it valid input for the code below (same columns / keys / format the code reads).
- Output only the file content.
"""


def _requested_data_files(skeleton: dict, query: str, folder: str) -> list[str]:
    """骨組みの ``data_files`` のうち、依頼文に名前があり、配信先にまだ無いもの (f_10 §11.1-1)。

    モデルが発明した名前 (依頼文に無い) は採らない。依頼のフォルダに既にあるファイルは入力なので作らない。
    """
    names = skeleton.get("data_files") or []
    if not names:
        return []
    in_query = {
        PurePosixPath(m.group()).name.lower() for m in languages.FILE_NAME_IN_TEXT_RE.finditer(query or "")
    }
    out = []
    for name in names:
        if name.lower() not in in_query:
            continue
        if _exists_in_delivery(name, query, folder):
            logger.info("staged v2: requested data file %s already exists; not generating it", name)
            continue
        out.append(name)
    return out


def _exists_in_delivery(name: str, query: str, folder: str) -> bool:
    """配信先の根 (f_03 §4.4) の出力フォルダか根の直下に ``name`` が既にあるか (あれば入力)。"""
    from backend.free.agent.meta_cognitive_task_exec import delivery_roots

    try:
        return any((root / folder / name).exists() or (root / name).exists() for root in delivery_roots(query))
    except OSError:
        return False


def named_but_missing_data_files(skeleton: dict, query: str, folder: str, produced: set[str]) -> list[str]:
    """依頼文が用意の対象として名指したのに、骨組みにも成果物にも配信先にも無いデータファイル (f_10 §11.1-1)。

    名前の抽出は字句の鍵、「用意の対象か」は判定点 ``requested_data_file`` (c_17 §3.10)。棄権は数えない
    (通知しない)。自動生成はしない — 保存先のファイル (``todos.json``) を種データ入りで作る誤りを避ける。
    """
    from backend.free.generation.requested_data_file import requested_names

    known = {PurePosixPath(n).name.lower() for n in [*(skeleton.get("data_files") or []), *produced]}
    return requested_names(
        query, skip=lambda name: name.lower() in known or _exists_in_delivery(name, query, folder),
    )


def _data_file_instruction(
    skeleton: dict, path: str, code_map: dict[str, str], *, request: str, brief: str,
) -> str:
    """依頼されたデータファイルの生成指示 (書き上がったコードを見て形式を合わせる)。"""
    shared = (f"{brief}\n\n" if brief else "") + render_skeleton_context(skeleton, request)
    code = "\n".join(
        f"```{languages.fence_language(p)}\n# {p}\n{c[:6000]}\n```" for p, c in code_map.items()
    )
    return f"{shared}{SHARED_CONTEXT_BOUNDARY}" + _DATA_FILE_TASK.format(path=path) + (
        f"\nCode of this deliverable:\n{code}\n" if code else ""
    )


def generation_waves(modules: list[dict]) -> list[list[dict]]:
    """生成の段 (最大 2 段)。依存の無いモジュールを先に同時生成し、残りは
    書き上がった依存先のコードを見て同時生成する。

    全部を同時に書くと、モジュール間の取り決め (``Todo.to_dict`` の有無など) が
    骨組みに無い部分で食い違う (2026-09-25 create ベンチ m1)。段を依存の深さ
    ぶん積むと鎖状の依頼で直列になるので 2 段で止める。
    """
    names = {m["path"] for m in modules}
    leaves = [m for m in modules if not [d for d in m.get("imports_from") or [] if d in names]]
    rest = [m for m in modules if m not in leaves]
    return [leaves, rest] if leaves and rest else [modules]


def _worker_slots(client, parallel: int) -> list[int]:
    """同時生成に使うスロット (チャットスロットは除く。重複・負値は除く)。"""
    slots: list[int] = []
    for name in ("longform_slot", "background_slot", "classifier_slot"):
        slot = getattr(client, name, None)
        chat = getattr(client, "chat_slot", None)
        if isinstance(slot, int) and slot >= 0 and slot != chat and slot not in slots:
            slots.append(slot)
    return slots[: max(1, parallel)] or [getattr(client, "longform_slot", -1)]


async def _run_jobs(jobs: list[Callable[[int], "asyncio.Future"]], slots: list[int]) -> list:
    """ジョブ (スロットを受け取る coroutine 関数) をスロット数まで同時に走らせる (順序を保って返す)。"""
    queue: asyncio.Queue[int] = asyncio.Queue()
    for s in slots:
        queue.put_nowait(s)
    results: list = [None] * len(jobs)

    async def _one(i: int) -> None:
        slot = await queue.get()
        try:
            results[i] = await jobs[i](slot)
        finally:
            queue.put_nowait(slot)

    await asyncio.gather(*(_one(i) for i in range(len(jobs))))
    return results


def _task_id(path: str) -> str:
    """ファイルごとの task id (``app.py`` と ``app.js`` を分ける)。"""
    return "code_" + re.sub(r"\W", "_", path)


def _errors_for(module_path: str, errors: list[str]) -> list[str]:
    """そのモジュール自身のエラー (smoke は ``<stem>: …``、静的検査は ``<path>: …`` で始まる)。

    ``main: ImportError: cannot import name 'x' from 'temperature'`` は import した側
    (``main``) の誤り — 骨組みにない名前を使ったのは main なので main を作り直す。
    """
    prefixes = (f"{module_path}:",)
    if module_path.endswith(".py"):
        prefixes += (f"{PurePosixPath(module_path).stem}:",)
    return [e for e in errors if e.startswith(prefixes)]


def _free_names(tree: ast.Module) -> set[str]:
    """モジュールの名前参照のうち、関数の引数・関数内の代入で隠れていないもの。

    ``from cli import done`` を足したモジュールが ``def __init__(self, done=False):
    self.done = done`` とだけ書いていれば、``done`` は引数を指していて import は
    使われていない (2026-09-25 create ベンチ m1 の Free 実機)。
    """
    free: set[str] = set()

    def _bound(func: ast.AST) -> set[str]:
        args = func.args  # type: ignore[attr-defined]
        names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        names |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
        return names

    def _visit(node: ast.AST, shadowed: frozenset[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    _visit(default, shadowed)
            for deco in getattr(node, "decorator_list", []):
                _visit(deco, shadowed)
            inner = shadowed | _bound(node)
            body = node.body if isinstance(node.body, list) else [node.body]
            for child in body:
                _visit(child, inner)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in shadowed:
            free.add(node.id)
        for child in ast.iter_child_nodes(node):
            _visit(child, shadowed)

    _visit(tree, frozenset())
    return free


def break_undeclared_cycles(
    code_map: dict[str, str], skeleton: dict,
) -> tuple[dict[str, str], list[str]]:
    """骨組みに無い import が作る循環を解く (決定論)。

    生成したモジュールが骨組みの ``imports_from`` に無い兄弟を import し、
    それが循環になると smoke は **循環の反対側** (import された側) の名前で
    落ちるので、作り直しが無関係なモジュールへ向く (2026-09-25 create ベンチ
    m1: models.py が ``from cli import done`` を足し、cli.py だけが作り直された)。
    循環上の宣言されていない辺は、取り込んだ名前が未使用なら import 文を消し、
    使っていれば import した側のエラー (``<stem>: …``) として返す。
    """
    stems = {PurePosixPath(p).stem: p for p in code_map if p.endswith(".py")}
    declared = {
        m["path"]: {PurePosixPath(d).stem for d in m.get("imports_from") or []}
        for m in skeleton.get("modules") or []
    }
    trees: dict[str, ast.Module] = {}
    edges: dict[str, dict[str, list[ast.stmt]]] = {}
    for path, source in code_map.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        trees[path] = tree
        own = PurePosixPath(path).stem
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # ``from .cli`` / ``from todo_app.cli`` も兄弟を指す
                targets = sorted({node.module.split(".")[0], node.module.split(".")[-1]})
            elif isinstance(node, ast.Import):
                targets = [a.name.split(".")[0] for a in node.names]
            else:
                continue
            for target in targets:
                if target in stems and target != own:
                    edges.setdefault(own, {}).setdefault(target, []).append(node)

    def _reaches(src: str, dst: str) -> bool:
        seen, stack = set(), [src]
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            if cur not in seen:
                seen.add(cur)
                stack.extend(edges.get(cur, {}))
        return False

    out = dict(code_map)
    errors: list[str] = []
    for own, targets in edges.items():
        path = stems[own]
        for target, nodes in targets.items():
            if target in declared.get(path, set()) or not _reaches(target, own):
                continue
            imported = {
                (a.asname or a.name).split(".")[0]
                for n in nodes for a in n.names  # type: ignore[attr-defined]
            }
            if imported & _free_names(trees[path]):
                allowed = ", ".join(sorted(declared.get(path, set()))) or "none"
                errors.append(
                    f"{own}: circular import — {own} imports {target}, but {target} already "
                    f"(directly or indirectly) imports {own}. {own} must not import {target} "
                    f"(its dependencies are: {allowed}).",
                )
                continue
            drop = {ln for n in nodes for ln in range(n.lineno, (n.end_lineno or n.lineno) + 1)}
            lines = out[path].splitlines(keepends=True)
            out[path] = "".join(line for i, line in enumerate(lines, 1) if i not in drop)
            logger.info("staged v2: dropped unused circular import %s -> %s", own, target)
    return out, errors



def _read_text(path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _unchecked_reason(gate) -> UncheckedReason:
    """skipped のゲートの理由 (``GateResult.skip_kind``)。知らない値は「実行できなかった」。"""
    try:
        return UncheckedReason(gate.skip_kind or "")
    except ValueError:
        return UncheckedReason.NOT_RUN


def _declared_signatures(modules: list[dict]) -> dict[str, str]:
    """骨組みが宣言した関数の signature (``"stem.関数名"`` → ``def f(...)``)。

    モジュール間の引数の数が食い違ったとき、どちらが骨組みから外れたかを決めるのに使う
    (``check_call_arity``、定義側が外れていれば定義側を作り直す)。
    """
    out: dict[str, str] = {}
    for m in modules:
        path = str(m.get("path") or "")
        if not path.endswith(".py"):
            continue
        stem = PurePosixPath(path).stem
        for c in m.get("components") or []:
            sig = str(c.get("signature") or "").strip()
            head = sig.removeprefix("async ").removeprefix("def ").split("(", 1)[0].strip()
            if sig and head.isidentifier():
                out[f"{stem}.{head}"] = sig
    return out


async def run_staged_v2_pipeline(
    *,
    query: str,
    session_id: str,
    state: "AppState",
    cfg: dict,
    output_target: str,
    client,
    brief: str = "",
    total_timeout_sec: float | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    resume_of: str | None = None,
    tests_enabled: bool = False,
) -> AsyncIterator[dict]:
    """staged v2 を駆動し、旧経路と同じ形の構造化イベントを yield する。"""
    from backend.config import get_path_resolver
    from backend.free.generation.as_built import render_flowchart, render_spec
    from backend.free.generation.contract_tests import (
        build_example_tests,
        lint_generated_tests,
        malformed_example_count,
        usable_examples,
    )
    from backend.free.generation.direct_codegen import generate_single_file
    from backend.free.loop.staged import RunEventLog, RunRecordStore, WorkspaceManager
    from backend.free.loop.staged.test_runner import StagedTestRunner, failed_count, failure_lines
    from backend.free.loop.staged.workspace import StageTestResult
    from backend.i18n_helper import get_locale, msg
    from backend.io.id_registry import new_id

    t_start = time.monotonic()
    create_cfg = cfg.get("create", {}) or {}
    staged_cfg = create_cfg.get("staged", {}) or {}
    _is_cancelled = is_cancelled or (lambda: cancel_requested(session_id))
    if total_timeout_sec is None:
        turn = float(create_cfg.get("turn_timeout_sec", _TURN_TIMEOUT_DEFAULT_SEC))
        total_timeout_sec = min(
            float(staged_cfg.get("total_timeout_sec", _STAGED_TOTAL_TIMEOUT_DEFAULT_SEC)),
            max(_STAGE_BUDGET_FLOOR_SEC, turn - 300.0),
        )
    deadline = t_start + float(total_timeout_sec)
    locale = get_locale()
    phase_sec: dict[str, float] = {}
    # 接頭辞 KV を使わずに作り直した回数 (骨組み + 本文、c_14 §2.2)。notes に残す
    gen_stats: dict = {}

    def _remaining() -> float:
        return deadline - time.monotonic()

    enabled_families = set(staged_cfg.get("v2_families") or languages.FAMILIES)

    def _unsupported_result(evidence: list[str], **notes) -> dict:
        # 第 1 段の対象外 (言語・系統の混在・無効な系統) は旧経路へ (Pro は v1、Free は longform、f_10 §12.1)
        logger.info("staged v2: request is outside the supported languages (%s); using the previous pipeline",
                    evidence)
        return {
            "exit_kind": "error",
            "notes": {"fallback": UNSUPPORTED_LANGUAGE_FALLBACK, "unsupported": evidence, **notes},
            "code_map": {}, "spec_md": None, "flowchart_md": None, "tasks_failed": 0,
            "runnability_issues": [], "metrics": {}, "truncated_steps": [],
            "truncated_max_tokens": None,
        }

    evidence = languages.unsupported_request(query)
    if evidence:
        yield {"kind": "result", "payload": _unsupported_result(evidence)}
        return

    run_id = new_id("run_")
    ws = WorkspaceManager.open_or_create(
        get_path_resolver().resolve_local("create_workspace_dir"),
        workspace_id=run_id, session_id=session_id, project_id=_STAGED_PROJECT_ID,
        goal=query, debug_logger=state.debug_logger,
    )

    def _fallback_result(reason: str) -> dict:
        return {
            "exit_kind": "error",
            "notes": {"fallback": "empty_task_graph", "reason": reason,
                      "run_id": run_id, "workspace_root": str(ws.root)},
            "code_map": {}, "spec_md": None, "flowchart_md": None, "tasks_failed": 0,
            "runnability_issues": [], "metrics": {}, "truncated_steps": [],
            "truncated_max_tokens": None,
        }

    # ── 1. 骨組み ────────────────────────────────────────────────
    yield {"kind": "step", "payload": {
        "type": "long_form_plan", "detail": "骨組み (モジュール・シグネチャ・入出力例) を設計中…",
        "status": "running",
    }}
    reference_doc = ""
    try:
        from backend.free.api.chat.chat_stream_output import read_reference_design_doc
        from backend.free.generation.strategy_common import condense_design_doc

        reference_doc = condense_design_doc(
            await read_reference_design_doc(query, state), _REFERENCE_DOC_MAX_CHARS,
        )
    except Exception as exc:  # noqa: BLE001 - 読めなければ参照なしで続ける
        logger.warning("staged v2: reference design document unavailable: %s", exc)
    t0 = time.monotonic()
    aux_client = state.aux_client
    raw: dict = {}
    if aux_client is not None:
        prompt = _SKELETON_PROMPT.format(
            request=query.strip(),
            reference=("\nReference design document:\n" + reference_doc + "\n") if reference_doc else "",
            language_name=_language_name(locale),
        )
        async def _skeleton(system: str | None, max_tokens: int, **cache_kwargs) -> dict:
            try:
                return await aux_client.generate_json(
                    prompt, system=system, purpose="create_skeleton",
                    max_tokens=max_tokens, temperature=0.2,
                    timeout=max(_CALL_FLOOR_SEC, min(240.0 + len(reference_doc) * 0.05, _remaining())),
                    **cache_kwargs,
                ) or {}
            except Exception as exc:  # noqa: BLE001 - 骨組みが無ければ longform へ倒す
                logger.warning("staged v2: skeleton generation failed: %s", exc)
                return {}

        raw = await _skeleton(brief or None, _SKELETON_MAX_TOKENS)
        problem = skeleton_problem(raw, query)
        if problem and _remaining() > _CALL_FLOOR_SEC * 2:
            # 空 (上限で切れた) / 依頼と無関係 (ブリーフの過去の作業に引きずられた) は 1 回だけ、
            # ブリーフを外し上限を上げて作り直す (2026-09-25 create ベンチ q1 / p1)
            # 接頭辞 KV も使わない: 壊れたスロット KV は空白だけの JSON を返す (2026-09-26
            # ライブ監査 #5、c_14 §2.2)
            logger.warning(
                "staged v2: skeleton %s; retrying once without the brief and the prompt cache", problem,
            )
            gen_stats["cache_bypass_retries"] = int(gen_stats.get("cache_bypass_retries") or 0) + 1
            raw = await _skeleton(None, _SKELETON_RETRY_MAX_TOKENS, cache_prompt=False)
    paths = skeleton_paths(raw)
    family, other = languages.family_of(paths)
    if paths and (family is None or family not in enabled_families):
        # 言語名の無い依頼 (「Web ページを作って」) は骨組みで初めて分かる
        yield {"kind": "result", "payload": _unsupported_result(
            other or [f"family:{family}"], run_id=run_id, workspace_root=str(ws.root),
        )}
        return
    skeleton, folder = normalize_skeleton(raw, query)
    phase_sec["skeleton"] = round(time.monotonic() - t0, 1)
    if not skeleton:
        logger.info("staged v2: empty skeleton; falling back (%s)", "no aux" if aux_client is None else "no modules")
        yield {"kind": "result", "payload": _fallback_result("empty_skeleton")}
        return
    # パッケージ形 (f_10 §11.1-1): 使い方が ``python -m <共通の親>`` の Python だけ。生成・検査は平置きのまま、
    # 配信形 (相対 import・__main__.py) は配信の直前に作る。文書・テスト・データはパッケージの外 (出力フォルダ直下)
    package = package_of(skeleton["modules"], skeleton.get("usage", ""), folder)
    out_folder = "" if package else folder
    command = usage_command(skeleton.get("usage", "")) if package else None
    # ``python -m units`` はパッケージそのものを起動する (__main__.py を合成)、``python -m units.cli`` は cli.py のガード
    runs_package = command is not None and command[0] == package.replace("/", ".")
    if package:
        logger.info("staged v2: delivering the Python package %s/ (usage: %s)", package, skeleton.get("usage", ""))

    run_store: RunRecordStore | None = None
    event_log: RunEventLog | None = None
    try:
        run_store = RunRecordStore(ws.root)
        event_log = RunEventLog(ws.root, debug_logger=state.debug_logger)
        run_store.start(
            run_id=run_id, session_id=session_id, request_id=get_trace_id(),
            mode="create", query=query, output_target=output_target,
            brief_tokens=_estimate_tokens(brief) if brief else 0, resume_of=resume_of,
        )
    except Exception as exc:  # noqa: BLE001 - 記録の失敗で制作は止めない
        logger.warning("staged v2 run record start failed: %s", exc)
        run_store, event_log = None, None
    yield {"kind": "run_started", "payload": {"run_id": run_id, "session_id": session_id}}
    try:
        ws.write_plan({m["path"]: list(m.get("imports_from") or []) for m in skeleton["modules"]})
        ws.write_file("skeleton.json", json.dumps(skeleton, ensure_ascii=False, indent=1),
                      kind="spec", stage="spec", task_id="skeleton")
    except Exception as exc:  # noqa: BLE001
        logger.warning("staged v2: skeleton persist failed: %s", exc)
    modules = skeleton["modules"]
    _kind, payload = _emit_check(event_log, "skeleton", {
        "type": "long_form_plan",
        "detail": f"骨組み: {len(modules)} モジュール ({', '.join(m['path'] for m in modules)})",
        "status": "done",
    })
    yield {"kind": "step", "payload": payload}

    # ── 2. 同時生成 (本文 + Pro の参考テスト) ─────────────────────
    parallel = int(staged_cfg.get("parallel_generation", _DEFAULT_PARALLEL))
    slots = _worker_slots(client, parallel)
    yield {"kind": "step", "payload": {
        "type": "task_progress",
        "detail": f"コードを生成中… ({len(modules)} ファイル、同時 {min(len(slots), len(modules))} 本)",
        "status": "running",
    }}
    t0 = time.monotonic()
    code_map: dict[str, str] = {}

    def _module_job(module: dict, instruction: str):
        async def _job(slot: int) -> str:
            if _remaining() < _CALL_FLOOR_SEC or _is_cancelled():
                return ""
            out = await generate_single_file(
                client, instruction, module["path"], max_tokens=_MODULE_MAX_TOKENS,
                request_timeout=max(_CALL_FLOOR_SEC, _remaining()), id_slot=slot, stats=gen_stats,
            )
            return out.get(module["path"], "")
        return _job

    waves = generation_waves(modules)
    advisory_path = ""
    py_modules = [m for m in modules if m["path"].endswith(".py")]
    advisory = tests_enabled and bool(py_modules)
    if advisory:
        advisory_path = f"test_{PurePosixPath(py_modules[0]['path']).stem}_generated.py"
        shared = (f"{brief}\n\n" if brief else "") + render_skeleton_context(skeleton, query)

        def _advisory_job(written: dict[str, str]):
            # 前の段の本文から拾ったデータの置き場の定数を渡す (1 段なら無し、f_10 §11.1-4 R12)
            instruction = f"{shared}{SHARED_CONTEXT_BOUNDARY}" + _ADVISORY_TEST_TASK.format(
                path=advisory_path, example_module=PurePosixPath(py_modules[0]["path"]).stem,
                data_constants=_advisory_data_constants(written),
            )

            async def _job(slot: int) -> str:
                if _remaining() < _CALL_FLOOR_SEC or _is_cancelled():
                    return ""
                out = await generate_single_file(
                    client, instruction, advisory_path,
                    max_tokens=_ADVISORY_TEST_MAX_TOKENS,
                    request_timeout=max(_CALL_FLOOR_SEC, _remaining()), id_slot=slot, stats=gen_stats,
                )
                return out.get(advisory_path, "")
            return _job
    advisory_src = ""
    advisory_done = not advisory
    # 既定引数に焼き込んだ置き場を書き換えた ``<ファイル>: <関数>(<引数>)`` (notes / finalize に残す、R17)
    unbaked_defaults: list[str] = []
    for i, wave in enumerate(waves):
        jobs = [
            _module_job(m, _module_instruction(
                skeleton, m, request=query, brief=brief, locale=locale, written=code_map,
            ))
            for m in wave
        ]
        # 参考テストの置き場所 (f_10 §11.1-2): 段が 1 つなら本文と同時、最後の段より前の本文に置き場の定数が
        # あれば最後の段と同時、無ければ全段の後 (置き場が最後の段にあっても名前を渡す、独立レビュー P3-1)
        together = not advisory_done and i == len(waves) - 1 and (
            len(waves) == 1 or bool(_advisory_data_constants(code_map))
        )
        if together:
            jobs.append(_advisory_job(dict(code_map)))
        wave_results = await _run_jobs(jobs, slots)
        landed = []
        for m, content in zip(wave, wave_results):
            if content or allows_empty_content(m["path"]):
                code_map[m["path"]] = content
                landed.append(m["path"])
        # 受け取った直後に直す — 参考テストのプロンプトに差し替えられる定数の名前を渡すため (R17)
        code_map.update(_unbake(code_map, landed, unbaked_defaults))
        if together:
            advisory_src, advisory_done = wave_results[len(wave)], True
    if not advisory_done:
        advisory_src = (await _run_jobs([_advisory_job(dict(code_map))], slots))[0]
    phase_sec["generate"] = round(time.monotonic() - t0, 1)

    # ── 3. 配線 → smoke → 1 回だけ作り直す ───────────────────────
    t0 = time.monotonic()
    internal = frozenset(
        [PurePosixPath(m["path"]).stem for m in py_modules]
        + [c.get("signature", "").split("(")[0].replace("def ", "").replace("class ", "").split(".")[-1].strip()
           for m in py_modules for c in m.get("components") or []]
    )
    smoke_timeout = float(staged_cfg.get("test_timeout_sec", 120.0))
    # 生成コードの配信先の配下の絶対パスは ``__file__`` 基準へ (作り直さない、f_10 §11.1-1)
    # (基準はモジュールを実際に配信するフォルダ。入力のフォルダは書き換えない)
    relative_of: dict[str, Callable[[str], str | None]] = {}
    rebased_paths: list[str] = []
    smoke_unchecked: list[str] = []
    # 構文検査器が無くて構文を検査できなかったファイル (f_10 §12.4)
    syntax_unchecked: dict[str, str] = {}

    def _rebase(cmap: dict[str, str]) -> dict[str, str]:
        out = dict(cmap)
        for path, content in cmap.items():
            if not path.endswith(".py"):
                continue
            if path not in relative_of:
                relative_of[path] = delivery_relative(query, folder, path)
            new, rebased = rebase_absolute_paths_in_source(content, relative_of[path])
            if rebased:
                logger.info("staged v2: rebased absolute paths in %s onto __file__: %s", path, rebased[:3])
                out[path] = new
                rebased_paths.extend(p for p in rebased if p not in rebased_paths)
        # 既定引数に焼き込んだ置き場は呼び出し時に読む形へ (作り直した版も。兄弟の定数は書き換え後の本文から、
        # f_10 §11.1-1 R17)
        return _unbake(out, list(out), unbaked_defaults)

    def _entry(cmap: dict[str, str]) -> str:
        # 骨組みの入口、無ければ __main__ ガードを持つモジュール、無ければ __init__.py (f_10 §11.1-1)
        return choose_entry({p: c for p, c in cmap.items() if p.endswith(".py")}, skeleton.get("entry_module", ""))

    def _package_code(cmap: dict[str, str]) -> dict[str, str]:
        py = {p: c for p, c in cmap.items() if p.endswith(".py")}
        return package_code_map(py, _entry(cmap), synthesize_main=runs_package)

    def _package_files(cmap: dict[str, str]) -> dict[str, str]:
        # 配信形のパッケージ (``units/__init__.py`` …、f_10 §11.1-1)
        return {f"{package}/{p}": c for p, c in _package_code(cmap).items()}

    def _flat_error(error: str, entry: str) -> str:
        # ``units.length: …`` → ``length: …`` (作り直しの宛先は平置きのモジュール名、合成した __main__ は入口)
        prefix = package.replace("/", ".") + "."
        if not error.startswith(prefix):
            return error
        name, sep, rest = error[len(prefix):].partition(":")
        if name == "__main__" and entry:
            name = PurePosixPath(entry).stem
        return f"{name}{sep}{rest}"

    async def _check(cmap: dict[str, str]) -> tuple[dict[str, str], list[str], list[str]]:
        # 循環の検出は配線の後 (``from .cli import`` / ``from todo_app.cli import`` が
        # 兄弟の bare import に揃ってから見る)
        wired, issues, _ = _staged_postprocess(_rebase(cmap))
        wired, cycle_errors = break_undeclared_cycles(wired, skeleton)
        smoke_unchecked.clear()
        # パッケージ形は配信形 (``<tmp>/units/…``) で import する (f_10 §11.1-3)
        errors = await _staged_import_smoke(
            _package_files(wired) if package else wired, smoke_timeout,
            internal_names=internal, unchecked_out=smoke_unchecked,
        )
        if package:
            errors = [_flat_error(e, _entry(wired)) for e in errors]
        if cycle_errors:
            # 循環の import エラーは反対側の名前で出るので、原因側の指摘に差し替える
            errors = cycle_errors + [
                e for e in errors if "cannot import name" not in e and "partially initialized" not in e
            ]
        # モジュール間呼び出しの引数の数 (実行すれば必ず TypeError になる確定的な欠陥。
        # 呼び出し側のモジュールを作り直しに回す、2026-09-27 残件)
        errors += check_call_arity(
            {p: c for p, c in wired.items() if p.endswith(".py")},
            declared=_declared_signatures(modules),
        )
        # Python 以外: 構文・参照の整合・実行環境による構文検査 (f_10 §12.3 / §12.4)
        errors += await run_in_executor_with_context(
            asyncio.get_running_loop(), None,
            lambda: languages.check(wired, cfg=cfg, skeleton=skeleton, query=query),
        )
        syntax_unchecked.clear()
        syntax_unchecked.update(languages.syntax_unchecked(wired, cfg=cfg))
        return wired, errors, issues

    missing = [m["path"] for m in modules if m["path"] not in code_map]
    code_map, smoke_errors, static_issues = await _check(code_map)
    max_repair = int(staged_cfg.get("max_repair_rounds", 1))
    for _round in range(max(0, max_repair)):
        failing = [
            m for m in modules
            if m["path"] in missing or _errors_for(m["path"], smoke_errors)
        ]
        if not failing and smoke_errors:
            failing = [m for m in modules if m["path"] in code_map]
        if not failing or _remaining() < _CALL_FLOOR_SEC:
            break
        yield {"kind": "step", "payload": {
            "type": "task_progress",
            "detail": msg("create.fixing_check_errors", paths=", ".join(m["path"] for m in failing)),
            "status": "running",
        }}
        repair_jobs = []
        for m in failing:
            instruction = _module_instruction(
                skeleton, m, request=query, brief=brief, locale=locale,
                written={k: v for k, v in code_map.items() if k != m["path"]},
            )
            if m["path"] in code_map:
                instruction += "\n\n" + _REPAIR_TASK.format(
                    path=m["path"], fence=languages.fence_language(m["path"]),
                    errors="\n".join(_errors_for(m["path"], smoke_errors) or smoke_errors)[:3000],
                    previous=code_map[m["path"]][:12000],
                )
            repair_jobs.append(_module_job(m, instruction))
        repaired = await _run_jobs(repair_jobs, slots)
        for m, content in zip(failing, repaired):
            if content or allows_empty_content(m["path"]):
                code_map[m["path"]] = content
        missing = [m["path"] for m in modules if m["path"] not in code_map]
        code_map, smoke_errors, static_issues = await _check(code_map)
    phase_sec["smoke"] = round(time.monotonic() - t0, 1)
    for path, content in code_map.items():
        try:
            ws.write_file(path, content, kind="src", stage="code", task_id=_task_id(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("staged v2: write src %s failed: %s", path, exc)
    tasks_failed = len(missing) + (1 if smoke_errors else 0)
    verification: list[str] = []
    # 検査ごとの 3 値 (合格 / 不合格 / 未検査、f_10 §12.4)。verification の行はここから描く
    checks: list[CheckOutcome] = []
    if missing:
        verification.append(msg("create.not_generated", names=", ".join(missing)))
    if smoke_errors:
        static_check = CheckOutcome.failed(CheckKind.STATIC, errors=len(smoke_errors))
    elif smoke_unchecked:
        static_check = CheckOutcome.unchecked(
            CheckKind.STATIC, UncheckedReason.SANDBOX_VIOLATION, detail=", ".join(smoke_unchecked[:3]),
        )
    elif syntax_unchecked:
        # 環境の欠け (コードの挙動ではない) — 未完了には数えない
        static_check = CheckOutcome.unchecked(
            CheckKind.STATIC, UncheckedReason.MISSING_DEPENDENCY,
            detail=", ".join(sorted(set(syntax_unchecked.values()))),
        )
    else:
        static_check = CheckOutcome.passed(CheckKind.STATIC)
    checks.append(static_check)
    verification.append(static_check.render())
    _kind, payload = _emit_check(event_log, "smoke", {
        "type": "task_result",
        "detail": "smoke 検査: " + (
            static_check.status_text() if not smoke_errors else "; ".join(smoke_errors[:3])
        ),
        "status": "done" if not smoke_errors else "failed",
    })
    yield {"kind": "step", "payload": payload}

    # ── 3b. 依頼されたデータファイル (コードの後、f_10 §11.1-1) ─────────
    # モジュールとは別に返す (smoke・as-built 文書の対象外)。依頼されたテストもここへ入る (§4)
    extra_files: dict[str, str] = {}
    notices: list[str] = []
    data_files = _requested_data_files(skeleton, query, out_folder)
    if data_files and code_map and _remaining() >= _CALL_FLOOR_SEC:
        yield {"kind": "step", "payload": {
            "type": "task_progress", "detail": msg("create.generating_data_files", names=", ".join(data_files)),
            "status": "running",
        }}
        data_out = await _run_jobs([
            _module_job({"path": name}, _data_file_instruction(
                skeleton, name, code_map, request=query, brief=brief,
            ))
            for name in data_files
        ], slots)
        for name, content in zip(data_files, data_out):
            if content:
                extra_files[name] = content
                ws.write_file(name, content, kind="src", stage="code", task_id=_task_id(name))
    missing_data = [name for name in data_files if name not in extra_files]
    if missing_data:
        tasks_failed += len(missing_data)
        verification.append(msg("create.not_generated", names=", ".join(missing_data)))
    # 依頼が「用意して」と名指したのに骨組みが挙げなかったもの — 作らずに本文で伝える (情報表示、tasks_failed に数えない)
    unlisted = named_but_missing_data_files(
        skeleton, query, out_folder, produced={*code_map, *extra_files, *AS_BUILT_DOC_NAMES},
    )
    if unlisted:
        verification.append(msg("create.not_generated", names=", ".join(unlisted)))
        notices.append(msg("create.requested_files_not_generated", names=", ".join(unlisted)))
    # 同梱していないローカル資産の読み込みは警告だけ (作り直し・tasks_failed へ波及させない、f_10 §12.3-5)
    asset_refs = languages.missing_assets({**code_map, **extra_files}, query=query, folder=folder)
    if asset_refs:
        warning = msg("create.missing_assets", refs=", ".join(f"{p} → {r}" for p, r in asset_refs))
        verification.append(warning)
        notices.append(warning)

    # ── 3c. パッケージ形は使い方を配信形で 1 回実行する (f_10 §11.1-3) ──────
    usage_timeout = min(smoke_timeout, _USAGE_TIMEOUT_SEC)

    # 実行するフォルダに置く依頼されたデータファイル (この時点の extra_files はデータだけ。テストは §4 で足す)
    usage_data = dict(extra_files)

    async def _run_usage_check(cmap: dict[str, str]) -> tuple[CheckOutcome, str, str]:
        # 戻り値は (結果, 作り直す stem, traceback の末尾)
        module, args = command
        blocker = usage_blocker(args, cmap, set(usage_data))
        if blocker:
            return CheckOutcome.unchecked(CheckKind.USAGE, UncheckedReason.NEEDS_INPUT, detail=blocker), "", ""
        run = await asyncio.to_thread(run_usage, {**_package_files(cmap), **usage_data}, module, args, usage_timeout)
        outcome, blamed = usage_outcome(run, package, args=args)
        return outcome, blamed, (run.stderr or "")[-3000:]

    usage_check: CheckOutcome | None = None
    usage_checked_code: dict[str, str] = {}
    if command is not None and code_map and not smoke_errors:
        t0 = time.monotonic()
        usage_check, blamed, trace = await _run_usage_check(code_map)
        target = next((m for m in modules if PurePosixPath(m["path"]).stem == blamed), None) or next(
            (m for m in modules if m["path"] == _entry(code_map)), None,
        )
        if (
            usage_check.is_failure and target is not None and target["path"] in code_map
            and _remaining() >= _CALL_FLOOR_SEC
        ):
            yield {"kind": "step", "payload": {
                "type": "task_progress",
                "detail": msg("create.fixing_usage_errors", path=target["path"]),
                "status": "running",
            }}
            instruction = _module_instruction(
                skeleton, target, request=query, brief=brief, locale=locale,
                written={k: v for k, v in code_map.items() if k != target["path"]},
            ) + "\n\n" + _REPAIR_TASK.format(
                path=target["path"], fence=languages.fence_language(target["path"]),
                errors=(
                    f"Running `{skeleton.get('usage', '')}` in the folder that contains the package "
                    f"`{package}/` (these files are delivered as that Python package) failed:\n"
                    + (trace or "; ".join(usage_check.failures))
                ),
                previous=code_map[target["path"]][:12000],
            )
            [content] = await _run_jobs([_module_job(target, instruction)], slots)
            if content:
                # 作り直した版は smoke と使い方の実行の両方に通ったときだけ採る (通らなければ修正前のまま)
                wired_trial, trial_errors, trial_issues = await _check({**code_map, target["path"]: content})
                if not trial_errors and not smoke_unchecked:
                    trial_check, _, _ = await _run_usage_check(wired_trial)
                    if trial_check.status is CheckStatus.PASSED:
                        for path, text in wired_trial.items():
                            if code_map.get(path) != text:
                                ws.write_file(path, text, kind="src", stage="code", task_id=_task_id(path))
                        code_map, static_issues, usage_check = dict(wired_trial), trial_issues, trial_check
                if usage_check.is_failure:
                    logger.warning(
                        "staged v2: the usage repair of %s did not make `%s` run; keeping the previous version",
                        target["path"], skeleton.get("usage", ""),
                    )
        usage_checked_code = dict(code_map)
        phase_sec["usage"] = round(time.monotonic() - t0, 1)

    # ── 4. (Pro) 契約テスト ──────────────────────────────────────
    advisory_note = ""
    # lint が参考テストから落とした理由 (finalize のイベントと notes へ、f_10 §11.1-4)
    advisory_dropped: list[str] = []
    # 依頼されたテストの名前 (Python の参考テストを配信するので .py だけ。無ければ test_<入口>.py)
    requested_tests = [n for n in skeleton.get("tests") or [] if n.endswith(".py")] or (
        [f"test_{PurePosixPath(py_modules[0]['path']).stem}.py"] if skeleton.get("tests") and py_modules
        else list(skeleton.get("tests") or [])
    )
    if tests_enabled and code_map and not smoke_errors:
        t0 = time.monotonic()
        runner = StagedTestRunner(
            workspace=ws, test_timeout_sec=float(staged_cfg.get("test_timeout_sec", 120.0)),
            debug_logger=state.debug_logger,
        )
        attempts: dict[str, int] = {}

        def _record_gate(task_id: str, gate, *, kind: str) -> None:
            # 実行ごとに tests/_runs と manifest へ (以前は debug JSONL にしか残らなかった、f_10 §11.1-4)
            attempts[task_id] = attempts.get(task_id, 0) + 1
            try:
                ws.record_test_result(StageTestResult(
                    task_id=task_id, passed=gate.ok, failed_count=failed_count(gate),
                    attempt=attempts[task_id],
                    summary=gate.error or gate.skip_reason or ("passed" if gate.ok else ""),
                    output_tail=(gate.stdout_tail or gate.stderr_tail or "")[-2000:],
                    ran_at=time.time(), kind="unchecked" if gate.skipped else kind,
                ))
            except Exception as exc:  # noqa: BLE001 - 記録の失敗で制作は止めない
                logger.warning("staged v2: test result record failed (%s): %s", task_id, exc)
            _emit_check(event_log, "test_gate", {
                "type": "task_result", "task_id": task_id, "attempt": attempts[task_id],
                "status": "done" if gate.ok else "skipped" if gate.skipped else "failed",
                "detail": gate.error or gate.skip_reason or "",
                "failures": failure_lines(gate),
            })

        cases = usable_examples(skeleton, list(code_map))
        example_src = build_example_tests(cases)
        if example_src:
            ws.write_file("test_examples.py", example_src, kind="test", stage="test", task_id="test_examples")
            gate = await asyncio.to_thread(runner.run, test_logical_path="test_examples.py")
            _record_gate("test_examples", gate, kind="pytest")
            if not gate.ok and not gate.skipped and _remaining() >= _CALL_FLOOR_SEC:
                gate_before = gate
                code_before = dict(code_map)
                # 例 (契約) に反するのはコード側 — 失敗の証拠を渡して対象モジュールを 1 回だけ作り直す
                failing = [m for m in modules if any(c.module == PurePosixPath(m["path"]).stem for c in cases)]
                evidence = (gate.stdout_tail or "")[-3000:]
                repair_jobs = [
                    _module_job(m, _module_instruction(skeleton, m, request=query, brief=brief, locale=locale)
                                + "\n\n" + _REPAIR_TASK.format(
                                    path=m["path"], fence=languages.fence_language(m["path"]),
                                    errors=evidence, previous=code_map.get(m["path"], "")[:12000]))
                    for m in failing if m["path"] in code_map
                ]
                repaired = await _run_jobs(repair_jobs, slots)
                candidates = {
                    m["path"]: content
                    for m, content in zip([m for m in failing if m["path"] in code_map], repaired)
                    if content
                }
                # 作り直した版は smoke を通してから採る (f_10 §11.1-4)。いまの code_map は smoke
                # 合格済みなので、落ちたモジュールは修正前の版に戻す (K05 run2: 構文エラー版が
                # smoke 合格の元コードに取って代わった)
                # smoke が未検査 (外へ書こうとした) になった版も採らない — いまの code_map の smoke は
                # 未検査ではなかったので、採ると「検査済みの不合格」が「未検査」に化ける (独立レビュー)
                static_was_checked = not static_check.is_unchecked
                while candidates:
                    wired_trial, trial_errors, _ = await _check({**code_map, **candidates})
                    if static_was_checked and smoke_unchecked and not trial_errors:
                        trial_errors = [f"{p}: import smoke could not be checked" for p in candidates]
                    if not trial_errors:
                        break
                    blamed = [p for p in candidates if _errors_for(p, trial_errors)] or list(candidates)
                    for path in blamed:
                        logger.warning(
                            "staged v2: the contract repair of %s failed the smoke check (%s); "
                            "keeping the previous version", path, "; ".join(trial_errors[:2]),
                        )
                        candidates.pop(path)
                    verification.append(msg("create.contract_repair_reverted", paths=", ".join(blamed)))
                adopted = False
                if candidates:
                    # smoke に通った組 (配線後) をそのまま採る
                    for path, content in wired_trial.items():
                        if code_map.get(path) != content:
                            code_map[path] = content
                            adopted = True
                            ws.write_file(path, content, kind="src", stage="code", task_id=_task_id(path))
                contract_now = _read_text(ws.path("tests/test_examples.py"))
                gate = await asyncio.to_thread(runner.run, test_logical_path="test_examples.py")
                _record_gate("test_examples", gate, kind="pytest")
                # 契約そのものを作り直した回は件数を比べられない
                comparable = adopted and contract_now == example_src
                # 修正前は不合格 (検査できた) — 修正版が未検査なら「不合格」が「未検査」に化ける
                unchecked_after = comparable and gate.skipped
                worse = comparable and not gate.skipped and failed_count(gate) > failed_count(gate_before)
                if worse or unchecked_after:
                    # 作り直しで悪化した — 修正前の版に戻す (K01: 2 → 4 failed のまま配信された)
                    reverted = [p for p in code_map if p in code_before and code_map[p] != code_before[p]]
                    logger.warning(
                        "staged v2: the contract repair made things worse (%d failed -> %s); "
                        "keeping the previous version of %s",
                        failed_count(gate_before),
                        "not checked" if unchecked_after else f"{failed_count(gate)} failed", reverted,
                    )
                    for path in reverted:
                        code_map[path] = code_before[path]
                        ws.write_file(path, code_before[path], kind="src", stage="code", task_id=_task_id(path))
                    verification.append(
                        msg("create.contract_repair_unchecked", paths=", ".join(reverted)) if unchecked_after
                        else msg(
                            "create.contract_repair_worse", before=failed_count(gate_before),
                            after=failed_count(gate), paths=", ".join(reverted),
                        )
                    )
                    gate = gate_before
                    # 最新の記録を戻した版の結果にする (悪化版のまま残さない)
                    _record_gate("test_examples", gate, kind="pytest")
            if gate.skipped:
                contract_check = CheckOutcome.unchecked(
                    CheckKind.CONTRACT, _unchecked_reason(gate), detail=gate.skip_detail or "", count=len(cases),
                )
            elif gate.ok:
                contract_check = CheckOutcome.passed(CheckKind.CONTRACT, count=len(cases))
            else:
                tasks_failed += 1
                contract_check = CheckOutcome.failed(
                    CheckKind.CONTRACT, count=len(cases), errors=failed_count(gate), failures=failure_lines(gate),
                )
        elif malformed_example_count(skeleton, list(code_map)):
            # 例はあるのに形が崩れていて組めない — 「例なし」と黙らない (2026-09-27 ライブ監査 M7)
            contract_check = CheckOutcome.unchecked(CheckKind.CONTRACT, UncheckedReason.INVALID_EXAMPLES)
        else:
            contract_check = CheckOutcome.unchecked(CheckKind.CONTRACT, UncheckedReason.NO_EXAMPLES)
        checks.append(contract_check)
        verification.append(contract_check.render())
        if contract_check.failures:
            verification.append(contract_check.render_failures())
        if advisory_src:
            linted, dropped = lint_generated_tests(
                advisory_src, allowed_text=query + json.dumps(skeleton, ensure_ascii=False),
                # 本文が __file__ 基準の置き場を持つのに chdir だけで隔離するテストも落とす (R12)
                data_locations=file_based_data_locations(code_map),
            )
            if dropped:
                # 件数だけでは実物で確かめられない — 理由をログへ、lint 前の本文を tests/_dropped/ へ (R8)
                advisory_dropped = list(dropped)
                try:
                    kept_at = ws.record_dropped_test(advisory_path, advisory_src)
                except Exception as exc:  # noqa: BLE001 - 記録の失敗で制作は止めない
                    logger.warning("staged v2: could not keep the dropped advisory test: %s", exc)
                    kept_at = "(not kept)"
                logger.info(
                    "staged v2: lint dropped %d brittle check(s) from the advisory test %s "
                    "(raw output kept at %s): %s",
                    len(dropped), advisory_path, kept_at, "; ".join(dropped),
                )
            if linted:
                ws.write_file(advisory_path, linted, kind="test", stage="test", task_id="test_advisory")
                agate = await asyncio.to_thread(runner.run, test_logical_path=advisory_path)
                _record_gate("test_advisory", agate, kind="advisory")
                if agate.skipped:
                    advisory_check = CheckOutcome.unchecked(
                        CheckKind.ADVISORY, _unchecked_reason(agate), detail=agate.skip_detail or "",
                    )
                elif agate.ok:
                    advisory_check = CheckOutcome.passed(CheckKind.ADVISORY)
                else:
                    advisory_check = CheckOutcome.failed(
                        CheckKind.ADVISORY, errors=failed_count(agate), failures=failure_lines(agate),
                    )
                if requested_tests:
                    # 依頼されたテストは参考テストを依頼の名前で配信する (f_10 §11.1-1 / §11.3)
                    extra_files[requested_tests[0]] = linted
                    if advisory_check.is_failure:
                        notices.append(msg("create.requested_tests_failing", name=requested_tests[0]))
                elif advisory_check.is_failure:
                    # 配信しない参考テストの不合格も本文に 1 行 (SPEC にしか無かった、2026-09-27 ライブ監査 M7)
                    notices.append(
                        msg("create.reference_tests_failing", count=advisory_check.errors)
                        if advisory_check.errors else msg("create.reference_tests_failing_some")
                    )
            elif dropped:
                # 生成したが lint が全件落とした — 「参考テストなし」と書かない (独立レビュー P3-6)
                advisory_check = CheckOutcome.unchecked(
                    CheckKind.ADVISORY, UncheckedReason.LINT_DROPPED, detail=str(len(dropped)),
                )
            else:
                advisory_check = CheckOutcome.unchecked(CheckKind.ADVISORY, UncheckedReason.NO_TESTS)
            checks.append(advisory_check)
            advisory_note = advisory_check.render()
            if advisory_check.is_failure:
                advisory_note += msg("create.check.advisory_warning_only")
            if dropped and linted:
                advisory_note += msg("create.check.advisory_dropped", count=len(dropped))
            verification.append(advisory_note)
            if advisory_check.failures:
                verification.append(advisory_check.render_failures())
        phase_sec["tests"] = round(time.monotonic() - t0, 1)
    if usage_check is not None:
        if code_map != usage_checked_code:
            # 契約テストの作り直しでコードが変わった — 作り直しなしでもう 1 回実行して差し替える
            usage_check, _, _ = await _run_usage_check(code_map)
        if usage_check.is_failure:
            tasks_failed += 1
        checks.append(usage_check)
        verification.append(usage_check.render())
        if usage_check.failures:
            verification.append(usage_check.render_failures())
        _emit_check(event_log, "usage", {
            "type": "task_result", "detail": usage_check.render(),
            "status": "failed" if usage_check.is_failure else "skipped" if usage_check.is_unchecked else "done",
            "failures": list(usage_check.failures),
        })
    if requested_tests and not tests_enabled:
        # テスト工程の無い構成 (Free) — 作っていないことを本文で伝える
        notices.append(msg("create.requested_tests_not_available", names=", ".join(requested_tests)))
    elif requested_tests:
        # 依頼 − 配信の差分を名前ごとに (1 本配ると残りが黙って消えた、2026-09-27 ライブ監査 K04)
        undelivered = [name for name in requested_tests if name not in extra_files]
        if undelivered:
            notices.append(msg("create.requested_tests_not_generated", names=", ".join(undelivered)))
    # 外へ書こうとして検査できなかった回は完了扱いにしない (成功の教師にしない)。本文でも明示する
    # (入出力例なし・外部依存なしの未検査は環境・契約の欠けで、コードの挙動ではないので数えない)
    blocked = [c for c in checks if c.is_unchecked and c.reason is UncheckedReason.SANDBOX_VIOLATION]
    if blocked:
        notices.append(msg("create.checks_not_run", lines="; ".join(c.render() for c in blocked)))

    # ── 5. as-built 文書 ─────────────────────────────────────────
    # 配信形 (パッケージ形はここで相対 import と __main__.py を持つ形にする。テストはパッケージ経由の import へ)
    delivered_code = code_map
    if package:
        delivered_code = _package_code(code_map)
        siblings = {PurePosixPath(p).stem for p in delivered_code}
        for name in [n for n in extra_files if n.endswith(".py")]:
            extra_files[name] = to_package_imports(extra_files[name], siblings, anchor=package.replace("/", "."))
    other_facts = {p: languages.facts(p, c) for p, c in code_map.items() if not p.endswith(".py")}
    refs = languages.references({p: c for p, c in code_map.items() if not p.endswith(".py")})
    spec_md = render_spec(
        skeleton=skeleton, request=query, locale=locale, verification=verification, other_facts=other_facts,
        # SPEC のモジュール一覧は配信形のパス (``units/length.py``)
        code_map={f"{package}/{p}": c for p, c in delivered_code.items()} if package else code_map,
    )
    # flowchart のノードも配信形のパス (辺は平置きの bare import から引く)
    flowchart_md = render_flowchart(
        skeleton=skeleton, locale=locale, references=refs,
        code_map={f"{package}/{p}": c for p, c in code_map.items()} if package else code_map,
    )
    try:
        ws.write_spec(spec_md, task_id="as_built")
        ws.write_flowchart(flowchart_md, task_id="as_built")
    except Exception as exc:  # noqa: BLE001
        logger.warning("staged v2: as-built docs persist failed: %s", exc)
    exit_kind = "cancelled" if _is_cancelled() else "timeout" if _remaining() <= 0 and missing else "done"
    _emit_check(event_log, "finalize", {
        "type": "task_result", "detail": "; ".join(verification),
        "status": "failed" if tasks_failed else "done",
        "phase_sec": phase_sec,
        "checks": [c.to_dict() for c in checks],
        "advisory_dropped": advisory_dropped,
        "unbaked_defaults": unbaked_defaults,
    })
    _finish_run_safely(run_store, event_log, exit_kind, tasks_failed=tasks_failed)
    logger.info(
        "staged v2 finished: run=%s modules=%d failed=%d phases=%s total=%.1fs",
        run_id, len(code_map), tasks_failed, phase_sec, time.monotonic() - t_start,
    )
    yield {"kind": "result", "payload": {
        "exit_kind": exit_kind,
        "notes": {
            "run_id": run_id, "workspace_root": str(ws.root), "pipeline": "staged_v2",
            # パッケージ形はパッケージの親が出力フォルダ、コードは ``package/`` の下 (ハーネスが前置する)
            "output_folder": out_folder, "package": package, "phase_sec": phase_sec,
            "static_issues": static_issues[:20], "smoke_errors": smoke_errors[:20],
            "verification": verification,
            "checks": [c.to_dict() for c in checks],
            "unchecked_checks": unchecked_count(checks),
            "advisory_dropped": advisory_dropped,
            "incomplete_checks": [c.check for c in blocked],
            "rebased_paths": rebased_paths,
            "unbaked_defaults": unbaked_defaults,
            "cache_bypass_retries": int(gen_stats.get("cache_bypass_retries") or 0),
            "notices": notices,
        },
        "run_id": run_id,
        "code_map": delivered_code,
        "extra_files": extra_files,
        "spec_md": spec_md,
        "flowchart_md": flowchart_md,
        "tasks_failed": tasks_failed,
        "runnability_issues": smoke_errors,
        "metrics": {
            "units_total": len(modules), "units_completed": len(code_map),
            "validation_errors": tasks_failed, "content_type": "code", "strategy": "staged_v2",
        },
        "truncated_steps": [],
        "truncated_max_tokens": None,
    }}


__all__ = ["normalize_skeleton", "render_skeleton_context", "run_staged_v2_pipeline"]
