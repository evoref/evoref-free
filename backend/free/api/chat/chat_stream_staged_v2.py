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
import re
import time
from pathlib import PurePosixPath
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

_SKELETON_PROMPT = """\
You design a small software deliverable before any code is written. Return ONLY JSON.

Request (the only specification — design exactly this):
{request}
{reference}
Rules:
- Any background in the system message (facts, memory, prior work) describes earlier and possibly
  unrelated work. Use it only where the request explicitly refers to it; never design a previous
  project instead of the request.
- modules: the source files to write. Use exactly the file names (and the folder) the request names;
  otherwise choose short snake_case names. Write every file in the language the request asks for
  and give it that language's extension (".py" for Python, ".html", ".js", ".ts", ".css", …);
  Python when the request does not say. Do not include test files, README,
  SPEC.md or data files. Prefer the fewest modules that satisfy the request (one file when the
  request names one file).
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

_MODULE_TASK = """\
Write the complete file `{path}`.

It must implement exactly these public components (same names and signatures):
{components}

Rules:
{language_rules}- From other files of this deliverable use only the components the design lists (or the code
  shown below) — do not call methods or attributes they do not define.
- Comments and user-facing messages in {language_name}; keep messages short and do not add
  features the request does not ask for.
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
- Output only the file content.
"""


def _language_name(locale: str) -> str:
    return "Japanese" if str(locale).startswith("ja") else "English"


def _norm_path(raw: str) -> str:
    p = str(raw or "").strip().replace("\\", "/").lstrip("./")
    parts = [x for x in PurePosixPath(p).parts if x not in ("", ".", "..")]
    return "/".join(parts)


#: 第 1 段の対象外 (言語・系統の混在) の結果 (``notes["fallback"]``)。ハーネスが旧経路へ回す (f_10 §12.1)。
UNSUPPORTED_LANGUAGE_FALLBACK = "unsupported_language"


def skeleton_problem(data: dict, query: str) -> str:
    """骨組みを作り直すべき理由 (空・依頼が名指したファイルが 1 つも無い)。問題が無ければ空文字列。

    依頼が ``schema.sql`` のようにファイルを名指しているのに、骨組みのどのモジュールも
    その名前でなければ、依頼ではなく別の何か (ブリーフの過去の作業) を設計している
    (ベンチ q1: SQL の依頼に以前のお題の単位変換ツールを設計した)。
    """
    paths = skeleton_paths(data)
    if not paths:
        return "empty"
    named = {
        PurePosixPath(m.group()).name.lower()
        for m in languages.FILE_NAME_IN_TEXT_RE.finditer(query or "")
        if PurePosixPath(m.group()).suffix.lower() in languages.SUPPORTED_SUFFIXES
    }
    if named and not named & {PurePosixPath(p).name.lower() for p in paths}:
        return "unrelated to the files the request names"
    return ""


def skeleton_paths(data: dict) -> list[str]:
    """骨組みのモジュールのパス (テスト・データファイルは除く)。"""
    out = []
    for m in data.get("modules") or []:
        path = _norm_path(m.get("path", ""))
        suffix = PurePosixPath(path).suffix.lower()
        if not path or PurePosixPath(path).name.startswith("test_") or suffix in languages.DATA_SUFFIXES:
            continue
        out.append(path)
    return out


def normalize_skeleton(data: dict) -> tuple[dict, str]:
    """骨組みを検証・正規化する。戻り値は (正規化済み骨組み, 出力フォルダ)。

    モジュールのパスは作業フォルダでは平置き (``src/<name>.py``) にし、依頼が名指した共通の
    フォルダ (``todo_app/``) は出力先の接頭辞として返す。使えなければ ``({}, "")``。
    """
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
    examples = [
        {**e, "module": _local(e.get("module", "") or "")}
        for e in data.get("examples") or []
    ]
    return {
        "summary": str(data.get("summary") or ""),
        "language": family or "python",
        "modules": flat,
        "entry_module": entry if entry in names else "",
        "usage": str(data.get("usage") or ""),
        "examples": [e for e in examples if e["module"] in names and e["module"].endswith(".py")],
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
        usable_examples,
    )
    from backend.free.generation.direct_codegen import generate_single_file
    from backend.free.loop.staged import RunEventLog, RunRecordStore, WorkspaceManager
    from backend.free.loop.staged.test_runner import StagedTestRunner
    from backend.i18n_helper import get_locale
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
        async def _skeleton(system: str | None, max_tokens: int) -> dict:
            try:
                return await aux_client.generate_json(
                    prompt, system=system, purpose="create_skeleton",
                    max_tokens=max_tokens, temperature=0.2,
                    timeout=max(_CALL_FLOOR_SEC, min(240.0 + len(reference_doc) * 0.05, _remaining())),
                ) or {}
            except Exception as exc:  # noqa: BLE001 - 骨組みが無ければ longform へ倒す
                logger.warning("staged v2: skeleton generation failed: %s", exc)
                return {}

        raw = await _skeleton(brief or None, _SKELETON_MAX_TOKENS)
        problem = skeleton_problem(raw, query)
        if problem and _remaining() > _CALL_FLOOR_SEC * 2:
            # 空 (上限で切れた) / 依頼と無関係 (ブリーフの過去の作業に引きずられた) は 1 回だけ、
            # ブリーフを外し上限を上げて作り直す (2026-09-25 create ベンチ q1 / p1)
            logger.info("staged v2: skeleton %s; retrying once without the brief", problem)
            raw = await _skeleton(None, _SKELETON_RETRY_MAX_TOKENS)
    paths = skeleton_paths(raw)
    family, other = languages.family_of(paths)
    if paths and (family is None or family not in enabled_families):
        # 言語名の無い依頼 (「Web ページを作って」) は骨組みで初めて分かる
        yield {"kind": "result", "payload": _unsupported_result(
            other or [f"family:{family}"], run_id=run_id, workspace_root=str(ws.root),
        )}
        return
    skeleton, folder = normalize_skeleton(raw)
    phase_sec["skeleton"] = round(time.monotonic() - t0, 1)
    if not skeleton:
        logger.info("staged v2: empty skeleton; falling back (%s)", "no aux" if aux_client is None else "no modules")
        yield {"kind": "result", "payload": _fallback_result("empty_skeleton")}
        return

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
                request_timeout=max(_CALL_FLOOR_SEC, _remaining()), id_slot=slot,
            )
            return out.get(module["path"], "")
        return _job

    waves = generation_waves(modules)
    jobs = [
        _module_job(m, _module_instruction(skeleton, m, request=query, brief=brief, locale=locale))
        for m in waves[0]
    ]
    advisory_path = ""
    py_modules = [m for m in modules if m["path"].endswith(".py")]
    advisory = tests_enabled and bool(py_modules)
    if advisory:
        advisory_path = f"test_{PurePosixPath(py_modules[0]['path']).stem}_generated.py"
        shared = (f"{brief}\n\n" if brief else "") + render_skeleton_context(skeleton, query)
        advisory_instruction = f"{shared}{SHARED_CONTEXT_BOUNDARY}" + _ADVISORY_TEST_TASK.format(
            path=advisory_path, example_module=PurePosixPath(py_modules[0]["path"]).stem,
        )

        async def _advisory_job(slot: int) -> str:
            if _remaining() < _CALL_FLOOR_SEC or _is_cancelled():
                return ""
            out = await generate_single_file(
                client, advisory_instruction, advisory_path,
                max_tokens=_ADVISORY_TEST_MAX_TOKENS,
                request_timeout=max(_CALL_FLOOR_SEC, _remaining()), id_slot=slot,
            )
            return out.get(advisory_path, "")
        jobs.append(_advisory_job)
    results = await _run_jobs(jobs, slots)
    for m, content in zip(waves[0], results[: len(waves[0])]):
        if content:
            code_map[m["path"]] = content
    advisory_src = results[len(waves[0])] if advisory else ""
    for wave in waves[1:]:
        wave_results = await _run_jobs([
            _module_job(m, _module_instruction(
                skeleton, m, request=query, brief=brief, locale=locale, written=code_map,
            ))
            for m in wave
        ], slots)
        for m, content in zip(wave, wave_results):
            if content:
                code_map[m["path"]] = content
    phase_sec["generate"] = round(time.monotonic() - t0, 1)

    # ── 3. 配線 → smoke → 1 回だけ作り直す ───────────────────────
    t0 = time.monotonic()
    internal = frozenset(
        [PurePosixPath(m["path"]).stem for m in py_modules]
        + [c.get("signature", "").split("(")[0].replace("def ", "").replace("class ", "").split(".")[-1].strip()
           for m in py_modules for c in m.get("components") or []]
    )
    smoke_timeout = float(staged_cfg.get("test_timeout_sec", 120.0))

    async def _check(cmap: dict[str, str]) -> tuple[dict[str, str], list[str], list[str]]:
        # 循環の検出は配線の後 (``from .cli import`` / ``from todo_app.cli import`` が
        # 兄弟の bare import に揃ってから見る)
        wired, issues, _ = _staged_postprocess(cmap)
        wired, cycle_errors = break_undeclared_cycles(wired, skeleton)
        errors = await _staged_import_smoke(wired, smoke_timeout, internal_names=internal)
        if cycle_errors:
            # 循環の import エラーは反対側の名前で出るので、原因側の指摘に差し替える
            errors = cycle_errors + [
                e for e in errors if "cannot import name" not in e and "partially initialized" not in e
            ]
        # Python 以外: 構文・参照の整合・実行環境による構文検査 (f_10 §12.3 / §12.4)
        errors += await run_in_executor_with_context(
            asyncio.get_running_loop(), None,
            lambda: languages.check(wired, cfg=cfg, skeleton=skeleton, query=query),
        )
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
            "detail": f"検査のエラーを修正中… ({', '.join(m['path'] for m in failing)})",
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
            if content:
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
    if missing:
        verification.append(f"未生成: {', '.join(missing)}")
    verification.append(
        "import / 静的検査: " + ("合格" if not smoke_errors else f"エラー {len(smoke_errors)} 件")
    )
    _kind, payload = _emit_check(event_log, "smoke", {
        "type": "task_result",
        "detail": "smoke 検査: " + ("合格" if not smoke_errors else "; ".join(smoke_errors[:3])),
        "status": "done" if not smoke_errors else "failed",
    })
    yield {"kind": "step", "payload": payload}

    # ── 4. (Pro) 契約テスト ──────────────────────────────────────
    advisory_note = ""
    if tests_enabled and code_map and not smoke_errors:
        t0 = time.monotonic()
        runner = StagedTestRunner(
            workspace=ws, test_timeout_sec=float(staged_cfg.get("test_timeout_sec", 120.0)),
            debug_logger=state.debug_logger,
        )
        cases = usable_examples(skeleton, list(code_map))
        example_src = build_example_tests(cases)
        if example_src:
            ws.write_file("test_examples.py", example_src, kind="test", stage="test", task_id="test_examples")
            gate = await asyncio.to_thread(runner.run, test_logical_path="test_examples.py")
            if not gate.ok and not gate.skipped and _remaining() >= _CALL_FLOOR_SEC:
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
                for m, content in zip([m for m in failing if m["path"] in code_map], repaired):
                    if content:
                        code_map[m["path"]] = content
                        ws.write_file(m["path"], content, kind="src", stage="code",
                                      task_id=_task_id(m["path"]))
                gate = await asyncio.to_thread(runner.run, test_logical_path="test_examples.py")
            if gate.skipped:
                verification.append(f"契約テスト: 実行できず ({gate.skip_reason})")
            elif gate.ok:
                verification.append(f"契約テスト (入出力例 {len(cases)} 件): 合格")
            else:
                tasks_failed += 1
                verification.append(f"契約テスト (入出力例 {len(cases)} 件): 不合格")
        else:
            verification.append("契約テスト: 入出力例なし")
        if advisory_src:
            linted, dropped = lint_generated_tests(
                advisory_src, allowed_text=query + json.dumps(skeleton, ensure_ascii=False),
            )
            if linted:
                ws.write_file(advisory_path, linted, kind="test", stage="test", task_id="test_advisory")
                agate = await asyncio.to_thread(runner.run, test_logical_path=advisory_path)
                advisory_note = (
                    "参考テスト: 合格" if agate.ok
                    else "参考テスト: 実行できず" if agate.skipped
                    else f"参考テスト: 一部不合格 ({agate.error or ''}) — 警告のみ"
                )
            else:
                advisory_note = "参考テスト: なし"
            if dropped:
                advisory_note += f" (壊れやすい検査を {len(dropped)} 件除外)"
            verification.append(advisory_note)
        phase_sec["tests"] = round(time.monotonic() - t0, 1)

    # ── 5. as-built 文書 ─────────────────────────────────────────
    other_facts = {p: languages.facts(p, c) for p, c in code_map.items() if not p.endswith(".py")}
    refs = languages.references({p: c for p, c in code_map.items() if not p.endswith(".py")})
    spec_md = render_spec(
        skeleton=skeleton, code_map=code_map, request=query, locale=locale,
        verification=verification, other_facts=other_facts,
    )
    flowchart_md = render_flowchart(skeleton=skeleton, code_map=code_map, locale=locale, references=refs)
    try:
        ws.write_spec(spec_md, task_id="as_built")
        ws.write_flowchart(flowchart_md, task_id="as_built")
    except Exception as exc:  # noqa: BLE001
        logger.warning("staged v2: as-built docs persist failed: %s", exc)
    exit_kind = "cancelled" if _is_cancelled() else "timeout" if _remaining() <= 0 and missing else "done"
    _emit_check(event_log, "finalize", {
        "type": "task_result", "detail": "; ".join(verification), "status": "done",
        "phase_sec": phase_sec,
    })
    _finish_run_safely(run_store, event_log, exit_kind)
    logger.info(
        "staged v2 finished: run=%s modules=%d failed=%d phases=%s total=%.1fs",
        run_id, len(code_map), tasks_failed, phase_sec, time.monotonic() - t_start,
    )
    yield {"kind": "result", "payload": {
        "exit_kind": exit_kind,
        "notes": {
            "run_id": run_id, "workspace_root": str(ws.root), "pipeline": "staged_v2",
            "output_folder": folder, "phase_sec": phase_sec,
            "static_issues": static_issues[:20], "smoke_errors": smoke_errors[:20],
            "verification": verification,
        },
        "run_id": run_id,
        "code_map": code_map,
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
