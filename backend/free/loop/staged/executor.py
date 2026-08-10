"""staged クリエイトの stage 対応 TaskExecutor。

``LoopDriver`` から 1 タスクずつ呼ばれ、``task.stage`` で分岐する:

- ``spec`` : アシストモデルで設計仕様 (spec.md) を生成し workspace に永続化。
- ``code`` : **base クリエイトモデル** (注入された codegen 委譲) で当該モジュール
  を単発生成。spec.md と既生成ファイル一覧を読み込んで instruction に埋め込み、
  1 ファイル = 1 回の base 呼び出しで完結させる (``LongFormOrchestrator`` 経由の
  plan/CodeSpec 再合成は経由しない。再合成は instruction の大半を lossy に圧縮し
  spec/flowchart が実コードへ反映されない原因になるため)。
- ``test`` : テストを生成 → ワークスペース限定 pytest 実行 → 失敗時リペアパス。

base コードゲンは ``CodegenDelegate`` (callable, ``(instruction, file_path) ->
{path: code}``) として DI 注入し、loop pillar が EvorefGen の具象を import
しないようにする。

SemMem への task.status / progress_marker / failure_pattern 書込は ``LoopDriver``
が ``ExecutionOutcome`` を見て一元管理する (本 executor は outcome を返すだけ)。
"""

from __future__ import annotations

import ast
import asyncio
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.core.dependency_constraint import (
    find_third_party_imports,
    requires_stdlib_only,
)
from backend.free.loop.executor import ArtifactEntry, ExecutionOutcome
from backend.free.loop.quality_gate import GateResult, QualityGateOutcome
from backend.free.loop.staged.flow_render import (
    MAX_FLOW_STEPS,
    FlowStep,
    assemble_flow_parts,
    fallback_flow,
    prefix_unit_steps,
    render_flow_section,
    render_mermaid,
    renumber_steps,
    repair_unreachable_tail,
    steps_from_payload,
    validate_flow,
)
from backend.free.loop.staged.spec_contract import (
    contract_drift_reason,
    declared_definition_names,
    extract_behavior_steps,
    normalized_text,
    parse_declared_contract,
    signature_snippets,
)
from backend.free.loop.staged.spec_parts import (
    ComponentGroup,
    FilePartsPlan,
    canonical_module_list,
    component_heading_names,
    count_component_headings,
    count_h2_headings,
    ensure_module_sections,
    extract_entry_point_section,
    extract_module_section,
    foreign_module_headings,
    merge_foreign_module_sections,
    module_entries_from_list,
    module_paths_from_list,
    plan_file_parts,
    replace_entry_point_section,
    replace_flow_section,
    replace_module_section,
)
from backend.free.loop.staged.synthesizer import MODULE_LIST_MARKER, os_constraint
from backend.free.loop.staged.test_runner import StagedTestRunner
from backend.free.loop.staged.workspace import WorkspaceManager, StageTestResult
from backend.i18n_helper import prose_language_name
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.loop.driver import TaskFactView
    from backend.free.loop.events import LoopEventBus

logger = get_logger("loop.staged.executor")

# (instruction, file_path) -> {logical_path: code}。base クリエイトモデル経由の
# コード生成委譲。file_path は生成対象の論理パス (戻り値の主キー) を明示する。
CodegenDelegate = Callable[[str, str], Awaitable[dict[str, str]]]

_SPEC_PROMPT = """\
Write a detailed software design specification in Markdown for the program below.

Structure the document EXACTLY as follows — the heading syntax is machine-parsed, \
so reproduce the heading forms verbatim:

## Overview
What the program does, the overall architecture, and how the modules interact \
(3-6 sentences).

## Shared data structures
Every data structure used by more than one module. For each: its name, the EXACT \
Python type of every field at runtime, and one concrete example value.

Then write ONE section per module. If the program design below lists specific \
file paths, use those EXACT paths verbatim in the headings — do not rename, \
merge, split, or add files:

## Module: <file_path>
One short paragraph: this module's responsibility and what it imports from \
sibling modules. Then one subsection per public class or top-level function \
(one class = one component; small related helper functions may share one \
component; the `if __name__ == "__main__"` entry belongs to its own final \
component):

### Component: <ClassName or function_name>
- Signature: every definition as a complete Python `def`/`class` line in \
backticks, ONE definition per line (for a class: its `__init__` and each \
public method's full signature).
- Attributes: (classes only) every instance attribute as `name: type` — one \
per line with a short purpose.
- Responsibility: what it does, in 1-3 sentences.
- Behavior: for EACH public method/function, 2-4 numbered steps describing \
exactly what it does, including the exact return value and every branch/error \
condition.
- Constants: every concrete constant this component fixes (sizes, key \
bindings, rates, limits, symbols) with its exact literal value.
- Inputs/Outputs: one concrete example of arguments in and the value returned.
- Errors: which exceptions it raises or handles, when, and what happens to \
control flow (propagates / caught by which component / terminates).
- Invariants: conditions that must always hold (state, value ranges, ordering).

## Entry point
Which module/function starts the program and the exact startup call sequence \
as a numbered list.

Be decisive and unambiguous. For every concrete value, parameter, or behavior, \
commit to ONE exact value/parameter name/type/default — never present it as an \
example or an open alternative left to the implementer's discretion (do not \
hedge with "e.g.", "such as", "or similar" for anything the implementation must \
actually produce). When you describe a data structure's element values, state \
the precise Python type they must hold at runtime, and call out any type that \
could easily be substituted for a similar-looking one (for instance, a field \
described as holding integers must say so explicitly enough that it is not \
implemented with booleans instead). Output ONLY the Markdown.

Program design:
{description}
"""

_FLOW_SPEC_PROMPT = """\
You are a software architect. Extract the program's runtime control flow from \
the design specification below as JSON: {{"steps": [...]}}.

Each step is {{"id": "S<n>", "module": "<file path or empty>", "label": \
"<what happens, 12 words max>", "kind": "start"|"process"|"decision"|"error"|\
"end", "next": [{{"to": "S<m>", "condition": "<branch label, empty if \
unconditional>"}}]}}.

Hard rules — the JSON is validated mechanically, so obey ALL of them:
- ids are S1, S2, ... numbered in execution order, each used exactly once.
- Exactly ONE step has kind "start" (the program entry from "## Entry point"); \
every path ends in a step with an empty "next" list (kind "end" or "error").
- "module" is copied VERBATIM from the canonical file list below ("" is \
allowed only for start/end steps). EVERY canonical file must own at least one \
step.
- Follow the specification's stated order of operations EXACTLY — never \
reorder steps.
- A "decision" step has 2 or more "next" entries, each with a distinct \
non-empty "condition" (e.g. "yes" / "no" / an exception class name).
- EVERY exception class named in the specification's Errors bullets that \
changes control flow MUST appear as the "condition" of a branch leading to a \
kind "error" step or to the step that handles it.
- If the specification describes a main loop with multiple distinct phases \
(input handling, state update, logic/collision checks, rendering, timing), \
decompose each described phase into a SEPARATE step — do not force \
decomposition beyond what the specification actually describes.
- Create one "decision" step for each branch condition the specification's \
Behavior steps and Errors bullets declare.
- Aim for {step_lo}-{step_hi} steps for a multi-phase program; 3 to \
{max_steps} steps total. Output ONLY the JSON object.

Example shape (structure only — every path reaches an end/error step, the \
decision has 2 distinct non-empty conditions, start has exactly one outgoing \
edge; do NOT copy its content, derive real steps from the specification below):
{{"steps": [
{{"id": "S1", "module": "", "label": "Program starts", "kind": "start", \
"next": [{{"to": "S2", "condition": ""}}]}},
{{"id": "S2", "module": "main.py", "label": "Load input", "kind": "process", \
"next": [{{"to": "S3", "condition": ""}}]}},
{{"id": "S3", "module": "main.py", "label": "Validate input", \
"kind": "decision", "next": [{{"to": "S4", "condition": "invalid"}}, \
{{"to": "S5", "condition": "valid"}}]}},
{{"id": "S4", "module": "main.py", "label": "Raise ValueError", \
"kind": "error", "next": []}},
{{"id": "S5", "module": "main.py", "label": "Print result", "kind": "end", \
"next": []}}
]}}

Canonical file list:
{module_list}

Design specification:
{spec}
"""

# フロー合成入力の合計予算。確定後 spec (深化 + モジュール節補完 + Canonical
# 追記済み) を渡し、超過時は Entry point 節を優先注入して先頭から切詰める。
# 深化パスで spec が伸びるため 12000 (旧 8000 では後方モジュール節が切れる)。
_FLOW_CONTEXT_CHARS = 12000
# FlowSpec JSON (steps ≈ 10-25 個の詳細フロー) の出力予算。切断は
# telemetry.truncated で全棄却されるため、詳細化要求に見合う 3072 を確保する
# (旧 1536 のままステップ分解を要求すると切断→2 回失敗→線形 fallback で
# かえって粗くなる逆効果経路がある)。
_FLOW_MAX_TOKENS = 3072


def _flow_step_count_hint(total_behavior: int, max_steps: int) -> tuple[int, int]:
    """spec の Behavior ステップ総数から FlowSpec のステップ数目安を右サイズ化する。

    固定 "10-25 steps" を単一モジュール等の小規模仕様にも一律要求すると、
    実際の複雑度に対して過大なグラフを一発合成させることになり、大域整合
    (到達可能性等) の検証不合格率を押し上げる (2026-07-07 live: 単一モジュール
    課題で 27-28 ステップ級のグラフを要求し、末尾ブロックの孤立が頻発した)。
    Behavior ステップ数が取得できない spec は従来通り (10, 25) を返す (非回帰)。
    """
    if total_behavior <= 0:
        return (10, 25)
    lo = max(3, min(total_behavior, 8))
    hi = min(max_steps - 2, max(total_behavior + 5, lo + 3))
    return (lo, hi)


# flow_spec_synthesis が (架橋修復込みで) 2 回とも検証不合格の場合のエスカ
# レーション向けプロンプト。Component/モジュール単位の小規模サブグラフのみ
# を合成させる (f_10 §9 の禁則1(a) 準拠: 該当ユニットの spec テキストは無改変
# で渡す)。ステップ数目安は右サイズ化の下限寄りに固定 (小規模サブグラフ)。
_FLOW_PART_STEP_LO = 2
_FLOW_PART_STEP_HI = 8

_FLOW_PART_PROMPT = """\
You are extracting ONE phase of a larger program's runtime control flow as \
JSON: {{"steps": [...]}}. This subgraph will be mechanically stitched \
together with other phases by another program afterward — follow this \
contract exactly.

Each step is {{"id": "S<n>", "module": "<file path>", "label": \
"<what happens, 12 words max>", "kind": "process"|"decision"|"error", \
"next": [{{"to": "S<m>", "condition": "<branch label, empty if \
unconditional>"}}]}}.

Hard rules — the JSON is validated mechanically, so obey ALL of them:
- ids are S1, S2, ... numbered in execution order WITHIN THIS PHASE ONLY, \
each used exactly once.
- "module" is copied VERBATIM as "{module_path}" for every step.
- The FIRST step in your "steps" array is this phase's entry point (control \
arrives here from the previous phase — do not write a "start" step).
- Do NOT use kind "start" or "end" — those are reserved for the assembler's \
synthetic entry/exit. Use kind "error" ONLY for a branch that terminates \
the ENTIRE program early (an unrecoverable exception); leave its "next" \
empty in that case.
- A step whose "next" is left empty (and is not kind "error") means \
"control continues to the next phase" — the assembler connects it \
automatically. Use this for the natural end of this phase's work.
- A "decision" step has 2 or more "next" entries, each with a distinct \
non-empty "condition" (e.g. "yes" / "no" / an exception class name).
- EVERY exception class named in this phase's Errors bullets that changes \
control flow MUST appear as the "condition" of a branch leading to a kind \
"error" step or the step that handles it.
- Aim for {step_lo}-{step_hi} steps for this phase. Output ONLY the JSON \
object.

Design specification for this phase:
{spec}
"""

_SPEC_DEEPEN_PROMPT = """\
Below is one module section of a software design specification, together with \
the document's Overview and shared data structures for context. Rewrite the \
COMPLETE module section, ADDING implementation-level detail so a developer \
can code it without making any design decision of their own:

- For each class: the full `__init__` signature, EVERY instance attribute as \
`name: type` with a short purpose, and for EACH public method 2-4 numbered \
Behavior steps (exact return value, every branch/error condition).
- For each function: 2-4 numbered Behavior steps (the algorithm) and its \
edge cases.
- Every concrete constant (sizes, key bindings, rates, limits, symbols) with \
its exact literal value, consistent with the shared data structures.

HARD RULES:
- Do NOT change any already-declared signature, constant value, type, or \
Component name. Do NOT add or remove `### Component:` subsections (only when \
the section is an empty placeholder may you introduce its components). Only \
ADD detail.
- Keep the exact heading `## Module: {file_path}` and every existing \
`### Component:` heading verbatim.
- Your response MUST start DIRECTLY with the `## Module: {file_path}` \
heading and contain ONLY that one module section. Do NOT restate, copy, or \
summarize the reference context below — it is given only so you can check \
consistency against it, not to be repeated. Do NOT output `## Overview`, \
`## Shared data structures`, `## Entry point`, or any other section.

>>> REFERENCE CONTEXT (for consistency checks only — do NOT copy or restate) >>>
{context}

>>> MODULE SECTION TO REWRITE (return this one section, expanded, and nothing else) >>>
{section}
"""

# 深化節の絶対上限と、プレースホルダ節でも実体化できる成長フロア。
_DEEPEN_SECTION_MAX_CHARS = 12000
_DEEPEN_GROWTH_FLOOR_CHARS = 4000
_SPEC_DEEPEN_MAX_TOKENS = 3072
# 深化入力に同梱する文書コンテキスト (Overview + Shared data structures) の上限。
_DEEPEN_CONTEXT_CHARS = 4000


def _fallback_flow_labels(
    spec: str, module_list: str,
) -> list[tuple[str, list[str]]]:
    """決定論フォールバックフロー用の (path, ラベル列) を組み立てる。

    各モジュールの Behavior 番号ステップ (:func:`extract_behavior_steps`) が
    取得できればそれを使い、フェーズ分解された多段フローを LLM 呼出なしで
    再構成する。取得できない節は module_list の purpose 先頭行 1 個へ
    縮退する (旧実装相当。2026-07-07 live: FlowSpec 合成が 2 回とも検証不能
    だった単一モジュール課題で、旧フォールバックは「モジュール概要を丸ごと
    1 行に圧縮した 3 step」まで粗くなっていた)。
    """
    out: list[tuple[str, list[str]]] = []
    for path, head in module_entries_from_list(module_list):
        found = extract_module_section(spec, path)
        labels = extract_behavior_steps(found[2]) if found is not None else []
        out.append((path, labels or [head or path]))
    return out


def _condensed_flow_context(spec: str, module_paths: list[str]) -> str:
    """予算超過 spec のフロー合成入力を全モジュール公平に圧縮する。

    先頭 (Overview + Shared data structures) + 各モジュール節の均等割り +
    Entry point 節を決定論結合する。深化後 spec は _FLOW_CONTEXT_CHARS を
    超えうるため、頭からの単純切詰めだと後方モジュールの Behavior/Errors が
    フロー合成に届かず、そのモジュールの分岐/例外がフローチャートから黙って
    欠落する。
    """
    entry = extract_entry_point_section(spec)
    idx = spec.find("## Module:")
    head = (spec[:idx] if idx >= 0 else spec).strip()[:2500]
    n = max(len(module_paths), 1)
    per_module = max(
        (_FLOW_CONTEXT_CHARS - len(head) - len(entry)) // n, 1200,
    )
    parts = [head]
    for path in module_paths:
        found = extract_module_section(spec, path)
        if found is not None:
            parts.append(found[2].strip()[:per_module])
    if entry:
        parts.append(entry.strip())
    return "\n\n".join(p for p in parts if p)


def _condensed_code_spec(spec: str, source_path: str) -> str:
    """予算超過 spec の単発コード生成 instruction 入力を圧縮する。

    生成対象 (``source_path``) 自身の ``## Module:`` 節は BINDING CONTRACT
    そのものであり、絶対に切り詰めない。他モジュールの節は cross-reference
    用の参考情報として per-module cap で圧縮する
    (:func:`_condensed_flow_context` と同じ全モジュール公平圧縮の思想)。
    ``## Canonical file list`` 以降 (Canonical file list + Processing flow、
    末尾に決定論追記される軽量な節) は常に全文保持する。
    """
    first_idx = spec.find("## Module:")
    if first_idx < 0:
        return spec[:_CODE_SPEC_MAX_CHARS]
    head = spec[:first_idx].strip()[:_CODE_SPEC_HEAD_MAX_CHARS]

    tail_idx = spec.find("## Canonical file list")
    modules_region = spec[first_idx:tail_idx if tail_idx >= 0 else len(spec)]
    tail = spec[tail_idx:].strip() if tail_idx >= 0 else ""

    target_heading = f"## Module: {source_path}"
    parts = [head]
    for chunk in re.split(r"\n(?=## Module: )", modules_region):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(
            chunk if chunk.startswith(target_heading)
            else chunk[:_CODE_SPEC_OTHER_MODULE_MAX_CHARS]
        )
    if tail:
        parts.append(tail)
    return "\n\n".join(p for p in parts if p)


def _missing_declared_names(group_spec: str, part_code: str) -> list[str]:
    """部分 spec が宣言する def/class 名のうち part コードに無いものを返す。

    per-part 準拠チェック (f_10 §4.2): smoke は「結合後に存在しない component」
    を構造的に検出できないため、部分生成の再試行ループで宣言名の存在を確認
    する。parse-or-skip — 宣言はパース成功した Signature 由来のみ、part が
    構文エラーなら判定不能として空を返す (結合後の ast ゲートが扱う)。
    """
    declared = declared_definition_names(parse_declared_contract(group_spec))
    if not declared:
        return []
    try:
        tree = ast.parse(part_code)
    except SyntaxError:
        return []
    present = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return [d for d in declared if d not in present]


def _undeclared_part_names(
    coherence_checker: "Callable[[dict[str, str]], list[str]] | None",
    assembler: "Callable[..., str | None] | None",
    parts: list[str],
    code: str,
    source_path: str,
) -> list[str]:
    """宣言済みチェックを通過した part に spec 未宣言の幻覚識別子が無いか確認する。

    ``_missing_declared_names`` は spec に宣言された component の実装漏れしか
    検知できず、LLM が自発的に持ち込んだ未宣言の幻覚名 (import も定義も無い
    bare identifier) は原理的に対象外になる。同一ファイル内のみを対象にした
    ``check_coherence`` プレビューで補完し、安価な part 単位リトライに前倒しする
    (finalize 直前の advisory チェックはタイムアウトで飛ばされ得るため)。

    クロスファイル参照との誤検知を避けるため、対象は生成中ファイル単体の
    結合プレビューに限定する (import 文がある名前は check_coherence 側の
    束縛名収集で解決済み扱いになるため、他ファイルの実在有無は見ない)。
    assembler/checker 未注入 or 結合失敗時は判定不能として空を返す。
    """
    if coherence_checker is None or assembler is None:
        return []
    try:
        preview = assembler([*parts, code], module_stem=Path(source_path).stem)
    except Exception:
        return []
    if not preview or not preview.strip():
        return []
    try:
        return coherence_checker({source_path: preview})
    except Exception:
        return []


def _deepen_context(spec: str) -> str:
    """深化入力に同梱する文書コンテキスト (先頭〜最初の Module 節の手前)。

    Overview と Shared data structures を必ず同梱し、深化節が定数・型を
    宣言済みの値と整合させる根拠を与える (矛盾書き換えは決定論検出不能な
    残余リスクなので、根拠の同梱 + プロンプト禁止文言で緩和する)。
    spec が Module 節から始まる場合 (Overview 欠落) はコンテキスト無し
    (他モジュール節を根拠として与えない)。
    """
    idx = spec.find("## Module:")
    head = spec[:idx] if idx >= 0 else spec
    return head.strip()[:_DEEPEN_CONTEXT_CHARS]


def _extract_deepened_section(raw: str, module_path: str) -> str:
    """深化応答から ``## Module:`` 節以降を決定論的に切り出す (前置き汚染の除去)。

    小型モデルは同梱した参照コンテキストやプロンプト自身の枠組みラベルを
    そのまま複写して応答冒頭に置くことがある (2026-07-06/07 live: 応答が
    "## Overview" や "## Document context" から始まった)。単純な startswith
    判定での prepend は、実際の見出しが複写された前置きの後方に埋もれていた
    場合に見出しを二重化させ ``extra_h2_heading`` を招くため、応答中に実際の
    ``## Module:`` 見出しが出現していればその位置から末尾までを節本体とみなす
    (前置きの複写・説明文を切り捨てる)。見出しが一切無ければ応答全体を節本体と
    みなし先頭に補う (フェイルセーフ、最終判定は :func:`_deepen_reject_reason`)。
    """
    idx = raw.find("## Module:")
    if idx > 0:
        return raw[idx:]
    if idx == 0:
        return raw
    return f"## Module: {module_path}\n{raw}"


def _deepen_reject_reason(
    original: str, deepened: str, module_path: str,
    taken_names: frozenset[str] = frozenset(),
) -> str:
    """深化節の決定論ガード。棄却理由を返す ("" = 合格)。

    深化は best-effort であり、棄却は常に原節維持 (fail-safe)。ガードは
    spec 見直しループの改訂ガードと同じ思想で、構造劣化・他節混入・
    宣言契約の無断変更を弾く。``taken_names`` は他モジュール節の component
    名集合 (プレースホルダ深化が他節と同名の component を発明して準拠ゲート
    と衝突する経路の遮断)。
    """
    if len(deepened) < len(original):
        return "shrunk"
    if len(deepened) > min(
        max(4 * len(original), _DEEPEN_GROWTH_FLOOR_CHARS),
        _DEEPEN_SECTION_MAX_CHARS,
    ):
        return "overgrown"
    if "## Canonical file list" in deepened or "## Processing flow" in deepened:
        return "reserved_section_contamination"
    if foreign_module_headings(deepened, module_path):
        return "foreign_module_heading"
    if count_h2_headings(deepened) != 1:
        return "extra_h2_heading"
    orig_names = component_heading_names(original)
    new_names = component_heading_names(deepened)
    if orig_names and new_names != orig_names:
        # 非プレースホルダ節では component の追加・削除・リネーム・並べ替え・
        # 重複を一切禁止 (多重集合でなく系列比較 — 同名見出しの重複挿入も弾く)。
        return "component_set_changed"
    if not orig_names:
        clash = set(new_names) & set(taken_names)
        if clash:
            # プレースホルダ深化が他モジュール節の component を再発明した
            # (同名クラスの矛盾宣言は準拠ゲートとリペアを空転させる)。
            return f"component_name_collision ({sorted(clash)[0]})"
    normalized = normalized_text(deepened)
    for snippet in signature_snippets(original):
        if snippet not in normalized:
            return f"signature_drift ({snippet[:60]})"
    # substring 保存だけでは「原文を残したまま別シグネチャを追記して契約を
    # 乗っ取る」経路を防げないため、宣言契約そのものの保存も要求する。
    drift = contract_drift_reason(original, deepened)
    if drift:
        return f"contract_drift ({drift})"
    return ""


_SPEC_REVISION_PROMPT = """\
The module `{source_path}` was implemented from the design-spec section below, \
but verification failed. Decide whether the SPEC SECTION ITSELF is wrong or \
ambiguous (missing behavior, contradictory types/signatures, unstated edge \
cases) — as opposed to a code- or test-side mistake.

## Spec section
{section}

## Current `## Entry point` section (global startup description)
{entry_section}

## Current source `{source_path}` (truncated)
```python
{src_head}
```

## Verification failure (kind: {kind})
{evidence}

Return JSON: {{"spec_ok": bool, "reason": str, "revised_section": str, \
"revised_entry_point": str}}. \
If the spec section is at fault, set \
spec_ok=false and put the COMPLETE corrected section in revised_section, \
starting with the exact heading `## Module: {source_path}` and keeping every \
`### Component:` subsection (fix or add the flawed parts; keep correct parts \
verbatim; do NOT include any other module's section or the canonical file \
list). If the global `## Entry point` section above contradicts your corrected \
section (e.g. it still references an approach the correction removed), ALSO \
return that section fully corrected — starting with the exact heading \
`## Entry point` — in revised_entry_point; otherwise leave revised_entry_point \
empty. If the spec is fine, set spec_ok=true and leave both empty.
"""

# spec 見直し judge に渡す evidence / 現ソースの truncate 上限。
_REVISION_EVIDENCE_CHARS = 3000
_REVISION_SRC_HEAD_CHARS = 4000
# 改訂節の決定論ガード: これを超える節は不採用 (spec 全文の巻き込み防止)。
_REVISION_SECTION_MAX_CHARS = 8000
# 随伴改訂される Entry point 節の上限 (起動列は短い列挙のはずで、超過は
# spec 全文の巻き込み兆候として不採用)。
_REVISION_ENTRY_MAX_CHARS = 3000

# 部分生成プロンプトに載せる既生成部分の実コード上限。超過時は AST 公開
# シグネチャ要約へ決定論的に縮退する (LLM 要約は使わない)。
_PART_CONTEXT_MAX_CHARS = 16000

# 単発 (非分割) コード生成 instruction に spec 全文をそのまま埋め込む上限。
# 深化後 spec (module 1 節あたり最大 _DEEPEN_SECTION_MAX_CHARS) は複数モジュール
# 構成で容易にこれを超えうる。超過時は _condensed_code_spec で圧縮する
# (生成対象モジュール自身の節は全文保持、他モジュールは per-module cap)。
_CODE_SPEC_MAX_CHARS = 20000
_CODE_SPEC_HEAD_MAX_CHARS = 3000
_CODE_SPEC_OTHER_MODULE_MAX_CHARS = 2500


# 実行 OS 制約は synthesizer.os_constraint を共有する (planner/spec/code で
# 同一文言を注入し、上流の設計決定と下流の制約が食い違わないようにする)。

# spec.md / flowchart.md の prose 出力言語 (GUI の言語設定 = i18n locale に追従)。
# 見出し (`## Module:` 等) は spec_parts の決定論パーサのアンカーなので英語構文の
# まま固定し、prose のみ対象言語で書かせる。マッピングの実体は共有ヘルパー
# (synthesizer.py と同じ import パターン) に委譲し、独自再実装を避ける。


def _prose_language() -> str:
    """成果物 prose の出力言語名 (生成時点の locale を反映)。"""
    return prose_language_name(english=True)


def _spec_language_constraint() -> str:
    """spec 生成に注入する出力言語制約 (見出し/バレットラベル/識別子は英語のまま)。"""
    return (
        f"\n\nWrite every prose part (overview, responsibilities, descriptions, "
        f"error/invariant notes) in {_prose_language()}. Keep the machine-parsed "
        f"heading forms verbatim in English exactly as specified above "
        f"(`## Overview`, `## Shared data structures`, `## Module: <file_path>`, "
        f"`### Component: <name>`, `## Entry point`), keep the component bullet "
        f"labels verbatim in English (`- Signature:`, `- Attributes:`, "
        f"`- Responsibility:`, `- Behavior:`, `- Constants:`, "
        f"`- Inputs/Outputs:`, `- Errors:`, `- Invariants:` — these are "
        f"machine-parsed anchors), and keep all code identifiers, signatures, "
        f"types, and file paths unchanged."
    )


def _deepen_language_constraint() -> str:
    """spec 深化に注入する出力言語制約。

    ``_spec_language_constraint`` はドキュメント全体の見出し (``## Overview`` 等)
    を列挙するが、深化プロンプトでこれを使うと無関係な見出しの再提示が
    「文書の続きを書く」パターンマッチを誘発しやすい (2026-07-06/07 live:
    応答が Document context をそのまま複写して常に extra_h2_heading で棄却
    された)。深化が実際に出力する見出し・ラベルのみに絞る。
    """
    return (
        f"\n\nWrite every prose part (responsibilities, descriptions, "
        f"error/invariant notes) in {_prose_language()}. Keep the machine-parsed "
        f"heading forms verbatim in English exactly as specified above "
        f"(`## Module: <file_path>`, `### Component: <name>`), keep the "
        f"component bullet labels verbatim in English (`- Signature:`, "
        f"`- Attributes:`, `- Responsibility:`, `- Behavior:`, `- Constants:`, "
        f"`- Inputs/Outputs:`, `- Errors:`, `- Invariants:` — these are "
        f"machine-parsed anchors), and keep all code identifiers, signatures, "
        f"types, and file paths unchanged."
    )


def _flow_language_constraint() -> str:
    """フロー構造合成に注入する出力言語制約 (パス/例外名/識別子は原語のまま)。"""
    return (
        f'\nWrite "label" and "condition" text in {_prose_language()}; keep '
        f"file paths, exception class names, and code identifiers as-is."
    )


def _code_language_constraint() -> str:
    """実コード/テスト生成に注入する出力言語制約 (locale 追従)。

    chat モードの write_file 全経路に注入される
    ``meta_cognitive_utils.content_language_directive()`` と同じ原則
    (コメント/docstring は locale、識別子・シグネチャ・契約名は英語のまま)
    を staged pipeline のコード生成 (_build_code_instruction /
    _build_part_instruction) とテスト生成 (advisory unit test の生成/修正)
    にも適用する。
    """
    return (
        f"\n\nWrite comments and docstrings in {_prose_language()} unless the "
        f"user's request explicitly specifies another language. Keep class "
        f"names, method/function names, signatures, types, imports, string "
        f"literals that are data (e.g. keys, URLs, log format strings), and "
        f"file paths unchanged (as declared in the design specification)."
    )


def _content(resp: dict) -> str:
    """assist generate() の dict 応答から content を取り出す (pillar 内に閉じる)。"""
    try:
        return str(resp["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return ""


def _finish_reason(resp: dict) -> str:
    """assist generate() 応答の finish_reason ('length' は max_tokens 切断を示す)。"""
    try:
        fr = resp["choices"][0].get("finish_reason")
        return fr if isinstance(fr, str) else ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


# spec 本文再生成時の max_tokens 上限 (backend/schemas/create.py の le=8192 と整合)。
_SPEC_MAX_TOKENS_CEILING = 8192


def _extract_module_list(description: str) -> str:
    """spec タスクの description から正準モジュール一覧ブロックを取り出す。

    ``synthesizer.synthesize_create_task_graph`` が ``MODULE_LIST_MARKER`` 以降に
    埋め込む ``file_path`` 群は、決定論的に展開された各 code タスクの
    ``source_path`` と 1:1 で一致する (= 正準)。spec 本文はここから **別の** LLM
    呼び出しで自由記述生成されるため、その出力がこの一覧を忠実に再現する保証は
    ない。呼出側はこの戻り値を spec.md に決定的に追記し、コード生成が参照する
    ファイル一覧を必ず正しいものにする。マーカーが無ければ空文字 (アシスト未接続
    等で modules が無かった場合)。
    """
    idx = description.find(MODULE_LIST_MARKER)
    if idx == -1:
        return ""
    return description[idx + len(MODULE_LIST_MARKER):].strip()


def _anchor_constraint(module_paths: list[str]) -> str:
    """アンカー遵守リトライ用の矯正制約 (必須 ``## Module:`` 見出しを列挙)。

    spec 生成 LLM が見出し規約を丸ごと無視した場合 (2026-07-06 live:
    ``## 1. Purpose`` 等のアドホック見出しで全モジュール節が不成立)、
    プレースホルダ補完では詳細が全損するため、再生成時に明示列挙で矯正する。
    """
    headings = "\n".join(f"`## Module: {p}`" for p in module_paths)
    return (
        "\n\nIMPORTANT: your previous draft did not use the required "
        "machine-parsed heading forms, so it could not be processed. Rewrite "
        "the COMPLETE specification now, reproducing the heading forms "
        "verbatim — the document MUST contain each of these exact H2 "
        "headings:\n" + headings
    )


def _pick_primary(files: dict[str, str], source_path: str) -> str:
    """生成結果 dict から source_path に対応するコードを 1 つ選ぶ。"""
    if source_path in files and files[source_path].strip():
        return files[source_path]
    nonempty = {p: c for p, c in files.items() if c and c.strip()}
    if not nonempty:
        return ""
    if len(nonempty) == 1:
        return next(iter(nonempty.values()))
    base = Path(source_path).name
    for p, c in nonempty.items():
        if Path(p).name == base:
            return c
    # 最長 (最も中身のある) ファイルを採用
    return max(nonempty.values(), key=len)


def _fmt_func_sig(node: "ast.FunctionDef | ast.AsyncFunctionDef") -> str:
    """関数/メソッドの 1 行シグネチャ (本体なし) を返す。"""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"


def _class_api_lines(node: ast.ClassDef) -> list[str]:
    """class の公開メンバ (公開メソッド + __init__ + 注釈付きクラス変数) を要約する。"""
    out: list[str] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name.startswith("_") and item.name != "__init__":
                continue
            out.append(_fmt_func_sig(item))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if not item.target.id.startswith("_"):
                out.append(f"{item.target.id}: {ast.unparse(item.annotation)}")
    return out


def summarize_module_api(code: str) -> str:
    """source コードの公開 API (class/メソッド/関数/注釈定数) を AST から要約する。

    既生成モジュールの実シグネチャを後続モジュール生成プロンプトへ注入し、
    モジュール間で API がドリフトする (クラス名/メソッド名/コンストラクタ引数が
    食い違う) のを防ぐ共有契約として使う。構文エラー (非 Python) 時は空文字。
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return ""
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                lines.append(_fmt_func_sig(node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            bases = (
                "(" + ", ".join(ast.unparse(b) for b in node.bases) + ")"
                if node.bases else ""
            )
            members = _class_api_lines(node)
            if members:
                lines.append(f"class {node.name}{bases}:")
                lines.extend(f"    {m}" for m in members)
            else:
                lines.append(f"class {node.name}{bases}: ...")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                lines.append(f"{node.target.id}: {ast.unparse(node.annotation)}")
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                    lines.append(f"{tgt.id} = ...")
    return "\n".join(lines).strip()


def build_sibling_api_block(
    files: dict[str, str], *, max_chars_per_file: int = 1200,
) -> str:
    """既コミットの兄弟 src 群の公開 API を生成プロンプト用ブロックに整形する。

    各ファイルを「import 名 (= stem) + 公開シグネチャ」として列挙する。
    要約不能 (構文エラー / 空) なファイルは省く。全ファイル省略時は空文字。
    """
    parts: list[str] = []
    for path, code in files.items():
        summary = summarize_module_api(code or "")
        if not summary:
            continue
        if len(summary) > max_chars_per_file:
            summary = summary[:max_chars_per_file] + "\n# ... (truncated)"
        parts.append(
            f"### `{path}` (import as `{Path(path).stem}`)\n"
            f"```python\n{summary}\n```"
        )
    return "\n\n".join(parts)


def _test_logical_for(source_path: str) -> str:
    return f"test_{Path(source_path).stem}.py"


def _is_valid_python(code: str) -> bool:
    """``code`` が構文的に有効な Python か (markdown 等の非コード検出用)。"""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


#: code 工程で ``_salvage_python_code`` の結果を採用する下限 (原文に対する文字比)。
#: 前置き/後書きの自然文はコード本体に比べて短いので高い比率が残る。逆に本物の
#: 構文エラーは、壊れた文そのものを削らないとパースが通らないため大きく縮む。
_SALVAGE_MIN_KEEP_RATIO = 0.9


def _compile_error_detail(code: str) -> str | None:
    """``compile`` が通らないときの位置と理由を 1 行で返す (通れば ``None``)。

    :func:`_is_valid_python` の ``ast.parse`` は構文木を作るだけで、
    ``continue`` / ``break`` の位置のような **コンパイル時にしか検出されない
    誤り**を見逃す (実測 2026-08-07: ループ外の ``continue`` を ``ast.parse`` は
    受理し、import スモークだけが落とした)。成果物として書き出す前の最終判定は
    import スモークと同じ ``compile`` で行う。
    """
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        where = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return f"{where}: {exc.msg}"
    except ValueError as exc:
        return str(exc)
    return None


_PYTEST_USAGE_RE = re.compile(r"\bpytest\.\w+")
_PYTEST_IMPORT_RE = re.compile(
    r"^\s*(import\s+pytest\b|from\s+pytest\s+import\b)", re.MULTILINE,
)


def _ensure_module_import(test_code: str, source_path: str) -> str:
    """``<module>.<attr>`` を使うのに import 忘れの生成テストへ import を補う。

    2026-07-27 live: brackets.py のテストが ``brackets.is_balanced("([)]")`` を
    ``import brackets`` 無しで呼び、収集時の ``NameError`` で失敗した (同型の
    失敗は wordcount.py でも記録済み)。``_ensure_pytest_import`` と同じ方針で、
    構文的に確証できる欠落だけを機械的に補う (既に import 済みなら何もしない)。
    """
    stem = Path(source_path).stem
    if not stem or not stem.isidentifier():
        return test_code
    if not re.search(rf"\b{re.escape(stem)}\.\w", test_code):
        return test_code
    already = re.search(
        rf"^\s*(?:import\s+{re.escape(stem)}\b"
        rf"|from\s+{re.escape(stem)}\s+import\b"
        rf"|import\s+.*\bas\s+{re.escape(stem)}\b)",
        test_code, re.MULTILINE,
    )
    if already:
        return test_code
    return f"import {stem}\n{test_code}"


def _ensure_pytest_import(test_code: str) -> str:
    """``pytest.xxx`` を使うのに import 忘れの生成テストへ決定論的に import を補う。

    2026-07-23 live: bubble_sort のテストが `pytest.raises`/マーカーを使い
    ながら `import pytest` を欠落させ、収集時の `NameError` で 11 件全滅した。
    LLM 再生成に頼らず、構文的に確証できる欠落は機械的に修復する
    (誤検知余地が無い — 既に import 済みなら何もしない)。
    """
    if not _PYTEST_USAGE_RE.search(test_code) or _PYTEST_IMPORT_RE.search(test_code):
        return test_code
    return f"import pytest\n{test_code}"


# repair/test 系 instruction の末尾に付与する出力制約。"Output only ... code." だけの
# 弱い指示だと前置き/後書きの自然文が混入し ast.parse に失敗する (2026-07-06 live:
# 非 Python 応答で repair round が空転)。code_repair.py の _PYTHON_REPAIR_PROMPT と
# 同じ明示的な禁止文言に揃える。
_NO_PROSE_OUTPUT_CONSTRAINT = (
    " Output ONLY the raw source code itself — no explanations, no markdown "
    "code fences, no commentary before or after the code."
)

# 前置き/後書きの自然文救出で試す最大トリム行数 (冒頭・末尾それぞれ)。
_SALVAGE_TRIM_LINES = 5


def _parses_to_nonempty_module(code: str) -> bool:
    """``code`` が構文的に有効 **かつ** コメント/空行だけでない実体を持つか。

    コメント行だけの断片は ``ast.parse`` を通ってしまう (body が空の Module) ため、
    救出候補としては無意味 (元の壊れたコードを空同然の断片へ差し替えてしまう)。
    """
    try:
        return bool(ast.parse(code).body)
    except (SyntaxError, ValueError):
        return False


def _salvage_python_code(code: str) -> str | None:
    """前置き/後書きの自然文が混じった応答から有効な Python 部分を救出する。

    コードフェンス除去 (``remove_code_fences``) はフェンス区切り行のみを除去し、
    その外側に残った説明文 ("Here's the fix:" 等) は保持するため、それだけで
    ``ast.parse`` が失敗するケースがある。冒頭・末尾から少数行ずつ削って
    再パースを試み、有効になった時点のコードを返す。有効にならなければ
    ``None`` (呼出側は従来通り「非 Python」として棄却する)。
    """
    lines = code.splitlines()
    n = len(lines)
    limit = min(_SALVAGE_TRIM_LINES, n)
    for start in range(limit + 1):
        for end in range(n, n - limit - 1, -1):
            if end <= start or (start == 0 and end == n):
                continue
            candidate = "\n".join(lines[start:end]).strip()
            if candidate and _parses_to_nonempty_module(candidate):
                return candidate
    return None


@dataclass
class StagedCreateExecutor:
    """stage 別 TaskExecutor。``LoopDriver(executor=...)`` に差し込む。

    Args:
        workspace: 工程間ハンドオフ用 :class:`WorkspaceManager`。
        assist_client: spec 工程で使うアシスト。``None`` なら spec は description
            をそのまま spec.md に書く degraded 動作。
        codegen: base クリエイトモデル経由の生成委譲
            (instruction, file_path -> {path: code})。
        smoke_runner: 生成 src 群を import スモーク検証する callable
            (``run_import_smoke`` をラップ注入)。``.errors`` / ``.warnings`` を
            duck-typed で参照。``None`` (degraded) ならスモークゲートはスキップ
            (=成功扱い)。**test 工程の合否ゲートはこのスモーク結果で決まる**。
        test_runner: ワークスペース限定 pytest 実行 (advisory ユニットテスト用)。
            ``None`` なら advisory テストは省略 (スモークゲートは別途実施)。
        max_repair_rounds: スモーク実エラー時のコード修正の最大回数。
        spec_max_tokens: spec.md 生成の最大トークン。
    """

    workspace: WorkspaceManager
    assist_client: "AssistModelClient | None"
    codegen: CodegenDelegate
    smoke_runner: "Callable[[dict[str, str]], object] | None" = None
    test_runner: StagedTestRunner | None = None
    # test↔src API 契約チェッカ (src_files, test_files) -> 違反メッセージ列。
    # EvorefGen の check_api_contract を api 層で注入 (loop→gen の越境回避)。
    contract_checker: "Callable[[dict[str, str], dict[str, str]], list[str]] | None" = None
    # 生成テストの決め打ち期待値を pytest 失敗出力の実測値で補正する
    # (test_code, pytest_output) -> (repaired_code, fixed_count)。EvorefGen の
    # repair_literal_assertions を api 層で注入 (loop→gen の越境回避)。
    # ``None`` (未注入) なら補正はスキップされる (従来動作)。
    value_repair: "Callable[[str, str], tuple[str, int]] | None" = None
    max_repair_rounds: int = 2
    max_test_regen_rounds: int = 2
    spec_max_tokens: int = 1536
    spec_timeout_sec: float = 120.0
    flowchart_enabled: bool = True
    # flow_spec_synthesis が (架橋修復込みで) 2 回とも検証不合格の場合の
    # エスカレーション。Component/モジュール単位で小規模サブグラフを部分合成
    # →決定論結合する (part_generation_enabled と同じ段階導入の作法)。
    flow_part_synthesis_enabled: bool = False
    # spec 工程のモジュール節深化 (1 節 1 assist 呼出でメソッド毎挙動・属性・
    # 定数まで書き下す)。ガード棄却時は原節維持の best-effort。
    spec_deepen_enabled: bool = True
    # spec 宣言契約と生成コードの決定論照合を test 工程の smoke gate に合流
    # させる (False = 観測記録のみに縮退する運用弁)。
    spec_conformance_enabled: bool = True
    # 宣言契約 (plain dict 列) と src の照合関数。EvorefGen の
    # check_spec_conformance を api 層で注入 (loop→gen の越境回避)。
    conformance_checker: (
        "Callable[..., list[str]] | None"
    ) = None
    # test 不合格時に spec 該当節を assist で点検・改訂して再生成するサイクルの
    # ワークスペース全体での上限 (0 で無効)。per-task では 1 回まで。
    max_spec_revision_rounds: int = 1
    # 部分ごと生成 (spec の ### Component: 単位で生成→決定論結合)。どちらか
    # 未注入なら無効で、従来の単発生成のみ。part_codegen は部分向け予算
    # (part_max_tokens) の別 delegate インスタンス。part_assembler は
    # EvorefGen の assemble_file_parts を api 層で注入 (loop→gen の越境回避)。
    # smoke 修復 / advisory テスト生成は「ファイル全体」意味論のため常に
    # self.codegen を使う (部分化しない)。
    part_codegen: "CodegenDelegate | None" = None
    # (parts, *, module_stem) -> merged | None。module_stem は自己 import 除去用。
    part_assembler: "Callable[..., str | None] | None" = None
    # 宣言済みコンポーネントチェック (_missing_declared_names) を通過した part
    # に、spec 未宣言の幻覚識別子 (import も定義も無い bare name) が紛れ込んで
    # いないかを補完検知する。EvorefGen の check_coherence を api 層で注入
    # (loop→gen の越境回避、conformance_checker と同パターン)。未注入時は
    # このチェックをスキップする (2026-07-24 実機で GameData/Contact_manager
    # のような自発的な幻覚名が finalize 直前の advisory チェックまで
    # すり抜けタイムアウトで未修正のまま出荷された問題への対策)。
    coherence_checker: "Callable[[dict[str, str]], list[str]] | None" = None
    part_max_parts: int = 4
    event_bus: "LoopEventBus | None" = None
    debug_logger: "DebugLogger | None" = None
    name: str = "staged_create"

    def _emit(self, stage: str, detail: str, status: str, task_id: str) -> None:
        """工程内サブステップ進捗を event_bus へ発行する (null-safe)。"""
        if self.event_bus is None:
            return
        try:
            self.event_bus.emit(
                "stage_progress",  # type: ignore[arg-type]
                iteration=0, project_id=None,
                data={"stage": stage, "detail": detail,
                      "status": status, "task_id": task_id},
            )
        except Exception as exc:
            logger.debug("stage_progress emit failed: %s", exc)

    def _fail_task(self, task: "TaskFactView", error: str) -> None:
        """manifest 上の task を failed に更新する (失敗パスの観測性向上)。"""
        try:
            self.workspace.upsert_task(
                task_id=task.task_id, title=task.title,
                stage=task.stage or "code", status="failed",
                depends_on=task.depends_on, last_error=error,
            )
        except Exception as exc:
            logger.debug("manifest fail upsert failed: %s", exc)

    async def execute(self, task: "TaskFactView") -> ExecutionOutcome:
        self.workspace.upsert_task(
            task_id=task.task_id, title=task.title,
            stage=task.stage or "code", status="in_progress",
            depends_on=task.depends_on,
        )
        match task.stage:
            case "spec":
                return await self._run_spec(task)
            case "code":
                return await self._run_code(task)
            case "test":
                return await self._run_test(task)
            case _:
                return ExecutionOutcome(
                    status="skipped",
                    notes={"executor": self.name, "reason": "no_stage"},
                )

    # ── spec 工程 ─────────────────────────────────────────────────────
    async def _run_spec(self, task: "TaskFactView") -> ExecutionOutcome:
        self._emit("spec", "設計仕様を生成中", "running", task.task_id)
        module_list = _extract_module_list(task.description)
        module_paths = module_paths_from_list(module_list)
        spec_text = await self._generate_spec_doc(task.description)

        # アンカー遵守リトライ: `## Module:` 見出しを 1 つも守れなかった自由記述
        # はプレースホルダ補完だけでは詳細が全損するため、必須見出しを列挙した
        # 矯正制約付きで 1 回だけ再生成する (アンカーが現れた場合のみ採用)。
        anchor_retry = False
        if spec_text and module_paths and not any(
            extract_module_section(spec_text, p) is not None for p in module_paths
        ):
            logger.warning(
                "spec draft has no module anchors; retrying once with a "
                "corrective heading constraint",
            )
            self._emit("spec", "設計仕様を再生成中 (見出し規約違反)",
                       "running", task.task_id)
            retry = await self._generate_spec_doc(
                task.description,
                extra_constraint=_anchor_constraint(module_paths),
            )
            if retry and any(
                extract_module_section(retry, p) is not None for p in module_paths
            ):
                spec_text = retry
                anchor_retry = True
        # 深化パスは LLM 由来の spec に対してのみ意味を持つ (description
        # フォールバック時は assist 劣化中の可能性が高く、深化 N 呼出は
        # ウォールクロックを無成果に消費するだけ)。
        spec_from_llm = bool(spec_text)
        if not spec_text:
            spec_text = task.description
        # `## Processing flow` は flow_render が挿入する予約決定論見出し。LLM
        # 出力に迷い込んだ同名節は剥離し、後段の決定論挿入と二重化させない。
        spec_text = replace_flow_section(spec_text, "")

        final_spec = spec_text
        sections_note = ""
        foreign_merged = 0
        deepen_applied = 0
        deepen_rejected = 0
        if module_list:
            # LLM の自由記述が一部モジュールの `## Module:` 見出しを落としても、
            # 部分生成・spec 見直しループのアンカーが常に存在するよう決定的に補完
            # する (プレースホルダ節。canonical list 追記の前に行う)。
            final_spec, found = ensure_module_sections(final_spec, module_paths)
            sections_note = f"{found}/{len(module_paths)}"
            # 正準に無い幻覚 `## Module:` 節は正準節へ決定論移送する (単一
            # 正準モジュール時のみ)。component が幻覚節へ泣き別れると部分生成が
            # 不発になるため、補完の後・canonical list 追記の前に行う。
            final_spec, foreign_merged = merge_foreign_module_sections(
                final_spec, module_paths,
            )
            # モジュール節の深化 (1 節 1 assist 呼出): メソッド毎の挙動・属性・
            # 定数まで実装水準に書き下す。単一呼出の spec は 4B の注意が文書
            # 全体に薄まり密度が出ないため、節単位の集中プロンプトで賄う
            # (部分生成と同じ確立パターン)。ガード棄却時は原節維持。
            if (
                self.spec_deepen_enabled
                and self.assist_client is not None
                and spec_from_llm
            ):
                context = _deepen_context(final_spec)
                for path in module_paths:
                    found = extract_module_section(final_spec, path)
                    if found is None:
                        continue
                    _, _, section = found
                    if len(section) >= _DEEPEN_SECTION_MAX_CHARS:
                        # 上限超の節は数学的に必ず棄却される (成長不能) ため
                        # assist 呼出自体をスキップする。
                        logger.info(
                            "spec deepen skipped for %s: section already at "
                            "max size", path,
                        )
                        continue
                    taken = frozenset(
                        set(component_heading_names(final_spec))
                        - set(component_heading_names(section)),
                    )
                    self._emit(
                        "spec", f"仕様を詳細化中: {path}", "running",
                        task.task_id,
                    )
                    deepened, abort = await self._deepen_module_section(
                        path, section, context, taken,
                    )
                    if abort:
                        # assist 劣化 (タイムアウト/接続断)。残りモジュールへ
                        # 直列で挑み続けるとウォールクロック予算を無成果に
                        # 食い潰すため深化を打ち切る。
                        deepen_rejected += 1
                        break
                    replaced = (
                        replace_module_section(final_spec, path, deepened)
                        if deepened is not None else None
                    )
                    if replaced is None:
                        deepen_rejected += 1
                        continue
                    final_spec = replaced
                    deepen_applied += 1

                if self.debug_logger is not None:
                    self.debug_logger.log_decision(
                        decision_point="spec_deepen_outcome",
                        chosen=("applied" if deepen_applied > 0 else "rejected_or_skipped"),
                        candidates=["applied", "rejected_or_skipped"],
                        reason=f"applied={deepen_applied} rejected={deepen_rejected}",
                        context={
                            "task_id": task.task_id,
                            "module_count": len(module_paths),
                            "deepen_applied": deepen_applied,
                            "deepen_rejected": deepen_rejected,
                        },
                        scope="loop_iter",
                    )

            # spec_text は task.description とは別の LLM 呼び出しの自由記述であり、
            # 与えたファイル一覧を忠実に再現する保証が無い。code タスクの
            # source_path と必ず一致する正準一覧を決定的に追記し、コード生成が
            # 常に正しいファイル一覧を参照できるようにする。
            final_spec = (
                f"{final_spec}\n\n"
                f"## Canonical file list (authoritative — generated files MUST "
                f"match these exact paths)\n{module_list}\n"
            )

        # フロー合成は確定後 spec (補完 + Canonical 追記済み) を入力に 1 回だけ
        # 行い、flowchart.md の mermaid と spec.md の `## Processing flow` 節を
        # 同一 steps から決定論レンダリングする。両成果物が同一データの派生物に
        # なるため内容相違は構造的に発生しない (f_10 §3)。mermaid コードブロック
        # 自体は spec.md へ埋め込まない (二重埋め込み回避は従来通り)。
        flow_source = "none"
        if self.flowchart_enabled:
            self._emit("spec", "処理フローを合成中", "running", task.task_id)
            try:
                steps, flow_source = await self._synthesize_flow_steps(
                    module_list, final_spec,
                )
                if steps:
                    # 2 種のレンダリングを書込み前に完了させる (途中例外で
                    # 片側だけ書かれた状態を作らない)。
                    mermaid = render_mermaid(steps, module_paths)
                    flow_section = render_flow_section(steps)
                    self.workspace.write_flowchart(mermaid, task_id=task.task_id)
                    final_spec = replace_flow_section(final_spec, flow_section)
            except Exception as exc:
                # レンダラ側の想定外エラー時は両成果物ともフロー無しに保つ
                # (片側だけ残すと相違が生まれる)。
                logger.warning(
                    "flow rendering failed; omitting flow from both "
                    "artifacts: %s", exc,
                )
                flow_source = "error"
        has_flow = flow_source in (
            "llm", "llm_retry", "llm_repaired", "llm_retry_repaired", "parts",
            "fallback",
        )
        if self.flowchart_enabled and self.debug_logger is not None:
            self.debug_logger.log_decision(
                decision_point="flow_spec_synthesis_path",
                chosen=flow_source,
                candidates=[
                    "llm", "llm_retry", "llm_repaired", "llm_retry_repaired",
                    "parts", "fallback", "none", "error",
                ],
                context={"task_id": task.task_id, "module_count": len(module_paths)},
                scope="loop_iter",
            )

        wf = self.workspace.write_spec(final_spec, task_id=task.task_id)
        self.workspace.upsert_task(
            task_id=task.task_id, title=task.title, stage="spec",
            status="done", depends_on=task.depends_on,
        )
        self._emit(
            "spec",
            "設計仕様" + ("・フローチャート" if has_flow else "") + "を確定",
            "done", task.task_id,
        )
        notes = {"executor": self.name, "stage": "spec",
                 "chars": str(len(final_spec)),
                 "flowchart": "true" if has_flow else "false",
                 "flow_source": flow_source}
        if sections_note:
            notes["spec_sections"] = sections_note
        if foreign_merged:
            notes["spec_foreign_merged"] = str(foreign_merged)
        if anchor_retry:
            notes["spec_anchor_retry"] = "true"
        if deepen_applied or deepen_rejected:
            notes["spec_deepen"] = (
                f"{deepen_applied}/{deepen_applied + deepen_rejected}"
            )
        # ExecutionOutcome.notes は成功時 LoopDriver に読まれず manifest にも
        # 保存されないため (失敗時のみ一部キーが参照される)、観測用の値は
        # ここで INFO ログへ明示的に出す (--develop 不要、既定で backend.log
        # に残る。2026-07-07 live: notes 追加時に見落とし、目視でしか裏取り
        # できなかった)。
        logger.info("spec stage notes for %s: %s", task.task_id, notes)
        return ExecutionOutcome(
            status="success",
            artifacts=(_artifact(wf.logical_path, wf.sha256),),
            notes=notes,
        )

    async def _generate_spec_doc(
        self, description: str, extra_constraint: str = "",
    ) -> str:
        """spec 本文をアシスト生成する。max_tokens 切断時のみ予算を倍に広げ 1 回再生成。

        ``generate()`` は finish_reason='length' を telemetry に露出しないため、
        応答 dict から直接 finish_reason を読み本文の途中切れを検知する (検知しないと
        §7 のように文末で切れた spec が後続コード生成の土台になり破損が連鎖する)。
        遅い iGPU を考慮し再生成は切断時のみ・明示 timeout 付きに限定する。
        ``None`` (degraded) / 失敗時は空文字を返し、呼出側が description に倒す。
        ``extra_constraint`` はアンカー遵守リトライの矯正制約 (末尾追記)。
        """
        if self.assist_client is None:
            return ""
        msgs = [{"role": "user", "content":
                 _SPEC_PROMPT.format(description=description) + os_constraint()
                 + _spec_language_constraint() + extra_constraint}]
        try:
            resp = await self.assist_client.generate(
                msgs, purpose="create_spec_doc", max_tokens=self.spec_max_tokens,
                temperature=0.3, timeout=self.spec_timeout_sec,
            )
        except Exception as exc:
            logger.warning("spec doc generation failed: %s", exc)
            return ""
        spec_text = _content(resp).strip()
        if _finish_reason(resp) != "length":
            return spec_text
        retry_tokens = min(self.spec_max_tokens * 2, _SPEC_MAX_TOKENS_CEILING)
        if retry_tokens <= self.spec_max_tokens:
            return spec_text  # 既に上限。再生成しても伸びない
        logger.warning(
            "spec doc truncated at max_tokens=%d; regenerating at %d",
            self.spec_max_tokens, retry_tokens,
        )
        try:
            resp2 = await self.assist_client.generate(
                msgs, purpose="create_spec_doc", max_tokens=retry_tokens,
                temperature=0.3, timeout=self.spec_timeout_sec * 1.5,
            )
        except Exception as exc:
            logger.warning("spec doc regeneration failed: %s", exc)
            return spec_text
        retry_text = _content(resp2).strip()
        # 非切断 or より長い結果のみ採用 (再切断でも初回より短くはしない)。
        if retry_text and (
            _finish_reason(resp2) != "length" or len(retry_text) > len(spec_text)
        ):
            return retry_text
        return spec_text

    async def _deepen_module_section(
        self, module_path: str, section: str, context: str,
        taken_names: frozenset[str] = frozenset(),
    ) -> tuple[str | None, bool]:
        """モジュール節を実装水準の詳細へ書き下した節を返す。

        戻り値 ``(deepened | None, abort)``。parse-or-skip の深化版: 切断
        (`finish_reason=='length'`)・空応答・決定論ガード棄却
        (:func:`_deepen_reject_reason`) は ``(None, False)`` に倒し、呼出側は
        原節を維持する。**例外 (タイムアウト/接続断) は ``(None, True)``** —
        assist 劣化のシグナルであり、残りモジュールへ 600s ずつ直列で挑み
        続けるとウォールクロック予算を無成果に食い潰すため、呼出側は深化
        ループ全体を打ち切る (レビュー確定指摘のサーキットブレーカ)。
        """
        if self.assist_client is None:
            return None, False
        msgs = [{"role": "user", "content":
                 _SPEC_DEEPEN_PROMPT.format(
                     file_path=module_path, context=context, section=section,
                 ) + os_constraint() + _deepen_language_constraint()}]
        try:
            resp = await self.assist_client.generate(
                msgs, purpose="create_spec_deepen",
                max_tokens=_SPEC_DEEPEN_MAX_TOKENS,
                temperature=0.3, timeout=self.spec_timeout_sec,
            )
        except Exception as exc:
            logger.warning(
                "spec deepen failed for %s (aborting remaining modules): %s",
                module_path, exc,
            )
            return None, True
        if _finish_reason(resp) == "length":
            # 半端に切れた節を契約 SSOT にしない (再試行もしない — 深化は
            # best-effort で、原節維持が常に安全側)。
            logger.warning(
                "spec deepen truncated at max_tokens for %s; keeping the "
                "original section", module_path,
            )
            return None, False
        deepened = _content(resp).strip()
        if not deepened:
            return None, False
        deepened = _extract_deepened_section(deepened, module_path)
        reason = _deepen_reject_reason(section, deepened, module_path, taken_names)
        if reason:
            logger.warning(
                "spec deepen rejected for %s: %s", module_path, reason,
            )
            return None, False
        return deepened, False

    async def _synthesize_flow_steps(
        self, module_list: str, spec: str,
    ) -> tuple[list[FlowStep], str]:
        """フロー構造 (FlowSpec) を合成し、検証済み steps と出所を返す。

        戻り値の第 2 要素は ``"llm" | "llm_retry" | "fallback" | "none"``。
        入力 ``spec`` は確定後 (モジュール節補完 + Canonical 追記済み) を渡す。
        検証違反・切断・崩れ JSON は違反文言を添えて 1 回だけ再試行し、なお
        不成立なら正準一覧からの線形フォールバック (``fallback_flow``、
        module_list 空なら成果物なし) へ決定論縮退する。検証合格 steps は
        ``renumber_steps`` で S1..Sn へ再採番済み (mermaid ノード id と
        Processing flow 節の番号が 1:1)。
        """
        module_paths = module_paths_from_list(module_list)
        entries = _fallback_flow_labels(spec, module_list)

        def _fallback() -> tuple[list[FlowStep], str]:
            steps = fallback_flow(entries)
            return (steps, "fallback") if steps else ([], "none")

        if self.assist_client is None:
            return _fallback()

        # 入力から旧 Processing flow 節を除く (改訂後の再合成で自己参照させ
        # ない)。予算超過時は全モジュール公平の圧縮 (先頭からの単純切詰めは
        # 深化後 spec で後方モジュール節を黙って落とす — レビュー確定指摘)。
        body = replace_flow_section(spec, "")
        if len(body) > _FLOW_CONTEXT_CHARS:
            body = _condensed_flow_context(body, module_paths)
        total_behavior = sum(len(labels) for _, labels in entries)
        step_lo, step_hi = _flow_step_count_hint(total_behavior, MAX_FLOW_STEPS)
        base_prompt = _FLOW_SPEC_PROMPT.format(
            max_steps=MAX_FLOW_STEPS, step_lo=step_lo, step_hi=step_hi,
            module_list=module_list or "(single ad-hoc program — no file list)",
            spec=body,
        ) + _flow_language_constraint()

        violations: list[str] = []
        for attempt in ("llm", "llm_retry"):
            prompt = base_prompt
            if violations:
                prompt = (
                    "Your previous JSON was rejected by the mechanical "
                    "validator:\n"
                    + "\n".join(f"- {v}" for v in violations)
                    + "\nRegenerate the complete JSON and fix EVERY listed "
                    "problem.\n\n"
                ) + base_prompt
            telemetry: dict = {}
            try:
                data = await self.assist_client.generate_json(
                    prompt, max_tokens=_FLOW_MAX_TOKENS, temperature=0.2,
                    purpose="flow_spec_synthesis", telemetry=telemetry,
                    # 出力予算 3072 tok は iGPU 実測 (7-13 t/s) で purpose 既定
                    # 120s を超えうるため明示 timeout で上書きする。
                    timeout=300.0,
                )
            except Exception as exc:
                logger.warning("flow spec synthesis failed (%s): %s",
                               attempt, exc)
                violations = []
                continue
            if telemetry.get("truncated"):
                # json_repair が切断 JSON を閉じて素通りし、途中欠けの steps を
                # 妥当と誤認しうるため、切断応答は全体を不信として棄却する。
                logger.warning(
                    "flow spec synthesis truncated at max_tokens (%s); "
                    "discarding", attempt,
                )
                violations = []
                continue
            steps = steps_from_payload(data, module_paths)
            violations = validate_flow(steps, module_paths)
            if not violations:
                return renumber_steps(steps), attempt
            repaired = repair_unreachable_tail(steps)
            if repaired is not steps and not validate_flow(repaired, module_paths):
                logger.info("flow spec repaired (%s): bridged unreachable tail",
                            attempt)
                return renumber_steps(repaired), f"{attempt}_repaired"
            logger.warning(
                "flow spec validation failed (%s): %s",
                attempt, "; ".join(violations[:6]),
            )
        if self.flow_part_synthesis_enabled and module_paths:
            parts_steps = await self._synthesize_flow_parts(module_paths, spec)
            if parts_steps is not None:
                return renumber_steps(parts_steps), "parts"
        return _fallback()

    async def _synthesize_flow_parts(
        self, module_paths: list[str], spec: str,
    ) -> list[FlowStep] | None:
        """flow_spec_synthesis エスカレーション: Component/モジュール単位で
        小規模サブグラフを部分合成→決定論結合する。

        各正準モジュールについて :func:`plan_file_parts` (code 工程の部分生成
        と同じ Component 分割) を試み、Component が 2 つ以上あればその単位、
        無ければモジュール全体を 1 単位とする。1 ユニットでも生成が枯渇したら
        部分結果を握り潰さず ``None`` を返す (呼出側は決定論フォールバックへ
        委譲する — "欠落したまま結合しない" という部分生成と同じ哲学)。
        結合後は必ず :func:`validate_flow` で全体を再検証し、合格した場合の
        みステップ列を返す。
        """
        assert self.assist_client is not None
        units_specs: list[tuple[str, str]] = []  # (module_path, unit_spec_text)
        for module_path in module_paths:
            plan = plan_file_parts(spec, module_path, max_parts=self.part_max_parts)
            if plan is not None:
                units_specs.extend(
                    (module_path, group.spec_text) for group in plan.groups
                )
                continue
            found = extract_module_section(spec, module_path)
            if found is None:
                return None
            units_specs.append((module_path, found[2]))

        units: list[list[FlowStep]] = []
        for i, (module_path, unit_spec) in enumerate(units_specs, start=1):
            prompt = _FLOW_PART_PROMPT.format(
                module_path=module_path,
                step_lo=_FLOW_PART_STEP_LO, step_hi=_FLOW_PART_STEP_HI,
                spec=unit_spec,
            ) + _flow_language_constraint()
            steps: list[FlowStep] = []
            for _attempt in range(2):  # 初回 + 再試行 1 回
                try:
                    data = await self.assist_client.generate_json(
                        prompt, max_tokens=768, temperature=0.2,
                        purpose="flow_spec_part_synthesis", timeout=60.0,
                    )
                except Exception as exc:
                    logger.warning(
                        "flow spec part synthesis failed (%d/%d, %s): %s",
                        i, len(units_specs), module_path, exc,
                    )
                    continue
                steps = steps_from_payload(data, [module_path])
                if steps:
                    break
            if not steps:
                logger.warning(
                    "flow spec part synthesis exhausted for %d/%d (%s); "
                    "abandoning escalation", i, len(units_specs), module_path,
                )
                return None
            units.append(prefix_unit_steps(steps, f"U{i}_"))

        assembled = assemble_flow_parts(units)
        if assembled is None:
            return None
        repaired = repair_unreachable_tail(assembled)
        if validate_flow(repaired, module_paths):
            return None
        return repaired

    # ── code 工程 (base モデル) ───────────────────────────────────────
    def _sibling_contract_block(self, source_path: str) -> str:
        """既生成兄弟 src の実 API 契約ブロック (無ければ中立な完全実装文言)。"""
        api_block = build_sibling_api_block(self._sibling_src(source_path))
        if api_block:
            return (
                f"## Already-implemented sibling modules — use their EXACT public API\n"
                f"These modules already exist in this program. Import from them and call "
                f"the EXACT class names, method names, and constructor signatures shown "
                f"below — do NOT invent different names or signatures. `{source_path}` "
                f"MUST be consistent with them.\n\n{api_block}\n\n"
            )
        # 兄弟が無い場合 (単一モジュール or 最初に生成されるモジュール)。
        # 「sibling modules が使う API を定義せよ」と煽るとスタブ化を招くため、
        # 完全実装を促す中立な文言にする。
        return (
            "## No sibling modules generated yet\n"
            "Implement this file completely with real, working logic and clear, "
            "importable public classes/functions.\n\n"
        )

    def _build_code_instruction(
        self, description: str, source_path: str, spec: str, flowchart: str,
    ) -> str:
        """単一ファイル生成の instruction を組み立てる (code 工程と spec 見直し後の
        再生成で共有。spec 全文・flowchart・兄弟 API を無改変で運ぶ)。

        spec が ``_CODE_SPEC_MAX_CHARS`` を超える場合 (深化後の複数モジュール
        構成で起こりうる) は :func:`_condensed_code_spec` で圧縮する。単一
        メッセージ (system+user) の instruction は
        ``local_client._enforce_context_budget`` の位置ベース中略に落ちる前に
        ここで生成対象モジュール自身の節を優先温存する。
        """
        if len(spec) > _CODE_SPEC_MAX_CHARS:
            spec = _condensed_code_spec(spec, source_path)
        flow_block = (
            f"## Module architecture diagram\n```mermaid\n{flowchart}\n```\n\n"
            if flowchart.strip() else ""
        )
        return (
            f"{description}\n\n"
            f"## Shared design specification\n{spec}\n\n"
            f"{flow_block}"
            f"{self._sibling_contract_block(source_path)}"
            f"The design specification above is a BINDING CONTRACT: implement "
            f"EVERY `### Component:` of `{source_path}` with EXACTLY the "
            f"declared signatures (class names, method names, parameters), the "
            f"declared attributes and constants, following the Behavior steps "
            f"and the `## Processing flow` order. The output is mechanically "
            f"checked against the declared signatures and non-conforming code "
            f"is rejected.\n"
            f"Produce the COMPLETE, fully working contents of the single file "
            f"`{source_path}` implementing the above with real logic — do NOT leave "
            f"function/method bodies as stubs (no bare `pass`, `# TODO`/`...` "
            f"placeholders, or NotImplementedError). Use the EXACT Python types "
            f"stated in the shared design specification's data structures — do "
            f"NOT substitute a different type that merely behaves similarly "
            f"(e.g. if a field is specified as holding `int` values, it must "
            f"hold real `int`s, not `bool`). Output only this file's code."
            + os_constraint()
            + self._dependency_constraint()
            + _code_language_constraint()
        )

    def _request_text(self) -> str:
        """依存制約の検出に使う「ユーザーの元の要求文」 (manifest の goal)。"""
        try:
            return str(self.workspace.read_manifest().get("goal") or "")
        except Exception:
            return ""

    def _dependency_constraint(self) -> str:
        """要求文が「標準ライブラリのみ」ならその制約を生成指示へ運ぶ。

        ゲート (:meth:`_run_code`) だけでは retry のたびに同じ違反を作り直す
        ので、制約は生成時点で伝える。
        """
        if not requires_stdlib_only(self._request_text()):
            return ""
        return (
            "\n\nThe user requires the STANDARD LIBRARY ONLY: do not import any "
            "third-party package (no pandas, numpy, requests, fastapi, ...). "
            "Use only modules from the Python standard library and the sibling "
            "modules of this project. Code importing a third-party package is "
            "rejected."
        )

    def _local_module_names(self, source_path: str) -> set[str]:
        """生成物の兄弟モジュール名 (import 検証で自作扱いにする集合)。"""
        names = {Path(source_path).stem}
        try:
            files = self.workspace.read_manifest().get("files") or {}
            names |= {Path(p).stem for p in files}
        except Exception:
            pass
        return names

    async def _run_code(self, task: "TaskFactView") -> ExecutionOutcome:
        source_path = task.source_path or f"{task.task_id}.py"
        self._emit("code", f"コード生成中: {source_path}", "running", task.task_id)
        spec, flowchart = self._read_spec_and_flowchart(f"code stage {source_path}")

        # 部分ごと生成 (発動条件: 両 delegate 注入済み + spec に component 構造)。
        # 失敗時は code="" のまま従来の単発生成へフォールバックする。
        code = ""
        part_notes: dict[str, str] = {}
        if self.part_codegen is not None and self.part_assembler is not None:
            plan = plan_file_parts(spec, source_path, max_parts=self.part_max_parts)
            if plan is not None:
                code, part_notes = await self._generate_in_parts(
                    task, source_path, plan, flowchart,
                )

        if not code.strip():
            instruction = self._build_code_instruction(
                task.description, source_path, spec, flowchart,
            )
            try:
                files = await self.codegen(instruction, source_path)
            except Exception as exc:
                logger.warning("code generation failed for %s: %s", source_path, exc)
                self._fail_task(task, f"codegen error: {exc}")
                return ExecutionOutcome(
                    status="failure", error=f"codegen error: {exc}",
                    notes={"executor": self.name, "stage": "code", **part_notes},
                )
            code = _pick_primary(files or {}, source_path)
        if not code.strip():
            logger.warning("staged code stage produced no code for %s", source_path)
            self._fail_task(task, "empty code generation")
            self._emit("code", f"コード生成失敗 (空): {source_path}", "failed", task.task_id)
            return ExecutionOutcome(
                status="failure", error="empty code generation",
                notes={"executor": self.name, "stage": "code", **part_notes},
            )
        # 構文チェックは code 工程で行う。ここを素通りさせると、壊れたファイルは
        # test 工程の smoke ゲートで初めて検出されるが、そのときの repair は
        # **当該 test タスクの source_path に固定** されているため、別ユニットの
        # 構文エラーは誰も直せないまま retry を消費して警告付きで配信される
        # (実インシデント 2026-08-07 ライブ監査: file_scanner.py の
        # ``'continue' not properly in loop`` が test: main.py の失敗として現れ、
        # 2 回の retry がどちらも main.py を書き直して同じエラーで終わった)。
        # 生成したタスク自身を失敗させれば、driver の retry が同じ code タスクを
        # 再実行し、修復対象と原因ファイルが一致する。
        # 同じ判定は生成テスト (_run_test) と repair 書き戻し
        # (_write_source_if_valid) では既に効いており、ここだけ非対称だった。
        # ``_salvage_python_code`` は前後の行を削って再パースするので、前置きの
        # 自然文は救えるが、**本物の構文エラーに当てると壊れた行を削った縮小版**
        # が出来てしまう (実測: 9 行の関数から except 節ごと 3 行消えて「有効な
        # Python」になった)。repair 経路 (_write_source_if_valid) は旧ファイルとの
        # 縮小率ガードが後段にあるが、code 工程には比較対象が無い。ここでは
        # 「ほぼ全量が残った場合だけ前置き除去とみなす」で同じ役割を果たす。
        detail = _compile_error_detail(code)
        if detail is not None:
            salvaged = _salvage_python_code(code)
            if (
                salvaged is not None
                and len(salvaged) >= _SALVAGE_MIN_KEEP_RATIO * len(code)
                and _compile_error_detail(salvaged) is None
            ):
                logger.warning(
                    "staged code stage returned prose around the code for %s; "
                    "salvaged the embedded Python (%d -> %d chars)",
                    source_path, len(code), len(salvaged),
                )
                code = salvaged
                detail = None
        if detail is not None:
            logger.warning(
                "staged code stage produced invalid Python for %s: %s",
                source_path, detail,
            )
            self._fail_task(task, f"syntax error: {detail}")
            self._emit(
                "code", f"コード生成失敗 (構文エラー): {source_path} — {detail}",
                "failed", task.task_id,
            )
            return ExecutionOutcome(
                status="failure", error=f"syntax error: {detail}",
                notes={"executor": self.name, "stage": "code", **part_notes},
            )
        # ユーザーが明示した依存制約の検証。import スモークは「その環境で import
        # できるか」しか見ないため、pandas が入っている開発機では
        # 「標準ライブラリのみ」違反が合格として通ってしまう (実インシデント
        # 2026-08-07 ライブ監査)。制約も違反も決定論で確定できるので LLM に
        # 判定させない。
        if requires_stdlib_only(self._request_text()):
            offenders = find_third_party_imports(
                code, self._local_module_names(source_path),
            )
            if offenders:
                detail = ", ".join(offenders)
                logger.warning(
                    "staged code stage violated the stdlib-only constraint for "
                    "%s: imports %s", source_path, detail,
                )
                self._fail_task(task, f"third-party import: {detail}")
                self._emit(
                    "code",
                    f"コード生成失敗 (標準ライブラリのみの指定に違反): "
                    f"{source_path} — {detail}",
                    "failed", task.task_id,
                )
                return ExecutionOutcome(
                    status="failure",
                    error=f"third-party import despite stdlib-only request: {detail}",
                    notes={"executor": self.name, "stage": "code", **part_notes},
                )
        wf = self.workspace.write_file(
            source_path, code, kind="src", stage="code", task_id=task.task_id,
        )
        self.workspace.upsert_task(
            task_id=task.task_id, title=task.title, stage="code",
            status="done", depends_on=task.depends_on,
        )
        loc = code.count("\n") + 1
        self._emit("code", f"コード生成: {source_path} ({loc} 行)", "done", task.task_id)
        notes = {"executor": self.name, "stage": "code", "file": source_path,
                 **part_notes}
        # 準拠照合の観測記録 (LLM 呼出ゼロ)。執行は test 工程の smoke 合流が
        # 担い、ここでは被覆率 (契約ゼロ = skipped の頻度) を可視化する。
        conf_status, conf_violations = self._conformance_status(source_path, spec)
        if conf_status != "unavailable":
            notes["spec_conformance"] = (
                f"violations:{len(conf_violations)}"
                if conf_violations else conf_status
            )
        # spec stage 同様、成功時の notes は既定でどこにも残らないため INFO
        # ログへ明示する (--develop 不要)。
        logger.info("code stage notes for %s: %s", task.task_id, notes)
        return ExecutionOutcome(
            status="success",
            artifacts=(_artifact(wf.logical_path, wf.sha256),),
            notes=notes,
        )

    def _build_part_instruction(
        self,
        description: str,
        source_path: str,
        plan: "FilePartsPlan",
        flowchart: str,
        prior_parts: list[str],
        part_index: int,
        group: "ComponentGroup",
    ) -> str:
        """部分 1 個の生成 instruction を組み立てる。

        自モジュール節は**全文・無改変**で運ぶ (他モジュール節は運ばない —
        prefill 予算の支配項。未生成 sibling の情報は sibling API block +
        smoke 修復が担保する既存トレードオフに従う)。既生成部分は実コード
        全文を運び、超過時のみ AST 公開シグネチャ要約へ決定論縮退する。
        """
        flow_block = (
            f"## Module architecture diagram\n```mermaid\n{flowchart}\n```\n\n"
            if flowchart.strip() else ""
        )
        canonical_block = (
            f"{plan.canonical_list.strip()}\n\n" if plan.canonical_list.strip() else ""
        )
        prior_block = ""
        if prior_parts:
            joined = "\n\n".join(p.rstrip() for p in prior_parts)
            if len(joined) > _PART_CONTEXT_MAX_CHARS:
                joined = summarize_module_api(joined) or joined[:_PART_CONTEXT_MAX_CHARS]
                prior_head = (
                    f"## Public API of parts 1..{part_index - 1} of `{source_path}` "
                    f"(already final — call these EXACT signatures)\n"
                )
            else:
                prior_head = (
                    f"## Parts 1..{part_index - 1} of `{source_path}` generated so "
                    f"far — already final\n"
                )
            prior_block = f"{prior_head}```python\n{joined}\n```\n\n"
        n = len(plan.groups)
        return (
            f"{description}\n\n"
            f"## Design specification for `{source_path}` (this module) — "
            f"implement faithfully\n{plan.module_section.strip()}\n\n"
            f"{canonical_block}"
            f"{flow_block}"
            f"{self._sibling_contract_block(source_path)}"
            f"{prior_block}"
            f"You are writing PART {part_index} of {n} of the single file "
            f"`{source_path}`. This part must contain ONLY the following "
            f"component(s), fully implemented:\n\n{group.spec_text.strip()}\n\n"
            f"Rules:\n"
            f"- Output ONLY this part's code, including the import statements "
            f"THIS part needs (imports are merged deterministically later).\n"
            f"- Do NOT repeat or redefine anything from the already-generated "
            f"parts above.\n"
            f"- Call components defined in other parts of this module EXACTLY as "
            f"specified in the module design above. They live in the SAME file "
            f"scope — do NOT import them, and never write `from "
            f"{Path(source_path).stem} import ...` (importing this module from "
            f"itself crashes at startup).\n"
            f"- Do NOT emit the `if __name__ == \"__main__\"` guard unless the "
            f"component above IS the entry point.\n"
            f"- Implement with real logic — do NOT leave function/method bodies "
            f"as stubs (no bare `pass`, `# TODO`/`...` placeholders, or "
            f"NotImplementedError). Use the EXACT Python types stated in the "
            f"design specification.\n"
            f"- The component spec above is a BINDING CONTRACT: define EXACTLY "
            f"the declared class/method/function signatures, attributes and "
            f"constants, following the Behavior steps. The output is "
            f"mechanically checked against the declared signatures and "
            f"non-conforming code is rejected."
            + os_constraint()
            + self._dependency_constraint()
            + _code_language_constraint()
        )

    async def _generate_in_parts(
        self,
        task: "TaskFactView",
        source_path: str,
        plan: "FilePartsPlan",
        flowchart: str,
    ) -> tuple[str, dict[str, str]]:
        """spec の component 単位で部分生成し決定論結合する。

        戻り値 ``(code, notes)``。失敗時は ``("", {"part_mode":
        "fallback_single", ...})`` を返し、呼出側が単発生成へ 1 回だけ
        フォールバックする (部分欠落のまま結合はしない — smoke は欠落 component
        を検出できないため)。各部分は空応答/例外時に同一 instruction で 1 回
        だけ再試行する。
        """
        assert self.part_codegen is not None and self.part_assembler is not None
        n = len(plan.groups)
        parts: list[str] = []
        for i, group in enumerate(plan.groups, start=1):
            self._emit(
                "code", f"部分生成中 ({i}/{n}): {group.label} — {source_path}",
                "running", task.task_id,
            )
            instr = self._build_part_instruction(
                task.description, source_path, plan, flowchart, parts, i, group,
            )
            code = ""
            last_err = ""
            for _attempt in range(2):  # 初回 + 再試行 1 回
                try:
                    files = await self.part_codegen(instr, source_path)
                except Exception as exc:
                    last_err = str(exc)
                    logger.warning(
                        "part %d/%d generation failed for %s: %s",
                        i, n, source_path, exc,
                    )
                    continue
                code = _pick_primary(files or {}, source_path)
                if code.strip():
                    # per-part 準拠チェック: 担当 component の宣言名が part に
                    # 無ければ空応答と同じ扱いで再試行 → 枯渇で単発フォール
                    # バック (欠落 component は結合後 smoke で検出できない)。
                    missing = _missing_declared_names(group.spec_text, code)
                    if not missing:
                        undeclared = _undeclared_part_names(
                            self.coherence_checker, self.part_assembler,
                            parts, code, source_path,
                        )
                        if not undeclared:
                            break
                        last_err = (
                            "part introduces undeclared name(s): "
                            + ", ".join(undeclared)
                        )
                        logger.warning(
                            "part %d/%d for %s %s; retrying",
                            i, n, source_path, last_err,
                        )
                        code = ""
                        continue
                    last_err = (
                        "part lacks declared component(s): "
                        + ", ".join(missing)
                    )
                    logger.warning(
                        "part %d/%d for %s %s; retrying",
                        i, n, source_path, last_err,
                    )
                    code = ""
                    continue
                last_err = "empty part generation"
            if not code.strip():
                self._emit(
                    "code",
                    f"部分生成失敗 ({i}/{n}) → 単発生成へフォールバック: {source_path}",
                    "failed", task.task_id,
                )
                return "", {
                    "part_mode": "fallback_single",
                    "part_fallback_reason": f"part {i}/{n}: {last_err}"[:200],
                }
            parts.append(code)

        try:
            # module_stem を渡し、自モジュールからの幻覚 import (from <stem>
            # import ...) を結合時に決定論除去する。
            assembled = self.part_assembler(
                parts, module_stem=Path(source_path).stem,
            )
        except Exception as exc:
            logger.warning("part assembly raised for %s: %s", source_path, exc)
            assembled = None
        if not assembled or not assembled.strip():
            self._emit(
                "code",
                f"部分結合失敗 → 単発生成へフォールバック: {source_path}",
                "failed", task.task_id,
            )
            return "", {
                "part_mode": "fallback_single",
                "part_fallback_reason": "assembly failed",
            }
        self._emit(
            "code", f"部分結合完了 ({n} 部分): {source_path}", "running", task.task_id,
        )
        return assembled, {"part_mode": "parts", "parts": str(n)}

    # ── test 工程 (スモーク検証ゲート + advisory ユニットテスト) ─────────
    async def _run_test(self, task: "TaskFactView") -> ExecutionOutcome:
        """合否 = 決定論的 import スモーク AND 生成テストの最終結果。

        コードは成果物 (権威)。スモークの実エラー時のみコードを修正する
        (劣化ガード付き)。生成テストが決定論的に赤と確認された場合のみ
        failure に集計する (成果物の配信自体は finalize が test 状態に依存せず
        継続する = 警告付き配信)。テスト未生成 / インフラ障害 / degraded は
        従来どおり success (環境起因を罰しない)。
        """
        source_path = task.source_path or ""
        src_code = self.workspace.read_file(source_path, kind="src") if source_path else None
        if not src_code:
            self._emit("test", "ソース未検出のためスキップ", "failed", task.task_id)
            return ExecutionOutcome(
                status="skipped", error="source not found for test stage",
                notes={"executor": self.name, "stage": "test", "source": source_path},
            )

        # driver リトライの高速パス: 初回実行で spec 見直しサイクルまで完了して
        # いて src が不変なら、advisory 再生成をスキップし pytest のみ再実行する
        # (フレーク救済。LLM 呼出ゼロで数秒)。record_stage_notes はマーカーを
        # 消さないようここでは呼ばない。
        if self._retry_fast_path_applicable(task, source_path):
            return await self._run_retry_fast_path(task, source_path)

        # driver リトライの smoke 側高速確定: 初回が smoke 失敗で終わり src が
        # 不変なら、修復ループ (LLM) を回しても結果は決定論的に同一。修復なしの
        # 再検証 1 回だけ行い、まだ失敗なら即 failure を確定する (兄弟ファイルの
        # 変更で解消していた場合のみ通常フローへ続行)。
        retry_smoke_failfast = self._retry_smoke_failfast_applicable(task, source_path)
        if retry_smoke_failfast:
            self._emit(
                "test", "リトライ: src 不変のため修復なしで再検証のみ",
                "running", task.task_id,
            )

        # 1) 決定論的スモークゲート (実エラー時のみコード修正)
        smoke_ok, gate, errors = await self._smoke_gate(
            task, source_path, src_code,
            max_rounds=(0 if retry_smoke_failfast else None),
            start_attempt=(30 if retry_smoke_failfast else 1),
        )
        if retry_smoke_failfast and not smoke_ok:
            err_summary = ("; ".join(errors)[:300] or "smoke errors")
            self.workspace.upsert_task(
                task_id=task.task_id, title=task.title, stage="test",
                status="failed", depends_on=task.depends_on,
                last_error=err_summary,
            )
            return ExecutionOutcome(
                status="failure",
                gate_outcome=_single_gate_outcome(gate) if gate is not None else None,
                error=err_summary,
                notes={"executor": self.name, "stage": "test",
                       "smoke": "errors",
                       "advisory_test": "skipped_retry_smoke_failed",
                       "retry_fast_path": "smoke_failfast"},
            )

        # 1.5) smoke 枯渇でもエラー残存 → spec 該当節の見直し (トリガ a)
        if not smoke_ok and await self._spec_revision_cycle(
            task, source_path,
            evidence="\n".join(errors), kind="import smoke",
        ):
            smoke_ok, gate, errors = await self._smoke_gate(
                task, source_path, src_code, max_rounds=1, start_attempt=10,
            )

        # 2) advisory ユニットテスト (ソース不変)。pytest 赤のみ最終合否へ波及する
        test_sha, advisory_status, _pytest_gate = await self._advisory_unit_test(
            task, source_path,
        )

        # 2.5) pytest 赤 → spec 該当節の見直し (トリガ b)。適用されたら
        # 再 smoke (修復 1 回) → 再テスト (既存 test 再利用) で最終結果を更新する
        if (
            smoke_ok
            and advisory_status == "generated_pytest_failed"
            and await self._spec_revision_cycle(
                task, source_path,
                evidence=(
                    (_pytest_gate.stdout_tail or _pytest_gate.stderr_tail or "")
                    if _pytest_gate is not None else ""
                ),
                kind="pytest",
            )
        ):
            smoke_ok, gate, errors = await self._smoke_gate(
                task, source_path, src_code, max_rounds=1, start_attempt=20,
            )
            if smoke_ok:
                test_sha, advisory_status, _pytest_gate = (
                    await self._advisory_retest(task, source_path)
                )

        pytest_final_failed = advisory_status == "generated_pytest_failed"
        passed = smoke_ok and not pytest_final_failed
        if smoke_ok and pytest_final_failed:
            err_summary: str | None = (
                f"generated tests failed: {(_pytest_gate.error if _pytest_gate else '')}"
            ).strip()
            self._emit(
                "test",
                f"テスト未合格 — 警告付きで配信されます ({err_summary})",
                "failed", task.task_id,
            )
        else:
            err_summary = (
                ("; ".join(errors)[:300] or "smoke errors") if errors else None
            )
        self.workspace.upsert_task(
            task_id=task.task_id, title=task.title, stage="test",
            status="done" if passed else "failed",
            depends_on=task.depends_on, last_error=err_summary,
        )
        artifacts: tuple = ()
        if test_sha is not None:
            artifacts = (_artifact(f"tests/{_test_logical_for(source_path)}", test_sha),)
        notes = {"executor": self.name, "stage": "test",
                 "smoke": "ok" if smoke_ok else "errors",
                 "advisory_test": advisory_status}
        if pytest_final_failed:
            notes["test_verdict"] = "pytest_unpassed"
        return ExecutionOutcome(
            status="success" if passed else "failure",
            # gate_outcome は smoke のまま (pytest を載せると UI 側が
            # 「起動可能性チェック失敗」と誤訳するため。pytest は notes/emit で表示)
            gate_outcome=_single_gate_outcome(gate) if gate is not None else None,
            artifacts=artifacts,
            error=None if passed else err_summary,
            notes=notes,
        )

    def _retry_smoke_failfast_applicable(
        self, task: "TaskFactView", source_path: str,
    ) -> bool:
        """driver リトライ時に「修復なし smoke 1 回」で確定してよい条件。

        条件: (a) 初回実行が spec 見直しサイクルまで完了 (stage_notes マーカー)
        (b) 直近 smoke が failed (c) 当該 src の sha が記録時と不変。
        smoke は全 src を検査するため、兄弟変更で解消しうるケースに備えて
        本判定は「修復スキップ」までに留め、失敗確定は再検証の結果で行う。
        """
        if self._revision_marker(task.task_id) is None:
            return False
        smoke_rec = self.workspace.get_test_result(task.task_id)
        if smoke_rec is None or smoke_rec.passed:
            return False
        recorded = self._revision_recorded_sha(task.task_id)
        current = self.workspace.file_map().get(source_path)
        return bool(recorded and current is not None and current.sha256 == recorded)

    def _retry_fast_path_applicable(
        self, task: "TaskFactView", source_path: str,
    ) -> bool:
        """driver リトライ時に pytest のみ再実行してよい条件を判定する。

        条件: (a) 初回実行で spec 見直しサイクルまで完了 (stage_notes マーカー)
        (b) 直近 pytest が failed (c) 直近 smoke が passed (d) src sha が記録時と
        不変 (e) 生成テストが存在。いずれか欠ければ通常フローをやり直す。
        """
        if self._revision_marker(task.task_id) is None:
            return False
        pytest_rec = self.workspace.get_test_result(f"{task.task_id}.pytest")
        if pytest_rec is None or pytest_rec.passed:
            return False
        smoke_rec = self.workspace.get_test_result(task.task_id)
        if smoke_rec is None or not smoke_rec.passed:
            return False
        recorded = self._revision_recorded_sha(task.task_id)
        current = self.workspace.file_map().get(source_path)
        if not recorded or current is None or current.sha256 != recorded:
            return False
        test_logical = _test_logical_for(source_path)
        return bool(self.workspace.read_file(test_logical, kind="test"))

    async def _run_retry_fast_path(
        self, task: "TaskFactView", source_path: str,
    ) -> ExecutionOutcome:
        """既存テストで pytest のみ再実行し、最終合否を更新する (LLM 呼出なし)。"""
        test_logical = _test_logical_for(source_path)
        self._emit(
            "test", "リトライ: 既存テストで pytest のみ再実行", "running", task.task_id,
        )
        status, gate = await self._run_advisory_pytest(task, test_logical, attempt=3)
        passed = status == "generated_pytest_ok"
        err = None if passed else (
            f"generated tests failed: {(gate.error if gate else '')}".strip()
        )
        if not passed:
            self._emit(
                "test", f"テスト未合格 — 警告付きで配信されます ({err})",
                "failed", task.task_id,
            )
        self.workspace.upsert_task(
            task_id=task.task_id, title=task.title, stage="test",
            status="done" if passed else "failed",
            depends_on=task.depends_on, last_error=err,
        )
        test_file = self.workspace.file_map().get(test_logical)
        artifacts: tuple = ()
        if test_file is not None:
            artifacts = (_artifact(f"tests/{test_logical}", test_file.sha256),)
        notes = {"executor": self.name, "stage": "test",
                 "smoke": "ok", "advisory_test": status,
                 "retry_fast_path": "true"}
        if not passed:
            notes["test_verdict"] = "pytest_unpassed"
        return ExecutionOutcome(
            status="success" if passed else "failure",
            artifacts=artifacts,
            error=err,
            notes=notes,
        )

    # ── スモークゲート ─────────────────────────────────────────────────
    async def _smoke_gate(
        self, task: "TaskFactView", source_path: str, src_code: str,
        *, max_rounds: int | None = None, start_attempt: int = 1,
    ) -> tuple[bool, "GateResult | None", list[str]]:
        """全 src を import スモークし、実エラーがあればコードを修正する (有界)。

        Returns ``(ok, gate, errors)``。``smoke_runner`` 未注入時はスキップ=成功。
        ``max_rounds`` は修復回数の上書き (spec 見直し後の再検証は 1 に制限する)。
        ``start_attempt`` は ``_record`` / ``tests/_runs`` の attempt 番号オフセット
        (再入時の衝突回避)。
        """
        if self.smoke_runner is None and (
            not self.spec_conformance_enabled
            or self.conformance_checker is None
        ):
            # smoke も準拠ゲートも走らせるものが無い時のみ完全スキップ
            # (準拠ゲートは smoke_runner 無効時でも独立に執行する — smoke だけ
            # 切る config で準拠検査まで黙って消えないように)。
            self._emit("test", "スモーク検証スキップ (degraded)", "done", task.task_id)
            return True, None, []
        repair_rounds = self.max_repair_rounds if max_rounds is None else max_rounds

        # spec/flowchart はループ外で 1 度だけ読む (毎試行同一、I/O 削減)。修正指示に
        # 含めないと base モデルは「エラーを消すこと」だけを目標にでき、spec の
        # ロジック (境界条件・データ構造・エントリポイント仕様等) を無視した修正
        # (機能削除・別ロジックへの置換) を許してしまう。準拠検査 (spec 宣言契約
        # との照合) も同じ spec を根拠に初回・修復後の両方で合成する。
        spec, flowchart = self._read_spec_and_flowchart(f"smoke repair {source_path}")

        self._emit("test", "import スモーク検証中", "running", task.task_id)
        errors, warnings = await self._smoke_once()
        errors += self._gate_conformance_errors(source_path, spec)
        attempt = start_attempt
        gate = _smoke_gate_result(errors, warnings)
        self._record(task.task_id, gate, attempt, kind="smoke")
        self._emit_smoke(task, errors, warnings, attempt)
        spec_block = f"## Shared design specification\n{spec}\n\n" if spec.strip() else ""
        repair_flow_block = (
            f"## Module architecture diagram\n```mermaid\n{flowchart}\n```\n\n"
            if flowchart.strip() else ""
        )

        repairs_done = 0
        while errors and repairs_done < repair_rounds:
            repairs_done += 1
            self.workspace.bump_task_attempt(task.task_id)
            self._emit("test", f"コード修正 (試行 {attempt + 1})", "running", task.task_id)
            cur_src = self.workspace.read_file(source_path, kind="src") or src_code
            api_block = build_sibling_api_block(self._sibling_src(source_path))
            sibling_block = (
                f"## Sibling modules' ACTUAL public API — align `{source_path}` to these "
                f"(use the EXACT names/signatures; do NOT invent different ones)\n"
                f"{api_block}\n\n"
                if api_block else ""
            )
            repair_instr = (
                f"The generated module(s) failed an import smoke test.\n\n"
                f"## Smoke errors\n" + "\n".join(f"- {e}" for e in errors) + "\n\n"
                f"{spec_block}"
                f"{repair_flow_block}"
                f"{sibling_block}"
                f"## Current source `{source_path}`\n```python\n{cur_src}\n```\n\n"
                f"Fix `{source_path}` so it imports and runs without these errors "
                f"AND conforms to the component signatures declared in the design "
                f"specification (spec-conformance errors name the exact missing/"
                f"mismatched declaration), while remaining faithful to the design "
                f"specification above. "
                f"KEEP ALL existing functionality — do NOT remove features, classes, or "
                f"functions, and do NOT shrink the program to a stub. "
                f"Output only the corrected `{source_path}` code."
                + _NO_PROSE_OUTPUT_CONSTRAINT
                + _code_language_constraint()
            )
            fixed = await self._safe_codegen(repair_instr, source_path)
            wrote = self._apply_source_fixes(fixed, source_path, task.task_id)
            if not wrote:
                # 修復応答が空 / 非 Python / 縮小ガード棄却 = モデルが修復不能の
                # シグナル。src が不変のまま再スモーク・追加修復を回しても結果は
                # 変わらず、低速環境では倍増リトライ込みの大コストだけ嵩むため
                # 残りの修復 round を打ち切る (2026-07-06 live: 非 Python 応答で
                # 2 round 空転 ×2 リトライ = 約 20 分の浪費)。
                logger.warning(
                    "smoke repair produced no applicable fix for %s; "
                    "aborting remaining repair rounds", source_path,
                )
                self._emit(
                    "test", "修復不能 (無効な修復応答) — 残りの修復を打ち切り",
                    "failed", task.task_id,
                )
                break
            attempt += 1
            errors, warnings = await self._smoke_once()
            errors += self._gate_conformance_errors(source_path, spec)
            gate = _smoke_gate_result(errors, warnings)
            self._record(task.task_id, gate, attempt, kind="smoke")
            self._emit_smoke(task, errors, warnings, attempt)

        return (not errors), gate, errors

    def _conformance_status(
        self, source_path: str, spec: str,
    ) -> tuple[str, list[str]]:
        """spec 宣言契約と現 src の準拠照合。(status, violations) を返す。

        status: ``"ok" | "violations" | "skipped"`` (節不在/契約ゼロ/例外) |
        ``"unavailable"`` (checker 未注入)。判定不能はすべて skipped 側へ倒す
        (保守的 — 誤検知は破壊的リペアを招く)。観測記録用に
        ``spec_conformance_enabled`` とは独立に動き、ゲート合流だけが
        フラグでガードされる (:meth:`_gate_conformance_errors`)。
        """
        if self.conformance_checker is None:
            return "unavailable", []
        found = extract_module_section(spec, source_path)
        if found is None:
            return "skipped", []
        try:
            declared = parse_declared_contract(found[2])
            if not declared:
                return "skipped", []
            src_files = {
                f.logical_path: (
                    self.workspace.read_file(f.logical_path, kind="src") or ""
                )
                for f in self.workspace.list_files(kind="src")
            }
            violations = list(self.conformance_checker(
                declared, src_files, primary_path=source_path,
            ))
        except Exception as exc:
            logger.warning(
                "spec conformance check failed for %s: %s", source_path, exc,
            )
            return "skipped", []
        return ("violations" if violations else "ok"), violations

    def _gate_conformance_errors(self, source_path: str, spec: str) -> list[str]:
        """smoke gate へ合流させる準拠違反列 (無効化時は空 = 観測のみに縮退)。

        'spec-conformance: ' プレフィックスで smoke 由来のエラーと区別する。
        合流先の有界リペア (spec 全文注入済み)・spec 見直しループ・retry
        failfast をそのまま再利用する (専用リペア機構は持たない — f_10 §5)。
        """
        if not self.spec_conformance_enabled:
            return []
        status, violations = self._conformance_status(source_path, spec)
        if status == "violations":
            return [f"spec-conformance: {v}" for v in violations]
        return []

    async def _smoke_once(self) -> tuple[list[str], list[str]]:
        """現在の全 src を import スモークし (errors, warnings) を返す。

        ``smoke_runner`` 未注入 (config off / degraded) は空を返す (準拠ゲート
        のみのモードで _smoke_gate が続行できるように null-safe)。
        """
        if self.smoke_runner is None:
            return [], []
        src_files = {
            f.logical_path: (self.workspace.read_file(f.logical_path, kind="src") or "")
            for f in self.workspace.list_files(kind="src")
        }
        try:
            result = await asyncio.to_thread(self.smoke_runner, src_files)  # type: ignore[misc]
        except Exception as exc:
            logger.warning("smoke runner failed: %s", exc)
            return [], []
        errors = [str(e) for e in (getattr(result, "errors", None) or [])]
        warnings = [str(w) for w in (getattr(result, "warnings", None) or [])]
        return errors, warnings

    def _emit_smoke(
        self, task: "TaskFactView", errors: list[str], warnings: list[str], attempt: int,
    ) -> None:
        if not errors:
            extra = f" (警告 {len(warnings)} 件)" if warnings else ""
            self._emit(
                "test", f"起動可能性チェック: OK (静的・未実行){extra}",
                "done", task.task_id,
            )
        else:
            self._emit(
                "test", f"スモーク: {len(errors)} 件のエラー (試行 {attempt})",
                "failed", task.task_id,
            )

    # ── advisory ユニットテスト (ソース不変) ────────────────────────────
    async def _advisory_unit_test(
        self, task: "TaskFactView", source_path: str,
    ) -> tuple[str | None, str, "GateResult | None"]:
        """参考用ユニットテストを 1 回生成・実行する (ソースは書き換えない)。

        戻り値は ``(test_sha, status, pytest_gate)``。status は
        ``ExecutionOutcome.notes`` の ``advisory_test`` に記録され、「テスト不要で
        未生成」と「インフラ障害 (タイムアウト等) で生成に失敗」を区別する
        (前者は無害、後者はテスト成果物の欠落として可観測にする)。pytest_gate は
        pytest 実行時のみ非 None で、失敗トレース (stdout_tail) を spec 見直し
        ループ・最終合否判定の材料として呼出側へ返す。
        """
        if self.test_runner is None:
            return None, "test_runner_unavailable", None
        spec, flowchart = self._read_spec_and_flowchart(f"advisory test {source_path}")
        src_code = self.workspace.read_file(source_path, kind="src") or ""
        test_logical = _test_logical_for(source_path)
        stem = Path(source_path).stem
        flow_block = (
            f"## Module architecture diagram\n```mermaid\n{flowchart}\n```\n\n"
            if flowchart.strip() else ""
        )
        self._emit("test", f"ユニットテスト生成 (参考): {test_logical}",
                   "running", task.task_id)
        gen_instr = (
            f"## Shared design specification\n{spec}\n\n"
            f"{flow_block}"
            f"## Source file `{source_path}`\n```python\n{src_code}\n```\n\n"
            f"Write pytest tests in a single file `{test_logical}`.\n"
            f"REQUIREMENTS:\n"
            f"- `import {stem}` (it is importable as a top-level module) and test its "
            f"PUBLIC API by calling its functions/classes. Do NOT redefine or copy its "
            f"classes, types, constants, or dataclasses — import them from `{stem}`.\n"
            f"- The file MUST contain at least one `def test_...` function with asserts.\n"
            f"- Use only runnable Python; never subscript typing.TypedDict.\n"
            f"- Derive the test cases from the specification's Behavior numbered "
            f"steps, Errors and Invariants bullets — verify the DECLARED behavior, "
            f"not merely that attributes exist.\n"
            f"Output only the test file's code."
            + _NO_PROSE_OUTPUT_CONSTRAINT
            + _code_language_constraint()
        )
        test_files, codegen_error = await self._codegen_or_error(gen_instr, test_logical)
        test_code = _pick_primary(test_files, test_logical)
        if not test_code.strip():
            if codegen_error:
                self._emit(
                    "test",
                    f"ユニットテスト生成失敗 (インフラ障害・参考): {codegen_error[:120]}",
                    "failed", task.task_id,
                )
                return None, f"generation_failed: {codegen_error}", None
            self._emit("test", "ユニットテスト生成なし (参考)", "done", task.task_id)
            return None, "not_generated", None

        # 契約準ゲート: test が src の実 API (arity / 属性) に整合するまで test のみ
        # 有界再生成する。src は権威・不変 (契約違反は src の正しさを否定しないので
        # 合否ゲートにはしない = 準ゲート)。契約 OK の test だけ pytest に回し、壊れた
        # test を握ったまま advisory pytest が無意味に赤くなるのを防ぐ。
        violations = self._contract_violations(source_path, test_code)
        regen = 0
        while violations and regen < self.max_test_regen_rounds:
            regen += 1
            self._emit("test", f"テスト整合性チェック: {len(violations)} 件 → "
                       f"再生成 (試行 {regen})", "running", task.task_id)
            fix_instr = (
                f"The generated tests do NOT match the ACTUAL public API of "
                f"`{source_path}`.\n\n## Contract violations\n"
                + "\n".join(f"- {v}" for v in violations) + "\n\n"
                f"## Source file `{source_path}` (THE source of truth)\n"
                f"```python\n{src_code}\n```\n\n"
                f"Rewrite `{test_logical}`: call the ACTUAL signatures/attributes shown "
                f"above (do NOT invent methods/attributes; do NOT redefine src symbols; "
                f"import them from `{stem}`). At least one `def test_...` with asserts. "
                f"Output only the test file's code."
                + _NO_PROSE_OUTPUT_CONSTRAINT
                + _code_language_constraint()
            )
            regen_code = _pick_primary(
                await self._safe_codegen(fix_instr, test_logical), test_logical,
            )
            if not regen_code.strip():
                break
            test_code = regen_code
            violations = self._contract_violations(source_path, test_code)

        test_code = _ensure_pytest_import(test_code)
        test_code = _ensure_module_import(test_code, source_path)

        # 構文的に無効な test (中間切断等) は書き込む前に弾く。契約チェッカは構文
        # エラーを例外握り潰しで違反ゼロ扱いにする (_contract_violations) ため、この
        # ゲートが無いと壊れたテストファイルがそのまま永続化・pytest 実行されうる。
        if not _is_valid_python(test_code):
            salvaged = _salvage_python_code(test_code)
            if salvaged is None:
                logger.warning(
                    "advisory unit test for %s is not valid Python; treating as "
                    "no test generated", source_path,
                )
                self._emit(
                    "test", f"ユニットテスト生成失敗 (構文エラー、参考): {test_logical}",
                    "failed", task.task_id,
                )
                return None, "invalid_syntax", None
            logger.warning(
                "advisory unit test for %s contained non-code prose; salvaged "
                "the embedded Python (%d -> %d chars)",
                source_path, len(test_code), len(salvaged),
            )
            test_code = salvaged

        wf = self.workspace.write_file(
            test_logical, test_code, kind="test", stage="test",
            task_id=task.task_id, covers=(source_path,),
        )
        if violations:
            # 残存違反は記録するが合否ゲートにはしない。壊れた test は pytest に回さない。
            self._emit(
                "test",
                f"テスト整合性: {len(violations)} 件の不一致が残存 (pytest スキップ)",
                "failed", task.task_id,
            )
            return wf.sha256, "contract_violations_remaining", None

        pytest_status, pytest_gate = await self._run_advisory_pytest(task, test_logical)
        if pytest_status == "generated_pytest_failed" and pytest_gate is not None:
            repaired = await self._maybe_repair_literal_assertions(
                task, source_path, test_logical, test_code, pytest_gate,
            )
            if repaired is not None:
                return repaired
        return wf.sha256, pytest_status, pytest_gate

    async def _maybe_repair_literal_assertions(
        self, task: "TaskFactView", source_path: str, test_logical: str,
        test_code: str, gate: "GateResult",
    ) -> tuple[str, str, "GateResult | None"] | None:
        """pytest 失敗の決め打ちリテラルを実測値へ補正できるか試す (golden-testing)。

        2026-07-23 live: 二分探索の重複値ケース (仕様上どのインデックスも正当
        だがテストが未検証の値を決め打ち) と、スタックの多型 push テストの
        期待値取り違えは、いずれも「実装は正しいがテストの決め打ち期待値だけ
        が誤り」という同型の失敗だった。src は成果物 (権威) というこの
        パイプラインの既存方針に沿い、pytest 自身が報告した実測値でテスト側
        を補正し、1 回だけ再実行する。1 回だけ (spec 見直しループとは独立、
        無限ループ化しない)。

        Returns:
            補正して再実行した場合のみ ``(sha, status, gate)``。補正対象が
            無い (=単純な決め打ちリテラル不一致ではない) 場合は ``None``
            (呼出側は従来の pytest_status/gate をそのまま使う)。
        """
        if self.value_repair is None:
            return None
        output = (gate.stdout_tail or "") + "\n" + (gate.stderr_tail or "")
        repaired_code, fixed_count = self.value_repair(test_code, output)
        if fixed_count == 0:
            return None
        self._emit(
            "test",
            f"テストの決め打ち期待値を実測値へ補正中 ({fixed_count} 件): {test_logical}",
            "running", task.task_id,
        )
        wf = self.workspace.write_file(
            test_logical, repaired_code, kind="test", stage="test",
            task_id=task.task_id, covers=(source_path,),
        )
        status, new_gate = await self._run_advisory_pytest(
            task, test_logical, attempt=2,
        )
        self._emit(
            "test",
            (
                f"テストの期待値を実測値へ補正し合格 ({fixed_count} 件)"
                if status == "generated_pytest_ok"
                else f"期待値補正 ({fixed_count} 件) 後もテスト未合格"
            ),
            "done" if status == "generated_pytest_ok" else "failed", task.task_id,
        )
        return wf.sha256, status, new_gate

    async def _run_advisory_pytest(
        self, task: "TaskFactView", test_logical: str, *, attempt: int = 1,
    ) -> tuple[str, "GateResult | None"]:
        """生成済みテストで pytest を実行し ``(status, gate)`` を返す。

        GateResult は ``<task_id>.pytest`` キーで manifest / tests/_runs へ永続化
        する (spec 見直しループ・リトライ高速パス・finalize 警告の情報源)。
        """
        if self.test_runner is None:
            return "test_runner_unavailable", None
        self._emit("test", f"pytest 実行 (参考): {test_logical}", "running", task.task_id)
        try:
            gate = await asyncio.to_thread(
                self.test_runner.run, test_logical_path=test_logical,
            )
        except Exception as exc:
            logger.warning("advisory pytest run failed: %s", exc)
            return "generated_pytest_error", None
        if gate.skipped:
            # 環境要因 (外部依存未インストール等) の collection error。コードの
            # 欠陥ではないため、spec 見直しトリガ・failure 集計・テスト未合格
            # 警告のいずれにも乗せない (.pytest の失敗記録も残さない — finalize
            # の未合格集計は failed 記録を数えるため)。
            self._emit(
                "test",
                f"pytest 環境スキップ: {gate.skip_reason or '環境要因'}",
                "done", task.task_id,
            )
            return "generated_pytest_env_skipped", gate
        self._emit(
            "test",
            f"pytest 参考結果: {'合格' if gate.ok else (gate.error or '失敗')}",
            "done" if gate.ok else "failed", task.task_id,
        )
        self._record(f"{task.task_id}.pytest", gate, attempt, kind="pytest")
        status = "generated_pytest_ok" if gate.ok else "generated_pytest_failed"
        return status, gate

    # ── spec 見直しループ (test 不合格時、有界) ──────────────────────────
    def _revision_marker(self, task_id: str) -> str | None:
        """当該タスクの spec 見直し記録 (``spec_revision_cycle: <result>``) を返す。"""
        notes = (self.workspace.read_manifest() or {}).get("stage_notes", {}) or {}
        for d in (notes.get(task_id) or {}).get("decisions", []) or []:
            if str(d).startswith("spec_revision_cycle:"):
                return str(d).split(":", 1)[1].strip()
        return None

    def _revision_recorded_sha(self, task_id: str) -> str:
        """spec 見直し記録に添えた src sha (リトライ高速パスの不変判定用)。"""
        notes = (self.workspace.read_manifest() or {}).get("stage_notes", {}) or {}
        for d in (notes.get(task_id) or {}).get("decisions", []) or []:
            if str(d).startswith("src_sha:"):
                return str(d).split(":", 1)[1].strip()
        return ""

    def _revision_budget_used(self) -> int:
        """ワークスペース全体で judge 呼出まで到達した見直しサイクル数。

        skip 系 (degraded / budget / アンカー不在) は予算を消費しない。
        """
        consuming = {
            "applied", "judged_spec_ok", "judge_unparseable", "judge_invalid",
            "judge_truncated",
        }
        notes = (self.workspace.read_manifest() or {}).get("stage_notes", {}) or {}
        used = 0
        for rec in notes.values():
            for d in (rec or {}).get("decisions", []) or []:
                text = str(d)
                if text.startswith("spec_revision_cycle:"):
                    result = text.split(":", 1)[1].strip()
                    if result in consuming:
                        used += 1
                    break
        return used

    def _record_revision(
        self, task: "TaskFactView", source_path: str, result: str, reason: str = "",
    ) -> None:
        """見直し判断を stage_notes へ記録する (回数制御・リトライ抑止・観測)。"""
        sha = ""
        try:
            f = self.workspace.file_map().get(source_path)
            sha = f.sha256 if f is not None else ""
        except Exception:  # noqa: BLE001 - 観測情報の欠落は本処理を止めない
            sha = ""
        decisions = [f"spec_revision_cycle: {result}"]
        if sha:
            decisions.append(f"src_sha: {sha}")
        self.workspace.record_stage_notes(
            task.task_id, decisions=decisions,
            open_questions=[reason] if reason else [],
        )

    async def _spec_revision_cycle(
        self, task: "TaskFactView", source_path: str, *, evidence: str, kind: str,
    ) -> bool:
        """spec 該当節を assist で点検し、欠陥なら改訂 + コード再生成する。

        戻り値 True = 改訂を適用しコードを再生成した (呼出側は再検証する)。
        ガード: per-task 1 回 (stage_notes マーカー) / ワークスペース全体
        ``max_spec_revision_rounds`` / assist degraded / アンカー不在。
        JSON 崩れ・非 dict は spec_ok 扱い (grammar 非強制モデル対策)。
        max_tokens 切断 (telemetry.truncated) は判定全体を不信として不採用
        (json_repair が切断 JSON を閉じて素通りし、途中切れの revised_section
        が spec を劣化させるため。2026-07-06 live で Component 3 節が消失)。
        改訂節は決定論ガード (長さ・Canonical 節混入・他モジュール見出し混入・
        Component 見出し数の減少) を通った場合のみ ``replace_module_section``
        で差し替える。judge プロンプトには ``os_constraint()`` を注入する
        (改訂文が OS 非対応方式や非標準ライブラリの「許容」を書き込まない
        ため)。改訂と矛盾する ``## Entry point`` 節は ``revised_entry_point``
        で随伴改訂できる (長さ・単一 H2 の決定論ガード付き、失敗しても
        Module 節の改訂は維持)。
        """
        if self.max_spec_revision_rounds <= 0:
            return False
        if self._revision_marker(task.task_id) is not None:
            return False
        if self._revision_budget_used() >= self.max_spec_revision_rounds:
            self._record_revision(task, source_path, "skipped_budget_exhausted")
            return False
        if self.assist_client is None:
            self._record_revision(task, source_path, "skipped_degraded")
            return False
        spec = self.workspace.read_spec() or ""
        found = extract_module_section(spec, source_path)
        if found is None:
            self._record_revision(task, source_path, "skipped_section_not_found")
            return False
        _, _, section = found

        self._emit(
            "test", f"仕様の該当節を点検中: {source_path}", "running", task.task_id,
        )
        src_head = (
            self.workspace.read_file(source_path, kind="src") or ""
        )[:_REVISION_SRC_HEAD_CHARS]
        prompt = _SPEC_REVISION_PROMPT.format(
            source_path=source_path,
            section=section.strip(),
            entry_section=extract_entry_point_section(spec).strip() or "(none)",
            src_head=src_head,
            kind=kind,
            evidence=evidence[-_REVISION_EVIDENCE_CHARS:],
        ) + os_constraint() + (
            f"\nWrite the prose in revised_section in {_prose_language()} "
            f"(keep the heading forms and code identifiers as-is)."
        )
        judge_telemetry: dict = {}
        try:
            data = await self.assist_client.generate_json(
                prompt, purpose="spec_revision_judge",
                max_tokens=1536, temperature=0.2, telemetry=judge_telemetry,
            )
        except Exception as exc:
            logger.warning("spec revision judge failed: %s", exc)
            self._record_revision(task, source_path, "judge_unparseable")
            return False
        if not isinstance(data, dict):
            self._record_revision(task, source_path, "judge_unparseable")
            return False
        if judge_telemetry.get("truncated"):
            logger.warning(
                "spec revision judge truncated at max_tokens for %s; "
                "discarding the revision (revised_section may be incomplete)",
                source_path,
            )
            self._record_revision(task, source_path, "judge_truncated")
            return False

        reason = str(data.get("reason", "") or "")[:500]
        if bool(data.get("spec_ok", True)):
            self._emit(
                "test", "仕様に欠陥なし (コード/テスト側の問題と判定)",
                "done", task.task_id,
            )
            self._record_revision(task, source_path, "judged_spec_ok", reason)
            return False

        revised = str(data.get("revised_section", "") or "").strip()
        heading = f"## Module: {source_path}"
        if revised and not revised.startswith("## Module:"):
            revised = f"{heading}\n{revised}"
        if (
            not revised
            or len(revised) > _REVISION_SECTION_MAX_CHARS
            or "## Canonical file list" in revised
            # `## Processing flow` は flow_render が管理する予約節。改訂節への
            # 混入は Canonical 節混入と同様に不採用。
            or "## Processing flow" in revised
            or foreign_module_headings(revised, source_path)
            # 自見出し以外の H2 混入 (## Overview 等) は replace_module_section
            # 後の span 演算前提を壊す構造汚染として不採用。
            or count_h2_headings(revised) != 1
            # Component 見出しが元節より減る改訂は構造劣化 (切断・書き漏らし)
            # とみなし不採用 (部分生成・再改訂のアンカーが失われるため)。
            or count_component_headings(revised) < count_component_headings(section)
            # 名前の多重集合包含も要求 (数は同じでもリネームされる構造劣化と、
            # 同名見出しの重複挿入による偽装を検出。追加は "fix or add" の
            # プロンプト許容範囲なので包含で判定)。
            or not Counter(component_heading_names(revised))
            >= Counter(component_heading_names(section))
        ):
            self._record_revision(task, source_path, "judge_invalid", reason)
            return False

        new_spec = replace_module_section(spec, source_path, revised)
        if new_spec is None:
            self._record_revision(task, source_path, "judge_invalid", reason)
            return False

        # Entry point 節の随伴改訂 (任意)。改訂は Module 節のみ差し替えるため、
        # 起動列に矛盾記述 (撤去した方式への言及等) が残ると、再合成フローが
        # それを忠実に反映してしまう (2026-07-07 live: S1 に curses が残存)。
        # 単一 H2 ガードで他節 (Module / Canonical / Processing flow) の
        # 巻き込みを決定論的に排除する。
        entry_revised = str(data.get("revised_entry_point", "") or "").strip()
        if entry_revised:
            if not entry_revised.startswith("## Entry point"):
                entry_revised = f"## Entry point\n{entry_revised}"
            if (
                len(entry_revised) > _REVISION_ENTRY_MAX_CHARS
                or count_h2_headings(entry_revised) != 1
            ):
                logger.warning(
                    "revised entry point rejected by deterministic guards "
                    "for %s; keeping the current entry point", source_path,
                )
            else:
                replaced = replace_entry_point_section(new_spec, entry_revised)
                if replaced is not None:
                    new_spec = replaced
                    logger.info(
                        "entry point section revised alongside %s", source_path,
                    )

        # 改訂で挙動記述が変わった可能性があるため、フローは judge の申告に
        # 依存せず常に再合成し、mermaid と Processing flow 節を同一 steps から
        # 再レンダリングしてから spec.md を書く (spec と flowchart の相違を
        # 構造的に残さない)。再合成不能時は旧フロー (両成果物とも同一 steps
        # 由来) を残す。
        if self.flowchart_enabled:
            module_list = canonical_module_list(new_spec)
            steps, flow_source = await self._synthesize_flow_steps(
                module_list, new_spec,
            )
            if steps:
                mermaid = render_mermaid(
                    steps, module_paths_from_list(module_list),
                )
                flow_section = render_flow_section(steps)
                self.workspace.write_flowchart(mermaid, task_id=task.task_id)
                new_spec = replace_flow_section(new_spec, flow_section)
                logger.info(
                    "flow re-rendered after spec revision for %s (source=%s)",
                    source_path, flow_source,
                )
            else:
                logger.warning(
                    "flow re-synthesis unavailable after spec revision for "
                    "%s; keeping the previous flow in both artifacts",
                    source_path,
                )
        self.workspace.write_spec(new_spec, task_id=task.task_id)
        self._emit(
            "test", f"仕様節を改訂 → {source_path} を再生成", "running", task.task_id,
        )

        regenerated = await self._regen_source_for_spec_change(task, source_path)
        self._record_revision(
            task, source_path,
            "applied" if regenerated else "applied_regen_failed", reason,
        )
        return regenerated

    async def _regen_source_for_spec_change(
        self, task: "TaskFactView", source_path: str,
    ) -> bool:
        """改訂 spec に基づいて source を再生成する (劣化ガード共用)。"""
        spec, flowchart = self._read_spec_and_flowchart(
            f"spec revision regen {source_path}",
        )
        description = (
            f"The design-spec section for `{source_path}` was just revised to fix "
            f"a defect found during verification. Regenerate the COMPLETE file "
            f"per the updated specification."
        )
        instruction = self._build_code_instruction(
            description, source_path, spec, flowchart,
        )
        files = await self._safe_codegen(instruction, source_path)
        code = _pick_primary(files or {}, source_path)
        if not code.strip():
            logger.warning(
                "spec revision regen produced no code for %s", source_path,
            )
            return False
        # 劣化ガードに弾かれて書き戻せなかった場合は False (再検証は不要)
        return self._write_source_if_valid(source_path, code, task.task_id)

    async def _advisory_retest(
        self, task: "TaskFactView", source_path: str,
    ) -> tuple[str | None, str, "GateResult | None"]:
        """spec 見直し後の再テスト。既存 test を再利用し pytest のみ再実行する。

        再生成された source と既存 test の契約違反が出た場合のみ、test を
        1 回だけ再生成する (LLM 節約。src は不変・権威)。
        """
        test_logical = _test_logical_for(source_path)
        test_code = self.workspace.read_file(test_logical, kind="test") or ""
        if not test_code.strip():
            # 初回で test が生成されなかった場合はフル経路をやり直す
            return await self._advisory_unit_test(task, source_path)

        violations = self._contract_violations(source_path, test_code)
        if violations:
            src_code = self.workspace.read_file(source_path, kind="src") or ""
            stem = Path(source_path).stem
            fix_instr = (
                f"The tests no longer match the ACTUAL public API of "
                f"`{source_path}` (the module was just regenerated from a revised "
                f"spec).\n\n## Contract violations\n"
                + "\n".join(f"- {v}" for v in violations) + "\n\n"
                f"## Source file `{source_path}` (THE source of truth)\n"
                f"```python\n{src_code}\n```\n\n"
                f"Rewrite `{test_logical}`: call the ACTUAL signatures/attributes "
                f"shown above (do NOT invent methods/attributes; do NOT redefine "
                f"src symbols; import them from `{stem}`). At least one "
                f"`def test_...` with asserts. Output only the test file's code."
                + _NO_PROSE_OUTPUT_CONSTRAINT
                + _code_language_constraint()
            )
            regen_code = _pick_primary(
                await self._safe_codegen(fix_instr, test_logical), test_logical,
            )
            if regen_code.strip():
                regen_code = _ensure_pytest_import(regen_code)
                regen_code = _ensure_module_import(regen_code, source_path)
            if regen_code.strip() and not _is_valid_python(regen_code):
                regen_code = _salvage_python_code(regen_code) or ""
            if regen_code.strip() and _is_valid_python(regen_code):
                remaining = self._contract_violations(source_path, regen_code)
                if not remaining:
                    self.workspace.write_file(
                        test_logical, regen_code, kind="test", stage="test",
                        task_id=task.task_id, covers=(source_path,),
                    )
                    test_code = regen_code

        sha = ""
        f = self.workspace.file_map().get(test_logical)
        if f is not None:
            sha = f.sha256
        status, gate = await self._run_advisory_pytest(task, test_logical, attempt=2)
        if status == "generated_pytest_failed" and gate is not None:
            repaired = await self._maybe_repair_literal_assertions(
                task, source_path, test_logical, test_code, gate,
            )
            if repaired is not None:
                sha_out, status, gate = repaired
                return sha_out, status, gate
        return (sha or None), status, gate

    def _contract_violations(self, source_path: str, test_code: str) -> list[str]:
        """注入された契約チェッカで test↔src API 不一致を返す (未注入時は空)。"""
        if self.contract_checker is None:
            return []
        src_files = {
            f.logical_path: (self.workspace.read_file(f.logical_path, kind="src") or "")
            for f in self.workspace.list_files(kind="src")
        }
        try:
            return list(self.contract_checker(src_files, {Path(source_path).name: test_code}))
        except Exception as exc:
            logger.warning("contract check failed: %s", exc)
            return []

    # ── ヘルパ ────────────────────────────────────────────────────────
    def _read_spec_and_flowchart(self, context: str) -> tuple[str, str]:
        """spec.md / flowchart.md を読み込む (診断ログ付き)。

        依存ゲーティング (code/test タスクは spec タスク done 後にのみ実行される)
        により、通常運用では spec.md は必ず存在する。読めない場合はワークスペース
        破損等の異常サインなので、黙ってフォールバックせず warning を残す。
        """
        raw_spec = self.workspace.read_spec()
        if raw_spec is None:
            logger.warning(
                "spec.md not found for %s (dependency gating should guarantee "
                "spec is done first) — proceeding with empty spec", context,
            )
        raw_flowchart = self.workspace.read_flowchart()
        if raw_flowchart is None and self.flowchart_enabled:
            logger.warning(
                "flowchart.md not found for %s despite flowchart_enabled=True "
                "— proceeding without it", context,
            )
        return raw_spec or "", raw_flowchart or ""

    def _sibling_src(self, source_path: str) -> dict[str, str]:
        """``source_path`` 以外の既コミット src ファイル {logical_path: code} を返す。"""
        return {
            f.logical_path: (self.workspace.read_file(f.logical_path, kind="src") or "")
            for f in self.workspace.list_files(kind="src")
            if f.logical_path != source_path
        }

    async def _safe_codegen(self, instruction: str, file_path: str) -> dict[str, str]:
        result, _ = await self._codegen_or_error(instruction, file_path)
        return result

    async def _codegen_or_error(
        self, instruction: str, file_path: str,
    ) -> tuple[dict[str, str], str | None]:
        """``_safe_codegen`` 相当だが失敗理由 (例外文字列) も返す。

        advisory テスト生成のように「インフラ障害で失敗した」ことと「生成が
        (意図通り) 空だった」ことを呼び出し側で区別する必要がある箇所で使う。
        """
        try:
            return (await self.codegen(instruction, file_path) or {}), None
        except Exception as exc:
            logger.warning("staged codegen failed: %s", exc)
            return {}, str(exc)

    def _apply_source_fixes(
        self, files: dict[str, str], source_path: str, task_id: str,
    ) -> bool:
        """スモーク修正生成物から source のみを書き戻す (ガード付き・test には触れない)。

        1 件でも有効な書き戻しがあれば True。False は「修復応答が空 / 非 Python /
        縮小ガード棄却」で src が不変のままであることを意味する (呼出側は残りの
        修復 round を打ち切ってよい)。
        """
        src_name = Path(source_path).name
        wrote = False
        for path, code in (files or {}).items():
            if not code or not code.strip():
                continue
            name = Path(path).name
            if path == source_path or name == src_name or len(files) == 1:
                wrote = self._write_source_if_valid(source_path, code, task_id) or wrote
        return wrote

    def _write_source_if_valid(self, source_path: str, code: str, task_id: str) -> bool:
        """source 上書きは (a) 有効な Python かつ (b) 極端に縮小しない ときのみ。

        非コード (markdown 等) や、機能を削ったスタブ (例: 175 行→49 行) で動作する
        生成コードを破壊する事故を防ぐ。書き込んだら True、ガード棄却なら False。
        """
        if not _is_valid_python(code):
            salvaged = _salvage_python_code(code)
            if salvaged is None:
                logger.warning(
                    "repair returned non-Python for %s; keeping existing code",
                    source_path,
                )
                return False
            logger.warning(
                "repair response for %s contained non-code prose; salvaged the "
                "embedded Python (%d -> %d chars)",
                source_path, len(code), len(salvaged),
            )
            code = salvaged
        old = self.workspace.read_file(source_path, kind="src") or ""
        if old and len(code) < 0.5 * len(old):
            logger.warning(
                "repair drastically shrank %s (%d -> %d chars); keeping existing code",
                source_path, len(old), len(code),
            )
            return False
        self.workspace.write_file(
            source_path, code, kind="src", stage="code", task_id=task_id,
        )
        return True

    def _record(
        self, task_id: str, gate: GateResult, attempt: int, *, kind: str,
    ) -> None:
        """ゲート結果を manifest / tests/_runs へ記録する。

        ``kind`` は出所 (``"pytest"`` = 生成テストを実行した / ``"smoke"`` =
        静的ゲートのみ)。``tests_passing`` の集計が両者を区別するために必須。
        """
        tail = (gate.stdout_tail or gate.stderr_tail or "")[-2000:]
        self.workspace.record_test_result(StageTestResult(
            task_id=task_id, passed=gate.ok,
            failed_count=0 if gate.ok else 1,
            attempt=attempt, summary=(gate.error or "passed"),
            output_tail=tail, ran_at=time.time(), kind=kind,
        ))


def _smoke_gate_result(errors: list[str], warnings: list[str]) -> GateResult:
    """import スモーク結果を GateResult 化する (failure_pattern 記録/可視化用)。"""
    ok = not errors
    return GateResult(
        name="import_smoke", ok=ok, skipped=False,
        returncode=0 if ok else 1, duration_ms=0,
        stdout_tail="\n".join(errors)[-2000:],
        stderr_tail="\n".join(warnings)[-500:],
        error=None if ok else f"{len(errors)} smoke error(s)",
    )


def _artifact(logical_path: str, sha256: str) -> ArtifactEntry:
    return ArtifactEntry(
        path=logical_path, diff_sha1=(sha256[:12] if sha256 else ""),
        lines_added=0, lines_removed=0, action_kind="edit_file",
    )


def _single_gate_outcome(gate: GateResult) -> QualityGateOutcome:
    return QualityGateOutcome(
        ok=gate.ok, results=(gate,),
        failed=() if gate.ok else (gate.name,),
        skipped=(gate.name,) if gate.skipped else (),
    )
