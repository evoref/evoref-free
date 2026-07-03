"""staged コーディングの stage 対応 TaskExecutor。

``LoopDriver`` から 1 タスクずつ呼ばれ、``task.stage`` で分岐する:

- ``spec`` : アシストモデルで設計仕様 (spec.md) を生成し workspace に永続化。
- ``code`` : **base コーディングモデル** (注入された codegen 委譲) で当該モジュール
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
import platform
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.loop.executor import ArtifactEntry, ExecutionOutcome
from backend.free.loop.quality_gate import GateResult, QualityGateOutcome
from backend.free.loop.staged.synthesizer import MODULE_LIST_MARKER
from backend.free.loop.staged.test_runner import StagedTestRunner
from backend.free.loop.staged.workspace import WorkspaceManager, StageTestResult
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.loop.driver import TaskFactView
    from backend.free.loop.events import LoopEventBus

logger = get_logger("loop.staged.executor")

# (instruction, file_path) -> {logical_path: code}。base コーディングモデル経由の
# コード生成委譲。file_path は生成対象の論理パス (戻り値の主キー) を明示する。
CodegenDelegate = Callable[[str, str], Awaitable[dict[str, str]]]

_SPEC_PROMPT = """\
Write a detailed software design specification in Markdown for the program below.
Cover: purpose, module/file breakdown, key data structures, public interfaces \
(function/class signatures), and the entry point. Be concrete enough that each \
module can be implemented independently against this spec. If the program design \
below lists specific file paths, use those EXACT paths verbatim in the module/file \
breakdown — do not rename, merge, split, or add files.

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

_FLOWCHART_PROMPT = """\
You are a software architect. From the design specification below, produce a Mermaid \
flowchart (start with "flowchart TD") showing the modules, their dependencies, and the \
control/data flow. Return JSON with a "mermaid" key (the flowchart code, NO ``` fences).

Design specification:
{spec}
"""

# フローチャート合成入力の合計予算 (module_list を優先し、残りを narrative に充てる)。
_FLOWCHART_CONTEXT_CHARS = 4000


def _os_constraint() -> str:
    """spec/code 生成に注入する実行 OS 制約 (OS 非対応 stdlib の生成を抑止)。"""
    return (
        f"\n\nTarget runtime OS: {platform.system() or 'unknown'}. Use only "
        f"cross-platform standard-library and APIs available on that OS. Avoid "
        f"OS-specific modules unavailable there (e.g. `curses`/`fcntl`/`termios` "
        f"on Windows); prefer portable approaches so the program actually runs."
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


# spec 本文再生成時の max_tokens 上限 (backend/schemas/coding.py の le=8192 と整合)。
_SPEC_MAX_TOKENS_CEILING = 8192


def _extract_module_list(description: str) -> str:
    """spec タスクの description から正準モジュール一覧ブロックを取り出す。

    ``synthesizer.synthesize_coding_task_graph`` が ``MODULE_LIST_MARKER`` 以降に
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


@dataclass
class StagedCodingExecutor:
    """stage 別 TaskExecutor。``LoopDriver(executor=...)`` に差し込む。

    Args:
        workspace: 工程間ハンドオフ用 :class:`WorkspaceManager`。
        assist_client: spec 工程で使うアシスト。``None`` なら spec は description
            をそのまま spec.md に書く degraded 動作。
        codegen: base コーディングモデル経由の生成委譲
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
    max_repair_rounds: int = 2
    max_test_regen_rounds: int = 2
    spec_max_tokens: int = 1536
    spec_timeout_sec: float = 120.0
    flowchart_enabled: bool = True
    event_bus: "LoopEventBus | None" = None
    debug_logger: "DebugLogger | None" = None
    name: str = "staged_coding"

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
        spec_text = await self._generate_spec_doc(task.description)
        if not spec_text:
            spec_text = task.description
        module_list = _extract_module_list(task.description)

        # 設計フローチャート (mermaid) を合成し flowchart.md へ保存。code/test 工程は
        # flowchart.md を読んで整合的に生成する。spec.md には埋め込まない
        # (flowchart.md を単一の真実とし、code/test 工程側の flow_block 注入との
        # 二重埋め込みを避ける)。
        flowchart = ""
        if self.flowchart_enabled and self.assist_client is not None:
            self._emit("spec", "フローチャートを生成中", "running", task.task_id)
            flowchart = await self._synthesize_flowchart(module_list, spec_text)
            if flowchart.strip():
                self.workspace.write_flowchart(flowchart, task_id=task.task_id)

        final_spec = spec_text
        if module_list:
            # spec_text は task.description とは別の LLM 呼び出しの自由記述であり、
            # 与えたファイル一覧を忠実に再現する保証が無い。code タスクの
            # source_path と必ず一致する正準一覧を決定的に追記し、コード生成が
            # 常に正しいファイル一覧を参照できるようにする。
            final_spec = (
                f"{spec_text}\n\n"
                f"## Canonical file list (authoritative — generated files MUST "
                f"match these exact paths)\n{module_list}\n"
            )
        wf = self.workspace.write_spec(final_spec, task_id=task.task_id)
        self.workspace.upsert_task(
            task_id=task.task_id, title=task.title, stage="spec",
            status="done", depends_on=task.depends_on,
        )
        self._emit(
            "spec",
            "設計仕様" + ("・フローチャート" if flowchart.strip() else "") + "を確定",
            "done", task.task_id,
        )
        return ExecutionOutcome(
            status="success",
            artifacts=(_artifact(wf.logical_path, wf.sha256),),
            notes={"executor": self.name, "stage": "spec",
                   "chars": str(len(final_spec)),
                   "flowchart": "true" if flowchart.strip() else "false"},
        )

    async def _generate_spec_doc(self, description: str) -> str:
        """spec 本文をアシスト生成する。max_tokens 切断時のみ予算を倍に広げ 1 回再生成。

        ``generate()`` は finish_reason='length' を telemetry に露出しないため、
        応答 dict から直接 finish_reason を読み本文の途中切れを検知する (検知しないと
        §7 のように文末で切れた spec が後続コード生成の土台になり破損が連鎖する)。
        遅い iGPU を考慮し再生成は切断時のみ・明示 timeout 付きに限定する。
        ``None`` (degraded) / 失敗時は空文字を返し、呼出側が description に倒す。
        """
        if self.assist_client is None:
            return ""
        msgs = [{"role": "user", "content":
                 _SPEC_PROMPT.format(description=description) + _os_constraint()}]
        try:
            resp = await self.assist_client.generate(
                msgs, purpose="coding_spec_doc", max_tokens=self.spec_max_tokens,
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
                msgs, purpose="coding_spec_doc", max_tokens=retry_tokens,
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

    async def _synthesize_flowchart(self, module_list: str, spec_text: str) -> str:
        """設計仕様から mermaid フローチャートを合成する (失敗時は空文字)。

        入力は正準モジュール一覧 (``module_list``、常に完全) を先頭に置き、残り予算を
        spec 本文の narrative に充てる。旧実装は spec_text の先頭 4000 文字のみを
        渡していたため、長い spec では narrative の途中で切れ、末尾のモジュールが
        フローチャートから欠落しうった。module_list は 1 モジュール 1 行の短い
        列挙のため、全モジュールがほぼ確実に予算内に収まる。

        既存の ``flowchart_synthesis`` purpose / ``FlowchartSpec`` を再利用する。
        """
        if self.assist_client is None:
            return ""
        if module_list:
            remaining = max(_FLOWCHART_CONTEXT_CHARS - len(module_list), 0)
            context = f"{module_list}\n\n{spec_text[:remaining]}"
        else:
            context = spec_text[:_FLOWCHART_CONTEXT_CHARS]
        try:
            data = await self.assist_client.generate_json(
                _FLOWCHART_PROMPT.format(spec=context),
                max_tokens=1024, temperature=0.3,
                purpose="flowchart_synthesis",
            )
            if isinstance(data, dict):
                return str(data.get("mermaid", "") or "").strip()
        except Exception as exc:
            logger.warning("flowchart synthesis failed: %s", exc)
        return ""

    # ── code 工程 (base モデル) ───────────────────────────────────────
    async def _run_code(self, task: "TaskFactView") -> ExecutionOutcome:
        source_path = task.source_path or f"{task.task_id}.py"
        self._emit("code", f"コード生成中: {source_path}", "running", task.task_id)
        spec, flowchart = self._read_spec_and_flowchart(f"code stage {source_path}")
        api_block = build_sibling_api_block(self._sibling_src(source_path))
        if api_block:
            contract_block = (
                f"## Already-implemented sibling modules — use their EXACT public API\n"
                f"These modules already exist in this program. Import from them and call "
                f"the EXACT class names, method names, and constructor signatures shown "
                f"below — do NOT invent different names or signatures. `{source_path}` "
                f"MUST be consistent with them.\n\n{api_block}\n\n"
            )
        else:
            # 兄弟が無い場合 (単一モジュール or 最初に生成されるモジュール)。
            # 「sibling modules が使う API を定義せよ」と煽るとスタブ化を招くため、
            # 完全実装を促す中立な文言にする。
            contract_block = (
                "## No sibling modules generated yet\n"
                "Implement this file completely with real, working logic and clear, "
                "importable public classes/functions.\n\n"
            )
        flow_block = (
            f"## Module architecture diagram\n```mermaid\n{flowchart}\n```\n\n"
            if flowchart.strip() else ""
        )
        instruction = (
            f"{task.description}\n\n"
            f"## Shared design specification\n{spec}\n\n"
            f"{flow_block}"
            f"{contract_block}"
            f"Produce the COMPLETE, fully working contents of the single file "
            f"`{source_path}` implementing the above with real logic — do NOT leave "
            f"function/method bodies as stubs (no bare `pass`, `# TODO`/`...` "
            f"placeholders, or NotImplementedError). Use the EXACT Python types "
            f"stated in the shared design specification's data structures — do "
            f"NOT substitute a different type that merely behaves similarly "
            f"(e.g. if a field is specified as holding `int` values, it must "
            f"hold real `int`s, not `bool`). Output only this file's code."
            + _os_constraint()
        )
        try:
            files = await self.codegen(instruction, source_path)
        except Exception as exc:
            logger.warning("code generation failed for %s: %s", source_path, exc)
            self._fail_task(task, f"codegen error: {exc}")
            return ExecutionOutcome(
                status="failure", error=f"codegen error: {exc}",
                notes={"executor": self.name, "stage": "code"},
            )
        code = _pick_primary(files or {}, source_path)
        if not code.strip():
            logger.warning("staged code stage produced no code for %s", source_path)
            self._fail_task(task, "empty code generation")
            self._emit("code", f"コード生成失敗 (空): {source_path}", "failed", task.task_id)
            return ExecutionOutcome(
                status="failure", error="empty code generation",
                notes={"executor": self.name, "stage": "code"},
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
        return ExecutionOutcome(
            status="success",
            artifacts=(_artifact(wf.logical_path, wf.sha256),),
            notes={"executor": self.name, "stage": "code", "file": source_path},
        )

    # ── test 工程 (スモーク検証ゲート + advisory ユニットテスト) ─────────
    async def _run_test(self, task: "TaskFactView") -> ExecutionOutcome:
        """合否は決定論的 import スモークで決める。ユニットテストは advisory。

        コードは成果物 (権威)。スモークの実エラー時のみコードを修正し (劣化ガード
        付き)、ユニットテストはソースを一切書き換えない。
        """
        source_path = task.source_path or ""
        src_code = self.workspace.read_file(source_path, kind="src") if source_path else None
        if not src_code:
            self._emit("test", "ソース未検出のためスキップ", "failed", task.task_id)
            return ExecutionOutcome(
                status="skipped", error="source not found for test stage",
                notes={"executor": self.name, "stage": "test", "source": source_path},
            )

        # 1) 決定論的スモークゲート (合否を決める。実エラー時のみコード修正)
        smoke_ok, gate, errors = await self._smoke_gate(task, source_path, src_code)

        # 2) advisory ユニットテスト (合否非ゲート・ソース不変)
        test_sha = await self._advisory_unit_test(task, source_path)

        err_summary = ("; ".join(errors)[:300] or "smoke errors") if errors else None
        self.workspace.upsert_task(
            task_id=task.task_id, title=task.title, stage="test",
            status="done" if smoke_ok else "failed",
            depends_on=task.depends_on, last_error=err_summary,
        )
        artifacts: tuple = ()
        if test_sha is not None:
            artifacts = (_artifact(f"tests/{_test_logical_for(source_path)}", test_sha),)
        return ExecutionOutcome(
            status="success" if smoke_ok else "failure",
            gate_outcome=_single_gate_outcome(gate) if gate is not None else None,
            artifacts=artifacts,
            error=None if smoke_ok else err_summary,
            notes={"executor": self.name, "stage": "test",
                   "smoke": "ok" if smoke_ok else "errors"},
        )

    # ── スモークゲート ─────────────────────────────────────────────────
    async def _smoke_gate(
        self, task: "TaskFactView", source_path: str, src_code: str,
    ) -> tuple[bool, "GateResult | None", list[str]]:
        """全 src を import スモークし、実エラーがあればコードを修正する (有界)。

        Returns ``(ok, gate, errors)``。``smoke_runner`` 未注入時はスキップ=成功。
        """
        if self.smoke_runner is None:
            self._emit("test", "スモーク検証スキップ (degraded)", "done", task.task_id)
            return True, None, []

        self._emit("test", "import スモーク検証中", "running", task.task_id)
        errors, warnings = await self._smoke_once()
        attempt = 1
        gate = _smoke_gate_result(errors, warnings)
        self._record(task.task_id, gate, attempt)
        self._emit_smoke(task, errors, warnings, attempt)

        # spec/flowchart はループ外で 1 度だけ読む (毎試行同一、I/O 削減)。修正指示に
        # 含めないと base モデルは「エラーを消すこと」だけを目標にでき、spec の
        # ロジック (境界条件・データ構造・エントリポイント仕様等) を無視した修正
        # (機能削除・別ロジックへの置換) を許してしまう。
        spec, flowchart = self._read_spec_and_flowchart(f"smoke repair {source_path}")
        spec_block = f"## Shared design specification\n{spec}\n\n" if spec.strip() else ""
        repair_flow_block = (
            f"## Module architecture diagram\n```mermaid\n{flowchart}\n```\n\n"
            if flowchart.strip() else ""
        )

        while errors and attempt <= self.max_repair_rounds:
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
                f"Fix `{source_path}` so it imports and runs without these errors, "
                f"while remaining faithful to the design specification above. "
                f"KEEP ALL existing functionality — do NOT remove features, classes, or "
                f"functions, and do NOT shrink the program to a stub. "
                f"Output only the corrected `{source_path}` code."
            )
            fixed = await self._safe_codegen(repair_instr, source_path)
            self._apply_source_fixes(fixed, source_path, task.task_id)
            attempt += 1
            errors, warnings = await self._smoke_once()
            gate = _smoke_gate_result(errors, warnings)
            self._record(task.task_id, gate, attempt)
            self._emit_smoke(task, errors, warnings, attempt)

        return (not errors), gate, errors

    async def _smoke_once(self) -> tuple[list[str], list[str]]:
        """現在の全 src を import スモークし (errors, warnings) を返す。"""
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

    # ── advisory ユニットテスト (合否非ゲート・ソース不変) ───────────────
    async def _advisory_unit_test(
        self, task: "TaskFactView", source_path: str,
    ) -> str | None:
        """参考用ユニットテストを 1 回生成・実行する (ソースは書き換えない)。"""
        if self.test_runner is None:
            return None
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
            f"Output only the test file's code."
        )
        test_code = _pick_primary(
            await self._safe_codegen(gen_instr, test_logical), test_logical,
        )
        if not test_code.strip():
            self._emit("test", "ユニットテスト生成なし (参考)", "done", task.task_id)
            return None

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
            )
            regen_code = _pick_primary(
                await self._safe_codegen(fix_instr, test_logical), test_logical,
            )
            if not regen_code.strip():
                break
            test_code = regen_code
            violations = self._contract_violations(source_path, test_code)

        # 構文的に無効な test (中間切断等) は書き込む前に弾く。契約チェッカは構文
        # エラーを例外握り潰しで違反ゼロ扱いにする (_contract_violations) ため、この
        # ゲートが無いと壊れたテストファイルがそのまま永続化・pytest 実行されうる。
        if not _is_valid_python(test_code):
            logger.warning(
                "advisory unit test for %s is not valid Python; treating as "
                "no test generated", source_path,
            )
            self._emit(
                "test", f"ユニットテスト生成失敗 (構文エラー、参考): {test_logical}",
                "failed", task.task_id,
            )
            return None

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
            return wf.sha256

        self._emit("test", f"pytest 実行 (参考): {test_logical}", "running", task.task_id)
        try:
            gate = await asyncio.to_thread(
                self.test_runner.run, test_logical_path=test_logical,
            )
            self._emit(
                "test",
                f"pytest 参考結果: {'合格' if gate.ok else (gate.error or '失敗')}",
                "done" if gate.ok else "failed", task.task_id,
            )
        except Exception as exc:
            logger.warning("advisory pytest run failed: %s", exc)
        return wf.sha256

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
        try:
            return await self.codegen(instruction, file_path) or {}
        except Exception as exc:
            logger.warning("staged codegen failed: %s", exc)
            return {}

    def _apply_source_fixes(
        self, files: dict[str, str], source_path: str, task_id: str,
    ) -> None:
        """スモーク修正生成物から source のみを書き戻す (ガード付き・test には触れない)。"""
        src_name = Path(source_path).name
        for path, code in (files or {}).items():
            if not code or not code.strip():
                continue
            name = Path(path).name
            if path == source_path or name == src_name or len(files) == 1:
                self._write_source_if_valid(source_path, code, task_id)

    def _write_source_if_valid(self, source_path: str, code: str, task_id: str) -> None:
        """source 上書きは (a) 有効な Python かつ (b) 極端に縮小しない ときのみ。

        非コード (markdown 等) や、機能を削ったスタブ (例: 175 行→49 行) で動作する
        生成コードを破壊する事故を防ぐ。
        """
        if not _is_valid_python(code):
            logger.warning(
                "repair returned non-Python for %s; keeping existing code", source_path,
            )
            return
        old = self.workspace.read_file(source_path, kind="src") or ""
        if old and len(code) < 0.5 * len(old):
            logger.warning(
                "repair drastically shrank %s (%d -> %d chars); keeping existing code",
                source_path, len(old), len(code),
            )
            return
        self.workspace.write_file(
            source_path, code, kind="src", stage="code", task_id=task_id,
        )

    def _record(self, task_id: str, gate: GateResult, attempt: int) -> None:
        tail = (gate.stdout_tail or gate.stderr_tail or "")[-2000:]
        self.workspace.record_test_result(StageTestResult(
            task_id=task_id, passed=gate.ok,
            failed_count=0 if gate.ok else 1,
            attempt=attempt, summary=(gate.error or "passed"),
            output_tail=tail, ran_at=time.time(),
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
