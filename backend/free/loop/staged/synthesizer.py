"""staged コーディングのタスクグラフ合成。

ユーザーのコーディング要求 (例: 「Xを作って」) を、アシストモデルに粗計画
(``summary`` + ``modules[]``) として返させ、Python 側で **決定的に**
spec → code → test の task ファクト三層へ展開する。

設計方針:
- LLM には粗計画のみ返させる (purpose=``coding_task_graph``、schema は
  ``PURPOSE_SCHEMAS`` から自動解決)。三層展開・依存配線・slug 衝突回避は
  本モジュールが Python で行い、非循環な依存グラフを保証する。
- アシスト未接続 / 解析失敗 / モジュール 0 件 のときは ``[]`` を返し、呼出側
  (チャットディスパッチ) に旧 longform 経路へのフォールバックを指示する。
- 依存配線: code タスクは spec タスクに依存し、test タスクは対応する code
  タスクに依存する。code タスク同士は相互依存させない (cross-file import 配線は
  code 工程内の既存 ``import_wirer`` が担うため、直列化・循環を避ける)。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from backend.free.loop.driver import make_task_fact
from backend.free.memory.types import SemanticFact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.llm.assist_client import AssistModelClient

logger = get_logger("loop.staged.synthesizer")

SPEC_TASK_ID = "spec_root"
"""spec タスクの固定 task_id。全 code タスクの依存先になる。"""

# salience: spec(0.9) > code(0.7..) > test(0.5)。逼迫時 (max_iterations 到達) でも
# spec→code を優先し、test を後回しにする (pick_next_task_with_deps は salience 降順)。
_SPEC_SALIENCE = 0.9
_CODE_SALIENCE = 0.7
_TEST_SALIENCE = 0.5
# code タスク間は hard 依存を張らない (依存先 failed で永久ブロックになるため)。
# 代わりに depends_on DAG 上の深さで salience を下げ、leaf モジュール→エントリの
# 順で生成させる。エントリは兄弟の実 API が既コミットの状態で最後に生成され、
# executor 側の兄弟 API 注入と相俟って API ドリフトを抑える。floor は test を
# 必ず上回るよう _TEST_SALIENCE より上に置く。
_CODE_DEPTH_STEP = 0.02
_CODE_SALIENCE_FLOOR = 0.55

_SYNTHESIS_PROMPT = """\
You are a software design planner. Given a coding request, produce a coarse \
plan as a JSON object with two fields:
- "summary": a concise design overview of the whole program (what it does, the \
overall structure, key data shapes / interfaces). This becomes the shared design spec.
- "modules": an array of the source files to implement. Each entry is an object \
{{"file_path": <relative path>, "purpose": <one line>, "depends_on": [<other file_path>...]}}.

IMPORTANT rules:
- Implement EXACTLY the program the user requested, using the user's own terms. \
NEVER substitute a different or merely "similar" program.
- "file_path" is the program's OWN relative module path (e.g. "main.py", \
"parser.py", "models.py"). Use forward slashes for sub-packages (e.g. "core/engine.py"). \
Do NOT use absolute paths and do NOT invent an unrelated output directory.
- One source file = ONE module entry. A simple single-file program is ONE module.
- Keep the module list minimal — only split into multiple files when the program \
genuinely needs separate concerns. Prefer fewer, cohesive modules.
- "depends_on" lists OTHER file_path values this module imports from (for code-stage \
import wiring). Leave it empty when there is no intra-program dependency. Do NOT \
create circular dependencies.

Request:
{request}

Output the JSON object and nothing else."""


def _slug(file_path: str) -> str:
    """file_path を task_id に使える ASCII slug に正規化する。"""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", file_path.strip().lower()).strip("_")
    return s or "mod"


def _unique(base: str, seen: set[str]) -> str:
    """``seen`` 内で一意な識別子を返す (衝突時は ``_2`` / ``_3`` ... を付与)。"""
    if base not in seen:
        seen.add(base)
        return base
    i = 2
    while f"{base}_{i}" in seen:
        i += 1
    out = f"{base}_{i}"
    seen.add(out)
    return out


def _dependency_depths(modules: list[dict]) -> dict[str, int]:
    """depends_on DAG 上の各モジュールの深さ (= 最長の依存連鎖長) を返す。

    leaf (依存なし) は 0。循環や未知の依存先は深さに寄与させない (0 打ち切り)。
    生成順制御 (salience) のヒント用途なので厳密な topo ソートでなくてよい。
    """
    by_path = {m["file_path"]: m["depends_on"] for m in modules}
    depths: dict[str, int] = {}

    def visit(fp: str, on_path: frozenset[str]) -> int:
        if fp in depths:
            return depths[fp]
        if fp in on_path or fp not in by_path:
            return 0  # 循環 or 外部参照 → 寄与なし
        d = 0
        for dep in by_path[fp]:
            d = max(d, 1 + visit(dep, on_path | {fp}))
        depths[fp] = d
        return d

    for fp in by_path:
        visit(fp, frozenset())
    return depths


def _render_module_list(modules: list[dict]) -> str:
    """spec タスク description 用にモジュール一覧を整形する。"""
    lines: list[str] = []
    for m in modules:
        fp = str(m.get("file_path", "")).strip()
        purpose = str(m.get("purpose", "")).strip()
        deps = [str(d).strip() for d in (m.get("depends_on") or []) if str(d).strip()]
        dep_note = f" (depends on: {', '.join(deps)})" if deps else ""
        lines.append(f"- {fp}: {purpose}{dep_note}")
    return "\n".join(lines)


def _normalize_modules(raw_modules: object) -> list[dict]:
    """LLM 応答の modules を検証し、file_path 重複を除いた dict リストに正規化する。"""
    if not isinstance(raw_modules, list):
        return []
    out: list[dict] = []
    seen_paths: set[str] = set()
    for item in raw_modules:
        if not isinstance(item, dict):
            continue
        fp = str(item.get("file_path", "")).strip()
        if not fp or fp in seen_paths:
            continue
        seen_paths.add(fp)
        deps = item.get("depends_on") or []
        out.append({
            "file_path": fp,
            "purpose": str(item.get("purpose", "")).strip(),
            "depends_on": [str(d).strip() for d in deps if str(d).strip()],
        })
    return out


async def synthesize_coding_task_graph(
    *,
    request: str,
    project_id: str,
    assist_client: "AssistModelClient | None",
    include_tests: bool = True,
    debug_logger: "DebugLogger | None" = None,
) -> list[SemanticFact]:
    """コーディング要求を spec/code/test の task ファクト群へ分解する。

    Args:
        request: ユーザーのコーディング指示。
        project_id: タスクを所属させる project_id。
        assist_client: アシストモデル。``None`` (degraded) なら ``[]`` を返す。
        include_tests: ``False`` なら test 工程を生成しない (config の
            ``coding.staged.test_stage_enabled=false`` 用)。
        debug_logger: 任意。

    Returns:
        spec → code* → test* の順に並んだ ``task`` 型 SemanticFact リスト
        (まだストアには追加されていない)。アシスト未接続 / 解析失敗 /
        モジュール 0 件 の場合は ``[]`` (= 呼出側で longform へフォールバック)。
    """
    if assist_client is None:
        logger.info("synthesize_coding_task_graph: assist_client is None — fallback")
        return []
    if not project_id:
        raise ValueError("project_id must be non-empty")

    prompt = _SYNTHESIS_PROMPT.format(request=request.strip())
    graph_telemetry: dict = {}
    try:
        result = await assist_client.generate_json(
            prompt,
            purpose="coding_task_graph",
            max_tokens=1024,
            temperature=0.3,
            telemetry=graph_telemetry,
        )
    except Exception as exc:  # noqa: BLE001 — 失敗時は longform へフォールバック
        logger.warning("coding_task_graph synthesis failed: %s", exc)
        return []
    # coarse plan は通常 1024 に収まるが、切断時はモジュール欠落の可能性を可視化。
    if graph_telemetry.get("truncated"):
        logger.warning(
            "coding_task_graph truncated: some modules may be missing from the "
            "staged plan (request too large for the planning budget)",
        )

    summary = str((result or {}).get("summary", "")).strip()
    modules = _normalize_modules((result or {}).get("modules"))
    if not modules:
        logger.info("coding_task_graph returned no modules — fallback to longform")
        return []

    facts: list[SemanticFact] = []
    seen_ids: set[str] = {SPEC_TASK_ID}

    # 1. spec タスク (依存なし、最初に必ず実行される)
    spec_desc = summary or f"Design specification for: {request.strip()}"
    module_block = _render_module_list(modules)
    if module_block:
        spec_desc = f"{spec_desc}\n\nModules to implement:\n{module_block}"
    facts.append(make_task_fact(
        project_id=project_id,
        task_id=SPEC_TASK_ID,
        title="設計仕様の作成",
        description=spec_desc,
        depends_on=[],
        salience=_SPEC_SALIENCE,
        stage="spec",
    ))

    # 2. code タスク (モジュール毎、spec に依存) + 3. test タスク (code に依存)
    depths = _dependency_depths(modules)
    for m in modules:
        fp = m["file_path"]
        slug = _unique(f"code_{_slug(fp)}", seen_ids)
        deps = m["depends_on"]
        dep_note = (
            f"\n\nThis module imports from: {', '.join(deps)}." if deps else ""
        )
        code_salience = max(
            _CODE_SALIENCE - depths.get(fp, 0) * _CODE_DEPTH_STEP,
            _CODE_SALIENCE_FLOOR,
        )
        facts.append(make_task_fact(
            project_id=project_id,
            task_id=slug,
            title=fp,
            description=(m["purpose"] or f"Implement {fp}") + dep_note,
            depends_on=[SPEC_TASK_ID],
            salience=code_salience,
            source_path=fp,
            stage="code",
        ))
        if include_tests:
            test_slug = _unique(f"test_{_slug(fp)}", seen_ids)
            facts.append(make_task_fact(
                project_id=project_id,
                task_id=test_slug,
                title=f"test: {fp}",
                description=f"Generate and run pytest tests for {fp}.",
                depends_on=[slug],
                salience=_TEST_SALIENCE,
                source_path=fp,
                stage="test",
            ))

    logger.info(
        "coding_task_graph synthesized: %d modules -> %d tasks (project=%s)",
        len(modules), len(facts), project_id,
    )
    return facts
