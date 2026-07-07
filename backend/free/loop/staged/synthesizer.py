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

import platform
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

MODULE_LIST_MARKER = "\n\nModules to implement:\n"
"""spec タスク description 内でモジュール一覧ブロックの直前に置くマーカー。

``executor._extract_module_list`` がこのマーカーを検索して正準ファイル一覧
(= 各 code タスクの ``source_path`` と一致する ``file_path`` 群) を取り出し、
spec.md へ決定的に付記する (LLM が自由記述で spec 本文を再生成する際に
ファイル名がドリフトしても、正準一覧は必ず spec.md に残る)。
"""

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
# LLM が depends_on を申告しない場合、深さヒントが効かずエントリモジュールが
# 先に生成され、未実装の兄弟 API を発明する (2026-07-06 live: main.py が
# game_state.py より先に生成され Game.setup_event_loop を幻覚)。エントリらしい
# stem は depends_on の有無に関わらず全 code タスクの最後 (ただし test より上)
# へ決定論的に落とす。
_ENTRY_STEMS = frozenset({"main", "app", "__main__", "cli", "run"})
_ENTRY_SALIENCE = 0.52

_SYNTHESIS_PROMPT = """\
You are a software design planner. Given a coding request, produce a coarse \
plan as a JSON object with two fields:
- "summary": a concise design overview of the whole program (what it does, the \
overall structure, key data shapes / interfaces). This becomes the shared design spec.
- "modules": an array of the source files to implement. Each entry is an object \
{{"file_path": <relative path>, "purpose": <2-4 sentences: the module's \
responsibilities, its key public classes/functions, and its inputs/outputs>, \
"key_components": [<names of the public classes / top-level functions this \
module will expose>], "depends_on": [<other file_path>...]}}.

IMPORTANT rules:
- Implement EXACTLY the program the user requested, using the user's own terms. \
NEVER substitute a different or merely "similar" program.
- "file_path" is the program's OWN relative module path (e.g. "main.py", \
"parser.py", "models.py"). Use forward slashes for sub-packages (e.g. "core/engine.py"). \
Do NOT use absolute paths and do NOT invent an unrelated output directory.
- One source file = ONE module entry. A simple single-file program is ONE module.
- Keep the module list minimal — only split into multiple files when the program \
genuinely needs separate concerns. Prefer fewer, cohesive modules.
- "key_components" lists 2-4 names per module (one class = one component; a small \
group of related helper functions may count as one component).
- "depends_on" lists OTHER file_path values this module imports from (for code-stage \
import wiring). Leave it empty when there is no intra-program dependency. Do NOT \
create circular dependencies.

Request:
{request}

Output the JSON object and nothing else."""


def os_constraint() -> str:
    """planner/spec/code 生成に注入する実行 OS 制約 (OS 非対応 stdlib の生成を抑止)。

    UI 技術の選定はタスクグラフ合成 (planner) の時点で確定し、下流の spec/code
    への制約注入では覆らないため、本 synthesizer のプロンプトにも必ず注入する
    (2026-07-06 live: planner が curses を採用し Windows で起動不能になった)。
    executor (spec/code/部分生成) も同一の制約文言を共有する。
    """
    return (
        f"\n\nTarget runtime OS: {platform.system() or 'unknown'}. Use only "
        f"cross-platform standard-library and APIs available on that OS. Avoid "
        f"OS-specific modules unavailable there (e.g. `curses`/`fcntl`/`termios` "
        f"on Windows); prefer portable approaches so the program actually runs."
    )


# OS 別の「ターゲット OS で import 不可な stdlib」既知リスト (SSOT)。
# planner 出力 (summary / module purpose) への決定論スクリーンに使う。
_OS_UNAVAILABLE_STDLIB: dict[str, tuple[str, ...]] = {
    "Windows": ("curses", "fcntl", "termios", "pty", "grp", "pwd",
                "posix", "resource", "syslog"),
    "Linux": ("msvcrt", "winreg", "winsound"),
    "Darwin": ("msvcrt", "winreg", "winsound"),
}


def _denied_stdlib_mentions(text: str, os_name: str) -> list[str]:
    """text 中で言及されている「対象 OS で利用不可な stdlib」名を返す (既知リスト順)。"""
    return [
        mod for mod in _OS_UNAVAILABLE_STDLIB.get(os_name, ())
        if re.search(rf"\b{re.escape(mod)}\b", text)
    ]


def annotate_os_unavailable_modules(
    summary: str, modules: list[dict], *, os_name: str | None = None,
) -> tuple[str, list[dict]]:
    """planner 出力に OS 非対応 stdlib の採用が焼き込まれていたら対抗ノートを追記する。

    ``os_constraint()`` のプロンプト注入だけでは小型モデルに無視されうる
    (2026-07-07 live: PR#240 の注入後も planner が purpose に「standard curses
    library」を採用し、正準ファイル一覧経由で spec/code 全プロンプトへ伝播して
    Windows 起動不能に至った)。採用文言そのものに決定論の否定注記を併記し、
    下流の合理化 (「curses に対応した環境」等) を汚染源で断つ。テキストの
    削除・書き換えは行わない (prose 手術は誤爆リスクがあるため追記のみ)。
    """
    resolved_os = os_name or platform.system() or "unknown"

    def _note(mods: list[str]) -> str:
        quoted = ", ".join(f"`{m}`" for m in mods)
        return (
            f" [OS NOTE: {quoted} is NOT importable on {resolved_os} — "
            f"do NOT use it; design a portable alternative instead]"
        )

    hits_total: set[str] = set()
    out_modules: list[dict] = []
    for m in modules:
        hits = _denied_stdlib_mentions(str(m.get("purpose", "")), resolved_os)
        if hits:
            m = {**m, "purpose": f"{m['purpose']}{_note(hits)}"}
            hits_total.update(hits)
        out_modules.append(m)
    summary_hits = _denied_stdlib_mentions(summary, resolved_os)
    if summary_hits:
        summary = f"{summary}{_note(summary_hits)}"
        hits_total.update(summary_hits)
    if hits_total:
        logger.warning(
            "planner adopted OS-unavailable stdlib %s; appended deterministic "
            "counter-note (os=%s)", sorted(hits_total), resolved_os,
        )
    return summary, out_modules


def _file_name(path: str) -> str:
    """OS 非依存のファイル名 (basename)。depends_on の一意解決用。"""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def resolve_unknown_dependencies(modules: list[dict]) -> list[dict]:
    """module list に無いパスを指す ``depends_on`` を決定論解決する。

    planner は「単一モジュール構成」と申告しながら実在しないモジュールへの
    依存を書く自己矛盾を出すことがある (2026-07-07 live: modules=[main.py] で
    depends_on=[game.py] + purpose「imports the Game class」→ 生成コードが
    `from game import Game` して起動不能。`game` は生成物 stem でも stdlib
    でもないため smoke/pytest 双方で外部依存に降格され偽 success で配信された)。

    exact 一致 → basename 一意一致の順で正準パスへ解決し、解決不能な依存は
    除去して purpose に PLAN NOTE を追記する (依存エントリの除去以外は
    追記のみ — prose の書き換えはしない)。ノートは正準ファイル一覧経由で
    spec/code 全プロンプトへ伝播し、幻覚モジュールからの import を抑止する。
    """
    known = {m["file_path"] for m in modules}
    by_name: dict[str, list[str]] = {}
    for path in known:
        by_name.setdefault(_file_name(path), []).append(path)

    def _note(removed: list[str]) -> str:
        quoted = ", ".join(f"`{d}`" for d in removed)
        return (
            f" [PLAN NOTE: {quoted} is NOT part of this program — do not "
            f"import from it; implement the needed behavior inside this "
            f"module or its listed dependencies]"
        )

    out: list[dict] = []
    dropped_total: set[str] = set()
    for m in modules:
        resolved: list[str] = []
        dropped: list[str] = []
        for dep in m["depends_on"]:
            if dep in known:
                target = dep
            else:
                candidates = by_name.get(_file_name(dep), [])
                if len(candidates) == 1:
                    target = candidates[0]
                else:
                    dropped.append(dep)
                    continue
            if target not in resolved:
                resolved.append(target)
        if dropped:
            m = {**m, "depends_on": resolved,
                 "purpose": f"{m['purpose']}{_note(dropped)}"}
            dropped_total.update(dropped)
        elif resolved != m["depends_on"]:
            m = {**m, "depends_on": resolved}
        out.append(m)
    if dropped_total:
        logger.warning(
            "planner declared dependencies on unplanned modules %s; removed "
            "them and appended a deterministic counter-note",
            sorted(dropped_total),
        )
    return out


def _slug(file_path: str) -> str:
    """file_path を task_id に使える ASCII slug に正規化する。"""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", file_path.strip().lower()).strip("_")
    return s or "mod"


def _module_stem(file_path: str) -> str:
    """file_path の stem (小文字) を返す (エントリモジュール判定用)。"""
    name = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0].lower()


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
    """spec タスク description 用にモジュール一覧を整形する。

    行頭 ``- <path>: `` の 1 行目形式は :func:`spec_parts.module_paths_from_list`
    の正規表現と契約している。複数行 purpose は 2 スペースインデントの継続行に
    折り返し、行ベースの形状を保つ。
    """
    lines: list[str] = []
    for m in modules:
        fp = str(m.get("file_path", "")).strip()
        purpose = str(m.get("purpose", "")).strip()
        components = [
            str(c).strip() for c in (m.get("key_components") or []) if str(c).strip()
        ]
        comp_note = f" [components: {', '.join(components)}]" if components else ""
        deps = [str(d).strip() for d in (m.get("depends_on") or []) if str(d).strip()]
        dep_note = f" (depends on: {', '.join(deps)})" if deps else ""
        purpose_lines = [ln.strip() for ln in purpose.splitlines() if ln.strip()]
        head = purpose_lines[0] if purpose_lines else ""
        lines.append(f"- {fp}: {head}{comp_note}{dep_note}")
        lines.extend(f"  {ln}" for ln in purpose_lines[1:])
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
        components = item.get("key_components")
        if not isinstance(components, list):
            components = []
        out.append({
            "file_path": fp,
            "purpose": str(item.get("purpose", "")).strip(),
            "key_components": [str(c).strip() for c in components if str(c).strip()],
            "depends_on": [str(d).strip() for d in deps if str(d).strip()],
        })
    return out


async def synthesize_coding_task_graph(
    *,
    request: str,
    project_id: str,
    assist_client: "AssistModelClient | None",
    include_tests: bool = True,
    debug_logger: "DebugLogger | None" = None,  # noqa: ARG001
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

    prompt = _SYNTHESIS_PROMPT.format(request=request.strip()) + os_constraint()
    graph_telemetry: dict = {}
    try:
        result = await assist_client.generate_json(
            prompt,
            purpose="coding_task_graph",
            max_tokens=1536,
            temperature=0.3,
            telemetry=graph_telemetry,
        )
    except Exception as exc:
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
    # OS 制約のプロンプト注入を planner が無視した場合の決定論バックストップ。
    summary, modules = annotate_os_unavailable_modules(summary, modules)
    # 実在しないモジュールへの依存 (planner の自己矛盾) を決定論解決する。
    modules = resolve_unknown_dependencies(modules)

    facts: list[SemanticFact] = []
    seen_ids: set[str] = {SPEC_TASK_ID}

    # 1. spec タスク (依存なし、最初に必ず実行される)
    spec_desc = summary or f"Design specification for: {request.strip()}"
    module_block = _render_module_list(modules)
    if module_block:
        spec_desc = f"{spec_desc}{MODULE_LIST_MARKER}{module_block}"
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
        # エントリらしいモジュールは depends_on 未申告でも最後に生成する
        # (兄弟 API が全て実在する状態でエントリを書かせ、API 幻覚を抑える)。
        if len(modules) > 1 and _module_stem(fp) in _ENTRY_STEMS:
            code_salience = _ENTRY_SALIENCE
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
