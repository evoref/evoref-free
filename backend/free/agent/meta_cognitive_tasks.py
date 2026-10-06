"""Meta-Cognitive タスク管理: データクラス・タスク状態判定・タスクマージ"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.free.agent.meta_cognitive_content_gate import (
    is_edit_request,
    looks_like_tool_selector_json,
)
from backend.free.agent.meta_cognitive_defs import _DATA_BEARING_TOOLS
from backend.free.agent.meta_cognitive_utils import is_tool_error
from backend.free.agent.task_write_gate import (
    record_task_write_verdict,
    task_write_verdict,
)
from backend.i18n_helper import msg
from backend.log_config import get_logger
from backend.free.agent.write_gate import WRITE_PATH_TOOLS
from backend.free.llm.json_schemas import PLAN_TASK_KINDS

if TYPE_CHECKING:
    from backend.free.agent.credit_assigner import StepCredit
    from backend.free.core.predicate import Band, Verdict

logger = get_logger("agent.meta_cognitive.tasks")


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class TaskItem:
    """タスクリストの1項目"""
    description: str
    status: str = "pending"  # "pending" | "done" | "failed"
    result: str = ""
    # collapse_fetch_save_tasks が付与する「取得専任」マーカー。True のタスクは
    # 取得 (fetch_url 等) のみ行い、書込みは行わない (出力は後続の write タスクが
    # 取得データを決定論的に書く)。小型モデルが取得タスクの tool-loop 内で余計な
    # write_file を出し、プレースホルダ/重複ファイルを生む退行を防ぐ。
    fetch_only: bool = False
    # 書込みを期待したタスクが失敗したときの利用者向けの注記 (i18n。どのファイルへの
    # 何が失敗したか / ファイルが変わっていないか)。meta が失敗の確定後に刻み、
    # 最終応答と chat_stream_meta の要約が読む (docs/f_03 §4.3)。
    failure_note: str = ""
    # ``result`` が取得系ツール (``_DATA_BEARING_TOOLS``) の出力そのものか。後続タスクの
    # 素材であって答えではないので、最終応答は全タスクが取得のターンでしか本文に
    # 出さない (docs/f_03 §4.2.1、2026-10-03 ライブ監査 run15)。
    retrieved: bool = False
    # 判定点 ``task_write_intent`` の結果 (判定したときの記述, 判定)。記述が書き換わったら
    # 判定し直す (``task_write_band``、docs/f_03 §4.3)。
    write_verdict: tuple[str, Verdict] | None = field(
        default=None, repr=False, compare=False,
    )
    # 計画モデルが付けたタスクの種別のラベル (``PLAN_TASK_KINDS`` の 1 つ、無ければ
    # ``None``)。読むのは判定点 ``task_write_intent`` の確認票だけ (``process`` /
    # ``retrieve`` は字句の書込みを棄権へ倒す、task_write_gate)。他の実行の経路は読まない。
    plan_kind: str | None = None
    # 答えがコードで集計した表の値と食い違ったときの末尾の注記 (i18n、docs/f_03 §4.2.2)。
    # 答え (``result``) には混ぜない — 後のタスクの素材になり、最終応答の途中に残ると
    # 記憶から落とせない。最終応答の末尾にまとめて足す。
    output_note: str = ""


@dataclass
class PlannerOutput:
    """計画モデルの出力 (記録のみ、docs/f_03 §4.3)。

    ``tasks`` は正規化 (束ね・集約・生成タスクへの置換) の **前** のタスク (同じ
    オブジェクト)。``kinds`` は応答の ``tasks`` の位置ごとのラベル (空のタスクの位置も含む)。
    """
    tasks: list[TaskItem]
    kinds: list[str | None]


@dataclass
class EditorArtifact:
    """エディタ出力用の生成コード片（ディスク書込せずフロントのエディタへ流す）"""
    content: str
    language: str = "python"
    filename: str | None = None


@dataclass
class MetaCognitiveResponse:
    """Meta-Cognitive 層の応答"""
    content: str
    tasks: list[TaskItem] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    steps: int = 0
    episode_id: str = ""
    step_credits: list[StepCredit] = field(default_factory=list)
    # 出力先パス未指定時にエディタペインへ流す生成コード（write_file は行わない）
    editor_artifacts: list[EditorArtifact] = field(default_factory=list)
    # 内部の LLM 生成 (ツールループ / コンテンツ生成 / フォールバック) のいずれかが
    # ``finish_reason=length`` で切れたか。エージェントがストリームを内部で消費する
    # ため、開示は呼出側 (chat_stream_meta) がこのフラグを見て SSE フレームで行う。
    truncated: bool = False
    truncated_steps: list[str] = field(default_factory=list)
    # 最後に切れた生成の生トークン数 / max_tokens (``sse.output_truncated`` 用)
    truncated_tokens: int = 0
    truncated_max_tokens: int | None = None
    # production_stage (制作ステージ、f_03 §4.4) の ``ProductionResult.metrics``。
    # 制作ステージを経由しなかった (production_stage 無し) ターンは空 dict のまま
    # (Level 0 経験記録の quality_signals へ chat_stream_meta が載せる)。
    production_metrics: dict = field(default_factory=dict)
    # 制作ステージの注記 (``ProductionResult.notes["notices"]``、UI の言語の文)。本文の後に
    # 添える (依頼されたテストを作っていない等、f_10 §11.1-1)。
    production_notices: list[str] = field(default_factory=list)
    # 制作ステージの検査の 3 値 (``ProductionResult.notes["checks"]``、``CheckOutcome.to_dict``
    # の列)。未検査の主要な検査は経験の成否をラベル無しにする (docs/f_04 §2.5)。
    production_checks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# タスク状態判定
# ---------------------------------------------------------------------------

def task_write_band(task: TaskItem) -> Band:
    """タスクの判定点 ``task_write_intent`` の帯域 (記述ごとに 1 回だけ判定する)。"""
    return task_write_verdict_of(task).band


def task_write_verdict_of(task: TaskItem) -> Verdict:
    """タスクの判定点 ``task_write_intent`` の判定 (記述ごとに 1 回、計画モデルの種別込み)。"""
    cached = task.write_verdict
    if cached is None or cached[0] != task.description:
        cached = (task.description, task_write_verdict(task.description, task.plan_kind))
        task.write_verdict = cached
    return cached[1]


def task_writes(task: TaskItem) -> bool:
    """書込みのタスクか (``fire`` だけ、docs/f_03 §4.3)。

    書込みを起こす読み手 (ツールの推論・テキストからの書込みの救出・再試行・計画への
    宛先の付記・計画の束ね直し) と、状態を ``failed`` にしたり答えを隠したりする帳簿の
    読み手 (``determine_task_status`` / 失敗の注記 / 本文) が読む。棄権 (動詞が無く
    宛先の標識だけ) は観察のみ — 「Compare a.txt to b.txt」で b.txt を書いたり、
    「Go to E:\\…\\README.md and summarize it」の答えを失敗にして隠したりしない。
    """
    return task_write_band(task) == "fire"


def task_may_write(task: TaskItem) -> bool:
    """書込みのタスク **かもしれない** か (``fire`` か ``abstain``)。

    書込みを起こさず近道を止めるだけの読み手に限る — 取得済みデータからの答え・
    取得の後の処理の対象外にする (ツールループへ回す) と、後のタスクへ書込みを委ねる
    (書込みを止める)。「Save the summary to X」を取得済みデータの答えにして書かずに
    終えないため (docs/f_03 §4.3)。状態や本文の判定には使わない (:func:`task_writes`)。
    """
    return task_write_band(task) != "skip"


def record_task_write_verdicts(tasks: list[TaskItem]) -> None:
    """計画の正規化が済んだタスクを判定点として 1 回ずつ記録し、判定を持たせる。"""
    for task in tasks:
        task.write_verdict = (
            task.description, record_task_write_verdict(task.description, task.plan_kind),
        )


#: 計画のタスク種別のラベルを記録する決定ログの判定点 (記録のみ、docs/f_03 §4.3)。
PLAN_TASK_KIND_DECISION = "plan_task_kind"
#: ラベルが無いタスクの ``chosen``。
NO_PLAN_KIND = "none"


def parse_plan_task_kinds(raw_kinds: object, task_count: int) -> list[str | None]:
    """計画の応答の ``kinds`` を ``tasks`` の位置ごとのラベルにする (純粋関数)。

    位置で対応付けるので、リストでない・長さが ``tasks`` と違うときは全部 ``None``
    (1 件ずれると全部ずれる)。未知の値はその要素だけ ``None``。どちらも件数を
    WARNING に出す (計画そのものは落とさない)。
    """
    if not isinstance(raw_kinds, list) or len(raw_kinds) != task_count:
        if task_count:
            logger.warning(
                "Plan kinds ignored: expected %d labels, got %s",
                task_count,
                len(raw_kinds) if isinstance(raw_kinds, list) else type(raw_kinds).__name__,
            )
        return [None] * task_count
    kinds = [k if isinstance(k, str) and k in PLAN_TASK_KINDS else None for k in raw_kinds]
    invalid = sum(1 for k in kinds if k is None)
    if invalid:
        logger.warning("Plan kinds: %d of %d labels are not known kinds", invalid, task_count)
    return kinds


def log_plan_task_kinds(
    tasks: list[TaskItem], planner: PlannerOutput, debug_logger: object | None,
) -> None:
    """タスクごとに計画のラベルと ``task_write_intent`` の判定・結末を 1 件ずつ記録する。

    実行の済んだ後に呼ぶ (結末を同じ行に並べるため)。判定にも状態にも影響しない。
    ``reason`` は、ラベルがある (``planner_label``)・計画モデルのタスクのままでラベルが
    無い (``no_label``)・正規化が作り直したタスク (``rebuilt``) を分ける。計画モデルの
    ラベル列とタスク数は最初の行の context にだけ載せる (正規化の前後を突き合わせる用)。
    """
    if debug_logger is None:
        return
    candidates = [*PLAN_TASK_KINDS, NO_PLAN_KIND]
    planned_ids = {id(t) for t in planner.tasks}
    for index, task in enumerate(tasks):
        cached = task.write_verdict
        verdict = (
            cached[1] if cached is not None and cached[0] == task.description
            else task_write_verdict(task.description, task.plan_kind)
        )
        if task.plan_kind:
            reason = "planner_label"
        elif id(task) in planned_ids:
            reason = "no_label"
        else:
            reason = "rebuilt"
        context: dict[str, object] = {
            "task_index": index,
            "task_count": len(tasks),
            "write_band": verdict.band,
            "write_evidence": verdict.evidence,
            "status": task.status,
            "retrieved": task.retrieved,
            "fetch_only": task.fetch_only,
            "has_failure_note": bool(task.failure_note),
        }
        if index == 0:
            context["planner_kinds"] = [k or NO_PLAN_KIND for k in planner.kinds]
            context["planner_task_count"] = len(planner.kinds)
        try:
            debug_logger.log_decision(  # type: ignore[attr-defined]
                decision_point=PLAN_TASK_KIND_DECISION,
                chosen=task.plan_kind or NO_PLAN_KIND,
                candidates=candidates,
                reason=reason,
                context=context,
                scope="request",
            )
        except Exception as e:  # pragma: no cover - ログで実行を落とさない
            logger.debug("Plan task kind decision log failed: %s", e)


def retrieval_failure_note(entry: dict) -> str:
    """エラーで終えた取得の呼出しの、利用者向けの 1 文 (i18n ``agent.retrieval_failed.*``)。

    対象は呼出しの引数 (URL / パス / フォルダ)、理由はツールのエラーの種別
    (``error_kind``、``meta_cognitive_tool_io.tool_error_kind``) から引く。生のエラー文は出さない。
    種別の分からないエラー (``error``) と対象の無い呼出しは空文字 — 注記で答えを上書きせず、
    モデルの答えを残す (2026-10-04 反証 MED-2)。
    """
    kind = entry.get("error_kind") or "error"
    if kind == "no_location":
        # 探す場所が無くて撃たなかった検索 (対象は無い、run23 反証 2 周目 LOW-1)
        return msg("agent.retrieval_failed.no_location")
    args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
    target = str(
        args.get("url") or args.get("file_path") or args.get("directory") or "",
    ).strip()
    if kind == "error" or not target:
        return ""
    return msg(f"agent.retrieval_failed.{kind}", target=target)


def determine_task_status(
    task: TaskItem, result: str, tool_calls: list[dict],
    *, destination_known: bool = True,
) -> str:
    """タスク実行結果からステータスを決定する。

    ``destination_known`` が False (書込み先がタスク文にも会話にも無い) のとき
    「書込みが期待されたのに実行されていない」判定を **行わない**。書込み先が
    無いタスクは write_file を撃ちようがなく、この判定を掛けると **原理的に
    達成不可能なタスク** になる。実インシデント (2026-09-06 監査 F-02) では
    パス指定の無い依頼が毎回この分岐で failed になり、生成済みの成果物 3 本を
    捨てて「(書き込みが実行されませんでした)」だけを返していた (26.6 分)。

    書込み先の有無は呼出側が解決する — タスク文だけでは「同じファイルに」型の
    参照解決ができず、ここで判定すると会話から解決できるケースまで落とす。

    取得がすべてエラーで失敗にしたタスクには、何の取得が何故失敗したかの注記
    (``retrieval_failure_note``) を ``task.failure_note`` に刻む (既にあれば残す)。
    """
    if is_tool_error(result) or "Step limit reached" in result:
        return "failed"

    # ツールの出力が分類器の {tool, arg} JSON なら本文ではない (2026-09-26 監査 C08#3)
    if looks_like_tool_selector_json(result):
        logger.warning(
            "Task marked failed: output is a tool-selector JSON: %s",
            task.description[:80],
        )
        return "failed"

    if destination_known and task_writes(task) and not any(
        tc.get("tool") in WRITE_PATH_TOOLS and tc.get("success")
        for tc in tool_calls
    ):
        logger.warning(
            "Task marked failed: write expected but not executed: %s",
            task.description[:80],
        )
        return "failed"

    # 取得がすべてエラーなら、モデルの文 (「ファイルを作成します」等) が何でも失敗
    # (2026-10-04 ライブ監査 run21 F1、f_03 §4.2.1)。0 件 (エラーでない空振り) は答えにできる
    # 生成系ツール (draft_document 等) の下書きは取得の代わりにならないので成功に数えない
    # (run23 F2: 取得の拒否の後に draft_document が成功して done になった)
    from backend.free.agent.deliberative import _GENERATED_DRAFT_TOOLS

    retrievals = [tc for tc in tool_calls if tc.get("tool") in _DATA_BEARING_TOOLS]
    if (
        retrievals
        and all(tc.get("error") for tc in retrievals)
        and not any(
            tc.get("success") and tc.get("tool") not in _GENERATED_DRAFT_TOOLS
            for tc in tool_calls
        )
    ):
        logger.warning(
            "Task marked failed: every retrieval errored: %s", task.description[:80],
        )
        # 何の取得が何故失敗したかを利用者へ (2026-10-04 ライブ監査 run22 F1)
        task.failure_note = task.failure_note or retrieval_failure_note(retrievals[-1])
        return "failed"

    return "done"


# ---------------------------------------------------------------------------
# タスクマージ
# ---------------------------------------------------------------------------

def merge_same_file_tasks(tasks: list[TaskItem]) -> list[TaskItem]:
    """同一ファイルを対象とする複数タスクを1つにマージする

    PLAN_SYSTEM_PROMPT で「1ファイル=1タスク」を指示しているが、
    ローカル LLM が無視して分割するケースへの防御策。

    グループ化のキーは **書き込み先** (:func:`extract_write_target_path`)。
    以前は ``_extract_file_path`` (= 先頭のパス) を使っていたため、
    **2 ファイルにまたがる操作が同一ファイル扱いで潰れていた**。

    実インシデント (2026-08-26 ライブ監査 T7-7)。プランナーの実出力:

        {"tasks": ["Read the content of E:\\tmp\\dest_b.txt",
                   "Append the content of E:\\tmp\\dest_b.txt to the end of
                    E:\\tmp\\dest_a.txt"]}

    どちらも **先頭のパスが dest_b.txt** なので同じグループに入り
    ``" / "`` で 1 タスクへ連結された (``Task merging: 2 tasks → 1 tasks``)。
    読み取りタスクが消えて ``read_file`` が一度も走らず、書き込み内容の
    供給元 (``_fetched_tool_outputs``) が空のままになる。プランナーは
    正しく 2 タスクに分けていたのに、防御策の側が壊していた。

    書き込み先で束ねれば「Read B」は B、「Append B→A」は A になり分離する。
    本来の目的 (同じファイルへの過分割をまとめる) は変わらない。
    """
    from backend.free.agent.tool_judge_args import extract_write_target_path as _key

    keys = [_key(t.description) for t in tasks]
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        if key:
            groups.setdefault(key, []).append(i)
    multi = {key: members for key, members in groups.items() if len(members) > 1}
    if not multi:
        return tasks

    # 区間が他のグループと重なるグループは束ねない。束ねると挟まるタスクが
    # 2 グループに入って 2 回走るか、書き戻しが挟まるタスクより前へ移る
    # (2026-09-26 レビュー、docs/f_03 §4.3)。重なるときは計画の順序のまま。
    spans = {key: (m[0], m[-1]) for key, m in multi.items()}

    def _overlaps(key: str) -> bool:
        a0, a1 = spans[key]
        return any(
            other != key and not (b1 < a0 or a1 < b0)
            for other, (b0, b1) in spans.items()
        )

    def _absorbable(task: TaskItem) -> bool:
        # 書込みを期待しないか、既存内容の変更依頼だけ。別の成果物を作るパス無し
        # タスク (「Create a new report」) は別ファイルのグループへ取り込まない。
        return not task_writes(task) or is_edit_request(task.description)

    merged_indices: set[int] = set()
    result_items: list[tuple[int, TaskItem]] = []

    for key, members in multi.items():
        if _overlaps(key):
            continue
        # 書込みの無いグループ (同じフォルダの TODO の検索と FIXME の検索) は別々の取得で、
        # 束ねると 2 つ目の取得がループのモデル任せになる (2026-10-04 run23 X1、f_03 §4.3)
        if not any(task_writes(tasks[j]) for j in members):
            continue
        # 読んで → (パス無しの) 変更 → 書き戻す、の区間は 1 件に束ねる。束ねた
        # グループは先頭の位置に置かれるので、挟まる変更タスクを残すと書き戻しの
        # 後に単独で走る (2026-09-26 監査 C08#3、docs/f_03 §4.3)。
        if (
            not task_writes(tasks[members[0]])
            and task_writes(tasks[members[-1]])
        ):
            members = sorted({
                *members,
                *(j for j in range(members[0] + 1, members[-1])
                  if not keys[j] and _absorbable(tasks[j])),
            })
        merged_desc = " / ".join(tasks[j].description for j in members)
        result_items.append((members[0], TaskItem(description=merged_desc)))
        merged_indices.update(members)

    if not merged_indices:
        return tasks
    for i, task in enumerate(tasks):
        if i not in merged_indices:
            result_items.append((i, task))

    result_items.sort(key=lambda x: x[0])

    merged = [item for _, item in result_items]
    logger.info(
        "Task merging: %d tasks → %d tasks", len(tasks), len(merged),
    )
    return merged


def collapse_editor_write_tasks(tasks: list[TaskItem]) -> list[TaskItem]:
    """editor/chat 出力 (パス未指定) で過分割された書き込みタスクを単一生成へ集約する。

    merge_same_file_tasks はパスを持つタスクのみ統合するため editor/chat 経路では
    機能しない。ローカル LLM が単一ファイル要求を複数の書き込みタスクに分割すると、
    タスクごとに独立生成され editor_artifact が複数 (= エディタに同名タブが複数) でき
    る。書き込みタスクが 2 つ以上ある場合、最初の書き込みタスクだけ残し残りを除去する。
    非書き込みタスク (read/verify 等) と順序は保持。コード生成は _generate_content が
    original_query (要求全体) を主体に行うため、残した 1 タスクで完全なファイルになる。
    1 リクエスト=1 生成=1 タブを保証する防御策。
    """
    write_idx = [i for i, t in enumerate(tasks) if task_writes(t)]
    if len(write_idx) <= 1:
        return tasks
    drop = set(write_idx[1:])  # 最初の書き込みタスクだけ残す
    collapsed = [t for i, t in enumerate(tasks) if i not in drop]
    logger.info(
        "Editor task collapsing: %d tasks → %d tasks (%d write tasks → 1)",
        len(tasks), len(collapsed), len(write_idx),
    )
    return collapsed


# 取得 (fetch) タスクの識別。動詞または URL の存在で判定する。
_FETCH_TASK_RE = re.compile(
    r"(?:fetch|retrieve|download|scrape|取得|読み込|読み取)|https?://",
    re.IGNORECASE,
)


def collapse_fetch_save_tasks(
    tasks: list[TaskItem], query: str,
) -> list[TaskItem]:
    """単一 URL 取得 → 単一ファイル保存の過分割を [fetch, write] へ集約する。

    planner が「fetch / extract / generate / save」と過分割すると、抽出・保存
    タスクで小型モデルが拒否/誤反応 (例: 「2026 は未来だから結果は無い」) し、
    出力が破綻する。取得データの書き込みは決定論経路 (取得テーブルを直接書込) が
    担うため、fetch を 1 つ残して残りを 1 つの write タスクへ集約し、モデルに
    「抽出」「保存」を別タスクで実行させる余地を断つ。

    安全側ガード: 出力が表計算/リッチ文書 (xlsx/csv/ods/docx/pptx) で、クエリに URL が
    ちょうど 1 つ、fetch 以外のタスクが 2 つ以上の場合のみ集約する。非対象形式
    (要約 → .md 等) はモデル駆動フローを保持し、複数 URL / 複数ファイル要求も対象外。
    """
    from backend.free.agent.output_format import (
        FETCHED_TABLE_EXTS,
        infer_output_extension,
    )
    if infer_output_extension(query, default="") not in FETCHED_TABLE_EXTS:
        return tasks
    if len(re.findall(r"https?://", query)) != 1:
        return tasks
    fetch_tasks = [t for t in tasks if _FETCH_TASK_RE.search(t.description)]
    other_tasks = [t for t in tasks if not _FETCH_TASK_RE.search(t.description)]
    if not fetch_tasks or len(other_tasks) <= 1:
        return tasks

    from backend.free.agent.tool_call_judge import _extract_file_path
    out_path = _extract_file_path(query)
    desc = "Write the fetched data to the file the user requested"
    if out_path:
        desc = f"Write the fetched data to {out_path}"
    # 取得タスクは fetch_only=True とし、書込みは後続の write タスクへ委ねる
    # (入力 TaskItem を変異させず新規生成する)。
    collapsed = [
        TaskItem(description=fetch_tasks[0].description, fetch_only=True),
        TaskItem(description=desc),
    ]
    logger.info(
        "Fetch/save task collapsing: %d tasks → 2 (fetch + write)",
        len(tasks),
    )
    return collapsed


# 「スクリプトを実行/run」系タスク。文書/データ出力では成果物を write_file →
# export Writer が描画するため、生成スクリプトの実行は常に誤り (実体の無い
# スクリプトを run_command で叩いて失敗する)。
_EXECUTE_TASK_RE = re.compile(
    r"(?<![A-Za-z])(?:execute|run)(?![A-Za-z])|実行",
    re.IGNORECASE,
)
# 「プログラム/スクリプトを生成」系タスク。文書/データ出力でこれらの語を含む
# プランは、内容ではなく「文書を作るコード」を生成しようとする退行シグナル。
_SCRIPT_TASK_RE = re.compile(
    r"(?<![A-Za-z])(?:python|script|program|openpyxl|vba|macro)(?![A-Za-z])"
    r"|python-pptx|python-docx|スクリプト|プログラム|マクロ",
    re.IGNORECASE,
)
# 入力取得 (read/fetch) タスク。集約時もデータ源として保持する。
_INPUT_TASK_RE = re.compile(
    r"(?<![A-Za-z])(?:fetch|retrieve|download|scrape|read|load)(?![A-Za-z])"
    r"|取得|読み込|読み取|読んで",
    re.IGNORECASE,
)


def collapse_document_generation_tasks(
    tasks: list[TaskItem], query: str,
) -> list[TaskItem]:
    """URL 無しの文書/データファイル出力で「スクリプト生成 → 実行」プランを単一 write へ集約する。

    planner が「Excel カレンダーを作成」を「Excel を作る Python スクリプトを生成 →
    そのスクリプトを実行」と誤分解すると、(1) スクリプトコードを .xlsx へ書こうとして
    "No table data found" エラー、(2) 実体の無い生成スクリプトを ``run_command`` で実行
    して失敗、となる。成果物 (表/文書) は write_file → export Writer が描画するため、
    内容を直接生成する単一 write タスクへ正規化し、スクリプト生成 / 実行タスクを排除する。

    安全側ガード: 出力が FETCHED_TABLE_EXTS (xlsx/csv/ods/docx/pptx)、クエリに URL 無し
    (URL は ``collapse_fetch_save_tasks`` 管轄)、かつ実行 / スクリプト生成タスクを実際に
    含む (退行シグナルがある) 場合のみ集約する。正常な単一 write タスクや、入力ファイルを
    読んで変換する正当なプランには手を入れない。入力 (read/fetch) タスクは保持する。
    """
    from backend.free.agent.output_format import (
        FETCHED_TABLE_EXTS,
        infer_output_extension,
    )
    if infer_output_extension(query, default="") not in FETCHED_TABLE_EXTS:
        return tasks
    if re.search(r"https?://", query):
        return tasks
    has_antipattern = any(
        _EXECUTE_TASK_RE.search(t.description)
        or _SCRIPT_TASK_RE.search(t.description)
        for t in tasks
    )
    if not has_antipattern:
        return tasks

    # データ源 (read/fetch) は保持。スクリプト生成 / 実行タスクは破棄する。
    input_tasks = [
        t for t in tasks
        if _INPUT_TASK_RE.search(t.description)
        and not _SCRIPT_TASK_RE.search(t.description)
        and not _EXECUTE_TASK_RE.search(t.description)
    ]

    from backend.free.agent.tool_call_judge import _extract_file_path
    out_path = _extract_file_path(query)
    desc = "Write the requested document to the file the user requested"
    if out_path:
        desc = f"Write the requested document to {out_path}"

    collapsed = [
        TaskItem(description=t.description, fetch_only=True) for t in input_tasks
    ]
    collapsed.append(TaskItem(description=desc))
    logger.info(
        "Document generation task collapsing: %d tasks → %d "
        "(script/execute steps removed)",
        len(tasks), len(collapsed),
    )
    return collapsed
