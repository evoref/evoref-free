"""Meta-Cognitive タスク管理: データクラス・タスク状態判定・タスクマージ"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.free.agent.meta_cognitive_utils import is_tool_error
from backend.log_config import get_logger
from backend.free.core.intent_vocab import WRITE_VERB_RE

if TYPE_CHECKING:
    from backend.free.agent.credit_assigner import StepCredit

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


# ---------------------------------------------------------------------------
# タスク状態判定
# ---------------------------------------------------------------------------

# 書き込み期待パターン（日本語・英語）。
# 定義は core.intent_vocab が SSOT (meta_cognitive_tools の write_file
# ルーティングが同一定義を持っていた)。
_WRITE_PATTERN = WRITE_VERB_RE


def task_expects_write(description: str) -> bool:
    """タスク記述がファイル書き込みを期待しているか判定する

    「作成」「追加」「実装」「修正」などの動詞を含むタスクは
    write_file の実行が期待される。
    読み取り専用の記述（"Read foo.py"）は書き込み動詞を含まないため
    自然に False になる。

    注: ファイルパスの有無は問わない。パスなしでも「書き込み期待」は成立する
    （determine_task_status でのステータス判定で使用）。
    """
    return bool(_WRITE_PATTERN.search(description))


def determine_task_status(
    task: TaskItem, result: str, tool_calls: list[dict],
) -> str:
    """タスク実行結果からステータスを決定する"""
    if is_tool_error(result) or "Step limit reached" in result:
        return "failed"

    if task_expects_write(task.description) and not any(
        tc.get("tool") == "write_file" and tc.get("success")
        for tc in tool_calls
    ):
        logger.warning(
            "Task marked failed: write expected but not executed: %s",
            task.description[:80],
        )
        return "failed"

    fetch_tools = {"fetch_url", "read_file", "search_code", "list_directory"}
    if any(
        tc.get("tool") in fetch_tools and tc.get("success")
        for tc in tool_calls
    ):
        return "done"

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

    file_groups: dict[str, tuple[int, list[str]]] = {}
    for i, task in enumerate(tasks):
        path = _key(task.description)
        if path:
            if path not in file_groups:
                file_groups[path] = (i, [task.description])
            else:
                file_groups[path][1].append(task.description)

    if all(len(descs) == 1 for _, descs in file_groups.values()):
        return tasks

    merged_indices: set[int] = set()
    result_items: list[tuple[int, TaskItem]] = []

    for path, (first_idx, descriptions) in file_groups.items():
        if len(descriptions) == 1:
            result_items.append((first_idx, tasks[first_idx]))
        else:
            merged_desc = " / ".join(descriptions)
            result_items.append((first_idx, TaskItem(description=merged_desc)))
            for j, task in enumerate(tasks):
                if _key(task.description) == path:
                    merged_indices.add(j)

    for i, task in enumerate(tasks):
        if i not in merged_indices and not _key(task.description):
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
    write_idx = [i for i, t in enumerate(tasks) if task_expects_write(t.description)]
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
