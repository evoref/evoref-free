"""TaskExecutionMixin — meta_cognitive_task_exec"""

from __future__ import annotations

import asyncio
import json
import posixpath
import re
import time

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from backend.config import resolve_context_size_for_mode
from backend.free.agent.agent_state import AgentState
from backend.free.agent.meta_cognitive_tasks import (
    TaskItem,
    task_may_write,
    task_write_band,
    task_write_verdict_of,
    task_writes,
)
from backend.free.agent.task_write_gate import PLAN_KIND_ABSTAIN_EVIDENCE
from backend.free.agent.meta_cognitive_defs import ALREADY_WRITTEN
from backend.free.agent.meta_cognitive_tools import (
    normalize_read_file_args,
    normalize_write_file_args,
)
from backend.free.agent.output_format import (
    anchor_relative_output_path,
    redirect_unnamed_overwrite,
    resolve_dir_output_path,
)
from backend.free.agent.meta_cognitive_content import note_stream_truncation
from backend.free.agent.meta_cognitive_content_gate import looks_like_tool_selector_json
from backend.free.agent.meta_cognitive_utils import (
    call_callback,
    content_language_directive,
    is_tool_error,
    summarize_tool_args,
    tool_error_kind,
    tool_result_lacks_information,
    tool_result_succeeded,
)
from backend.free.agent.meta_cognitive_tool_io import fetch_error_type
from backend.free.agent.step_compactor import StepResult
from backend.free.agent.table_aggregate_intent import aggregate_retrieved_table
from backend.free.agent.tool_judge_args import _URL_IN_QUERY_RE
from backend.free.agent.tools_registry import FILESYSTEM_TOOL_NAMES
from backend.free.api.chat.chat_constants import (
    TOOL_EXECUTION_TIMEOUT_SEC,
    TOOL_RESULT_HEAD_RATIO,
    TOOL_RESULT_MAX_CHARS,
    TOOL_RESULT_OMISSION_CHARS,
)
from backend.free.constants import (
    FETCH_UNREACHABLE_ERRORS,
    SEARCH_NO_LOCATION_ERROR,
    search_history_window_note,
)
from backend.free.core.inference import build_messages_for_loop
from backend.free.core.intent_vocab import EXPLICIT_WINDOWS_PATH_RE, mentions_filesystem
from backend.free.core.table_aggregate import (
    mismatch_note as table_aggregate_mismatch_note,
    result_block as table_aggregate_block,
)
from backend.i18n_helper import msg
from backend.utils import estimate_tokens as _estimate_tokens

from backend.free.agent.meta_cognitive_defs import (
    EXECUTE_SYSTEM_PROMPT,
    RETRIEVED_ANSWER_LATER_STEPS,
    RETRIEVED_ANSWER_SYSTEM_PROMPT,
    RETRIEVED_ANSWER_USER_PROMPT,
    RETRIEVED_DATA_BLOCK_NOTE,
    _DATA_BEARING_TOOLS,
    _is_placeholder_write_path,
    resolve_read_path,
)

from backend.log_config import get_logger

logger = get_logger("agent.meta_cognitive")


def tool_mode_error(tools_registry, tool_name: str, mode: str) -> str | None:
    """``ToolDefinition.modes`` に基づく実行時の mode ゲート (deliberative と同じ規則)。

    ``modes`` は元々 LLM 向け説明文のフィルタにしか使われず、meta 経路の
    実行時には無視されていた (search_code だけ個別にガードしていた)。
    create 専用ツール (run_command / apply_diff / verify_syntax 等) が chat
    モードのタスクから実行されないよう、登録済み定義があれば必ず照合する。
    例外は ``write_file`` のみ: 選択は create 限定だが chat の書き出し経路から
    正規に実行されるため、``inventory_modes`` に載っているモードでは許可する
    (:attr:`ToolDefinition.inventory_modes` 参照)。

    Returns:
        拒否時は ``Error:`` 文字列、許可時は None。
    """
    tool_def = tools_registry.get(tool_name)
    if tool_def is None or mode in tool_def.modes:
        return None
    if tool_name == "write_file" and tool_def.listed_in(mode):
        return None
    logger.warning(
        "Tool not allowed in mode=%s: %s (allowed modes: %s)",
        mode, tool_name, tool_def.modes,
    )
    return f"Error: {tool_name} is not available in mode '{mode}'"


async def execute_tool_with_timeout(
    tools_registry, tool_name: str, tool_args: dict,
) -> str:
    """ツールを timeout 付きで実行し結果テキストを返す (deliberative と同じ規則)。

    timeout は ``ToolsRegistry.timeout_for`` (ツール宣言 > 既定 30 秒)。
    超過時は ``Error:`` 文字列を返し、``tool_result_succeeded`` が失敗と
    扱えるようにする。timeout 以外の例外は呼出側で扱う (経路ごとに
    state / ログの処理が異なるため)。ツール台帳への失敗記録は
    ``ToolsRegistry.execute`` (キャンセルが配送される合流点) が行う。
    """
    timeout_sec = tools_registry.timeout_for(tool_name, TOOL_EXECUTION_TIMEOUT_SEC)
    try:
        result = await asyncio.wait_for(
            tools_registry.execute(tool_name, **tool_args), timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Tool execution timed out: %s (%.0fs)", tool_name, timeout_sec,
        )
        return f"Error: tool '{tool_name}' timed out after {timeout_sec:g}s"
    return str(result)


def truncate_tool_result(text: str, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    """ツール結果が max_chars を超える場合、先頭と末尾を残して切り詰める。

    deliberative の ``_truncate_tool_result`` と同じ体裁。meta のツールループは
    最新 step の出力を全文で LLM に渡す (StepCompactor は最新反復を圧縮しない)
    ため、巨大な結果がそのままコンテキストを食い潰さないようここで上限を掛ける。
    """
    if len(text) <= max_chars:
        return text
    head_size = int(max_chars * TOOL_RESULT_HEAD_RATIO)
    tail_size = max_chars - head_size - TOOL_RESULT_OMISSION_CHARS
    omitted = len(text) - head_size - tail_size
    return (
        text[:head_size]
        + f"\n\n... ({omitted} chars omitted) ...\n\n"
        + text[-tail_size:]
    )


def _is_unanchored_relative(file_path: str) -> bool:
    """ディレクトリ成分を持つ錨の無い相対パスか (``./`` ``../`` ``~`` / 絶対パスは除く)。"""
    normalized = file_path.replace("\\", "/")
    if normalized.startswith(("./", "../", "~", "/")):
        return False
    if len(file_path) > 1 and file_path[1] == ":":
        return False
    return "/" in normalized.strip("/")


def explicit_query_dirs(query: str) -> list[tuple[Path, bool]]:
    """依頼が明示したフォルダを出現順に ``(パス, 出力先の候補か)`` で返す (f_03 §4.4 / §4.x)。

    - 実在するディレクトリ → 出力先の候補
    - 拡張子の無いパス (``E:\\tmp\\new フォルダに``) → **未作成でも** 出力先の候補。日本語が続いて
      区切れない表記 (``E:\\tmp\\newフォルダに``) は ASCII のセグメントだけを取る (``_DIR_PATH_RE``)
    - ファイルのパス → その親 (入力ファイルでありうるので出力先の候補にはしない)

    - 英文の手前の空白で切った名前 (``Q:\\projects\\new app`` → ``new``) → 出力先の候補に加えて、
      その親を出力先ではない根として足す (切らない読み方も配信先・書込みの門が覆う)

    パスの末尾の境界 (文末の ``.`` / 空白の後の地の文 / 括られた空白入りの名前) は
    ``iter_drive_dir_paths`` が決める (f_03 §4.x)。

    書込み先の確定 (``_resolve_write_path_from_query``) と配信先の根 (:func:`delivery_roots`) が
    共有する部品。以前は実在するディレクトリしか見ず、未作成の出力フォルダの名指しのファイルが
    ``outputs_dir`` へ落ちて配信で拒否された (独立レビュー 2026-09-26)。
    """
    from backend.free.agent.tool_judge_args import iter_drive_dir_paths

    found: list[tuple[int, Path, bool]] = []
    for m in EXPLICIT_WINDOWS_PATH_RE.finditer(query or ""):
        candidate = Path(m.group(0).rstrip("。、,.\\/"))
        try:
            if candidate.is_dir():
                found.append((m.start(), candidate, True))
            elif candidate.suffix:
                found.append((m.start(), candidate.parent, False))
        except OSError:
            continue
    for drive_path in iter_drive_dir_paths(query or ""):
        start, end = drive_path.start, drive_path.end
        if end < len(query) and query[end] in "\\/":
            continue  # 非 ASCII のセグメントの手前で切れた接頭辞
        candidate = Path(drive_path.path.rstrip("\\/"))
        if len(candidate.parts) <= 1:
            continue  # ドライブ直下だけ
        try:
            is_dir = candidate.is_dir()
        except OSError:
            is_dir = False
        if is_dir or not candidate.suffix:
            found.append((start, candidate, True))
            if drive_path.ambiguous and len(candidate.parent.parts) > 1:
                # 英文の手前の空白で切った名前 (``new app`` → ``new``)。切らない読み方も
                # ありうるので、親を出力先ではない根として足し、配信先・書込みの門が両方を覆う。
                found.append((start, candidate.parent, False))
        else:
            found.append((start, candidate.parent, False))
    out: list[tuple[Path, bool]] = []
    for _pos, path, is_output in sorted(found, key=lambda item: item[0]):
        seen = next((i for i, (p, _) in enumerate(out) if p == path), None)
        if seen is None:
            out.append((path, is_output))
        elif is_output:
            out[seen] = (path, True)
    return out


def _explicit_output_dir(query: str) -> Path | None:
    """依頼が明示した出力先のフォルダ (最後に現れたもの、未作成でもよい)。無ければ ``None``。"""
    outputs = [path for path, is_output in explicit_query_dirs(query) if is_output]
    return outputs[-1] if outputs else None


def _explicit_path_named(query: str, name: str) -> str | None:
    """クエリの明示パスのうち、末尾名が ``name`` のもの (最初の 1 件) を返す。"""
    for raw in EXPLICIT_WINDOWS_PATH_RE.findall(query):
        candidate = raw.rstrip("。、,.\\/")
        if Path(candidate).name == name:
            return candidate
    return None


def delivery_roots(
    query: str, *, implicit: bool = True, fallback: bool = True,
) -> list[Path]:
    """制作物を配信してよいフォルダ (f_03 §4.4)。

    部品は書込み先の確定 (``_resolve_write_path_from_query``) と同じ: 依頼が明示した
    フォルダ (:func:`explicit_query_dirs`、未作成も含む)、``_extract_file_path`` が拾うパス
    (ディレクトリ形ならそれ、ファイル形なら親)。依頼にフォルダが無ければ ``outputs_dir``
    だけ。裸の名前・相対パスの成果物はこのどれかの下に着地する。

    ``implicit=False`` は暗黙参照 (「その中身」→ 直近に触れたファイル) を根にしない
    — 書込みゲート (f_03 §4.y) は依頼文の文字列だけを証拠に数える。
    ``fallback=False`` は依頼にフォルダが無くても ``outputs_dir`` を足さない
    (裸のファイル名の探し場所、``file_ledger.resolve_bare_filename``)。
    """
    from backend.free.agent.tool_call_judge import _extract_file_path
    from backend.free.agent.tool_judge_args import _extract_file_path_literal

    roots: list[Path] = []

    def _add(root: Path) -> None:
        if root not in roots:
            roots.append(root)

    for path, _is_output in explicit_query_dirs(query or ""):
        _add(path)
    qpath = (_extract_file_path if implicit else _extract_file_path_literal)(query or "")
    if qpath and ("\\" in qpath or "/" in qpath):
        qp = Path(qpath)
        try:
            is_dir = qp.is_dir()
        except OSError:
            is_dir = False
        _add(qp if is_dir or not qp.suffix else qp.parent)
    if not roots and fallback:
        from backend.config import resolve_outputs_dir

        roots.append(resolve_outputs_dir())
    return roots


def _path_key(path: str) -> str:
    """包含判定用の正規形 (区切りを ``/`` に、``..`` を畳む。ドライブ / UNC は大小文字を無視)。"""
    text = posixpath.normpath(str(path).replace("\\", "/"))
    if (len(text) > 1 and text[1] == ":") or text.startswith("//"):
        return text.lower()
    return text


def relative_to_roots(path: str, roots: list[Path]) -> str | None:
    """``path`` がどれかの根の配下ならその根からの相対パス (``/`` 区切り、根そのものなら "")。外なら ``None``。"""
    key = _path_key(path)
    normalized = posixpath.normpath(str(path).replace("\\", "/"))
    for root in roots:
        root_key = _path_key(str(root)).rstrip("/")
        if key == root_key:
            return ""
        if key.startswith(root_key + "/"):
            return normalized[len(root_key) + 1:]
    return None


def fetch_cache_key(tool_name: str, tool_args: dict | None) -> str:
    """同じターンで取り直さない取得の鍵 (``fetch_url`` の URL。対象外なら空文字列)。

    ``read_file`` は対象外 — ローカルの読み直しは安く、同じターンの書込みで中身が
    変わりうる (docs/f_03 §4.2.1)。
    """
    if tool_name != "fetch_url":
        return ""
    return normalize_fetch_url(str((tool_args or {}).get("url") or ""))


def retrieval_call_key(tool_name: str, tool_args: dict | None) -> str:
    """同じ場所への取得の呼出しかを比べる鍵 (ツール名 + 場所。パスは ``_path_key``、URL は取得の鍵の正規形)。

    綴りの違う同じ場所 (ドライブの大小文字・区切りの向き・末尾の区切り) を同じに数える
    (#887 の残件 LOW-2、f_03 §4.2.1)。パターン・件数などの付随の引数は含めない — 出所は
    ループの中の繰り返しと結果の一致が確かめる (``# TODO`` / ``max_results`` を足した繰り返し)。
    """
    args = tool_args or {}
    url_key = fetch_cache_key(tool_name, args)
    if url_key:
        return f"{tool_name}:{url_key}"
    location = args.get("file_path") or args.get("directory") or ""
    return f"{tool_name}:{_path_key(str(location)) if location else ''}"


#: ``search_code`` の該当の行 (``<パス>:<行番号>: <本文>``) の先頭のパス
_SEARCH_HIT_PATH_RE = re.compile(r"^(?P<path>.+?)(?=:\d+: )", re.MULTILINE)


def same_retrieval_result(tool_name: str, earlier: str, later: str) -> bool:
    """同じ場所の 2 つの結果が同じ中身か (引数の綴りが写る部分は比べない)。

    ``read_file`` は先頭のメタ行 (引数の綴りのまま) を除き、``search_code`` は該当の行の先頭の
    パス (場所の引数の綴りのまま) を ``_path_key`` で畳んで比べる。
    """
    if tool_name == "read_file":
        from backend.free.agent.deliberative import strip_read_file_meta_line

        return strip_read_file_meta_line(earlier) == strip_read_file_meta_line(later)
    if tool_name == "search_code":
        def fold(text: str) -> str:
            return _SEARCH_HIT_PATH_RE.sub(lambda m: _path_key(m.group("path")), text)

        return fold(earlier) == fold(later)
    return earlier == later


def normalize_fetch_url(url: str) -> str:
    """取得の鍵にする URL の最小限の正規化 (純粋関数)。

    スキームとホストを小文字に、フラグメントとパス末尾の ``/`` を落とす。
    クエリ文字列とパスの大小文字は残す (別の資源でありうる)。
    """
    raw = url.strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    return urlunsplit((
        parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"),
        parts.query, "",
    ))


def continues_after_last_url(description: str) -> bool:
    """タスク文の最後の URL の後に文字 (英数字・かな漢字) が続くか (純粋関数)。

    続かないタスク (「Fetch the content of https://…」) は取得が目的語で終わる取得
    専任とみなす。続くタスク (「Fetch https://… and summarize the content」) は取得の
    後に処理を求める。語彙は見ない (不変則 #14、docs/f_03 §4.2.1)。URL の無い
    タスクは偽。
    """
    matches = list(_URL_IN_QUERY_RE.finditer(description))
    if not matches:
        return False
    return any(ch.isalnum() for ch in description[matches[-1].end():])


#: ``_try_fast_path`` にツール判定が渡されていない印 (``None`` は「ツール無し」の判定)。
_UNJUDGED = object()

#: 取得の後に処理 (要約・計算・抽出) を求める計画の種別 (``json_schemas.PLAN_TASK_KINDS``)。
PROCESSING_PLAN_KINDS: frozenset[str] = frozenset({"process", "retrieve_then_process"})


@dataclass(frozen=True)
class _AnswerFromRepeat:
    """ループの 1 手が「繰り返した取得の結果からツール無しで答えよ」を返した印 (f_03 §4.2.1)。"""

    data: list[str]


def _fetch_timeout(tool_args: dict | None, default: float) -> float:
    """``fetch_url`` の呼出しの実効の ``timeout`` (明示が無い・読めなければ ``default``)。"""
    try:
        return float((tool_args or {}).get("timeout") or default)
    except (TypeError, ValueError):
        return float(default)


def search_without_location(tool_name: str, tool_args: dict | None) -> bool:
    """場所の無い ``search_code`` か (``directory`` が無い・``.``・絶対パスでない)。meta はこれを撃たない。

    場所の無い・相対の検索はプロセスの CWD (インストール根) を基準に舐める — meta 経路で利用者が
    求める場所ではない (run23 反証 HIGH-1 / 2 周目 MED-1: 相対の ``docsx`` がインストール根の文書を
    書いた、f_03 §4.2.1)。依頼が名指したフォルダの下に在る相対の場所は、ループの引数の正規化が
    絶対パスへ直してから届く。
    """
    if tool_name != "search_code":
        return False
    directory = str((tool_args or {}).get("directory") or "").strip()
    return not directory or not Path(directory).is_absolute()


def write_scope_only_folders(query: str, write_destinations: tuple[str, ...]) -> frozenset[str]:
    """依頼文が書込みの節の前置きでだけ名指した、このターンの書込み先のフォルダ (``_path_key`` の集合)。

    「In <work>, save the TODO lines you find to <work>/r.md」の <work> — 依頼は出力の場所として
    名指しただけで、探す場所ではない。計画のタスク文が「… under <work>」とそこを場所に書いても
    探さない (run23 O4、f_03 §4.2.1)。依頼文の検索の節で名指したフォルダ (「Search for TODO in X
    and save … to X/r.md」「In X, search for TODO …」) は含めない。語彙は足さない
    (``_named_directories`` の ``skip_write_scopes`` と同じ部品)。
    """
    from backend.free.agent.meta_cognitive_tools import _named_directories

    destination_dirs = {_path_key(str(Path(dest).parent)) for dest in write_destinations if dest}
    if not destination_dirs:
        return frozenset()
    searched = {_path_key(d) for d in _named_directories(query, skip_write_scopes=True)}
    return frozenset(
        key for key in (_path_key(d) for d in _named_directories(query))
        if key in destination_dirs and key not in searched
    )


def _fast_path_miss_context(tool_name: str | None, text: str) -> str:
    """空振りしたツール結果を、回答ではなく文脈として渡すための 1 行。"""
    window_note = search_history_window_note(text or "") if tool_name == "search_history" else ""
    if window_note:
        # 日付の窓の空振りは期間の注記を切らずに渡す。「会話の内容から答え」を添えると
        # 今日の記憶をその日の会話として語る (実機 2026-10-03 run9)。
        return f"[ツール実行結果] search_history は情報を返さなかった。{window_note}"
    return (
        f"[ツール実行結果] {tool_name or 'tool'} は情報を返さなかった "
        f"({(text or '').strip()[:120]})。この結果は回答ではない。会話の内容から"
        "答え、依頼が「覚えておいて」等の保存指示なら受け取った内容を言い換えて確認する。"
    )


class _TaskExecutionMixin:
    """タスク 1 件の実行 — ツールループ / LLM 呼び出し / ツール実行。

    計画された ``TaskItem`` を 1 件受け取り、ツールを撃ちながら完了まで
    運ぶ層。``MetaCognitiveAgent`` の一部として mixin される
    (``self`` は同一インスタンスで、他の責務のメソッドも参照できる)。
    """

    async def _execute_task(
        self,
        task: TaskItem,
        original_query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        tools_registry,
        context_parts: list[str],
        on_step=None,
        task_index: int = 1,
        total_tasks: int = 1,
        generation_params: dict | None = None,
    ) -> tuple[str, list[dict]]:
        """1つのタスクを実行（ツールループ付き）"""
        prefix = f"[{task_index}/{total_tasks}]"
        # 取得の印は結果を返した経路が立てる (書込みの再試行で前の印を持ち越さない)
        task.retrieved = False
        task.output_note = ""
        # ループの繰り返しの扱いが見る: このタスクの開始時の素材の数 / 前のタスクまでの
        # 主の取得の数 / 自分の主の取得の空振り
        self._materials_at_task_start = len(getattr(self, "_fetched_tool_outputs", None) or [])
        self._primary_at_task_start = len(getattr(self, "_primary_results", None) or [])
        self._primary_retrieval_missed = False
        # ループを同じ結果の繰り返しで打ち切ったか (付随の素材を捨てるかを決める、run23 Z1r)
        self._stopped_by_repeat = False

        if tools_registry is not None:
            judgement = await self._judge_tool_for_task(
                task.description, tools_registry,
            )
            # ── 取得済みデータからの答え (ツールループに入れない、f_03 §4.2.1) ──
            if self._answers_from_retrieved(task, judgement):
                answered = await self._answer_from_retrieved(
                    task, original_query, system_prompt, llm_client,
                    context_parts, on_step, prefix, generation_params,
                )
                if answered is not None:
                    return answered
                if judgement is not None and judgement.tool_name == "fetch_url":
                    # ファストパスは取得済みの結果を取り直さずに返すだけ — それを答えに
                    # せず、作り直さずに失敗にする (タスク文が URL で終わるタスクも)。
                    # 判定がツール無しのタスクは従来どおりツールループへ
                    return self._retrieved_answer_failed(task), []

            # ── ファストパス ──
            fast_result = await self._try_fast_path(
                task, original_query, tools_registry, llm_client,
                on_step, prefix, judgement=judgement,
            )
            if fast_result is not None:
                text, entries = fast_result
                if self._listing_stands_in_for_a_read(judgement):
                    # 中身を読む依頼が一覧しか得られなかった: 取得は試みたが素材は無い。後の
                    # 書込みは門で no_source_data になり、一覧から本文を作話しない (run23 O3)
                    self._retrieval_attempted = True
                written = await self._write_after_search(
                    task, judgement, entries, original_query,
                    llm_client, tools_registry, on_step, prefix,
                )
                if written is not None:
                    return written
                # 読み取り系ツールが **情報を得られなかった** (0 件 / エラー) 結果は
                # 回答ではない。そのまま返すと「No results found for: …」が
                # ユーザーへの返答になる (2026-09-10 ライブ監査 (g) G-03:
                # 「ポンドの計算結果を覚えておいてください」に search_history の
                # 空振り文字列を返した)。deliberative は空振りを注記にして LLM に
                # 答えさせる。こちらも同じく、空振りを文脈に添えて通常経路へ落とす。
                if entries and not any(e.get("success") for e in entries):
                    logger.info(
                        "Tool fast path yielded no information; falling back to "
                        "the LLM with the result as context: %s", text[:80],
                    )
                    context_parts = [
                        *context_parts,
                        _fast_path_miss_context(entries[0].get("tool"), text),
                    ]
                    # ファストパスの呼出しもこのタスクの呼出しに数える (結末の判定が見る)。
                    # 自分の取得の空振りはループが直しうるので、前の取得の門は掛けない
                    # (原本のパスを直して読む、反証レビュー MED-1)
                    self._primary_retrieval_missed = True
                    loop_text, loop_calls = await self._run_tool_loop(
                        task, original_query, system_prompt, conversation,
                        llm_client, tools_registry, context_parts,
                        on_step, prefix, generation_params,
                    )
                    return loop_text, [*entries, *loop_calls]
                elif (
                    self._continues_after_fetch(task, judgement)
                    or self._processes_after_retrieval(task, judgement)
                ):
                    # 取得と処理を 1 タスクに持つタスクは取得の生の出力で終えない
                    # (f_03 §4.2.1、2026-10-03 ライブ監査 run17)。URL 以外の取得
                    # (read_file 等) は計画の種別で読み、渡すのはこのタスクの取得だけ
                    # (2026-10-05 ライブ監査: 「Summarize the content of …notes.txt」が
                    # ファイルの中身の生の出力のまま答えになった)
                    answered = await self._answer_from_retrieved(
                        task, original_query, system_prompt, llm_client,
                        context_parts, on_step, prefix, generation_params,
                        data=(
                            self._task_url_data(task)
                            if judgement.tool_name == "fetch_url" else [text]
                        ),
                    )
                    task.retrieved = False
                    if answered is None:
                        # 取得の生の出力を答えにしない (同じページが本文に 2 度並んだ)
                        return self._retrieved_answer_failed(task), entries
                    return answered[0], entries
                else:
                    return fast_result

        # ── 通常パス（ツールループ） ──
        return await self._loop_unless_no_source(
            task, original_query, system_prompt, conversation,
            llm_client, tools_registry, context_parts,
            on_step, prefix, generation_params,
        )

    async def _loop_unless_no_source(
        self, task: TaskItem, original_query: str, system_prompt: str,
        conversation: list[dict], llm_client, tools_registry, context_parts: list[str],
        on_step, prefix: str, generation_params: dict | None,
    ) -> tuple[str, list[dict]]:
        """ツールループを回す。ただし前の取得が素材を返さなかった書込みのタスクは回さない。

        書込みのファストパスを通らない書込みのタスク (棄権の「Save the search results to R」)
        は、ループに入れると同じ 0 件の検索を繰り返して Step limit で終えた (2026-10-04
        ライブ監査 run21 C1、f_03 §4.2.1)。このタスクは合流点 (``_resolve_write_content``) を
        通す書込みへ進め、決定論の素材 (依頼の引用・直前の応答) を先に見て、無ければ合流点の
        門が本文を生成せずに ``no_source_data`` で失敗にする。
        """
        file_path = (
            self._requested_write_path(task)
            if getattr(self, "_earlier_retrieval_attempted", False) and not task.fetch_only
            else ""
        )
        if (
            getattr(self, "_earlier_retrieval_attempted", False)
            and not self._has_source_material(original_query, file_path or "")
            and not task.fetch_only
        ):
            # 宛先以外のパス・URL を名指すタスクは自分で原本を読みうるのでループへ (反証 MED-B)
            if file_path and not self._names_a_source(task.description, file_path):
                logger.info(
                    "Write task sent to the write funnel instead of the tool loop "
                    "because this turn's retrieval returned no data: %s",
                    task.description[:80],
                )
                return await self._execute_write_fast(
                    task, original_query, file_path, llm_client, tools_registry,
                    on_step=on_step, prefix=prefix,
                )
        return await self._run_tool_loop(
            task, original_query, system_prompt, conversation,
            llm_client, tools_registry, context_parts,
            on_step, prefix, generation_params,
        )

    async def _write_after_search(
        self, task: TaskItem, judgement, entries: list[dict], original_query: str,
        llm_client, tools_registry, on_step, prefix: str,
    ) -> tuple[str, list[dict]] | None:
        """検索と保存を 1 タスクに持つタスクを、検索の後の書込みまで進める (f_03 §4.2.1)。

        判定器の ``search_code`` だけで終えると、保存先を作らないまま検索の生の出力を
        答えにして done になる (2026-10-04 反証レビュー MED-2)。検索の結果は素材に
        積んであり、0 件なら書込みの門 (``no_source_data``) が失敗にする。

        書込みを **起こす** ので、判定点 ``task_write_intent`` の ``fire`` か、棄権でも宛先を
        含む節が書込みの命令形で始まる (「… and save the matching lines to R」) タスクに
        限る。宛先の標識だけの棄権 (「draft a report I can paste into R」「I will later save
        them to R myself」) と、書込みを断った依頼 (「but do not save to R」) では書かない
        (2026-10-04 反証レビュー 3 周目 HIGH-1)。それ以外は None (従来の経路)。
        """
        if judgement is None or judgement.tool_name != "search_code":
            return None
        file_path = self._requested_write_path(task)
        if not file_path:
            return None
        if not task_writes(task):
            logger.info(
                "Write after search: abstained task upgraded to a write "
                "(evidence=destination_clause_imperative): %s", task.description[:80],
            )
        # このタスクの検索は書込みにとって「前の手順」
        self._earlier_retrieval_attempted = self._retrieval_attempted
        task.retrieved = False
        text, write_entries = await self._execute_write_fast(
            task, original_query, file_path, llm_client, tools_registry,
            on_step=on_step, prefix=prefix,
        )
        return text, [*entries, *write_entries]

    async def _try_fast_path(
        self,
        task: TaskItem,
        original_query: str,
        tools_registry,
        llm_client,
        on_step,
        prefix: str,
        *,
        judgement=_UNJUDGED,
    ) -> tuple[str, list[dict]] | None:
        """ファストパス判定・実行。適用不可なら None を返す"""
        if judgement is _UNJUDGED:
            judgement = await self._judge_tool_for_task(
                task.description, tools_registry,
            )
        if judgement is None or not judgement.tool_needed or not judgement.tool_name:
            return None

        if judgement.tool_name == "write_file" and (
            task_write_verdict_of(task).evidence == PLAN_KIND_ABSTAIN_EVIDENCE
        ):
            # 計画モデルが処理 / 取得と付け、タスク文に宛先の無いタスク (「Identify
            # the bugs and determine the fixes」) は書かない (2026-10-05 ライブ監査 T5、
            # 判定点 task_write_intent の確認票)。
            logger.info(
                "Write fast path skipped: planner labelled the task %s without a "
                "destination: %s", task.plan_kind, task.description[:80],
            )
            return None
        if (
            judgement.tool_name == "write_file"
            and tools_registry.has("write_file")
        ):
            # パスが無い (「同じファイルに保存し直して」型) か、ディレクトリを
            # 伴わない裸のファイル名 (「notes.txt に追記して」) は保存先が
            # 確定していないので直近会話から引く。ディレクトリ付きパスはその
            # まま返るので無条件に通してよい (実測 2026-07-27: パス不明のまま
            # 生成だけ走りファイルは旧内容のままだった / 2026-08-09: 裸名を
            # そのまま渡すとカレントディレクトリに別物を作る)。
            file_path = judgement.tool_args.get("file_path", "")
            file_path = self._referential_write_path(file_path or None) or file_path
            if file_path:
                return await self._execute_write_fast(
                    task, original_query, file_path,
                    llm_client, tools_registry,
                    on_step=on_step, prefix=prefix,
                )
        elif tools_registry.has(judgement.tool_name):
            if judgement.tool_name == "search_code" and _path_key(
                str((judgement.tool_args or {}).get("directory") or "."),
            ) in write_scope_only_folders(
                original_query, self._plan_write_destinations(original_query),
            ):
                # 依頼が出力の場所としてだけ名指したフォルダを、計画のタスク文が探す場所に
                # 書いた: 名指しの無い検索と同じくループへ委ねる (run23 O4、f_03 §4.2.1)
                logger.info(
                    "Fast path skipped: search_code directory is only the output folder "
                    "named by the request: %s", judgement.tool_args.get("directory"),
                )
                return None
            # ツールの必須引数が揃っているか確認（不足時は通常ループに委譲）
            tool_def = tools_registry.get(judgement.tool_name)
            if tool_def and tool_def.parameters and not judgement.tool_args:
                logger.debug(
                    "Fast path skipped: %s requires args but none provided",
                    judgement.tool_name,
                )
                return None
            return await self._execute_tool_fast(
                judgement.tool_name, judgement.tool_args, task,
                tools_registry,
                on_step=on_step, prefix=prefix,
                original_query=original_query,
            )

        return None

    def _turn_fetched(self, tool_name: str, tool_args: dict) -> str | None:
        """このターンで既に取得した同じ取得の結果 (無ければ ``None``。参照だけ)。"""
        key = fetch_cache_key(tool_name, tool_args)
        if not key:
            return None
        return (getattr(self, "_turn_fetches", None) or {}).get(key)

    def _turn_fetch_error(self, tool_name: str, tool_args: dict) -> str | None:
        """このターンで同じ URL の取得が返したエラー (取りに行き直してよければ ``None``)。

        鍵は ``fetch_cache_key`` (URL だけ) — 拒否・名前解決の失敗・接続の失敗を ``timeout`` を
        変えて取りに行き直した (run23 F2 / F2b、f_03 §4.2.1)。時間切れは、前より長い ``timeout`` を
        明示した取り直しだけ通す (反証 LOW-1)。
        """
        key = fetch_cache_key(tool_name, tool_args)
        if not key:
            return None
        remembered = (getattr(self, "_turn_fetch_errors", None) or {}).get(key)
        if remembered is None:
            return None
        text, timeout_used = remembered
        if timeout_used is not None and _fetch_timeout(
            tool_args, self._default_fetch_timeout(),
        ) > timeout_used:
            return None
        return text

    def _default_fetch_timeout(self) -> float:
        """``fetch_url`` の既定の timeout (設定 ``tools.fetch_url_timeout``、既定 10。tools/builtin.py と同じ)。"""
        cfg = getattr(self, "config", None)
        tools_cfg = cfg.get("tools", {}) if isinstance(cfg, dict) else {}
        try:
            return float(tools_cfg.get("fetch_url_timeout", 10))
        except (TypeError, ValueError):
            return 10.0

    def _remember_fetch_error(self, tool_name: str, tool_args: dict, result_text: str) -> None:
        """時間で直らない取得のエラー (拒否・名前解決・接続の拒否) と時間切れを URL の鍵で覚える。

        時間切れは使った ``timeout`` (明示が無ければ 0) と一緒に覚える。それ以外のエラー (HTTP の
        状態エラー等) は覚えない。
        """
        key = fetch_cache_key(tool_name, tool_args)
        if not key or not is_tool_error(result_text):
            return
        exc_type = fetch_error_type(result_text)
        if tool_error_kind(result_text) in ("blocked", "unresolved_host") or exc_type == "ConnectError":
            timeout_used = None
        elif exc_type in FETCH_UNREACHABLE_ERRORS:
            timeout_used = _fetch_timeout(tool_args, self._default_fetch_timeout())
        else:
            return
        errors = getattr(self, "_turn_fetch_errors", None)
        if errors is None:
            errors = self._turn_fetch_errors = {}
        errors[key] = (result_text, timeout_used)

    def _remember_retrieval(
        self, tool_name: str, tool_args: dict, result_text: str,
    ) -> None:
        """取得の結果を後続タスクの素材とこのターンの取得済み URL へ積む。"""
        sources = getattr(self, "_material_tools", None)
        if sources is None:
            sources = self._material_tools = {}
        sources[len(self._fetched_tool_outputs)] = tool_name
        self._fetched_tool_outputs.append(result_text)
        key = fetch_cache_key(tool_name, tool_args)
        if key:
            self._turn_fetches[key] = result_text

    def _settle_failed_task_materials(
        self, task: TaskItem, tool_calls: list[dict], before: tuple[int, int],
    ) -> None:
        """主の取得が素材を返さなかったタスクの、その間に積んだ付随の素材と取得済み URL を捨てる。

        主の取得は判定したツールを撃つファストパスの取得 (記録に ``fast_path``)。

        - 失敗したタスクが主の取得に成功していなければ、積んだ素材をすべて捨てる。成功して
          後の処理で失敗したタスクの取得は依頼の対象なので残す (f_03 §4.2.1、run21 Z1 / 反証 MED-2)。
        - 主の取得が情報を返さなかった (0 件、エラーでない) タスクを同じ結果の繰り返しで打ち切った
          なら、done でも、ループが主と別のツールで積んだ素材 (一覧の後の無関係な readme.txt) を
          捨てる。主と同じツールの取得 (TODO が無ければ FIXME の検索) は残す (run23 Z1r)。モデルが
          自分の答えで終えたタスク (正しいファイルを読んで答えた) の読込みは残す (反証 MED-1)。
        """
        outputs_len, fetches_len = before
        primary = next((tc for tc in tool_calls if tc.get("fast_path")), None)
        if task.status == "failed" and not (primary and primary.get("success")):
            keep_tool = None
        elif (
            primary and not primary.get("success") and not primary.get("error")
            and getattr(self, "_stopped_by_repeat", False)
        ):
            keep_tool = primary.get("tool")
        else:
            return
        sources = getattr(self, "_material_tools", None) or {}
        kept = [
            (sources.get(i), text)
            for i, text in enumerate(self._fetched_tool_outputs[outputs_len:], outputs_len)
            if keep_tool is not None and sources.get(i) == keep_tool
        ]
        del self._fetched_tool_outputs[outputs_len:]
        for i in [i for i in sources if i >= outputs_len]:
            del sources[i]
        for tool_name, text in kept:
            sources[len(self._fetched_tool_outputs)] = tool_name
            self._fetched_tool_outputs.append(text)
        # 取得済み URL は fetch_url の取得だけが積む (情報ゼロの主の取得は検索なので別のツール)
        for key in list(self._turn_fetches)[fetches_len:]:
            del self._turn_fetches[key]

    def _repeated_in_task(self, tool_name: str, tool_args: dict, result_text: str) -> bool:
        """同じ呼出しがこのタスクの中で既に同じ結果を返したか (``_repeated_result`` の更新の前に呼ぶ)。

        ``_task_repeat_counts`` はタスクの開始時に空にするので、鍵があればこのタスクで撃った呼出し。
        """
        key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"
        return (
            key in self._task_repeat_counts
            and self._turn_call_results.get(key) == result_text
        )

    def _repeated_result(self, tool_name: str, tool_args: dict, result_text: str) -> int:
        """取得系の同じ呼出しが同じ結果を返した繰り返しの、実行中のタスクでの回数 (初めてなら 0)。

        0 件も該当ありも同じに数える (該当を返した検索を繰り返して Step limit で終えた、
        run22 B1r)。エラーは連続エラーの打ち切りに任せる。結果の記憶はターン全体で持ち、
        回数はタスクの開始時に空にする (前のタスクの繰り返しで後のタスクを注記なしに
        止めない、f_03 §4.2.1)。
        """
        if tool_name not in _DATA_BEARING_TOOLS or is_tool_error(result_text):
            return 0
        key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"
        counts = self._task_repeat_counts
        repeats = counts.get(key, 0) + 1 if self._turn_call_results.get(key) == result_text else 0
        self._turn_call_results[key] = result_text
        counts[key] = repeats
        return repeats

    @staticmethod
    def _listing_stands_in_for_a_read(judgement) -> bool:
        """判定が、中身を読む依頼の対象のフォルダを ``list_directory`` に振り替えたか。

        判定器の明示パスの層 (``explicit_path``) が ``list_directory`` を選ぶのは、読込みの依頼の
        パスがフォルダだったときだけ (``ToolCallJudge._infer_tool``)。一覧を求める依頼は列挙の
        規則が選ぶ。語彙は見ず、判定の出所で分ける (f_03 §4.2.1)。
        """
        return (
            judgement is not None
            and judgement.tool_name == "list_directory"
            and getattr(judgement, "decided_reason", "") == "explicit_path"
        )

    @staticmethod
    def _names_a_source(description: str, file_path: str) -> bool:
        """タスク文が宛先以外のドライブ付きパスか URL を名指すか (純粋関数、語彙は見ない)。"""
        if _URL_IN_QUERY_RE.search(description):
            return True
        destination = _path_key(file_path)
        return any(
            _path_key(m.group(0).rstrip("。、,.\\/")) != destination
            for m in EXPLICIT_WINDOWS_PATH_RE.finditer(description)
        )

    def _requested_write_path(self, task: TaskItem) -> str | None:
        """書込みを求めるタスクの宛先 (求めなければ None)。

        判定点 ``task_write_intent`` の ``fire`` か、棄権でも宛先を含む節が書込みの命令形で
        始まるタスク (f_03 §4.2.1)。書込みを断った依頼 (``write_prohibited``) は求めない。
        """
        from backend.free.agent.task_write_gate import destination_clause_requests_write
        from backend.free.agent.tool_judge_args import extract_write_target_path
        from backend.free.core.intent_vocab import write_prohibited

        if write_prohibited(task.description):
            return None
        file_path = (
            extract_write_target_path(task.description)
            or self._referential_write_path(None)
        )
        if not file_path:
            return None
        if task_writes(task) or (
            task_write_band(task) == "abstain"
            and destination_clause_requests_write(task.description, file_path)
        ):
            return file_path
        return None

    def _answers_from_retrieved(self, task: TaskItem, judgement) -> bool:
        """取得済みデータから答えるタスクか (ツールループに入れない、docs/f_03 §4.2.1)。

        このターンで取得したデータがあり、書込みのタスク (判定点
        ``task_write_intent`` の ``fire`` / ``abstain``) でも取得専任でもなく、タスク文の URL がすべて取得済みのタスクで、ツール判定が
        「ツール無し」か「取得済みの URL への取り直し」のとき。ループはタスクを元の
        依頼文のまま渡し、取得済みデータは先頭 200 字しか見えないため、モデルが
        取り直していた (2026-10-03 ライブ監査 run16)。
        """
        if not getattr(self, "_fetched_tool_outputs", None):
            return False
        if task.fetch_only or task_may_write(task):
            return False
        if not self._task_urls_retrieved(task):
            return False
        if judgement is None or not judgement.tool_needed or not judgement.tool_name:
            return True
        return self._turn_fetched(
            judgement.tool_name, judgement.tool_args or {},
        ) is not None

    def _task_urls_retrieved(self, task: TaskItem) -> bool:
        """タスク文の URL がすべてこのターンで取得済みか (URL が無ければ真)。"""
        fetched = getattr(self, "_turn_fetches", None) or {}
        return all(
            normalize_fetch_url(m.group(1)) in fetched
            for m in _URL_IN_QUERY_RE.finditer(task.description)
        )

    def _task_url_data(self, task: TaskItem) -> list[str]:
        """タスク文が名指す URL のこのターンの取得結果 (重複を除き、文中の順)。"""
        fetched = getattr(self, "_turn_fetches", None) or {}
        keys = dict.fromkeys(
            normalize_fetch_url(m.group(1))
            for m in _URL_IN_QUERY_RE.finditer(task.description)
        )
        return [fetched[k] for k in keys if k in fetched]

    def _continues_after_fetch(self, task: TaskItem, judgement) -> bool:
        """ファストパスの ``fetch_url`` で取得したタスクが、取得の後に処理を求めるか。

        取得専任の印 (``fetch_only``) は計画の集約でしか立たないので、タスク文の
        構造 (``continues_after_last_url``) で補う。書込みのタスク (``task_may_write``) と、
        タスク文の URL に未取得のものが残るタスクは対象外 (docs/f_03 §4.2.1)。
        """
        if not task.retrieved or task.fetch_only:
            return False
        if judgement is None or judgement.tool_name != "fetch_url":
            return False
        if task_may_write(task):
            return False
        return (
            continues_after_last_url(task.description)
            and self._task_urls_retrieved(task)
        )

    @staticmethod
    def _processes_after_retrieval(task: TaskItem, judgement) -> bool:
        """ファストパスで URL 以外の取得 (``read_file`` 等) をしたタスクが、取得の後に処理を求めるか。

        処理を求めるかは計画モデルの種別で読む (``process`` / ``retrieve_then_process``)。
        「Summarize the content of E:\\…\\notes.txt」はタスク文がパスで終わるので、URL の
        ``continues_after_last_url`` の構造では読めない。種別の誤りの害は小さい — 取得だけの
        タスクを処理と誤れば取得したデータからの答えになり (本文に生の出力が出ない)、
        処理を取得と誤れば従来どおり。書込みのタスク (``task_may_write``) と取得専任は
        対象外 (docs/f_03 §4.2.1、2026-10-05 ライブ監査)。
        """
        if not task.retrieved or task.fetch_only:
            return False
        if judgement is None or judgement.tool_name == "fetch_url":
            return False
        if judgement.tool_name not in _DATA_BEARING_TOOLS:
            return False
        if task_may_write(task):
            return False
        return task.plan_kind in PROCESSING_PLAN_KINDS

    @staticmethod
    def _retrieved_answer_failed(task: TaskItem) -> str:
        """取得の後の処理 (答えの生成) に失敗したタスクを失敗で終える (docs/f_03 §4.2.1)。

        利用者向けの注記を刻み、``determine_task_status`` が失敗と読む結果を返す。
        取得の生の出力は答えにしない (後続タスクの素材としては既に積んである)。
        """
        logger.warning(
            "Retrieved data could not be processed; task failed: %s",
            task.description[:80],
        )
        task.failure_note = msg("agent.retrieved_answer_failed")
        return "Error: the answer could not be generated from the retrieved data"

    def _answer_context_parts(
        self, task: TaskItem, context_parts: list[str],
    ) -> list[str]:
        """前段の結果のうち、取得でない完了タスクの答えを 200 字で切らずに戻す。

        「要約 → 英訳」の英訳は前段の要約そのものが素材 (docs/f_03 §4.2.1)。
        """
        full: dict[str, str] = {}
        for earlier in getattr(self, "_plan_tasks", None) or []:
            if earlier is task:
                break
            if earlier.status == "done" and not earlier.retrieved and earlier.result:
                full[f"[{earlier.description}]: {earlier.result[:200]}"] = (
                    f"[{earlier.description}]: {earlier.result}"
                )
        return [full.get(part, part) for part in context_parts]

    def _later_steps_text(self, task: TaskItem) -> str:
        """計画でこのタスクより後のステップを並べた注記 (無ければ空文字列)。"""
        tasks = getattr(self, "_plan_tasks", None) or []
        index = next((i for i, t in enumerate(tasks) if t is task), None)
        if index is None or index + 1 >= len(tasks):
            return ""
        return RETRIEVED_ANSWER_LATER_STEPS.format(
            steps="\n".join(f"- {t.description}" for t in tasks[index + 1:]),
        )

    async def _answer_from_retrieved(
        self,
        task: TaskItem,
        original_query: str,
        system_prompt: str,
        llm_client,
        context_parts: list[str],
        on_step,
        prefix: str,
        generation_params: dict | None,
        *,
        data: list[str] | None = None,
    ) -> tuple[str, list[dict]] | None:
        """取得済みデータを渡した 1 回の生成でタスクの答えを作る。

        ツールの選択肢を見せないので取り直しが起きない。前段の結果と記憶 / 参考
        コンテキスト / 添付ファイル / 参考例はツールループと同じ部品
        (``_loop_context_text`` / ``_with_turn_context_blocks``) で、言語の指示と
        現在日時は書込み本文の生成と同じもので渡す。データは
        ``_inject_fetched_data`` (実際の system の長さを引いた予算) で渡し、``data``
        を渡すとターンの取得全体の代わりにそれだけを渡す。依頼文は全体の参考として
        示し、計画の後のステップはここでは行わないと書く。取得でない前段の答えは
        切り詰めずに渡す。生成が空・時間切れ・ツール呼出の JSON なら ``None`` を
        返す (呼出側が従来の経路へ落とす)。
        """
        if on_step:
            await call_callback(on_step, {
                "type": "llm",
                "detail": f"{prefix} 取得済みデータから回答を生成中...",
                "status": "running",
            })
        ctx_size = resolve_context_size_for_mode(self.config, self._mode)
        step_prompt = self._with_turn_context_blocks(
            RETRIEVED_ANSWER_SYSTEM_PROMPT.format(
                task=task.description,
                context=self._loop_context_text(
                    self._answer_context_parts(task, context_parts),
                ),
            ),
        )
        system_content = (
            f"{system_prompt}\n\n{step_prompt}" if system_prompt else step_prompt
        )
        system_content = f"{system_content}\n{content_language_directive()}"
        request = RETRIEVED_ANSWER_USER_PROMPT.format(
            query=original_query, task=task.description,
        )
        later = self._later_steps_text(task)
        if later:
            request = f"{request}\n\n{later}"
        # 取得した表の数値集計はコードで行い、確定値として渡す (docs/f_03 §4.2.2)
        aggregate = await aggregate_retrieved_table(
            llm_client, original_query, self._aggregate_materials(data),
            task=task.description,
        )
        if aggregate is not None:
            request = f"{request}\n\n{table_aggregate_block(aggregate)}"
        user_prompt = self._inject_fetched_data(
            self._inject_current_date(request),
            ctx_size,
            system_text=system_content,
            note=RETRIEVED_DATA_BLOCK_NOTE,
            outputs=data,
        )
        gen_kwargs = self._build_gen_kwargs(generation_params, llm_client)
        sampling = {
            k: v for k, v in gen_kwargs.items() if k not in ("max_tokens", "id_slot")
        }
        text, timed_out = await self._stream_text_with_idle_timeout(
            llm_client,
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._calc_gen_max_tokens(system_content + user_prompt, ctx_size),
            id_slot=gen_kwargs["id_slot"],
            step="retrieved_answer",
            **sampling,
        )
        text = text.strip()
        if (
            timed_out or not text
            or self._parse_tool_call(text) is not None
            or looks_like_tool_selector_json(text)
        ):
            logger.warning(
                "Answer from retrieved data unusable (timed_out=%s, chars=%d); "
                "falling back: %s",
                timed_out, len(text), task.description[:80],
            )
            return None
        logger.info(
            "Task answered from this turn's retrieved data (no tool loop): %s "
            "(%d chars)", task.description[:80], len(text),
        )
        # 渡した集計と別の値を述べたら正しい値を示す注記を持たせる。答えには混ぜない —
        # 答えは後のタスクの素材 (書込みの本文など) になり、最終応答の途中に注記が残ると
        # ``strip_system_notes`` (末尾だけを落とす) が記憶から落とせない。最終応答の末尾に
        # まとめて足す (``_build_final_response``)
        task.output_note = table_aggregate_mismatch_note(text, aggregate)
        if task.output_note:
            logger.info(
                "Answer contradicts the table aggregate computed by code: %s",
                task.description[:80],
            )
        return text, []

    def _aggregate_materials(self, data: list[str] | None) -> list[str]:
        """表の集計に使う取得結果 (このタスクの分。無ければこのターンの分)。

        ``data`` (呼出側がこのタスクのものとして渡した取得) があればそれだけ。無ければ
        このタスクが積んだ素材、それも無ければ (取得の後の計算のタスク) ターン全体の
        素材。ターン全体のときに別のファイルの表が 2 つ以上あれば ``find_table`` が
        集計しない (2026-10-05 レビュー)。
        """
        if data is not None:
            return list(data)
        outputs = list(getattr(self, "_fetched_tool_outputs", None) or [])
        own = outputs[getattr(self, "_materials_at_task_start", len(outputs)):]
        return own or outputs

    def _plan_write_destinations(self, query: str) -> tuple[str, ...]:
        """計画の書込みのタスク (``task_may_write``) の宛先 (書込み時と同じく解決したもの)。"""
        from backend.free.agent.tool_judge_args import extract_write_target_path

        destinations: list[str] = []
        for task in getattr(self, "_plan_tasks", None) or []:
            if not task_may_write(task):
                continue
            target = extract_write_target_path(task.description)
            if target:
                destinations.append(self._resolve_write_path(target, query, task.description))
        return tuple(destinations)

    def _write_left_to_later_task(
        self, task: TaskItem, file_path: str, query: str,
    ) -> bool:
        """書込みのタスクでないものの書込み先を、計画の後の書込みのタスクが名指しているか。

        保存先のある依頼でツールループが依頼文の「保存して」に従って書き、後の
        書込みタスクがもう一度書いていた (2026-10-03 ライブ監査 run16)。書込みの
        タスク同士の順次の書込み (下書き → 手直し) は対象外。書込みのタスクかは
        判定点 ``task_write_intent`` の ``fire`` / ``abstain`` で見る (``task_may_write``)。
        """
        from backend.free.agent.tool_judge_args import extract_write_target_path

        if not file_path or task_may_write(task):
            return False
        tasks = getattr(self, "_plan_tasks", None) or []
        index = next((i for i, t in enumerate(tasks) if t is task), None)
        if index is None:
            return False
        key = _path_key(file_path)
        for later in tasks[index + 1:]:
            if not task_may_write(later):
                continue
            target = extract_write_target_path(later.description)
            if target and _path_key(
                self._resolve_write_path(target, query, later.description),
            ) == key:
                return True
        return False

    def _build_loop_system_prompt(
        self,
        task: TaskItem,
        context_parts: list[str],
        tools_registry,
        original_query: str = "",
    ) -> str:
        """ツールループ用 system プロンプトを構築する。

        コンテキストは 3000 文字で truncate し、ツール記述は registry から取得。
        ユーザーの質問がファイル / ディレクトリに一切触れていなければ、
        ファイル系ツールはメニューに載せない (無関係な ``list_directory()`` で
        リポジトリ一覧がコンテキストへ混入した 2026-09-05 F-08 の対策)。
        """
        tool_descriptions = ""
        if tools_registry is not None:
            exclude = (
                frozenset() if mentions_filesystem(original_query) or not original_query
                else FILESYSTEM_TOOL_NAMES
            )
            # 1 回の process() 内でタスク/反復ごとに再構築されていたので (mode, 除外) 別に cache
            cache = self._tool_descriptions_cache
            key = (self._mode, exclude)
            if key not in cache:
                cache[key] = tools_registry.get_descriptions_text(
                    mode=self._mode, exclude=exclude,
                )
            tool_descriptions = cache[key]
        prompt = EXECUTE_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions or "(no tools available)",
            task=task.description,
            context=self._loop_context_text(context_parts),
        )
        return self._with_turn_context_blocks(prompt)

    @staticmethod
    def _loop_context_text(context_parts: list[str]) -> str:
        """前段の結果を system に載せる形 (3000 文字で切る)。"""
        context_text = "\n".join(context_parts) if context_parts else "(none)"
        if len(context_text) > 3000:
            context_text = context_text[:3000] + "\n... (truncated)"
        return context_text

    def _with_turn_context_blocks(self, prompt: str) -> str:
        """ターンの記憶 / 参考コンテキスト / 添付ファイル / 参考例を system の後ろへ足す。

        ツールループと取得済みデータからの答え (``_answer_from_retrieved``) が共有する。
        """
        # SemMem メモリをループ全反復の system に維持する
        if self._semmem_block:
            prompt = f"{prompt}\n\n[関連する記憶]\n{self._semmem_block}"
        # search pipeline 取得済み RAG チャンクを参考コンテキストとして維持する
        if self._rag_block:
            prompt = f"{prompt}\n\n[参考コンテキスト]\n{self._rag_block}"
        # ユーザー添付ファイルをループ全反復の system に維持する
        if self._file_block:
            prompt = f"{prompt}\n\n[添付ファイル]\n{self._file_block}"
        # Level 1 で進化した few-shot を参考例として維持する
        if self._fewshot_block:
            prompt = f"{prompt}\n\n[参考例]\n{self._fewshot_block}"
        return prompt

    def _rebuild_loop_messages(
        self,
        prompt: str,
        conversation: list[dict],
        step_results: list[StepResult],
        original_query: str,
        compact_budget: int,
    ) -> list[dict]:
        """直前 step_results を圧縮して loop 用 messages を再構築する。"""
        compacted = self.compactor.compact(step_results, compact_budget)
        messages = build_messages_for_loop(
            prompt, conversation, compacted, self.config,
        )
        last = step_results[-1]
        last_context = (
            f"Your last action: {last.tool_name} → "
            f"{last.output[:200]}\n\n"
        )
        messages.append({
            "role": "user",
            "content": last_context + original_query,
        })
        return messages

    async def _call_llm_in_loop(
        self,
        llm_client,
        injected_messages: list[dict],
        gen_kwargs: dict,
        loop: int,
    ) -> tuple[str, bool]:
        """ツールループの LLM 呼び出しをストリーミングで実行し ``(text, timed_out)`` を返す。

        非ストリーミング + 総ウォールクロック打ち切りだと低速 GPU が長い応答を生成しきる
        前に殺されるため、``_stream_text_with_idle_timeout`` (first-token / idle / total) を
        使う。``max_tokens`` は利用可能コンテキストに収める。後処理はしない (ツールコール
        JSON のパースに生テキストが必要なため)。
        """
        prompt_text = "".join(m.get("content", "") for m in injected_messages)
        ctx_size = resolve_context_size_for_mode(self.config, self._mode)
        sampling = {
            k: v for k, v in gen_kwargs.items()
            if k not in ("max_tokens", "id_slot", "stream")
        }
        text, timed_out = await self._stream_text_with_idle_timeout(
            llm_client, injected_messages,
            max_tokens=self._calc_gen_max_tokens(prompt_text, ctx_size),
            id_slot=gen_kwargs.get("id_slot", -1),
            **sampling,
        )
        if timed_out:
            logger.warning("LLM call timed out in tool loop iteration %d", loop)
        return text.strip(), timed_out

    async def _stream_text_with_idle_timeout(
        self,
        llm_client,
        messages: list[dict],
        *,
        max_tokens: int,
        id_slot: int = -1,
        step: str = "tool_loop",
        **sampling,
    ) -> tuple[str, bool]:
        """ストリーミング生成をトークン間アイドルタイムアウトで読み取り ``(text, timed_out)`` を返す。

        最初の1トークンは ``content_gen_first_token_timeout``、以降はトークン間アイドル
        ``content_gen_idle_timeout`` で待ち、総上限 ``content_gen_timeout`` まで継続する。
        総ウォールクロックでは一律に打ち切らない (進行中の生成を殺さない)。無出力で停止した
        時だけ ``timed_out=True``。後処理はしない (生テキストを返す)。``_call_llm_in_loop`` /
        ``_fallback_plain_llm`` が共有する。``step`` は ``finish_reason=length`` を
        記録するときのラベル (``MetaCognitiveResponse.truncated_steps``)。
        """
        try:
            stream = await llm_client.generate(
                messages, stream=True,
                max_tokens=max_tokens, id_slot=id_slot, **sampling,
            )
        except asyncio.TimeoutError:
            return "", True
        agen = stream.__aiter__()
        chunks: list[str] = []
        start = time.monotonic()
        first_token = True
        try:
            while True:
                wait_timeout = (
                    self._content_gen_first_token_timeout
                    if first_token
                    else self._content_gen_idle_timeout
                )
                try:
                    token = await asyncio.wait_for(
                        agen.__anext__(), timeout=wait_timeout,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    return "", True
                first_token = False
                chunks.append(token)
                if time.monotonic() - start > self._content_gen_timeout:
                    return "", True
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
        note_stream_truncation(self, stream, step)
        return "".join(chunks), False

    @staticmethod
    def _append_timeout_recovery_messages(messages: list[dict]) -> None:
        """LLM タイムアウト後にループ続行する際のリトライメッセージを追記する。"""
        messages.append({"role": "assistant", "content": "(timeout)"})
        messages.append({
            "role": "user",
            "content": "Previous call timed out. Respond concisely.",
        })

    async def _try_recover_no_tool_call(
        self,
        text: str,
        task: TaskItem,
        original_query: str,
        llm_client,
        tools_registry,
        on_step,
        prefix: str,
        tool_calls: list[dict],
    ) -> tuple[str, list[dict]] | None:
        """`_parse_tool_call` が None のとき write_file 期待タスクならテキスト→ツール変換を試みる。

        成功時は `tool_calls` を mutate して `(result_text, tool_calls)` を返す。
        対象外 / 失敗時は `None` を返し、呼び出し側はテキストのまま終了する。
        """
        if not (
            task_writes(task)
            and tools_registry is not None
            and tools_registry.has("write_file")
        ):
            return None
        recovery = await self._recover_write_from_text(
            text, task, original_query, llm_client,
            tools_registry, on_step, prefix,
        )
        if recovery is None:
            return None
        tool_calls.append(recovery)
        return recovery.get("result", text), tool_calls

    @staticmethod
    def _next_consecutive_errors(
        step_results: list[StepResult],
        current: int,
    ) -> int:
        """直近 step_results に応じて consecutive_errors を更新する。"""
        last_step = step_results[-1] if step_results else None
        if last_step and is_tool_error(last_step.output):
            return current + 1
        return 0

    async def _run_tool_loop(
        self,
        task: TaskItem,
        original_query: str,
        system_prompt: str,
        conversation: list[dict],
        llm_client,
        tools_registry,
        context_parts: list[str],
        on_step,
        prefix: str,
        generation_params: dict | None,
    ) -> tuple[str, list[dict]]:
        """ツールループ本体: LLM 推論 → ツール実行を繰り返す"""
        tool_calls: list[dict] = []
        prompt = self._build_loop_system_prompt(
            task, context_parts, tools_registry, original_query=original_query,
        )

        state = AgentState(
            agent_layer="meta_cognitive",
            max_iterations=self.max_tool_iterations,
            expected_format="json",
        )

        step_results: list[StepResult] = []
        consecutive_errors = 0
        max_consecutive_errors = 3
        compact_budget = self.loop_budget - self._reminder_budget

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": original_query},
        ]

        for loop in range(self.max_tool_iterations):
            state.current_iteration = loop

            if step_results:
                messages = self._rebuild_loop_messages(
                    prompt, conversation, step_results,
                    original_query, compact_budget,
                )

            if on_step:
                await call_callback(on_step, {
                    "type": "llm",
                    "detail": f"{prefix} LLM 推論中... (ループ {loop + 1})",
                    "status": "running",
                })

            self._update_context_usage(state, messages)
            injected_messages = self.reminder_system.inject(messages, state)
            gen_kwargs = self._build_gen_kwargs(generation_params, llm_client)

            text, timed_out = await self._call_llm_in_loop(
                llm_client, injected_messages, gen_kwargs, loop,
            )
            if timed_out:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    return "Error: LLM call timed out repeatedly", tool_calls
                self._append_timeout_recovery_messages(messages)
                continue
            state.last_output = text

            tool_call = self._parse_tool_call(text)
            if tool_call is None:
                recovered = await self._try_recover_no_tool_call(
                    text, task, original_query, llm_client,
                    tools_registry, on_step, prefix, tool_calls,
                )
                if recovered is not None:
                    return recovered
                return text, tool_calls

            loop_result = await self._execute_loop_tool_call(
                tool_call, text, task, original_query,
                llm_client, tools_registry, on_step, prefix,
                state, tool_calls, step_results, messages,
                consecutive_errors, max_consecutive_errors,
                loop, generation_params,
            )
            if isinstance(loop_result, _AnswerFromRepeat):
                answered = await self._answer_from_retrieved(
                    task, original_query, system_prompt, llm_client,
                    context_parts, on_step, prefix, generation_params,
                    data=loop_result.data,
                )
                if answered is None:
                    return self._retrieved_answer_failed(task), tool_calls
                return answered[0], tool_calls
            if loop_result is not None:
                return loop_result

            consecutive_errors = self._next_consecutive_errors(
                step_results, consecutive_errors,
            )

        # 生のツールの出力ではなく、打ち切った理由を利用者へ伝える (f_03 §4.2.1)
        task.failure_note = task.failure_note or msg("agent.step_limit_reached")
        return "Step limit reached during task execution.", tool_calls

    def _update_context_usage(
        self, state: AgentState, messages: list[dict],
    ) -> None:
        """コンテキスト使用率を計算して AgentState に設定"""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        ctx_size = resolve_context_size_for_mode(self.config, self._mode)
        state.context_usage_pct = min(
            100,
            int(_estimate_tokens("x" * total_chars) / ctx_size * 100),
        )

    def _build_gen_kwargs(
        self, generation_params: dict | None, llm_client=None,
    ) -> dict:
        """LLM 生成パラメータを組み立てる (stream は呼び出し側で指定)。

        ``id_slot`` はチャットスロットに固定する。``-1`` (llama-server の
        LCP / LRU 自動割当) だとタスクループがチャットスロットの接頭辞を追い出す
        か分類器スロットに乗るため (2026-09-11)。
        """
        gen_kwargs: dict = {
            "max_tokens": self._execute_max_tokens,
            "id_slot": getattr(llm_client, "chat_slot", -1),
        }
        if generation_params:
            for k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "repetition_penalty"):
                if k in generation_params:
                    gen_kwargs[k] = generation_params[k]
        return gen_kwargs

    @staticmethod
    def _normalize_loop_tool_args(
        tool_name: str, tool_args: dict, original_query: str,
        conversation: list[dict] | None = None,
        task_description: str = "",
        *,
        single_task: bool = True,
        write_destinations: tuple[str, ...] = (),
    ) -> dict:
        """`write_file` / `read_file` / `search_code` の args を正規化する。それ以外は素通し。

        ``single_task`` が偽 (計画が 2 タスク以上) なら、場所の無い ``search_code`` は
        タスク文が名指すフォルダを継ぎ、依頼文へは探す場所が 1 つだけのときに限って落ちる
        (2 つ以上なら別のタスクの対象でありうる)。``write_destinations`` (このターンの書込み先)
        のフォルダは、書込みの節の前置きの名指し (「In <出力のフォルダ>, write …」) なら継がない。
        """
        if tool_name == "write_file":
            args = normalize_write_file_args(tool_args)
            fp = args.get("file_path", "")
            if fp:
                args["file_path"] = _TaskExecutionMixin._resolve_write_path(
                    fp, original_query, task_description,
                )
            return args
        if tool_name == "read_file":
            args = normalize_read_file_args(tool_args, original_query)
            fp = args.get("file_path", "")
            if fp:
                args["file_path"] = _TaskExecutionMixin._resolve_read_path(
                    fp, original_query, conversation,
                )
            return args
        write_only = (
            write_scope_only_folders(original_query, write_destinations)
            if tool_name == "search_code" else frozenset()
        )
        if write_only and _path_key(str(tool_args.get("directory") or ".")) in write_only:
            # 依頼が出力の場所としてだけ名指したフォルダは探さない — 場所の無い検索として扱う
            # (計画のタスク文の「under <work>」をモデルが写した、run23 O4、f_03 §4.2.1)
            tool_args = {k: v for k, v in tool_args.items() if k != "directory"}
        if tool_name != "search_code":
            return tool_args
        from backend.free.agent.meta_cognitive_tools import _named_directories

        destination_dirs = {
            _path_key(str(Path(dest).parent)) for dest in write_destinations if dest
        }

        def candidates(text: str) -> list[str]:
            # 書込みの節の前置きで名指した書込み先のフォルダは探す場所ではない — そこには
            # 前の報告がありうる (#887 の残件 LOW-1、f_03 §4.2.1)。依頼が出力の場所として
            # だけ名指したフォルダは、タスク文が検索の節で名指しても継がない (run23 O4)
            searched = _named_directories(text, skip_write_scopes=True)
            return [
                d for d in _named_directories(text)
                if (d in searched or _path_key(d) not in destination_dirs)
                and _path_key(d) not in write_only
            ]

        directory = str(tool_args.get("directory") or ".").strip()
        if directory not in (".", "./", ".\\") and not Path(directory).is_absolute():
            # 相対の場所は依頼が名指したフォルダの下に在るときだけそこへ直す。無ければそのまま返し、
            # 実行の前に場所が無いとして撃たない — CWD (インストール根) を基準にしない (run23 反証
            # 2 周目 MED-1、f_03 §4.2.1)
            for base in [*candidates(task_description), *candidates(original_query)]:
                joined = Path(base) / directory
                if joined.is_dir():
                    return {**tool_args, "directory": str(joined)}
            return tool_args
        if directory in (".", "./", ".\\"):
            # 場所の無い検索は撃たない (CWD を舐める)。依頼が名指したフォルダを継ぐ
            # (CLAUDE.md §6 #5、f_03 §4.2.1)。名指しが無ければ実行の前に場所が無いとして断る
            in_query = candidates(original_query)
            in_task = candidates(task_description)
            # 依頼文へ落ちるのは、計画が 1 タスクか、依頼文の探す場所が 1 つだけのとき (反証 LOW-A)
            named = (in_task[0] if in_task else "") or (
                in_query[0] if in_query and (single_task or len(in_query) == 1) else ""
            )
            if named:
                return {**tool_args, "directory": named}
        return tool_args

    @staticmethod
    def _resolve_read_path(
        file_path: str, original_query: str, conversation: list[dict] | None,
    ) -> str:
        """read_file の対象パスを文脈から解決する (裸のファイル名の救済)。

        書込み側は ``_resolve_write_path`` + ``_referential_write_path`` で
        裸の名前を会話中のフルパスへ寄せているのに、**読取側にはその解決が
        無かった**。``_resolve_referenced_path`` の docstring は最初から
        「書込み/読取の対象パスを会話から解決する」と書いており、読取だけ
        配線が漏れていた。

        実インシデント (2026-08-26 ライブ監査の修正検証): 「mrg_b.txt の中身を
        mrg_a.txt の末尾に追記してください。」でプランナーが裸の名前で
        ``["Read the content of mrg_b.txt", ...]`` を出し、
        ``read_file({'file_path': 'mrg_b.txt'})`` が **File not found** で失敗した
        (ディレクトリはこのターンのクエリに無く、前のターンの会話にしかない)。
        読み取りが失敗すると ``is_tool_error`` で ``_fetched_tool_outputs`` にも
        入らないため、書込み内容の供給元が空のままになる。
        """
        resolved = resolve_read_path(file_path, original_query, conversation)
        if resolved and resolved != file_path:
            logger.info(
                "Read target resolved from conversation: %s (bare=%r)",
                resolved, file_path,
            )
            return resolved
        return file_path

    @staticmethod
    def _resolve_write_path(
        file_path: str, query: str, task_description: str = "",
    ) -> str:
        """write_file の出力先を確定する。

        - 引数名プレースホルダ (``file_path`` / ``<path>`` 等) → クエリ中の
          明示パスへ差し替える
        - 既存ディレクトリ指定 (例: C:\\...\\aa) → ``output_<UTC><ext>``
          (write_file はディレクトリをエラーにするため、書込み前にファイル名へ)
        - ディレクトリ成分の無い bare ファイル名で、クエリが出力ディレクトリを
          指定している場合 → そのディレクトリ配下へ寄せる (planner が CWD 相対の
          名前を発明したとき、ユーザー指定の場所へ揃える)
        - それでも錨の無い相対パスが残った場合 → ``local_paths.outputs_dir``
          配下へ寄せる (``anchor_relative_output_path``)。``./`` ``../`` ``~``
          始まりと絶対パスはユーザーの明示指定なので触らない。

        最後の 1 段が無いと、``compose.yaml`` のような裸名がプロセスの CWD
        (= リポジトリ直下) に着地する (2026-09-08 監査 F-05)。錨付けは
        **この 1 箇所を出口** にする (前段の分岐が増えても漏れないため)。

        錨付けの直前に、**誰も名指ししていない既存ファイルへの上書き** を
        名指しの対象へ戻す (``redirect_unnamed_overwrite``、2026-09-16 監査
        F-13)。モデルは文脈に出ているだけのファイル名 (直前のターンの一覧等)
        を書込み先に選ぶことがあり、書込みは成功するので失敗がどこにも出ない。
        """
        return anchor_relative_output_path(
            redirect_unnamed_overwrite(
                _TaskExecutionMixin._resolve_write_path_from_query(
                    file_path, query,
                ),
                task_description, query,
            ),
        )

    @staticmethod
    def _resolve_write_path_from_query(file_path: str, query: str) -> str:
        """クエリ・プレースホルダから出力先を解決する (錨付けの前段)。"""
        from backend.free.agent.tool_call_judge import _extract_file_path

        if _is_placeholder_write_path(file_path):
            # 小型 aux はパラメータ名をそのまま値として返すことがある
            # (実インシデント 2026-07-28 ライブ検証:
            # `{"tool": "write_file", "args": {"file_path": "file_path", ...}}`
            # がそのまま実行され、リポジトリ直下に `file_path` という名前の
            # ファイルが作られた)。クエリに明示パスがあればそこへ寄せる。
            qpath = _extract_file_path(query)
            if qpath:
                logger.warning(
                    "Write path placeholder %r replaced with query path %s",
                    file_path, qpath,
                )
                return qpath
            logger.warning(
                "Write path is a parameter-name placeholder (%r) and the query "
                "has no explicit path; falling through to normal resolution",
                file_path,
            )
        resolved = resolve_dir_output_path(file_path, query)
        if resolved != file_path:
            return resolved
        p = Path(file_path)
        if str(p.parent) in ("", "."):  # ディレクトリ成分の無い bare ファイル名
            # 依頼が同名のファイルを絶対パスで名指ししていれば、それがそのファイル
            # 自身。先頭の明示パスだけを見ていたため、「X\DESIGN.md の設計に従って
            # X\logmon.py を作成」の logmon.py が outputs_dir へ落ちた
            # (2026-09-21 ライブ監査 K03)。
            named = _explicit_path_named(query, p.name)
            if named is not None:
                return named
            # クエリが出力先ディレクトリを明示していればそこへ置く。先頭の明示パスが
            # 入力ファイル (「X\DESIGN.md の設計に基づいて … を X に作成」) だと
            # 以下の分岐では名指しの index.html 等が outputs_dir へ落ちた
            # (2026-09-19 ライブ監査 K05)。
            out_dir = _explicit_output_dir(query)
            if out_dir is not None:
                return str(out_dir / p.name)
            qpath = _extract_file_path(query)
            if qpath and ("\\" in qpath or "/" in qpath):
                qp = Path(qpath)
                if qp.is_dir() or not qp.suffix:
                    return str(qp / p.name)
                if p.name == qp.name:
                    # bare 名がクエリの明示パスの basename と同じなら、それは
                    # そのファイル自身 (production_stage の artifact は論理名
                    # だけを持つ。2026-09-18 実機: `...\p3\idgen.py に保存して`
                    # の idgen.py が outputs 直下へ落ちた)。
                    return qpath
                if p.name not in query:
                    # クエリが挙げているのは別のファイルで、この bare 名は
                    # planner の発明。そのまま書くとプロセスの CWD
                    # (= リポジトリ直下) にゴミが残る (実インシデント
                    # 2026-07-29 ライブ監査: 「E:\tmp\audit_r4b.md の5番目の
                    # 項目を削除して保存し直してください。」が 2 タスクに割れ、
                    # 2 番目が `document.txt` へタスク文を書き込んだ)。
                    # 少なくともユーザーが作業しているディレクトリへ寄せる。
                    logger.warning(
                        "Invented bare write path %r redirected next to the "
                        "query path %s", file_path, qpath,
                    )
                    return str(qp.parent / p.name)
        elif _is_unanchored_relative(file_path):
            # ``ledger/store.py`` のような相対サブパスも、依頼が出力フォルダを
            # 指していればその配下へ置く (2026-09-20 実機: 裸名の SPEC.md は依頼
            # フォルダへ、ledger/store.py は outputs_dir へ割れた)。入力ファイルしか
            # 指していない依頼は従来どおり outputs_dir へ。
            out_dir = _explicit_output_dir(query)
            if out_dir is not None:
                return str(out_dir / p)
            qpath = _extract_file_path(query)
            if qpath and ("\\" in qpath or "/" in qpath):
                qp = Path(qpath)
                if qp.is_dir() or not qp.suffix:
                    return str(qp / p)
                # ファイル形 (依頼文のモジュール名が付いた ``<folder>\store.py``) は
                # 裸名と同じく親を作業ディレクトリとみなす。親の末尾が相対パスの
                # ディレクトリ成分と重なるなら (``...\ledger\store.py`` と
                # ``ledger/cli.py``) 重複させない。
                base = qp.parent
                tail = p.parent.parts
                if base.parts[-len(tail):] == tail:
                    base = Path(*base.parts[:-len(tail)])
                return str(base / p)
        return file_path

    async def _emit_loop_tool_running(
        self,
        on_step,
        prefix: str,
        tool_name: str,
        tool_args: dict,
    ) -> None:
        """ツール実行開始 (`status=running`) のコールバック emit。"""
        if on_step is None:
            return
        args_summary = summarize_tool_args(tool_name, tool_args)
        await call_callback(on_step, {
            "type": "tool_call",
            "detail": f"{prefix} {tool_name}({args_summary})",
            "status": "running",
        })

    async def _emit_loop_tool_result(
        self,
        on_step,
        prefix: str,
        tool_name: str,
        tool_result_text: str,
    ) -> None:
        """ツール実行結果 (`status=done|failed`) のコールバック emit。"""
        if on_step is None:
            return
        # deliberative と同じ規則: run_command の非ゼロ終了 (``[exit code: N]``)
        # も失敗として表示する (Error: プレフィックスだけでは ✓ になっていた)。
        succeeded = tool_result_succeeded(tool_name, tool_result_text)
        await call_callback(on_step, {
            "type": "tool_call",
            "detail": f"{prefix} {tool_name}: {tool_result_text[:100]}",
            "status": "done" if succeeded else "failed",
        })

    async def _execute_loop_tool_call(
        self,
        tool_call: dict,
        text: str,  # noqa: ARG002
        task: TaskItem,
        original_query: str,
        llm_client,
        tools_registry,
        on_step,
        prefix: str,
        state: AgentState,
        tool_calls: list[dict],
        step_results: list[StepResult],
        messages: list[dict],
        consecutive_errors: int,
        max_consecutive_errors: int,
        loop: int,
        generation_params: dict | None,  # noqa: ARG002
    ) -> tuple[str, list[dict]] | _AnswerFromRepeat | None:
        """ツールループ内の1回のツール実行

        ループ終了条件に達した場合は結果タプルを返す。繰り返した取得の結果から答える
        ときは ``_AnswerFromRepeat`` を返す (生成は呼出側)。
        ループ続行の場合は None を返し、messages を更新する。
        """
        tool_name = tool_call.get("tool", "")
        tool_args = self._normalize_loop_tool_args(
            tool_name, tool_call.get("args", {}), original_query,
            getattr(self, "_conversation", None),
            getattr(task, "description", "") or "",
            single_task=len(getattr(self, "_plan_tasks", None) or []) <= 1,
            write_destinations=self._plan_write_destinations(original_query),
        )

        # 取得専任タスク (collapse 済み) では write_file を実行しない。出力は後続の
        # write タスクが取得データを決定論的に書く。小型モデルが取得タスクの途中で
        # 余計な write_file を出してプレースホルダ/重複ファイルを生む退行を防ぐ。
        if getattr(task, "fetch_only", False) and tool_name == "write_file":
            logger.info(
                "fetch-only task: suppressing write_file to %s (delegated to "
                "write task)", tool_args.get("file_path", ""),
            )
            step_results.append(StepResult(
                tool_name="write_file",
                output=(
                    "Skipped: this step only fetches data; the fetched data is "
                    "written to the file by a later step. Do not write files here."
                ),
                iteration=loop,
            ))
            return None

        if tool_name == "write_file" and self._write_left_to_later_task(
            task, tool_args.get("file_path", ""), original_query,
        ):
            logger.info(
                "Suppressing write_file to %s: a later task in the plan writes "
                "this file", tool_args.get("file_path", ""),
            )
            step_results.append(StepResult(
                tool_name="write_file",
                output=(
                    "Skipped: a later step writes this file. Give the result of "
                    "this step as plain text; do not write files here."
                ),
                iteration=loop,
            ))
            return None

        tool_call_entry = {"tool": tool_name, "args": tool_args, "success": False}
        tool_calls.append(tool_call_entry)

        # write_file で content が不足 → 別途コンテンツ生成
        if tool_name == "write_file" and not tool_args.get("content"):
            gen_result = await self._generate_write_content(
                tool_name, tool_args, original_query, task,
                llm_client, on_step, prefix,
                tool_calls, step_results, messages,
                consecutive_errors, max_consecutive_errors, loop,
            )
            if gen_result is not None:
                return gen_result

        await self._emit_loop_tool_running(on_step, prefix, tool_name, tool_args)

        state.pending_tool = tool_name
        state.pending_args = tool_args

        cached = self._turn_fetched(tool_name, tool_args)
        failed_before = None if cached is not None else self._turn_fetch_error(
            tool_name, tool_args,
        )
        if cached is not None:
            logger.info(
                "%s served from this turn's retrieval (not fetched again): %s",
                tool_name, fetch_cache_key(tool_name, tool_args),
            )
            tool_result_text = cached
            # 取り直さなかった実行は学習の正例に数えない (不変則 #15、f_03 §4.2.1)
            tool_call_entry["cached"] = True
        elif failed_before is not None:
            logger.info(
                "%s not fetched again: the same URL failed earlier in this turn: %s",
                tool_name, fetch_cache_key(tool_name, tool_args),
            )
            tool_result_text = failed_before
            tool_call_entry["cached"] = True
        elif search_without_location(tool_name, tool_args):
            logger.info("search_code not run: no folder to search (not the CWD): %s", tool_args)
            tool_result_text = SEARCH_NO_LOCATION_ERROR
        else:
            tool_result_text = await self._execute_tool(
                tool_name, tool_args, tools_registry, state,
            )
            self._remember_fetch_error(tool_name, tool_args, tool_result_text)
        tool_call_entry["success"] = tool_result_succeeded(
            tool_name, tool_result_text,
        )
        if is_tool_error(tool_result_text):
            tool_call_entry["error"] = True
            tool_call_entry["error_kind"] = tool_error_kind(tool_result_text)

        state.pending_tool = None
        state.pending_args = {}

        await self._emit_loop_tool_result(on_step, prefix, tool_name, tool_result_text)

        # 連続エラー検出 → ループ打ち切り
        if is_tool_error(tool_result_text):
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                logger.warning(
                    "Stopping loop: %d consecutive errors in task '%s'",
                    consecutive_errors, task.description[:50],
                )
                return tool_result_text, tool_calls
        else:
            consecutive_errors = 0

        # 同じ呼出しが同じ結果を返した: 1 度目は注記して続け、2 度目でループを終える。
        # 生の 0 件の文字列は答えにしない (f_03 §4.2.1、run21 Z1 / C1・反証 HIGH-1 / run22 B1r)
        repeated_in_task = self._repeated_in_task(tool_name, tool_args, tool_result_text)
        repeats = self._repeated_result(tool_name, tool_args, tool_result_text)
        no_hit = tool_result_lacks_information(tool_name, tool_result_text)
        write_path = self._requested_write_path(task) if repeats else None
        earlier_primary = (getattr(self, "_primary_results", None) or [])[
            :getattr(self, "_primary_at_task_start", 0)
        ]
        call_key = retrieval_call_key(tool_name, tool_args)
        if (
            repeated_in_task and not no_hit and write_path
            and any(
                key == call_key and same_retrieval_result(tool_name, result, tool_result_text)
                for key, result in earlier_primary
            )
        ):
            # 書込みのタスクが、前のタスクの主の取得と同じ呼出しをこのタスクの中で繰り返して
            # 同じ該当を得た: 繰り返しがその素材を原本と示すので、取り直しを続けずに合流点で
            # 書く (run22 B1r、反証 HIGH-1)。ループの付随の読込みは出所に数えない (z1b)、
            # ターンの記憶の上の 1 度目は数えない (fool1、反証 HIGH-A / MED-A)
            logger.info(
                "Write task repeated an earlier retrieval (%s) with the same result; "
                "writing from that material: %s", tool_name, task.description[:50],
            )
            text, write_entries = await self._execute_write_fast(
                task, original_query, write_path, llm_client, tools_registry,
                on_step=on_step, prefix=prefix,
            )
            tool_calls.extend(write_entries)
            return text, tool_calls
        if repeats >= 2:
            self._stopped_by_repeat = True
            logger.info(
                "Stopping loop: repeated the same %s call with the same result "
                "(no_hit=%s) in task '%s'", tool_name, no_hit, task.description[:50],
            )
            if no_hit and write_path is None:
                # 検索だけのタスクは 0 件が答え (生の結果ではなく 1 文で返す)
                return msg("agent.repeated_no_match"), tool_calls
            if no_hit:
                task.failure_note = msg("agent.repeated_no_result")
                return (
                    f"Error: {tool_name} kept returning no results for the same call",
                    tool_calls,
                )
            if write_path is None and not getattr(self, "_primary_retrieval_missed", False):
                # 書込みでないタスクは繰り返した結果から答える (反証 MED-1)。主の取得が
                # 空振りした後のループの繰り返し (無関係なファイルの読込みでありうる、Z1) は除く
                # このタスクの取得をすべて渡す (繰り返した結果が前のタスクのものなら足す、反証 LOW-B)
                data = list((getattr(self, "_fetched_tool_outputs", None) or [])[
                    getattr(self, "_materials_at_task_start", 0):
                ])
                if tool_result_text not in data:
                    data.append(tool_result_text)
                return _AnswerFromRepeat(data)
            # 書込みのタスク / 主の取得が空振りしたタスクは、Step limit と同じく失敗で終える
            # (早く、理由を添えて。結果を素材にしない)
            task.failure_note = msg("agent.repeated_same_result")
            return (
                f"Error: {tool_name} kept returning the same result for the same call",
                tool_calls,
            )
        repeat_note = ""
        if repeats:
            repeat_note = (
                "\n(This exact call already returned no results earlier in this turn. "
                "Do not repeat it; try a different action or give your answer.)"
                if no_hit else
                "\n(This exact call already returned this same result earlier in this "
                "turn. Do not repeat it; use the result or give your answer.)"
            )
        step_results.append(StepResult(
            tool_name=tool_name,
            output=(
                "(Already retrieved earlier in this turn; not fetched again.)\n"
                if cached is not None else
                "(This URL already failed earlier in this turn; not fetched again.)\n"
                if failed_before is not None else ""
            ) + truncate_tool_result(tool_result_text) + repeat_note,
            iteration=loop,
        ))
        # データ取得結果をタスク横断アキュムレータへ (write タスクの素材に再利用)。
        # ここは書込み素材なので切り詰めない (全文を保持する)。取り直さなかった
        # 結果は既に積んである。
        if tool_name in _DATA_BEARING_TOOLS:
            self._retrieval_attempted = True
        if tool_name in _DATA_BEARING_TOOLS and not is_tool_error(tool_result_text):
            # 素材に積むのは役に立つ結果だけ (0 件の検索は積まない)。ファストパスと
            # 同じ判定 — 積むと書込みが「No matches found」を素材に TODO 行を作話した
            succeeded = tool_result_succeeded(tool_name, tool_result_text)
            if cached is None and succeeded:
                self._remember_retrieval(tool_name, tool_args, tool_result_text)
            # 取得専任タスクは取得成功時点で完了 (後続 write タスクへ委譲)。
            # ここで止めないと小型モデルが次ループで余計な write_file を出す。
            if getattr(task, "fetch_only", False):
                logger.info(
                    "fetch-only task: fetch succeeded, ending step "
                    "(write delegated): %s", tool_name,
                )
                task.retrieved = succeeded
                return tool_result_text, tool_calls

        # write_file 成功 → タスク完了
        if not is_tool_error(tool_result_text) and tool_name == "write_file":
            logger.info(
                "Stopping loop: write_file succeeded (%s)",
                tool_args.get("file_path", ""),
            )
            return tool_result_text, tool_calls

        # 再ループ。次反復の messages は step_results から ``_rebuild_loop_messages``
        # が再構築する (ここで messages に追記しても上書きされて LLM に届かない)。
        logger.info("Tool call #%d: %s", loop + 1, tool_name)
        return None

    async def _generate_write_content(
        self,
        tool_name: str,
        tool_args: dict,
        original_query: str,
        task: TaskItem,
        llm_client,
        on_step,
        prefix: str,
        tool_calls: list[dict],
        step_results: list[StepResult],
        messages: list[dict],  # noqa: ARG002
        consecutive_errors: int,
        max_consecutive_errors: int,
        loop: int,
    ) -> tuple[str, list[dict]] | None:
        """write_file の content が不足時にコンテンツを生成する

        生成失敗時は結果タプルを返す（ループ終了）。
        成功時は tool_args["content"] を更新して None を返す（ループ続行）。
        """
        file_path = tool_args.get("file_path", "")

        async def _notify_generating() -> None:
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} コンテンツ生成中 → {file_path}",
                    "status": "running",
                })

        content, rejection = await self._resolve_write_content(
            file_path=file_path,
            original_query=original_query,
            task_description=task.description,
            llm_client=llm_client,
            notify_generating=_notify_generating,
        )
        if content.startswith("(Content generation failed:"):
            logger.warning(
                "Skipping write_file due to content generation failure: %s",
                file_path,
            )
            tool_result_text = f"Error: {content}"
            consecutive_errors += 1
            step_results.append(StepResult(
                tool_name=tool_name,
                output=tool_result_text,
                iteration=loop,
            ))
            if consecutive_errors >= max_consecutive_errors:
                return tool_result_text, tool_calls
            # ループ続行のためメッセージを追加（呼び出し元の text は使えない）
            return None
        if rejection == ALREADY_WRITTEN:
            # このターンに同じ本文を書き済み。同じ本文の書込みは無害なので棄却にしない
            rejection = None
        if rejection:
            logger.warning(
                "Tool-loop write: generated content still rejected (%s) "
                "after retry, aborting: %r",
                rejection, content[:120],
            )
            tool_result_text = (
                f"Error: Content generation produced invalid output ({rejection}), "
                "not actual content"
            )
            consecutive_errors += 1
            step_results.append(StepResult(
                tool_name=tool_name,
                output=tool_result_text,
                iteration=loop,
            ))
            if consecutive_errors >= max_consecutive_errors:
                return tool_result_text, tool_calls
            return None
        tool_args["content"] = content
        logger.info(
            "Content generated for write_file: %d chars → %s",
            len(content), file_path,
        )
        return None

    def _extract_fetched_table_markdown(self) -> str:
        """タスク横断で取得したツール結果から GFM テーブルを抽出・結合する。

        ``fetch_url`` 等が返した本文中の table ブロックを集め、複数テーブル
        (日付ごと等で繰り返されるヘッダ) を 1 ヘッダ + 全データ行へ正規化した
        GFM 文字列を返す。テーブルが無ければ空文字列。
        """
        from backend.export.content_converter import ContentConverter

        outputs = getattr(self, "_fetched_tool_outputs", [])
        header: list[str] | None = None
        data_rows: list[list[str]] = []
        for out in outputs:
            for block in ContentConverter().convert(out):
                if block.type != "table" or not block.rows:
                    continue
                if header is None:
                    header = block.rows[0]
                    data_rows.extend(block.rows[1:])
                else:
                    # 繰り返しヘッダ行はスキップして本文行のみ連結
                    start = 1 if block.rows[0] == header else 0
                    data_rows.extend(block.rows[start:])
        if header is None or not data_rows:
            return ""
        ncol = len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * ncol) + " |",
        ]
        for r in data_rows:
            cells = (list(r) + [""] * ncol)[:ncol]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
        tools_registry,
        state: AgentState,
    ) -> str:
        """ツールを実行して結果テキストを返す"""
        if tools_registry is not None and tools_registry.has(tool_name):
            # ToolDefinition.modes は元々 LLM 向け説明文のフィルタにしか使われず、
            # LLM プランナーが自由選択するこのループ経路では実行時に無視されて
            # いた (chat モードの search_code による CWD 全域 os.walk が実
            # インシデント)。deliberative と同じ規則で全ツールを照合する。
            mode_error = tool_mode_error(tools_registry, tool_name, self._mode)
            if mode_error is not None:
                state.on_tool_failure(tool_name, mode_error)
                return mode_error
            try:
                result_text = await execute_tool_with_timeout(
                    tools_registry, tool_name, tool_args,
                )
                if is_tool_error(result_text):
                    state.on_tool_failure(tool_name, result_text)
                else:
                    state.on_tool_success(tool_name)
            except Exception as e:
                result_text = f"Error: {e}"
                state.on_tool_failure(tool_name, str(e))
                logger.warning("Tool execution failed: %s - %s", tool_name, e)
        else:
            result_text = f"Error: Unknown tool '{tool_name}'"
            state.on_tool_failure(tool_name, result_text)
        return result_text
