"""FastPathMixin — meta_cognitive_fast_path"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from backend.free.agent.meta_cognitive_tasks import TaskItem
from backend.free.agent.meta_cognitive_task_exec import (
    execute_tool_with_timeout,
    fetch_cache_key,
    retrieval_call_key,
    search_without_location,
    tool_mode_error,
)
from backend.free.constants import SEARCH_NO_LOCATION_ERROR
from backend.free.agent.output_format import wants_fetched_table
from backend.free.agent.file_ledger import forget_current_file
from backend.free.agent.tool_ledger import mark_last_failed
from backend.free.agent.write_gate import WRITE_DENIED_RE, request_names_file
from backend.free.agent.meta_cognitive_utils import (
    call_callback,
    extract_literal_write_content,
    generated_content_rejection,
    is_tool_error,
    previous_answer_write_content,
    rescue_quoted_write_literal,
    strip_generator_scaffold_block,
    strip_markdown_wrapper,
    strip_output_lead_in,
    strip_prompt_scaffold_lines,
    strip_task_log_scaffold,
    summarize_tool_args,
    tool_error_kind,
    text_looks_like_code,
    tool_result_succeeded,
)

from backend.free.agent.meta_cognitive_defs import (
    ALREADY_WRITTEN,
    _DATA_BEARING_TOOLS,
    resolve_read_path,
)
from backend.free.agent.meta_cognitive_content import (
    ExistingContentRefused,
    InsufficientSourceRefused,
)
from backend.free.agent.meta_cognitive_tool_io import read_file_result_path
from backend.free.agent.tools.filesystem import append_existing_text
from backend.free.agent.meta_cognitive_content_gate import is_pure_append_request
from backend.free.agent.edit_mode_gate import is_append_verdict
from backend.io.text_file import Unreadable

from backend.free.core.response_dates import fix_weekday_claims
from backend.free.core.script_ranges import KANJI, KANJI_MARKS, KATAKANA_WORD
from backend.i18n_helper import msg
from backend.log_config import get_logger
from backend.free.core.prompt_blocks import local_today

logger = get_logger("agent.meta_cognitive")

#: 既存ファイル編集の fast path が扱う本文の上限 (文字数)。
_EDIT_MAX_CHARS = 2_000_000
#: 同 (バイト数。読込時の上限で、文字数上限と同じ値にしている)。
_EDIT_MAX_BYTES = 2_000_000

#: 既存と同一の生成を棄却した後の再生成に添える指示。同じプロンプトのまま
#: 再生成すると同じ写しが返り、長い生成を無駄に繰り返す (2026-10-02 ライブ監査 D07#3)。
_UNCHANGED_EDIT_RETRY_HINT = (
    "[再生成の指示] 直前の出力は既存ファイルの内容と同一で、依頼の変更が一つも"
    "反映されていなかった。次のユーザー指示を必ず反映した内容を出力すること"
    "(既存内容の写しは不可): {instruction}"
)


def _turn_path_key(path: str) -> str:
    """このターンの書込み記録の鍵 (絶対パスの正規形)。"""
    try:
        return os.path.normcase(os.path.abspath(path))
    except (OSError, ValueError):
        return os.path.normcase(path)


def _content_digest(content: str) -> str:
    """書いた本文の指紋 (改行の種類と末尾の改行 1 つの差は同じとみなす)。"""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").removesuffix("\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


#: 文中のファイル名らしい語 (ASCII の名前 + 拡張子)。素材の名指しの照合に使う (字句の鍵)。
_FILE_TOKEN_RE = re.compile(r"(?<![\w.\-])[\w\-]+\.[A-Za-z][A-Za-z0-9]{0,5}(?![\w\-])")
#: 内容語の連なり (漢字・カタカナ・英数字)。ひらがな (助詞・語尾) で切る。
_CONTENT_RUN_RE = re.compile(rf"[{KANJI}{KANJI_MARKS}{KATAKANA_WORD}A-Za-z0-9]+")
#: 前の発話が今の依頼に関係するとみなす、内容語の 2 文字の共有数の下限。
_RELEVANT_BIGRAMS = 2


def _content_bigrams(text: str) -> set[str]:
    """パスとファイル名を除いた内容語の 2 文字の組 (英字は小文字)。"""
    from backend.free.agent.tool_judge_args import without_drive_paths

    stripped = _FILE_TOKEN_RE.sub(" ", without_drive_paths(text or "", " "))
    out: set[str] = set()
    for run in _CONTENT_RUN_RE.findall(stripped):
        run = run.lower()
        out.update(run[i:i + 2] for i in range(len(run) - 1))
    return out


def relevant_earlier_turn(query: str, earlier_user_texts: list[str]) -> str:
    """今の依頼に関係する、ファイルを名指した最も新しい user の発話 (無ければ空)。

    関係の判定は内容語の 2 文字の組の共有 (:data:`_RELEVANT_BIGRAMS` 以上)。「部署ごとの
    平均給与を…に書き出して」と「staff.csv を読んで、部署ごとの平均給与を計算して」は
    部署・平均・均給・給与を共有し、「buggy.py のバグを見つけて」とは何も共有しない。
    ``earlier_user_texts`` は新しい順 (``file_ledger.user_texts``)。今の依頼文自身は飛ばす。
    """
    wanted = _content_bigrams(query)
    if not wanted:
        return ""
    for text in earlier_user_texts:
        if text.strip() == (query or "").strip() or not _FILE_TOKEN_RE.search(text):
            continue
        if len(wanted & _content_bigrams(text)) >= _RELEVANT_BIGRAMS:
            return text
    return ""


class _FastPathMixin:
    """ツールループを介さない直接実行 (ファストパス)。

    ツール名と引数が決定論で確定しているタスクを、LLM のツールループを
    回さずに実行する層。書込みでは本文の解決・検証・救出までを担う。
    """

    def _pure_append_requested(self, query: str) -> bool:
        """追記だけの依頼か (判定点 ``edit_mode`` = append)。

        ``process()`` の冒頭でターンに 1 回評価した結果 (事例の確認と記録つき) を
        読む。同じ依頼文の評価が無いとき (単体の呼出) は字句段だけで決める。
        """
        if query is not None and getattr(self, "_edit_mode_query", None) == query:
            return is_append_verdict(getattr(self, "_edit_mode_verdict", None))
        return is_pure_append_request(query)

    def _resolve_read_args(self, tool_args: dict, query: str) -> dict:
        """read_file の ``file_path`` を文脈から解決する (裸の名前の救済)。

        ``_resolve_referenced_path`` の docstring は最初から「書込み/**読取**の
        対象パスを会話から解決する」と書いているのに、配線されていたのは
        書込み側だけだった。解決できないときは元の値を返す (退行させない)。
        """
        file_path = tool_args.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            return tool_args
        resolved = resolve_read_path(
            file_path, query, getattr(self, "_conversation", None),
        )
        if resolved == file_path:
            return tool_args
        logger.info(
            "Read target resolved from context: %s (bare=%r)",
            resolved, file_path,
        )
        return {**tool_args, "file_path": resolved}

    # ── このターンの書込みの記録 ({宛先: 本文の指紋}) ──────────────────
    #
    # 2026-10-05 ライブ監査 T5: タスク 2 (救出) が fixed.py を書いた後、タスク 3 が同じ
    # 本文を生成し、既存 (= タスク 2 が書いた内容) と同じだとして edit_without_change で
    # 棄却 → 再生成 → 失敗、本文と「書き込みができませんでした」が並んだ。比べる相手は
    # ターンの開始時のファイルで、このターンの自分の書込みではない。

    def _turn_write_baseline(self, file_path: str, existing: str) -> str:
        """無変更の検査で比べる既存内容 (このターンに書いた宛先はターン開始時の内容)。"""
        starts = getattr(self, "_turn_start_contents", None) or {}
        return starts.get(_turn_path_key(file_path), existing)

    def _written_this_turn(self, file_path: str, content: str) -> bool:
        """このターンに同じ宛先へ同じ本文を既に書いたか。"""
        writes = getattr(self, "_turn_writes", None) or {}
        return writes.get(_turn_path_key(file_path)) == _content_digest(content)

    async def _note_turn_write_start(self, file_path: str) -> None:
        """宛先のターン開始時の内容を、このターンの最初の書込みの前に 1 回だけ控える。"""
        starts = getattr(self, "_turn_start_contents", None)
        if starts is None:
            starts = self._turn_start_contents = {}
        key = _turn_path_key(file_path)
        if key in starts:
            return
        loaded = await self._load_existing_for_edit(file_path)
        if isinstance(loaded, Unreadable):
            return
        starts[key] = loaded or ""

    def _note_turn_write_done(self, file_path: str, content: str) -> None:
        """書けた本文の指紋をこのターンの記録へ積む。"""
        writes = getattr(self, "_turn_writes", None)
        if writes is None:
            writes = self._turn_writes = {}
        writes[_turn_path_key(file_path)] = _content_digest(content)

    def _already_saved_entry(self, file_path: str, content: str) -> tuple[str, dict]:
        """同じ本文をこのターンに書き済みの宛先の結果 (書かずに done)。"""
        logger.info(
            "Write skipped: this turn already wrote the same content to %s", file_path,
        )
        return msg("agent.write_already_saved", path=file_path), {
            "tool": "write_file",
            "args": {"file_path": file_path, "content": content},
            "success": True,
            "already_written": True,
        }

    def _has_source_material(self, original_query: str, destination: str = "") -> bool:
        """このターンの取得に、依頼に関係する素材があるか (``no_source_data`` の門)。

        読込み (``read_file``) の結果を素材に数えるのは、対象を利用者が **この依頼のために**
        名指したときだけ:

        1. 名前かフルパスが今の依頼文にある
        2. 「そのファイル」型の暗黙参照が指す直近のファイル
        3. 今の依頼文が宛先 (``destination``) のほかにファイルを名指さず、前の user の
           発話のうち **今の依頼に関係する** (:func:`relevant_earlier_turn`) 最も新しいものが
           名指したファイル — T6「部署ごとの平均給与を …\\summary.md に表で書き出して」の
           素材は T1「staff.csv を読んで、部署ごとの平均給与を計算して」の staff.csv で、
           T4「buggy.py のバグを…」の buggy.py ではない

        依頼文が挙げたフォルダ (書込みの宛先の親) の中にあるだけでは数えない。2026-10-05
        ライブ監査 T6 では計画が無関係な buggy.py を読み、謝罪文を summary.md へ書いて成功と
        報告した。他の取得 (URL / 検索) は従来どおり素材に数える。
        """
        from backend.free.agent.file_ledger import user_texts
        from backend.free.agent.tool_judge_args import _extract_file_path

        query = original_query or ""
        # 「そのファイルを要約して…」型の暗黙参照が指すファイル (台帳の直近) も依頼の対象
        referred = _extract_file_path(query)
        referred_key = _turn_path_key(referred) if referred else ""
        dest_name = os.path.basename(destination or "").lower()
        names_other_source = any(
            token.lower() != dest_name for token in _FILE_TOKEN_RE.findall(query)
        )
        earlier_turn = "" if names_other_source else relevant_earlier_turn(
            query, user_texts(getattr(self, "_conversation", None)),
        )
        for output in getattr(self, "_fetched_tool_outputs", None) or ():
            path = read_file_result_path(output)
            if path is None or request_names_file(path, query):
                return True
            if referred_key and _turn_path_key(path) == referred_key:
                return True
            if earlier_turn and request_names_file(path, earlier_turn):
                return True
        return False

    async def _execute_tool_fast(
        self,
        tool_name: str,
        tool_args: dict,
        task: TaskItem,
        tools_registry,
        on_step=None,
        prefix: str = "",
        original_query: str = "",
    ) -> tuple[str, list[dict]]:
        """read_file / run_command / search_code のファストパス実行"""
        # 裸のファイル名を会話中のフルパスへ寄せる。書込み側は
        # ``_resolve_write_path`` / ``_referential_write_path`` で解決済みなのに
        # **読取側だけ解決が無く**、プランナーが裸の名前を出すと
        # ``read_file`` がプロセスの CWD を見て File not found になる。
        #
        # 実インシデント (2026-08-26 の修正検証): 「rd_b.txt の中身を rd_a.txt の
        # 末尾に追記してください。」でプランが
        # ``["Read the content of rd_b.txt", ...]`` と裸の名前になり、
        # ``read_file({'file_path': 'rd_b.txt'})`` が失敗した。ディレクトリは
        # このターンのクエリに無く、前のターンの会話にしかない。
        # 読み取りが失敗すると ``is_tool_error`` で ``_fetched_tool_outputs`` に
        # 入らないため、後続の書込みは供給元を失って内容を捏造する。
        #
        # ⚠ 最初この解決を ``_normalize_loop_tool_args`` (ツールループ側) に
        # 入れたが、read_file は **このファストパス** を通るため一度も効かな
        # かった。実機のログ (``Tool fast path: read_file``) で判明。
        if tool_name == "read_file":
            tool_args = self._resolve_read_args(tool_args, original_query)
        logger.info("Tool fast path: %s(%s)", tool_name, tool_args)

        # ToolCallJudge の rule/aux 判定結果をそのまま実行する経路なので
        # ToolDefinition.modes を経由しない。deliberative / _execute_tool と
        # 同じ規則で全ツールを mode ゲートに通す (create 専用の run_command /
        # search_code 等が chat のタスクから走らないように)。
        mode_error = tool_mode_error(tools_registry, tool_name, self._mode)
        if mode_error is not None:
            return mode_error, [{
                "tool": tool_name,
                "args": tool_args,
                "success": False,
            }]

        if on_step:
            args_summary = summarize_tool_args(tool_name, tool_args)
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} {tool_name}({args_summary})",
                "status": "running",
            })

        # 同じターンで取得済みの URL は取り直さない (docs/f_03 §4.2.1)
        cached = self._turn_fetched(tool_name, tool_args)
        failed_before = None if cached is not None else self._turn_fetch_error(
            tool_name, tool_args,
        )
        if cached is not None:
            logger.info(
                "%s served from this turn's retrieval (not fetched again): %s",
                tool_name, fetch_cache_key(tool_name, tool_args),
            )
            result_text, is_success = cached, True
        elif failed_before is not None:
            # 同じ URL がこのターンで既にエラーを返した (timeout 等だけが違う取り直し、f_03 §4.2.1)
            logger.info(
                "%s not fetched again: the same URL failed earlier in this turn: %s",
                tool_name, fetch_cache_key(tool_name, tool_args),
            )
            result_text, is_success = failed_before, False
        elif search_without_location(tool_name, tool_args):
            # 場所の無い検索はインストール根を舐めるので撃たない (run23 反証 HIGH-1、f_03 §4.2.1)
            logger.info("search_code not run: no folder to search (not the CWD): %s", tool_args)
            result_text, is_success = SEARCH_NO_LOCATION_ERROR, False
        else:
            try:
                result_text = await execute_tool_with_timeout(
                    tools_registry, tool_name, tool_args,
                )
                is_success = tool_result_succeeded(tool_name, result_text)
            except Exception as e:
                result_text = f"Error: {e}"
                is_success = False
                logger.error("Tool fast path failed: %s - %s", tool_name, e)
            self._remember_fetch_error(tool_name, tool_args, result_text)

        # データ取得結果をタスク横断アキュムレータへ (後続 write タスクの素材に再利用)。
        # ファストパス経由の fetch_url 等もここで蓄積する (ツールループ経路と対称)。
        # 取り直さなかった結果は既に積んである。
        task.retrieved = tool_name in _DATA_BEARING_TOOLS and is_success
        if tool_name in _DATA_BEARING_TOOLS:
            self._retrieval_attempted = True
        if task.retrieved and cached is None:
            self._remember_retrieval(tool_name, tool_args, result_text)
        if task.retrieved:
            self._primary_results.append(
                (retrieval_call_key(tool_name, tool_args), result_text),
            )

        tool_entry = {
            "tool": tool_name,
            "args": tool_args,
            "success": is_success,
        }
        if cached is not None or failed_before is not None:
            # 取り直さなかった実行は学習の正例に数えない (不変則 #15、f_03 §4.2.1)
            tool_entry["cached"] = True
        if tool_name in _DATA_BEARING_TOOLS:
            # タスクの主の取得 (判定したツール)。失敗したタスクの素材を残すかを決める
            tool_entry["fast_path"] = True
        if is_tool_error(result_text):
            tool_entry["error"] = True
            tool_entry["error_kind"] = tool_error_kind(result_text)
        # 結果はループで同じ呼出しを繰り返したときの打ち切りに使う (f_03 §4.2.1)
        self._repeated_result(tool_name, tool_args, result_text)

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} {tool_name}: {result_text[:100]}",
                "status": "done" if is_success else "failed",
            })

        # 結果本文は出さない (read_file ならファイルの中身そのもの、f_03 §3.2)
        logger.info(
            "Tool fast path completed: %s, result_length=%d, success=%s",
            tool_name, len(result_text), is_success,
        )
        return result_text, [tool_entry]

    async def _execute_write_fast(
        self,
        task: TaskItem,
        original_query: str,
        file_path: str,
        llm_client,
        tools_registry,
        on_step=None,
        prefix: str = "",
    ) -> tuple[str, list[dict]]:
        """書き込みタスクのファストパス実行"""
        # 出力先を確定 (ディレクトリ→output ファイル / bare 名→クエリ指定ディレクトリ配下)。
        # planner/judge が発明した CWD 相対の bare 名をユーザー指定の場所へ寄せる。
        # 直近ファイルの宛先 (判定点 recent_file_reference = write) は依頼が名指した
        # ファイルとして扱い、名指しの無い既存ファイルの振り替えを掛けない
        # (2026-09-28 レビュー H2: 「report.md にも」の名前へ振り替わった)。
        recent_target = getattr(self, "_recent_file_target", "")
        file_path = self._resolve_write_path(
            file_path, original_query,
            recent_target if recent_target and file_path == recent_target else "",
        )
        logger.info("Write fast path: %s → %s", task.description[:60], file_path)

        async def _notify_generating() -> None:
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成中 → {file_path}",
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
            logger.warning("Write fast path: content generation failed for %s", file_path)
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成失敗",
                    "status": "failed",
                })
            return f"Error: {content}", []

        if rejection == ALREADY_WRITTEN:
            text, entry = self._already_saved_entry(file_path, content)
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: {text[:100]}",
                    "status": "done",
                })
            return text, [entry]

        if rejection == "insufficient_source":
            # 素材に書く内容が無いと生成側が答えた。何が足りないかを利用者へ (書かない)
            missing = (getattr(self, "_insufficient_source_missing", None) or {}).get(
                _turn_path_key(file_path), "",
            )
            task.failure_note = msg(
                "agent.write_insufficient_source", path=file_path,
                missing=missing or msg("agent.write_rejection.insufficient_source"),
            )

        if rejection:
            logger.warning(
                "Write fast path: generated content still rejected (%s) "
                "after retry, aborting: %r",
                rejection, content[:120],
            )
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成失敗（{rejection}）",
                    "status": "failed",
                })
            return (
                f"Error: Content generation produced invalid output ({rejection}), "
                "not actual content",
                [],
            )

        return await self._write_file(
            file_path, content, tools_registry, on_step, prefix,
        )

    async def _resolve_write_content(
        self,
        *,
        file_path: str,
        original_query: str,
        task_description: str,
        llm_client,
        notify_generating=None,
    ) -> tuple[str, str | None]:
        """書込み本文を「決定論 → 生成 → 救済」の順で解決する単一の合流点。

        fast path と tool-loop が本文解決を各々持っており、決定論経路
        (ユーザー literal / 直前応答) が fast path にしか無かったため、
        同じ依頼でも経路が変わると実況文が書き込まれた。両経路をここへ
        集約し、解決順序を 1 箇所で決める。

        出力先の既存ファイル (docs/f_11 §5): 追記だけの依頼
        (判定点 ``edit_mode`` = append。本文を変える指示が同居すれば書き直し) なら
        どの段で得た本文も ``既存内容 + 区切り + 本文`` へ決定論で連結する
        (生成段はモデルに追記分だけを出させる)。既存ファイルが読めなければ、
        追記と生成 (既存内容に依存する本文) は書かずに ``existing_unreadable``
        で断る。書き直しで既存内容が予算に収まらなければ ``existing_too_large``。

        Returns:
            ``(content, rejection)``。``rejection`` が None なら書込み可。
        """
        loaded = await self._load_existing_for_edit(file_path)
        unreadable = isinstance(loaded, Unreadable)
        existing = loaded if isinstance(loaded, str) else ""
        append = loaded is not None and self._pure_append_requested(original_query)
        if unreadable and append:
            logger.warning(
                "Write: refusing to append to unreadable file (%s): %s",
                loaded.reason, file_path,
            )
            return "", "existing_unreadable"

        # 1. 取得済みの実テーブル (転記させるとハルシネーション/行脱落が出る)
        if wants_fetched_table(file_path):
            fetched_table = self._extract_fetched_table_markdown()
            if fetched_table:
                logger.info(
                    "Write content from fetched table (deterministic): "
                    "%d chars -> %s", len(fetched_table), file_path,
                )
                return (
                    append_existing_text(existing, fetched_table) if append else fetched_table
                ), None

        # 2. ユーザーが引用符で本文そのものを指定している (高精度マッチ)
        literal = extract_literal_write_content(original_query, file_path)
        if literal:
            logger.info(
                "Write content from user literal (deterministic): "
                "%d chars -> %s", len(literal), file_path,
            )
            return (append_existing_text(existing, literal) if append else literal), None

        # 3. 「この案内文を保存して」型: 書くべき本文は直前の応答そのもの
        previous = previous_answer_write_content(
            original_query, getattr(self, "_conversation", None), file_path,
        )
        if previous:
            logger.info(
                "Write content from previous answer (deterministic): "
                "%d chars -> %s", len(previous), file_path,
            )
            return (append_existing_text(existing, previous) if append else previous), None

        # 既存内容を知らずに生成すると、既存ファイルを別物で上書きする
        if unreadable:
            logger.warning(
                "Write: refusing to generate over unreadable file (%s): %s",
                loaded.reason, file_path,
            )
            return "", "existing_unreadable"

        # 前のタスクの取得が素材を 1 件も返さなかった (0 件・エラー) なら、本文を
        # 生成しない。素材の無い生成は依頼文から本文を作話する — 「TODO を探して
        # 該当行を保存して」で、存在しない TODO 行がファイルに書かれ done で
        # 終わった (2026-10-04 ライブ監査 run20、不変則 #15、f_03 §4.2.1)。
        # 上の決定論の素材 (引用・直前の応答) は依頼と会話に在るので先に通す。
        if (
            getattr(self, "_earlier_retrieval_attempted", False)
            and not self._has_source_material(original_query, file_path)
        ):
            logger.warning(
                "Write: refusing to generate content because this turn's "
                "retrieval returned no data: %s", file_path,
            )
            return "", "no_source_data"

        if notify_generating:
            await notify_generating()

        # 4. 生成 (棄却されたら 1 度だけ再生成)。追記では生成物は追記分だけ。
        # 無変更の検査はターン開始時の内容と比べる (このターンの自分の書込みと比べない)。
        compare_with = "" if append else self._turn_write_baseline(file_path, existing)
        async def _generate(retry_hint: str = "") -> str:
            return await self._generate_content(
                original_query, task_description, llm_client, file_path=file_path,
                existing=existing, append=append, retry_hint=retry_hint,
            )

        try:
            content = await _generate()
            content, rejection = self._validate_generated_content(
                content, file_path, original_query, existing_content=compare_with,
            )
            if not append and self._written_this_turn(file_path, content):
                # このターンに同じ本文を書き済み — 再生成も書込みもしない
                return content, ALREADY_WRITTEN
            if rejection and not content.startswith("(Content generation failed:"):
                logger.warning(
                    "Write: generated content rejected (%s), retrying content "
                    "generation: %r", rejection, content[:120],
                )
                content = await _generate(
                    _UNCHANGED_EDIT_RETRY_HINT.format(instruction=original_query)
                    if rejection == "edit_without_change" else "",
                )
                content, rejection = self._validate_generated_content(
                    content, file_path, original_query, existing_content=compare_with,
                )
        except InsufficientSourceRefused as refusal:
            # 素材に書く内容が無いと生成側が答えた (何が足りないかを控えて書かない)
            missing = getattr(self, "_insufficient_source_missing", None)
            if missing is None:
                missing = self._insufficient_source_missing = {}
            missing[_turn_path_key(file_path)] = refusal.missing
            return "", refusal.code
        except ExistingContentRefused as refusal:
            return "", refusal.code

        # 5. 救済: 生成が失敗と確定した後に限り、緩い引用抽出で本文を拾う。
        #    誤爆リスクは「既に生成が棄却されている」状態に閉じ込めてある。
        if rejection:
            rescued = rescue_quoted_write_literal(original_query, file_path)
            if rescued:
                logger.info(
                    "Write content rescued from user quote after %s: "
                    "%d chars -> %s", rejection, len(rescued), file_path,
                )
                return (append_existing_text(existing, rescued) if append else rescued), None

        if rejection is None:
            # 月日に添えた曜日を暦で照合する (ユーザーが述べた月日だけ。f_08 §6.3)。
            # 追記では連結前に掛け、既存内容には手を入れない。
            grounded = "\n".join([original_query, *(
                str(m.get("content") or "")
                for m in getattr(self, "_conversation", None) or []
                if isinstance(m, dict) and m.get("role") == "user"
            )])
            content = fix_weekday_claims(
                content, today=local_today(), grounded=grounded,
            )
            if append:
                content = append_existing_text(existing, content)
        return content, rejection

    @staticmethod
    def _validate_generated_content(
        content: str, file_path: str, instruction: str = "",
        *, existing_content: str,
    ) -> tuple[str, str | None]:
        """生成コンテンツを scaffold 除去してから書込み適性を検証する。

        「タスクログ + 本文」の連結は本文だけに救済し、本文が残らない
        エコー (task_log_echo / prompt_echo / instruction_echo / path_only /
        csv_without_rows) は棄却理由を返して呼出側で再生成・中断させる。

        ``instruction`` (元のユーザー依頼文) を渡すと、依頼文の逐語コピーを
        本文として書き込む退化も棄却できる。

        上書き対象が既に存在する場合は既存内容も渡し、編集依頼なのに 1 文字も
        変わっていない生成 (edit_without_change) を棄却する。書込み自体は
        成功してしまうため、これを見ないと「完了しました」と誤報告される。
        ``existing_content`` は呼び出し側が :meth:`_load_existing_for_edit` で
        (イベントループの外で) 読んだもの。新規作成・追記分の検証では空文字。
        """
        if content.startswith("(Content generation failed:"):
            return content, "generation_failed"
        descaffolded = strip_prompt_scaffold_lines(content)
        if descaffolded != content:
            logger.info(
                "Write: stripped prompt scaffold labels from generated content "
                "(%d -> %d chars)", len(content), len(descaffolded),
            )
            content = descaffolded
        stripped = strip_task_log_scaffold(content)
        if stripped and stripped != content:
            logger.info(
                "Write fast path: stripped task-log scaffold from generated "
                "content (%d -> %d chars)", len(content), len(stripped),
            )
            content = stripped
        descaffolded_block = strip_generator_scaffold_block(content, file_path)
        if descaffolded_block != content:
            logger.info(
                "Write: stripped generator-directed scaffold block "
                "(%d -> %d chars)", len(content), len(descaffolded_block),
            )
            content = descaffolded_block
        without_lead_in = strip_output_lead_in(content, file_path)
        if without_lead_in != content:
            logger.info(
                "Write: stripped answer lead-in naming the output path "
                "(%d -> %d chars)", len(content), len(without_lead_in),
            )
            content = without_lead_in
        return content, generated_content_rejection(
            content, file_path, instruction,
            existing_content=existing_content,
        )

    async def _write_file(
        self,
        file_path: str,
        content: str,
        tools_registry,
        on_step,
        prefix: str,
    ) -> tuple[str, list[dict]]:
        """write_file を実行して結果を返す"""
        tool_args = {"file_path": file_path, "content": content}
        mode_error = tool_mode_error(tools_registry, "write_file", self._mode)
        if mode_error is not None:
            return mode_error, [{
                "tool": "write_file", "args": tool_args, "success": False,
            }]
        await self._note_turn_write_start(file_path)
        try:
            result_text = await execute_tool_with_timeout(
                tools_registry, "write_file", tool_args,
            )
            is_success = not is_tool_error(result_text)
            if WRITE_DENIED_RE.match(result_text) and content not in result_text:
                # 書込みゲートが断った (docs/f_03 §4.y)。生成した本文は捨てずに
                # 結果へ添え、最終応答が断りの理由と一緒に見せる (上書きの門の
                # 断りはレジストリが添え済みなので 2 度添えない)。
                result_text = f"{result_text}\n\n{content}"
            if is_success:
                verify_error = self._verify_written_file(file_path, content)
                if verify_error:
                    result_text = f"Error: {verify_error}"
                    is_success = False
                    # ToolsRegistry.execute は戻り値だけで台帳へ「成功」を記録済み。
                    # 実ファイル突合で失敗と分かった時点で台帳側も失敗へ揃え、
                    # 壊れたファイルを「直近に触れたファイル」から外す。
                    mark_last_failed("write_file")
                    forget_current_file(file_path)
                    logger.error(
                        "write_file post-verification failed: %s (%s)",
                        file_path, verify_error,
                    )
                else:
                    self._note_turn_write_done(file_path, content)
        except Exception as e:
            result_text = f"Error: {e}"
            is_success = False
            logger.error("write_file failed: %s", e)

        tool_entry = {
            "tool": "write_file",
            "args": tool_args,
            "success": is_success,
        }

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} write_file: {result_text[:100]}",
                "status": "done" if is_success else "failed",
            })

        logger.info("write_file completed: %s → %s", file_path, result_text[:80])
        return result_text, [tool_entry]

    @staticmethod
    def _verify_written_file(file_path: str, content: str) -> str | None:
        """書込み後にディスク上の実ファイルを読み戻して内容を突合する。

        「Written N bytes」という成功申告と実ファイルの乖離 (書込み経路の
        取り違え・変換事故) を success にしないための最終ガード。リッチ文書
        (xlsx/docx 等) は export 変換で内容が変わるため対象外。書き手は既存
        ファイルの符号化・改行・末尾改行を保つ (docs/f_11 §5.5) ので、読み戻しは
        その符号化のまま行い (``read_text_for_edit``)、改行 (CRLF/LF) と末尾改行
        1 つの差は正規化して比較する。検証自体の失敗 (読み戻し不可等) は書込み
        失敗と区別できないためエラーにせず None (成功維持) を返す。
        """
        from backend.free.agent.tools.builtin import _EXPORT_DOC_EXTS
        from backend.io.text_file import TextFile, read_text_for_edit

        try:
            p = Path(file_path)
            if p.suffix.lower() in _EXPORT_DOC_EXTS:
                return None
            if len(content) > _EDIT_MAX_CHARS:
                return None
            read = read_text_for_edit(p, max_bytes=_EDIT_MAX_BYTES)
        except Exception:
            return None
        if not isinstance(read, TextFile):
            return None
        on_disk = read.text
        expected = content.replace("\r\n", "\n").replace("\r", "\n")
        if on_disk.removesuffix("\n") != expected.removesuffix("\n"):
            return (
                f"post-write verification failed: on-disk content of "
                f"'{file_path}' does not match the generated content "
                f"({len(on_disk)} vs {len(content)} chars)"
            )
        return None

    async def _recover_write_from_text(
        self,
        text: str,
        task: TaskItem,
        original_query: str,
        llm_client,
        tools_registry,
        on_step,
        prefix: str,
    ) -> dict | None:
        """LLM がツールコール JSON を出力しなかった場合の自動リカバリー"""
        from backend.free.agent.tool_judge_args import extract_write_target_path

        # 「同じファイルに追記して」型はタスク文にパスが無く、直前ターンに
        # しか無い。プラン生成側 (write fast path) は既に
        # ``_referential_write_path`` で会話から解決しているのに、リカバリー
        # 側はタスク文しか見ておらず非対称だった。その結果、書込みが 2 回とも
        # 失敗して「実行されませんでした」で終わる (実インシデント
        # 2026-08-08 ライブ監査 ターン6: ルータは local_write_intent へ
        # 正しく振ったが、ここで毎回 no file path になっていた)。
        # 裸のファイル名も同じく未確定として会話から解決する (2026-08-09)。
        # 2 ファイルが登場するタスクでは先頭は常に source (読む側)。
        # 先頭一致だと書き込み先が source に化ける
        # (extract_write_target_path の docstring 参照)。
        file_path = extract_write_target_path(task.description)
        file_path = self._referential_write_path(file_path or None) or file_path
        if not file_path:
            logger.warning(
                "Auto-recovery skipped: no file path in task or conversation: %s",
                task.description[:80],
            )
            return None

        # 出力先を確定 (ディレクトリ→output / bare→クエリ dir)。
        file_path = self._resolve_write_path(file_path, original_query)

        logger.info(
            "Auto-recovery: LLM returned plain text for write task, "
            "extracting content for %s",
            file_path,
        )

        # この経路だけの特殊性: モデルが既に成果物 (コード) を平文で吐いている
        # なら、それが書くべき本文そのもの。合流点より前に採用する。
        content = ""
        if not wants_fetched_table(file_path):
            candidate = strip_markdown_wrapper(text)
            if text_looks_like_code(candidate):
                loaded = await self._load_existing_for_edit(file_path)
                # 既存ファイルへの追記 / 読めない既存ファイルは合流点に任せる (f_11 §5)
                defer = isinstance(loaded, Unreadable) or (
                    loaded is not None and self._pure_append_requested(original_query)
                )
                if not defer:
                    validated, rejection = self._validate_generated_content(
                        candidate, file_path, original_query,
                        existing_content=self._turn_write_baseline(file_path, loaded or ""),
                    )
                    if not rejection:
                        content = validated

        # それ以外は共通の合流点へ (取得テーブル / ユーザー literal / 直前応答 /
        # 生成 + 再生成 + 救済)。この経路だけ決定論解決を持たず、実況文が本文
        # として書き込まれる穴になっていた。
        if not content:
            async def _notify_generating() -> None:
                if on_step:
                    await call_callback(on_step, {
                        "type": "tool_call",
                        "detail": f"{prefix} コンテンツ生成中（自動リカバリー） → {file_path}",
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
                logger.warning("Auto-recovery content generation failed: %s", file_path)
                return None
            if rejection and rejection != ALREADY_WRITTEN:
                logger.warning(
                    "Auto-recovery: generated content rejected (%s), aborting: %r",
                    rejection, content[:120],
                )
                return None

        if self._written_this_turn(file_path, content):
            text, entry = self._already_saved_entry(file_path, content)
            return {**entry, "result": text}

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} write_file({file_path}, {len(content)}文字) [自動リカバリー]",
                "status": "running",
            })

        tool_args = {"file_path": file_path, "content": content}
        if tool_mode_error(tools_registry, "write_file", self._mode) is not None:
            return None
        await self._note_turn_write_start(file_path)
        try:
            result_text = await execute_tool_with_timeout(
                tools_registry, "write_file", tool_args,
            )
            is_success = not is_tool_error(result_text)
            if is_success:
                self._note_turn_write_done(file_path, content)
            logger.info("Auto-recovery write_file: %s → %s", file_path, result_text[:100])

            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: {result_text[:100]}",
                    "status": "done" if is_success else "failed",
                })

            return {
                "tool": "write_file",
                "args": tool_args,
                "success": is_success,
                "result": result_text,
            }
        except Exception as e:
            logger.error("Auto-recovery write_file failed: %s", e)
            return None
