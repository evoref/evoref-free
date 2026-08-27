"""ビルトインツール群の定義と登録"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.api.chat.chat_constants import LLM_TOOL_EXECUTION_TIMEOUT_SEC
from backend.free.core.system_info import (
    format_hardware_facts,
    format_runtime_facts,
)
from backend.free.llm.utils import extract_content
from backend.i18n_helper import prose_language_name
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.history.history_manager import HistoryManager
    from backend.free.llm.local_client import LocalClient

logger = get_logger("agent.tools.builtin")

# 道具の種類ごとにモジュールを分けた。本モジュールは LLM へ委譲する道具
# (要約 / 翻訳 / 下書き / 履歴検索) と、レジストリへの登録、そして分割前から
# ``tools.builtin.<名前>`` で見えていた公開名の再エクスポートを担う。
from backend.free.agent.tools.calc import (  # noqa: F401
    calculate,
    _DISALLOWED_NODE_HINTS,
    _format_calc_result,
    _SAFE_NAMES,
    _SAFE_NODES,
)
from backend.free.agent.tools.filesystem import (  # noqa: F401
    apply_diff,
    _block_has_renderable_content,
    _check_calendar_table,
    _check_path_traversal,
    _EXPORT_DOC_EXTS,
    list_directory,
    _LIST_DIRECTORY_MAX_LINES,
    _MONTH_MAX_DAYS,
    read_file,
    _reject_unrenderable_rich_content,
    search_code,
    _TOOL_MAX_FILE_READ_BYTES,
    verify_syntax,
    _walk_tree,
    write_file,
    _write_rich_document,
)
from backend.free.agent.tools.shell import (  # noqa: F401
    # ``_last_full_output`` / ``_last_full_output_lines`` は再エクスポートしない。
    # shell 側が ``global`` で書き換える可変状態なので、別モジュールへ束縛すると
    # 値が固定されて実体と食い違う。読み書きは下の 2 関数を通す。
    clear_last_full_output,
    _decode_subprocess_output,
    get_last_full_output,
    _mkdir_safe,
    run_command,
    _run_command_async_impl,
)
from backend.free.agent.tools.html_text import (  # noqa: F401
    _BOILERPLATE_ATTR_RE,
    _collapse_blank_lines,
    _contains_markdown_table,
    _extract_main_content,
    _extract_naive,
    _EXTRACTION_MIN_RETAIN_RATIO,
    _flatten_tables,
    _has_boilerplate_attr,
    _html_to_text,
    _LINK_DENSE_BLOCK_TAGS,
    _LINK_DENSITY_MIN_LINKS,
    _LINK_DENSITY_MIN_TEXT,
    _LINK_DENSITY_THRESHOLD,
    _MAIN_CONTENT_SELECTORS,
    _MD_TABLE_SEP_LINE_RE,
    _prune_link_dense_blocks,
    _select_main_root,
    _strip_html_fallback,
    _STRIP_TAGS,
)
from backend.free.agent.tools.web_fetch import (  # noqa: F401
    fetch_url,
    _FETCH_URL_ALLOWED_SCHEMES,
    _FETCH_URL_MAX_BYTES,
    _FETCH_URL_MAX_REDIRECTS,
    _FETCH_URL_MAX_TEXT_CHARS,
    _FETCH_URL_MAX_TEXT_CHARS_TABLE,
    _FETCH_URL_TEXT_CONTENT_TYPE_PREFIXES,
    _FETCH_URL_USER_AGENT,
    _make_fetch_url,
    _redact_url_for_log,
    _validate_fetch_url,
)

from backend.free.constants import (
    SEARCH_HISTORY_CURRENT_SESSION_HEADER,
    SEARCH_HISTORY_NO_RESULTS_PREFIX as _SEARCH_HISTORY_NO_RESULTS_PREFIX,
    SEARCH_HISTORY_OTHER_SESSIONS_HEADER,
)


def _make_summarize(client: LocalClient):
    """summarize ツールハンドラを生成（LocalClient をクロージャでバインド）"""

    async def summarize(text: str) -> str:
        """テキストを要約する"""
        messages = [
            {"role": "system", "content": (
                "You are a summarization assistant. Summarize the given text "
                f"concisely. Write the summary in {prose_language_name(english=True)}."
            )},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
        ]
        try:
            result = await client.generate(
                messages, stream=False, temperature=0.3,
                id_slot=client.background_slot,
            )
            return extract_content(result)
        except Exception as e:
            logger.error("summarize tool failed: %s", e)
            return f"Error: {e}"

    return summarize


def _make_translate(client: LocalClient):
    """translate ツールハンドラを生成（LocalClient をクロージャでバインド）"""

    async def translate(text: str, target_lang: str) -> str:
        """テキストを指定言語に翻訳する

        システムプロンプトは「訳文だけを 1 案出す」ことを肯定形で指示する。
        指定が緩いとモデルは丁寧さの異なる複数案とその解説を返し、それが
        ツール結果として下流へ渡る (2026-08-05 ライブ監査: 「今の英文を日本語に
        訳して」に対し ``Here are a few ways to translate this, depending on the
        desired level of politeness: **Option 1: Po...`` という 1088 文字の
        英語の解説文が返った。最終応答は base 側が持ち直したが、ツールの
        戻り値としては不正)。
        """
        messages = [
            {"role": "system", "content": (
                "You are a translation assistant. Translate the given text "
                "accurately. Reply with the translated text only, as a single "
                "version, keeping the original formatting and line breaks."
            )},
            {"role": "user", "content": f"Translate the following text to {target_lang}:\n\n{text}"},
        ]
        try:
            result = await client.generate(
                messages, stream=False, temperature=0.3,
                id_slot=client.background_slot,
            )
            return extract_content(result)
        except Exception as e:
            logger.error("translate tool failed: %s", e)
            return f"Error: {e}"

    return translate


def _make_draft_document(client: LocalClient):
    """draft_document ツールハンドラを生成（LocalClient をクロージャでバインド）"""

    async def draft_document(instruction: str, format: str = "markdown") -> str:
        """指示に基づいてドキュメントを生成する

        本文の言語は生成時点の locale に従う。指示文はツール判定層の aux が
        書くため常に英語で届き (実測)、言語指示が無いと日本語ユーザーに英語の
        成果物が返る (2026-07-28 ライブ検証:「ここまでの数値を表にまとめて
        ください。」→ ``| Speed (km/h) | Time |`` の英語表を生成)。
        """
        messages = [
            {"role": "system", "content": (
                "You are a document drafting assistant. Generate documents in "
                f"{format} format. Write the document in "
                f"{prose_language_name(english=True)} unless the instruction "
                "explicitly asks for a different language."
            )},
            {"role": "user", "content": instruction},
        ]
        try:
            result = await client.generate(
                messages, stream=False, temperature=0.7,
                id_slot=client.background_slot,
            )
            return extract_content(result)
        except Exception as e:
            logger.error("draft_document tool failed: %s", e)
            return f"Error: {e}"

    return draft_document


def _make_run_command(config: dict):
    """run_command ハンドラを生成（config をクロージャでバインド）

    ToolsRegistry 経由の呼び出しで config.agent.dangerous_command_block 設定を反映するため、
    config をクロージャで捕捉する（設計書 §6.8.8）。
    """

    async def wrapped(command: str, timeout: int = 30) -> str:
        return await run_command(command, timeout, config)

    return wrapped


def _make_run_command_readonly(config: dict):
    """run_command_readonly ハンドラを生成（config をクロージャでバインド）

    chat モードの executable query (時刻 / OS / スペック等) 専用。実行前に
    ``reject_readonly_violation`` で読み取り専用 (書込 / 削除 / 導入 /
    ネットワーク送信なし) を検証し、通過したコマンドのみ ``run_command``
    本体へ委譲する (危険コマンドガード / 対話コマンドガードは本体側で適用)。
    判定層のバグや将来コードの誤用があっても、chat から破壊コマンドを実行
    できない構造的保証をこのラッパが担う。
    """

    async def wrapped(command: str, timeout: int = 30) -> str:
        from backend.free.agent.safety_patterns import reject_readonly_violation

        reject = reject_readonly_violation(command)
        if reject is not None:
            logger.warning(
                "Readonly command rejected (%s): %s", reject, command[:100],
            )
            return f"Error: readonly violation: {reject}"
        return await run_command(command, timeout, config)

    return wrapped


# search_history が 1 件もヒットしなかったときの戻り値プレフィックス。
# 実体は共通定数モジュール (成否判定側と emit 側で同じ文字列を共有するため)。
# 既存の import 経路を保つためここから再エクスポートする。
SEARCH_HISTORY_NO_RESULTS_PREFIX = _SEARCH_HISTORY_NO_RESULTS_PREFIX

#: 「この会話の最初/最後に何を言ったか」という **位置指定** の自己参照。
#:
#: この種の質問は逐語一致では絶対に当たらない。``session_id`` スコープ検索の
#: 唯一のヒット源は ``_find_matched_turns`` (逐語/トークン一致) であり、
#: 「一番最初に話しかけた内容」という語は当の 1 通目には出てこないためである。
#: 進行中セッションは ``summary`` も空 (要約は sleep-time でしか付かない) なので、
#: 「summary も matched_turns も無いヒットは捨てる」ガードで最後の 1 件も落ち、
#: 結果は必ず「該当なし」になる。
#:
#: 2026-08-16 ライブ監査ターン 35「今日の会話で、私が一番最初に話しかけた内容は
#: 何でしたか？」: index には ``first_user_preview`` として正解
#: (「こんばんは。今ちょうど夜中の0時半で、まだ起きてます。」) が入っており、
#: セッション本体にも 80 ターン全てが残っていた。それでも検索は該当なしを返し、
#: モデルは窓に残っていた最古の発話 (21 ターン目) を「最初」と答えた。
#:
#: 位置指定の質問には検索ではなく **境界ターンの直接取得** で答える。
_POSITIONAL_SESSION_QUERY_RE = re.compile(
    r"(一番最初|いちばん最初|最初に|冒頭|初めに|はじめに|書き出し"
    r"|一番最後|いちばん最後|最後に|末尾|直近|さいご"
    r"|\bfirst\b|\bearliest\b|\bbeginning\b|\blast\b|\blatest\b|\bend\b)",
    re.IGNORECASE,
)
#: 末尾側 (最後/last) を指しているか。上のどちらにも当たらなければ先頭側。
_POSITIONAL_TAIL_RE = re.compile(
    r"(一番最後|いちばん最後|最後に|末尾|直近|さいご|\blast\b|\blatest\b|\bend\b)",
    re.IGNORECASE,
)
#: 境界ターンとして返す user 発話の件数。1 件だと「最初の方に何を話したか」の
#: ような幅のある問いに答えられず、多すぎると窓を圧迫する。
_POSITIONAL_TURN_LIMIT = 3


def _boundary_turns_answer(
    manager: HistoryManager, session_id: str, query: str,
) -> str | None:
    """位置指定の自己参照質問に、セッションの境界ターンで直接答える。

    位置指定でない / セッションが取れない / user 発話が無い場合は ``None`` を
    返し、呼出側は従来どおり「該当なし」に落ちる。

    ``role`` は user に限る: 「私が最初に話しかけた内容」に対して assistant の
    応答を混ぜると、どちらが誰の発言か曖昧なまま「最初の発言」として提示される。
    """
    if not _POSITIONAL_SESSION_QUERY_RE.search(query):
        return None
    try:
        session = manager.get_session(session_id)
    except Exception as e:  # pragma: no cover - 履歴 I/O 失敗は従来経路へ
        logger.warning("Boundary turn lookup failed for %s: %s", session_id, e)
        return None
    if session is None:
        return None
    user_turns = [
        (i, t) for i, t in enumerate(session.turns)
        if t.get("role") == "user" and (t.get("content") or "").strip()
    ]
    if not user_turns:
        return None

    tail = bool(_POSITIONAL_TAIL_RE.search(query))
    picked = (
        user_turns[-_POSITIONAL_TURN_LIMIT:] if tail
        else user_turns[:_POSITIONAL_TURN_LIMIT]
    )
    where = "最後" if tail else "最初"
    lines = [
        SEARCH_HISTORY_CURRENT_SESSION_HEADER,
        f"[この会話の{where}の user 発話 {len(picked)} 件 "
        f"(全 {len(session.turns)} ターン中)]",
    ]
    for idx, turn in picked:
        content = (turn.get("content") or "").strip()
        preview = content if len(content) <= 200 else content[:200] + "…"
        lines.append(f"  turn#{idx} (user): {preview}")
    logger.info(
        "search_history answered a positional query from session boundaries "
        "(session=%s, tail=%s, turns=%d)", session_id, tail, len(picked),
    )
    return "\n".join(lines)


def _make_search_history(manager: HistoryManager):
    """search_history ツールハンドラを生成（HistoryManager をクロージャでバインド）"""

    def search_history(query: str, mode: str | None = None, limit: int = 10,
                       date_from: str | None = None, date_to: str | None = None,
                       session_id: str | None = None,
                       exclude_session_id: str | None = None) -> str:
        """過去の会話履歴を検索する

        ``session_id`` / ``exclude_session_id`` は LLM 向けツールスキーマには
        公開しない (ToolCallJudge が code 側で強制注入する。LLM が任意の
        session_id を指定できると他セッションの意図的な絞り込み回避に
        使われかねないため)。

        ``exclude_session_id`` は現在進行中のセッションを結果から外す。
        現在セッションの発言は既に会話コンテキストへ全文が載っており、
        検索結果として再注入しても情報は増えない一方、要約 (= 会話冒頭の
        発言) や断片が「独立した根拠」の顔で入り、後から訂正された内容を
        訂正前の値へ巻き戻す (2026-07-26 ライブ検証: 火曜→水曜と訂正した
        歯科の予約が、2 ターン後に検索結果のセッション要約経由で火曜へ
        戻った)。自己参照質問で ``session_id`` を明示スコープした場合は
        現在セッションこそが検索対象なので除外しない。
        """
        if session_id:
            exclude_session_id = None

        def _search(*, search_turns: bool) -> list[dict]:
            found = manager.search_sessions(
                query=query, mode=mode, limit=limit, search_turns=search_turns,
                date_from=date_from, date_to=date_to, session_id=session_id,
            )
            if exclude_session_id:
                found = [
                    r for r in found if r.get("session_id") != exclude_session_id
                ]
            return found

        try:
            results = _search(search_turns=False)
            # 結果が少ない場合のみターン検索で再検索
            if len(results) < limit:
                results = _search(search_turns=True)
            # summary も matched_turns も無いヒットは **会話の中身を 1 文字も
            # 運んでいない**。``search_sessions`` は ``max(score, 0.1)`` で全件に
            # 下駄を履かせ、``session_id`` 指定時はクエリ絞り込み自体を行わない
            # ため、無関係なセッションが必ず score=0.1 のヘッダだけで返る。
            # これを「根拠」の枠で base に渡すと、中身が無いのに検索が当たった
            # ことになり、モデルは仕方なく手元の (切り詰め済み) 文脈から適当な
            # 発言を選んで断定する。実インシデント (2026-08-14 ライブ監査
            # ターン19): 「この会話で一番最初に送ったメッセージは？」に対し
            # ``[2026-08-14T04:14:43Z] mode=chat score=0.1`` だけが返り、
            # 実際の 1 通目ではなく窓の先頭 (7 ターン目の質問) を答えた。
            results = [
                r for r in results
                if r.get("summary") or r.get("matched_turns")
            ]
            if not results:
                # 位置指定の自己参照 (「この会話の一番最初に言ったこと」) は
                # 逐語一致では構造的に当たらない。セッションが特定できている
                # ときに限り、境界ターンを直接返す (_POSITIONAL_SESSION_QUERY_RE)。
                boundary = (
                    _boundary_turns_answer(manager, session_id, query)
                    if session_id else None
                )
                if boundary:
                    return boundary
                return f"{SEARCH_HISTORY_NO_RESULTS_PREFIX}{query}"

            lines: list[str] = []
            # 由来 (今回の会話 / 別の会話) を本文先頭で必ず宣言する。
            # exclude_session_id が入っている = 現在セッションを除外した検索
            # なので、ヒットは構造的に全て別セッション。session_id 明示時は
            # 逆に全て現在セッション。どちらでもない (スコープ未注入) 場合は
            # 混在し得るので宣言しない。
            if exclude_session_id:
                lines.append(SEARCH_HISTORY_OTHER_SESSIONS_HEADER)
            elif session_id:
                lines.append(SEARCH_HISTORY_CURRENT_SESSION_HEADER)
            for r in results:
                header = f"[{r['started_at']}] mode={r['mode']} score={r['relevance_score']:.1f}"
                if r.get("summary"):
                    # summary はそのセッションの「最初のユーザ発話」そのもの。
                    # 同じ会話の後半で訂正されていてもここには反映されないため、
                    # 裸で出すと現在も有効な事実として読まれ、訂正済みの値へ
                    # 巻き戻す (2026-07-26 ライブ検証: 火曜→水曜と訂正済みの
                    # 予約が、過去セッションの要約「来週の火曜日に歯科の予約を
                    # 入れました。」経由で火曜へ戻った)。由来を明示して、
                    # 会話冒頭の発言にすぎないことが読み取れるようにする。
                    header += f" | first_message: {r['summary']}"
                lines.append(header)
                for turn in r.get("matched_turns", []):
                    lines.append(f"  turn#{turn['index']} ({turn['role']}): {turn['content_preview']}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("search_history tool failed: %s", e)
            return f"Error: {e}"

    return search_history


def register_builtin_tools(
    registry,
    config: dict | None = None,
    local_client: LocalClient | None = None,
    history_manager: HistoryManager | None = None,
) -> None:
    """ビルトインツールをレジストリに一括登録"""
    cfg = config or {}

    registry.register(
        name="calculate",
        func=calculate,
        description=(
            "Evaluate a Python-syntax arithmetic expression safely (numeric "
            "literals + - * / % ** // and parentheses only)"
        ),
        parameters={
            "expression": {
                "type": "string",
                "description": (
                    "Use ** for exponentiation, NOT ^ (which is bitwise XOR in "
                    "this sandbox and will error). Function calls (e.g. gcd(), "
                    "sqrt()) and symbolic constants (e.g. pi, e) are NOT "
                    "supported -- inline the numeric value instead (e.g. 3.14159 "
                    "instead of pi), and compute functions like gcd manually "
                    "step-by-step rather than calling them."
                ),
            },
        },
    )

    registry.register(
        name="read_file",
        func=read_file,
        description=(
            "Read the contents of a file. Line 1 is metadata in brackets "
            "(total lines / total chars) — read counts from there and keep it "
            "out of your answer. The file content itself starts on line 2"
        ),
        parameters={
            "file_path": {"type": "string", "description": "Path to the file to read"},
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-based, inclusive). Optional",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (1-based, inclusive). Optional",
            },
        },
    )

    registry.register(
        name="write_file",
        func=write_file,
        description="Write content to a file (parent directories are created automatically, no mkdir needed)",
        parameters={
            "file_path": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        # 選択は create 限定のまま。chat では meta_cognitive の書き出し経路が
        # パス解決と action_blocked のガードを通したうえで execute するので、
        # 分類器へ直接選ばせてそのガードを飛ばしたくない。
        modes=["create"],
        # ただし chat でも **実際に実行される** ので目録には載せる。
        # 実インシデント (2026-08-27 ライブ監査): chat で write_file を 3 回
        # 実行した直後の「使えるツールの一覧」に write_file だけが無かった。
        # 同じ会話の別の問いには「write_file を 3 回呼んだ」と正しく答えており、
        # 目録だけが実態とずれていた。
        inventory_modes=["chat", "create"],
    )

    # chat でも使える。read_file (ファイル全文) と list_directory (ツリー全体) は
    # 既に chat 可なのに、**両者より読み取り範囲が狭い** search_code だけが
    # create 限定だった。結果、chat で所在を問われたモデルには「全文を読む」か
    # 「ツリーを列挙する」しか手が無い。
    #
    # 2026-08-16 ライブ監査ターン 19「このプロジェクトで LangChain はどこで
    # 使われていますか？」: list_directory が 5,477 文字のツリーを返し、その
    # 再 prefill で **218.6 秒** (当セッション最長) を消費したうえ、
    # 「一覧は途中が省略されているため確認できません」で終わった。
    # grep 1 回で済む質問だった。
    registry.register(
        name="search_code",
        func=search_code,
        description="Search code files using a regex pattern",
        parameters={
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "directory": {"type": "string", "description": "Directory to search in"},
        },
        modes=["chat", "create"],
    )

    registry.register(
        name="list_directory",
        func=list_directory,
        description="List the directory structure as an indented tree",
        parameters={
            "directory": {"type": "string", "description": "Directory to list"},
            # 関数は元から max_depth を持っていたがスキーマに出しておらず、
            # 「直下だけ一覧して」という依頼を表現する手段が無かった。常に 3 階層
            # の木が返り、受け取ったモデルがインデントを読み違えて入れ子の項目を
            # 直下の項目として並べた (実インシデント 2026-08-01 再検証)。
            "max_depth": {
                "type": "integer",
                "description": (
                    "How many levels to descend. Use 1 for the immediate "
                    "children of the directory only. Default 3."
                ),
            },
        },
    )

    registry.register(
        name="apply_diff",
        func=apply_diff,
        description="Apply a unified diff to a file",
        parameters={
            "file_path": {"type": "string", "description": "Path to the file to patch"},
            "diff_text": {"type": "string", "description": "Unified diff content"},
        },
        modes=["create"],
    )

    registry.register(
        name="run_command",
        func=_make_run_command(cfg),
        description="Execute a shell command (CLI only)",
        parameters={
            "command": {"type": "string", "description": "Shell command to execute"},
        },
        modes=["create"],
    )

    # chat モードの executable query (時刻 / OS / スペック等) 専用。
    # hidden=True で LLM プロンプトのツール一覧には出さず、tool_call_judge の
    # executable 経路 (_executable_tool_for_mode) がコード側から注入する。
    # mode ゲート (2026-07-18) を変えずに chat のシステム情報クエリを復活
    # させるための登録 (2026-07-21 回帰対策、docs/f_03_agent_engine.md §3.1)。
    registry.register(
        name="run_command_readonly",
        func=_make_run_command_readonly(cfg),
        description=(
            "Execute a read-only shell command for environment facts "
            "(injected by the tool judge; not directly selectable)"
        ),
        parameters={
            "command": {"type": "string", "description": "Read-only shell command to execute"},
        },
        modes=["chat"],
        hidden=True,
    )

    # 搭載 / 空き RAM・CPU・OS・CPU 使用率・GPU VRAM を **シェルを介さず**
    # 返す chat 専用ツール。
    # readonly allow-list (_READONLY_SAFE_MODULES) はチャットから渡される
    # コマンド文字列にしか掛からないため、backend 内の実装で測る
    # (free/core/vram_monitor が nvidia-smi を直接叩いているのと同じ立て付け)。
    # hidden=True: LLM のツール一覧には出さず、tool_call_judge の層0.6 が注入する。
    registry.register(
        name="system_hardware_info",
        func=lambda: format_hardware_facts(),
        description=(
            "Report host hardware facts (OS / CPU / cores / RAM / CPU usage / "
            "GPU VRAM) without a shell (injected by the tool judge; not "
            "directly selectable)"
        ),
        parameters={},
        modes=["chat"],
        hidden=True,
    )

    # 自己構成 (どのモデルを serve しているか / n_ctx / ポート)。ハードウェアと
    # 同じ立て付けで、シェル経由では取れない (readonly allow-list は config も
    # /props も読めない)。hidden=True で層0.6b が注入する。
    # 実測 (/props 由来の metadata) と宣言 (config) を区別して返す —
    # config を書き替えても llama-server を再起動しなければ反映されない。
    registry.register(
        name="evoref_runtime_info",
        func=lambda: format_runtime_facts(
            cfg, getattr(local_client, "metadata", None),
        ),
        description=(
            "Report this assistant's own runtime configuration (served base "
            "model, embedding model, context size, slots, llama-server ports) "
            "without a shell (injected by the tool judge; not directly "
            "selectable)"
        ),
        parameters={},
        modes=["chat"],
        hidden=True,
    )

    registry.register(
        name="verify_syntax",
        func=verify_syntax,
        description="Verify Python file syntax (checks existence and runs py_compile)",
        parameters={
            "file_path": {"type": "string", "description": "Path to the Python file to verify"},
        },
        modes=["create"],
    )

    # fetch_url（デフォルト有効、config で無効化可能）
    if cfg.get("tools", {}).get("fetch_url_enabled", True):
        fetch_timeout = cfg.get("tools", {}).get("fetch_url_timeout", 10)
        registry.register(
            name="fetch_url",
            func=_make_fetch_url(cfg),
            description="Fetch a URL and extract text content",
            parameters={
                "url": {"type": "string", "description": "URL to fetch"},
                "timeout": {"type": "integer", "description": f"Request timeout (default: {fetch_timeout})"},
            },
        )

    # LLM ツール（summarize / translate / draft_document）
    if local_client is not None:
        registry.register(
            name="summarize",
            func=_make_summarize(local_client),
            description="Summarize the given text concisely",
            parameters={
                "text": {"type": "string", "description": "Text to summarize"},
            },
            modes=["chat"],
            timeout_sec=LLM_TOOL_EXECUTION_TIMEOUT_SEC,
        )

        registry.register(
            name="translate",
            func=_make_translate(local_client),
            description="Translate text to a target language",
            parameters={
                "text": {"type": "string", "description": "Text to translate"},
                "target_lang": {"type": "string", "description": "Target language (e.g. 'English', 'Japanese')"},
            },
            modes=["chat"],
            timeout_sec=LLM_TOOL_EXECUTION_TIMEOUT_SEC,
        )

        registry.register(
            name="draft_document",
            func=_make_draft_document(local_client),
            description="Generate a document based on instructions",
            parameters={
                "instruction": {"type": "string", "description": "Instructions for document generation"},
                "format": {"type": "string", "description": "Output format (e.g. 'markdown', 'plain')"},
            },
            modes=["chat"],
            timeout_sec=LLM_TOOL_EXECUTION_TIMEOUT_SEC,
        )

    # 会話履歴検索ツール
    if history_manager is not None:
        registry.register(
            name="search_history",
            func=_make_search_history(history_manager),
            description="Search past conversation history by keyword and/or date range",
            parameters={
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords to search for, NOT the user's question verbatim "
                        "(e.g. use 'Rust' rather than 'what is the user's favorite "
                        "programming language?'). Extract the key noun(s)/proper "
                        "noun(s) from the request."
                    ),
                },
                "mode": {"type": "string", "description": "Filter by mode (chat/create)"},
                "limit": {"type": "integer", "description": "Maximum number of results (default: 10)"},
                "date_from": {"type": "string", "description": "Start date in ISO 8601 format (e.g. '2026-03-01')"},
                "date_to": {"type": "string", "description": "End date in ISO 8601 format (e.g. '2026-03-31')"},
            },
        )

    logger.info("Registered %d builtin tools", registry.count)
