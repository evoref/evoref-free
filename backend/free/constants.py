"""Free エディション共通定数

CLI / API / エージェント間で共有する定数を定義する。
"""

# 出力切り詰めマーカー（エージェントの run_command() と CLI の自動ヒント検出で共有）
TRUNCATION_MARKER = "行省略"

# run_command が非ゼロ終了したとき結果末尾へ付与する行頭マーカー。
# tools/builtin.py が emit し、deliberative の成否判定 (command_run_failed) が参照する。
# 非ゼロ終了時のみ付与されるため、このマーカーの有無がコマンド失敗の信号になる。
COMMAND_EXIT_CODE_PREFIX = "[exit code:"

# search_history が 0 件だったとき返す先頭マーカー。tools/builtin.py が emit し、
# 成否判定 (meta_cognitive_utils.tool_result_lacks_information) と
# フロントの表示置換 (AgenticSteps.svelte) が参照する。
SEARCH_HISTORY_NO_RESULTS_PREFIX = "No results found for: "

# fetch_url がホスト名を名前解決できなかったときの先頭。tools/web_fetch.py が emit し、
# 取得の失敗の注記 (meta_cognitive_tool_io.tool_error_kind) が参照する。
FETCH_UNRESOLVED_HOST_PREFIX = "Error: Failed to resolve hostname"
# fetch_url が private/reserved のアドレスへの取得を拒んだときの先頭 (取得の失敗の注記の種別 blocked)。
FETCH_BLOCKED_PREFIX = "Error: Access to private/reserved addresses is not allowed"

# fetch_url の取得が例外で終えたときの先頭 (``<先頭> (<例外の型>): <理由>``)。``Error:`` で
# 始めてエラーと読ませる (以前は ``Error fetching URL`` で始まり、接続の失敗が成功に数えられた)。
# url_curator の失敗の印もこの定数から組む。
FETCH_ERROR_PREFIX = "Error: fetching URL failed"
#: 到達できなかった (接続できない・時間切れ) とみなす例外の型名 (取得の失敗の注記の種別)。
FETCH_UNREACHABLE_ERRORS = frozenset({
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "TimeoutException",
})
# meta の検索に探す場所が無い (依頼が名指さず、継ぐフォルダも無い) ときの結果。プロセスの CWD
# (インストール根) は探さない (2026-10-04 run23 反証 HIGH-1、f_03 §4.2.1)
SEARCH_NO_LOCATION_ERROR = (
    "Error: No folder to search: the request names no folder, so the search was not run"
)
# 読込み・検索の対象が無いときの先頭 (tools/filesystem.py が emit、取得の失敗の注記が参照)。
FILE_NOT_FOUND_PREFIX = "Error: File not found"
DIRECTORY_NOT_FOUND_PREFIX = "Error: Directory not found"
# 裸のファイル名が複数のフォルダにあって決められないとき (file_ledger.resolve_bare_filename)。
FILE_AMBIGUOUS_PREFIX = "Error: Ambiguous file name"

# search_history の結果が「今のセッション」か「別のセッション」かを本文先頭で
# 明示する見出し。スコープ注入 (ToolCallJudge._maybe_scope_session_search) は
# 正規表現で自己参照を推定しており、外れると別セッションの内容が「さっきの話」
# として提示される (2026-08-05 ライブ監査: 「さっき ...txt に書き込んで
# もらったはず」に対し、別セッションの `監査メモ.txt に「検証コードは
# アオサギ42」` を今回の会話で依頼された操作として列挙した)。
# 推定精度に依存せず、由来をデータ側に必ず載せることで混同を構造的に防ぐ。
#
# 「今回の会話で起きたことではありません」だけだと、**継続する事実** まで
# 無効化される。実インシデント 2026-08-22 ライブ監査 (新規セッション ターン1):
# 「私の名前を覚えていますか？」に対し search_history は過去会話の「小川」を
# 引き当てていたのに、回答は「過去の会話記録には『小川』という名前の記載が
# ありますが、今回の会話ではまだご自身から直接お聞かせいただいていないため、
# 現時点では特定できません」。同じセッションの後続ターンでは (search_history を
# 経由しない注入経路から) 「小川です。」と即答しており、抑止していたのはこの
# 見出しだった。禁じたいのは **出来事・操作の取り違え** だけなので、そこに
# 限定して書く。
SEARCH_HISTORY_OTHER_SESSIONS_HEADER = (
    "[以下は**別の（過去の）会話**の記録です。"
    "ここでの操作や出来事を「今回の会話でのこと」として説明しないでください。"
    "ただしユーザーが述べた名前・設定・好みなど、**継続する事実**は"
    "現在も有効なものとして答えてかまいません]"
)
SEARCH_HISTORY_CURRENT_SESSION_HEADER = "[以下は**今回の会話**の記録です]"

#: 件数上限で切った search_history の末尾に添える 1 行 (モデル向け)。切ったことを
#: 書かないと、モデルは表示分を全件として答える (実機 2026-10-03: 当日 42 件のうち
#: 古い 12 件が出ないまま一覧を全件のように答えた)。
SEARCH_HISTORY_TRUNCATED_LISTING_NOTE = "(この期間の会話は全 {total} 件、新しい順に {shown} 件を表示)"
SEARCH_HISTORY_TRUNCATED_RESULTS_NOTE = "(該当する会話は全 {total} 件、上位 {shown} 件を表示)"

#: 日付の窓で 0 件だった search_history の 2 行目 (モデル向け、1 行目は空振りの接頭辞のまま)。
#: 期間を書かずに空振りだけ返すと、モデルはプロンプトの記憶 (今日の内容) をその日の会話として
#: 語る (実機 2026-10-03: 前日のセッションが無いのに「昨日は、私が Alice であることを確認し…」)。
#: 窓が今を含み、今回の会話を除いて検索したときは、今回の会話を会話履歴から答えさせる。
SEARCH_HISTORY_EMPTY_WINDOW_LEAD = "[期間 "
SEARCH_HISTORY_EMPTY_WINDOW_NOTE = (
    SEARCH_HISTORY_EMPTY_WINDOW_LEAD + "{period} の会話の記録はありません。"
    "この期間に何を話したかは「記録がない」と答え、"
    "記憶や今回の会話の内容をこの期間の会話として述べないでください]"
)
#: 窓が今を含む版の目印 (受け手が「前の話題は無視せよ」を添えないために見る)。
SEARCH_HISTORY_EMPTY_WINDOW_CURRENT_MARK = "の会話の記録は、今回の会話のほかにはありません。"
SEARCH_HISTORY_EMPTY_WINDOW_CURRENT_NOTE = (
    SEARCH_HISTORY_EMPTY_WINDOW_LEAD + "{period} " + SEARCH_HISTORY_EMPTY_WINDOW_CURRENT_MARK
    + "今回の会話で話したことは会話履歴から答え、"
    "記憶の内容など、それ以外をこの期間の会話として述べないでください]"
)


def search_history_window_note(result: str) -> str:
    """空振りした search_history の結果から日付の窓の注記を取り出す (無ければ空文字)。"""
    if not result.startswith(SEARCH_HISTORY_NO_RESULTS_PREFIX):
        return ""
    _, _, rest = result.partition("\n")
    return rest if rest.startswith(SEARCH_HISTORY_EMPTY_WINDOW_LEAD) else ""

#: 由来見出し → UI に出す短いラベルの i18n キー。見出しはモデルへの指示を含むので
#: step 表示にはそのまま出さない (2026-09-27 監査 F12、docs/f_03 §3.2)。
_SEARCH_HISTORY_DISPLAY_KEYS: dict[str, str] = {
    SEARCH_HISTORY_OTHER_SESSIONS_HEADER: "agent.search_history.display_other_sessions",
    SEARCH_HISTORY_CURRENT_SESSION_HEADER: "agent.search_history.display_current_session",
}


def search_history_display_text(result: str) -> str:
    """search_history の結果を UI 表示用にする (先頭の由来見出しを短いラベルへ)。

    モデルへ渡す文 (ツール結果の注入・会話履歴) には使わない。
    """
    from backend.i18n_helper import msg

    if search_history_window_note(result):
        # 期間の注記はモデルへの指示文。表示は 1 行目 (空振り) だけにする (F12 と同じ)。
        return result.partition("\n")[0]
    for header, key in _SEARCH_HISTORY_DISPLAY_KEYS.items():
        if result.startswith(header):
            rest = result[len(header):].strip()
            return f"{msg(key)} {rest}" if rest else msg(key)
    return result

# read_file が結果の先頭へ付けるメタ行の開始マーカー。行数・文字数をモデルに
# 数えさせないための**モデル向け**の補助情報であり、ユーザーに見せる本文では
# ない。emit 側 (tools/builtin.read_file) と除去側 (deliberative の逐語エコー)
# で同じ定数を共有する。
# 逐語エコー (「そのまま見せて」) は生成を迂回してツール結果をそのまま返すため、
# 除去しないとこのメタ行がそのまま回答に出る (2026-08-05 ライブ監査で 2 件)。
READ_FILE_META_PREFIX = "[file: "
