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
