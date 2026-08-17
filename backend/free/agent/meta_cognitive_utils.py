"""Meta-Cognitive ユーティリティ — 責務別モジュールへの窓口

実体は ``meta_cognitive_text`` (整形・修復) / ``meta_cognitive_write_rescue``
(依頼文からの本文救出) / ``meta_cognitive_scaffold`` (scaffold 除去) /
``meta_cognitive_content_gate`` (棄却判定) / ``meta_cognitive_tool_io``
(ツール入出力の解釈)。既存の呼出元とテストが ``meta_cognitive_utils.<名前>``
を参照しているため、公開名はここへ集約する。

``call_callback`` だけはどの責務にも属さない小さな実行時ヘルパーなのでここに残す。
"""

from __future__ import annotations

import inspect

from backend.free.agent.meta_cognitive_text import (
    _ANSWER_FRAMING_LEAD_RE,
    _ANSWER_FRAMING_RE,
    content_language_directive,
    _ENUMERATED_LINE_QUOTES,
    _ENUMERATED_LINE_RE,
    extract_enumerated_line_content,
    extract_literal_write_content,
    fix_json_backslashes,
    _HEADING_PATH_RE,
    _LITERAL_WRITE_CONTENT_RE,
    _LITERAL_WRITE_EXTENSIONS,
    _LITERAL_WRITE_REJECT_RE,
    _NARRATION_HEADING_TEXT_RE,
    _PATH_COMMENT_RE,
    strip_answer_framing,
    _strip_enclosing_quotes,
    strip_leading_narration_headings,
    strip_leading_path_comment,
    strip_markdown_wrapper,
    summarize_file_content,
    summarize_tool_args,
    _truncate_block_repetition,
    truncate_repetition,
)
from backend.free.agent.meta_cognitive_write_rescue import (
    _artifact_shape_for,
    _ARTIFACT_SHAPES,
    _GODAN_WRITE_REPORT,
    _HERE_STRING_ASSIGN_RE,
    _LEAD_IN_LINE_RE,
    _LITERAL_WRAPPER_MAX_FRAMING,
    _LITERAL_WRAPPER_MAX_RATIO,
    _LITERAL_WRAPPER_MIN_EXTRA,
    looks_like_literal_wrapper,
    looks_like_write_report,
    looks_like_write_script,
    _PREVIOUS_ANSWER_MIN_CHARS,
    _PREVIOUS_ANSWER_REF_RE,
    previous_answer_write_content,
    _QUOTE_IS_PATH_RE,
    _QUOTED_SPAN_RE,
    quoted_write_literals,
    rescue_quoted_write_literal,
    _SAHEN_AUX,
    _SCRIPT_SUFFIXES,
    _SOLE_CODE_FENCE_RE,
    _strip_lead_in,
    _TRANSFORM_VERB_RE,
    unwrap_sole_code_fence,
    _WRITE_REPORT_LABEL_RE,
    _WRITE_REPORT_RE,
    _WRITE_REPORT_STEM,
    _WRITE_REQUEST_RE,
    _WRITE_SCRIPT_RE,
)
from backend.free.agent.meta_cognitive_scaffold import (
    EXISTING_CONTENT_BLOCK_HEADING,
    FETCHED_DATA_BLOCK_HEADING,
    FETCHED_DATA_BLOCK_NOTE,
    fewshot_contains_task_log,
    _GENERATOR_DIRECTIVE_RE,
    _heading_titles_output_path,
    looks_like_task_log_echo,
    looks_like_task_log_residue,
    _PROMPT_SCAFFOLD_MARKERS,
    _SCAFFOLD_LABEL_LINE_RE,
    strip_generator_scaffold_block,
    strip_output_lead_in,
    strip_prompt_scaffold_lines,
    strip_task_log_scaffold,
    _TASK_LOG_FRAGMENT_RE,
    _TASK_LOG_LINE_RE,
    _TASK_SCAFFOLD_LINE_RE,
)
from backend.free.agent.meta_cognitive_content_gate import (
    _char_bigrams,
    _CJK_RE,
    _CODE_INDICATORS,
    contains_code_indicator,
    csv_content_lacks_rows,
    edit_produced_no_change,
    _EDIT_REQUEST_RE,
    _FEWSHOT_EXAMPLE_HEADING_RE,
    fewshot_seems_relevant,
    _FEWSHOT_TURN_LINE_RE,
    _FEWSHOT_USER_LINE_RE,
    generated_content_rejection,
    _INSTRUCTION_ECHO_MAX_LEN_RATIO,
    _INSTRUCTION_ECHO_MIN_CHARS,
    _INSTRUCTION_ECHO_MIN_LEN_RATIO,
    _INSTRUCTION_ECHO_MIN_SIMILARITY,
    _JA_APOLOGY_RE,
    _JA_INABILITY_RE,
    looks_like_fewshot_echo,
    looks_like_instruction_echo,
    looks_like_path_not_content,
    looks_like_prompt_echo,
    looks_like_refusal_or_missing_info,
    looks_like_task_restatement,
    looks_like_tool_call_syntax,
    _normalize_for_content_compare,
    _PATH_ONLY_RE,
    _REFUSAL_MARKERS,
    _TASK_RESTATEMENT_LABEL_RE,
    text_looks_like_code,
    _TOOL_CALL_SYNTAX_RE,
)
from backend.free.agent.meta_cognitive_tool_io import (
    command_run_failed,
    _EXIT_CODE_TOOLS,
    _find_matching_close_brace,
    is_tool_error,
    iter_balanced_brace_substrings,
    _parse_template_args,
    parse_template_tool_call,
    _TEMPLATE_JSON_CALL_RE,
    _TEMPLATE_TOOL_CALL_RE,
    _TOOL_EMPTY_RESULT_PREFIXES,
    TOOL_ERROR_PREFIX,
    tool_result_lacks_information,
    tool_result_succeeded,
    try_parse_tool_dict,
)


# ---------------------------------------------------------------------------
# コールバック
# ---------------------------------------------------------------------------

async def call_callback(callback, data) -> None:
    """コールバックを呼び出す（async/sync 両対応）"""
    if inspect.iscoroutinefunction(callback):
        await callback(data)
    else:
        callback(data)
