"""ストリーミング関数 — 層ごとのモジュールへの窓口

実体は ``chat_stream_common`` / ``chat_stream_output`` と、層ごとの
``chat_stream_{meta,long_form,deliberative,staged}``。既存の呼出元とテストが
``chat_streaming.<名前>`` を参照しているため、公開名はここへ集約する。
"""

from __future__ import annotations

from backend.free.api.chat.chat_stream_common import (
    _cancel_flags,
    cancel_scope,
    _emit_timing,
    logger,
    _make_step_queue_callback,
    meta_tool_routing_false_positive,
    meta_tool_routing_success,
    rag_signals_from_chunks,
    sse,
)
from backend.free.api.chat.chat_stream_output import (
    _APPEND_HINT_RE,
    clean_generated_text,
    _DEDUP_MIN_PARAGRAPH_CHARS,
    dedup_verbatim_paragraphs,
    _EDITOR_BLANK_LINE_RE,
    _editor_language_for_extension,
    _EDITOR_LANGUAGE_MAP,
    _HEADING_LINE_RE,
    _infer_output_extension,
    long_form_write_file,
    _MD_EXT_HINT_RE,
    _NEEDS_EXISTING_RE,
    _normalize_editor_text,
    read_existing_for_append,
    _READ_REF_RE,
    _resolve_editor_output_format,
    _resolve_long_form_target_path,
    _resolve_split_unit_path,
    _slug_for_split_file,
    _SPLIT_FILE_NAME_SAFE_RE,
    split_write_index,
    split_write_single_unit,
    _WRITE_HINT_RE,
)
from backend.free.api.chat.chat_stream_meta import (
    _build_meta_cognitive_agent_runner,
    _drain_meta_cognitive_steps,
    _emit_meta_cognitive_result_frames,
    _finalize_meta_cognitive_stream,
    _meta_cognitive_body_text,
    meta_cognitive_recorded_text,
    _STEP_DESCRIPTION_MAX_CHARS,
    stream_meta_cognitive,
    stream_reactive,
    sync_meta_cognitive,
    _truncate_step_description,
    _WRITTEN_PATH_RE,
    _written_paths,
)
from backend.free.api.chat.chat_stream_long_form import (
    _emit_long_form_episode,
    _emit_long_form_init_steps,
    _finalize_long_form_stream,
    _flush_step_queue_split_aware,
    _flush_step_queue_to_sse,
    _LongFormStreamState,
    stream_long_form,
    sync_long_form,
)
from backend.free.api.chat.chat_stream_deliberative import (
    _apply_generation_params,
    _DeliberativeStreamState,
    _drain_deliberative_step_queue,
    _finalize_deliberative_stream,
    _maybe_cache_reactive_response,
    _retry_zero_tokens_deliberative,
    stream_deliberative,
    _stream_filtered_token_pipeline,
    stream_reactive_light,
    sync_deliberative,
    sync_reactive_light,
)
from backend.free.api.chat.chat_stream_staged import (
    _finalize_staged_stream,
    _stage_label_for_task,
    _STAGE_LABELS,
    _staged_deliverable_path,
    _staged_import_smoke,
    _staged_internal_names,
    _staged_output_dir,
    _staged_postprocess,
    _STAGED_PROJECT_ID,
    _staged_pytest_counts,
    _STAGED_TOTAL_TIMEOUT_DEFAULT_SEC,
    _staged_write_file,
    stream_staged_create,
    _translate_loop_event,
)
