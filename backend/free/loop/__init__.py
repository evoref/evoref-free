"""自律実行ループ

EvorefMem 統合仕様 における自律ループ層。`task` 型 SemanticFact を駆動源
とし、状態遷移 (open → in_progress → done|failed) を SemMem に即書き込みする。

- ループ driver (クリエイトの staged パイプラインが専用インスタンスを駆動する)
  `pick_next_task` / `update_task_status`
- 品質ゲートの結果データ型
- failure_pattern 即時記録 + sleep-time Step 13 統合
- クリーンコンテキスト再起動 + bootstrap
- progress_marker 即時記録
"""

from backend.free.loop.bootstrap import (
    BootstrapResult,
    MaybeResetReport,
    ResetReport,
    bootstrap_project_context,
    estimate_episodic_tokens,
    maybe_reset_and_bootstrap,
    reset_episodic_context,
    should_reset_episodic,
)
from backend.free.loop.driver import (
    LoopDriver,
    LoopDriverState,
    LoopNotRunningError,
    TaskFactView,
    decode_task_fact,
    encode_task_object,
    list_tasks,
    pick_next_task,
    reopen_orphan_in_progress_tasks,
    update_task_status,
)
from backend.free.loop.failure_note import (
    ConsolidationSummary,
    FailurePayload,
    compute_failure_signature_from_gate,
    compute_failure_signatures_from_outcome,
    consolidate_failure_patterns,
    extract_actions_from_steps,
    extract_error_type,
    extract_file_path,
    parse_failure_object,
    write_failure_note,
)
from backend.free.loop.progress_marker import (
    PROGRESS_MARKER_PREFIX,
    PROGRESS_PREDICATE,
    ProgressPayload,
    list_progress_markers,
    parse_progress_object,
    write_progress_marker,
)
from backend.free.loop.quality_gate import (
    GateResult,
    QualityGateOutcome,
)

__all__ = [
    "PROGRESS_MARKER_PREFIX",
    "PROGRESS_PREDICATE",
    "BootstrapResult",
    "ConsolidationSummary",
    "FailurePayload",
    "GateResult",
    "LoopDriver",
    "LoopDriverState",
    "LoopNotRunningError",
    "MaybeResetReport",
    "ProgressPayload",
    "QualityGateOutcome",
    "ResetReport",
    "TaskFactView",
    "bootstrap_project_context",
    "compute_failure_signature_from_gate",
    "compute_failure_signatures_from_outcome",
    "consolidate_failure_patterns",
    "decode_task_fact",
    "encode_task_object",
    "estimate_episodic_tokens",
    "extract_actions_from_steps",
    "extract_error_type",
    "extract_file_path",
    "list_progress_markers",
    "list_tasks",
    "maybe_reset_and_bootstrap",
    "parse_failure_object",
    "parse_progress_object",
    "pick_next_task",
    "reopen_orphan_in_progress_tasks",
    "reset_episodic_context",
    "should_reset_episodic",
    "update_task_status",
    "write_failure_note",
    "write_progress_marker",
]
