"""自律実行ループ

EvorefMem 統合仕様 における自律ループ層。`task` 型 SemanticFact を駆動源
とし、状態遷移 (open → in_progress → done|failed) を SemMem に即書き込みする。

- ループ driver の **骨組み** と PRD JSON パーサ
  `pick_next_task` / `update_task_status`
- 品質ゲート層 (pytest / typecheck / lint アダプタ)
- failure_pattern 即時記録 + sleep-time Step 13 統合
- クリーンコンテキスト再起動 + bootstrap
- progress_marker 即時記録 + LoopReport
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
from backend.free.loop.prd_parser import (
    PRDParseError,
    parse_prd_json,
    parse_prd_json_file,
)
from backend.free.loop.progress_marker import (
    PROGRESS_MARKER_PREFIX,
    PROGRESS_PREDICATE,
    ProgressPayload,
    list_progress_markers,
    parse_progress_object,
    write_progress_marker,
)
from backend.free.loop.report import (
    LoopReport,
    aggregate_failure_patterns,
    generate_loop_report,
)
from backend.free.loop.quality_gate import (
    GateAction,
    GateResult,
    LintGate,
    PytestGate,
    QualityGate,
    QualityGateOutcome,
    TypecheckGate,
    build_default_gates,
    decide_gate_action,
    rollback_working_tree,
    run_quality_gates,
)

__all__ = [
    "PROGRESS_MARKER_PREFIX",
    "PROGRESS_PREDICATE",
    "BootstrapResult",
    "ConsolidationSummary",
    "FailurePayload",
    "GateAction",
    "GateResult",
    "LintGate",
    "LoopDriver",
    "LoopDriverState",
    "LoopNotRunningError",
    "LoopReport",
    "MaybeResetReport",
    "PRDParseError",
    "ProgressPayload",
    "PytestGate",
    "QualityGate",
    "QualityGateOutcome",
    "ResetReport",
    "TaskFactView",
    "TypecheckGate",
    "aggregate_failure_patterns",
    "bootstrap_project_context",
    "build_default_gates",
    "compute_failure_signature_from_gate",
    "compute_failure_signatures_from_outcome",
    "consolidate_failure_patterns",
    "decide_gate_action",
    "decode_task_fact",
    "encode_task_object",
    "estimate_episodic_tokens",
    "extract_actions_from_steps",
    "extract_error_type",
    "extract_file_path",
    "generate_loop_report",
    "list_progress_markers",
    "list_tasks",
    "maybe_reset_and_bootstrap",
    "parse_failure_object",
    "parse_prd_json",
    "parse_prd_json_file",
    "parse_progress_object",
    "pick_next_task",
    "reopen_orphan_in_progress_tasks",
    "reset_episodic_context",
    "rollback_working_tree",
    "run_quality_gates",
    "should_reset_episodic",
    "update_task_status",
    "write_failure_note",
    "write_progress_marker",
]
