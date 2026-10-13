"""staged v2 のテスト工程 (契約テスト・参考テスト) の差込口 — 実体は Pro (docs/f_10 §11.3)

Free の ``run_staged_v2_pipeline`` は handler ``staged_test_stage`` が登録されているときだけ
テスト工程を呼ぶ。工程が読み書きするパイプラインの状態を :class:`StagedTestStageContext`
で渡し、書き換えた値は呼出し側が読み戻す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

#: Pro が登録する handler 名。
STAGED_TEST_STAGE_HANDLER = "staged_test_stage"


@dataclass
class StagedTestStageContext:
    """テスト工程が参照・更新するパイプラインの状態。

    ``checks`` / ``verification`` / ``notices`` / ``extra_files`` / ``phase_sec`` は呼出し側と同じ
    オブジェクトで、工程が書き足す。再代入されうる値 (``code_map`` / ``delivered_state`` /
    ``usage_check`` / ``usage_checked_code`` / ``tasks_failed`` / ``repair_reverts`` /
    ``repair_rounds_run`` / ``spec_examples`` / ``advisory_dropped``) は工程の後に読み戻す。
    """

    query: str
    brief: str
    locale: str
    skeleton: dict
    modules: list[dict]
    staged_cfg: dict
    state: Any
    ws: Any
    event_log: Any
    slots: Any
    missing: Any
    smoke_errors: list[str]
    smoke_unchecked: list[str]
    static_check: Any
    revert_if_worse: bool
    contract_history: Any
    requested_tests: list[str]
    is_named: Callable[[str], bool]
    advisory_path: str
    advisory_src: str
    check: Callable
    module_job: Callable
    remaining: Callable[[], float]
    run_usage_check: Callable
    verified: Callable
    checks: list
    verification: list[str]
    notices: list[str]
    extra_files: dict[str, str]
    phase_sec: dict[str, float]
    code_map: dict[str, str]
    delivered_state: Any
    usage_check: Any
    usage_checked_code: dict[str, str]
    tasks_failed: int
    repair_reverts: int
    repair_rounds_run: int
    spec_examples: list
    advisory_dropped: list[str]
