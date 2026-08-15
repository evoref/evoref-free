"""`/api/optimize` ハンドラから抽出した純粋な収集ロジック

`backend/free/api/optimize.py` の `optimize_status` / `optimize_history`
ハンドラ内に直書きされていた以下のロジックを抽出した純粋関数群:
- PromptMeta / AuxPromptMeta → レスポンススキーマへのマッピング
- LearningScheduler 設定値の既定値マージ
- Unix epoch → ISO8601 タイムスタンプ整形
- 履歴エントリのマッピング

レイヤー責務:
- `optimize.py` (API 層)        — HTTP / Pydantic / Pro/Free ガード / 状態取得
- `_optimize_collectors.py` (helper) — 純粋なマッピング / 既定値マージ / 整形

すべて引数のみに依存し、FastAPI / app_state / グローバル設定 / I/O に
触れない pure 関数 (LearningScheduler / PromptManager のメソッド呼び出しを
含むため副作用ゼロの保証は呼び出し側のオブジェクトに委ねる)。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from backend.free.api.learning._learning_collectors import latest_level2_run
from backend.free.api.learning._optimize_schemas import (
    AuxTaskStatus,
    PromptHistoryEntry,
    PromptModeStatus,
)

if TYPE_CHECKING:
    from backend.free.agent.aux_prompt_manager import AuxPromptManager
    from backend.free.agent.prompt_manager import SystemPromptManager


# ── スケジューラ既定値 (元: optimize_status 内のリテラル) ─────────────────


SCHEDULER_DEFAULTS: dict[str, int] = {
    "generations": 10,
    "population_size": 5,
    "min_experiences": 20,
    "spsa_iterations": 500,
    "sparse_params": 200,
    "min_failures": 50,
}


# ── PromptMeta / AuxPromptMeta → スキーマへのマッピング ──────────────


def collect_prompt_mode_statuses(
    prompt_mgr: SystemPromptManager | None,
    modes: list[str],
) -> list[PromptModeStatus]:
    """指定モードリストから `PromptModeStatus` のリストを構築する。

    `prompt_mgr` が `None` の場合は空リストを返す。`get_meta` が
    `ValueError` を投げたモードはスキップする (元の handler 動作と互換)。
    """
    if prompt_mgr is None:
        return []
    result: list[PromptModeStatus] = []
    for mode in modes:
        try:
            meta = prompt_mgr.get_meta(mode)
        except ValueError:
            continue
        result.append(PromptModeStatus(
            mode=meta.mode,
            version=meta.version,
            source=meta.source,
            updated_at=meta.updated_at,
            model_calibrated_for=meta.model_calibrated_for,
        ))
    return result


def collect_aux_task_statuses(
    aux_prompt_mgr: AuxPromptManager | None,
    tasks: list[str],
) -> list[AuxTaskStatus]:
    """指定タスクリストから `AuxTaskStatus` のリストを構築する。

    `aux_prompt_mgr` が `None` の場合は空リストを返す。`get_meta` が
    `ValueError` を投げたタスクはスキップする。Pro/Free ガードは呼び出し側の
    責務 (この関数自体は edition-agnostic)。
    """
    if aux_prompt_mgr is None:
        return []
    result: list[AuxTaskStatus] = []
    for task in tasks:
        try:
            meta = aux_prompt_mgr.get_meta(task)
        except ValueError:
            continue
        result.append(AuxTaskStatus(
            task=meta.task,
            version=meta.version,
            source=meta.source,
            updated_at=meta.updated_at,
            fitness_score=meta.fitness_score,
        ))
    return result


# ── スケジューラ設定値の既定値マージ ─────────────────────────────────────


def extract_scheduler_params(scheduler: object | None) -> dict[str, int]:
    """`LearningScheduler` から Level 1/2 設定値を取り出し、未初期化時は
    既定値で埋めた dict を返す。

    元 handler では if scheduler is not None: の中で 6 個の値を上書き代入
    していたが、純粋なマージ操作として抽出。
    """
    params = dict(SCHEDULER_DEFAULTS)
    if scheduler is None:
        return params
    for key in SCHEDULER_DEFAULTS:
        value = getattr(scheduler, key, None)
        if isinstance(value, int):
            params[key] = value
    return params


# ── タイムスタンプ整形 ──────────────────────────────────────────────────


def _format_iso_or_none(epoch_seconds: float) -> str | None:
    """正の epoch 秒を `YYYY-MM-DDTHH:MM:SSZ` 形式に整形。0 以下は `None`。"""
    if epoch_seconds <= 0:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))


def format_run_timestamps(
    scheduler_status: dict | None,
) -> tuple[str | None, str | None]:
    """`scheduler.get_status()` の結果から `(last_l1_iso, last_l2_iso)` を
    返す。未存在 / 0 値は `None`。
    """
    if not scheduler_status:
        return None, None
    last_l1 = _format_iso_or_none(float(scheduler_status.get("last_level1_run", 0.0) or 0.0))
    last_l2 = _format_iso_or_none(latest_level2_run(scheduler_status.get("last_level2_run")))
    return last_l1, last_l2


# ── プロンプト履歴の収集 (optimize_history 用) ──────────────────────────


def collect_prompt_history(
    prompt_mgr: SystemPromptManager | None,
    modes: list[str],
) -> dict[str, list[PromptHistoryEntry]]:
    """システムプロンプトのモード別履歴を `{mode: [entries...]}` 形式で返す。

    `prompt_mgr` が `None` の場合は空 dict。`ValueError` のモードはスキップ。
    """
    if prompt_mgr is None:
        return {}
    result: dict[str, list[PromptHistoryEntry]] = {}
    for mode in modes:
        try:
            entries = prompt_mgr.get_history(mode)
        except ValueError:
            continue
        result[mode] = [
            PromptHistoryEntry(mode=mode, version=e["version"], file=e["file"])
            for e in entries
        ]
    return result


def collect_aux_prompt_history(
    aux_prompt_mgr: AuxPromptManager | None,
    tasks: list[str],
) -> dict[str, list[PromptHistoryEntry]]:
    """補助タスクのプロンプト別履歴を `{f"aux_{task}": [entries...]}` で返す。

    `aux_prompt_mgr` が `None` の場合は空 dict。`ValueError` はスキップ。
    Pro/Free ガードは呼び出し側の責務。
    """
    if aux_prompt_mgr is None:
        return {}
    result: dict[str, list[PromptHistoryEntry]] = {}
    for task in tasks:
        try:
            entries = aux_prompt_mgr.get_history(task)
        except ValueError:
            continue
        key = f"aux_{task}"
        result[key] = [
            PromptHistoryEntry(mode=key, version=e["version"], file=e["file"])
            for e in entries
        ]
    return result
