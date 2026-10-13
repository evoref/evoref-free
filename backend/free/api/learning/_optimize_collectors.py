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

from typing import TYPE_CHECKING

from backend.free.api.learning._optimize_schemas import (
    PromptHistoryEntry,
    PromptModeStatus,
)
from backend.utils import epoch_to_utc

if TYPE_CHECKING:
    from backend.free.agent.prompt_manager import SystemPromptManager


# ── スケジューラ既定値 (元: optimize_status 内のリテラル) ─────────────────


SCHEDULER_DEFAULTS: dict[str, int] = {
    "generations": 10,
    "population_size": 5,
    "min_experiences": 20,
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


# ── スケジューラ設定値の既定値マージ ─────────────────────────────────────


def extract_scheduler_params(scheduler: object | None) -> dict[str, int]:
    """`LearningScheduler` から Level 1 設定値を取り出し、未初期化時は
    既定値で埋めた dict を返す。
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
    """正の epoch 秒を永続化と同じ ISO 8601 (μs, ``Z``) に整形。0 以下は `None`。"""
    return epoch_to_utc(epoch_seconds)


def format_last_level1_run(scheduler_status: dict | None) -> str | None:
    """`scheduler.get_status()` の結果から Level 1 の最終実行時刻 (ISO) を返す。
    未存在 / 0 値は `None`。
    """
    if not scheduler_status:
        return None
    return _format_iso_or_none(float(scheduler_status.get("last_level1_run", 0.0) or 0.0))


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
