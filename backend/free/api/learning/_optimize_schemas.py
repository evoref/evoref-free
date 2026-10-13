"""`/api/optimize` のレスポンス Pydantic スキーマ

`backend.free.api.learning.optimize` から Pydantic スキーマを切り出した型モジュール。
循環 import 防止のため、`_optimize_collectors` (helper) と `optimize.py`
(handler) の双方から参照される型はここに集約する。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PromptModeStatus(BaseModel):
    """モード別プロンプト最適化状態"""
    mode: str
    version: int
    source: str  # "default" | "manual" | "evolution"
    updated_at: str
    model_calibrated_for: str = ""


class AuxTaskStatus(BaseModel):
    """補助タスクタスク別プロンプト最適化状態"""
    task: str
    version: int
    source: str  # "default" | "manual" | "evolution"
    updated_at: str
    fitness_score: float = 0.0


class Level1Status(BaseModel):
    """Level 1 (プロンプト進化) の状態"""
    modes: list[PromptModeStatus] = Field(default_factory=list)
    aux_tasks: list[AuxTaskStatus] = Field(default_factory=list)
    generations: int
    population_size: int
    min_experiences: int


class OptimizeStatusResponse(BaseModel):
    """最適化状態レスポンス"""
    running: bool
    last_level1_run: str | None = None
    level1: Level1Status


class OptimizeTriggerRequest(BaseModel):
    """最適化トリガーリクエスト"""
    level: str  # "level1"
    mode: str | None = None  # Level 1 対象モード (省略時は全モード)


class OptimizeTriggerResponse(BaseModel):
    """最適化トリガーレスポンス"""
    triggered: bool
    level: str
    mode: str | None = None
    message: str


class PromptHistoryEntry(BaseModel):
    """プロンプト進化履歴エントリ"""
    type: str = "prompt_evolution"
    mode: str
    version: int
    file: str


class OptimizeHistoryResponse(BaseModel):
    """最適化履歴レスポンス"""
    prompt_history: dict[str, list[PromptHistoryEntry]] = Field(default_factory=dict)
