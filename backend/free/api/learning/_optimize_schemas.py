"""`/api/optimize` のレスポンス Pydantic スキーマ

`backend.free.api.learning.optimize` から Pydantic スキーマを切り出した型モジュール。
循環 import 防止のため、`_optimize_collectors` (helper) と `optimize.py`
(handler) の双方から参照される型はここに集約する。

`optimize.py` は本モジュールから schemas を import + re-export し、外部から
`from backend.free.api.learning.optimize import PromptModeStatus` の互換性を維持する。
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


class Level2Status(BaseModel):
    """Level 2 (SPSA LoRA 微調整) の状態"""
    spsa_iterations: int
    sparse_params: int
    min_failures: int
    lora_adapter_exists: bool
    # #3a: 既定 (base=lora) は no-op でトリガ段階 skip される。UI が
    # 「設定済・待機中」と誤解しないよう、有効メソッドと実行可否を明示する。
    base_method: str = "lora"          # lora | cvector
    active_method: str = "none"        # cvector | spsa-real-eval | none
    will_run: bool = False             # 実メソッドが有効か (no-op skip でないか)


class OptimizeStatusResponse(BaseModel):
    """最適化状態レスポンス"""
    running: bool
    last_level1_run: str | None = None
    last_level2_run: str | None = None
    level1: Level1Status
    level2: Level2Status


class OptimizeTriggerRequest(BaseModel):
    """最適化トリガーリクエスト"""
    level: str  # "level1" | "level2"
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


class LoRAHistoryEntry(BaseModel):
    """LoRA バージョン履歴エントリ"""
    type: str = "lora_version"
    version: int
    eval_score: float
    created_at: str
    metadata: dict = Field(default_factory=dict)


class OptimizeHistoryResponse(BaseModel):
    """最適化履歴レスポンス"""
    prompt_history: dict[str, list[PromptHistoryEntry]] = Field(default_factory=dict)
    lora_history: list[LoRAHistoryEntry] = Field(default_factory=list)
