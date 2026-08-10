"""長文生成エンジン基盤モジュール

設計書 f_08_long_form_generation.md 準拠。
CogWriter / Recurrent 両戦略で共有するデータモデル・ユーティリティ。
"""

from backend.free.generation.code_skeleton import CodeSkeleton, update_skeleton
from backend.free.generation.content_detector import detect_content_type
from backend.free.generation.models import (
    CodeUnit,
    ContentType,
    GenerationPlan,
    SectionPlan,
)
from backend.free.generation.rolling_context import RollingContext
from backend.free.generation.token_budget import TokenBudget
from backend.free.generation.strategy_common import resolve_generation_order
from backend.free.generation.strategy_cogwriter import (
    CogWriterStrategy,
    ReviewIssue,
)
from backend.free.generation.orchestrator import LongFormOrchestrator
from backend.free.generation.strategy_recurrent import RecurrentStrategy
from backend.free.generation.validators import ValidationError, validate_python

__all__ = [
    "CodeSkeleton",
    "CodeUnit",
    "CogWriterStrategy",
    "ContentType",
    "GenerationPlan",
    "LongFormOrchestrator",
    "RecurrentStrategy",
    "ReviewIssue",
    "RollingContext",
    "SectionPlan",
    "TokenBudget",
    "ValidationError",
    "detect_content_type",
    "resolve_generation_order",
    "update_skeleton",
    "validate_python",
]
