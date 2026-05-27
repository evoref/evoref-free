"""LearnedPattern dataclass の単独モジュール

`LearnedPatternStore` (ドメイン) と `LearnedPatternRepository` (infra) の
両方から参照される dataclass を切り出すことで循環 import を防ぐ。
"""

from __future__ import annotations

from dataclasses import dataclass

# NOTE: 既定値は `learned_patterns` モジュールのデフォルトと一致させること。
_DEFAULT_INITIAL_WEIGHT = 0.5


@dataclass
class LearnedPattern:
    """学習済みパターンの1エントリ"""
    keyword: str
    category: str = "correction"  # "correction" | "intent" | "rephrase" | "tool_routing"
    weight: float = _DEFAULT_INITIAL_WEIGHT
    hit_count: int = 0
    source_count: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_hit: float = 0.0
