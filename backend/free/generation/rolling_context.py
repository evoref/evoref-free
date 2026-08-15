"""RollingContext — セクション間の文脈受け渡し

設計書 f_08_long_form_generation.md §3.4 準拠。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.free.generation.code_skeleton import CodeSkeleton
from backend.free.generation.models import GenerationPlan
from backend.free.generation.token_budget import TokenBudget


@dataclass
class RollingContext:
    """セクション間で受け渡す文脈情報"""

    plan: GenerationPlan
    budget: TokenBudget
    current_unit_idx: int = 0
    generated_units: list[str] = field(default_factory=list)

    # コード用: スケルトン（ルールベース更新）
    skeleton: CodeSkeleton | None = None

    # テキスト用: ローリングサマリ（LLM更新 or 補助タスク更新）
    long_term_summary: str = ""

    # 共通: 直前ユニットの末尾テキスト
    short_term: str = ""

    # 現ユニット向けに選抜した RAG コンテキスト (config ``long_form.rag_per_unit``)。
    # 空文字なら unit プロンプトの参考情報スロットは「(なし)」になる。
    unit_rag: str = ""

    # 既存テキスト参照モード: 追記・修正・加筆など既存ファイルを踏まえた生成
    has_existing_context: bool = False
