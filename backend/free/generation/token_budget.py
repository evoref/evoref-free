"""トークン予算の動的配分

設計書 f_08_long_form_generation.md §4 準拠。
コンテキストサイズ × 比率で各スロットの予算を算出する。
比率テーブルは local/prompts/token_budget.json に保存し、
Level 1 進化させる
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from backend.exceptions import InsufficientContextError
from backend.free.generation.models import ContentType
from backend.utils import estimate_tokens

logger = logging.getLogger("backend.free.generation.token_budget")

# ── デフォルト比率テーブル ──
# 各スロット: [比率, 最低保証トークン数]
DEFAULT_RATIOS: dict[str, dict[str, list[float | int]]] = {
    "cogwriter_code": {
        "system_prompt": [0.06, 128],
        "plan_overview": [0.06, 128],
        "skeleton_or_summary": [0.18, 256],
        "short_term": [0.10, 256],
        "unit_spec": [0.06, 64],
        "rag_chunks": [0.06, 128],
    },
    "cogwriter_text": {
        "system_prompt": [0.06, 128],
        "plan_overview": [0.08, 128],
        "skeleton_or_summary": [0.0, 0],
        "short_term": [0.12, 256],
        "unit_spec": [0.04, 64],
        "rag_chunks": [0.10, 128],
    },
    "recurrent_code": {
        "system_prompt": [0.06, 128],
        "plan_overview": [0.06, 128],
        "skeleton_or_summary": [0.20, 256],
        "short_term": [0.10, 256],
        "unit_spec": [0.06, 64],
        "rag_chunks": [0.06, 128],
    },
    "recurrent_text": {
        "system_prompt": [0.06, 128],
        "plan_overview": [0.08, 128],
        "skeleton_or_summary": [0.08, 128],
        "short_term": [0.12, 256],
        "unit_spec": [0.04, 64],
        "rag_chunks": [0.08, 128],
    },
}

# generation 以外のスロット名（フィールド順）
_SLOT_NAMES = [
    "system_prompt",
    "plan_overview",
    "skeleton_or_summary",
    "short_term",
    "unit_spec",
    "rag_chunks",
]


def truncate_head(text: str, token_limit: int) -> str:
    """先頭を優先保持（末尾を切り詰め）"""
    if not text or estimate_tokens(text) <= token_limit:
        return text
    # 文字数ベースで近似的に切り詰め
    # estimate_tokens: CJK=1tok, ASCII=0.25tok → 平均的に1トークン≈2文字と仮定
    char_limit = max(token_limit * 2, 1)
    while char_limit > 0 and estimate_tokens(text[:char_limit]) > token_limit:
        char_limit = int(char_limit * 0.8)
    return text[:char_limit]


def truncate_tail(text: str, token_limit: int) -> str:
    """末尾を優先保持（先頭を切り詰め）"""
    if not text or estimate_tokens(text) <= token_limit:
        return text
    char_limit = max(token_limit * 2, 1)
    while char_limit > 0 and estimate_tokens(text[-char_limit:]) > token_limit:
        char_limit = int(char_limit * 0.8)
    return text[-char_limit:]


def load_ratios(
    prompts_dir: Path | None = None,
) -> dict[str, dict[str, list[float | int]]]:
    """local/prompts/token_budget.json から比率テーブルを読み込む

    ファイルが存在しない場合はデフォルトを返す。
    """
    if prompts_dir is None:
        return DEFAULT_RATIOS.copy()

    path = prompts_dir / "token_budget.json"
    if not path.exists():
        return DEFAULT_RATIOS.copy()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("ratios", DEFAULT_RATIOS.copy())
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed to load token_budget.json, using defaults")
        return DEFAULT_RATIOS.copy()


def save_ratios(
    ratios: dict[str, dict[str, list[float | int]]],
    prompts_dir: Path,
) -> None:
    """local/prompts/token_budget.json に比率テーブルを保存"""
    from backend.utils import utc_now

    prompts_dir.mkdir(parents=True, exist_ok=True)
    path = prompts_dir / "token_budget.json"
    data = {
        "version": 1,
        "updated_at": utc_now(),
        "source": "learned" if path.exists() else "default",
        "ratios": ratios,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_ratios(
    content_type: ContentType,
    strategy: Literal["cogwriter", "recurrent"],
    ratio_override: dict[str, float] | None,
    prompts_dir: Path | None,
) -> dict[str, list[float | int]]:
    """コンテンツ種別・戦略からスロット別の [比率, 最低保証] を解決"""
    all_ratios = load_ratios(prompts_dir)
    key = f"{strategy}_{content_type.value}"
    ratios = all_ratios.get(key, DEFAULT_RATIOS[key])

    if ratio_override:
        for slot, value in ratio_override.items():
            if slot in ratios:
                ratios[slot] = [value, ratios[slot][1]]

    return ratios


@dataclass
class TokenBudget:
    """コンテキストサイズから動的に算出されるトークン予算"""

    context_size: int
    system_prompt: int
    plan_overview: int
    skeleton_or_summary: int
    short_term: int
    unit_spec: int
    rag_chunks: int
    generation: int

    @classmethod
    def from_context_size(
        cls,
        context_size: int,
        content_type: ContentType,
        strategy: Literal["cogwriter", "recurrent"],
        ratio_override: dict[str, float] | None = None,
        prompts_dir: Path | None = None,
    ) -> TokenBudget:
        """コンテキストサイズ・コンテンツ種別・戦略から予算を自動算出"""
        ratios = _resolve_ratios(content_type, strategy, ratio_override, prompts_dir)

        budget: dict[str, int] = {}
        used = 0
        for slot in _SLOT_NAMES:
            ratio, minimum = ratios[slot]
            value = max(int(context_size * ratio), int(minimum))
            budget[slot] = value
            used += value

        budget["generation"] = context_size - used

        return cls(context_size=context_size, **budget)

    def adjust_for_small_context(self) -> None:
        """generation < 512 の場合に機能を段階的に削減"""
        # Stage 1: RAG を無効化
        if self.generation < 512 and self.rag_chunks > 0:
            self.generation += self.rag_chunks
            self.rag_chunks = 0

        # Stage 2: スケルトン/要約を縮小（最低保証の半分まで）
        if self.generation < 512 and self.skeleton_or_summary > 128:
            freed = self.skeleton_or_summary - 128
            self.generation += freed
            self.skeleton_or_summary = 128

        # Stage 3: 生成不可
        if self.generation < 256:
            raise InsufficientContextError(
                f"context_size={self.context_size} is too small for long-form "
                f"generation. Minimum recommended: 2048"
            )

    def fit_content(self, slot: str, text: str) -> str:
        """スロットの予算内にテキストを収める"""
        tokens = estimate_tokens(text)
        limit = getattr(self, slot)
        if tokens <= limit:
            return text
        match slot:
            case "short_term":
                return truncate_tail(text, limit)
            case "skeleton_or_summary" | "plan_overview":
                return truncate_head(text, limit)
            case _:
                return truncate_head(text, limit)
