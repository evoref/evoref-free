"""トークン予算の動的配分

設計書 f_08_long_form_generation.md §3.3.1 準拠。
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
from backend.io import atomic_write_text
from backend.utils import estimate_tokens

logger = logging.getLogger("backend.free.generation.token_budget")

# ── デフォルト比率テーブル ──
# 各スロット: [比率, 最低保証トークン数]
#
# ``plan_overview`` と leftover の ``generation`` は 2026-09-18 (Phase 1) に
# 撤去した (f_08 §3.3.1) — どちらも消費側がゼロで、窓の一部を予約するだけの
# 死んだ枠だった (実際の unit 生成上限は ``long_form.unit_max_tokens`` が別途
# 持つ)。旧 ``plan_overview`` の比率はそのぶん ``skeleton_or_summary`` /
# ``short_term`` / ``rag_chunks`` へ均等に近い形で再配分し、4 組とも合計
# (旧 6 スロットの比率合計 = 新 5 スロットの比率合計) は不変にしてある。
DEFAULT_RATIOS: dict[str, dict[str, list[float | int]]] = {
    "cogwriter_code": {
        "system_prompt": [0.06, 128],
        "skeleton_or_summary": [0.20, 256],
        "short_term": [0.12, 256],
        "unit_spec": [0.06, 64],
        "rag_chunks": [0.08, 128],
    },
    "cogwriter_text": {
        "system_prompt": [0.06, 128],
        # TextSkeleton (f_08 §3.3.1、2026-09-18) の予算。以前は 0.0 で TEXT は
        # 直前 1000 字の窓しか持たず前半の固有値を忘れていた。
        "skeleton_or_summary": [0.08, 200],
        "short_term": [0.15, 256],
        "unit_spec": [0.04, 64],
        "rag_chunks": [0.12, 128],
    },
    "recurrent_code": {
        "system_prompt": [0.06, 128],
        "skeleton_or_summary": [0.22, 256],
        "short_term": [0.12, 256],
        "unit_spec": [0.06, 64],
        "rag_chunks": [0.08, 128],
    },
    "recurrent_text": {
        "system_prompt": [0.06, 128],
        "skeleton_or_summary": [0.11, 128],
        "short_term": [0.15, 256],
        "unit_spec": [0.04, 64],
        "rag_chunks": [0.10, 128],
    },
}

# TokenBudget が持つスロット名（フィールド順）
_SLOT_NAMES = [
    "system_prompt",
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
        ratios = data.get("ratios", DEFAULT_RATIOS.copy())
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed to load token_budget.json, using defaults")
        return DEFAULT_RATIOS.copy()
    return _drop_unknown_slots(ratios)


def _drop_unknown_slots(
    ratios: dict[str, dict[str, list[float | int]]],
) -> dict[str, dict[str, list[float | int]]]:
    """撤去済みスロット名 (``plan_overview`` 等) が残る比率テーブルを無害化する。

    ``token_budget.json`` は Level 1 進化対象で、撤去済みスロット名を含んだ
    旧テーブルが残っている可能性がある。未知スロット名は無視して 1 行
    WARNING を出す。既知スロットを揃えられない ``(content_type, strategy)``
    キーはそのエントリごと落とし、``_resolve_ratios`` の
    ``DEFAULT_RATIOS[key]`` フォールバックへ安全に縮退させる
    (部分的に古いキーだけを残すと ``from_context_size`` が KeyError になる)。
    """
    unknown: set[str] = set()
    cleaned: dict[str, dict[str, list[float | int]]] = {}
    for key, slots in ratios.items():
        kept = {slot: value for slot, value in slots.items() if slot in _SLOT_NAMES}
        unknown |= set(slots) - set(kept)
        if set(kept) == set(_SLOT_NAMES):
            cleaned[key] = kept
    if unknown:
        logger.warning(
            "token_budget.json contains unknown slot name(s), ignoring: %s",
            sorted(unknown),
        )
    return cleaned


def save_ratios(
    ratios: dict[str, dict[str, list[float | int]]],
    prompts_dir: Path,
    *,
    source: str = "learned",
) -> None:
    """local/prompts/token_budget.json に比率テーブルを保存"""
    from backend.utils import utc_now

    prompts_dir.mkdir(parents=True, exist_ok=True)
    path = prompts_dir / "token_budget.json"
    data = {
        "version": 1,
        "updated_at": utc_now(),
        # 呼び手が何を書いているかを記録する。以前は path.exists() から
        # 決めており、2 回目以降は既定値を書き戻しても "learned" になった。
        "source": source,
        "ratios": ratios,
    }
    atomic_write_text(
        path, json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
    )


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
        # ``load_ratios(None)`` は ``DEFAULT_RATIOS.copy()`` (shallow) を返す
        # ため、``ratios`` はネストした辞書としては ``DEFAULT_RATIOS[key]`` と
        # 同一オブジェクトになりうる。呼出元の共有辞書を書き換えないよう、
        # 上書き前に浅いコピーを 1 段作る。
        ratios = dict(ratios)
        for slot, value in ratio_override.items():
            if slot in ratios:
                ratios[slot] = [value, ratios[slot][1]]

    return ratios


@dataclass
class TokenBudget:
    """コンテキストサイズから動的に算出されるトークン予算

    2026-09-18 (Phase 1) に ``plan_overview`` / ``generation`` (leftover) を
    撤去した (f_08 §3.3.1)。生成そのものの上限は ``long_form.unit_max_tokens``
    が別途持つため、本クラスは実際にプロンプトへ注入する 5 スロットの予算
    のみを持つ。
    """

    context_size: int
    system_prompt: int
    skeleton_or_summary: int
    short_term: int
    unit_spec: int
    rag_chunks: int

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
        for slot in _SLOT_NAMES:
            ratio, minimum = ratios[slot]
            budget[slot] = max(int(context_size * ratio), int(minimum))

        return cls(context_size=context_size, **budget)

    def _slots_total(self) -> int:
        """5 スロットの合計 (プロンプトに実際に載る予約分)。"""
        return (
            self.system_prompt + self.skeleton_or_summary + self.short_term
            + self.unit_spec + self.rag_chunks
        )

    def adjust_for_small_context(self) -> None:
        """生成に残る余白 (``context_size - 5 スロット合計``) が乏しい場合に
        機能を段階的に削減する。

        以前は leftover を ``generation`` フィールドに貯めて直接書き換えて
        いたが、``generation`` は撤去済み (docstring 参照) のため、余白は
        都度 ``context_size - _slots_total()`` で計算し直す。判定の閾値
        (512 / 256) は従来と同じ。
        """
        # Stage 1: RAG を無効化
        if self.context_size - self._slots_total() < 512 and self.rag_chunks > 0:
            self.rag_chunks = 0

        # Stage 2: スケルトン/要約を縮小（最低保証の半分まで）
        if (
            self.context_size - self._slots_total() < 512
            and self.skeleton_or_summary > 128
        ):
            self.skeleton_or_summary = 128

        # Stage 3: 生成不可
        if self.context_size - self._slots_total() < 256:
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
            case _:
                return truncate_head(text, limit)
