"""トークン予算の動的配分

設計書 f_08_long_form_generation.md §3.3.1 準拠。
コンテキストサイズ × 比率で各スロットの予算を算出する。
比率テーブルはプロンプトディレクトリの token_budget.json に保存し、
Level 1 進化させる
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.exceptions import InsufficientContextError
from backend.free.generation.models import ContentType
from backend.io.codec import codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedPayloadFile
from backend.utils import estimate_tokens

logger = logging.getLogger("backend.free.generation.token_budget")


@persisted()
@dataclass
class SlotRatios:
    """1 つの ``<strategy>_<content_type>`` の比率表 (スロット → ``[比率, 最低保証]``)。

    スロットの追加は任意フィールドの追加 (同じ版)。この版が知らないスロットは
    ``_extra`` に残り、書き戻しで元の位置へ戻る (予算には使わない)。
    """

    system_prompt: list[Any]
    skeleton_or_summary: list[Any]
    short_term: list[Any]
    unit_spec: list[Any]
    rag_chunks: list[Any]
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class TokenBudgetFile:
    """``token_budget.json`` のペイロード。"""

    updated_at: str = ""
    source: str = "learned"
    ratios: dict[str, SlotRatios] = field(default_factory=dict)
    _extra: dict[str, Any] | None = None


#: ``token_budget.json`` の形式。ペイロードは :class:`TokenBudgetFile`。
TOKEN_BUDGET_FORMAT = register_format(FormatSpec(
    format_id="learning.token_budget",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/prompts/token_budget.json",
    retention="one per partition",
    export=True,
    records=(TokenBudgetFile,),
))


def _ratios_file(prompts_dir: Path) -> VersionedPayloadFile:
    """ペイロードを :class:`TokenBudgetFile` で読み書きする (型の合わないファイルは退避)。"""
    codec = codec_for(TokenBudgetFile)
    return VersionedPayloadFile(
        TOKEN_BUDGET_FORMAT, prompts_dir / "token_budget.json",
        component="token_budget", state_logger=logger,
        decode=codec.decode, encode=codec.encode,
    )

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


def read_budget_file(prompts_dir: Path) -> TokenBudgetFile | None:
    """``<prompts_dir>/token_budget.json`` を読む。無い / 読めないなら ``None``。"""
    f = _ratios_file(prompts_dir)
    return f.payload if f.load() else None


def write_budget_file(budget: TokenBudgetFile, prompts_dir: Path) -> None:
    """``budget`` を ``<prompts_dir>/token_budget.json`` へ書く。

    ディスク上のファイルが G1 の封筒でない / 版が新しいなら上書きしない (WARNING)。
    書き込みの失敗は送出する。
    """
    prompts_dir.mkdir(parents=True, exist_ok=True)
    f = _ratios_file(prompts_dir)
    f.RAISE_ON_SAVE_ERROR = True
    f.load()
    f.payload = budget
    f.save()


def load_ratios(
    prompts_dir: Path | None = None,
) -> dict[str, dict[str, list[float | int]]]:
    """``<prompts_dir>/token_budget.json`` から比率テーブルを読み込む

    ファイルが存在しない / 読めない (G1 の封筒でない・版が新しい・壊れている・型が
    合わない) 場合はデフォルトを返す。値は ``{key: {slot: [比率, 最低保証]}}`` で、
    この版が知らないスロットもそのまま載る (予算の計算は既知のスロットだけを引く)。
    """
    if prompts_dir is None:
        return DEFAULT_RATIOS.copy()
    budget = read_budget_file(prompts_dir)
    if budget is None:
        return DEFAULT_RATIOS.copy()
    slots = codec_for(SlotRatios)
    return {key: slots.encode(row) for key, row in budget.ratios.items()}


def save_ratios(
    ratios: dict[str, dict[str, list[float | int]]],
    prompts_dir: Path,
    *,
    source: str = "learned",
) -> None:
    """``<prompts_dir>/token_budget.json`` に比率テーブルを保存

    ディスク上のファイルが G1 の封筒でない / 版が新しいなら上書きしない (WARNING)。
    読めたファイルのトップの未知キーは残す。書き込みの失敗は従来どおり送出する。
    """
    from backend.utils import utc_now

    current = read_budget_file(prompts_dir)
    slots = codec_for(SlotRatios)
    write_budget_file(TokenBudgetFile(
        updated_at=utc_now(),
        # 呼び手が何を書いているかを記録する。以前は path.exists() から
        # 決めており、2 回目以降は既定値を書き戻しても "learned" になった。
        source=source,
        ratios={key: slots.decode(row) for key, row in ratios.items()},
        _extra=current._extra if current is not None else None,
    ), prompts_dir)


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
