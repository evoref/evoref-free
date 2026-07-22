"""

EvorefMem 統合仕様 におけるモード別の意味記憶 / 短期記憶注入器
``SemanticFact`` と ``MemoryNote`` の集合を入力として、モード別の Tier 配分
(チャット 800 トークン / コーディング 2000 トークン) に従って LLM プロンプト
へ注入する候補列を生成する。

設計仕様:

1. **モード別予算と Tier 比率**

   ============  =============  =================================
   モード        予算 (tokens)  Tier 比率 (T1, T2, T3, T4)
   ============  =============  =================================
   chat            800          (0.40, 0.35, 0.15, 0.10)
   coding         2000          (0.40, 0.35, 0.15, 0.10)
   ============  =============  =================================

2. **Tier 1 ボーナス**: ``pinned=True`` の項目はスコアに ``+1.0`` を付与し、
   かつ Tier 1 へ強制配置する (タグ種別を問わない)。

3. **Coding Tier 1 拡張**:

   - ``policy`` ファクトは ``confidence >= 0.7`` (active) のみ採用。
   - ``failure_pattern`` ファクトは ``failure_signature`` が呼び出し側
     から提供された signature 集合と一致したものだけ Tier 1 注入。

4. **Coding Tier 2 拡張**: ``current_project_id`` と一致する ``task``
   ファクト (current task) を Tier 2 に注入する。

5. **予算オーバー時の挙動**: 各 Tier はその予算を厳密上限とする
   (``int(total_budget * ratio)``)。さらに最終的に総使用量が総予算を
   超えた場合は **Tier 4 から順に削除** する

本モジュールは外部 I/O を一切持たない純粋関数的な計画器であり、
チャット応答パスへの配線は別層で行う。本モジュールのスコープは
「ロジック実装 + ユニットテスト」までとする。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Literal

from backend.free.core.session_mode import (
    is_chat_mode,
    is_coding_mode,
    is_valid_session_mode,
)
from backend.free.memory.stores.short_term import MemoryNote
from backend.free.memory.types import MemoryMode, SemanticFact
from backend.log_config import get_logger
from backend.utils import estimate_tokens

logger = get_logger("memory.injector")

# ──────────────────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_CHAT_BUDGET_TOKENS = 800
DEFAULT_CODING_BUDGET_TOKENS = 2000
DEFAULT_TIER_RATIOS: tuple[float, float, float, float] = (0.40, 0.35, 0.15, 0.10)

#: pin ボーナス値
PINNED_BONUS = 1.0

#: policy ファクトを active と見なす最小 confidence
DEFAULT_POLICY_ACTIVATION_MIN_CONFIDENCE = 0.7

#: スコア計算で用いる recency 半減期 (日)
_RECENCY_HALF_LIFE_DAYS = 14.0


# ──────────────────────────────────────────────────────────────────────────
# 結果データクラス
# ──────────────────────────────────────────────────────────────────────────

ItemSource = Literal["fact", "note"]


@dataclass(frozen=True)
class InjectedItem:
    """注入候補 1 件 (ファクト or ノート)"""

    tier: int
    source: ItemSource
    item_id: str
    text: str
    tokens: int
    score: float


@dataclass
class InjectionPlan:
    """:class:`MemoryInjector.inject` の結果。

    :attr:`items` は Tier 1 → 4 の順、各 Tier 内はスコア降順。
    :attr:`dropped` は予算オーバーや Tier 不適合で除外された候補。
    """

    mode: MemoryMode
    budget_tokens: int
    tier_budgets: list[int]
    items: list[InjectedItem] = field(default_factory=list)
    dropped: list[InjectedItem] = field(default_factory=list)
    used_tokens: int = 0

    def render(self) -> str:
        """注入対象を 1 つのテキストに連結する (改行区切り)。"""
        return "\n".join(it.text for it in self.items)

    def by_tier(self, tier: int) -> list[InjectedItem]:
        return [it for it in self.items if it.tier == tier]


# ──────────────────────────────────────────────────────────────────────────
# MemoryInjector
# ──────────────────────────────────────────────────────────────────────────


class MemoryInjector:
    """モード別 Tier 注入器。

    config 例 (``config.yaml`` 全体をそのまま渡す想定だが、未指定でも
    デフォルト値で動作する)::

        memory:
          injection:
            chat_budget_tokens: 800
            coding_budget_tokens: 2000
            tier_ratios:
              chat: [0.40, 0.35, 0.15, 0.10]
              coding: [0.40, 0.35, 0.15, 0.10]
        learning:
          policy:
            activation_min_confidence: 0.7

    ``policy_activation_min_confidence`` は
    ``learning.policy.activation_min_confidence`` を SSOT とし、
    ``PolicyInterpreter`` / ``LoopFactView`` 等と必ず同一閾値で動作する
    (旧 ``harness:`` セクションから移行済)。
    """

    def __init__(
        self,
        config: dict | None = None,
        *,
        now_provider=None,
    ) -> None:
        cfg_root = config or {}
        cfg = ((cfg_root.get("memory") or {}).get("injection") or {})
        learning_policy_cfg = (
            (cfg_root.get("learning") or {}).get("policy") or {}
        )
        self.chat_budget: int = int(
            cfg.get("chat_budget_tokens", DEFAULT_CHAT_BUDGET_TOKENS),
        )
        self.coding_budget: int = int(
            cfg.get("coding_budget_tokens", DEFAULT_CODING_BUDGET_TOKENS),
        )
        ratios = cfg.get("tier_ratios") or {}
        self.chat_ratios = self._normalize_ratios(
            ratios.get("chat"), DEFAULT_TIER_RATIOS,
        )
        self.coding_ratios = self._normalize_ratios(
            ratios.get("coding"), DEFAULT_TIER_RATIOS,
        )
        self.policy_min_confidence: float = float(
            learning_policy_cfg.get(
                "activation_min_confidence",
                DEFAULT_POLICY_ACTIVATION_MIN_CONFIDENCE,
            ),
        )
        self._now_provider = now_provider or time.time

    # ── public API ────────────────────────────────────────────────────

    def inject(
        self,
        *,
        mode: MemoryMode,
        facts: Iterable[SemanticFact] = (),
        stm_notes: Iterable[MemoryNote] = (),
        current_project_id: str | None = None,
        failure_signatures: Iterable[str] | None = None,
    ) -> InjectionPlan:
        """注入計画を構築する。

        Args:
            mode: ``"chat"`` または ``"coding"``。
            facts: 候補 ``SemanticFact`` の集合 (global / project 混在可)。
            stm_notes: 候補 ``MemoryNote`` の集合 (Tier 2 配置)。
            current_project_id: コーディングモードで「現在プロジェクト」と
                見なすプロジェクト ID。``None`` の場合 project ファクトは
                すべて他プロジェクト扱いになる。
            failure_signatures: Tier 1 注入を許可する failure_pattern の
                ``failure_signature`` 集合 (signature 一致時のみ Tier 1)。

        Returns:
            :class:`InjectionPlan` — 採用候補 / 削除候補 / 使用トークン。
        """
        if not is_valid_session_mode(mode):
            raise ValueError(f"unsupported mode: {mode}")

        sigs: set[str] = set(failure_signatures or ())
        budget = self.chat_budget if is_chat_mode(mode) else self.coding_budget
        ratios = self.chat_ratios if is_chat_mode(mode) else self.coding_ratios
        tier_budgets = self._tier_budgets(budget, ratios)

        # Tier ごとに分類
        buckets: dict[int, list[InjectedItem]] = {1: [], 2: [], 3: [], 4: []}

        for fact in facts:
            if fact.superseded_by:
                continue
            tier = self._classify_fact(
                fact, mode, current_project_id, sigs,
            )
            if tier is None:
                continue
            score = self._score_fact(fact)
            text = self._render_fact(fact)
            tokens = estimate_tokens(text)
            buckets[tier].append(
                InjectedItem(
                    tier=tier,
                    source="fact",
                    item_id=fact.id,
                    text=text,
                    tokens=tokens,
                    score=score,
                ),
            )

        for note in stm_notes:
            tier = self._classify_note(note, mode, current_project_id)
            if tier is None:
                continue
            score = self._score_note(note)
            text = self._render_note(note)
            tokens = estimate_tokens(text)
            buckets[tier].append(
                InjectedItem(
                    tier=tier,
                    source="note",
                    item_id=note.id,
                    text=text,
                    tokens=tokens,
                    score=score,
                ),
            )

        plan = InjectionPlan(
            mode=mode,
            budget_tokens=budget,
            tier_budgets=tier_budgets,
        )

        # Tier 1 → 4 の順にパック
        for tier in (1, 2, 3, 4):
            cap = tier_budgets[tier - 1]
            accepted, dropped = self._pack_tier(buckets[tier], cap)
            plan.items.extend(accepted)
            plan.dropped.extend(dropped)
            plan.used_tokens += sum(it.tokens for it in accepted)

        # 総予算オーバー時は Tier 4 から削除
        if plan.used_tokens > budget:
            self._spill_from_tier4(plan, budget)

        logger.debug(
            "MemoryInjector.inject: mode=%s budget=%d used=%d items=%d "
            "dropped=%d project=%s sigs=%d",
            mode, budget, plan.used_tokens, len(plan.items),
            len(plan.dropped), current_project_id, len(sigs),
        )
        return plan

    # ── tier 分類 ────────────────────────────────────────────────────

    def _classify_fact(
        self,
        fact: SemanticFact,
        mode: MemoryMode,
        current_project_id: str | None,
        failure_signatures: set[str],
    ) -> int | None:
        """ファクトの Tier を返す。配置対象外なら ``None``。"""
        if fact.private:
            # private ファクトは注入対象外
            return None
        if fact.pinned:
            # pinned は常に Tier 1 に強制配置
            return 1

        is_current_project = (
            current_project_id is not None
            and fact.scope == f"project:{current_project_id}"
        )
        is_other_project = fact.is_project_scoped() and not is_current_project
        t = fact.type

        if is_chat_mode(mode):
            if t in ("personal_fact", "preference", "emotion"):
                return 1
            if t in ("decision", "commitment"):
                return 2
            if t in ("belief", "opinion"):
                return 3
            if t in ("world_fact", "project"):
                return 4
            return None

        # coding mode
        if t == "policy":
            if fact.confidence >= self.policy_min_confidence:
                return 1
            return None
        if t == "failure_pattern":
            if (
                fact.failure_signature is not None
                and fact.failure_signature in failure_signatures
            ):
                return 1
            return None
        if t == "project":
            return 1 if is_current_project else 4
        if t == "decision":
            if is_current_project:
                return 1
            if is_other_project:
                return 3
            # global decision
            return 1
        if t == "commitment":
            return 1
        if t == "task":
            return 2 if is_current_project else None
        if t == "coding":
            return 2 if is_current_project else 4
        if t == "preference":
            return 3
        if t == "personal_fact":
            return 4
        if t == "world_fact":
            return 4
        if t == "model":
            return 4
        # progress_marker などは別層で扱う
        return None

    def _classify_note(
        self,
        note: MemoryNote,
        mode: MemoryMode,
        current_project_id: str | None,
    ) -> int | None:
        """STM ノートの Tier を返す。"""
        if note.private:
            return None
        if note.is_tool_output:
            # ツール出力は WM までで止める
            return None
        if note.pin_flag:
            return 1
        # モード不一致は対象外
        if note.mode != mode:
            return None
        if is_coding_mode(mode) and current_project_id is not None:
            if note.project_id != current_project_id:
                return None
        return 2

    # ── スコアリング ─────────────────────────────────────────────────

    def _score_fact(self, fact: SemanticFact) -> float:
        """ファクトのスコア (高いほど優先)。

        ``confidence`` を基準とし、``access_count`` の対数増分と recency
        による減衰を加える。Tier 1 配置時に pinned ボーナスを足す。
        """
        base = float(fact.confidence)
        base += 0.2 * math.log1p(max(0, fact.access_count))
        base += self._recency_term(fact.accessed_at)
        if fact.pinned:
            base += PINNED_BONUS
        return base

    def _score_note(self, note: MemoryNote) -> float:
        base = float(note.confidence)
        base += 0.2 * math.log1p(max(0, note.access_count))
        base += self._recency_term(note.accessed_at)
        if note.pin_flag:
            base += PINNED_BONUS
        return base

    def _recency_term(self, accessed_at: float) -> float:
        if accessed_at <= 0:
            return 0.0
        age_days = max(0.0, (self._now_provider() - accessed_at) / 86400.0)
        # 半減期に基づく指数減衰 (0..1)
        return math.exp(-age_days * math.log(2) / _RECENCY_HALF_LIFE_DAYS)

    # ── レンダリング ─────────────────────────────────────────────────

    def _render_fact(self, fact: SemanticFact) -> str:
        return f"- ({fact.type}) {fact.subject} {fact.predicate}: {fact.object}"

    def _render_note(self, note: MemoryNote) -> str:
        return f"- (note) {note.content}"

    # ── パッキング ───────────────────────────────────────────────────

    def _pack_tier(
        self,
        items: list[InjectedItem],
        cap: int,
    ) -> tuple[list[InjectedItem], list[InjectedItem]]:
        """1 Tier 分の greedy パッキング。

        スコア降順に詰め、cap 超過分は dropped に回す。
        """
        items.sort(key=lambda it: it.score, reverse=True)
        accepted: list[InjectedItem] = []
        dropped: list[InjectedItem] = []
        used = 0
        for it in items:
            if it.tokens <= 0:
                # 空テキスト等は無視
                continue
            if used + it.tokens <= cap:
                accepted.append(it)
                used += it.tokens
            else:
                dropped.append(it)
        return accepted, dropped

    def _spill_from_tier4(self, plan: InjectionPlan, budget: int) -> None:
        """総予算超過時に Tier 4 から低スコア順に削除する。"""
        # Tier 4 をスコア昇順に並べ、削除対象を決定
        tier4 = [it for it in plan.items if it.tier == 4]
        tier4.sort(key=lambda it: it.score)
        removed_ids: set[str] = set()
        for it in tier4:
            if plan.used_tokens <= budget:
                break
            removed_ids.add(it.item_id)
            plan.used_tokens -= it.tokens
            plan.dropped.append(it)
        if removed_ids:
            plan.items = [
                it for it in plan.items
                if not (it.tier == 4 and it.item_id in removed_ids)
            ]

    # ── 補助 ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_ratios(
        raw: list | tuple | None,
        default: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if not raw or len(raw) != 4:
            return default
        try:
            r = tuple(float(x) for x in raw)
        except (TypeError, ValueError):
            return default
        return r  # type: ignore[return-value]

    @staticmethod
    def _tier_budgets(total: int, ratios: tuple[float, ...]) -> list[int]:
        """各 Tier の token cap を整数で返す。

        合計が total を超えないよう ``int()`` で切り捨てる。
        """
        return [max(0, int(total * r)) for r in ratios]
