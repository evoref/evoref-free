"""Few-shot 候補プール: 経験バッファから高品質な応答例を収集・管理する

経験バッファの成功例（高 fitness）を候補プールに蓄積し、
進化時に指示テキストと Few-shot 例の組み合わせを同時に最適化する。
候補間の多様性は文字 bi-gram コサイン類似度で保証する。

参考論文: PromptWizard (arXiv:2405.18369) の Few-shot 同時最適化
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

# FewShotExample / format_fewshot_section は EvorefLoop pillar
# (agent) 所属の純粋 util に移動済。Learn 側はここから import する。
# format_fewshot_section は他モジュール (tests / 一部呼出元) が本モジュール経由で
# import するため re-export として保持する。
from backend.free.agent.prompt_utils import (
    FewShotExample,
    format_fewshot_section,  # noqa: F401  (re-export for tests)
)
from backend.free.learning.json_state_store import JsonPayload, JsonStateStore
from backend.free.memory.types import make_fact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.memory.types import SemanticFact
    from backend.free.memory.views.learn import LearnFactView

logger = get_logger("learning.fewshot_pool")

# デフォルト設定
DEFAULT_POOL_SIZE = 50            # モード別最大プールサイズ
DEFAULT_MIN_FITNESS = 0.7         # プール追加の最低 fitness
DEFAULT_MAX_EXAMPLES = 3          # プロンプトに埋め込む最大 Few-shot 数
DEFAULT_DIVERSITY_THRESHOLD = 0.8  # コサイン類似度の上限（これ以上は重複とみなす）

# SemMem 書き戻し時の subject prefix
# ``harness.fewshot.*`` から ``learn.fewshot.*`` に移行済。owner は EvorefLearn。
LEARN_FEWSHOT_SUBJECT_PREFIX: str = "learn.fewshot."
"""SemMem 上の Few-shot ファクト subject prefix。
``learn.fewshot.<mode>.<example_id>`` の形式で書き出す。"""

DEFAULT_FEWSHOT_PREDICATE: str = "example_for"
"""Few-shot ファクトの predicate (常に固定)"""

EvolveWriteback = Literal["yaml", "semmem"]


def _char_bigrams(text: str) -> Counter:
    """テキストから文字 bi-gram の出現頻度を返す"""
    t = text.lower().strip()
    if len(t) < 2:
        return Counter({t: 1}) if t else Counter()
    return Counter(t[i:i + 2] for i in range(len(t) - 1))


def _cosine_similarity(a: Counter, b: Counter) -> float:
    """2つの Counter 間のコサイン類似度を計算する"""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    norm_a = sqrt(sum(v * v for v in a.values()))
    norm_b = sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _calc_experience_fitness(signals: dict) -> float:
    """経験のシグナルから fitness を計算する（PromptEvolver と同じ基準）"""
    score = 0.0
    if signals.get("conversation_ended", False):
        score += 1.0
    if signals.get("rephrased_query", False):
        score -= 0.5
    if signals.get("user_correction") is not None:
        score -= 0.8
    # 正規化: [-1.3, 1.0] → [0.0, 1.0]
    return max(0.0, min(1.0, (score + 1.3) / 2.3))


class FewShotPool(JsonStateStore):
    """Few-shot 候補プール

    経験バッファから高品質な応答例を収集し、多様性を維持しながら
    候補プールを管理する。進化時のソースとして使用される。
    """

    _state_logger = logger

    def __init__(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        min_fitness: float = DEFAULT_MIN_FITNESS,
        max_examples: int = DEFAULT_MAX_EXAMPLES,
        diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
        debug_logger: DebugLogger | None = None,
        *,
        learn_view: LearnFactView | None = None,
        semmem_writeback_scope: str = "global",
        evolve_writeback: EvolveWriteback = "yaml",
    ) -> None:
        """
        Args:
            pool_size: モード別最大プールサイズ (yaml モード時のみ有効。
                semmem モードではプール側 GC を停止し、SemMem 側
                ``semmem_limits.policy`` + ``gc_strategy=lowest_score`` に委譲)
            min_fitness: プール追加の最低 fitness
            max_examples: プロンプトに埋め込む最大 Few-shot 数
            diversity_threshold: コサイン類似度の上限
            debug_logger: DebugLogger (任意)
                bootstrap / SemMem 書込の経路は全て本 view 経由に一本化される。
                ``evolve_writeback="semmem"`` 時に必須。
            semmem_writeback_scope: 書き込み先 scope (``global`` または
                ``project:<id>``)
            evolve_writeback: ``"yaml"`` (従来動作) / ``"semmem"``
                (SemMem に新規 fewshot ファクトを書き込み、永続化を SemMem に
                委譲する)
        """
        self.pool_size = pool_size
        self.min_fitness = min_fitness
        self.max_examples = max_examples
        self.diversity_threshold = diversity_threshold
        self._debug_logger = debug_logger

        # LearnFactView 経由の writeback に一本化
        self._learn_view: LearnFactView | None = learn_view
        self._semmem_writeback_scope: str = semmem_writeback_scope
        self._evolve_writeback: EvolveWriteback = evolve_writeback

        # モード別のプール: mode → list[FewShotExample]
        self._pools: dict[str, list[FewShotExample]] = {}
        # キャッシュ: id → bi-gram Counter
        self._bigram_cache: dict[str, Counter] = {}

    # ── SemMem 書き戻しヘルパ ───────────────────────────

    @property
    def evolve_writeback(self) -> EvolveWriteback:
        """現在の書き戻しモード"""
        return self._evolve_writeback

    def is_semmem_writeback_active(self) -> bool:
        """SemMem 書き戻しが有効かつ LearnFactView が注入済か"""
        return (
            self._evolve_writeback == "semmem"
            and self._learn_view is not None
        )

    def set_learn_view(
        self,
        learn_view: LearnFactView | None,
        *,
        writeback_scope: str | None = None,
        evolve_writeback: EvolveWriteback | None = None,
    ) -> None:
        """LearnFactView を動的に差し替える (テスト・lifespan 後注入用)。"""
        if learn_view is not None:
            self._learn_view = learn_view
        if writeback_scope is not None:
            self._semmem_writeback_scope = writeback_scope
        if evolve_writeback is not None:
            self._evolve_writeback = evolve_writeback

    @staticmethod
    def _build_subject(mode: str, example_id: str) -> str:
        """``learn.fewshot.<mode>.<example_id>`` 形式の subject を構築する"""
        return f"{LEARN_FEWSHOT_SUBJECT_PREFIX}{mode}.{example_id}"

    @staticmethod
    def _example_to_object(example: FewShotExample) -> str:
        """FewShotExample を SemMem ファクトの object 用 JSON 文字列に変換"""
        return json.dumps(
            {
                "query": example.query,
                "response": example.response,
                "mode": example.mode,
                "fitness": float(example.fitness),
                "added_at": example.added_at,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _example_from_fact(fact: SemanticFact) -> FewShotExample | None:
        """SemMem ファクトから FewShotExample を復元する。

        破損ファクトは ``None`` を返してスキップする。
        ``id`` は subject 末尾セグメント (``learn.fewshot.<mode>.<id>``) を
        信頼する
        """
        try:
            payload = json.loads(fact.object)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        # subject 末尾セグメントを ID として採用
        subject = fact.subject
        if not subject.startswith(LEARN_FEWSHOT_SUBJECT_PREFIX):
            return None
        rest = subject[len(LEARN_FEWSHOT_SUBJECT_PREFIX):]
        # rest = "<mode>.<id>"
        if "." not in rest:
            return None
        mode_part, _, id_part = rest.partition(".")
        return FewShotExample(
            id=id_part,
            query=str(payload.get("query", "")),
            response=str(payload.get("response", "")),
            mode=str(payload.get("mode", mode_part)),
            fitness=float(payload.get("fitness", 0.0)),
            added_at=str(payload.get("added_at", "")),
        )

    def _writeback_example_fact(
        self,
        example: FewShotExample,
    ) -> SemanticFact | None:
        """新規 FewShotExample を SemMem に fewshot ファクトとして書き出す。

        LearnFactView が未注入 / writeback モードが ``yaml`` の場合は
        ``None`` を返して no-op する。``type="policy"``
        → ``type="fewshot"`` に変更 (FACT_OWNERSHIP の fewshot 所有権に整合)。
        """
        if not self.is_semmem_writeback_active():
            return None
        view = self._learn_view
        assert view is not None  # is_semmem_writeback_active で保証

        subject = self._build_subject(example.mode, example.id)
        fact_mode: str = example.mode if example.mode in ("chat", "coding") else "chat"
        new_fact = make_fact(
            subject=subject,
            predicate=DEFAULT_FEWSHOT_PREDICATE,
            object_=self._example_to_object(example),
            type="fewshot",
            scope=self._semmem_writeback_scope,
            mode_origin=fact_mode,  # type: ignore[arg-type]
            confidence=float(example.fitness),
            auto_evolved=True,
            eval_metric={"fitness": float(example.fitness)},
        )
        try:
            added = view.add_fewshot_fact(new_fact)
        except ValueError as exc:
            logger.warning(
                "fewshot_pool semmem writeback failed: subject=%s err=%s",
                subject, exc,
            )
            return None
        logger.debug(
            "fewshot_pool semmem writeback: subject=%s id=%s fitness=%.3f",
            subject, added.id, example.fitness,
        )
        return added

    def bootstrap_from_semmem(self) -> int:
        """SemMem の ``learn.fewshot.*`` ファクトから in-memory プールを再構築する

        LearnFactView 経由で active な fewshot (legacy の type="policy" も
        受容) を集め、subject 末尾の ``<mode>.<id>`` をキーにモード別プール
        へ展開する。同一 ``id`` が複数ストアにあれば最初に見つけたものを
        優先する。

        Returns:
            復元した FewShotExample 数の合計
        """
        view = self._learn_view
        if view is None:
            return 0
        seen_ids: set[str] = set()
        loaded = 0
        try:
            facts = view.search_fewshot_by_prefix(LEARN_FEWSHOT_SUBJECT_PREFIX)
        except ValueError:
            return 0
        for fact in facts:
            example = self._example_from_fact(fact)
            if example is None:
                continue
            if example.id in seen_ids:
                continue
            seen_ids.add(example.id)
            pool = self._pools.setdefault(example.mode, [])
            pool.append(example)
            loaded += 1
        if loaded:
            logger.info(
                "fewshot_pool bootstrap_from_semmem: loaded=%d pools=%s",
                loaded, {m: len(p) for m, p in self._pools.items()},
            )
        return loaded

    @property
    def total_count(self) -> int:
        """全モードの候補数合計"""
        return sum(len(pool) for pool in self._pools.values())

    def count(self, mode: str) -> int:
        """指定モードの候補数"""
        return len(self._pools.get(mode, []))

    def get_pool(self, mode: str) -> list[FewShotExample]:
        """指定モードのプール全体を返す"""
        return list(self._pools.get(mode, []))

    def _get_bigrams(self, example: FewShotExample) -> Counter:
        """例の bi-gram を取得（キャッシュ付き）"""
        if example.id not in self._bigram_cache:
            text = f"{example.query} {example.response}"
            self._bigram_cache[example.id] = _char_bigrams(text)
        return self._bigram_cache[example.id]

    def _is_diverse(self, new: FewShotExample, existing: list[FewShotExample]) -> bool:
        """新しい例が既存の例と十分に異なるか判定する"""
        if not existing:
            return True
        new_bg = self._get_bigrams(new)
        for ex in existing:
            sim = _cosine_similarity(new_bg, self._get_bigrams(ex))
            if sim >= self.diversity_threshold:
                return False
        return True

    def add_from_experiences(self, experiences: list[dict]) -> int:
        """経験バッファから高品質な例を候補プールに追加する

        Args:
            experiences: 経験バッファのエントリリスト（dict 形式）

        Returns:
            追加された候補数
        """
        added = 0
        for exp in experiences:
            signals = exp.get("signals", {})
            fitness = _calc_experience_fitness(signals)

            # fitness 閾値チェック
            if fitness < self.min_fitness:
                continue

            # 成功例のみ（訂正・言い直しがない）
            if signals.get("user_correction") is not None:
                continue
            if signals.get("rephrased_query", False):
                continue

            query = exp.get("query", "").strip()
            response = exp.get("response_summary", "").strip()
            mode = exp.get("mode", "chat")

            # クエリと応答が存在するか
            if not query or not response:
                continue

            example = FewShotExample(
                query=query,
                response=response,
                mode=mode,
                fitness=fitness,
                added_at=exp.get("timestamp", ""),
            )

            pool = self._pools.setdefault(mode, [])

            # 多様性チェック
            if not self._is_diverse(example, pool):
                continue

            pool.append(example)
            added += 1

            # SemMem 書き戻しモードでは
            # 新規 example を ``learn.fewshot.<mode>.<id>`` ファクトとして書き出す。
            self._writeback_example_fact(example)

            # プールサイズ制限: fitness が最も低い候補を除去
            # SemMem 書き戻しモードでは GC を SemMem 側
            # ``semmem_limits.policy`` + ``gc_strategy=lowest_score`` に
            # 委譲し、プール側の局所 GC は停止する。
            if (
                len(pool) > self.pool_size
                and not self.is_semmem_writeback_active()
            ):
                pool.sort(key=lambda e: e.fitness)
                removed = pool.pop(0)
                self._bigram_cache.pop(removed.id, None)

        if added:
            pool_counts = {m: len(p) for m, p in self._pools.items()}
            logger.info(
                "Added %d examples to fewshot pool (total: %s)",
                added, pool_counts,
            )
            dl = self._debug_logger
            if dl:
                dl.log_learning_cycle(cycle_num=0, data={
                    "component": "fewshot_pool",
                    "op": "add_from_experiences",
                    "input_count": len(experiences),
                    "added": added,
                    "pool_counts": pool_counts,
                })
        return added

    def accept_from_artifact(
        self,
        *,
        query: str,
        response: str,
        mode: str = "coding",
        fitness: float,
        added_at: str = "",
    ) -> FewShotExample | None:
        """ラルフループの成果物 (task.content + artifact) を Few-shot 候補として採否判定する。

        シグナルから fitness を計算するのに対し、本メソッドは SemMem の
        ``progress_marker`` (gate_passed=true) + ``task`` + ``artifact`` から
        算出された fitness を外部から受け取り、閾値・多様性・プールサイズ
        制限・SemMem 書き戻しを既存ロジックで共有する。

        Returns:
            採用された ``FewShotExample``。閾値未達 / 多様性不足時は ``None``。
        """
        query = (query or "").strip()
        response = (response or "").strip()
        if not query or not response:
            return None
        if fitness < self.min_fitness:
            return None

        example = FewShotExample(
            query=query,
            response=response,
            mode=mode,
            fitness=float(fitness),
            added_at=added_at,
        )
        pool = self._pools.setdefault(mode, [])
        if not self._is_diverse(example, pool):
            return None

        pool.append(example)
        self._writeback_example_fact(example)
        if (
            len(pool) > self.pool_size
            and not self.is_semmem_writeback_active()
        ):
            pool.sort(key=lambda e: e.fitness)
            removed = pool.pop(0)
            self._bigram_cache.pop(removed.id, None)

        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "accept_from_artifact",
                "mode": mode,
                "fitness": float(fitness),
                "example_id": example.id,
                "pool_size": len(pool),
            })
        return example

    def select(
        self,
        mode: str,
        n: int | None = None,
        seed: int | None = None,
    ) -> list[FewShotExample]:
        """指定モードから多様な候補をランダム選択する

        Args:
            mode: 対象モード
            n: 選択数（デフォルト: max_examples）
            seed: 乱数シード

        Returns:
            選択された候補リスト
        """
        n = n or self.max_examples
        pool = self._pools.get(mode, [])
        if not pool:
            return []
        if len(pool) <= n:
            return list(pool)

        rng = np.random.default_rng(seed)
        # fitness による重み付きサンプリング（高 fitness を優先）
        fitnesses = np.array([ex.fitness for ex in pool])
        weights = fitnesses / fitnesses.sum()
        indices = rng.choice(len(pool), size=min(n, len(pool)), replace=False, p=weights)
        selected = [pool[i] for i in indices]

        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "select",
                "mode": mode,
                "pool_size": len(pool),
                "requested": n,
                "selected_ids": [ex.id for ex in selected],
                "selected_fitnesses": [ex.fitness for ex in selected],
            })
        return selected

    def mutate_selection(
        self,
        current_ids: list[str],
        mode: str,
        seed: int | None = None,
    ) -> list[str]:
        """Few-shot 選択を変異させる（add/remove/swap）

        Args:
            current_ids: 現在選択されている例の ID リスト
            mode: 対象モード
            seed: 乱数シード

        Returns:
            変異後の ID リスト
        """
        pool = self._pools.get(mode, [])
        if not pool:
            return []

        pool_ids = {ex.id for ex in pool}
        # 現在の選択からプールに存在する ID のみ保持
        valid_ids = [i for i in current_ids if i in pool_ids]

        rng = np.random.default_rng(seed)
        op = rng.choice(["add", "remove", "swap"])

        available = [ex.id for ex in pool if ex.id not in set(valid_ids)]

        before_ids = list(valid_ids)

        if op == "add" and available and len(valid_ids) < self.max_examples:
            new_id = rng.choice(available)
            valid_ids.append(new_id)
        elif op == "remove" and valid_ids:
            idx = rng.integers(0, len(valid_ids))
            valid_ids.pop(idx)
        elif op == "swap" and valid_ids and available:
            idx = rng.integers(0, len(valid_ids))
            new_id = rng.choice(available)
            valid_ids[idx] = new_id

        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "mutate_selection",
                "mode": mode,
                "mutation": op,
                "before_ids": before_ids,
                "after_ids": valid_ids,
            })

        return valid_ids

    def get_by_ids(self, ids: list[str], mode: str) -> list[FewShotExample]:
        """ID リストから例を取得する（存在しない ID は無視）"""
        pool = self._pools.get(mode, [])
        id_map = {ex.id: ex for ex in pool}
        return [id_map[i] for i in ids if i in id_map]

    # ── Step 14 — Few-shot プール GC ───────────────────

    def garbage_collect(self) -> dict:
        """Few-shot プール全体に対する明示的 GC を実行する。

        sleep-time scheduler の **Step 14** から呼び出される。動作モードは
        ``evolve_writeback`` 設定で切り替わる:

        - ``yaml`` (従来動作): 各モードプールを fitness 昇順でソートし、
          ``pool_size`` を超える分を最低 fitness から除去する。除去した
          example は ``_bigram_cache`` からも追い出す。
        - ``semmem``: GC は SemMem 側 ``semmem_limits.policy``
          + ``gc_strategy=lowest_score`` に委譲されるため、本メソッドは
          in-memory プールに触れず ``delegated_to_semmem=True`` を返す
          (no-op + ログのみ)。

        Returns:
            ``{"removed_per_mode": {mode: int}, "removed_total": int,
              "remaining_per_mode": {mode: int}, "delegated_to_semmem": bool}``
        """
        if self.is_semmem_writeback_active():
            remaining = {m: len(p) for m, p in self._pools.items()}
            logger.info(
                "Step 14 fewshot GC: delegated to SemMem (semmem_limits.policy)",
            )
            dl = self._debug_logger
            if dl:
                dl.log_learning_cycle(cycle_num=0, data={
                    "component": "fewshot_pool",
                    "op": "garbage_collect",
                    "delegated_to_semmem": True,
                    "remaining_per_mode": remaining,
                })
            return {
                "removed_per_mode": {},
                "removed_total": 0,
                "remaining_per_mode": remaining,
                "delegated_to_semmem": True,
            }

        removed_per_mode: dict[str, int] = {}
        for mode, pool in self._pools.items():
            if len(pool) <= self.pool_size:
                continue
            pool.sort(key=lambda e: e.fitness)
            excess = len(pool) - self.pool_size
            for _ in range(excess):
                removed = pool.pop(0)
                self._bigram_cache.pop(removed.id, None)
            removed_per_mode[mode] = excess

        removed_total = sum(removed_per_mode.values())
        remaining = {m: len(p) for m, p in self._pools.items()}
        if removed_total:
            logger.info(
                "Step 14 fewshot GC: removed=%d per_mode=%s remaining=%s",
                removed_total, removed_per_mode, remaining,
            )
        else:
            logger.debug(
                "Step 14 fewshot GC: nothing to evict (pool_size=%d remaining=%s)",
                self.pool_size, remaining,
            )
        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "garbage_collect",
                "delegated_to_semmem": False,
                "removed_per_mode": removed_per_mode,
                "removed_total": removed_total,
                "remaining_per_mode": remaining,
            })
        return {
            "removed_per_mode": removed_per_mode,
            "removed_total": removed_total,
            "remaining_per_mode": remaining,
            "delegated_to_semmem": False,
        }

    # ── 永続化 (JsonStateStore) ──

    def _to_payload(self) -> JsonPayload:
        return {
            mode: [asdict(ex) for ex in pool]
            for mode, pool in self._pools.items()
        }

    def _from_payload(self, payload: JsonPayload) -> None:
        if not isinstance(payload, dict):
            raise TypeError(
                f"fewshot_pool.json must be a dict, got {type(payload).__name__}"
            )
        for mode, entries in payload.items():
            self._pools[mode] = [FewShotExample(**entry) for entry in entries]
        # bi-gram キャッシュをリビルド
        self._bigram_cache.clear()

    def _on_save_success(self, path: Path) -> None:
        logger.info("Fewshot pool saved: %s (%d total)", path, self.total_count)

    def _on_load_success(self, path: Path) -> None:
        logger.info(
            "Fewshot pool loaded: %s (%s)",
            path,
            {m: len(p) for m, p in self._pools.items()},
        )

    def _on_load_missing(self, path: Path) -> None:
        logger.info("Fewshot pool file not found: %s", path)
