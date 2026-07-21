"""Few-shot 候補プール: 経験バッファから高品質な応答例を収集・管理する

経験バッファの成功例（高 fitness）を候補プールに蓄積し、
進化時に指示テキストと Few-shot 例の組み合わせを同時に最適化する。
候補間の多様性は文字 bi-gram コサイン類似度で保証する。

参考論文: PromptWizard (arXiv:2405.18369) の Few-shot 同時最適化
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, fields
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

# _calc_experience_fitness の生スコア理論域 (係数の総和)。
# min = -(rephrase 0.5 + correction 0.8 + loops 0.3 + verr 0.3) = -1.9
# max = (ended 1.0 + rag 0.3 + long_form 完了 0.3 + lf_success 0.1) = +1.7
# 係数を変えたらこの 2 定数も更新する。
_FITNESS_LO = -1.9
_FITNESS_HI = 1.7

# select_top_k のスコア合成: query 適合を主項に fitness を従に。
_TOPK_SIM_WEIGHT = 0.7
# select_top_k で類似計算する候補数の上限 (fitness 上位で足切りし hot path 遅延を抑制)。
_TOPK_SELECT_CAP = 64
# select_top_k の合成スコア下限。これ未満の候補はタスク無関連とみなして
# 返さない (2026-07-15: 第 3 スロットに 0.28-0.33 帯の無関連例 (W杯話題等) が
# 毎ターン混入していた)。
_TOPK_MIN_SCORE = 0.40

# タスク進捗ノート行 (エージェントの最終応答フォーマット)。
# meta_cognitive_utils._TASK_LOG_LINE_RE と同旨だが、pillar 境界
# (EvorefLearn → EvorefLoop の utils は import 対象外) のため最小実装を持つ。
_TASK_LOG_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\[(?:done|failed|skipped)\]\s"
    r"|^\s*Written\s+\d+\s+bytes\s+to\s+\S",
)


def _response_is_task_log_only(response: str) -> bool:
    """応答がタスク進捗ノート行だけで構成されるかを判定する。"""
    lines = [ln for ln in response.split("\n") if ln.strip()]
    if not lines:
        return False
    return all(_TASK_LOG_LINE_RE.match(ln) for ln in lines)

# SemMem 書き戻し時の subject prefix
# ``harness.fewshot.*`` から ``learn.fewshot.*`` に移行済。owner は EvorefLearn。
LEARN_FEWSHOT_SUBJECT_PREFIX: str = "learn.fewshot."
"""SemMem 上の Few-shot ファクト subject prefix。
``learn.fewshot.<mode>.<example_id>`` の形式で書き出す。"""

DEFAULT_FEWSHOT_PREDICATE: str = "example_for"
"""Few-shot ファクトの predicate (常に固定)"""

EvolveWriteback = Literal["yaml", "semmem"]

#: _from_payload で JSON から復元する FewShotExample のキー集合。
#: 未知キー (旧スキーマ / フィールド削除) を無視して TypeError によるプール
#: 全消失を防ぐ (level0_instant の FeedbackSignals 復元と対称)。
_EXAMPLE_FIELD_NAMES = frozenset(f.name for f in fields(FewShotExample))


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
    """経験 1 件のシグナルから段階的 fitness を計算する。

    基準項 (conversation_ended / rephrased_query / user_correction) の係数は
    PromptEvolver._calc_fitness と揃える。これに RAG ヒット品質・エージェント
    反復・長文生成の完了率/検証エラーを段階項として加減算し [0,1] に正規化する。
    欠損シグナル (None / 0 / False) は加点も減点もせず中立に倒すため、シグナル
    未配線のモードでも従来の基準項のみで評価される。連続値を返すことで、
    プールに入った例の fitness が 1.0 に縮退せず select() の重み付けと
    garbage_collect の lowest-fitness eviction が意味を持つ。
    """
    score = 0.0

    # ── 基準項 (係数は PromptEvolver._calc_fitness と同一) ──
    if signals.get("conversation_ended", False):
        score += 1.0
    if signals.get("rephrased_query", False):
        score -= 0.5
    if signals.get("user_correction") is not None:
        score -= 0.8

    # ── 段階項: RAG ヒット品質 (top1 cos 類似が高いほど根拠が強い)。欠損は中立 ──
    rag_top1 = signals.get("rag_top1_score")
    if rag_top1 is not None:
        score += 0.3 * max(0.0, min(1.0, float(rag_top1)))

    # ── 段階項: エージェント反復。<=1 は中立、多反復ほど停滞として減点 ──
    loops = int(signals.get("agent_loops", 0) or 0)
    if loops > 1:
        score -= min(0.3, 0.1 * (loops - 1))

    # ── 段階項: 長文生成 (used のときだけ評価)。完了率で加点・検証エラーで減点 ──
    if signals.get("long_form_used", False):
        total = int(signals.get("long_form_units_total", 0) or 0)
        completed = int(signals.get("long_form_units_completed", 0) or 0)
        if total > 0:
            score += 0.3 * min(1.0, completed / total)
        verr = int(signals.get("long_form_validation_errors", 0) or 0)
        if verr > 0:
            score -= min(0.3, 0.1 * verr)
        if signals.get("long_form_success", False):
            score += 0.1

    # 正規化: score の理論域 [_FITNESS_LO, _FITNESS_HI] を [0,1] へ線形写像。
    return max(0.0, min(1.0, (score - _FITNESS_LO) / (_FITNESS_HI - _FITNESS_LO)))


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
        base_model_id: str = "",
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
        # base 学習パーティションの active モデルスラグ。空 = partition 無効
        # (subject はレガシー ``learn.fewshot.<mode>.<id>`` 形式に縮退)。
        self._base_model_id: str = base_model_id

        # モード別のプール: mode → list[FewShotExample]
        self._pools: dict[str, list[FewShotExample]] = {}
        # キャッシュ: id → bi-gram Counter (採用済 example のみ)
        self._bigram_cache: dict[str, Counter] = {}
        # 明示 dedup 用: mode → {content_hash}。プール内の example と 1:1 で同期する
        # (採用で add、eviction で discard)。diversity_threshold と独立に厳密重複を排除。
        self._seen_hashes: dict[str, set[str]] = {}

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

    def set_base_model_id(self, base_model_id: str) -> None:
        """base 学習パーティションの active モデルスラグを差し替える。

        モデル切替時に :func:`backend.factory._learning_rebind.rebind_base_learning`
        から呼ばれ、以後の writeback / bootstrap が当該モデルの fewshot ファクト
        (``learn.fewshot.<model>.*``) のみを対象にする。空文字でレガシー縮退。
        """
        self._base_model_id = base_model_id or ""

    @staticmethod
    def _build_subject(base_model_id: str, mode: str, example_id: str) -> str:
        """``learn.fewshot.<model>.<mode>.<example_id>`` 形式の subject を構築する。

        ``base_model_id`` が空のときは partition 無効としてレガシー
        ``learn.fewshot.<mode>.<example_id>`` (2 段) へ縮退する。
        """
        if base_model_id:
            return f"{LEARN_FEWSHOT_SUBJECT_PREFIX}{base_model_id}.{mode}.{example_id}"
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
        # rest は新形式 "<model>.<mode>.<id>" または レガシー "<mode>.<id>"。
        # mode は {chat, coding} に限られるため先頭/2 番目セグメントで判別する。
        segs = rest.split(".")
        if len(segs) >= 2 and segs[0] in ("chat", "coding"):
            mode_part, id_part = segs[0], ".".join(segs[1:])
        elif len(segs) >= 3 and segs[1] in ("chat", "coding"):
            mode_part, id_part = segs[1], ".".join(segs[2:])
        else:
            return None
        if not id_part:
            return None
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

        subject = self._build_subject(self._base_model_id, example.mode, example.id)
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
        # partition 有効時は active モデルの fewshot ファクトのみを hydrate する
        # (``learn.fewshot.<model>.``)。未知モデルは 0 件 = 空プール = ゼロから学習。
        # partition 無効時は従来どおり全 fewshot を対象 (レガシー縮退)。
        prefix = (
            f"{LEARN_FEWSHOT_SUBJECT_PREFIX}{self._base_model_id}."
            if self._base_model_id
            else LEARN_FEWSHOT_SUBJECT_PREFIX
        )
        try:
            facts = view.search_fewshot_by_prefix(prefix)
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
            # dedup ハッシュをプールと同期 (#6 整合)
            self._seen_hashes.setdefault(example.mode, set()).add(
                self._content_hash(example.query, example.response),
            )
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
        """新しい例が既存の例と十分に異なるか判定する。

        ``new`` の bi-gram はキャッシュに登録しない (不採用候補が
        ``_bigram_cache`` に残留するリークを防ぐ)。採用済 ``existing`` 側のみ
        ``_get_bigrams`` でキャッシュする。
        """
        if not existing:
            return True
        new_bg = _char_bigrams(f"{new.query} {new.response}")
        for ex in existing:
            sim = _cosine_similarity(new_bg, self._get_bigrams(ex))
            if sim >= self.diversity_threshold:
                return False
        return True

    @staticmethod
    def _content_hash(query: str, response: str) -> str:
        """query + response の正規化 (trim + lower) ハッシュ。厳密重複判定用。"""
        norm = f"{query.strip().lower()}\x00{response.strip().lower()}"
        return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()

    def _try_accept(self, example: FewShotExample) -> FewShotExample | None:
        """候補 1 件の採否判定 + 採用処理を一手に行う (両投入経路で共有)。

        手順: 厳密重複 dedup → 多様性チェック → append → SemMem 書き戻し →
        プールサイズ GC (yaml モードのみ)。採用なら ``example`` を、棄却なら
        ``None`` を返す。``_seen_hashes`` はプール内 example と 1:1 に保つ。
        """
        pool = self._pools.setdefault(example.mode, [])
        seen = self._seen_hashes.setdefault(example.mode, set())
        h = self._content_hash(example.query, example.response)
        if h in seen:
            return None
        if not self._is_diverse(example, pool):
            return None

        pool.append(example)
        seen.add(h)
        self._writeback_example_fact(example)

        # プールサイズ制限: SemMem 書き戻しモードでは GC を SemMem 側
        # (semmem_limits.policy + gc_strategy=lowest_score) に委譲し局所 GC を停止。
        if len(pool) > self.pool_size and not self.is_semmem_writeback_active():
            pool.sort(key=lambda e: e.fitness)
            removed = pool.pop(0)
            self._bigram_cache.pop(removed.id, None)
            seen.discard(self._content_hash(removed.query, removed.response))
        return example

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

            # 成功例のみ（訂正・言い直し・失敗ターンがない）
            if signals.get("user_correction") is not None:
                continue
            if signals.get("rephrased_query", False):
                continue
            if signals.get("turn_outcome") == "failed":
                continue

            query = exp.get("query", "").strip()
            # few-shot 例には切り詰めていない全文を優先採用 (採用例の途中切れ防止)。
            # 旧 experience.json / response_full 欠落時は要約へフォールバック。
            response = (exp.get("response_full") or exp.get("response_summary", "")).strip()
            mode = exp.get("mode", "chat")

            # クエリと応答が存在するか
            if not query or not response:
                continue

            # 応答がタスク進捗ノート形式 (- [done] ... Written N bytes) のみの
            # 例は「報告だけ出せば正解」バイアスを注入するため採用しない
            # (2026-07-15: この形式の例が毎ターン選択され本文なしの極小
            # ファイル生成を誘発した)。
            if _response_is_task_log_only(response):
                continue

            example = FewShotExample(
                query=query,
                response=response,
                mode=mode,
                fitness=fitness,
                added_at=exp.get("timestamp", ""),
            )

            if self._try_accept(example) is not None:
                added += 1

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
        accepted = self._try_accept(example)
        if accepted is None:
            return None

        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "accept_from_artifact",
                "mode": mode,
                "fitness": float(fitness),
                "example_id": accepted.id,
                "pool_size": len(self._pools.get(mode, [])),
            })
        return accepted

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
        # fitness による重み付きサンプリング（高 fitness を優先）。
        # 全ゼロ fitness (bootstrap で fitness 欠損=0.0 復元等) だと sum=0 →
        # 0/0=NaN で rng.choice がクラッシュする。非ゼロ重み数 < size でも
        # replace=False で ValueError。これらを一様重みにフォールバックして防ぐ。
        size = min(n, len(pool))
        fitnesses = np.array([ex.fitness for ex in pool], dtype=float)
        fitnesses = np.nan_to_num(fitnesses, nan=0.0, posinf=0.0, neginf=0.0)
        fitnesses = np.clip(fitnesses, 0.0, None)
        total = float(fitnesses.sum())
        nonzero = int((fitnesses > 0).sum())
        if total <= 0.0 or nonzero < size:
            weights = np.full(len(pool), 1.0 / len(pool))
        else:
            weights = fitnesses / total
        indices = rng.choice(len(pool), size=size, replace=False, p=weights)
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

    def select_top_k(
        self,
        mode: str,
        query: str,
        k: int | None = None,
    ) -> list[FewShotExample]:
        """query 類似度 (char bi-gram cosine) × fitness 重み付けで上位 k を返す。

        推論時の動的 few-shot 選択器 (FewShotSelector Protocol の実装)。埋め込み
        サーバは使わず同期・低遅延。``combined = SIM_W * sim + (1-SIM_W) * fitness``。
        pool が大きい場合は fitness 上位 ``_TOPK_SELECT_CAP`` 件に足切りしてから
        類似計算する (hot path のレイテンシ抑制)。query/pool が空なら ``[]``。
        """
        k = k or self.max_examples
        pool = self._pools.get(mode, [])
        q = (query or "").strip()
        if not pool or not q:
            return []
        # fitness 上位 cap 件に足切り (全件 cosine を避ける)
        if len(pool) > _TOPK_SELECT_CAP:
            pool = sorted(pool, key=lambda e: -e.fitness)[:_TOPK_SELECT_CAP]
        q_bg = _char_bigrams(q)  # query bi-gram は 1 回だけ計算
        scored: list[tuple[float, FewShotExample]] = [
            (
                _TOPK_SIM_WEIGHT * _cosine_similarity(q_bg, self._get_bigrams(ex))
                + (1.0 - _TOPK_SIM_WEIGHT) * ex.fitness,
                ex,
            )
            for ex in pool
        ]
        scored.sort(key=lambda t: -t[0])
        # 合成スコアが下限未満の候補は返さない (無関連例の混入防止)
        selected = [ex for s, ex in scored[:k] if s >= _TOPK_MIN_SCORE]

        dl = self._debug_logger
        if dl:
            selected_ids = [ex.id for ex in selected]
            scores = [round(s, 4) for s, _ in scored[:k]]
            # evolve 専用 learning カテゴリ (従来通り)
            dl.log_learning_cycle(cycle_num=0, data={
                "component": "fewshot_pool",
                "op": "select_top_k",
                "mode": mode,
                "pool_considered": len(pool),
                "query_len": len(q),
                "selected_ids": selected_ids,
                "scores": scores,
            })
            # debug / investigate でも見える rag カテゴリへ並行出力
            dl.log_fewshot_select(
                mode=mode,
                query_len=len(q),
                pool_considered=len(pool),
                selected_ids=selected_ids,
                scores=scores,
            )
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
            seen = self._seen_hashes.setdefault(mode, set())
            for _ in range(excess):
                removed = pool.pop(0)
                self._bigram_cache.pop(removed.id, None)
                seen.discard(self._content_hash(removed.query, removed.response))
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
        # 二重 load / bootstrap 後 load で旧モードが残らないよう全状態をリセット。
        self._pools.clear()
        self._bigram_cache.clear()
        self._seen_hashes.clear()
        for mode, entries in payload.items():
            pool: list[FewShotExample] = []
            seen = self._seen_hashes.setdefault(mode, set())
            for entry in entries:
                # 未知キーを無視して復元 (旧スキーマ耐性、TypeError 全消失を防ぐ)。
                ex = FewShotExample(**{
                    k: v for k, v in entry.items() if k in _EXAMPLE_FIELD_NAMES
                })
                pool.append(ex)
                seen.add(self._content_hash(ex.query, ex.response))
            self._pools[mode] = pool

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
