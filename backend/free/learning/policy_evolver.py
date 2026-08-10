"""PolicyParamEvolver: ポリシーパラメータのモード別自動進化

Stage 1 で外部化されたポリシーの数値パラメータを、
ガウス摂動 + fitness 評価で自動最適化する。
LearningScheduler の Level 1 サイクルとして実行される

LLM 不要。数値摂動 + fitness 評価のみ。
chat モードと create モードは独立して進化する。

ACE (arXiv:2510.04618) のデルタ更新パターンを参考に、
全パラメータ同時変更ではなく部分的な差分更新で進化品質を維持する。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np

from backend.free.core.session_mode import normalize_session_mode
from backend.free.learning.json_state_store import JsonPayload, JsonStateStore
from backend.free.memory.types import make_fact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.core.policy_interpreter import PolicyInterpreter
    from backend.free.learning.exploration_controller import ExplorationController
    from backend.free.memory.types import SemanticFact
    from backend.free.memory.views.learn import LearnFactView

logger = get_logger("learning.policy_evolver")

# 進化対象ドメイン（learning は進化のメタパラメータなので対象外）
EVOLVABLE_DOMAINS: list[str] = [
    "router",
    "memory",
    "search",
    "agent",
    "long_form",
]

# 進化対象外の型
_NON_EVOLVABLE_TYPES: set[str] = {"bool", "str"}

# ACE 式: 1 回の進化ステップで摂動するパラメータの最大数
MAX_PARAMS_PER_STEP: int = 5

# 自動ロールバック: 連続低下回数の閾値
ROLLBACK_THRESHOLD: int = 3

# fitness 改善判定の最小差
FITNESS_EPSILON: float = 0.001

# fitness 履歴の保持上限 (無上限 append によるメモリ・永続化肥大を防ぐ)
_FITNESS_HISTORY_CAP: int = 100

# SemMem 書き戻し時の subject prefix と初期値。owner pillar は EvorefLearn。
LEARN_POLICY_SUBJECT_PREFIX: str = "learn.policy."
"""SemMem 上のポリシーファクト subject prefix。
``learn.policy.<mode>.<domain>.<param_path>`` の形式で書き出す。"""

DEFAULT_AUTO_EVOLVED_CONFIDENCE: float = 0.5
"""自動進化された policy ファクトの初期 confidence。
``learning.policy.activation_min_confidence`` (デフォルト 0.7) より低くする
ことで、評価期間中は active 化されない (旧 ``harness.*`` から移行済)。"""

DEFAULT_ACTIVATION_MIN_CONFIDENCE: float = 0.7
"""PolicyInterpreter が policy ファクトを適用する最小 confidence のデフォルト。
昇格 (promotion) 時の目標値として使う (``learning.policy.activation_min_confidence``)。"""

PROMOTION_SURVIVAL_CYCLES: int = 2
"""rollback されずに連続生存した evolve サイクル数がこの値に達したら、active な
policy ファクトを activation_min まで昇格する (proven とみなす評価期間)。"""

EvolveWriteback = Literal["yaml", "semmem"]


class PolicyParamEvolver(JsonStateStore):
    """ポリシーパラメータのモード別自動進化

    各ドメイン・モードごとに独立して進化を行う。
    1回の進化ステップでは ACE 式にランダムなパラメータサブセットのみを摂動し、
    fitness 評価に基づいてパラメータを更新またはロールバックする。
    """

    _state_logger = logger

    def __init__(
        self,
        policy: PolicyInterpreter,
        exploration: ExplorationController,
        debug_logger: DebugLogger | None = None,
        *,
        learn_view: LearnFactView | None = None,
        semmem_writeback_scope: str = "global",
        evolve_writeback: EvolveWriteback = "yaml",
        policy_confidence_init: float = DEFAULT_AUTO_EVOLVED_CONFIDENCE,
        activation_min_confidence: float = DEFAULT_ACTIVATION_MIN_CONFIDENCE,
        base_model_id: str = "",
    ) -> None:
        """
        Args:
            policy: PolicyInterpreter インスタンス
            exploration: ExplorationController インスタンス
            debug_logger: DebugLogger (任意)
            learn_view: EvorefLearn pillar の Fact View
                書き戻し時に同一 ``subject`` の prior fact を
                :meth:`LearnFactView.find_active_policy_by_subject` で探索し、
                新規追加後に :meth:`LearnFactView.supersede_policy` で
                チェーン構築する。``None`` 時は SemMem 書き戻しを行わない。
            semmem_writeback_scope: 書き込み先 scope (``global`` または
                ``project:<id>``)
            evolve_writeback: ``"yaml"`` (従来動作) / ``"semmem"``
                (SemMem に新規 policy ファクトを書き込み、supersedes チェーン
                を構築する)
            policy_confidence_init: 自動進化ファクトの初期 confidence
                (デフォルト 0.5)。``learning.policy.activation_min_confidence``
                ``harness.*`` から移行済)。
        """
        self._policy = policy
        self._exploration = exploration
        self._debug_logger = debug_logger

        # LearnFactView 経由の writeback に一本化
        self._learn_view: LearnFactView | None = learn_view
        self._semmem_writeback_scope: str = semmem_writeback_scope
        self._evolve_writeback: EvolveWriteback = evolve_writeback
        self._policy_confidence_init: float = float(policy_confidence_init)
        self._activation_min_confidence: float = float(activation_min_confidence)
        # base 学習パーティションの active モデルスラグ。空 = partition 無効
        # (subject はレガシー ``learn.policy.<mode>.<domain>.<key>`` に縮退)。
        self._base_model_id: str = base_model_id

        # 経路 3: 実環境成功率プロバイダ
        # callable(domain: str, mode: str) -> float | None
        self._semmem_success_provider: Callable[[str, str], float | None] | None = None
        self._semmem_success_weight: float = 0.0

        # (domain, mode) → fitness 履歴
        self._fitness_history: dict[tuple[str, str], list[float]] = {}

        # (domain, mode) → 連続低下カウント
        self._decline_count: dict[tuple[str, str], int] = {}

        # (domain, mode) → 最良 fitness
        self._best_fitness: dict[tuple[str, str], float] = {}

        # (domain, mode) → best_fitness 達成時の params スナップショット。
        # rollback 時に単段 _snapshots ではなくこれで丸ごと復元する (#1)。
        self._best_params: dict[tuple[str, str], dict] = {}

        # (domain, mode) → rollback されずに連続生存した evolve サイクル数 (#8)。
        # PROMOTION_SURVIVAL_CYCLES 到達で active fact を昇格する。rollback でリセット。
        self._survived_count: dict[tuple[str, str], int] = {}

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
        から呼ばれ、以後の policy 書き戻し subject に当該モデルを埋め込む。空文字で
        レガシー縮退。
        """
        self._base_model_id = base_model_id or ""

    @staticmethod
    def _build_subject(base_model_id: str, mode: str, domain: str, key: str) -> str:
        """``learn.policy.<model>.<mode>.<domain>.<key>`` 形式の subject を構築する。

        ``base_model_id`` が空のときは partition 無効としてレガシー
        ``learn.policy.<mode>.<domain>.<key>`` (model セグメント無し) へ縮退する。
        """
        if base_model_id:
            return f"{LEARN_POLICY_SUBJECT_PREFIX}{base_model_id}.{mode}.{domain}.{key}"
        return f"{LEARN_POLICY_SUBJECT_PREFIX}{mode}.{domain}.{key}"

    def _find_active_fact_in_writeback_store(
        self,
        subject: str,
    ) -> SemanticFact | None:
        """LearnFactView 経由で同一 subject の active policy を返す。

        supersession チェーンの末端 (= ``superseded_by is None``) のうち、最も
        新しい (``created_at`` 最大) ものを返す。
        """
        view = self._learn_view
        if view is None:
            return None
        return view.find_active_policy_by_subject(subject)

    def _writeback_evolved_fact(
        self,
        domain: str,
        mode: str,
        key: str,
        new_value,
        fitness: float,
        sigma: float,
        phase: str,  # noqa: ARG002
        *,
        rollback: bool = False,
    ) -> SemanticFact | None:
        """進化結果を SemMem に新規 policy ファクトとして書き出す。

        SemMem writeback のコア処理。同一 ``subject`` の prior active fact が
        存在する場合は supersedes チェーンに連結する。
        書き込み先ストアが未設定 / writeback モードが ``yaml`` の場合は
        ``None`` を返して no-op する。
        """
        if not self.is_semmem_writeback_active():
            return None
        view = self._learn_view
        assert view is not None  # is_semmem_writeback_active で保証

        subject = self._build_subject(self._base_model_id, mode, domain, key)
        fact_mode: str = normalize_session_mode(mode)
        eval_metric = {
            "fitness": float(fitness),
            "sigma": float(sigma),
        }
        if rollback:
            eval_metric["rollback"] = 1.0

        # prior を新規追加前に捕捉 (追加後だと自身も候補に入るため)
        prior = view.find_active_policy_by_subject(subject)

        new_fact = make_fact(
            subject=subject,
            predicate="value",
            object_=json.dumps(new_value),
            type="policy",
            scope=self._semmem_writeback_scope,
            mode_origin=fact_mode,  # type: ignore[arg-type]
            confidence=self._policy_confidence_init,
            auto_evolved=True,
            eval_metric=eval_metric,
        )
        try:
            added = view.add_policy_fact(new_fact)
        except ValueError as exc:
            logger.warning(
                "policy_evolver semmem writeback failed: subject=%s err=%s",
                subject, exc,
            )
            return None

        if prior is not None and prior.id != added.id:
            view.supersede_policy(old_id=prior.id, new_id=added.id)
        logger.debug(
            "policy_evolver semmem writeback: subject=%s value=%r "
            "rollback=%s prior_id=%s new_id=%s",
            subject, new_value, rollback,
            prior.id if prior else None, added.id,
        )
        return added

    def _maybe_rollback(
        self,
        domain: str,
        mode: str,
        history: list[float],
        fitness: float,
        prev_best: float,
        sigma: float,
        phase: str,
    ) -> dict | None:
        """連続 fitness 低下検出時にロールバックを実行する。

        ロールバック実施時は結果 dict を返す。継続可能なら ``None``。
        副作用として `self._decline_count` / `self._best_fitness` を更新する。
        """
        key = (domain, mode)
        is_decline = (
            len(history) >= 2 and fitness < prev_best - FITNESS_EPSILON
        )
        if not is_decline:
            # 改善または維持 → カウンタリセット
            self._decline_count[key] = 0
            if fitness > prev_best:
                self._best_fitness[key] = fitness
                # best 達成時の params を丸ごと捕捉 (rollback の復元先)。
                try:
                    self._best_params[key] = dict(self._policy.get_all(domain, mode))
                except KeyError:
                    pass
            return None

        self._decline_count[key] = self._decline_count.get(key, 0) + 1
        if self._decline_count[key] < ROLLBACK_THRESHOLD:
            return None

        # 単段 _snapshots は「直近 apply の 1 段前」までしか戻れず低下起点デルタが
        # 焼き付くため、best 達成時の params スナップショットで丸ごと復元する。
        # best 未確立時のみ従来の単段 rollback にフォールバックする。
        best = self._best_params.get(key)
        if best is not None:
            self._policy.restore_params(domain, best, mode)
            self._policy.save()
            rolled_back = True
        else:
            rolled_back = self._policy.rollback(domain, mode)
            if rolled_back:
                self._policy.save()
        if rolled_back and self.is_semmem_writeback_active():
            # SemMem 履歴を evolved → rollback のチェーンとして残す。
            self._writeback_rollback_facts(domain, mode, fitness, sigma, phase)
        self._decline_count[key] = 0
        # rollback したので生存カウントをリセット (#8: 評価期間を最初からやり直す)。
        self._survived_count[key] = 0
        # 直近に積んだ「低下した」fitness を履歴から除き、復元後の best から
        # 再評価を始められるようにする (良好値の消去ではない)。
        if history:
            history.pop()

        logger.info(
            "Policy rollback: domain=%s, mode=%s, "
            "fitness=%.4f, prev_best=%.4f, rolled_back=%s",
            domain, mode, fitness, prev_best, rolled_back,
        )
        self._log_evolution(domain, mode, "rollback", fitness, sigma, phase)
        return {
            "action": "rollback",
            "fitness": fitness,
            "sigma": sigma,
            "phase": phase,
        }

    def _writeback_rollback_facts(
        self,
        domain: str,
        mode: str,
        fitness: float,
        sigma: float,
        phase: str,
    ) -> int:
        """ロールバック後の値で SemMem に rollback ファクトを書き戻す。

        書き込み対象は「書き込み先ストアに同一 subject の active fact が
        既に存在するキー」のみ。これにより SemMem 履歴に存在しないキーまで
        勝手に追加してしまうのを防ぐ。

        Returns:
            書き出した rollback ファクト数
        """
        if self._learn_view is None:
            return 0
        try:
            params = self._policy.get_all(domain, mode)
        except KeyError:
            return 0
        count = 0
        for key, value in params.items():
            subject = self._build_subject(self._base_model_id, mode, domain, key)
            prior = self._find_active_fact_in_writeback_store(subject)
            if prior is None:
                continue
            written = self._writeback_evolved_fact(
                domain, mode, key, value, fitness, sigma, phase,
                rollback=True,
            )
            if written is not None:
                count += 1
        return count

    def _apply_evolved_step(
        self,
        domain: str,
        mode: str,
        fitness: float,
        sigma: float,
        phase: str,
    ) -> dict:
        """ガウス摂動でデルタ生成 → ACE 適用。デルタ無しなら ``skipped``。"""
        delta = self._generate_delta(domain, mode, sigma)
        if not delta:
            logger.debug(
                "Policy evolution skipped: domain=%s, mode=%s "
                "(no evolvable params)",
                domain, mode,
            )
            return {
                "action": "skipped",
                "fitness": fitness,
                "sigma": sigma,
                "phase": phase,
            }

        self._policy.apply_delta(domain, delta, mode)
        self._policy.save()
        delta_keys = list(delta.keys())

        # SemMem に新規 policy ファクトを書き戻し
        # 同一 subject の prior fact を supersede する。
        if self.is_semmem_writeback_active():
            try:
                applied_params = self._policy.get_all(domain, mode)
            except KeyError:
                applied_params = {}
            for key in delta_keys:
                if key not in applied_params:
                    continue
                self._writeback_evolved_fact(
                    domain, mode, key, applied_params[key],
                    fitness, sigma, phase,
                )

        # 生存カウント更新 + proven 昇格 (#8、semmem モードのみ)。rollback されずに
        # PROMOTION_SURVIVAL_CYCLES 連続で evolved できた = proven とみなし、今書いた
        # active fact を activation_min まで昇格して PolicyInterpreter が適用できるように
        # する。次サイクルの摂動が新 0.5 fact で supersede するため適用は断続的だが、
        # 安定領域では周期的に活性化される (conservative)。
        if self.is_semmem_writeback_active():
            key = (domain, mode)
            self._survived_count[key] = self._survived_count.get(key, 0) + 1
            if self._survived_count[key] >= PROMOTION_SURVIVAL_CYCLES:
                self._promote_active_facts(domain, mode, delta_keys)

        logger.info(
            "Policy evolved: domain=%s, mode=%s, fitness=%.4f, "
            "sigma=%.4f, phase=%s, delta_keys=%s",
            domain, mode, fitness, sigma, phase, delta_keys,
        )
        self._log_evolution(
            domain, mode, "evolved", fitness, sigma, phase, delta_keys,
        )
        return {
            "action": "evolved",
            "fitness": fitness,
            "delta_keys": delta_keys,
            "sigma": sigma,
            "phase": phase,
        }

    def _promote_active_facts(
        self, domain: str, mode: str, keys: list[str],
    ) -> None:
        """指定 delta キーの active policy fact を activation_min まで昇格する (#8)。"""
        view = self._learn_view
        if view is None:
            return
        target = self._activation_min_confidence
        for key in keys:
            subject = self._build_subject(self._base_model_id, mode, domain, key)
            fact = self._find_active_fact_in_writeback_store(subject)
            if fact is not None and fact.confidence < target:
                view.promote_policy_confidence(fact_id=fact.id, confidence=target)
                logger.info(
                    "Policy fact promoted: subject=%s, confidence=%.2f->%.2f",
                    subject, fact.confidence, target,
                )

    def set_semmem_success_provider(
        self,
        provider: Callable[[str, str], float | None] | None,
        weight: float,
    ) -> None:
        """SemMem 由来の実環境成功率プロバイダを設定する

        Args:
            provider: ``(domain, mode) -> success_rate | None`` を返す callable。
                ``None`` を返すか、本メソッドに ``None`` を渡すと補正を無効化する。
            weight: ``[0.0, 1.0]`` の重み。最終 fitness は
                ``(1-w) * calc_fitness + w * success_rate`` で混合される。
        """
        self._semmem_success_provider = provider
        self._semmem_success_weight = max(0.0, min(1.0, float(weight)))

    def _blend_semmem_success_rate(
        self,
        domain: str,
        mode: str,
        base: float | None,
    ) -> float | None:
        """calc_fitness に SemMem 成功率項を加重合成する。

        プロバイダ未設定 / 重み 0 / プロバイダが ``None`` を返した場合は
        ``base`` をそのまま返す。``base`` が ``None`` (無情報) なら素通し。
        """
        if base is None:
            return None
        if (
            self._semmem_success_provider is None
            or self._semmem_success_weight <= 0.0
        ):
            return base
        try:
            rate = self._semmem_success_provider(domain, mode)
        except Exception as exc:
            logger.warning(
                "semmem_success_provider failed (domain=%s mode=%s): %s",
                domain, mode, exc,
            )
            return base
        if rate is None:
            return base
        rate = max(0.0, min(1.0, float(rate)))
        w = self._semmem_success_weight
        blended = (1.0 - w) * base + w * rate
        logger.debug(
            "policy fitness blended: domain=%s mode=%s base=%.3f rate=%.3f w=%.2f -> %.3f",
            domain, mode, base, rate, w, blended,
        )
        return blended

    def evolve(
        self,
        domain: str,
        mode: str,
        experiences: list[dict],
    ) -> dict:
        """1ドメイン・1モードのパラメータ進化を1ステップ実行する

        Args:
            domain: ポリシードメイン ("router", "memory", "search", ...)
            mode: "chat" | "create"
            experiences: そのモードの経験リスト

        Returns:
            進化結果 dict:
            - action: "evolved" | "rollback" | "skipped"
            - fitness: 現在の fitness 値
            - delta_keys: 摂動したパラメータ名リスト（evolved 時のみ）
            - sigma: 使用した変異スケール
            - phase: ExplorationController のフェーズ
        """
        key = (domain, mode)

        # 1. 現在の fitness を算出。無情報 (該当シグナルゼロ) なら None が返る。
        # この場合は履歴に積まず摂動もせず即 skip し、無情報ドメインの乱歩を防ぐ。
        base_fitness = calc_fitness(domain, experiences)
        if base_fitness is None:
            sigma = self._exploration.get_mutation_scale(domain, mode)
            phase = self._exploration.get_phase(domain, mode)
            self._log_evolution(domain, mode, "skipped_no_signal", 0.0, sigma, phase)
            return {
                "action": "skipped_no_signal", "fitness": None,
                "sigma": sigma, "phase": phase,
            }

        fitness = self._blend_semmem_success_rate(domain, mode, base_fitness)
        history = self._fitness_history.setdefault(key, [])
        history.append(fitness)
        if len(history) > _FITNESS_HISTORY_CAP:
            del history[:-_FITNESS_HISTORY_CAP]

        # 2. ExplorationController に fitness 履歴をフィードバック
        self._exploration.update(domain, mode, history)
        sigma = self._exploration.get_mutation_scale(domain, mode)
        phase = self._exploration.get_phase(domain, mode)

        # 3. 連続低下判定 → 自動ロールバック
        prev_best = self._best_fitness.get(key, 0.0)
        rollback_result = self._maybe_rollback(
            domain, mode, history, fitness, prev_best, sigma, phase,
        )
        if rollback_result is not None:
            return rollback_result

        # 3.5. decline 継続中 (ロールバック閾値未満の連続低下) は新摂動を打たず hold。
        # 劣化中に摂動を重ねて複利的に悪化するのを断つ。
        if self._decline_count.get(key, 0) > 0:
            self._log_evolution(domain, mode, "hold", fitness, sigma, phase)
            return {
                "action": "hold", "fitness": fitness,
                "sigma": sigma, "phase": phase,
                "decline_count": self._decline_count[key],
            }

        # 4-5. ガウス摂動でデルタ生成 → ACE 適用
        return self._apply_evolved_step(domain, mode, fitness, sigma, phase)

    def evolve_all(
        self,
        experiences: list[dict],
        modes: list[str] | None = None,
    ) -> dict[str, dict]:
        """全ドメイン・モードの進化を一括実行する

        Args:
            experiences: 全経験リスト（内部でモード別にフィルタ）
            modes: 対象モードリスト（デフォルト: ["chat", "create"]）

        Returns:
            "{domain}_{mode}" → 進化結果 dict
        """
        if modes is None:
            modes = ["chat", "create"]

        results: dict[str, dict] = {}

        for domain in EVOLVABLE_DOMAINS:
            for mode in modes:
                mode_exp = [e for e in experiences if e.get("mode") == mode]
                if not mode_exp:
                    continue
                result = self.evolve(domain, mode, mode_exp)
                results[f"{domain}_{mode}"] = result

        return results

    def get_fitness_history(
        self,
        domain: str,
        mode: str,
    ) -> list[float]:
        """指定ドメイン・モードの fitness 履歴を返す"""
        return list(self._fitness_history.get((domain, mode), []))

    def get_status(self) -> dict:
        """全ドメイン・モードの進化状態を返す"""
        status = {}
        for (domain, mode), history in self._fitness_history.items():
            key = f"{domain}_{mode}"
            status[key] = {
                "fitness_history_len": len(history),
                "current_fitness": history[-1] if history else None,
                "best_fitness": self._best_fitness.get((domain, mode), 0.0),
                "decline_count": self._decline_count.get((domain, mode), 0),
                "sigma": self._exploration.get_mutation_scale(domain, mode),
                "phase": self._exploration.get_phase(domain, mode),
            }
        return status

    # ── 永続化 (JsonStateStore) ──

    def _to_payload(self) -> JsonPayload:
        return {
            "fitness_history": {
                f"{d}:{m}": h
                for (d, m), h in self._fitness_history.items()
            },
            "decline_count": {
                f"{d}:{m}": c
                for (d, m), c in self._decline_count.items()
            },
            "best_fitness": {
                f"{d}:{m}": f
                for (d, m), f in self._best_fitness.items()
            },
            "best_params": {
                f"{d}:{m}": p
                for (d, m), p in self._best_params.items()
            },
            "survived_count": {
                f"{d}:{m}": c
                for (d, m), c in self._survived_count.items()
            },
        }

    def _from_payload(self, payload: JsonPayload) -> None:
        if not isinstance(payload, dict):
            raise TypeError(
                f"policy_evolver_state.json must be a dict, "
                f"got {type(payload).__name__}"
            )

        self._fitness_history.clear()
        for key_str, history in payload.get("fitness_history", {}).items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                self._fitness_history[(parts[0], parts[1])] = history

        self._decline_count.clear()
        for key_str, count in payload.get("decline_count", {}).items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                self._decline_count[(parts[0], parts[1])] = count

        self._best_fitness.clear()
        for key_str, fitness in payload.get("best_fitness", {}).items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                self._best_fitness[(parts[0], parts[1])] = fitness

        self._best_params.clear()
        for key_str, params in payload.get("best_params", {}).items():
            parts = key_str.split(":", 1)
            if len(parts) == 2 and isinstance(params, dict):
                self._best_params[(parts[0], parts[1])] = params

        self._survived_count.clear()
        for key_str, count in payload.get("survived_count", {}).items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                self._survived_count[(parts[0], parts[1])] = count

    def _on_save_success(self, path: Path) -> None:
        logger.debug("PolicyEvolver state saved: %s", path)

    def _on_load_success(self, path: Path) -> None:
        logger.info("PolicyEvolver state loaded from %s", path)

    def _generate_delta(
        self,
        domain: str,
        mode: str,
        sigma: float,
    ) -> dict:
        """ガウス摂動でパラメータのデルタを生成する

        ACE 式に、全パラメータではなくランダムなサブセットを摂動する。
        bool / str 型のパラメータはスキップする。

        Args:
            domain: ポリシードメイン
            mode: "chat" | "create"
            sigma: 変異スケール（制約の range に対する比率）

        Returns:
            {param_name: new_value, ...}
        """
        try:
            current_params = self._policy.get_all(domain, mode)
            # 制約情報を取得 (公開 API 経由。_data 直アクセスを避ける)
            constraints = self._policy.get_constraints(domain)
        except KeyError:
            return {}

        # 進化可能なパラメータを抽出（bool / str 除外）
        evolvable_keys = []
        for key in current_params:
            constraint = constraints.get(key, {})
            param_type = constraint.get("type", "float")
            if param_type in _NON_EVOLVABLE_TYPES:
                continue
            if "min" not in constraint or "max" not in constraint:
                continue
            evolvable_keys.append(key)

        if not evolvable_keys:
            return {}

        # ACE 式: ランダムにサブセットを選択
        n_params = min(MAX_PARAMS_PER_STEP, len(evolvable_keys))
        selected_keys = random.sample(evolvable_keys, n_params)

        delta: dict = {}
        for key in selected_keys:
            current_val = current_params[key]
            constraint = constraints[key]
            param_min = constraint["min"]
            param_max = constraint["max"]
            param_type = constraint.get("type", "float")
            param_range = param_max - param_min

            # ガウスノイズ: σ * range のスケール
            noise = float(np.random.normal(0, sigma * param_range))
            new_val = current_val + noise

            # 型変換
            if param_type == "int":
                new_val = int(round(new_val))
            else:
                new_val = float(new_val)

            # PolicyInterpreter.apply_delta() が clamp するので
            # ここでは範囲外でも構わないが、明示的にクランプする
            new_val = max(param_min, min(param_max, new_val))

            # 変化がない場合はスキップ
            if param_type == "int" and new_val == current_val:
                continue
            if param_type == "float" and abs(new_val - current_val) < 1e-9:
                continue

            delta[key] = new_val

        return delta

    def _log_evolution(
        self,
        domain: str,
        mode: str,
        action: str,
        fitness: float,
        sigma: float,
        phase: str,
        delta_keys: list[str] | None = None,
    ) -> None:
        """デバッグログに進化ステップを記録する"""
        dl = self._debug_logger
        if dl is None:
            return
        dl.log_learning_cycle(cycle_num=1, data={
            "level": 1,
            "phase": 8,
            "component": "policy_evolver",
            "domain": domain,
            "mode": mode,
            "action": action,
            "fitness": fitness,
            "sigma": sigma,
            "exploration_phase": phase,
            "delta_keys": delta_keys or [],
        })


# ── ドメイン別 fitness 関数 ──


def calc_fitness(domain: str, experiences: list[dict]) -> float | None:
    """ドメインに適した fitness 値を算出する

    Args:
        domain: ポリシードメイン
        experiences: そのモードの経験リスト

    Returns:
        fitness 値 [0.0, 1.0]。該当シグナルが無く評価不能なら ``None``
        (呼出側は履歴に積まず進化を skip する)。
    """
    fn = _FITNESS_FUNCTIONS.get(domain, _calc_fitness_default)
    return fn(experiences)


def _calc_fitness_router(experiences: list[dict]) -> float | None:
    """router fitness: ルーティング精度（ユーザー修正率の逆数）

    修正・言い換えが少ないほど高スコア。経験ゼロは評価不能 (None)。
    """
    total = len(experiences)
    if total == 0:
        return None

    bad = sum(
        1 for e in experiences
        if (e.get("signals", {}).get("user_correction")
            or e.get("signals", {}).get("rephrased_query"))
    )
    return 1.0 - (bad / total)


def _calc_fitness_memory(experiences: list[dict]) -> float | None:
    """memory fitness: 会話品質指標

    会話完了（good）と修正・言い換え（bad）のバランスで評価。経験ゼロは None。
    """
    total = len(experiences)
    if total == 0:
        return None

    good = sum(
        1 for e in experiences
        if e.get("signals", {}).get("conversation_ended")
    )
    bad = sum(
        1 for e in experiences
        if (e.get("signals", {}).get("user_correction")
            or e.get("signals", {}).get("rephrased_query"))
    )
    # [-0.5, 1.0] を [0.0, 1.0] にマッピング
    raw = (good - bad * 0.5) / total
    return max(0.0, min(1.0, raw + 0.5))


def _calc_fitness_search(experiences: list[dict]) -> float | None:
    """search fitness: RAG チャンク使用率

    rag_top1_score の平均値。RAG 未使用経験は除外。RAG 経験ゼロは評価不能 (None)。
    """
    rag_exps = [
        e for e in experiences
        if e.get("signals", {}).get("rag_used")
    ]
    if not rag_exps:
        return None

    scores = [
        e["signals"]["rag_top1_score"]
        for e in rag_exps
        if e.get("signals", {}).get("rag_top1_score") is not None
    ]
    if not scores:
        return None

    return max(0.0, min(1.0, sum(scores) / len(scores)))


def _calc_fitness_agent(experiences: list[dict]) -> float | None:
    """agent fitness: ステップ効率

    エージェントループ数が少なく、会話が完了しているほど高スコア。
    agent 経験ゼロは評価不能 (None)。
    """
    agent_exps = [
        e for e in experiences
        if e.get("signals", {}).get("agent_loops", 0) > 0
    ]
    if not agent_exps:
        return None

    scores = []
    for e in agent_exps:
        signals = e.get("signals", {})
        loops = max(1, signals.get("agent_loops", 1))
        efficiency = 1.0 / loops
        if signals.get("conversation_ended"):
            efficiency += 0.3
        if signals.get("user_correction"):
            efficiency -= 0.3
        scores.append(max(0.0, min(1.0, efficiency)))

    return sum(scores) / len(scores)


def _calc_fitness_long_form(experiences: list[dict]) -> float | None:
    """long_form fitness: 長文生成の完了率と品質

    ユニット完了率とバリデーションエラー率で評価。長文経験ゼロは評価不能 (None)。
    """
    lf_exps = [
        e for e in experiences
        if e.get("signals", {}).get("long_form_used")
    ]
    if not lf_exps:
        return None

    scores = []
    for e in lf_exps:
        signals = e.get("signals", {})
        total_units = signals.get("long_form_units_total", 0)
        completed = signals.get("long_form_units_completed", 0)
        errors = signals.get("long_form_validation_errors", 0)

        if total_units > 0:
            completion_rate = completed / total_units
            error_penalty = min(0.3, errors * 0.1)
            scores.append(max(0.0, min(1.0, completion_rate - error_penalty)))

    if not scores:
        return None

    return sum(scores) / len(scores)


def _calc_fitness_default(experiences: list[dict]) -> float | None:
    """デフォルト fitness: 会話完了率ベース。経験ゼロは評価不能 (None)。"""
    total = len(experiences)
    if total == 0:
        return None

    good = sum(
        1 for e in experiences
        if e.get("signals", {}).get("conversation_ended")
    )
    bad = sum(
        1 for e in experiences
        if (e.get("signals", {}).get("user_correction")
            or e.get("signals", {}).get("rephrased_query"))
    )
    return max(0.0, min(1.0, (good - bad * 0.5) / total + 0.5))


_FITNESS_FUNCTIONS: dict[str, Callable[..., float | None]] = {
    "router": _calc_fitness_router,
    "memory": _calc_fitness_memory,
    "search": _calc_fitness_search,
    "agent": _calc_fitness_agent,
    "long_form": _calc_fitness_long_form,
}
