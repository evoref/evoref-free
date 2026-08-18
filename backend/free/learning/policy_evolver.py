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

# fitness の算出に必要な、ドメイン固有シグナルの最小サンプル数。これ未満だと
# 1 件の増減が fitness を 1/N 動かすため、摂動の良し悪しではなくサンプルの
# 偏りを見ていることになる。実測 (2026-08-18、経験 206 件): agent ドメインの
# 母集合 (agent_loops > 0) は **1 件** しかなく、その 1 件から fitness=1.0 が
# 確定して best_fitness に焼き付き、以後ロールバック不能なまま摂動だけが
# 続いていた。
MIN_FITNESS_SAMPLES: int = 10

# 恒真ガードの評価窓。直近この件数の fitness が FITNESS_EPSILON 未満の幅しか
# 動いていなければ、その fitness は当該ドメインの摂動を判別できていないと
# みなす。実測データでは window=4 が memory / agent (span=0.0) を捕捉し、
# router (span=0.0135) / search (span=0.0972) は素通しする。
DEGENERATE_WINDOW: int = 4

#: 永続化された評価状態 (fitness_history / best_fitness / best_params …) が
#: どの fitness 定義の下で採られたかを示すバージョン。**fitness 関数を変更したら
#: 上げる**。ロード時に不一致なら評価状態を捨てる — 旧定義下の平坦な履歴を新定義の
#: 履歴として続けると、恒真ガードが「新 fitness も恒真」と誤判定して凍結し、
#: 尺度の違う best_fitness が更新不能な基準として残る。
#:
#: v3: コスト項を fitness に導入 (:data:`COST_WEIGHT`)。
FITNESS_SCHEMA_VERSION: int = 3

#: fitness に占めるコスト項の重み。``fitness = (1-w) * 品質 + w * (1 - コスト)``。
#:
#: **実測から導出** (2026-08-18、``scripts/calibrate_cost_weight.py``)。fitness は
#: 経験集合の平均を見るので、選択圧になるのは各項の **tick 間 span** (プレフィックス
#: 平均が tick ごとに動く幅) である。両項の寄与を釣り合わせる重みは
#: ``w* = span_Q / (span_Q + span_C)``。
#:
#: 実機で 56 ターン回してコスト付き経験を 50 件貯めた時点の実測:
#:
#: ===============  ==========  ======================================
#: 項                span        出所
#: ===============  ==========  ======================================
#: コスト            0.0922      経験バッファ (再プリフィル率、n=50)
#: コスト            0.1280      llama ログ復元 (slot 0、n=162) — 独立系統
#: 品質 memory       0.0181      3 回の測定で 0.0203 / 0.0183 / 0.0181 と安定
#: ===============  ==========  ======================================
#:
#: → ``w*`` = 0.0181 / (0.0181 + 0.0922) = **0.164**、llama ログ側の span を使うと
#: 0.123。**0.12〜0.17 のレンジ**なので中央の 0.15 を採る。
#:
#: 独立した 2 系統が同オーダーで一致していることが根拠の主。当初は 0.3 と見積もって
#: いたが、実測ではコスト側の自然変動が想定より大きく、**釣り合わせるのに必要な重みは
#: 小さい**。
#:
#: 下限側の妥当性: 取得件数の膨張を止めるのに必要な重みは ``w > 0.03`` 程度
#: (top_k 5→10 で ΔC ≈ +0.19 に対し Δ品質 ≲ 0.005)。0.15 は十分に上回る。
#:
#: 解除対象を持つドメインのうち品質 span を実測できたのは memory だけ。
#: agent / long_form は該当経験がまだ足りない (それぞれ agent_loops>0 と
#: long_form_used の母集合が :data:`MIN_FITNESS_SAMPLES` 未満)。溜まったら
#: ``scripts/calibrate_cost_weight.py`` を再実行して測り直す。
#:
#: **本値を変えたら fitness の尺度が変わるので :data:`FITNESS_SCHEMA_VERSION` も
#: 上げること** (旧尺度の best_fitness が更新不能な基準として残る)。
COST_WEIGHT: float = 0.15

#: プロンプト側コストで評価するドメイン。
#:
#: **コスト項は :data:`MONOTONE_PARAMS` を解除するために入れる**ので、解除対象を
#: 持つドメインにだけ適用する。持たないドメイン (``router`` / ``search``) では、
#: 残る進化キー (重み・閾値の類) がプロンプト量に影響しないため、コスト項は
#: **相関のないノイズ**として fitness に乗るだけで摂動の判別力を落とす。
#: 両集合の一致はテストで固定する。
_PROMPT_COST_DOMAINS: frozenset[str] = frozenset({"memory", "agent"})

#: 生成側コストで評価するドメイン。凍結キーが生成トークン量を増やす方向のもの。
_COMPLETION_COST_DOMAINS: frozenset[str] = frozenset({"long_form"})

#: **コスト項を入れても解除しない**パラメータと、その理由。
#:
#: 解除の条件は「コストが観測できること」だけでは足りず、**品質項とコスト項の
#: 双方がそのパラメータに反応すること**が要る。片方でも無反応だと最適解が制約の
#: 端に固定される — コストが見えなければ最大へ、品質が見えなければ最小へ張り付く
#: だけで、病理の向きが変わるにすぎない。
#:
#: 値は「なぜ恒久凍結か」の根拠。テストが policy 側のキー実在を検証する。
PERMANENTLY_FROZEN_PARAMS: dict[str, dict[str, str]] = {
    "search": {
        # 品質項 mean(rag_top1_score) は **top-1 の類似度** なので、取得件数を
        # 増やしても減らしても動かない。コスト項だけ入れると「最大へ膨張」が
        # 「最小へ縮退」に変わるだけ。解除には取得幅に反応する品質指標が要る。
        "top_k": "品質項 mean(rag_top1_score) が取得件数に無反応",
        "stm_top_k": "品質項 mean(rag_top1_score) が取得件数に無反応",
        # int8 粗検索 → float32 rescore の候補数 (vector_store.search)。返す
        # チャンク数は top_k が決めるので、増やしてもプロンプトは 1 トークンも
        # 増えない。代償は CPU 時間だけで、配線済みのどのシグナルにも出ない。
        "rescore_candidates": "コストが CPU 時間にのみ出てトークン数に現れない",
        # _default_policies() 以外に参照が無い (2026-08-18 時点)。摂動しても
        # 何も起きないので進化スロットの無駄。キー自体の存廃は別途判断する。
        "candidates_multiplier": "消費側が存在しない dead key",
    },
    "agent": {
        # 「残コンテキストがこの値以上なら meta-cognitive を許可」という**ゲート
        # 閾値** (router._can_use_meta_cognitive)。上げるほど meta-cognitive が
        # 発火せず、ループ数もトークンも減る = 品質項 (1/loops) とコスト項の
        # 両方が「機能を止めること」を報酬する。解除には meta-cognitive が
        # 使えたことの価値を測るシグナルが要る。
        "meta_cognitive_min_budget": "上げるほど機能が停止し品質項・コスト項の両方が改善する",
    },
    "memory": {
        # sleep-time の LLM 呼び出し回数を食うだけで、チャットターンの
        # プロンプト/生成トークンには一切現れない。
        "conflict_batch_size": "コストが sleep-time の LLM 呼び出しにのみ出る",
    },
}

#: 品質項だけでは **単調** なパラメータ。増やす (または減らす) ほど品質指標が
#: 改善する一方、その代償 (コンテキスト量・レイテンシ) が品質項には現れない。
#: 最適化器へ渡すと必ず制約の端へ張り付く。
#:
#: **凍結は動的**: その tick の経験集合にコスト実データが
#: :data:`MIN_FITNESS_SAMPLES` 件以上あれば ``fitness`` にコスト項が入り、
#: 単調性が解けるので本表のキーも進化対象に戻る (:func:`is_cost_observable`)。
#: コストが観測できない tick では従来どおり凍結する。フラグではなくデータで
#: 切り替えるのは、コスト項が ``None`` に縮退したまま解除されると単調膨張が
#: そのまま戻るため。
#:
#: コスト項が入っても解除されないキーは :data:`PERMANENTLY_FROZEN_PARAMS` を参照。
#:
#: 既に個別の対症療法が入っていた 2 件 — ``search.top_k`` の上限 50→10 と
#: ``long_form.unit_target_tokens`` の下限 128→512 (いずれも
#: :func:`backend.free.core.policy_interpreter._default_policies` のコメント参照) —
#: と同じ病理を、ドメイン横断で構造的に塞ぐ。
#:
#: **constraints (policy JSON) ではなくコード側に置く理由**: constraints は
#: 「値として妥当な範囲」、本表は「最適化器が触ってよいか」で関心が違う。加えて
#: ``PolicyInterpreter._merge_with_defaults`` の constraints マージはパラメータ名
#: 単位の浅いマージなので、既存インストールの policy JSON に後からサブキーを
#: 足しても伝播しない (凍結が静かに無効化される)。
MONOTONE_PARAMS: dict[str, frozenset[str]] = {
    "agent": frozenset({
        # 圧縮を弱める / スケルトン化を減らすほどステップの情報量が増え、代償は
        # プロンプトのみ。品質項 (1/loops) は情報量が足りないとループが増える形で
        # 反応するため、双方が効く。
        "file_skeleton_threshold",
        "step_compaction_rag_lines", "step_compaction_command_head_tail",
    }),
    "long_form": frozenset({
        # ユニット予算と再試行回数を増やすほど完了率 (品質項) は上がり、
        # 予算消費率 (コスト項) も上がる。双方が反応する最も素直なケース。
        "unit_max_tokens", "unit_target_tokens", "max_extend_rounds",
    }),
    "memory": frozenset({
        # 保持期間を延ばすほど想起は増え (欠陥率が下がる)、代償は想起注入による
        # プロンプト増。
        "decay_days",
    }),
}

#: コスト未観測時に **全パラメータ** が凍結されるドメイン。その tick の進化
#: ステップは常に ``skipped`` になる。
#:
#: ``agent`` の 4 パラメータはステップ圧縮 2 種 / スケルトン閾値 / 最小予算ゲート
#: しかなく、前 3 者は :data:`MONOTONE_PARAMS`、最後は
#: :data:`PERMANENTLY_FROZEN_PARAMS` に入るため、品質項だけで評価できるものが
#: 1 つも残らない。ドメインごと凍結されるのは事故ではなく意図した状態なので
#: 明示しておく (テストが両者の一致を検証する)。プロンプト側コストが観測できる
#: tick では前 3 者が解除される。
FULLY_FROZEN_DOMAINS: frozenset[str] = frozenset({"agent"})

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


def is_degenerate_fitness(history: list[float]) -> bool:
    """直近 :data:`DEGENERATE_WINDOW` 件の fitness が実質同一かを返す。

    ``True`` = その fitness は摂動を判別できていない。このとき摂動を続けても
    選択圧が無く、かつ「低下」が起きないので ``_maybe_rollback`` も永久に発火
    しない = 制約範囲内の乱歩になる。窓が埋まるまで (``< DEGENERATE_WINDOW``)
    は判定を保留する。
    """
    if len(history) < DEGENERATE_WINDOW:
        return False
    recent = history[-DEGENERATE_WINDOW:]
    return (max(recent) - min(recent)) < FITNESS_EPSILON


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
        *,
        cost_observed: bool = False,
    ) -> dict:
        """ガウス摂動でデルタ生成 → ACE 適用。デルタ無しなら ``skipped``。"""
        delta = self._generate_delta(
            domain, mode, sigma, cost_observed=cost_observed,
        )
        if not delta:
            logger.debug(
                "Policy evolution skipped: domain=%s, mode=%s "
                "(no evolvable params, cost_observed=%s)",
                domain, mode, cost_observed,
            )
            return {
                "action": "skipped",
                "fitness": fitness,
                "sigma": sigma,
                "phase": phase,
                "cost_observed": cost_observed,
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
            "sigma=%.4f, phase=%s, cost_observed=%s, delta_keys=%s",
            domain, mode, fitness, sigma, phase, cost_observed, delta_keys,
        )
        self._log_evolution(
            domain, mode, "evolved", fitness, sigma, phase, delta_keys,
            cost_observed=cost_observed,
        )
        return {
            "action": "evolved",
            "fitness": fitness,
            "delta_keys": delta_keys,
            "sigma": sigma,
            "phase": phase,
            "cost_observed": cost_observed,
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
            - cost_observed: fitness にコスト項が入り、単調パラメータの凍結が
              解けている tick か
        """
        key = (domain, mode)

        # 0. コスト項が fitness に入るだけの標本があるか。入っていれば単調性が
        # 解けるので MONOTONE_PARAMS の凍結を外す。fitness の合成と同じ判定を
        # 使うので、「コスト項が効いていないのに凍結だけ解ける」状態は作れない。
        cost_observed = is_cost_observable(domain, experiences)

        # 1. 現在の fitness を算出。無情報 (該当シグナルゼロ) なら None が返る。
        # この場合は履歴に積まず摂動もせず即 skip し、無情報ドメインの乱歩を防ぐ。
        base_fitness = calc_fitness(domain, experiences)
        if base_fitness is None:
            sigma = self._exploration.get_mutation_scale(domain, mode)
            phase = self._exploration.get_phase(domain, mode)
            self._log_evolution(
                domain, mode, "skipped_no_signal", 0.0, sigma, phase,
                cost_observed=cost_observed,
            )
            return {
                "action": "skipped_no_signal", "fitness": None,
                "sigma": sigma, "phase": phase,
                "cost_observed": cost_observed,
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

        # 2.5. 恒真ガード。直近窓の fitness が動いていない = この fitness は当該
        # ドメインの摂動を判別できていない。摂動を続けても選択圧が無く、低下も
        # 起きないのでロールバックも発火せず、制約範囲内の乱歩になる。
        # 実測 (2026-08-18、経験 206 件): memory / agent の 2 ドメインが span=0.0
        # でこの状態にあり、memory の conflict_similarity_threshold が上限 1.0 へ
        # 張り付いて STM の競合検出が実質無効化されていた。
        if is_degenerate_fitness(history):
            logger.info(
                "Policy evolution frozen (degenerate fitness): domain=%s, mode=%s, "
                "fitness=%.4f, window=%d",
                domain, mode, fitness, DEGENERATE_WINDOW,
            )
            self._log_evolution(
                domain, mode, "skipped_degenerate", fitness, sigma, phase,
                cost_observed=cost_observed,
            )
            return {
                "action": "skipped_degenerate",
                "fitness": fitness,
                "sigma": sigma,
                "phase": phase,
                "cost_observed": cost_observed,
            }

        # 3. 連続低下判定 → 自動ロールバック
        prev_best = self._best_fitness.get(key, 0.0)
        rollback_result = self._maybe_rollback(
            domain, mode, history, fitness, prev_best, sigma, phase,
        )
        if rollback_result is not None:
            rollback_result["cost_observed"] = cost_observed
            return rollback_result

        # 3.5. decline 継続中 (ロールバック閾値未満の連続低下) は新摂動を打たず hold。
        # 劣化中に摂動を重ねて複利的に悪化するのを断つ。
        if self._decline_count.get(key, 0) > 0:
            self._log_evolution(
                domain, mode, "hold", fitness, sigma, phase,
                cost_observed=cost_observed,
            )
            return {
                "action": "hold", "fitness": fitness,
                "sigma": sigma, "phase": phase,
                "decline_count": self._decline_count[key],
                "cost_observed": cost_observed,
            }

        # 4-5. ガウス摂動でデルタ生成 → ACE 適用
        return self._apply_evolved_step(
            domain, mode, fitness, sigma, phase, cost_observed=cost_observed,
        )

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
                # 恒真ガードで凍結中か。乱歩は「静かに」起きるので、状態として
                # 外から見えるようにしておく (/api/learning/status 経由)。
                "degenerate": is_degenerate_fitness(history),
            }
        return status

    # ── 永続化 (JsonStateStore) ──

    def _to_payload(self) -> JsonPayload:
        return {
            "fitness_schema_version": FITNESS_SCHEMA_VERSION,
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

        stored_version = payload.get("fitness_schema_version", 1)
        if stored_version != FITNESS_SCHEMA_VERSION:
            # fitness の定義が変わった = 旧履歴を新定義の履歴として続けられない。
            # 続けると (a) 恒真ガードが旧値の平坦さを見て新 fitness を誤って凍結し、
            # (b) 尺度の違う best_fitness が更新されない基準として残り、
            # (c) 旧 fitness 下で採られた best_params が復元先として焼き付く。
            # 評価状態だけを捨てて測り直す (現行 params には触れない — 値の是正は
            # PolicyInterpreter 側の責務)。
            logger.info(
                "PolicyEvolver fitness schema changed (%s -> %s): "
                "discarding stored evaluation state (params are left untouched)",
                stored_version, FITNESS_SCHEMA_VERSION,
            )
            self._fitness_history.clear()
            self._decline_count.clear()
            self._best_fitness.clear()
            self._best_params.clear()
            self._survived_count.clear()
            return

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
        *,
        cost_observed: bool = False,
    ) -> dict:
        """ガウス摂動でパラメータのデルタを生成する

        ACE 式に、全パラメータではなくランダムなサブセットを摂動する。
        bool / str 型のパラメータはスキップする。

        Args:
            domain: ポリシードメイン
            mode: "chat" | "create"
            sigma: 変異スケール（制約の range に対する比率）
            cost_observed: この tick の fitness にコスト項が入っているか。
                ``True`` なら :data:`MONOTONE_PARAMS` の凍結を解く
                (:data:`PERMANENTLY_FROZEN_PARAMS` は解かない)。

        Returns:
            {param_name: new_value, ...}
        """
        try:
            current_params = self._policy.get_all(domain, mode)
            # 制約情報を取得 (公開 API 経由。_data 直アクセスを避ける)
            constraints = self._policy.get_constraints(domain)
        except KeyError:
            return {}

        # 進化可能なパラメータを抽出（bool / str / 凍結パラメータを除外）。
        # 単調パラメータはコストが観測できた tick でのみ解除する。
        frozen = set(PERMANENTLY_FROZEN_PARAMS.get(domain, {}))
        if not cost_observed:
            frozen |= MONOTONE_PARAMS.get(domain, frozenset())
        evolvable_keys = []
        for key in current_params:
            if key in frozen:
                continue
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
        *,
        cost_observed: bool | None = None,
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
            # 単調パラメータが解除されている tick かどうか。凍結/解除の切替が
            # データ次第で起きるので、後から追えるように残す。
            "cost_observed": cost_observed,
        })


# ── ドメイン別 fitness 関数 ──

#: 欠陥シグナルと重み。**加点シグナルを使わない** のが要点。
#: ``conversation_ended`` は :meth:`ExperienceBuffer._mark_loaded_conversations_ended`
#: が読み込み時に全エントリへ立てるため構造的に恒真へ寄り (実測 205/206 = 99.5%)、
#: ``turn_outcome`` も実測 206/206 が ``"success"`` だった。これらを加点項に置くと
#: fitness が上限へ張り付いて選択圧が消える。観測された欠陥の率だけを見れば、
#: 加点シグナルの恒真性に引きずられない。
#:
#: :class:`~backend.free.learning.generation_param_evolver.GenerationParamEvolver`
#: が同じ問題に対して先に採った設計と揃えてある。
DEFECT_WEIGHTS: dict[str, float] = {
    "user_correction": 1.0,
    "assistant_self_retraction": 1.0,
    "rephrased_query": 0.6,
    "tool_routing_false_negative": 0.5,
    "tool_routing_false_positive": 0.5,
}


def _defect_rate_fitness(
    experiences: list[dict],
    weights: dict[str, float] | None = None,
) -> float:
    """観測された欠陥の重み付き率から fitness を返す (1.0 = 欠陥なし)。

    呼出側が非空を保証すること (サンプル数の下限判定は各 fitness 関数の責務)。
    """
    table = DEFECT_WEIGHTS if weights is None else weights
    defects = 0.0
    for e in experiences:
        signals = e.get("signals") or {}
        for key, weight in table.items():
            if signals.get(key):
                defects += weight
    return max(0.0, min(1.0, 1.0 - defects / len(experiences)))


# ── コスト項 ──


def _prompt_cost_samples(experiences: list[dict]) -> list[float]:
    """ターンごとの **再プリフィル率** ``(prompt - cached) / prompt`` を集める。

    生のプロンプトトークン数ではなく比率を使う理由は 3 つ:

    1. 無次元 [0, 1] なので、正規化定数 (context_size 等) を fitness 関数へ
       持ち込まずに品質項と合成できる。
    2. 再プリフィルは実測でレイテンシの支配項 (ライブ監査で 64〜80%)。
    3. **キャッシュに乗る膨張を罰しない**。静的な前置きを増やしても接頭辞
       キャッシュに乗るので比率は上がらない。一方 top_k を上げて増える取得
       テキストはクエリ依存で毎ターン再計算されるため比率が上がる。罰したい
       種類の膨張だけが効く。

    ``prompt_tokens`` / ``cached_prompt_tokens`` が揃っていないエントリは除外する
    (未計測を 0 と扱わない)。
    """
    samples: list[float] = []
    for e in experiences:
        signals = e.get("signals") or {}
        prompt = signals.get("prompt_tokens")
        cached = signals.get("cached_prompt_tokens")
        if not isinstance(prompt, int) or not isinstance(cached, int):
            continue
        if prompt <= 0:
            continue
        samples.append(max(0.0, min(1.0, (prompt - cached) / prompt)))
    return samples


def _completion_cost_samples(experiences: list[dict]) -> list[float]:
    """長文生成の **予算消費率** を集める。

    ``long_form_budget_used_pct`` は長文経路が自前で持つ「割り当て予算のうち
    どれだけ使ったか」で、既に [0, 100] 正規化済み。``unit_max_tokens`` /
    ``unit_target_tokens`` / ``max_extend_rounds`` を上げれば単調に増える =
    凍結を解くのに必要なコスト項として過不足がない。
    """
    samples: list[float] = []
    for e in experiences:
        signals = e.get("signals") or {}
        pct = signals.get("long_form_budget_used_pct")
        if not isinstance(pct, (int, float)) or isinstance(pct, bool):
            continue
        samples.append(max(0.0, min(1.0, float(pct) / 100.0)))
    return samples


def cost_samples(domain: str, experiences: list[dict]) -> list[float]:
    """ドメインに対応するコスト標本 (0.0 = 無コスト / 1.0 = 最大) を返す。

    解除対象 (:data:`MONOTONE_PARAMS`) を持たないドメイン (``router`` /
    ``search``) は空リスト = 品質項のまま。これらの残る進化キーはプロンプト量に
    影響しないので、コスト項を足しても摂動と無相関なノイズが乗るだけになる。
    """
    if domain in _PROMPT_COST_DOMAINS:
        return _prompt_cost_samples(experiences)
    if domain in _COMPLETION_COST_DOMAINS:
        return _completion_cost_samples(experiences)
    return []


def is_cost_observable(domain: str, experiences: list[dict]) -> bool:
    """コスト項が ``fitness`` に実際に入るだけの標本があるか。

    ``True`` のとき :data:`MONOTONE_PARAMS` の凍結が解ける。品質項と同じ
    :data:`MIN_FITNESS_SAMPLES` を要求するのは、少数標本のコスト平均で
    単調性が解けたと誤判定すると膨張が再発するため。
    """
    return len(cost_samples(domain, experiences)) >= MIN_FITNESS_SAMPLES


def calc_fitness(domain: str, experiences: list[dict]) -> float | None:
    """ドメインに適した fitness 値を算出する

    ``(1 - COST_WEIGHT) * 品質 + COST_WEIGHT * (1 - コスト)``。コスト標本が
    :data:`MIN_FITNESS_SAMPLES` に満たない場合は**品質項のみ**へ縮退する
    (コスト未計測を 0 コストと扱わない)。縮退時は :func:`is_cost_observable`
    も ``False`` を返すので、単調パラメータは凍結されたままになる。

    Args:
        domain: ポリシードメイン
        experiences: そのモードの経験リスト

    Returns:
        fitness 値 [0.0, 1.0]。該当シグナルが無く評価不能なら ``None``
        (呼出側は履歴に積まず進化を skip する)。
    """
    fn = _FITNESS_FUNCTIONS.get(domain, _calc_fitness_default)
    quality = fn(experiences)
    if quality is None:
        return None

    samples = cost_samples(domain, experiences)
    if len(samples) < MIN_FITNESS_SAMPLES:
        return quality

    cost = sum(samples) / len(samples)
    blended = (1.0 - COST_WEIGHT) * quality + COST_WEIGHT * (1.0 - cost)
    return max(0.0, min(1.0, blended))


def _calc_fitness_router(experiences: list[dict]) -> float | None:
    """router fitness: ルーティング精度（ユーザー修正率の逆数）

    修正・言い換えが少ないほど高スコア。サンプル不足は評価不能 (None)。
    """
    total = len(experiences)
    if total < MIN_FITNESS_SAMPLES:
        return None

    bad = sum(
        1 for e in experiences
        if (e.get("signals", {}).get("user_correction")
            or e.get("signals", {}).get("rephrased_query"))
    )
    return 1.0 - (bad / total)


def _calc_fitness_memory(experiences: list[dict]) -> float | None:
    """memory fitness: 観測された欠陥の率 (1.0 = 欠陥なし)。

    旧実装は ``conversation_ended`` を加点の主項に置いていたが、この信号は
    :meth:`ExperienceBuffer._mark_loaded_conversations_ended` が読み込み時に
    全エントリへ立てるため構造的に恒真へ寄る。結果 fitness は clamp 上限の
    1.0 に張り付き、**選択圧ゼロ・ロールバック不能**の乱歩になっていた
    (実測 2026-08-18、経験 206 件: 現行 span=0.0000 / distinct=1)。

    欠陥率へ差し替えると同じ実データで span=0.0203 / distinct=9 の判別力が出る。
    サンプル不足は評価不能 (None)。
    """
    if len(experiences) < MIN_FITNESS_SAMPLES:
        return None
    return _defect_rate_fitness(experiences)


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
    if len(scores) < MIN_FITNESS_SAMPLES:
        return None

    return max(0.0, min(1.0, sum(scores) / len(scores)))


def _calc_fitness_agent(experiences: list[dict]) -> float | None:
    """agent fitness: ステップ効率 (ループ数が少ないほど高スコア)。

    ``conversation_ended`` による +0.3 の加点は**外した**。この信号は実測 99.5%
    が True なうえ、``1/loops`` が 1.0 (= 単発ループ) のとき 1.3 → clamp 1.0 と
    なり、**単発ループのターンが全て厳密に 1.0 に潰れる**。ループ数という唯一の
    判別項を clamp の天井が消していた。

    agent 母集合 (``agent_loops > 0``) のサンプル不足は評価不能 (None)。実測
    2026-08-18 では 206 件中 1 件しかなく、この 1 件から fitness=1.0 が
    best_fitness へ焼き付いていた。
    """
    agent_exps = [
        e for e in experiences
        if e.get("signals", {}).get("agent_loops", 0) > 0
    ]
    if len(agent_exps) < MIN_FITNESS_SAMPLES:
        return None

    scores = []
    for e in agent_exps:
        signals = e.get("signals", {})
        loops = max(1, signals.get("agent_loops", 1))
        efficiency = 1.0 / loops
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

    if len(scores) < MIN_FITNESS_SAMPLES:
        return None

    return sum(scores) / len(scores)


def _calc_fitness_default(experiences: list[dict]) -> float | None:
    """デフォルト fitness: 観測された欠陥の率。サンプル不足は評価不能 (None)。

    新ドメイン追加時のフォールバック。旧実装は ``conversation_ended`` 加点
    ベースだったが、それは恒真で選択圧を持たない (``_calc_fitness_memory`` の
    docstring 参照)。既定を欠陥率にしておくことで、新ドメインが黙って乱歩を
    始めることを防ぐ。
    """
    if len(experiences) < MIN_FITNESS_SAMPLES:
        return None
    return _defect_rate_fitness(experiences)


_FITNESS_FUNCTIONS: dict[str, Callable[..., float | None]] = {
    "router": _calc_fitness_router,
    "memory": _calc_fitness_memory,
    "search": _calc_fitness_search,
    "agent": _calc_fitness_agent,
    "long_form": _calc_fitness_long_form,
}
