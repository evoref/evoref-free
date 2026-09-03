"""LearningScheduler: Level 1〜2 の学習サイクルを管理

Level 1: アイドル時のモード別システムプロンプト進化（Darwinian Evolver）
         + 補助タスクのタスク別プロンプト進化（§7.1.2, Pro 以上）
Level 2: 夜間 SPSA による LoRA 微調整（Pro 専用、Level2Runner で注入）
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from backend.i18n_helper import msg
from backend.io import atomic_write_text
from backend.log_config import get_logger
from backend.policy_helpers import get_policy_value
from backend.utils import utc_now
from backend.free.core.session_mode import is_chat_mode, is_create_mode
from backend.free.learning.fitness import defect_rate_fitness
from backend.free.learning.learning_state_store import (
    LearningState,
    LearningStateStore,
)
from backend.free.learning.level0_instant import ExperienceBuffer
from backend.free.learning.level1_session import (
    Level1Session,
    PriorityRequest,
    archive_session,
    discard_active_session,
    load_active_session,
    save_active_session,
)
from backend.free.optimizer.prompt_evolver import (
    PROMPT_DEFECT_WEIGHTS,
    PromptEvolver,
)
from backend.free.agent.prompt_manager import SystemPromptManager
from backend.free.agent.prompt_utils import (
    restore_protected_sections,
    validate_protected_sections,
)

if TYPE_CHECKING:
    from backend.free.core.policy_interpreter import PolicyInterpreter
    from backend.free.learning.critique_synthesizer import CritiqueResult

logger = get_logger("learning.scheduler")


# 変異 (prompt mutation) が systemic に失敗したと判定する失敗率閾値。
# この比率以上の変異が失敗し、かつ採用改善が 0 の場合、learning_cycle_l1 の
# success を True と偽らない (ReadTimeout 等で実質何も学習できていないため。
# 一方、変異は成功したが改善が無い「健全な収束」は success=True を維持する)。
_MUTATION_DEGRADED_RATE = 0.5

#: 1 つの Level1Session が yield できる上限。これを超えたら session を破棄して
#: 経験カーソルを進める (同じ snapshot を何日も抱えたまま resume を繰り返さない)。
MAX_SESSION_YIELDS = 5

#: 採用したプロンプトを事後監視する Level 1 窓の数と、rollback を発火させる
#: 欠陥率 fitness の悪化幅 (採用時基準との差)。
PROMPT_ROLLBACK_WINDOWS = 2
PROMPT_ROLLBACK_EPSILON = 0.02


def _aggregate_mutation_health(results: dict) -> tuple[int, int]:
    """Level 1 結果集合から変異 (mutation) の総試行/総失敗回数を集計する。

    ``results`` の値は ``EvolutionResult`` (prompt 進化) か結果 dict
    (extra phase 別 dict) のいずれか。変異統計を持たない phase
    (aux/embed/extra) は 0 として扱う。``(attempts, failures)`` を返す。
    """
    attempts = 0
    failures = 0
    for r in results.values():
        if isinstance(r, dict):
            attempts += int(r.get("mutation_attempts", 0) or 0)
            failures += int(r.get("mutation_failures", 0) or 0)
        else:
            attempts += int(getattr(r, "mutation_attempts", 0) or 0)
            failures += int(getattr(r, "mutation_failures", 0) or 0)
    return attempts, failures


def _mutation_health_signals(results: dict, improved_count: int) -> tuple[bool, dict]:
    """変異の失敗状況から degraded 判定と outcome 用シグナルを返す。

    Returns:
        ``(mutations_degraded, signals)``。``mutations_degraded`` は
        「変異が systemic に失敗し、かつ改善が 0」のとき ``True``。``signals`` は
        ``quality_signals`` に合流させる観測値 dict。
    """
    attempts, failures = _aggregate_mutation_health(results)
    rate = (failures / attempts) if attempts else 0.0
    degraded = attempts > 0 and rate >= _MUTATION_DEGRADED_RATE and improved_count == 0
    signals = {
        "mutation_attempts": attempts,
        "mutation_failures": failures,
        "mutation_failure_rate": round(rate, 3),
        "mutations_degraded": degraded,
    }
    return degraded, signals


class _CachedCritiqueProxy:
    """1 Level 1 run 内で ``CritiqueResult`` を再利用するための薄いプロキシ。

    **Step 11 (Critique-Synthesis)** で先に
    ``CritiqueSynthesizer.critique()`` を呼んだ結果を、後段の
    ``PromptEvolver._darwinian_evolve`` に渡す際に二重実行されないようにする。

    元の ``CritiqueSynthesizer`` と同一の ``critique()`` インターフェース
    のみを提供し、内部状態の差し替えはしない。
    """

    def __init__(self, result: "CritiqueResult") -> None:
        self._result = result

    async def critique(self, experiences: list[dict]):  # noqa: ARG002
        return self._result


class LearningScheduler:
    """Level 1〜2 学習サイクルのスケジューラ"""

    MODES = ["create", "chat"]

    def _init_level1_params(self, learning: dict) -> None:
        """Level 1 学習サイクル関連パラメータをポリシー優先で設定する。"""
        self.min_experiences: int = self._lp(
            "level1_min_experiences", learning.get("level1_min_experiences", 20),
        )
        self.generations: int = self._lp(
            "level1_generations", learning.get("level1_generations", 10),
        )
        self.population_size: int = self._lp(
            "level1_population_size", learning.get("level1_population_size", 5),
        )
        # Level 1 アイドル閾値
        self.level1_idle_minutes: float = self._lp(
            "level1_idle_minutes", learning.get("level1_idle_minutes", 30),
        )
        # level1_recheck_interval_sec は常駐ループ (SleepTimeScheduler) 側だけが
        # 読むので本クラスには持たない。level1_idle_minutes はポリシー解決値を
        # SleepTimeScheduler が本インスタンス経由で参照する (L-C1)。
        self.priority_threshold_ratio: float = learning.get(
            "priority_threshold_ratio", 0.5,
        )

    def _init_level2_params(self, learning: dict) -> None:
        """Level 2 (ベースモデル §7.5.1) パラメータを設定する。"""
        self.min_failures: int = learning.get("level2_min_failures", 50)
        # モード別の発火閾値 (未指定モードは min_failures にフォールバック)。
        # create は経験が溜まりにくく、chat と同じ閾値だと永久に発火しないか、
        # chat が回るたび無駄に評価されるかのどちらかになる。
        self.min_failures_by_mode: dict[str, int] = {
            str(k): int(v)
            for k, v in (learning.get("level2_min_failures_by_mode") or {}).items()
        }
        self.spsa_iterations: int = learning.get("level2_spsa_iterations", 500)
        self.sparse_params: int = learning.get("level2_sparse_params", 200)
        self.active_minutes: float = learning.get("active_minutes", 5)
        # Level 2 自動発火のタイミングゲート (SleepTimeWorker と同一 config 由来)。
        # ダッシュボードの発火条件表示で参照する (get_level2_status 経由)。
        self.level2_overdue_hours: float = float(
            learning.get("level2_overdue_hours", 24.0),
        )
        self.level2_recheck_interval_sec: float = float(
            learning.get("level2_recheck_interval_sec", 300),
        )
        # 連続で改善が採用されなかった target を延長クールダウンへ落とす閾値と、
        # そのときに使う待機時間。1 サイクル 1 時間規模の実推論最適化が空振りし
        # 続ける局面 (探索が局所最適に張り付いた状態) で 24h 間隔を回し続けても
        # 得るものが無いため、間隔を伸ばして次のデータ蓄積を待つ。
        self.level2_stale_streak: int = max(
            1, int(learning.get("level2_stale_streak", 2)),
        )
        self.level2_stale_backoff_hours: float = float(
            learning.get("level2_stale_backoff_hours", 72.0),
        )
        # 採用に必要な最小改善幅 (baseline_loss に対する相対比)。実推論 eval は
        # 決定論的なので微小差も「実差」ではあるが、少数ケースの held-out に対する
        # 0.0x% の改善は汎化ではなく当該ケースへの過適合で、版だけが増える。
        self.level2_min_improvement_ratio: float = max(
            0.0, float(learning.get("level2_min_improvement_ratio", 0.001)),
        )
        # SPSA の best が更新されないまま何反復続いたら打ち切るか (0 で無効)。
        # 実推論 eval は 1 反復 = 候補サーバ 2 起動と高価なため、空振りが確定
        # している探索を最後まで回さない。
        self.level2_early_stop_patience: int = max(
            0, int(learning.get("level2_early_stop_patience", 10)),
        )
        # 候補評価用の使い捨てサーバに --no-warmup を付けるか (旧 llama.cpp で
        # 非対応の場合のみ false)。
        self.level2_eval_no_warmup: bool = bool(
            learning.get("level2_eval_no_warmup", True),
        )
        # Level 2 最適化器切替
        self.optimizer_type: str = learning.get("optimizer", "spsa")
        self.cma_sigma0: float = learning.get("cma_sigma0", 0.1)
        self.cma_popsize: int = learning.get("cma_popsize", 0)
        # Level 2 初回 full-train (bootstrap)。既定 OFF (実 llama-server ロード検証後に有効化)
        self.bootstrap_enabled: bool = bool(learning.get("level2_bootstrap_enabled", False))
        self.bootstrap_rank: int = int(learning.get("level2_bootstrap_rank", 8))
        self.bootstrap_init_sigma: float = float(
            learning.get("level2_bootstrap_init_sigma", 1e-4)
        )
        self.bootstrap_min_failures: int = int(
            learning.get("level2_bootstrap_min_failures", 20)
        )
        # Level 2 base=C: control vector (既定 'lora' = 既存 SPSA/LoRA 経路、挙動変更なし)
        self.level2_base_method: str = learning.get("level2_base_method", "lora")
        # Level 2 base アダプタの (mode) パーティション粒度。既定 "model_mode"
        # (chat/create で 1 アダプタ共有、挙動変更なし)。"model_mode" で
        # Level2Runner が chat/create を交互に別サイクルとして訓練する。
        self.level2_adapter_partition: str = learning.get(
            "level2_adapter_partition", "model_mode",
        )
        # base=spsa-real-eval: 候補 LoRA を ephemeral サーバで実推論評価する
        self.base_realeval_spsa_iterations: int = int(
            learning.get("base_realeval_spsa_iterations", 30)
        )
        self.base_eval_scratch_port: int = int(
            learning.get("base_eval_scratch_port", 8091)
        )
        self.base_eval_loss_w1: float = float(learning.get("base_eval_loss_w1", 0.7))
        self.base_eval_loss_w2: float = float(learning.get("base_eval_loss_w2", 0.3))
        self.base_eval_max_tokens: int = int(learning.get("base_eval_max_tokens", 64))
        self.base_eval_max_cases: int = int(learning.get("base_eval_max_cases", 20))
        self.base_eval_health_timeout: int = int(
            learning.get("base_eval_health_timeout", 120)
        )
        self.cvector_method: str = learning.get("cvector_method", "pca")
        self.cvector_pca_batch: int = int(learning.get("cvector_pca_batch", 100))
        self.cvector_pca_iter: int = int(learning.get("cvector_pca_iter", 1000))
        # cvector_scale は適用時に scripts/launch_llama.py が config から直接読むため、
        # scheduler 属性としては保持しない (runtime reader 不在)。
        self.cvector_seed_pairs_file: str = learning.get("cvector_seed_pairs_file", "")
        self.cvector_min_experiences: int = int(
            learning.get("cvector_min_experiences", 40)
        )

    def _init_phase4_and_mutator_config(self, learning: dict) -> None:
        """トークン予算進化 (phase4) の設定。

        **現状これらのノブは inert。** phase4 は候補比率ごとに異なる outcome
        シグナルが経験に残っていない (生成時の比率が記録されない) ため候補を
        識別できず、``no_outcome_signal`` で skip する (L-A1)。config スキーマ
        (``backend/schemas/learning.py``) のキーは互換のため残す。per-candidate
        の outcome が配線されたら再び読む。
        """
        self.budget_generations: int = learning.get("budget_generations", 5)
        self.budget_sigma: float = learning.get("budget_sigma", 0.05)
        self.budget_max_total_ratio: float = learning.get(
            "budget_max_total_ratio", 0.7,
        )

    def _init_lazy_injected_slots(self) -> None:
        """`lifespan` / Pro プラグインから後注入されるコンポーネントの初期値を設定。"""
        # エンベッド検索指示プロンプト進化
        self._embedder = None
        self._embed_instruction_evolver = None
        # 候補 instruction の実測評価器 (EmbedEvalProtocol)。後注入され、
        # set_embedder 順に依存しないよう scheduler 側にも保持する。
        self._embed_eval = None
        # パターン重み進化
        self._learned_patterns = None
        # 生成パラメータ進化
        self._generation_param_evolver = None
        # ポリシーパラメータ進化
        self._policy_param_evolver = None
        # Critique-Synthesis Loop
        self._critique_synthesizer = None
        # Step 11 で生成された CritiqueResult のキャッシュ
        # (1 Level 1 run の中だけ有効。run の冒頭で None リセット)
        self._last_critique_result = None
        # Few-shot Pool
        self._fewshot_pool = None
        # 品質ゲート 還流パイプ
        self._feedback_pipe = None
        self._last_feedback_summary: dict = {}
        # ランタイム注入用 callable / runner
        self._task: asyncio.Task | None = None
        self._user_active_checker = None  # callable: () -> bool
        self._level2_runner = None  # Pro: Level2Runner インスタンス

    def _prompt_rule_delete_gate(self, line: str) -> bool:
        """Level 1 の delete を台帳の計数で裁く (f_04 §4.5.2)。"""
        can = getattr(self.prompt_manager, "can_delete_rule_text", None)
        if can is None:
            return False
        min_turns = int((self._config.get("prompt") or {}).get("rule_stats_min_turns", 200))
        return bool(can(line, min_turns=min_turns))

    def _init_runtime_state(self, prompt_manager: SystemPromptManager) -> None:
        """ランタイム mutable state + パス + Level1Session 関連の初期化。"""
        self._cancelled = False
        self._running = False
        # Level 2 学習タスク実行中の対象 ("base" | None)。
        # 「学習中」表示用に Level2Runner が set/clear する。
        self._running_target: str | None = None
        self._last_run: float = 0.0
        # target 別の最終 Level 2 実行時刻 (現在は "base" のみ)。単一 float 共有
        # だった旧仕様では、base (cvector) の失敗が別 target の overdue
        # 判定まで巻き込み、経験が閾値を超えていても 24h ブロックしていた
        # 回帰 (2026-07-18) の修正で target 別 dict に分離した。
        self._last_level2_run: dict[str, float] = {}
        # target 別の「連続で改善が採用されなかった回数」。`is_level2_overdue`
        # が延長クールダウンへ落とす判断に使う (採用成功で 0 へリセット)。
        self._level2_no_improve_streak: dict[str, int] = {}
        self._level1_run_count: int = 0
        self._last_level1_results: dict = {}
        self._fitness_history: dict[str, list[dict]] = {}
        self._prev_correction_rate: float | None = None
        self._prev_rag_usage_rate: float | None = None
        # mode → 採用直後プロンプトの事後監視レコード (L-A7、learning_state に永続化)
        self._prompt_adoptions: dict[str, dict] = {}
        # _get_filtered_experiences のメモ化 (バッファ長 / 末尾 timestamp /
        # ロード済みカートリッジ集合が変わらない限り再変換しない、L-D1)
        self._exp_cache_key: tuple | None = None
        self._exp_cache: list[dict] = []
        # 学習コンポーネントが束ねられている base モデル stem (起動時 = resolver の
        # active stem、rebind_learning_partition で更新)。ModelState と食い違えば
        # ランタイム切替が起きている (L-A10、_base_model_changed)。
        self._model_stem_at_init: str | None = (
            getattr(self._resolver, "active_model_stem", None)
            if self._resolver is not None else None
        )
        self._model_change_warned = False
        # ランタイム切替を検知したときの自己修復 (新モデル GGUF 名 → rebind 結果)。
        # ``_learning_rebind.install_rebind_hook`` が注入。None なら従来どおり
        # Level 1 を止めるだけ。
        self._partition_rebind_hook: Callable[[str], dict] | None = None
        # Level1Session / 優先キュー / 協調 yield 用の状態
        self._priority_queue: list[PriorityRequest] = []
        self._yield_event: asyncio.Event = asyncio.Event()
        self._bind_partition_paths(prompt_manager.prompt_dir)

    def _bind_partition_paths(self, prompt_dir: Path) -> None:
        """パーティション (prompt_dir) 依存の状態ファイルパスを導出する。"""
        self._state_file: Path = prompt_dir / "learning_state.json"
        self._active_session_file: Path = prompt_dir / "level1_session_active.json"
        self._level1_history_dir: Path = prompt_dir / "level1_history"

    def __init__(
        self,
        config: dict,
        experience_buf: ExperienceBuffer,
        prompt_manager: SystemPromptManager,
        cartridge_mgr=None,
        version_manager=None,
        eval_core_manager=None,
        debug_logger=None,
        policy: PolicyInterpreter | None = None,
        disabled: bool = False,
        resolver=None,
    ):
        self._policy = policy
        # embed_instruction の保存先解決用 (embedding モデル単位パーティション)。
        # 未注入 (レガシー構築 / 一部テスト) 時は prompt_manager.prompt_dir
        # (base パーティション) にフォールバックする。
        self._resolver = resolver
        learning = config.get("learning", {})

        # 設定値ロード (4 グループ)
        self._init_level1_params(learning)
        self._init_level2_params(learning)
        self._init_phase4_and_mutator_config(learning)

        # 必須コンポーネント参照を保持
        self._config = config
        self.experience_buf = experience_buf
        self.prompt_manager = prompt_manager
        self.cartridge_mgr = cartridge_mgr
        self.version_manager = version_manager
        self.eval_core_manager = eval_core_manager
        self._debug_logger = debug_logger
        self._evolver = PromptEvolver()
        prompt_cfg = config.get("prompt") or {}
        self._evolver.max_deletes_per_run = int(prompt_cfg.get("max_deletes_per_run", 1))
        self._evolver.delete_gate = self._prompt_rule_delete_gate

        # 自己学習無効化フラグ (--no-learning 経由)。True の場合 Level 1/2 サイクルは
        # 全て早期 return し副作用なし。get_status は ``is_disabled: True`` を返す
        self._disabled = disabled
        if disabled:
            logger.info(
                "LearningScheduler initialized in disabled mode "
                "(Level 1/2 tick / cancel are no-op)",
            )

        # 後注入スロットとランタイム状態
        self._init_lazy_injected_slots()
        self._init_runtime_state(prompt_manager)

        # 永続化済み状態の復元 (state_file が必要なため runtime_state 後)
        self._load_state()

    def _lp(self, key: str, default: int | float) -> int | float:
        """learning ポリシーからパラメータ取得（フォールバック付き）"""
        return get_policy_value(self._policy, "learning", key, default)

    @property
    def running(self) -> bool:
        return self._running

    def _load_state(self) -> None:
        """前回の学習実行時刻を永続ファイルから復元する。

        永続化 I/O は `LearningStateStore` に委譲し、本メソッドは
        ロード後の scheduler フィールドへの hydration に専念する。
        """
        state = LearningStateStore.load(self._state_file)
        if state is None:
            logger.info("Learning state file not found: %s", self._state_file)
            return
        self._last_run = state.last_level1_run
        self._last_level2_run = dict(state.last_level2_run)
        self._level2_no_improve_streak = dict(state.level2_no_improve_streak)
        self._level1_run_count = state.level1_run_count
        self._last_level1_results = state.last_level1_results
        self._fitness_history = state.fitness_history
        self._prev_correction_rate = state.prev_correction_rate
        self._prev_rag_usage_rate = state.prev_rag_usage_rate
        self._prompt_adoptions = dict(state.prompt_adoptions)
        # 優先キューを復元
        self._priority_queue = state.priority_queue
        logger.info(
            "Learning state loaded: L1=%.0f, L2=%s, runs=%d, path=%s",
            self._last_run, self._last_level2_run,
            self._level1_run_count, self._state_file,
        )

    def _save_state(self) -> None:
        """学習実行時刻を永続ファイルに保存する (`LearningStateStore` に委譲)"""
        try:
            LearningStateStore.save(
                LearningState(
                    last_level1_run=self._last_run,
                    last_level2_run=self._last_level2_run,
                    level2_no_improve_streak=self._level2_no_improve_streak,
                    level1_run_count=self._level1_run_count,
                    last_level1_results=self._last_level1_results,
                    fitness_history=self._fitness_history,
                    prev_correction_rate=self._prev_correction_rate,
                    prev_rag_usage_rate=self._prev_rag_usage_rate,
                    priority_queue=list(self._priority_queue),
                    prompt_adoptions=dict(self._prompt_adoptions),
                ),
                self._state_file,
            )
        except OSError as e:
            logger.warning("Failed to save learning state: %s (path=%s)", e, self._state_file)

    def _save_policy_evolver_state(self) -> None:
        """PolicyEvolver + ExplorationController の状態を永続化する。

        ``policy_evolver_state.json`` は **ランタイム評価状態** (fitness_history /
        decline_count / best_params / survived_count) であり、SemMem の policy ファクト
        (= 活性化された policy 値) とは別物。semmem モードでも保存する (#8 Part B)。
        従来は semmem モードでスキップしていたため、rollback の best_params 復元先や
        昇格用 survived_count が再起動ごとにリセットされていた。
        """
        if self._policy_param_evolver is None:
            return
        try:
            state_dir = self.prompt_manager.prompt_dir
            self._policy_param_evolver.save(
                state_dir / "policy_evolver_state.json",
            )
            self._policy_param_evolver._exploration.save(
                state_dir / "exploration_state.json",
            )
        except Exception as e:
            logger.warning("Failed to save policy evolver state: %s", e)

    def set_partition_rebind_hook(self, hook: Callable[[str], dict] | None) -> None:
        """ランタイム base 切替検知時の自己修復フック (新 GGUF 名 → rebind 結果) を注入する。"""
        self._partition_rebind_hook = hook

    def save_partition_state(self) -> None:
        """現在のパーティションへ学習状態を書き出す (rebind 前の退避)。

        learning_state / policy_evolver_state / exploration_state / (yaml モードの)
        fewshot_pool を ``prompt_manager.prompt_dir`` 配下へ保存する。prompt_dir を
        差し替えた後では旧パスが分からなくなるため、rebind は「退避 → prompt_dir
        差替え → :meth:`rebind_learning_partition`」の順で呼ぶ。
        """
        self._save_state()
        self._save_policy_evolver_state()
        pool = self._fewshot_pool
        if pool is not None and not pool.is_semmem_writeback_active():
            pool.save(self.prompt_manager.prompt_dir / "fewshot_pool.json")

    def rebind_learning_partition(
        self,
        *,
        model_stem: str | None,
        base_model_id: str,
        generation_deltas_file: Path | None = None,
    ) -> None:
        """ランタイム base 切替後、学習状態を新パーティションへ向け直す。

        前提: 呼出側が :meth:`save_partition_state` で旧パーティションへ退避し、
        ``prompt_manager.rebind_prompt_dir`` で prompt_dir を新パーティションへ
        差し替え済であること。ここでは prompt_dir から状態パスを再導出し、
        learning_state / PolicyParamEvolver (+ExplorationController) /
        FewShotPool / FeedbackPipe / GenerationParamEvolver を新モデルへ再スコープ
        して再ロードする。新パーティションが空なら各状態は初期値 (ゼロから学習)。
        Level 1 実行中は呼ばない (``running`` を呼出側で確認する)。
        """
        prompt_dir = self.prompt_manager.prompt_dir
        self._bind_partition_paths(prompt_dir)

        # learning_state: 永続化対象フィールドを初期値へ戻してから新パスを読む
        self._last_run = 0.0
        self._last_level2_run = {}
        self._level2_no_improve_streak = {}
        self._level1_run_count = 0
        self._last_level1_results = {}
        self._fitness_history = {}
        self._prev_correction_rate = None
        self._prev_rag_usage_rate = None
        self._prompt_adoptions = {}
        self._priority_queue = []
        self._exp_cache_key = None
        self._exp_cache = []
        self._load_state()

        evolver = self._policy_param_evolver
        if evolver is not None:
            evolver.set_base_model_id(base_model_id)
            evolver.reset_evaluation_state()
            evolver.load(prompt_dir / "policy_evolver_state.json")
            exploration = getattr(evolver, "_exploration", None)
            if exploration is not None:
                exploration.reset()
                exploration.load(prompt_dir / "exploration_state.json")

        pool = self._fewshot_pool
        if pool is not None:
            pool.set_base_model_id(base_model_id)
            pool.reset()
            if pool.is_semmem_writeback_active():
                pool.bootstrap_from_semmem()
            else:
                pool.load(prompt_dir / "fewshot_pool.json")

        if self._feedback_pipe is not None:
            self._feedback_pipe.set_base_model_id(base_model_id)

        if (
            self._generation_param_evolver is not None
            and generation_deltas_file is not None
        ):
            self._generation_param_evolver.rebind_delta_file(generation_deltas_file)

        self._model_stem_at_init = model_stem
        self._model_change_warned = False
        logger.info(
            "Learning scheduler rebound to partition: stem=%s slug=%s dir=%s",
            model_stem, base_model_id, prompt_dir,
        )

    def record_level2_run(
        self, target: str = "base", *, improved: bool | None = None,
    ) -> None:
        """target ("base") の Level 2 実行時刻を記録して永続化する。

        Args:
            target: "base"
            improved: そのサイクルで改善版を採用できたか。``True`` で連続無改善
                ストリークをリセット、``False`` でインクリメントする。``None``
                (既定) は判定不能 (early return / 例外 / 手動記録) を意味し、
                ストリークを触らない — 空振りと区別できない事象でクールダウンを
                伸ばすと、データが揃った直後の 1 回目まで巻き添えで待たされる。
        """
        self._last_level2_run[target] = time.time()
        if improved is True:
            self._level2_no_improve_streak[target] = 0
        elif improved is False:
            self._level2_no_improve_streak[target] = (
                self._level2_no_improve_streak.get(target, 0) + 1
            )
        self._save_state()

    def level2_no_improve_streak(self, target: str = "base") -> int:
        """target の連続無改善回数 (採用に成功すると 0 に戻る)。"""
        return self._level2_no_improve_streak.get(target, 0)

    def level2_cooldown_hours(self, target: str = "base") -> float:
        """target に現在適用される Level 2 クールダウン時間 (h)。

        連続無改善が ``level2_stale_streak`` 回に達した target は
        ``level2_stale_backoff_hours`` へ落ちる。
        """
        if self.level2_no_improve_streak(target) >= self.level2_stale_streak:
            return max(self.level2_overdue_hours, self.level2_stale_backoff_hours)
        return self.level2_overdue_hours

    def is_level2_overdue(self, target: str = "base") -> bool:
        """target の Level 2 が overdue (クールダウン経過済み) か。

        Level 2 発火のタイミングゲートの **単一の述語**。常駐ループ
        (`SleepTimeScheduler._schedule_level2_loop`) の安価な事前チェックと、
        Level2Runner が実際に起動する target ごとの本ゲートの双方がこれを使う。

        両者が別々の判定を持っていた旧実装では、ループが「次に試す target」
        (= 実行されない target) で判定する一方で実際に走るのは base だったため、
        一度も実行されない target の経過時間が恒久的に ``inf`` となりゲートが
        開きっぱなしになっていた (実機で Level 2 が 10 時間連続稼働)。判定を
        実行 target と同じ軸へ揃えるのが本メソッドの存在理由。
        """
        return self.seconds_since_level2_run(target) >= (
            self.level2_cooldown_hours(target) * 3600.0
        )

    def seconds_since_level2_run(self, target: str = "base") -> float:
        """target の前回 Level 2 実行からの経過秒。未実行は inf。

        `_last_level2_run` は learning_state に永続化されるため、再起動を跨いで
        overdue 判定が継続する (SleepTimeScheduler の Level 2 常駐ループが参照)。
        target 別に分離しているため、一方の失敗がもう一方の overdue 判定を
        巻き込まない (2026-07-18 修正: 以前は単一値を共有していた)。
        """
        last = self._last_level2_run.get(target, 0.0)
        if last <= 0:
            return float("inf")
        return max(0.0, time.time() - last)

    def set_user_active_checker(self, checker) -> None:
        """ユーザーアクティブ判定関数を設定（Level 2 中断用）"""
        self._user_active_checker = checker

    def set_version_manager(self, version_manager) -> None:
        """ベースモデル LoRA バージョンマネージャを設定（Pro、lifespan から注入）。

        構築時は None 既定で、Pro ハンドラ登録後に running scheduler へ後注入する。
        未注入だと Level 2 (base) は version_manager is None で停止する。
        """
        self.version_manager = version_manager

    def set_eval_core_manager(self, eval_core_manager) -> None:
        """評価コアマネージャを設定（Pro、lifespan から注入）。"""
        self.eval_core_manager = eval_core_manager

    def set_embedder(self, embedder) -> None:
        """embed instruction 進化用のエンベッダを設定（lifespan から注入）

        supports_instructions() が True の場合のみ embed instruction 進化が有効になる
        """
        self._embedder = embedder
        if hasattr(embedder, "supports_instructions") and embedder.supports_instructions():
            from backend.free.optimizer.embed_instruction_evolver import (
                EmbedInstructionEvolver,
            )
            # set_embed_eval が先に呼ばれていれば実測評価器を引き継ぐ。
            self._embed_instruction_evolver = EmbedInstructionEvolver(
                embed_eval=self._embed_eval,
            )
            logger.info(
                "embed instruction evolver enabled (real_eval=%s)",
                self._embed_eval is not None,
            )
            # 起動時反映: 過去に進化採用された embed_instruction.md が存在すれば
            # runtime embedder へ適用する (再起動後も進化結果を runtime に乗せる)。
            # ファイル不在時は何もしない (config 既定 instruction を尊重し、
            # _load_embed_instruction がデフォルトを書き出す副作用も避ける)。
            # 保存先は _embed_instruction_dir() (embedding モデル単位パーティション)
            # と一致させる必要がある — prompt_manager.prompt_dir 直読みだと
            # _save_embed_instruction が書いた新パーティションを見落として
            # 旧 (base パーティション) の古い内容で runtime を上書きしてしまう。
            ei_path = self._embed_instruction_dir() / "embed_instruction.md"
            if ei_path.exists():
                try:
                    txt = ei_path.read_text(encoding="utf-8").strip()
                    if txt:
                        self._apply_embed_instruction_runtime(txt)
                        logger.info("Loaded evolved embed instruction at startup")
                except OSError as e:
                    logger.warning("Failed to load embed_instruction.md at startup: %s", e)
        else:
            logger.info("embed instruction evolver disabled (embedder lacks instruction support)")

    def set_embed_eval(self, embed_eval) -> None:
        """候補 instruction の実測評価器を注入する (wire_pillars から後注入)。

        set_embedder の前後どちらで呼ばれても整合するよう scheduler 側に保持し、
        既に evolver が生成済みなら即反映する。
        """
        self._embed_eval = embed_eval
        if self._embed_instruction_evolver is not None:
            self._embed_instruction_evolver.set_embed_eval(embed_eval)

    def set_learned_patterns(self, learned_patterns) -> None:
        """学習済みパターンストアを設定（lifespan から注入）"""
        self._learned_patterns = learned_patterns
        logger.info("learned pattern evolver enabled (%d patterns)", learned_patterns.count)

    def set_generation_param_evolver(self, evolver) -> None:
        """生成パラメータ進化器を設定（lifespan から注入）"""
        self._generation_param_evolver = evolver
        logger.info("generation param evolver enabled")

    def set_critique_synthesizer(self, synthesizer) -> None:
        """Critique-Synthesis Loop を設定"""
        self._critique_synthesizer = synthesizer
        logger.info("Critique-Synthesis Loop enabled")

    def set_fewshot_pool(self, pool) -> None:
        """Few-shot 候補プールを設定"""
        self._fewshot_pool = pool
        logger.info("Fewshot pool enabled (total: %d)", pool.total_count)

    def set_policy_param_evolver(self, evolver) -> None:
        """ポリシーパラメータ進化器を設定"""
        self._policy_param_evolver = evolver
        logger.info("Policy param evolver enabled")
        # 経路 3: feedback_pipe が先に設定済なら即プロバイダを結線
        self._wire_feedback_success_provider()

    def set_feedback_pipe(self, pipe) -> None:
        """品質ゲート 還流パイプを設定"""
        self._feedback_pipe = pipe
        logger.info(
            "Feedback pipe enabled=%s", getattr(pipe, "enabled", False),
        )
        # FewShotPool / CritiqueSynthesizer が先に注入されていれば同期
        if pipe is not None:
            pipe.set_components(
                fewshot_pool=self._fewshot_pool,
                critique_synthesizer=self._critique_synthesizer,
            )
        self._wire_feedback_success_provider()

    def _wire_feedback_success_provider(self) -> None:
        """feedback_pipe と policy_param_evolver の両方が揃ったら fitness プロバイダを結線する"""
        pipe = self._feedback_pipe
        evolver = self._policy_param_evolver
        if pipe is None or evolver is None:
            return
        if not getattr(pipe, "enabled", False):
            return
        evolver.set_semmem_success_provider(
            pipe.compute_semmem_success_rate,
            pipe.weight_semmem_success,
        )
        logger.info(
            "Wired feedback_pipe -> policy_evolver success provider (weight=%.2f)",
            pipe.weight_semmem_success,
        )

    def set_level2_runner(self, runner) -> None:
        """Level 2 Runner を設定する（Pro プラグインから注入）

        Args:
            runner: Level2Runner インスタンス（check_and_run メソッドを持つ）
        """
        self._level2_runner = runner
        logger.info("Level 2 runner set")

    def set_learn_components(self, components) -> None:
        """補助タスク + Level 2 拡張コンポーネントを一括注入する

        旧 ``inject_learn_components(scheduler, buf, vm, eval_mgr, ...)`` の
        位置引数 7 本を :class:`~backend.free.learning.protocols.LearnComponentsProtocol`
        インスタンス 1 本に集約する。Protocol の各 Property から
        scheduler の個別フィールドに hydrate する。Protocol の契約どおり、
        Free 版 (:class:`~backend.free.learning.protocols.NoopLearnComponents`)
        では全 Property が ``None`` を返すため本メソッドは no-op に近い動作
        となる。

        Args:
            components: :class:`LearnComponentsProtocol` 準拠インスタンス
                (Pro 版は ``backend.pro.learn_components.ProLearnComponents``,
                Free 版は ``NoopLearnComponents``)。
        """
        if components.version_manager is not None:
            self.version_manager = components.version_manager
        # Level 2 拡張 (Pro 専用): level2_runner は従来の setter 経由でも
        # 注入可能。ここで Protocol 側の値が非 None であれば優先する。
        if components.level2_runner is not None:
            self._level2_runner = components.level2_runner
        logger.info(
            "Learn components injected: vm=%s, l2_runner=%s",
            self.version_manager is not None,
            self._level2_runner is not None,
        )

    def resolve_idle_task_client(self, llm_client=None):
        """批評合成 / few-shot 品質採点に使うクライアントを解決する。

        ベースモデルを `AuxClient` 越しに使う。両 purpose
        (`critique_synthesis` / `fewshot_quality_score`) は `PURPOSE_SCHEMAS`
        登録済みなので文法制約経路を通る。ベースが未接続 / 文法制約非対応なら
        `None` を返し、呼出側が no-op になる。
        """
        local = getattr(llm_client, "local", llm_client)
        if local is None or not hasattr(local, "generate_constrained"):
            return None
        from backend.free.llm.aux_client import AuxClient

        return AuxClient(
            local, config=self._config, debug_logger=self._debug_logger,
        )

    def cancel(self, *, graceful: bool = True) -> None:
        """学習を中断する（f_04 §7.1）

        Args:
            graceful: True (デフォルト) の場合、_yield_event を立てるだけで
                現世代を完走させ進捗を session に保存する経路を取る。
                ユーザー入力時の通常経路。
                False の場合、上記に加えて task.cancel() を呼び、
                CancelledError → finally で session 保存を試みる。
                シャットダウン時専用。
        """
        if self._disabled:
            # 学習サイクルが走っていないため cancel すべきものが無い
            return
        self._cancelled = True
        self._yield_event.set()
        if not graceful and self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None
            logger.info("Learning task cancelled (destructive)")
        elif graceful:
            logger.info("Learning yield requested (graceful)")

    # ── 協調 yield 用ヘルパー ────────

    def should_yield(self) -> bool:
        """Evolver から呼ばれる協調 yield 判定（f_02 §4.3）

        以下のいずれかが True なら yield すべきと返す:
        - _yield_event が立っている（cancel/ユーザー入力）
        - user_active_checker が True（in_flight_chat_count > 0 等）
        """
        if self._yield_event.is_set():
            return True
        if self._user_active_checker is not None:
            try:
                if self._user_active_checker():
                    return True
            except Exception as e:
                logger.warning("user_active_checker raised: %s", e)
        return False

    def reset_yield_event(self) -> None:
        """新しい Level 1 セッション開始時に呼び、yield フラグをクリアする"""
        self._yield_event.clear()
        self._cancelled = False

    # ── 優先キュー ────────────────────

    def push_priority_request(self, request: PriorityRequest) -> int:
        """優先キューに要求を追加する。

        同じ `reason` の既存要求は最新値で上書きされる（冪等）。

        Returns:
            push 後のキュー長
        """
        self._priority_queue = [
            r for r in self._priority_queue if r.reason != request.reason
        ]
        self._priority_queue.append(request)
        self._save_state()
        logger.info(
            "Priority request pushed: reason=%s relax=%.2f queue_len=%d",
            request.reason, request.relax_ratio, len(self._priority_queue),
        )
        return len(self._priority_queue)

    def peek_priority_request(self) -> PriorityRequest | None:
        """先頭の優先要求を取得（pop しない）。空なら None"""
        return self._priority_queue[0] if self._priority_queue else None

    def pop_priority_request(self) -> PriorityRequest | None:
        """先頭の優先要求を取り出す。空なら None"""
        if not self._priority_queue:
            return None
        req = self._priority_queue.pop(0)
        self._save_state()
        return req

    @property
    def priority_queue_length(self) -> int:
        return len(self._priority_queue)

    def priority_queue_snapshot(self) -> list[PriorityRequest]:
        """API/CLI 表示用のスナップショット（コピー）"""
        return list(self._priority_queue)

    # ── Level1Session ヘルパー ──────

    def has_active_session(self) -> bool:
        """SUSPENDED な Level1Session が存在するかどうか"""
        return self._active_session_file.exists()

    def load_active_session(self) -> Level1Session | None:
        """active session ファイルがあればロードする"""
        return load_active_session(self._active_session_file)

    def save_active_session(self, session: Level1Session) -> None:
        """SUSPENDED な session を永続化する（yield 時、フェーズ完了時に呼ぶ）"""
        save_active_session(self._active_session_file, session)

    def archive_active_session(self, session: Level1Session) -> Path:
        """完了した session を history に移動する"""
        return archive_session(
            self._active_session_file, self._level1_history_dir, session,
        )

    def discard_active_session(self) -> None:
        """active session ファイルを破棄する（復旧不能ケース）"""
        discard_active_session(self._active_session_file)

    # ── 経験数判定 ──────────────────

    def has_enough_experiences(
        self,
        *,
        strict: bool = True,
        relax_ratio: float | None = None,
    ) -> bool:
        """Level 1 を起動するのに十分な経験があるかを判定する。

        Args:
            strict: True なら通常閾値、False なら priority_threshold_ratio
                による緩和率を適用する。
            relax_ratio: 明示的な緩和率（PriorityRequest.relax_ratio）。
                指定された場合は strict よりも優先される。
        """
        safe_exp = self._get_filtered_experiences()
        if relax_ratio is not None:
            ratio = relax_ratio
        elif strict:
            ratio = 1.0
        else:
            ratio = float(self.priority_threshold_ratio)
        threshold = max(1, int(self.min_experiences * ratio))
        return len(safe_exp) >= threshold

    def has_enough_new_experiences(self) -> bool:
        """前回 Level 1 実行以降の新規経験数が min_experiences 以上か判定する。

        通常 idle トリガー専用ガード。前回成功実行 (`_last_run`) 以降に
        追加された経験が閾値に満たない場合は同じデータで Level 1 を
        空回しさせないために False を返す。
        `_last_run == 0.0`（初回）の場合は全経験を新規扱いするため、
        実質 `has_enough_experiences(strict=True)` と同じ挙動になる。

        Note:
            - resume / priority queue 経路では適用しない（明示的トリガーや
              既存セッション再開のため、新規経験有無で握り潰すべきでない）。
            - カートリッジ unload で除外された経験は new カウントから
              落ちる（`_get_filtered_experiences` 経由）。
        """
        return self._get_new_experience_count() >= self.min_experiences

    # ── Level1Session 駆動の進化エントリ ──

    def _load_or_create_level1_session(
        self, reason: str, relax_threshold: bool,
    ) -> Level1Session | dict:
        """既存の SUSPENDED session を復元するか、新規作成する。

        経験数が閾値未満なら ``{"skipped": True, ...}`` の dict を返す。
        正常時は `Level1Session` を返す。
        """
        session = self.load_active_session()
        if session is not None:
            logger.info(
                "Level 1 session resumed: id=%s reason=%s yield_count=%d "
                "completed_phases=%s",
                session.session_id, session.reason, session.yield_count,
                session.completed_phases,
            )
            return session

        safe_exp = self._get_filtered_experiences()
        ratio = float(self.priority_threshold_ratio) if relax_threshold else 1.0
        min_required = max(1, int(self.min_experiences * ratio))
        if len(safe_exp) < min_required:
            return {
                "skipped": True,
                "reason": "insufficient_experiences",
                "available": len(safe_exp),
                "required": min_required,
            }
        cart_ids: list[str] = []
        if self.cartridge_mgr is not None:
            cart_ids = sorted(self.cartridge_mgr.loaded.keys())
        session = Level1Session.new(
            cartridge_ids=cart_ids,
            experiences=safe_exp,
            reason=reason,
            experience_cutoff=self._last_run,
        )
        self.save_active_session(session)
        logger.info(
            "Level 1 session started: id=%s reason=%s experiences=%d",
            session.session_id, session.reason, len(safe_exp),
        )
        return session

    @staticmethod
    def _format_modes_summary(results: dict) -> dict:
        """`evolve_all_modes` の結果を JSON-friendly な summary dict に変換する。"""
        return {
            m: {
                "generations_run": r.generations_run,
                "initial_fitness": r.initial_fitness,
                "final_fitness": r.final_fitness,
                "yielded": r.yielded,
            }
            for m, r in results.items()
        }

    def _finalize_completed_level1(
        self,
        session: Level1Session,
        results: dict,
        *,
        extra_results: dict[str, dict] | None = None,
        experiences: list[dict] | None = None,
        skipped_phases: list[str] | None = None,
    ) -> str:
        """全モード完了時の後処理: 最良 prompt 保存 + state 更新 + session archive.

        採用ゲート: fitness が initial を上回った mode のみ update_evolved する。
        無改善 (final <= initial) で採用すると、目的関数を改善しないまま version を
        bump し続ける (embed_instruction 系の採用パスと挙動を揃える)。

        採用したプロンプトは ``_prompt_adoptions`` に基準 (採用時の欠陥率 fitness と
        rollback 先 version) を記録し、以後の Level 1 完了時に
        :meth:`_check_prompt_adoptions` が事後監視する (L-A7)。

        ``skipped_phases`` (cancel で飛ばした extra phase) が残っていれば
        ``_last_run`` を進めない — 同じ経験でもう一度 phase を回す機会を残す (L-A15)。
        """
        experiences = experiences or []
        rolled_back = self._check_prompt_adoptions(experiences)
        for mode, result in results.items():
            if result.final_fitness <= result.initial_fitness:
                logger.info(
                    "Level 1 %s: no adoption (fitness %.4f → %.4f, no improvement)",
                    mode, result.initial_fitness, result.final_fitness,
                )
                continue
            if mode in rolled_back:
                # 直前に rollback した mode の候補は rollback 対象から派生した
                # ものなので、このサイクルでは採用しない。
                logger.info(
                    "Level 1 %s: adoption skipped (prompt rolled back this cycle)",
                    mode,
                )
                continue
            try:
                rollback_to = self.prompt_manager.get_meta(mode).version
                self.prompt_manager.update_evolved(
                    mode,
                    result.best_candidate.text,
                    result.final_fitness,
                )
                self._record_prompt_adoption(mode, rollback_to, experiences)
                logger.info(
                    "Level 1 %s: adopted evolved prompt (fitness %.4f → %.4f)",
                    mode, result.initial_fitness, result.final_fitness,
                )
            except Exception as e:
                logger.warning("Failed to save prompt for %s: %s", mode, e, exc_info=True)

        stats: dict[str, dict] = {
            mode: {
                "improved": result.final_fitness > result.initial_fitness,
                "fitness_before": result.initial_fitness,
                "fitness_after": result.final_fitness,
                "mutation_attempts": getattr(result, "mutation_attempts", 0),
                "mutation_failures": getattr(result, "mutation_failures", 0),
            }
            for mode, result in results.items()
        }
        if extra_results:
            stats.update(extra_results)
        self._level1_finalize(
            experiences, stats,
            advance_cursor=not skipped_phases,
            skipped_phases=skipped_phases,
        )
        archived_path = self.archive_active_session(session)
        logger.info(
            "Level 1 session completed and archived: %s", archived_path,
        )
        return archived_path

    # ── 採用プロンプトの事後監視 (L-A7) ──

    def _record_prompt_adoption(
        self, mode: str, rollback_to: int, experiences: list[dict],
    ) -> None:
        """採用直後の基準を記録する (欠陥率 fitness と rollback 先 version)。"""
        mode_exp = [e for e in experiences if e.get("mode") == mode]
        baseline = defect_rate_fitness(mode_exp, weights=PROMPT_DEFECT_WEIGHTS)
        now = utc_now()
        self._prompt_adoptions[mode] = {
            "rollback_to": int(rollback_to),
            "baseline": round(baseline, 4) if baseline is not None else None,
            "adopted_at": now,
            "window_since": now,
            "windows": [],
        }

    def _check_prompt_adoptions(self, experiences: list[dict]) -> set[str]:
        """採用済みプロンプトの事後監視。悪化していれば rollback する。

        採用後に記録された経験 (``window_since`` より新しい) が
        ``min_experiences // 2`` 件以上溜まった Level 1 完了を 1 窓と数え、
        :data:`PROMPT_ROLLBACK_WINDOWS` 窓の欠陥率 fitness 平均が採用時基準を
        :data:`PROMPT_ROLLBACK_EPSILON` 超悪化していたら
        ``prompt_manager.rollback(mode, rollback_to)`` を呼ぶ。

        Returns:
            このサイクルで rollback した mode の集合。
        """
        rolled_back: set[str] = set()
        min_samples = max(1, self.min_experiences // 2)
        for mode, rec in list(self._prompt_adoptions.items()):
            since = str(rec.get("window_since") or "")
            new_exp = [
                e for e in experiences
                if e.get("mode") == mode and str(e.get("timestamp") or "") > since
            ]
            if len(new_exp) < min_samples:
                continue
            fitness = defect_rate_fitness(new_exp, weights=PROMPT_DEFECT_WEIGHTS)
            if fitness is None:
                continue
            windows = list(rec.get("windows") or [])
            windows.append(round(fitness, 4))
            rec["windows"] = windows
            rec["window_since"] = max(str(e.get("timestamp") or "") for e in new_exp)
            if len(windows) < PROMPT_ROLLBACK_WINDOWS:
                continue
            baseline = rec.get("baseline")
            mean = sum(windows) / len(windows)
            if baseline is not None and mean < baseline - PROMPT_ROLLBACK_EPSILON:
                try:
                    self.prompt_manager.rollback(mode, int(rec["rollback_to"]))
                    rolled_back.add(mode)
                    logger.warning(
                        "Level 1 %s: evolved prompt rolled back to v%03d "
                        "(post-adoption fitness %.4f < baseline %.4f - %.2f)",
                        mode, int(rec["rollback_to"]), mean, baseline,
                        PROMPT_ROLLBACK_EPSILON,
                    )
                except Exception as e:
                    logger.warning(
                        "Level 1 %s: prompt rollback failed: %s", mode, e, exc_info=True,
                    )
            else:
                logger.info(
                    "Level 1 %s: adopted prompt passed post-adoption check "
                    "(fitness %.4f vs baseline %s)",
                    mode, mean, baseline,
                )
            del self._prompt_adoptions[mode]
        return rolled_back

    async def _run_extra_optimizations(
        self,
        experiences: list[dict],
        llm_client,
        results: dict[str, dict],
        phase_durations: dict[str, float],
        experience_cutoff: float = 0.0,
    ) -> list[str]:
        """Level 1 追加最適化 (f_04 §4): prompt 進化以外の evolver 群を一括実行する

        各 phase は未注入コンポーネント・経験不足を自身でガードして no-op する。
        cancel (``_cancelled``) が立った以降の phase は実行せず名前を返す —
        呼出側は ``_last_run`` を進めない (同じ経験でやり直せる、L-A15)。

        Returns:
            cancel で飛ばした phase 名のリスト (空 = 全 phase 実行)。
        """
        skipped: list[str] = []

        async def _phase(name: str, fn) -> None:
            if self._cancelled:
                skipped.append(name)
                return
            out = fn()
            if asyncio.iscoroutine(out):
                await out

        await _phase("phase3_embed_instruction", lambda: self._level1_phase3_embed_instruction(
            experiences, llm_client, results, phase_durations,
        ))
        await _phase("phase4_token_budget", lambda: self._level1_phase4_token_budget(
            experiences, results, phase_durations,
        ))
        await _phase("phase6_generation_params", lambda: self._level1_phase6_generation_params(
            experiences, results, phase_durations,
        ))
        await _phase("phase7_tool_routing", lambda: self._level1_phase7_tool_routing(
            experiences, results, phase_durations, experience_cutoff,
        ))
        # Step 12 — PolicyEvolver 評価/書き戻し
        await _phase("step12_policy_evolver", lambda: self._step12_policy_evolver(
            experiences, results, phase_durations,
        ))
        # Step 14 — Few-shot プール GC
        # finalize() 前に実行することで、yaml モードでは保存される
        # fewshot_pool.json が GC 後の状態で書き出される。
        await _phase("step14_fewshot_gc", lambda: self._step14_fewshot_gc(
            results, phase_durations,
        ))
        if skipped:
            logger.info("Level 1 extra optimizations skipped (cancelled): %s", skipped)
        return skipped

    async def run_or_resume_level1(
        self,
        llm_client,
        *,
        reason: str = "idle",
        relax_threshold: bool = False,
    ) -> dict:
        """Level 1 進化を新規実行 or SUSPENDED から再開する。

        `_schedule_level1_loop` から呼ばれるエントリポイント
        本メソッドは prompt evolution 部分を Level1Session で管理し、
        全モード完了時に追加最適化 (`_run_extra_optimizations`: aux prompt /
        embed instruction / token budget / pattern weights / generation params /
        tool routing / policy evolver / fewshot GC) を続けて実行する。
        yield された場合は prompt evolution の再開を優先し、追加最適化は
        セッション完了まで持ち越す。

        Args:
            llm_client: 学習に使用する LLM クライアント
            reason: トリガー理由（"idle" / "manual" / "cartridge_unload" / "resume"）
            relax_threshold: True なら priority_threshold_ratio を適用

        Returns:
            {
                "session_id": str,
                "yielded": bool,
                "completed_phases": list[str],
                "modes": dict[mode, EvolutionResult.summary],
            }
        """
        if self._disabled:
            return {"skipped": True, "reason": "learning_disabled"}
        if llm_client is None:
            return {"skipped": True, "reason": "no_llm_client"}
        if self._running:
            return {"skipped": True, "reason": "already_running"}
        if self._base_model_changed():
            return {"skipped": True, "reason": "base_model_changed"}

        session_or_skip = self._load_or_create_level1_session(reason, relax_threshold)
        if isinstance(session_or_skip, dict):
            return session_or_skip
        session = session_or_skip

        # 1 cycle = 1 trace_id を発行 (bg_task wrapper の trace_id を退避)
        from backend.trace_context import generate_trace_id, trace_id_var
        trace_token = trace_id_var.set(generate_trace_id())
        started_at = utc_now()
        t0 = time.monotonic()
        success = False
        any_yielded = False
        results: dict = {}
        extra_results: dict[str, dict] = {}
        phase_durations: dict[str, float] = {}
        try:
            self.reset_yield_event()
            self._running = True
            # per-run の critique キャッシュを毎回リセット
            self._last_critique_result = None
            try:
                # Level 1 tick 冒頭で feedback_pipe を実行し、品質ゲート結果を
                # FewShotPool / 失敗クリティーク / policy fitness プロバイダへ還流する
                await self._run_feedback_pipe()
                # 進化前に経験から few-shot プールを補充する (f_04 §3)。session
                # snapshot は応答本文を持たない圧縮射影 (L-D3) なので live バッファ
                # から読む。resume 時は同一データの再投入になるが FewShotPool 側の
                # 多様性チェックが重複を弾く。
                await self._update_fewshot_pool_from_experiences(
                    self._get_filtered_experiences(), llm_client,
                )
                # Step 11 — Critique-Synthesis を base prompt 進化より前に 1 回だけ
                # 実行し、_CachedCritiqueProxy 経由で PromptEvolver に渡す
                # (モードごとの二重 critique を防ぐ)。
                await self._step11_critique_synthesis(
                    session.experience_snapshot, extra_results, phase_durations,
                )
                # 進化対象は名前プレフィックス無しの raw 本文。get_prompt() の
                # 出力 (プレフィックス付き) を渡すと進化後本文へ名前が焼き込まれ、
                # ランタイムの二重付与で設定名が反映されなくなる。
                prompt_texts = self._collect_mode_prompt_texts()
                if not prompt_texts:
                    logger.warning("No prompts available for evolution")
                tp = time.monotonic()
                results = await self._evolver.evolve_all_modes(
                    experiences=session.experience_snapshot,
                    prompt_texts=prompt_texts,
                    llm_client=llm_client,
                    generations=self.generations,
                    population_size=self.population_size,
                    critique_synthesizer=self._select_critique_for_phase1(),
                    session=session,
                    yield_check=self.should_yield,
                    save_session=self.save_active_session,
                )
                phase_durations["phase1_base_prompt"] = round(time.monotonic() - tp, 3)

                any_yielded = any(r.yielded for r in results.values())
                if any_yielded:
                    # session は save_session 経由で最新状態が保存済み
                    logger.info(
                        "Level 1 session yielded: id=%s completed_phases=%s "
                        "yield_count=%d",
                        session.session_id, session.completed_phases,
                        session.yield_count,
                    )
                    success = True
                    discarded = self._discard_if_yield_cap_hit(session)
                    return {
                        "session_id": session.session_id,
                        "yielded": True,
                        "discarded": discarded,
                        "completed_phases": list(session.completed_phases),
                        "modes": self._format_modes_summary(results),
                    }

                # 追加最適化 (f_04 §4)。prompt 進化の採用結果と session archive
                # を失わないため、失敗しても警告のみで finalize へ進む。
                skipped_phases: list[str] = []
                try:
                    skipped_phases = await self._run_extra_optimizations(
                        session.experience_snapshot, llm_client,
                        extra_results, phase_durations,
                        session.experience_cutoff,
                    )
                except Exception as exc:
                    logger.warning(
                        "Level 1 extra optimizations failed: %s: %s",
                        type(exc).__name__, exc, exc_info=True,
                    )

                self._finalize_completed_level1(
                    session, results,
                    extra_results=extra_results,
                    experiences=session.experience_snapshot,
                    skipped_phases=skipped_phases,
                )
                elapsed = round(time.monotonic() - t0, 3)
                self._level1_log_debug(
                    started_at, elapsed, phase_durations,
                    session.experience_snapshot, self._last_level1_results,
                )
                success = True
                return {
                    "session_id": session.session_id,
                    "yielded": False,
                    "completed_phases": list(session.completed_phases),
                    "modes": self._format_modes_summary(results),
                    "extra_phases": sorted(extra_results.keys()),
                    "skipped_phases": skipped_phases,
                }
            finally:
                self._running = False
        finally:
            # outcome.jsonl への結末記録 (evolve 限定)
            dl = self._debug_logger
            if dl is not None:
                duration_ms = (time.monotonic() - t0) * 1000
                improved_count = sum(
                    1 for r in results.values()
                    if r.final_fitness > r.initial_fitness
                )
                # 変異が systemic に失敗して実質何も学習できていない場合は
                # success を True と偽らない (健全な収束 = 改善なしとは区別する)。
                mutations_degraded, mutation_signals = _mutation_health_signals(
                    results, improved_count,
                )
                dl.log_outcome(
                    kind="learning_cycle_l1",
                    success=success and not mutations_degraded,
                    duration_ms=duration_ms,
                    quality_signals={
                        "entry": "run_or_resume_level1",
                        "session_id": session.session_id,
                        "reason": reason,
                        "yielded": any_yielded,
                        "completed_phases": len(session.completed_phases),
                        "modes_improved": improved_count,
                        "extra_phases": sorted(extra_results.keys()),
                        **mutation_signals,
                    },
                )
            trace_id_var.reset(trace_token)

    def _discard_if_yield_cap_hit(self, session: Level1Session) -> bool:
        """yield 回数が :data:`MAX_SESSION_YIELDS` に達した session を破棄する。

        破棄時は ``_last_run`` を現在時刻へ進めて経験カーソルを更新する (同じ
        snapshot で永久に resume を繰り返さない)。次の Level 1 は新規経験が
        ``min_experiences`` 溜まってから新しい session で始まる (L-A6)。
        """
        if session.yield_count < MAX_SESSION_YIELDS:
            return False
        logger.warning(
            "Level 1 session %s discarded after %d yields (cap=%d); "
            "advancing experience cursor",
            session.session_id, session.yield_count, MAX_SESSION_YIELDS,
        )
        self.discard_active_session()
        self._last_run = time.time()
        self._save_state()
        return True

    def _base_model_changed(self) -> bool:
        """base モデルが学習コンポーネントの束ね先と食い違っていないかを検知する (L-A10)。

        学習コンポーネント (experience / prompts / PolicyParamEvolver の subject)
        は ``_model_stem_at_init`` の stem で束ねられている。``ModelState.
        current_filename`` の stem がこれと食い違っていたら、ランタイム切替が
        ``_learning_rebind.rebind_base_learning`` を経由せずに起きている
        (/api/model/reload を踏まずに llama-server だけ差し替えた等)。
        自己修復フック (``set_partition_rebind_hook``) があればその場で rebind
        して続行し、無ければ WARNING を 1 回出して Level 1 を止める
        (旧パーティションへ新モデルの経験で書き込まないため)。
        """
        if self._resolver is None or not self._model_stem_at_init:
            return False
        try:
            from backend.free.core.model_migration import ModelState

            current = ModelState(
                self._resolver.resolve_local("model_state_file"),
            ).current_filename
        except Exception as e:
            logger.debug("base model change check skipped: %s", e)
            return False
        if not current:
            return False
        current_stem = Path(current).stem
        if current_stem == self._model_stem_at_init:
            return False
        if self._partition_rebind_hook is not None:
            try:
                result = self._partition_rebind_hook(Path(current).name)
            except Exception as e:
                logger.warning(
                    "Base model changed at runtime (%s -> %s) but self-heal "
                    "rebind failed: %s; Level 1 suspended",
                    self._model_stem_at_init, current_stem, e,
                )
                return True
            if result.get("rebound"):
                logger.info(
                    "Base model changed at runtime (%s -> %s); learning "
                    "partition rebound in place",
                    result.get("old_stem"), result.get("new_stem"),
                )
                return False
        if not self._model_change_warned:
            self._model_change_warned = True
            logger.warning(
                "Base model changed at runtime (%s -> %s); Level 1 is suspended "
                "until the learning partition is rebound (/api/model/reload) "
                "so the old learning partition is not written to",
                self._model_stem_at_init, current_stem,
            )
        return True

    async def _run_feedback_pipe(self) -> None:
        """品質ゲート 還流パイプを実行する

        パイプが未設定 / disabled の場合は何もしない。例外は握り潰して
        学習サイクル本体の継続を保証する。
        """
        pipe = self._feedback_pipe
        if pipe is None or not getattr(pipe, "enabled", False):
            return
        try:
            summary = await pipe.run(cycle_num=self._level1_run_count)
            self._last_feedback_summary = summary
        except Exception as exc:
            logger.warning("feedback_pipe.run failed: %s", exc)

    def _filter_experiences(self, experiences: list[dict]) -> list[dict]:
        """カートリッジ依存の経験を除外"""
        if self.cartridge_mgr is None:
            return experiences

        current_ids = set(self.cartridge_mgr.loaded.keys())
        filtered = []
        for exp in experiences:
            exp_ids = set(exp.get("cartridge_ids", []))
            if exp_ids <= current_ids or not exp_ids:
                filtered.append(exp)
        return filtered

    def _filter_experiences_for_level2(
        self,
        experiences: list[dict],
        current_model: str,
        mode: str | None = None,
    ) -> list[dict]:
        """Level 2 用: カートリッジフィルタ + ベースモデルフィルタ（§22.5.4）

        旧モデルで収集された経験は新モデルの LoRA 微調整に適さないため、
        現在のベースモデルで収集された経験のみを使用する。``mode`` 指定時は
        さらにそのモード ("chat"/"create") の経験のみに絞る (省略時は全モード
        横断、後方互換)。
        """
        filtered = self._filter_experiences(experiences)
        filtered = [
            e for e in filtered
            if e.get("base_model", current_model) == current_model
        ]
        if mode is not None:
            filtered = [e for e in filtered if e.get("mode") == mode]
        return filtered

    def _get_filtered_experiences(self) -> list[dict]:
        """経験バッファを dict リストに変換しカートリッジフィルタを適用

        1 tick に 3〜4 回呼ばれ、毎回 1000 件規模の dataclass → dict 変換を
        繰り返していた (L-D1)。バッファ長 / 末尾 timestamp / ロード済み
        カートリッジ集合をキーにメモ化する (ExperienceBuffer は世代カウンタを
        持たないため、この 3 つ組を安価な代替キーとする)。
        """
        entries = self.experience_buf.entries
        cart_ids = (
            frozenset(self.cartridge_mgr.loaded.keys())
            if self.cartridge_mgr is not None else None
        )
        key = (len(entries), entries[-1].timestamp if entries else None, cart_ids)
        if key == self._exp_cache_key:
            return list(self._exp_cache)
        raw_experiences = [
            {
                "timestamp": e.timestamp,
                "mode": e.mode,
                "query": e.query,
                "response_summary": e.response_summary,
                "response_full": e.response_full,
                "base_model": e.base_model,
                "cartridge_ids": e.cartridge_ids,
                "signals": asdict(e.signals),
            }
            for e in entries
        ]
        filtered = self._filter_experiences(raw_experiences)
        self._exp_cache_key = key
        self._exp_cache = filtered
        return list(filtered)

    @staticmethod
    def _select_new_experiences(
        experiences: list[dict], cutoff: float,
    ) -> list[dict]:
        """``cutoff`` (epoch 秒) より新しい経験だけを返す。0.0 なら全件。

        比較は ``_get_new_experience_count`` と同じ ISO タイムスタンプ文字列の
        辞書順比較。
        """
        if cutoff <= 0.0:
            return experiences
        cutoff_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cutoff))
        return [e for e in experiences if e.get("timestamp", "") > cutoff_str]

    def _get_new_experience_count(self) -> int:
        """前回 Level 1 実行以降の新規経験数を返す"""
        if self._last_run <= 0.0:
            return len(self._get_filtered_experiences())
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_run))
        return sum(
            1 for e in self._get_filtered_experiences()
            if e.get("timestamp", "") > cutoff
        )

    def get_status(self) -> dict:
        """現在の学習状態を返す"""
        safe_exp = self._get_filtered_experiences()
        total = len(safe_exp)
        new_count = self._get_new_experience_count()

        # モード別経験数
        chat_count = sum(1 for e in safe_exp if is_chat_mode(e.get("mode")))
        create_count = sum(1 for e in safe_exp if is_create_mode(e.get("mode")))

        # 最終 Level 0 記録日時
        last_level0_record: str | None = None
        if self.experience_buf.entries:
            last_level0_record = self.experience_buf.entries[-1].timestamp

        # 修正検出率
        corrections = sum(
            1 for e in safe_exp
            if e.get("signals", {}).get("correction_detected_by")
        )
        correction_rate = corrections / total if total > 0 else 0.0

        # RAG 利用率
        rag_used = sum(
            1 for e in safe_exp
            if e.get("signals", {}).get("rag_used")
        )
        rag_usage_rate = rag_used / total if total > 0 else 0.0

        # phase3 (embed_instruction) / phase4 (token_budget) が実際に判定する
        # 部分集合の経験数 (発火条件の可視化用。閾値は min_experiences // 2)。
        rag_score_experience_count = sum(
            1 for e in safe_exp
            if e.get("signals", {}).get("rag_top1_score") is not None
        )
        long_form_experience_count = sum(
            1 for e in safe_exp
            if e.get("signals", {}).get("long_form_used")
        )

        # 優先キューと active session のスナップショット
        priority_queue_view = [
            {
                "reason": r.reason,
                "requested_at": r.requested_at,
                "relax_ratio": r.relax_ratio,
                "payload": r.payload,
            }
            for r in self._priority_queue
        ]
        active = self.load_active_session()
        active_session_view: dict | None = None
        if active is not None:
            active_session_view = {
                "session_id": active.session_id,
                "started_at": active.started_at,
                "reason": active.reason,
                "completed_phases": list(active.completed_phases),
                "yield_count": active.yield_count,
                "cartridge_snapshot": list(active.cartridge_snapshot),
                "experience_count": len(active.experience_snapshot),
            }

        return {
            "running": self._running,
            "is_disabled": self._disabled,
            "experience_count": total,
            "new_experience_count": new_count,
            "min_experiences": self.min_experiences,
            "conditions_met": new_count >= self.min_experiences,
            "last_level1_run": self._last_run,
            "last_level2_run": self._last_level2_run,
            # 実行中の Level 2 対象 ("base"/None) と base の状態。
            # level2 は Pro (Level2Runner 注入時) のみ非 None。
            "running_target": self._running_target,
            "level2": (
                self._level2_runner.get_level2_status()
                if self._level2_runner is not None
                else None
            ),
            # Level 0 詳細
            "last_level0_record": last_level0_record,
            "experience_by_mode": {"chat": chat_count, "create": create_count},
            "correction_rate": round(correction_rate, 3),
            "rag_usage_rate": round(rag_usage_rate, 3),
            # phase3/phase4 部分集合条件の可視化 (閾値は min_experiences // 2)
            "rag_score_experience_count": rag_score_experience_count,
            "long_form_experience_count": long_form_experience_count,
            "phase_subset_min_experiences": max(1, self.min_experiences // 2),
            # 前回 Level 1 時点からのトレンド
            "prev_correction_rate": self._prev_correction_rate,
            "prev_rag_usage_rate": self._prev_rag_usage_rate,
            # Level 1 詳細
            "level1_run_count": self._level1_run_count,
            "last_level1_results": self._last_level1_results,
            "fitness_history": self._fitness_history,
            # 優先キュー / SUSPENDED session
            "priority_queue": priority_queue_view,
            "active_session": active_session_view,
        }

    def get_pro_status(self) -> dict:
        """Pro 専用の拡張学習状態を返す（探索フェーズ・ポリシー進化）"""
        result: dict = {}

        if self._policy_param_evolver is not None:
            result["policy_evolver"] = self._policy_param_evolver.get_status()

        if (
            self._policy_param_evolver is not None
            and hasattr(self._policy_param_evolver, "_exploration")
        ):
            result["exploration"] = (
                self._policy_param_evolver._exploration.get_status()
            )

        return result

    @staticmethod
    def _summarize_level1_results(results: dict[str, dict]) -> dict:
        """Level 1 結果を永続化用にサマリ化する"""
        summary: dict = {}
        executed_phases: list[str] = []
        for key, val in results.items():
            improved = val.get("improved", False)
            fb = val.get("fitness_before")
            fa = val.get("fitness_after")
            summary[key] = {
                "improved": improved,
                "fitness_before": round(fb, 4) if fb is not None else None,
                "fitness_after": round(fa, 4) if fa is not None else None,
            }
            executed_phases.append(key)
        summary["_executed_phases"] = executed_phases
        return summary

    def _append_fitness_history(self, results: dict[str, dict]) -> None:
        """Level 1 完了時にモード別 fitness 履歴を蓄積する（直近10件保持）"""
        for key, val in results.items():
            fa = val.get("fitness_after")
            if fa is None:
                continue
            if key not in self._fitness_history:
                self._fitness_history[key] = []
            self._fitness_history[key].append({
                # ``_level1_finalize`` が **この呼出の前に** カウンタを
                # インクリメント済みなので、今回の run 番号はそのままの値。
                # ``+ 1`` すると初回完了時点の点が run=2 になり、run=1 の点が
                # 永久に存在しないまま improvement-curve の X 軸が
                # ``level1_run_count`` と 1 ずれる (2026-08-18 ライブ監査:
                # level1_run_count=1 に対し fitness_history が run=2)。
                "run": self._level1_run_count,
                "fitness": round(fa, 4),
            })
            # 直近 10 件のみ保持
            if len(self._fitness_history[key]) > 10:
                self._fitness_history[key] = self._fitness_history[key][-10:]

    def _snapshot_rates(self, experiences: list[dict]) -> None:
        """Level 1 完了時の修正率・RAG利用率を記録（トレンド表示用）"""
        total = len(experiences)
        if total == 0:
            return
        corrections = sum(
            1 for e in experiences
            if e.get("signals", {}).get("correction_detected_by")
        )
        rag_used = sum(
            1 for e in experiences
            if e.get("signals", {}).get("rag_used")
        )
        self._prev_correction_rate = round(corrections / total, 3)
        self._prev_rag_usage_rate = round(rag_used / total, 3)

    def trigger_level1(self, llm_client) -> tuple[bool, str]:
        """Level 1 を手動トリガー (時間条件バイパス) — 優先キューへ ``manual`` を積む。

        以前はここから ``_run_level1`` (Level1Session / yield を持たない第 2 の
        Level 1 実装) を直接 create_task していた。常駐ループと同じ
        :meth:`run_or_resume_level1` に一本化し、本メソッドは要求を enqueue する
        薄いラッパにする (L-C3)。

        Returns:
            (triggered, message) — message は i18n 済み。
        """
        if self._disabled:
            # --no-learning 中は手動トリガーでも副作用 (プロンプト書戻し等) を起こさない。
            return False, msg("api.learning_disabled")

        if self._running:
            return False, msg("api.learning_cycle_running")

        if llm_client is None:
            return False, msg("api.learning_llm_unavailable")

        available = len(self._get_filtered_experiences())
        if available < self.min_experiences:
            return False, msg(
                "api.learning_not_enough_experiences",
                available=available, required=self.min_experiences,
            )

        position = self.push_priority_request(PriorityRequest(
            reason="manual",
            requested_at=time.time(),
            relax_ratio=1.0,
            payload={"level": "level1"},
        ))
        logger.info(
            "Level 1 manually triggered: queued (%d experiences, position=%d)",
            available, position,
        )
        return True, msg(
            "api.learning_level1_queued", count=available, position=position,
        )

    # ── Level 1 ヘルパー: 各サブステップの実装 ──

    async def _update_fewshot_pool_from_experiences(
        self, experiences: list[dict], llm_client=None,
    ) -> None:
        """Few-shot プール更新 + 未採点例への品質採点。

        採点はプールに入った例の **順位付け** にのみ効き、採用可否は決めない
        (``FewShotPool.score_pending_quality`` の docstring 参照)。採点クライアント
        が無ければ no-op で、従来どおり fitness だけで順位付けされる。
        """
        if self._fewshot_pool is None:
            return
        added = self._fewshot_pool.add_from_experiences(experiences)
        if added:
            logger.info("Fewshot pool updated: %d new examples added", added)
        await self._fewshot_pool.score_pending_quality(
            self.resolve_idle_task_client(llm_client),
        )

    def _collect_mode_prompt_texts(self) -> dict[str, str]:
        """モード別の生プロンプトテキストを収集（取得失敗モードはスキップ）"""
        prompt_texts: dict[str, str] = {}
        for mode in self.MODES:
            try:
                prompt_texts[mode] = self.prompt_manager.get_raw_prompt(mode)
            except ValueError:
                continue
        return prompt_texts

    async def _level1_phase3_embed_instruction(
        self,
        experiences: list[dict],
        llm_client,
        results: dict[str, dict],
        phase_durations: dict[str, float],
    ) -> None:
        """エンベッド検索指示プロンプト進化（f_04 §4.2）"""
        if self._cancelled or self._embed_instruction_evolver is None:
            return

        rag_exp = [
            e for e in experiences
            if e.get("signals", {}).get("rag_top1_score") is not None
        ]
        # 部分集合 (RAG 使用ターンのみ) への閾値は全体 min_experiences ではなく
        # phase6 (generation_params) と同じ min_experiences // 2 を使う。全体閾値
        # をそのまま課すと、Level 1 発火自体は total >= min_experiences で足りる
        # のに部分集合はそれより小さくなりがちで、この phase だけ実質発火不能に
        # なっていた (2026-07-17 の自己学習ログ監査で判明)。
        threshold = max(1, self.min_experiences // 2)
        if len(rag_exp) < threshold:
            logger.debug(
                "Level 1 embed instruction skipped: %d RAG experiences < %d",
                len(rag_exp), threshold,
            )
            return

        logger.info(
            "Level 1 embed instruction evolution (%d RAG experiences)",
            len(rag_exp),
        )
        base_mutator = llm_client
        current_instruction = self._load_embed_instruction()
        tp = time.monotonic()
        evo_result = await self._embed_instruction_evolver.evolve_instruction(
            experiences=rag_exp,
            current_instruction=current_instruction,
            mutator_client=base_mutator,
            generations=self.generations,
            population_size=self.population_size,
        )
        phase_durations["phase3_embed_instruction"] = round(time.monotonic() - tp, 3)

        improved = evo_result.final_fitness > evo_result.initial_fitness
        results["embed_instruction"] = {
            "improved": improved,
            "fitness_before": evo_result.initial_fitness,
            "fitness_after": evo_result.final_fitness,
        }
        from backend.free.optimizer.embed_instruction_evolver import (
            ADOPTION_THRESHOLD,
        )
        if improved and evo_result.final_fitness > ADOPTION_THRESHOLD:
            self._save_embed_instruction(evo_result.best_candidate.text)
            # 永続化に加えて稼働中の embedder へ即時反映 (再起動不要)。
            self._apply_embed_instruction_runtime(evo_result.best_candidate.text)
            logger.info(
                "Embed instruction evolved: fitness %.4f → %.4f",
                evo_result.initial_fitness, evo_result.final_fitness,
            )
        else:
            logger.info(
                "Embed instruction: no adoption "
                "(fitness %.4f → %.4f, threshold %.4f)",
                evo_result.initial_fitness,
                evo_result.final_fitness,
                ADOPTION_THRESHOLD,
            )

    async def _level1_phase4_token_budget(
        self,
        experiences: list[dict],
        results: dict[str, dict],
        phase_durations: dict[str, float],
    ) -> None:
        """トークン予算比率進化（f_04 §4）— 現状は評価不能として skip する。

        旧実装は比率テーブルを摂動して ``_calc_budget_fitness(ratios, exp)`` で
        採点していたが、この関数は ``ratios`` を一切読まず (経験に「その応答を
        生成したときの比率」が残っていない)、候補と現行が常に同点で採用ゲートは
        構造的に閉じていた。候補生成だけが走る無駄を止め、per-candidate の
        outcome シグナルが配線されるまで ``no_outcome_signal`` を返す (L-A1)。
        ``budget_*`` config ノブは inert (``_init_phase4_and_mutator_config``)。
        """
        if self._cancelled:
            return
        lf_exp = [
            e for e in experiences
            if e.get("signals", {}).get("long_form_used")
        ]
        threshold = max(1, self.min_experiences // 2)
        if len(lf_exp) < threshold:
            logger.debug(
                "Level 1 token budget skipped: %d long-form experiences < %d",
                len(lf_exp), threshold,
            )
            return
        tp = time.monotonic()
        logger.info(
            "Level 1 token budget: no_outcome_signal (candidate evaluation "
            "disabled; %d long-form experiences)", len(lf_exp),
        )
        results["token_budget"] = {
            "improved": False,
            "skipped": True,
            "reason": "no_outcome_signal",
        }
        phase_durations["phase4_token_budget"] = round(time.monotonic() - tp, 3)

    # 旧 phase5 (correction パターン重み進化) は 2026-07-21 に learned
    # correction 機構ごと廃止した (feedback._detect_correction docstring 参照)。
    # phase 番号は欠番のまま維持する (phase6/7 のリネームは行わない)。

    def _level1_phase6_generation_params(
        self,
        experiences: list[dict],
        results: dict[str, dict],
        phase_durations: dict[str, float],
    ) -> None:
        """生成パラメータ進化"""
        if self._cancelled or self._generation_param_evolver is None:
            return
        tp = time.monotonic()
        for mode in self.MODES:
            if self._cancelled:
                break
            mode_exp = [e for e in experiences if e.get("mode") == mode]
            if len(mode_exp) >= self.min_experiences // 2:
                gen_result = self._generation_param_evolver.evolve(
                    mode=mode,
                    experiences=mode_exp,
                    population_size=self.population_size,
                )
                results[f"generation_params_{mode}"] = gen_result
            else:
                logger.debug(
                    "Level 1 generation params skipped for %s: %d experiences < %d",
                    mode, len(mode_exp), self.min_experiences // 2,
                )
        phase_durations["phase6_generation_params"] = round(time.monotonic() - tp, 3)

    def _level1_phase7_tool_routing(
        self,
        experiences: list[dict],
        results: dict[str, dict],
        phase_durations: dict[str, float],
        experience_cutoff: float = 0.0,
    ) -> None:
        """ツール誘導パターン重み進化 + 長文ルーティングパターン重み進化

        f_04 §3.4: ブースト / 減衰は非冪等なので、``experience_cutoff``
        (session 開始時の ``_last_run``) 以降の新規経験だけに適用する。
        snapshot 全件へ再適用すると同じ成功ターンが毎 tick ブーストされ、
        重みが上限へ飽和する。
        """
        if self._cancelled or self._learned_patterns is None:
            return
        tp = time.monotonic()
        experiences = self._select_new_experiences(experiences, experience_cutoff)
        tool_exp = [
            e for e in experiences
            if (e.get("signals", {}).get("tool_routing_success")
                or e.get("signals", {}).get("tool_routing_false_positive")
                or e.get("signals", {}).get("tool_routing_false_negative"))
        ]
        if tool_exp:
            tool_result = self._evolve_tool_routing_patterns(tool_exp)
            if tool_result:
                results["tool_routing_patterns"] = tool_result

        long_form_exp = [
            e for e in experiences
            if (e.get("signals", {}).get("long_form_success")
                or e.get("signals", {}).get("long_form_false_positive")
                or e.get("signals", {}).get("long_form_false_negative"))
        ]
        if long_form_exp:
            lf_result = self._evolve_long_form_patterns(long_form_exp)
            if lf_result:
                results["long_form_patterns"] = lf_result
        phase_durations["phase7_tool_routing"] = round(time.monotonic() - tp, 3)

    # ── sleep-time Step 11 / 12 / 14 統合 ─────────

    def _select_critique_for_phase1(self):
        """base prompt 進化 (`run_or_resume_level1` の evolve_all_modes) に渡す critique を選ぶ

        Step 11 が ``CritiqueResult`` を生成済みであればキャッシュ化した
        プロキシを返し、PromptEvolver 内での再 critique を防ぐ。Step 11
        が skip (synthesizer 未注入 など) されていれば従来通り synthesizer
        本体 (もしくは ``None``) を返す。
        """
        if self._last_critique_result is not None:
            return _CachedCritiqueProxy(self._last_critique_result)
        return self._critique_synthesizer

    async def _step11_critique_synthesis(
        self,
        experiences: list[dict],
        results: dict[str, dict],
        phase_durations: dict[str, float],
    ) -> None:
        """Step 11: Critique-Synthesis

        ``CritiqueSynthesizer.critique()`` を base prompt 進化より前に明示的に呼び
        失敗パターン → 改善ヒントを生成する。結果は
        ``self._last_critique_result`` に格納し、後段の 
        ``_CachedCritiqueProxy`` 経由で再利用される。

        synthesizer が未注入の場合は no-op (results / phase_durations
        への記録もなし) で抜ける。
        """
        if self._cancelled or self._critique_synthesizer is None:
            return
        tp = time.monotonic()
        try:
            critique_result = await self._critique_synthesizer.critique(experiences)
        except Exception as exc:
            logger.warning(
                "Step 11 critique-synthesis failed: %s: %s",
                type(exc).__name__, exc,
            )
            phase_durations["step11_critique_synthesis"] = round(
                time.monotonic() - tp, 3,
            )
            return
        self._last_critique_result = critique_result
        phase_durations["step11_critique_synthesis"] = round(
            time.monotonic() - tp, 3,
        )
        results["critique_synthesis"] = {
            "source": critique_result.source,
            "failure_patterns": list(critique_result.failure_patterns),
            "improvement_hints": list(critique_result.improvement_hints),
            "summary": critique_result.summary,
        }
        logger.info(
            "Step 11 critique-synthesis: source=%s patterns=%d hints=%d",
            critique_result.source,
            len(critique_result.failure_patterns),
            len(critique_result.improvement_hints),
        )

    def _step12_policy_evolver(
        self,
        experiences: list[dict],
        results: dict[str, dict],
        phase_durations: dict[str, float],
    ) -> None:
        """Step 12: PolicyEvolver 評価/書き戻し

        ``PolicyParamEvolver.evolve_all()`` を呼び
        evolve_writeback=semmem の場合は SemMem に新規 ``policy`` ファクト
        を書き戻す。yaml モードでは
        ``policy_evolver_state.json`` への永続化を行う。
        """
        if self._cancelled or self._policy_param_evolver is None:
            return
        tp = time.monotonic()
        min_for_policy = max(1, self.min_experiences // 2)
        mode_counts = {
            mode: sum(1 for e in experiences if e.get("mode") == mode)
            for mode in self.MODES
        }

        if any(c >= min_for_policy for c in mode_counts.values()):
            logger.info(
                "Step 12 policy evolver: evolving (experiences: %s, writeback=%s)",
                mode_counts, self._policy_param_evolver.evolve_writeback,
            )
            policy_results = self._policy_param_evolver.evolve_all(
                experiences=experiences,
                modes=self.MODES,
            )
            if policy_results:
                results["policy_params"] = policy_results
                self._save_policy_evolver_state()
        else:
            logger.debug(
                "Step 12 policy evolver skipped: %s (min=%d)",
                mode_counts, min_for_policy,
            )
        phase_durations["step12_policy_evolver"] = round(
            time.monotonic() - tp, 3,
        )

    def _step14_fewshot_gc(
        self,
        results: dict[str, dict],
        phase_durations: dict[str, float],
    ) -> None:
        """Step 14: Few-shot プール GC

        ``FewShotPool.garbage_collect()`` を呼ぶ。SemMem writeback モード
 では SemMem 側 ``semmem_limits.policy`` +
        ``gc_strategy=lowest_score`` に GC を委譲するため in-memory プールは
        触らず、``delegated_to_semmem=True`` を返す。yaml モードではモード別
        プールを fitness 昇順でソートし ``pool_size`` 超過分を除去する。

        GC 実行後、yaml モードでは fewshot プールを ``fewshot_pool.json``
        へ保存する (旧 ``_level1_finalize`` で行っていた永続化を移動)。
        """
        if self._cancelled or self._fewshot_pool is None:
            return
        tp = time.monotonic()
        gc_summary = self._fewshot_pool.garbage_collect()
        # SemMem 書き戻しモードでは fewshot_pool.json への保存は廃止 (D-3)
        if not self._fewshot_pool.is_semmem_writeback_active():
            try:
                self._fewshot_pool.save(
                    self.prompt_manager.prompt_dir / "fewshot_pool.json",
                )
            except OSError as exc:
                logger.warning(
                    "Step 14 fewshot pool save failed: %s", exc,
                )
        phase_durations["step14_fewshot_gc"] = round(
            time.monotonic() - tp, 3,
        )
        results["fewshot_gc"] = gc_summary

    def _level1_finalize(
        self,
        experiences: list[dict],
        results: dict[str, dict],
        *,
        advance_cursor: bool = True,
        skipped_phases: list[str] | None = None,
    ) -> None:
        """Level 1 後処理: 統計更新 + 状態保存

        Few-shot プールの GC + 永続化は ``_step14_fewshot_gc``
        に移管したため、本メソッドでは扱わない。

        ``advance_cursor=False`` (cancel で extra phase を飛ばした) のときは
        ``_last_run`` を進めない — 次の idle tick が同じ経験で残りの phase を
        回せるようにする (L-A15)。
        """
        if advance_cursor:
            self._last_run = time.time()
        self._level1_run_count += 1
        self._last_level1_results = self._summarize_level1_results(results)
        self._last_level1_results["_skipped_phases"] = list(skipped_phases or [])
        self._append_fitness_history(results)
        self._snapshot_rates(experiences)
        self._save_state()

    def _level1_log_debug(
        self,
        started_at: str,
        elapsed: float,
        phase_durations: dict[str, float],
        experiences: list[dict],
        results: dict[str, dict],
    ) -> None:
        """デバッグログに Level 1 学習サイクルを記録"""
        dl = self._debug_logger
        if dl is None:
            return
        dl.log_learning_cycle(cycle_num=1, data={
            "level": 1,
            "started_at": started_at,
            "elapsed_sec": elapsed,
            "phase_durations_sec": phase_durations,
            "experiences": len(experiences),
            "generations": self.generations,
            "results": results,
        })

    # ── エンベッド検索指示プロンプト I/O ──

    def _apply_embed_instruction_runtime(self, text: str) -> None:
        """採用 instruction を稼働中の embedder へ即時反映 + query キャッシュ clear。

        instruction-aware embedder かつ ``set_instruction`` を実装する場合のみ。
        進化器・eval が mode 非フィルタなため ``mode=None`` で全 mode に共通適用する。
        非 instruction-aware モデル / embedder 未注入時は no-op (安全縮退)。
        """
        emb = self._embedder
        if emb is None or not (
            hasattr(emb, "supports_instructions")
            and emb.supports_instructions()
            and hasattr(emb, "set_instruction")
        ):
            return
        emb.set_instruction(text, mode=None)
        logger.info("Applied evolved embed instruction to live embedder")

    def _embed_instruction_dir(self) -> Path:
        """embed_instruction.md 系データの保存先ディレクトリ (embedding モデル単位)。

        resolver 未注入 (レガシー構築 / 一部テスト) の場合は従来通り
        ``prompt_manager.prompt_dir`` (base パーティション) にフォールバックする。
        """
        if self._resolver is not None:
            return self._resolver.resolve_embed_instruction_dir()
        return self.prompt_manager.prompt_dir

    def _load_embed_instruction(self) -> str:
        """embed_instruction.md を読み込む（なければデフォルト生成）

        2026-07-18: 保存先を base モデルパーティションから embedding モデル
        パーティションへ変更 (embed_instruction は埋め込みモデル向けのクエリ
        指示であり base モデル切替と無関係に保持されるべきだったため)。旧
        base パーティションに残る従来データがあれば一度だけ新パスへコピーして
        引き継ぐ (非破壊: 旧ファイルは残置)。
        """
        from backend.free.optimizer.embed_instruction_evolver import (
            DEFAULT_EMBED_INSTRUCTION,
        )
        embed_dir = self._embed_instruction_dir()
        path = embed_dir / "embed_instruction.md"
        if path.exists():
            return path.read_text(encoding="utf-8")

        embed_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = self._find_legacy_embed_instruction_source()
        if embed_dir != self.prompt_manager.prompt_dir and legacy_path is not None:
            content = legacy_path.read_text(encoding="utf-8")
            atomic_write_text(path, content, encoding="utf-8")
            logger.info(
                "Migrated embed_instruction.md from legacy base partition "
                "(%s) to %s", legacy_path, path,
            )
            return content

        # デフォルトを書き込んで返す
        atomic_write_text(path, DEFAULT_EMBED_INSTRUCTION, encoding="utf-8")
        logger.info("Created default embed_instruction.md")
        return DEFAULT_EMBED_INSTRUCTION

    def _find_legacy_embed_instruction_source(self) -> Path | None:
        """旧 base パーティション群から embed_instruction.md の移行元を探す。

        embed_instruction は本来 base モデルと独立のはずが、旧実装では base
        パーティション (prompt_manager.prompt_dir) に同居していた。base モデルを
        複数回切り替えた履歴がある場合、active なモデルのものだけを見ると、
        過去に切り替えた別モデル配下でより進化した instruction を見落とし
        永久に不可視化する (2026-07-18 のレビューで判明)。resolver が使える
        場合は全 base パーティションを走査し、更新日時が最も新しいものを選ぶ
        (どのモデルが「最も進化しているか」を厳密には判定できないため、直近の
        Level 1 サイクルで更新されたものを優先する近似)。
        """
        legacy_path = self.prompt_manager.prompt_dir / "embed_instruction.md"
        candidates: set[Path] = set()
        if legacy_path.exists():
            candidates.add(legacy_path.resolve())
        if self._resolver is not None:
            learning_dir = self._resolver.resolve_local("learning_dir")
            if learning_dir.is_dir():
                candidates.update(
                    p.resolve()
                    for p in learning_dir.glob("*/prompts/embed_instruction.md")
                )
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _save_embed_instruction(self, content: str) -> None:
        """embed_instruction.md を保存（保護セクション検証付き）"""
        embed_dir = self._embed_instruction_dir()
        path = embed_dir / "embed_instruction.md"

        # 保護セクション最終ゲート
        current = self._load_embed_instruction()
        if not validate_protected_sections(current, content):
            logger.warning(
                "Evolved embed instruction lost protected sections, force-restoring",
            )
            content = restore_protected_sections(current, content)

        # 履歴に退避
        history_dir = embed_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # 簡易バージョン番号: 既存履歴の数 + 1
            existing = list(history_dir.glob("embed_instruction_v*.md"))
            version = len(existing) + 1
            dst = history_dir / f"embed_instruction_v{version:03d}.md"
            atomic_write_text(dst, path.read_text(encoding="utf-8"), encoding="utf-8")

        atomic_write_text(path, content, encoding="utf-8")
        logger.info("Embed instruction updated: %s", path)


    # ── ツール誘導パターン重み進化──

    def _evolve_tool_routing_patterns(
        self, experiences: list[dict],
    ) -> dict | None:
        """ツール誘導パターンの重みをフィードバックに基づいて進化させる

        成功時: パターンをブースト、誤検出時: 減衰、未検出時: 新パターン追加。
        LLM 不要。

        Returns:
            {"boosted": int, "decayed": int, "added": int, "total": int} or None
        """
        if self._learned_patterns is None:
            return None

        store = self._learned_patterns
        learning_cfg = self._config.get("learning", {})
        boost_amount = learning_cfg.get("tool_pattern_boost_success", 0.03)
        decay_amount = learning_cfg.get("tool_pattern_decay_false_pos", 0.1)

        boosted = 0
        decayed = 0
        added = 0

        for exp in experiences:
            signals = exp.get("signals", {})
            query = exp.get("query", "")

            if signals.get("tool_routing_success"):
                # 成功: マッチしたパターンの重みをブースト
                matches = store.match(query, category="tool_routing", record_hit=False)
                for kw, _ in matches:
                    store.boost(kw, amount=boost_amount)
                    boosted += 1

            if signals.get("tool_routing_false_positive"):
                # 誤検出: マッチしたパターンの重みを減衰
                matches = store.match(query, category="tool_routing", record_hit=False)
                for kw, _ in matches:
                    store.decay_one(kw, amount=decay_amount)
                    decayed += 1

            if signals.get("tool_routing_false_negative"):
                # 未検出: クエリからキーワードを抽出して追加。
                # extract_tool_routing_keywords がツールシグナル無しクエリ /
                # 話題名詞 / 言語タスク語 (「説明」等) を除外する
                # (FeedbackCollector._learn_tool_routing_from_false_negative
                # と同一ゲート。従来ここは無ゲートの extract_intent_keywords
                # で、feedback 側 2026-07-18 ガードで弾いた経験からも Level 1
                # バッチで再学習される穴があり、「コーヒー」「天気」等の
                # 話題名詞 62 件が tool_routing に蓄積していた)
                keywords = store.extract_tool_routing_keywords(query)
                for kw in keywords:
                    store.add_pattern(kw, category="tool_routing")
                    added += 1

        # 永続化
        try:
            from backend.config import get_path_resolver
            resolver = get_path_resolver()
            patterns_file = resolver.resolve_local("learned_patterns_file")
            store.save(patterns_file)
        except Exception:
            logger.warning("tool routing pattern persistence failed")

        logger.info(
            "tool routing evolution: boosted=%d, decayed=%d, added=%d, total=%d",
            boosted, decayed, added, store.count,
        )
        return {
            "boosted": boosted,
            "decayed": decayed,
            "added": added,
            "total": store.count,
        }

    def _evolve_long_form_patterns(
        self, experiences: list[dict],
    ) -> dict | None:
        """長文ルーティングパターンの重みをフィードバックに基づいて進化させる

        ``_evolve_tool_routing_patterns`` のロジックを ``category="long_form"``
        に置換した同型実装。ハードコード regex (LONG_FORM_PATTERNS) と並列に
        ``ComplexityClassifier._detect_long_form_learned()`` で OR 評価される
        学習語彙を更新する。

        Returns:
            {"boosted": int, "decayed": int, "added": int, "total": int} or None
        """
        if self._learned_patterns is None:
            return None

        store = self._learned_patterns
        learning_cfg = self._config.get("learning", {})
        boost_amount = learning_cfg.get("long_form_pattern_boost_success", 0.03)
        decay_amount = learning_cfg.get("long_form_pattern_decay_false_pos", 0.1)

        boosted = 0
        decayed = 0
        added = 0

        for exp in experiences:
            signals = exp.get("signals", {})
            query = exp.get("query", "")

            if signals.get("long_form_success"):
                matches = store.match(query, category="long_form", record_hit=False)
                for kw, _ in matches:
                    store.boost(kw, amount=boost_amount)
                    boosted += 1

            if signals.get("long_form_false_positive"):
                matches = store.match(query, category="long_form", record_hit=False)
                for kw, _ in matches:
                    store.decay_one(kw, amount=decay_amount)
                    decayed += 1

            if signals.get("long_form_false_negative"):
                # パス片 / URL 片 / 汎用ファイル操作語は文書種別シグナルでないため除外
                keywords = [
                    kw for kw in store.extract_intent_keywords(query)
                    if store.is_long_form_learnable(kw)
                ]
                for kw in keywords:
                    store.add_pattern(kw, category="long_form")
                    added += 1

        try:
            from backend.config import get_path_resolver
            resolver = get_path_resolver()
            patterns_file = resolver.resolve_local("learned_patterns_file")
            store.save(patterns_file)
        except Exception:
            logger.warning("long_form pattern persistence failed")

        logger.info(
            "long_form pattern evolution: boosted=%d, decayed=%d, added=%d, total=%d",
            boosted, decayed, added, store.count,
        )
        return {
            "boosted": boosted,
            "decayed": decayed,
            "added": added,
            "total": store.count,
        }

    # ── Level 2: Pro プラグインに委譲 ──

    def check_level2(
        self,
        is_user_active: bool,
        lora_path: Path | None = None,
        current_model: str = "",
        base_model_path: Path | None = None,
        force: bool = False,
    ) -> bool:
        """Level 2 トリガー判定と実行開始（ベースモデル）

        Pro 版: Level2Runner に委譲。
        Free 版: _level2_runner が None のため常に False。

        Args:
            is_user_active: ユーザーがアクティブかどうか
            lora_path: ベースモデル LoRA アダプタパス
            current_model: 現在のベースモデルファイル名（base_model フィルタ用）
            base_model_path: ベースモデル GGUF パス（LoRA ターゲット自動判定用）
            force: overdue クールダウンを無視して即時試行する（手動トリガ用）。
                データ量・実行中・アダプタ互換の各ゲートは維持する。

        Returns:
            学習を開始したかどうか
        """
        if self._disabled:
            return False
        if self._level2_runner is None:
            logger.debug("Level 2 runner not set (Free edition)")
            return False

        return self._level2_runner.check_and_run(
            is_user_active=is_user_active,
            lora_path=lora_path,
            current_model=current_model,
            base_model_path=base_model_path,
            force=force,
        )
