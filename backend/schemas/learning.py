"""学習サイクル / スケジュール関連スキーマ"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScheduleConfig(BaseModel):
    """スケジューラの tz 設定

    evoref の内部時刻は全て UTC で扱うが、「夜間 N 時に Level 2 学習を
    起動する」のように *ユーザ体感のローカル時刻* で判定したい処理が
    一部存在する。そうした箇所は本設定の ``local_tz`` を UTC→ローカル
    変換に用いる。サーバの OS tz 設定には依存しない。
    """

    model_config = ConfigDict(extra="forbid")

    local_tz: str = Field(
        default="Asia/Tokyo",
        description="ZoneInfo で解決可能な IANA tz 名 (例: Asia/Tokyo, UTC)",
    )


class FeedbackPipeConfig(BaseModel):
    """品質ゲート結果 → 学習サイクル 還流パイプ設定

    ラルフループが SemMem に書く ``progress_marker`` / ``failure_pattern`` /
    ``artifact`` を Level 1/2 tick 時に走査し、以下 3 経路で学習サイクルへ
    還流する:

    - 経路 1: 成功 ``progress_marker`` + ``task`` + ``artifact`` → FewShotPool
    - 経路 2: 連続 ``failure_pattern`` クラスタ → CritiqueSynthesizer →
      ``policy`` ファクト書き戻し
    - 経路 3: ``progress_marker`` / ``failure_pattern`` 比率 →
      PolicyParamEvolver の fitness に実環境成功率項を加重合成
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    """全経路の ON/OFF。``false`` 時は走査自体をスキップする。"""

    max_scan_per_tick: int = Field(default=50, ge=1)
    """1 tick で走査する SemMem ファクト数上限 (暴走防止)"""

    progress_marker_lookback: int = Field(default=20, ge=1)
    """経路 1 で走査する直近 progress_marker 件数"""

    failure_pattern_lookback: int = Field(default=30, ge=1)
    """経路 2 で走査する直近 failure_pattern 件数"""

    failure_cluster_min_size: int = Field(default=3, ge=2)
    """経路 2 でクラスタとして扱う最小件数 (これ未満は critique しない)"""

    weight_semmem_success: float = Field(default=0.3, ge=0.0, le=1.0)
    """経路 3 の fitness 重み ``w``。
    最終 fitness = ``(1-w) * calc_fitness(...) + w * semmem_success_rate``"""


class LearningPolicyConfig(BaseModel):
    """学習ポリシー設定

    PolicyInterpreter / PolicyParamEvolver / FewShotPool の永続化先と
    進化挙動を制御する。owner pillar は EvorefLearn。

    - ``source`` :
        - ``yaml`` (デフォルト) : 従来通り ``local/policies/*.json`` のみを参照
        - ``hybrid`` : YAML を seed としてロード後、SemMem の active な
          ``policy`` ファクト (``subject`` 先頭が ``learn.policy.``) で上書き
        - ``semmem`` : 導入予定 (現状は ``hybrid`` と同等動作)
    - ``activation_min_confidence`` : SemMem 上の policy ファクトを active
      と見なす最小 ``confidence``。これ未満は YAML 値が温存される
    - ``evolve_writeback`` : 進化結果書き戻し先
      ``yaml`` (従来動作) / ``semmem`` (PolicyParamEvolver / FewShotPool で
      SemMem へ書き戻し)。
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["yaml", "semmem", "hybrid"] = "yaml"
    activation_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evolve_writeback: Literal["yaml", "semmem"] = "yaml"


class LearningConfig(BaseModel):
    """学習サイクル設定"""

    model_config = ConfigDict(extra="allow")  # Pro 拡張フィールドを許可

    # base モデルの自己学習データを (base モデル識別子 × モード) でパーティション化する。
    # True: experience / base prompts / base LoRA・cvector / learning_state を
    #   ``local/learning/<base_model_stem>/`` 配下へ分離し、モデル切替でそのモデルの
    #   保存済みセットへ差し替える (未知モデルは空=ゼロから学習)。
    # False: レガシー flat レイアウト (resolve_learning が resolve_local へ素通し)。
    #   既存環境の即時ロールバック用キルスイッチ。
    partition_by_base_model: bool = True
    # Level 2 base LoRA アダプタ (Pro) のパーティション粒度。
    # 対象は LoRA のみ (control vector / cvector は本設定の対象外、常にモデル単位で
    # 不変 — ランタイム hot-swap 不可 + GGUF 再ロードが高コストなため)。
    # "model" (既定): LoRA はモデル単位 (両モードの経験を 1 アダプタにプール)。
    #   chat↔create 切替で base サーバを LoRA 差分では再起動しない。
    #   プロンプト/fewshot/policy/experience は (model×mode) で分離されるため
    #   軽量側のモード差は保たれる。
    # "model_mode": LoRA を (model×mode) で分け、モード切替時に
    #   LoRA パスが変わっていれば該当サーバを再起動して切り替える
    #   (プロセス再起動方式、切替レイテンシ増を許容する strict モード)。
    #   partition_by_base_model=true を前提とする (model_validator で強制)。
    level2_adapter_partition: Literal["model", "model_mode"] = "model"

    level1_min_experiences: int = Field(default=20, ge=1)
    level1_generations: int = Field(default=10, ge=1)
    level1_population_size: int = Field(default=5, ge=1)
    level2_min_failures: int = Field(default=50, ge=1)
    # モード別の Level 2 発火閾値。未指定のモードは level2_min_failures に
    # フォールバックする。create は経験が溜まりにくく (2026-07-26 実測: chat 180 件に対し
    # create 2 件)、chat と同じ閾値では「chat が回るたびに create も無駄に評価される」か
    # 「create が永久に発火しない」のどちらかになるため、モード別に持てるようにする。
    level2_min_failures_by_mode: dict[Literal["chat", "create"], int] = Field(
        default_factory=dict,
    )
    # アイドル判定タイマー
    full_idle_minutes: int = Field(default=10, ge=1)        # Trigger B: Full sleep-time
    level1_idle_minutes: int = Field(default=30, ge=1)      # Trigger C: Level 1
    level1_recheck_interval_sec: int = Field(default=60, ge=1)  # Level 1 ループ再評価間隔
    priority_threshold_ratio: float = Field(default=0.5, ge=0.0, le=1.0)  # 優先 Level 1 の経験数閾値緩和率
    active_minutes: int = Field(default=5, ge=1)
    level2_spsa_iterations: int = Field(default=500, ge=1)
    level2_sparse_params: int = Field(default=200, ge=1)
    level2_schedule_hour: int = Field(default=3, ge=0, le=23)  # 優先窓のヒント (強制 sleep ではない)
    # Level 2 常駐ループ (再起動耐性): 03:00 固定 sleep を廃し overdue/idle 判定で発火
    level2_recheck_interval_sec: int = Field(default=300, ge=10)
    level2_overdue_hours: float = Field(default=24.0, ge=0.0)
    # 連続で改善版を採用できなかった回数がこの値に達した target は
    # level2_stale_backoff_hours の延長クールダウンへ落ちる。1 サイクルが実推論で
    # 1 時間規模になりうるため、探索が局所最適に張り付いた状態で 24h 間隔を
    # 回し続けないための天井 (採用に成功したらストリークは 0 へ戻る)。
    level2_stale_streak: int = Field(default=2, ge=1)
    level2_stale_backoff_hours: float = Field(default=72.0, ge=0.0)
    # 採用に必要な最小改善幅 (baseline_loss に対する相対比)。実推論 eval は
    # 決定論なので微小差も実差だが、少数の held-out ケースに対する 0.0x% の
    # 改善は汎化ではなく過適合で、LoRA の版だけが増える。既定 0.1%。
    level2_min_improvement_ratio: float = Field(default=0.001, ge=0.0, le=1.0)
    # SPSA の best が更新されないまま連続でこの反復数を超えたら打ち切る (0 で無効)。
    # 実推論 eval 経路のみ有効 (1 反復 = 候補サーバ 2 起動)。
    level2_early_stop_patience: int = Field(default=10, ge=0)
    # 候補評価用の使い捨て llama-server に --no-warmup を付ける。1 サイクルで
    # 数十回起動するため空回しのウォームアップが積み上がる。旧 llama.cpp で
    # フラグ非対応の場合のみ false にする。
    level2_eval_no_warmup: bool = True
    # Level 2 初回 full-train (bootstrap): 既存 LoRA が無いとき初期アダプタを生成して
    # パイプラインを起動可能にする。llama.cpp の LoRA GGUF 形式を生成するため、実
    # llama-server へのロード検証が済むまで既定 False (誤った形式は server 起動を妨げる)。
    level2_bootstrap_enabled: bool = False
    level2_bootstrap_rank: int = Field(default=8, ge=1, le=256)
    level2_bootstrap_init_sigma: float = Field(default=1e-4, ge=0.0)
    level2_bootstrap_min_failures: int = Field(default=20, ge=1)
    # Level 2 base=C: ベースモデルの自己進化を control vector (残差ストリーム操舵) で行う。
    # llama-cvector-generator (forward-only) で生成し、llama-server --control-vector-scaled
    # で次回起動時に適用する。'cvector' 指定が enable (Pro 限定)。既定 'lora' の SPSA 改良は
    # no-op (対称摂動で勾配 0) のためトリガ段階で skip され Level 2 base は何も起動しない
    # (#3a)。base の実学習は 'cvector' または 'spsa-real-eval' で有効化する
    # (実 llama-server ロード検証後)。
    # 'spsa-real-eval': CandidateEvalHarness を
    # base 用に配線し、候補 LoRA を ephemeral llama-server (base モデル + scratch port) に
    # 実ロードして eval_core.json ケースで実推論評価する。CPU 推論 (-ngl 0 相当) のため
    # base モデルサイズ次第で 1 サイクルが長時間化しうる。
    level2_base_method: Literal["lora", "cvector", "spsa-real-eval"] = "lora"
    # base=spsa-real-eval 実推論 eval 有効時の SPSA 反復数
    # (1 反復で候補サーバを複数回起動するためコスト天井として低く設定)。
    base_realeval_spsa_iterations: int = Field(default=30, ge=1)
    base_eval_scratch_port: int = Field(default=8091, ge=1024, le=65535)
    base_eval_loss_w1: float = Field(default=0.7, ge=0.0, le=1.0)  # keyword/pattern 一致項
    base_eval_loss_w2: float = Field(default=0.3, ge=0.0, le=1.0)  # mean neg-logprob 項
    base_eval_max_tokens: int = Field(default=64, ge=1)
    base_eval_max_cases: int = Field(default=20, ge=1)
    base_eval_health_timeout: int = Field(default=120, ge=10)
    cvector_method: Literal["pca", "mean"] = "pca"
    cvector_pca_batch: int = Field(default=100, ge=1)
    cvector_pca_iter: int = Field(default=1000, ge=1)
    # 空 = arch の block_count から全層自動。"START END" / "START,END" 指定可。
    cvector_layer_range: str = ""
    cvector_scale: float = Field(default=1.0)  # --control-vector-scaled FNAME:SCALE
    # キュレーション種ペア JSON のパス (空 = 組込デフォルト軸)。HYBRID 対照例の種。
    cvector_seed_pairs_file: str = ""
    cvector_min_experiences: int = Field(default=40, ge=1)  # HYBRID 対照例の最小数
    # パターン検出キーワード学習
    pattern_max_patterns: int = Field(default=200, ge=10, le=1000)
    pattern_initial_weight: float = Field(default=0.5, ge=0.1, le=1.0)
    pattern_decay_rate: float = Field(default=0.05, ge=0.01, le=0.5)
    pattern_boost_amount: float = Field(default=0.15, ge=0.01, le=0.5)
    pattern_min_weight: float = Field(default=0.1, ge=0.01, le=0.5)
    pattern_match_threshold: float = Field(default=0.3, ge=0.1, le=0.9)
    # ツールパターン学習
    tool_pattern_boost_success: float = Field(default=0.03, ge=0.0, le=1.0)
    tool_pattern_decay_false_pos: float = Field(default=0.1, ge=0.0, le=1.0)
    tool_pattern_match_threshold: float = Field(default=0.4, ge=0.1, le=0.9)
    # Few-shot プール
    fewshot_pool_size: int = Field(default=50, ge=1)
    fewshot_min_fitness: float = Field(default=0.7, ge=0.0, le=1.0)
    fewshot_max_examples: int = Field(default=3, ge=1)
    fewshot_diversity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # Level 2 補助タスク
    # Level 2 最適化器切替
    # "spsa" は Free 版にも実装あり (`backend/free/optimizer/spsa.py`)。
    # "full-cma-es" は Pro 版で `cma` パッケージ利用 (`backend/pro/cma_es_full.py`)。
    # `cma` 未インストール時は Level2Runner が SPSA に自動フォールバックする。
    optimizer: Literal["spsa", "full-cma-es"] = "spsa"
    # CMA-ES 初期ステップサイズ（探索範囲の標準偏差）
    cma_sigma0: float = Field(default=0.1, gt=0.0)
    # CMA-ES 1 世代あたりの個体数 (0 = 自動: 4 + floor(3 * ln(N)))
    cma_popsize: int = Field(default=0, ge=0)
    # トークン予算進化
    budget_generations: int = Field(default=5, ge=1)
    budget_sigma: float = Field(default=0.05, ge=0.0, le=1.0)
    # 各戦略の比率合計の上限 (generation 余地を確保するため 1.0 未満に抑える)
    budget_max_total_ratio: float = Field(default=0.7, gt=0.0, le=1.0)
    # 品質ゲート結果 還流パイプ
    feedback_pipe: FeedbackPipeConfig = Field(default_factory=FeedbackPipeConfig)
    # ポリシー / 進化結果書き戻し設定
    policy: LearningPolicyConfig = Field(default_factory=LearningPolicyConfig)

    @model_validator(mode="after")
    def _require_partition_for_adapter_mode_split(self) -> "LearningConfig":
        """``level2_adapter_partition="model_mode"`` は base 学習パーティション必須

        chat/create 別アダプタは ``local/learning/<base_model_stem>/<mode>/...``
        という base パーティション配下のサブディレクトリとして実装されるため、
        ``partition_by_base_model=false`` (flat レイアウト) との組合せは意味を
        持たない。単純化のため起動時に明示的に拒否する。
        """
        if self.level2_adapter_partition == "model_mode" and not self.partition_by_base_model:
            raise ValueError(
                "learning.level2_adapter_partition='model_mode' requires "
                "learning.partition_by_base_model=true",
            )
        return self
