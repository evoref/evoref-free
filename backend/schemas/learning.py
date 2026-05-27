"""学習サイクル / スケジュール関連スキーマ"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

    level1_min_experiences: int = Field(default=20, ge=1)
    level1_generations: int = Field(default=10, ge=1)
    level1_population_size: int = Field(default=5, ge=1)
    level2_min_failures: int = Field(default=50, ge=1)
    # アイドル判定タイマー
    full_idle_minutes: int = Field(default=10, ge=1)        # Trigger B: Full sleep-time
    level1_idle_minutes: int = Field(default=30, ge=1)      # Trigger C: Level 1
    level1_recheck_interval_sec: int = Field(default=60, ge=1)  # Level 1 ループ再評価間隔
    priority_threshold_ratio: float = Field(default=0.5, ge=0.0, le=1.0)  # 優先 Level 1 の経験数閾値緩和率
    active_minutes: int = Field(default=5, ge=1)
    level2_spsa_iterations: int = Field(default=500, ge=1)
    level2_sparse_params: int = Field(default=200, ge=1)
    level2_schedule_hour: int = Field(default=3, ge=0, le=23)
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
    # Level 2 アシストモデル
    assist_level2_min_experiences: int = Field(default=30, ge=1)
    assist_spsa_iterations: int = Field(default=300, ge=1)
    assist_sparse_params: int = Field(default=100, ge=1)
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
    # 変異生成 LLM 設定
    prompt_mutator_base: str = "main"
    prompt_mutator_assist: str = "main"
    # 品質ゲート結果 還流パイプ
    feedback_pipe: FeedbackPipeConfig = Field(default_factory=FeedbackPipeConfig)
    # ポリシー / 進化結果書き戻し設定
    policy: LearningPolicyConfig = Field(default_factory=LearningPolicyConfig)
