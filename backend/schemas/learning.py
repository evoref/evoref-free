"""学習サイクル / スケジュール関連スキーマ"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.config_normalize import reject_retired_keys


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

    @field_validator("local_tz")
    @classmethod
    def _warn_unknown_tz(cls, value: str) -> str:
        """解決できない tz 名は警告だけ出して残す (読み手がそれぞれ既定の tz へ倒す。起動は止めない)。"""
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            from backend.log_config import get_logger

            get_logger("config").warning(
                "schedule.local_tz=%r is not a known IANA time zone; local-time features will use a fallback zone.",
                value,
            )
        return value


class LearningPolicyConfig(BaseModel):
    """学習ポリシー設定

    PolicyInterpreter / PolicyParamEvolver / FewShotPool の永続化先と
    進化挙動を制御する。owner pillar は EvorefLearn。

    - ``source`` :
        - ``yaml`` (デフォルト) : 従来通り policy ファイル (``<data_root>/g1/store/learning/``
          の ``shared/policies/`` と ``<mk>/policies/``) のみを参照
        - ``hybrid`` : YAML を seed としてロード後、SemMem の active な
          ``policy`` ファクト (``subject`` 先頭が ``learn.policy.``) で上書き
        - ``semmem`` : 導入予定 (現状は ``hybrid`` と同等動作)
    - ``activation_min_confidence`` : SemMem 上の policy ファクトを active
      と見なす最小 ``confidence``。これ未満は YAML 値が温存される
    進化結果 (手本プール・進化した policy 値) の書き戻し先は policy ファイルだけ。
    SemMem へ書き戻す ``evolve_writeback`` は 2026-10-12 に撤去した (f_04 §3.2)。
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["yaml", "semmem", "hybrid"] = "yaml"
    activation_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class GrowthRegressionConfig(BaseModel):
    """Level 2 (成長) の回帰台帳 (f_04 §6.2 / §6.4 / §6.5)。"""

    model_config = ConfigDict(extra="forbid")

    max_cases: int = Field(default=200, ge=1)
    """台帳の件数上限 (同梱の種ケースは数えない)。超えたら retired → quarantined → active の順に古い行から消す"""

    max_guard_cases: int = Field(default=12, ge=0)
    """採用ゲートの標本に足す guard の上限 (同じモード)"""

    max_cases_per_tick: int = Field(default=6, ge=1)
    """Level 1 の 1 tick で再生するケースの上限"""

    guard_min_pass_streak: int = Field(default=2, ge=1)
    """guard とみなす連続合格の回数"""

    use_seed: bool = True
    """同梱の種ケース (ガード専用) を読むか"""


class GrowthHarvestConfig(BaseModel):
    """sleep-time Step 13.6 の収穫源 (f_04 §6.3)。"""

    model_config = ConfigDict(extra="forbid")

    consistency: bool = True
    """会話の整合性検査 (``conversation_consistency``) の検出を収穫するか"""


class GrowthRecallKeysConfig(BaseModel):
    """段 3: episodic 想起の鍵 (f_04 §6.9、条件付き・未実装)。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class GrowthConfig(BaseModel):
    """Level 2 (成長) — 回帰台帳・提案キュー・成長の台帳 (f_04 §6.7)。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    """``false`` で収穫・再生・提案を止める (sleep-time の結果辞書には 0 を返す)"""

    auto_adopt: bool = False
    """既定は提案のみ。``true`` でも f_04 §6.6 の条件を満たすまで自動採用は効かない"""

    min_guards_for_auto: int = Field(default=3, ge=0)
    max_pending_proposals: int = Field(default=20, ge=1)
    proposal_ttl_days: int = Field(default=14, ge=1)
    regression: GrowthRegressionConfig = Field(default_factory=GrowthRegressionConfig)
    harvest: GrowthHarvestConfig = Field(default_factory=GrowthHarvestConfig)
    recall_keys: GrowthRecallKeysConfig = Field(default_factory=GrowthRecallKeysConfig)


class LearningConfig(BaseModel):
    """学習サイクル設定"""

    # 未知のキーは typo とみなして拒否する。``extra="allow"`` は Pro Level 2 の
    # 拡張キーのためだったが、それは撤去した (撤去キーは normalize が先に落とす)。
    model_config = ConfigDict(extra="forbid")

    # base モデルの自己学習データは常に ``store/learning/<model_key>/`` で
    # パーティション化する (c_05 §0.5.12)。flat に戻すキー
    # (``partition_by_base_model``) は G1 で撤去した (:meth:`_reject_flat_layout`)。
    # Pro Level 2 (LoRA / control vector / SPSA / CMA-ES) のキーは config v2 で
    # 撤去した (:meth:`_reject_retired_level2`、f_04 §6.7)。

    #: Level 1 の起動に要る経験数。前回実行以降の **新規** 件数にも同じ値が
    #: 掛かる (``LearningScheduler.has_enough_new_experiences``)。
    #: 20 は実運用の蓄積速度に対して大きすぎた — 実測 161 件 / 約 2 か月に対し
    #: ``Level 1 loop: skipped, not enough NEW experiences`` が 2228 回、起動は
    #: 46 回。ライブ監査 1 回 (30〜50 ターン) で 4〜6 サイクル回る値へ下げる
    #: (2026-09-21)。緩めた分の安全装置は採用ゲート側 (``PromptCandidateEval``)。
    level1_min_experiences: int = Field(default=8, ge=1)
    #: 欠陥率 fitness が平坦なモードでは停滞 3 世代で打ち切られ、10 世代の
    #: 予定が毎回 3 世代で終わっていた (2026-09-14 実測)。選択圧は採用ゲート
    #: 側にあるので、変異生成は少数で足りる。
    level1_generations: int = Field(default=5, ge=1)
    level1_population_size: int = Field(default=5, ge=1)
    # アイドル判定タイマー
    full_idle_minutes: int = Field(default=10, ge=1)        # Trigger B: Full sleep-time
    # Trigger B: アイドルを待ち続ける上限 (分)。会話が途切れないと毎ターン
    # タイマーが張り直されて Full が一度も走らない (2026-08-16 ライブ監査:
    # 40 ターン / 53 分で起動試行 39 件が全て debounced、実行 0 回)。この時間を
    # 超えたらユーザーが会話中でも 1 回走らせる。0 で無効 (従来のアイドル専用)。
    full_max_defer_minutes: int = Field(default=30, ge=0)
    level1_idle_minutes: int = Field(default=30, ge=1)      # Trigger C: Level 1
    level1_recheck_interval_sec: int = Field(default=60, ge=1)  # Level 1 ループ再評価間隔
    priority_threshold_ratio: float = Field(default=0.5, ge=0.0, le=1.0)  # 優先 Level 1 の経験数閾値緩和率
    active_minutes: int = Field(default=5, ge=1)
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
    # 長文生成パターン学習 (router._contains_learned_long_form_patterns)
    long_form_pattern_decay_false_pos: float = Field(default=0.1, ge=0.0, le=1.0)
    long_form_pattern_match_threshold: float = Field(
        default=0.4, ge=0.1, le=0.9,
        description=(
            "学習済み long_form パターンを一致とみなす重みの下限。"
            "閾値以上の一致が 2 語以上、または重み合計 0.8 以上で long_form へ振る"
        ),
    )
    # Few-shot プール
    fewshot_pool_size: int = Field(default=50, ge=1)
    fewshot_min_fitness: float = Field(default=0.7, ge=0.0, le=1.0)
    # GC で無条件に破棄する quality_score の下限。採点は採用の後に走るため
    # intake では掛けられず、従来の GC はサイズ超過時の下位切りしか無かった。
    fewshot_min_quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    fewshot_max_examples: int = Field(default=3, ge=1)
    fewshot_diversity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # 手本の寿命 (f_04 §3.2.2)。使用実績 (注入した経験の成否) から Step 14 が導く。
    # stale = 最終使用がこの日数より古い (選択には効かない、trim の順序のみ)。
    fewshot_stale_after_days: int = Field(default=30, ge=1)
    # 有害判定の最小使用回数 (harmful 率がプール平均の 2 倍超で退避)。
    fewshot_harmful_min_uses: int = Field(default=10, ge=1)
    # トークン予算進化
    budget_generations: int = Field(default=5, ge=1)
    budget_sigma: float = Field(default=0.05, ge=0.0, le=1.0)
    # 各戦略の比率合計の上限 (generation 余地を確保するため 1.0 未満に抑える)
    budget_max_total_ratio: float = Field(default=0.7, gt=0.0, le=1.0)
    # ポリシー / 進化結果書き戻し設定
    policy: LearningPolicyConfig = Field(default_factory=LearningPolicyConfig)
    # Level 2 (成長) (f_04 §6.7)
    growth: GrowthConfig = Field(default_factory=GrowthConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_flat_layout(cls, data: object) -> object:
        """撤去した ``partition_by_base_model`` を扱う (未知キーの一般の拒否より先に理由を付ける)。

        ``true`` は現行の挙動そのものなので黙って落とす。``false`` (flat
        レイアウト) はもう無いので起動時に拒否する — 設定したのに効かない
        状態を作らない。
        """
        if isinstance(data, dict) and "partition_by_base_model" in data:
            data = dict(data)
            if not data.pop("partition_by_base_model"):
                raise ValueError(
                    "learning.partition_by_base_model was removed: learning data is "
                    "always partitioned by model_key (docs/c_05 §0.5.12). "
                    "Remove the key from config.yaml.",
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def _drop_tolerated_removed_keys(cls, data: object) -> object:
        """PR #952 で廃止した ``long_form_pattern_boost_success`` は既存 config を止めないよう黙って落とす。"""
        if isinstance(data, dict) and "long_form_pattern_boost_success" in data:
            data = {k: v for k, v in data.items() if k != "long_form_pattern_boost_success"}
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_retired_level2(cls, data: object) -> object:
        """撤去した Pro Level 2 のキーを normalize の案内付きで拒否する (未知キーの一般の拒否より先)。

        設定したのに効かない状態を作らない。``evoref config normalize`` が
        config.yaml を退避してから落とす (f_04 §6.7)。
        """
        if isinstance(data, dict):
            # partition_by_base_model は _reject_flat_layout が扱う (true は黙って落とす)
            reject_retired_keys("learning", {k: v for k, v in data.items() if k != "partition_by_base_model"})
        return data
