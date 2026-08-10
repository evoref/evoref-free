"""EvorefMem (WM/STM/LTM + SemMem) 関連スキーマ"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# FadeMem タグ別半減期の有効タグ集合
# `backend.free.memory.types.FactType` と同期させること。
# 同期はテスト (test_config_schema.py::test_valid_fact_types_in_sync) で検証。
VALID_FACT_TYPES: frozenset[str] = frozenset(
    {
        "personal_fact",
        "world_fact",
        "preference",
        "emotion",
        "opinion",
        "belief",
        "decision",
        "commitment",
        "project",
        "policy",
        "fewshot",          # policy subtype から独立昇格
        "failure_pattern",
        "learned_failure_pattern",  # PolicyAdjuster 由来 (Learn owned)
        "progress_marker",
        "task",
        "create_task",
        "artifact",
        "create",
        "model",
    }
)


class SubjectDictionaryConfig(BaseModel):
    """Subject 正規化辞書設定

    EvorefMem 統合仕様 における意味記憶 subject の表記ゆれ吸収
    に使う辞書ファイルの位置付けと挙動を定義する。
    自動拡張は仕様で禁止 (`auto_expand: false` 固定が原則)。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    file: str = "local/memory/semantic/subject_dictionary.json"
    auto_expand: bool = False


class FactsExtractionMaxPerSession(BaseModel):
    """セッションあたりの抽出上限"""

    model_config = ConfigDict(extra="forbid")

    chat: int = Field(default=10, ge=0)
    create: int = Field(default=5, ge=0)


class FactsConfig(BaseModel):
    """ファクト抽出関連設定

    バイパス正規表現のみだったが、sleep-time
    Step 8 (Chat/Create/MDP Extractor) のスイッチと上限値を追加する。
    """

    model_config = ConfigDict(extra="forbid")

    extraction_skip_subject_canonicalize_regex: str = r"^(loop|learn|mem)\."

    # ── 統合追加 ──
    enable_extraction: bool = True
    """Step 8 Extractor 全体のオンオフ"""

    trigger: str = "idle_full_only"
    """抽出トリガ。``idle_full_only`` のみ対応 (Trigger B = run_full)"""

    extraction_max_per_session: FactsExtractionMaxPerSession = Field(
        default_factory=FactsExtractionMaxPerSession,
    )
    """セッションあたりの抽出上限 (chat=10 / create=5)"""

    extraction_max_pinned_per_session: int = -1
    """pinned ノート由来の抽出上限。``-1`` で無制限"""

    extract_from_mdp_trace: bool = True
    """``agent_trace.jsonl`` から MDPTraceExtractor を実行するか"""

    ingest_mdp_trace_to_ltm: bool = True
    """``agent_trace*.jsonl`` を episodic LTM に取り込むか

    Step 7.5 (``_step7_5_ingest_mdp_traces``) で MDPIngester を起動し、
    エピソード単位で ``MemoryNote`` を生成 → ``LongTermMemory`` に投入する。
    ``trace_id`` が contextvars 経由で MemoryNote / SemanticFact に伝播し、
    ファクトとエピソード記憶を相互参照可能にする。"""


class NoteEvolverConfig(BaseModel):
    """NoteEvolver の LLM スキップ閾値

    A-MEM ノート進化 (sleep-time Step 7) での不要な LLM 呼び出しを削減する。
    ノートの ``confidence`` が ``confidence_threshold`` 以上の場合は LLM による
    ``context_description`` 生成を行わず、``evolution_pending=False`` のみ立てて
    ルールベース evolution 扱いとする。``max_per_cycle`` は 1 サイクルあたりの
    LLM 呼び出し上限で、memory が肥大化しても処理時間が線形増加しないように
    するための安全弁。いずれも ``memory.note_evolver`` ネスト以下に配置する。
    """

    model_config = ConfigDict(extra="forbid")

    # confidence がこの値以上のノートは LLM 進化をスキップ (ルールベースのみ)、
    # 未満のノートのみ LLM 進化 (context_description 生成) の対象になる。
    # ノートの初期 confidence は発生源で決まる (NoteBuilder.source_confidence:
    # user=1.0 / assistant=0.5 / rag=system=0.6)。既定 0.7 では user 発話は
    # ルールベース、assistant / rag / system 由来が LLM 進化に回る。
    # 0.0 にすると全ノートがスキップ (LLM 進化なし)、1.0 にすると
    # confidence<1.0 の全ノートが LLM 進化対象になる。1.0 超は設定不可。
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # 1 サイクルあたり LLM 進化呼び出しの上限。既存 ``note_evolution_batch`` と
    # 比較して小さい方が採用される。0 を設定すると LLM 進化を完全に停止する。
    max_per_cycle: int = Field(default=10, ge=0)


class ConflictResolverConfig(BaseModel):
    """ConflictResolver の LLM スキップ閾値

    FadeMem コンフリクト解消 (sleep-time Step 6) での不要な LLM 呼び出しを
    削減する。``detect_conflicts`` で候補として抽出されたペアのうち、
    実際の類似度が ``min_merge_similarity`` 未満のものは LLM 統合を行わず
    低優先度扱いでスキップする (detect 側の閾値 ``conflict_similarity_threshold``
    よりも厳しい値に設定することで、「候補に入るが LLM 呼び出しはしない」
    中間帯を作れる)。``max_per_cycle`` は 1 サイクルあたりの LLM 呼び出し上限。
    いずれも ``memory.conflict_resolver`` ネスト以下に配置する。
    """

    model_config = ConfigDict(extra="forbid")

    # LLM 統合を実行する最小類似度。detect の閾値以上・この値未満のペアは
    # LLM 統合をスキップする。0.0 にすると detect された全ペアを LLM に回す。
    min_merge_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    # 1 サイクルあたり LLM 統合呼び出しの上限。既存 ``conflict_batch_size``
    # と比較して小さい方が採用される。0 を設定すると LLM 統合を完全に停止する。
    max_per_cycle: int = Field(default=5, ge=0)


class ConflictChatReviewConfig(BaseModel):
    """pending 競合のチャット確認フロー設定

    ``review_status="pending"`` の競合をチャットの SemMem 注入で提示し、
    ユーザー回答を assist (``conflict_chat_judge``) で判定して即時解決する
    フローを制御する。
    """

    model_config = ConfigDict(extra="forbid")

    # チャットでの pending 確認・解決フロー全体の有効/無効
    enabled: bool = True
    # 注入するグループ数の上限 (0 = 無制限)。超過分は件数のみ要約する。
    # 2026-07-25: 既定 0 (無制限) では pending が溜まるほどブロックが膨張し、
    # Tier 予算の外で数百トークンを消費していた (実測 16 件 = 840 tokens)。
    max_groups: int = Field(default=3, ge=0)
    # ブロック全体のトークン上限 (0 = 無制限)。グループ数だけでは member 数の
    # 多いグループを抑えられないため併用する。設計書 §203 の「drop されない」
    # 保証のため、上限を超えても最低 1 グループは必ず注入する。
    max_tokens: int = Field(default=400, ge=0)
    # conflict_chat_judge (assist) のセッション内発火上限。上限到達後は
    # 同一セッションでの判定と pending 注入を停止し、滞留 pending が
    # 毎ターン realtime スロットを専有するのを防ぐ。0 = 無制限 (従来動作)。
    max_judge_per_session: int = Field(default=3, ge=0)


class SemMemConflictConfig(BaseModel):
    """SemMem コンフリクト解消設定

    EvorefMem 統合仕様 における sleep-time Step 6 (SemanticFact 競合
    解消) の挙動を定義する。

    - ``project_tag_always_manual``: ``project`` / ``policy`` タグの競合は基本
      ユーザー確認必須
    - ``auto_for_evolved_policies``: 例外として ``auto_evolved=True`` の
      ``policy`` ファクトのみ自動マージする (PolicyEvolver 由来)。
    - ``confirm_window_hours``: 同 source / N 時間以内の対は微妙ケースとして
      確認モードに振り分ける。
    - ``default_mode``: ``auto`` で例外を除き自動マージ、``manual`` で全件を
      ``review_status="pending"`` に振り分ける。
    - ``chat_review``: pending をチャットで確認・解決するフローの制御。
    """

    model_config = ConfigDict(extra="forbid")

    default_mode: Literal["auto", "manual"] = "auto"
    confirm_window_hours: float = Field(default=1.0, ge=0.0)
    project_tag_always_manual: bool = True
    auto_for_evolved_policies: bool = True
    # pending 競合の TTL 自動解消 (日)。グループ内最新ファクトの created_at から
    # N 日経過した pending を sleep-time Step 6B が keep_new で自動解消する。
    # pinned / project / policy タグを含むグループは対象外 (チャット回答でのみ解決)。
    # default_mode=manual でも有効 (無効化は 0 を指定)。
    pending_auto_resolve_days: float = Field(default=3.0, ge=0.0)
    chat_review: ConflictChatReviewConfig = Field(
        default_factory=ConflictChatReviewConfig,
    )


class InjectionTierRatios(BaseModel):
    """モード別 Tier 比率

    4 要素 (Tier 1〜4) の合計が 1.0 になることを期待する。検証はソフトで、
    多少のずれは ``MemoryInjector`` 側で許容する。
    """

    model_config = ConfigDict(extra="forbid")

    chat: list[float] = Field(default_factory=lambda: [0.40, 0.35, 0.15, 0.10])
    create: list[float] = Field(default_factory=lambda: [0.40, 0.35, 0.15, 0.10])

    @model_validator(mode="after")
    def validate_lengths(self) -> "InjectionTierRatios":
        for name, vec in (("chat", self.chat), ("create", self.create)):
            if len(vec) != 4:
                raise ValueError(
                    f"injection.tier_ratios.{name} は 4 要素 (Tier 1〜4) "
                    f"である必要があります (現在: {len(vec)})",
                )
        return self


class InjectionConfig(BaseModel):
    """MemoryInjector 設定

    EvorefMem 統合仕様 におけるモード別 Tier 注入器の設定
    チャット 800 / クリエイト 2000 トークンを既定予算とする。

    ``policy_activation_min_confidence`` は
    ``learning.policy.activation_min_confidence`` (LearningPolicyConfig) に
    統合された (silent inconsistency 解消)。MemoryInjector も learning
    側の値を参照する
    """

    model_config = ConfigDict(extra="forbid")

    chat_budget_tokens: int = Field(default=800, ge=0)
    create_budget_tokens: int = Field(default=2000, ge=0)
    tier_ratios: InjectionTierRatios = Field(default_factory=InjectionTierRatios)
    relevance_enabled: bool = Field(
        default=True,
        description=(
            "クエリ埋め込みとの関連度で注入候補を足切りするか。False で"
            "従来の静的スコアのみ (recency / type / tier) に戻る"
        ),
    )
    relevance_min_score: float = Field(
        default=0.35, ge=0.0, le=1.0,
        description=(
            "関連度ゲートのコサイン類似度閾値。実測 (LFM2.5-Embedding-350M) で"
            "真陽性 0.38〜0.44 / ノイズ中央値 0.12〜0.17"
        ),
    )
    pinned_relevance_min_score: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description=(
            "pinned ファクトに課す関連度の下限。pin は優先度の指定であって"
            "「常に関連する」の宣言ではなく、しかも「覚えておいてください」等の"
            "語で自動 pin されるため、0 (完全迂回) だと無関係なターンにも毎回"
            "載り続ける。実測 2026-08-09: 0 → 0.10 で記憶不要ターンの注入が"
            "297→81 token (約 2.2→0.6 秒) に減り、想起ターンは注入量・正答とも"
            "無変化。0.35 で想起が壊れ始める"
        ),
    )


class PinConfig(BaseModel):
    """Pin 機能設定

    EvorefMem 統合仕様 における Pin 機能の有効化と
    自動 Pin 検出 (「覚えて」「重要」「忘れないで」等) の挙動を定義する。
    """

    model_config = ConfigDict(extra="forbid")

    auto_detect: bool = True
    """自動 Pin 検出を有効化するか"""

    auto_detect_confirm: bool = False
    """検出時にユーザー確認を要求するか"""

    unlimited: bool = True
    """pin 数上限なし"""


class PrivateSessionConfig(BaseModel):
    """プライベートセッション設定

    `private: true` のターンは memory_only で動作し、LTM/SemMem に書き込まない。
    会話履歴のディスク永続化もスキップする。セッション終了時に揮発する。
    """

    model_config = ConfigDict(extra="forbid")

    default: bool = False
    """セッション開始時のデフォルト private モード"""

    history_storage: str = Field(default="memory_only", pattern=r"^(memory_only|skip)$")
    """history 保存方針。

    - ``memory_only`` (既定): private ターンだけをディスク永続化から除外し、
      WM/STM のみで保持する。同じセッションの通常ターンは履歴に残る。
    - ``skip``: private ターンを 1 度でも含んだセッションは、通常ターンも含めて
      セッションファイルごと永続化しない。
    """


class SemMemLimitsConfig(BaseModel):
    """SemMem 容量上限と GC 戦略

    EvorefMem 統合仕様 における意味記憶の type 別上限と
    超過時の削除戦略を定義する。sleep-time Step 9 内で実施される
    GC が本設定を参照する。``enforcement=hard`` の場合、上限を超えた
    type は ``gc_strategy`` に従って下位スコアから削除される。
    pinned ファクトは常に削除対象外。

    スコア計算順位:
    1. ``eval_metric.fitness`` が存在すればこれを使用 (policy 用)
    2. それ以外は ``confidence``
    タイブレーカ: ``access_count``、``accessed_at`` (古い順)。
    """

    model_config = ConfigDict(extra="forbid")

    personal_fact: int = Field(default=10000, ge=0)
    world_fact: int = Field(default=5000, ge=0)
    preference: int = Field(default=5000, ge=0)
    emotion: int = Field(default=3000, ge=0)
    opinion: int = Field(default=3000, ge=0)
    belief: int = Field(default=1000, ge=0)
    decision: int = Field(default=5000, ge=0)
    commitment: int = Field(default=2000, ge=0)
    project: int = Field(default=10000, ge=0)
    create: int = Field(default=10000, ge=0)
    task: int = Field(default=5000, ge=0)
    create_task: int = Field(default=5000, ge=0)
    policy: int = Field(default=5000, ge=0)
    fewshot: int = Field(default=2000, ge=0)  # policy subtype から独立
    failure_pattern: int = Field(default=2000, ge=0)
    learned_failure_pattern: int = Field(default=2000, ge=0)  # PolicyAdjuster 由来
    progress_marker: int = Field(default=5000, ge=0)
    artifact: int = Field(default=10000, ge=0)
    model: int = Field(default=1000, ge=0)
    enforcement: Literal["hard", "soft"] = "hard"
    gc_strategy: Literal["lowest_score"] = "lowest_score"

    def limit_for(self, fact_type: str) -> int | None:
        """``fact_type`` の上限を返す。未定義 type は ``None``。"""
        return getattr(self, fact_type, None) if fact_type in VALID_FACT_TYPES else None


class SemMemProjectConfig(BaseModel):
    """プロジェクトアーカイブ設定

    sleep-time Step 10 がアクセスのないプロジェクトを ``archive_dir``
    へ移動する閾値とパスを指定する。``auto_archive_inactive_days=0``
    でアーカイブを無効化できる (検証用)。
    """

    model_config = ConfigDict(extra="forbid")

    auto_archive_inactive_days: int = Field(default=180, ge=0)
    archive_dir: str = "local/memory/semantic/archive"


class MemoryConfig(BaseModel):
    """メモリシステム設定"""

    model_config = ConfigDict(extra="forbid")

    # EvorefMem スキーマバージョン
    # 不一致時は scripts/init_evorefmem.py による初期化を促す。
    # code 側の backend.free.memory.init_evorefmem.SCHEMA_VERSION が最終的な
    # source of truth で、cfg 値は参考情報として扱われる。divergent な場合は
    # 起動時に WARN ログを出した上で code 側の値が採用される。
    schema_version: int = Field(default=1, ge=1)
    # Subject 正規化辞書
    subject_dictionary: SubjectDictionaryConfig = Field(
        default_factory=SubjectDictionaryConfig,
    )
    # ファクト抽出
    facts: FactsConfig = Field(default_factory=FactsConfig)
    # Pin 機能
    pin: PinConfig = Field(default_factory=PinConfig)
    # プライベートセッション
    private: PrivateSessionConfig = Field(default_factory=PrivateSessionConfig)
    # SemMem コンフリクト解消
    conflict: SemMemConflictConfig = Field(default_factory=SemMemConflictConfig)
    # MemoryInjector
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    # SemMem 容量上限 + GC 戦略
    semmem_limits: SemMemLimitsConfig = Field(default_factory=SemMemLimitsConfig)
    # プロジェクトアーカイブ
    project: SemMemProjectConfig = Field(default_factory=SemMemProjectConfig)
    # ターン数はトークン上限より先に効かせない。ターン数超過は無圧縮の
    # ハード eviction (古い発言が丸ごと消える) だが、トークン超過は
    # compress_turn による段階的縮退なので、後者を主たる制約にする。
    # 10 だと実測 ~59 tok/turn で 590 tok しか使わず working_max_tokens=2048 の
    # 3 割未満で足切りされ、5 往復前の自己紹介が文脈から消えていた
    # (2026-07-25 実測: 20 ターン会話で名前・職業・趣味を想起できず
    #  search_history へ不要フォールバック)。
    #
    # **単位はメッセージ数** (WorkingMemory.add_turn は user / assistant を
    # それぞれ 1 件として積む)。名前は "turns" だが 30 = 15 往復であり、
    # 30 往復ではない (2026-08-05 ライブ監査: 40 ターンの会話でターン 22 以降
    # しか見えず、会話全体を走査する質問が前半を「無い」と断定した)。
    # 上限を超えた分は WorkingMemory.session_evicted_turns に計上され、
    # 全体走査質問には切り詰め注記が付く (chat_service._append_truncated_history_note)。
    working_max_turns: int = Field(default=30, ge=1)
    working_max_tokens: int = Field(default=2048, ge=256)
    # 上限に達したときに **まとめて** 押し出すターン数 (ヒステリシス)。
    # 1 ターンずつ削ると窓の先頭が毎ターン動き、llama-server の接頭辞 KV
    # キャッシュが system プロンプト以降まるごと無効化される。実測
    # (2026-08-09 ライブ監査、base=gemma-4-12b / 8192 ctx): prompt eval が
    # 2,898 トークンで 27.7 秒かかる一方 decode は 54 トークンで 7.1 秒
    # (7.6 tok/s = iGPU の正常値)。LCP 類似度 f_sim_best は毎ターン 0.35 前後で、
    # 連続プロンプトの共通接頭辞は常に system プロンプト長ちょうどだった。
    # 8〜14 トークンの短い応答でも 28〜40 秒かかる原因。
    # 1 で従来どおり (1 件ずつ)。既定 6 で押し出し回数が約 1/3 になり、
    # 保持ターン数の平均は max_turns の約 90% を維持する。
    working_evict_block: int = Field(default=6, ge=1)
    # 過去履歴の最低確保トークン数 (床)。動的ブロック (few-shot/file/semmem/RAG) の
    # 配分前に予約し、予算圧迫時でも直近の会話文脈が丸ごと締め出されるのを防ぐ。
    # 実履歴量・残予算・working_max_tokens でキャップされ、履歴が現在の質問のみの
    # 新規セッションでは 0 に縮退する。0 で無効。
    # フォールバック定数は chat_constants.DEFAULT_HISTORY_MIN_TOKENS と同期させること。
    history_min_tokens: int = Field(default=1024, ge=0)
    short_term_max_notes: int = Field(default=100, ge=1)
    lightmem_decay_days: int = Field(default=7, ge=1)
    # FadeMem タグ別半減期 (日)
    # キーは VALID_FACT_TYPES (FactType と同期) のいずれか。値は正の整数 (日)。
    # 未指定タグは lightmem_decay_days (デフォルト半減期) を使用。
    half_life_days_by_tag: dict[str, int] = Field(default_factory=dict)
    fade_alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    fade_beta: float = Field(default=0.3, ge=0.0, le=1.0)
    fade_gamma: float = Field(default=0.3, ge=0.0, le=1.0)
    fade_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    conflict_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    conflict_batch_size: int = Field(default=5, ge=1)
    note_evolution_enabled: bool = True
    note_evolution_batch: int = Field(default=10, ge=1)
    note_evolution_context_k: int = Field(default=3, ge=1)
    llm_call_base_interval: float = Field(default=1.0, ge=0.0)
    # A-MEM リンク張り直し + クラスタリング
    note_link_rebuild_enabled: bool = True
    note_link_top_k: int = Field(default=5, ge=1, le=50)
    note_link_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    note_clustering_enabled: bool = True
    # NoteEvolver / ConflictResolver の LLM スキップ閾値
    note_evolver: NoteEvolverConfig = Field(default_factory=NoteEvolverConfig)
    conflict_resolver: ConflictResolverConfig = Field(
        default_factory=ConflictResolverConfig,
    )

    @model_validator(mode="after")
    def validate_fade_weights(self) -> "MemoryConfig":
        """FadeMem の重み合計が 1.0 であることを検証"""
        total = self.fade_alpha + self.fade_beta + self.fade_gamma
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"fade_alpha + fade_beta + fade_gamma = 1.0 である必要があります"
                f"（現在: {self.fade_alpha} + {self.fade_beta} + {self.fade_gamma} = {total}）"
            )
        return self

    @model_validator(mode="after")
    def validate_half_life_days_by_tag(self) -> "MemoryConfig":
        """half_life_days_by_tag のキーは FactType、値は正の整数であることを検証"""
        invalid_keys: list[str] = []
        invalid_values: list[str] = []
        for key, value in self.half_life_days_by_tag.items():
            if key not in VALID_FACT_TYPES:
                invalid_keys.append(key)
            if not isinstance(value, int) or value <= 0:
                invalid_values.append(f"{key}={value!r}")
        errors: list[str] = []
        if invalid_keys:
            errors.append(
                "half_life_days_by_tag に未知のタグがあります "
                f"(FactType 以外): {sorted(invalid_keys)}"
            )
        if invalid_values:
            errors.append(
                "half_life_days_by_tag の値は正の整数 (日) である必要があります: "
                f"{invalid_values}"
            )
        if errors:
            raise ValueError("; ".join(errors))
        return self
