"""RAG / 埋め込み / リランカー関連スキーマ"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CartridgeGateConfig(BaseModel):
    """Cartridge Gate (centroid 事前フィルタ) 設定"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # cosine 類似度閾値。この値未満の centroid を持つカートリッジはスキップ
    threshold: float = Field(default=0.3, ge=-1.0, le=1.0)
    # gate 通過させる上限件数 (0 以下で無制限)
    max_cartridges: int = Field(default=10, ge=0)
    # 全カートリッジが threshold 未満で 0 件通過になったときの挙動:
    #   False (既定): カートリッジ検索を skip し空リストを返す。
    #                 雑談・ファイル生成依頼など、ロード中カートリッジと
    #                 無関係な発話で RAG chunk が混入するのを防ぐ。
    #   True       : 全件フォールバック (旧挙動)。recall 重視。
    fallback_when_empty: bool = False


class ClusterIndexConfig(BaseModel):
    """Cluster Index (IVF-KMeans) 設定"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # この件数以上の VectorStore で cluster index を構築
    threshold: int = Field(default=5000, ge=100)
    # K = max(16, sqrt(N)) のうち n_probe_ratio × K 個のクラスタを探索
    n_probe_ratio: float = Field(default=0.125, gt=0.0, le=1.0)


class ContextualPrefixConfig(BaseModel):
    """Contextual Retrieval プレフィックス生成設定

    Anthropic 方式の Contextual Retrieval プレフィックスを
    チャンクに付与するための設定。eager (Sleep-time 一括生成) と
    lazy (retrieval 時 on-demand) の 2 モードをサポートする。
    """

    model_config = ConfigDict(extra="forbid")

    # 機能自体の有効/無効。無効時はプレフィックス生成を完全スキップ。
    enabled: bool = True
    # 生成モード: "eager" (Sleep-time 一括) / "lazy" (retrieval 時 on-demand)
    mode: str = Field(default="eager", pattern=r"^(eager|lazy)$")
    # アシストモデルに要求する最大トークン数 (プレフィックス 1 件あたり)
    max_tokens: int = Field(default=128, ge=1)
    # プロンプトに埋め込むドキュメント本文の最大文字数 (超過時はトランケート)
    max_doc_chars: int = Field(default=6000, ge=100)
    # Sleep-time 1 サイクルで処理するチャンクの上限 (eager/lazy 両方で適用)
    batch_size: int = Field(default=10, ge=1)
    # 生成対象となる最小チャンクトークン数。この値未満はスキップする。
    # metadata の `tokens` フィールド (tiktoken cl100k_base 概算) を基準とする。
    min_chunk_tokens: int = Field(default=200, ge=0)
    # lazy モード時のヒット閾値。retrieval でこの回数以上ヒットした chunk のみ
    # プレフィックスを永続化する。1 回目のヒットでは生成せずカウントのみ更新。
    lazy_hit_threshold: int = Field(default=2, ge=1)
    # プレフィックス生成プロンプトのテンプレート。プレースホルダ
    # ``{document}`` ``{chunk}`` をサポート。多言語対応や、英語専用 / 中国語
    # 専用アシストモデルに切替えた際にこのテンプレートを差し替えることで、
    # アシストモデルの学習言語と一致させる。空文字列にすると default
    # (日本語) が利用される。
    prompt_template: str = ""


class SelfRagAssistJudgeConfig(BaseModel):
    """Self-RAG アシスト品質再判定設定

    ルールベース Self-RAG の quality 判定が ``only_when_quality`` に
    該当した場合に、アシストモデル LLM による品質再判定で marginal な
    検索結果を救済する。セッション / クエリ単位で発火上限を設けて、
    ユーザ体感レイテンシとアシストモデル負荷への影響を抑える。
    """

    model_config = ConfigDict(extra="forbid")

    # 機能の有効/無効。無効時はルールベース判定のみで確定する。
    enabled: bool = True
    # 1 セッション (session_id 単位) で発火可能な最大回数。0 以下で無制限。
    # 上限超過時は raw hybrid 結果で返し、DebugLogger に
    # ``assist_judge_skipped_reason="session_cap"`` を記録する。
    max_per_session: int = Field(default=5, ge=0)
    # 1 クエリ (= 1 unified_search 呼び出し) 内の最大発火回数。
    # 現行実装では LLM 再判定は 1 クエリあたり 1 回しか呼ばれないが、
    # 将来の多段判定導入に備えた防御線として残す。0 以下で無制限。
    max_per_query: int = Field(default=1, ge=0)
    # 発火条件の quality ラベル集合。ルールベース判定がこのいずれかに
    # 該当した場合のみ LLM 再判定を呼ぶ。既定は marginal ラベルのみ
    # (high = 高信頼なので不要、low = クエリ拡張フォールバック側で処理)。
    only_when_quality: list[str] = Field(default_factory=lambda: ["medium"])

    @model_validator(mode="after")
    def validate_only_when_quality(self) -> "SelfRagAssistJudgeConfig":
        valid = {"high", "medium", "low"}
        invalid = [q for q in self.only_when_quality if q not in valid]
        if invalid:
            raise ValueError(
                "self_rag.assist_judge.only_when_quality に不正な値があります: "
                f"{invalid}。許容値は {sorted(valid)}",
            )
        return self


class SelfRagAssistNecessityConfig(BaseModel):
    """Self-RAG アシスト検索必要性判定設定

    ルールで `retrieve` / `skip` を確定できない短い質問 (uncertain) のみ
    アシストモデルに 1 bit JSON (`{"need_rag": bool}`) を問い、`true` なら
    retrieve、`false` なら skip。失敗 / タイムアウト / 上限到達時は安全側
    (`retrieve`) にフォールバック。
    """

    model_config = ConfigDict(extra="forbid")

    # 機能の有効/無効。無効時は純ルール判定のみで確定する。
    enabled: bool = True
    # 1 セッション (session_id 単位) で発火可能な最大回数。0 以下で無制限。
    max_per_session: int = Field(default=10, ge=0)
    # 1 クエリ内の最大発火回数。0 以下で無制限。
    max_per_query: int = Field(default=1, ge=0)
    # tracker キー互換 (本機能の quality ラベルは "uncertain" のみ)
    only_when_quality: list[str] = Field(default_factory=lambda: ["uncertain"])
    # アシスト wall-clock 上限 (秒)。超過時は安全側 retrieve にフォールバック。
    timeout_s: float = Field(default=5.0, gt=0.0, le=60.0)
    # アシストプロンプトに埋め込む直前ターン数 (user + assistant 合計)。
    # 0 で会話文脈を含めない (旧挙動)。1〜2 が推奨。
    context_turns: int = Field(default=2, ge=0, le=10)


class SelfRagContentGateConfig(BaseModel):
    """取得直後の chunk 内容精査ゲート設定 (heuristics-first + 境界 assist)

    ``unified_search`` の Step 4 マージ直後に低価値 chunk を pruning し、
    quality judge / query expansion の候補数を縮小
    する。coding mode を主対象とし (chat mode は近似重複除去のみ)、安価な
    ヒューリスティック (relevance floor / 近似重複除去 / コードシグナル) で
    大半を裁き、判断に迷う marginal band の prose チャンクだけアシストモデル
    で 1 回判定する。``assist_client=None`` / cap 超過 / error 時はヒューリス
    ティックのみで確定する。
    """

    model_config = ConfigDict(extra="forbid")

    # 機能の有効/無効。スキーマ既定は保守的に OFF (config.yaml.example 側で
    # true を記載し、新規セットアップでは coding-mode 精査が有効になる)。
    enabled: bool = False
    # relevance スコア下限 (cosine スケール 0-1)。これ未満を pruning する。
    # unified_search は STM/LTM/カートリッジの cosine 類似度が基準。
    relevance_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    # [floor, floor+marginal_band) を「判断に迷う帯」とし、coding mode では
    # この帯の prose チャンクのみ assist 判定に回す。
    marginal_band: float = Field(default=0.10, ge=0.0, le=0.5)
    # 最低保持件数。pruning がこれを下回ったら上位から補填し生成を枯渇させない。
    min_keep: int = Field(default=3, ge=1)
    # 近似重複除去の token-set Jaccard しきい値 (1.0 で重複除去を無効化)。
    dedup_jaccard: float = Field(default=0.85, ge=0.0, le=1.0)
    # coding mode で marginal 帯のコードシグナル判定を有効化する。
    coding_code_signal: bool = True
    # marginal band の assist 救済を有効化する (false で純ヒューリスティック)。
    assist_enabled: bool = True
    # 1 セッションでの assist 発火上限 (0 以下で無制限)。
    max_per_session: int = Field(default=5, ge=0)
    # 1 クエリでの assist 発火上限 (0 以下で無制限)。
    max_per_query: int = Field(default=1, ge=0)


class SelfRagConfig(BaseModel):
    """Self-RAG 設定ルート

    ルールベース Self-RAG の補助機能 (アシスト LLM 再判定 / 検索必要性
    判定) をネスト構造に集約する。
    """

    model_config = ConfigDict(extra="forbid")

    assist_judge: SelfRagAssistJudgeConfig = Field(
        default_factory=SelfRagAssistJudgeConfig,
    )
    assist_necessity: SelfRagAssistNecessityConfig = Field(
        default_factory=SelfRagAssistNecessityConfig,
    )
    content_gate: SelfRagContentGateConfig = Field(
        default_factory=SelfRagContentGateConfig,
    )


class RAGConfig(BaseModel):
    """RAG 設定"""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=128, ge=0)
    chunking_strategy: str = Field(default="semantic", pattern=r"^(semantic|fixed)$")
    semantic_min_chunk: int = Field(default=64, ge=1)
    semantic_max_chunk: int = Field(default=512, ge=1)
    top_k: int = Field(default=5, ge=1, le=50)
    # LTM / cartridge の取得件数を top_k*N へ拡張する倍率
    # (STM は stm_top_k 固定で拡張対象外)。
    # 既定 1 = 拡張なし。「広く取って絞る」第1段として候補プールを広げる。
    # LTM は内部で更に top_k*2 するため VectorStore 実 fetch は
    # top_k*N*2 となり、上限 5 を超えると N>=5000+IVF 環境で recall ガード経由の
    # 全件走査フォールバックに倒れやすくなるため le=5 で制限する。
    fetch_multiplier: int = Field(default=1, ge=1, le=5)
    # クロスレイヤ・スコア正規化方式。STM (cosine*0.6+lightmem*0.4)
    # / LTM (raw cosine) / cartridge (cosine*priority, priority 無上限) という異種スケールを
    # 層内正規化で [0,1] に揃えてから融合し、最終順位付けの歪みを抑える。
    # none (既定) = 従来 (生スコア降順)。minmax = 層内 min-max。rank = 層内ランク減衰。
    score_normalization: str = Field(default="none", pattern=r"^(none|minmax|rank)$")
    hybrid_search: bool = True
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    fusion_method: str = Field(default="rrf", pattern=r"^(rrf|weighted)$")
    rrf_k: int = Field(default=60, ge=1)
    # --- BM25 チューニング ---
    bm25_k1: float = Field(default=1.5, ge=0.0, le=5.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    bm25_delta: float = Field(default=1.0, ge=0.0, le=5.0)
    bm25_use_trigrams: bool = False
    bm25_split_ascii: bool = True
    # None → 既定ストップワード (DEFAULT_STOPWORD_BIGRAMS) を使用
    # []   → ストップワード無効
    # list → 指定リストを使用
    bm25_stopword_bigrams: list[str] | None = None
    # --- Contextual Retrieval ---
    contextual_prefix: ContextualPrefixConfig = Field(
        default_factory=ContextualPrefixConfig,
    )
    # --- ベクトル量子化 ---
    quantization: str = Field(default="none", pattern=r"^(none|int8)$")
    rescore_candidates: int = Field(default=50, ge=0)
    # --- memmap ---
    memmap_threshold: int = Field(default=10000, ge=100)
    # --- Cartridge スケーラビリティ ---
    # 同時 loaded 可能なカートリッジ上限。超過時は LRU で最古参を unload する。
    max_loaded_cartridges: int = Field(default=20, ge=1, le=500)
    # load 時にチャンク数がこの閾値を超えるカートリッジは WARNING ログを出す。
    large_cartridge_warn_chunks: int = Field(default=50000, ge=1000)
    # カートリッジ検索全体のタイムアウト (ミリ秒)。超過時は途中結果を返す。
    # 0 以下でタイムアウト無効。
    cartridge_search_timeout_ms: int = Field(default=3000, ge=0)
    # Cartridge Gate: centroid ベースの事前フィルタ
    cartridge_gate: CartridgeGateConfig = Field(default_factory=CartridgeGateConfig)
    # Cluster Index: IVF-KMeans による大規模 VectorStore 高速化
    cluster_index: ClusterIndexConfig = Field(default_factory=ClusterIndexConfig)
    # --- Self-RAG 品質判定閾値 ---
    relevance_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    support_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    hysteresis_band: float = Field(default=0.02, ge=0.0, le=0.5)
    # --- Self-RAG 補助機能 ---
    # 旧 `assist_judge_enabled` は本ネスト構造 (`self_rag.assist_judge.enabled`)
    # へ移行された。後方互換は維持しない。
    self_rag: SelfRagConfig = Field(default_factory=SelfRagConfig)

    @model_validator(mode="after")
    def validate_rag_constraints(self) -> "RAGConfig":
        """RAG 固有の制約を検証"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) は "
                f"chunk_size ({self.chunk_size}) より小さい必要があります"
            )
        if self.semantic_min_chunk > self.semantic_max_chunk:
            raise ValueError(
                f"semantic_min_chunk ({self.semantic_min_chunk}) は "
                f"semantic_max_chunk ({self.semantic_max_chunk}) 以下である必要があります"
            )
        if self.fusion_method == "weighted":
            total = self.bm25_weight + self.vector_weight
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"fusion_method='weighted' の場合、"
                    f"bm25_weight + vector_weight = 1.0 である必要があります（現在: {total}）"
                )
        return self


_DEFAULT_INSTRUCTIONS: dict[str, str] = {
    "chat": "Given a user question, retrieve relevant passages that answer the query",
    "coding": "Given a code search query, retrieve relevant code snippets",
}

# 埋め込みクエリ整形テンプレート (Qwen3-Embedding 公式仕様)
# {task} = instructions[mode] / {query} = 元クエリ
_DEFAULT_EMBED_QUERY_TEMPLATE = "Instruct: {task}\nQuery: {query}"


class EmbeddingConfig(BaseModel):
    """埋め込みモデル設定"""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(default="llama-cpp", pattern=r"^(llama-cpp)$")
    # llama-cpp バックエンド用
    llama_host: str = "localhost"
    llama_port: int = Field(default=8082, ge=1024, le=65535)
    dim: int = Field(default=1024, ge=1)
    timeout: float = Field(default=30.0, ge=0.1)
    max_length: int = Field(default=8192, ge=1)
    # 文脈長 (llama-server ``-c``)。埋め込みは max_length トークンまでの入力を
    # 扱うため、必ず max_length 以上にすること (下回ると長い入力で 500)。モデル
    # 既定 n_ctx (例: Qwen3-Embedding-0.6B=32768) は過剰で KV を浪費するため、
    # max_length に合わせて縮小する (実測 0.6B で active WorkingSet 7.5GB→4.3GB、
    # 速度・次元は同等)。
    context_size: int = Field(default=8192, ge=1)
    # 共通
    model_name: str = "qwen3-embedding"
    # Qwen3-Embedding 系の instruction-aware プレフィックス
    # ``is_query=True`` のときに ``query_template`` で整形する。``mode`` は
    # ``chat`` / ``coding`` のいずれか。ドキュメント側 (``is_query=False``)
    # は ``doc_template`` で整形する (既定では空のため prefix なし)。
    instructions: dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_INSTRUCTIONS),
    )
    # クエリ整形テンプレート。プレースホルダ ``{task}`` (= instructions[mode])
    # と ``{query}`` をサポートする。空文字列にすると instruction prefix を
    # 一切付与しない (BGE-M3 等の非 instruction-aware モデル運用)。
    # Qwen3-Embedding 既定: ``"Instruct: {task}\nQuery: {query}"``
    query_template: str = Field(default=_DEFAULT_EMBED_QUERY_TEMPLATE)
    # ドキュメント整形テンプレート。Qwen3 仕様では空 (prefix なし)。
    # 対称型埋め込み (例: e5) で文書側にも prefix が要る場合のみ設定する。
    # プレースホルダ ``{task}`` ``{query}`` (= 文書本文) をサポート。
    doc_template: str = ""
    # Pooling 方式 (llama-server --pooling)。既定 None はフラグを付与せず
    # llama-server 側のモデル既定 pooling (GGUF pooling_type メタデータ or
    # ビルトイン既定) に委ねる — 既存モデル (Qwen3-Embedding 等) の挙動を
    # 変えない。BGE-M3 (arch "bert") は CLS pooling が正しく、embed 切替時に
    # models/profiles/bert.yaml から "cls" が自動転写される (手動設定は
    # 通常不要)。値は llama-server --pooling が受理する 5 値のみ許容する。
    pooling: Literal["none", "mean", "cls", "last", "rank"] | None = None
    # GPU オフロード層数
    # None の場合は CPU フォールバック (``-ngl 0``) が既定となる。
    # GPU に乗せたい場合は明示的に 999 等を指定する。ベースモデルの
    # ``llama.gpu_layers`` には追従しない (CPU default はユーザ明示の opt-in で
    # のみ上書きされる)。
    gpu_layers: int | None = Field(default=None, ge=-1)
    # 物理バッチサイズ。max_length トークン分の単一入力を 1 回で処理できる値に揃える。
    # llama-server デフォルト 512 のままだと、長い STM ノート (例: 933 tok) の埋め込み
    # リクエストが 500 エラーになり EvorefMem sleep-time update が連鎖失敗する
    batch_size: int | None = Field(default=None, ge=1)
    ubatch_size: int | None = Field(default=None, ge=1)
    # スレッド数 (llama-server ``-t``)。0 = 省略し llama.cpp 自動検出 (全物理コア)。
    # CPU 埋め込み (gpu_layers=null/0) 時に base/assist と CPU を分け合うため、
    # 物理コア数より小さい値を明示してヘッドルームを残せる。
    threads: int = Field(default=0, ge=0)
    # 並列スロット数 (llama-server ``-np``)。明示しないと n_parallel=auto (=4) が
    # 選ばれ slots × context_size 分の KV を無駄に確保する。埋め込みは概ね逐次
    # アクセスのため既定 2 (前景クエリ + 背景ノート埋め込みの最小並列) で十分。
    slots: int = Field(default=2, ge=1, le=16)
    # idle slot offload。埋め込みは chat slots を使わないため
    # 上流既定 8192 MiB の RAM 予約は無意味。0 で明示 disable する。
    cache_ram_mib: int = Field(default=0, ge=-1)
    # --- 永続埋め込みキャッシュ ---
    cache_enabled: bool = True
    cache_max_mb: int = Field(default=100, ge=1)
    cache_dir: str = "local/cache/embeddings/"
    # 起動時に embedder dim と既存ベクトルの dim 不整合を検出した際、
    # 自動で ``run_reindex`` を実行するか。``false`` (既定) では
    # ``state.embedding_dim_mismatch`` を立てて WARNING ログのみ。
    # ``true`` で運用者が ``evoref reindex`` を手動実行する手間を省く
    # (モデル切替直後の RAG ダウンタイムを縮める)。
    auto_reindex_on_mismatch: bool = False


