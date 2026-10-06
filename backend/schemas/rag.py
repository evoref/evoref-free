"""RAG / 埋め込み関連スキーマ"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CartridgeGateConfig(BaseModel):
    """Cartridge Gate (centroid 事前フィルタ) 設定"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # cosine 類似度閾値。この値未満の centroid を持つカートリッジはスキップ。
    # cos の絶対値は埋め込みモデルの sim 分布に依存する (無関係ペアの中央値が
    # LFM2.5 0.105 / Qwen3 0.273 / bge-m3 0.459)。None (既定) = 有効な埋め込み
    # モデルプロファイル (models/profiles/<arch>.yaml の
    # embedding.rag.cartridge_gate_threshold) の値を使い、プロファイルに無ければ
    # 0.3。明示値はプロファイルより優先 (cartridge_manager.resolve_cartridge_gate_threshold)。
    threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    # gate 通過させる上限件数 (0 以下で無制限)
    max_cartridges: int = Field(default=10, ge=0)
    # 全カートリッジが threshold 未満で 0 件通過になったときの挙動:
    #   False (既定): カートリッジ検索を skip し空リストを返す。
    #                 雑談・ファイル生成依頼など、ロード中カートリッジと
    #                 無関係な発話で RAG chunk が混入するのを防ぐ。
    #   True       : 全件フォールバック (旧挙動)。recall 重視。
    fallback_when_empty: bool = False


class PackagesConfig(BaseModel):
    """corpus パッケージ install の入口検査 (c_16 §4.3)。

    展開前の zip 本体サイズと、展開後の合計サイズ (zip bomb 対策) を検査する。
    """

    model_config = ConfigDict(extra="forbid")

    # .evocart zip 本体のサイズ上限 (バイト)。既定 256MiB。
    max_package_bytes: int = Field(default=268_435_456, ge=1)
    # 展開後の合計サイズ上限 (バイト)。既定 1GiB。
    max_unpacked_bytes: int = Field(default=1_073_741_824, ge=1)


class ClusterIndexConfig(BaseModel):
    """Cluster Index (IVF-KMeans) 設定"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # K = max(16, sqrt(N)) のうち n_probe_ratio × K 個のクラスタを探索
    n_probe_ratio: float = Field(default=0.125, gt=0.0, le=1.0)


class PseudoQueryConfig(BaseModel):
    """corpus パッケージの疑似クエリ索引 (f_01 §6)。

    チャンクごとに「このチャンクが答える問い」を sleep-time (Step 5.9) で
    補助タスクに作らせ、その埋め込みを別リストとして検索し疑似クエリ側を
    先頭に interleave する。応答パスに LLM は入らない。
    """

    model_config = ConfigDict(extra="forbid")

    # 生成と検索の両方を切る (既存の索引は残る)。
    enabled: bool = True
    # 1 チャンクあたりの問いの本数。decode は問いの数に比例する (27B で 1 問
    # ≈ 7 秒) ので既定 1。取りこぼした問い (misses) のあるチャンクは言い換えを
    # 1 つ足す (f_01 §6.4)。
    questions_per_chunk: int = Field(default=1, ge=1, le=5)
    # Full サイクル 1 回で生成するチャンク数の上限 (27B で 1 件 20 秒級)。静穏窓
    # + 横取りでサイクルを畳む協調 yield があるので、上限を小さく保つ理由は
    # 「Full を短く終える」だけ。20 では 1124 チャンクの充足 50% に 28 サイクル要る。
    max_per_cycle: int = Field(default=20, ge=1)
    # ヒットの無いチャンクも snapshot 行順に埋める件数。疑似クエリの充足率が
    # 関連性ゲートの前提なので既定で埋める。0 で lazy のみ。
    backfill_per_cycle: int = Field(default=20, ge=0)
    # 1 サイクルの生成に使う壁時計の予算 (秒)。件数より先にこちらで畳む —
    # 1 件 20 秒級なので件数上限だけでは Full が 20 分伸びる (2026-09-12 実測、
    # 記憶の整理を待たせていた)。チャンク境界で判定し、書いた分は commit する。
    # 0 で無効 (件数だけ)。
    budget_seconds: float = Field(default=180.0, ge=0.0)
    # 生成プロンプトへ渡す本文の上限文字数。
    max_chunk_chars: int = Field(default=1200, ge=100)
    # 疑似クエリの関連性ゲート (Step 3d の拒否) を有効にする充足率 (問いを持つ
    # チャンク / 全チャンク) の下限。較正済みの棒そのものは充足率に関わらず使う
    # (棒の妥当性は標本数で決まる、f_01 §6.6)。
    gate_min_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    # 静穏窓 (秒)。チャットが終わってからこの秒数は Step 5.9 を始めない。
    # 1 件 20 秒級の生成をターンの合間に始めると次のターンに横取りされ、
    # その往復で TTFT を壊す (2026-09-12 実測 20 s → 55 s)。
    quiet_seconds: float = Field(default=60.0, ge=0.0)


class ProjectMapUpdateConfig(BaseModel):
    """ProjectMap の fingerprint 差分更新の分類しきい値 (c_16 §4.4)。"""

    model_config = ConfigDict(extra="forbid")

    # 構造変更ファイル数がこれを超える (またはトップレベルディレクトリ集合が
    # 変化する) と ``architecture`` 分類になる。
    architecture_threshold: int = Field(default=10, ge=1)
    # 構造変更ファイル数がこれを超える (または総ファイル数の full_ratio 以上)
    # だと ``full`` 分類になる。
    full_threshold: int = Field(default=30, ge=1)
    full_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    # 静穏窓 (秒)。5.9 (疑似クエリ) と同じ「チャットが終わってから N 秒は
    # 始めない」判定を使う。
    quiet_seconds: float = Field(default=120.0, ge=0.0)


class ProjectMapConfig(BaseModel):
    """既存プロジェクトの code グラフ (ProjectMap、c_16 §4.4)。

    tree-sitter による決定論抽出で file / class / function ノードと
    contains / imports / calls / inherits 辺を corpus の 1 パッケージとして
    持つ。書き手は sleep-time Full の Step 5.87 のみ (応答パスは読むだけ)。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # 走査対象 (プロジェクトルート相対)。
    roots: list[str] = Field(default_factory=lambda: ["."])
    # 既定の除外ディレクトリ (.git / node_modules / __pycache__ / .venv / venv /
    # models / local / dist / build / htmlcov / coverage) への追加 glob。
    exclude_globs: list[str] = Field(default_factory=list)
    max_file_bytes: int = Field(default=1_000_000, ge=1)
    neighborhood_depth: int = Field(default=2, ge=1, le=5)
    neighborhood_budget_tokens: int = Field(default=800, ge=1)
    update: ProjectMapUpdateConfig = Field(default_factory=ProjectMapUpdateConfig)


class SelfRagContentGateConfig(BaseModel):
    """取得直後の chunk 内容精査ゲート設定 (heuristics-first + 境界 LLM 判定)

    ``unified_search`` の Step 4 マージ直後に低価値 chunk を pruning し、
    quality judge / query expansion の候補数を縮小
    する。create mode を主対象とし (chat mode は近似重複除去のみ)、安価な
    ヒューリスティック (relevance floor / 近似重複除去 / コードシグナル) で
    大半を裁き、判断に迷う marginal band の prose チャンクだけ補助タスク
    で 1 回判定する。``aux_client=None`` / cap 超過 / error 時はヒューリス
    ティックのみで確定する。
    """

    model_config = ConfigDict(extra="forbid")

    # 機能の有効/無効。スキーマ既定は保守的に OFF (config.yaml.example 側で
    # true を記載し、新規セットアップでは create-mode 精査が有効になる)。
    enabled: bool = False
    # relevance スコア下限 (cosine スケール 0-1)。これ未満を pruning する。
    # unified_search は STM/LTM/カートリッジの cosine 類似度が基準。
    relevance_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    # [floor, floor+marginal_band) を「判断に迷う帯」とし、create mode では
    # この帯の prose チャンクのみ LLM 判定に回す。
    marginal_band: float = Field(default=0.10, ge=0.0, le=0.5)
    # 最低保持件数。pruning がこれを下回ったら上位から補填し生成を枯渇させない。
    min_keep: int = Field(default=3, ge=1)
    # 近似重複除去の token-set Jaccard しきい値 (1.0 で重複除去を無効化)。
    dedup_jaccard: float = Field(default=0.85, ge=0.0, le=1.0)
    # create mode で marginal 帯のコードシグナル判定を有効化する。
    create_code_signal: bool = True
    # marginal band の LLM 救済を有効化する (false で純ヒューリスティック)。
    judge_enabled: bool = True
    # 1 セッションでの LLM 判定発火上限 (0 以下で無制限)。
    max_per_session: int = Field(default=5, ge=0)
    # 1 クエリでの LLM 判定発火上限 (0 以下で無制限)。
    max_per_query: int = Field(default=1, ge=0)


class SelfRagConfig(BaseModel):
    """Self-RAG 設定ルート

    ルールベース Self-RAG の閾値と content gate をネスト構造に集約する。
    """

    model_config = ConfigDict(extra="forbid")

    # チャンク単位の関連度フロア (生スコア = cosine スケール)。
    # フロア以上のチャンクだけを残して添付する (1 件も残らなければ空)。
    # **集計判定 (quality) と独立に常時**掛かる。quality は merged 全体に対する
    # 単一スカラで「セットとして使えるか」しか見ず、top1 が強ければ high になる
    # ため、以前は同じ検索の 2 位以下が無関連でも全通しだった (2026-08-16 実測:
    # [参考情報] の 67% が別セッションの生ログで、対クエリ類似度は較正済み閾値を
    # 1 件も超えていなかった)。キー名は歴史的経緯で low_ が残っている。
    # 0.0 で無効化 = 従来どおり「low はクエリ単位で全件破棄 / それ以外は全通し」。
    # 既定 0.40 の根拠と実測は config.yaml.example / search_pipeline Step 6.5 を参照。
    # Level 1 進化の対象外 (policy_interpreter の search ドメインに登録しない)。
    low_quality_keep_floor: float = Field(default=0.40, ge=0.0, le=1.0)

    # 「そのターンの最良証拠 (top1 の生スコア)」に対する相対の棒。
    # low_quality_keep_floor が「ノイズより上か」を見るのに対し、こちらは
    # 「このクエリで取れた最良証拠と比べられるか」を見る。較正が効いていると
    # 絶対の棒は background_p95 になり、構造上ノイズの 5% が通る。実測
    # (2026-08-19、chat 56 ターン / 採用 183 チャンク、background_p95=0.302 /
    # match_top1_p25=0.475): 採用スコアの 75% が「真の一致の下位 25%」より下で、
    # 他人のペルソナを含む挨拶文が 0.32〜0.40 で通っていた。
    # 絶対値を 0.40 へ上げると 15/42 ターンが空になるのに対し、相対 0.75 は
    # 採用を 73% に絞りつつ空になるターンが 0 (top1 は定義上必ず越えるため)。
    # 0.0 で無効化。Level 1 進化の対象外 (low_quality_keep_floor と同じ理由)。
    relative_keep_ratio: float = Field(default=0.75, ge=0.0, le=1.0)

    # 品質 3 閾値 (relevance / support / confidence) の決め方。
    # ``auto``  : 実ストアのスコア分布から較正した値を使う (埋め込みモデル指紋で
    #             キャッシュ)。較正が無ければ config の静的値へ縮退する。
    # ``manual``: config の静的値をそのまま使う (較正を無視)。
    # 静的既定 0.65/0.50/0.80 は Qwen3-Embedding の分布前提の値で、他モデルでは
    # 到達不能になりゲートが閉じたままになる (実測 LFM2.5-Embedding-350M で
    # 記憶採用 0 件)。既定を ``auto`` にしてモデル差を吸収する。
    threshold_mode: Literal["auto", "manual"] = "auto"
    content_gate: SelfRagContentGateConfig = Field(
        default_factory=SelfRagContentGateConfig,
    )


#: ``shadow`` (並べ替えを記録だけ) は採用しない (c_16 §7.2.1、2026-09-29)。書くと検証エラー。
RerankMode = Literal["off", "on"]


def normalize_rerank_mode(value: object) -> str:
    """``rag.rerank.mode`` の生の値を正規化する (起動スクリプト・backend・スキーマの SSOT)。

    YAML 1.1 で引用符無しの ``off`` / ``on`` は真偽値になるので文字列へ戻す。明示の ``null`` /
    ``false`` は ``off`` (キー自体が無いときの既定 ``on`` は :func:`rerank_mode_of` が決める)。
    """
    if value is False or value is None:
        return "off"
    if value is True:
        return "on"
    return str(value)


def rerank_mode_of(cfg: dict) -> str:
    """素の config dict から ``rag.rerank.mode`` を読む (検証前の dict でも使える)。

    キーが無ければ既定の ``on`` (:class:`RerankConfig` と同じ)。モデル
    (``model_paths.rerank_model``) が無ければ ``on`` でも何も起動しない。
    """
    rerank = (cfg.get("rag") or {}).get("rerank") or {}
    return normalize_rerank_mode(rerank.get("mode", "on"))


class RerankConfig(BaseModel):
    """再順位段 (リランカー) の設定 (c_16 §7.2.1)。

    ``mode: on`` で検索経路 (``unified_search`` の Step 6.8) が floor を通った corpus と
    episodic の候補をストアごとに並べ替える (履歴検索ツールと sleep-time の疑似クエリ採点も同じクライアントを使う)。``off`` なら rerank 用 llama-server は起動せず、並べ替えもしない。
    既定は ``on`` だが、``model_paths.rerank_model`` が未設定かファイルが無ければ起動せず
    並べ替えもしない (cosine 順のまま)。モデルは利用者が ``models/`` に置く (自動ダウンロードは無い)。
    自己テストは PC の指紋が変わったときだけ起動スクリプトが走らせる。
    """

    model_config = ConfigDict(extra="forbid")

    mode: RerankMode = Field(
        default="on",
        description=(
            "off: 起動しない・並べ替えない / on (既定): model_paths.rerank_model のファイルがあれば"
            "起動して corpus と episodic の候補を並べ替える (無ければ起動しない)"
        ),
    )
    port: int = Field(default=8083, ge=1024, le=65535, description="rerank 用 llama-server のポート")
    gpu_layers: Literal["auto"] | int = Field(
        default="auto",
        description="auto: GPU の空きがモデル + 計算バッファの見積りを満たせば GPU、でなければ CPU。整数なら -ngl にそのまま渡す (0 = CPU)",
    )
    deadline_ms: int = Field(default=1000, ge=50, description="1 回の rerank 呼出の締切 (ミリ秒)。超えたら並べ替えを諦める")
    max_candidates: int = Field(default=12, ge=1, le=200, description="1 回に並べ替える候補数の上限 (実際に送る量はこの上限と、自己テストの速度から決まる近似トークンの予算の早い方。12 は締切超過を p90 で抑える値、c_16 §7.2.1)")
    min_candidates: int = Field(
        default=3, ge=1,
        description="締切に収まる候補数がこれ未満なら自己テストで無効にする (too_slow)",
    )
    threads: int = Field(default=0, ge=0, description=(
        "-t (0 = 自動: GPU 配置なら 2、CPU 配置なら環境調整 (c_16 §7.2.3 の項目 threads) が物理コア数から配分した値)"
    ))

    @field_validator("mode", mode="before")
    @classmethod
    def coerce_yaml_bool_mode(cls, value: object) -> object:
        """YAML 1.1 で引用符無しの ``off`` / ``on`` は真偽値になるので文字列へ戻す"""
        if isinstance(value, bool):
            return normalize_rerank_mode(value)
        return value

    @model_validator(mode="after")
    def validate_candidates(self) -> "RerankConfig":
        """gpu_layers の範囲と、候補数の下限 <= 上限を検証"""
        if isinstance(self.gpu_layers, int) and self.gpu_layers < 0:
            raise ValueError(f"rag.rerank.gpu_layers must be 'auto' or >= 0 (got {self.gpu_layers})")
        if self.min_candidates > self.max_candidates:
            raise ValueError(
                f"rag.rerank.min_candidates ({self.min_candidates}) は "
                f"max_candidates ({self.max_candidates}) 以下である必要があります"
            )
        return self


class QueryDecomposeConfig(BaseModel):
    """決定論の問い分解 (D2) の設定 (f_01 §8.1 の 7.9)。

    ``shadow`` は構文で割れる問いを分解の項ごとに検索し、元の問いの採用で空いた枠に
    入る id を ``decision.jsonl`` に記録するだけで、応答には何も足さない。
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "shadow"] = Field(
        default="off",
        description=(
            "off (既定): 分解しない / shadow: 分解の項ごとに埋め込み 1 回 + 統合検索 1 回を"
            "応答と並走させ、空いた枠に入る id を記録だけする"
        ),
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
    # 転置索引の上位に予約する corpus の席 (f_01 §8.1 の 4.3)。cosine 1 本の順位
    # では固有語で当てた行が top-k に残らないため、棒を越える語彙 top-N を
    # 疑似クエリ由来と同じ列 (先頭側) に置く。0 で無効。実測 (golden 111 件)
    # 席 1 で recall@5 0.640 → 0.703、席 2 は 0.649 に下がる。
    lexical_seats: int = Field(default=1, ge=0, le=2)
    # 採用した corpus チャンクに随伴させる直前チャンク (同文書・同大節) の末尾文字数
    # (f_01 §8.1 の 7.65)。0 で無効。実測 (v3、golden): 0.830 → 0.859 (+31% 文字)。
    previous_context_chars: int = Field(default=300, ge=0, le=2000)
    # --- 順位付け (c_16 §7.2) ---
    #
    # 順位式は 3 ストア共通の 1 本 (``cos × freshness × confidence ×
    # store_prior``) で、係数は ``memory.evidence.ranking.store_prior``。
    # 廃止したキー (c_16 §8): ``score_normalization`` (層内正規化) /
    # ``rrf_k`` / ``fusion_method`` / ``hybrid_search`` / ``bm25_weight`` /
    # ``vector_weight`` / ``bm25_k1`` / ``bm25_b`` / ``bm25_delta`` /
    # ``bm25_use_trigrams`` / ``bm25_split_ascii`` / ``bm25_stopword_bigrams``。
    # BM25 (``rank-bm25``) は ``backend/free/rag/evidence/lexical_index.py`` の
    # numpy CSR 転置索引に置き換わり、走査上限は
    # ``memory.evidence.lexical`` が持つ。転置索引は候補生成器であって
    # スコアを持ち込まないので (c_16 §6.3)、重み付け融合のキーは意味を失った。
    # --- 疑似クエリ索引 (f_01 §6) ---
    pseudo_query: PseudoQueryConfig = Field(default_factory=PseudoQueryConfig)
    # --- 問い分解の影 (f_01 §8.1 の 7.9) ---
    query_decompose: QueryDecomposeConfig = Field(default_factory=QueryDecomposeConfig)
    # --- code グラフ (c_16 §4.4) ---
    project_map: ProjectMapConfig = Field(default_factory=ProjectMapConfig)

    # --- ベクトル量子化 ---
    quantization: str = Field(default="int8", pattern=r"^(none|int8)$")
    # int8 粗検索後に float32 で rescore する候補数。チャット応答経路の LTM /
    # カートリッジ検索 (``search_pipeline`` → ``VectorStore.search``) へ渡る。
    # 0 でストア側の既定 ``max(50, top_k*3)``。
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
    # --- corpus パッケージ install の入口検査 (c_16 §4.3) ---
    packages: PackagesConfig = Field(default_factory=PackagesConfig)
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
    self_rag: SelfRagConfig = Field(default_factory=SelfRagConfig)
    # --- 再順位段 (リランカー、c_16 §7.2.1) ---
    rerank: RerankConfig = Field(default_factory=RerankConfig)

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
        return self


_DEFAULT_INSTRUCTIONS: dict[str, str] = {
    "chat": "Given a user question, retrieve relevant passages that answer the query",
    "create": "Given a code search query, retrieve relevant code snippets",
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
    # 1 要求の HTTP timeout (秒)。``auto`` / ``null`` は環境調整の値 (c_16 §7.2.3 の項目
    # ``embed_params``: 配置の判別の ms/トークンから。下限 30、遅い CPU では延びる)。
    timeout: Annotated[float, Field(ge=0.1)] | Literal["auto"] | None = None
    # チャット応答パスの **単一クエリ** 埋め込みだけに掛かるデッドライン (秒)。
    # 0 で無効 (``timeout`` のみ)。バッチ / ドキュメント側には掛からない
    # (sleep-time のバッチは本来長い)。``auto`` / ``null`` は環境調整の値 (下限 3.0、
    # 判別の p50 から見積る p95 × 係数)。
    #
    # 実測 (2026-08-18、chat 136 ターン / 実往復 262 件): 中央値 216.7ms /
    # p90 1292.1ms / p95 3122ms / p99 5977ms / 最大 8057ms。分布は二峰で、
    # 1.0s 超が 16.0% ある一方 2.0s 超は 6.1% しかない。3.0s 超の 5.7% は
    # 「埋め込みサーバが詰まっている」区間で、そのままターンの TTFT に前置き
    # される (TTFT 中央値 17s に対し最大 8s = +47%)。
    # 既定 3.0 は ``rag.cartridge_search_timeout_ms`` (3000) と同じ水準に揃え、
    # p95 までは通しつつ病的な裾だけを切る。
    query_timeout: Annotated[float, Field(ge=0.0)] | Literal["auto"] | None = None
    max_length: int = Field(default=8192, ge=1)
    # 文脈長 (llama-server ``-c``)。埋め込みは max_length トークンまでの入力を
    # 扱うため、必ず max_length 以上にすること (下回ると長い入力で 500)。モデル
    # 既定 n_ctx (例: Qwen3-Embedding-0.6B=32768) は過剰で KV を浪費するため、
    # max_length に合わせて縮小する (実測 0.6B で active WorkingSet 7.5GB→4.3GB、
    # 速度・次元は同等)。
    context_size: int = Field(default=8192, ge=1)
    # 共通
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    # Qwen3-Embedding 系の instruction-aware プレフィックス
    # ``is_query=True`` のときに ``query_template`` で整形する。``mode`` は
    # ``chat`` / ``create`` のいずれか。ドキュメント側 (``is_query=False``)
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
    # None の場合は CPU フォールバック (``-ngl 0``) が既定となる (既存の config の値のまま)。
    # 整数はそのまま ``-ngl`` へ。``auto`` は PC の指紋が変わったときだけ一時ポートで CPU と
    # GPU を測って決める (c_16 §7.2.2、雛形の既定)。ベースモデルの ``llama.gpu_layers``
    # には追従しない。
    gpu_layers: Literal["auto"] | int | None = Field(default=None)

    @field_validator("gpu_layers")
    @classmethod
    def _gpu_layers_range(cls, v: "Literal['auto'] | int | None") -> "Literal['auto'] | int | None":
        """整数は -1 以上 (``auto`` / ``null`` はそのまま)。"""
        if isinstance(v, int) and v < -1:
            raise ValueError(f"embedding.gpu_layers must be 'auto', null or >= -1 (got {v})")
        return v

    # 物理バッチサイズ。max_length トークン分の単一入力を 1 回で処理できる値に揃える。
    # llama-server デフォルト 512 のままだと、長い STM ノート (例: 933 tok) の埋め込み
    # リクエストが 500 エラーになり EvorefMem sleep-time update が連鎖失敗する。
    # ``auto`` / ``null`` は環境調整の値 (``ubatch >= max_length`` を保証、c_16 §7.2.3)。
    # 明示値は保証せず、破っていれば起動時に WARNING だけ出す。
    batch_size: Annotated[int, Field(ge=1)] | Literal["auto"] | None = None
    ubatch_size: Annotated[int, Field(ge=1)] | Literal["auto"] | None = None
    # スレッド数 (llama-server ``-t``)。0 = 環境調整 (c_16 §7.2.3 の項目 threads) が物理コア数を
    # base / 埋め込み / リランカーへ配分した値 (GPU 配置は 2)。正の整数は明示 (そのまま使う)。
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
    # 起動時に embedder dim と既存ベクトルの dim 不整合を検出した際、
    # 自動で ``run_reindex`` を実行するか。``false`` (既定) では
    # ``state.embedding_dim_mismatch`` を立てて WARNING ログのみ。
    # ``true`` で運用者が ``evoref reindex`` を手動実行する手間を省く
    # (モデル切替直後の RAG ダウンタイムを縮める)。
    auto_reindex_on_mismatch: bool = False


