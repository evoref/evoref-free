"""設定管理とパス解決"""

import yaml
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from backend.io.atomic import atomic_write_text
from backend.log_config import get_logger

logger = get_logger("config")

_config: dict | None = None
_path_resolver: "PathResolver | None" = None

#: base_model 未指定時のフォールバック (歴史的既定)。
_DEFAULT_BASE_MODEL = "models/gemma-4-12b-it-qat-q4_0.gguf"


def mode_base_model_raw(
    model_paths: dict, mode: str, *, default: str = _DEFAULT_BASE_MODEL,
) -> str:
    """指定モードで実際にロードされる base モデルの生パス文字列を返す。

    chat は常に ``model_paths.base_model``。create は ``create_model`` 指定が
    あればそれ、無い/空なら ``base_model`` へフォールバックする。

    ``get_mode_generation_params`` (起動 / モード切替が使う解決) と
    ``PathResolver`` (LoRA パーティション根の決定) の**単一情報源**。両者が
    別々に同じ規則を書くと、片方だけ変えたときに「表示・保存先と実際に
    ロードされるモデル」がズレる (2026-07-26 に同種のズレを 2 件修正済み)。

    ``default`` は ``base_model`` 未宣言時の戻り値。起動側は歴史的既定へ倒すが、
    パーティション根の決定では ``""`` を渡して「未宣言」を区別する
    (宣言されていないモデル名でディレクトリを作らないため)。
    """
    base_model = model_paths.get("base_model") or default
    if mode == "create":
        return model_paths.get("create_model") or base_model
    return base_model


class PathResolver:
    """モデルパスとデータパスを統一的に解決する (c_05 §0.2)。

    モデルは ``model_paths`` (インストール根基準)。データ・状態は **データ根
    ``data_root`` からだけ** 導出する (:data:`LAYOUT`)。``local_paths`` の個別
    キーは撤去済みで、利用者が変えられるのは ``outputs_dir`` だけ。
    """

    MODEL_DEFAULTS = {
        "base_model": "models/gemma-4-12b-it-qat-q4_0.gguf",
    }
    #: データ根からの相対パス (c_03 §10.1)。末尾 ``/`` はディレクトリ。
    #: ``store/learning/shared/`` はモデル非依存の学習データだけの置き場
    #: (c_05 §0.5.12)。モデル依存の学習データは :data:`_LEARNING_SUBPATH`。
    LAYOUT = {
        "liveness_file": "logs/liveness.json",
        # ディレクトリ指定の無い生成物の既定の書込み先 (CWD へ書かないため)。
        # 利用者が ``local_paths.outputs_dir`` で変えられる唯一のパス。
        "outputs_dir": "outputs/",
        "model_state_file": "store/model_state.json",
        # model_key の計算結果 (derived、c_05 §0.5.7)。
        "model_registry_file": "store/model_registry.json",
        "model_quality_file": "store/model_quality.json",
        # EvorefMem ローカル状態
        "local_state_file": "store/state.json",
        "memory_dir": "store/memory/",
        # AgentTracer の MDP トレース常設ストア (エピソード記憶の入力)。
        "agent_trace_dir": "store/agent_trace/",
        "history_dir": "store/history/",
        "learned_patterns_file": "store/learning/shared/learned_patterns.json",
        "themes_dir": "themes/",
        # EvorefMem トリガ辞書 (pin / fact / classify) のユーザー上書き先。
        # 同梱 default は ``backend/free/memory/_defaults/triggers/``。
        "triggers_dir": "store/overrides/triggers/",
        # staged クリエイトパイプラインの一時ワークスペース。
        "create_workspace_dir": "store/create/",
        # base 学習データの (model_key × mode) パーティションルート。
        "learning_dir": "store/learning/",
        # develop=evolve の LogIngestor 進捗 (読み込みオフセット)。
        "log_ingestor_file": "store/state/log_ingestor.json",
        # ── ここから G1 で足した置き場 (c_03 §10.1) ──
        "store_dir": "store/",
        # corpus ストア (文書由来チャンク)。``resolve_corpus_dir()`` の実体。
        "corpus_dir": "store/corpus/",
        "logs_dir": "logs/",
        "tmp_dir": "tmp/",
        "run_dir": "run/",
        "cache_dir": "cache/",
        "embedding_cache_dir": "cache/embeddings/",
        # リランカー自己テストの結果 (volatile、PC 固有の測定値。c_16 §7.2.1)。
        # 書き手は起動スクリプト、backend は読むだけ。store/ の外なので readonly でも書ける。
        "rerank_selftest_file": "cache/rerank_selftest.json",
        # 埋め込みサーバの配置 (GPU / CPU) の判別結果 (PC 固有の測定値。c_16 §7.2.2)。
        # 書き手は起動スクリプト、backend と再起動の経路は読むだけ。store/ の外。
        "embed_placement_file": "cache/embed_placement.json",
        # 環境調整 (auto-tune) の結果と環境移行の確認状態 (PC 固有、c_16 §7.2.3)。
        # 書き手は起動スクリプト / evoref tune / 管理画面の実行、backend は読むだけ。store/ の外。
        "auto_tune_file": "cache/auto_tune.json",
        # 生成速度 (prefill / decode tok/s) の実測 (model_key 単位、PC 固有、c_16 §7.2.3)。
        # 書き手は backend (応答の timings を間引いて保存)。store/ の外。
        "tps_calibration_file": "cache/tps_calibration.json",
        # 疑似クエリの品質検査の閾値 (リランカーの model_key 単位、c_17 §3.15)。
        # 書き手は較正、backend は起動時に読むだけ。store/ の外。
        "pseudo_query_gate_file": "cache/pseudo_query_gate.json",
        "profiles_dir": "profiles/",
        # 利用者のファイルを上書きする前の退避 (f_11 §5.6)。store/ の外 (readonly でも書ける)。
        "backup_overwrite_dir": "bk/overwrite/",
        "subject_dictionary_file": "store/memory/semantic/subject_dictionary.json",
        # PolicyInterpreter のポリシー (進化対象外のドメイン・モデル非依存)。
        # 進化対象 (agent / long_form) は ``evolved_policies_dir`` (パーティション)。
        "policies_dir": "store/learning/shared/policies/",
        # CLI の ``/save`` ``/load`` が書く手動保存セッション。
        "cli_sessions_dir": "store/cli_sessions/",
        # Pro だけが書くデータの根 (c_05 §0.4.2 所有表)。Free は開かない・作らない・
        # 消さない (:data:`PRO_LAYOUT_KEYS` は ``ensure_local_dirs`` の対象外)。
        "pro_dir": "store/pro/",
    }
    #: Pro の置き場のキー。Free のプロセスはディレクトリを作らない。
    PRO_LAYOUT_KEYS = frozenset({"pro_dir"})
    #: ``local_paths`` で上書きできるキー (c_05 §0.2: ``outputs_dir`` だけ)。
    USER_OVERRIDABLE = frozenset({"outputs_dir"})

    # resolve_learning で active モデルの model_key パーティション配下へ rebase する
    # base 学習キーと、``learning_dir/<model_key>/`` からの相対サブパス。ここに無い
    # キー (共有 / embed) は resolve_local へ素通しする (memory_dir を巻き込まない
    # ための allow-list)。flat の置き場は持たない (c_05 §0.5.12)。
    _LEARNING_SUBPATH = {
        "experience_file": "experience.jsonl",
        "prompts_dir": "prompts",
        # 補助タスクのプロンプト。実行するのはベースモデルなので base 軸で
        # 分離する (モデルを替えたら既定から作り直す)。
        "aux_prompts_dir": "aux_prompts",
        # 補助タスクの timeout 較正 (derived、keep_on_reset)。
        "aux_calibration_file": "aux_calibration.json",
        # PolicyParamEvolver の進化対象ドメイン (agent / long_form) のポリシー。
        "evolved_policies_dir": "policies",
        "generation_deltas_file": "generation_deltas.json",
    }

    # Pro の学習データ (c_05 §0.4.2 所有表)。``resolve_pro_learning`` が
    # ``<pro_dir>/learning/<model_key>/`` の下に解決する。eval_core 以外は
    # アダプタ系で、``<model_key>/<mode>/`` の mode パーティションに置く。
    _PRO_LEARNING_SUBPATH = {
        # Level 1 採用ゲート / Level 2 目的関数の合格基準。auto ケースは
        # そのモデルの訂正から作られるので base 軸で分離する (書き手は Pro)。
        "eval_core_file": "eval_core.json",
        "lora_adapter": "adapter.gguf",
        "lora_versions_dir": "lora_versions",
        "lora_spsa_checkpoint": "lora_spsa_checkpoint.json",
        "control_vector_adapter": "control_vector.gguf",
        "control_vector_versions_dir": "control_vector_versions",
        "cvector_work_dir": "cvector",
    }
    #: ``_PRO_LEARNING_SUBPATH`` のうち mode で分けないキー。
    _PRO_MODELESS_KEYS = frozenset({"eval_core_file"})

    def __init__(self, config: dict, project_root: Path, data_root: Path | None = None):
        from backend.data_root import resolve_data_root

        self.root = project_root
        #: データ根 (c_05 §0.2)。未指定なら ``--data-root`` を反映した環境変数
        #: ``EVOREF_DATA_ROOT`` → ``<project_root>/userdata``。
        self.data_root: Path = (
            Path(data_root) if data_root is not None else resolve_data_root(root=project_root)
        )
        self.models = config.get("model_paths", {})
        self.local = config.get("local_paths", {}) or {}
        # base 学習パーティションの active モデル (model_key と GGUF の stem)。
        # 未束縛のまま resolve_learning が呼ばれたら ``model_paths.base_model``
        # から遅延で導出する (CLI など束縛しないプロセス向け)。
        self._active_key: str | None = None
        self._active_stem: str | None = None
        # embed_instruction 系データの (embedding モデル) パーティション。
        # base 学習パーティションとは独立した軸。未束縛なら ``embed_model`` から導出。
        self._active_embed_key: str | None = None
        # Level 2 アダプタの (mode) パーティション state。レガシー "model" では
        # resolve_pro_learning は mode 引数を無視し、chat/create が "chat" の
        # 1 つを共有する。AppState.current_mode の初期値と揃え、active_mode の
        # 既定は "chat"。
        self._active_mode: str = "chat"
        # llama-server が実際に載せているモデルのパス (``/props`` の ``model_path``) を
        # 返す関数。学習データの置き場を「その生成を返したモデル」で決める
        # (:meth:`generating_model_key`、f_04 §1.2.0)。未登録なら宣言へ倒す。
        self._served_model_source: Callable[[], str] | None = None
        #: ``/props`` の生の文字列 → model_key (``None`` = 宣言へ倒す)。束ね直しで捨てる。
        self._served_key_cache: dict[str, str | None] = {}
        self._adapter_partition_mode: str = str(
            (config.get("learning", {}) or {}).get(
                "level2_adapter_partition", "model_mode",
            ),
        )

    def resolve_model(self, key: str) -> Path:
        """モデルパス解決

        config の ``model_paths[key]`` を優先し、無ければ ``MODEL_DEFAULTS[key]``
        を使う。``dict.get(key, MODEL_DEFAULTS[key])`` は default 引数を先に評価
        するため、``MODEL_DEFAULTS`` に無いキー (embed_model /
        create_model 等) を config 側に持っていても KeyError で落ちていた。
        """
        raw = self.models.get(key) or self.MODEL_DEFAULTS.get(key)
        if raw is None:
            raise KeyError(f"unknown model_paths key: {key!r}")
        return self._to_absolute(raw)

    def resolve_local(self, key: str) -> Path:
        """データパス解決 (読み書きリソース)。全て ``data_root`` の下。

        ``store/`` ``cache/`` は世代フォルダ ``g<N>/`` の下 (:func:`backend.data_root.data_path`)。
        ``outputs_dir`` だけは ``local_paths.outputs_dir`` で変えられる
        (相対ならデータ根基準)。
        """
        if key in self.USER_OVERRIDABLE and self.local.get(key):
            override = Path(str(self.local[key]))
            return override if override.is_absolute() else self.data_root / override
        return self.layout_path(self.data_root, key)

    @classmethod
    def layout_path(cls, data_root: Path, key: str) -> Path:
        """``LAYOUT[key]`` をデータ根 ``data_root`` の下の実パスにする (利用者の上書きは見ない)。"""
        from backend.data_root import data_path

        return data_path(data_root, cls.LAYOUT[key])

    # ── モデル識別子 (c_05 §0.5.7) ──

    def model_key_for(self, model_path: Path | str) -> str:
        """``model_path`` の ``model_key`` (相対ならインストール根基準)。

        GGUF が読めなければファイル名由来の仮 key (:func:`backend.model_key.model_key_for`)。
        """
        from backend.model_key import model_key_for

        return model_key_for(self._to_absolute(str(model_path)))

    def bind_active_model(self, model_path: Path | str | None) -> str | None:
        """base 学習パーティションを ``model_path`` の ``model_key`` に束ねる。

        起動時は ``model_paths.base_model``、ランタイムのモデル切替では新しい
        モデルで呼ぶ。``None`` / 空なら束縛を外す (次の resolve_learning が
        ``model_paths.base_model`` から導出し直す)。

        Returns:
            束ねた ``model_key`` (外したときは ``None``)。
        """
        self._served_key_cache.clear()
        name = Path(str(model_path or "")).name
        if not name:
            self._active_key = None
            self._active_stem = None
            return None
        self._active_key = self.model_key_for(model_path)
        self._active_stem = Path(name).stem
        return self._active_key

    def set_active_model_key(self, model_key: str, *, stem: str | None = None) -> None:
        """``model_key`` を直接束ねる (計算済みの key を持っている呼出元・テスト用)。"""
        self._served_key_cache.clear()
        self._active_key = model_key
        self._active_stem = stem

    @property
    def active_model_key(self) -> str:
        """base 学習パーティションの ``model_key`` (未束縛なら base_model から導出)。"""
        if self._active_key is None:
            self.bind_active_model(self.models.get("base_model") or self.MODEL_DEFAULTS["base_model"])
        assert self._active_key is not None
        return self._active_key

    @property
    def active_model_stem(self) -> str | None:
        """束ねたモデルの GGUF ファイル名の stem (未束縛なら ``None``)。

        ランタイム切替の検知 (``LearningScheduler._base_model_changed``) が
        ``model_state`` のファイル名と突き合わせる。パーティションのディレクトリ名
        は :attr:`active_model_key`。
        """
        return self._active_stem

    def bind_active_embedding_model(self, model_path: Path | str | None) -> str | None:
        """embed_instruction のパーティションを埋め込みモデルの ``model_key`` に束ねる。"""
        name = Path(str(model_path or "")).name
        self._active_embed_key = self.model_key_for(model_path) if name else None
        return self._active_embed_key

    @property
    def active_embedding_model_key(self) -> str:
        """埋め込みモデルの ``model_key`` (未束縛なら ``model_paths.embed_model`` から導出)。

        ``embed_model`` が宣言されていなければ ``embedding.model_name`` の無い
        構成なので、空のファイル名の仮 key に倒す。
        """
        if self._active_embed_key is None:
            raw = self.models.get("embed_model") or ""
            if raw:
                self.bind_active_embedding_model(raw)
            else:
                from backend.model_key import provisional_model_key

                self._active_embed_key = provisional_model_key("")
        assert self._active_embed_key is not None
        return self._active_embed_key

    def resolve_embed_instruction_dir(self) -> Path:
        """embed_instruction 系データの保存先を **埋め込みモデルの model_key 単位**で解決する。

        embed_instruction は埋め込みモデル向けのクエリ指示文であり、base モデル
        切替とは無関係に保持されるべきなので、base 学習パーティションとは別軸の
        ``learning_dir/embed/<embed model_key>/`` に置く (c_05 §0.7.1)。
        """
        return self.resolve_local("learning_dir") / "embed" / self.active_embedding_model_key

    def resolve_aux_prompt_dir(self) -> Path:
        """補助タスクプロンプトの保存先を **ベースモデル単位**で解決する。

        補助プロンプト (rag_necessity / rag_quality / tool_call / note_evolve)
        は進化の対象で、進化した文面は **そのモデルの癖に合わせて最適化される**。
        判定を実行するのはベースモデルなので、base 学習パーティション
        (``resolve_learning``) と同じ軸に置き、モデルを差し替えたら既定から
        作り直す。
        """
        return self.resolve_learning("aux_prompts_dir")

    def resolve_corpus_dir(self) -> Path:
        """corpus ストア (文書由来チャンク) の置き場を解決する。

        G1 のレイアウト (c_03 §10.1) では ``<data_root>/g1/store/corpus/`` で、
        パッケージは ``packages/<id>/<version>/`` に置く。
        """
        return self.resolve_local("corpus_dir")

    def resolve_outputs_dir(self) -> Path:
        """ディレクトリ指定の無い生成物の既定の書込み先を解決する。

        「compose.yaml に保存して」のようにファイル名だけを指定された書込みは、
        素で ``write_file`` へ渡すとプロセスの CWD (= リポジトリ直下) に着地
        する (2026-09-08 監査 F-05: リポジトリ直下に ``compose.yaml`` が
        作られた)。宛先の錨が無い相対パスはすべてここへ寄せる。明示パス
        (絶対パス / ``./`` ``../`` 始まり) はユーザーの指定なのでそのまま。
        """
        return self.resolve_local("outputs_dir")

    @property
    def active_mode(self) -> str:
        """Level 2 アダプタパーティションの active モード (``"chat"``/``"create"``)。"""
        return self._active_mode

    def set_active_mode(self, mode: str | None) -> None:
        """モード切替時に呼ぶ。未指定/不明値は安全側で ``"chat"`` に丸める。"""
        self._active_mode = mode if mode in ("chat", "create") else "chat"

    def resolve_learning(self, key: str) -> Path:
        """base 学習データのパスを **active** モデルの model_key パーティション配下で解決する。

        ``_LEARNING_SUBPATH`` の base 学習キーのみ ``learning_dir/<model_key>/...``
        配下へ rebase する。非対象キー (共有) は ``resolve_local`` へ素通しする。
        active モデルが未束縛なら ``model_paths.base_model`` から導出する
        (:attr:`active_model_key`)。``learning_dir`` は ``resolve_local`` 経由で
        データ根の下に解決する (``--isolate-data`` は別のデータ根になる)。
        Pro の学習データは :meth:`resolve_pro_learning`。
        """
        if key not in self._LEARNING_SUBPATH:
            return self.resolve_local(key)
        return self.learning_path_for(key, self.active_model_key)

    def learning_path_for(self, key: str, model_key: str) -> Path:
        """**指定** ``model_key`` のパーティション配下で base 学習パスを解決する。

        active モデルに依存せず任意モデルのパーティションパスを得る。
        ``_LEARNING_SUBPATH`` 非対象キーは ``resolve_local`` へ素通しする。
        """
        if key not in self._LEARNING_SUBPATH:
            return self.resolve_local(key)
        return self.resolve_local("learning_dir") / model_key / self._LEARNING_SUBPATH[key]

    def resolve_pro_learning(self, key: str, mode: str | None = None) -> Path:
        """Pro の学習データのパスを ``<pro_dir>/learning/<model_key>/`` の下に解決する。

        ``eval_core_file`` は active モデルの ``<model_key>/`` 直下。アダプタ系
        (LoRA / control vector / SPSA checkpoint / cvector 作業場) は
        ``<model_key>/<mode>/`` で、``<model_key>`` は **そのモードが実際に
        ロードするモデル** (アダプタは特定モデルの重みへの差分なので、「どの
        モデル向けか」が保存先の第一キー)。create が別モデル (``create_model``)
        を使う構成で chat の ``base_model`` を差し替えても、create のアダプタは
        create のモデルの下に残る。

        ``mode`` 省略時は :attr:`active_mode`。レガシー
        ``learning.level2_adapter_partition == "model"`` では mode に依らず
        active モデルの ``chat`` を chat/create で共有する。
        """
        sub = self._PRO_LEARNING_SUBPATH[key]
        if key in self._PRO_MODELESS_KEYS:
            return self._pro_learning_root(self.active_model_key) / sub
        if self._adapter_partition_mode != "model_mode":
            return self._pro_learning_root(self.active_model_key) / "chat" / sub
        effective_mode = mode if mode in ("chat", "create") else self._active_mode
        return self._pro_learning_root(self._mode_model_key(effective_mode)) / effective_mode / sub

    def _pro_learning_root(self, model_key: str) -> Path:
        return self.resolve_local("pro_dir") / "learning" / model_key

    def _mode_model_key(self, mode: str) -> str:
        """``mode`` が実際にロードするモデルの ``model_key`` (宣言が無ければ active)。"""
        active = self.active_model_key
        raw = mode_base_model_raw(self.models, mode, default="")
        if not raw or Path(raw).stem == self._active_stem:
            return active
        return self.model_key_for(raw)

    # ── 生成したモデルのパーティション (f_04 §1.2.0) ──

    def declared_model_key(self, mode: str) -> str:
        """``mode`` のターンを生成する **はずの** モデルの ``model_key``。

        chat は束ねた active (``/api/model/reload`` の後も config の base_model では
        なく束ねたモデル)。create は ``create_model`` を宣言していればそのモデル、
        していなければ active。
        """
        if mode == "create" and self.models.get("create_model"):
            return self._mode_model_key("create")
        return self.active_model_key

    def set_served_model_source(self, source: Callable[[], str] | None) -> None:
        """llama-server が実際に載せているモデルのパスを返す関数を登録する (``None`` で外す)。

        起動時の配線 (``_learning_rebind.wire_generating_partitions``) が
        ``/props`` の ``model_path`` を読む関数を渡す。
        """
        self._served_model_source = source

    def served_model_path(self) -> str:
        """llama-server が実際に載せているモデルのパス (登録が無い・取れなければ空文字)。"""
        source = self._served_model_source
        if source is None:
            return ""
        try:
            raw = source()
        except Exception as exc:  # noqa: BLE001 — 置き場の解決は宣言へ倒して続ける
            logger.debug("served model unavailable: %s", exc)
            return ""
        return raw if isinstance(raw, str) else ""

    def generating_model_key(
        self, mode: str | None = None, *, served_model: str | Path | None = None,
    ) -> str:
        """その生成を返したモデルの ``model_key`` (学習データの置き場の鍵)。

        鍵は設定上の対応 (mode → model) ではなく、実際に載っているモデルから取る:
        ``served_model`` (明示) → 登録された :meth:`set_served_model_source` →
        ``mode`` が宣言するモデル (:meth:`declared_model_key`) → active の順。
        ``/props`` の名前がファイルでも宣言済みでもない (alias だけ) ときも宣言へ倒す。
        モード切替でパーティション全体を束ね直さず、書込み・読込みのたびに
        これで置き場を決める。
        """
        raw = str(served_model or "") or self.served_model_path()
        if raw:
            key = self._served_model_key(raw)
            if key is not None:
                return key
        if mode in ("chat", "create"):
            return self.declared_model_key(mode)
        return self.active_model_key

    def _served_model_key(self, raw: str) -> str | None:
        """``/props`` の ``model_path`` (または ``model_id``) を ``model_key`` にする。

        読めるファイルならその重みから。読めない (名前だけ / 別の場所) ときは宣言した
        モデル (base / create) の中から同じファイル名を探し、次に base_model の
        ディレクトリの同名ファイル (モデル移行はその中で行う)。どれでもない名前
        (alias だけ) は ``None`` — 仮 key の孤立パーティションを作らず、呼出側が
        そのモードの宣言へ倒す。結果は生の文字列ごとにメモする (毎ターン 10 回前後
        呼ばれ、models/ が NAS だと stat 1 回が重い)。束ね直しでメモを捨てる。
        """
        cached = self._served_key_cache.get(raw)
        if cached is not None or raw in self._served_key_cache:
            return cached
        key = self._served_model_key_uncached(raw)
        self._served_key_cache[raw] = key
        return key

    def _served_model_key_uncached(self, raw: str) -> str | None:
        path = self._to_absolute(raw)
        if path.is_file():
            return self.model_key_for(path)
        for declared in (self.models.get("base_model"), self.models.get("create_model")):
            if declared and Path(str(declared)).name == path.name:
                if Path(str(declared)).stem == self._active_stem:
                    return self.active_model_key
                return self.model_key_for(declared)
        base_model = self.models.get("base_model")
        if base_model:
            sibling = self._to_absolute(str(base_model)).parent / path.name
            if sibling.is_file():
                return self.model_key_for(sibling)
        return None

    def resolve_pro_created_dir(self) -> Path:
        """Pro が作る ``.evocart`` の置き場 (``<pro_dir>/created/``)。"""
        return self.resolve_local("pro_dir") / "created"

    def _to_absolute(self, raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else self.root / path

    def ensure_local_dirs(self) -> None:
        """データ根のディレクトリを作成する (ファイルのキーは親だけ)。"""
        for key, rel in self.LAYOUT.items():
            if key in self.PRO_LAYOUT_KEYS:
                continue
            path = self.resolve_local(key)
            if rel.endswith("/"):
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict, override: dict) -> dict:
    """ネストされた辞書を再帰的にマージ（override が優先）"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None, project_root: Path | None = None) -> dict:
    """エディション別設定ファイルをマージ読込みし、グローバルに保持する

    config.yaml（共通）を読込み、backend/pro/ 存在時は config.pro.yaml を
    deep merge する。
    """
    global _config, _path_resolver

    if project_root is None:
        project_root = Path(__file__).parent.parent

    if path is None:
        path = project_root / "config.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    # Pro 設定: backend/pro/ が存在する場合のみ読込み
    pro_cfg_path = project_root / "config.pro.yaml"
    if (project_root / "backend" / "pro").is_dir() and pro_cfg_path.exists():
        with open(pro_cfg_path, encoding="utf-8") as f:
            pro_cfg = yaml.safe_load(f)
        if pro_cfg:
            _config = _deep_merge(_config, pro_cfg)
            logger.info("Merged config.pro.yaml")
    elif pro_cfg_path.exists():
        logger.warning(
            "config.pro.yaml found but backend/pro/ does not exist. "
            "Pro settings will be ignored. Install Pro edition or remove config.pro.yaml."
        )

    # Pydantic スキーマでバリデーション + デフォルト値補完
    from backend.schemas import validate_config

    try:
        _config = validate_config(_config)
    except ValidationError as e:
        logger.error("Config validation failed:\n%s", e)
        raise

    _path_resolver = PathResolver(_config, project_root)
    return _config


def get_config() -> dict:
    """現在の設定を取得（未ロード時はエラー）"""
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _config


def get_path_resolver() -> PathResolver:
    """PathResolver を取得（未ロード時はエラー）"""
    if _path_resolver is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _path_resolver


def resolve_outputs_dir() -> Path:
    """既定の成果物書込み先を返す (config 未ロードでも解決する)。

    エージェントの書込み経路は config を持たない静的ヘルパから呼ばれるため、
    ``get_path_resolver()`` の「未ロードなら RuntimeError」をここで吸収し、
    未ロード時は既定値 (``<data_root>/outputs/``) を返す。書込み先の
    解決が config のロード順に依存して CWD へ落ちることを防ぐ。
    """
    if _path_resolver is not None:
        return _path_resolver.resolve_outputs_dir()
    from backend.data_root import resolve_data_root

    return PathResolver.layout_path(resolve_data_root(root=get_project_root()), "outputs_dir")


def resolve_data_path(key: str, project_root: Path | None = None) -> Path:
    """``PathResolver.LAYOUT`` のキーをデータ根の下に解決する (config 未ロードでも解決する)。

    config ロード後 (``project_root`` 未指定か、グローバル ``PathResolver`` と同じ根)
    はグローバル ``PathResolver``。それ以外 (CLI の起動前・ログ設定・単体テスト) は
    ``--data-root`` → ``EVOREF_DATA_ROOT`` → ``<project_root>/userdata`` の 3 段で
    決めたデータ根に ``LAYOUT`` の相対パスを足す。
    """
    if _path_resolver is not None and (
        project_root is None or Path(project_root) == _path_resolver.root
    ):
        return _path_resolver.resolve_local(key)
    from backend.data_root import resolve_data_root

    root = project_root if project_root is not None else get_project_root()
    return PathResolver.layout_path(resolve_data_root(root=root), key)


def get_project_root() -> Path:
    """プロジェクトルートを取得"""
    if _path_resolver is not None:
        return _path_resolver.root
    return Path(__file__).parent.parent


def save_config_section(section: str, data: dict) -> dict:
    """設定セクションを保存してリロード

    1. config.yaml を読み込み
    2. config.yaml.bak にバックアップ
    3. 対象セクションを更新
    4. バリデーション
    5. 書き込み + リロード
    """
    from backend.io.readonly import guard_write
    from backend.schemas import validate_config

    project_root = get_project_root()
    config_path = project_root / "config.yaml"
    # 設定は store/ の外 (インストール根) だが、readonly の間は設定 API も止める
    # (c_05 §0.4.2)。API では 423 / E0423 になる。
    guard_write(config_path, outside_store=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 読み込み
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # バックアップ
    backup_path = project_root / "config.yaml.bak"
    import shutil
    shutil.copy2(config_path, backup_path)

    # セクション更新
    if section in raw and isinstance(raw[section], dict):
        raw[section] = _deep_merge(raw[section], data)
    else:
        raw[section] = data

    # バリデーション（Pro マージ前の raw を検証するため一時的に全体構築）
    merged = dict(raw)
    pro_cfg_path = project_root / "config.pro.yaml"
    if (project_root / "backend" / "pro").is_dir() and pro_cfg_path.exists():
        with open(pro_cfg_path, encoding="utf-8") as f:
            pro_cfg = yaml.safe_load(f)
        if pro_cfg:
            merged = _deep_merge(merged, pro_cfg)

    validated = validate_config(merged)

    # config.yaml に書き込み（Pro マージ前の raw を保存）
    # config.yaml が truncate 途中で壊れると起動不能になる。atomic + fsync。
    atomic_write_text(
        config_path,
        yaml.dump(raw, default_flow_style=False, allow_unicode=True, sort_keys=False),
        fsync=True,
    )

    logger.info("Config section '%s' saved to %s", section, config_path)

    # グローバル設定をリロード
    load_config(config_path, project_root)

    return validated


# モデルプロファイルのキャッシュ。((絶対パス, mtime_ns) -> 正規化済み profile dict)
# チャット応答パスから毎リクエスト呼ばれるため GGUF ヘッダと YAML の読取を
# キャッシュする。モデル差し替えで mtime が変わり自動 miss する
# (プロファイル YAML 自体の編集は再起動で反映)。
_profile_cache: dict[tuple[str, int], dict] = {}


# config 明示も arch プロファイル宣言も無い場合の context_size 既定。
# scripts/launch_llama.py::_CONTEXT_SIZE_DEFAULTS と一致させること
# (サーバ起動値 ``-c`` とランタイム token budget の値を揃えるため)。
_CONTEXT_SIZE_FALLBACK = 8192

# template family -> reasoning mode の fallback 対応 (プロファイル無宣言時に
# detect_template_family の結果から推定する)。
#   toggle = enable_thinking で ON/OFF 可 (Qwen3)
#   always = 常時 reasoning・OFF 不可 (DeepSeek-R1 / gpt-oss harmony)。enable_thinking は
#            送らないが reasoning_budget は honor しうる
#   none   = reasoning 非対応 (Gemma / Llama3 / 素の ChatML)
# unknown は map に載せず None (= 不明、ゲートしない / 後方互換) とする。
_TEMPLATE_FAMILY_REASONING_MODE: dict[str, str] = {
    "qwen3_thinking": "toggle",
    "deepseek_r1": "always",
    "harmony": "always",
    "gemma": "none",
    "llama3": "none",
    "chatml": "none",
}
_REASONING_MODES = frozenset({"toggle", "always", "none"})


def _model_path_for(cfg: dict, target: str) -> Path | None:
    """target のモデル GGUF 絶対パスを解決する。

    target は slot (``"base"``) と mode (``"chat"`` / ``"create"``) の 3 値。
    ``"create"`` のみ ``model_paths.create_model`` を見て、未設定なら base へ
    フォールバックする。モデル未設定時は ``None``。
    """
    model_paths = cfg.get("model_paths", {}) or {}
    base_model = model_paths.get("base_model") or ""
    if target == "create":
        model_rel = model_paths.get("create_model") or base_model
    else:
        model_rel = base_model
    if not model_rel:
        return None
    model_path = Path(model_rel)
    if not model_path.is_absolute():
        model_path = get_project_root() / model_path
    return model_path


def _normalize_profile(raw: dict) -> dict:
    """検証付きセクションを正規化する (キャッシュ格納前に 1 回だけ走る)。

    ``reasoning`` は ``ProfileReasoningConfig``、``sampling`` は
    ``ProfileSamplingConfig`` で検証する。宣言が無い / 検証に失敗したセクションは
    キーごと落とす (プロファイル全体は落とさない)。他のキーは素通しで、
    プロファイル YAML への寛容さ (綴り違いで起動を落とさない) を維持する。
    """
    profile = dict(raw)

    # reasoning は宣言されている場合のみ検証する (docs/c_15、profile=SSOT)。
    # 未宣言時はキーを落とし、template family fallback に委ねる。
    reasoning = profile.get("reasoning")
    if isinstance(reasoning, dict) and reasoning:
        from backend.schemas.llm import ProfileReasoningConfig

        try:
            profile["reasoning"] = ProfileReasoningConfig(**reasoning).model_dump()
        except Exception as e:
            logger.warning("Invalid reasoning profile (ignored): %s", e)
            profile.pop("reasoning", None)
    else:
        profile.pop("reasoning", None)

    sampling = profile.get("sampling")
    if isinstance(sampling, dict) and sampling:
        from backend.schemas.llm import ProfileSamplingConfig

        try:
            profile["sampling"] = ProfileSamplingConfig(
                **sampling,
            ).model_dump(exclude_none=True)
        except Exception as e:
            logger.warning("Invalid sampling profile (ignored): %s", e)
            profile.pop("sampling", None)
    else:
        profile.pop("sampling", None)

    return profile


def _profile_for(cfg: dict, target: str) -> dict:
    """target のモデルの有効プロファイルを返す (プロファイル解決の単一入口)。

    ``llama.auto_model_flags`` が false / モデル未設定 / GGUF 読取失敗 /
    プロファイル不在のときは ``{}``。((絶対パス, mtime_ns) でキャッシュ)。
    起動フラグ側 (scripts/launch_llama.py) と同じローダ (arch 層 + モデル別層)
    を流用し、SSOT を一本化する。
    """
    if not (cfg.get("llama", {}) or {}).get("auto_model_flags", True):
        return {}
    model_path = _model_path_for(cfg, target)
    if model_path is None:
        return {}
    try:
        mtime = model_path.stat().st_mtime_ns
    except OSError:
        return {}
    cache_key = (str(model_path), mtime)
    cached = _profile_cache.get(cache_key)
    if cached is not None:
        return cached

    profile: dict = {}
    try:
        from scripts.launch_llama import load_model_profile_for

        raw = load_model_profile_for(model_path, get_project_root())
        if raw:
            profile = _normalize_profile(raw)
    except Exception as e:
        logger.debug("Profile resolution failed for %s: %s", target, e)
        profile = {}

    _profile_cache[cache_key] = profile
    return profile


def _resolve_profile_sampling_for_mode(cfg: dict, mode: str) -> dict:
    """アクティブモデル ("chat"|"create") のプロファイルから sampling 既定を返す。"""
    return _profile_for(cfg, mode).get("sampling") or {}


def _resolve_profile_reasoning(cfg: dict, slot: str) -> dict:
    """slot ("base") のモデルプロファイルから ``reasoning`` を返す。"""
    return _profile_for(cfg, slot).get("reasoning") or {}


def resolve_reasoning_mode(
    cfg: dict, slot: str, *, chat_template: str | None = None,
    observed_reasoning_mode: str | None = None,
) -> str | None:
    """slot ("base") のモデル arch の reasoning mode を返す。

    戻り値: ``"toggle"`` | ``"always"`` | ``"none"`` | ``None`` (不明)。
      - toggle : ``enable_thinking`` で ON/OFF 可 (Qwen3)
      - always : 常時 reasoning・OFF 不可 (DeepSeek-R1 / gpt-oss)。``enable_thinking`` は
                 送らないが ``reasoning_budget`` は honor しうる
      - none   : reasoning 非対応 (Qwen2 / dense LFM2 等)。reasoning 系 kwarg を送らない

    優先順位 (docs/c_15、profile=SSOT): プロファイル ``reasoning.mode`` を**権威**とする。
    プロファイル未宣言時のみ、実機プローブ観測 (``observed_reasoning_mode``、未知モデルの
    シード) → ``chat_template`` の template family fallback の順で補う。**観測は profile を
    上書きしない** (宣言と実機の食い違いはプローブが WARNING + Status で可視化し、ユーザーが
    ``<data_root>/profiles/<arch>.yaml`` で是正する)。どれでも不明なら ``None``。
    """
    # プロファイル宣言が最優先 (profile = SSOT)。
    mode = _resolve_profile_reasoning(cfg, slot).get("mode")
    if mode in _REASONING_MODES:
        return mode
    # 以降は profile 未宣言時の fallback: 観測 (未知モデルのシード) → template family。
    if observed_reasoning_mode in _REASONING_MODES:
        return observed_reasoning_mode
    if chat_template:
        try:
            from backend.free.llm.model_metadata import detect_template_family

            return _TEMPLATE_FAMILY_REASONING_MODE.get(
                detect_template_family(chat_template),
            )
        except Exception:
            return None
    return None


def resolve_enable_thinking(
    cfg: dict,
    slot: str,
    *,
    explicit: bool | None,
    chat_template: str | None = None,
    observed_reasoning_mode: str | None = None,
) -> bool | None:
    """slot のモデルへ送る ``enable_thinking`` 値を解決する (能力判定 + 優先順位)。

    reasoning mode が ``none`` (思考しない) / ``always`` (常時思考・OFF 不可) の場合は
    ``enable_thinking`` を送らない (``None``)。``toggle`` / 不明 (``None``) の場合は
    優先順位 config 明示 (``explicit``) > プロファイル ``reasoning.enable_thinking`` 既定
    > ``None`` で解決する。

    Args:
        slot: ``"base"``。
        explicit: config 明示値 (``llama.enable_thinking``)。未指定は ``None``。
        chat_template: 取得済みなら渡す (base は ``metadata.chat_template``)。能力 fallback 用。

    Returns:
        送信すべき ``enable_thinking``、または ``None`` (送らない)。
    """
    mode = resolve_reasoning_mode(
        cfg, slot, chat_template=chat_template,
        observed_reasoning_mode=observed_reasoning_mode,
    )
    if mode in ("none", "always"):
        if explicit is not None:
            logger.warning(
                "enable_thinking ignored for %s model: reasoning mode=%s does not "
                "support enable_thinking toggle (override via "
                "models/profiles/<arch>.yaml reasoning.mode)",
                slot, mode,
            )
        return None

    if explicit is not None:
        return explicit
    default = _resolve_profile_reasoning(cfg, slot).get("enable_thinking")
    return default if isinstance(default, bool) else None


def resolve_client_reasoning(cfg: dict, slot: str) -> tuple[int, str]:
    """slot のモデル profile から client 側 reasoning watchdog 設定を返す (docs/c_15 B3)。

    戻り値: ``(client_think_budget, on_runaway)``。profile 未宣言時は ``(0, "fallback")``
    で watchdog 無効。``LocalClient`` に渡され、未閉じ ``<think>`` が budget chunk を超えたら
    ストリームを中断する。サーバ側で reasoning が分離されるモデルは content に ``<think>`` が
    出ないため発火しない。
    """
    reasoning = _resolve_profile_reasoning(cfg, slot)
    try:
        budget = int(reasoning.get("client_think_budget", 0) or 0)
    except (TypeError, ValueError):
        budget = 0
    on_runaway = reasoning.get("on_runaway") or "fallback"
    return max(0, budget), str(on_runaway)


def _profile_context_size(cfg: dict, target: str) -> int | None:
    """target のモデルプロファイルから ``context_size`` を返す。

    未宣言 / 512 未満 / 数値でない場合は ``None`` (呼び出し側が既定へ倒す)。
    起動フラグ側 (scripts/launch_llama.py) と同じプロファイルを読み、サーバ
    ``-c`` とランタイム値を揃える。
    """
    raw = _profile_for(cfg, target).get("context_size")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 512 else None


#: 接続中の llama-server (8080) が ``/props`` で報告した実際の n_ctx。``build_local_client`` が
#: 接続のたびに記録し、``/props`` に n_ctx が無ければ外す (不明 = 従来どおり config 由来の値)。
#:
#: config 由来の値 (明示 / 環境調整の保存値) は llama-server の ``-c`` と一致する前提だが、起動側が
#: 保守側へ倒れた回 (例: GPU 名だけ変わって環境移行の確認待ち → ``-c 8192``) には backend だけが
#: 保存値 (32768) を読み、8192 を超えるプロンプトを組んで HTTP 400 になる。予算は実際の窓を超えない。
_served_context_size: int | None = None


def note_served_context_size(n_ctx: object) -> None:
    """接続した llama-server の実際の n_ctx を記録する (正の整数でなければ「不明」へ戻す)。"""
    global _served_context_size
    valid = isinstance(n_ctx, int) and not isinstance(n_ctx, bool) and n_ctx > 0
    _served_context_size = n_ctx if valid else None


def _clamp_to_served(value: int) -> int:
    """config 由来の context_size を、実際に起動している llama-server の n_ctx 以下へ丸める。"""
    served = _served_context_size
    return min(int(value), served) if served else int(value)


def _resolve_profile_context_size_for_mode(cfg: dict, mode: str) -> int | None:
    """アクティブモード ("chat"|"create") のプロファイルから ``context_size`` を返す。"""
    return _profile_context_size(cfg, mode)


def resolve_context_size_for_mode(cfg: dict, mode: str) -> int:
    """アクティブモード ("chat"|"create") の有効 context_size を解決する。

    create モードで ``model_paths.create_model`` が base と別 arch (別 context
    window) の場合に、その実窓を反映する。``llama.context_size`` の明示はモードに
    依らず手動 pin として優先する。``auto`` / ``null`` は base と同じモデルなら環境調整の
    値 (:func:`resolve_context_size`)、別モデルの create はそのモデルの arch プロファイル > 既定
    (調整は ``base_model`` について決めるため。起動フラグ側の ``model_override`` と同じ)。
    いずれも接続中の llama-server の実際の n_ctx を超えない。
    """
    explicit = (cfg.get("llama") or {}).get("context_size")
    if explicit is not None and explicit != "auto":
        return _clamp_to_served(int(explicit))
    model_paths = cfg.get("model_paths") or {}
    create_model = model_paths.get("create_model")
    if mode != "create" or not create_model or create_model == model_paths.get("base_model"):
        return _clamp_to_served(_tuned_context_size(cfg))
    profile_ctx = _resolve_profile_context_size_for_mode(cfg, mode)
    if profile_ctx is not None:
        return _clamp_to_served(profile_ctx)
    return _clamp_to_served(_CONTEXT_SIZE_FALLBACK)


def _tuned_context_size(cfg: dict) -> int:
    """``llama.context_size: auto`` / ``null`` の値 (c_16 §7.2.3)。起動スクリプトが決めた保存値を読む。

    backend は測らない (``allow_decide=False``): 保存値が無い / 別の PC の値なら保守側 (8192)。
    llama-server の ``-c`` より小さい側に倒れるだけで、窓を超えるプロンプトは組まない。
    """
    from backend.free.core.tuning.resolve import resolve_tuned

    try:
        value = resolve_tuned(cfg, "ctx", project_root=get_project_root(), allow_decide=False).value
    except Exception as e:  # noqa: BLE001 - 読めなければ保守側 (チャットを止めない)
        logger.warning("Tuned context size unreadable: %s", e)
        return _CONTEXT_SIZE_FALLBACK
    return int(value) if isinstance(value, int) else _CONTEXT_SIZE_FALLBACK


def resolve_context_size(cfg: dict, slot: str) -> int:  # noqa: ARG001 - slot は "base" だけ (呼び手との互換)
    """slot ("base") の有効 context_size を解決する (docs/c_15)。

    config 明示 (``llama.context_size``) はそのまま。``auto`` / ``null`` は環境調整の項目 ctx
    (c_16 §7.2.3) で、起動スクリプト (scripts/launch_llama.py::resolve_context_size_for) が
    決めて保存した値を読み、llama-server 起動値とランタイム値 (token budget 表示等) を一致させる。
    起動側が保守側へ倒れて食い違ったときに備え、接続中の llama-server の実際の n_ctx
    を超えない。
    """
    explicit = (cfg.get("llama") or {}).get("context_size")
    if explicit is not None and explicit != "auto":
        return _clamp_to_served(int(explicit))
    return _clamp_to_served(_tuned_context_size(cfg))


def get_mode_generation_params(mode: str) -> dict:
    """指定モードの生成パラメータを取得

    Args:
        mode: モード名（"chat" または "create"）

    Returns:
        {"model": str, "temperature": float, "top_p": float,
         "top_k": int, "presence_penalty": float}

    Raises:
        ValueError: 不明なモード名
        RuntimeError: Config 未ロード
    """
    cfg = get_config()
    modes_cfg = cfg.get("modes", {})

    # デフォルト値
    # モデルパスは生成パラメータと分離し、create は model_paths.create_model から引く。
    defaults = {
        "chat": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "create": {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    }

    if mode not in defaults:
        raise ValueError(f"Unknown mode: {mode!r} (available: chat, create)")

    mode_cfg = dict(modes_cfg.get(mode, {}))
    # 生成パラメータのみを採用 (model はここには存在しない)
    mode_cfg.pop("model", None)
    # モデル arch プロファイルの sampling 既定を、汎用 modes.* より優先で適用する
    # (モデル切替時にモデル推奨値を自動反映する目的)。空 {} なら従来どおり。
    # 上書きしたい場合は <data_root>/profiles/<arch>.yaml か auto_model_flags:false。
    # 学習デルタは後段 (apply_deltas) で最優先に適用される。
    profile_sampling = _resolve_profile_sampling_for_mode(cfg, mode)
    params = {**defaults[mode], **mode_cfg, **profile_sampling}

    # ベースモデル。chat は常にここを採用。
    # create は model_paths.create_model 指定が無い/空の場合のみフォールバック。
    params["model"] = mode_base_model_raw(cfg.get("model_paths", {}), mode)

    # 学習デルタの適用（Level 1 生成パラメータ進化の結果）
    try:
        from backend.free.learning.generation_delta_store import GenerationDeltaStore
        from backend.free.learning.generation_param_evolver import apply_deltas
        resolver = get_path_resolver()
        # デルタはそれを学習したモデルのもの。いま生成するモデルのパーティションから読む
        # (create_model に base の create デルタを当てない、f_04 §1.2.0)。
        delta_path = resolver.learning_path_for(
            "generation_deltas_file", resolver.generating_model_key(mode),
        )
        mode_deltas = GenerationDeltaStore.load_mode(delta_path, mode)
        if mode_deltas:
            params = apply_deltas(params, mode_deltas)
            logger.debug("Applied generation deltas for mode %s: %s", mode, mode_deltas)
    except Exception as e:
        logger.warning("Failed to apply generation deltas: %s", e)

    return params
