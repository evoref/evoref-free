"""設定管理とパス解決"""

import yaml
from pathlib import Path

from pydantic import ValidationError

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
    """モデルパスとローカルパスを統一的に解決"""

    MODEL_DEFAULTS = {
        "base_model": "models/gemma-4-12b-it-qat-q4_0.gguf",
    }
    LOCAL_DEFAULTS = {
        "lora_adapter": "local/models/adapter.gguf",
        "lora_versions_dir": "local/lora_versions/",
        "lora_spsa_checkpoint": "local/models/lora_spsa_checkpoint.json",
        "assist_lora_adapter": "local/models/assist_adapter.gguf",
        "assist_lora_versions_dir": "local/models/assist_lora_versions/",
        "assist_lora_spsa_checkpoint": "local/models/assist_lora_spsa_checkpoint.json",
        # Level 2 base=C: control vector 本体 / 版管理 / 作業用ディレクトリ
        "control_vector_adapter": "local/models/control_vector.gguf",
        "control_vector_versions_dir": "local/models/control_vector_versions/",
        "cvector_work_dir": "local/cvector/",
        "experience_assist_file": "local/experience_assist.json",
        "eval_assist_file": "local/eval_assist.json",
        "assist_calibration_file": "local/assist_calibration.json",
        "rag_judge_events_file": "local/rag_judge_events.jsonl",
        "lora_archive_dir": "local/lora_archive/",
        "embed_lora_adapter": "local/models/embed_adapter.gguf",
        "embed_lora_versions_dir": "local/models/embed_lora_versions/",
        "vectors_dir": "local/vectors/",
        "knowledge_dir": "local/knowledge/",
        "experience_file": "local/experience.json",
        "eval_core_file": "local/eval_core.json",
        "model_state_file": "local/model_state.json",
        "model_quality_file": "local/model_quality.json",
        # EvorefMem ローカル状態
        "local_state_file": "local/state.json",
        "memory_dir": "local/memory/",
        "prompts_dir": "local/prompts/",
        "cartridges_dir": "local/cartridges/",
        "history_dir": "local/history/",
        "learned_patterns_file": "local/learned_patterns.json",
        "themes_dir": "local/themes/",
        # SchemaMigrator バックアップ / 旧 prompts 退避先
        "migration_archive_dir": "local/migration_archive/",
        # EvorefMem トリガ辞書 (pin / fact / classify) のユーザー上書き先。
        # 同梱 default は ``backend/free/memory/_defaults/triggers/``。
        "triggers_dir": "local/triggers/",
        # staged クリエイトパイプラインの一時ワークスペース。
        "create_workspace_dir": "local/create/",
        # base 学習データの (model × mode) パーティションルート。
        "learning_dir": "local/learning/",
    }

    # resolve_learning で active モデルパーティション配下へ rebase する base 学習キーと、
    # ``learning_dir/<stem>/`` からの相対サブパス。ここに無いキー (assist / 共有 / embed)
    # は resolve_local へ素通しする (assist_* や memory_dir を巻き込まないため allow-list)。
    _LEARNING_SUBPATH = {
        "experience_file": "experience.json",
        "prompts_dir": "prompts",
        "lora_adapter": "models/adapter.gguf",
        "lora_versions_dir": "models/lora_versions",
        "lora_spsa_checkpoint": "models/lora_spsa_checkpoint.json",
        "control_vector_adapter": "models/control_vector.gguf",
        "control_vector_versions_dir": "models/control_vector_versions",
        "cvector_work_dir": "cvector",
    }

    # ``_LEARNING_SUBPATH`` のうち、``learning.level2_adapter_partition=="model_mode"``
    # の時に ``<stem>/<mode>/...`` と mode サブディレクトリを追加で挟む対象キー。
    # experience_file / prompts_dir は既に別の方法 (mode.md ファイル名 / mode タグ) で
    # モード分離済みのため対象外。control_vector 系 / cvector_work_dir は本機能の
    # スコープ外 (Level 2 base=cvector 手法は今回のモード分離の対象としない)。
    _MODE_PARTITIONED_KEYS = frozenset(
        {"lora_adapter", "lora_versions_dir", "lora_spsa_checkpoint"},
    )

    # resolve_assist_learning で使う assist LoRA 用の (mode のみ) パーティション先。
    # base の ``_LEARNING_SUBPATH`` と異なり model stem 軸を持たない
    # (``learning_dir/assist/<mode>/...``) — assist は経験/プロンプトがモデル
    # 非依存という既存設計 (docs/f_04_self_learning.md) を維持するため。
    _ASSIST_MODE_SUBPATH = {
        "assist_lora_adapter": "adapter.gguf",
        "assist_lora_versions_dir": "lora_versions",
        "assist_lora_spsa_checkpoint": "lora_spsa_checkpoint.json",
    }

    def __init__(self, config: dict, project_root: Path):
        self.root = project_root
        self.models = config.get("model_paths", {})
        self.local = config.get("local_paths", {})
        # base 学習データの (model × mode) パーティション state。
        # active stem 未設定 or flag false なら resolve_learning は resolve_local 同等
        # (レガシー flat レイアウト)。set_active_model_stem で起動時に確定する。
        self._active_stem: str | None = None
        self._partition_enabled: bool = bool(
            (config.get("learning", {}) or {}).get("partition_by_base_model", True)
        )
        # embed_instruction 系データの (embedding モデル) パーティション state。
        # base 学習パーティション (_active_stem) とは独立した軸。
        self._active_embed_stem: str | None = None
        self._active_assist_stem: str | None = None
        # Level 2 base/assist LoRA アダプタの (mode) パーティション state。
        # "model" (既定) では resolve_learning/resolve_assist_learning は mode 引数を
        # 無視し、従来どおりモデル単位で 1 アダプタを共有する。"model_mode" のときのみ
        # chat/create で別ファイルへ分離する。AppState.current_mode の初期値と揃え、
        # active_mode の既定は "chat"。
        self._active_mode: str = "chat"
        self._adapter_partition_mode: str = str(
            (config.get("learning", {}) or {}).get("level2_adapter_partition", "model"),
        )

    def resolve_model(self, key: str) -> Path:
        """モデルパス解決

        config の ``model_paths[key]`` を優先し、無ければ ``MODEL_DEFAULTS[key]``
        を使う。``dict.get(key, MODEL_DEFAULTS[key])`` は default 引数を先に評価
        するため、``MODEL_DEFAULTS`` に無いキー (assist_model / embed_model /
        create_model 等) を config 側に持っていても KeyError で落ちていた。
        """
        raw = self.models.get(key) or self.MODEL_DEFAULTS.get(key)
        if raw is None:
            raise KeyError(f"unknown model_paths key: {key!r}")
        return self._to_absolute(raw)

    def resolve_local(self, key: str) -> Path:
        """ローカルパス解決（読み書きリソース）"""
        raw = self.local.get(key, self.LOCAL_DEFAULTS[key])
        return self._to_absolute(raw)

    @property
    def active_model_stem(self) -> str | None:
        """base 学習パーティションの active モデル stem (未設定なら ``None``)。

        downstream の構築側が SemMem ``learn.*`` subject 用スラグ
        (``model_slug``) を導出するために参照する。
        """
        return self._active_stem

    @property
    def partition_enabled(self) -> bool:
        """base 学習パーティションが有効か (``partition_by_base_model``)。"""
        return self._partition_enabled

    def set_active_model_stem(self, stem: str | None) -> None:
        """base 学習パーティションの active モデル stem を設定する。

        ``None`` / 空 のときは partition を無効化し、``resolve_learning`` が
        ``resolve_local`` へ素通しする (レガシー flat レイアウト)。起動時に
        ``ModelState.current_filename`` の stem で確定し、モデル切替時に更新する。
        """
        self._active_stem = stem or None

    @property
    def active_embedding_model_stem(self) -> str | None:
        """embed_instruction パーティションの active 埋め込みモデル stem。"""
        return self._active_embed_stem

    def set_active_embedding_model_stem(self, stem: str | None) -> None:
        """embed_instruction パーティションの active 埋め込みモデル stem を設定する。

        base 学習パーティション (``set_active_model_stem``) とは独立した軸。
        ``None`` / 空のときは ``resolve_embed_instruction_dir`` が flat レイアウト
        (``resolve_local("prompts_dir")``) へ素通しする。
        """
        self._active_embed_stem = stem or None

    def resolve_embed_instruction_dir(self) -> Path:
        """embed_instruction 系データの保存先を **embedding モデル単位**で解決する。

        embed_instruction は埋め込みモデル向けのクエリ指示文であり、base モデル
        切替とは無関係に保持されるべきだが、従来 ``SystemPromptManager.prompt_dir``
        (base 学習パーティション) に同居しており base モデル切替で誤って
        切り替わっていた (2026-07-18)。partition 無効 / active embed stem 未確定
        時は ``resolve_local("prompts_dir")`` (従来の flat 配置) へ素通しし、
        後方互換を保つ。
        """
        if not self._partition_enabled or not self._active_embed_stem:
            return self.resolve_local("prompts_dir")
        return self.resolve_local("learning_dir") / "embed" / self._active_embed_stem

    @property
    def active_assist_model_stem(self) -> str | None:
        """assist プロンプトパーティションの active アシストモデル stem。"""
        return self._active_assist_stem

    def set_active_assist_model_stem(self, stem: str | None) -> None:
        """assist プロンプトパーティションの active モデル stem を設定する。

        base 学習 (``set_active_model_stem``) / embed_instruction
        (``set_active_embedding_model_stem``) とは独立した第 3 の軸。
        """
        self._active_assist_stem = stem or None

    def resolve_assist_prompt_dir(self) -> Path:
        """アシストプロンプトの保存先を **アシストモデル単位**で解決する。

        アシストプロンプト (rag_necessity / rag_quality / tool_call / note_evolve)
        は進化の対象で、進化した文面は **そのモデルの癖に合わせて最適化される**。
        従来 ``local/prompts/`` に flat 配置されており、アシストモデルを差し替えても
        前モデル向けに進化した文面がそのまま使われていた。base 学習
        (``resolve_learning``) と embed_instruction
        (``resolve_embed_instruction_dir``) は既にモデル別に分かれており、
        assist だけが取り残されていた。

        partition 無効 / active assist stem 未確定時は ``resolve_local("prompts_dir")``
        (従来の flat 配置) へ素通しし、後方互換を保つ。
        """
        if not self._partition_enabled or not self._active_assist_stem:
            return self.resolve_local("prompts_dir")
        return self.resolve_local("learning_dir") / "assist" / self._active_assist_stem

    @property
    def active_mode(self) -> str:
        """Level 2 アダプタパーティションの active モード (``"chat"``/``"create"``)。"""
        return self._active_mode

    def set_active_mode(self, mode: str | None) -> None:
        """モード切替時に呼ぶ。未指定/不明値は安全側で ``"chat"`` に丸める。"""
        self._active_mode = mode if mode in ("chat", "create") else "chat"

    @property
    def adapter_partition_mode(self) -> str:
        """``learning.level2_adapter_partition`` の値 (``"model"``/``"model_mode"``)。"""
        return self._adapter_partition_mode

    def resolve_learning(self, key: str, mode: str | None = None) -> Path:
        """base 学習データのパスを **active** モデルパーティション配下で解決する。

        ``_LEARNING_SUBPATH`` の base 学習キーのみ ``learning_dir/<active_stem>/...``
        配下へ rebase する。partition 無効 (``partition_by_base_model=false`` /
        active stem 未設定) または非対象キーは ``resolve_local`` へ素通しする
        (assist・共有・レガシー)。``learning_dir`` は ``resolve_local`` 経由で解決
        するため ``--isolate-data`` の prefix 書換えを自動継承する。

        ``mode`` は ``key`` が ``_MODE_PARTITIONED_KEYS`` に属し、かつ
        ``adapter_partition_mode=="model_mode"`` のときだけ効く (``learning_path_for``
        参照)。省略時は ``active_mode`` を使う。"model" (既定) スキームでは mode に
        関わらず常に同一パスを返す (後方互換)。
        """
        if (
            not self._partition_enabled
            or not self._active_stem
            or key not in self._LEARNING_SUBPATH
        ):
            return self.resolve_local(key)
        effective_mode = mode if mode is not None else self._active_mode
        return self.learning_path_for(
            key, self._stem_for(key, effective_mode), mode=effective_mode,
        )

    def _stem_for(self, key: str, mode: str | None) -> str:
        """``key`` / ``mode`` に対するパーティション根のモデル stem を返す。

        LoRA アダプタ系 (``_MODE_PARTITIONED_KEYS``) は **そのモードが実際に
        ロードするモデル** を根にする。アダプタは特定モデルの重みへの差分なので、
        「どのモデル向けか」が保存先の第一キーであるべき。

        従来は常に ``_active_stem`` (= ``model_paths.base_model`` = chat のモデル)
        を根にしていたため、create が別モデル (``create_model``) を使う構成では
        create のアダプタが chat モデルのパーティション配下に置かれていた。この
        状態で chat の ``base_model`` を差し替えると、``create_model`` は変わって
        いないのに create のアダプタが参照されなくなる (根が別 stem に移るため)。

        アダプタ以外 (experience / prompts / cvector 系) は従来どおり
        ``_active_stem``。経験とプロンプトは base 学習パーティション単位で
        まとまっている既存設計 (docs/f_04_self_learning.md) を変えない。

        ``model_paths`` に該当モデルの宣言が無い場合も ``_active_stem`` に倒す
        (宣言されていないモデル名でパーティションを作らない)。
        """
        assert self._active_stem is not None  # 呼出元でガード済み
        if (
            mode is None
            or key not in self._MODE_PARTITIONED_KEYS
            or self._adapter_partition_mode != "model_mode"
        ):
            return self._active_stem
        raw = mode_base_model_raw(self.models, mode, default="")
        return Path(raw).stem if raw else self._active_stem

    def learning_path_for(self, key: str, stem: str, mode: str | None = None) -> Path:
        """**指定** モデル stem のパーティション配下で base 学習パスを解決する。

        active stem に依存せず任意モデルのパーティションパスを得る。flat→partition
        移行 (producer 別 / experience の base_model タグ別バケツ) で、active モデル
        とは異なるモデルのパーティションへ振り分けるために使う。``_LEARNING_SUBPATH``
        非対象キーは ``resolve_local`` へ素通しする。

        ``mode`` は ``key in _MODE_PARTITIONED_KEYS`` かつ
        ``adapter_partition_mode=="model_mode"`` のときのみ ``<stem>/<mode>/...`` と
        サブディレクトリを追加する。それ以外 (既定 "model" スキーム、または
        mode 分割対象外キー) は従来どおり ``<stem>/...`` のまま (mode を無視する)。
        """
        if key not in self._LEARNING_SUBPATH:
            return self.resolve_local(key)
        root = self.resolve_local("learning_dir") / stem
        if (
            mode is not None
            and key in self._MODE_PARTITIONED_KEYS
            and self._adapter_partition_mode == "model_mode"
        ):
            root = root / mode
        return root / self._LEARNING_SUBPATH[key]

    def resolve_assist_learning(self, key: str, mode: str | None = None) -> Path:
        """assist LoRA アダプタ/バージョン履歴/チェックポイントを **mode 単位**で解決する。

        base の ``resolve_learning`` と異なり model stem 軸を持たない — assist の
        経験・プロンプトはモデル識別子で分離しない既存方針 (docs/f_04_self_learning.md)
        を踏襲し、``adapter_partition_mode=="model_mode"`` のときのみ
        ``learning_dir/assist/<mode>/...`` という mode のみの軸を新設する。
        非対象キー / "model" (既定) スキームは ``resolve_local`` へ素通しする
        (従来の flat 配置、後方互換)。``mode`` 省略時は ``resolve_learning`` と同様
        ``active_mode`` を使う。
        """
        if key not in self._ASSIST_MODE_SUBPATH or self._adapter_partition_mode != "model_mode":
            return self.resolve_local(key)
        effective_mode = mode if mode is not None else self._active_mode
        return (
            self.resolve_local("learning_dir")
            / "assist" / effective_mode / self._ASSIST_MODE_SUBPATH[key]
        )

    def _to_absolute(self, raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else self.root / path

    def ensure_local_dirs(self) -> None:
        """ローカルパスのディレクトリを自動作成"""
        for key in self.LOCAL_DEFAULTS:
            path = self.resolve_local(key)
            # ファイルパスの場合は親ディレクトリを作成
            if "." in path.name:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
        # ログディレクトリも作成
        logs_dir = self.root / "local" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)


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
    from backend.schemas import validate_config

    project_root = get_project_root()
    config_path = project_root / "config.yaml"

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
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

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

    target は slot (``"base"`` / ``"assist"``) と mode (``"chat"`` / ``"create"``)
    の 4 値。``"create"`` のみ ``model_paths.create_model`` を見て、未設定なら
    base へフォールバックする。モデル未設定時は ``None``。
    """
    model_paths = cfg.get("model_paths", {}) or {}
    base_model = model_paths.get("base_model") or ""
    if target == "assist":
        model_rel = model_paths.get("assist_model") or ""
    elif target == "create":
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
    """slot ("base"|"assist") のモデルプロファイルから ``reasoning`` を返す。"""
    return _profile_for(cfg, slot).get("reasoning") or {}


def resolve_sampling_params(cfg: dict, target: str) -> dict:
    """target のモデルプロファイルが宣言する sampling パラメータを返す。

    target は slot (``"base"`` / ``"assist"``) と mode (``"chat"`` / ``"create"``)。
    宣言の無いキーは含まれないので、呼び出し側の既定を潰さない。base 側は
    :func:`get_mode_generation_params` が ``modes.*`` より優先で適用し、assist 側は
    ``AssistModelClient`` がリクエスト payload の既定として使う。
    """
    return _profile_for(cfg, target).get("sampling") or {}


def resolve_reasoning_mode(
    cfg: dict, slot: str, *, chat_template: str | None = None,
    observed_reasoning_mode: str | None = None,
) -> str | None:
    """slot ("base"|"assist") のモデル arch の reasoning mode を返す。

    戻り値: ``"toggle"`` | ``"always"`` | ``"none"`` | ``None`` (不明)。
      - toggle : ``enable_thinking`` で ON/OFF 可 (Qwen3)
      - always : 常時 reasoning・OFF 不可 (DeepSeek-R1 / gpt-oss)。``enable_thinking`` は
                 送らないが ``reasoning_budget`` は honor しうる
      - none   : reasoning 非対応 (Qwen2 / dense LFM2 等)。reasoning 系 kwarg を送らない

    優先順位 (docs/c_15、profile=SSOT): プロファイル ``reasoning.mode`` を**権威**とする。
    プロファイル未宣言時のみ、実機プローブ観測 (``observed_reasoning_mode``、未知モデルの
    シード) → ``chat_template`` の template family fallback の順で補う。**観測は profile を
    上書きしない** (宣言と実機の食い違いはプローブが WARNING + Status で可視化し、ユーザーが
    ``local/profiles/<arch>.yaml`` で是正する)。どれでも不明なら ``None``。
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
        slot: ``"base"`` | ``"assist"``。
        explicit: config 明示値 (base=``llama.enable_thinking`` / assist=
            ``chat_template_kwargs.enable_thinking``)。未指定は ``None``。
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


def _resolve_profile_context_size(cfg: dict, slot: str) -> int | None:
    """slot ("base"|"assist") のプロファイルから ``context_size`` を返す。"""
    return _profile_context_size(cfg, slot)


def _resolve_profile_context_size_for_mode(cfg: dict, mode: str) -> int | None:
    """アクティブモード ("chat"|"create") のプロファイルから ``context_size`` を返す。"""
    return _profile_context_size(cfg, mode)


def resolve_context_size_for_mode(cfg: dict, mode: str) -> int:
    """アクティブモード ("chat"|"create") の有効 context_size を解決する。

    create モードで ``model_paths.create_model`` が base と別 arch (別 context
    window) の場合に、その実窓を反映する。優先順位は :func:`resolve_context_size`
    と同じ: config 明示 (``llama.context_size``) > arch プロファイル > 既定。
    ``llama.context_size`` の明示はモードに依らず手動 pin として優先する。
    """
    explicit = (cfg.get("llama") or {}).get("context_size")
    if explicit is not None:
        return int(explicit)
    profile_ctx = _resolve_profile_context_size_for_mode(cfg, mode)
    if profile_ctx is not None:
        return profile_ctx
    return _CONTEXT_SIZE_FALLBACK


def resolve_context_size(cfg: dict, slot: str) -> int:
    """slot ("base"|"assist") の有効 context_size を解決する (docs/c_15)。

    優先順位: config 明示 (``llama.context_size`` / ``assist_model.local.context_size``)
    > arch プロファイル ``context_size`` > 既定 (``_CONTEXT_SIZE_FALLBACK``)。
    起動フラグ ``-c`` (scripts/launch_llama.py::resolve_context_size_for) と同じ
    優先順位で解決し、llama-server 起動値とランタイム値 (token budget 表示等) を
    一致させる。
    """
    if slot == "assist":
        local = (cfg.get("assist_model") or {}).get("local") or {}
        explicit = local.get("context_size")
    else:
        explicit = (cfg.get("llama") or {}).get("context_size")
    if explicit is not None:
        return int(explicit)
    profile_ctx = _resolve_profile_context_size(cfg, slot)
    if profile_ctx is not None:
        return profile_ctx
    return _CONTEXT_SIZE_FALLBACK


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
    # 上書きしたい場合は local/profiles/<arch>.yaml か auto_model_flags:false。
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
        delta_path = get_project_root() / "local" / "generation_deltas.json"
        mode_deltas = GenerationDeltaStore.load_mode(delta_path, mode)
        if mode_deltas:
            params = apply_deltas(params, mode_deltas)
            logger.debug("Applied generation deltas for mode %s: %s", mode, mode_deltas)
    except Exception as e:
        logger.warning("Failed to apply generation deltas: %s", e)

    return params


def get_mode_assist_model_path(mode: str) -> str:
    """指定モードで使用すべきアシストモデルの GGUF パスを解決する

    create モードで ``model_paths.assist_create_model`` が指定されていれば
    それを、未指定/空文字列なら ``model_paths.assist_model`` にフォールバック
    する (``get_mode_generation_params`` の ``create_model`` 解決と対称)。

    Args:
        mode: "chat" または "create"

    Returns:
        GGUF パス文字列 (config.yaml からの相対 or 絶対)
    """
    cfg = get_config()
    model_paths = cfg.get("model_paths", {})
    assist_model = model_paths.get(
        "assist_model", "models/gemma-4-E4B_q4_0-it.gguf",
    )
    if mode == "create":
        return model_paths.get("assist_create_model") or assist_model
    return assist_model


def get_mode_lora_path(mode: str) -> Path:
    """指定モードで使用すべき base LoRA アダプタの絶対パスを解決する (存在確認はしない)。

    ``learning.level2_adapter_partition=="model"`` (既定) のときは mode に依らず
    常に同一パスを返す — ``/api/mode/switch`` の再起動判定がこの関数の戻り値を
    比較するだけで LoRA 差分による再起動要否を導けるようにするため
    (既定運用では常に等しい = 再起動トリガーなし、後方互換)。
    "model_mode" のときのみ mode 別の実際のパスを返す。

    Args:
        mode: "chat" または "create"

    Returns:
        解決された絶対パス (ファイルの存在は保証しない)
    """
    resolver = get_path_resolver()
    effective_mode = mode if resolver.adapter_partition_mode == "model_mode" else None
    return resolver.resolve_learning("lora_adapter", mode=effective_mode)


def get_mode_assist_lora_path(mode: str) -> Path:
    """指定モードで使用すべき assist LoRA アダプタの絶対パスを解決する (存在確認はしない)。

    ``get_mode_lora_path`` と対称。"model" (既定) では常に flat 共有パスを返す。

    Args:
        mode: "chat" または "create"

    Returns:
        解決された絶対パス (ファイルの存在は保証しない)
    """
    resolver = get_path_resolver()
    if resolver.adapter_partition_mode != "model_mode":
        return resolver.resolve_local("assist_lora_adapter")
    return resolver.resolve_assist_learning("assist_lora_adapter", mode=mode)
