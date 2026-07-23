"""設定管理とパス解決"""

import yaml
from pathlib import Path

from pydantic import ValidationError

from backend.log_config import get_logger

logger = get_logger("config")

_config: dict | None = None
_path_resolver: "PathResolver | None" = None


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
        "lora_archive_dir": "local/lora_archive/",
        "embed_lora_adapter": "local/models/embed_adapter.gguf",
        "embed_lora_versions_dir": "local/models/embed_lora_versions/",
        "vectors_dir": "local/vectors/",
        "knowledge_dir": "local/knowledge/",
        "experience_file": "local/experience.json",
        "eval_core_file": "local/eval_core.json",
        "model_state_file": "local/model_state.json",
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
        # staged コーディングパイプラインの一時ワークスペース。
        "coding_workspace_dir": "local/coding/",
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
        # Level 2 base/assist LoRA アダプタの (mode) パーティション state。
        # "model" (既定) では resolve_learning/resolve_assist_learning は mode 引数を
        # 無視し、従来どおりモデル単位で 1 アダプタを共有する。"model_mode" のときのみ
        # chat/coding で別ファイルへ分離する。AppState.current_mode の初期値と揃え、
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
        coding_model 等) を config 側に持っていても KeyError で落ちていた。
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
    def active_mode(self) -> str:
        """Level 2 アダプタパーティションの active モード (``"chat"``/``"coding"``)。"""
        return self._active_mode

    def set_active_mode(self, mode: str | None) -> None:
        """モード切替時に呼ぶ。未指定/不明値は安全側で ``"chat"`` に丸める。"""
        self._active_mode = mode if mode in ("chat", "coding") else "chat"

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
        return self.learning_path_for(key, self._active_stem, mode=effective_mode)

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


# モデル arch プロファイルの sampling 既定キャッシュ。((絶対パス, mtime_ns) -> dict)
# GGUF ヘッダ読取を毎回走らせないため。モデル差し替えで mtime が変わり自動 miss する。
_profile_sampling_cache: dict[tuple[str, int], dict] = {}

# プロファイル sampling として採用する既知キー
_PROFILE_SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty",
)

# モデル arch プロファイルの reasoning 既定キャッシュ。((絶対パス, mtime_ns) -> dict)
_profile_reasoning_cache: dict[tuple[str, int], dict] = {}

# モデル arch プロファイルの context_size キャッシュ。((絶対パス, mtime_ns) -> int | None)
# チャット応答パスから毎リクエスト呼ばれるため GGUF ヘッダ読取をキャッシュする。
_profile_context_cache: dict[tuple[str, int], int | None] = {}

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


def _resolve_profile_sampling_for_mode(cfg: dict, mode: str) -> dict:
    """アクティブモデルの arch プロファイルから sampling 既定を解決する。

    ``llama.auto_model_flags`` が false / GGUF 読取失敗 / arch 不明 / プロファイル
    不在のときは ``{}`` を返す。結果は (絶対パス, mtime) でキャッシュする。
    起動フラグ側 (scripts/launch_llama.py) と同じ GGUF reader / プロファイル
    ローダを流用し、SSOT を一本化する。
    """
    if not (cfg.get("llama", {}) or {}).get("auto_model_flags", True):
        return {}

    model_paths = cfg.get("model_paths", {})
    base_model = model_paths.get("base_model", "")
    if mode == "coding":
        model_rel = model_paths.get("coding_model") or base_model
    else:
        model_rel = base_model
    if not model_rel:
        return {}

    project_root = get_project_root()
    model_path = Path(model_rel)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    try:
        mtime = model_path.stat().st_mtime_ns
    except OSError:
        return {}

    cache_key = (str(model_path), mtime)
    if cache_key in _profile_sampling_cache:
        return _profile_sampling_cache[cache_key]

    sampling: dict = {}
    try:
        from scripts.launch_llama import load_model_profile, read_gguf_metadata

        meta = read_gguf_metadata(model_path)
        profile = load_model_profile(meta.get("architecture"), project_root)
        raw = profile.get("sampling") or {}
        if isinstance(raw, dict):
            sampling = {
                k: v
                for k, v in raw.items()
                if k in _PROFILE_SAMPLING_KEYS and v is not None
            }
    except Exception as e:
        logger.debug("Profile sampling resolution failed for mode %s: %s", mode, e)
        sampling = {}

    _profile_sampling_cache[cache_key] = sampling
    return sampling


def _resolve_profile_reasoning(cfg: dict, slot: str) -> dict:
    """slot ("base"|"assist") のモデル arch プロファイルから ``reasoning`` を返す。

    ``auto_model_flags=false`` / モデル未設定 / GGUF 読取失敗 / プロファイル不在 /
    ``reasoning`` 不在のときは ``{}``。((絶対パス, mtime) でキャッシュ)。
    """
    if not (cfg.get("llama", {}) or {}).get("auto_model_flags", True):
        return {}
    model_paths = cfg.get("model_paths", {}) or {}
    key = "base_model" if slot == "base" else "assist_model"
    model_rel = model_paths.get(key, "")
    if not model_rel:
        return {}
    project_root = get_project_root()
    model_path = Path(model_rel)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    try:
        mtime = model_path.stat().st_mtime_ns
    except OSError:
        return {}
    cache_key = (str(model_path), mtime)
    if cache_key in _profile_reasoning_cache:
        return _profile_reasoning_cache[cache_key]

    reasoning: dict = {}
    try:
        from scripts.launch_llama import load_model_profile, read_gguf_metadata

        meta = read_gguf_metadata(model_path)
        profile = load_model_profile(meta.get("architecture"), project_root)
        raw = profile.get("reasoning")
        # reasoning セクションが宣言されている場合のみ ProfileReasoningConfig で
        # 検証し、既定を埋めた dict を返す (docs/c_15、profile=SSOT)。未宣言時は
        # ``{}`` を保ち template family fallback (resolve_reasoning_mode) に委ねる。
        if isinstance(raw, dict) and raw:
            from backend.schemas.llm import ProfileReasoningConfig

            try:
                reasoning = ProfileReasoningConfig(**raw).model_dump()
            except Exception as e:
                logger.warning(
                    "Invalid reasoning profile for slot %s (using defaults): %s",
                    slot, e,
                )
                reasoning = {}
    except Exception as e:
        logger.debug("Profile reasoning resolution failed for slot %s: %s", slot, e)
        reasoning = {}

    _profile_reasoning_cache[cache_key] = reasoning
    return reasoning


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


def _resolve_profile_context_size(cfg: dict, slot: str) -> int | None:
    """slot ("base"|"assist") の arch プロファイルから ``context_size`` を返す。

    ``auto_model_flags=false`` / モデル未設定 / GGUF 読取失敗 / プロファイル不在 /
    ``context_size`` 未宣言・512 未満のときは ``None``。((絶対パス, mtime) でキャッシュ)。
    起動フラグ側 (scripts/launch_llama.py) と同じ GGUF reader / プロファイルローダを
    流用し、サーバ ``-c`` とランタイム値を揃える。
    """
    if not (cfg.get("llama", {}) or {}).get("auto_model_flags", True):
        return None
    model_paths = cfg.get("model_paths", {}) or {}
    key = "base_model" if slot == "base" else "assist_model"
    model_rel = model_paths.get(key, "")
    if not model_rel:
        return None
    project_root = get_project_root()
    model_path = Path(model_rel)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    try:
        mtime = model_path.stat().st_mtime_ns
    except OSError:
        return None
    cache_key = (str(model_path), mtime)
    if cache_key in _profile_context_cache:
        return _profile_context_cache[cache_key]

    value: int | None = None
    try:
        from scripts.launch_llama import load_model_profile, read_gguf_metadata

        meta = read_gguf_metadata(model_path)
        profile = load_model_profile(meta.get("architecture"), project_root)
        raw = profile.get("context_size")
        if raw is not None:
            ivalue = int(raw)
            if ivalue >= 512:
                value = ivalue
    except Exception as e:
        logger.debug("Profile context_size resolution failed for slot %s: %s", slot, e)
        value = None

    _profile_context_cache[cache_key] = value
    return value


def _resolve_profile_context_size_for_mode(cfg: dict, mode: str) -> int | None:
    """アクティブモードのモデル arch プロファイルから ``context_size`` を返す。

    ``_resolve_profile_sampling_for_mode`` と対称で、``mode=="coding"`` のときは
    ``model_paths.coding_model`` (未設定なら base_model) を、それ以外は base_model
    を参照する。``auto_model_flags=false`` / モデル未設定 / 読取失敗 / プロファイル
    不在・512 未満は ``None``。((絶対パス, mtime) でキャッシュ、slot 版と共有)。
    """
    if not (cfg.get("llama", {}) or {}).get("auto_model_flags", True):
        return None
    model_paths = cfg.get("model_paths", {}) or {}
    base_model = model_paths.get("base_model", "")
    if mode == "coding":
        model_rel = model_paths.get("coding_model") or base_model
    else:
        model_rel = base_model
    if not model_rel:
        return None
    project_root = get_project_root()
    model_path = Path(model_rel)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    try:
        mtime = model_path.stat().st_mtime_ns
    except OSError:
        return None
    cache_key = (str(model_path), mtime)
    if cache_key in _profile_context_cache:
        return _profile_context_cache[cache_key]

    value: int | None = None
    try:
        from scripts.launch_llama import load_model_profile, read_gguf_metadata

        meta = read_gguf_metadata(model_path)
        profile = load_model_profile(meta.get("architecture"), project_root)
        raw = profile.get("context_size")
        if raw is not None:
            ivalue = int(raw)
            if ivalue >= 512:
                value = ivalue
    except Exception as e:
        logger.debug("Profile context_size resolution failed for mode %s: %s", mode, e)
        value = None

    _profile_context_cache[cache_key] = value
    return value


def resolve_context_size_for_mode(cfg: dict, mode: str) -> int:
    """アクティブモード ("chat"|"coding") の有効 context_size を解決する。

    coding モードで ``model_paths.coding_model`` が base と別 arch (別 context
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
        mode: モード名（"chat" または "coding"）

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
    # モデルパスは生成パラメータと分離し、coding は model_paths.coding_model から引く。
    defaults = {
        "chat": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "presence_penalty": 0.0,
        },
        "coding": {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
        },
    }

    if mode not in defaults:
        raise ValueError(f"Unknown mode: {mode!r} (available: chat, coding)")

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
    # coding は model_paths.coding_model 指定が無い/空の場合のみフォールバック。
    model_paths = cfg.get("model_paths", {})
    base_model = model_paths.get("base_model", "models/gemma-4-12b-it-qat-q4_0.gguf")
    if mode == "coding":
        params["model"] = model_paths.get("coding_model") or base_model
    else:
        params["model"] = base_model

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

    coding モードで ``model_paths.assist_coding_model`` が指定されていれば
    それを、未指定/空文字列なら ``model_paths.assist_model`` にフォールバック
    する (``get_mode_generation_params`` の ``coding_model`` 解決と対称)。

    Args:
        mode: "chat" または "coding"

    Returns:
        GGUF パス文字列 (config.yaml からの相対 or 絶対)
    """
    cfg = get_config()
    model_paths = cfg.get("model_paths", {})
    assist_model = model_paths.get(
        "assist_model", "models/gemma-4-E4B_q4_0-it.gguf",
    )
    if mode == "coding":
        return model_paths.get("assist_coding_model") or assist_model
    return assist_model


def get_mode_lora_path(mode: str) -> Path:
    """指定モードで使用すべき base LoRA アダプタの絶対パスを解決する (存在確認はしない)。

    ``learning.level2_adapter_partition=="model"`` (既定) のときは mode に依らず
    常に同一パスを返す — ``/api/mode/switch`` の再起動判定がこの関数の戻り値を
    比較するだけで LoRA 差分による再起動要否を導けるようにするため
    (既定運用では常に等しい = 再起動トリガーなし、後方互換)。
    "model_mode" のときのみ mode 別の実際のパスを返す。

    Args:
        mode: "chat" または "coding"

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
        mode: "chat" または "coding"

    Returns:
        解決された絶対パス (ファイルの存在は保証しない)
    """
    resolver = get_path_resolver()
    if resolver.adapter_partition_mode != "model_mode":
        return resolver.resolve_local("assist_lora_adapter")
    return resolver.resolve_assist_learning("assist_lora_adapter", mode=mode)
