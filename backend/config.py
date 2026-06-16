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
        "assist_lora_adapter": "local/models/assist_adapter.gguf",
        "assist_lora_versions_dir": "local/models/assist_lora_versions/",
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
    }

    def __init__(self, config: dict, project_root: Path):
        self.root = project_root
        self.models = config.get("model_paths", {})
        self.local = config.get("local_paths", {})

    def resolve_model(self, key: str) -> Path:
        """モデルパス解決"""
        raw = self.models.get(key, self.MODEL_DEFAULTS[key])
        return self._to_absolute(raw)

    def resolve_local(self, key: str) -> Path:
        """ローカルパス解決（読み書きリソース）"""
        raw = self.local.get(key, self.LOCAL_DEFAULTS[key])
        return self._to_absolute(raw)

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
    except Exception as e:  # noqa: BLE001
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
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Invalid reasoning profile for slot %s (using defaults): %s",
                    slot, e,
                )
                reasoning = {}
    except Exception as e:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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
