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
        "base_model": "models/qwen3.5-4b-q4_k_m.gguf",
    }
    LOCAL_DEFAULTS = {
        "lora_adapter": "local/models/adapter.gguf",
        "lora_versions_dir": "local/lora_versions/",
        "assist_lora_adapter": "local/models/assist_adapter.gguf",
        "assist_lora_versions_dir": "local/models/assist_lora_versions/",
        "experience_assist_file": "local/experience_assist.json",
        "eval_assist_file": "local/eval_assist.json",
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
    params = {**defaults[mode], **mode_cfg}

    # ベースモデル。chat は常にここを採用。
    # coding は model_paths.coding_model 指定が無い/空の場合のみフォールバック。
    model_paths = cfg.get("model_paths", {})
    base_model = model_paths.get("base_model", "models/qwen3.5-4b-q4_k_m.gguf")
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
