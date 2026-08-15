"""起動シーケンスの前提条件チェック（設計書 14, 19.9）"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from backend.error_handlers import E1001
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.startup_checks")


class CheckLevel(Enum):
    """チェック結果のレベル"""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class CheckStatus(Enum):
    """チェック結果のステータス"""
    OK = "ok"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class CheckResult:
    """個別チェックの結果"""
    level: CheckLevel
    status: CheckStatus
    name: str
    message: str
    code: str | None = None


def run_serve_checks(
    project_root: Path,
    config: dict,
    *,
    skip_llama_check: bool = False,
    skip_port_check: bool = False,
) -> list[CheckResult]:
    """evoref serve 用の全チェックを実行

    Args:
        project_root: プロジェクトルート
        config: 読込み済み config.yaml の辞書
        skip_llama_check: llama.cpp 接続チェックをスキップ（serve が起動するため）
        skip_port_check: ポート競合チェックをスキップ（--force 時）

    Returns:
        チェック結果のリスト
    """
    results: list[CheckResult] = []

    # 1. config.yaml は呼び出し元で既に読込み済み（ここでは構文検証済みとして扱う）
    # → 呼び出し元の validate_config() で CRITICAL チェック済み

    # 2. [WARNING] local_paths ディレクトリの存在確認 → 自動作成
    results.extend(_check_local_paths(project_root, config))

    # 3. [WARNING] model_paths の存在確認 → 警告のみ
    results.extend(_check_model_paths(project_root, config))

    # 4. [WARNING] llama.cpp 接続確認（--no-llama 時のみ意味がある）
    if not skip_llama_check:
        results.append(_check_llama_connection(config))

    # 5. [WARNING] LoRA アダプタの存在確認
    results.append(_check_lora_adapter(project_root, config))

    # 6. [INFO] ベクトルインデックスの存在確認
    results.append(_check_vector_index(project_root, config))

    # 6b. [WARNING] ベクトルインデックス次元整合性
    dim_result = _check_vector_dim_consistency(project_root, config)
    if dim_result is not None:
        results.append(dim_result)

    # 7. [INFO] メモリファイルの存在確認
    results.append(_check_memory_files(project_root, config))

    # 8. [WARNING] ポート競合チェック
    if not skip_port_check:
        results.extend(_check_port_conflicts(config))

    # 9. [WARNING] model_state.json と config.yaml.base_model の整合性
    results.append(_check_model_state_consistency(project_root, config))

    return results


def validate_config(project_root: Path) -> tuple[dict | None, CheckResult]:
    """config.yaml の読込みと構文検証

    Returns:
        (config 辞書 or None, チェック結果)
    """
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        logger.error("config.yaml not found at %s", config_path)
        return None, CheckResult(
            level=CheckLevel.CRITICAL,
            status=CheckStatus.FAIL,
            name="config.yaml",
            message=msg("cli.startup_config_missing", path=str(config_path)),
        )

    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("config.yaml is not a valid YAML mapping")
    except Exception as e:
        logger.error("config.yaml parse error: %s", e)
        return None, CheckResult(
            level=CheckLevel.CRITICAL,
            status=CheckStatus.FAIL,
            name="config.yaml",
            message=msg("cli.startup_config_invalid", detail=str(e)),
        )

    logger.debug("config.yaml loaded successfully")
    return config, CheckResult(
        level=CheckLevel.CRITICAL,
        status=CheckStatus.OK,
        name="config.yaml",
        message=msg("cli.startup_config_ok"),
    )


def run_interactive_checks(project_root: Path) -> list[CheckResult]:
    """対話モード用の簡易チェック（config.yaml のみ）

    Returns:
        チェック結果のリスト
    """
    results: list[CheckResult] = []
    config, config_result = validate_config(project_root)
    results.append(config_result)
    return results


def has_critical_failure(results: list[CheckResult]) -> bool:
    """CRITICAL レベルの失敗があるか"""
    return any(
        r.level == CheckLevel.CRITICAL and r.status == CheckStatus.FAIL
        for r in results
    )


def _resolve_path(project_root: Path, path_str: str) -> Path:
    """設定パスを解決"""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def _check_local_paths(project_root: Path, config: dict) -> list[CheckResult]:
    """local_paths の各ディレクトリ存在確認 → 自動作成"""
    results: list[CheckResult] = []
    local_paths = config.get("local_paths", {})

    for key, path_str in local_paths.items():
        path = _resolve_path(project_root, path_str)

        # ファイルパスの場合はディレクトリ部分を確認
        if key.endswith("_file"):
            target_dir = path.parent
        elif key.endswith("_dir"):
            target_dir = path
        else:
            # adapter 等: ファイルの親ディレクトリ
            target_dir = path.parent

        if not target_dir.exists():
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory: %s", target_dir)
                results.append(CheckResult(
                    level=CheckLevel.WARNING,
                    status=CheckStatus.WARN,
                    name=f"local:{key}",
                    message=msg("cli.startup_dir_created", path=str(target_dir)),
                ))
            except OSError as e:
                logger.error("Failed to create directory %s: %s", target_dir, e)
                results.append(CheckResult(
                    level=CheckLevel.WARNING,
                    status=CheckStatus.FAIL,
                    name=f"local:{key}",
                    message=msg("cli.startup_dir_create_failed", path=str(target_dir)),
                ))

    if not results:
        results.append(CheckResult(
            level=CheckLevel.WARNING,
            status=CheckStatus.OK,
            name="local_paths",
            message=msg("cli.startup_local_paths_ok"),
        ))

    return results


def _check_model_paths(project_root: Path, config: dict) -> list[CheckResult]:
    """model_paths の各パス存在確認 → 警告のみ"""
    results: list[CheckResult] = []
    model_paths = config.get("model_paths", {})

    for key, path_str in model_paths.items():
        path = _resolve_path(project_root, path_str)

        # ディレクトリ型キーの存在確認
        if key.endswith("_dir"):
            if not path.exists():
                logger.warning("Model path not found: %s", path)
                results.append(CheckResult(
                    level=CheckLevel.WARNING,
                    status=CheckStatus.WARN,
                    name=f"model:{key}",
                    message=msg("cli.startup_model_path_missing", path=str(path)),
                ))

    if not results:
        results.append(CheckResult(
            level=CheckLevel.WARNING,
            status=CheckStatus.OK,
            name="model_paths",
            message=msg("cli.startup_model_paths_ok"),
        ))

    return results


def _check_llama_connection(config: dict) -> CheckResult:
    """llama.cpp サーバーへの接続確認（同期版）"""
    llama_cfg = config.get("llama", {})
    host = llama_cfg.get("host", "localhost")
    port = llama_cfg.get("port", 8080)

    try:
        import httpx
        resp = httpx.get(f"http://{host}:{port}/health", timeout=3.0)
        if resp.status_code == 200:
            logger.debug("llama.cpp server is reachable at %s:%d", host, port)
            return CheckResult(
                level=CheckLevel.WARNING,
                status=CheckStatus.OK,
                name="llama.cpp",
                message=msg("cli.startup_llama_ok"),
            )
    except Exception:
        pass

    logger.warning("llama.cpp server not reachable at %s:%d", host, port)
    return CheckResult(
        level=CheckLevel.WARNING,
        status=CheckStatus.WARN,
        name="llama.cpp",
        message=msg("cli.startup_llama_unreachable", host=host, port=port),
        code=E1001,
    )


def _check_lora_adapter(project_root: Path, config: dict) -> CheckResult:
    """LoRA アダプタの存在確認"""
    local_paths = config.get("local_paths", {})
    lora_path_str = local_paths.get("lora_adapter", "local/models/adapter.gguf")
    lora_path = _resolve_path(project_root, lora_path_str)

    if not lora_path.exists():
        logger.info("LoRA adapter not found: %s (starting without LoRA)", lora_path)
        return CheckResult(
            level=CheckLevel.WARNING,
            status=CheckStatus.WARN,
            name="LoRA adapter",
            message=msg("cli.startup_lora_missing"),
        )

    logger.debug("LoRA adapter found: %s", lora_path)
    return CheckResult(
        level=CheckLevel.WARNING,
        status=CheckStatus.OK,
        name="LoRA adapter",
        message=msg("cli.startup_lora_ok"),
    )


def _check_vector_index(project_root: Path, config: dict) -> CheckResult:
    """ベクトルインデックスの存在確認"""
    local_paths = config.get("local_paths", {})
    vectors_dir_str = local_paths.get("vectors_dir", "local/vectors/")
    vectors_dir = _resolve_path(project_root, vectors_dir_str)
    index_file = vectors_dir / "index_q8.npy"

    if not index_file.exists():
        logger.info("Vector index not found: %s (will be created on first use)", index_file)
        return CheckResult(
            level=CheckLevel.INFO,
            status=CheckStatus.WARN,
            name="vector index",
            message=msg("cli.startup_vector_empty"),
        )

    logger.debug("Vector index found: %s", index_file)
    return CheckResult(
        level=CheckLevel.INFO,
        status=CheckStatus.OK,
        name="vector index",
        message=msg("cli.startup_vector_ok"),
    )


def _check_vector_dim_consistency(
    project_root: Path, config: dict,
) -> CheckResult | None:
    """ベクトルインデックスと設定の埋め込み次元の整合性チェック

    metadata.json の `_store_info.embedding_dim`（なければ index_q8.npy の
    shape[1]）を読み、`embedding.dim` と比較する。
    インデックスが空のときは None を返してチェック自体を省略する。
    """
    local_paths = config.get("local_paths", {})
    vectors_dir_str = local_paths.get("vectors_dir", "local/vectors/")
    vectors_dir = _resolve_path(project_root, vectors_dir_str)
    metadata_file = vectors_dir / "metadata.json"
    index_file = vectors_dir / "index_q8.npy"

    if not index_file.exists():
        return None

    embedding_cfg = config.get("embedding", {})
    config_dim = int(embedding_cfg.get("dim", 1024))

    stored_dim: int | None = None
    if metadata_file.exists():
        try:
            import json
            with open(metadata_file, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list) and raw and isinstance(raw[0], dict) \
                    and raw[0].get("_store_info") is True:
                stored_dim = int(raw[0].get("embedding_dim", 0)) or None
        except Exception as e:
            logger.warning("Failed to parse metadata.json for dim check: %s", e)

    if stored_dim is None:
        try:
            import numpy as np
            arr = np.load(str(index_file), mmap_mode="r")
            if arr.ndim == 2:
                stored_dim = int(arr.shape[1])
        except Exception as e:
            logger.warning("Failed to read index_q8.npy for dim check: %s", e)
            return None

    if stored_dim is None:
        return None

    if stored_dim != config_dim:
        logger.warning(
            "Vector dim mismatch: stored=%d, config=%d", stored_dim, config_dim,
        )
        return CheckResult(
            level=CheckLevel.WARNING,
            status=CheckStatus.WARN,
            name="vector dimension",
            message=msg(
                "cli.startup_vector_dim_mismatch",
                stored=stored_dim, current=config_dim,
            ),
        )

    return CheckResult(
        level=CheckLevel.INFO,
        status=CheckStatus.OK,
        name="vector dimension",
        message=msg("cli.startup_vector_dim_ok", dim=config_dim),
    )


def _check_memory_files(project_root: Path, config: dict) -> CheckResult:
    """メモリファイルの存在確認"""
    local_paths = config.get("local_paths", {})
    memory_dir_str = local_paths.get("memory_dir", "local/memory/")
    memory_dir = _resolve_path(project_root, memory_dir_str)

    if not memory_dir.exists() or not any(memory_dir.iterdir()):
        logger.info("Memory files not found in %s (will be created on first use)", memory_dir)
        return CheckResult(
            level=CheckLevel.INFO,
            status=CheckStatus.WARN,
            name="memory files",
            message=msg("cli.startup_memory_empty"),
        )

    logger.debug("Memory files found in %s", memory_dir)
    return CheckResult(
        level=CheckLevel.INFO,
        status=CheckStatus.OK,
        name="memory files",
        message=msg("cli.startup_memory_ok"),
    )


def _check_model_state_consistency(
    project_root: Path, config: dict,
) -> CheckResult:
    """model_state.json と config.yaml の整合性確認

    不一致が検出された場合は、/migrate-model の利用を CLI で明示的に通知する。
    """
    local_paths = config.get("local_paths", {})
    state_file_str = local_paths.get("model_state_file", "local/model_state.json")
    state_path = _resolve_path(project_root, state_file_str)

    if not state_path.exists():
        return CheckResult(
            level=CheckLevel.INFO,
            status=CheckStatus.OK,
            name="model_state",
            message=msg("cli.startup_model_state_init"),
        )

    try:
        from backend.free.core.model_migration import ModelState, detect_mismatches
        ms = ModelState(state_path)
        mismatches = detect_mismatches(ms, config)
    except Exception as e:
        logger.warning("Failed to read model_state.json: %s", e)
        return CheckResult(
            level=CheckLevel.WARNING,
            status=CheckStatus.WARN,
            name="model_state",
            message=msg("cli.startup_model_state_read_failed", detail=str(e)),
        )

    if not mismatches:
        return CheckResult(
            level=CheckLevel.INFO,
            status=CheckStatus.OK,
            name="model_state",
            message=msg("cli.startup_model_state_ok"),
        )

    # base mismatch を優先表示 (ユーザが起動時に見る主エラー)
    base_mm = mismatches.get("base_model")
    if base_mm:
        config_base_model = config.get("model_paths", {}).get("base_model") or ""
        logger.error(
            "Model mismatch: model_state.json=%s vs config.yaml=%s",
            base_mm["model_state"], base_mm["config"],
        )
        return CheckResult(
            level=CheckLevel.WARNING,
            status=CheckStatus.WARN,
            name="model_state",
            message=msg(
                "cli.startup_model_state_mismatch",
                current=base_mm["model_state"],
                config=base_mm["config"],
                new_model=config_base_model,
            ),
        )

    # component (embed) のみの不一致
    keys = ", ".join(sorted(mismatches))
    logger.error("Component model mismatch: %s", keys)
    return CheckResult(
        level=CheckLevel.WARNING,
        status=CheckStatus.WARN,
        name="model_state",
        message=msg("cli.startup_component_model_state_mismatch", keys=keys),
    )


def _check_port_conflicts(config: dict) -> list[CheckResult]:
    """起動に必要なポートが既に使用中でないか確認"""
    from backend.free.cli.pid_manager import (
        collect_configured_ports,
        find_port_occupants,
    )

    ports = collect_configured_ports(config)
    occupants = find_port_occupants(ports)

    if not occupants:
        return [CheckResult(
            level=CheckLevel.WARNING,
            status=CheckStatus.OK,
            name="ports",
            message=msg("cli.startup_ports_ok"),
        )]

    results: list[CheckResult] = []
    for occ in occupants:
        logger.warning(
            "Port %d is occupied by PID %d (%s)",
            occ.port, occ.pid, occ.process_name,
        )
        results.append(CheckResult(
            level=CheckLevel.WARNING,
            status=CheckStatus.WARN,
            name=f"port:{occ.port}",
            message=msg(
                "cli.startup_port_occupied",
                port=occ.port,
                pid=occ.pid,
                name=occ.process_name or "unknown",
            ),
        ))
    return results
