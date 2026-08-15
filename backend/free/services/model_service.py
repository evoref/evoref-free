"""モデル情報・ヘルスチェック統合サービス

config.yaml と各ポートのヘルスチェックからモデル情報一覧を構築する。
CLI (main.py, command_handlers.py) と GUI から共通利用される。
sync / async 両対応。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.config import resolve_context_size
from backend.log_config import get_logger

logger = get_logger("services.model_service")


@dataclass
class ModelInfoItem:
    """モデル情報の1行分"""
    label: str
    name: str
    connected: bool


def build_model_info_sync(
    project_root: Path,
    status_data: dict | None,
) -> tuple[list[ModelInfoItem], int | None]:
    """config.yaml + /api/status からモデル情報一覧を構築（同期版）"""
    import yaml

    from backend.free.services.health_service import check_health_sync

    models: list[ModelInfoItem] = []
    context_size: int | None = None

    try:
        cfg = yaml.safe_load(
            (project_root / "config.yaml").read_text(encoding="utf-8"),
        )
    except Exception:
        return models, context_size

    sp = cfg.get("model_paths", {})
    context_size = resolve_context_size(cfg, "base")

    # llama-server 接続状態
    llama_connected = False
    if status_data:
        ls = status_data.get("llama_server", {})
        llama_connected = ls.get("connected", False)

    # ベースモデル
    base = sp.get("base_model") or ""
    if base:
        models.append(ModelInfoItem(
            label="base",
            name=Path(base).stem,
            connected=llama_connected,
        ))

    # 埋め込みモデル
    embed = sp.get("embed_model") or ""
    embed_cfg = cfg.get("embedding", {})
    if embed or embed_cfg.get("backend"):
        if embed:
            embed_port = embed_cfg.get("port", 8082)
            embed_connected = check_health_sync("127.0.0.1", embed_port)
            models.append(ModelInfoItem(
                label="embed",
                name=Path(embed).stem,
                connected=embed_connected,
            ))

    return models, context_size


async def build_model_info_async(
    project_root: Path,
    status_data: dict | None,
) -> tuple[list[ModelInfoItem], int | None]:
    """config.yaml + 各ポートの非同期ヘルスチェックからモデル情報を構築（非同期版）"""
    import yaml

    from backend.free.services.health_service import check_health_async

    models: list[ModelInfoItem] = []
    context_size: int | None = None

    try:
        cfg = yaml.safe_load(
            (project_root / "config.yaml").read_text(encoding="utf-8"),
        )
    except Exception:
        return models, context_size

    sp = cfg.get("model_paths", {})
    context_size = resolve_context_size(cfg, "base")

    # llama-server 接続状態
    llama_connected = False
    if status_data:
        ls = status_data.get("llama_server", {})
        llama_connected = ls.get("connected", False)

    # ベースモデル
    base = sp.get("base_model") or ""
    if base:
        models.append(ModelInfoItem(
            label="base",
            name=Path(base).stem,
            connected=llama_connected,
        ))

    # 埋め込みモデル
    embed = sp.get("embed_model") or ""
    embed_cfg = cfg.get("embedding", {})
    if embed or embed_cfg.get("backend"):
        if embed:
            embed_port = embed_cfg.get("port", 8082)
            embed_connected = await check_health_async("127.0.0.1", embed_port)
            models.append(ModelInfoItem(
                label="embed",
                name=Path(embed).stem,
                connected=embed_connected,
            ))

    return models, context_size
