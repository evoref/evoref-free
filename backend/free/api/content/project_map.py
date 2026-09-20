"""ProjectMap (c_16 §4.4) の手動トリガと状態表示 API

書き手は変わらず :func:`~backend.free.memory.sleep.project_map.update_project_map`
(sleep-time Step 5.87) だけ — この router はその関数を呼ぶか、版ディレクトリを
読むだけの薄い層。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.config import get_config, get_path_resolver
from backend.free.api.content._rag_helpers import rag_error
from backend.free.rag.corpus import (
    PACKAGES_DIR,
    CorpusManifest,
    PackageMeta,
    meta_from_record,
)
from backend.free.rag.projectmap import project_map_package_id
from backend.free.rag.projectmap.fingerprint import load_fingerprint_store
from backend.log_config import get_logger
from backend.utils import format_utc

logger = get_logger("api.project_map")

router = APIRouter(prefix="/api/rag/project_map", tags=["rag"])


def _pm_config(config: dict) -> dict:
    """``rag.project_map`` を読む (sleep-time Step 5.87 と同じ規則)。"""
    return ((config.get("rag") or {}).get("project_map") or {})


def _version_written_at(directory: Path) -> str | None:
    """版ディレクトリの更新時刻 (ISO 8601 UTC)。取れなければ ``None``。"""
    try:
        stamp = directory.stat().st_mtime
    except OSError:
        return None
    return format_utc(datetime.fromtimestamp(stamp, tz=UTC))


def _read_package_meta(directory: Path) -> PackageMeta | None:
    """版ディレクトリの ``package.json`` を読む (壊れていれば ``None``)。"""
    path = directory / "package.json"
    try:
        return meta_from_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        logger.warning("project_map: unreadable package.json at %s: %s", directory, e)
        return None


def _root_status(
    root_rel: str, root_path: Path, corpus_dir: Path, manifest: CorpusManifest,
) -> dict[str, Any]:
    """1 root 分の現在版の状態 (c_16 §4.4)。"""
    package_id = project_map_package_id(root_path)
    version = manifest.active.get(package_id)
    update_kind: str | None = None
    languages: dict[str, int] = {}
    nodes = 0
    edges = 0
    written_at: str | None = None
    if version:
        directory = corpus_dir / PACKAGES_DIR / package_id / version
        meta = _read_package_meta(directory)
        if meta is not None:
            update_kind = str(meta._extra.get("update_kind") or "") or None
            raw_languages = meta._extra.get("languages")
            if isinstance(raw_languages, dict):
                languages = {str(k): int(v) for k, v in raw_languages.items()}
        fp_store = load_fingerprint_store(directory)
        nodes = fp_store.node_count
        edges = fp_store.edge_count
        written_at = _version_written_at(directory)
    return {
        "root": root_rel,
        "package_id": package_id,
        "version": version,
        "update_kind": update_kind,
        "languages": languages,
        "nodes": nodes,
        "edges": edges,
        "written_at": written_at,
    }


@router.get("")
async def get_project_map_status() -> dict[str, Any]:
    """ProjectMap の設定状態と root ごとの現在版 (c_16 §4.4)。

    版ディレクトリを直接読む (sleep-time Step 5.87 と同じ経路) — corpus の
    実行時ロード (``CartridgeManager``) は埋め込みバックエンドを要るため、
    状態表示だけのこのエンドポイントでは使わない。
    """
    config = get_config()
    cfg = _pm_config(config)
    resolver = get_path_resolver()
    corpus_dir = resolver.resolve_corpus_dir()
    manifest = CorpusManifest(corpus_dir)
    manifest.load()
    roots = cfg.get("roots") or ["."]
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "roots": [
            _root_status(root_rel, (resolver.root / root_rel).resolve(), corpus_dir, manifest)
            for root_rel in roots
        ],
    }


@router.post("/update")
async def trigger_project_map_update(
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    """ProjectMap を手動で更新する (c_16 §4.4)。

    走査 + tree-sitter 抽出は数分掛かりうるため、reindex と同様に同期で待つ。
    Step 5.87 (Full サイクル) と同時に同じ root へ版を書かないよう、既に
    実行中なら 409 で拒否する (``update_project_map`` 自体もロックするが、
    ロック待ちで数分ブロックしないよう先に確認する)。
    """
    from backend.free.memory.sleep.project_map import (
        is_project_map_update_running,
        update_project_map,
    )

    config = get_config()
    cfg = _pm_config(config)
    if not bool(cfg.get("enabled", True)):
        raise rag_error(
            400, "E0400", "ProjectMap is disabled",
            "api.project_map_disabled",
        )
    if is_project_map_update_running():
        raise rag_error(
            409, "E0409", "ProjectMap update already running",
            "api.project_map_update_running",
        )

    resolver = get_path_resolver()
    corpus = getattr(state.cartridge_manager, "corpus", None)
    language_overlay = corpus.language_overlay() if corpus is not None else None
    updated = await update_project_map(
        config=config, resolver=resolver, language_overlay=language_overlay,
    )
    logger.info("Manual ProjectMap update: %d root(s) updated", updated)
    return {"updated_roots": updated, "running": False}


__all__ = ["router"]
