"""SemMem fact 埋め込みの stale 検知ガード

embed モデルを (同 dim でも) 別モデルへ切り替えると、既存 fact の embedding は
旧モデル空間のまま取り残され、``search_by_embedding`` (URL / コマンドリコール)
が非互換ベクトルで空振りする。RAG 側の ``.embed_reindex_required`` マーカー
(:mod:`backend.free.rag.dimension_check`) と対称に、SemMem 側にも reembed 要求
マーカーを置き、起動時 WARNING で ``evorefmem_cli reembed-facts`` を案内する。

RAG はベクトルが stale だと「誤った検索結果」を返すため search をブロックするが、
SemMem の stale は recall が「ヒットしない (= no-tool 縮退)」だけで誤結果には
ならないため、ここでは **WARNING のみ** (ブロックしない)。

マーカーは:

- :func:`set_semmem_reembed_required` — embed component-migrate 時に立てる
  (:mod:`backend.free.api.model.model`)。
- :func:`warn_if_semmem_reembed_required` — 起動時に存在を確認して WARN
  (:mod:`backend.factory._pillar_wirer`)。
- :func:`clear_semmem_reembed_required` — ``reembed-facts --apply`` 成功後に消す
  (:mod:`backend.free.memory.semantic.cli.reembed_facts_cmd`)。
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("memory.semantic.stale_guard")

#: embed モデル切替で SemMem fact 埋め込みが stale になったことを示すマーカー名。
#: ``<memory_dir>/semantic/`` 直下に置く。
_SEMMEM_REEMBED_MARKER = ".reembed_facts_required"


def _resolve_memory_dir() -> Path | None:
    """PathResolver 経由で ``memory_dir`` を解決する (未初期化時は ``None``)。"""
    try:
        from backend.config import get_path_resolver

        return get_path_resolver().resolve_local("memory_dir")
    except Exception:
        return None


def _marker_path(memory_dir: Path | None = None) -> Path | None:
    """マーカーの絶対パス。``memory_dir`` 明示時はそれを、None なら resolver。"""
    md = memory_dir if memory_dir is not None else _resolve_memory_dir()
    if md is None:
        return None
    return Path(md) / "semantic" / _SEMMEM_REEMBED_MARKER


def set_semmem_reembed_required(
    new_model: str, *, memory_dir: Path | None = None,
) -> None:
    """embed モデル変更により SemMem fact 埋め込みが stale になったと記録する。"""
    p = _marker_path(memory_dir)
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"new_model": new_model, "at": utc_now()}, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            "SemMem reembed marker set (embed model -> %s); fact vectors are "
            "stale. Click the Reembed button in the admin UI (POST "
            "/api/model/reembed-facts) or run 'python scripts/evorefmem_cli.py "
            "reembed-facts --apply' to rebuild URL/command recall vectors.",
            new_model,
        )
    except Exception as exc:
        logger.warning("Failed to write SemMem reembed marker: %s", exc)


def clear_semmem_reembed_required(*, memory_dir: Path | None = None) -> None:
    """SemMem reembed 要求マーカーを消す (reembed-facts 成功後に呼ぶ)。"""
    p = _marker_path(memory_dir)
    if p is None:
        return
    try:
        p.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to clear SemMem reembed marker: %s", exc)


def is_semmem_reembed_required(
    *, memory_dir: Path | None = None,
) -> dict | None:
    """マーカーが存在すれば中身の dict を、無ければ ``None`` を返す。

    マーカーは存在するが内容が壊れている場合は空 dict ``{}`` を返す
    (= 「要再 embed だが詳細不明」)。
    """
    p = _marker_path(memory_dir)
    if p is None or not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def warn_if_semmem_reembed_required(
    *, memory_dir: Path | None = None,
) -> bool:
    """マーカーがあれば WARNING を出して ``True`` を返す (ブロックしない)。"""
    info = is_semmem_reembed_required(memory_dir=memory_dir)
    if info is None:
        return False
    model = info.get("new_model", "?")
    logger.warning(
        "SemMem fact embeddings are STALE (embed model changed to %s). "
        "URL/command recall (search_by_embedding) will MISS until rebuilt. "
        "Click the Reembed button in the admin UI (POST /api/model/reembed-facts) "
        "or run 'python scripts/evorefmem_cli.py reembed-facts --apply'.",
        model,
    )
    return True


__all__ = [
    "clear_semmem_reembed_required",
    "is_semmem_reembed_required",
    "set_semmem_reembed_required",
    "warn_if_semmem_reembed_required",
]
