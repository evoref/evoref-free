"""config + ``/props`` メタデータから :class:`LocalClient` を組み立てる唯一の入口。

起動時 (``factory/_pillar_wirer._init_llama_server``) / 遅延再接続
(``api/system/status._try_lazy_connect``) / モデル切替 (``api/model/model.reload_model``)
の 3 箇所が別々に ``LocalClient(...)`` を組んでいて、後者 2 つは
``context_size`` (送信前コンテキスト超過ガード) と ``stream_first_token_timeout``
を渡さず、``slots`` も llama-server の ``total_slots`` へ丸めていなかった。
遅延接続 / 切替の後だけガードが無効化され、``id_slot=2`` がスロット数の少ない
サーバに 400 を返す状態になっていた (2026-09-11)。
"""

from __future__ import annotations

from typing import Any

from backend.config import (
    resolve_client_reasoning,
    resolve_context_size,
    resolve_enable_thinking,
)
from backend.free.llm.model_metadata import ModelMetadata
from backend.log_config import get_logger

logger = get_logger("llm.client_builder")


def build_local_client(
    cfg: dict[str, Any],
    llama_url: str,
    metadata: ModelMetadata,
    *,
    debug_logger=None,
):
    """``cfg["llama"]`` と ``metadata`` からベースモデルの :class:`LocalClient` を組む。

    ``enable_thinking`` / 暴走 reasoning watchdog / ``context_size`` は起動フラグと
    同じ優先順位 (``backend.config.resolve_*``) で解決する。``slots`` は config の
    宣言値と ``/props`` の ``total_slots`` の少ない方へ丸める。
    """
    # 関数内 import: 既存テストが ``backend.free.llm.local_client.LocalClient`` を
    # monkeypatch する (呼出時に解決させないと差し替えが効かない)。
    from backend.free.llm.local_client import LocalClient

    llama_cfg = cfg.get("llama", {})
    base_enable_thinking = resolve_enable_thinking(
        cfg, "base",
        explicit=llama_cfg.get("enable_thinking"),
        chat_template=getattr(metadata, "chat_template", None),
    )
    think_budget, on_runaway = resolve_client_reasoning(cfg, "base")
    # config の slots は宣言値。llama-server が実際に確保したスロット数
    # (``/props`` の total_slots) より多いと、``id_slot=2`` 等の要求が
    # 存在しないスロットを指して 400 になる。少ない方へ丸めて警告する。
    cfg_slots = int(llama_cfg.get("slots", 1) or 1)
    total_slots = int(getattr(metadata, "total_slots", 0) or 0)
    slots = cfg_slots
    if total_slots > 0 and total_slots != cfg_slots:
        slots = min(cfg_slots, total_slots)
        logger.warning(
            "llama.slots=%d does not match llama-server total_slots=%d; "
            "using %d (restart llama-server after changing config.yaml)",
            cfg_slots, total_slots, slots,
        )
    return LocalClient(
        llama_url,
        metadata,
        cache_prompt=llama_cfg.get("cache_prompt", True),
        slots=slots,
        enable_thinking=base_enable_thinking,
        stream_first_token_timeout=llama_cfg.get(
            "stream_first_token_timeout_sec", 60.0,
        ),
        debug_logger=debug_logger,
        client_think_budget=think_budget,
        on_runaway=on_runaway,
        # 送信前コンテキスト超過ガード用 (slots>1 でも launch_llama が
        # --kv-unified を自動付与するため per-slot でも full n_ctx)。
        # ``llama.context_size`` の既定は None (プロファイル委譲) なので
        # ``.get(..., 4096)`` では None が入ってガードが無効化されていた。
        # 起動フラグ ``-c`` と同じ優先順位で解決する。
        context_size=resolve_context_size(cfg, "base"),
    )
