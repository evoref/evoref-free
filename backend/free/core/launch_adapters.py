"""base llama-server に渡すアダプタ (``--lora`` / ``--control-vector``) の取得口。

アダプタは Pro の学習データ (``store/pro/``) なので、パスは Pro が登録する
ハンドラ ``adapter_paths_for_launch`` からだけ得る (c_05 §0.4.2)。Free では
ハンドラが無く、アダプタ無しで起動する。

backend プロセスでは ``setup_pro`` がハンドラを登録する。CLI プロセス
(``evoref serve`` / evoref-ctl の起動) は FastAPI の Pro 初期化を通らないので、
Pro が同梱されていれば ``backend.pro.adapters`` の import で登録する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.edition import get_pro_handler
from backend.log_config import get_logger

logger = get_logger("launch_adapters")

HANDLER_NAME = "adapter_paths_for_launch"


@dataclass(frozen=True, slots=True)
class LaunchAdapters:
    """起動に使うアダプタ (存在と互換を確認済み)。無ければ ``None``。"""

    lora: Path | None = None
    control_vector: Path | None = None

    def launch_args(self) -> list[str]:
        """``scripts/launch_llama.py`` の引数列。"""
        args: list[str] = []
        if self.lora is not None:
            args += ["--lora", str(self.lora)]
        if self.control_vector is not None:
            args += ["--control-vector", str(self.control_vector)]
        return args


NO_ADAPTERS = LaunchAdapters()


def _handler():
    handler = get_pro_handler(HANDLER_NAME)
    if handler is not None:
        return handler
    from backend.free.cli.cli_mode import is_cli_pro_edition

    if not is_cli_pro_edition():
        return None
    try:
        import backend.pro.adapters  # noqa: F401  (import でハンドラを登録する)
    except ImportError:
        return None
    return get_pro_handler(HANDLER_NAME)


def adapters_for_launch(cfg: dict, project_root: Path, mode: str = "chat") -> LaunchAdapters:
    """``mode`` で base llama-server に当てるアダプタを返す (Free では常に無し)。"""
    handler = _handler()
    if handler is None:
        return NO_ADAPTERS
    return handler(cfg, project_root, mode)


__all__ = ["HANDLER_NAME", "NO_ADAPTERS", "LaunchAdapters", "adapters_for_launch"]
