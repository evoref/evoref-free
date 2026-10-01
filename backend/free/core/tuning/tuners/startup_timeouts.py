"""項目 ``startup_timeouts`` — 起動の health 待ちの秒数 (c_16 §7.2.3「health 待ち」)。

base / 埋め込み / リランカーの GGUF サイズから、各経路が実際に使う待ち秒を導いて見せる。式は
:func:`backend.free.llm._base_client.resolve_health_wait` (15 秒/GB、上限 600) の 1 実装を呼ぶだけで、
ここには持たない。サーバを起こさず、ファイルサイズを読むだけ (停止中でも稼働中でも安全)。

反映は経路ごとに即時 (起動のたびにその場で導く)。保存結果は読まれず、画面と CLI の表示用。
``process_manager.health_timeout`` に整数を明示していれば、その値が全経路の本番待ちに使われる
(提案値は結果に残し、``manual`` で未適用と示す)。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from backend.free.core.tuning.base_model import base_model_path
from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, register
from backend.free.llm._base_client import HEALTH_WAIT_FLOOR_SEC, resolve_health_wait

KEY = "startup_timeouts"


def _size_gb(path: Path | None) -> float | None:
    try:
        return round(path.stat().st_size / (1024 ** 3), 2) if path is not None else None
    except OSError:
        return None


def decide_startup_timeouts(
    models: dict[str, Path | None], gpu_floors: dict[str, float], *, probe_floor: float,
) -> dict[str, Any]:
    """モデル別の待ち秒 (純関数、サイズが読めないモデルは現行の下限)。

    ``models`` は ``base`` / ``embed`` / ``rerank`` の GGUF パス。``gpu_floors`` は embed / rerank の
    GPU 起動待ちの下限、``probe_floor`` は埋め込みの判別プローブの下限 (起動スクリプトの定数)。
    """
    out: dict[str, Any] = {}
    for name, path in models.items():
        entry: dict[str, Any] = {
            "model_gb": _size_gb(path),
            "health_wait_sec": resolve_health_wait("auto", path, floor=HEALTH_WAIT_FLOOR_SEC, label=name),
        }
        if name in gpu_floors:
            entry["gpu_start_wait_sec"] = resolve_health_wait("auto", path, floor=gpu_floors[name], label=name)
        if name == "embed":
            entry["probe_wait_sec"] = resolve_health_wait("auto", path, floor=probe_floor, label=name)
        out[name] = entry
    return out


def run(ctx: TuneContext) -> TuneOutcome:
    """base / 埋め込み / リランカーのモデルサイズから各待ち秒を返す。"""
    try:
        ll = importlib.import_module("scripts.launch_llama")
    except ImportError:
        return TuneOutcome("failed", reason="launcher_unavailable", environmental=True)
    rerank_raw = (ctx.cfg.get("model_paths") or {}).get("rerank_model")
    models: dict[str, Path | None] = {
        "base": base_model_path(ctx.cfg, ctx.project_root),
        "embed": ll._embed_model_path(ctx.cfg, ctx.project_root),
    }
    if rerank_raw:
        models["rerank"] = ll.resolve_rerank_model_path(ctx.cfg, ctx.project_root)
    value = decide_startup_timeouts(
        models,
        {"embed": ll.EMBED_GPU_START_TIMEOUT_SEC, "rerank": ll.RERANK_GPU_START_TIMEOUT_SEC},
        probe_floor=ll.EMBED_PROBE_HEALTH_TIMEOUT_SEC,
    )
    return TuneOutcome("ok", value=value, reason="from_gguf_size")


SPEC = register(TuneSpec(
    key=KEY,
    config_key="process_manager.health_timeout",
    method="estimated",
    requires_restart=False,
    run=run,
    auto_values=("auto", None),
    config_default="auto",
    needs_servers=False,
    safe_while_running=True,  # ファイルサイズを読むだけ
    description="startup health-wait seconds derived from the GGUF sizes (15 s/GB, floor 120, cap 600), c_16 §7.2.3",
))
