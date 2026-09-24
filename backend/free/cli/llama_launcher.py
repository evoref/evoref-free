"""evoref-ctl の llama-server 起動口 (``python -m backend.free.cli.llama_launcher``)。

``scripts/launch_llama.py`` はアダプタを config から解決しない。base を起こすときは
ここで :func:`adapters_for_launch` (Free では常に無し) から ``--lora`` /
``--control-vector`` を得て、引数に足してから同じスクリプトを実行する。
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from backend.free.core.launch_adapters import adapters_for_launch

_LAUNCH_PY = Path(__file__).resolve().parents[3] / "scripts" / "launch_llama.py"
#: base を起こさない (アダプタが要らない) 呼び方。
_NO_BASE_FLAGS = frozenset({"--embed", "--print-health-ports", "--wait-health"})


def _launch_args(argv: list[str]) -> list[str]:
    """``argv`` に base のアダプタを足した引数列。"""
    if any(a.split("=", 1)[0] in _NO_BASE_FLAGS for a in argv):
        return argv
    config = next((a for a in argv if not a.startswith("-")), "config.yaml")
    cfg_path = Path(config).resolve()
    if not cfg_path.exists():
        return argv  # スクリプトが自分でエラーを出す
    from backend.config import load_config

    cfg = load_config(cfg_path, project_root=cfg_path.parent)
    return [*argv, *adapters_for_launch(cfg, cfg_path.parent, "chat").launch_args()]


def main(argv: list[str] | None = None) -> None:
    """アダプタを足して ``scripts/launch_llama.py`` を実行する (終了はスクリプトに任せる)。"""
    args = _launch_args(list(sys.argv[1:] if argv is None else argv))
    sys.argv = [str(_LAUNCH_PY), *args]
    runpy.run_path(str(_LAUNCH_PY), run_name="__main__")


if __name__ == "__main__":
    main()
