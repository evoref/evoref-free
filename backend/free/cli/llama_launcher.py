"""evoref-ctl の llama-server 起動口 (``python -m backend.free.cli.llama_launcher``)。

引数をそのまま ``scripts/launch_llama.py`` に渡して実行する。以前はここで学習済み
アダプタ (Pro の ``--lora`` / ``--control-vector``) を足していたが、Pro Level 2 の
撤去 (2026-10-10、f_04 §6.7) で足すものは無くなった。evoref-ctl の起動口として残す。
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_LAUNCH_PY = Path(__file__).resolve().parents[3] / "scripts" / "launch_llama.py"


def main(argv: list[str] | None = None) -> None:
    """``scripts/launch_llama.py`` を実行する (終了はスクリプトに任せる)。"""
    args = list(sys.argv[1:] if argv is None else argv)
    sys.argv = [str(_LAUNCH_PY), *args]
    runpy.run_path(str(_LAUNCH_PY), run_name="__main__")


if __name__ == "__main__":
    main()
