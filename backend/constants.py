"""全 pillar から参照できる共有定数 (横断基盤)。

値の意味が 1 つに定まるものだけを置く。モジュール固有の閾値・タイムアウトは
各モジュールの先頭で名前を付ける。
"""

from __future__ import annotations

SECONDS_PER_HOUR: float = 3600.0
SECONDS_PER_DAY: float = 86400.0

#: 除算ゼロ除け・ノルム 0 判定に使う数値誤差の下限。
FLOAT_EPS: float = 1e-9

#: ベース LLM (llama-server) の既定ポート。実際の値は ``llama.port`` (config) が正。
DEFAULT_LLAMA_PORT: int = 8080
