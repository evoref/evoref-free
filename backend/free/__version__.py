"""Free エディションのバージョン情報

Free と Pro のバージョンは独立して管理する。
本ファイルは Free 配布物に必ず含まれる。

- `__version__`: Free 配布のセマンティックバージョン
- `DATA_GENERATION`: Free が書く形式 (Free と Pro の共有形式) のデータ世代。
  リリース用の定数でディスクには持たない (真実は形式ごとの版、c_05 §0.4)。
  共有形式の版を上げたら 1 上げ、Free と Pro を同日に出す。
  ``python -m backend.formats --check`` が形式台帳の lock と突き合わせる
- `__build__`: ビルド時にスクリプトが書き換える任意フィールド
"""

from __future__ import annotations

__version__ = "0.0.100"
DATA_GENERATION = 1
__build__ = "dev"
