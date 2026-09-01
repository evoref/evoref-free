"""Free エディションのバージョン情報

Free と Pro のバージョンは独立して管理する。
本ファイルは Free 配布物に必ず含まれる。

- `__version__`: Free 配布のセマンティックバージョン
- `__schema_version__`: データ互換性の軸 (config.yaml / RAG index / memory)
  Free と Pro で共通の値を持つこと。互換破壊変更時にインクリメントする
- `__build__`: ビルド時にスクリプトが書き換える任意フィールド
"""

from __future__ import annotations

__version__ = "0.0.66"
__schema_version__ = 1
__build__ = "dev"
