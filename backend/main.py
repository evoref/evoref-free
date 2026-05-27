"""FastAPI uvicorn エントリーポイント

`uvicorn backend.main:app` から参照される最小エントリ。
アプリの組み立ては `backend.app_factory.create_app()` に委譲する。

レイヤー責務:
- 本モジュールは uvicorn ターゲットに必要な `app` シンボルだけを公開する
- ライフサイクル / 初期化 / プラグイン登録は `backend.app_factory` に集約
"""

from __future__ import annotations

from backend.app_factory import create_app

app = create_app()
