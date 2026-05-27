"""FastAPI アプリケーションファクトリ (薄いファサード)

`backend.main` (uvicorn エントリ) からは :func:`create_app` のみを呼ぶ。

縮退した。実体は以下のサブモジュールに分離されている:

- :mod:`backend.factory._bootstrap`   — config / logging / i18n / local_dirs
- :mod:`backend.factory._memory_init` — EvorefMem 初期化 + 起動時 SemMem bootstrap
- :mod:`backend.factory._pillar_wirer` — pillar 配線 + 個別 ``_init_*``
- :mod:`backend.factory._lifespan`    — shutdown 系 + ``lifespan``
- :mod:`backend.factory._health`      — CORS / Free・Pro プラグイン登録 / health

レイヤー責務:

- ``backend.main``        — uvicorn ターゲット (``app = create_app()`` のみ)
- ``backend.app_factory`` — ``create_app()`` ファサード (この層)
- ``backend.factory.*``   — 起動 / シャットダウン / pillar 配線の実体
- ``backend/free/api/*``  — エンドポイント実装 (FastAPI ルータ)
"""

from __future__ import annotations

from fastapi import FastAPI

from backend.error_handlers import register_exception_handlers
from backend.factory._health import (
    _compose_app_version,
    _register_cors,
    _register_health_route,
    _register_plugins,
    _register_version_route,
)
from backend.factory._lifespan import lifespan


def create_app() -> FastAPI:
    """FastAPI アプリケーションを構築して返す。

    `backend.main` から uvicorn 用のグローバル `app` インスタンス生成のため、
    およびテスト等から個別のアプリインスタンスを得るために呼び出される。

    本関数は副作用として:
    - lifespan ハンドラを束ねた FastAPI を生成
    - CORS / エラーハンドラ / プラグイン (Free + Pro) / health を登録

    なお `_init_*` 系の起動処理は lifespan 起動時に走るため、`create_app()`
    自体は I/O や重い初期化を行わない (CORS の frontend_port 解決のみ
    `get_config()` をベストエフォートで参照する)。
    """
    app = FastAPI(
        title="evoref",
        # 仮のバージョン。プラグイン登録後に Free/Pro 解決値で上書きする
        version="0.0.0",
        lifespan=lifespan,
    )
    _register_cors(app)
    register_exception_handlers(app)
    _register_plugins(app)
    # Pro プラグイン登録後にバージョン情報を解決し、FastAPI と app.state の
    # 双方に反映する。current_edition() は setup_pro() で _pro_handlers が
    # 設定されて初めて PRO を返すため、必ず _register_plugins の後で呼ぶ。
    from backend.version import get_version_info
    app.state.version = get_version_info()
    app.version = _compose_app_version()
    _register_health_route(app)
    _register_version_route(app)
    return app
