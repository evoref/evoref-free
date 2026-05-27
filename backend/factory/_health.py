"""FastAPI ファクトリ補助 (CORS / プラグイン / health / version)

含まれる関数:

- :func:`_resolve_frontend_port`  : `config.yaml` の frontend_port をベスト
  エフォートで参照
- :func:`_register_cors`          : フロントエンドポート向け CORS ミドル
  ウェアを登録
- :func:`_register_plugins`       : Free (必須) + Pro (任意) プラグイン
  登録 + Pro ルータ取り込み
- :func:`_compose_app_version`    : FastAPI ``version`` フィールドの組み立て
- :func:`_register_version_route` : ``/api/version`` 登録
- :func:`_register_health_route`  : ``/api/health`` 登録

純粋な move であり、関数本体・引数・default 値は変更していない。
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_config


def _resolve_frontend_port() -> int:
    """`config.yaml` が読み込み済みなら frontend_port を返し、未読込みなら 5173。"""
    try:
        cfg = get_config()
    except RuntimeError:
        return 5173
    return cfg.get("server", {}).get("frontend_port", 5173)


def _register_cors(app: FastAPI) -> None:
    """フロントエンドポート向け CORS ミドルウェアを登録。"""
    frontend_port = _resolve_frontend_port()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://localhost:{frontend_port}",
            f"http://127.0.0.1:{frontend_port}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _register_plugins(app: FastAPI) -> None:
    """Free (必須) + Pro / Develop (任意) プラグインの登録 + ルータ取り込み。

    エディション解決順序:
    - ``EVOREF_EDITION=free`` → Pro / Develop を両方無効化
    - ``EVOREF_EDITION=pro``  → Pro のみ有効化、Develop は無効化
    - ``EVOREF_EDITION=develop`` または未設定 → Pro と Develop の両方を
      パッケージ存在時に有効化 (Develop ⊇ Pro)
    """
    try:
        from backend.free import setup_free
        setup_free(app)
    except ImportError as e:
        raise RuntimeError("backend/free/ is required for evoref to run") from e

    edition_env = os.environ.get("EVOREF_EDITION", "").lower()

    # Pro: EVOREF_EDITION=free 以外で有効化試行
    if edition_env != "free":
        try:
            from backend.pro import setup_pro  # type: ignore[import-not-found]
            setup_pro(app)
        except ImportError:
            pass  # Free エディションには Pro パッケージ自体が無い

    # Develop: EVOREF_EDITION=develop または未設定でパッケージ存在時のみ
    # 有効化 (free / pro 明示時はスキップ)。
    if edition_env in ("", "develop"):
        try:
            from backend.develop import setup_develop  # type: ignore[import-not-found]
            setup_develop(app)
        except ImportError:
            pass  # Free / Pro 配布には Develop パッケージ自体が無い

    from backend.edition import get_develop_routers, get_pro_routers
    for pro_router in get_pro_routers():
        app.include_router(pro_router)
    for develop_router in get_develop_routers():
        app.include_router(develop_router)


def _compose_app_version() -> str:
    """FastAPI の `version` フィールド用の文字列を組み立てる。

    Free / Pro のバージョンを独立管理しつつ、OpenAPI / Swagger 上で
    全エディションを把握できる形 (`free=X / pro=Y`) で表現する。
    未同梱エディションは省略。
    """
    from backend.version import get_version_info
    info = get_version_info()
    parts = [f"free={info.free}"]
    if info.pro is not None:
        parts.append(f"pro={info.pro}")
    return " / ".join(parts)


def _register_version_route(app: FastAPI) -> None:
    """`/api/version` エンドポイントを登録。

    Free / Pro のバージョン情報を JSON で返す。Pro 未同梱時は
    `pro` / `pro_build` が `null` になる。
    """

    @app.get("/api/version")
    async def api_version():
        """Free / Pro バージョンとデータ互換性軸を返す"""
        v = app.state.version
        return v.to_dict()


def _register_health_route(app: FastAPI) -> None:
    """`/api/health` エンドポイントを登録。"""

    @app.get("/api/health")
    async def health():
        """ヘルスチェック"""
        from backend.i18n_helper import msg
        return {"status": msg("api.health_ok")}
