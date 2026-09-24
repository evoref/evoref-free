"""API の到達範囲を絞る ASGI ミドルウェア (c_06 §1.5)

backend は利用者 1 人のローカルアプリで、認証を持たない。そのため「誰が叩けるか」を
接続の形で絞る:

- **Host 許可リスト**: ``localhost`` / ``127.0.0.1`` / ``[::1]`` に、実際に待ち受けている
  ポートか frontend のポートを付けたものだけを通す (DNS rebinding 対策)。
- **状態を変えるメソッドは ``X-Evoref-Client`` ヘッダ必須**: ブラウザの単純リクエスト
  (``<form>`` の POST 等) では付けられないので、他サイトからの CSRF は CORS preflight に
  掛かって止まる。
- **Origin**: 付いていれば許可リスト (frontend の origin) と照合する。``Origin: null`` は拒否。
- ``server.allow_remote: true`` のときは Host / Origin を問わず、全リクエストで
  ``X-Evoref-Token`` を照合する (LAN から使う構成)。

BaseHTTPMiddleware は SSE の切断・キャンセルの伝播を壊すので、素の ASGI で書く。
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("factory.access_guard")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

CLIENT_HEADER = "x-evoref-client"
TOKEN_HEADER = "x-evoref-token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")
#: テスト環境などで追加の Host を許すための環境変数 (カンマ区切り)。同じ OS ユーザーの
#: プロセスは脅威モデルの外 (c_06 §1.2) なので、環境変数で広げられてよい。
EXTRA_HOSTS_ENV = "EVOREF_EXTRA_ALLOWED_HOSTS"
TOKEN_ENV = "EVOREF_API_TOKEN"


@dataclass(frozen=True)
class GuardSettings:
    frontend_port: int = 5173
    token: str | None = None
    extra_hosts: frozenset[str] = frozenset()


class AccessGuard:
    """Host / Origin / クライアントヘッダ / トークンを検査する ASGI ミドルウェア。

    設定は lifespan で読まれるので (``create_app`` の時点では未ロード)、``settings`` は
    リクエストのたびに呼ぶ提供関数で受ける。提供関数は読めた値をキャッシュする。
    """

    def __init__(self, app: ASGIApp, *, settings: Callable[[], GuardSettings]) -> None:
        self.app = app
        self._settings = settings

    @staticmethod
    def _host_allowed(host: str, server_port: int | None, cfg: GuardSettings) -> bool:
        if host in cfg.extra_hosts:
            return True
        ports = {str(cfg.frontend_port)}
        if server_port is not None:
            ports.add(str(server_port))
        return any(host == f"{h}:{p}" for h in LOOPBACK_HOSTS for p in ports)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers: dict[str, bytes] = {}
        for key, value in scope.get("headers") or ():
            headers[key.decode("latin-1").lower()] = value
        denial = self._check(scope, headers)
        if denial is not None:
            await _reject(scope, send, *denial)
            return
        await self.app(scope, receive, send)

    def _check(
        self, scope: Scope, headers: dict[str, bytes],
    ) -> tuple[int, str, str] | None:
        """拒否するなら (status, code, i18n_key) を返す。"""
        cfg = self._settings()
        method = str(scope.get("method") or "GET").upper()
        if cfg.token is not None:
            presented = headers.get(TOKEN_HEADER, b"")
            if not hmac.compare_digest(presented, cfg.token.encode()):
                return 401, "E0401", "error.access.token_required"
        else:
            host = headers.get("host", b"").decode("latin-1").lower()
            server = scope.get("server")
            server_port = server[1] if server and len(server) > 1 else None
            if not self._host_allowed(host, server_port, cfg):
                return 403, "E0403", "error.access.host_not_allowed"
            origin = headers.get("origin")
            if origin is not None:
                origin_s = origin.decode("latin-1").lower()
                allowed = {f"http://{h}:{cfg.frontend_port}" for h in LOOPBACK_HOSTS}
                if origin_s not in allowed:
                    return 403, "E0403", "error.access.origin_not_allowed"
        if scope["type"] == "http" and method not in SAFE_METHODS:
            if not headers.get(CLIENT_HEADER, b"").strip():
                return 403, "E0403", "error.access.client_header_required"
        return None


async def _reject(scope: Scope, send: Send, status: int, code: str, i18n_key: str) -> None:
    logger.warning(
        "Rejected %s %s: %s", scope.get("method", ""), scope.get("path", ""), i18n_key,
    )
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    body = json.dumps(
        {"detail": {"code": code, "message": msg(i18n_key), "i18n_key": i18n_key, "context": {}}},
        ensure_ascii=False,
    ).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})


def resolve_api_token(token_file: Path) -> str:
    """``allow_remote`` 用のトークンを決める。環境変数が優先、無ければファイルを作る。"""
    env_token = os.environ.get(TOKEN_ENV, "").strip()
    if env_token:
        return env_token
    if token_file.exists():
        existing = token_file.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    from backend.io.atomic import atomic_write_text

    token = secrets.token_urlsafe(32)
    atomic_write_text(token_file, token + "\n", fsync=True)
    logger.info("API token for remote access written to %s", token_file)
    return token


def extra_allowed_hosts() -> frozenset[str]:
    raw = os.environ.get(EXTRA_HOSTS_ENV, "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def _token_file() -> Path:
    # 実行時ファイル置き場 (pid と同じデータ根の run/)。CWD ではなくデータ根から引く。
    from backend.config import resolve_data_path

    return resolve_data_path("run_dir") / "api_token"


def _settings_provider() -> Callable[[], GuardSettings]:
    """設定が読めるまでは既定値 (loopback のみ・トークンなし) を返し、読めたら固定する。"""
    cached: list[GuardSettings] = []
    extra = extra_allowed_hosts()

    def provide() -> GuardSettings:
        if cached:
            return cached[0]
        from backend.config import get_config

        try:
            server_cfg = get_config().get("server", {}) or {}
        except RuntimeError:
            return GuardSettings(extra_hosts=extra)
        token = resolve_api_token(_token_file()) if server_cfg.get("allow_remote") else None
        cached.append(GuardSettings(
            frontend_port=int(server_cfg.get("frontend_port", 5173)),
            token=token,
            extra_hosts=extra,
        ))
        return cached[0]

    return provide


def register_access_guard(app: Any) -> None:
    """``create_app`` から呼ぶ。CORS より外側 (最後に add) に置く。"""
    app.add_middleware(AccessGuard, settings=_settings_provider())
