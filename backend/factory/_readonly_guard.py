"""readonly 中の書き込み拒否を API の 423 / ``E0423`` に揃える (c_05 §0.4.2)。

書き込み層 (:mod:`backend.io.readonly`) の拒否はリクエストごとに記録される。
ハンドラが ``except Exception`` で握り潰して 200 / 500 を返そうとしても、応答の
開始時点で記録があれば 423 に置き換える。既に始まったストリーミング応答
(チャットの SSE) には手を出さない — readonly でもチャットは動く。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.i18n_helper import msg
from backend.io.readonly import DataReadonlyError, begin_request_scope, readonly_reason

READONLY_CODE = "E0423"
READONLY_STATUS = 423
READONLY_I18N_KEY = "api.data_readonly"


def _readonly_body() -> dict[str, Any]:
    reason = readonly_reason() or ""
    return {
        "detail": {
            "code": READONLY_CODE,
            "message": msg(READONLY_I18N_KEY),
            "i18n_key": READONLY_I18N_KEY,
            "context": {"reason": reason},
        },
    }


class ReadonlyViolationMiddleware:
    """書き込みを拒否したリクエストの応答を 423 に置き換える ASGI ミドルウェア。"""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        violations = begin_request_scope()
        replaced = False

        async def guarded_send(message: dict[str, Any]) -> None:
            nonlocal replaced
            if replaced:
                return
            if (
                message.get("type") == "http.response.start"
                and violations
                and message.get("status") != READONLY_STATUS
            ):
                replaced = True
                body = json.dumps(_readonly_body(), ensure_ascii=False).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": READONLY_STATUS,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                })
                await send({"type": "http.response.body", "body": body, "more_body": False})
                return
            await send(message)

        await self.app(scope, receive, guarded_send)


def register_readonly_guard(app: FastAPI) -> None:
    """例外ハンドラとミドルウェアを登録する (アプリ組み立て時に 1 回)。"""

    @app.exception_handler(DataReadonlyError)
    async def _readonly_handler(request: Request, exc: DataReadonlyError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=READONLY_STATUS, content=_readonly_body())

    app.add_middleware(ReadonlyViolationMiddleware)


__all__ = [
    "READONLY_CODE",
    "READONLY_I18N_KEY",
    "READONLY_STATUS",
    "ReadonlyViolationMiddleware",
    "register_readonly_guard",
]
